// Shading shared by the closest-hit stage and the raygen: the hit description,
// environment fog per segment, the debug views and shade_and_bounce. Nothing
// here reads a hit-stage built-in; the ray that reached the surface is passed
// in, so the raygen can shade the G-buffer's primary surface with the code
// the hit shaders use for every other hit.
//
// Required includes (before this file):
//   raytracing_inc.glsl, brdf_inc.glsl, raytracing_hit_inc.glsl, raytracing_lights_inc.glsl,
//   raytracing_emissive_inc.glsl, raytracing_sky_inc.glsl, raytracing_light_tree_inc.glsl,
//   raytracing_material_eval_inc.glsl, radiance_octmap_sample()
//
// Required bindings (before this file):
//   tlas, payload, scene_data_block, geometries[], materials[], bindless_textures[],
//   SAMPLER_* (12 material samplers), rt_params, DLSS-RR images (ifdef DLSS_RR_ENABLED)

// clang-format off
#include "raytracing_texture_lod_inc.glsl"
// clang-format on

// ============================================================================
// HIT DATA
// ============================================================================

struct HitData {
	vec3 hit_pos;
	vec3 geometry_normal; // World space, flipped for back-face hits.
	vec3 tangent; // World space.
	vec3 bitangent; // World space.
	vec2 uv; // Raw UV (no material scale/offset applied).
	vec2 uv2; // The raw second UV set (D118), zero unless the hit fetched it.
	vec4 color; // Vertex color (white if not present).
	bool is_front_face;
	bool custom_group; // VAL: a custom hit group's surface (P3a): a walk's exit takes the entry's perturbation (D120).
	uint geometry_idx;
	uint primitive_id; // The triangle hit; 0 for a G-buffer surface, which never weighs its emission (D50).
	float texture_lod; // The ray cone's texture-independent level (D116), RT_TEXTURE_LOD_BASE when unknown.
	float texture_footprint; // The cone's footprint as a level in metres (D116): a texture of d texels per metre reads at this + log2(d).
	vec4 texture_grad; // The cone's elliptical footprint as the mesh's UV gradients (g1, g2) (D116), zero when the hit samples at texture_lod.
	float texture_lod2; // The cone's level at the second UV set (D118), RT_TEXTURE_LOD_BASE when unknown.
	vec4 texture_grad2; // And its elliptical footprint as the second set's gradients, zero as texture_grad is.
	float strand_inset; // VAL: how deep inside a strand its rays start (D126 H2, rt_strand_inset), world metres; 0 elsewhere.
	vec3 strand_axis; // VAL: a strand's unit axis, world space (D126 H2): the far field's frame, which tangent (orthogonal to the normal) is not on a taper; zero elsewhere.
};

// ============================================================================
// ENVIRONMENT FOG (per ray segment)
// ============================================================================

vec3 fog_get_directional_color(uint index) {
	return rt_lights[index].emission;
}

vec3 fog_get_directional_direction(uint index) {
	vec3 world_dir = -normalize(rt_lights[index].position);
	mat3 view_rot = transpose(mat3(
			scene_data_block.data.view_matrix[0].xyz,
			scene_data_block.data.view_matrix[1].xyz,
			scene_data_block.data.view_matrix[2].xyz));
	return view_rot * world_dir;
}

#define FOG_HAS_RADIANCE

vec3 fog_sample_radiance(vec3 vertex, float mip_level) {
	vec3 cube_view = scene_data_block.data.radiance_inverse_xform * vertex;
	vec2 border = vec2(scene_data_block.data.radiance_border_size,
			1.0 - scene_data_block.data.radiance_border_size * 2.0);
	vec2 cube_uv = vec3_to_oct_with_border(cube_view, border);
	// mip_level is a normalized roughness (0..1); radiance_octmap_sample maps it
	// onto the prefiltered roughness array layers. The octmap holds the exposure it was baked
	// with, so the aerial perspective takes the ratio to the frame's, as sky_radiance does.
	return radiance_octmap_sample(cube_uv, mip_level) * scene_data_block.data.IBL_exposure_normalization;
}

#include "../fog_inc.glsl"

/// Media on a path segment after the camera's (D37): the Environment's exponential fog charged
/// over the segment's own length, with the fog colour and the sun scatter along the segment's
/// direction. The camera segment is composed in the raygen over the whole path from the froxel
/// volume and the analytic fog at the primary hit, and the miss stage tints a sky seen after a
/// bounce by fog_sky_affect. Named simplifications on these later segments: depth-mode fog and
/// the height term are camera-anchored and carry no per-metre density; aerial perspective is not
/// sampled, so the fog colour stays flat instead of tinting toward the sky; and neither the
/// froxel volume nor any fog volume is evaluated, both existing only inside the camera frustum.
/// The directional lights are the first directional_light_count entries of rt_lights, which is
/// the order gather_lights packs them in and what fog_process already assumed.
void apply_segment_fog(vec3 ray_dir, float segment_length, inout vec3 radiance, inout vec3 throughput) {
	if ((RT_FLAGS & RT_FLAG_FOG_ENABLED) == 0u) {
		return;
	}
	if ((scene_data_block.data.flags & SCENE_DATA_FLAGS_USE_DEPTH_FOG) != 0u) {
		return;
	}
	float transmittance = exp(min(0.0, -segment_length * scene_data_block.data.fog_density));
	vec3 fog_color = scene_data_block.data.fog_light_color;
	if (scene_data_block.data.fog_sun_scatter > 0.001) {
		for (uint i = 0u; i < scene_data_block.data.directional_light_count; i++) {
			vec3 light_dir = -normalize(rt_lights[i].position);
			float light_amount = pow(max(dot(ray_dir, light_dir), 0.0), 8.0);
			fog_color += rt_lights[i].emission * light_amount * scene_data_block.data.fog_sun_scatter;
		}
	}
	radiance += throughput * fog_color * (1.0 - transmittance);
	throughput *= transmittance;
}

/// Converts specular parameter [0..1] to dielectric F0.
float specular_to_f0(float specular) {
	return 0.16 * specular * specular;
}

/// Exact Fresnel reflectance of a dielectric interface for unpolarised light.
/// p_eta is the ratio of the refractive indices, incident over transmitted;
/// returns 1.0 beyond the critical angle (total internal reflection).
float fresnel_dielectric(float p_cos_i, float p_eta) {
	float sin2_t = p_eta * p_eta * (1.0 - p_cos_i * p_cos_i);
	if (sin2_t >= 1.0) {
		return 1.0;
	}
	float cos_t = sqrt(1.0 - sin2_t);
	float r_s = (p_eta * p_cos_i - cos_t) / (p_eta * p_cos_i + cos_t);
	float r_p = (p_cos_i - p_eta * cos_t) / (p_cos_i + p_eta * cos_t);
	return 0.5 * (r_s * r_s + r_p * r_p);
}

/// The F0 at which Schlick's approximation reproduces p_fresnel at p_cos_i, so
/// the microfacet lobe (Schlick on the half vector) follows an exact dielectric
/// Fresnel, total internal reflection included.
float schlick_f0_for_fresnel(float p_fresnel, float p_cos_i) {
	float w = pow(1.0 - p_cos_i, 5.0);
	if (w >= 0.999) {
		// Grazing: Schlick gives its F90 whatever F0 is, so hand back the exact value.
		return p_fresnel;
	}
	return clamp((p_fresnel - w) / (1.0 - w), 0.0, 1.0);
}

// ============================================================================
// DEBUG VISUALIZATION
// ============================================================================

#ifdef RT_DEBUG_ENABLED
void debug_visualize(
		int vis_mode,
		vec3 geometry_normal,
		vec3 final_normal,
		vec3 tangent_space_normal,
		vec3 world_tangent,
		vec3 world_bitangent,
		vec2 uv,
		vec3 albedo,
		vec3 orm,
		float metalness,
		float roughness,
		float specular,
		vec3 emissive,
		vec3 V,
		float NdotV,
		float hit_t,
		bool is_front_face,
		float texture_lod) {
	PathState ps = path_unpack(payload);

	if (vis_mode == 1) {
		if (get_total_bounces(ps.packed_bounces_flags) == 0u) {
			ps.packed_bounces_flags = inc_total_bounce(ps.packed_bounces_flags);
			ps.hit_t = hit_t;
			ps.offset_normal = geometry_normal;
			ps.next_ray_dir = reflect(-V, geometry_normal);
			path_pack(payload, ps);
			return;
		} else {
			ps.radiance = geometry_normal * 0.5 + 0.5;
		}
	} else if (vis_mode == 2) {
		ps.radiance = geometry_normal * 0.5 + 0.5;
	} else if (vis_mode == 3) {
		ps.radiance = final_normal * 0.5 + 0.5;
	} else if (vis_mode == 4) {
		ps.radiance = tangent_space_normal * 0.5 + 0.5;
	} else if (vis_mode == 5) {
		ps.radiance = world_tangent * 0.5 + 0.5;
	} else if (vis_mode == 6) {
		ps.radiance = world_bitangent * 0.5 + 0.5;
	} else if (vis_mode == 7) {
		ps.radiance = vec3(fract(uv), 0.0);
	} else if (vis_mode == 8) {
		ps.radiance = albedo;
	} else if (vis_mode == 9) {
		ps.radiance = orm;
	} else if (vis_mode == 10) {
		ps.radiance = DLSSRR_computeDiffuseAlbedo(albedo, metalness);
	} else if (vis_mode == 11) {
		ps.radiance = DLSSRR_computeSpecularAlbedo(albedo, metalness, specular_to_f0(specular), roughness, NdotV);
	} else if (vis_mode == 12) {
		ps.radiance = (final_normal * 0.5 + 0.5) * (1.0 - roughness * 0.5);
	} else if (vis_mode == 13) {
		if (get_total_bounces(ps.packed_bounces_flags) == 0u) {
			if (roughness < MAX_DENOISER_SPECULAR_HIT_THRESHOLD) {
				ps.packed_bounces_flags = inc_total_bounce(ps.packed_bounces_flags);
				ps.radiance = vec3(0.1, 0.1, 0.4);
				ps.hit_t = hit_t;
				ps.offset_normal = final_normal;
				ps.next_ray_dir = reflect(-V, final_normal);
				path_pack(payload, ps);
				return;
			} else {
				ps.radiance = vec3(0.1, 0.1, 0.4);
			}
		} else {
			float spec_hit_t = hit_t;
			float v = clamp(log(spec_hit_t + 1.0) / log(1000.0), 0.0, 1.0);
			vec3 color;
			if (v < 0.33) {
				color = mix(vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), v * 3.0);
			} else if (v < 0.66) {
				color = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), (v - 0.33) * 3.0);
			} else {
				color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0), (v - 0.66) * 3.0);
			}
			ps.radiance = color;
		}
	} else if (vis_mode == 14) {
		ps.radiance = vec3(metalness);
	} else if (vis_mode == 15) {
		ps.radiance = vec3(roughness);
	} else if (vis_mode == 16) {
		mat3 world_to_view = mat3(scene_data_block.data.inv_view_matrix);
		ps.radiance = normalize(world_to_view * final_normal) * 0.5 + 0.5;
	} else if (vis_mode == 17) {
		vec3 diffuse_albedo = DLSSRR_computeDiffuseAlbedo(albedo, metalness);
		vec3 specular_albedo = DLSSRR_computeSpecularAlbedo(albedo, metalness, specular_to_f0(specular), roughness, NdotV);
		ps.radiance = mix(diffuse_albedo, specular_albedo, metalness);
	} else if (vis_mode == 18) {
		ps.radiance = baseColorToSpecularF0(albedo, metalness, specular_to_f0(specular));
	} else if (vis_mode == 19) {
		ps.radiance = is_front_face ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
	} else if (vis_mode == 20) {
		float depth_range = scene_data_block.data.z_far - scene_data_block.data.z_near;
		float d = clamp(hit_t / depth_range, 0.0, 1.0);
		ps.radiance = vec3(d);
	} else if (vis_mode == 21) {
		ps.radiance = emissive;
	} else if (vis_mode == 22) {
		// BRDF below-hemisphere fallback visualization. Reflects the two-layer
		// recovery applied in shade_and_bounce:
		//   Layer 1 - shading-normal clamp toward geometry (energy preserving
		//             but flattens detail at grazing angles)
		//   Layer 2 - mirror rejected directions across the geometry plane
		//             (biased; reuses the rejected sample's BRDF weight)
		//
		// We sample both BRDF lobes once with the un-clamped shading normal
		// and once with the clamped shading normal, reusing the same random
		// pair so the two passes are directly comparable.
		//
		// Color legend (ordered cleanest -> most biased):
		//   green   = no rejection in either pass (no fallback needed)
		//   blue    = clamp alone resolves rejection (energy preserving)
		//   yellow  = clamp + mirror (one lobe still mirrored after clamp)
		//   red     = clamp + mirror (both lobes mirrored - most biased pixel)
		//
		// Production output is never black anymore -- yellow/red just indicate
		// where the cheaper Layer-1 clamp could not catch the bump and the
		// biased Layer-2 mirror had to step in. Raise
		// RT_SHADING_NORMAL_CLAMP_THRESHOLD if yellow/red dominate.
		MaterialProperties dbg_mat;
		dbg_mat.baseColor = albedo;
		dbg_mat.metalness = metalness;
		dbg_mat.roughness = roughness;
		dbg_mat.dielectricF0 = specular_to_f0(specular);
		dbg_mat.emissive = vec3(0.0);
		dbg_mat.transmissivness = 0.0;
		dbg_mat.opacity = 1.0;
		dbg_mat.diffuseTransmission = vec3(0.0);
		dbg_mat.anisotropy = 0.0;
		dbg_mat.anisotropyT = vec3(0.0);
		dbg_mat.clearcoat = 0.0;
		dbg_mat.clearcoatAlpha = 0.0;
		dbg_mat.clearcoatN = vec3(0.0, 0.0, 1.0);
		dbg_mat.sheenColor = vec3(0.0);
		dbg_mat.sheenRoughness = 1.0;
		dbg_mat.subsurface = 0.0;
		dbg_mat.hair = false;
		dbg_mat.hairChiang = false;
		dbg_mat.hairT = vec3(0.0);
		dbg_mat.hairAxis = vec3(0.0);
		dbg_mat.hairAbsorption = vec3(0.0);
		dbg_mat.hairIor = 1.55;
		dbg_mat.hairTilt = 0.0;
		dbg_mat.hairInset = 0.0;

		vec3 dbg_dir;
		vec3 dbg_weight;
		// Draw both random pairs up front so each (specular/diffuse) test in
		// the un-clamped and clamped pass uses the exact same numbers.
		vec2 u_spec_rng = rand2(ps.rng_state);
		vec2 u_diff_rng = rand2(ps.rng_state);

		bool u_spec_ok = evalIndirectCombinedBRDF(u_spec_rng, final_normal, geometry_normal, V, dbg_mat, SPECULAR_TYPE, dbg_dir, dbg_weight, vec4(0.0));
		bool u_diff_ok = evalIndirectCombinedBRDF(u_diff_rng, final_normal, geometry_normal, V, dbg_mat, DIFFUSE_TYPE, dbg_dir, dbg_weight, vec4(0.0));

		// Uses the same threshold as shade_and_bounce so the visualization
		// stays in sync with production sampling.
		vec3 N_clamped = clampShadingNormal(final_normal, geometry_normal, V, RT_SHADING_NORMAL_CLAMP_THRESHOLD);
		bool c_spec_ok = evalIndirectCombinedBRDF(u_spec_rng, N_clamped, geometry_normal, V, dbg_mat, SPECULAR_TYPE, dbg_dir, dbg_weight, vec4(0.0));
		bool c_diff_ok = evalIndirectCombinedBRDF(u_diff_rng, N_clamped, geometry_normal, V, dbg_mat, DIFFUSE_TYPE, dbg_dir, dbg_weight, vec4(0.0));

		bool all_uncl_ok = u_spec_ok && u_diff_ok;
		bool all_cl_ok = c_spec_ok && c_diff_ok;
		bool any_cl_ok = c_spec_ok || c_diff_ok;

		if (all_uncl_ok) {
			ps.radiance = vec3(0.0, 1.0, 0.0);
		} else if (all_cl_ok) {
			ps.radiance = vec3(0.0, 0.4, 1.0);
		} else if (any_cl_ok) {
			ps.radiance = vec3(1.0, 1.0, 0.0);
		} else {
			ps.radiance = vec3(1.0, 0.0, 0.0);
		}
	} else if (vis_mode == RT_VIS_MODE_TEXTURE_LOD) {
		// The level a 1024 x 1024 texture reads at the primary hit (D116), darkened on odd levels so
		// whole levels show as bands; grey where the level is unknown and every texture reads its base.
		if (texture_lod <= RT_TEXTURE_LOD_BASE) {
			ps.radiance = vec3(0.25);
		} else {
			float level = texture_lod + 10.0;
			ps.radiance = rt_texture_lod_ramp(level) * ((int(floor(level)) & 1) != 0 ? 0.7 : 1.0);
		}
	}

	ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
	path_pack(payload, ps);
}
#endif // RT_DEBUG_ENABLED

// ============================================================================
// MEDIA (D49)
// ============================================================================

/// The radiance a white Lambertian surface with normal p_normal would reflect, unshadowed, from
/// the scene's directional lights and the sky: the hit shaders' diffuse convention
/// (emission x NdotL / pi) and the prefiltered radiance at the roughest level. A medium's
/// in-scatter is its SCATTERING times this at the interface the path entered through, so
/// SCATTERING = 1 adds what white paper facing the light would reflect.
vec3 medium_irradiance(vec3 p_normal) {
	vec3 e = vec3(0.0);
	for (uint i = 0u; i < scene_data_block.data.directional_light_count; i++) {
		vec3 light_dir = -normalize(rt_lights[i].position);
		e += rt_lights[i].emission * (max(dot(p_normal, light_dir), 0.0) * ONE_OVER_PI);
	}
	vec3 sky_dir = scene_data_block.data.radiance_inverse_xform * p_normal;
	vec2 border = vec2(scene_data_block.data.radiance_border_size, 1.0 - scene_data_block.data.radiance_border_size * 2.0);
	e += radiance_octmap_sample(vec3_to_oct_with_border(sky_dir, border), 1.0) * scene_data_block.data.IBL_exposure_normalization;
	return e;
}

/// The segment inside a medium: Beer-Lambert over its length with the payload's extinction, and
/// the in-scatter the medium adds in place of what it took (the closed form the segment fog uses).
void apply_segment_medium(float segment_length, inout PathState ps) {
	vec3 transmittance = exp(-path_medium_absorption(ps) * segment_length);
	ps.radiance += ps.throughput * path_medium_inscatter(ps) * (vec3(1.0) - transmittance);
	ps.throughput *= transmittance;
}

// ============================================================================
// SHADE AND BOUNCE
// ============================================================================

/// Production shading: emissive + NEE direct lighting + BRDF importance sampling + next bounce.
/// Also handles DLSS-RR G-buffer output on primary ray.
/// The vertex the ray that reached h left from: the hit stage's own origin, or reconstructed
/// from the hit for the raygen's G-buffer surface (whose camera ray never weighs an emitter).
vec3 shade_previous_origin(HitData h, vec3 ray_dir, float hit_t) {
#ifdef RT_STAGE_CLOSEST_HIT
	return gl_WorldRayOriginEXT;
#else
	return h.hit_pos - ray_dir * hit_t;
#endif
}

#ifdef RT_STAGE_CLOSEST_HIT
// VAL (GPU audit P3e): a shadow ray the inline test leaves unresolved is traced through the pipeline at the end of the
// closest hit (shade_resolve_shadows), after the payload is packed, rather than where its light sample is drawn: a
// trace in the middle of the shading split the hit into continuations that saved and restored everything live across
// it. A hit draws at most two light samples, slot 0 at the hit and slot 1 at a subsurface walk's exit, and nothing
// writes the radiance between or after them, so the radiance each outcome would have is formed here in the original
// order, in fp32 (rt_nee_radiance1 with slot 0's light, 2 with slot 1's, 3 with both, ps.radiance with neither), and
// rounded once by path_pack_radiance, as path_pack rounds the radiance: the same bits. That makes it a rule: nothing
// may write ps.radiance after a hit's first shade_direct_light, since a pending outcome's sum replaces the radiance and
// would drop the write (shade_and_bounce's contract). A sum is read only for a set of slots whose bits are pending, so
// one left unset is never chosen; they are not initialised, as a value set at the hit's entry would hold registers
// through the whole shading (4 bytes more spill, tools/shader-stats.ps1). Both slots pass the hit's arriving diffuse
// count, so one set of opacity micromap flags serves both.
uint rt_shadow_pending = 0u; // Bit s: slot s's ray waits for the pipeline.
vec3 rt_nee_radiance1;
vec3 rt_nee_radiance2;
vec3 rt_nee_radiance3;
vec3 rt_shadow_origin0;
vec3 rt_shadow_dir0;
float rt_shadow_distance0;
vec3 rt_shadow_origin1;
vec3 rt_shadow_dir1;
float rt_shadow_distance1;
uint rt_shadow_omm_flags;
#endif

/// One light sample at a surface for its next-event estimate (D50 to D72): the strategy split, the sample,
/// the shadow ray and the BSDF, added to ps.radiance at the path's throughput (in a closest hit, a ray the inline test
/// cannot decide waits for the end of the hit with the sum it would make, GPU audit P3e). The caller pins the
/// sampler's dimension first. p_slot is the sample's place at its hit, 0 at the surface and 1 at a subsurface walk's
/// exit. p_hit_pos and p_geometry_normal are the vertex and its normal facing the side the path arrived from, p_hit_t
/// the length of the segment that reached it (for the origin's offset); diffuse_transmit_max a thin surface's
/// transmission (D103), whose samples below N leave from the back.
void shade_direct_light(vec3 p_hit_pos, vec3 p_geometry_normal, float p_hit_t, vec3 N, vec3 V, MaterialProperties brdf_mat, float diffuse_transmit_max, uint diffuse_bounces, uint rt_emissive_count, bool in_medium, uint p_slot, inout PathState ps) {
	uint rt_light_count = uint(get_rt_param(RT_PARAM_LIGHT_COUNT));
	// The selection's receiver: the geometry normal as the payload carries it to the next vertex,
	// so the chance-hit and miss densities there read the same function of the same normal (the
	// light tree's since D55, the sky's share by variance since D65).
	vec3 N_select = oct_to_vec3(unpackUnorm2x16(packUnorm2x16(vec3_to_oct(p_geometry_normal))) * 2.0 - 1.0);
	// The sky's share of the light samples (D54), 0 without a sky light; the surface lights
	// split the rest, so their densities carry the remainder on both sides of the weight.
	float p_sky = light_tree_enabled() ? 0.0 : sky_light_probability(N_select);
	// What the next vertex weighs a sky or an emitter it finds against (GPU audit 4a); the tree sets its own below,
	// and the top's total (light 1a), 0 until the tree scores one.
	ps.sky_probability = p_sky;
	ps.tree_total = 0.0;
	// Under the tree (D55) the sky's share is per hit, so a frame whose only light is the sky still
	// takes its sample here (the miss weighs against it either way).
	bool tree_sky = light_tree_enabled() && sky_light_enabled() && sky_total > 0.0;
	if (rt_light_count > 0u || rt_emissive_count > 0u || p_sky > 0.0 || tree_sky) {
		// A strand's rays start inside the fibre (D126 H2), toward a light on either side.
		const bool strand_inside = hasHair(brdf_mat) && brdf_mat.hairInset > 0.0;
		vec3 hit_pos_offset = strand_inside ? p_hit_pos - p_geometry_normal * brdf_mat.hairInset : offset_ray_origin(p_hit_pos, p_geometry_normal, p_hit_t);
		bool is_indirect = (diffuse_bounces > 0u);
		// One light sample per hit (D50): the analytic lights or the emissive list, chosen by the
		// per-frame share, the contribution divided by its share. Nothing is drawn for the choice
		// when the emissive list is empty, so such a frame keeps its random sequence. Only the
		// sample's generation diverges; the shadow ray and the BRDF below are shared, so a warp
		// whose threads chose differently traces once. The sky is the third strategy (D54), drawn
		// first by its share; the surface lights split the rest as before, so a frame without a
		// sky light keeps its random sequence.
		bool sample_sky = p_sky > 0.0;
		float strategy_probability = 1.0;
		if (sample_sky && (rt_light_count > 0u || rt_emissive_count > 0u)) {
			sample_sky = rand(ps.rng_state) < p_sky;
			strategy_probability = sample_sky ? p_sky : (1.0 - p_sky);
		}
		bool sample_emissive = !sample_sky && rt_emissive_count > 0u;
		if (!light_tree_enabled() && !sample_sky && rt_light_count > 0u && rt_emissive_count > 0u) {
			float p_emissive = get_rt_param(RT_PARAM_EMISSIVE_PROBABILITY);
			sample_emissive = rand(ps.rng_state) < p_emissive;
			strategy_probability *= sample_emissive ? p_emissive : (1.0 - p_emissive);
		}
		LightSampleNEE light_sample;
		bool have_sample;
		if (light_tree_enabled()) {
			// The light tree (D55) chooses among every strategy by its importance here, judged on
			// N_select, so the next vertex recovers the same probability; the shading normal still
			// decides what the sample delivers.
			uint tree_candidates = uint(get_rt_param(RT_PARAM_EMISSIVE_CANDIDATES));
			// Past the first diffuse bounce the sample is drawn over the coarser indirect cut (D59);
			// the payload tells the next vertex, whose chance-hit and miss densities read the same cut.
			if (is_indirect) {
				ps.packed_bounces_flags = set_light_tree_indirect(ps.packed_bounces_flags);
			}
			have_sample = light_tree_sample_direct(hit_pos_offset, N_select, N, V, brdf_mat, ps.rng_state, is_indirect, tree_candidates, light_sample, strategy_probability, ps.sky_probability, ps.tree_total);
		} else if (sample_sky) {
			have_sample = sky_sample_direct(N, V, brdf_mat, ps.rng_state, strategy_probability, light_sample);
		} else if (sample_emissive) {
			uint candidates = uint(get_rt_param(RT_PARAM_EMISSIVE_CANDIDATES));
			have_sample = emissive_sample_direct(hit_pos_offset, N, V, brdf_mat, ps.rng_state, strategy_probability, rt_emissive_count, candidates, light_sample);
		} else {
			have_sample = lights_sample_direct(hit_pos_offset, N, brdf_mat, ps.rng_state, is_indirect, rt_light_count, light_sample);
		}
		// A sample a thin surface takes through its transmission (D103) leaves from the other side,
		// aimed at the point the light was sampled at from the front origin, so it ends where the
		// front ray would; the BRDF keeps the sample's own direction.
		vec3 shadow_origin = hit_pos_offset;
		vec3 shadow_dir = light_sample.direction;
		float shadow_distance = light_sample.shadow_distance;
		// A strand's already starts inside it (D126 H2), from where a light on either side is reached through the fibre, which
		// a ray starting inside a linear swept sphere cannot hit; a strand without a radius leaves from the back.
		if (!strand_inside && (diffuse_transmit_max > 0.0 || hasHair(brdf_mat)) && dot(N, light_sample.direction) < 0.0 && !(hasClearcoat(brdf_mat) && dot(brdf_mat.clearcoatN, light_sample.direction) > 0.0)) {
			shadow_origin = offset_ray_origin(p_hit_pos, -p_geometry_normal, p_hit_t);
			vec3 to_light = hit_pos_offset + light_sample.direction * light_sample.shadow_distance - shadow_origin;
			shadow_distance = length(to_light);
			shadow_dir = to_light / max(shadow_distance, 1e-6);
		}
#ifdef RT_STAGE_CLOSEST_HIT
		uint shadow = have_sample ? lights_shadow_ray_inline(shadow_origin, shadow_dir, shadow_distance, RT_OMM_RAY_FLAGS(diffuse_bounces)) : RT_SHADOW_OCCLUDED;
#else
		uint shadow = (have_sample && lights_trace_shadow_ray(shadow_origin, shadow_dir, shadow_distance, RT_OMM_RAY_FLAGS(diffuse_bounces), ps.rng_state)) ? RT_SHADOW_VISIBLE : RT_SHADOW_OCCLUDED;
#endif
		if (shadow != RT_SHADOW_OCCLUDED) {
			// Evaluate BRDF (diffuse + specular separately for specular_amount control).
			vec3 brdf_diffuse, brdf_specular;
			if (light_sample.hair_value.x >= 0.0) {
				// A strand's sample shades with the evaluation its MIS weight took (D126 H2), as evalCombinedBRDFSeparate
				// would return it: all in the diffuse output.
				brdf_diffuse = light_sample.hair_value;
				brdf_specular = vec3(0.0);
			} else {
				evalCombinedBRDFSeparate(N, light_sample.direction, V, brdf_mat, brdf_diffuse, brdf_specular);
			}
			// NdotL is already included in the BRDF value (evalLambertian/evalMicrofacet bake it in).
			vec3 direct_light = (brdf_diffuse + brdf_specular * light_sample.specular_multiplier) * light_sample.radiance * (light_sample.weight / strategy_probability);
			if (in_medium) {
				// The light's own leg through the medium (D49): to the light, or up to the medium's
				// top, the horizontal plane through the point the path entered at, whichever is
				// nearer. The shadow ray is a visibility test that does not resolve the interface, so
				// the leg is this estimate (a named limit); a light under the plane takes its distance.
				float leg = min(light_sample.leg_distance, max(ps.medium_entry_height - p_hit_pos.y, 0.0) / max(light_sample.direction.y, 0.1));
				direct_light *= exp(-path_medium_absorption(ps) * leg);
			}
			// The clamp (D53): direct at the surface the camera sees, indirect after a scatter.
			vec3 contribution = rt_clamp_contribution(ps.throughput * direct_light, get_rt_param(is_scattered(ps.packed_bounces_flags) ? RT_PARAM_CLAMP_INDIRECT : RT_PARAM_CLAMP_DIRECT));
#ifdef RT_STAGE_CLOSEST_HIT
			if (shadow == RT_SHADOW_UNRESOLVED) {
				// The ray waits for the end of the hit; the radiance it would bring is formed now, in order.
				if (p_slot == 0u) {
					rt_nee_radiance1 = ps.radiance + contribution;
					rt_shadow_origin0 = shadow_origin;
					rt_shadow_dir0 = shadow_dir;
					rt_shadow_distance0 = shadow_distance;
				} else {
					rt_nee_radiance2 = ps.radiance + contribution;
					rt_nee_radiance3 = rt_nee_radiance1 + contribution;
					rt_shadow_origin1 = shadow_origin;
					rt_shadow_dir1 = shadow_dir;
					rt_shadow_distance1 = shadow_distance;
				}
				rt_shadow_pending |= 1u << p_slot;
				rt_shadow_omm_flags = RT_OMM_RAY_FLAGS(diffuse_bounces);
			} else {
				ps.radiance += contribution;
				if (p_slot == 1u) {
					rt_nee_radiance1 += contribution; // Slot 0's outcome, if it waits, gains the exit's light after it.
				}
			}
#else
			ps.radiance += contribution;
#endif
		}
	}
}

#ifdef RT_SUBSURFACE_COMPILED
// ============================================================================
// SUBSURFACE WALK (D120, docs/r2-subsurface.md)
// ============================================================================

#define RT_SUBSURFACE_FREE_EVENTS 64u // Scatter events before Russian roulette (nvpro's free budget).
#define RT_SUBSURFACE_MAX_EVENTS 256u // A walk this long ends its path (Cycles' cap).
#define RT_SUBSURFACE_MAX_SKIPS 4u // Front faces stepped past in one step; past them the step scatters at its distance.
#define RT_SUBSURFACE_EXITED 0u
#define RT_SUBSURFACE_DIED 1u
#define RT_SUBSURFACE_CAPPED 2u

#ifdef RT_SUBSURFACE_STATS
// The walk's counters (D120), one pixel in 64: walks, steps (queries), the most steps of one walk, walks that left the
// surface, walks Russian roulette or a zero throughput ended, walks the cap ended, front faces stepped past. The host
// reads and clears them on the verbose report's cadence.
layout(set = 0, binding = 52, std430) buffer SubsurfaceStats {
	uint words[];
}
subsurface_stats;
#endif

float subsurface_rand(inout uint state) {
	state = pcg_hash(state);
	return float(state) * (1.0 / 4294967296.0);
}

/// The single-scattering albedo of a medium whose semi-infinite slab reflects A in all (Chiang et al. 2016, the form
/// Cycles' volume_subsurface_albedo_to_single_scattering takes at g = 0).
vec3 subsurface_single_scattering_albedo(vec3 A) {
	vec3 s = vec3(4.09712) + 4.20863 * A - sqrt(vec3(9.59217) + 41.6808 * A + 17.7126 * A * A);
	return vec3(1.0) - s * s;
}

/// A texture at a walk's exit, at a ray-cone level carried from the entry for the texture's UV set (level 0 without one).
vec4 subsurface_exit_sample(uint p_texture_idx, vec2 p_uv, float p_cone_lod, vec2 p_uv_scale) {
	// The cone's level is per unit of UV; the texture's own size makes it a mip level, as sample_material_texture does.
	float lod = 0.0;
	if (rt_texture_lod_enabled() && p_cone_lod > RT_TEXTURE_LOD_BASE) {
		lod = max(p_cone_lod + 0.5 * log2(max(abs(p_uv_scale.x * p_uv_scale.y), 1e-20)) + rt_texel_log(p_texture_idx), 0.0);
	}
	return textureLod(sampler2D(bindless_textures[nonuniformEXT(p_texture_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), p_uv, lod);
}

/// The shading normal at a walk's exit (D120): the whole material's normal, as Cycles evaluates it at the exit; the forms
/// a normal map carries (a spine's groove, a collarbone, a layered overlay's own anatomy) belong to the surface the light
/// leaves.
/// - A standard material's normal map (flag bit 0) at the exit's UV1 and its detail layer (flag bit 14) at the layer's UV
///   set, mixed as the hit's HG0 fill mixes them: the detail normal (the flat one when unassigned, which flattens the base
///   under an albedo-only layer's alpha) under the detail albedo's alpha and the mask's red, read on UV1; through the
///   exit's interpolated tangent frame.
/// - Each map is read at the entry's ray-cone level for its UV set, taken through the exit material's UV scale as HG0
///   takes it, or level 0 without texture LOD or when the entry carried no level for that set (an entry row that did not
///   read UV2). The level is the entry mesh's density, carried over the millimetres the walk moved, also across rows.
/// - Otherwise, from a custom hit group (whose fragment cannot run here), the entry's perturbation: the rotation taking the
///   entry's interpolated normal to its shading normal, applied to the exit's when the exit faces the entry's side.
/// - Otherwise the exit's interpolated normal, also on a row stored without tangents.
/// A result at or below the exit's plane is lifted just above it, as the hit's clamp keeps its shading normal.
vec3 subsurface_exit_shading_normal(HitData h, vec3 N, uint row, GeometryData geom, uint i0, uint i1, uint i2, vec3 bary, TBNResult tbn, mat3 normal_matrix, float handedness, vec3 n) {
	vec3 ns = n;
	MaterialData mat = materials[row];
	vec3 t = normal_matrix * tbn.tangent;
	t -= n * dot(n, t);
	// A row stored without tangents carries fetch_tbn's placeholder; the hit derives its frame there instead, which this
	// include cannot, so such an exit keeps its interpolated normal.
	const bool has_tangent = (geom.flags & FLAG_COMPRESSED) != 0u || geom.normal_stride >= 8u;
	const bool base_mapped = (mat.flags & 1u) != 0u;
	const bool detail_layer = (mat.flags & (1u << 14)) != 0u;
	if ((base_mapped || (detail_layer && mat.detail_normal_texture_idx != 0u)) && has_tangent && dot(t, t) > 1e-10) {
		t = normalize(t);
		vec3 b = cross(n, t) * tbn.bitangent_sign * handedness;
		vec2 uv = fetch_uv(geom, i0, i1, i2, bary) * mat.uv1_scale + mat.uv1_offset;
		// Encoded until the detail layer has mixed its own in, as the hit does; without a base map it is the flat 0.5.
		vec2 sample_xy = base_mapped ? subsurface_exit_sample(mat.normal_texture_idx, uv, h.texture_lod, mat.uv1_scale).xy : vec2(0.5);
		if (detail_layer) {
			const bool detail_on_uv2 = (mat.flags & (1u << 17)) != 0u;
			const vec2 detail_uv = detail_on_uv2 ? fetch_uv2(geom, i0, i1, i2, bary) * mat.uv2_scale + mat.uv2_offset : uv;
			const float detail_cone = detail_on_uv2 ? h.texture_lod2 : h.texture_lod;
			const vec2 detail_scale = detail_on_uv2 ? mat.uv2_scale : mat.uv1_scale;
			const float detail_alpha = mat.detail_albedo_texture_idx != 0u ? subsurface_exit_sample(mat.detail_albedo_texture_idx, detail_uv, detail_cone, detail_scale).a : 1.0;
			const float detail_mask = mat.detail_mask_texture_idx != 0u ? subsurface_exit_sample(mat.detail_mask_texture_idx, uv, h.texture_lod, mat.uv1_scale).r : 1.0;
			const vec2 detail_xy = mat.detail_normal_texture_idx != 0u ? subsurface_exit_sample(mat.detail_normal_texture_idx, detail_uv, detail_cone, detail_scale).xy : vec2(0.5);
			sample_xy = mix(sample_xy, mix(sample_xy, detail_xy, detail_alpha), detail_mask);
		}
		vec3 ts = vec3(sample_xy * 2.0 - 1.0, 0.0);
		ts.z = sqrt(max(0.0, 1.0 - dot(ts.xy, ts.xy)));
		ns = normalize(mix(n, t * ts.x + b * ts.y + n * ts.z, mat.normal_map_depth));
	}
#ifdef RT_CUSTOM_HIT_GROUP
	else if (dot(n, h.geometry_normal) > 0.0) {
		// Rodrigues: the rotation taking the entry's interpolated normal to its shading normal, about their common axis.
		vec3 axis = cross(h.geometry_normal, N);
		float sine = length(axis);
		if (sine > 1e-6) {
			axis /= sine;
			float cosine = dot(h.geometry_normal, N);
			ns = normalize(n * cosine + cross(axis, n) * sine + axis * dot(axis, n) * (1.0 - cosine));
		}
	}
#endif
	float above = dot(ns, n);
	if (above < 0.05) {
		ns = normalize(ns + n * (0.05 - above));
	}
	return ns;
}

/// A random walk under the surface h (Cycles' shape): in by a cosine direction about -N from the hit, through a
/// homogeneous medium whose coefficients come from the albedo and the mean free path (the setting times skin's channel
/// ratios or equal ones), a channel drawn per step by the throughput it carries times its albedo and the step weighed
/// by the balance heuristic over the three, isotropic scattering, Russian roulette past the free events and a hard cap.
/// The walk's rays see only subsurface instances. The first back face of any of them ends the walk: a head and a body
/// joined at an open seam pass walks across it, and two touching subsurface objects exchange them. A front face is
/// another walking surface met from outside inside this body (the head's neck shell from the body, teeth in the gums):
/// the step goes on past it, at most RT_SUBSURFACE_MAX_SKIPS times, then scatters at its distance. On RT_SUBSURFACE_EXITED the
/// throughput has carried the walk and r_exit_* name where it left: the point, the interpolated vertex normal facing
/// out, the shading normal the exit is lit and leaves about (r_exit_shading_normal), the step's length. The walk draws from a hash of its own, seeded per vertex.
uint subsurface_walk(HitData h, vec3 p_albedo, bool p_skin, vec3 N, float hit_t, uint p_seed, inout vec3 throughput, out vec3 r_exit_position, out vec3 r_exit_normal, out vec3 r_exit_shading_normal, out float r_exit_t, out uint r_events, out uint r_skips) {
	r_exit_position = h.hit_pos;
	r_exit_normal = h.geometry_normal;
	r_exit_shading_normal = N;
	r_exit_t = hit_t;
	r_events = 0u;
	r_skips = 0u;
	const vec3 A = clamp(p_albedo, vec3(0.0), vec3(1.0));
	// The diffuse lobe's weight carries the albedo, which the walk adds back by scattering: a slab of this medium reflects
	// A (Cycles divides it out the same way).
	throughput = vec3(A.r > 0.0 ? throughput.r / A.r : 0.0, A.g > 0.0 ? throughput.g / A.g : 0.0, A.b > 0.0 ? throughput.b / A.b : 0.0);
	const vec3 ratio = p_skin ? vec3(1.0, 0.373, 0.185) : vec3(1.0);
	const vec3 sigma_t = 1.0 / max(get_rt_param(RT_PARAM_SUBSURFACE_SCALE) * ratio, vec3(1e-7));
	const vec3 alpha = subsurface_single_scattering_albedo(A);
	const vec3 sigma_s = alpha * sigma_t;

	uint rng = p_seed;
	// The walk's position is kept as an offset from the hit, so trace space rounds each query's origin once and the steps
	// do not accumulate its error: 1 km from the relative origin (a 1024 m rendering/pathtracing/relative_origin_grid) a
	// float's step is about 0.1 mm, a fair share of a mean free path. The walk starts on the hit itself, its first query
	// stepping over the entered triangle by a few of those steps (offset_ray_origin moves a point 256 of them, about
	// 1.5 cm there, which started walks centimetres deep); a step past a front face starts again as far beyond it. The hit
	// was found as origin + direction * hit_t, so its error follows the larger of the two magnitudes, as in
	// offset_ray_origin's floor: near a small grid's anchor, a hit seen from further away carries more than its own.
	vec3 local = vec3(0.0);
	const float entry_tmin = max(max(max(abs(h.hit_pos.x), max(abs(h.hit_pos.y), abs(h.hit_pos.z))), hit_t) * (4.0 / 8388608.0), 1e-6);
	vec3 direction = rotatePoint(getRotationFromZAxis(-N), sampleHemisphere(vec2(subsurface_rand(rng), subsurface_rand(rng))));
	// A shading normal tilted far from the geometry can aim the entry back out; mirror it under the plane.
	float above = dot(direction, h.geometry_normal);
	if (above > 0.0) {
		direction = normalize(direction - 2.0 * above * h.geometry_normal);
	}

	for (uint event = 0u; event < RT_SUBSURFACE_MAX_EVENTS; event++) {
		r_events = event;
		vec3 channel_weight = abs(throughput * alpha);
		float weight_sum = channel_weight.r + channel_weight.g + channel_weight.b;
		if (!(weight_sum > 0.0)) {
			return RT_SUBSURFACE_DIED; // Nothing left to carry (or not a number).
		}
		vec3 channel_pdf = channel_weight / weight_sum;
		float u_channel = subsurface_rand(rng);
		float sigma = u_channel < channel_pdf.r ? sigma_t.r : (u_channel < channel_pdf.r + channel_pdf.g ? sigma_t.g : sigma_t.b);
		float t = -log(max(1.0 - subsurface_rand(rng), 1e-10)) / sigma;

		// The step: the first back face along it is the exit; front faces are stepped past.
		bool exited = false;
		float t_exit = t;
		float travelled = 0.0;
		float step_tmin = event == 0u ? min(entry_tmin, t) : 0.0;
		for (uint skip = 0u; skip <= RT_SUBSURFACE_MAX_SKIPS && travelled < t; skip++) {
			// VAL: past a front face the query starts entry_tmin on, and a step with less than that left has an empty
			// interval: no face ends it, and a query whose tmin exceeds its tmax is undefined (lights_shadow_interval_empty).
			if (!(t - travelled >= step_tmin)) {
				break;
			}
			rayQueryEXT q;
			rayQueryInitializeEXT(q, tlas, gl_RayFlagsOpaqueEXT, RT_MASK_SUBSURFACE, h.hit_pos + (local + direction * travelled), step_tmin, direction, t - travelled);
			while (rayQueryProceedEXT(q)) {
			}
			if (rayQueryGetIntersectionTypeEXT(q, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
				break;
			}
			float t_hit = travelled + rayQueryGetIntersectionTEXT(q, true);
			if (rayQueryGetIntersectionFrontFaceEXT(q, true)) {
				r_skips++;
				travelled = t_hit;
				step_tmin = entry_tmin; // Over the face just met.
				continue;
			}
			exited = true;
			t_exit = t_hit;
			// The exit's interpolated vertex normal, which faces out of the body it leaves, as compute_hit_data builds the hit's.
			int cluster_id = RT_CLUSTER_ID_NONE;
#ifdef USE_CLUSTER_AS
			cluster_id = rayQueryGetIntersectionClusterIdNV(q, true);
#endif
			GeometryData geom = geometries[rayQueryGetIntersectionInstanceCustomIndexEXT(q, true)];
			uint i0, i1, i2;
			rt_resolve_triangle(geom, cluster_id, uint(rayQueryGetIntersectionPrimitiveIndexEXT(q, true)), i0, i1, i2);
			vec2 b = rayQueryGetIntersectionBarycentricsEXT(q, true);
			TBNResult tbn = fetch_tbn(geom, i0, i1, i2, vec3(1.0 - b.x - b.y, b.x, b.y));
			mat3 basis = rt_instance_basis(geom, mat3(rayQueryGetIntersectionObjectToWorldEXT(q, true)));
			mat3 normal_matrix = rt_normal_basis(basis);
			r_exit_normal = normalize(normal_matrix * tbn.normal);
			r_exit_shading_normal = subsurface_exit_shading_normal(h, N, rayQueryGetIntersectionInstanceCustomIndexEXT(q, true), geom, i0, i1, i2, vec3(1.0 - b.x - b.y, b.x, b.y), tbn, normal_matrix, determinant(basis) < 0.0 ? -1.0 : 1.0, r_exit_normal);
			break;
		}

		if (exited) {
			vec3 transmittance = exp(-sigma_t * t_exit);
			throughput *= transmittance / dot(channel_pdf, transmittance);
			r_exit_position = h.hit_pos + (local + direction * t_exit);
			r_exit_t = t_exit;
			r_events = event + 1u;
			return RT_SUBSURFACE_EXITED;
		}
		vec3 transmittance = exp(-sigma_t * t);
		throughput *= sigma_s * transmittance / dot(channel_pdf, sigma_t * transmittance);
		local += direction * t;
		float z = 1.0 - 2.0 * subsurface_rand(rng);
		float r = sqrt(max(0.0, 1.0 - z * z));
		float phi = TWO_PI * subsurface_rand(rng);
		direction = vec3(r * cos(phi), r * sin(phi), z);

		if (event + 1u >= RT_SUBSURFACE_FREE_EVENTS) {
			float survive = min(max(throughput.r, max(throughput.g, throughput.b)), 0.999);
			if (!(subsurface_rand(rng) < survive)) {
				r_events = event + 1u;
				return RT_SUBSURFACE_DIED;
			}
			throughput /= survive;
		}
	}
	r_events = RT_SUBSURFACE_MAX_EVENTS;
	return RT_SUBSURFACE_CAPPED;
}
#endif

#ifdef RT_SUBSURFACE_COMPILED
/// Whether this hit group walks (D120 SS2): every custom hit group (each folds the share by its own SSS_STRENGTH), and the
/// standard hit group only in its walking variant, which the instances whose material can walk select; the plain one
/// folds the walk away, which would otherwise cost every standard hit the walk's registers. The surface record carries
/// the share this gives (P3a).
bool rt_subsurface_walk_group() {
#ifdef RT_CUSTOM_HIT_GROUP
	return true;
#else
	return (RT_FLAGS & RT_FLAG_SUBSURFACE_WALK_GROUP) != 0u;
#endif
}
#endif

/// ray_dir and hit_t describe the ray that reached h.hit_pos: the hit stage's
/// built-ins, or the raygen's own primary ray for a G-buffer surface. ps is the path
/// as it arrived; on return it is the path the raygen continues or ends (the caller
/// packs it into the payload, P3a). In a closest hit, ps.radiance on return leaves out
/// the light of any shadow ray still pending (GPU audit P3e): the caller packs ps with
/// path_pack and then calls shade_resolve_shadows(), which replaces the payload's
/// radiance with the outcome's sum. So nothing may write ps.radiance after the first
/// shade_direct_light: the sums were formed from it there and would drop the write.
void shade_and_bounce(HitData h, MaterialResult m, vec3 ray_dir, float hit_t, inout PathState ps) {
	// Every exit below hands the path back, and the raygen reads hit_t for the camera segment's
	// length before it checks for termination, so the distance is recorded once, here. The
	// previous vertex's back offset (D103) is spent: only a transmission below sets it again.
	ps.hit_t = hit_t;
	ps.packed_bounces_flags = set_back_offset(ps.packed_bounces_flags, false);
	// And so is a strand's origin (D126 H2): this hit's rays take their own. Whether the ray left a strand is read once
	// more, by the emitter weight below: the light tree scored that vertex's receiver on both sides.
	const bool incoming_strand = (ps.packed_bounces_flags & HAIR_RAY_FLAG) != 0u;
	ps.packed_bounces_flags &= ~(HAIR_RAY_FLAG | HAIR_INSET_MASK);
	rt_shadow_tmin = m.hair ? RT_HAIR_TMIN : RT_SHADOW_TMIN;
#ifdef RT_SUBSURFACE_COMPILED
	ps.packed_bounces_flags &= ~SUBSURFACE_EXIT_FLAG; // And so is the previous vertex's walk exit (D120).
	// The raygen reads the exit only under the flag, which only a walk below sets again with its own exit, so the
	// incoming one is not carried through the shading (GPU audit P3e step 3).
	ps.exit_position = vec3(0.0);
#endif
	// The ray cone moves to the hit (D116), which is the next ray's origin on every exit.
	ps.cone_width += ps.cone_spread * hit_t;
	// The densities the previous vertex drew with are read once more, by the emitter weight below. A path that ends here
	// returns none of them, and every exit that continues writes them again (shade_direct_light, the scatter, the
	// refraction), so the incoming values are not carried through the shading (GPU audit P3e step 3).
	float incoming_bsdf_pdf = ps.bsdf_pdf;
	float incoming_sky_probability = ps.sky_probability;
	float incoming_tree_total = ps.tree_total;
	ps.bsdf_pdf = RT_BSDF_PDF_DELTA;
	ps.sky_probability = 0.0;
	ps.tree_total = 0.0;

	vec3 V = -ray_dir;

	// Clamp shading normal toward geometry at grazing view angles to keep BRDF above the geometry hemisphere.
	vec3 N = clampShadingNormal(m.normal, h.geometry_normal, V, RT_SHADING_NORMAL_CLAMP_THRESHOLD);
	float NdotV = max(dot(N, V), 0.0001);

	// The bounce counts are read from the flags where they are used, rather than cached here: the flags are live through
	// the shading anyway, and nothing below changes the counts before their last use (GPU audit P3e step 3).

	// Media on this segment (D37): the camera segment is composed in the raygen over the whole
	// path; every later segment is charged here, over its own length. Inside a medium (D49) the
	// segment takes the payload's medium instead of the Environment's fog, the camera segment too
	// (the raygen seeds the medium the camera is in and composes no fog over it).
	bool in_medium = is_in_medium(ps.packed_bounces_flags);
	if (in_medium) {
		apply_segment_medium(hit_t, ps);
	} else if (get_total_bounces(ps.packed_bounces_flags) > 0u) {
		apply_segment_fog(ray_dir, hit_t, ps.radiance, ps.throughput);
	}

	// Emissive contribution, weighed against the emissive light sampling that competes for it
	// after a scatter (D50): the camera ray and a delta lobe keep the whole weight, and so does
	// a surface the emissive list does not carry, since light sampling never reaches it.
	uint rt_emissive_count = uint(get_rt_param(RT_PARAM_EMISSIVE_COUNT));
	float emissive_weight = 1.0;
	if (get_total_bounces(ps.packed_bounces_flags) > 0u && rt_emissive_count > 0u && incoming_bsdf_pdf != RT_BSDF_PDF_DELTA && any(greaterThan(m.emissive, vec3(0.0)))) {
		float pdf_light;
		if (light_tree_enabled()) {
			// The light tree's probability of this entry at the previous vertex (D55): its cut node scored at
			// the ray's origin and the offset normal the payload carried from there, over the top's total
			// the payload carried too (GPU audit light 1a).
			uint entry = emissive_find_entry(h.geometry_idx, rt_emissive_count);
			float p_instance = (entry < rt_emissive_count) ? light_tree_emissive_probability(entry, shade_previous_origin(h, ray_dir, hit_t), ps.offset_normal, is_light_tree_indirect(ps.packed_bounces_flags), incoming_strand, incoming_tree_total) : 0.0;
			pdf_light = (entry < rt_emissive_count) ? emissive_hit_pdf_entry(entry, p_instance, h.primitive_id, ray_dir, hit_t, h.is_front_face) : 0.0;
		} else {
			// The sky's share of the light samples (D54), 0 without a sky light; the surface lights
			// split the rest, so their densities carry the remainder on both sides of the weight. It is
			// the previous vertex's share as drawn there, carried in the payload (GPU audit 4a).
			pdf_light = (1.0 - incoming_sky_probability) * get_rt_param(RT_PARAM_EMISSIVE_PROBABILITY) * emissive_hit_pdf(h.geometry_idx, h.primitive_id, ray_dir, hit_t, h.is_front_face, rt_emissive_count);
		}
		emissive_weight = emissive_mis_weight(incoming_bsdf_pdf, pdf_light);
	}
	// A decal's emission (D104) is not in the emissive list, so the BSDF ray is its only estimator.
	vec3 emitted = ps.throughput * m.emissive * emissive_weight + ps.throughput * m.decal_emissive;
	if (is_scattered(ps.packed_bounces_flags)) {
		// An emitter found after a scatter is indirect light; the one the camera sees, through
		// any delta interfaces, is the image (D53).
		emitted = rt_clamp_contribution(emitted, get_rt_param(RT_PARAM_CLAMP_INDIRECT));
	}
	ps.radiance += emitted;

	// Bounce limit check.
	if (rt_bounce_limit_reached(ps.packed_bounces_flags)) {
		ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
		return;
	}

	// BRDF material setup.
	MaterialProperties brdf_mat;
	brdf_mat.baseColor = m.albedo;
	brdf_mat.metalness = m.metalness;
	brdf_mat.roughness = m.roughness;
	brdf_mat.dielectricF0 = specular_to_f0(m.specular);
	brdf_mat.emissive = m.emissive;
	brdf_mat.transmissivness = 0.0;
	brdf_mat.opacity = 1.0;
	// A thin surface's diffuse transmission (D103), RTXPT's split: the diffuse reflection keeps
	// what the transmission's strongest channel leaves; the transmission's albedo is BACKLIGHT times
	// the diffuse albedo, as the raster weighs it.
	brdf_mat.diffuseTransmission = m.diffuse_transmission;
	float diffuse_transmit_max = max(m.diffuse_transmission.r, max(m.diffuse_transmission.g, m.diffuse_transmission.b));
	if (diffuse_transmit_max > 0.0) {
		brdf_mat.baseColor *= 1.0 - diffuse_transmit_max;
	}
	// VAL: an anisotropic specular lobe (D118 MM1) in an orthonormal frame about the normal every BRDF call below takes:
	// its tangent off N by Gram-Schmidt, any vector perpendicular to N where the tangent lies along it.
	brdf_mat.anisotropy = m.anisotropy;
	brdf_mat.anisotropyT = vec3(0.0);
	if (m.anisotropy != 0.0) {
		vec3 t = m.anisotropy_tangent - N * dot(N, m.anisotropy_tangent);
		float t_len = length(t);
		brdf_mat.anisotropyT = t_len > 1e-4 ? t / t_len : normalize(abs(N.x) > abs(N.z) ? vec3(-N.y, N.x, 0.0) : vec3(0.0, -N.z, N.y));
	}
	// VAL: Godot's clearcoat (D118 MM2): its strength, its alpha as the raster maps CLEARCOAT_ROUGHNESS, its normal
	// the vertex normal, which the shading normal's clamp does not move.
	// A coat seen from below its normal (a silhouette of interpolated normals) is absent, so its pick, its terms,
	// the base's attenuation and the guides agree.
	brdf_mat.clearcoatN = m.clearcoat > 0.0 ? normalize(m.clearcoat_normal) : N;
	brdf_mat.clearcoat = dot(brdf_mat.clearcoatN, V) > 0.0 ? m.clearcoat : 0.0;
	brdf_mat.clearcoatAlpha = mix(0.001, 0.1, m.clearcoat_roughness);
	// VAL: Godot's rim as a sheen layer (D118 MM3), coloured as the raster's rim term ends up: RIM times
	// mix(1, albedo, RIM_TINT), times the albedo and 1 - metallic its diffuse light takes; r the surface's roughness.
	// Seen from below N (a silhouette of interpolated normals) the sheen is absent, as the coat is from below its normal,
	// so its terms, the base's attenuation and the guides agree.
	brdf_mat.sheenColor = vec3(0.0);
	brdf_mat.sheenRoughness = clamp(m.roughness, 0.1, 1.0); // Below 0.1 the albedo fit leans on its r = 0 row and under-states the lobe.
	// VAL: the diffuse lobe's share that walks under the surface (D120); constant 0 while the walk is not compiled in, and
	// outside a walking group (rt_subsurface_walk_group). A back face (an open mesh seen from inside) never walks, so it
	// keeps the whole diffuse lobe; nor does a strand (D126), whose instance the walk's rays never see.
	brdf_mat.subsurface = 0.0;
#ifdef RT_SUBSURFACE_COMPILED
	brdf_mat.subsurface = (h.is_front_face && rt_subsurface_walk_group() && m.subsurface > 0.0 && (geometries[h.geometry_idx].flags & FLAG_LSS) == 0u) ? m.subsurface : 0.0;
#endif
	if (m.rim > 0.0 && dot(N, V) > 0.0) {
		brdf_mat.sheenColor = clamp(m.rim * m.albedo * mix(vec3(1.0), m.albedo, m.rim_tint) * (1.0 - m.metalness), vec3(0.0), vec3(1.0));
	}
	// VAL: hair (D126 H2): a hair material on a strand shades with the hair BCSDF alone, in the fibre's frame: N the
	// strand's normal (the shading normal is the geometry normal there); Chiang's tangent the axis made orthogonal to it,
	// the far field's the axis itself (HitData.strand_axis, which a surface record carries too).
	brdf_mat.hair = m.hair;
	brdf_mat.hairChiang = get_rt_param(RT_PARAM_HAIR_BCSDF) > 0.5;
	brdf_mat.hairT = vec3(0.0);
	brdf_mat.hairAxis = vec3(0.0);
	brdf_mat.hairAbsorption = vec3(0.0);
	brdf_mat.hairIor = m.hair_ior;
	brdf_mat.hairTilt = m.hair_tilt;
	brdf_mat.hairInset = m.hair ? h.strand_inset : 0.0;
	if (m.hair) {
		vec3 t = h.strand_axis - N * dot(N, h.strand_axis);
		float t_len = length(t);
		brdf_mat.hairT = t_len > 1e-4 ? t / t_len : normalize(abs(N.x) > abs(N.z) ? vec3(-N.y, N.x, 0.0) : vec3(0.0, -N.z, N.y));
		brdf_mat.hairAxis = t_len > 1e-4 ? h.strand_axis : brdf_mat.hairT;
		brdf_mat.hairAbsorption = hairAbsorptionCoefficient(m.hair_color, m.albedo, m.hair_melanin, m.hair_redness, m.roughness);
	}

	// Transmission: the surface is a dielectric interface. TRANSMISSION is the
	// fraction (per channel) of the unreflected light that continues through
	// it, IOR gives the refracted direction and the exact Fresnel of the
	// interface (total internal reflection from inside), which replaces the
	// SPECULAR-derived F0 so the reflection and the refraction agree. The
	// diffuse lobe keeps the rest. The transmitted lobe is chosen after NEE
	// with its own probability; opaque materials never draw for it, so their
	// random sequence is unchanged.
	float transmit_max = max(m.transmission.r, max(m.transmission.g, m.transmission.b));
	float transmit_p = 0.0;
	float refract_eta = 1.0;
	if (transmit_max > 0.0) {
		refract_eta = h.is_front_face ? (1.0 / m.ior) : m.ior;
		float fresnel = fresnel_dielectric(NdotV, refract_eta);
		brdf_mat.dielectricF0 = schlick_f0_for_fresnel(fresnel, NdotV);
		brdf_mat.baseColor *= (vec3(1.0) - m.transmission);
		transmit_p = (1.0 - fresnel) * transmit_max;
	}

	vec3 specularF0 = baseColorToSpecularF0(brdf_mat.baseColor, brdf_mat.metalness, brdf_mat.dielectricF0);
	if (hasClearcoat(brdf_mat)) {
		specularF0 = clearcoatBaseF0(specularF0, brdf_mat.clearcoat); // As prepareBRDFData remaps it (D118 MM2).
	}
	vec3 diffuseReflectance = baseColorToDiffuseReflectance(brdf_mat.baseColor, brdf_mat.metalness);

	// =================================================================
	// DLSS Ray Reconstruction output (primary ray, sample 0 only)
	// =================================================================
#ifdef DLSS_RR_ENABLED
	if (get_total_bounces(ps.packed_bounces_flags) == 0u && is_sample_zero(ps.packed_bounces_flags)) {
		ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);

		// The transmitted radiance is demodulated as diffuse: the guide carries the transmitted fraction on top of the diffuse remainder.
		// A thin surface's diffuse transmission (D103) is diffuse light too, as RTXPT's estimate adds
		// it; T already carries (1 - metallic).
		vec3 diffuse_albedo = DLSSRR_computeDiffuseAlbedo(brdf_mat.baseColor + m.transmission, m.metalness) + m.diffuse_transmission;
		vec3 specular_albedo = DLSSRR_computeSpecularAlbedo(m.albedo, m.metalness, brdf_mat.dielectricF0, m.roughness, NdotV);
		vec3 guide_N = N;
		float guide_roughness = m.roughness;
		// VAL: under a sheen (D118 MM3) the base's albedos are what the sheen passes at V (its normalised attenuation
		// integrated over L), and the sheen, a broad lobe where the raster's rim is diffuse light, adds its albedo to the
		// diffuse guide; the coat below covers both.
		float sheen_V = 1.0;
		if (hasSheen(brdf_mat)) {
			sheen_V = sheenAttenuation(brdf_mat, N, V);
			specular_albedo *= sheen_V;
			diffuse_albedo = diffuse_albedo * sheen_V + brdf_mat.sheenColor * sheenAlbedo(brdf_mat.sheenRoughness, NdotV);
		}
		// VAL: under a coat (D118 MM2) the specular albedo is the coat's plus the base's it passes, the diffuse what
		// the coat passes; the normal, roughness and hit distance follow whichever specular reflects more, so a rough
		// coat does not hide a sharp metal under it.
		if (hasClearcoat(brdf_mat)) {
			float coat_V = clearcoatReflectance(brdf_mat, V);
			float coat_roughness = sqrt(brdf_mat.clearcoatAlpha);
			vec3 coat_albedo = brdf_mat.clearcoat * DLSSRR_envBRDFApprox(vec3(0.04), coat_roughness, max(dot(brdf_mat.clearcoatN, V), 0.0001));
			vec3 base_albedo = (1.0 - coat_V) * sheen_V * DLSSRR_envBRDFApprox(clearcoatBaseF0(baseColorToSpecularF0(m.albedo, m.metalness, brdf_mat.dielectricF0), brdf_mat.clearcoat), m.roughness, NdotV);
			specular_albedo = coat_albedo + base_albedo;
			diffuse_albedo *= 1.0 - coat_V;
			if (luminance(coat_albedo) > luminance(base_albedo)) {
				guide_N = brdf_mat.clearcoatN;
				guide_roughness = coat_roughness;
			}
		}
		// VAL: a strand (D126 H2) is guided by the hair's own colour and cuticle (docs/r6-strand-hair.md, "RR guides"): the
		// colour its absorption maps back to as the diffuse albedo (the authored colour in the colour mode, as RTXCR's sample
		// guides with; in the melanin mode the albedo is only a dye tint), its Fresnel at the fibre's IOR as the specular
		// (where RTXCR's sample takes the mesh material's F0), the fibre's normal and roughness.
		if (hasHair(brdf_mat)) {
			diffuse_albedo = hairGuideColor(brdf_mat.hairAbsorption, m.roughness);
			float hair_f0 = (brdf_mat.hairIor - 1.0) / (brdf_mat.hairIor + 1.0);
			specular_albedo = DLSSRR_computeSpecularAlbedo(vec3(0.0), 0.0, hair_f0 * hair_f0, m.roughness, NdotV);
			guide_N = N;
			guide_roughness = m.roughness;
		}
		imageStore(dlss_rr_diffuse_albedo, pixel, vec4(DLSSRR_encodeDiffuseAlbedo(diffuse_albedo), 1.0));

		imageStore(dlss_rr_specular_albedo, pixel, vec4(clamp(specular_albedo, vec3(0.04), vec3(1.0)), 1.0)); // match UNORM8 like before - fixes some issues with garbling..

		imageStore(dlss_rr_normal_roughness, pixel, vec4(guide_N, guide_roughness));

		// Specular hit distance via inline ray query (only for smooth surfaces).
		float spec_hit_dist = -1.0;
		if (guide_roughness < MAX_DENOISER_SPECULAR_HIT_THRESHOLD) {
			// The ray leaves the surface as the bounce ray does: a direction under the geometry plane
			// is mirrored above it, and the origin is offset along the normal. The reflected direction
			// has almost no normal component at grazing angles, so offsetting along it left the origin
			// on the surface and a cull-disabled one hit itself.
			vec3 spec_dir = mirrorDirectionAcrossPlane(reflect(-V, guide_N), h.geometry_normal);
			// A strand's starts inside the fibre (D126 H2).
			vec3 spec_origin = brdf_mat.hairInset > 0.0 ? h.hit_pos - h.geometry_normal * brdf_mat.hairInset : offset_ray_origin(h.hit_pos, h.geometry_normal, hit_t);

			rayQueryEXT spec_rq;
			// The closest hit: terminating on the first one would record whichever surface
			// traversal order reaches first, not the reflected one.
			rayQueryInitializeEXT(spec_rq, tlas, RT_RAY_FLAGS,
					0xFF, spec_origin, hasHair(brdf_mat) ? RT_HAIR_TMIN : 0.001, spec_dir, 10000.0);
			while (rayQueryProceedEXT(spec_rq)) {
				if (rayQueryGetIntersectionTypeEXT(spec_rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
					if (ray_query_alpha_test(spec_rq)) {
						rayQueryConfirmIntersectionEXT(spec_rq);
					}
				}
			}
			if (rayQueryGetIntersectionTypeEXT(spec_rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
				spec_hit_dist = rayQueryGetIntersectionTEXT(spec_rq, true);
			}
		}
		imageStore(dlss_rr_specular_hit_dist, pixel, vec4(spec_hit_dist));
	}
#endif

	// =================================================================
	// NEE: Next Event Estimation (direct light sampling)
	// =================================================================

	// This bounce's block of Sobol dimensions (D52): the light sampling below draws a variable
	// count, so the scatter's draws are pinned again after it. The analytic path draws the two
	// strategy choices (the sky's, then the emissive list's), a pick and an acceptance per
	// reservoir slot and the cone pair, which must stay below the scatter's offset or the
	// scatter would redraw the light's numbers; the light tree (D55) draws one for its reservoir,
	// at most two per candidate (the top split, then its place in the cut, D56) and the leaf's
	// pair, plus the sky's.
#if 2 * RT_LIGHT_RESERVOIR_SIZE + 5 > 40 || 2 * RT_LIGHT_TREE_MAX_CANDIDATES + 4 > 40
#error RT_SOBOL_DIM_SCATTER must exceed the light sampling paths' draw counts
#endif
	rng_set_dimension(ps.rng_state, get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE);
	shade_direct_light(h.hit_pos, h.geometry_normal, hit_t, N, V, brdf_mat, diffuse_transmit_max, get_diffuse_bounces(ps.packed_bounces_flags), rt_emissive_count, in_medium, 0u, ps);

	// =================================================================
	// Transmission: refract through the surface with probability transmit_p
	// =================================================================
	rng_set_dimension(ps.rng_state, get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_SCATTER);
	if (transmit_p > 0.0) {
		if (rand(ps.rng_state) < transmit_p) {
			vec3 refracted = refract(-V, N, refract_eta);
			if (dot(refracted, refracted) < 0.5) {
				// No refracted direction (total internal reflection leaves transmit_p at zero, so only a degenerate normal gets here).
				ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
				return;
			}
			// A shading normal tilted far from the geometry can bend the direction back above the surface; mirror it under the plane.
			float above = dot(refracted, h.geometry_normal);
			if (above > 0.0) {
				refracted = normalize(refracted - 2.0 * above * h.geometry_normal);
			}
			ps.throughput *= m.transmission / transmit_max;
			if (hasClearcoat(brdf_mat)) {
				ps.throughput *= clearcoatAttenuation(brdf_mat, V) * clearcoatAttenuation(brdf_mat, refracted); // Through the coat (D118 MM2).
			}
			if (hasSheen(brdf_mat)) {
				// And the sheen (D118 MM3), on V's side alone: what it does not reflect enters. Kulla and Conty's
				// normalised pair holds over a diffuse base only; over this delta lobe it would add up to 18 %.
				ps.throughput *= sheenAttenuation(brdf_mat, N, V);
			}
			ps.packed_bounces_flags = inc_total_bounce(ps.packed_bounces_flags);
			// The medium behind the interface (D49): the surface's ABSORPTION and SCATTERING,
			// which a front face sets and a back face writes as zero to clear (no nesting: the
			// last interface crossed decides). A reflection or a scatter leaves the state as it was.
			bool enters_medium = any(greaterThan(m.absorption, vec3(0.0)));
			ps.packed_bounces_flags = set_in_medium(ps.packed_bounces_flags, enters_medium);
			if (enters_medium) {
				path_set_medium(ps, m.absorption, m.scattering * medium_irradiance(N), h.hit_pos.y);
			} else {
				path_clear_medium(ps);
			}
			ps.hit_t = hit_t;
			ps.offset_normal = -h.geometry_normal;
			ps.next_ray_dir = refracted;
			ps.bsdf_pdf = RT_BSDF_PDF_DELTA;
			return;
		}
		ps.throughput /= (1.0 - transmit_p);
	}

	// =================================================================
	// BRDF importance sampling for next bounce
	// =================================================================
	// VAL: a coat (D118 MM2) is chosen before the base's lobes, with the probability brdf_pdf gives it, on a
	// dimension of its own so the base's draws keep theirs; a surface without one draws nothing here.
	float coat_p = clearcoatProbability(brdf_mat, N, V, specularF0, diffuseReflectance);
	bool coat_sampled = coat_p >= 1.0;
	if (coat_p > 0.0 && coat_p < 1.0) {
		rng_set_dimension(ps.rng_state, get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_CLEARCOAT);
		coat_sampled = rand(ps.rng_state) < coat_p;
		rng_set_dimension(ps.rng_state, get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_SCATTER + (transmit_p > 0.0 ? 1u : 0u));
		ps.throughput /= coat_sampled ? coat_p : (1.0 - coat_p);
	}
	// VAL: a sheen (D118 MM3) is chosen next, under the coat and over the base's lobes, with the probability brdf_pdf
	// gives it, on a dimension of its own; a surface without one draws nothing here.
	float sheen_p = coat_sampled ? 0.0 : sheenProbability(brdf_mat, N, V, specularF0, diffuseReflectance);
	bool sheen_sampled = sheen_p >= 1.0;
	if (sheen_p > 0.0 && sheen_p < 1.0) {
		rng_set_dimension(ps.rng_state, get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_SHEEN);
		sheen_sampled = rand(ps.rng_state) < sheen_p;
		rng_set_dimension(ps.rng_state, get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_SCATTER + (transmit_p > 0.0 ? 1u : 0u));
		ps.throughput /= sheen_sampled ? sheen_p : (1.0 - sheen_p);
	}

	// A thin surface's diffuse transmission (D103) is chosen first, with the probability brdf_pdf
	// gives it, on a dimension of its own so the lobe and direction draws keep theirs; a surface
	// without it draws nothing here.
	float diffuse_transmit_p = (diffuse_transmit_max > 0.0 && !coat_sampled && !sheen_sampled) ? diffuseTransmissionProbability(specularF0, diffuseReflectance, m.diffuse_transmission) : 0.0;
	bool diffuse_transmitted = false;
	if (diffuse_transmit_p > 0.0) {
		rng_set_dimension(ps.rng_state, get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_DIFFUSE_TRANSMISSION);
		diffuse_transmitted = rand(ps.rng_state) < diffuse_transmit_p;
		rng_set_dimension(ps.rng_state, get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_SCATTER + (transmit_p > 0.0 ? 1u : 0u));
		ps.throughput /= diffuse_transmitted ? diffuse_transmit_p : (1.0 - diffuse_transmit_p);
	}

	int brdfType = DIFFUSE_TYPE;
	// The sheen is a broad lobe, a diffuse bounce for the bounce budget (D74), as the raster's rim is diffuse light. Hair
	// (D126 H2) is glossy transport, a specular bounce, so the diffuse budget does not cut its multiple scattering.
	float brdfProbability = (coat_sampled || hasHair(brdf_mat)) ? 1.0 : ((diffuse_transmitted || sheen_sampled) ? 0.0 : brdfSpecularProbability(specularF0, diffuseReflectance));
	if (brdfProbability >= 1.0) {
		brdfType = SPECULAR_TYPE;
	} else if (brdfProbability <= 0.0) {
		brdfType = DIFFUSE_TYPE;
	} else {
		if (rand(ps.rng_state) < brdfProbability) {
			brdfType = SPECULAR_TYPE;
			ps.throughput /= brdfProbability;
		} else {
			brdfType = DIFFUSE_TYPE;
			ps.throughput /= (1.0 - brdfProbability);
		}
	}

	vec2 u = rand2(ps.rng_state);
	vec3 next_dir;
	vec3 brdf_weight;
	// The hair sampler's density of its direction (D126 H2), the one brdf_pdf would evaluate again below.
	float hair_pdf = 0.0;
	if (hasHair(brdf_mat)) {
		// VAL: the hair BCSDF's own sample (D126 H2) on a block of dimensions of its own: its direction leaves on either
		// side of N, so nothing is mirrored; a rejected direction ends the path, as in RTXCR's sample.
		rng_set_dimension(ps.rng_state, get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_HAIR);
		vec3 hair_u = vec3(rand2(ps.rng_state), rand(ps.rng_state));
		vec3 hair_w = vec3(rand2(ps.rng_state), rand(ps.rng_state));
		if (!sampleHair(hair_u, hair_w, N, V, brdf_mat, next_dir, brdf_weight, hair_pdf)) {
			ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
			return;
		}
	} else if (diffuse_transmitted) {
		// Lambertian about -N: the cosine cancels its density, leaving the albedo. A direction the
		// shading normal tilts back above the geometry plane is mirrored under it.
		next_dir = rotatePoint(getRotationFromZAxis(-N), sampleHemisphere(u));
		float above = dot(next_dir, h.geometry_normal);
		if (above > 0.0) {
			next_dir = normalize(next_dir - 2.0 * above * h.geometry_normal);
		}
		brdf_weight = m.diffuse_transmission;
	} else if (coat_sampled ? !sampleClearcoat(u, h.geometry_normal, V, brdf_mat, next_dir, brdf_weight) : (sheen_sampled ? !sampleSheen(u, N, h.geometry_normal, V, brdf_mat, next_dir, brdf_weight) : !evalIndirectCombinedBRDF(u, N, h.geometry_normal, V, brdf_mat, brdfType, next_dir, brdf_weight, vec4(0.0)))) {
		// Two failure modes:
		//   1) Sample weight is zero (no contribution). Terminate.
		//   2) Sampled direction is below the geometry plane. Recover by
		//      mirroring across the geometry plane (see brdf_inc.glsl). The
		//      mirror is gated by RT_BELOW_HEMISPHERE_RECOVERY_ENABLED; when
		//      disabled, recovery returns false and we terminate the path.
		vec3 recovered_dir;
		if (luminance(brdf_weight) == 0.0 ||
				!recoverBelowHemisphereSample(next_dir, h.geometry_normal, recovered_dir)) {
			ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
			return;
		}
		next_dir = recovered_dir;
	}
	// VAL: the base under a sheen (D118 MM3) passes what the sheen does not reflect, in both directions.
	if (!coat_sampled && !sheen_sampled && hasSheen(brdf_mat)) {
		brdf_weight *= sheenBaseAttenuation(brdf_mat, N, V, next_dir);
	}
	// VAL: the sheen and the base under a coat (D118 MM2) pass what the coat does not reflect, in both directions.
	if (!coat_sampled && hasClearcoat(brdf_mat)) {
		brdf_weight *= clearcoatAttenuation(brdf_mat, V) * clearcoatAttenuation(brdf_mat, next_dir);
	}
	// A base lobe's direction below N under a coat, on a surface without transmission (D118 MM2).
	const bool base_below_coat = !coat_sampled && diffuse_transmit_max <= 0.0 && hasClearcoat(brdf_mat) && dot(N, next_dir) <= 0.0;
	// A mirror: a zero-roughness specular lobe, or a coat below CLEARCOAT_DELTA_ALPHA (D118 MM2).
	const bool delta_lobe = coat_sampled ? brdf_mat.clearcoatAlpha < CLEARCOAT_DELTA_ALPHA : (brdfType == SPECULAR_TYPE && !hasHair(brdf_mat) && brdf_mat.roughness * brdf_mat.roughness <= 0.0);

	ps.throughput *= brdf_weight;

#ifdef RT_SUBSURFACE_COMPILED
	// VAL: subsurface scattering (D120). A diffuse pick walks under the surface with the subsurface share s. The direct
	// lobe's NEE and density carry 1 - s and the walk s, so the choice divides neither weight. The walk replaces the
	// diffuse bounce: its exit samples a light with a white Lambert and the path leaves by a cosine direction about the
	// exit's normal, counted as the diffuse bounce.
	bool walked = false;
	vec3 exit_position = h.hit_pos;
	vec3 exit_normal = h.geometry_normal;
	vec3 exit_shading_normal = N;
	MaterialProperties exit_mat = brdf_mat;
	// Only from outside: a back face (seen from inside an open mesh) keeps the plain diffuse bounce.
	if (brdfType == DIFFUSE_TYPE && !coat_sampled && !sheen_sampled && !diffuse_transmitted && brdf_mat.subsurface > 0.0 && h.is_front_face) {
		rng_set_dimension(ps.rng_state, get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_SUBSURFACE);
		if (rand(ps.rng_state) < brdf_mat.subsurface) {
			float exit_t;
			uint events, skips;
			uint seed = pcg_hash(ps.rng_state ^ pcg_hash(rt_pixel_seed() + get_total_bounces(ps.packed_bounces_flags) * 0x9e3779b9u));
			uint status = subsurface_walk(h, m.albedo, m.subsurface_skin, N, hit_t, seed, ps.throughput, exit_position, exit_normal, exit_shading_normal, exit_t, events, skips);
#ifdef RT_SUBSURFACE_STATS
			if (((gl_LaunchIDEXT.x | gl_LaunchIDEXT.y) & 7u) == 0u) {
				atomicAdd(subsurface_stats.words[0], 1u);
				atomicAdd(subsurface_stats.words[1], events);
				atomicMax(subsurface_stats.words[2], events);
				atomicAdd(subsurface_stats.words[3u + status], 1u);
				atomicAdd(subsurface_stats.words[6], skips);
			}
#endif
			if (status != RT_SUBSURFACE_EXITED) {
				ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
				return;
			}
			walked = true;
			exit_mat.baseColor = vec3(1.0);
			exit_mat.metalness = 0.0;
			exit_mat.roughness = 1.0;
			exit_mat.dielectricF0 = 0.0;
			exit_mat.emissive = vec3(0.0);
			exit_mat.transmissivness = 0.0;
			exit_mat.opacity = 1.0;
			exit_mat.diffuseTransmission = vec3(0.0);
			exit_mat.anisotropy = 0.0;
			exit_mat.anisotropyT = vec3(0.0);
			exit_mat.clearcoat = 0.0;
			exit_mat.clearcoatAlpha = 0.001;
			exit_mat.clearcoatN = exit_shading_normal;
			exit_mat.sheenColor = vec3(0.0);
			exit_mat.sheenRoughness = 1.0;
			exit_mat.subsurface = 0.0;
			exit_mat.hair = false;
			// The exit's light sample about its shading normal (D120), viewed along it (a Lambert is the same from every side),
			// with the interpolated normal as the receiver the densities read and the origin's offset, as at a hit; on its own
			// block. Past
			// the walk the path has scattered, so its light is indirect for the clamp (D53).
			ps.packed_bounces_flags = set_scattered(ps.packed_bounces_flags);
			rng_set_dimension(ps.rng_state, RT_SOBOL_DIM_SUBSURFACE_EXIT_BASE + get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE);
			shade_direct_light(exit_position, exit_normal, exit_t, exit_shading_normal, exit_shading_normal, exit_mat, 0.0, get_diffuse_bounces(ps.packed_bounces_flags), rt_emissive_count, in_medium, 1u, ps);
			// And its direction: the cosine cancels its density, leaving a weight of 1.
			rng_set_dimension(ps.rng_state, RT_SOBOL_DIM_SUBSURFACE_EXIT_BASE + get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_SCATTER);
			next_dir = rotatePoint(getRotationFromZAxis(exit_shading_normal), sampleHemisphere(rand2(ps.rng_state)));
			// A direction the shading normal tilts below the exit's plane is mirrored above it, as the hit's recovery does.
			float exit_below = dot(next_dir, exit_normal);
			if (exit_below < 0.0) {
				next_dir = normalize(next_dir - 2.0 * exit_below * exit_normal);
			}
		}
	}
#endif

	// Russian roulette (D73): once an earlier vertex has scattered (D53's flag: a non-delta lobe,
	// so a surface seen through glass, water or a mirror is exempt as a directly seen one is), the
	// path continues with the probability of its throughput's brightest channel and, when it does,
	// carries the weight 1 / p, so the estimator is unchanged in expectation while a path that has
	// gone dim stops paying for traversal. Direct light at this vertex is already counted above, so
	// only the next segment is at stake. A path at full throughput draws nothing; the draw takes
	// its own dimension so this bounce's other draws are unchanged, and the negated comparison
	// ends a path whose throughput is not a number.
	if ((RT_FLAGS & RT_FLAG_RUSSIAN_ROULETTE) != 0u && is_scattered(ps.packed_bounces_flags)) {
		float survive = max(ps.throughput.r, max(ps.throughput.g, ps.throughput.b));
		if (!(survive >= 1.0)) {
			rng_set_dimension(ps.rng_state, get_total_bounces(ps.packed_bounces_flags) * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_ROULETTE);
			if (!(rand(ps.rng_state) < survive)) {
				ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
				return;
			}
			ps.throughput /= survive;
		}
	}

	if (brdfType == DIFFUSE_TYPE) {
		ps.packed_bounces_flags = inc_diffuse_bounce(ps.packed_bounces_flags);
	} else {
		ps.packed_bounces_flags = inc_total_bounce(ps.packed_bounces_flags);
	}
	// Past a non-delta lobe the path carries sampling variance (D53); a mirror does not.
	if (!delta_lobe) {
		ps.packed_bounces_flags = set_scattered(ps.packed_bounces_flags);
	}

	// The density of the direction just chosen, for the next hit's emission weight (D50): a
	// zero-roughness specular lobe is a delta light sampling cannot hit, so it keeps the whole
	// weight; every other direction is weighed with the density brdf_pdf assigns it, which is
	// the same function light sampling evaluates for its own directions. The miss weighs the sky
	// with it too when the sky is a light (D54).
	vec3 pdf_N = N;
	vec3 pdf_V = V;
	MaterialProperties pdf_mat = brdf_mat;
	bool pdf_unowned = (diffuse_transmit_p > 0.0 && diffuse_transmitted != (dot(N, next_dir) < 0.0)) || base_below_coat || (sheen_sampled && dot(N, next_dir) <= 0.0);
#ifdef RT_SUBSURFACE_COMPILED
	if (walked) {
		// After a walk the direction left the exit by its Lambert (D120).
		pdf_N = exit_shading_normal;
		pdf_V = exit_shading_normal;
		pdf_mat = exit_mat;
		pdf_unowned = false;
	}
#endif
	if ((rt_emissive_count == 0u && !sky_light_enabled()) || delta_lobe) {
		// With no emitter listed and no sky light nothing reads the density, so both off cost no ALU here.
		ps.bsdf_pdf = RT_BSDF_PDF_DELTA;
	} else {
		float pdf = hasHair(brdf_mat) ? hair_pdf : brdf_pdf(pdf_N, next_dir, pdf_V, pdf_mat);
		// A direction the density cannot own (the below-hemisphere recovery; on a thin surface, a
		// direction on the side of N the chosen lobe does not sample: a transmitted one mirrored back
		// above N, a reflected one in the wedge between N and the geometry plane) is one light
		// sampling does not produce this way, so the BSDF strategy keeps the whole weight. So is a base lobe's
		// direction below N under a coat (D118 MM2): light sampling evaluates only the coat there. So is a sheen
		// sample the recovery mirrored below N (D118 MM3), which the sheen's density does not cover.
		if (pdf_unowned) {
			pdf = 0.0;
		}
		ps.bsdf_pdf = pdf > 0.0 ? pdf : RT_BSDF_PDF_DELTA;
	}

	// A non-delta lobe widens the ray cone by the density of the direction it drew (D116); a mirror,
	// like a refraction above, keeps the spread. Only a cone the hit shaders read is widened.
	if (rt_texture_lod_enabled() && !delta_lobe) {
		float pdf = ps.bsdf_pdf;
		if (pdf == RT_BSDF_PDF_DELTA) {
			pdf = hasHair(brdf_mat) ? hair_pdf : brdf_pdf(pdf_N, next_dir, pdf_V, pdf_mat);
			if (pdf_unowned) {
				pdf = 0.0;
			}
		}
		ps.cone_spread = rt_cone_scatter_spread(ps.cone_spread, pdf);
	}

	// Hand the next ray back to raygen. PATH_TERMINATED_FLAG stays clear so
	// the raygen loop continues with the reconstructed origin and next_ray_dir.
	ps.hit_t = hit_t;
	ps.offset_normal = h.geometry_normal;
	// A strand's next ray starts inside it (D126 H2) whichever way it leaves; one without a radius leaves from the back.
	ps.packed_bounces_flags = set_back_offset(ps.packed_bounces_flags, diffuse_transmitted || (hasHair(brdf_mat) && brdf_mat.hairInset == 0.0 && dot(h.geometry_normal, next_dir) < 0.0));
	if (hasHair(brdf_mat)) {
		ps.packed_bounces_flags |= HAIR_RAY_FLAG | rt_strand_inset_code(brdf_mat.hairInset);
	}
#ifdef RT_SUBSURFACE_COMPILED
	if (walked) {
		// The next ray leaves from where the walk left the surface (D120), offset along its exit normal.
		ps.offset_normal = exit_normal;
		ps.exit_position = exit_position;
		ps.packed_bounces_flags |= SUBSURFACE_EXIT_FLAG;
	}
#endif
	ps.next_ray_dir = next_dir;
}
