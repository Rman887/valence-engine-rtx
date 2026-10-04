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

// ============================================================================
// HIT DATA
// ============================================================================

struct HitData {
	vec3 hit_pos;
	vec3 geometry_normal; // World space, flipped for back-face hits.
	vec3 tangent; // World space.
	vec3 bitangent; // World space.
	vec2 uv; // Raw UV (no material scale/offset applied).
	vec4 color; // Vertex color (white if not present).
	bool is_front_face;
	uint geometry_idx;
	uint primitive_id; // The triangle hit; 0 for a G-buffer surface, which never weighs its emission (D50).
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
	// onto the prefiltered roughness array layers.
	return radiance_octmap_sample(cube_uv, mip_level);
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
		bool is_front_face) {
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
	vec3 transmittance = exp(-ps.medium_absorption * segment_length);
	ps.radiance += ps.throughput * ps.medium_inscatter * (vec3(1.0) - transmittance);
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

/// ray_dir and hit_t describe the ray that reached h.hit_pos: the hit stage's
/// built-ins, or the raygen's own primary ray for a G-buffer surface.
void shade_and_bounce(HitData h, MaterialResult m, vec3 ray_dir, float hit_t) {
	PathState ps = path_unpack(payload);
	// Every exit below packs the payload, and the raygen reads hit_t for the camera segment's
	// length before it checks for termination, so the distance is recorded once, here. The
	// previous vertex's back offset (D103) is spent: only a transmission below sets it again.
	ps.hit_t = hit_t;
	ps.packed_bounces_flags = set_back_offset(ps.packed_bounces_flags, false);

	vec3 V = -ray_dir;

	// Clamp shading normal toward geometry at grazing view angles to keep BRDF above the geometry hemisphere.
	vec3 N = clampShadingNormal(m.normal, h.geometry_normal, V, RT_SHADING_NORMAL_CLAMP_THRESHOLD);
	float NdotV = max(dot(N, V), 0.0001);

	uint total_bounces = get_total_bounces(ps.packed_bounces_flags);
	uint diffuse_bounces = get_diffuse_bounces(ps.packed_bounces_flags);

	// Media on this segment (D37): the camera segment is composed in the raygen over the whole
	// path; every later segment is charged here, over its own length. Inside a medium (D49) the
	// segment takes the payload's medium instead of the Environment's fog, the camera segment too
	// (the raygen seeds the medium the camera is in and composes no fog over it).
	bool in_medium = is_in_medium(ps.packed_bounces_flags);
	if (in_medium) {
		apply_segment_medium(hit_t, ps);
	} else if (total_bounces > 0u) {
		apply_segment_fog(ray_dir, hit_t, ps.radiance, ps.throughput);
	}

	// Emissive contribution, weighed against the emissive light sampling that competes for it
	// after a scatter (D50): the camera ray and a delta lobe keep the whole weight, and so does
	// a surface the emissive list does not carry, since light sampling never reaches it.
	uint rt_emissive_count = uint(get_rt_param(RT_PARAM_EMISSIVE_COUNT));
	float emissive_weight = 1.0;
	if (total_bounces > 0u && rt_emissive_count > 0u && ps.bsdf_pdf != RT_BSDF_PDF_DELTA && any(greaterThan(m.emissive, vec3(0.0)))) {
		float pdf_light;
		if (light_tree_enabled()) {
			// The light tree's exact probability of this entry at the previous vertex (D55): the ray's
			// origin and the offset normal the payload carried from there.
			uint entry = emissive_find_entry(h.geometry_idx, rt_emissive_count);
			float p_instance = (entry < rt_emissive_count) ? light_tree_emissive_probability(entry, shade_previous_origin(h, ray_dir, hit_t), ps.offset_normal, is_light_tree_indirect(ps.packed_bounces_flags)) : 0.0;
			pdf_light = (entry < rt_emissive_count) ? emissive_hit_pdf_entry(entry, p_instance, h.primitive_id, ray_dir, hit_t, h.is_front_face) : 0.0;
		} else {
			// The sky's share of the light samples (D54), 0 without a sky light; the surface lights
			// split the rest, so their densities carry the remainder on both sides of the weight. It is
			// the previous vertex's share, for its receiver as the payload carried it (D65).
			float p_sky_previous = sky_light_probability(ps.offset_normal);
			pdf_light = (1.0 - p_sky_previous) * get_rt_param(RT_PARAM_EMISSIVE_PROBABILITY) * emissive_hit_pdf(h.geometry_idx, h.primitive_id, ray_dir, hit_t, h.is_front_face, rt_emissive_count);
		}
		emissive_weight = emissive_mis_weight(ps.bsdf_pdf, pdf_light);
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
		path_pack(payload, ps);
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
	vec3 diffuseReflectance = baseColorToDiffuseReflectance(brdf_mat.baseColor, brdf_mat.metalness);

	// =================================================================
	// DLSS Ray Reconstruction output (primary ray, sample 0 only)
	// =================================================================
#ifdef DLSS_RR_ENABLED
	if (total_bounces == 0u && is_sample_zero(ps.packed_bounces_flags)) {
		ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);

		// The transmitted radiance is demodulated as diffuse: the guide carries the transmitted fraction on top of the diffuse remainder.
		// A thin surface's diffuse transmission (D103) is diffuse light too, as RTXPT's estimate adds
		// it; T already carries (1 - metallic).
		vec3 diffuse_albedo = DLSSRR_encodeDiffuseAlbedo(DLSSRR_computeDiffuseAlbedo(brdf_mat.baseColor + m.transmission, m.metalness) + m.diffuse_transmission);
		imageStore(dlss_rr_diffuse_albedo, pixel, vec4(diffuse_albedo, 1.0));

		vec3 specular_albedo = DLSSRR_computeSpecularAlbedo(m.albedo, m.metalness, brdf_mat.dielectricF0, m.roughness, NdotV);
		imageStore(dlss_rr_specular_albedo, pixel, vec4(clamp(specular_albedo, vec3(0.04), vec3(1.0)), 1.0)); // match UNORM8 like before - fixes some issues with garbling..

		imageStore(dlss_rr_normal_roughness, pixel, vec4(N, m.roughness));

		// Specular hit distance via inline ray query (only for smooth surfaces).
		float spec_hit_dist = -1.0;
		if (m.roughness < MAX_DENOISER_SPECULAR_HIT_THRESHOLD) {
			// The ray leaves the surface as the bounce ray does: a direction under the geometry plane
			// is mirrored above it, and the origin is offset along the normal. The reflected direction
			// has almost no normal component at grazing angles, so offsetting along it left the origin
			// on the surface and a cull-disabled one hit itself.
			vec3 spec_dir = mirrorDirectionAcrossPlane(reflect(-V, N), h.geometry_normal);
			vec3 spec_origin = offset_ray_origin(h.hit_pos, h.geometry_normal, hit_t);

			rayQueryEXT spec_rq;
			// The closest hit: terminating on the first one would record whichever surface
			// traversal order reaches first, not the reflected one.
			rayQueryInitializeEXT(spec_rq, tlas, RT_RAY_FLAGS,
					0xFF, spec_origin, 0.001, spec_dir, 10000.0);
			while (rayQueryProceedEXT(spec_rq)) {
				if (rayQueryGetIntersectionTypeEXT(spec_rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
					if (ray_query_alpha_test(
								rayQueryGetIntersectionInstanceCustomIndexEXT(spec_rq, false),
								rt_ray_query_cluster_id(spec_rq),
								uint(rayQueryGetIntersectionPrimitiveIndexEXT(spec_rq, false)),
								rayQueryGetIntersectionBarycentricsEXT(spec_rq, false))) {
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
	path_pack(payload, ps);

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
	rng_set_dimension(ps.rng_state, total_bounces * RT_SOBOL_DIMS_PER_BOUNCE);
	uint rt_light_count = uint(get_rt_param(RT_PARAM_LIGHT_COUNT));
	// The selection's receiver: the geometry normal as the payload carries it to the next vertex,
	// so the chance-hit and miss densities there read the same function of the same normal (the
	// light tree's since D55, the sky's share by variance since D65).
	vec3 N_select = oct_to_vec3(unpackUnorm2x16(packUnorm2x16(vec3_to_oct(h.geometry_normal))) * 2.0 - 1.0);
	// The sky's share of the light samples (D54), 0 without a sky light; the surface lights
	// split the rest, so their densities carry the remainder on both sides of the weight.
	float p_sky = light_tree_enabled() ? 0.0 : sky_light_probability(N_select);
	// Under the tree (D55) the sky's share is per hit, so a frame whose only light is the sky still
	// takes its sample here (the miss weighs against it either way).
	bool tree_sky = light_tree_enabled() && sky_light_enabled() && sky_total > 0.0;
	if (rt_light_count > 0u || rt_emissive_count > 0u || p_sky > 0.0 || tree_sky) {
		vec3 hit_pos_offset = offset_ray_origin(h.hit_pos, h.geometry_normal, hit_t);
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
			have_sample = light_tree_sample_direct(hit_pos_offset, N_select, N, V, brdf_mat, ps.rng_state, is_indirect, tree_candidates, light_sample, strategy_probability);
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
		if (diffuse_transmit_max > 0.0 && dot(N, light_sample.direction) < 0.0) {
			shadow_origin = offset_ray_origin(h.hit_pos, -h.geometry_normal, hit_t);
			vec3 to_light = hit_pos_offset + light_sample.direction * light_sample.shadow_distance - shadow_origin;
			shadow_distance = length(to_light);
			shadow_dir = to_light / max(shadow_distance, 1e-6);
		}
		if (have_sample && lights_trace_shadow_ray(shadow_origin, shadow_dir, shadow_distance, RT_OMM_RAY_FLAGS(diffuse_bounces), ps.rng_state)) {
			// Evaluate BRDF (diffuse + specular separately for specular_amount control).
			vec3 brdf_diffuse, brdf_specular;
			evalCombinedBRDFSeparate(N, light_sample.direction, V, brdf_mat, brdf_diffuse, brdf_specular);
			// NdotL is already included in the BRDF value (evalLambertian/evalMicrofacet bake it in).
			vec3 direct_light = (brdf_diffuse + brdf_specular * light_sample.specular_multiplier) * light_sample.radiance * (light_sample.weight / strategy_probability);
			if (in_medium) {
				// The light's own leg through the medium (D49): to the light, or up to the medium's
				// top, the horizontal plane through the point the path entered at, whichever is
				// nearer. The shadow ray is a visibility test that does not resolve the interface, so
				// the leg is this estimate (a named limit); a light under the plane takes its distance.
				float leg = min(light_sample.leg_distance, max(ps.medium_entry_height - h.hit_pos.y, 0.0) / max(light_sample.direction.y, 0.1));
				direct_light *= exp(-ps.medium_absorption * leg);
			}
			// The clamp (D53): direct at the surface the camera sees, indirect after a scatter.
			ps.radiance += rt_clamp_contribution(ps.throughput * direct_light, get_rt_param(is_scattered(ps.packed_bounces_flags) ? RT_PARAM_CLAMP_INDIRECT : RT_PARAM_CLAMP_DIRECT));
		}
	}

	// =================================================================
	// Transmission: refract through the surface with probability transmit_p
	// =================================================================
	rng_set_dimension(ps.rng_state, total_bounces * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_SCATTER);
	if (transmit_p > 0.0) {
		if (rand(ps.rng_state) < transmit_p) {
			vec3 refracted = refract(-V, N, refract_eta);
			if (dot(refracted, refracted) < 0.5) {
				// No refracted direction (total internal reflection leaves transmit_p at zero, so only a degenerate normal gets here).
				ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
				path_pack(payload, ps);
				return;
			}
			// A shading normal tilted far from the geometry can bend the direction back above the surface; mirror it under the plane.
			float above = dot(refracted, h.geometry_normal);
			if (above > 0.0) {
				refracted = normalize(refracted - 2.0 * above * h.geometry_normal);
			}
			ps.throughput *= m.transmission / transmit_max;
			ps.packed_bounces_flags = inc_total_bounce(ps.packed_bounces_flags);
			// The medium behind the interface (D49): the surface's ABSORPTION and SCATTERING,
			// which a front face sets and a back face writes as zero to clear (no nesting: the
			// last interface crossed decides). A reflection or a scatter leaves the state as it was.
			bool enters_medium = any(greaterThan(m.absorption, vec3(0.0)));
			ps.packed_bounces_flags = set_in_medium(ps.packed_bounces_flags, enters_medium);
			if (enters_medium) {
				ps.medium_absorption = m.absorption;
				ps.medium_inscatter = m.scattering * medium_irradiance(N);
				ps.medium_entry_height = h.hit_pos.y;
			} else {
				path_clear_medium(ps);
			}
			ps.hit_t = hit_t;
			ps.offset_normal = -h.geometry_normal;
			ps.next_ray_dir = refracted;
			ps.bsdf_pdf = RT_BSDF_PDF_DELTA;
			path_pack(payload, ps);
			return;
		}
		ps.throughput /= (1.0 - transmit_p);
	}

	// =================================================================
	// BRDF importance sampling for next bounce
	// =================================================================
	// A thin surface's diffuse transmission (D103) is chosen first, with the probability brdf_pdf
	// gives it, on a dimension of its own so the lobe and direction draws keep theirs; a surface
	// without it draws nothing here.
	float diffuse_transmit_p = diffuse_transmit_max > 0.0 ? diffuseTransmissionProbability(specularF0, diffuseReflectance, m.diffuse_transmission) : 0.0;
	bool diffuse_transmitted = false;
	if (diffuse_transmit_p > 0.0) {
		rng_set_dimension(ps.rng_state, total_bounces * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_DIFFUSE_TRANSMISSION);
		diffuse_transmitted = rand(ps.rng_state) < diffuse_transmit_p;
		rng_set_dimension(ps.rng_state, total_bounces * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_SCATTER + (transmit_p > 0.0 ? 1u : 0u));
		ps.throughput /= diffuse_transmitted ? diffuse_transmit_p : (1.0 - diffuse_transmit_p);
	}

	int brdfType = DIFFUSE_TYPE;
	float brdfProbability = diffuse_transmitted ? 0.0 : brdfSpecularProbability(specularF0, diffuseReflectance);
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
	if (diffuse_transmitted) {
		// Lambertian about -N: the cosine cancels its density, leaving the albedo. A direction the
		// shading normal tilts back above the geometry plane is mirrored under it.
		next_dir = rotatePoint(getRotationFromZAxis(-N), sampleHemisphere(u));
		float above = dot(next_dir, h.geometry_normal);
		if (above > 0.0) {
			next_dir = normalize(next_dir - 2.0 * above * h.geometry_normal);
		}
		brdf_weight = m.diffuse_transmission;
	} else if (!evalIndirectCombinedBRDF(u, N, h.geometry_normal, V, brdf_mat, brdfType, next_dir, brdf_weight, vec4(0.0))) {
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
			path_pack(payload, ps);
			return;
		}
		next_dir = recovered_dir;
	}

	ps.throughput *= brdf_weight;

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
			rng_set_dimension(ps.rng_state, total_bounces * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_ROULETTE);
			if (!(rand(ps.rng_state) < survive)) {
				ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
				path_pack(payload, ps);
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
	if (!(brdfType == SPECULAR_TYPE && brdf_mat.roughness * brdf_mat.roughness <= 0.0)) {
		ps.packed_bounces_flags = set_scattered(ps.packed_bounces_flags);
	}

	// The density of the direction just chosen, for the next hit's emission weight (D50): a
	// zero-roughness specular lobe is a delta light sampling cannot hit, so it keeps the whole
	// weight; every other direction is weighed with the density brdf_pdf assigns it, which is
	// the same function light sampling evaluates for its own directions. The miss weighs the sky
	// with it too when the sky is a light (D54).
	if ((rt_emissive_count == 0u && !sky_light_enabled()) || (brdfType == SPECULAR_TYPE && brdf_mat.roughness * brdf_mat.roughness <= 0.0)) {
		// With no emitter listed and no sky light nothing reads the density, so both off cost no ALU here.
		ps.bsdf_pdf = RT_BSDF_PDF_DELTA;
	} else {
		float pdf = brdf_pdf(N, next_dir, V, brdf_mat);
		// A direction the density cannot own (the below-hemisphere recovery; on a thin surface, a
		// direction on the side of N the chosen lobe does not sample: a transmitted one mirrored back
		// above N, a reflected one in the wedge between N and the geometry plane) is one light
		// sampling does not produce this way, so the BSDF strategy keeps the whole weight.
		if (diffuse_transmit_p > 0.0 && diffuse_transmitted != (dot(N, next_dir) < 0.0)) {
			pdf = 0.0;
		}
		ps.bsdf_pdf = pdf > 0.0 ? pdf : RT_BSDF_PDF_DELTA;
	}

	// Hand the next ray back to raygen. PATH_TERMINATED_FLAG stays clear so
	// the raygen loop continues with the reconstructed origin and next_ray_dir.
	ps.hit_t = hit_t;
	ps.offset_normal = h.geometry_normal;
	ps.packed_bounces_flags = set_back_offset(ps.packed_bounces_flags, diffuse_transmitted);
	ps.next_ray_dir = next_dir;
	path_pack(payload, ps);
}
