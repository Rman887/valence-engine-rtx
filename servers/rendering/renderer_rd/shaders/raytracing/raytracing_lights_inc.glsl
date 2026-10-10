// Light sampling and Next Event Estimation (NEE) for raytracing.
// Requires: raytracing_inc.glsl, brdf_inc.glsl, tlas at binding 1, payload at location 0.
// Note: ray_query_alpha_test() still requires GL_EXT_ray_query (used by DLSS-RR path or when USE_RAY_QUERY_SHADOWS is defined).

// ============================================================================
// Light Types and Constants
// ============================================================================

#define RT_LIGHT_TYPE_OMNI 0 // Point light with radius (soft shadows)
#define RT_LIGHT_TYPE_DIRECTIONAL 1 // Sun/moon with angular size
#define RT_LIGHT_TYPE_RECT 2 // VAL (D124): AreaLight3D, a one-sided rectangle (the raster's)
#define RT_LIGHT_TYPE_SPOT 3 // Spot light with cone falloff

// Reservoir sampling batch size for stochastic light selection.
#ifndef RT_LIGHT_RESERVOIR_SIZE
#define RT_LIGHT_RESERVOIR_SIZE 16
#endif

// ============================================================================
// Light Data (matches C++ RT_LightData, 80 bytes, std430)
// ============================================================================

struct RTLightData {
	vec3 position; // World pos (omni/spot) or direction (directional).
	uint type; // RT_LIGHT_TYPE_*.
	vec3 emission; // HDR emission color (color * energy). RECT: the rectangle's radiance.
	float radius; // Light size (omni/spot) or angular radius (directional). RECT: the half diagonal.
	float attenuation; // Attenuation exponent (2=inverse-square, 1=linear).
	float inv_max_range; // 1/range (-1 = infinite).
	float max_range_squared; // range^2 (0 = infinite).
	float specular_amount; // Godot specular multiplier [0..1].
	float indirect_energy; // Godot indirect energy multiplier.
	float inv_spot_attenuation; // Spot cone softness. RECT: the half height axis, x.
	float cos_spot_angle; // Cosine of spot cone half-angle. RECT: y.
	float _pad0; // RECT: z.
	vec3 spot_direction; // Spot direction (normalized, world space). RECT: the half width axis.
	float _pad1; // RECT: the sign turning cross(width axis, height axis) into the emitting normal.
};

// Light buffer SSBO (binding provided by the including shader via RT_LIGHT_BUFFER_BINDING).
#ifndef RT_LIGHT_BUFFER_BINDING
#define RT_LIGHT_BUFFER_BINDING 13
#endif

layout(set = 0, binding = RT_LIGHT_BUFFER_BINDING, std430) readonly buffer LightBuffer {
	RTLightData rt_lights[];
};

// ============================================================================
// Unified Cone Sampling (for sphere, directional, spot lights)
// ============================================================================

struct LightSample {
	vec3 cone_axis; // Direction to sample around (normalized).
	float cos_theta_max; // Cone half-angle cosine (1.0 = point, 0.0 = hemisphere).
	vec3 emission; // Light radiance.
	float distance_sq; // Squared distance to light (0 = directional).
	float max_distance; // Max shadow ray distance.
};

// Build orthonormal basis from a single direction.
void lights_build_basis(vec3 dir, out vec3 tangent, out vec3 bitangent) {
	vec3 up = abs(dir.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	tangent = normalize(cross(up, dir));
	bitangent = cross(dir, tangent);
}

// Transform local direction to world space around axis.
vec3 lights_local_to_world(vec3 local_dir, vec3 axis) {
	vec3 tangent, bitangent;
	lights_build_basis(axis, tangent, bitangent);
	return local_dir.x * tangent + local_dir.y * bitangent + local_dir.z * axis;
}

// Prepare unified cone sample from any light type.
LightSample lights_prepare_sample(vec3 hit_pos, RTLightData light) {
	LightSample s;

	vec3 to_light = light.position - hit_pos;
	float dist_sq = dot(to_light, to_light);
	float inv_dist = inversesqrt(dist_sq + 1e-10);
	float dist = dist_sq * inv_dist;

	float is_directional = (light.type == RT_LIGHT_TYPE_DIRECTIONAL) ? 1.0 : 0.0;

	// Cone axis.
	vec3 sphere_axis = to_light * inv_dist;
	vec3 dir_axis = -normalize(light.position);
	s.cone_axis = mix(sphere_axis, dir_axis, is_directional);

	// Distance (0 for directional = no falloff).
	s.distance_sq = mix(dist_sq, 0.0, is_directional);

	// Cone angle from subtended solid angle.
	float sin_theta_sphere = clamp(light.radius * inv_dist, 0.0, 1.0);
	float cos_theta_sphere = sqrt(max(0.0, 1.0 - sin_theta_sphere * sin_theta_sphere));
	float cos_theta_dir = cos(light.radius);
	s.cos_theta_max = mix(cos_theta_sphere, cos_theta_dir, is_directional);
	s.cos_theta_max = min(s.cos_theta_max, 0.999999);

	s.emission = light.emission;

	// Max shadow ray distance.
	float sphere_max = dist + light.radius;
	float dir_max = 10000.0;
	s.max_distance = mix(sphere_max, dir_max, is_directional);

	return s;
}

// Sample a direction within the light's cone.
vec3 lights_sample_cone(LightSample ls, vec2 u, out float pdf) {
	float cos_theta = 1.0 - u.x * (1.0 - ls.cos_theta_max);
	float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
	float phi = 2.0 * PI * u.y;

	vec3 local_dir = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
	vec3 L = lights_local_to_world(local_dir, ls.cone_axis);

	float solid_angle = 2.0 * PI * (1.0 - ls.cos_theta_max);
	pdf = 1.0 / max(solid_angle, 1e-10);

	return L;
}

// ============================================================================
// Attenuation
// ============================================================================

// VAL (D123): the nearest a receiver is taken to be to an omni or spot light's centre, in metres.
#define RT_LIGHT_MIN_DISTANCE 0.01

// Godot-style windowed distance attenuation.
// window = (1 - (d/range)^4)^2, combined with pow(d, -decay).
// VAL (D123): the falloff is the raster's outside the light's radius, uncapped: NVIDIA's min(pow(d, -decay), 1) held a light at its
// 1 m value for every receiver inside a metre (0.09 of the inverse square at 0.3 m). The distance is
// floored at the light's radius instead (a sphere's falloff outside it is exactly the inverse square
// of the centre distance), and at 1 cm for a point light, which keeps a receiver touching it finite.
float lights_get_attenuation(LightSample ls, float inv_max_range, float decay, float radius) {
	if (ls.distance_sq <= 0.0) {
		return 1.0; // Directional: no distance attenuation.
	}

	float distance = sqrt(ls.distance_sq);
	float atten = pow(max(distance, max(radius, RT_LIGHT_MIN_DISTANCE)), -decay);

	// Windowed falloff if range is finite (inv_max_range >= 0).
	if (inv_max_range >= 0.0) {
		float nd = distance * inv_max_range;
		nd *= nd;
		nd *= nd; // nd^4
		nd = max(1.0 - nd, 0.0);
		nd *= nd; // nd^2 window
		return atten * nd;
	}

	return atten;
}

// ============================================================================
// Rectangle lights (VAL, D124)
// ============================================================================

// A rectangle light's half axes and the normal it emits along (the light's -Z, as the raster's).
vec3 lights_rect_half_x(RTLightData light) {
	return light.spot_direction;
}

vec3 lights_rect_half_y(RTLightData light) {
	return vec3(light.inv_spot_attenuation, light.cos_spot_angle, light._pad0);
}

vec3 lights_rect_normal(RTLightData light) {
	return normalize(cross(light.spot_direction, lights_rect_half_y(light))) * light._pad1;
}

// The raster's area-light attenuation (light_process_area): the range window and the decay taken from the
// distance to the rectangle's closest point, relative to the inverse square the solid angle already falls
// off by (d^(2 - decay): 1 at the default decay of 2). Floored at 1 cm, as the omni falloff (D123).
float lights_rect_attenuation(RTLightData light, vec3 hit_pos, vec3 half_x, vec3 half_y, vec3 normal) {
	vec3 rel = hit_pos - light.position;
	float hx = length(half_x);
	float hy = length(half_y);
	vec2 local = vec2(dot(rel, half_x) / max(hx, 1e-8), dot(rel, half_y) / max(hy, 1e-8));
	vec2 closest = clamp(local, -vec2(hx, hy), vec2(hx, hy));
	float d = length(vec3(local - closest, dot(rel, normal)));
	float atten = pow(max(d, RT_LIGHT_MIN_DISTANCE), 2.0 - light.attenuation);
	if (light.inv_max_range >= 0.0) {
		float nd = d * light.inv_max_range;
		nd *= nd;
		nd *= nd;
		nd = max(1.0 - nd, 0.0);
		atten *= nd * nd;
	}
	return atten;
}

// Urena, Fajardo and King 2013, "An Area-Preserving Parametrization for Spherical Rectangles": a point on the
// rectangle (corner p_corner, edges p_ex and p_ey) drawn uniformly in the solid angle it subtends from p_o,
// and that solid angle. False where the rectangle is too small or too oblique to set up in float; the caller
// samples its area instead.
bool lights_rect_sample_solid_angle(vec3 p_o, vec3 p_corner, vec3 p_ex, vec3 p_ey, vec2 p_u, out vec3 r_point, out float r_solid_angle) {
	r_point = p_corner;
	r_solid_angle = 0.0;
	float exl = length(p_ex);
	float eyl = length(p_ey);
	if (!(exl > 0.0) || !(eyl > 0.0)) {
		return false;
	}
	vec3 x = p_ex / exl;
	vec3 y = p_ey / eyl;
	vec3 z = cross(x, y);
	vec3 d = p_corner - p_o;
	float z0 = dot(d, z);
	if (z0 > 0.0) {
		z = -z;
		z0 = -z0;
	}
	float x0 = dot(d, x);
	float y0 = dot(d, y);
	float x1 = x0 + exl;
	float y1 = y0 + eyl;
	vec3 v00 = vec3(x0, y0, z0);
	vec3 v01 = vec3(x0, y1, z0);
	vec3 v10 = vec3(x1, y0, z0);
	vec3 v11 = vec3(x1, y1, z0);
	vec3 c0 = cross(v00, v10);
	vec3 c1 = cross(v10, v11);
	vec3 c2 = cross(v11, v01);
	vec3 c3 = cross(v01, v00);
	float l0 = length(c0);
	float l1 = length(c1);
	float l2 = length(c2);
	float l3 = length(c3);
	if (!(min(min(l0, l1), min(l2, l3)) > 1e-12)) {
		return false;
	}
	vec3 n0 = c0 / l0;
	vec3 n1 = c1 / l1;
	vec3 n2 = c2 / l2;
	vec3 n3 = c3 / l3;
	float g0 = acos(clamp(-dot(n0, n1), -1.0, 1.0));
	float g1 = acos(clamp(-dot(n1, n2), -1.0, 1.0));
	float g2 = acos(clamp(-dot(n2, n3), -1.0, 1.0));
	float g3 = acos(clamp(-dot(n3, n0), -1.0, 1.0));
	float b0 = n0.z;
	float b1 = n2.z;
	float k = 2.0 * PI - g2 - g3;
	float S = g0 + g1 - k;
	// A solid angle the acos sums cannot resolve (a small or edge-on rectangle) is sampled by area.
	if (!(S > 1e-4)) {
		return false;
	}
	float au = p_u.x * S + k;
	float sin_au = sin(au);
	if (abs(sin_au) < 1e-7) {
		return false;
	}
	float fu = (cos(au) * b0 - b1) / sin_au;
	float cu = clamp((fu > 0.0 ? 1.0 : -1.0) / sqrt(fu * fu + b0 * b0), -1.0, 1.0);
	float xu = -(cu * z0) / max(sqrt(1.0 - cu * cu), 1e-12);
	xu = clamp(xu, x0, x1);
	float dd = sqrt(xu * xu + z0 * z0);
	float h0 = y0 / sqrt(dd * dd + y0 * y0);
	float h1 = y1 / sqrt(dd * dd + y1 * y1);
	float hv = h0 + p_u.y * (h1 - h0);
	float hv2 = hv * hv;
	float yv = (hv2 < 1.0 - 1e-6) ? (hv * dd) / sqrt(1.0 - hv2) : y1;
	yv = clamp(yv, y0, y1);
	r_point = p_o + xu * x + yv * y + z0 * z;
	r_solid_angle = S;
	return !any(isnan(r_point)) && !any(isinf(r_point));
}

// Per-light specular multiplier.
float lights_get_specular_multiplier(float specular_amount, float roughness) {
	if (specular_amount >= 0.0) {
		return specular_amount;
	} else {
		float r3 = roughness * roughness * roughness;
		return mix(0.0, r3, -specular_amount);
	}
}

// The miss stage reads the light buffer for the light tree's selection (D55) and traces nothing.
#ifndef RT_STAGE_MISS

// ============================================================================
// Inline Alpha Test (shared by all ray query proceed loops)
// ============================================================================

/// The candidate's cluster ID, the ray-query twin of rt_hit_cluster_id() (D79), which
/// rt_resolve_triangle takes with the candidate's primitive index.
int rt_ray_query_cluster_id(rayQueryEXT q) {
#ifdef USE_CLUSTER_AS
	return rayQueryGetIntersectionClusterIdNV(q, false);
#else
	return RT_CLUSTER_ID_NONE;
#endif
}

/// Inline alpha test for ray query candidates. Returns true if the hit is opaque (alpha >= the material's
/// alpha_threshold, sampled at level 0 as HG0's any-hit does).
/// Mirrors the any-hit shader logic for use with inline ray queries.
bool ray_query_alpha_test(uint geometry_idx, int cluster_id, uint primitive, vec2 candidate_bary) {
	vec3 bary = vec3(1.0 - candidate_bary.x - candidate_bary.y, candidate_bary.x, candidate_bary.y);

	GeometryData geom = geometries[geometry_idx];
	uint i0, i1, i2;
	rt_resolve_triangle(geom, cluster_id, primitive, i0, i1, i2);
	vec2 uv = fetch_uv(geom, i0, i1, i2, bary);

	MaterialData mat = materials[geometry_idx];
	uv = uv * mat.uv1_scale + mat.uv1_offset;
	float alpha = textureLod(sampler2D(bindless_textures[nonuniformEXT(mat.albedo_texture_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), uv, 0.0).a;
	alpha *= mat.albedo_color.a;

	// As the any-hit's `alpha < threshold` ignores: a NaN alpha stays a hit in both.
	return !(alpha < mat.alpha_threshold);
}

// ============================================================================
// Shadow Ray (an inline ray query, or the traceRayEXT pipeline)
// ============================================================================

/// The shadow ray through the pipeline: every candidate is decided by its hit group's any-hit (an alpha
/// test) or intersection shader. Returns true if the light is visible.
/// SkipClosestHitShader: only any_hit (alpha test) and miss are invoked.
/// TerminateOnFirstHit causes early exit on first confirmed opaque hit.
bool lights_trace_shadow_ray_pipeline(vec3 origin, vec3 direction, float max_dist, uint omm_flags) {
#if defined(USE_SER)
	hitObjectEXT hitObject;
	hitObjectTraceRayEXT(hitObject, tlas,
			gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | omm_flags,
			RT_MASK_SHADOW, 0, 0, 0,
			origin, 0.001, direction, max_dist - 0.001, 0);

	return !(hitObjectIsHitEXT(hitObject));
#else
	/// The miss shader writes radiance = vec3(1.0) for shadow rays (visible).
	/// If an opaque hit occurs, miss is never called and radiance stays vec3(0.0).
	// Save full payload, set up shadow ray, then restore after trace.
	PathPayload saved_payload = payload;

	PathState shadow_ps;
	shadow_ps.radiance = vec3(0.0);
	shadow_ps.throughput = vec3(0.0);
	shadow_ps.packed_bounces_flags = set_shadow_ray(0u);
	shadow_ps.rng_state = 0u;
	shadow_ps.hit_t = 0.0;
	shadow_ps.bsdf_pdf = 0.0;
	shadow_ps.cone_width = 0.0;
	shadow_ps.cone_spread = 0.0;
#ifdef RT_SUBSURFACE_COMPILED
	shadow_ps.exit_position = vec3(0.0);
#endif
	shadow_ps.offset_normal = vec3(0.0, 0.0, 1.0);
	shadow_ps.next_ray_dir = vec3(0.0, 0.0, 1.0);
	path_clear_medium(shadow_ps);
	path_pack(payload, shadow_ps);

	traceRayEXT(tlas,
			gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | omm_flags,
			RT_MASK_SHADOW, 0, 0, 0,
			origin, 0.001, direction, max_dist - 0.001, 0);

	// Unpack to check visibility (miss shader packs radiance = 1.0).
	shadow_ps = path_unpack(payload);
	bool visible = shadow_ps.radiance.x > 0.5;

	payload = saved_payload;

	return visible;
#endif
}

#ifdef RT_SHADOW_STATS
// The inline shadow ray's counters (P1k), one pixel in 64: rays, rays that met a candidate the inline test cannot
// decide, rays traced through the pipeline for it (nothing committed). The host reads and clears them on the verbose
// report's cadence.
layout(set = 0, binding = 53, std430) buffer ShadowStats {
	uint words[];
}
shadow_stats;
#endif

#ifdef USE_RAY_QUERY_SHADOWS
/// Whether the inline test decides a ray query's candidate as the pipeline would: a triangle of the standard
/// hit group, whose any-hit ray_query_alpha_test mirrors. An AABB needs its intersection shader and a custom
/// group's triangle its own any-hit, which run the material's code.
bool lights_shadow_candidate_inline(rayQueryEXT p_query) {
	if (rayQueryGetIntersectionTypeEXT(p_query, false) != gl_RayQueryCandidateIntersectionTriangleEXT) {
		return false;
	}
	const uint record = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(p_query, false);
	return record == 0u || record == RT_HIT_SBT_WALK_RECORD;
}
#endif

/// Returns true if light is visible.
/// omm_flags: RT_OMM_RAY_FLAGS of the vertex's arriving diffuse count (D96 OMMe).
bool lights_trace_shadow_ray(vec3 origin, vec3 direction, float max_dist, uint omm_flags, inout uint rng_state) {
#ifdef USE_RAY_QUERY_SHADOWS
	// VAL (P1k, docs/gpu-audit-2026-10.md): an inline query, which spills nothing of the hit's shading state around a
	// nested trace. Opaque instances commit without a candidate, and a standard material's alpha is tested here as its
	// any-hit tests it. A candidate the inline test cannot decide (a procedural AABB, a custom hit group's alpha) is
	// passed over: occlusion is an OR, so any committed hit settles the ray, and only a ray with such a candidate and
	// nothing committed is traced through the pipeline, whose shaders decide it. The same visibility as the
	// pipeline's for every ray; a ray that falls back pays both traversals.
	rayQueryEXT shadow_rq;
	rayQueryInitializeEXT(shadow_rq, tlas,
			gl_RayFlagsTerminateOnFirstHitEXT | omm_flags,
			RT_MASK_SHADOW, origin, 0.001, direction, max_dist - 0.001);

	bool unresolved = false;
	while (rayQueryProceedEXT(shadow_rq)) {
		if (!lights_shadow_candidate_inline(shadow_rq)) {
			unresolved = true;
		} else if (ray_query_alpha_test(
						   rayQueryGetIntersectionInstanceCustomIndexEXT(shadow_rq, false),
						   rt_ray_query_cluster_id(shadow_rq),
						   uint(rayQueryGetIntersectionPrimitiveIndexEXT(shadow_rq, false)),
						   rayQueryGetIntersectionBarycentricsEXT(shadow_rq, false))) {
			rayQueryConfirmIntersectionEXT(shadow_rq);
		}
	}

	const bool occluded = rayQueryGetIntersectionTypeEXT(shadow_rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
#ifdef RT_SHADOW_STATS
	if (((gl_LaunchIDEXT.x | gl_LaunchIDEXT.y) & 7u) == 0u) {
		atomicAdd(shadow_stats.words[0], 1u);
		if (unresolved) {
			atomicAdd(shadow_stats.words[1], 1u);
			if (!occluded) {
				atomicAdd(shadow_stats.words[2], 1u);
			}
		}
	}
#endif
	if (occluded || !unresolved) {
		return !occluded;
	}
#endif
	return lights_trace_shadow_ray_pipeline(origin, direction, max_dist, omm_flags);
}

#endif // !RT_STAGE_MISS

// ============================================================================
// Next Event Estimation (NEE) - Direct Light Sampling
// ============================================================================

// A light sample before its shadow ray (D50): the direction, the distance the shadow ray is
// traced over, the distance a medium charges the light's leg over (unbounded for a directional
// light, D49), the light's radiance at the receiver with its attenuation, the specular multiplier
// and the weight the contribution is multiplied by (the reciprocal of the selection density, and
// for a mesh light its MIS weight too). The shadow ray and the BRDF are the caller's, shared by
// every strategy, so a warp whose threads chose different strategies traces once.
struct LightSampleNEE {
	vec3 direction;
	float shadow_distance;
	float leg_distance;
	vec3 radiance;
	float weight;
	float specular_multiplier;
};

bool lights_sample_light(RTLightData light, vec3 hit_pos, vec3 N, MaterialProperties material, vec2 u, bool is_indirect_bounce, out LightSampleNEE r_sample);

// Sample one analytic light for NEE with stochastic light selection.
// Uses mini-batch reservoir sampling for importance-weighted light selection.
// Returns false when the sample carries nothing (no valid light, out of range, outside the spot
// cone, below the horizon); the random draws are the same either way.
bool lights_sample_direct(
		vec3 hit_pos,
		vec3 N,
		MaterialProperties material,
		inout uint rng_state,
		bool is_indirect_bounce,
		uint light_count,
		out LightSampleNEE r_sample) {
	r_sample.direction = vec3(0.0, 1.0, 0.0);
	r_sample.shadow_distance = 0.0;
	r_sample.leg_distance = 0.0;
	r_sample.radiance = vec3(0.0);
	r_sample.weight = 0.0;
	r_sample.specular_multiplier = 1.0;
	if (light_count == 0u) {
		return false;
	}

	// Mini-batch reservoir sampling: sample k lights, pick the best valid one.
	const uint k = uint(min(RT_LIGHT_RESERVOIR_SIZE, int(light_count)));
	uint valid_found = 0u;
	uint selected_idx = 0u;

	for (uint i = 0u; i < k; i++) {
		uint idx = min(uint(rand(rng_state) * float(light_count)), light_count - 1u);
		RTLightData test_light = rt_lights[idx];

		// Range check for positional lights.
		bool is_positional = (test_light.type == RT_LIGHT_TYPE_OMNI || test_light.type == RT_LIGHT_TYPE_SPOT || test_light.type == RT_LIGHT_TYPE_RECT);
		bool is_valid = !is_positional;
		if (!is_valid) {
			vec3 to_l = test_light.position - hit_pos;
			float d2 = dot(to_l, to_l);
			// A rectangle's range is from its closest point (D124): bounded here by its half diagonal.
			float reach = test_light.type == RT_LIGHT_TYPE_RECT ? sqrt(test_light.max_range_squared) + test_light.radius : 0.0;
			is_valid = (test_light.max_range_squared == 0.0 || d2 <= (test_light.type == RT_LIGHT_TYPE_RECT ? reach * reach : test_light.max_range_squared));
		}

		if (is_valid) {
			valid_found++;
			if (rand(rng_state) < 1.0 / float(valid_found)) {
				selected_idx = idx;
			}
		}
	}

	if (valid_found == 0u) {
		return false;
	}

	// Estimate valid count from sample ratio, PDF = 1/validCount.
	float valid_count_estimate = float(light_count) * (float(valid_found) / float(k));
	float light_select_pdf = 1.0 / max(valid_count_estimate, 1.0);

	RTLightData light = rt_lights[selected_idx];
	vec2 u = rand2(rng_state);
	bool have_sample = lights_sample_light(light, hit_pos, N, material, u, is_indirect_bounce, r_sample);
	r_sample.weight = 1.0 / max(light_select_pdf, 1e-10);
	return have_sample;
}

// VAL: whether a light sample can deliver anything, so a sample that cannot traces no shadow ray. Behind
// the receiver only a thin surface's transmission (D103) or a coat above it (D118 MM2) takes light, as
// evalCombinedBRDFSeparate decides on the same shading normal; a light with zero indirect energy delivers
// nothing at an indirect vertex. Tested after the sample's random numbers are drawn, so the sequence is
// the same either way and the image is bit-identical.
bool lights_sample_reaches_brdf(vec3 N, MaterialProperties material, vec3 L, float indirect_mul) {
	return indirect_mul != 0.0 && (dot(N, L) > 0.0 || brdfTakesLightBelowN(material, L));
}

// One analytic light's sample from its pair of random numbers: an omni or spot light's point or
// cone with its attenuation, a directional light's cone, a rectangle's point by solid angle (area
// when that is degenerate) with the raster's closest-point attenuation (D124); the weight is left at
// 1 for the caller's selection density (the reservoir's estimate above, the light tree's exact
// probability, D55). Returns false when the sample carries nothing (out of range, outside the spot
// cone, behind the rectangle, below the horizon).
bool lights_sample_light(
		RTLightData light,
		vec3 hit_pos,
		vec3 N,
		MaterialProperties material,
		vec2 u,
		bool is_indirect_bounce,
		out LightSampleNEE r_sample) {
	r_sample.direction = vec3(0.0, 1.0, 0.0);
	r_sample.shadow_distance = 0.0;
	r_sample.leg_distance = 0.0;
	r_sample.radiance = vec3(0.0);
	r_sample.weight = 1.0;
	float indirect_mul = is_indirect_bounce ? light.indirect_energy : 1.0;
	r_sample.specular_multiplier = lights_get_specular_multiplier(light.specular_amount, material.roughness);

	// === RECTANGLE LIGHT PATH (AreaLight3D, VAL D124) ===
	// One-sided toward its normal, its colour a radiance, as the raster's. A point drawn uniformly in the
	// solid angle the rectangle subtends (Urena 2013) carries the radiance over that density, the solid
	// angle; where that cannot be set up, a point drawn by area carries it over the area density. The
	// shadow ray ends at the point. Attenuation as the raster's (lights_rect_attenuation).
	if (light.type == RT_LIGHT_TYPE_RECT) {
		vec3 half_x = lights_rect_half_x(light);
		vec3 half_y = lights_rect_half_y(light);
		vec3 normal = lights_rect_normal(light);
		if (dot(normal, hit_pos - light.position) <= 0.0) {
			return false; // Behind the light.
		}
		float atten = lights_rect_attenuation(light, hit_pos, half_x, half_y, normal);
		if (!(atten > 0.0)) {
			return false; // Beyond the range.
		}
		vec3 point;
		float solid_angle;
		float inv_pdf;
		if (lights_rect_sample_solid_angle(hit_pos, light.position - half_x - half_y, 2.0 * half_x, 2.0 * half_y, u, point, solid_angle)) {
			inv_pdf = solid_angle;
		} else {
			point = light.position + (2.0 * u.x - 1.0) * half_x + (2.0 * u.y - 1.0) * half_y;
			vec3 to_p = point - hit_pos;
			float d2 = dot(to_p, to_p);
			float cos_l = dot(normal, -to_p) * inversesqrt(max(d2, 1e-20));
			if (!(cos_l > 0.0)) {
				return false;
			}
			inv_pdf = 4.0 * length(half_x) * length(half_y) * cos_l / max(d2, 1e-20);
		}
		vec3 to_point = point - hit_pos;
		float dist = length(to_point);
		if (!(dist > 0.0)) {
			return false;
		}
		r_sample.direction = to_point / dist;
		r_sample.shadow_distance = dist;
		r_sample.leg_distance = dist;
		r_sample.radiance = light.emission * (atten * inv_pdf * indirect_mul);
		return lights_sample_reaches_brdf(N, material, r_sample.direction, indirect_mul);
	}
	// === POSITIONAL LIGHT PATH (omni + spot) ===
	if (light.type == RT_LIGHT_TYPE_OMNI || light.type == RT_LIGHT_TYPE_SPOT) {
		vec3 to_light = light.position - hit_pos;
		float dist_sq = dot(to_light, to_light);

		// Early out: outside max range.
		if (light.max_range_squared != 0.0 && dist_sq > light.max_range_squared) {
			return false;
		}

		float dist = sqrt(dist_sq);
		vec3 L;
		float shadow_dist;
		float sphere_norm = 1.0;

		if (light.radius <= 0.01) {
			// True point light: exact direction.
			L = to_light / max(dist, 0.0001);
			shadow_dist = dist;
		} else {
			// Sphere light: cone sampling for soft shadows.
			LightSample ls = lights_prepare_sample(hit_pos, light);
			float light_pdf;
			L = lights_sample_cone(ls, u, light_pdf);
			// VAL (D123): a direction drawn uniformly in the subtended cone and charged the centre's falloff
			// delivers (1 + cos(theta_max)) / 2 of a uniform sphere's irradiance (half at contact); this
			// restores it, exact outside the sphere and continuous with the radius floor inside it.
			sphere_norm = 2.0 / (1.0 + ls.cos_theta_max);
			float t_center = dot(to_light, L);
			vec3 perp = to_light - t_center * L;
			float perp_sq = dot(perp, perp);
			float dt = sqrt(max(0.0, light.radius * light.radius - perp_sq));
			shadow_dist = max(0.0, t_center - dt);
		}

		r_sample.direction = L;
		r_sample.shadow_distance = shadow_dist;
		r_sample.leg_distance = shadow_dist;

		// Spot cone early-out.
		float spot_atten = 1.0;
		if (light.type == RT_LIGHT_TYPE_SPOT) {
			float scos = dot(-L, light.spot_direction);
			if (scos <= light.cos_spot_angle) {
				return false;
			}
			float spot_rim = max(1e-4, (1.0 - scos) / (1.0 - light.cos_spot_angle));
			spot_atten = 1.0 - pow(spot_rim, light.inv_spot_attenuation);
		}

		// Distance attenuation.
		LightSample ls_atten;
		ls_atten.distance_sq = dist_sq;
		float atten = lights_get_attenuation(ls_atten, light.inv_max_range, light.attenuation, light.radius) * spot_atten;

		r_sample.radiance = light.emission * atten * sphere_norm * indirect_mul;
		return lights_sample_reaches_brdf(N, material, L, indirect_mul);
	}
	// === CONE LIGHT PATH (directional) ===
	else {
		LightSample ls = lights_prepare_sample(hit_pos, light);
		float light_pdf;
		vec3 L = lights_sample_cone(ls, u, light_pdf);
		r_sample.direction = L;
		r_sample.shadow_distance = ls.max_distance;
		r_sample.leg_distance = 1e30;

		r_sample.radiance = light.emission * indirect_mul;
		return lights_sample_reaches_brdf(N, material, L, indirect_mul);
	}
}
