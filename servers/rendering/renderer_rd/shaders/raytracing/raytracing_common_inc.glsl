// Shared defines and common bindings for all RT shader stages.
// Include AFTER raytracing_inc.glsl and scene_data_inc.glsl.
// The includer must set exactly one of RT_STAGE_{RAYGEN,MISS,CLOSEST_HIT,ANY_HIT,INTERSECTION}.

// Specialization constant (bits 0-10: flags, 11-12: the SER hint, 13: cluster BLAS, 14-20: free, 21-28: samples, 29-31: bounces).
// Bit 13 is read at compile time as USE_CLUSTER_AS (D79), never from RT_FLAGS.
layout(constant_id = 0) const uint RT_FLAGS = 0u;

#define RT_FLAG_DLSS_RR_ENABLED (1u << 1)
#define RT_FLAG_FOG_ENABLED (1u << 2)
#define RT_FLAG_SOBOL_SAMPLER (1u << 5)
#define RT_FLAG_SKY_LIGHT (1u << 6)
#define RT_FLAG_LIGHT_TREE (1u << 7)
#define RT_FLAG_LIGHT_TREE_POWER_PICK (1u << 8)
#define RT_FLAG_SKY_LIGHT_VARIANCE_SHARE (1u << 9)
#define RT_FLAG_RUSSIAN_ROULETTE (1u << 10)

// The SER coherence hint's mode (D75), bits 11-12; the values of
// rendering/pathtracing/shader_execution_reordering_hint.
#define RT_SER_HINT_SHIFT 11u
#define RT_SER_HINT_MASK 0x3u
#define RT_GET_SER_HINT() ((RT_FLAGS >> RT_SER_HINT_SHIFT) & RT_SER_HINT_MASK)
#define RT_SER_HINT_NONE 0u
#define RT_SER_HINT_INSTANCE 1u
#define RT_SER_HINT_TERMINATION 2u
#define RT_SER_HINT_TERMINATION_DISTANCE 3u

#define RT_SAMPLE_COUNT_SHIFT 21u
#define RT_SAMPLE_COUNT_MASK 0xFFu
#define RT_GET_SAMPLE_COUNT() max(1u, (RT_FLAGS >> RT_SAMPLE_COUNT_SHIFT) & RT_SAMPLE_COUNT_MASK)

#define RT_MAX_BOUNCES_SHIFT 29u
#define RT_MAX_BOUNCES_MASK 0x7u
#define RT_GET_MAX_BOUNCES() (((RT_FLAGS >> RT_MAX_BOUNCES_SHIFT) & RT_MAX_BOUNCES_MASK) + 1u)

// Cull back faces by default; double-sided instances override via CULL_DISABLE flag.
#define RT_RAY_FLAGS gl_RayFlagsCullBackFacingTrianglesEXT

// Instance masks match RT_INSTANCE_MASK_* in render_raytracing.h. All geometry
// carries bit 0; shadow casters also carry bit 1, procedural instances bit 2.
// Camera and bounce rays select all bits; the G-buffer probe selects only bit 2.
#define RT_MASK_SHADOW 0x02u
#define RT_MASK_PROCEDURAL 0x04u

layout(set = 0, binding = 2, std140) uniform SceneDataBlock {
	SceneData data;
	SceneData prev_data;
}
scene_data_block;

layout(set = 0, binding = 14, std430) readonly buffer GlobalShaderUniformData {
	vec4 data[256];
}
global_shader_uniforms;

layout(set = 0, binding = 6, std140) uniform RaytracingParams {
	vec4 rt_params[5];
	mat4 prev_vp_unjittered;
	mat4 curr_vp_unjittered;
	// World-space anchor the trace is relative to (camera-relative tracing).
	vec4 rt_origin;
	// The medium the camera is inside of (Environment.camera_medium_*, D49): the per-metre
	// extinction with 1 in w while enabled, and the single-scatter colour with the medium's top
	// (trace space) in w.
	vec4 camera_medium_absorption;
	vec4 camera_medium_scatter;
};

float get_rt_param(uint idx) {
	return rt_params[idx >> 2u][idx & 3u];
}

/// Whether a path whose packed counts are p_packed has reached a bounce limit, the whole-path
/// limit (RT_GET_MAX_BOUNCES(), a specialization constant) or the diffuse limit (D74, a uniform):
/// the vertex it arrives at adds its emission and ends. The one test for shade_and_bounce and for
/// the SER termination hint (D75), so the two cannot drift apart.
bool rt_bounce_limit_reached(uint p_packed) {
	return get_total_bounces(p_packed) >= RT_GET_MAX_BOUNCES() || get_diffuse_bounces(p_packed) >= uint(get_rt_param(RT_PARAM_MAX_DIFFUSE_BOUNCES));
}

/// One path contribution under a luminance clamp (D53): above the limit it is scaled down to the
/// limit with its colour kept, and a limit of 0 is no clamp. A contribution that is not finite is
/// dropped either way: path_pack's saturation is undefined for a NaN, and one such pixel poisons
/// the denoiser's history.
vec3 rt_clamp_contribution(vec3 c, float limit) {
	if (any(isnan(c)) || any(isinf(c))) {
		return vec3(0.0);
	}
	if (limit > 0.0) {
		float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
		if (l > limit) {
			c *= limit / l;
		}
	}
	return c;
}

/// The sampler the pipeline was specialised for (D52): a constant, so the other one costs nothing.
bool rng_is_sobol() {
	return (RT_FLAGS & RT_FLAG_SOBOL_SAMPLER) != 0u;
}

/// The launch pixel's hash, the Sobol sampler's per-pixel key (raytracing_inc.glsl, D52).
uint rt_pixel_seed() {
	return pcg_hash(gl_LaunchIDEXT.x + (gl_LaunchIDEXT.y << 16u));
}

/// Project a world-space point to UV through an unjittered VP.
vec2 project_uv(vec3 world_pos, mat4 vp) {
	vec4 clip = vp * vec4(world_pos, 1.0);
	return clip.xy / clip.w * 0.5 + 0.5;
}

#ifndef RT_STAGE_ANY_HIT

#ifdef DLSS_RR_ENABLED
layout(set = 0, binding = 9, rgba16f) uniform image2D dlss_rr_diffuse_albedo;
layout(set = 0, binding = 10, rgba16f) uniform image2D dlss_rr_specular_albedo;
layout(set = 0, binding = 11, rgba16f) uniform image2D dlss_rr_normal_roughness;
layout(set = 0, binding = 12, r16f) uniform image2D dlss_rr_specular_hit_dist;
#endif

// Binding 14 is reserved for GlobalShaderUniformData (declared above).
// Samplers occupy 16-27 (see raytracing_samplers_inc.glsl); velocity sits at
// the first free slot past them so we do not collide with either.
layout(set = 0, binding = 28, rg16f) uniform image2D rt_velocity_image;
layout(set = 0, binding = 15, r32f) uniform image2D rt_depth_image;

#endif // !RT_STAGE_ANY_HIT

// Shared hitAttributeEXT layout for all hit-group stages.
// Vulkan requires every shader in a hit group to agree on this layout.
#if defined(RT_STAGE_CLOSEST_HIT) || defined(RT_STAGE_ANY_HIT) || defined(RT_STAGE_INTERSECTION)
struct HitAttribs {
	vec2 bary_or_uv;
#ifdef ENABLE_INTERSECTION_SHADERS
	uint packed_normal; // xyz: snorm8 normal,  w: delta.x FP16 low byte.
	uint packed_tangent; // xyz: snorm8 tangent, w: delta.x FP16 high byte.
	uint prev_pos_delta_yz; // packHalf2x16(delta.y, delta.z).
#endif
};
hitAttributeEXT HitAttribs hit_attribs;
#endif
