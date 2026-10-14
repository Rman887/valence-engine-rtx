#[raygen]

#version 460

#extension GL_EXT_control_flow_attributes : enable

#VERSION_DEFINES

// clang-format off
#include "raytracing_inc.glsl"
#include "../scene_data_inc.glsl"
// clang-format on

#pragma shader_stage(raygen)
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_nonuniform_qualifier : require
#ifdef USE_SER
#extension GL_EXT_shader_invocation_reorder : enable
#endif
#ifdef USE_CLUSTER_AS
#extension GL_NV_cluster_acceleration_structure : require
#endif
#ifdef USE_OMM_INDIRECT_2_STATE
#extension GL_EXT_opacity_micromap : require
#endif

#define GLSL 1
#define RT_STAGE_RAYGEN 1
// clang-format off
#include "brdf_inc.glsl"
#include "raytracing_common_inc.glsl"
#include "raytracing_hit_inc.glsl"
// clang-format on

layout(set = 0, binding = 0, rgba16f) uniform image2D image;
layout(set = 0, binding = 1) uniform accelerationStructureEXT tlas;

layout(location = 0) rayPayloadEXT PathPayload payload;

// The G-buffer primary surface (Environment.pathtracing_primary_surface): the
// raster pass's material at the visible surface and the depth it was
// rasterized with, shaded here with the hit shaders' shade_and_bounce. The
// bindings below are what shade_and_bounce needs from the raygen stage.
layout(set = 1, binding = 0) uniform texture2D bindless_textures[];

#include "raytracing_samplers_inc.glsl"

// clang-format off
layout(set = 0, binding = 3, std430) readonly buffer GeometryBuffer {
	GeometryData geometries[];
};

layout(set = 0, binding = 5, std430) readonly buffer MaterialBuffer {
	MaterialData materials[];
};
// clang-format on

#include "raytracing_lights_inc.glsl"
#include "raytracing_emissive_inc.glsl"

#ifdef USE_RADIANCE_OCTMAP_ARRAY

layout(set = 0, binding = 7) uniform texture2DArray radiance_octmap;
layout(set = 0, binding = 8) uniform sampler radiance_sampler;

vec3 radiance_octmap_sample(vec2 p_oct_uv, float p_roughness) {
	float layer;
	float blend = modf(clamp(p_roughness, 0.0, 1.0) * MAX_ROUGHNESS_LOD, layer);
	vec3 a = textureLod(sampler2DArray(radiance_octmap, radiance_sampler), vec3(p_oct_uv, layer), 0.0).rgb;
	vec3 b = textureLod(sampler2DArray(radiance_octmap, radiance_sampler), vec3(p_oct_uv, layer + 1.0), 0.0).rgb;
	return mix(a, b, blend);
}

#else

layout(set = 0, binding = 7) uniform texture2D radiance_octmap;
layout(set = 0, binding = 8) uniform sampler radiance_sampler;

vec3 radiance_octmap_sample(vec2 p_oct_uv, float p_roughness) {
	return textureLod(sampler2D(radiance_octmap, radiance_sampler), p_oct_uv, clamp(p_roughness, 0.0, 1.0) * MAX_ROUGHNESS_LOD).rgb;
}

#endif // USE_RADIANCE_OCTMAP_ARRAY

// clang-format off
#include "raytracing_sky_inc.glsl"
#include "raytracing_light_tree_inc.glsl"
#include "raytracing_material_eval_inc.glsl"
#include "raytracing_shade_inc.glsl"
#include "raytracing_surface_inc.glsl"
// clang-format on

layout(set = 0, binding = 33) uniform texture2D gbuffer_albedo_specular;
layout(set = 0, binding = 34) uniform texture2D gbuffer_normal_roughness;
layout(set = 0, binding = 35) uniform texture2D gbuffer_geo_normal_metallic;
layout(set = 0, binding = 36) uniform texture2D gbuffer_emission_ior;
layout(set = 0, binding = 37) uniform texture2D gbuffer_transmission_flags;
layout(set = 0, binding = 38) uniform texture2D gbuffer_depth;
// The frame's froxel volumetric fog (Environment.volumetric_fog_*), built in the path-traced
// frame and shadowed against the TLAS; composed over the camera segment in apply_camera_media.
layout(set = 0, binding = 39) uniform texture3D volumetric_fog_texture;
// The medium behind the visible surface (D49), the G-buffer's sixth and seventh targets.
layout(set = 0, binding = 40) uniform texture2D gbuffer_absorption;
layout(set = 0, binding = 41) uniform texture2D gbuffer_scattering;

/// The next ray's origin from the hit it leaves: inside the fibre at a strand (D126 H2, rt_strand_inset), or offset off the
/// surface along the payload's normal, to its back after a transmission (D103).
vec3 path_next_origin(vec3 p_hit_pos, PathState p_ps) {
	const float inset = (p_ps.packed_bounces_flags & HAIR_RAY_FLAG) != 0u ? rt_strand_inset_decode(p_ps.packed_bounces_flags) : 0.0;
	if (inset > 0.0) {
		return p_hit_pos - p_ps.offset_normal * inset;
	}
	return offset_ray_origin(p_hit_pos, is_back_offset(p_ps.packed_bounces_flags) ? -p_ps.offset_normal : p_ps.offset_normal, p_ps.hit_t);
}

void trace_path_ray(vec3 ray_origin, vec3 ray_dir, uint mask, float max_distance) {
	// A ray that leaves a strand starts inside it (D126 H2).
	const float tmin = (payload.packed_bounces_flags & HAIR_RAY_FLAG) != 0u ? RT_HAIR_TMIN : 0.001;
#ifdef USE_SER
	hitObjectEXT hitObject;
	hitObjectTraceRayEXT(hitObject, tlas, RT_RAY_FLAGS | RT_OMM_RAY_FLAGS(get_diffuse_bounces(payload.packed_bounces_flags)), mask, 0, 0, 0, ray_origin, tmin, ray_dir, max_distance, 0);

	// The reorder sorts the threads on the hit's shader by itself (the hit group, or the miss);
	// the hint refines that order. Its form is the pipeline's (D75, RT_GET_SER_HINT(), the
	// rendering/pathtracing/shader_execution_reordering_hint setting), so the other forms fold away.
	const uint hint_mode = RT_GET_SER_HINT();
	if (hint_mode == RT_SER_HINT_INSTANCE) {
		// NVIDIA's: the instance index in 8 bits, which the shader sort already separates by material.
		uint hint = 0u;
		if (hitObjectIsHitEXT(hitObject)) {
			hint = hitObjectGetInstanceIdEXT(hitObject);
		}
		reorderThreadEXT(hitObject, hint, 8);
	} else if (hint_mode == RT_SER_HINT_NONE) {
		reorderThreadEXT(hitObject);
	} else {
		// Whether the path ends at this hit (RTXPT's one hint bit): the counts packed before the
		// trace are the ones shade_and_bounce tests on arrival, so a set bit is a thread that only
		// adds the hit's emission, kept apart from one that samples the lights and a direction and
		// traces again. A miss ends the path too, and the sort keys it by the miss shader already.
		uint hint = rt_bounce_limit_reached(payload.packed_bounces_flags) ? 1u : 0u;
		if (hint_mode == RT_SER_HINT_TERMINATION_DISTANCE) {
			// Under the termination bit, a 3-bit bin of the hit distance in factors of 4 (under 4 m,
			// 4 to 16 m, ... 4096 m and beyond; a miss sits in the last).
			uint bin = 7u;
			if (hitObjectIsHitEXT(hitObject)) {
				bin = uint(clamp(log2(max(hitObjectGetRayTMaxEXT(hitObject), 1.0)) * 0.5, 0.0, 7.0));
			}
			reorderThreadEXT(hitObject, (hint << 3u) | bin, 4);
		} else {
			reorderThreadEXT(hitObject, hint, 1);
		}
	}

	hitObjectExecuteShaderEXT(hitObject, 0);
#else
	traceRayEXT(tlas, RT_RAY_FLAGS | RT_OMM_RAY_FLAGS(get_diffuse_bounces(payload.packed_bounces_flags)), mask, 0, 0, 0, ray_origin, tmin, ray_dir, max_distance, 0);
#endif
}

/// Shades the G-buffer's surface or a nearer procedural surface as the primary
/// hit. Returns false when the raster drew nothing so the caller traces the
/// full primary ray (sky and procedural geometry without a raster backdrop).
/// ray_origin and ray_dir are the raygen's own primary ray, anchor-relative;
/// the hit is rebuilt along it from the raster's reverse-Z depth, never from
/// a written position, so it sits on the same float grid as the TLAS. Its depth
/// error grows with the distance, which offset_ray_origin's floor covers.
bool shade_gbuffer_primary(ivec2 pixel, vec2 ndc, vec3 ray_origin, vec3 ray_dir) {
	float depth = texelFetch(sampler2D(gbuffer_depth, SAMPLER_NEAREST_CLAMP), pixel, 0).r;
	if (depth <= 0.0) {
		return false;
	}

	vec4 view_pos = scene_data_block.data.inv_projection_matrix * vec4(ndc, depth, 1.0);
	float hit_t = length(view_pos.xyz / view_pos.w);

	// Procedural surfaces have no raster representation. Shade the nearest one
	// before the raster surface, using its normal closest-hit depth/velocity/RR
	// writes. A miss must leave all of those images and the path untouched.
	if (get_rt_param(RT_PARAM_HAS_PROCEDURAL_INSTANCES) > 0.5 && hit_t > 0.001) {
		payload.packed_bounces_flags |= PRIMARY_PROBE_FLAG;
		trace_path_ray(ray_origin, ray_dir, RT_MASK_PROCEDURAL, min(hit_t, 10000.0));
		bool procedural_hit = (payload.packed_bounces_flags & PRIMARY_PROBE_FLAG) != 0u;
		payload.packed_bounces_flags &= ~PRIMARY_PROBE_FLAG;
		if (procedural_hit) {
			return true;
		}
	}

	// The RT depth image is the frame's depth after the trace (a hole in the G-buffer, procedural
	// geometry or the sky, gets the traced value there), so the raster's depth goes in for this pixel.
	if (is_sample_zero(payload.packed_bounces_flags)) {
		imageStore(rt_depth_image, pixel, vec4(depth));
	}

	vec4 albedo_specular = texelFetch(sampler2D(gbuffer_albedo_specular, SAMPLER_NEAREST_CLAMP), pixel, 0);
	vec4 normal_roughness = texelFetch(sampler2D(gbuffer_normal_roughness, SAMPLER_NEAREST_CLAMP), pixel, 0);
	vec4 geo_normal_metallic = texelFetch(sampler2D(gbuffer_geo_normal_metallic, SAMPLER_NEAREST_CLAMP), pixel, 0);
	vec4 emission_ior = texelFetch(sampler2D(gbuffer_emission_ior, SAMPLER_NEAREST_CLAMP), pixel, 0);
	vec4 transmission_flags = texelFetch(sampler2D(gbuffer_transmission_flags, SAMPLER_NEAREST_CLAMP), pixel, 0);
	vec4 absorption = texelFetch(sampler2D(gbuffer_absorption, SAMPLER_NEAREST_CLAMP), pixel, 0);
	vec4 scattering = texelFetch(sampler2D(gbuffer_scattering, SAMPLER_NEAREST_CLAMP), pixel, 0);

	HitData h;
	h.hit_pos = ray_origin + ray_dir * hit_t;
	h.geometry_normal = normalize(geo_normal_metallic.xyz);
	h.tangent = vec3(0.0);
	h.bitangent = vec3(0.0);
	h.uv = vec2(0.0);
	h.uv2 = vec2(0.0);
	h.color = vec4(1.0);
	h.is_front_face = transmission_flags.a > 0.5;
	h.custom_group = false;
	h.geometry_idx = 0u;
	h.primitive_id = 0u;
	h.texture_lod = RT_TEXTURE_LOD_BASE; // The raster sampled the surface's textures.
	h.texture_footprint = RT_TEXTURE_LOD_BASE; // And drew its decals.
	h.texture_grad = vec4(0.0);
	h.texture_lod2 = RT_TEXTURE_LOD_BASE;
	h.texture_grad2 = vec4(0.0);
	h.strand_inset = 0.0;
	h.strand_axis = vec3(0.0);

	MaterialResult m;
	m.albedo = albedo_specular.rgb;
	m.alpha = 1.0;
	m.roughness = normal_roughness.a;
	m.metalness = geo_normal_metallic.a;
	m.specular = albedo_specular.a;
	m.emissive = emission_ior.rgb;
	m.decal_emissive = vec3(0.0); // The raster's decals are in the G-buffer's emission, which the camera ray sees at full weight.
	m.normal = normalize(normal_roughness.xyz);
	m.transmission = clamp(transmission_flags.rgb, vec3(0.0), vec3(1.0));
	m.ior = max(emission_ior.a, 1.0);
	m.absorption = max(absorption.rgb, vec3(0.0));
	m.scattering = clamp(scattering.rgb, vec3(0.0), vec3(1.0));
	m.diffuse_transmission = vec3(0.0); // The G-buffer carries no backlight (D103).
	m.anisotropy = 0.0; // Nor anisotropy (D118 MM1).
	m.anisotropy_tangent = vec3(0.0);
	m.clearcoat = 0.0; // Nor a coat (D118 MM2).
	m.clearcoat_roughness = 0.0;
	m.clearcoat_normal = h.geometry_normal;
	m.rim = 0.0; // Nor a rim (D118 MM3).
	m.rim_tint = 0.0;
	m.subsurface = 0.0; // Nor subsurface scattering (D120).
	m.subsurface_skin = false;
	material_clear_hair(m); // Nor hair (D126 H2): the raster draws strands as lines.

#ifdef RT_DEBUG_ENABLED
	int vis_mode = int(get_rt_param(RT_PARAM_VIS_MODE));
	if (vis_mode != 0) {
		vec3 V = -ray_dir;
		float NdotV = max(dot(m.normal, V), 0.0001);
		debug_visualize(vis_mode, h.geometry_normal, m.normal, vec3(0.5, 0.5, 1.0),
				h.tangent, h.bitangent, h.uv, m.albedo, vec3(1.0, m.roughness, m.metalness), m.metalness, m.roughness, m.specular, m.emissive, V, NdotV, hit_t, h.is_front_face, h.texture_lod);
		return true;
	}
#endif // RT_DEBUG_ENABLED
	// The surface as the hit shaders will hand it back (P3a): the record of the G-buffer's own precision, so the codec
	// runs on a supported path and loses only the normals' octahedral rounding.
#ifdef RT_SURFACE_CODEC_CHECK
	surface_codec_check_hit(h, m, ray_origin, ray_dir, hit_t);
#endif
	const SurfaceRecord record = surface_pack(h, m, hit_t);
	HitData h_record;
	MaterialResult m_record;
	surface_unpack(record, ray_origin, ray_dir, h_record, m_record);
	PathState ps = path_unpack(payload);
	shade_and_bounce(h_record, m_record, ray_dir, hit_t, ps);
	path_pack(payload, ps);
	return true;
}

/// The camera segment's media (D37): the Environment's analytic fog at the primary hit and the
/// froxel volumetric fog, composed as the raster composes them, over the radiance of the whole
/// path. A primary miss keeps the miss stage's sky tint for the analytic fog and takes the
/// froxel's far slice scaled by volumetric_fog_sky_affect, as the sky pass does. Every later
/// segment is charged in shade_and_bounce.
vec3 apply_camera_media(vec3 radiance, vec2 uv, vec3 ray_origin, vec3 ray_dir, float hit_t, bool primary_miss) {
	if (camera_medium_absorption.w > 0.5) {
		// The camera is inside a medium (D49): the camera segment was charged with it in
		// shade_and_bounce, and neither the analytic fog nor the froxel (both air) applies.
		return radiance;
	}
	vec4 fog = vec4(0.0, 0.0, 0.0, 1.0); // Premultiplied in-scatter, transmittance.
	float depth_z = scene_data_block.data.z_far;
	if (!primary_miss) {
		mat4 view_mat = transpose(mat4(
				scene_data_block.data.view_matrix[0],
				scene_data_block.data.view_matrix[1],
				scene_data_block.data.view_matrix[2],
				vec4(0.0, 0.0, 0.0, 1.0)));
		vec3 vertex = (view_mat * vec4(ray_origin + ray_dir * hit_t, 1.0)).xyz;
		depth_z = -vertex.z;
		if ((RT_FLAGS & RT_FLAG_FOG_ENABLED) != 0u) {
			vec4 f = fog_process(scene_data_block.data, vertex);
			fog = vec4(f.rgb * f.a, 1.0 - f.a);
		}
	}
	if (get_rt_param(RT_PARAM_VFOG_ENABLED) > 0.5) {
		vec4 vf = vec4(0.0, 0.0, 0.0, 1.0);
		// A miss takes the far slice outright, as the sky pass does; mapping z_far through the
		// volume's own length would land mid-volume whenever the volume is shorter than the far plane.
		vec3 fog_pos = vec3(uv, primary_miss ? 1.0 : depth_z * get_rt_param(RT_PARAM_VFOG_INV_LENGTH));
		if (fog_pos.z >= 0.0) {
			if (fog_pos.z < 1.0) {
				fog_pos.z = pow(fog_pos.z, get_rt_param(RT_PARAM_VFOG_INV_SPREAD));
			}
			vf = textureLod(sampler3D(volumetric_fog_texture, SAMPLER_LINEAR_CLAMP), fog_pos, 0.0);
		}
		if (primary_miss) {
			vf = mix(vec4(0.0, 0.0, 0.0, 1.0), vf, get_rt_param(RT_PARAM_VFOG_SKY_AFFECT));
		}
		fog = vec4(fog.rgb * vf.a + vf.rgb, fog.a * vf.a);
	}
	return radiance * fog.a + fog.rgb;
}

void main() {
	uvec2 pixel = gl_LaunchIDEXT.xy;
	const vec2 pixel_center = vec2(pixel) + vec2(0.5);
	const vec2 in_uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);
	vec2 d = in_uv * 2.0 - 1.0;

	mat4 inv_view = transpose(mat4(scene_data_block.data.inv_view_matrix[0],
			scene_data_block.data.inv_view_matrix[1],
			scene_data_block.data.inv_view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

	vec4 target = scene_data_block.data.inv_projection_matrix * vec4(d.x, d.y, 1.0, 1.0);
	vec4 origin = inv_view * vec4(0.0, 0.0, 0.0, 1.0);
	vec4 direction = inv_view * vec4(normalize(target.xyz), 0);
	// The ray cone's spread at the camera (D116), the same for every sample of the pixel.
	const float cone_spread = rt_cone_primary_spread(scene_data_block.data.inv_projection_matrix, d, float(gl_LaunchSizeEXT.y));

	// Sample count from specialization constant, frame index from uniform
	const uint samples_per_pixel = RT_GET_SAMPLE_COUNT();
	uint frame_index = uint(get_rt_param(RT_PARAM_FRAME_INDEX));

	// Accumulate multiple samples per pixel
	vec3 total_radiance = vec3(0.0);
	// The camera segment every sample shares: its hit distance, or a miss (D37).
	float primary_t = 0.0;
	bool primary_miss = false;

	const uint max_bounces = RT_GET_MAX_BOUNCES();

	// The primary surface comes from the G-buffer in that raygen variant (RT_GBUFFER_PRIMARY, P1e), so the
	// default raygen carries no G-buffer path; the traced primary ray otherwise.
#ifdef RT_GBUFFER_PRIMARY
	const bool primary_from_gbuffer = true;
#else
	const bool primary_from_gbuffer = false;
#endif

	// The camera's medium (D49), the same for every sample: its in-scatter at a horizontal
	// interface, charged over the light's leg from the medium's top down to the camera (the
	// first directional light's elevation, the leg formula NEE uses), so the water darkens
	// with the camera's depth.
	const bool camera_in_medium = camera_medium_absorption.w > 0.5;
	vec3 camera_medium_inscatter = vec3(0.0);
	if (camera_in_medium) {
		float sun_y = 1.0;
		if (scene_data_block.data.directional_light_count > 0u) {
			sun_y = max(-normalize(rt_lights[0].position).y, 0.1);
		}
		float leg = max(camera_medium_scatter.w - origin.y, 0.0) / sun_y;
		camera_medium_inscatter = camera_medium_scatter.xyz * medium_irradiance(vec3(0.0, 1.0, 0.0)) * exp(-camera_medium_absorption.xyz * leg);
	}

	// TODO: when we have a spp > 0 the first raycast is always identical,
	// we should move it out of the loop

	[[dont_unroll]] for (uint sample_idx = 0u; sample_idx < samples_per_pixel; sample_idx++) {
		PathState ps;
		ps.radiance = vec3(0.0);
		ps.throughput = vec3(1.0);
		ps.packed_bounces_flags = (sample_idx == 0u) ? set_sample_zero(0u) : 0u;
		ps.rng_state = init_rng(pixel, frame_index, sample_idx, samples_per_pixel);
		ps.hit_t = 0.0;
		ps.bsdf_pdf = 0.0;
		ps.cone_width = 0.0;
		ps.cone_spread = cone_spread;
#ifdef RT_SUBSURFACE_COMPILED
		ps.exit_position = vec3(0.0);
#endif
		ps.sky_probability = 0.0;
		ps.tree_total = 0.0;
		path_clear_medium(ps);
		if (camera_in_medium) {
			// The camera is inside a medium (D49): the path starts in it and leaves it through the
			// first back face it refracts out of.
			ps.packed_bounces_flags = set_in_medium(ps.packed_bounces_flags, true);
			path_set_medium(ps, camera_medium_absorption.xyz, camera_medium_inscatter, camera_medium_scatter.w);
		}

		vec3 ray_origin = origin.xyz;
		vec3 ray_dir = direction.xyz;

		[[dont_unroll]] for (uint bounce = 0u; bounce <= max_bounces; bounce++) {
			path_pack(payload, ps);

			if (bounce == 0u && primary_from_gbuffer && shade_gbuffer_primary(ivec2(pixel), d, ray_origin, ray_dir)) {
				ps = path_unpack(payload);
				primary_t = ps.hit_t;
				if (is_path_terminated(ps.packed_bounces_flags)) {
					break;
				}
				vec3 hit_pos = ray_origin + ray_dir * ps.hit_t;
#ifdef RT_SUBSURFACE_COMPILED
				if ((ps.packed_bounces_flags & SUBSURFACE_EXIT_FLAG) != 0u) {
					hit_pos = ps.exit_position; // Where the subsurface walk left the surface (D120).
				}
#endif
				ray_origin = path_next_origin(hit_pos, ps);
				ray_dir = ps.next_ray_dir;
				continue;
			}

			trace_path_ray(ray_origin, ray_dir, 0xFF, 10000.0);

			ps = path_unpack(payload);
			if (bounce == 0u) {
				primary_t = ps.hit_t;
				primary_miss = (ps.packed_bounces_flags & PRIMARY_MISS_FLAG) != 0u;
			}
			if (is_path_terminated(ps.packed_bounces_flags)) {
				break;
			}
			// Reconstruct the next ray origin from the current ray + hit distance, apply bias
			vec3 hit_pos = ray_origin + ray_dir * ps.hit_t;
#ifdef RT_SUBSURFACE_COMPILED
			if ((ps.packed_bounces_flags & SUBSURFACE_EXIT_FLAG) != 0u) {
				hit_pos = ps.exit_position; // Where the subsurface walk left the surface (D120).
			}
#endif
			ray_origin = path_next_origin(hit_pos, ps);
			ray_dir = ps.next_ray_dir;
		}

		total_radiance += ps.radiance;
	}

	vec3 final_radiance = total_radiance / float(samples_per_pixel);

#ifdef RT_DEBUG_ENABLED
	if (int(get_rt_param(RT_PARAM_VIS_MODE)) == 0)
#endif
	{
		final_radiance = apply_camera_media(final_radiance, in_uv, origin.xyz, direction.xyz, primary_t, primary_miss);
	}

	imageStore(image, ivec2(pixel), vec4(final_radiance, 1.0));
}

#[miss]

#version 460

#VERSION_DEFINES

#pragma shader_stage(miss)
#extension GL_EXT_ray_tracing : enable
#ifdef USE_CLUSTER_AS
#extension GL_NV_cluster_acceleration_structure : require
#endif

#define GLSL 1
#define RT_STAGE_MISS 1

// clang-format off
#include "raytracing_inc.glsl"
#include "../scene_data_inc.glsl"
#include "brdf_inc.glsl"
#include "raytracing_common_inc.glsl"
// clang-format on

layout(location = 0) rayPayloadInEXT PathPayload payload;

#ifdef USE_RADIANCE_OCTMAP_ARRAY

layout(set = 0, binding = 7) uniform texture2DArray radiance_octmap;
layout(set = 0, binding = 8) uniform sampler radiance_sampler;

// Array-backed octahedral radiance: roughness selects the prefiltered array
// layer (with a blend between adjacent layers). Ray/hit stages have no
// screen-space derivatives, so the intra-layer mip is fixed at 0.0; roughness is
// represented entirely by the layer selection.
vec3 radiance_octmap_sample(vec2 p_oct_uv, float p_roughness) {
	float layer;
	float blend = modf(clamp(p_roughness, 0.0, 1.0) * MAX_ROUGHNESS_LOD, layer);
	vec3 a = textureLod(sampler2DArray(radiance_octmap, radiance_sampler), vec3(p_oct_uv, layer), 0.0).rgb;
	vec3 b = textureLod(sampler2DArray(radiance_octmap, radiance_sampler), vec3(p_oct_uv, layer + 1.0), 0.0).rgb;
	return mix(a, b, blend);
}

#else

layout(set = 0, binding = 7) uniform texture2D radiance_octmap;
layout(set = 0, binding = 8) uniform sampler radiance_sampler;

// Single-texture octahedral radiance: roughness maps to the mip LOD.
vec3 radiance_octmap_sample(vec2 p_oct_uv, float p_roughness) {
	return textureLod(sampler2D(radiance_octmap, radiance_sampler), p_oct_uv, clamp(p_roughness, 0.0, 1.0) * MAX_ROUGHNESS_LOD).rgb;
}

#endif // USE_RADIANCE_OCTMAP_ARRAY

// clang-format off
#include "raytracing_sky_inc.glsl"
#include "raytracing_lights_inc.glsl"
#include "raytracing_light_tree_inc.glsl"
// clang-format on

void main() {
	if ((payload.packed_bounces_flags & PRIMARY_PROBE_FLAG) != 0u) {
		payload.packed_bounces_flags &= ~PRIMARY_PROBE_FLAG;
		return;
	}

	PathState ps = path_unpack(payload);

#if !defined(USE_SER)
	// Shadow rays that miss mean the light is visible (no occluder).
	if (is_shadow_ray(ps.packed_bounces_flags)) {
		ps.radiance = vec3(1.0);
		path_pack(payload, ps);
		return;
	}
#endif

	// Miss always ends the path.
	ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
	if (get_total_bounces(ps.packed_bounces_flags) == 0u) {
		ps.packed_bounces_flags |= PRIMARY_MISS_FLAG;
	}

#ifdef RT_DEBUG_ENABLED
	{
		int VIS_MODE = int(get_rt_param(RT_PARAM_VIS_MODE));
		// Specular hit distance: the first-bounce closest_hit pre-seeds the
		// "no hit" color into radiance, so a missed reflection just keeps it.
		if (VIS_MODE == 13 && get_total_bounces(ps.packed_bounces_flags) > 0u) {
			path_pack(payload, ps);
			return;
		}
	}
#endif // RT_DEBUG_ENABLED

	// Primary ray miss: write depth, velocity, and DLSS RR defaults (sample 0 only).
	{
		uint total_bounces = get_total_bounces(ps.packed_bounces_flags);
		if (total_bounces == 0u && is_sample_zero(ps.packed_bounces_flags)) {
			ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);

			imageStore(rt_depth_image, pixel, vec4(0.0));

			// Sky velocity: reproject a far-plane point using unjittered VPs.
			{
				vec3 far_world = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * 10000.0;
				vec2 curr_uv = project_uv(far_world, curr_vp_unjittered);
				vec2 prev_uv = project_uv(far_world, prev_vp_unjittered);
				imageStore(rt_velocity_image, pixel, vec4(prev_uv - curr_uv, 0.0, 0.0));
			}
		}
	}

	mat3 camera_basis = mat3(scene_data_block.data.inv_view_matrix);
	mat3 world_to_sky = scene_data_block.data.radiance_inverse_xform * camera_basis;
	vec3 sky_dir = world_to_sky * gl_WorldRayDirectionEXT;

	// The sky along the ray (D54): the one evaluation the sky light sample reads too.
	vec3 sky_color = sky_radiance(sky_dir);

	// The sky a scattered path finds is indirect light under the clamp (D53); the sky the camera
	// sees, through any delta interfaces, is the image. The medium's in-scatter below is charged
	// analytically and stays outside the clamp, as it does on a hit.
	vec3 sky_add = ps.throughput * sky_color;
	// The sky a bounce found on its own is weighed against the sky light sample that competes
	// for it (D54): the camera ray and a delta lobe keep the whole weight, as with an emitter.
	if (sky_light_enabled() && get_total_bounces(ps.packed_bounces_flags) > 0u && ps.bsdf_pdf != RT_BSDF_PDF_DELTA) {
		// The sky's share as the previous vertex drew it (GPU audit 4a), carried in the payload: the tree's
		// (D55) or the share's (D65) for the receiver it was judged at. A vertex that gave the sky no share
		// reads no leaf (4c): its density is 0 whatever the leaf, and most vertices take the share 0 (D62).
		sky_add *= sky_mis_weight(ps.bsdf_pdf, (ps.sky_probability > 0.0) ? ps.sky_probability * sky_pdf(sky_dir) : 0.0);
	}
	if (is_scattered(ps.packed_bounces_flags)) {
		sky_add = rt_clamp_contribution(sky_add, get_rt_param(RT_PARAM_CLAMP_INDIRECT));
	}
	if (is_in_medium(ps.packed_bounces_flags)) {
		// A miss inside a medium (D49): the segment is charged over the trace's own reach (the
		// 10000 m tmax), so a channel the medium does not extinguish stays continuous with a far
		// hit and every other one takes the full column; the guides below carry the result.
		vec3 medium_inscatter = path_medium_inscatter(ps);
		vec3 t = exp(-path_medium_absorption(ps) * 10000.0);
		sky_color = mix(medium_inscatter, sky_color, t);
		sky_add = mix(ps.throughput * medium_inscatter, sky_add, t);
	}

#ifdef DLSS_RR_ENABLED
	{
		uint total_bounces = get_total_bounces(ps.packed_bounces_flags);
		if (total_bounces == 0u && is_sample_zero(ps.packed_bounces_flags)) {
			ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);
			imageStore(dlss_rr_diffuse_albedo, pixel, vec4(DLSSRR_encodeDiffuseAlbedo(sky_color), 1.0));
			imageStore(dlss_rr_specular_albedo, pixel, vec4(0.0));
			imageStore(dlss_rr_normal_roughness, pixel, vec4(-gl_WorldRayDirectionEXT, 0.0));
			imageStore(dlss_rr_specular_hit_dist, pixel, vec4(-1.0));
		}
	}
#endif

#ifdef RT_DEBUG_ENABLED
	{
		int VIS_MODE = int(get_rt_param(RT_PARAM_VIS_MODE));
		if (VIS_MODE == 20) {
			ps.radiance = vec3(1.0);
		} else if (VIS_MODE == 0) {
			ps.radiance += sky_add;
		} else {
			ps.radiance = sky_color;
		}
	}
#else
	ps.radiance += sky_add;
#endif // RT_DEBUG_ENABLED

	path_pack(payload, ps);
}

#[closest_hit]

#version 460

#VERSION_DEFINES

#pragma shader_stage(closest_hit)
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_nonuniform_qualifier : require
#ifdef USE_SER
#extension GL_EXT_shader_invocation_reorder : enable
#endif

#ifdef USE_CLUSTER_AS
#extension GL_NV_cluster_acceleration_structure : require
#endif
#ifdef USE_OMM_INDIRECT_2_STATE
#extension GL_EXT_opacity_micromap : require
#endif
#ifdef USE_POSITION_FETCH
#extension GL_EXT_ray_tracing_position_fetch : require
#endif
#ifdef USE_LSS
// A line list's hits (D126) read their segment's end points and radii.
#extension GL_NV_linear_swept_spheres : require
#endif
#ifdef RT_SURFACE_CODEC_CHECK
// The surface record's check counts per subgroup (P3a), an asserts build only.
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif

#define GLSL 1
#define RT_STAGE_CLOSEST_HIT 1

// clang-format off
#include "raytracing_inc.glsl"
#include "../scene_data_inc.glsl"
#include "brdf_inc.glsl"
#include "raytracing_common_inc.glsl"
// clang-format on

#define attribs hit_attribs.bary_or_uv
#define RT_HIT_ATTRIBS_DECLARED

#include "raytracing_hit_inc.glsl"

layout(set = 0, binding = 1) uniform accelerationStructureEXT tlas;
layout(location = 0) rayPayloadInEXT PathPayload payload;

layout(set = 1, binding = 0) uniform texture2D bindless_textures[];

#include "raytracing_samplers_inc.glsl"

// clang-format off
layout(set = 0, binding = 3, std430) readonly buffer GeometryBuffer {
	GeometryData geometries[];
};

layout(set = 0, binding = 4, std430) readonly buffer MotionIndexBuffer {
	int motion_indices[];
};

layout(set = 0, binding = 5, std430) readonly buffer MaterialBuffer {
	MaterialData materials[];
};
// clang-format on

#include "raytracing_lights_inc.glsl"
#include "raytracing_emissive_inc.glsl"

// clang-format off
layout(set = 0, binding = 32, std430) readonly buffer MotionTransforms {
	InstanceMotionData motion_transforms[];
};
// clang-format on

#ifdef USE_RADIANCE_OCTMAP_ARRAY

layout(set = 0, binding = 7) uniform texture2DArray radiance_octmap;
layout(set = 0, binding = 8) uniform sampler radiance_sampler;

// Array-backed octahedral radiance: roughness selects the prefiltered array
// layer (with a blend between adjacent layers). Ray/hit stages have no
// screen-space derivatives, so the intra-layer mip is fixed at 0.0; roughness is
// represented entirely by the layer selection.
vec3 radiance_octmap_sample(vec2 p_oct_uv, float p_roughness) {
	float layer;
	float blend = modf(clamp(p_roughness, 0.0, 1.0) * MAX_ROUGHNESS_LOD, layer);
	vec3 a = textureLod(sampler2DArray(radiance_octmap, radiance_sampler), vec3(p_oct_uv, layer), 0.0).rgb;
	vec3 b = textureLod(sampler2DArray(radiance_octmap, radiance_sampler), vec3(p_oct_uv, layer + 1.0), 0.0).rgb;
	return mix(a, b, blend);
}

#else

layout(set = 0, binding = 7) uniform texture2D radiance_octmap;
layout(set = 0, binding = 8) uniform sampler radiance_sampler;

// Single-texture octahedral radiance: roughness maps to the mip LOD.
vec3 radiance_octmap_sample(vec2 p_oct_uv, float p_roughness) {
	return textureLod(sampler2D(radiance_octmap, radiance_sampler), p_oct_uv, clamp(p_roughness, 0.0, 1.0) * MAX_ROUGHNESS_LOD).rgb;
}

#endif // USE_RADIANCE_OCTMAP_ARRAY

// clang-format off
#include "raytracing_sky_inc.glsl"
#include "raytracing_light_tree_inc.glsl"
#include "raytracing_material_eval_inc.glsl"
#include "raytracing_closest_hit_common_inc.glsl"
#include "raytracing_decals_inc.glsl"
#include "raytracing_surface_inc.glsl"
// clang-format on

/// The shadow rays shade_direct_light left to the pipeline (GPU audit P3e), traced at the end of the hit with the payload
/// packed, one call site whatever their count; the payload's radiance becomes the outcome's, rounded as the pack rounds.
void shade_resolve_shadows() {
	uint pending = rt_shadow_pending;
	if (pending == 0u) {
		return;
	}
	uvec2 radiance1 = path_pack_radiance(rt_nee_radiance1);
	uvec2 radiance2 = path_pack_radiance(rt_nee_radiance2);
	uvec2 radiance3 = path_pack_radiance(rt_nee_radiance3);
	uint visible = 0u;
	// A loop over the waiting slots, its count the data's, so the trace is one call site.
	uint left = pending;
	while (left != 0u) {
		uint s = uint(findLSB(left));
		left &= left - 1u;
		bool first = s == 0u;
		if (lights_trace_shadow_ray_pipeline(first ? rt_shadow_origin0 : rt_shadow_origin1, first ? rt_shadow_dir0 : rt_shadow_dir1, first ? rt_shadow_distance0 : rt_shadow_distance1, rt_shadow_omm_flags)) {
			visible |= 1u << s;
		}
	}
	if (visible != 0u) {
		uvec2 r = (visible == 3u) ? radiance3 : ((visible == 1u) ? radiance1 : radiance2);
		payload.packed_rt[0] = r.x;
		payload.packed_rt[1] = (payload.packed_rt[1] & 0xFFFF0000u) | r.y;
	}
}

/// Shades the evaluated surface and hands the path back to the raygen in the payload. A function of its own, so a custom
/// hit group's fragment code, spliced into main, cannot shadow its names.
void shade_hit(HitData h, MaterialResult m) {
#ifdef RT_SURFACE_CODEC_CHECK
	surface_codec_check_hit(h, m, gl_WorldRayOriginEXT, gl_WorldRayDirectionEXT, gl_HitTEXT);
#endif
	PathState ps = path_unpack(payload);
	shade_and_bounce(h, m, gl_WorldRayDirectionEXT, gl_HitTEXT, ps);
	path_pack(payload, ps);
	shade_resolve_shadows(); // After the pack: it replaces the packed radiance with a pending outcome's (P3e).
}

// ============================================================================
// CUSTOM SHADER GLOBALS (injected by ShaderCompiler for HG1+)
// ============================================================================
#ifdef RT_CUSTOM_HIT_GROUP
#include "raytracing_custom_globals_inc.glsl"
#endif

// ============================================================================
// MAIN
// ============================================================================
void main() {
	// A shader that calls texture_projected() needs the cone's footprint at a mesh without UVs too (D116).
	// And the elliptical footprint's gradients only where a fetch reads them (D116): in a custom shader with a
	// lookup rewritten to textureGrad, in HG0 for an anisotropic-filtered material.
#ifdef RT_CUSTOM_HIT_GROUP
#if defined(RT_CUSTOM_TEXTURE_GRAD) && !defined(RT_CUSTOM_UV_WRITTEN)
	const bool gradients = true;
#else
	const bool gradients = false;
#endif
#else
	const bool gradients = (materials[gl_InstanceCustomIndexEXT].flags & 64u) != 0u;
#endif
	// The second UV set only where it is read (D118): by a custom shader that uses UV2, in HG0 by a detail layer or the
	// emission on UV2 (flag bits 17 and 18).
#ifdef RT_CUSTOM_HIT_GROUP
#ifdef RT_CUSTOM_UV2_USED
	const bool fetch_uv2 = true;
#else
	const bool fetch_uv2 = false;
#endif
#else
	const bool fetch_uv2 = (materials[gl_InstanceCustomIndexEXT].flags & ((1u << 17) | (1u << 18))) != 0u;
#endif
	// The vertex colour where HG0's material multiplies by it (D126 H3), converted per corner as BaseMaterial3D's vertex()
	// does with vertex_color_is_srgb; a custom hit group fetches every attribute.
	const uint hg0_flags = materials[gl_InstanceCustomIndexEXT].flags;
	const uint color_fetch = (hg0_flags & RT_MAT_FLAG_VERTEX_COLOR) != 0u ? (FETCH_COLOR | ((hg0_flags & RT_MAT_FLAG_VERTEX_COLOR_SRGB) != 0u ? FETCH_COLOR_SRGB : 0u)) : 0u;
#ifdef RT_CUSTOM_PROJECTED_LOD
	HitData h = compute_hit_data(true, gradients, fetch_uv2, color_fetch);
#else
	HitData h = compute_hit_data(false, gradients, fetch_uv2, color_fetch);
#endif
	write_primary_hit_depth(h.hit_pos);
	write_primary_hit_velocity(h.hit_pos);

#ifdef RT_CUSTOM_HIT_GROUP
	uint rt_geometry_idx = h.geometry_idx;
	vec3 rt_hit_pos = h.hit_pos;
	vec2 rt_uv = h.uv;
	vec2 rt_uv2 = h.uv2;
	vec4 rt_color = h.color;
	vec3 rt_normal = h.geometry_normal;
	vec3 rt_tangent = h.tangent;
	vec3 rt_bitangent = h.bitangent;
	bool rt_front_face = h.is_front_face;
	// The level the fragment's texture() calls at the mesh's UVs read and the footprint its texture_projected()
	// calls read (D116), handed over after the vertex function.
	float rt_hit_texture_lod = rt_texture_lod_enabled() ? h.texture_lod : RT_TEXTURE_LOD_BASE;
	float rt_hit_texture_footprint = rt_texture_lod_enabled() ? h.texture_footprint : RT_TEXTURE_LOD_BASE;
	// Zero unless the anisotropic switch is on and the path has not bounced diffusely (GPU audit hot 8).
	vec4 rt_hit_texture_grad = h.texture_grad;
	float rt_hit_texture_lod2 = rt_texture_lod_enabled() ? h.texture_lod2 : RT_TEXTURE_LOD_BASE; // At UV2 (D118).

#include "raytracing_custom_fragment_inc.glsl"

	// Build MaterialResult from fragment outputs.
	MaterialResult m;
	m.albedo = albedo;
	m.alpha = alpha;
	m.roughness = roughness;
	m.metalness = metallic;
	m.specular = specular;
	m.emissive = emission * scene_data_block.data.emissive_exposure_normalization;
	m.decal_emissive = vec3(0.0);
	m.normal = normalize(mat3(inv_view_matrix) * normal); // view space -> world space
	m.transmission = clamp(transmission, vec3(0.0), vec3(1.0));
	m.ior = max(ior, 1.0);
	m.absorption = max(absorption, vec3(0.0));
	m.scattering = clamp(scattering, vec3(0.0), vec3(1.0));
	m.diffuse_transmission = vec3(0.0); // No BACKLIGHT here: the surface record's thin and refracting words overlay (raytracing_surface_inc.glsl).
	// VAL: ANISOTROPY and ANISOTROPY_FLOW (D118 MM1) against the fragment's TANGENT and BINORMAL, as the raster turns
	// them; in a shader that writes neither it is the constant 0, on which the compiler drops the lobe. Capped at 1,
	// past which the raster's aspect soon has no root; below -1 it stretches further along B, as in the raster.
	m.anisotropy = min(anisotropy, 1.0);
	m.anisotropy_tangent = vec3(0.0);
	if (m.anisotropy != 0.0) {
		m.anisotropy_tangent = rt_anisotropy_tangent(mat3(inv_view_matrix) * tangent, mat3(inv_view_matrix) * binormal, anisotropy_flow, m.anisotropy);
	}
	// VAL: CLEARCOAT and CLEARCOAT_ROUGHNESS (D118 MM2) about the fragment's NORMAL before the normal map, the raster's
	// geo_normal; in a shader that writes no CLEARCOAT it is the constant 0, on which the compiler drops the lobe.
	m.clearcoat = clamp(clearcoat, 0.0, 1.0);
	m.clearcoat_roughness = clamp(clearcoat_roughness, 0.0, 1.0);
	m.clearcoat_normal = m.normal;
	// VAL: RIM and RIM_TINT (D118 MM3), read as a sheen layer whose colour shade_and_bounce takes from the albedo
	// after the decals; in a shader that writes no RIM it is the constant 0, on which the compiler drops the lobe.
	m.rim = clamp(rim, 0.0, 1.0);
	m.rim_tint = clamp(rim_tint, 0.0, 1.0);
	// VAL: SSS_STRENGTH (D120), the diffuse share that walks under the surface, over skin's channel ratios under
	// render_mode sss_mode_skin.
	m.subsurface = 0.0;
	m.subsurface_skin = false;
#ifdef RT_SUBSURFACE_COMPILED
	m.subsurface = clamp(sss_strength, 0.0, 1.0);
#ifdef RT_CUSTOM_SSS_SKIN
	m.subsurface_skin = true;
#endif
#endif
	material_clear_hair(m); // VAL: the hair BCSDF is BaseMaterial3D's feature (D126 H2); a custom shader has no twin yet.

	// Apply normal map if it was written.
	if (normal_map != vec3(0.5, 0.5, 1.0)) {
		vec3 ts_normal;
		ts_normal.xy = normal_map.xy * 2.0 - 1.0;
		ts_normal.z = sqrt(max(0.0, 1.0 - dot(ts_normal.xy, ts_normal.xy)));
		m.normal = apply_normal_map(h, ts_normal, normal_map_depth);
	}

	// Decals over the evaluated material (D104).
	apply_decals(h, m);

#ifdef RT_DEBUG_ENABLED
	{
		int VIS_MODE = int(get_rt_param(RT_PARAM_VIS_MODE));
		if (VIS_MODE != 0) {
			vec3 V = -gl_WorldRayDirectionEXT;
			float NdotV = max(dot(m.normal, V), 0.0001);
			vec3 orm = vec3(1.0, m.roughness, m.metalness);
			debug_visualize(VIS_MODE, h.geometry_normal, m.normal, normal_map,
					h.tangent, h.bitangent, h.uv, m.albedo, orm, m.metalness, m.roughness, m.specular, m.emissive + m.decal_emissive, V, NdotV, gl_HitTEXT, h.is_front_face, h.texture_lod);
			return;
		}
	}
#endif // RT_DEBUG_ENABLED
	shade_hit(h, m);
#else
	// HG0: StandardMaterial3D evaluation.
	MaterialData mat = materials[h.geometry_idx];
	vec2 uv = h.uv * mat.uv1_scale + mat.uv1_offset;
	// The ray cone's level for this material (D116): the material's UV scale is part of the texture's
	// density on the surface.
	float material_lod = RT_TEXTURE_LOD_BASE;
	if (h.texture_lod > RT_TEXTURE_LOD_BASE) {
		material_lod = h.texture_lod + 0.5 * log2(max(abs(mat.uv1_scale.x * mat.uv1_scale.y), 1e-20));
	}
	const float lod = rt_texture_lod_enabled() ? material_lod : RT_TEXTURE_LOD_BASE;
	// And its elliptical footprint, the mesh's gradients through the same scale.
	const vec4 grad = h.texture_grad * vec4(mat.uv1_scale, mat.uv1_scale);
	// VAL: the second UV set (D118 MM4), for a detail layer (flag bit 17) or the emission (bit 18) on UV2, through
	// uv2_scale and uv2_offset, which the packer leaves at the identity unless the detail UV layer is UV2, where
	// BaseMaterial3D's vertex() applies them, detail on or off; at that set's level and footprint. UV1's otherwise, unread.
	vec2 uv2 = uv;
	float lod2 = lod;
	vec4 grad2 = grad;
	if ((mat.flags & ((1u << 17) | (1u << 18))) != 0u) {
		uv2 = h.uv2 * mat.uv2_scale + mat.uv2_offset;
		float material_lod2 = RT_TEXTURE_LOD_BASE;
		if (h.texture_lod2 > RT_TEXTURE_LOD_BASE) {
			material_lod2 = h.texture_lod2 + 0.5 * log2(max(abs(mat.uv2_scale.x * mat.uv2_scale.y), 1e-20));
		}
		lod2 = rt_texture_lod_enabled() ? material_lod2 : RT_TEXTURE_LOD_BASE;
		grad2 = h.texture_grad2 * vec4(mat.uv2_scale, mat.uv2_scale);
	}
	const bool detail_on_uv2 = (mat.flags & (1u << 17)) != 0u;
	const vec2 detail_uv = detail_on_uv2 ? uv2 : uv;
	const float detail_lod = detail_on_uv2 ? lod2 : lod;
	const vec4 detail_grad = detail_on_uv2 ? grad2 : grad;
	// The numbers the five lookups share under stochastic texture filtering (D117), drawn only for a material the
	// filter applies to (not point-filtered), with the major axis's position for an anisotropic one with the cone's
	// elliptical footprint (F2), the same test as the lookup's.
	const vec4 stf_u = (rt_texture_filtering() != 0u && (mat.flags & 4u) == 0u) ? rt_texture_filtering_random((mat.flags & 64u) != 0u && (grad != vec4(0.0) || grad2 != vec4(0.0))) : vec4(0.0);

	// Normal mapping. The map stays encoded until the detail layer has mixed its own into it, as the raster's
	// NORMAL_MAP does before the scene shader decodes it once; without a normal map it is the flat 0.5.
	vec3 tangent_space_normal = vec3(0.0, 0.0, 1.0);
	vec3 final_normal = h.geometry_normal;
	bool normal_mapped = (mat.flags & 1u) != 0u;
	vec2 normal_sample = vec2(0.5);
	if (normal_mapped) {
		normal_sample = sample_material_texture(mat.normal_texture_idx, uv, mat.flags, lod, grad, stf_u).xy;
	}

	// Texture sampling.
	vec4 albedo_tex = sample_material_texture(mat.albedo_texture_idx, uv, mat.flags, lod, grad, stf_u);
	// VAL: BaseMaterial3D's vertex colour as albedo (D126 H3), its alpha included, as its fragment() multiplies
	// albedo_tex by COLOR (fetched linear where its vertex() converts it, compute_hit_data's p_color).
	if ((mat.flags & RT_MAT_FLAG_VERTEX_COLOR) != 0u) {
		albedo_tex *= h.color;
	}
	vec3 albedo = albedo_tex.rgb * mat.albedo_color.rgb;

	// VAL: the detail layer (D118 MM4) as BaseMaterial3D's fragment writes it, last: the detail albedo blended by the
	// material's mode under its alpha, the detail normal mixed into the encoded normal map under the same alpha, both
	// under the mask's red, which is read on UV1. Unassigned, the albedo and mask read the white default and the
	// normal the flat one. The blended albedo is kept within 0..1, where the raster's Add and Subtract can leave it.
	if ((mat.flags & (1u << 14)) != 0u) {
		const vec4 detail_tex = mat.detail_albedo_texture_idx != 0u ? sample_material_texture(mat.detail_albedo_texture_idx, detail_uv, mat.flags, detail_lod, detail_grad, stf_u) : vec4(1.0);
		const vec2 detail_normal = mat.detail_normal_texture_idx != 0u ? sample_material_texture(mat.detail_normal_texture_idx, detail_uv, mat.flags, detail_lod, detail_grad, stf_u).xy : vec2(0.5);
		const float detail_mask = mat.detail_mask_texture_idx != 0u ? sample_material_texture(mat.detail_mask_texture_idx, uv, mat.flags, lod, grad, stf_u).r : 1.0;
		const uint detail_blend = (mat.flags >> 15u) & 3u;
		vec3 blended = detail_tex.rgb;
		if (detail_blend == 1u) {
			blended = albedo + detail_tex.rgb;
		} else if (detail_blend == 2u) {
			blended = albedo - detail_tex.rgb;
		} else if (detail_blend == 3u) {
			blended = albedo * detail_tex.rgb;
		}
		const vec3 detail = mix(albedo, blended, detail_tex.a);
		albedo = clamp(mix(albedo, detail, detail_mask), vec3(0.0), vec3(1.0));
		normal_sample = mix(normal_sample, mix(normal_sample, detail_normal, detail_tex.a), detail_mask);
		normal_mapped = true;
	}

	if (normal_mapped) {
		tangent_space_normal.xy = normal_sample * 2.0 - 1.0;
		tangent_space_normal.z = sqrt(max(0.0, 1.0 - dot(tangent_space_normal.xy, tangent_space_normal.xy)));
		final_normal = apply_normal_map(h, tangent_space_normal, mat.normal_map_depth);
	}

	// VAL: roughness and metallic through their textures' channels (D118): an ORM texture's green and blue, one
	// lookup when both are one texture; an unassigned texture (index 0) is the white default.
	const vec4 roughness_tex = mat.orm_texture_idx != 0u ? sample_material_texture(mat.orm_texture_idx, uv, mat.flags, lod, grad, stf_u) : vec4(1.0);
	vec4 metallic_tex = roughness_tex;
	if (mat.metallic_texture_idx != mat.orm_texture_idx) {
		metallic_tex = mat.metallic_texture_idx != 0u ? sample_material_texture(mat.metallic_texture_idx, uv, mat.flags, lod, grad, stf_u) : vec4(1.0);
	}
	const vec2 roughness_metallic = material_roughness_metallic(mat, roughness_tex, metallic_tex);
	float roughness = roughness_metallic.x;
	float metalness = roughness_metallic.y;

	// Emission as the raster's fragment computes it: the texture (black when there is none) is added to
	// or multiplied with the colour by the material's emission operator, then scaled by the energy.
	vec3 emission_tex = vec3(0.0);
	if ((mat.flags & 2u) != 0u) {
		// VAL: on UV2 with flag bit 18 (D118 MM4).
		const bool emission_on_uv2 = (mat.flags & (1u << 18)) != 0u;
		emission_tex = sample_material_texture(mat.emission_texture_idx, emission_on_uv2 ? uv2 : uv, mat.flags, emission_on_uv2 ? lod2 : lod, emission_on_uv2 ? grad2 : grad, stf_u).rgb;
	}
	vec3 emissive = ((mat.flags & 8u) != 0u) ? mat.emission_color * emission_tex : mat.emission_color + emission_tex;
	emissive *= mat.emission_strength * scene_data_block.data.emissive_exposure_normalization;

	// Build MaterialResult.
	MaterialResult m;
	m.albedo = albedo;
	m.alpha = albedo_tex.a * mat.albedo_color.a;
	m.roughness = roughness;
	m.metalness = metalness;
	m.specular = mat.specular;
	m.emissive = emissive;
	m.decal_emissive = vec3(0.0);
	m.normal = final_normal;
	m.transmission = vec3(0.0); // No TRANSMISSION here: the surface record's thin and refracting words overlay (raytracing_surface_inc.glsl).
	m.ior = 1.5;
	m.absorption = vec3(0.0);
	m.scattering = vec3(0.0);

	// Decals over the evaluated material (D104), before the back lighting reads the albedo, as in the raster.
	apply_decals(h, m);

	// Back lighting (D103) as the raster weighs it: its BACKLIGHT (the sRGB colour decoded, plus the
	// texture read linear) times the diffuse albedo, as the albedo of the thin diffuse-transmission lobe.
	m.diffuse_transmission = vec3(0.0);
	if ((mat.flags & 32u) != 0u) {
		vec3 c = unpackUnorm4x8(mat.backlight_rgba8).rgb;
		vec3 backlight = mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
		if (mat.backlight_texture_idx != 0u) {
			backlight += sample_material_texture(mat.backlight_texture_idx, uv, mat.flags, lod, grad, stf_u).rgb;
		}
		m.diffuse_transmission = clamp(backlight * m.albedo * (1.0 - m.metalness), vec3(0.0), vec3(1.0));
	}

	// VAL: anisotropy (D118 MM1) as BaseMaterial3D's fragment writes it: the ratio times the flow map's alpha, the
	// flow its red and green; without a flow map, along the tangent at full strength.
	m.anisotropy = mat.anisotropy;
	m.anisotropy_tangent = vec3(0.0);
	if (mat.anisotropy != 0.0) {
		vec2 flow = vec2(1.0, 0.0);
		if (mat.anisotropy_flowmap_idx != 0u) {
			const vec4 flowmap = sample_material_texture(mat.anisotropy_flowmap_idx, uv, mat.flags, lod, grad, stf_u);
			flow = flowmap.rg * 2.0 - 1.0;
			m.anisotropy *= flowmap.a;
		}
		m.anisotropy = clamp(m.anisotropy, -1.0, 1.0);
		m.anisotropy_tangent = rt_anisotropy_tangent(h.tangent, h.bitangent, flow, m.anisotropy);
	}

	// VAL: clearcoat (D118 MM2) as BaseMaterial3D's fragment writes it: the strength times the texture's red, the
	// roughness times its green, about the vertex normal (the raster's geo_normal ignores the normal map).
	m.clearcoat = 0.0;
	m.clearcoat_roughness = 0.0;
	m.clearcoat_normal = h.geometry_normal;
	if (mat.clearcoat_half2 != 0u) {
		vec2 clearcoat_params = unpackHalf2x16(mat.clearcoat_half2);
		if (mat.clearcoat_texture_idx != 0u) {
			clearcoat_params *= sample_material_texture(mat.clearcoat_texture_idx, uv, mat.flags, lod, grad, stf_u).xy;
		}
		m.clearcoat = clamp(clearcoat_params.x, 0.0, 1.0);
		m.clearcoat_roughness = clamp(clearcoat_params.y, 0.0, 1.0);
	}

	// VAL: rim (D118 MM3) as BaseMaterial3D's fragment writes it: the rim times the texture's red, the tint times its
	// green. shade_and_bounce reads the pair as a sheen layer.
	m.rim = 0.0;
	m.rim_tint = 0.0;
	if (mat.rim_half2 != 0u) {
		vec2 rim_params = unpackHalf2x16(mat.rim_half2);
		if (mat.rim_texture_idx != 0u) {
			rim_params *= sample_material_texture(mat.rim_texture_idx, uv, mat.flags, lod, grad, stf_u).xy;
		}
		m.rim = clamp(rim_params.x, 0.0, 1.0);
		m.rim_tint = clamp(rim_params.y, 0.0, 1.0);
	}

	// VAL: subsurface scattering (D120) as BaseMaterial3D's fragment writes it: the strength times the texture's red on
	// UV1, the diffuse share that walks under the surface; the skin mode from its flag. Constant 0 while the walk is not
	// compiled in.
	m.subsurface = 0.0;
	m.subsurface_skin = false;
	// VAL: hair (D126 H2): a hair material on a strand shades with the hair BCSDF, its parameters from the record, in
	// the fibre's own frame (no normal map). On triangles (cards) the material keeps the standard lobes.
	material_clear_hair(m);
	if ((mat.flags & RT_MAT_FLAG_HAIR) != 0u && (geometries[h.geometry_idx].flags & FLAG_LSS) != 0u) {
		const vec2 melanin_redness = unpackHalf2x16(mat.hair_melanin_half2);
		const vec2 ior_tilt = unpackHalf2x16(mat.hair_ior_tilt);
		m.hair = true;
		m.hair_color = (mat.flags & RT_MAT_FLAG_HAIR_COLOR) != 0u;
		m.hair_melanin = melanin_redness.x;
		m.hair_redness = melanin_redness.y;
		m.hair_ior = ior_tilt.x;
		m.hair_tilt = ior_tilt.y;
		m.normal = h.geometry_normal;
	}
#ifdef RT_SUBSURFACE_COMPILED
	m.subsurface_skin = (mat.flags & (1u << 19)) != 0u;
	if (mat.subsurface_strength > 0.0) {
		m.subsurface = mat.subsurface_strength;
		if (mat.subsurface_texture_idx != 0u) {
			m.subsurface *= sample_material_texture(mat.subsurface_texture_idx, uv, mat.flags, lod, grad, stf_u).r;
		}
		m.subsurface = clamp(m.subsurface, 0.0, 1.0);
	}
#endif

#ifdef RT_DEBUG_ENABLED
	{
		int VIS_MODE = int(get_rt_param(RT_PARAM_VIS_MODE));
		if (VIS_MODE != 0) {
			vec3 V = -gl_WorldRayDirectionEXT;
			float NdotV = max(dot(m.normal, V), 0.0001);
			vec3 orm = vec3(1.0, roughness, metalness);
			debug_visualize(VIS_MODE, h.geometry_normal, m.normal, tangent_space_normal,
					h.tangent, h.bitangent, uv, m.albedo, orm, m.metalness, m.roughness, mat.specular, m.emissive + m.decal_emissive, V, NdotV, gl_HitTEXT, h.is_front_face, material_lod);
			return;
		}
	}
#endif // RT_DEBUG_ENABLED
	shade_hit(h, m);
#endif
}

#[any_hit]

#version 460

#VERSION_DEFINES

#pragma shader_stage(any_hit)
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

#ifdef USE_CLUSTER_AS
#extension GL_NV_cluster_acceleration_structure : require
#endif
#ifdef USE_POSITION_FETCH
// Enabled with the closest hit's; the any-hit reads no position (its alpha test takes UVs from the
// cluster's blob, D93b).
#extension GL_EXT_ray_tracing_position_fetch : require
#endif

#define GLSL 1
#define RT_STAGE_ANY_HIT 1

// clang-format off
#include "raytracing_inc.glsl"
#include "../scene_data_inc.glsl"
#include "raytracing_common_inc.glsl"
// clang-format on

#define attribs hit_attribs.bary_or_uv
#define RT_HIT_ATTRIBS_DECLARED

#include "raytracing_hit_inc.glsl"

layout(location = 0) rayPayloadInEXT PathPayload payload;

// clang-format off
layout(set = 0, binding = 3, std430) readonly buffer GeometryBuffer {
	GeometryData geometries[];
};

layout(set = 0, binding = 4, std430) readonly buffer MotionIndexBuffer {
	int motion_indices[];
};

layout(set = 0, binding = 5, std430) readonly buffer MaterialBuffer {
	MaterialData materials[];
};
// clang-format on

layout(set = 1, binding = 0) uniform texture2D bindless_textures[];

#include "raytracing_samplers_inc.glsl"

// clang-format off
layout(set = 0, binding = 32, std430) readonly buffer MotionTransforms {
	InstanceMotionData motion_transforms[];
};
// clang-format on

// ============================================================================
// CUSTOM SHADER GLOBALS (injected for per-HG any-hit)
// ============================================================================
#ifdef RT_CUSTOM_HIT_GROUP
#include "raytracing_custom_globals_inc.glsl"
#endif

void main() {
	uint geometry_idx = gl_InstanceCustomIndexEXT;
	GeometryData geom = geometries[geometry_idx];

	uint i0, i1, i2;
	rt_resolve_hit_triangle(geom, i0, i1, i2);
	vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

#ifdef RT_CUSTOM_HIT_GROUP
	// Compute hit data inline (cannot include closest_hit_common_inc here).
	uint rt_geometry_idx = geometry_idx;
	vec2 rt_uv = fetch_uv(geom, i0, i1, i2, bary);
#ifdef RT_CUSTOM_UV2_USED
	vec2 rt_uv2 = fetch_uv2(geom, i0, i1, i2, bary);
#else
	vec2 rt_uv2 = vec2(0.0);
#endif
	TBNResult ah_tbn = fetch_tbn(geom, i0, i1, i2, bary);

	mat3 basis = rt_instance_basis(geom, mat3(gl_ObjectToWorldEXT));
	mat3 normal_matrix = rt_normal_basis(basis);

	vec3 rt_normal = normalize(normal_matrix * ah_tbn.normal);
	vec3 rt_tangent = normalize(normal_matrix * ah_tbn.tangent);
	vec3 rt_bitangent = cross(rt_normal, rt_tangent) * ah_tbn.bitangent_sign * (determinant(basis) < 0.0 ? -1.0 : 1.0);

	bool rt_front_face = (gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT);
	if (!rt_front_face) {
		rt_normal = -rt_normal;
	}

	vec3 rt_hit_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	vec4 rt_color = fetch_color(geom, i0, i1, i2, bary);

#include "raytracing_custom_fragment_inc.glsl"

	if (alpha_scissor_threshold > 0.0 && alpha < alpha_scissor_threshold) {
		ignoreIntersectionEXT;
	}
#else
	// HG0: Standard material alpha test.
	vec2 uv = fetch_uv(geom, i0, i1, i2, bary);
	MaterialData mat = materials[geometry_idx];
	uv = uv * mat.uv1_scale + mat.uv1_offset;
	// The base level, explicitly: an opacity micromap is baked from it (D96).
	float alpha = textureLod(sampler2D(bindless_textures[nonuniformEXT(mat.albedo_texture_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), uv, 0.0).a;
	alpha *= mat.albedo_color.a;
	if ((mat.flags & RT_MAT_FLAG_VERTEX_COLOR) != 0u) {
		alpha *= fetch_color(geom, i0, i1, i2, bary).a; // The vertex colour's alpha (D126 H3), as the closest hit's.
	}

	if (alpha < mat.alpha_threshold) {
		ignoreIntersectionEXT;
	}
#endif
}

#[intersection]

#version 460

#VERSION_DEFINES

#pragma shader_stage(intersection)
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

#define GLSL 1
#define RT_STAGE_INTERSECTION 1

// clang-format off
#include "raytracing_inc.glsl"
#include "../scene_data_inc.glsl"
#include "raytracing_data_inc.glsl"
#include "raytracing_common_inc.glsl"
// clang-format on

// Write all attributes and report the intersection. Transparently delta-compresses
// PREV_POSITION into spare .w bytes of packed_normal/tangent + prev_pos_delta_yz.
#define report_intersection(t_hit, kind)                                                      \
	{                                                                                         \
		vec3 _obj_hit = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * (t_hit);           \
		vec3 _delta = any(isnan(m_PREV_POSITION)) ? vec3(0.0) : (m_PREV_POSITION - _obj_hit); \
		uint _n4 = packSnorm4x8(vec4(m_HIT_NORMAL, 0.0));                                     \
		uint _t4 = packSnorm4x8(vec4(m_HIT_TANGENT, 0.0));                                    \
		uint _dx = packHalf2x16(vec2(_delta.x, 0.0));                                         \
		hit_attribs.bary_or_uv = m_HIT_UV;                                                    \
		hit_attribs.packed_normal = (_n4 & 0x00FFFFFFu) | ((_dx & 0xFFu) << 24u);             \
		hit_attribs.packed_tangent = (_t4 & 0x00FFFFFFu) | (((_dx >> 8u) & 0xFFu) << 24u);    \
		hit_attribs.prev_pos_delta_yz = packHalf2x16(vec2(_delta.y, _delta.z));               \
		reportIntersectionEXT(t_hit, kind);                                                   \
	}

#ifdef RT_CUSTOM_HIT_GROUP

// clang-format off
layout(set = 0, binding = 3, std430) readonly buffer GeometryBuffer {
	GeometryData geometries[];
};

layout(set = 0, binding = 5, std430) readonly buffer MaterialBuffer {
	MaterialData materials[];
};
// clang-format on

layout(set = 1, binding = 0) uniform texture2D bindless_textures[];

#include "raytracing_samplers_inc.glsl"

layout(buffer_reference, std140) readonly buffer CustomMaterialUniforms{
	/* RT_CUSTOM_UNIFORM_MEMBERS */
};

#include "raytracing_custom_texture_inc.glsl"

/* RT_CUSTOM_TEXTURE_DEFINES */

// File-scope built-ins accessible from user helper functions in globals.
#define ViewIndex 0
#define SHADER_IS_SRGB false
#define SHADER_SPACE_FAR 0.0
float global_time = scene_data_block.data.time;
float global_prev_time = 0.0;
mat4 read_model_matrix = mat4(0.0);
mat4 m_INV_MODEL_MATRIX = mat4(0.0);
mat4 read_view_matrix = transpose(mat4(scene_data_block.data.view_matrix[0], scene_data_block.data.view_matrix[1], scene_data_block.data.view_matrix[2], vec4(0.0, 0.0, 0.0, 1.0)));
mat4 inv_view_matrix = transpose(mat4(scene_data_block.data.inv_view_matrix[0], scene_data_block.data.inv_view_matrix[1], scene_data_block.data.inv_view_matrix[2], vec4(0.0, 0.0, 0.0, 1.0)));
mat4 projection_matrix = scene_data_block.data.projection_matrix;
mat4 inv_projection_matrix = scene_data_block.data.inv_projection_matrix;
vec2 read_viewport_size = scene_data_block.data.viewport_size;
float m_Z_NEAR = scene_data_block.data.z_near;
float m_Z_FAR = scene_data_block.data.z_far;

uint64_t _rt_material_address;
#define material CustomMaterialUniforms(_rt_material_address)

/* RT_CUSTOM_INTERSECTION_GLOBALS */

#endif

void main() {
#ifdef RT_CUSTOM_HIT_GROUP
	// Writable outputs.
	vec2 m_HIT_UV = vec2(0.0);
	vec3 m_HIT_NORMAL = vec3(0.0, 1.0, 0.0);
	vec3 m_HIT_TANGENT = vec3(1.0, 0.0, 0.0);
	vec3 m_PREV_POSITION = vec3(uintBitsToFloat(0x7FC00000u)); // NaN sentinel = not set.

	// Per-invocation built-ins (require RT intrinsics, only available in main).
	vec3 m_ORIGIN = gl_ObjectRayOriginEXT;
	vec3 m_DIRECTION = gl_ObjectRayDirectionEXT;
	vec3 m_WORLD_ORIGIN = gl_WorldRayOriginEXT;
	vec3 m_WORLD_DIRECTION = gl_WorldRayDirectionEXT;
	float m_T_MIN = gl_RayTminEXT;
	float m_T_MAX = gl_RayTmaxEXT;
	global_prev_time = scene_data_block.prev_data.time;

	// Resolve custom material uniforms via BDA (assigns file-scope address).
	uint rt_geometry_idx = gl_InstanceCustomIndexEXT;
	MaterialData rt_mat = materials[rt_geometry_idx];
	_rt_material_address = rt_mat.uniform_address;

	// Per-primitive AABB bounds (available when expose_aabb_bounds is enabled).
	GeometryData rt_geom = geometries[rt_geometry_idx];
	vec3 m_AABB_MIN = vec3(0.0);
	vec3 m_AABB_MAX = vec3(0.0);
	if (rt_geom.vertex_address != 0ul) {
		FloatBuffer aabb_buf = FloatBuffer(rt_geom.vertex_address);
		int base = int(gl_PrimitiveID) * 6;
		m_AABB_MIN = vec3(aabb_buf.v[base + 0], aabb_buf.v[base + 1], aabb_buf.v[base + 2]);
		m_AABB_MAX = vec3(aabb_buf.v[base + 3], aabb_buf.v[base + 4], aabb_buf.v[base + 5]);
	}

	mat4 rt_aabb_xform;
	mat4 rt_inv_aabb_xform;
	get_aabb_compression_xforms(rt_geom, rt_aabb_xform, rt_inv_aabb_xform);
	read_model_matrix = mat4(gl_ObjectToWorldEXT) * rt_inv_aabb_xform;
	m_INV_MODEL_MATRIX = rt_aabb_xform * mat4(gl_WorldToObjectEXT);
	// Absolute world for user code; the trace itself is relative to rt_origin.
	read_model_matrix[3].xyz += rt_origin.xyz;
	m_INV_MODEL_MATRIX[3].xyz -= mat3(m_INV_MODEL_MATRIX) * rt_origin.xyz;
	inv_view_matrix[3].xyz += rt_origin.xyz;
	read_view_matrix[3].xyz -= mat3(read_view_matrix) * rt_origin.xyz;
	m_WORLD_ORIGIN += rt_origin.xyz;

	/* RT_CUSTOM_INTERSECTION_CODE */

#else
	// Base-variant fallback: never executed at runtime (the intersection
	// stage is always rebuilt per-HG with RT_CUSTOM_HIT_GROUP defined).
	// Only touch unconditional HitAttribs fields so the base variant parses.
	hit_attribs.bary_or_uv = vec2(0.0);
	reportIntersectionEXT(gl_RayTminEXT, 0u);
#endif
}
