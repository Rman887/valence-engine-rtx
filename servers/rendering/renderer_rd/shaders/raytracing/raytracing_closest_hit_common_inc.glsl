// Common closest_hit utilities shared by all hit groups.
//
// Required includes (before this file):
//   raytracing_inc.glsl, brdf_inc.glsl, raytracing_hit_inc.glsl, raytracing_lights_inc.glsl,
//   raytracing_material_eval_inc.glsl
//
// Required bindings (before this file):
//   tlas, payload, scene_data_block, geometries[], motion_indices[], materials[], motion_transforms[], bindless_textures[],
//   SAMPLER_* (12 material samplers), rt_params, rt_depth_image,
//   DLSS-RR images (ifdef DLSS_RR_ENABLED)
//
// The shading itself (HitData, fog, debug views, shade_and_bounce) lives in
// raytracing_shade_inc.glsl, which has no hit-stage built-ins so the raygen
// can shade a G-buffer surface with the same code.

// clang-format off
#include "raytracing_shade_inc.glsl"
// clang-format on

// ============================================================================
// POSITION FETCH CHECK (D93a)
// ============================================================================

#if defined(RT_POSITION_FETCH_CHECK) && defined(USE_CLUSTER_AS)
// A verbose run's comparison of the positions a cluster hit fetches from its CLAS with the ones
// its vertex buffer holds, in the space the CLAS was built in (the compressed format's unorm
// box, which is what emissive_fetch_position decodes). Words per kind (float, compressed,
// deformed): hits compared, hits that differ, the largest absolute difference as float bits. Then
// (D93b) the cluster geometry hits whose blob words were compared with the surface's buffers at
// the triangle's own vertices, and those with a word that differs; with the positions' check at
// the same triangle id, this proves the records' first triangles too. One pixel in 64 compares;
// the host clears the words after each read.
layout(set = 0, binding = 46, std430) buffer PositionFetchCheck {
	uint words[];
}
position_fetch_check;

void position_fetch_check_hit(GeometryData geom, uint primitive_id) {
	if (gl_ClusterIDNV == gl_ClusterIDNoneNV || ((gl_LaunchIDEXT.x | gl_LaunchIDEXT.y) & 7u) != 0u || geom.vertex_address == 0ul) {
		return;
	}
	uint i0, i1, i2;
	get_triangle_indices_ex(geom, primitive_id, i0, i1, i2);
	vec3 d0 = abs(gl_HitTriangleVertexPositionsEXT[0] - emissive_fetch_position(geom, i0));
	vec3 d1 = abs(gl_HitTriangleVertexPositionsEXT[1] - emissive_fetch_position(geom, i1));
	vec3 d2 = abs(gl_HitTriangleVertexPositionsEXT[2] - emissive_fetch_position(geom, i2));
	vec3 d = max(max(d0, d1), d2);
	float difference = max(max(d.x, d.y), d.z);
	uint base = (geom.flags & FLAG_DEFORMED) != 0u ? 6u : ((geom.flags & FLAG_COMPRESSED) != 0u ? 3u : 0u);
	atomicAdd(position_fetch_check.words[base], 1u);
	if (difference != 0.0 || isnan(difference)) {
		atomicAdd(position_fetch_check.words[base + 1u], 1u);
		atomicMax(position_fetch_check.words[base + 2u], floatBitsToUint(isnan(difference) ? uintBitsToFloat(0x7F800000u) : difference));
	}
}

// p_surface is the record as bound, p_blob as rt_resolve_hit_triangle rewrote it, b0..b2 its indices.
void cluster_geometry_check_hit(GeometryData p_surface, GeometryData p_blob, uint p_triangle, uint b0, uint b1, uint b2) {
	if (gl_ClusterIDNV == gl_ClusterIDNoneNV || ((gl_LaunchIDEXT.x | gl_LaunchIDEXT.y) & 7u) != 0u || (p_surface.flags & FLAG_CLUSTER_GEOMETRY) == 0u || p_surface.vertex_address == 0ul) {
		return;
	}
	uvec3 g;
	get_triangle_indices_ex(p_surface, p_triangle, g.x, g.y, g.z);
	uvec3 b = uvec3(b0, b1, b2);
	Uint32Buffer surface_vertices = Uint32Buffer(p_surface.vertex_address);
	Uint32Buffer blob_vertices = Uint32Buffer(p_blob.vertex_address);
	Uint32Buffer surface_attributes = Uint32Buffer(p_surface.attribute_address);
	Uint32Buffer blob_attributes = Uint32Buffer(p_blob.attribute_address);
	uint attribute_words = p_surface.attribute_address != 0ul ? p_surface.attribute_stride >> 2u : 0u;
	bool compressed = (p_surface.flags & FLAG_COMPRESSED) != 0u;
	bool differs = false;
	for (uint k = 0u; k < 3u; k++) {
		if (p_surface.normal_byte_offset != OFFSET_NONE) {
			uint surface_normal = (p_surface.normal_byte_offset >> 2u) + g[k] * (p_surface.normal_stride >> 2u);
			uint blob_normal = b[k] * (p_blob.normal_stride >> 2u);
			differs = differs || surface_vertices.v[surface_normal] != blob_vertices.v[blob_normal];
			if (compressed) {
				differs = differs || surface_vertices.v[g[k] * 2u + 1u] != blob_vertices.v[blob_normal + 1u]; // The tangent angle's word.
			} else if (p_surface.normal_stride >= 8u) {
				differs = differs || surface_vertices.v[surface_normal + 1u] != blob_vertices.v[blob_normal + 1u];
			}
		}
		for (uint w = 0u; w < attribute_words; w++) {
			differs = differs || surface_attributes.v[g[k] * attribute_words + w] != blob_attributes.v[b[k] * attribute_words + w];
		}
	}
	atomicAdd(position_fetch_check.words[9], 1u);
	if (differs) {
		atomicAdd(position_fetch_check.words[10], 1u);
	}
}
#endif

// ============================================================================
// HIT DATA
// ============================================================================

/// Fetch vertex attributes and transform to world space.
/// Requires hitAttributeEXT HitAttribs and GeometryBuffer/MaterialBuffer bindings.
/// p_footprint: the cone's footprint is wanted whether or not the mesh has UVs or the frame decals.
/// p_gradients: a fetch reads the elliptical footprint's UV gradients (D116), with the anisotropic switch on.
HitData compute_hit_data(bool p_footprint, bool p_gradients) {
	HitData h;
	h.geometry_idx = gl_InstanceCustomIndexEXT;
	GeometryData geom = geometries[h.geometry_idx];
#if defined(RT_POSITION_FETCH_CHECK) && defined(USE_CLUSTER_AS)
	GeometryData surface = geom;
#endif
	uint i0, i1, i2;
	h.primitive_id = rt_resolve_hit_triangle(geom, i0, i1, i2);
#if defined(RT_POSITION_FETCH_CHECK) && defined(USE_CLUSTER_AS)
	position_fetch_check_hit(surface, h.primitive_id);
	cluster_geometry_check_hit(surface, geom, h.primitive_id, i0, i1, i2);
#endif

	// Custom hit groups may reference vertex color in their fragment code, so
	// pull all attributes; default HGs can skip color for perf.
#ifdef RT_CUSTOM_HIT_GROUP
	VertexAttributes attrs = fetch_vertex_attributes(geom, i0, i1, i2, attribs, FETCH_ALL);
#else
	VertexAttributes attrs = fetch_vertex_attributes(geom, i0, i1, i2, attribs, FETCH_UV | FETCH_TBN);
#endif
	h.uv = attrs.uv;
	h.color = attrs.color;

	mat3 basis = rt_instance_basis(geom, mat3(gl_ObjectToWorldEXT));
	mat3 normal_matrix = rt_normal_basis(basis);
	// A mirrored instance turns the frame's handedness, which the cross products below drop.
	float handedness = determinant(basis) < 0.0 ? -1.0 : 1.0;

#ifdef ENABLE_INTERSECTION_SHADERS
	if ((geom.flags & FLAG_PROCEDURAL) != 0u) {
		h.uv = hit_attribs.bary_or_uv;
		vec3 obj_normal = normalize(unpackSnorm4x8(hit_attribs.packed_normal).xyz);
		vec3 obj_tangent = normalize(unpackSnorm4x8(hit_attribs.packed_tangent).xyz);
		h.geometry_normal = normalize(normal_matrix * obj_normal);
		h.tangent = normalize(normal_matrix * obj_tangent);
		h.bitangent = cross(h.geometry_normal, h.tangent) * handedness;

		h.is_front_face = (dot(h.geometry_normal, -gl_WorldRayDirectionEXT) > 0.0);
		if (!h.is_front_face) {
			h.geometry_normal = -h.geometry_normal;
		}
	} else
#endif
	{
		// Triangle hit: reuse `attrs` from the top-level fetch (already has
		// UV / TBN from FETCH_UV | FETCH_TBN or FETCH_ALL).
		h.geometry_normal = normalize(normal_matrix * attrs.normal);
		h.tangent = normalize(normal_matrix * attrs.tangent);
		h.bitangent = cross(h.geometry_normal, h.tangent) * attrs.bitangent_sign * handedness;

		h.is_front_face = (gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT);
		if (!h.is_front_face) {
			h.geometry_normal = -h.geometry_normal;
		}
	}

	h.hit_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

	// The ray cone's footprint and level (D116), only where something reads them: the triangle's
	// corners in object space (a compressed surface's unorm box, which the instance transform maps),
	// from the vertex buffer, or on a cluster geometry hit from the CLAS (D93b), whose blob holds no
	// positions. A surface without UVs still has a footprint, which its decals read, when there are any.
	h.texture_lod = RT_TEXTURE_LOD_BASE;
	h.texture_footprint = RT_TEXTURE_LOD_BASE;
	h.texture_grad = vec4(0.0);
	if (rt_texture_lod_wanted() && (geom.flags & FLAG_PROCEDURAL) == 0u && (p_footprint || attrs.has_uv || get_rt_param(RT_PARAM_DECAL_COUNT) > 0.5)) {
		vec3 p0, p1, p2;
		bool has_positions = true;
		if ((geom.flags & FLAG_CLUSTER_GEOMETRY) != 0u && rt_hit_cluster_id() != RT_CLUSTER_ID_NONE) {
#ifdef USE_POSITION_FETCH
			p0 = gl_HitTriangleVertexPositionsEXT[0];
			p1 = gl_HitTriangleVertexPositionsEXT[1];
			p2 = gl_HitTriangleVertexPositionsEXT[2];
#else
			has_positions = false;
#endif
		} else if (geom.vertex_address == 0ul) {
			has_positions = false;
		} else {
			p0 = emissive_fetch_position(geom, i0);
			p1 = emissive_fetch_position(geom, i1);
			p2 = emissive_fetch_position(geom, i2);
		}
		if (has_positions) {
			mat3 object_to_world = mat3(gl_ObjectToWorldEXT);
			vec3 n = cross(object_to_world * (p1 - p0), object_to_world * (p2 - p0));
			vec2 cone = unpackHalf2x16(payload.packed_cone);
			h.texture_footprint = rt_texture_footprint(n, gl_WorldRayDirectionEXT, cone.x + cone.y * gl_HitTEXT);
			if (attrs.has_uv) {
				h.texture_lod = rt_texture_lod_triangle(h.texture_footprint, n, attrs.uv_corners[0], attrs.uv_corners[1], attrs.uv_corners[2]);
				// The anisotropic footprint where the level is known; a degenerate one leaves the level.
				if (p_gradients && h.texture_lod > RT_TEXTURE_LOD_BASE && rt_texture_lod_anisotropic()) {
					float radius = (cone.x + cone.y * gl_HitTEXT) * exp2(get_rt_param(RT_PARAM_TEXTURE_LOD_BIAS));
					h.texture_grad = rt_texture_grad_triangle(object_to_world * (p1 - p0), object_to_world * (p2 - p0), gl_WorldRayDirectionEXT, radius, attrs.uv_corners[0], attrs.uv_corners[1], attrs.uv_corners[2]);
				}
			}
		}
	}

	return h;
}

// ============================================================================
// HELPERS
// ============================================================================

vec4 sample_bindless_texture(uint tex_idx, vec2 uv) {
	return texture(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), uv);
}

/// Sample with point/nearest filtering (for pixel art textures).
vec4 sample_bindless_texture_point(uint tex_idx, vec2 uv) {
	return texture(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_NEAREST_REPEAT), uv);
}

/// Sample with the appropriate filter based on material flags (bit 2 = point filtering).
vec4 sample_material_texture(uint tex_idx, vec2 uv, uint mat_flags) {
	if ((mat_flags & 4u) != 0u) {
		return sample_bindless_texture_point(tex_idx, uv);
	}
	return sample_bindless_texture(tex_idx, uv);
}

/// The texture at the ray cone's level (D116): p_lod_t, the hit's texture-independent level, plus half
/// the log2 of this texture's texel count; its base level where p_lod_t is RT_TEXTURE_LOD_BASE.
vec4 sample_bindless_texture(uint tex_idx, vec2 uv, float p_lod_t) {
	if (p_lod_t <= RT_TEXTURE_LOD_BASE) {
		return sample_bindless_texture(tex_idx, uv);
	}
	vec2 size = vec2(textureSize(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), 0));
	return textureLod(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), uv, p_lod_t + 0.5 * log2(max(size.x * size.y, 1.0)));
}

/// Nearest filtering with mipmaps at the ray cone's level, as sample_bindless_texture's linear form; level 0 where
/// p_lod_t is RT_TEXTURE_LOD_BASE.
vec4 sample_bindless_texture_point(uint tex_idx, vec2 uv, float p_lod_t) {
	float level = 0.0;
	if (p_lod_t > RT_TEXTURE_LOD_BASE) {
		vec2 size = vec2(textureSize(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT), 0));
		level = p_lod_t + 0.5 * log2(max(size.x * size.y, 1.0));
	}
	return textureLod(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT), uv, level);
}

/// The texture over the cone's elliptical footprint (D116): textureGrad at the material's UV gradients (g1, g2)
/// through the anisotropic sampler, where the material's filter is anisotropic. A material filtered without
/// anisotropy, or a hit without gradients, reads at the isotropic level p_lod_t, which beats the long axis's
/// trilinear level the raster would take.
vec4 sample_bindless_texture(uint tex_idx, vec2 uv, float p_lod_t, vec4 p_grad, bool p_anisotropic) {
	if (!p_anisotropic || p_grad == vec4(0.0)) {
		return sample_bindless_texture(tex_idx, uv, p_lod_t);
	}
	return textureGrad(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_REPEAT), uv, p_grad.xy, p_grad.zw);
}

/// A material's texture through the filter it asks for: nearest (bit 2) or linear, without mipmaps (bit 7) at the
/// base level as the raster's mip-less samplers read, with them at the ray cone's level or over its elliptical
/// footprint (bit 6, D116).
vec4 sample_material_texture(uint tex_idx, vec2 uv, uint mat_flags, float p_lod_t, vec4 p_grad) {
	const bool nearest = (mat_flags & 4u) != 0u;
	if ((mat_flags & 128u) != 0u) {
		if (nearest) {
			return sample_bindless_texture_point(tex_idx, uv);
		}
		return textureLod(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_LINEAR_REPEAT), uv, 0.0);
	}
	if (nearest) {
		return sample_bindless_texture_point(tex_idx, uv, p_lod_t);
	}
	return sample_bindless_texture(tex_idx, uv, p_lod_t, p_grad, (mat_flags & 64u) != 0u);
}

/// Apply tangent-space normal map to geometry normal.
vec3 apply_normal_map(HitData h, vec3 tangent_space_normal, float normal_map_depth) {
	vec3 mapped = h.tangent * tangent_space_normal.x + h.bitangent * tangent_space_normal.y + h.geometry_normal * tangent_space_normal.z;
	return normalize(mix(h.geometry_normal, mapped, normal_map_depth));
}

// ============================================================================
// DEPTH WRITE (primary ray only)
// ============================================================================

/// Write NDC depth for primary ray hits (bounce 0, sample 0 only).
void write_primary_hit_depth(vec3 hit_pos) {
	if (get_total_bounces(payload.packed_bounces_flags) == 0u && is_sample_zero(payload.packed_bounces_flags)) {
		mat4 view_mat = transpose(mat4(scene_data_block.data.view_matrix[0],
				scene_data_block.data.view_matrix[1],
				scene_data_block.data.view_matrix[2],
				vec4(0.0, 0.0, 0.0, 1.0)));
		vec3 view_pos = (view_mat * vec4(hit_pos, 1.0)).xyz;
		vec4 clip_pos = scene_data_block.data.projection_matrix * vec4(view_pos, 1.0);
		float ndc_depth = clip_pos.z / clip_pos.w;
		imageStore(rt_depth_image, ivec2(gl_LaunchIDEXT.xy), vec4(ndc_depth));
	}
}

// ============================================================================
// VELOCITY WRITE (primary ray only, MV-gated)
// ============================================================================

#ifdef ENABLE_INTERSECTION_SHADERS
/// Decode the FP16-compressed PREV_POSITION delta from HitAttribs.
vec3 decode_prev_pos_delta() {
	uint dx_low = (hit_attribs.packed_normal >> 24u) & 0xFFu;
	uint dx_high = (hit_attribs.packed_tangent >> 24u) & 0xFFu;
	float delta_x = unpackHalf2x16(dx_low | (dx_high << 8u)).x;
	vec2 delta_yz = unpackHalf2x16(hit_attribs.prev_pos_delta_yz);
	return vec3(delta_x, delta_yz.x, delta_yz.y);
}
#endif

/// Reconstruct previous-frame mat4 from a compact motion transform entry.
mat4 decode_prev_object_to_world(int motion_idx) {
	InstanceMotionData m = motion_transforms[motion_idx];
	return transpose(mat4(
			vec4(m.prev_xform[0], m.prev_xform[1], m.prev_xform[2], m.prev_xform[3]),
			vec4(m.prev_xform[4], m.prev_xform[5], m.prev_xform[6], m.prev_xform[7]),
			vec4(m.prev_xform[8], m.prev_xform[9], m.prev_xform[10], m.prev_xform[11]),
			vec4(0.0, 0.0, 0.0, 1.0)));
}

/// Write motion vectors for primary ray hits (bounce 0, sample 0 only).
/// Uses unjittered VP matrices matching the raster motion_vectors_store convention.
void write_primary_hit_velocity(vec3 hit_pos) {
	if (get_total_bounces(payload.packed_bounces_flags) != 0u || !is_sample_zero(payload.packed_bounces_flags)) {
		return;
	}

	uint geom_idx = gl_InstanceCustomIndexEXT;
	int mi = motion_indices[geom_idx];

	// Resolve previous-frame model matrix: compact entry if moved, current transform otherwise.
	mat4 prev_model = (mi >= 0) ? decode_prev_object_to_world(mi) : mat4(gl_ObjectToWorldEXT);

	vec3 obj_pos = (mat4(gl_WorldToObjectEXT) * vec4(hit_pos, 1.0)).xyz;
	vec3 prev_obj_pos = obj_pos;

	GeometryData geom = geometries[geom_idx];

#ifdef ENABLE_INTERSECTION_SHADERS
	if ((geom.flags & FLAG_PROCEDURAL) != 0u) {
		prev_obj_pos += decode_prev_pos_delta();
	}
#endif

#ifdef RT_HIT_ATTRIBS_DECLARED
	if ((geom.flags & FLAG_DEFORMED) != 0u) {
		uint64_t prev_addr = packUint2x32(uvec2(geom.prev_vertex_address_lo, geom.prev_vertex_address_hi));
		if (prev_addr != 0ul) {
			uint i0, i1, i2;
			rt_resolve_hit_triangle(geom, i0, i1, i2); // A deformed surface reads its own buffers.
			vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
			FloatBuffer prev_vb = FloatBuffer(prev_addr);
			uint stride_floats = geom.position_stride >> 2;
			vec3 p0 = vec3(prev_vb.v[i0 * stride_floats + 0u],
					prev_vb.v[i0 * stride_floats + 1u],
					prev_vb.v[i0 * stride_floats + 2u]);
			vec3 p1 = vec3(prev_vb.v[i1 * stride_floats + 0u],
					prev_vb.v[i1 * stride_floats + 1u],
					prev_vb.v[i1 * stride_floats + 2u]);
			vec3 p2 = vec3(prev_vb.v[i2 * stride_floats + 0u],
					prev_vb.v[i2 * stride_floats + 1u],
					prev_vb.v[i2 * stride_floats + 2u]);
			prev_obj_pos = bary.x * p0 + bary.y * p1 + bary.z * p2;
		}
	}
#endif

	vec3 prev_world_pos = (prev_model * vec4(prev_obj_pos, 1.0)).xyz;

	vec2 curr_uv = project_uv(hit_pos, curr_vp_unjittered);
	vec2 prev_uv = project_uv(prev_world_pos, prev_vp_unjittered);

	imageStore(rt_velocity_image, ivec2(gl_LaunchIDEXT.xy), vec4(prev_uv - curr_uv, 0.0, 0.0));
}
