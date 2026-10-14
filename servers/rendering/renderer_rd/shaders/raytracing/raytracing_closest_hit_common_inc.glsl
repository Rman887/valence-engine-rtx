// Common closest_hit utilities shared by all hit groups.
//
// Required includes (before this file):
//   raytracing_inc.glsl, brdf_inc.glsl, raytracing_hit_inc.glsl, raytracing_lights_inc.glsl,
//   raytracing_material_eval_inc.glsl
//
// Required bindings (before this file):
//   tlas, payload, scene_data_block, geometries[], motion_indices[], materials[], motion_transforms[], bindless_textures[],
//   SAMPLER_* (12 material samplers), rt_material_samplers[] (the same twelve as one array), bindless_texel_log2[],
//   rt_params, rt_depth_image,
//   DLSS-RR images (ifdef DLSS_RR_ENABLED)
//
// The shading itself (HitData, fog, debug views, shade_and_bounce) lives in
// raytracing_shade_inc.glsl, which has no hit-stage built-ins so the raygen
// can shade a G-buffer surface with the same code.

// clang-format off
#include "raytracing_shade_inc.glsl"
#include "rtxtf_stf_inc.glsl"
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

/// The hit triangle's corners in object space (a compressed surface's unorm box, which the instance transform maps):
/// from the acceleration structure where the device has position fetch, which every triangle and cluster build then
/// allows (P1d; the build's input is the vertex buffer a geometry row names, so the corners are the same); otherwise
/// from the vertex buffer, or nothing on a cluster geometry hit, whose blob holds no positions (D93b).
bool rt_hit_triangle_positions(GeometryData geom, uint i0, uint i1, uint i2, out vec3 r_p0, out vec3 r_p1, out vec3 r_p2) {
#ifdef USE_POSITION_FETCH
	r_p0 = gl_HitTriangleVertexPositionsEXT[0];
	r_p1 = gl_HitTriangleVertexPositionsEXT[1];
	r_p2 = gl_HitTriangleVertexPositionsEXT[2];
	return true;
#else
	r_p0 = vec3(0.0);
	r_p1 = vec3(0.0);
	r_p2 = vec3(0.0);
	if ((geom.flags & FLAG_CLUSTER_GEOMETRY) != 0u && rt_hit_cluster_id() != RT_CLUSTER_ID_NONE) {
		return false;
	}
	if (geom.vertex_address == 0ul) {
		return false;
	}
	r_p0 = emissive_fetch_position(geom, i0);
	r_p1 = emissive_fetch_position(geom, i1);
	r_p2 = emissive_fetch_position(geom, i2);
	return true;
#endif
}

/// A tangent for a surface stored without one (D118): the triangle's dP/du from its corners and UVs, in the object
/// space the vertex normal is in, its sign MikkTSpace's (the sign of the UV area); any vector perpendicular to the
/// normal where the UVs or the positions give none. Both tests are relative, so they hold at any scale.
vec3 rt_derived_tangent(GeometryData geom, uint i0, uint i1, uint i2, VertexAttributes p_attrs, out float r_sign) {
	r_sign = 1.0;
	vec3 p0, p1, p2;
	if (p_attrs.has_uv && rt_hit_triangle_positions(geom, i0, i1, i2, p0, p1, p2)) {
		const vec3 e1 = p1 - p0;
		const vec3 e2 = p2 - p0;
		const vec2 d1 = p_attrs.uv_corners[1] - p_attrs.uv_corners[0];
		const vec2 d2 = p_attrs.uv_corners[2] - p_attrs.uv_corners[0];
		const float det = d1.x * d2.y - d2.x * d1.y;
		if (abs(det) > 1e-6 * length(d1) * length(d2)) {
			// dP/du times |det|: the direction without the division.
			const vec3 dpdu = (e1 * d2.y - e2 * d1.y) * sign(det);
			const vec3 t = dpdu - p_attrs.normal * dot(p_attrs.normal, dpdu);
			if (dot(t, t) > 1e-6 * dot(dpdu, dpdu)) {
				r_sign = det < 0.0 ? -1.0 : 1.0;
				return t;
			}
		}
	}
	const vec3 n = p_attrs.normal;
	return abs(n.x) > abs(n.z) ? vec3(-n.y, n.x, 0.0) : vec3(0.0, -n.z, n.y);
}

#ifdef USE_LSS
/// A line list's segment (D126): its two vertex indices, from the surface's index pairs, or its vertex pairs without indices.
void rt_lss_segment_indices(GeometryData geom, uint p_segment, out uint i0, out uint i1) {
	if (geom.index_format == 2u) {
		i0 = p_segment * 2u;
		i1 = p_segment * 2u + 1u;
	} else if (geom.index_address != 0ul) {
		Uint32Buffer idx = Uint32Buffer(geom.index_address);
		if (geom.index_format == 0u) {
			uint pair = idx.v[p_segment]; // Two 16-bit indices, one word a segment.
			i0 = pair & 0xFFFFu;
			i1 = pair >> 16;
		} else {
			i0 = idx.v[p_segment * 2u];
			i1 = idx.v[p_segment * 2u + 1u];
		}
	} else {
		i0 = i1 = 0u;
	}
}
#endif

/// Fetch vertex attributes and transform to world space.
/// Requires hitAttributeEXT HitAttribs and GeometryBuffer/MaterialBuffer bindings.
/// p_footprint: the cone's footprint is wanted whether or not the mesh has UVs or the frame decals.
/// p_gradients: a fetch reads the elliptical footprint's UV gradients (D116), with the anisotropic switch on, until the
/// path's first diffuse bounce (GPU audit hot 8).
/// p_uv2: a fetch reads the second UV set, which the hit fetches with its level (D118).
HitData compute_hit_data(bool p_footprint, bool p_gradients, bool p_uv2) {
	HitData h;
	h.geometry_idx = gl_InstanceCustomIndexEXT;
#ifdef RT_CUSTOM_HIT_GROUP
	h.custom_group = true;
#else
	h.custom_group = false;
#endif
	GeometryData geom = geometries[h.geometry_idx];
#if defined(RT_POSITION_FETCH_CHECK) && defined(USE_CLUSTER_AS)
	GeometryData surface = geom;
#endif
	uint i0, i1, i2;
	vec2 fetch_bary = attribs;
#ifdef USE_LSS
	// A segment (D126): its two vertices as the triangle's first two and third, so the fetch's barycentrics (1 - u, u, 0)
	// blend the segment's ends at the hit's u, the position along the midsection the traversal reports in the hit
	// attribute: the centre of the sphere the hit point lies on, also on a tapered segment, where the axis's foot is not.
	const bool lss = (geom.flags & FLAG_LSS) != 0u;
	vec3 lss_a = vec3(0.0);
	vec3 lss_b = vec3(0.0);
	vec3 lss_obj_pos = vec3(0.0);
	float lss_u = 0.0;
	if (lss) {
		h.primitive_id = uint(gl_PrimitiveID);
		rt_lss_segment_indices(geom, h.primitive_id, i0, i1);
		i2 = i1;
		lss_a = gl_HitLSSPositionsNV[0];
		lss_b = gl_HitLSSPositionsNV[1];
		lss_obj_pos = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * gl_HitTEXT;
		lss_u = clamp(attribs.x, 0.0, 1.0);
		fetch_bary = vec2(lss_u, 0.0);
	} else
#endif
	{
		h.primitive_id = rt_resolve_hit_triangle(geom, i0, i1, i2);
#if defined(RT_POSITION_FETCH_CHECK) && defined(USE_CLUSTER_AS)
		position_fetch_check_hit(surface, h.primitive_id);
		cluster_geometry_check_hit(surface, geom, h.primitive_id, i0, i1, i2);
#endif
	}

	// Custom hit groups may reference vertex color in their fragment code, so
	// pull all attributes; default HGs can skip color for perf.
#ifdef RT_CUSTOM_HIT_GROUP
	VertexAttributes attrs = fetch_vertex_attributes(geom, i0, i1, i2, fetch_bary, FETCH_ALL | (p_uv2 ? FETCH_UV2 : 0u));
#else
	VertexAttributes attrs = fetch_vertex_attributes(geom, i0, i1, i2, fetch_bary, FETCH_UV | FETCH_TBN | (p_uv2 ? FETCH_UV2 : 0u));
#endif
	h.uv = attrs.uv;
	h.uv2 = attrs.uv2;
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
#ifdef USE_LSS
			if (lss) {
		// A segment (D126): the normal from the touched sphere's centre, the tangent along the axis (a direction on the
		// surface, so through the basis), made perpendicular to the normal, which a tapered segment tilts off the axis.
		// The surface has no inside a ray could leave by: every hit is a front face.
		vec3 obj_normal = lss_obj_pos - mix(lss_a, lss_b, lss_u);
		obj_normal = dot(obj_normal, obj_normal) > 0.0 ? normalize(obj_normal) : vec3(0.0, 1.0, 0.0);
		h.geometry_normal = normalize(normal_matrix * obj_normal);
		vec3 axis = basis * (lss_b - lss_a);
		axis -= h.geometry_normal * dot(h.geometry_normal, axis);
		const vec3 n = h.geometry_normal;
		h.tangent = dot(axis, axis) > 0.0 ? normalize(axis) : normalize(abs(n.x) > abs(n.z) ? vec3(-n.y, n.x, 0.0) : vec3(0.0, -n.z, n.y));
		h.bitangent = cross(h.geometry_normal, h.tangent) * handedness;
		h.is_front_face = true;
	} else
#endif
	{
		// Triangle hit: reuse `attrs` from the top-level fetch (already has
		// UV / TBN from FETCH_UV | FETCH_TBN or FETCH_ALL).
		vec3 obj_tangent = attrs.tangent;
		float bitangent_sign = attrs.bitangent_sign;
		if (attrs.has_normal && !attrs.has_tangent) {
			obj_tangent = rt_derived_tangent(geom, i0, i1, i2, attrs, bitangent_sign);
		}
		h.geometry_normal = normalize(normal_matrix * attrs.normal);
		h.tangent = normalize(normal_matrix * obj_tangent);
		h.bitangent = cross(h.geometry_normal, h.tangent) * bitangent_sign * handedness;

		h.is_front_face = (gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT);
		if (!h.is_front_face) {
			h.geometry_normal = -h.geometry_normal;
		}
	}

	h.hit_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

	// The ray cone's footprint and level (D116), only where something reads them, from the triangle's
	// corners (rt_hit_triangle_positions). A surface without UVs still has a footprint, which its decals
	// read, when there are any.
	h.texture_lod = RT_TEXTURE_LOD_BASE;
	h.texture_footprint = RT_TEXTURE_LOD_BASE;
	h.texture_grad = vec4(0.0);
	h.texture_lod2 = RT_TEXTURE_LOD_BASE;
	h.texture_grad2 = vec4(0.0);
	if (rt_texture_lod_wanted() && (geom.flags & (FLAG_PROCEDURAL | FLAG_LSS)) == 0u && (p_footprint || attrs.has_uv || attrs.has_uv2 || get_rt_param(RT_PARAM_DECAL_COUNT) > 0.5)) {
		vec3 p0, p1, p2;
		if (rt_hit_triangle_positions(geom, i0, i1, i2, p0, p1, p2)) {
			mat3 object_to_world = mat3(gl_ObjectToWorldEXT);
			vec3 n = cross(object_to_world * (p1 - p0), object_to_world * (p2 - p0));
			vec2 cone = unpackHalf2x16(payload.packed_cone);
			h.texture_footprint = rt_texture_footprint(n, gl_WorldRayDirectionEXT, cone.x + cone.y * gl_HitTEXT);
			// The anisotropic footprint (D116) only until the path's first diffuse bounce (GPU audit hot 8): the camera's
			// view and a mirror's or a glossy reflection's of it keep it, since a near-mirror lobe is narrower than the
			// pixel's cone, which D116 widens by only a fraction of the lobe; past a diffuse bounce the lobe is wider than
			// any footprint's long axis, and the isotropic level costs one fetch where the elliptical footprint takes up
			// to the sampler's anisotropy in taps.
			const bool anisotropic = p_gradients && rt_texture_lod_anisotropic() && get_diffuse_bounces(payload.packed_bounces_flags) == 0u;
			if (attrs.has_uv) {
				h.texture_lod = rt_texture_lod_triangle(h.texture_footprint, n, attrs.uv_corners[0], attrs.uv_corners[1], attrs.uv_corners[2]);
				// The anisotropic footprint where the level is known; a degenerate one leaves the level.
				if (anisotropic && h.texture_lod > RT_TEXTURE_LOD_BASE) {
					float radius = (cone.x + cone.y * gl_HitTEXT) * exp2(get_rt_param(RT_PARAM_TEXTURE_LOD_BIAS));
					h.texture_grad = rt_texture_grad_triangle(object_to_world * (p1 - p0), object_to_world * (p2 - p0), gl_WorldRayDirectionEXT, radius, attrs.uv_corners[0], attrs.uv_corners[1], attrs.uv_corners[2]);
				}
			}
			if (attrs.has_uv2) {
				h.texture_lod2 = rt_texture_lod_triangle(h.texture_footprint, n, attrs.uv2_corners[0], attrs.uv2_corners[1], attrs.uv2_corners[2]);
				if (anisotropic && h.texture_lod2 > RT_TEXTURE_LOD_BASE) {
					float radius = (cone.x + cone.y * gl_HitTEXT) * exp2(get_rt_param(RT_PARAM_TEXTURE_LOD_BIAS));
					h.texture_grad2 = rt_texture_grad_triangle(object_to_world * (p1 - p0), object_to_world * (p2 - p0), gl_WorldRayDirectionEXT, radius, attrs.uv2_corners[0], attrs.uv2_corners[1], attrs.uv2_corners[2]);
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

/// A material's texture through the filter it asks for: nearest (bit 2) or linear, without mipmaps (bit 7) at the
/// base level as the raster's mip-less samplers read, with them at the ray cone's level or over its elliptical
/// footprint (bit 6, D116). One fetch (GPU audit hot 5): the sampler is picked from the material's flags in the
/// sampler array and the level is given as gradients, where a branch per form issued one of four fetches through four
/// bindings. At the cone's level the gradients are that level's length along each axis, from the slot's log2 width and
/// height, so the hardware's level is the level (a non-square texture included); without a level they are zero, the
/// base level; over the footprint they are the material's own, through the anisotropic sampler.
vec4 sample_material_texture(uint tex_idx, vec2 uv, uint mat_flags, float p_lod_t, vec4 p_grad) {
	const bool nearest = (mat_flags & 4u) != 0u;
	const bool mipmaps = (mat_flags & 128u) == 0u;
	const bool footprint = mipmaps && !nearest && (mat_flags & 64u) != 0u && p_grad != vec4(0.0);
	const uint sampler_index = !mipmaps ? (nearest ? RT_MATERIAL_SAMPLER_NEAREST_REPEAT : RT_MATERIAL_SAMPLER_LINEAR_REPEAT) : (nearest ? RT_MATERIAL_SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT : (footprint ? RT_MATERIAL_SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_REPEAT : RT_MATERIAL_SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT));
	vec4 grad = p_grad;
	if (!footprint) {
		grad = vec4(0.0);
		if (mipmaps && p_lod_t > RT_TEXTURE_LOD_BASE) {
			const vec2 size_log = bindless_texel_log2[tex_idx];
			const float level = p_lod_t + 0.5 * (size_log.x + size_log.y);
			grad = vec4(exp2(level - size_log.x), 0.0, 0.0, exp2(level - size_log.y));
		}
	}
	return textureGrad(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], rt_material_samplers[nonuniformEXT(sampler_index)]), uv, grad.xy, grad.zw);
}

/// The stochastic texture filter the hit shaders use (D117): 0 for none, else the STF_FILTER_TYPE_* of
/// rendering/pathtracing/texture_filtering. The pipeline flag compiles the filter in only while the setting is on, so
/// with it off every use below folds away; the filter's type is a uniform.
uint rt_texture_filtering() {
	if ((RT_FLAGS & RT_FLAG_TEXTURE_FILTERING) == 0u) {
		return 0u;
	}
	return uint(get_rt_param(RT_PARAM_TEXTURE_FILTERING));
}

/// Whether an anisotropic lookup takes the library's gradient path (D117 F2): at the camera ray's hit of a frame the
/// camera jitters (a temporal upscaler or TAA; the frame's taa_jitter is the jittered projection's offset, never zero
/// on such a frame), where the jitter across frames sweeps the footprint's major axis. Anywhere else the lookup moves
/// along the major axis itself. The random numbers and the lookup read this one test, so u.z is drawn exactly where
/// the lookup uses it.
bool rt_texture_filtering_gradient_path() {
	return get_total_bounces(payload.packed_bounces_flags) == 0u && scene_data_block.data.taa_jitter != vec2(0.0);
}

/// The uniform numbers one hit's stochastic lookups share (D117), as RTX Texture Filtering's state takes them: the
/// texel's pair in xy and the level's in w. All of a material's textures take the same numbers (the library's
/// reseed off, as RTXPT), so textures of one size pick the same texel and level. With p_major, off the gradient
/// path, z is the position along an anisotropic footprint's major axis (F2; the library leaves z unused in 2D).
/// They come from the payload's sampler without moving it: under Sobol (D52) from this bounce's block at
/// RT_SOBOL_DIM_TEXTURE_FILTERING on a copy of the state, and under the hash from a salted hash of the chain's
/// current state, which the chain never advances through, so turning the filter on changes no other number in the
/// path.
vec4 rt_texture_filtering_random(bool p_major) {
	uint state = payload.rng_state;
	const uint bounce = get_total_bounces(payload.packed_bounces_flags);
	const bool major = p_major && !rt_texture_filtering_gradient_path();
	if (rng_is_sobol()) {
		rng_set_dimension(state, bounce * RT_SOBOL_DIMS_PER_BOUNCE + RT_SOBOL_DIM_TEXTURE_FILTERING);
		vec2 texel = rand2(state);
		float level = rand(state);
		return vec4(texel, major ? rand(state) : 0.0, level);
	}
	state = pcg_hash(state ^ 0x5f356495u);
	float x = float(state) / 4294967296.0;
	state = pcg_hash(state);
	float y = float(state) / 4294967296.0;
	state = pcg_hash(state);
	float w = float(state) / 4294967296.0;
	float z = 0.0;
	if (major) {
		state = pcg_hash(state);
		z = float(state) / 4294967296.0;
	}
	return vec4(x, y, z, w);
}

/// Whether a stochastic lookup at this full level keeps the hardware's filter (D117): the texture magnifies it (its
/// full level at most 0, or level 0 everywhere without the cone) and
/// rendering/pathtracing/texture_filtering_magnification is Hardware. For an isotropic lookup under the bilinear filter
/// both have the same expectation there; under the bicubic and Gaussian filters the hardware's bilinear is the sharper,
/// a seam in sharpness at level 0 that D117's F4 judges. Under every filter an anisotropic lookup on the gradient path
/// keeps a seam along the footprint's long axis where the rule switches: T5's textureGrad under the camera's jitter
/// filters that axis with two boxes (its probes and the jitter), the gradient path with the jitter's alone.
bool rt_texture_filtering_magnified(float p_level) {
	return p_level <= 0.0 && get_rt_param(RT_PARAM_TEXTURE_FILTERING_MAGNIFICATION) < 0.5;
}

/// The texel centre (xy) and integer level (z) a stochastic lookup reads (D117), for a texture of p_size texels and
/// p_levels levels, at the full level p_level (rt_texture_filtering_level's for a footprint), picked by p_u under
/// p_filter. The lookups of HG0 and of custom hit groups share it; each reads the texel through the nearest sampler
/// with mipmaps of its own texture, which wraps a texel past the edge and decodes sRGB.
///
/// An elliptical footprint (p_footprint, T5's gradients p_grad, F2) takes its level from the ellipse's minor axis,
/// the anisotropy clamped at 16:1 as the library clamps it; the hardware's own cap is the viewport's anisotropic
/// filtering level, so the two agree only at 16x. A gradient is the UV change across one pixel step, the footprint's
/// full extent (rt_texture_grad_triangle's radius is the cone's width), so the footprint spans half a gradient either
/// side of the hit. On the gradient path (the camera ray's hit under a jittered projection) the library reads that
/// level alone and leaves the major axis to the camera's sub-pixel jitter across frames (the STF paper's section
/// 4.2.1); elsewhere, where no jitter sweeps the footprint, the lookup moves along the major axis by (p_u.z - 1/2) of
/// its gradient, a box over the footprint's extent, before the level's pick.
vec3 rt_texture_filtering_position(uint p_filter, vec4 p_u, uvec2 p_size, uint p_levels, bool p_mipmaps, vec2 p_uv, float p_level, vec4 p_grad, bool p_footprint) {
	STF_SamplerState stf = STF_Create(p_u);
	STF_SetFilterType(stf, p_filter);
	STF_SetSigma(stf, get_rt_param(RT_PARAM_TEXTURE_FILTERING_SIGMA));
	if (p_footprint && rt_texture_filtering_gradient_path()) {
		return STF_Texture2DGetSamplePosGrad(stf, p_size.x, p_size.y, p_levels, p_uv, p_grad.xy, p_grad.zw);
	}
	vec2 at = p_uv;
	if (p_footprint) {
		// The major axis as the library picks it, the longer gradient in texels.
		const vec2 g1 = p_grad.xy * vec2(p_size);
		const vec2 g2 = p_grad.zw * vec2(p_size);
		at += (p_u.z - 0.5) * (dot(g1, g1) > dot(g2, g2) ? p_grad.xy : p_grad.zw);
	}
	return STF_Texture2DGetSamplePosLevel(stf, p_size.x, p_size.y, p_levels, at, p_mipmaps ? max(p_level, 0.0) : 0.0);
}

/// The full level of an elliptical footprint's stochastic lookup (D117 F2): the ellipse's minor axis, the anisotropy
/// clamped at 16:1 (rt_texture_filtering_position).
float rt_texture_filtering_level(uvec2 p_size, vec4 p_grad) {
	return STF_ComputeTextureLod_STF(p_size, p_grad, RT_TEXTURE_LOD_BASE, 64.0, 0.0);
}

/// A material's texture through the filter it asks for, as sample_material_texture above, with stochastic texture
/// filtering (D117) on a material filtered linearly (with mipmaps or without, anisotropically or not): one texel of
/// the filter's footprint at one of the two levels around the cone's, picked by p_stf_u
/// (rt_texture_filtering_position), unless the hardware's filter keeps the lookup (rt_texture_filtering_magnified).
/// Point-filtered materials already read one texel. An anisotropic material without gradients is filtered as an
/// isotropic one, as sample_material_texture above does.
vec4 sample_material_texture(uint tex_idx, vec2 uv, uint mat_flags, float p_lod_t, vec4 p_grad, vec4 p_stf_u) {
	const uint stf_filter = rt_texture_filtering();
	if (stf_filter == 0u || (mat_flags & 4u) != 0u) {
		return sample_material_texture(tex_idx, uv, mat_flags, p_lod_t, p_grad);
	}
	const bool mipmaps = (mat_flags & 128u) == 0u;
	// The elliptical footprint where T5's path would have read it.
	const bool footprint = (mat_flags & 64u) != 0u && p_grad != vec4(0.0);
	const uvec2 size = uvec2(textureSize(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT), 0));
	float level;
	if (footprint) {
		level = rt_texture_filtering_level(size, p_grad);
	} else {
		level = p_lod_t > RT_TEXTURE_LOD_BASE ? p_lod_t + 0.5 * log2(max(float(size.x) * float(size.y), 1.0)) : RT_TEXTURE_LOD_BASE;
	}
	if (rt_texture_filtering_magnified(level)) {
		return sample_material_texture(tex_idx, uv, mat_flags, p_lod_t, p_grad);
	}
	const uint levels = mipmaps ? uint(textureQueryLevels(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT))) : 1u;
	const vec3 pos = rt_texture_filtering_position(stf_filter, p_stf_u, size, levels, mipmaps, uv, level, p_grad, footprint);
	return textureLod(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT), pos.xy, pos.z);
}

/// One channel of a material texture by BaseMaterial3D's TextureChannel (D118): 0 red to 3 alpha, 4 the grey average.
float material_texture_channel(vec4 p_texel, uint p_channel) {
	return p_channel >= 4u ? dot(p_texel.rgb, vec3(1.0 / 3.0)) : p_texel[p_channel];
}

/// A standard material's roughness (x) and metallic (y) as the raster reads them (D118), from the lookups of its
/// orm_texture_idx and metallic_texture_idx (vec4(1.0), the white default, for an unassigned index 0; one lookup for
/// both when the indices are equal): each through its channel (flags bits 8-10 and 11-13), times its parameter.
vec2 material_roughness_metallic(MaterialData p_mat, vec4 p_roughness_tex, vec4 p_metallic_tex) {
	return clamp(vec2(material_texture_channel(p_roughness_tex, (p_mat.flags >> 8u) & 7u) * p_mat.roughness,
						 material_texture_channel(p_metallic_tex, (p_mat.flags >> 11u) & 7u) * p_mat.metallic),
			vec2(0.0), vec2(1.0));
}

/// VAL: the direction an anisotropic lobe's roughness stretches along (D118 MM1), as the raster's fragment turns its frame:
/// above an anisotropy of 0.01 the tangent becomes the flow's direction in the tangent and binormal's plane; otherwise
/// (a negative anisotropy among them) it stays the tangent. shade_and_bounce orthogonalises it against the normal.
vec3 rt_anisotropy_tangent(vec3 p_tangent, vec3 p_binormal, vec2 p_flow, float p_anisotropy) {
	vec3 t = normalize(p_tangent);
	if (p_anisotropy > 0.01) {
		t = p_flow.x * t + p_flow.y * normalize(p_binormal);
		float len = length(t);
		t = len > 1e-6 ? t / len : normalize(p_tangent);
	}
	return t;
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
#ifdef USE_LSS
		if (prev_addr != 0ul && (geom.flags & FLAG_LSS) != 0u) {
			// A segment (D126) moves the hit with its axis point at the hit's u: the midline's motion, the azimuth ignored,
			// as RTXCR's hair motion vectors take it.
			uint i0, i1;
			rt_lss_segment_indices(geom, uint(gl_PrimitiveID), i0, i1);
			const vec3 a = gl_HitLSSPositionsNV[0];
			const vec3 b = gl_HitLSSPositionsNV[1];
			const float s = clamp(attribs.x, 0.0, 1.0);
			FloatBuffer prev_vb = FloatBuffer(prev_addr);
			uint stride_floats = geom.position_stride >> 2;
			vec3 p0 = vec3(prev_vb.v[i0 * stride_floats + 0u], prev_vb.v[i0 * stride_floats + 1u], prev_vb.v[i0 * stride_floats + 2u]);
			vec3 p1 = vec3(prev_vb.v[i1 * stride_floats + 0u], prev_vb.v[i1 * stride_floats + 1u], prev_vb.v[i1 * stride_floats + 2u]);
			prev_obj_pos = obj_pos + (mix(p0, p1, s) - mix(a, b, s));
			prev_addr = 0ul; // Settled: no triangle below.
		}
#endif
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
