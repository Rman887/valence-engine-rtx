// Shared utilities for hit shaders (closest_hit, any_hit)
// Requires: GL_EXT_buffer_reference, GL_ARB_gpu_shader_int64, oct_inc.glsl, raytracing_inc.glsl

#include "raytracing_data_inc.glsl"

// ============================================================================
// VERTEX ATTRIBUTES
// ============================================================================
struct VertexAttributes {
	vec2 uv;
	vec3 normal;
	vec3 tangent;
	float bitangent_sign;
	vec4 color;
	bool has_uv;
	bool has_normal;
	bool has_tangent;
	bool has_color;
};

struct TBNResult {
	vec3 normal;
	vec3 tangent;
	vec3 binormal;
	float bitangent_sign;
};

// ============================================================================
// TRIANGLE INDEX FETCHING
// ============================================================================

/// Core implementation taking an explicit primitive ID (works in any shader stage).
void get_triangle_indices_ex(in GeometryData geom, uint primitive_id, out uint i0, out uint i1, out uint i2) {
	if (geom.index_format == 2u) {
		i0 = primitive_id * 3u;
		i1 = primitive_id * 3u + 1u;
		i2 = primitive_id * 3u + 2u;
	} else if (geom.index_address != 0ul) {
		Uint32Buffer idx = Uint32Buffer(geom.index_address);
		if (geom.index_format == 0u) {
			uint byte_off = primitive_id * 6u;
			uint word0 = idx.v[byte_off >> 2];
			uint word1 = idx.v[(byte_off >> 2) + 1u];
			if ((byte_off & 3u) == 0u) {
				i0 = word0 & 0xFFFFu;
				i1 = word0 >> 16;
				i2 = word1 & 0xFFFFu;
			} else {
				i0 = word0 >> 16;
				i1 = word1 & 0xFFFFu;
				i2 = word1 >> 16;
			}
		} else {
			i0 = idx.v[primitive_id * 3u];
			i1 = idx.v[primitive_id * 3u + 1u];
			i2 = idx.v[primitive_id * 3u + 2u];
		}
	} else {
		i0 = i1 = i2 = 0u;
	}
}

// ============================================================================
// TRIANGLE RESOLUTION (D93b)
// ============================================================================

// A triangle BLAS's hit, or a stage without cluster structures (gl_ClusterIDNoneNV's value).
const int RT_CLUSTER_ID_NONE = -1;
// A cluster geometry record's words and a blob section's alignment (RT_CLUSTER_GEOMETRY_RECORD_WORDS
// and RT_CLUSTER_GEOMETRY_ALIGNMENT on the host).
const uint CLUSTER_GEOMETRY_RECORD_WORDS = 4u;
const uint CLUSTER_GEOMETRY_ALIGNMENT = 16u;

#ifdef USE_CLUSTER_AS
// The cluster geometry records (D93b), in the LOD table: for each cluster of a
// FLAG_CLUSTER_GEOMETRY surface, from its cluster_record_base, the cluster's blob address (words 0
// and 1), its vertex count | triangle count << 16, and its first triangle in [LOD 0 | coarse tail].
layout(set = 0, binding = 45, std430) restrict readonly buffer ClusterGeometryRecords {
	uint cluster_geometry_records[];
};
#endif

/// A triangle of a hit or a ray-query candidate: its id in the surface's [LOD 0 | coarse tail],
/// which emission weighting (D50) keys on, and its three vertex indices. In a cluster BLAS (D79)
/// the primitive counts inside the cluster and the cluster's ID carries its first triangle, or on a
/// FLAG_CLUSTER_GEOMETRY surface its index (D93b): there the indices are local to the cluster's
/// blob, and geom is rewritten to read it (the normal block at its base, the attribute block after
/// it), so the fetches below read the blob unchanged. Its vertex and index addresses then no longer
/// name the surface's buffers, so a reader of positions keeps a copy of the record.
uint rt_resolve_triangle(inout GeometryData geom, int cluster_id, uint primitive, out uint i0, out uint i1, out uint i2) {
	uint triangle = primitive;
#ifdef USE_CLUSTER_AS
	if (cluster_id != gl_ClusterIDNoneNV) {
		if ((geom.flags & FLAG_CLUSTER_GEOMETRY) != 0u) {
			uint record = geom.cluster_record_base + uint(cluster_id) * CLUSTER_GEOMETRY_RECORD_WORDS;
			uint64_t blob = packUint2x32(uvec2(cluster_geometry_records[record], cluster_geometry_records[record + 1u]));
			uint vertex_count = cluster_geometry_records[record + 2u] & 0xFFFFu;
			// The blob's strides, as the population derives them: the compressed format's normal entry
			// widened to 8 bytes with the tangent angle's word, each section 16-byte aligned.
			bool has_normal = geom.normal_byte_offset != OFFSET_NONE;
			uint normal_stride = has_normal ? ((geom.flags & FLAG_COMPRESSED) != 0u ? 8u : geom.normal_stride) : 0u;
			uint attribute_stride = geom.attribute_stride; // A surface whose geometry streams addresses no buffer (D93d).
			const uint align_mask = CLUSTER_GEOMETRY_ALIGNMENT - 1u;
			uint64_t attributes = blob + uint64_t((vertex_count * normal_stride + align_mask) & ~align_mask);
			Uint32Buffer words = Uint32Buffer(attributes + uint64_t((vertex_count * attribute_stride + align_mask) & ~align_mask));
			uint b = primitive * 3u;
			i0 = (words.v[b >> 2u] >> ((b & 3u) << 3u)) & 0xFFu;
			b++;
			i1 = (words.v[b >> 2u] >> ((b & 3u) << 3u)) & 0xFFu;
			b++;
			i2 = (words.v[b >> 2u] >> ((b & 3u) << 3u)) & 0xFFu;
			geom.vertex_address = blob;
			geom.attribute_address = attribute_stride != 0u ? attributes : 0ul;
			geom.index_address = 0ul;
			geom.vertex_count = vertex_count;
			if (has_normal) {
				geom.normal_byte_offset = 0u;
				geom.normal_stride = normal_stride;
			}
			if (geom.tangent_byte_offset != OFFSET_NONE) {
				geom.tangent_byte_offset = 0u;
				geom.tangent_stride = normal_stride;
			}
			return cluster_geometry_records[record + 3u] + primitive;
		}
		triangle = uint(cluster_id) + primitive;
	}
#endif
	get_triangle_indices_ex(geom, triangle, i0, i1, i2);
	return triangle;
}

#ifndef RT_STAGE_RAYGEN
/// The hit's cluster ID (hit shaders only).
int rt_hit_cluster_id() {
#if defined(USE_CLUSTER_AS) && (defined(RT_STAGE_CLOSEST_HIT) || defined(RT_STAGE_ANY_HIT))
	return gl_ClusterIDNV;
#else
	return RT_CLUSTER_ID_NONE;
#endif
}

/// rt_resolve_triangle for the hit (hit shaders only).
uint rt_resolve_hit_triangle(inout GeometryData geom, out uint i0, out uint i1, out uint i2) {
	return rt_resolve_triangle(geom, rt_hit_cluster_id(), uint(gl_PrimitiveID), i0, i1, i2);
}
#endif

// ============================================================================
// UV FETCHING
// ============================================================================
vec2 fetch_uv(in GeometryData geom, uint i0, uint i1, uint i2, vec3 bary) {
	if (geom.uv_byte_offset == OFFSET_NONE || geom.attribute_address == 0ul) {
		return vec2(0.0);
	}

	uint stride_words = geom.attribute_stride >> 2;
	uint uv_word_off = geom.uv_byte_offset >> 2;
	bool compressed = (geom.flags & FLAG_COMPRESSED) != 0u;

	vec2 uv0, uv1, uv2;

	if (compressed) {
		Uint32Buffer attr = Uint32Buffer(geom.attribute_address);
		uint p0 = attr.v[i0 * stride_words + uv_word_off];
		uint p1 = attr.v[i1 * stride_words + uv_word_off];
		uint p2 = attr.v[i2 * stride_words + uv_word_off];

		uv0 = vec2(float(p0 & 0xFFFFu), float(p0 >> 16)) / 65535.0;
		uv1 = vec2(float(p1 & 0xFFFFu), float(p1 >> 16)) / 65535.0;
		uv2 = vec2(float(p2 & 0xFFFFu), float(p2 >> 16)) / 65535.0;

		vec2 scale = unpackHalf2x16(geom.uv_scale_packed);
		if (scale.x != 0.0 || scale.y != 0.0) {
			uv0 = (uv0 - 0.5) * scale;
			uv1 = (uv1 - 0.5) * scale;
			uv2 = (uv2 - 0.5) * scale;
		}
	} else {
		FloatBuffer attr_f = FloatBuffer(geom.attribute_address);
		uint stride_floats = geom.attribute_stride >> 2;
		uint uv_float_off = geom.uv_byte_offset >> 2;

		uv0 = vec2(attr_f.v[i0 * stride_floats + uv_float_off],
				attr_f.v[i0 * stride_floats + uv_float_off + 1u]);
		uv1 = vec2(attr_f.v[i1 * stride_floats + uv_float_off],
				attr_f.v[i1 * stride_floats + uv_float_off + 1u]);
		uv2 = vec2(attr_f.v[i2 * stride_floats + uv_float_off],
				attr_f.v[i2 * stride_floats + uv_float_off + 1u]);
	}

	return bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
}

// ============================================================================
// VERTEX COLOR FETCHING (RGBA8 packed in uint32)
// ============================================================================
vec4 fetch_color(in GeometryData geom, uint i0, uint i1, uint i2, vec3 bary) {
	if (geom.color_byte_offset == OFFSET_NONE || geom.attribute_address == 0ul) {
		return vec4(1.0);
	}

	Uint32Buffer attr = Uint32Buffer(geom.attribute_address);
	uint stride_words = geom.attribute_stride >> 2;
	uint color_word_off = geom.color_byte_offset >> 2;

	vec4 col0 = unpackUnorm4x8(attr.v[i0 * stride_words + color_word_off]);
	vec4 col1 = unpackUnorm4x8(attr.v[i1 * stride_words + color_word_off]);
	vec4 col2 = unpackUnorm4x8(attr.v[i2 * stride_words + color_word_off]);

	return bary.x * col0 + bary.y * col1 + bary.z * col2;
}

// ============================================================================
// TBN FETCHING
// ============================================================================
void axis_angle_to_tbn(vec3 axis, float angle, out vec3 tangent, out vec3 binormal, out vec3 normal) {
	float c = cos(angle);
	float s = sin(angle);
	vec3 omc_axis = (1.0 - c) * axis;
	vec3 s_axis = s * axis;
	tangent = omc_axis.xxx * axis + vec3(c, -s_axis.z, s_axis.y);
	binormal = omc_axis.yyy * axis + vec3(s_axis.z, c, -s_axis.x);
	normal = omc_axis.zzz * axis + vec3(-s_axis.y, s_axis.x, c);
}

TBNResult fetch_tbn_compressed_vertex(in GeometryData geom, uint idx) {
	TBNResult result;
	Uint32Buffer vb = Uint32Buffer(geom.vertex_address);

	uint norm_base = geom.normal_byte_offset >> 2;
	uint norm_stride = geom.normal_stride >> 2;
	uint n_packed = vb.v[norm_base + idx * norm_stride];
	vec2 axis_oct = vec2(float(n_packed & 0xFFFFu), float(n_packed >> 16)) / 65535.0 * 2.0 - 1.0;
	vec3 axis = oct_to_vec3(axis_oct);

	uint pos_word1 = vb.v[idx * 2u + 1u];
	float angle_raw = float(pos_word1 >> 16) / 65535.0;

	result.bitangent_sign = angle_raw > 0.5 ? 1.0 : -1.0;
	float angle = abs(angle_raw * 2.0 - 1.0) * PI;

	axis_angle_to_tbn(axis, angle, result.tangent, result.binormal, result.normal);
	result.binormal *= result.bitangent_sign;

	return result;
}

TBNResult fetch_tbn_uncompressed_vertex(in GeometryData geom, uint idx) {
	TBNResult result;
	Uint32Buffer vb = Uint32Buffer(geom.vertex_address);

	uint norm_base = geom.normal_byte_offset >> 2;
	uint norm_stride = geom.normal_stride >> 2;

	uint n_packed = vb.v[norm_base + idx * norm_stride];
	vec2 n_oct = vec2(float(n_packed & 0xFFFFu), float(n_packed >> 16)) / 65535.0 * 2.0 - 1.0;
	result.normal = oct_to_vec3(n_oct);

	if (geom.normal_stride >= 8u) {
		uint t_packed = vb.v[norm_base + idx * norm_stride + 1u];
		vec2 t_raw = vec2(float(t_packed & 0xFFFFu), float(t_packed >> 16)) / 65535.0 * 2.0 - 1.0;

		vec2 t_oct = vec2(t_raw.x, abs(t_raw.y) * 2.0 - 1.0);
		result.tangent = oct_to_vec3(t_oct);
		result.bitangent_sign = sign(t_raw.y);
		result.binormal = normalize(cross(result.normal, result.tangent) * result.bitangent_sign);
	} else {
		result.tangent = vec3(1.0, 0.0, 0.0);
		result.binormal = vec3(0.0, 0.0, 1.0);
		result.bitangent_sign = 1.0;
	}

	return result;
}

TBNResult fetch_tbn(in GeometryData geom, uint i0, uint i1, uint i2, vec3 bary) {
	TBNResult result;
	result.normal = vec3(0.0, 1.0, 0.0);
	result.tangent = vec3(1.0, 0.0, 0.0);
	result.binormal = vec3(0.0, 0.0, 1.0);
	result.bitangent_sign = 1.0;

	if (geom.normal_byte_offset == OFFSET_NONE || geom.vertex_address == 0ul) {
		return result;
	}

	bool compressed = (geom.flags & FLAG_COMPRESSED) != 0u;

	TBNResult t0, t1, t2;
	if (compressed) {
		t0 = fetch_tbn_compressed_vertex(geom, i0);
		t1 = fetch_tbn_compressed_vertex(geom, i1);
		t2 = fetch_tbn_compressed_vertex(geom, i2);
	} else {
		t0 = fetch_tbn_uncompressed_vertex(geom, i0);
		t1 = fetch_tbn_uncompressed_vertex(geom, i1);
		t2 = fetch_tbn_uncompressed_vertex(geom, i2);
	}

	result.normal = normalize(bary.x * t0.normal + bary.y * t1.normal + bary.z * t2.normal);
	result.tangent = normalize(bary.x * t0.tangent + bary.y * t1.tangent + bary.z * t2.tangent);
	result.binormal = normalize(bary.x * t0.binormal + bary.y * t1.binormal + bary.z * t2.binormal);
	result.bitangent_sign = sign(bary.x * t0.bitangent_sign + bary.y * t1.bitangent_sign + bary.z * t2.bitangent_sign);

	return result;
}

/// The instance's own linear part. The instance transform of a compressed surface folds in its unit box (the BLAS is built in it),
/// while the surface's normals and tangents are in its own space, so the box comes back out.
mat3 rt_instance_basis(GeometryData geom, mat3 object_to_world) {
	if ((geom.flags & FLAG_COMPRESSED) != 0u) {
		object_to_world[0] /= geom.aabb_size_x;
		object_to_world[1] /= geom.aabb_size_y;
		object_to_world[2] /= geom.aabb_size_z;
	}
	return object_to_world;
}

/// The normal matrix of a basis: its inverse transpose up to a positive factor (the cofactor
/// matrix signed by the determinant), so normals stay perpendicular under non-uniform scale
/// without inverting a basis that may be nearly flat. Tangents take it too, as in the raster.
mat3 rt_normal_basis(mat3 basis) {
	mat3 cofactor = mat3(cross(basis[1], basis[2]), cross(basis[2], basis[0]), cross(basis[0], basis[1]));
	return dot(basis[0], cofactor[0]) < 0.0 ? -cofactor : cofactor;
}

// ============================================================================
// VERTEX ATTRIBUTE FETCHING (requires hitAttributeEXT vec2 attribs to be declared)
// ============================================================================
#define FETCH_UV (1u << 0)
#define FETCH_TBN (1u << 1)
#define FETCH_COLOR (1u << 2)
#define FETCH_ALL (FETCH_UV | FETCH_TBN | FETCH_COLOR)

#ifdef RT_HIT_ATTRIBS_DECLARED
/// geom and the indices as rt_resolve_hit_triangle returned them.
VertexAttributes fetch_vertex_attributes(in GeometryData geom, uint i0, uint i1, uint i2, vec2 hit_attribs, uint fetch_flags) {
	VertexAttributes attrs;
	attrs.uv = vec2(0.0);
	attrs.normal = vec3(0.0, 1.0, 0.0);
	attrs.tangent = vec3(1.0, 0.0, 0.0);
	attrs.bitangent_sign = 1.0;
	attrs.color = vec4(1.0);
	attrs.has_uv = false;
	attrs.has_normal = false;
	attrs.has_tangent = false;
	attrs.has_color = false;

	vec3 bary = vec3(1.0 - hit_attribs.x - hit_attribs.y, hit_attribs.x, hit_attribs.y);

	if ((fetch_flags & FETCH_UV) != 0u && geom.uv_byte_offset != OFFSET_NONE) {
		attrs.uv = fetch_uv(geom, i0, i1, i2, bary);
		attrs.has_uv = true;
	}

	if ((fetch_flags & FETCH_COLOR) != 0u && geom.color_byte_offset != OFFSET_NONE) {
		attrs.color = fetch_color(geom, i0, i1, i2, bary);
		attrs.has_color = true;
	}

	if ((fetch_flags & FETCH_TBN) != 0u && geom.normal_byte_offset != OFFSET_NONE) {
		TBNResult tbn = fetch_tbn(geom, i0, i1, i2, bary);
		attrs.normal = tbn.normal;
		attrs.tangent = tbn.tangent;
		attrs.bitangent_sign = tbn.bitangent_sign;
		attrs.has_normal = true;
		attrs.has_tangent = true;
	}

	return attrs;
}
#endif
