#[compute]

#version 450

#VERSION_DEFINES

// The runtime passes of a subdivided surface (D122 S2, docs/character-subdivision.md in the Valence
// repository). The payload's layout is RenderingServerTypes::MeshSubdivisionConstants'; every offset
// below is in words from the start of a payload. The passes run in order, with a barrier between them:
// the cage (morphed, then skinned), the refined vertices as stencil sums of it, their normals, and the
// surface's vertices written into the instance's vertex buffer in the mesh's own layout. Each pass is
// its own variant (MODE_CAGE, MODE_STENCIL, MODE_NORMAL, MODE_WRITE), so each is compiled alone (S3b).
// The normals and tangents follow SubdivisionBuilder's rest-pose functions operation for operation, so
// both owners of a seam vertex compute it from the same bits in the same order.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer restrict readonly OwnerPayload {
	uint data[];
}
owner_payload;

// float x, y, z per local cage vertex: own, then ghosts.
layout(set = 0, binding = 1, std430) buffer restrict CageVertices {
	float data[];
}
cage;

// float x, y, z per refined vertex U: own, then ghosts.
layout(set = 0, binding = 2, std430) buffer restrict RefinedPositions {
	float data[];
}
refined_positions;

// float x, y, z per own refined vertex.
layout(set = 0, binding = 3, std430) buffer restrict RefinedNormals {
	float data[];
}
refined_normals;

// Every blend shape's weight, in the mesh's order.
layout(set = 0, binding = 4, std430) buffer restrict readonly BlendShapeWeights {
	float data[];
}
blend_weights;

layout(set = 1, binding = 0, std430) buffer restrict readonly SurfacePayload {
	uint data[];
}
surface_payload;

layout(set = 1, binding = 1, std430) buffer restrict writeonly DstVertexData {
	uint data[];
}
dst_vertices;

layout(set = 1, binding = 2, std430) buffer restrict readonly AttributeData {
	uint data[];
}
attributes;

layout(set = 1, binding = 3, std430) buffer restrict readonly IndexData {
	uint data[];
}
indices;

layout(set = 2, binding = 0, std430) buffer restrict readonly SkeletonData {
	vec4 data[];
}
bone_transforms;

layout(push_constant, std430) uniform Params {
	uint pad0;
	uint count;
	bool has_skin;
	bool has_blend;

	uint cage_positions;
	uint cage_skin;
	uint blend_offsets;
	uint blend_entries;

	uint stencil_offsets;
	uint stencil_indices;
	uint stencil_weights;
	bool stencil_index_16;

	uint quads;
	uint normal_offsets;
	uint normal_entries;
	bool index_16;

	uint r_to_u;
	uint tangent_offsets;
	uint tangent_entries;
	uint attribute_stride;

	uint uv_offset;
	bool has_normal;
	bool has_tangent;
	uint normal_tangent_stride;

	bool has_uv;
	uint pad1;
	uint pad2;
	uint pad3;
}
params;

// The encoders skeleton.glsl writes deformed vertices with.

vec2 signNotZero(vec2 v) {
	return mix(vec2(-1.0), vec2(1.0), greaterThanEqual(v.xy, vec2(0.0)));
}

uint vec2_to_uint(vec2 base) {
	uvec2 enc = uvec2(clamp(ivec2(base * vec2(65535, 65535)), ivec2(0), ivec2(0xFFFF, 0xFFFF))) << uvec2(0, 16);
	return enc.x | enc.y;
}

vec2 vec3_to_oct(vec3 e) {
	e /= abs(e.x) + abs(e.y) + abs(e.z);
	vec2 oct = e.z >= 0.0f ? e.xy : (vec2(1.0f) - abs(e.yx)) * signNotZero(e.xy);
	return oct * 0.5f + 0.5f;
}

uint encode_norm_to_uint_oct(vec3 base) {
	return vec2_to_uint(vec3_to_oct(base));
}

uint encode_tang_to_uint_oct(vec4 base) {
	vec2 oct = vec3_to_oct(base.xyz);
	// Encode binormal sign in y component
	oct.y = oct.y * 0.5f + 0.5f;
	oct.y = base.w >= 0.0f ? oct.y : 1 - oct.y;

	if (oct.x == 0.0 && oct.y == 1.0) {
		// (1, 1) and (0, 1) decode to the same value, but (0, 1) messes with our compression detection.
		// So we sanitize here.
		oct.x = 1.0;
	}

	return vec2_to_uint(oct);
}

// Godot's Vector3::normalized() and is_zero_approx(), as the builder calls them.
vec3 godot_normalized(vec3 v) {
	float length_squared = v.x * v.x + v.y * v.y + v.z * v.z;
	return length_squared == 0.0 ? vec3(0.0) : v / sqrt(length_squared);
}

bool godot_is_zero_approx(vec3 v) {
	return all(lessThan(abs(v), vec3(0.00001)));
}

// The angle between a corner's two edges, the weight Blender gives a face's normal at a vertex.
float corner_angle(vec3 prev, vec3 corner, vec3 next) {
	vec3 to_prev = godot_normalized(prev - corner);
	vec3 to_next = godot_normalized(next - corner);
	return acos(clamp(to_prev.x * to_next.x + to_prev.y * to_next.y + to_prev.z * to_next.z, -1.0, 1.0));
}

vec3 owner_vec3(uint offset) {
	return uintBitsToFloat(uvec3(owner_payload.data[offset], owner_payload.data[offset + 1], owner_payload.data[offset + 2]));
}

vec3 cage_vertex(uint c) {
	return vec3(cage.data[c * 3], cage.data[c * 3 + 1], cage.data[c * 3 + 2]);
}

vec3 refined_position(uint u) {
	return vec3(refined_positions.data[u * 3], refined_positions.data[u * 3 + 1], refined_positions.data[u * 3 + 2]);
}

vec3 refined_normal(uint u) {
	return vec3(refined_normals.data[u * 3], refined_normals.data[u * 3 + 1], refined_normals.data[u * 3 + 2]);
}

mat4 bone_matrix(uint bone) {
	uint b = bone * 3;
	return mat4(bone_transforms.data[b], bone_transforms.data[b + 1], bone_transforms.data[b + 2], vec4(0.0, 0.0, 0.0, 1.0));
}

uint surface_index(uint i) {
	if (params.index_16) {
		return (indices.data[i >> 1] >> ((i & 1) * 16)) & 0xFFFF;
	}
	return indices.data[i];
}

vec2 surface_uv(uint r) {
	uint offset = r * params.attribute_stride + params.uv_offset;
	return uintBitsToFloat(uvec2(attributes.data[offset], attributes.data[offset + 1]));
}

vec3 surface_position(uint r) {
	return refined_position(surface_payload.data[params.r_to_u + r]);
}

void main() {
	uint index = gl_GlobalInvocationID.x;
	if (index >= params.count) {
		return;
	}

#if defined(MODE_CAGE)
	{
		// The rest position plus the weighted deltas in ascending shape order, then skinned as
		// skeleton.glsl skins a position with 8 bones.
		vec3 vertex = owner_vec3(params.cage_positions + index * 3);
		if (params.has_blend) {
			vec3 blend_vertex = vec3(0.0);
			uint last = owner_payload.data[params.blend_offsets + index + 1];
			for (uint e = owner_payload.data[params.blend_offsets + index]; e < last; e++) {
				uint entry = params.blend_entries + e * 4;
				float w = blend_weights.data[owner_payload.data[entry]];
				if (abs(w) > 0.0001) {
					blend_vertex += owner_vec3(entry + 1) * w;
				}
			}
			vertex += blend_vertex;
		}
		if (params.has_skin) {
			uint skin = params.cage_skin + index * 8;
			uvec4 bones = uvec4(owner_payload.data[skin], owner_payload.data[skin + 1], owner_payload.data[skin + 2], owner_payload.data[skin + 3]);
			uvec4 weights = uvec4(owner_payload.data[skin + 4], owner_payload.data[skin + 5], owner_payload.data[skin + 6], owner_payload.data[skin + 7]);
			vec2 weights_01 = unpackUnorm2x16(weights.x);
			vec2 weights_23 = unpackUnorm2x16(weights.y);
			vec2 weights_45 = unpackUnorm2x16(weights.z);
			vec2 weights_67 = unpackUnorm2x16(weights.w);
			mat4 m = bone_matrix(bones.x & 0xFFFF) * weights_01.x;
			m += bone_matrix(bones.x >> 16) * weights_01.y;
			m += bone_matrix(bones.y & 0xFFFF) * weights_23.x;
			m += bone_matrix(bones.y >> 16) * weights_23.y;
			m += bone_matrix(bones.z & 0xFFFF) * weights_45.x;
			m += bone_matrix(bones.z >> 16) * weights_45.y;
			m += bone_matrix(bones.w & 0xFFFF) * weights_67.x;
			m += bone_matrix(bones.w >> 16) * weights_67.y;
			//reverse order because its transposed
			vertex = (vec4(vertex, 1.0) * m).xyz;
		}
		cage.data[index * 3] = vertex.x;
		cage.data[index * 3 + 1] = vertex.y;
		cage.data[index * 3 + 2] = vertex.z;
	}
#elif defined(MODE_STENCIL)
	{
		// The refined vertex as its stencil's weighted sum of deformed cage vertices, in stored order.
		vec3 sum = vec3(0.0);
		uint last = owner_payload.data[params.stencil_offsets + index + 1];
		for (uint i = owner_payload.data[params.stencil_offsets + index]; i < last; i++) {
			uint c;
			if (params.stencil_index_16) {
				c = (owner_payload.data[params.stencil_indices + (i >> 1)] >> ((i & 1) * 16)) & 0xFFFF;
			} else {
				c = owner_payload.data[params.stencil_indices + i];
			}
			sum += cage_vertex(c) * uintBitsToFloat(owner_payload.data[params.stencil_weights + i]);
		}
		refined_positions.data[index * 3] = sum.x;
		refined_positions.data[index * 3 + 1] = sum.y;
		refined_positions.data[index * 3 + 2] = sum.z;
	}
#elif defined(MODE_NORMAL)
	{
		// Blender's vertex normal: each quad's normal_quad_v3 (the cross product of its diagonals)
		// weighted by its corner's angle, summed in welded quad order. A quad's normal is recomputed per
		// corner, from the same bits, so it is the same at each.
		vec3 sum = vec3(0.0);
		uint last = owner_payload.data[params.normal_offsets + index + 1];
		for (uint e = owner_payload.data[params.normal_offsets + index]; e < last; e++) {
			uint entry = owner_payload.data[params.normal_entries + e];
			uint quad = params.quads + (entry >> 2) * 4;
			uint corner = entry & 3;
			vec3 p0 = refined_position(owner_payload.data[quad]);
			vec3 p1 = refined_position(owner_payload.data[quad + 1]);
			vec3 p2 = refined_position(owner_payload.data[quad + 2]);
			vec3 p3 = refined_position(owner_payload.data[quad + 3]);
			vec3 quad_normal = godot_normalized(cross(p0 - p2, p1 - p3));
			vec3 prev = refined_position(owner_payload.data[quad + ((corner + 3) & 3)]);
			vec3 here = refined_position(owner_payload.data[quad + corner]);
			vec3 next = refined_position(owner_payload.data[quad + ((corner + 1) & 3)]);
			sum += quad_normal * corner_angle(prev, here, next);
		}
		vec3 normal = godot_is_zero_approx(sum) ? vec3(0.0, 1.0, 0.0) : godot_normalized(sum);
		refined_normals.data[index * 3] = normal.x;
		refined_normals.data[index * 3 + 1] = normal.y;
		refined_normals.data[index * 3 + 2] = normal.z;
	}
#else // MODE_WRITE
	{
		// The surface's vertex R: its refined vertex's position and normal, and its tangent from the
		// triangles' UV gradients, each weighted by its corner's angle, orthogonalized against the normal.
		uint u = surface_payload.data[params.r_to_u + index];
		vec3 vertex = refined_position(u);
		uvec3 uvertex = floatBitsToUint(vertex);
		dst_vertices.data[index * 3] = uvertex.x;
		dst_vertices.data[index * 3 + 1] = uvertex.y;
		dst_vertices.data[index * 3 + 2] = uvertex.z;

		uint dst_normal = params.count * 3 + index * params.normal_tangent_stride;
		vec3 normal = refined_normal(u);
		if (params.has_normal) {
			dst_vertices.data[dst_normal] = encode_norm_to_uint_oct(normal);
			dst_normal++;
		}
		if (params.has_tangent && !params.has_uv) {
			// Without UVs, the arbitrary tangent RenderingServer gives a surface with normals.
			vec3 tangent = godot_normalized(cross(vec3(normal.z, -normal.x, normal.y), godot_normalized(normal)));
			dst_vertices.data[dst_normal] = encode_tang_to_uint_oct(vec4(tangent, 1.0));
		} else if (params.has_tangent) {
			vec3 tangent_sum = vec3(0.0);
			vec3 bitangent_sum = vec3(0.0);
			uint last = surface_payload.data[params.tangent_offsets + index + 1];
			for (uint e = surface_payload.data[params.tangent_offsets + index]; e < last; e++) {
				uint entry = surface_payload.data[params.tangent_entries + e];
				uint triangle = entry / 3;
				uint corner = entry - triangle * 3;
				uint r0 = surface_index(triangle * 3);
				uint r1 = surface_index(triangle * 3 + 1);
				uint r2 = surface_index(triangle * 3 + 2);
				vec3 p0 = surface_position(r0);
				vec2 uv0 = surface_uv(r0);
				vec3 e1 = surface_position(r1) - p0;
				vec3 e2 = surface_position(r2) - p0;
				vec2 d1 = surface_uv(r1) - uv0;
				vec2 d2 = surface_uv(r2) - uv0;
				float det = d1.x * d2.y - d2.x * d1.y;
				if (det == 0.0) {
					continue;
				}
				float orientation = det < 0.0 ? -1.0 : 1.0;
				vec3 tangent = godot_normalized((e1 * d2.y - e2 * d1.y) * orientation);
				vec3 bitangent = godot_normalized((e2 * d1.x - e1 * d2.x) * orientation);
				vec3 prev = surface_position(surface_index(triangle * 3 + (corner + 2) % 3));
				vec3 here = surface_position(surface_index(triangle * 3 + corner));
				vec3 next = surface_position(surface_index(triangle * 3 + (corner + 1) % 3));
				float angle = corner_angle(prev, here, next);
				tangent_sum += tangent * angle;
				bitangent_sum += bitangent * angle;
			}
			vec3 t = tangent_sum - normal * dot(normal, tangent_sum);
			if (godot_is_zero_approx(t)) {
				// Vector3::get_any_perpendicular().
				bool right = abs(normal.x) <= abs(normal.y) && abs(normal.x) <= abs(normal.z);
				t = godot_normalized(cross(normal, right ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
			}
			t = godot_normalized(t);
			float binormal_sign = dot(cross(normal, t), bitangent_sum) < 0.0 ? 1.0 : -1.0;
			dst_vertices.data[dst_normal] = encode_tang_to_uint_oct(vec4(t, binormal_sign));
		}
	}
#endif
}
