#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Cluster LOD selection (D86): one thread per cluster of one instance, the instances along y.
// A cluster is traced when its group's simplification would show at this instance's distance (the
// group's simplified error over its distance to the camera reaches the threshold, so the coarser
// level it feeds is not usable) and the finer group it was simplified from would not show (so this
// level is not needed finer), one level covering every part of the surface (meshoptimizer's
// clusterlod.h, its selection rule). The selected clusters' CLAS addresses are appended to the
// instance's reference list, and the count lands in the first word of its bottom-level build info,
// which the cluster BLAS build that follows reads.
//
// FrameBuf is the frame's argument buffer: the build infos (count, stride, address; count is the
// atomic), then per instance 16 words (the object-to-trace transform as three rows of four, the
// uniform scale, the cluster count, the surface's word offset in TableBuf, the reference list's word
// offset in FrameBuf), then the reference lists. TableBuf holds every hierarchy's region: per cluster
// its CLAS address (lo, hi), then per cluster its group and the group it refines (-1 at LOD 0), then
// per group eight words (center, radius, error, depth, first cluster, cluster count).

layout(set = 0, binding = 0) restrict buffer FrameBuf {
	uint v[];
}
frame;
layout(set = 0, binding = 1) restrict readonly buffer TableBuf {
	uint v[];
}
table;

layout(push_constant, std430) uniform PC {
	vec4 camera_threshold; // xyz: the camera in trace space; w: the error-over-distance threshold.
	float near_distance; // The distance the error is judged at, at least.
	uint entry_count;
	uint instances_offset; // Words into FrameBuf.
	uint blas_infos_offset; // Words into FrameBuf.
	uint blas_info_stride; // Words.
	uint flags;
	uint pad0;
	uint pad1;
}
pc;

#define FLAG_ORTHOGONAL 1u
#define INSTANCE_WORDS 16u
#define GROUP_WORDS 8u

// The group's sphere in trace space against the camera: its simplification shows when
// error * scale / max(near, distance - radius * scale) >= threshold, written without the division
// (the distance is positive). A terminal group's error is FLT_MAX, so it always shows.
bool group_error_visible(uint p_group, vec4 p_row0, vec4 p_row1, vec4 p_row2, float p_scale) {
	vec4 center = vec4(uintBitsToFloat(table.v[p_group + 0u]), uintBitsToFloat(table.v[p_group + 1u]), uintBitsToFloat(table.v[p_group + 2u]), 1.0);
	float radius = uintBitsToFloat(table.v[p_group + 3u]);
	float error = uintBitsToFloat(table.v[p_group + 4u]);
	float distance = 1.0;
	if ((pc.flags & FLAG_ORTHOGONAL) == 0u) {
		vec3 world = vec3(dot(p_row0, center), dot(p_row1, center), dot(p_row2, center));
		distance = max(pc.near_distance, length(world - pc.camera_threshold.xyz) - radius * p_scale);
	}
	return error * p_scale >= pc.camera_threshold.w * distance;
}

void main() {
	uint entry = gl_GlobalInvocationID.y;
	uint cluster = gl_GlobalInvocationID.x;
	if (entry >= pc.entry_count) {
		return;
	}
	uint inst = pc.instances_offset + entry * INSTANCE_WORDS;
	uint cluster_count = frame.v[inst + 13u];
	if (cluster >= cluster_count) {
		return;
	}
	vec4 row0 = vec4(uintBitsToFloat(frame.v[inst + 0u]), uintBitsToFloat(frame.v[inst + 1u]), uintBitsToFloat(frame.v[inst + 2u]), uintBitsToFloat(frame.v[inst + 3u]));
	vec4 row1 = vec4(uintBitsToFloat(frame.v[inst + 4u]), uintBitsToFloat(frame.v[inst + 5u]), uintBitsToFloat(frame.v[inst + 6u]), uintBitsToFloat(frame.v[inst + 7u]));
	vec4 row2 = vec4(uintBitsToFloat(frame.v[inst + 8u]), uintBitsToFloat(frame.v[inst + 9u]), uintBitsToFloat(frame.v[inst + 10u]), uintBitsToFloat(frame.v[inst + 11u]));
	float scale = uintBitsToFloat(frame.v[inst + 12u]);
	uint table_offset = frame.v[inst + 14u];
	uint list_offset = frame.v[inst + 15u];

	uint record = table_offset + cluster_count * 2u + cluster * 2u;
	uint group = table.v[record];
	int refined = int(table.v[record + 1u]);
	uint groups = table_offset + cluster_count * 4u;

	bool selected = group_error_visible(groups + group * GROUP_WORDS, row0, row1, row2, scale);
	if (selected && refined >= 0) {
		selected = !group_error_visible(groups + uint(refined) * GROUP_WORDS, row0, row1, row2, scale);
	}
	if (!selected) {
		return;
	}
	uint slot = atomicAdd(frame.v[pc.blas_infos_offset + entry * pc.blas_info_stride], 1u);
	frame.v[list_offset + slot * 2u] = table.v[table_offset + cluster * 2u];
	frame.v[list_offset + slot * 2u + 1u] = table.v[table_offset + cluster * 2u + 1u];
}
