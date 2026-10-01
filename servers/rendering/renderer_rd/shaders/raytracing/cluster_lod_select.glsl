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
// A deformed instance (D87) has no CLAS until this frame builds them: its selected cluster takes
// the next slot of the instance's destination range as its CLAS address (the reference list and
// the CLAS build's destination array both get it) and its build info, written once on the CPU into
// the instance's table region, is copied to the CLAS build's info array at the next free index,
// whose count the build reads from the device.
//
// A cached level (D90) is an item with no instance: it takes the clusters of one depth's discrete
// level, depth standing for the error (a cluster is taken when its group is at that depth or
// deeper, or terminal, and the group it was simplified from is above it), the whole surface at
// that depth for any instance. The CPU counts the same clusters to size its slot and puts the
// count in the record's word 18; the shader takes no more, so the two rules must stay one.
//
// The traversal (D91): an item flagged INSTANCE_FLAG_TREE is not tested cluster by cluster but
// walks its hierarchy's tree of spheres (words 20 and 21 of its record are the tree's nodes and
// refs in TableBuf, word 22 the basis's Frobenius norm), one dispatch per breadth-first level with
// FLAG_NODES set: each item of the level's queue is a node, half a workgroup tests its children (at
// most 32), passing child nodes go to the next level's queue and passing groups to the group queue,
// whose workgroup count DispatchBuf counts; then one indirect dispatch with FLAG_CLUSTERS gives each
// queued group half a workgroup, a lane per cluster, testing it against the group it refines. The
// group and the cluster tests are the exhaustive test's rule on the same words, so the selection is
// the same; a lane per cluster rather than a loop per group keeps each thread's chain of dependent
// loads and returning atomics short. A node's sphere encloses its children's and its error is
// their largest, and its test takes the Frobenius norm for the scale and a slack for the float
// arithmetic, so it passes whenever any child's does. With FLAG_CHECK the exhaustive test runs
// over the tree items and only counts, for a verbose run's comparison.
//
// FrameBuf is the frame's argument buffer: the build infos (count, stride, address; count is the
// atomic), then per instance INSTANCE_WORDS words (the object-to-trace transform as three rows of
// four, the uniform scale, the cluster count, the surface's word offset in TableBuf, the reference
// list's word offset in FrameBuf, the flags, the word offset of the instance's build infos in
// TableBuf, the destination base address as lo and hi), the CLAS count, then the reference lists,
// the CLAS destinations and the CLAS build infos. TableBuf holds every hierarchy's region: per
// cluster its CLAS address (lo, hi; zero for a deformed hierarchy), then per cluster its group and
// the group it refines (-1 at LOD 0), then per group eight words (center, radius, error, depth,
// first cluster, cluster count), then, for a deformed hierarchy, per cluster its build info.
//
// CLAS residency (D92): an item whose record's word 23 is non-zero is a streamed surface's, and
// word 23 the word offset of its per-group state in TableBuf (STATE_WORDS a group: the frame it
// was last requested and the frame its use was last reported, both written here by atomics, then
// flags the CPU writes once). Only the groups whose CLAS are resident have addresses (a group is
// resident when its first cluster's address is non-zero), and the rule is clamped by residency: a
// group is taken only if resident, and a cluster's refined group stands in for it only if resident
// too, so what is traced is the finest resident cut (the CPU loads coarse to fine and unloads fine
// to coarse, which keeps residency monotone along the DAG as the error is). A cluster whose refined
// group would be taken but is not resident requests it, once a frame; a group no cluster refines
// from (STATE_FLAG_ROOT: its simplification left nothing, as debris's does) has no such cluster,
// and is requested by its own test instead. A resident group taken reports its use every
// STREAM_TOUCH_PERIOD frames. Both go to capped lists, each after its count, which the CPU reads
// back a frame queue later.

// FrameBuf and DispatchBuf are not restrict: the cluster pass binds the frame's arguments to both.
layout(set = 0, binding = 0) buffer FrameBuf {
	uint v[];
}
frame;
// Written only in the streamed items' per-group state (D92), by atomics.
layout(set = 0, binding = 1) restrict buffer TableBuf {
	uint v[];
}
table;
// The cluster pass's indirect arguments (x, y, z), which the leaves' appends size; the cluster pass
// itself, which reads them as its dispatch, has the frame's arguments bound here instead.
layout(set = 0, binding = 2) buffer DispatchBuf {
	uint v[];
}
dispatch;

layout(push_constant, std430) uniform PC {
	vec4 camera_threshold; // xyz: the camera in trace space; w: the error-over-distance threshold.
	float near_distance; // The distance the error is judged at, at least.
	uint entry_count;
	uint instances_offset; // Words into FrameBuf.
	uint blas_infos_offset; // Words into FrameBuf.
	uint blas_info_stride; // Words.
	uint flags;
	uint clas_count_offset; // Words into FrameBuf: the CLAS build's count.
	uint clas_dst_offset; // Words into FrameBuf: the CLAS destinations (lo, hi).
	uint clas_infos_offset; // Words into FrameBuf: the CLAS build infos.
	uint clas_info_words; // The driver's build info, in words.
	uint clas_slot_size; // Bytes between destinations.
	uint queue_in; // Words into FrameBuf: the level's queue (entry, node), with FLAG_NODES.
	uint queue_out; // Words into FrameBuf: the next level's queue.
	uint count_in; // Word of FrameBuf: the level's queue length.
	uint count_out; // Word of FrameBuf: the next level's queue length, an atomic.
	uint check_offset; // Words into FrameBuf: per item, the count FLAG_CHECK makes.
	uint group_queue; // Words into FrameBuf: the group queue (entry, group record).
	uint group_count; // Word of FrameBuf: the group queue's length, an atomic.
	uint entry_first; // The exhaustive test's first item; its dispatch covers its range along y.
	uint frame_index; // The frame's number, for the streamed items' state (D92).
	uint stream_requests; // Words into FrameBuf: the request count, an atomic, then from 4 on the requests (table offset, group), STREAM_REQUEST_CAP of them (D92).
	uint stream_touches; // Words into FrameBuf: the use report count, then from 4 on the reports, STREAM_TOUCH_CAP of them.
	uint pad0;
	uint pad1;
	uint pad2;
	uint pad3;
}
pc;

#define FLAG_ORTHOGONAL 1u
#define FLAG_CHECK 2u
#define FLAG_NODES 4u
#define FLAG_CLUSTERS 8u
#define INSTANCE_WORDS 24u
#define INSTANCE_FLAG_DEFORMED 1u
#define INSTANCE_FLAG_LEVEL 2u
#define INSTANCE_FLAG_TREE 4u
#define GROUP_WORDS 8u
#define NODE_WORDS 8u
// Must match RT_LOD_STREAM_* in render_raytracing.h (D92).
#define STREAM_REQUEST_CAP 8192u
#define STREAM_TOUCH_CAP 16384u
#define STREAM_TOUCH_PERIOD 8u
#define STATE_WORDS 3u
#define STATE_FLAG_ROOT 1u

// The group's sphere in trace space against the camera: its simplification shows when
// error * scale / max(near, distance - radius * scale) >= threshold, written without the division
// (the distance is positive). A terminal group's error is FLT_MAX, so it always shows.
// The test is precise (D91): it is inlined into the exhaustive test, the traversal's leaves and its
// cluster pass, and the traversal selects what the exhaustive test does only if every copy rounds
// the same way, which contraction into fused operations could break.
bool group_error_visible(uint p_group, vec4 p_row0, vec4 p_row1, vec4 p_row2, float p_scale) {
	vec4 center = vec4(uintBitsToFloat(table.v[p_group + 0u]), uintBitsToFloat(table.v[p_group + 1u]), uintBitsToFloat(table.v[p_group + 2u]), 1.0);
	float radius = uintBitsToFloat(table.v[p_group + 3u]);
	float error = uintBitsToFloat(table.v[p_group + 4u]);
	precise float distance = 1.0;
	if ((pc.flags & FLAG_ORTHOGONAL) == 0u) {
		precise vec3 world = vec3(dot(p_row0, center), dot(p_row1, center), dot(p_row2, center));
		distance = max(pc.near_distance, length(world - pc.camera_threshold.xyz) - radius * p_scale);
	}
	precise float shown = error * p_scale;
	precise float limit = pc.camera_threshold.w * distance;
	return shown >= limit;
}

// A tree node against the camera (D91): the group test with the node's sphere and error, the
// sphere scaled by the basis's Frobenius norm (a bound on how far the transform moves any point of
// it, under shear too) and its distance lowered by a slack over every magnitude the arithmetic
// meets, so it passes whenever a child's test does.
bool node_error_visible(uint p_node, vec4 p_row0, vec4 p_row1, vec4 p_row2, float p_scale, float p_stretch) {
	vec3 center = vec3(uintBitsToFloat(table.v[p_node + 0u]), uintBitsToFloat(table.v[p_node + 1u]), uintBitsToFloat(table.v[p_node + 2u]));
	float radius = uintBitsToFloat(table.v[p_node + 3u]);
	float error = uintBitsToFloat(table.v[p_node + 4u]);
	float distance = 1.0;
	if ((pc.flags & FLAG_ORTHOGONAL) == 0u) {
		vec4 point = vec4(center, 1.0);
		vec3 world = vec3(dot(p_row0, point), dot(p_row1, point), dot(p_row2, point));
		vec3 origin = vec3(p_row0.w, p_row1.w, p_row2.w);
		float magnitude = p_stretch * (length(center) + radius) + length(origin) + length(pc.camera_threshold.xyz);
		distance = max(pc.near_distance, length(world - pc.camera_threshold.xyz) - radius * p_stretch - magnitude * (1.0 / 65536.0));
	}
	return error * p_scale >= pc.camera_threshold.w * distance;
}

// A streamed group's residency (D92): its first cluster's CLAS address, zero while it has none.
bool group_resident(uint p_table_offset, uint p_group_record) {
	uint first = table.v[p_group_record + 6u];
	return (table.v[p_table_offset + first * 2u] | table.v[p_table_offset + first * 2u + 1u]) != 0u;
}

// A visible group that is not resident (D92): requested once a frame, whatever the number of
// clusters and instances that want it, into the capped list (the count runs on past the cap).
void stream_request(uint p_state, uint p_table_offset, uint p_group) {
	if (atomicExchange(table.v[p_state + p_group * STATE_WORDS], pc.frame_index) == pc.frame_index) {
		return;
	}
	uint k = atomicAdd(frame.v[pc.stream_requests], 1u);
	if (k < STREAM_REQUEST_CAP) {
		frame.v[pc.stream_requests + 4u + k * 2u] = p_table_offset;
		frame.v[pc.stream_requests + 4u + k * 2u + 1u] = p_group;
	}
}

// A visible group that is not resident and that no cluster refines from (D92): requested by its
// own test, since no resident cluster stands in for it.
void stream_request_root(uint p_state, uint p_table_offset, uint p_group) {
	if ((table.v[p_state + p_group * STATE_WORDS + 2u] & STATE_FLAG_ROOT) != 0u) {
		stream_request(p_state, p_table_offset, p_group);
	}
}

// A taken resident group that is not persistent (D92): its use reported once in a period, which
// the CPU's least-recently-used order reads.
void stream_touch(uint p_state, uint p_table_offset, uint p_group, uint p_group_record) {
	if (uintBitsToFloat(table.v[p_group_record + 4u]) >= 3.0e38) {
		return; // Terminal: persistent, never evicted.
	}
	uint word = p_state + p_group * STATE_WORDS + 1u;
	uint last = table.v[word];
	if (pc.frame_index - last < STREAM_TOUCH_PERIOD || atomicCompSwap(table.v[word], last, pc.frame_index) != last) {
		return;
	}
	uint k = atomicAdd(frame.v[pc.stream_touches], 1u);
	if (k < STREAM_TOUCH_CAP) {
		frame.v[pc.stream_touches + 4u + k * 2u] = p_table_offset;
		frame.v[pc.stream_touches + 4u + k * 2u + 1u] = p_group;
	}
}

// Appends a selected cluster to its item's list: its CLAS address, or for a deformed item the
// destination its build gets and the build info it is made from.
void select_cluster(uint p_entry, uint p_inst, uint p_cluster, uint p_table_offset, uint p_list_offset, uint p_instance_flags) {
	uint slot = atomicAdd(frame.v[pc.blas_infos_offset + p_entry * pc.blas_info_stride], 1u);
	if ((p_instance_flags & INSTANCE_FLAG_LEVEL) != 0u && slot >= frame.v[p_inst + 18u]) {
		// Past the slot's count: taken back so the build stays in its slot, and counted.
		atomicAdd(frame.v[pc.blas_infos_offset + p_entry * pc.blas_info_stride], 0xFFFFFFFFu);
		atomicAdd(frame.v[pc.clas_count_offset + 1u], 1u);
		return;
	}
	if ((p_instance_flags & INSTANCE_FLAG_DEFORMED) == 0u) {
		frame.v[p_list_offset + slot * 2u] = table.v[p_table_offset + p_cluster * 2u];
		frame.v[p_list_offset + slot * 2u + 1u] = table.v[p_table_offset + p_cluster * 2u + 1u];
		return;
	}
	// The CLAS the build makes for this cluster: the slot's address in the instance's range (a
	// 64-bit add as lo with carry, the range being under 4 GB), for the BLAS's list and the
	// build's destination; the build info copied from the table to the build's next index.
	uint infos_offset = frame.v[p_inst + 17u];
	uint carry = 0u;
	uint dst_lo = uaddCarry(frame.v[p_inst + 18u], slot * pc.clas_slot_size, carry);
	uint dst_hi = frame.v[p_inst + 19u] + carry;
	frame.v[p_list_offset + slot * 2u] = dst_lo;
	frame.v[p_list_offset + slot * 2u + 1u] = dst_hi;
	uint k = atomicAdd(frame.v[pc.clas_count_offset], 1u);
	frame.v[pc.clas_dst_offset + k * 2u] = dst_lo;
	frame.v[pc.clas_dst_offset + k * 2u + 1u] = dst_hi;
	uint src = infos_offset + p_cluster * pc.clas_info_words;
	uint dst = pc.clas_infos_offset + k * pc.clas_info_words;
	for (uint w = 0u; w < pc.clas_info_words; w++) {
		frame.v[dst + w] = table.v[src + w];
	}
}

// The exhaustive test: one thread per cluster of one item, the items along y. Tree items are
// skipped, or with FLAG_CHECK are the only ones tested and are counted, not appended.
void exhaustive_main() {
	uint entry = pc.entry_first + gl_GlobalInvocationID.y;
	uint cluster = gl_GlobalInvocationID.x;
	if (entry >= pc.entry_count) {
		return;
	}
	uint inst = pc.instances_offset + entry * INSTANCE_WORDS;
	uint cluster_count = frame.v[inst + 13u];
	if (cluster >= cluster_count) {
		return;
	}
	uint instance_flags = frame.v[inst + 16u];
	bool check = (pc.flags & FLAG_CHECK) != 0u;
	if (((instance_flags & INSTANCE_FLAG_TREE) != 0u) != check) {
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

	bool selected;
	if ((instance_flags & INSTANCE_FLAG_LEVEL) != 0u) {
		uint level = frame.v[inst + 17u];
		uint group_record = groups + group * GROUP_WORDS;
		// A terminal group's error is FLT_MAX; the CPU's count tests the same bound.
		selected = table.v[group_record + 5u] >= level || uintBitsToFloat(table.v[group_record + 4u]) >= 3.0e38;
		if (selected && refined >= 0) {
			selected = table.v[groups + uint(refined) * GROUP_WORDS + 5u] < level;
		}
	} else {
		// A streamed item's rule is clamped by residency (D92); the CLAS a frame on the exhaustive
		// test needs are requested and reported here as the traversal does, the check only counts.
		uint state = frame.v[inst + 23u];
		uint group_record = groups + group * GROUP_WORDS;
		selected = group_error_visible(group_record, row0, row1, row2, scale);
		if (selected && state != 0u) {
			selected = group_resident(table_offset, group_record);
			if (!check) {
				if (selected) {
					stream_touch(state, table_offset, group, group_record);
				} else {
					stream_request_root(state, table_offset, group);
				}
			}
		}
		if (selected && refined >= 0) {
			uint refined_record = groups + uint(refined) * GROUP_WORDS;
			bool finer = group_error_visible(refined_record, row0, row1, row2, scale);
			if (finer && state != 0u && !group_resident(table_offset, refined_record)) {
				finer = false;
				if (!check) {
					stream_request(state, table_offset, uint(refined));
				}
			}
			selected = !finer;
		}
	}
	if (!selected) {
		return;
	}
	if (check) {
		atomicAdd(frame.v[pc.check_offset + entry], 1u);
		return;
	}
	select_cluster(entry, inst, cluster, table_offset, list_offset, instance_flags);
}

// The workgroup's appends to the two queues (D91), gathered so each queue takes one global atomic
// per workgroup rather than one per passing child: thousands of items append to the same two
// counters, and same-address atomics serialize.
shared uint shared_nodes;
shared uint shared_groups;
shared uint shared_node_base;
shared uint shared_group_base;

// One breadth-first level of the traversal (D91): two queue items per workgroup, one lane per child.
// Every invocation reaches the barriers, so nothing returns before the appends.
void traversal_main() {
	uint item = gl_WorkGroupID.x * 2u + (gl_LocalInvocationID.x >> 5u);
	uint lane = gl_LocalInvocationID.x & 31u;
	if (gl_LocalInvocationIndex == 0u) {
		shared_nodes = 0u;
		shared_groups = 0u;
	}
	barrier();

	bool append_node = false;
	bool append_group = false;
	uint entry = 0u;
	uint value = 0u; // The child node, or the passing group's record.
	if (item < frame.v[pc.count_in]) {
		entry = frame.v[pc.queue_in + item * 2u];
		uint node = frame.v[pc.queue_in + item * 2u + 1u];
		uint inst = pc.instances_offset + entry * INSTANCE_WORDS;
		uint nodes = frame.v[inst + 20u];
		uint node_record = nodes + node * NODE_WORDS;
		uint packed = table.v[node_record + 6u];
		if (lane < (packed & 0x7FFFFFFFu)) {
			vec4 row0 = vec4(uintBitsToFloat(frame.v[inst + 0u]), uintBitsToFloat(frame.v[inst + 1u]), uintBitsToFloat(frame.v[inst + 2u]), uintBitsToFloat(frame.v[inst + 3u]));
			vec4 row1 = vec4(uintBitsToFloat(frame.v[inst + 4u]), uintBitsToFloat(frame.v[inst + 5u]), uintBitsToFloat(frame.v[inst + 6u]), uintBitsToFloat(frame.v[inst + 7u]));
			vec4 row2 = vec4(uintBitsToFloat(frame.v[inst + 8u]), uintBitsToFloat(frame.v[inst + 9u]), uintBitsToFloat(frame.v[inst + 10u]), uintBitsToFloat(frame.v[inst + 11u]));
			float scale = uintBitsToFloat(frame.v[inst + 12u]);
			uint child = table.v[node_record + 5u] + lane;
			if ((packed & 0x80000000u) == 0u) {
				append_node = node_error_visible(nodes + child * NODE_WORDS, row0, row1, row2, scale, uintBitsToFloat(frame.v[inst + 22u]));
				value = child;
			} else {
				uint groups = frame.v[inst + 14u] + frame.v[inst + 13u] * 4u;
				uint group = table.v[frame.v[inst + 21u] + child];
				value = groups + group * GROUP_WORDS;
				append_group = group_error_visible(value, row0, row1, row2, scale);
				// A streamed item queues a group only once it is resident, and reports its use (D92).
				uint state = frame.v[inst + 23u];
				if (append_group && state != 0u) {
					append_group = group_resident(frame.v[inst + 14u], value);
					if (append_group) {
						stream_touch(state, frame.v[inst + 14u], group, value);
					} else {
						stream_request_root(state, frame.v[inst + 14u], group);
					}
				}
			}
		}
	}
	uint local = 0u;
	if (append_node) {
		local = atomicAdd(shared_nodes, 1u);
	} else if (append_group) {
		local = atomicAdd(shared_groups, 1u);
	}
	barrier();
	if (gl_LocalInvocationIndex == 0u) {
		if (shared_nodes > 0u) {
			shared_node_base = atomicAdd(frame.v[pc.count_out], shared_nodes);
		}
		if (shared_groups > 0u) {
			shared_group_base = atomicAdd(frame.v[pc.group_count], shared_groups);
			atomicMax(dispatch.v[0], (shared_group_base + shared_groups + 1u) / 2u);
		}
	}
	barrier();
	if (append_node) {
		uint k = shared_node_base + local;
		frame.v[pc.queue_out + k * 2u] = entry;
		frame.v[pc.queue_out + k * 2u + 1u] = value;
	} else if (append_group) {
		uint k = shared_group_base + local;
		frame.v[pc.group_queue + k * 2u] = entry;
		frame.v[pc.group_queue + k * 2u + 1u] = value;
	}
}

// The traversal's clusters (D91): two queued groups per workgroup, a lane per cluster of the group's
// contiguous run (checked at population), each tested against the group it refines.
void cluster_main() {
	uint item = gl_WorkGroupID.x * 2u + (gl_LocalInvocationID.x >> 5u);
	uint lane = gl_LocalInvocationID.x & 31u;
	if (item >= frame.v[pc.group_count]) {
		return;
	}
	uint entry = frame.v[pc.group_queue + item * 2u];
	uint group_record = frame.v[pc.group_queue + item * 2u + 1u];
	uint first = table.v[group_record + 6u];
	uint count = table.v[group_record + 7u];
	if (lane >= count) {
		return;
	}
	uint inst = pc.instances_offset + entry * INSTANCE_WORDS;
	vec4 row0 = vec4(uintBitsToFloat(frame.v[inst + 0u]), uintBitsToFloat(frame.v[inst + 1u]), uintBitsToFloat(frame.v[inst + 2u]), uintBitsToFloat(frame.v[inst + 3u]));
	vec4 row1 = vec4(uintBitsToFloat(frame.v[inst + 4u]), uintBitsToFloat(frame.v[inst + 5u]), uintBitsToFloat(frame.v[inst + 6u]), uintBitsToFloat(frame.v[inst + 7u]));
	vec4 row2 = vec4(uintBitsToFloat(frame.v[inst + 8u]), uintBitsToFloat(frame.v[inst + 9u]), uintBitsToFloat(frame.v[inst + 10u]), uintBitsToFloat(frame.v[inst + 11u]));
	float scale = uintBitsToFloat(frame.v[inst + 12u]);
	uint cluster_count = frame.v[inst + 13u];
	uint table_offset = frame.v[inst + 14u];
	uint list_offset = frame.v[inst + 15u];
	uint instance_flags = frame.v[inst + 16u];
	uint state = frame.v[inst + 23u];
	uint groups = table_offset + cluster_count * 4u;
	// A group of more than 32 clusters takes a lane's further clusters in strides of 32.
	for (uint i = lane; i < count; i += 32u) {
		uint cluster = first + i;
		int refined = int(table.v[table_offset + cluster_count * 2u + cluster * 2u + 1u]);
		if (refined >= 0) {
			uint refined_record = groups + uint(refined) * GROUP_WORDS;
			if (group_error_visible(refined_record, row0, row1, row2, scale)) {
				// The finer group stands in, if resident (D92); else this cluster does and it is
				// requested: the frontier of the resident cut, its parents all resident.
				if (state == 0u || group_resident(table_offset, refined_record)) {
					continue;
				}
				stream_request(state, table_offset, uint(refined));
			}
		}
		select_cluster(entry, inst, cluster, table_offset, list_offset, instance_flags);
	}
}

void main() {
	if ((pc.flags & FLAG_CLUSTERS) != 0u) {
		cluster_main();
	} else if ((pc.flags & FLAG_NODES) != 0u) {
		traversal_main();
	} else {
		exhaustive_main();
	}
}
