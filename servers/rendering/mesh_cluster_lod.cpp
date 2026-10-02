/**************************************************************************/
/*  mesh_cluster_lod.cpp                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "mesh_cluster_lod.h"

#include "core/math/math_funcs.h"
#include "servers/rendering/rendering_server_types.h"

bool mesh_cluster_lod_streamable(const Vector<uint8_t> &p_cluster_data, uint32_t p_index_count, uint32_t p_index_total, uint32_t p_max_triangles, uint32_t p_max_vertices) {
	const uint32_t HEADER_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS;
	const uint32_t CLUSTER_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
	const uint32_t GROUP_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS;
	const uint64_t words = p_cluster_data.size() / sizeof(uint32_t);
	if (p_cluster_data.size() % sizeof(uint32_t) != 0 || words < HEADER_WORDS) {
		return false;
	}
	const uint32_t *table = reinterpret_cast<const uint32_t *>(p_cluster_data.ptr());
	if (table[0] != RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION) {
		return false;
	}
	const uint32_t cluster_count = table[1];
	const uint32_t group_count = table[2];
	const uint32_t level_count = table[3];
	const uint32_t lod0_clusters = table[4];
	const uint32_t lod0_triangles = table[5];
	const uint32_t coarse_triangles = table[6];
	const uint64_t expected_words = HEADER_WORDS + uint64_t(CLUSTER_WORDS) * cluster_count + uint64_t(GROUP_WORDS) * group_count + uint64_t(coarse_triangles) * 3;

	// The layout every level is traced in: the path tracer's _cluster_layout_for_surface with the
	// coarse levels taken.
	if (words != expected_words || lod0_clusters == 0 || lod0_clusters > cluster_count || lod0_triangles != p_index_count / 3 || (table[7] & RenderingServerTypes::MESH_CLUSTER_LOD_FLAG_STALE) || coarse_triangles == 0 || uint64_t(p_index_total) < (uint64_t(lod0_triangles) + coarse_triangles) * 3) {
		return false;
	}
	const uint32_t *cluster_words = table + HEADER_WORDS;
	const uint32_t *group_words = cluster_words + uint64_t(CLUSTER_WORDS) * cluster_count;
	uint64_t lod0_sum = 0;
	uint64_t coarse_sum = 0;
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t packed = cluster_words[uint64_t(c) * CLUSTER_WORDS];
		const uint32_t triangles = packed & 0xFFFFu;
		const uint32_t vertices = packed >> 16;
		if (triangles == 0 || triangles > p_max_triangles || vertices == 0 || vertices > p_max_vertices) {
			return false;
		}
		(c < lod0_clusters ? lod0_sum : coarse_sum) += triangles;
	}
	if (lod0_sum != lod0_triangles || coarse_sum != coarse_triangles) {
		return false;
	}

	// The residency: the path tracer's _create_lod_stream (D92).
	if (group_count == 0 || level_count < 2 || level_count > group_count) {
		return false;
	}
	uint64_t covered = 0;
	uint64_t persistent_clusters = 0;
	for (uint32_t g = 0; g < group_count; g++) {
		const uint32_t *w = group_words + uint64_t(g) * GROUP_WORDS;
		const uint32_t depth = w[0];
		const uint32_t first_cluster = w[1];
		const uint32_t count = w[2];
		if (depth >= level_count || count == 0 || uint64_t(first_cluster) + count > cluster_count) {
			return false;
		}
		float floats[5];
		memcpy(floats, w + 3, sizeof(floats));
		for (const float f : floats) {
			if (Math::is_nan(f)) {
				return false;
			}
		}
		for (uint32_t c = first_cluster; c < first_cluster + count; c++) {
			if (cluster_words[uint64_t(c) * CLUSTER_WORDS + 1] != g) {
				return false;
			}
		}
		covered += count;
		if (floats[4] >= 3.0e38f) {
			persistent_clusters += count;
		}
	}
	if (covered != cluster_count) {
		return false;
	}
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t *w = cluster_words + uint64_t(c) * CLUSTER_WORDS;
		const int32_t refined = int32_t(w[2]);
		const uint32_t depth = group_words[uint64_t(w[1]) * GROUP_WORDS];
		if (refined >= int32_t(group_count) || (refined < 0 ? depth != 0 : depth == 0 || group_words[uint64_t(refined) * GROUP_WORDS] != depth - 1)) {
			return false;
		}
	}
	return persistent_clusters > 0 && persistent_clusters < cluster_count;
}

bool mesh_cluster_lod_local_vertices(const Vector<uint8_t> &p_cluster_data, const uint8_t *p_index_data, uint64_t p_index_bytes, bool p_index_16, uint32_t p_vertex_count, LocalVector<uint16_t> &r_scratch, LocalVector<uint32_t> &r_distinct) {
	const uint32_t CLUSTER_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
	const uint32_t *table = reinterpret_cast<const uint32_t *>(p_cluster_data.ptr());
	const uint32_t cluster_count = table[1];
	const uint32_t *cluster_words = table + RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS;
	const uint64_t index_size = p_index_16 ? sizeof(uint16_t) : sizeof(uint32_t);
	uint32_t largest = 0;
	uint64_t index_count = 0;
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t packed = cluster_words[uint64_t(c) * CLUSTER_WORDS];
		largest = MAX(largest, packed >> 16);
		index_count += uint64_t(packed & 0xFFFFu) * 3;
	}
	const uint32_t cap = MIN(MESH_CLUSTER_GEOMETRY_MAX_VERTICES, largest);
	if (p_index_data == nullptr || p_index_bytes < index_count * index_size) {
		return false;
	}
	if (r_scratch.size() < p_vertex_count) {
		const uint32_t from = r_scratch.size();
		r_scratch.resize(p_vertex_count);
		memset(r_scratch.ptr() + from, 0xFF, sizeof(uint16_t) * (p_vertex_count - from));
	}
	uint16_t *local = r_scratch.ptr();
	r_distinct.resize(cluster_count);
	LocalVector<uint32_t> seen;
	uint64_t first = 0;
	bool valid = true;
	for (uint32_t c = 0; c < cluster_count && valid; c++) {
		const uint64_t end = first + uint64_t(cluster_words[uint64_t(c) * CLUSTER_WORDS] & 0xFFFFu) * 3;
		for (uint64_t i = first; i < end && valid; i++) {
			const uint32_t index = p_index_16 ? reinterpret_cast<const uint16_t *>(p_index_data)[i] : reinterpret_cast<const uint32_t *>(p_index_data)[i];
			valid = index < p_vertex_count;
			if (valid && local[index] == 0xFFFF) {
				local[index] = uint16_t(seen.size());
				seen.push_back(index);
			}
		}
		first = end;
		r_distinct[c] = seen.size();
		for (const uint32_t vertex : seen) {
			local[vertex] = 0xFFFF;
		}
		seen.clear();
		valid = valid && r_distinct[c] <= cap;
	}
	return valid;
}

void mesh_cluster_lod_fallback_level(const Vector<uint8_t> &p_cluster_data, float p_ratio, LocalVector<uint32_t> &r_runs, uint32_t &r_depth, uint64_t &r_triangles) {
	const uint32_t CLUSTER_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
	const uint32_t GROUP_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS;
	const uint32_t *table = reinterpret_cast<const uint32_t *>(p_cluster_data.ptr());
	const uint32_t cluster_count = table[1];
	const uint32_t level_count = table[3];
	const uint32_t *cluster_words = table + RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS;
	const uint32_t *group_words = cluster_words + uint64_t(CLUSTER_WORDS) * cluster_count;

	// Each cluster's depths, and each depth's triangles.
	LocalVector<uint64_t> level_triangles;
	level_triangles.resize(level_count);
	for (uint64_t &triangles : level_triangles) {
		triangles = 0;
	}
	LocalVector<uint32_t> first_depth;
	LocalVector<uint32_t> last_depth;
	first_depth.resize(cluster_count);
	last_depth.resize(cluster_count);
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t *w = cluster_words + uint64_t(c) * CLUSTER_WORDS;
		const uint32_t *group = group_words + uint64_t(w[1]) * GROUP_WORDS;
		const int32_t refined = int32_t(w[2]);
		float error;
		memcpy(&error, group + 7, sizeof(float));
		first_depth[c] = refined < 0 ? 0 : group_words[uint64_t(refined) * GROUP_WORDS] + 1;
		last_depth[c] = error >= 3.0e38f ? level_count - 1 : MIN(group[0], level_count - 1);
		for (uint32_t l = first_depth[c]; l <= last_depth[c]; l++) {
			level_triangles[l] += w[0] & 0xFFFFu;
		}
	}
	const uint64_t wanted = uint64_t(Math::ceil(double(level_triangles[0]) * double(CLAMP(p_ratio, 0.0f, 1.0f))));
	r_depth = 0;
	for (uint32_t l = level_count; l-- > 0;) {
		if (level_triangles[l] >= MAX(wanted, uint64_t(1))) {
			r_depth = l;
			break;
		}
	}
	r_triangles = level_triangles[r_depth];

	r_runs.clear();
	uint32_t first_triangle = 0;
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t triangles = cluster_words[uint64_t(c) * CLUSTER_WORDS] & 0xFFFFu;
		if (first_depth[c] <= r_depth && r_depth <= last_depth[c]) {
			r_runs.push_back(first_triangle);
			r_runs.push_back(triangles);
		}
		first_triangle += triangles;
	}
}
