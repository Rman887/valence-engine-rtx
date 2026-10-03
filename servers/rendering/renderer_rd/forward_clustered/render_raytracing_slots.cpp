/**************************************************************************/
/*  render_raytracing_slots.cpp                                           */
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

// Slot-resident instances (D94b): the walk in build_tlas names each instance's persistent slot
// and pushes only the instances it cannot prove unchanged; the commit here turns those into the
// PTLAS's instance writes and the rows' scatter, disables the slots whose owner the walk did not
// reach, and keeps what the emissive gather reads. See docs/r1-partitioned-tlas.md.

#include "core/os/os.h"
#include "servers/rendering/renderer_rd/forward_clustered/render_forward_clustered.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererSceneRenderImplementation;

// ---------------------------------------------------------------------------
// Slot indices: lowest first fit over the freed intervals, then the high water, so a scene put in
// the trace in one frame gets its slots in walk order.
// ---------------------------------------------------------------------------

uint32_t RenderRaytracing::_slot_interval_alloc(uint32_t p_count) {
	for (uint32_t i = 0; i < slot_free_intervals.size(); i++) {
		SlotInterval &interval = slot_free_intervals[i];
		if (interval.count >= p_count) {
			const uint32_t first = interval.first;
			interval.first += p_count;
			interval.count -= p_count;
			if (interval.count == 0) {
				slot_free_intervals.remove_at(i);
			}
			return first;
		}
	}
	const uint32_t first = slot_high_water;
	slot_high_water += p_count;
	return first;
}

void RenderRaytracing::_slot_interval_free(uint32_t p_first, uint32_t p_count) {
	// The position of the first interval past it, by binary search.
	uint32_t lo = 0;
	uint32_t hi = slot_free_intervals.size();
	while (lo < hi) {
		const uint32_t mid = (lo + hi) / 2;
		if (slot_free_intervals[mid].first < p_first) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	uint32_t first = p_first;
	uint32_t count = p_count;
	if (lo > 0 && slot_free_intervals[lo - 1].first + slot_free_intervals[lo - 1].count == first) {
		lo--;
		first = slot_free_intervals[lo].first;
		count += slot_free_intervals[lo].count;
		slot_free_intervals.remove_at(lo);
	}
	if (lo < slot_free_intervals.size() && first + count == slot_free_intervals[lo].first) {
		count += slot_free_intervals[lo].count;
		slot_free_intervals.remove_at(lo);
	}
	if (first + count == slot_high_water) {
		slot_high_water = first; // The top of the slots returns to the high water.
		return;
	}
	slot_free_intervals.insert(lo, SlotInterval{ first, count });
}

// ---------------------------------------------------------------------------
// Ranges
// ---------------------------------------------------------------------------

uint32_t RenderRaytracing::_slot_range_acquire(RTViewportState *p_state, RTSlotOwner &r_owner, uint32_t p_key, uint32_t p_count) {
	uint32_t range_id = UINT32_MAX;
	for (uint32_t i = 0; i < r_owner.ranges.size(); i++) {
		if (uint32_t(r_owner.ranges[i] >> 32) == p_key) {
			range_id = uint32_t(r_owner.ranges[i] & 0xFFFFFFFFu);
			if (slot_ranges[range_id].count != p_count) {
				// A MultiMesh merged or expanded, or its visible count changed: new slots, the old
				// ones disabled by every viewport's sweep before they return.
				_slot_range_release(range_id);
				r_owner.ranges.remove_at_unordered(i);
				range_id = UINT32_MAX;
			}
			break;
		}
	}
	if (range_id == UINT32_MAX) {
		if (!slot_range_free_ids.is_empty()) {
			range_id = slot_range_free_ids[slot_range_free_ids.size() - 1];
			slot_range_free_ids.resize(slot_range_free_ids.size() - 1);
		} else {
			range_id = slot_ranges.size();
			slot_ranges.push_back(RTSlotRange());
		}
		RTSlotRange &range = slot_ranges[range_id];
		range.first = _slot_interval_alloc(p_count);
		range.count = p_count;
		range.serial++;
		range.live = true;
		range.released = false;
		range.enabled_viewports = 0;
		slot_live += p_count;
		r_owner.ranges.push_back((uint64_t(p_key) << 32) | range_id);
	}

	RTSlotRange &range = slot_ranges[range_id];
	if (p_state->slot_range_views.size() <= range_id) {
		p_state->slot_range_views.resize(slot_ranges.size());
	}
	RTSlotRangeView &view = p_state->slot_range_views[range_id];
	if (view.serial != range.serial) {
		DEV_ASSERT(!view.enabled);
		view = RTSlotRangeView();
		view.serial = range.serial;
	}
	view.present_frame = p_state->slot_build;
	if (!view.enabled) {
		view.enabled = true;
		range.enabled_viewports++;
	}
	if (p_state->slots.size() < range.first + range.count) {
		p_state->slots.resize(MAX(slot_high_water, range.first + range.count));
	}
	return range_id;
}

void RenderRaytracing::_slot_range_release(uint32_t p_range) {
	RTSlotRange &range = slot_ranges[p_range];
	ERR_FAIL_COND(!range.live || range.released);
	range.released = true;
	slot_live -= range.count;
	_slot_range_try_free(p_range);
}

void RenderRaytracing::_slot_range_try_free(uint32_t p_range) {
	RTSlotRange &range = slot_ranges[p_range];
	if (!range.live || !range.released || range.enabled_viewports > 0) {
		return;
	}
	_slot_interval_free(range.first, range.count);
	range.live = false;
	range.released = false;
	slot_local_rows.erase(p_range);
	slot_range_free_ids.push_back(p_range);
}

void RenderRaytracing::release_slot_ranges(RTSlotOwner &r_owner) {
	for (uint64_t entry : r_owner.ranges) {
		_slot_range_release(uint32_t(entry & 0xFFFFFFFFu));
	}
	r_owner.ranges.clear();
}

void RenderRaytracing::release_slot_ranges_from(RTSlotOwner &r_owner, uint32_t p_surface_count) {
	for (uint32_t i = 0; i < r_owner.ranges.size();) {
		const uint32_t key = uint32_t(r_owner.ranges[i] >> 32);
		if (key >= p_surface_count && key != RT_SLOT_KEY_PROCEDURAL) {
			_slot_range_release(uint32_t(r_owner.ranges[i] & 0xFFFFFFFFu));
			r_owner.ranges.remove_at_unordered(i);
		} else {
			i++;
		}
	}
}

void RenderRaytracing::_release_viewport_slots(RTViewportState *p_state) {
	for (uint32_t r = 0; r < p_state->slot_range_views.size() && r < slot_ranges.size(); r++) {
		RTSlotRangeView &view = p_state->slot_range_views[r];
		RTSlotRange &range = slot_ranges[r];
		if (view.enabled && range.live && view.serial == range.serial) {
			view.enabled = false;
			range.enabled_viewports--;
			_slot_range_try_free(r);
		}
	}
	p_state->slot_range_views.clear();
	if (p_state->slot_row_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->slot_row_buffer);
		p_state->slot_row_buffer = RID();
	}
	p_state->slot_row_capacity = 0;
}

// A viewport that stops building (a SubViewport no longer updated) would hold its claims, and with
// them every slot index it ever enabled, for as long as it lives. Past RT_SLOT_IDLE_FRAMES frames
// without a commit, longer than any frame in flight could still read its structure, another
// viewport's commit takes them back and frees its PTLAS and rows (the frees are deferred by the
// device); its next build starts from scratch.
void RenderRaytracing::_reset_viewport_slots(RTViewportState *p_state) {
	RD *rd = RD::get_singleton();
	_release_viewport_slots(p_state);
	RID *resources[4] = { &p_state->tlas, &p_state->geometry_buffer, &p_state->material_buffer, &p_state->motion_index_buffer };
	for (RID *rid : resources) {
		if (rid->is_valid()) {
			rd->free_rid(*rid);
			*rid = RID();
		}
	}
	p_state->tlas_max_instances = 0;
	p_state->geometry_buffer_capacity = 0;
	p_state->material_buffer_capacity = 0;
	p_state->motion_index_buffer_capacity = 0;
	p_state->slot_capacity = 0;
	p_state->slots.reset();
	p_state->slot_mm_keys.clear();
	p_state->emissive_slots.clear();
	p_state->motion_slots.reset();
	p_state->motion_slots_next.reset();
	p_state->slot_cells.clear();
	p_state->partition_occupancy.reset();
	p_state->partition_cells.reset();
	p_state->partition_translations.reset();
	p_state->partition_touched.reset();
	p_state->partition_free.reset();
	p_state->partition_high_water = 0;
	p_state->partition_capacity = 0;
	p_state->partitions_live = 0;
	p_state->global_slots.reset();
	p_state->global_return_buckets.reset();
	p_state->global_capacity = 0;
	p_state->slot_rebuild = true;
	p_state->slot_origin_valid = false;
	p_state->slot_commit_frame = 0;
}

void RenderRaytracing::_reclaim_idle_viewports(RTViewportState *p_current) {
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	for (KeyValue<RenderSceneBuffersRD *, RTViewportState *> &E : viewport_states) {
		RTViewportState *state = E.value;
		if (state == p_current || state->slot_commit_frame == 0 || frames_drawn - state->slot_commit_frame <= RT_SLOT_IDLE_FRAMES) {
			continue;
		}
		print_verbose(vformat("RT: a viewport idle for %d frames gives back its slots (D94b): %d range views, %d slots shadowed.", frames_drawn - state->slot_commit_frame, state->slot_range_views.size(), state->slots.size()));
		_reset_viewport_slots(state);
		slot_counters_reclaimed++;
	}
}

// ---------------------------------------------------------------------------
// Row generations: a source's row compared with what it held when its generation was last
// taken, so every writer of a row is covered without each having to bump it.
// ---------------------------------------------------------------------------

uint64_t RenderRaytracing::_surface_row_generation(RTSurfaceData *p_surf_data) {
	if (p_surf_data->row_generation == 0 || memcmp(&p_surf_data->row_snapshot, &p_surf_data->geometry, sizeof(RT_GeometryData)) != 0) {
		p_surf_data->row_snapshot = p_surf_data->geometry;
		p_surf_data->row_generation = ++row_generation_counter;
	}
	return p_surf_data->row_generation;
}

uint64_t RenderRaytracing::_material_row_generation(RTMaterialData *p_mat_data) {
	if (p_mat_data->row_generation == 0 || memcmp(&p_mat_data->row_snapshot, &p_mat_data->data, sizeof(RT_MaterialData)) != 0) {
		p_mat_data->row_snapshot = p_mat_data->data;
		p_mat_data->row_generation = ++row_generation_counter;
	}
	return p_mat_data->row_generation;
}

uint64_t RenderRaytracing::_local_row_generation(uint32_t p_range, const RT_GeometryData &p_row) {
	// Only merged MultiMesh and procedural ranges have one, so it is kept beside the ranges.
	RTSlotLocalRow &local = slot_local_rows[p_range];
	if (local.generation == 0 || memcmp(&local.row, &p_row, sizeof(RT_GeometryData)) != 0) {
		local.row = p_row;
		local.generation = ++row_generation_counter;
	}
	return local.generation;
}

bool RenderRaytracing::_slot_multimesh_key_matches(RTViewportState *p_state, uint32_t p_range, const RTSlotMultiMeshKey &p_key) {
	// The key of an expanded MultiMesh range (decision 6), kept beside the views: unchanged, the
	// walk skips the range whole; changed, it is stored for the next build.
	RTSlotMultiMeshKey &key = p_state->slot_mm_keys[p_range];
	const uint32_t serial = slot_ranges[p_range].serial;
	if (key.serial == serial && key.count == p_key.count && key.buffer_generation == p_key.buffer_generation && key.blas == p_key.blas &&
			key.geometry_generation == p_key.geometry_generation && key.material_generation == p_key.material_generation && key.sbt_offset == p_key.sbt_offset &&
			key.flags == p_key.flags && key.mask == p_key.mask && key.instance_transform == p_key.instance_transform) {
		return true;
	}
	key = p_key;
	key.serial = serial;
	return false;
}

void RenderRaytracing::_slot_relative_transform(const Transform3D &p_transform, const Vector3 &p_origin, float *r_rows) {
	// The narrowing the instance record makes, from the transform relative to an origin (the slot's
	// cell's, D94c) in real_t.
	const Vector3 origin = p_transform.origin - p_origin;
	r_rows[0] = float(p_transform.basis.rows[0][0]);
	r_rows[1] = float(p_transform.basis.rows[0][1]);
	r_rows[2] = float(p_transform.basis.rows[0][2]);
	r_rows[3] = float(origin.x);
	r_rows[4] = float(p_transform.basis.rows[1][0]);
	r_rows[5] = float(p_transform.basis.rows[1][1]);
	r_rows[6] = float(p_transform.basis.rows[1][2]);
	r_rows[7] = float(origin.y);
	r_rows[8] = float(p_transform.basis.rows[2][0]);
	r_rows[9] = float(p_transform.basis.rows[2][1]);
	r_rows[10] = float(p_transform.basis.rows[2][2]);
	r_rows[11] = float(origin.z);
}

static _FORCE_INLINE_ Transform3D _slot_rows_to_transform(const float *p_rows) {
	Transform3D t;
	t.basis.rows[0][0] = p_rows[0];
	t.basis.rows[0][1] = p_rows[1];
	t.basis.rows[0][2] = p_rows[2];
	t.origin.x = p_rows[3];
	t.basis.rows[1][0] = p_rows[4];
	t.basis.rows[1][1] = p_rows[5];
	t.basis.rows[1][2] = p_rows[6];
	t.origin.y = p_rows[7];
	t.basis.rows[2][0] = p_rows[8];
	t.basis.rows[2][1] = p_rows[9];
	t.basis.rows[2][2] = p_rows[10];
	t.origin.z = p_rows[11];
	return t;
}

Vector3 RenderRaytracing::_slot_cell_translation(const RTViewportState *p_state, const Vector3i &p_cell) const {
	// A partition's translation (D94c): its cell's origin less rt_origin, in double, rounded to the
	// float the build takes. The records, the emissive transforms and the structure all use this float.
	const Vector3 translation = _slot_cell_origin(p_cell) - p_state->rt_origin;
	return Vector3(float(translation.x), float(translation.y), float(translation.z));
}

Transform3D RenderRaytracing::_slot_traced_transform(const RTViewportState *p_state, const RTSlotShadow &p_shadow) const {
	// Where the trace finds the slot, relative to rt_origin: its cell-relative rows with its cell's
	// translation added in float, as the build adds a partition's translation to an instance's.
	const Vector3 translation = _slot_cell_translation(p_state, p_shadow.cell);
	Transform3D t = _slot_rows_to_transform(p_shadow.transform);
	t.origin.x = p_shadow.transform[3] + float(translation.x);
	t.origin.y = p_shadow.transform[7] + float(translation.y);
	t.origin.z = p_shadow.transform[11] + float(translation.z);
	return t;
}

Transform3D RenderRaytracing::_slot_record_transform(const RTViewportState *p_state, const RTSlotShadow &p_shadow) const {
	// The transform the slot's record carries: a cell partition's translation is the structure's, but
	// the global partition has none, so a global record carries its cell's translation itself and is
	// rewritten when rt_origin moves.
	if (p_shadow.partition == RD::PARTITION_GLOBAL) {
		return _slot_traced_transform(p_state, p_shadow);
	}
	return _slot_rows_to_transform(p_shadow.transform);
}

void RenderRaytracing::_slot_emit(uint32_t p_slot, uint64_t p_geometry_generation, uint64_t p_material_generation) {
	frame_slots.push_back(p_slot);
	frame_geometry_generations.push_back(p_geometry_generation);
	frame_material_generations.push_back(p_material_generation);
}

// ---------------------------------------------------------------------------
// Capacity: the PTLAS and the rows hold slot_capacity slots; a larger one is a new PTLAS built
// from scratch from the shadows, and new row buffers the old rows are copied into.
// ---------------------------------------------------------------------------

bool RenderRaytracing::_ensure_slot_capacity(RTViewportState *p_state) {
	RD *rd = RD::get_singleton();
	const uint32_t needed = MAX(slot_high_water, 1u);
	const uint32_t partitions_needed = MAX(p_state->partition_high_water, 1u);
	if (p_state->tlas.is_valid() && p_state->slot_capacity >= needed && p_state->partition_capacity >= partitions_needed) {
		return true;
	}
	uint32_t capacity = MAX(p_state->slot_capacity, RT_SLOT_MIN_CAPACITY);
	while (capacity < needed) {
		capacity = capacity + capacity / 2;
	}
	capacity = (capacity + 1023) & ~1023u;
	ERR_FAIL_COND_V_MSG(capacity > (1u << 24), false, "RT: the partitioned TLAS would need more slots than a custom index can name.");
	uint32_t partition_capacity = MAX(p_state->partition_capacity, 16u);
	while (partition_capacity < partitions_needed) {
		partition_capacity *= 2;
	}

	// A partition may hold every slot: the bucketing keeps ptlas_partition_max_instances enabled
	// slots in each, but the specification does not say whether a disabled instance's record
	// counts against maxInstancePerPartitionCount, and a disabled record stays in the partition
	// it was written to (D94b, review finding 9).
	RD::PartitionedAccelerationStructureLimits limits;
	limits.instance_count = capacity;
	limits.partition_count = partition_capacity;
	limits.max_instances_per_partition = capacity;
	limits.max_global_instances = MIN(ptlas_global_max_instances, capacity);
	limits.partition_translation = true; // Cells relative to their origin (D94c).

	// Everything is made before the state changes: a failure frees what was made and leaves the
	// viewport's structure and rows as they were.
	struct RowBuffer {
		RID *buffer;
		uint32_t stride;
		const char *name;
		RID created;
	};
	RowBuffer row_buffers[3] = {
		{ &p_state->geometry_buffer, uint32_t(sizeof(RT_GeometryData)), "RT Slot Geometry", RID() },
		{ &p_state->material_buffer, uint32_t(sizeof(RT_MaterialData)), "RT Slot Materials", RID() },
		{ &p_state->motion_index_buffer, uint32_t(sizeof(int32_t)), "RT Slot Motion Indices", RID() },
	};
	RID ptlas = rd->ptlas_create(limits);
	bool made = ptlas.is_valid();
	for (RowBuffer &row : row_buffers) {
		if (made) {
			row.created = rd->storage_buffer_create(capacity * row.stride);
			made = row.created.is_valid();
		}
	}
	if (!made) {
		if (ptlas.is_valid()) {
			rd->free_rid(ptlas);
		}
		for (RowBuffer &row : row_buffers) {
			if (row.created.is_valid()) {
				rd->free_rid(row.created);
			}
		}
		ERR_FAIL_V_MSG(false, vformat("RT: could not make a partitioned TLAS of %d slots and %d partitions with its rows.", capacity, partition_capacity));
	}

	rd->set_resource_name(ptlas, "RT PTLAS");
	for (RowBuffer &row : row_buffers) {
		rd->set_resource_name(row.created, row.name);
		if (row.buffer->is_valid()) {
			if (p_state->slot_capacity > 0) {
				rd->buffer_copy(*row.buffer, row.created, 0, 0, p_state->slot_capacity * row.stride);
			}
			rd->free_rid(*row.buffer);
		}
		*row.buffer = row.created;
	}
	p_state->geometry_buffer_capacity = capacity * sizeof(RT_GeometryData);
	p_state->material_buffer_capacity = capacity * sizeof(RT_MaterialData);
	p_state->motion_index_buffer_capacity = capacity * sizeof(int32_t);

	if (p_state->tlas.is_valid()) {
		rd->free_rid(p_state->tlas);
	}
	p_state->tlas = ptlas;
	p_state->tlas_max_instances = capacity;
	p_state->slot_capacity = capacity;
	p_state->partition_capacity = partition_capacity;
	p_state->global_capacity = limits.max_global_instances;
	p_state->slot_rebuild = true;
	print_verbose(vformat("RT: partitioned TLAS of %d slots and %d partitions (D94b), %d live, %d partitions in use.", capacity, partition_capacity, slot_live, p_state->partitions_live));
	return true;
}

// ---------------------------------------------------------------------------
// The scatter: the changed rows, assembled from the frame's arrays in chunks of RT_SLOT_CHUNK so
// neither the host scratch nor the upload buffer is sized by the largest frame, each chunk
// written into the slot-indexed buffers by one dispatch.
// ---------------------------------------------------------------------------

bool RenderRaytracing::_scatter_slot_rows(RTViewportState *p_state) {
	const uint32_t total = slot_row_refs.size();
	if (total == 0) {
		return true;
	}
	RD *rd = RD::get_singleton();
	const uint32_t chunk_capacity = MIN(total, RT_SLOT_CHUNK);
	if (chunk_capacity > p_state->slot_row_capacity) {
		if (p_state->slot_row_buffer.is_valid()) {
			rd->free_rid(p_state->slot_row_buffer);
			p_state->slot_row_buffer = RID();
			p_state->slot_row_capacity = 0;
		}
		const uint32_t capacity = MIN(MAX(Math::next_power_of_2(chunk_capacity), 256u), RT_SLOT_CHUNK);
		RID buffer = rd->storage_buffer_create(capacity * sizeof(RT_SlotRowWrite));
		ERR_FAIL_COND_V_MSG(buffer.is_null(), false, "RT: could not make the slot rows' upload buffer.");
		rd->set_resource_name(buffer, "RT Slot Row Writes");
		p_state->slot_row_buffer = buffer;
		p_state->slot_row_capacity = capacity;
	}
	RD::Uniform u_rows(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, p_state->slot_row_buffer);
	RD::Uniform u_geometry(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, p_state->geometry_buffer);
	RD::Uniform u_materials(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, p_state->material_buffer);
	RD::Uniform u_motion(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, p_state->motion_index_buffer);
	RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(slot_rows_shader.version_shader, 0, u_rows, u_geometry, u_materials, u_motion);

	for (uint32_t base = 0; base < total; base += RT_SLOT_CHUNK) {
		const uint32_t count = MIN(total - base, RT_SLOT_CHUNK);
		slot_row_writes.resize(count);
		for (uint32_t n = 0; n < count; n++) {
			const SlotRowRef &ref = slot_row_refs[base + n];
			RT_SlotRowWrite &row = slot_row_writes[n];
			row.slot = ref.slot;
			row.parts = ref.source != UINT32_MAX ? 1 : 0;
			row.motion_index = ref.motion_index;
			row._pad = 0;
			if (ref.source != UINT32_MAX) {
				row.geometry = geometry_data[ref.source];
				row.material = material_data[ref.source];
			} else {
				memset(&row.geometry, 0, sizeof(RT_GeometryData) + sizeof(RT_MaterialData));
			}
		}
		rd->buffer_update(p_state->slot_row_buffer, 0, count * sizeof(RT_SlotRowWrite), slot_row_writes.ptr());
		struct {
			uint32_t count;
			uint32_t pad[3];
		} push_constant = { count, { 0, 0, 0 } };
		RD::ComputeListID compute_list = rd->compute_list_begin();
		rd->compute_list_bind_compute_pipeline(compute_list, slot_rows_shader.pipeline);
		rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
		rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		rd->compute_list_dispatch(compute_list, (count + 63) / 64, 1, 1);
		rd->compute_list_end();
	}
	return true;
}

// ---------------------------------------------------------------------------
// Partitions (decision 4): a slot is placed in a bucket of the world cell (a column on x and z) its
// box's centre is in or, rewritten while enabled, in the global partition while it has room, so
// that what changes every frame rebuilds no cell; it returns to its cell after ptlas_stable_frames
// unwritten builds.
// ---------------------------------------------------------------------------

Vector3i RenderRaytracing::_slot_cell(const Transform3D &p_transform, const RT_GeometryData &p_geometry) const {
	// A compressed surface's positions are a unit cube under its transform.
	const Vector3 local = (p_geometry.flags & RT_GEOM_FLAG_COMPRESSED) ? Vector3(0.5, 0.5, 0.5) : Vector3(p_geometry.aabb_pos_x + 0.5f * p_geometry.aabb_size_x, p_geometry.aabb_pos_y + 0.5f * p_geometry.aabb_size_y, p_geometry.aabb_pos_z + 0.5f * p_geometry.aabb_size_z);
	const Vector3 centre = p_transform.xform(local);
	const double cell_size = ptlas_cell_size;
	auto axis = [cell_size](real_t p_value) {
		return int32_t(CLAMP(Math::floor(double(p_value) / cell_size), -2147483648.0, 2147483647.0));
	};
	// Columns on x and z, as the world's grid and rt_origin's: ground-level content would otherwise
	// split at every multiple of the cell's height.
	return Vector3i(axis(centre.x), 0, axis(centre.z));
}

uint32_t RenderRaytracing::_slot_cell_partition(RTViewportState *p_state, const Vector3i &p_cell) {
	RTViewportState::SlotCell &cell = p_state->slot_cells[p_cell];
	for (uint32_t partition : cell.partitions) {
		if (p_state->partition_occupancy[partition] < ptlas_partition_max_instances) {
			return partition;
		}
	}
	uint32_t partition;
	if (!p_state->partition_free.is_empty()) {
		partition = p_state->partition_free[p_state->partition_free.size() - 1];
		p_state->partition_free.resize(p_state->partition_free.size() - 1);
	} else {
		partition = p_state->partition_high_water++;
		p_state->partition_occupancy.push_back(0);
		p_state->partition_cells.push_back(Vector3i());
		p_state->partition_translations.push_back(Vector3(Math::NaN, Math::NaN, Math::NaN));
		p_state->partition_touched.push_back(0);
	}
	p_state->partition_cells[partition] = p_cell;
	cell.partitions.push_back(partition);
	p_state->partitions_live++;
	return partition;
}

void RenderRaytracing::_slot_unplace(RTViewportState *p_state, RTSlotShadow &r_shadow) {
	if (r_shadow.partition == RD::PARTITION_GLOBAL) {
		const uint32_t index = r_shadow.global_index;
		ERR_FAIL_COND(index >= p_state->global_slots.size());
		const uint32_t last = p_state->global_slots[p_state->global_slots.size() - 1];
		p_state->global_slots[index] = last;
		p_state->slots[last].global_index = index;
		p_state->global_slots.resize(p_state->global_slots.size() - 1);
		r_shadow.global_index = UINT32_MAX;
		return;
	}
	const uint32_t partition = r_shadow.partition;
	ERR_FAIL_COND(partition >= p_state->partition_high_water || p_state->partition_occupancy[partition] == 0);
	if (--p_state->partition_occupancy[partition] > 0) {
		return;
	}
	// An empty bucket leaves its cell, and any cell may take its id: every partition is sized for
	// every slot, so the disabled records left in it never count against its next cell.
	HashMap<Vector3i, RTViewportState::SlotCell>::Iterator E = p_state->slot_cells.find(p_state->partition_cells[partition]);
	if (E) {
		E->value.partitions.erase(partition);
		if (E->value.partitions.is_empty()) {
			p_state->slot_cells.remove(E);
		}
	}
	p_state->partition_free.push_back(partition);
	p_state->partitions_live--;
}

void RenderRaytracing::_slot_place(RTViewportState *p_state, uint32_t p_slot, RTSlotShadow &r_shadow, bool p_was_enabled, const Vector3i &p_cell, bool p_prefer_global) {
	const bool global_room = p_state->global_slots.size() < p_state->global_capacity;
	if (p_was_enabled) {
		if (p_prefer_global && r_shadow.partition == RD::PARTITION_GLOBAL) {
			r_shadow.cell = p_cell;
			return;
		}
		if ((!p_prefer_global || !global_room) && r_shadow.partition != RD::PARTITION_GLOBAL && p_state->partition_cells[r_shadow.partition] == p_cell) {
			if (p_prefer_global) {
				p_state->slot_counters.global_overflow++;
			}
			r_shadow.cell = p_cell;
			return; // It stays in its own cell's bucket.
		}
		_slot_unplace(p_state, r_shadow);
	}
	r_shadow.cell = p_cell;
	if (p_prefer_global && global_room) {
		r_shadow.partition = RD::PARTITION_GLOBAL;
		r_shadow.global_index = p_state->global_slots.size();
		p_state->global_slots.push_back(p_slot);
		p_state->slot_counters.global_entries_total++;
		return;
	}
	if (p_prefer_global) {
		p_state->slot_counters.global_overflow++;
	}
	r_shadow.partition = _slot_cell_partition(p_state, p_cell);
	p_state->partition_occupancy[r_shadow.partition]++;
}

// ---------------------------------------------------------------------------
// The commit, in build_acceleration_structures after the frame's BLAS builds: the sweep, the
// frame's instances against their shadows, the returns from the global partition, the motion rows
// to reset, the builds, then the scatter. False when the structure could not be brought up to its
// shadows, and the frame does not trace.
// ---------------------------------------------------------------------------

bool RenderRaytracing::_flush_slot_writes(RTViewportState *p_state, bool p_from_scratch, bool p_translations) {
	// One build of the staged writes: the first build of a from-scratch commit disables every slot
	// it does not write, so a later chunk of the same commit builds in place. The commit's last build
	// also writes the partition translations that changed (D94c).
	RD *rd = RD::get_singleton();
	RTViewportState::SlotCounters &counters = p_state->slot_counters;
	const uint32_t build = p_state->slot_build;
	for (const RD::PartitionedInstanceWrite &write : slot_writes) {
		uint32_t *stamp = nullptr;
		if (write.partition == RD::PARTITION_GLOBAL) {
			stamp = &p_state->global_touched;
		} else if (write.partition < p_state->partition_touched.size()) {
			stamp = &p_state->partition_touched[write.partition];
		}
		if (stamp && *stamp != build) {
			*stamp = build;
			counters.partitions_touched++;
		}
	}
	counters.writes += slot_writes.size();
	if (p_from_scratch) {
		// A structure built from scratch holds no translation it is not given; with several chunks
		// that is the commit's first build, and its last writes them.
		for (Vector3 &translation : p_state->partition_translations) {
			translation = Vector3(Math::NaN, Math::NaN, Math::NaN);
		}
	}
	slot_translation_writes.clear();
	if (p_translations) {
		for (uint32_t partition = 0; partition < p_state->partition_high_water; partition++) {
			if (p_state->partition_occupancy[partition] == 0) {
				continue; // Free: its translation is written when a cell takes it.
			}
			const Vector3 translation = _slot_cell_translation(p_state, p_state->partition_cells[partition]);
			if (p_state->partition_translations[partition] == translation) {
				continue;
			}
			p_state->partition_translations[partition] = translation;
			RD::PartitionTranslationWrite write;
			write.partition = partition;
			write.translation = translation;
			slot_translation_writes.push_back(write);
		}
		counters.translations += slot_translation_writes.size();
	}
	const uint64_t start_usec = OS::get_singleton()->get_ticks_usec();
	const Error err = rd->ptlas_build(p_state->tlas, slot_writes, slot_translation_writes, p_from_scratch);
	counters.build_usec += OS::get_singleton()->get_ticks_usec() - start_usec;
	slot_writes.clear();
	if (err != OK) {
		return false;
	}
	uint32_t rd_writes = 0, stale_rewrites = 0, dangling = 0;
	uint64_t scan_usec = 0;
	rd->ptlas_get_last_build_counts(p_state->tlas, rd_writes, stale_rewrites, dangling, scan_usec);
	counters.stale_rewrites += stale_rewrites;
	counters.scan_usec += scan_usec;
	counters.dangling_last += dangling;
	return true;
}

bool RenderRaytracing::_commit_slots(RTViewportState *p_state) {
	const uint64_t start_usec = OS::get_singleton()->get_ticks_usec();
	RD *rd = RD::get_singleton();
	_reclaim_idle_viewports(p_state);
	p_state->slot_commit_frame = rd->get_frames_drawn();

	RTViewportState::SlotCounters &counters = p_state->slot_counters;
	counters.writes = 0;
	counters.writes_new = 0;
	counters.writes_moved = 0;
	counters.writes_blas = 0;
	counters.writes_rebuilt = 0;
	counters.writes_return = 0;
	counters.writes_origin = 0;
	counters.translations = 0;
	counters.disables = 0;
	counters.rows = 0;
	counters.motion_rows = 0;
	counters.partitions_touched = 0;
	counters.global_overflow = 0;
	counters.stale_rewrites = 0;
	counters.build_usec = 0;
	counters.scan_usec = 0;
	counters.dangling_last = 0;
	counters.from_scratch_last = 0;
	counters.builds = 0;

	const uint32_t build = p_state->slot_build;
	slot_writes.clear();
	slot_row_refs.clear();
	if (p_state->slots.size() < slot_high_water) {
		p_state->slots.resize(slot_high_water);
	}
	if (p_state->slot_range_views.size() < slot_ranges.size()) {
		p_state->slot_range_views.resize(slot_ranges.size());
	}
	const uint32_t return_bucket_count = ptlas_stable_frames + 1;
	if (p_state->global_return_buckets.size() != return_bucket_count) {
		p_state->global_return_buckets.resize(return_bucket_count);
	}

	// The writes are staged and built in chunks of RT_SLOT_CHUNK once a capacity is known; until
	// the structure exists (its first commit, or after a reset), they are only staged.
	bool from_scratch = p_state->slot_rebuild || p_state->tlas.is_null();
	bool build_ok = true;
	auto write_slot = [&](uint32_t p_slot, const RTSlotShadow &p_shadow) {
		if (from_scratch) {
			return; // The build from scratch writes every enabled slot from the shadows.
		}
		RD::PartitionedInstanceWrite write;
		write.transform = _slot_record_transform(p_state, p_shadow);
		write.slot = p_slot;
		write.partition = p_shadow.partition;
		write.id = p_slot;
		write.mask = p_shadow.mask;
		write.hit_sbt_offset = p_shadow.sbt_offset;
		write.flags = BitField<RD::AccelerationStructureInstanceFlagBits>(p_shadow.flags);
		write.blas = p_shadow.blas;
		slot_writes.push_back(write);
		if (slot_writes.size() >= RT_SLOT_CHUNK && build_ok) {
			if (p_state->partition_high_water > p_state->partition_capacity) {
				// A partition past the structure's: the commit starts from scratch on a new one.
				from_scratch = true;
				slot_writes.clear();
			} else {
				build_ok = _flush_slot_writes(p_state, false);
				counters.builds++;
			}
		}
	};
	auto disable_slot = [&](uint32_t p_slot) {
		RTSlotShadow &shadow = p_state->slots[p_slot];
		if (shadow.blas.is_null()) {
			return;
		}
		_slot_unplace(p_state, shadow);
		if (shadow.partition == RD::PARTITION_GLOBAL) {
			// Its disabled record goes to partition 0, which is sized for every slot, so the
			// global partition holds no dead records (review finding 9).
			shadow.partition = 0;
		}
		shadow.blas = RID();
		write_slot(p_slot, shadow);
		counters.disables++;
		counters.enabled--;
		p_state->emissive_slots.erase(p_slot);
	};

	// The structure must cover every slot this commit can write; a new one starts from scratch.
	if (!p_state->tlas.is_valid() || p_state->slot_capacity < slot_high_water) {
		from_scratch = true;
		slot_writes.clear();
	}

	// The sweep: the slots of a range the walk did not reach in this build, or whose owner let it
	// go, are disabled, and the viewport's claim on the range ends.
	for (uint32_t r = 0; r < slot_ranges.size(); r++) {
		RTSlotRange &range = slot_ranges[r];
		if (!range.live) {
			continue;
		}
		RTSlotRangeView &view = p_state->slot_range_views[r];
		if (view.serial != range.serial) {
			DEV_ASSERT(!view.enabled);
			view = RTSlotRangeView();
			view.serial = range.serial;
			continue;
		}
		if (!view.enabled || (!range.released && view.present_frame == build)) {
			continue;
		}
		for (uint32_t slot = range.first; slot < range.first + range.count; slot++) {
			disable_slot(slot);
		}
		view.enabled = false;
		p_state->slot_mm_keys.erase(r);
		range.enabled_viewports--;
		_slot_range_try_free(r);
	}

	// The instances the walk pushed: each against its slot's shadow.
	const bool any_stale = p_state->tlas.is_valid() && rd->ptlas_get_stale_blas_count(p_state->tlas) > 0;
	// An origin shift changes the partitions' translations and the global partition's records, not the
	// cell-relative shadows (D94c).
	const bool origin_shifted = p_state->slot_origin_valid && p_state->slot_origin != p_state->rt_origin;
	p_state->slot_origin = p_state->rt_origin;
	p_state->slot_origin_valid = true;
	LocalVector<uint32_t> &return_bucket_now = p_state->global_return_buckets[build % return_bucket_count];
	for (uint32_t i = 0; i < frame_slots.size(); i++) {
		const uint32_t slot = frame_slots[i];
		RTSlotShadow &shadow = p_state->slots[slot];
		const RID blas = blass[i];
		if (blas.is_null()) {
			disable_slot(slot); // A deformed surface whose build the frame dropped (D82).
			continue;
		}
		// Compared as the shadow holds it, relative to its cell.
		const Vector3i rows_cell = shadow.cell;
		float rows[12];
		_slot_relative_transform(blas_transforms[i], _slot_cell_origin(rows_cell), rows);
		const uint8_t flags = uint8_t(instance_flags[i]);
		const uint8_t mask = instance_masks[i];
		const uint32_t sbt_offset = sbt_offsets[i];
		const bool was_enabled = shadow.blas.is_valid();
		const bool moved = memcmp(shadow.transform, rows, sizeof(rows)) != 0;
		const bool instance_changed = !was_enabled || shadow.blas != blas || moved || shadow.flags != flags || shadow.mask != mask || shadow.sbt_offset != sbt_offset;
		const bool rebuilt = !instance_changed && any_stale && rd->ptlas_is_blas_stale(p_state->tlas, blas);
		const bool rows_changed = !shadow.rows_valid || shadow.geometry_generation != frame_geometry_generations[i] || shadow.material_generation != frame_material_generations[i];
		const int32_t motion_index = motion_indices[i];

		if (instance_changed || rebuilt) {
			if (!was_enabled) {
				counters.writes_new++;
				counters.enabled++;
			} else if (shadow.blas != blas) {
				counters.writes_blas++;
			} else if (moved) {
				counters.writes_moved++;
			} else if (rebuilt) {
				counters.writes_rebuilt++;
			} else {
				counters.writes_blas++; // Flags, mask or hit group: counted with the BLAS.
			}
			// A slot rewritten while enabled goes global; a new one to its cell.
			_slot_place(p_state, slot, shadow, was_enabled, _slot_cell(blas_transforms[i], geometry_data[i]), was_enabled);
			if (shadow.cell != rows_cell) {
				_slot_relative_transform(blas_transforms[i], _slot_cell_origin(shadow.cell), rows);
			}
			memcpy(shadow.transform, rows, sizeof(rows));
			shadow.blas = blas;
			shadow.flags = flags;
			shadow.mask = mask;
			shadow.sbt_offset = sbt_offset;
			shadow.write_frame = build;
			if (shadow.partition == RD::PARTITION_GLOBAL) {
				return_bucket_now.push_back(slot); // Looked at again ptlas_stable_frames builds on.
			}
			write_slot(slot, shadow);
		}
		if (rows_changed || motion_index >= 0) {
			SlotRowRef ref;
			ref.slot = slot;
			ref.source = rows_changed ? i : UINT32_MAX;
			ref.motion_index = motion_index;
			slot_row_refs.push_back(ref);
			if (rows_changed) {
				shadow.rows_valid = true;
				shadow.geometry_generation = frame_geometry_generations[i];
				shadow.material_generation = frame_material_generations[i];
				counters.rows++;
			} else {
				counters.motion_rows++;
			}
		}
		if (motion_index >= 0) {
			if (!shadow.motion || shadow.motion_frame != build) {
				p_state->motion_slots_next.push_back(slot);
			}
			shadow.motion = true;
			shadow.motion_frame = build;
		}
		if (instance_changed || rows_changed) {
			if (material_data[i].flags & RT_MAT_FLAG_EMISSIVE_CONSTANT) {
				RTSlotEmissive &emissive = p_state->emissive_slots[slot];
				emissive.transform = blas_transforms[i];
				emissive.geometry = geometry_data[i];
				emissive.material = material_data[i];
				emissive.sbt_offset = sbt_offset;
				emissive.flags = instance_flags[i];
			} else {
				p_state->emissive_slots.erase(slot);
			}
		}
	}

	// The global partition's slots unwritten for ptlas_stable_frames builds return to their cell: the
	// slots written ptlas_stable_frames builds ago, from that build's bucket, that are still global
	// and have not been written since.
	if (build > ptlas_stable_frames) {
		const uint32_t written = build - ptlas_stable_frames;
		LocalVector<uint32_t> &bucket = p_state->global_return_buckets[written % return_bucket_count];
		for (uint32_t slot : bucket) {
			RTSlotShadow &shadow = p_state->slots[slot];
			if (shadow.blas.is_null() || shadow.partition != RD::PARTITION_GLOBAL || shadow.write_frame != written) {
				continue;
			}
			_slot_place(p_state, slot, shadow, true, shadow.cell, false);
			shadow.write_frame = build;
			write_slot(slot, shadow);
			counters.writes_return++;
			counters.returns_total++;
		}
		bucket.clear();
	}

	// An origin shift: the global partition has no translation, so its records carry their cells'
	// and are rewritten, those this build has not written already (D94c). Their write build stays,
	// so they return to their cells as they would have.
	if (origin_shifted) {
		for (uint32_t slot : p_state->global_slots) {
			const RTSlotShadow &shadow = p_state->slots[slot];
			if (shadow.write_frame == build || !rd->acceleration_structure_is_valid(shadow.blas)) {
				continue; // A freed BLAS is left to the structure's net (D94b, review finding 5).
			}
			write_slot(slot, shadow);
			if (!from_scratch) {
				counters.writes_origin++; // A build from scratch writes it anyway.
			}
		}
	}

	// The slots whose motion row the last build wrote and this one did not get -1 back.
	for (uint32_t slot : p_state->motion_slots) {
		RTSlotShadow &shadow = p_state->slots[slot];
		if (!shadow.motion || shadow.motion_frame == build) {
			continue;
		}
		shadow.motion = false;
		SlotRowRef ref;
		ref.slot = slot;
		ref.source = UINT32_MAX;
		ref.motion_index = -1;
		slot_row_refs.push_back(ref);
		counters.motion_rows++;
	}
	SWAP(p_state->motion_slots, p_state->motion_slots_next);
	p_state->motion_slots_next.clear();

	// A failure leaves the structure behind its shadows: the next build starts from scratch, the
	// walk pushes every instance again and its rows are written whole, and this frame does not
	// trace (the caller skips the fog and the trace). The rows are scattered only once the
	// structure holds what they describe.
	auto fail = [&]() {
		p_state->slot_rebuild = true;
		p_state->slot_mm_keys.clear();
		for (RTSlotShadow &shadow : p_state->slots) {
			shadow.rows_valid = false;
		}
		slot_writes.clear();
		slot_row_refs.clear();
		counters.commit_usec = OS::get_singleton()->get_ticks_usec() - start_usec;
		return false;
	};
	if (!build_ok) {
		return fail();
	}
	if (p_state->partition_capacity < p_state->partition_high_water) {
		from_scratch = true; // New partitions past the structure's: a new one, from scratch.
	}
	if (from_scratch) {
		if (!_ensure_slot_capacity(p_state)) {
			return fail();
		}
		// Every enabled slot, from the shadows: the first chunk starts from scratch.
		slot_writes.clear();
		counters.writes = 0;
		counters.partitions_touched = 0;
		counters.enabled = 0;
		bool first = true;
		for (uint32_t slot = 0; slot < p_state->slots.size(); slot++) {
			const RTSlotShadow &shadow = p_state->slots[slot];
			if (shadow.blas.is_null()) {
				continue;
			}
			RD::PartitionedInstanceWrite write;
			write.transform = _slot_record_transform(p_state, shadow);
			write.slot = slot;
			write.partition = shadow.partition;
			write.id = slot;
			write.mask = shadow.mask;
			write.hit_sbt_offset = shadow.sbt_offset;
			write.flags = BitField<RD::AccelerationStructureInstanceFlagBits>(shadow.flags);
			write.blas = shadow.blas;
			slot_writes.push_back(write);
			counters.enabled++;
			if (slot_writes.size() >= RT_SLOT_CHUNK) {
				if (!_flush_slot_writes(p_state, first)) {
					return fail();
				}
				counters.builds++;
				first = false;
			}
		}
		if (!_flush_slot_writes(p_state, first, true)) {
			return fail();
		}
		counters.builds++;
		counters.from_scratch++;
		counters.from_scratch_last = 1;
	} else {
		// The rest of the writes; with none, the build call still writes the slots of a BLAS rebuilt
		// in place again (or finds nothing to do).
		const bool any = !slot_writes.is_empty() || origin_shifted;
		if (!_flush_slot_writes(p_state, false, true)) {
			return fail();
		}
		counters.builds += any ? 1 : 0;
	}
	p_state->slot_rebuild = false;

	// Slots still on a BLAS that was freed were disabled by the structure's safety net: their
	// shadows follow it, so nothing believes them enabled (review finding 5).
	if (counters.dangling_last > 0) {
		counters.dangling_disables += counters.dangling_last;
		for (uint32_t slot = 0; slot < p_state->slots.size(); slot++) {
			RTSlotShadow &shadow = p_state->slots[slot];
			if (shadow.blas.is_valid() && !rd->acceleration_structure_is_valid(shadow.blas)) {
				_slot_unplace(p_state, shadow);
				if (shadow.partition == RD::PARTITION_GLOBAL) {
					shadow.partition = 0;
				}
				shadow.blas = RID();
				p_state->emissive_slots.erase(slot);
				counters.enabled--;
			}
		}
		p_state->slot_mm_keys.clear();
	}

	if (!_scatter_slot_rows(p_state)) {
		return fail(); // The rows the structure now names were never written.
	}

	counters.partitions_live = p_state->partitions_live;
	counters.global = p_state->global_slots.size();
	counters.global_peak = MAX(counters.global_peak, counters.global);
	counters.commit_usec = OS::get_singleton()->get_ticks_usec() - start_usec;
	counters.commit_usec_sum += counters.commit_usec;
	counters.build_usec_sum += counters.build_usec;
	counters.scan_usec_sum += counters.scan_usec;

	if (origin_shifted) {
		// The evidence for partition translation (D94c): a shift writes the live partitions'
		// translations and the global partition's records, and no cell slot for the shift's sake.
		print_verbose(vformat("RT: slots (D94c) build %d: rt_origin moved to (%.0f, %.0f, %.0f); %d partition translations of %d live partitions, %d global records rewritten for it, %d other instance writes (%d moved, %d new, %d BLAS, %d rebuilt, %d returned, %d disables).",
				build, p_state->rt_origin.x, p_state->rt_origin.y, p_state->rt_origin.z, counters.translations, counters.partitions_live, counters.writes_origin, counters.writes - counters.writes_origin, counters.writes_moved, counters.writes_new, counters.writes_blas, counters.writes_rebuilt, counters.writes_return, counters.disables));
	}
	if (build == 1 || build % 300 == 0) {
		print_verbose(vformat("RT: slots (D94c) build %d: %d live (high water %d, capacity %d), %d enabled; %d instance writes in %d builds (%d new, %d moved, %d BLAS, %d rebuilt, %d returned, %d for an origin shift, %d disables), %d partition translations, %d stale rewrites, %d row writes, %d motion rows; %d of %d partitions touched, %d global (%d overflowed), %d from-scratch builds, %d entries to the global partition (peak %d), %d returns, %d dangling disables and %d viewports reclaimed over the run; %d emissive slots; commit %d us, of which the build calls %d us (their stale scans %d us); means over the run %.1f, %.1f and %.1f us.",
				build, slot_live, slot_high_water, p_state->slot_capacity, counters.enabled, counters.writes, counters.builds, counters.writes_new, counters.writes_moved, counters.writes_blas, counters.writes_rebuilt, counters.writes_return, counters.writes_origin, counters.disables, counters.translations, counters.stale_rewrites, counters.rows, counters.motion_rows,
				counters.partitions_touched, counters.partitions_live, counters.global, counters.global_overflow, counters.from_scratch, counters.global_entries_total, counters.global_peak, counters.returns_total, counters.dangling_disables, slot_counters_reclaimed, p_state->emissive_slots.size(), counters.commit_usec, counters.build_usec, counters.scan_usec, double(counters.commit_usec_sum) / build, double(counters.build_usec_sum) / build, double(counters.scan_usec_sum) / build));
	}
	return true;
}
