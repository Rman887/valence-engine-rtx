/**************************************************************************/
/*  render_raytracing.cpp                                                 */
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

#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "servers/rendering/renderer_rd/environment/fog.h"
#include "servers/rendering/renderer_rd/environment/sky.h"
#include "servers/rendering/renderer_rd/forward_clustered/render_forward_clustered.h"
#include "servers/rendering/renderer_rd/forward_clustered/scene_shader_raytracing.h"
#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/light_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/mesh_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/rendering_server_globals.h"
#include "servers/rendering/storage/environment_storage.h"

using namespace RendererSceneRenderImplementation;

RenderRaytracing *RenderRaytracing::singleton = nullptr;

static _FORCE_INLINE_ uint64_t _align_up(uint64_t p_value, uint64_t p_alignment) {
	return p_alignment > 1 ? (p_value + p_alignment - 1) / p_alignment * p_alignment : p_value;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// The cluster limits a deformed layout is built under: the mesh table's limits clamped to the
// device, so a table's clusters are taken as they are and the runs fallback stays within them.
static _FORCE_INLINE_ void _deformed_cluster_maxima(const RD::ClusterAccelerationStructureCapabilities &p_caps, uint32_t &r_max_triangles, uint32_t &r_max_vertices) {
	r_max_triangles = MIN(uint32_t(RenderingServerTypes::MESH_CLUSTER_MAX_TRIANGLES), p_caps.max_triangles_per_cluster);
	r_max_vertices = MIN(uint32_t(RenderingServerTypes::MESH_CLUSTER_MAX_VERTICES), p_caps.max_vertices_per_cluster);
}

// The cluster arenas' allocator (D81, instanced twice since D93b), defined with the arena below.
static bool _arena_allocate(LocalVector<RTClusterArenaChunk> &r_arena, uint64_t &r_reserved_bytes, uint64_t &r_live_bytes, uint64_t p_alignment, BitField<RD::BufferCreationBits> p_usage, BitField<RDD::PipelineStageBits> p_address_read_stages, const String &p_name, uint64_t p_size, uint32_t &r_chunk, uint64_t &r_offset);
static void _arena_free(LocalVector<RTClusterArenaChunk> &r_arena, uint64_t &r_reserved_bytes, uint64_t &r_live_bytes, uint32_t p_chunk, uint64_t p_offset, uint64_t p_size);
static void _arena_sweep(LocalVector<RTClusterArenaChunk> &r_arena, LocalVector<RTClusterRetiredRegion> &r_retired, uint64_t &r_reserved_bytes, uint64_t &r_live_bytes);

void RenderRaytracing::initialize(RenderForwardClustered *p_owner) {
	owner = p_owner;
	singleton = this;
	bindless_block = memnew(BindlessBlock);
	// A surface its mesh drops releases its RT cache entry at once (D81), not at the slot's reuse.
	RendererRD::MeshStorage::get_singleton()->set_surface_freed_callback(&RenderRaytracing::_mesh_surface_freed);

	// The build budget (D97): deferrable builds admitted within a frame's triangles and builds.
	build_budget.enabled = GLOBAL_GET("rendering/pathtracing/build_budget");
	if (build_budget.enabled) {
		build_budget.triangles_per_frame = uint64_t(MAX(int64_t(1024), int64_t(GLOBAL_GET("rendering/pathtracing/build_budget_triangles_per_frame"))));
		build_budget.builds_per_frame = uint32_t(MAX(1, int(GLOBAL_GET("rendering/pathtracing/build_budget_builds_per_frame"))));
		build_budget.max_wait_frames = uint32_t(MAX(1, int(GLOBAL_GET("rendering/pathtracing/build_budget_max_wait_frames"))));
		print_line(vformat("RT: acceleration-structure build budget on: %d triangles and %d builds a frame, first in line after %d frames' wait (D97).", build_budget.triangles_per_frame, build_budget.builds_per_frame, build_budget.max_wait_frames));
	}

	cluster_blas_enabled = RD::get_singleton()->has_feature(RD::SUPPORTS_CLUSTER_ACCELERATION_STRUCTURE);
	if (cluster_blas_enabled) {
		cluster_compaction_enabled = GLOBAL_GET("rendering/pathtracing/cluster_blas_compaction");
		cluster_deformed_enabled = GLOBAL_GET("rendering/pathtracing/cluster_blas_deformed");
		// Their CLAS instantiated from per-mesh templates (D107) where the device writes template infos.
		const RD::ClusterAccelerationStructureCapabilities &template_caps = RD::get_singleton()->get_cluster_acceleration_structure_capabilities();
		deformed_cluster_templates_enabled = cluster_deformed_enabled && template_caps.instantiate_info_size > 0 && template_caps.triangle_template_build_info_size > 0 && template_caps.cluster_template_byte_alignment > 0;
		cluster_lod_select_enabled = GLOBAL_GET("rendering/pathtracing/cluster_lod") && GLOBAL_GET("rendering/pathtracing/cluster_lod_select");
		cluster_lod_pixel_error = MAX(0.0f, float(GLOBAL_GET("rendering/pathtracing/cluster_lod_pixel_error")));
		cluster_lod_share_enabled = cluster_lod_select_enabled && GLOBAL_GET("rendering/pathtracing/cluster_lod_share");
		cluster_lod_share_tolerance = uint32_t(CLAMP(int(GLOBAL_GET("rendering/pathtracing/cluster_lod_share_tolerance")), 0, 16));
		cluster_lod_cache_enabled = cluster_lod_share_enabled && GLOBAL_GET("rendering/pathtracing/cluster_lod_cache");
		cluster_lod_traversal_enabled = cluster_lod_select_enabled && GLOBAL_GET("rendering/pathtracing/cluster_lod_traversal");
		cluster_lod_max_instances = uint32_t(MIN(RD::get_singleton()->limit_get(RD::LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Y), uint64_t(UINT32_MAX)));
		const RD::ClusterAccelerationStructureCapabilities &caps = RD::get_singleton()->get_cluster_acceleration_structure_capabilities();
		// CLAS residency (D92) needs the traversal (its requests), the cached levels (the finest
		// level is one) and the compaction (a load is its two steps).
		const bool stream_wanted = cluster_lod_traversal_enabled && cluster_lod_cache_enabled && cluster_compaction_enabled && GLOBAL_GET("rendering/pathtracing/cluster_lod_stream");
		// Geometry residency (D93) takes a cluster hit's positions from its CLAS, so every cluster
		// build carries the data access flag while it is on: one build operation batches surfaces,
		// static and deformed alike, and the BLAS over the clusters carries it too, as the
		// position-fetch rule names the structure of the intersection. Set before any size query.
		if (GLOBAL_GET("rendering/pathtracing/cluster_lod_stream_geometry")) {
			if (caps.position_fetch) {
				// The mesh storage's own test, which keeps the RAM copy the blobs come from (D93b): stream_wanted and support.
				cluster_build_data_access = RendererRD::MeshStorage::is_cluster_geometry_streaming_wanted();
				DEV_ASSERT(cluster_build_data_access == stream_wanted);
			} else {
				WARN_PRINT("rendering/pathtracing/cluster_lod_stream_geometry is on but the device has no VK_KHR_ray_tracing_position_fetch; surfaces keep their geometry resident.");
			}
		}
		// The deformed selection (D87) builds each selected CLAS into its own slot: the worst case
		// of one cluster under the limits every cluster obeys, at the cluster alignment.
		cluster_lod_deformed_enabled = cluster_lod_select_enabled && cluster_deformed_enabled && GLOBAL_GET("rendering/pathtracing/cluster_lod_deformed");
		if (cluster_lod_deformed_enabled) {
			uint32_t max_triangles = 0, max_vertices = 0;
			_deformed_cluster_maxima(caps, max_triangles, max_vertices);
			RD::ClusterAccelerationStructureInput slot_input;
			slot_input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_TRIANGLE_CLUSTER;
			slot_input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
			slot_input.max_acceleration_structure_count = 1;
			slot_input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT);
			slot_input.vertex_format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
			slot_input.max_cluster_triangle_count = max_triangles;
			slot_input.max_cluster_vertex_count = max_vertices;
			slot_input.max_total_triangle_count = max_triangles;
			slot_input.max_total_vertex_count = max_vertices;
			uint64_t slot_size = 0, slot_scratch = 0;
			RD::get_singleton()->cluster_acceleration_structure_get_build_sizes(slot_input, slot_size, slot_scratch);
			// With templates (D108) the selection instantiates the CLAS instead: the slot holds the
			// worst case of an instantiation too, under the instantiation's flags and the templates'.
			if (deformed_cluster_templates_enabled && slot_size > 0) {
				RD::ClusterAccelerationStructureInput instantiate_input = slot_input;
				instantiate_input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_INSTANTIATE_TRIANGLE_CLUSTER;
				uint64_t fast_build_size = 0, fast_trace_size = 0;
				RD::get_singleton()->cluster_acceleration_structure_get_build_sizes(instantiate_input, fast_build_size, slot_scratch);
				instantiate_input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
				RD::get_singleton()->cluster_acceleration_structure_get_build_sizes(instantiate_input, fast_trace_size, slot_scratch);
				cluster_lod_deformed_templated = fast_build_size > 0 && fast_trace_size > 0;
				if (cluster_lod_deformed_templated) {
					slot_size = MAX(slot_size, MAX(fast_build_size, fast_trace_size));
				} else {
					WARN_PRINT("RT: the driver sized an instantiated cluster acceleration structure at 0 bytes; the deformed selection builds its CLAS each frame (D87).");
				}
			}
			cluster_lod_clas_slot_size = _align_up(slot_size, uint64_t(MAX(caps.cluster_byte_alignment, 1u)));
			// Persistent items (D109) take arguments, not region, a slot: an instantiation info, a
			// destination and a cluster index, under the same bound.
			cluster_lod_persistent_ceiling_slots = uint32_t(MIN(RT_LOD_CLAS_REGION_CEILING / (uint64_t(caps.instantiate_info_size) + sizeof(uint64_t) + sizeof(uint32_t)), uint64_t(UINT32_MAX)));
			const uint64_t region_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
			cluster_lod_clas_ceiling_clusters = slot_size > 0 && RT_LOD_CLAS_REGION_CEILING > region_alignment ? uint32_t(MIN((RT_LOD_CLAS_REGION_CEILING - region_alignment) / cluster_lod_clas_slot_size, uint64_t(UINT32_MAX))) : 0;
			if (slot_size == 0) {
				WARN_PRINT("RT: the driver sized one cluster acceleration structure at 0 bytes; deformed meshes trace their finest level.");
				cluster_lod_deformed_enabled = false;
			}
		}
		// CLAS residency (D92). A load's budget estimate is the worst case of one CLAS under the
		// limits every cluster obeys, in either position format.
		cluster_lod_stream_enabled = stream_wanted;
		if (cluster_lod_stream_enabled) {
			cluster_lod_stream_budget = uint64_t(MAX(16, int(GLOBAL_GET("rendering/pathtracing/cluster_lod_stream_budget_mb")))) * 1024 * 1024;
			cluster_lod_stream_builds_per_frame = uint32_t(MAX(64, int(GLOBAL_GET("rendering/pathtracing/cluster_lod_stream_builds_per_frame"))));
			cluster_lod_stream_upload_cap = uint64_t(MAX(1, int(GLOBAL_GET("rendering/pathtracing/cluster_lod_stream_upload_mb_per_frame")))) * 1024 * 1024;
			// The floor outlasts a use report lost past the read prefix (its group reports again a
			// period later) and the readback's frame queue.
			cluster_lod_stream_min_age = MAX(uint32_t(MAX(1, int(GLOBAL_GET("rendering/pathtracing/cluster_lod_stream_min_age_frames")))), RT_LOD_STREAM_TOUCH_PERIOD * 3 + uint32_t(RD::get_singleton()->get_frame_delay()));
			uint32_t max_triangles = 0, max_vertices = 0;
			_deformed_cluster_maxima(caps, max_triangles, max_vertices);
			const RD::DataFormat formats[2] = { RD::DATA_FORMAT_R32G32B32_SFLOAT, RD::DATA_FORMAT_R16G16B16A16_UNORM };
			for (const RD::DataFormat format : formats) {
				RD::ClusterAccelerationStructureInput input;
				input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_TRIANGLE_CLUSTER;
				input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
				input.max_acceleration_structure_count = 1;
				input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
				input.vertex_format = format;
				input.max_cluster_triangle_count = max_triangles;
				input.max_cluster_vertex_count = max_vertices;
				input.max_total_triangle_count = max_triangles;
				input.max_total_vertex_count = max_vertices;
				uint64_t size = 0, scratch = 0;
				RD::get_singleton()->cluster_acceleration_structure_get_build_sizes(input, size, scratch);
				cluster_lod_stream_cluster_worst = MAX(cluster_lod_stream_cluster_worst, _align_up(size, uint64_t(MAX(caps.cluster_byte_alignment, 1u))));
			}
			if (cluster_lod_stream_cluster_worst == 0) {
				WARN_PRINT("RT: the driver sized one cluster acceleration structure at 0 bytes; cluster LOD streaming is off.");
				cluster_lod_stream_enabled = false;
			}
		}
		cluster_lod_stream_geometry_enabled = cluster_lod_stream_enabled && cluster_build_data_access;
		if (cluster_lod_stream_geometry_enabled && OS::get_singleton()->is_stdout_verbose()) {
			Vector<uint8_t> zeros;
			zeros.resize_initialized(sizeof(uint32_t) * RT_POSITION_FETCH_CHECK_WORDS);
			position_fetch_check_buffer = RD::get_singleton()->storage_buffer_create(zeros.size(), zeros);
			RD::get_singleton()->set_resource_name(position_fetch_check_buffer, "RT position fetch check");
		}
		// One line on the console (the setting is off by default): the harness reads it as proof the path is active.
		print_line(vformat("RT: static BLAS built from clusters of up to %d triangles and %d vertices (D79), compaction %s, deformed meshes %s, cluster LOD selection %s, deformed selection %s, CLAS streaming %s.", caps.max_triangles_per_cluster, caps.max_vertices_per_cluster, cluster_compaction_enabled ? "on" : "off", cluster_deformed_enabled ? (deformed_cluster_templates_enabled ? "instantiated from cluster templates (D107)" : "rebuilt from clusters (D82)") : "refit as triangle BLAS", cluster_lod_select_enabled ? vformat("on at %.2f px (D86), traversal %s (D91), BLAS sharing %s", cluster_lod_pixel_error, cluster_lod_traversal_enabled ? "on" : "off", cluster_lod_share_enabled ? vformat("on within %d levels (D89), cached levels %s (D90)", cluster_lod_share_tolerance, cluster_lod_cache_enabled ? "on" : "off") : String("off (D89)")) : "off", cluster_lod_deformed_enabled ? vformat("on, %d byte slots (D87)%s", cluster_lod_clas_slot_size, cluster_lod_deformed_templated ? ", instantiated from templates (D108)" : "") : "off", cluster_lod_stream_enabled ? vformat("on within %d MB, %d clusters a frame, unused %d frames before unloading, %d bytes a CLAS at worst (D92)", cluster_lod_stream_budget / (1024 * 1024), cluster_lod_stream_builds_per_frame, cluster_lod_stream_min_age, cluster_lod_stream_cluster_worst) : "off"));
		if (cluster_lod_stream_geometry_enabled) {
			print_line(vformat("RT: geometry residency on: cluster builds allow data access, hits fetch CLAS positions (D93), blobs load with their groups within %d MB of uploads a frame (D93c).", cluster_lod_stream_upload_cap / (1024 * 1024)));
		}
	} else if (GLOBAL_GET("rendering/pathtracing/cluster_blas")) {
		WARN_PRINT("rendering/pathtracing/cluster_blas is on but the device has no VK_NV_cluster_acceleration_structure; static BLAS stay triangle BLAS.");
	}

	// Initialize merged MultiMesh BLAS compute shader.
	Vector<String> merge_modes;
	merge_modes.push_back("\n");
	merge_modes.push_back("\n#define MODE_INDEXED\n");
	mm_merge_shader.shader.initialize(merge_modes);
	mm_merge_shader.version = mm_merge_shader.shader.version_create();
	for (int i = 0; i < MergeShader::MODE_MAX; i++) {
		mm_merge_shader.version_shader[i] = mm_merge_shader.shader.version_get_shader(mm_merge_shader.version, i);
		mm_merge_shader.pipeline[i] = RD::get_singleton()->compute_pipeline_create(mm_merge_shader.version_shader[i]);
	}

	// Slot-resident instances (D94b): with every TLAS a PTLAS (D94a), the walk writes only the
	// instances that changed, and the scatter pass writes their rows.
	ptlas_slots_enabled = RD::get_singleton()->get_partitioned_acceleration_structure_capabilities().supported;
	if (ptlas_slots_enabled) {
		ptlas_cell_size = MAX(double(GLOBAL_GET("rendering/pathtracing/ptlas_cell_size")), 1.0);
		ptlas_partition_max_instances = uint32_t(MAX(int(GLOBAL_GET("rendering/pathtracing/ptlas_partition_max_instances")), 16));
		ptlas_global_max_instances = uint32_t(MAX(int(GLOBAL_GET("rendering/pathtracing/ptlas_global_max_instances")), 0));
		ptlas_stable_frames = uint32_t(MAX(int(GLOBAL_GET("rendering/pathtracing/ptlas_stable_frames")), 1));
		Vector<String> slot_rows_modes;
		slot_rows_modes.push_back("\n");
		slot_rows_shader.shader.initialize(slot_rows_modes);
		slot_rows_shader.version = slot_rows_shader.shader.version_create();
		slot_rows_shader.version_shader = slot_rows_shader.shader.version_get_shader(slot_rows_shader.version, 0);
		slot_rows_shader.pipeline = RD::get_singleton()->compute_pipeline_create(slot_rows_shader.version_shader);
		print_line(vformat("RT: slot-resident instances in a partitioned TLAS (D94b): instances written only when they change, in %.0f m cells of up to %d instances, %d in the global partition until %d frames unchanged; cells relative to their origin under a partition translation (D94c).", ptlas_cell_size, ptlas_partition_max_instances, ptlas_global_max_instances, ptlas_stable_frames));
		print_verbose(vformat("RT: slot bookkeeping (D94b): %d bytes a slot's shadow, %d a range, %d a range's view, %d a MultiMesh range's key, %d a local row.", int(sizeof(RTSlotShadow)), int(sizeof(RTSlotRange)), int(sizeof(RTSlotRangeView)), int(sizeof(RTSlotMultiMeshKey)), int(sizeof(RTSlotLocalRow))));
	}

	// The cluster LOD selection compute (D86).
	Vector<String> lod_select_modes;
	lod_select_modes.push_back("\n");
	lod_select_shader.shader.initialize(lod_select_modes);
	lod_select_shader.version = lod_select_shader.shader.version_create();
	lod_select_shader.version_shader = lod_select_shader.shader.version_get_shader(lod_select_shader.version, 0);
	lod_select_shader.pipeline = RD::get_singleton()->compute_pipeline_create(lod_select_shader.version_shader);

	// The sky's sampling table (D54): the pyramid's leaves, a level from the one below, the top
	// levels with the header (D69), and its moments per receiver normal in both forms of the radiance
	// octmap (D64, D65), clipped at multiples of the receiver's irradiance (D72).
	Vector<String> sky_light_modes;
	sky_light_modes.push_back("\n#define MODE_LEAF\n");
	sky_light_modes.push_back("\n#define MODE_REDUCE\n");
	sky_light_modes.push_back("\n#define MODE_HEADER\n");
	sky_light_modes.push_back("\n#define MODE_MOMENTS\n");
	sky_light_modes.push_back("\n#define MODE_MOMENTS\n#define USE_RADIANCE_OCTMAP_ARRAY\n");
	sky_light_shader.shader.initialize(sky_light_modes);
	sky_light_shader.version = sky_light_shader.shader.version_create();
	for (int i = 0; i < SkyLightShader::MODE_MAX; i++) {
		sky_light_shader.version_shader[i] = sky_light_shader.shader.version_get_shader(sky_light_shader.version, i);
		sky_light_shader.pipeline[i] = RD::get_singleton()->compute_pipeline_create(sky_light_shader.version_shader[i]);
	}

	// Opacity micromaps (D96).
	_omm_initialize();
}

RenderRaytracing::~RenderRaytracing() {
	if (RendererRD::MeshStorage::get_singleton()) {
		RendererRD::MeshStorage::get_singleton()->set_surface_freed_callback(nullptr);
	}
	for (KeyValue<RenderSceneBuffersRD *, RTViewportState *> &kv : viewport_states) {
		_free_viewport_state_internal(kv.value);
	}
	viewport_states.clear();

	cleanup_caches();
	singleton = nullptr;

	if (position_fetch_check_buffer.is_valid()) {
		RD::get_singleton()->free_rid(position_fetch_check_buffer);
		position_fetch_check_buffer = RID();
	}

	if (mat_ubo_pool_buffer.is_valid()) {
		RD::get_singleton()->free_rid(mat_ubo_pool_buffer);
		mat_ubo_pool_buffer = RID();
	}

	if (bindless_block) {
		memdelete(bindless_block);
		bindless_block = nullptr;
	}
	if (shader) {
		memdelete(shader);
		shader = nullptr;
	}

	mm_merge_shader.shader.version_free(mm_merge_shader.version);
	lod_select_shader.shader.version_free(lod_select_shader.version);
	if (omm_shader.version.is_valid()) {
		omm_shader.shader.version_free(omm_shader.version);
	}
	if (omm_sampler.is_valid()) {
		RD::get_singleton()->free_rid(omm_sampler);
		omm_sampler = RID();
	}
	sky_light_shader.shader.version_free(sky_light_shader.version);
	if (slot_rows_shader.version.is_valid()) {
		slot_rows_shader.shader.version_free(slot_rows_shader.version);
	}
}

// ---------------------------------------------------------------------------
// Per-viewport state lifecycle
// ---------------------------------------------------------------------------

RTViewportState *RenderRaytracing::_get_or_create_viewport_state(const RenderDataRD *p_render_data) {
	if (!p_render_data || p_render_data->render_buffers.is_null()) {
		return nullptr;
	}
	RenderSceneBuffersRD *key = p_render_data->render_buffers.ptr();
	HashMap<RenderSceneBuffersRD *, RTViewportState *>::Iterator it = viewport_states.find(key);
	if (it != viewport_states.end()) {
		return it->value;
	}
	RTViewportState *state = memnew(RTViewportState);
	viewport_states.insert(key, state);
	return state;
}

RTViewportState *RenderRaytracing::_get_viewport_state(const RenderDataRD *p_render_data) const {
	if (!p_render_data || p_render_data->render_buffers.is_null()) {
		return nullptr;
	}
	RenderSceneBuffersRD *key = p_render_data->render_buffers.ptr();
	HashMap<RenderSceneBuffersRD *, RTViewportState *>::ConstIterator it = viewport_states.find(key);
	return (it != viewport_states.end()) ? it->value : nullptr;
}

void RenderRaytracing::_free_viewport_state_internal(RTViewportState *p_state) {
	if (!p_state) {
		return;
	}
	if (ptlas_slots_enabled) {
		_release_viewport_slots(p_state);
	}
	if (p_state->tlas.is_valid()) {
		RD::get_singleton()->free_rid(p_state->tlas);
	}
	if (p_state->geometry_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->geometry_buffer);
	}
	if (p_state->material_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->material_buffer);
	}
	if (p_state->motion_index_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->motion_index_buffer);
	}
	if (p_state->motion_transform_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->motion_transform_buffer);
	}
	if (p_state->emissive_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->emissive_buffer);
	}
	if (p_state->light_tree_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->light_tree_buffer);
	}
	if (p_state->light_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->light_buffer);
	}
	if (p_state->params_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->params_buffer);
	}
	if (p_state->decal_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->decal_buffer);
	}
	if (p_state->decal_node_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->decal_node_buffer);
	}
	if (p_state->sky_light_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->sky_light_buffer);
	}
	if (p_state->scene_uniform_buffer.is_valid()) {
		RD::get_singleton()->free_rid(p_state->scene_uniform_buffer);
	}
	if (p_state->scene_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(p_state->scene_uniform_set)) {
		RD::get_singleton()->free_rid(p_state->scene_uniform_set);
	}
	memdelete(p_state);
}

void RenderRaytracing::free_viewport_state(RenderSceneBuffersRD *p_render_buffers) {
	HashMap<RenderSceneBuffersRD *, RTViewportState *>::Iterator it = viewport_states.find(p_render_buffers);
	if (it == viewport_states.end()) {
		return;
	}
	_free_viewport_state_internal(it->value);
	viewport_states.remove(it);
}

// ---------------------------------------------------------------------------
// Raytracing / DLSS-RR output textures (stored on the render buffers via named
// scopes) and the per-viewport DLSS upscaler context (in RTViewportState).
// ---------------------------------------------------------------------------

void RenderRaytracing::rt_ensure_textures(RenderSceneBuffersRD *p_render_buffers) {
	ERR_FAIL_NULL(p_render_buffers);

	uint32_t usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT |
			RD::TEXTURE_USAGE_SAMPLING_BIT |
			RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

	if (!p_render_buffers->has_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_RAYTRACING)) {
		p_render_buffers->create_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_RAYTRACING, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1);
	}
	if (!p_render_buffers->has_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_RT_DEPTH)) {
		p_render_buffers->create_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_RT_DEPTH, RD::DATA_FORMAT_R32_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1);
	}
}

bool RenderRaytracing::rt_has_texture(RenderSceneBuffersRD *p_render_buffers) const {
	return p_render_buffers && p_render_buffers->has_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_RAYTRACING);
}

RID RenderRaytracing::rt_get_texture(RenderSceneBuffersRD *p_render_buffers) const {
	ERR_FAIL_NULL_V(p_render_buffers, RID());
	return p_render_buffers->get_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_RAYTRACING);
}

bool RenderRaytracing::rt_has_depth_texture(RenderSceneBuffersRD *p_render_buffers) const {
	return p_render_buffers && p_render_buffers->has_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_RT_DEPTH);
}

RID RenderRaytracing::rt_get_depth_texture(RenderSceneBuffersRD *p_render_buffers) const {
	ERR_FAIL_NULL_V(p_render_buffers, RID());
	return p_render_buffers->get_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_RT_DEPTH);
}

static_assert(RenderRaytracing::RT_GBUFFER_TARGETS == RenderForwardClustered::RenderBufferDataForwardClustered::GBUFFER_TARGETS, "The G-buffer target count must match the render buffer data's.");

StringName RenderRaytracing::rt_gbuffer_texture_name(int p_target) {
	switch (p_target) {
		case 0:
			return SNAME("albedo_specular");
		case 1:
			return SNAME("normal_roughness");
		case 2:
			return SNAME("geo_normal_metallic");
		case 3:
			return SNAME("emission_ior");
		case 4:
			return SNAME("transmission_flags");
		case 5:
			return SNAME("absorption");
		case 6:
			return SNAME("scattering");
		default:
			ERR_FAIL_V_MSG(StringName(), vformat("No G-buffer target %d.", p_target));
	}
}

void RenderRaytracing::rt_ensure_gbuffer_textures(RenderSceneBuffersRD *p_render_buffers) {
	ERR_FAIL_NULL(p_render_buffers);

	if (p_render_buffers->has_texture(RB_SCOPE_RT_GBUFFER, rt_gbuffer_texture_name(0))) {
		return;
	}

	// The format and usage are the render buffer data's, shared with the precompiled pipelines' framebuffer format.
	for (int i = 0; i < RT_GBUFFER_TARGETS; i++) {
		p_render_buffers->create_texture(RB_SCOPE_RT_GBUFFER, rt_gbuffer_texture_name(i), RenderForwardClustered::RenderBufferDataForwardClustered::get_gbuffer_format(), RenderForwardClustered::RenderBufferDataForwardClustered::get_gbuffer_usage_bits(), RD::TEXTURE_SAMPLES_1);
	}
}

void RenderRaytracing::rt_free_gbuffer_textures(RenderSceneBuffersRD *p_render_buffers) {
	ERR_FAIL_NULL(p_render_buffers);
	p_render_buffers->clear_context(RB_SCOPE_RT_GBUFFER);
}

bool RenderRaytracing::rt_has_gbuffer_textures(RenderSceneBuffersRD *p_render_buffers) const {
	return p_render_buffers && p_render_buffers->has_texture(RB_SCOPE_RT_GBUFFER, rt_gbuffer_texture_name(0));
}

RID RenderRaytracing::rt_get_gbuffer_framebuffer(RenderSceneBuffersRD *p_render_buffers) {
	ERR_FAIL_NULL_V(p_render_buffers, RID());
	ERR_FAIL_COND_V(!rt_has_gbuffer_textures(p_render_buffers), RID());

	// The seven targets, then the velocity buffer the variant writes at location 7, on the
	// single-sample depth buffer the raygen and DLSS read.
	p_render_buffers->ensure_velocity();
	RID velocity = p_render_buffers->get_velocity_buffer(false);
	RID depth = p_render_buffers->get_depth_texture();
	return FramebufferCacheRD::get_singleton()->get_cache_multiview(1,
			p_render_buffers->get_texture(RB_SCOPE_RT_GBUFFER, rt_gbuffer_texture_name(0)),
			p_render_buffers->get_texture(RB_SCOPE_RT_GBUFFER, rt_gbuffer_texture_name(1)),
			p_render_buffers->get_texture(RB_SCOPE_RT_GBUFFER, rt_gbuffer_texture_name(2)),
			p_render_buffers->get_texture(RB_SCOPE_RT_GBUFFER, rt_gbuffer_texture_name(3)),
			p_render_buffers->get_texture(RB_SCOPE_RT_GBUFFER, rt_gbuffer_texture_name(4)),
			p_render_buffers->get_texture(RB_SCOPE_RT_GBUFFER, rt_gbuffer_texture_name(5)),
			p_render_buffers->get_texture(RB_SCOPE_RT_GBUFFER, rt_gbuffer_texture_name(6)),
			velocity, depth);
}

void RenderRaytracing::dlss_rr_ensure_buffers(RenderSceneBuffersRD *p_render_buffers) {
	ERR_FAIL_NULL(p_render_buffers);

	if (p_render_buffers->has_texture(RB_SCOPE_DLSS_RR, RB_TEX_DLSS_RR_DIFFUSE_ALBEDO)) {
		return;
	}

	uint32_t usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT |
			RD::TEXTURE_USAGE_SAMPLING_BIT |
			RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

	// Diffuse Albedo: linear RGB surface color for non-metals (RGBA16F; DLSS-RR expects
	// linear albedo, and float precision avoids 8-bit dark-value banding without gamma packing).
	p_render_buffers->create_texture(RB_SCOPE_DLSS_RR, RB_TEX_DLSS_RR_DIFFUSE_ALBEDO, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1);
	// Specular Albedo: RGB specular reflection color (RGBA16F; needs the accuracy, fixes banding artifacts).
	p_render_buffers->create_texture(RB_SCOPE_DLSS_RR, RB_TEX_DLSS_RR_SPECULAR_ALBEDO, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1);
	// Normal + Roughness: World space normals (RGB) + roughness (A).
	p_render_buffers->create_texture(RB_SCOPE_DLSS_RR, RB_TEX_DLSS_RR_NORMAL_ROUGHNESS, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1);
	// Specular Hit Distance: Single channel distance (R16F is sufficient).
	p_render_buffers->create_texture(RB_SCOPE_DLSS_RR, RB_TEX_DLSS_RR_SPECULAR_HIT_DIST, RD::DATA_FORMAT_R16_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1);
}

void RenderRaytracing::dlss_rr_free_buffers(RenderSceneBuffersRD *p_render_buffers) {
	ERR_FAIL_NULL(p_render_buffers);
	p_render_buffers->clear_context(RB_SCOPE_DLSS_RR);
}

bool RenderRaytracing::dlss_rr_has_buffers(RenderSceneBuffersRD *p_render_buffers) const {
	return p_render_buffers && p_render_buffers->has_texture(RB_SCOPE_DLSS_RR, RB_TEX_DLSS_RR_DIFFUSE_ALBEDO);
}

RID RenderRaytracing::dlss_rr_get_diffuse_albedo(RenderSceneBuffersRD *p_render_buffers) const {
	ERR_FAIL_NULL_V(p_render_buffers, RID());
	return p_render_buffers->get_texture(RB_SCOPE_DLSS_RR, RB_TEX_DLSS_RR_DIFFUSE_ALBEDO);
}

RID RenderRaytracing::dlss_rr_get_specular_albedo(RenderSceneBuffersRD *p_render_buffers) const {
	ERR_FAIL_NULL_V(p_render_buffers, RID());
	return p_render_buffers->get_texture(RB_SCOPE_DLSS_RR, RB_TEX_DLSS_RR_SPECULAR_ALBEDO);
}

RID RenderRaytracing::dlss_rr_get_normal_roughness(RenderSceneBuffersRD *p_render_buffers) const {
	ERR_FAIL_NULL_V(p_render_buffers, RID());
	return p_render_buffers->get_texture(RB_SCOPE_DLSS_RR, RB_TEX_DLSS_RR_NORMAL_ROUGHNESS);
}

RID RenderRaytracing::dlss_rr_get_specular_hit_dist(RenderSceneBuffersRD *p_render_buffers) const {
	ERR_FAIL_NULL_V(p_render_buffers, RID());
	return p_render_buffers->get_texture(RB_SCOPE_DLSS_RR, RB_TEX_DLSS_RR_SPECULAR_HIT_DIST);
}

// ---------------------------------------------------------------------------
// Material UBO sub-allocation pool
//
// Single device-address storage buffer of MAT_UBO_POOL_TOTAL_BYTES, divided
// into MAT_UBO_POOL_CAPACITY fixed-size slots. Allocate/release just bump a
// next-slot counter and a free-list. Per-material UBO writes are
// buffer_update at slot offset, no allocation. mat.uniform_address is
// pool_bda + slot * slot_size, so the closest-hit shader's
// CustomMaterialUniforms(addr) cast lands on the correct slot.
// ---------------------------------------------------------------------------

namespace {
constexpr uint32_t MAT_UBO_POOL_SLOT_SIZE = 512;
constexpr uint32_t MAT_UBO_POOL_CAPACITY = 100000;
constexpr uint64_t MAT_UBO_POOL_TOTAL_BYTES = uint64_t(MAT_UBO_POOL_SLOT_SIZE) * MAT_UBO_POOL_CAPACITY;
} // namespace

void RenderRaytracing::mat_ubo_pool_ensure_initialized() {
	if (mat_ubo_pool_buffer.is_valid()) {
		return;
	}
	Vector<uint8_t> init;
	init.resize(MAT_UBO_POOL_TOTAL_BYTES);
	memset(init.ptrw(), 0, MAT_UBO_POOL_TOTAL_BYTES);
	mat_ubo_pool_buffer = RD::get_singleton()->storage_buffer_create(
			MAT_UBO_POOL_TOTAL_BYTES, init, 0,
			RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT);
	if (mat_ubo_pool_buffer.is_valid()) {
		RD::get_singleton()->set_resource_name(mat_ubo_pool_buffer, "RT Material UBO Pool");
		mat_ubo_pool_bda = RD::get_singleton()->buffer_get_device_address(mat_ubo_pool_buffer);
	}
}

uint32_t RenderRaytracing::mat_ubo_pool_allocate() {
	mat_ubo_pool_ensure_initialized();
	if (!mat_ubo_pool_buffer.is_valid()) {
		return UINT32_MAX;
	}
	if (mat_ubo_pool_free_count > 0) {
		--mat_ubo_pool_free_count;
		return mat_ubo_pool_free_slots[mat_ubo_pool_free_count];
	}
	if (mat_ubo_pool_next_slot >= MAT_UBO_POOL_CAPACITY) {
		ERR_PRINT_ONCE("RT Material UBO Pool exhausted; falling back to dedicated buffer.");
		return UINT32_MAX;
	}
	return mat_ubo_pool_next_slot++;
}

void RenderRaytracing::mat_ubo_pool_release(uint32_t p_slot) {
	if (p_slot >= MAT_UBO_POOL_CAPACITY) {
		return;
	}
	if (mat_ubo_pool_free_count < mat_ubo_pool_free_slots.size()) {
		mat_ubo_pool_free_slots[mat_ubo_pool_free_count] = p_slot;
	} else {
		mat_ubo_pool_free_slots.push_back(p_slot);
	}
	++mat_ubo_pool_free_count;
}

void RenderRaytracing::mat_ubo_pool_update(uint32_t p_slot, const void *p_data, uint32_t p_size) {
	if (!mat_ubo_pool_buffer.is_valid() || p_slot >= MAT_UBO_POOL_CAPACITY) {
		return;
	}
	uint32_t size = MIN(p_size, MAT_UBO_POOL_SLOT_SIZE);
	RD::get_singleton()->buffer_update(mat_ubo_pool_buffer, uint64_t(p_slot) * MAT_UBO_POOL_SLOT_SIZE, size, p_data);
}

uint64_t RenderRaytracing::mat_ubo_pool_get_address(uint32_t p_slot) const {
	return mat_ubo_pool_bda + uint64_t(p_slot) * MAT_UBO_POOL_SLOT_SIZE;
}

// ---------------------------------------------------------------------------
// Cache management
// ---------------------------------------------------------------------------

void RenderRaytracing::cleanup_caches() {
	for (uint32_t i = 0; i < surface_chunks.size(); i++) {
		if (surface_chunks[i]) {
			for (uint32_t j = 0; j < RT_CACHE_CHUNK_SIZE; j++) {
				RTCacheEntry *entry = &surface_chunks[i][j];
				if (entry->ptr) {
					_release_surface_data(entry->ptr);
					memdelete(entry->ptr);
					entry->ptr = nullptr;
				}
			}
			memdelete_arr(surface_chunks[i]);
		}
	}
	surface_chunks.clear();

	// The micromaps (D96), while the cluster arena their cluster copies live in (OMMd) and the
	// merged and deformed entries whose BLASes carry them still exist: their release takes those along.
	_omm_finalize();

	RD *rd = RD::get_singleton();

	// Free all cached deformed surface data, before the LOD table below: a deformed entry's
	// region of the table (D87) is returned to the table's free ranges as it goes.
	{
		LocalVector<RID> live = deformed_pool.get_owned_list();
		for (uint32_t i = 0; i < live.size(); i++) {
			RTDeformedCacheEntry *e = deformed_pool.get_or_null(live[i]);
			if (!e) {
				continue;
			}
			if (e->ptr) {
				_release_deformed_blas(*e);
				memdelete(e->ptr);
				e->ptr = nullptr;
			}
			_release_deformed_cluster_storage(*e);
			if (e->owned_vb_full.is_valid()) {
				rd->free_rid(e->owned_vb_full);
				e->owned_vb_full = RID();
			}
			if (e->prev_pos_vb.is_valid()) {
				rd->free_rid(e->prev_pos_vb);
				e->prev_pos_vb = RID();
			}
			deformed_pool.free(live[i]);
		}
	}
	// Their templates (D107) went with the last entry holding each; any left are freed here.
	for (KeyValue<uint64_t, RTDeformedClusterTemplate> &kv : deformed_cluster_templates) {
		if (kv.value.storage.is_valid()) {
			rd->free_rid(kv.value.storage);
		}
	}
	deformed_cluster_templates.clear();
	deformed_cluster_batch.pending_templates.clear();

	// The cluster LOD instance slots and the frame's batch (D86), before the arena chunks their
	// handles depend on go; then the LOD table, whose regions went with their surfaces and the
	// deformed entries above.
	{
		LocalVector<RID> live = lod_pool.get_owned_list();
		for (uint32_t i = 0; i < live.size(); i++) {
			RTLodInstanceEntry *e = lod_pool.get_or_null(live[i]);
			if (e) {
				_release_lod_entry(*e);
			}
			lod_pool.free(live[i]);
		}
		// The cached levels (D90) went with their surfaces' release above.
		DEV_ASSERT(cluster_lod_cached_surfaces.is_empty());
	}
	if (lod_batch.dispatch_buffer.is_valid()) {
		rd->free_rid(lod_batch.dispatch_buffer);
		lod_batch.dispatch_buffer = RID();
	}
	if (lod_batch.args_buffer.is_valid()) {
		rd->free_rid(lod_batch.args_buffer);
		lod_batch.args_buffer = RID();
		lod_batch.args_capacity = 0;
	}
	if (cluster_lod_table.is_valid()) {
		rd->free_rid(cluster_lod_table);
		cluster_lod_table = RID();
	}
	cluster_lod_table_capacity = 0;
	cluster_lod_table_live_words = 0;
	cluster_lod_table_free.clear();

	// The cluster arena (D81): every region went with its surface above; a result still in
	// flight has no surface to land on.
	cluster_compaction_pending.clear();
	cluster_compaction_results.clear();
	// The gathers in flight (D93c) read only their own copies; they finish before they go.
	for (RTLodStreamGatherJob *job : cluster_lod_stream_gathers) {
		WorkerThreadPool::get_singleton()->wait_for_group_task_completion(job->group);
		memdelete(job);
	}
	cluster_lod_stream_gathers.clear();
	// The streamed groups' loads in flight (D92): their surfaces went above.
	for (const RTLodStreamLoad &load : cluster_lod_stream_loads) {
		rd->free_rid(load.staging);
		rd->free_rid(load.args);
	}
	cluster_lod_stream_loads.clear();
	cluster_lod_stream_load_results.clear();
	cluster_lod_stream_reads.clear();
	DEV_ASSERT(cluster_lod_stream_surfaces.is_empty());
	for (const RTClusterRetiredRegion &retired : cluster_arena_retired) {
		_cluster_arena_free(retired.chunk, retired.offset, retired.size);
	}
	cluster_arena_retired.clear();
	for (uint32_t i = 0; i < cluster_arena.size(); i++) {
		if (cluster_arena[i].buffer.is_valid()) {
			rd->free_rid(cluster_arena[i].buffer);
		}
	}
	if (cluster_compactions_total > 0) {
		print_verbose(vformat("RT: cluster arena released: %d compactions, %d KB staged -> %d KB compacted in all.", cluster_compactions_total, cluster_bytes_staged_total / 1024, cluster_bytes_compacted_total / 1024));
	}
	cluster_arena.clear();
	cluster_arena_reserved_bytes = 0;
	cluster_arena_live_bytes = 0;
	// The cluster geometry arena (D93b): its surfaces retired their blobs above.
	for (const RTClusterRetiredRegion &retired : cluster_geometry_arena_retired) {
		_arena_free(cluster_geometry_arena, cluster_geometry_arena_reserved_bytes, cluster_geometry_arena_live_bytes, retired.chunk, retired.offset, retired.size);
	}
	cluster_geometry_arena_retired.clear();
	for (uint32_t i = 0; i < cluster_geometry_arena.size(); i++) {
		if (cluster_geometry_arena[i].buffer.is_valid()) {
			rd->free_rid(cluster_geometry_arena[i].buffer);
		}
	}
	cluster_geometry_arena.clear();
	cluster_geometry_arena_reserved_bytes = 0;
	cluster_geometry_arena_live_bytes = 0;

	if (deformed_cluster_batch.args_buffer.is_valid()) {
		rd->free_rid(deformed_cluster_batch.args_buffer);
		deformed_cluster_batch.args_buffer = RID();
		deformed_cluster_batch.args_capacity = 0;
	}
	for (const RTDeformedClusterRegion &region : deformed_cluster_batch.regions) {
		if (region.buffer.is_valid()) {
			rd->free_rid(region.buffer);
		}
	}
	deformed_cluster_batch.regions.clear();

	// Free all merged MultiMesh BLAS data.
	{
		LocalVector<RID> live = merged_mm_pool.get_owned_list();
		for (uint32_t i = 0; i < live.size(); i++) {
			RTMergedMMEntry *e = merged_mm_pool.get_or_null(live[i]);
			if (!e) {
				continue;
			}
			if (e->blas.is_valid()) {
				rd->free_rid(e->blas);
				e->blas = RID();
			}
			if (e->merged_vtx_buffer.is_valid()) {
				rd->free_rid(e->merged_vtx_buffer);
				e->merged_vtx_buffer = RID();
			}
			if (e->merged_attr_buffer.is_valid()) {
				rd->free_rid(e->merged_attr_buffer);
				e->merged_attr_buffer = RID();
			}
			if (e->replicated_idx_buffer.is_valid()) {
				rd->free_rid(e->replicated_idx_buffer);
				e->replicated_idx_buffer = RID();
			}
			_omm_release_merged(*e);
			merged_mm_pool.free(live[i]);
		}
	}
	mm_handles.clear();
	deformed_active_this_frame.clear();
	merged_mm_active_this_frame.clear();

	// Free all cached material data
	for (uint32_t i = 0; i < material_chunks.size(); i++) {
		if (material_chunks[i]) {
			for (uint32_t j = 0; j < RT_CACHE_CHUNK_SIZE; j++) {
				RTMaterialCacheEntry *entry = &material_chunks[i][j];
				if (entry->ptr) {
					if (entry->ptr->uniform_buffer.is_valid()) {
						RD::get_singleton()->free_rid(entry->ptr->uniform_buffer);
					}
					if (entry->ptr->uniform_pool_slot != UINT32_MAX) {
						mat_ubo_pool_release(entry->ptr->uniform_pool_slot);
					}
					memdelete(entry->ptr);
					entry->ptr = nullptr;
				}
			}
			memdelete_arr(material_chunks[i]);
		}
	}
	material_chunks.clear();

	// Reset counters
	material_free_slots.clear();
	next_material_slot = 0;
	vram_used = 0;
	cache_hits = 0;
	cache_misses = 0;
}

// ---------------------------------------------------------------------------
// RID helpers
// ---------------------------------------------------------------------------

uint32_t RenderRaytracing::get_rid_index(RID p_rid) {
	return static_cast<uint32_t>(p_rid.get_id() & 0xFFFFFFFFULL);
}

uint32_t RenderRaytracing::get_rid_version(RID p_rid) {
	return static_cast<uint32_t>(p_rid.get_id() >> 32);
}

RTCacheEntry *RenderRaytracing::get_surface_cache_entry(uint32_t p_index) {
	uint32_t chunk_idx = p_index >> RT_CACHE_CHUNK_SHIFT;
	uint32_t entry_idx = p_index & RT_CACHE_CHUNK_MASK;

	// Grow vector if needed, initializing new slots to nullptr
	while (chunk_idx >= surface_chunks.size()) {
		surface_chunks.push_back(nullptr);
	}

	if (!surface_chunks[chunk_idx]) {
		surface_chunks.set(chunk_idx, memnew_arr(RTCacheEntry, RT_CACHE_CHUNK_SIZE));
		for (uint32_t i = 0; i < RT_CACHE_CHUNK_SIZE; i++) {
			surface_chunks[chunk_idx][i] = RTCacheEntry();
		}
	}

	return &surface_chunks[chunk_idx][entry_idx];
}

RTMaterialCacheEntry *RenderRaytracing::get_material_cache_entry(uint32_t p_index) {
	uint32_t chunk_idx = p_index >> RT_CACHE_CHUNK_SHIFT;
	uint32_t entry_idx = p_index & RT_CACHE_CHUNK_MASK;

	// Grow vector if needed, initializing new slots to nullptr
	while (chunk_idx >= material_chunks.size()) {
		material_chunks.push_back(nullptr);
	}

	if (!material_chunks[chunk_idx]) {
		material_chunks.set(chunk_idx, memnew_arr(RTMaterialCacheEntry, RT_CACHE_CHUNK_SIZE));
		for (uint32_t i = 0; i < RT_CACHE_CHUNK_SIZE; i++) {
			material_chunks[chunk_idx][i] = RTMaterialCacheEntry();
		}
	}

	return &material_chunks[chunk_idx][entry_idx];
}

uint32_t RenderRaytracing::allocate_material_slot() {
	if (!material_free_slots.is_empty()) {
		uint32_t slot = material_free_slots[material_free_slots.size() - 1];
		material_free_slots.resize(material_free_slots.size() - 1);
		return slot;
	}
	return next_material_slot++;
}

// ---------------------------------------------------------------------------
// Slot-pool access helpers
// ---------------------------------------------------------------------------

RTDeformedCacheEntry *RenderRaytracing::_access_deformed_slot(RID &r_handle) {
	RTDeformedCacheEntry *entry = deformed_pool.get_or_null(r_handle);
	if (!entry) {
		// Stale or never-allocated handle -> fresh slot. RID_Owner default-
		// constructs the entry, so all members start at their cleared values.
		r_handle = deformed_pool.make_rid();
		entry = deformed_pool.get_or_null(r_handle);
	}
	deformed_active_this_frame.push_back(r_handle);
	return entry;
}

// The deformed BLAS depends on the mesh's index buffer (only the vertex
// buffer is overridden), so RenderingDevice cascade-frees it with the mesh
// or its surfaces: free it only while it still exists, and forget the RID
// either way.
void RenderRaytracing::_release_deformed_blas(RTDeformedCacheEntry &p_entry) {
	if (p_entry.cluster_templated) {
		// The templates (D107) go with the layout that instantiates them; the CLAS are in the storage.
		_deformed_cluster_template_release(p_entry.cluster_template_key);
		p_entry.cluster_templated = false;
		p_entry.cluster_template_key = 0;
		p_entry.cluster_clas_address = 0;
		p_entry.cluster_clas_stride = 0;
	}
	if (!p_entry.ptr || !p_entry.ptr->blas.is_valid()) {
		return;
	}
	RD *rd = RD::get_singleton();
	if (rd->acceleration_structure_is_valid(p_entry.ptr->blas)) {
		rd->free_rid(p_entry.ptr->blas);
	}
	p_entry.ptr->blas = RID();
	p_entry.blas_built_once = false;
	p_entry.omm_serial = 0; // Its micromap (D96) was the BLAS's.
	// The cluster layout (D82) is tied to the handle's slot; the storage stays for the next layout.
	p_entry.cluster_count = 0;
	p_entry.cluster_built_frame = 0;
	p_entry.cluster_frame_kind = RTDeformedCacheEntry::CLUSTER_QUEUED_NONE;
	p_entry.cluster_infos.clear();
	p_entry.cluster_omm = false;
	_release_deformed_cluster_lod(p_entry);
}

RTMergedMMEntry *RenderRaytracing::_access_merged_mm_slot(RID &r_handle) {
	RTMergedMMEntry *entry = merged_mm_pool.get_or_null(r_handle);
	if (!entry) {
		r_handle = merged_mm_pool.make_rid();
		entry = merged_mm_pool.get_or_null(r_handle);
	}
	merged_mm_active_this_frame.push_back(r_handle);
	return entry;
}

// ---------------------------------------------------------------------------
// Per-frame preparation
// ---------------------------------------------------------------------------

void RenderRaytracing::prepare_frame() {
	// The build budget's frame (D97): its stand-ins below spend what the last frame left.
	_build_budget_begin_frame();
	// Sizes read back since the last frame move their CLAS into the arena before the TLAS
	// instances of this frame take each surface's handle (D81).
	_process_cluster_compactions();
	// Opacity micromaps (D96): bakes whose setup is back, and the setups of surfaces asked for since.
	_omm_process();
	// Then the streamed groups (D92): loads whose sizes are back become resident, and the requests
	// and uses read back since admit, evict and start loads, all before this frame's selection.
	if (cluster_lod_stream_enabled) {
		_process_lod_stream();
	}
	// The position-fetch check's counts (D93a), read and cleared on the verbose report's cadence,
	// so each report covers the frames since the last and no counter wraps.
	if (position_fetch_check_buffer.is_valid() && RD::get_singleton()->get_frames_drawn() % 300 == 0) {
		RD::get_singleton()->buffer_get_data_async(position_fetch_check_buffer, callable_mp_static(&RenderRaytracing::_on_position_fetch_check_read));
		RD::get_singleton()->buffer_clear(position_fetch_check_buffer, 0, sizeof(uint32_t) * RT_POSITION_FETCH_CHECK_WORDS);
	}

	blass.clear();
	blas_transforms.clear();
	instance_flags.clear();
	instance_masks.clear();
	sbt_offsets.clear();
	geometry_data.clear();
	material_data.clear();
	motion_indices.clear();
	motion_transforms.clear();
	frame_slots.clear();
	frame_geometry_generations.clear();
	frame_material_generations.clear();
	deformed_active_this_frame.clear();
	merged_mm_active_this_frame.clear();

	const uint32_t current_frame = RSG::rasterizer->get_frame_number();
	RD *rd = RD::get_singleton();

	// TTL-evict stale deformed-surface entries.
	{
		static const uint32_t DEFORMED_CACHE_TTL = (uint32_t)GLOBAL_GET("rendering/pathtracing/deformed_mesh_cache_ttl_frames");
		LocalVector<RID> live = deformed_pool.get_owned_list();
		for (uint32_t i = 0; i < live.size(); i++) {
			RTDeformedCacheEntry *e = deformed_pool.get_or_null(live[i]);
			if (!e) {
				continue;
			}
			if (e->last_used_frame == 0 || current_frame - e->last_used_frame <= DEFORMED_CACHE_TTL) {
				continue;
			}
			if (e->ptr) {
				_release_deformed_blas(*e);
				memdelete(e->ptr);
				e->ptr = nullptr;
			}
			_release_deformed_cluster_storage(*e);
			if (e->owned_vb_full.is_valid()) {
				rd->free_rid(e->owned_vb_full);
				e->owned_vb_full = RID();
			}
			if (e->prev_pos_vb.is_valid()) {
				rd->free_rid(e->prev_pos_vb);
				e->prev_pos_vb = RID();
			}
			deformed_pool.free(live[i]);
		}
	}

	// TTL-evict stale cluster LOD instance slots (D86), on the deformed entries' TTL.
	{
		static const uint32_t LOD_SLOT_TTL = (uint32_t)GLOBAL_GET("rendering/pathtracing/deformed_mesh_cache_ttl_frames");
		LocalVector<RID> live = lod_pool.get_owned_list();
		for (uint32_t i = 0; i < live.size(); i++) {
			RTLodInstanceEntry *e = lod_pool.get_or_null(live[i]);
			if (!e) {
				continue;
			}
			if (e->last_used_frame == 0 || current_frame - e->last_used_frame <= LOD_SLOT_TTL) {
				// An instance that has traced another's build for as long (D89) gives its slot back.
				if (e->blas.is_valid() && current_frame - e->last_built_frame > LOD_SLOT_TTL) {
					_release_lod_entry(*e);
				}
				continue;
			}
			_release_lod_entry(*e);
			lod_pool.free(live[i]);
		}
		// The cached levels (D90) no instance has traced for as long.
		LocalVector<RTSurfaceData *> emptied;
		for (RTSurfaceData *surface : cluster_lod_cached_surfaces) {
			bool any_left = false;
			for (RTLodCachedLevel &level : surface->cluster_lod_cache) {
				if (level.blas.is_valid() && current_frame - level.last_used_frame > LOD_SLOT_TTL) {
					_release_lod_cached_level(level);
				}
				any_left = any_left || level.blas.is_valid();
			}
			if (!any_left) {
				emptied.push_back(surface);
			}
		}
		for (RTSurfaceData *surface : emptied) {
			surface->cluster_lod_cache.clear();
			cluster_lod_cached_surfaces.erase(surface);
		}
	}

	// TTL-evict stale merged-MultiMesh entries.
	{
		static const uint32_t MM_BLAS_CACHE_TTL = (uint32_t)GLOBAL_GET("rendering/pathtracing/multimesh_blas_cache_ttl_frames");
		LocalVector<RID> live = merged_mm_pool.get_owned_list();
		for (uint32_t i = 0; i < live.size(); i++) {
			RTMergedMMEntry *e = merged_mm_pool.get_or_null(live[i]);
			if (!e) {
				continue;
			}
			if (e->last_used_frame == 0 || current_frame - e->last_used_frame <= MM_BLAS_CACHE_TTL) {
				continue;
			}
			if (e->blas.is_valid()) {
				rd->free_rid(e->blas);
				e->blas = RID();
			}
			if (e->merged_vtx_buffer.is_valid()) {
				rd->free_rid(e->merged_vtx_buffer);
				e->merged_vtx_buffer = RID();
			}
			if (e->merged_attr_buffer.is_valid()) {
				rd->free_rid(e->merged_attr_buffer);
				e->merged_attr_buffer = RID();
			}
			if (e->replicated_idx_buffer.is_valid()) {
				rd->free_rid(e->replicated_idx_buffer);
				e->replicated_idx_buffer = RID();
			}
			_omm_release_merged(*e);
			merged_mm_pool.free(live[i]);
		}
	}

	// Finish async HG compiles so live_ready_mask matches this frame (sync path fills at build_tlas end).
	SceneShaderRaytracing::get_singleton()->drain_completed_compiles();

	// Reset per-frame metrics
	cache_hits = 0;
	cache_misses = 0;

	if (!bindless_block->is_initialized()) {
		bindless_block->initialize(RD::get_singleton());
	}
	bindless_block->begin_frame();
}

// ---------------------------------------------------------------------------
// The acceleration-structure build budget (D97)
// ---------------------------------------------------------------------------

// A new surface's cost before it is populated: its triangles at full detail (a hierarchy's coarser
// levels and a streamed surface's fallback cut are not counted apart).
static uint64_t _build_budget_surface_cost(void *p_mesh_surface) {
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();
	const uint32_t index_count = mesh_storage->mesh_surface_get_rt_index_count(p_mesh_surface);
	const uint32_t vertex_count = mesh_storage->mesh_surface_get_rt_vertex_count(p_mesh_surface);
	return MAX(uint64_t(index_count > 0 ? index_count / 3 : vertex_count / 3), uint64_t(1));
}

// The surfaces a walk traced (D97): one bit a surface index, the frame's and the frame before's
// kept apart, so a stamp written this frame never counts as traced the frame before. Surface
// indices 63 and above share bit 63, which errs toward forcing: such a surface is held back only
// when no surface of its instance from index 63 up was traced the frame before.
static _FORCE_INLINE_ uint64_t _build_budget_surface_bit(uint32_t p_surface_index) {
	return uint64_t(1) << MIN(p_surface_index, 63u);
}

void RenderRaytracing::_build_budget_stamp(const void *p_instance, uint32_t p_surface_index, uint64_t p_frames_drawn) {
	const RenderForwardClustered::GeometryInstanceForwardClustered *p_inst = static_cast<const RenderForwardClustered::GeometryInstanceForwardClustered *>(p_instance);
	if (p_inst->rt_traced_frame != p_frames_drawn) {
		p_inst->rt_traced_prev_mask = (p_inst->rt_traced_frame != UINT64_MAX && p_inst->rt_traced_frame + 1 == p_frames_drawn) ? p_inst->rt_traced_mask : 0;
		p_inst->rt_traced_mask = 0;
		p_inst->rt_traced_frame = p_frames_drawn;
	}
	p_inst->rt_traced_mask |= _build_budget_surface_bit(p_surface_index);
}

bool RenderRaytracing::_build_budget_traced(const void *p_instance, uint32_t p_surface_index, uint64_t p_frames_drawn) {
	const RenderForwardClustered::GeometryInstanceForwardClustered *p_inst = static_cast<const RenderForwardClustered::GeometryInstanceForwardClustered *>(p_instance);
	uint64_t previous = 0;
	if (p_inst->rt_traced_frame == p_frames_drawn) {
		previous = p_inst->rt_traced_prev_mask;
	} else if (p_inst->rt_traced_frame != UINT64_MAX && p_inst->rt_traced_frame + 1 == p_frames_drawn) {
		previous = p_inst->rt_traced_mask;
	}
	return (previous & _build_budget_surface_bit(p_surface_index)) != 0;
}

// An instance's projected size: its bounding sphere's radius over its distance, at most 1.
static float _build_budget_priority(const AABB &p_aabb, const Vector3 &p_camera) {
	const real_t radius = p_aabb.size.length() * 0.5;
	if (!(radius > 0.0)) {
		return 0.0f;
	}
	const real_t distance = p_camera.distance_to(p_aabb.get_center());
	return float(radius / MAX(distance, radius));
}

// Once a frame, from the first prepare_frame: what the last frame left is what prepare_frame's
// stand-ins may spend, and the frame's own budget starts whole, so no frame admits more than
// the budget, the items forced past it aside (one overdue item a queue, the pre-pass's first, an
// item larger than the whole budget in a frame nothing else spent).
void RenderRaytracing::_build_budget_begin_frame() {
	BuildBudget &budget = build_budget;
	if (!budget.enabled) {
		return;
	}
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	if (budget.frame == frames_drawn) {
		return;
	}
	const bool first = budget.frame == UINT64_MAX;
	budget.carry_triangles = first ? budget.triangles_per_frame : budget.left_triangles;
	budget.carry_builds = first ? budget.builds_per_frame : budget.left_builds;
	budget.left_triangles = budget.triangles_per_frame;
	budget.left_builds = budget.builds_per_frame;
	budget.frame = frames_drawn;
	budget.walk_overdue_taken = false;
	budget.compaction_overdue_taken = false;
	budget.spent_triangles = 0;
	budget.spent_builds = 0;
	budget.built = 0;
	budget.deferred = 0;
	budget.waiting = 0;
	budget.held = 0;
	if (frames_drawn % 300 == 0 && OS::get_singleton()->is_stdout_verbose()) {
		uint32_t wait_p50 = 0, wait_max = 0;
		if (!budget.waits.is_empty()) {
			budget.waits.sort();
			wait_p50 = budget.waits[budget.waits.size() / 2];
			wait_max = budget.waits[budget.waits.size() - 1];
		}
		print_verbose(vformat("RT: build budget (D97): %d triangles and %d builds a frame; over the run %d admitted and %d refusals; since the last report %d items admitted after a wait, waited p50 %d, max %d frames (first in line at %d), %d of them merges, which waited %d frames at most; %d merges waiting now; the pre-pass took %d us at most in a frame since the last report, %d us over the run.", budget.triangles_per_frame, budget.builds_per_frame, budget.admitted_total, budget.deferred_total, budget.waits.size(), wait_p50, wait_max, budget.max_wait_frames, budget.merges_admitted_after_wait, budget.merge_wait_max, budget.merge_waits.size(), budget.prepass_usec_max, budget.prepass_usec_total));
		budget.waits.clear();
		budget.prepass_usec_max = 0;
		budget.merges_admitted_after_wait = 0;
		budget.merge_wait_max = 0;
	}
}

// Whether an item fits the last frame's remainder (p_carry, prepare_frame's stand-ins) or this
// frame's, with no side effect. An untouched pool takes anything: an item larger than the whole
// budget goes alone in a frame nothing else spent, so no queue holds such an item back for good.
bool RenderRaytracing::_build_budget_fits(uint64_t p_cost, bool p_carry) const {
	const BuildBudget &budget = build_budget;
	const uint64_t triangles = p_carry ? budget.carry_triangles : budget.left_triangles;
	const uint32_t builds = p_carry ? budget.carry_builds : budget.left_builds;
	return builds > 0 && (p_cost <= triangles || triangles == budget.triangles_per_frame);
}

void RenderRaytracing::_build_budget_spend(uint64_t p_cost, bool p_carry) {
	BuildBudget &budget = build_budget;
	if (p_carry) {
		budget.carry_triangles -= MIN(p_cost, budget.carry_triangles);
		budget.carry_builds -= MIN(1u, budget.carry_builds);
	}
	budget.left_triangles -= MIN(p_cost, budget.left_triangles);
	budget.left_builds -= MIN(1u, budget.left_builds);
	budget.spent_triangles += p_cost;
	budget.spent_builds++;
	budget.admitted_total++;
}

// One item's admission, one build and p_cost triangles: within the limits, or past them when
// p_force (an overdue item, the pre-pass's first), which the result says so the caller spends its
// one overdue slot only when it was used. A refusal is counted here.
RenderRaytracing::BuildBudgetTake RenderRaytracing::_build_budget_take(uint64_t p_cost, bool p_carry, bool p_force) {
	BuildBudget &budget = build_budget;
	if (!budget.enabled) {
		return BUILD_BUDGET_ADMITTED;
	}
	const bool fits = _build_budget_fits(p_cost, p_carry);
	if (!fits && !p_force) {
		budget.deferred++;
		budget.deferred_total++;
		return BUILD_BUDGET_REFUSED;
	}
	_build_budget_spend(p_cost, p_carry);
	return fits ? BUILD_BUDGET_ADMITTED : BUILD_BUDGET_ADMITTED_PAST;
}

// An item the walk or the selection's flush decides (a new surface it did not see in the pre-pass,
// a static micromap BLAS, a cached level), from this frame's remainder. Its wait is kept on the
// item: it carries over only from a refusal the frame before, and once it reaches the max wait the
// item is forced, one such item past the limits a frame.
bool RenderRaytracing::_build_budget_take_walk(uint64_t p_cost, uint64_t &r_budget_frame, uint64_t &r_wait_since) {
	BuildBudget &budget = build_budget;
	if (!budget.enabled) {
		return true;
	}
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	if (r_budget_frame == UINT64_MAX || r_budget_frame + 1 < frames_drawn) {
		r_wait_since = 0;
	}
	const uint64_t since = r_wait_since;
	const bool overdue = since != 0 && frames_drawn - since >= budget.max_wait_frames;
	const BuildBudgetTake take = _build_budget_take(p_cost, false, overdue && !budget.walk_overdue_taken);
	budget.walk_overdue_taken = budget.walk_overdue_taken || take == BUILD_BUDGET_ADMITTED_PAST;
	r_budget_frame = frames_drawn;
	if (take == BUILD_BUDGET_REFUSED) {
		r_wait_since = since != 0 ? since : frames_drawn;
		return false;
	}
	_build_budget_note_wait(since);
	r_wait_since = 0;
	return true;
}

void RenderRaytracing::_build_budget_note_wait(uint64_t p_since) {
	if (p_since != 0 && build_budget.waits.size() < RT_BUILD_BUDGET_WAITS_CAP) {
		const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
		build_budget.waits.push_back(uint32_t(MIN(frames_drawn - p_since, uint64_t(UINT32_MAX))));
	}
}

// A surface the walk will populate unless it waits: one whose cache entry misses and holds no BLAS
// the trace already takes (a rebuild under one is forced, as is any miss of a surface its instance
// traced the frame before). Its wait carries over only from a decision made the frame before.
void RenderRaytracing::_build_budget_consider_surface(RID p_mesh, uint32_t p_surface_index, void *p_mesh_surface, float p_priority, bool p_traced) {
	if (p_traced) {
		return; // Its instance traced it the frame before: whatever it misses is forced.
	}
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();
	BuildBudget &budget = build_budget;
	const uint32_t cache_key = (p_mesh.get_local_index() << 8) | (p_surface_index & 0xFF);
	const uint32_t mesh_version = get_rid_version(p_mesh);
	RTCacheEntry *entry = get_surface_cache_entry(cache_key);
	const bool needs_refresh = !entry->ptr || entry->cached_rid_version != mesh_version || entry->cached_counter != mesh_storage->mesh_surface_get_rt_invalidation_counter(p_mesh_surface);
	if (!needs_refresh && entry->ptr->blas.is_valid()) {
		return;
	}
	if (entry->ptr && entry->cached_rid_version == mesh_version && entry->ptr->blas.is_valid()) {
		return; // Forced: the trace takes its BLAS, which no longer matches its buffers.
	}
	uint32_t *found = budget.candidate_surfaces.getptr(entry);
	if (found != nullptr) {
		budget.candidates[*found].priority = MAX(budget.candidates[*found].priority, p_priority);
		return;
	}
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	if (entry->budget_frame == UINT64_MAX || entry->budget_frame + 1 < frames_drawn) {
		entry->budget_wait_since = 0;
	}
	BuildBudget::Candidate candidate;
	candidate.entry = entry;
	candidate.cost = _build_budget_surface_cost(p_mesh_surface);
	candidate.wait_since = entry->budget_wait_since;
	candidate.priority = p_priority;
	candidate.order = budget.candidates.size();
	candidate.kind = BuildBudget::CANDIDATE_SURFACE;
	budget.candidate_surfaces.insert(entry, budget.candidates.size());
	budget.candidates.push_back(candidate);
}

// Once a frame, before the first walk: the new surfaces and merges' first bakes the walk would
// build, admitted in order (overdue first, oldest first; surfaces before merges; by projected
// size; in walk order). It populates nothing: a surface that waits is stamped, and the walk's
// process_surface returns no surface for it, so the walk drops the instance where it drops a
// failed population, before its slot, material row, micromap request or selection candidate. With
// everything admitted the walk is the walk without the budget. A surface its instance traced the
// frame before is never listed: whatever it misses (a swapped mesh, a rebuilt part) is forced.
void RenderRaytracing::_build_budget_prepass(const RenderDataRD *p_render_data, uint32_t p_rt_flags) {
	BuildBudget &budget = build_budget;
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	if (!budget.enabled || budget.prepass_frame == frames_drawn) {
		return;
	}
	budget.prepass_frame = frames_drawn;
	const uint64_t start_usec = OS::get_singleton()->get_ticks_usec();
	budget.candidates.clear();
	budget.candidate_surfaces.clear();
	budget.candidate_merges.clear();
	budget.merges_waiting.clear();

	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();
	SceneShaderRaytracing *rt_shader = SceneShaderRaytracing::get_singleton();
	const Vector3 camera = p_render_data->scene_data->cam_transform.origin;
	const PagedArray<RenderGeometryInstance *> &rt_instances = *p_render_data->rt_instances;
	for (uint32_t i = 0; i < (uint32_t)rt_instances.size(); i++) {
		const RenderForwardClustered::GeometryInstanceForwardClustered *inst =
				static_cast<const RenderForwardClustered::GeometryInstanceForwardClustered *>(rt_instances[i]);
		if (!inst || !inst->data || inst->rt_procedural) {
			continue; // A procedural AABB BLAS builds when it is dirty, as before.
		}
		const float priority = _build_budget_priority(inst->transformed_aabb, camera);
		const RenderForwardClustered::GeometryInstanceSurfaceDataCache *surf = inst->surface_caches;
		if (inst->data->base_type == RSE::INSTANCE_MULTIMESH) {
			const RID mm_rid = inst->data->base;
			if (mesh_storage->multimesh_get_transform_format(mm_rid) != RSE::MULTIMESH_TRANSFORM_3D) {
				continue;
			}
			const uint32_t mm_count = mesh_storage->multimesh_get_instances_to_draw(mm_rid);
			if (mm_count == 0) {
				continue;
			}
			const RID mm_gpu_buffer = mesh_storage->multimesh_get_gpu_buffer(mm_rid);
			const RID mesh = mesh_storage->multimesh_get_mesh(mm_rid);
			for (; surf; surf = surf->next) {
				if (surf->rt_pass_flags & RenderForwardClustered::GeometryInstanceSurfaceDataCache::FLAG_PASS_ALPHA) {
					continue;
				}
				// The walk skips a surface whose hit group still compiles before anything is built;
				// read without side effects from the material cache, an uncached material left to the
				// walk, which decides such a surface from what the frame has left.
				RID material_rid;
				if (inst->data->material_override.is_valid()) {
					material_rid = inst->data->material_override;
				} else if (surf->surface_index < inst->data->surface_materials.size() && inst->data->surface_materials[surf->surface_index].is_valid()) {
					material_rid = inst->data->surface_materials[surf->surface_index];
				} else if (mesh.is_valid() && mesh_storage->owns_mesh(mesh)) {
					material_rid = mesh_storage->mesh_surface_get_material(mesh, surf->surface_index);
				}
				if (material_rid.is_valid()) {
					const RTMaterialCacheEntry *material = get_material_cache_entry(get_rid_index(material_rid));
					if (!material->ptr || material->cached_rid_version != get_rid_version(material_rid) || material->cached_counter != material_storage->material_get_rt_invalidation_counter(material_rid)) {
						continue;
					}
					if (material->ptr->rt_sbt_offset > 0 && !rt_shader->is_hg_ready_in_bundle(material->ptr->rt_sbt_offset, p_rt_flags)) {
						continue;
					}
				}
				void *mesh_surface = surf->surface;
				bool bakes = false;
				bool first = false;
				const bool merges = _plan_merged_mm(mm_rid, mm_gpu_buffer, mesh_surface, mm_count, surf->surface_index, mesh_storage->mesh_surface_get_rt_invalidation_counter(mesh_surface), bakes, &first);
				if (merges && bakes && first) {
					// A merge with no entry yet: its expanded instances stand in while it waits. A
					// rebake of an existing entry is forced.
					const uint64_t key = (uint64_t(mm_rid.get_local_index()) << 32) | surf->surface_index;
					uint32_t *found = budget.candidate_merges.getptr(key);
					if (found != nullptr) {
						budget.candidates[*found].priority = MAX(budget.candidates[*found].priority, priority);
					} else {
						const BuildBudget::MergeWait *wait = budget.merge_waits.getptr(key);
						BuildBudget::Candidate candidate;
						candidate.merge_key = key;
						candidate.merge_id = mm_rid.get_id();
						candidate.cost = uint64_t(mm_count) * _build_budget_surface_cost(mesh_surface);
						candidate.wait_since = (wait != nullptr && wait->mm_id == candidate.merge_id && wait->seen + 1 == frames_drawn) ? wait->since : 0;
						candidate.priority = priority;
						candidate.order = budget.candidates.size();
						candidate.kind = BuildBudget::CANDIDATE_MERGE;
						budget.candidate_merges.insert(key, budget.candidates.size());
						budget.candidates.push_back(candidate);
					}
				}
				// The expanded path populates the surface, and so does a streamed surface's merge (D93d).
				if ((!merges || mesh_storage->mesh_surface_is_geometry_streamed(mesh_surface)) && mesh.is_valid()) {
					_build_budget_consider_surface(mesh, surf->surface_index, mesh_surface, priority, _build_budget_traced(inst, surf->surface_index, frames_drawn));
				}
			}
			continue;
		}
		for (; surf; surf = surf->next) {
			if (surf->rt_pass_flags & RenderForwardClustered::GeometryInstanceSurfaceDataCache::FLAG_PASS_ALPHA) {
				continue;
			}
			if (inst->mesh_instance.is_valid() && mesh_storage->mesh_instance_get_vertex_buffer(inst->mesh_instance, surf->surface_index).is_valid()) {
				continue; // Deformed: forced.
			}
			_build_budget_consider_surface(inst->data->base, surf->surface_index, surf->surface, priority, _build_budget_traced(inst, surf->surface_index, frames_drawn));
		}
	}

	for (BuildBudget::Candidate &candidate : budget.candidates) {
		candidate.overdue = candidate.wait_since != 0 && frames_drawn - candidate.wait_since >= budget.max_wait_frames;
	}
	budget.candidates.sort();
	bool admitted_any = false;
	for (const BuildBudget::Candidate &candidate : budget.candidates) {
		// The frame's first item goes whatever its size, so one larger than the budget still
		// progresses, and an overdue one is first in line: past the limits one at a time, so a
		// burst that waited is never let through at once.
		const bool admitted = _build_budget_take(candidate.cost, false, !admitted_any) != BUILD_BUDGET_REFUSED;
		admitted_any = admitted_any || admitted;
		if (admitted) {
			_build_budget_note_wait(candidate.wait_since);
			if (candidate.kind == BuildBudget::CANDIDATE_MERGE && candidate.wait_since != 0) {
				budget.merges_admitted_after_wait++;
				budget.merge_wait_max = MAX(budget.merge_wait_max, uint32_t(MIN(frames_drawn - candidate.wait_since, uint64_t(UINT32_MAX))));
			}
		} else {
			budget.waiting++;
		}
		const uint64_t since = admitted ? 0 : (candidate.wait_since != 0 ? candidate.wait_since : frames_drawn);
		if (candidate.kind == BuildBudget::CANDIDATE_SURFACE) {
			candidate.entry->budget_frame = frames_drawn;
			candidate.entry->budget_admitted = admitted;
			candidate.entry->budget_wait_since = since;
			continue;
		}
		budget.merges_waiting.insert(candidate.merge_key, !admitted);
		if (admitted) {
			budget.merge_waits.erase(candidate.merge_key);
		} else {
			BuildBudget::MergeWait wait;
			wait.mm_id = candidate.merge_id;
			wait.since = since;
			wait.seen = frames_drawn;
			budget.merge_waits.insert(candidate.merge_key, wait);
		}
	}
	// A merge no longer asked for stops waiting.
	budget.stale_merges.clear();
	for (const KeyValue<uint64_t, BuildBudget::MergeWait> &E : budget.merge_waits) {
		if (E.value.seen != frames_drawn) {
			budget.stale_merges.push_back(E.key);
		}
	}
	for (const uint64_t key : budget.stale_merges) {
		budget.merge_waits.erase(key);
	}

	const uint64_t usec = OS::get_singleton()->get_ticks_usec() - start_usec;
	budget.prepass_usec_total += usec;
	budget.prepass_usec_max = MAX(budget.prepass_usec_max, usec);
}

// A surface the walk populates: as the pre-pass decided this frame, or, one it did not see (a
// later viewport's, a waiting merge's expanded instances, a MultiMesh whose material was not cached
// yet), from what the frame has left, in walk order.
bool RenderRaytracing::_build_budget_admit_surface(RTCacheEntry *p_entry, void *p_mesh_surface) {
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	if (p_entry->budget_frame == frames_drawn) {
		return p_entry->budget_admitted;
	}
	p_entry->budget_admitted = _build_budget_take_walk(_build_budget_surface_cost(p_mesh_surface), p_entry->budget_frame, p_entry->budget_wait_since);
	if (!p_entry->budget_admitted) {
		build_budget.waiting++;
	}
	return p_entry->budget_admitted;
}

// ---------------------------------------------------------------------------
// Surface processing
// ---------------------------------------------------------------------------

RTSurfaceData *RenderRaytracing::process_surface(
		const void *p_surf,
		void *p_mesh_surface,
		uint32_t p_surface_invalidation_counter,
		const Transform3D &p_transform,
		LocalVector<RID> &r_dirty_blas_list) {
	const RenderForwardClustered::GeometryInstanceSurfaceDataCache *surf =
			static_cast<const RenderForwardClustered::GeometryInstanceSurfaceDataCache *>(p_surf);

	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();

	RID mesh_rid = surf->owner->data->base;
	if (surf->owner->data->base_type == RSE::INSTANCE_MULTIMESH) {
		RID underlying = mesh_storage->multimesh_get_mesh(mesh_rid);
		if (underlying.is_valid()) {
			mesh_rid = underlying;
		}
	}

	uint32_t cache_key = (mesh_rid.get_local_index() << 8) | (surf->surface_index & 0xFF);
	uint32_t mesh_version = get_rid_version(mesh_rid);
	RTCacheEntry *entry = get_surface_cache_entry(cache_key);

	uint32_t current_frame = RSG::rasterizer->get_frame_number();
	bool needs_refresh = !entry->ptr ||
			entry->cached_rid_version != mesh_version ||
			entry->cached_counter != p_surface_invalidation_counter;

	if (!needs_refresh && entry->ptr->blas.is_valid()) {
		cache_hits++;
		entry->last_used_frame = current_frame;
		return entry->ptr;
	}

	cache_misses++;

	// The build budget (D97): a new surface waits until it is admitted, with no surface for the walk
	// to trace (its entry may still hold an earlier mesh's freed BLAS). A rebuild under a BLAS the
	// trace takes is forced, as is any miss of a surface its instance traced the frame before (its
	// mesh swapped, a part rebuilt, a merge grown past its cap and expanded): it never leaves the
	// trace for the budget.
	const bool budgeted = build_budget.enabled && !(entry->ptr && entry->cached_rid_version == mesh_version && entry->ptr->blas.is_valid()) && !_build_budget_traced(surf->owner, surf->surface_index, RD::get_singleton()->get_frames_drawn());
	if (budgeted && !_build_budget_admit_surface(entry, p_mesh_surface)) {
		return nullptr;
	}

	// Allocate or reuse entry
	if (!entry->ptr) {
		entry->ptr = memnew(RTSurfaceData);
	} else {
		// Surface data changed on a live mesh: the BLAS is still there, free it.
		// A deleted mesh or cleared surface cascade-freed a triangle BLAS through its buffers;
		// a cluster BLAS and its storage wait here for the slot's reuse.
		_release_surface_data(entry->ptr);
	}

	RTSurfaceData *surf_data = entry->ptr;

	_populate_surface_blas(p_mesh_surface, RID(), false, false, false, cache_key, surf_data, r_dirty_blas_list);
	if (budgeted && surf_data->blas.is_valid()) {
		build_budget.built++;
	}

	surf->cached_final_transform_valid = false;

	if (!surf_data->blas.is_valid()) {
		return surf_data;
	}

	entry->cached_counter = p_surface_invalidation_counter;
	entry->cached_rid_version = mesh_version;
	entry->last_used_frame = current_frame;

	return surf_data;
}

// ---------------------------------------------------------------------------
// Deformed surface processing
// ---------------------------------------------------------------------------

RTSurfaceData *RenderRaytracing::process_deformed_surface(
		const void *p_surf,
		void *p_mesh_surface,
		const RTDeformedGeometrySource &p_source,
		LocalVector<RID> &r_dirty_blas_list,
		LocalVector<RID> &r_dirty_blas_update_list,
		bool p_lod_select) {
	const RenderForwardClustered::GeometryInstanceSurfaceDataCache *surf =
			static_cast<const RenderForwardClustered::GeometryInstanceSurfaceDataCache *>(p_surf);

	if (!p_source.current_vb.is_valid()) {
		return nullptr;
	}

	RD *rd = RD::get_singleton();
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();

	// Compute the deformed VB layout.
	uint64_t fmt = mesh_storage->mesh_surface_get_format(p_mesh_surface);
	uint32_t vertex_count = mesh_storage->mesh_surface_get_vertex_count(p_mesh_surface);
	bool is_2d = fmt & RSE::ARRAY_FLAG_USE_2D_VERTICES;
	bool has_n = fmt & RSE::ARRAY_FORMAT_NORMAL;
	bool has_t = fmt & RSE::ARRAY_FORMAT_TANGENT;
	uint32_t position_stride = is_2d ? (uint32_t)sizeof(float) * 2u : (uint32_t)sizeof(float) * 3u;
	uint32_t tbn_section_per_vertex = has_n ? (has_t ? 8u : 4u) : 0u;
	uint32_t per_vertex_bytes = position_stride + tbn_section_per_vertex;
	uint32_t full_size = per_vertex_bytes * vertex_count;
	uint32_t pos_size = position_stride * vertex_count;

	if (vertex_count == 0 || full_size == 0) {
		return nullptr;
	}

	uint32_t current_frame = RSG::rasterizer->get_frame_number();
	uint64_t buffer_id = p_source.current_vb.get_id();

	RTDeformedCacheEntry *entry_ptr = _access_deformed_slot(surf->rt_deformed_handle);
	ERR_FAIL_NULL_V(entry_ptr, nullptr);
	RTDeformedCacheEntry &entry = *entry_ptr;
	entry.cluster_lod_pending = false; // Decided below for this frame; taken at the TLAS push (D87).
	entry.cluster_lod_pending_kind = RTDeformedCacheEntry::CLUSTER_QUEUED_NONE;

	bool layout_changed = entry.cached_vertex_count != vertex_count || entry.cached_full_size != full_size;
	bool data_changed = layout_changed ||
			entry.cached_change_stamp != p_source.change_stamp ||
			entry.cached_buffer_id != buffer_id;
	bool needs_full_rebuild = !entry.ptr || !entry.blas_built_once || layout_changed ||
			entry.cached_mesh_surface != p_mesh_surface || // A kept handle on another surface (D110).
			entry.cached_key_version != p_source.cache_version ||
			entry.cached_surface_counter != p_source.surface_counter ||
			(entry.cluster_count > 0 && !cluster_deformed_enabled); // The cluster path went off (D82): a triangle BLAS from here.

	BitField<RD::BufferCreationBits> owned_full_flags = RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT |
			RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT;
	BitField<RD::BufferCreationBits> prev_flags = RD::BUFFER_CREATION_AS_STORAGE_BIT |
			RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT;

	// Reallocate owned storage if the surface grew.
	if (entry.owned_vb_full_capacity < full_size) {
		_release_deformed_blas(entry);
		if (entry.owned_vb_full.is_valid()) {
			rd->free_rid(entry.owned_vb_full);
		}
		entry.owned_vb_full = rd->vertex_buffer_create(full_size, Span<uint8_t>(), owned_full_flags);
		ERR_FAIL_COND_V(!entry.owned_vb_full.is_valid(), nullptr);
		entry.owned_vb_full_capacity = full_size;
		rd->set_resource_name(entry.owned_vb_full, "RT deformed owned VB [" + itos(p_source.cache_key) + "]");
		entry.prev_pos_seeded = false;
		needs_full_rebuild = true;
	}
	if (entry.prev_pos_vb_capacity < pos_size) {
		if (entry.prev_pos_vb.is_valid()) {
			rd->free_rid(entry.prev_pos_vb);
		}
		entry.prev_pos_vb = rd->storage_buffer_create(pos_size, Vector<uint8_t>(), 0, prev_flags);
		ERR_FAIL_COND_V(!entry.prev_pos_vb.is_valid(), nullptr);
		entry.prev_pos_vb_capacity = pos_size;
		rd->set_resource_name(entry.prev_pos_vb, "RT deformed prev pos VB [" + itos(p_source.cache_key) + "]");
		entry.prev_pos_seeded = false;
	}

	// A kept handle on another surface or mesh instance (D110) holds positions that are not this
	// surface's: the previous positions are seeded again, so motion vectors start at zero.
	if (entry.cached_mesh_surface != p_mesh_surface || entry.cached_key_version != p_source.cache_version) {
		entry.prev_pos_seeded = false;
	}
	// Per-frame copies run exactly once per frame, even when the same surface is visible from multiple viewports.
	const bool first_touch_this_frame = entry.last_used_frame != current_frame;
	if (first_touch_this_frame) {
		if (entry.prev_pos_seeded) {
			rd->buffer_copy(entry.owned_vb_full, entry.prev_pos_vb, 0, 0, pos_size);
		}
		if (data_changed || !entry.prev_pos_seeded) {
			rd->buffer_copy(p_source.current_vb, entry.owned_vb_full, 0, 0, full_size);
			entry.cluster_lod_slots_stale = true; // The persistent selection's slots hold older positions (D109).
		}
		if (!entry.prev_pos_seeded) {
			// Seed prev = current so motion vectors are zero on the first frame.
			rd->buffer_copy(entry.owned_vb_full, entry.prev_pos_vb, 0, 0, pos_size);
			entry.prev_pos_seeded = true;
		}
	}

	// The cluster path (D82): a cluster BLAS cannot be refit, so on every frame the data changes
	// the surface's CLAS and BLAS are rebuilt in place, in the frame's batch; a triangle BLAS is
	// refit as before where the path is off or the surface does not fit.
	if (needs_full_rebuild) {
		cache_misses++;
		if (!entry.ptr) {
			entry.ptr = memnew(RTSurfaceData);
		}
		_release_deformed_blas(entry);
		bool clustered = false;
		if (cluster_blas_enabled && cluster_deformed_enabled) {
			clustered = _build_deformed_cluster_layout(surf->rt_deformed_handle, entry, p_mesh_surface, static_cast<uint32_t>(p_source.cache_key), p_source.surface_counter);
		}
		if (clustered && p_lod_select && entry.cluster_lod_table_offset != UINT32_MAX) {
			entry.cluster_lod_pending = true; // The selection builds it at the push (D87).
			entry.cluster_lod_pending_kind = RTDeformedCacheEntry::CLUSTER_QUEUED_FIRST_BUILD;
		} else if (clustered) {
			_queue_deformed_cluster_rebuild(entry, RTDeformedCacheEntry::CLUSTER_QUEUED_FIRST_BUILD);
		} else {
			_build_deformed_triangle_blas(surf->rt_deformed_handle, entry, p_mesh_surface, static_cast<uint32_t>(p_source.cache_key), p_source.surface_counter, r_dirty_blas_list);
		}
		entry.blas_built_once = entry.ptr->blas.is_valid();
	} else if (entry.cluster_count > 0 && p_lod_select && entry.cluster_lod_table_offset != UINT32_MAX) {
		// The selection rebuilds it on every frame it is traced, changed or not (D87): the level
		// follows the camera. What the batch would have queued is kept for the counters, and the
		// push queues the batch's rebuild instead where the instance cannot select.
		entry.cluster_lod_pending = true;
		if (data_changed) {
			cache_misses++;
			entry.cluster_lod_pending_kind = RTDeformedCacheEntry::CLUSTER_QUEUED_CHANGED;
		} else {
			cache_hits++;
			entry.cluster_lod_pending_kind = _deformed_cluster_region_expired(entry, rd->get_frames_drawn()) ? RTDeformedCacheEntry::CLUSTER_QUEUED_EXPIRY : RTDeformedCacheEntry::CLUSTER_QUEUED_NONE;
		}
	} else if (data_changed || (entry.cluster_count > 0 && ((entry.cluster_lod_last && entry.cluster_lod_pushed_frame != rd->get_frames_drawn()) || (!entry.cluster_templated && _deformed_cluster_region_expired(entry, rd->get_frames_drawn()))))) {
		// A last build that was a selection (D87) is rebuilt at LOD 0 before it is traced again
		// here, unless this frame's first viewport made it: the first viewport of a frame decides,
		// and a G-buffer viewport rendered after a traced one traces the selection for deformed meshes.
		// A templated entry's CLAS are its own (D107): nothing expires, an unchanged one does no work.
		cache_misses++;
		if (entry.cluster_count > 0) {
			_queue_deformed_cluster_rebuild(entry, data_changed ? RTDeformedCacheEntry::CLUSTER_QUEUED_CHANGED : RTDeformedCacheEntry::CLUSTER_QUEUED_EXPIRY);
		} else {
			r_dirty_blas_update_list.push_back(entry.ptr->blas);
		}
	} else {
		cache_hits++;
	}

	_mark_deformed_geometry(entry);

	surf->cached_final_transform_valid = false;

	entry.cached_change_stamp = p_source.change_stamp;
	entry.cached_key_version = p_source.cache_version;
	entry.cached_surface_counter = p_source.surface_counter;
	entry.cached_mesh_surface = p_mesh_surface;
	entry.cached_buffer_id = buffer_id;
	entry.cached_vertex_count = vertex_count;
	entry.cached_full_size = full_size;
	entry.last_used_frame = current_frame;

	return entry.ptr;
}

void RenderRaytracing::_mark_deformed_geometry(RTDeformedCacheEntry &r_entry) {
	if (r_entry.ptr && r_entry.ptr->blas.is_valid()) {
		r_entry.ptr->geometry.flags |= RT_GEOM_FLAG_DEFORMED;
		uint64_t prev_addr = RD::get_singleton()->buffer_get_device_address(r_entry.prev_pos_vb);
		r_entry.ptr->geometry.prev_vertex_buffer_address_lo = static_cast<uint32_t>(prev_addr & 0xFFFFFFFFULL);
		r_entry.ptr->geometry.prev_vertex_buffer_address_hi = static_cast<uint32_t>(prev_addr >> 32);
	}
}

void RenderRaytracing::_build_deformed_triangle_blas(RID p_handle, RTDeformedCacheEntry &r_entry, void *p_mesh_surface, uint32_t p_cache_key, uint32_t p_surface_counter, LocalVector<RID> &r_dirty_blas_list) {
	// Opacity micromaps (D96 OMMc): the BLAS carries the micromap its instance's walk wants, once
	// baked over the surface's current contents; refits keep it, as the build info does.
	RTOmmEntry *omm = _omm_deformed_entry(r_entry.omm_wanted_serial, p_surface_counter);
	bool omm_attached = false;
	_populate_surface_blas(p_mesh_surface, r_entry.owned_vb_full, true, true, true, p_cache_key, r_entry.ptr, r_dirty_blas_list, omm, &omm_attached);
	if (omm_attached) {
		_omm_deformed_attached(p_handle, r_entry, omm);
	} else if (omm != nullptr) {
		r_entry.omm_failed_serial = omm->serial;
	}
}

bool RenderRaytracing::_build_deformed_cluster_layout(RID p_handle, RTDeformedCacheEntry &r_entry, void *p_mesh_surface, uint32_t p_cache_key, uint32_t p_surface_counter) {
	// Opacity micromaps (D96 OMMd): the CLAS carry the micromap the walk wants, as the triangle BLAS
	// above does; the infos are written once with it, so every rebuild of the layout's CLAS keeps it.
	RTOmmEntry *omm = _omm_deformed_entry(r_entry.omm_wanted_serial, p_surface_counter);
	bool omm_attached = false;
	const bool clustered = _populate_deformed_cluster_layout(p_mesh_surface, p_cache_key, p_surface_counter, r_entry, omm, &omm_attached);
	if (clustered && omm_attached) {
		_omm_deformed_attached(p_handle, r_entry, omm);
	} else if (clustered && omm != nullptr) {
		r_entry.omm_failed_serial = omm->serial;
	}
	return clustered;
}

// ---------------------------------------------------------------------------
// Surface BLAS helper (shared between static and deformed paths)
// ---------------------------------------------------------------------------

// Fills RTSurfaceData geometry metadata from the surface format.
// Returns the RIDs needed for BLAS creation so _populate_surface_blas can use them.
void RendererSceneRenderImplementation::_fill_surface_geometry_data(
		void *p_mesh_surface,
		bool p_force_uncompressed,
		RTSurfaceData *r_surf_data,
		RID *r_vertex_buffer,
		RID *r_attribute_buffer,
		RID *r_index_buffer) {
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();

	uint64_t surface_format = mesh_storage->mesh_surface_get_format(p_mesh_surface);
	bool compressed = (surface_format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES) && !p_force_uncompressed;
	bool is_2d = surface_format & RSE::ARRAY_FLAG_USE_2D_VERTICES;

	r_surf_data->is_compressed = compressed;

	if (compressed) {
		AABB surface_aabb = mesh_storage->mesh_surface_get_aabb(p_mesh_surface);
		r_surf_data->aabb_transform.basis = Basis::from_scale(surface_aabb.size);
		r_surf_data->aabb_transform.origin = surface_aabb.position;
	} else {
		r_surf_data->aabb_transform = Transform3D();
	}

	RT_GeometryData &geom = r_surf_data->geometry;
	memset(&geom, 0, sizeof(geom));

	RID vertex_buffer = mesh_storage->mesh_surface_get_vertex_buffer(p_mesh_surface);
	RID attribute_buffer = mesh_storage->mesh_surface_get_attribute_buffer(p_mesh_surface);
	RID index_buffer = mesh_storage->mesh_surface_get_index_buffer(p_mesh_surface, 0);

	if (r_vertex_buffer) {
		*r_vertex_buffer = vertex_buffer;
	}
	if (r_attribute_buffer) {
		*r_attribute_buffer = attribute_buffer;
	}
	if (r_index_buffer) {
		*r_index_buffer = index_buffer;
	}

	// The surface's whole counts (D93d), which its buffers hold unless it streams its geometry.
	uint32_t vertex_count = mesh_storage->mesh_surface_get_rt_vertex_count(p_mesh_surface);
	uint32_t index_count = mesh_storage->mesh_surface_get_rt_index_count(p_mesh_surface);

	geom.vertex_count = vertex_count;

	// Position stride
	uint32_t position_stride;
	if (is_2d) {
		position_stride = sizeof(float) * 2;
	} else if (compressed) {
		position_stride = sizeof(uint16_t) * 4;
	} else {
		position_stride = sizeof(float) * 3;
	}
	geom.position_stride = position_stride;

	// Normal/tangent layout
	uint32_t normal_stride;
	uint32_t tangent_stride = 0;
	geom.normal_byte_offset = RT_OFFSET_NONE;
	geom.tangent_byte_offset = RT_OFFSET_NONE;
	uint32_t current_offset = position_stride * vertex_count;

	bool has_normal = surface_format & RSE::ARRAY_FORMAT_NORMAL;
	bool has_tangent = surface_format & RSE::ARRAY_FORMAT_TANGENT;

	if (compressed) {
		normal_stride = sizeof(uint16_t) * 2;
		if (has_normal) {
			geom.normal_byte_offset = current_offset;
			current_offset += normal_stride * vertex_count;
		}
	} else {
		if (has_normal && has_tangent) {
			normal_stride = sizeof(uint16_t) * 4;
			tangent_stride = sizeof(uint16_t) * 4;
			geom.normal_byte_offset = current_offset;
			geom.tangent_byte_offset = current_offset;
			current_offset += normal_stride * vertex_count;
		} else if (has_normal) {
			normal_stride = sizeof(uint16_t) * 2;
			geom.normal_byte_offset = current_offset;
			current_offset += normal_stride * vertex_count;
		} else {
			normal_stride = 0;
		}
	}
	geom.normal_stride = normal_stride;
	geom.tangent_stride = tangent_stride;
	geom.flags = compressed ? RT_GEOM_FLAG_COMPRESSED : 0;

	// The surface's box for every surface: the shader decodes compressed positions through it
	// (gated on the flag) and the emissive gather (D50) reads it as an area proxy for both.
	{
		AABB surface_aabb = mesh_storage->mesh_surface_get_aabb(p_mesh_surface);
		geom.aabb_size_x = surface_aabb.size.x;
		geom.aabb_size_y = surface_aabb.size.y;
		geom.aabb_size_z = surface_aabb.size.z;
		geom.aabb_pos_x = surface_aabb.position.x;
		geom.aabb_pos_y = surface_aabb.position.y;
		geom.aabb_pos_z = surface_aabb.position.z;
	}

	// Attribute buffer layout
	uint32_t attrib_offset = 0;
	geom.uv_byte_offset = RT_OFFSET_NONE;
	geom.color_byte_offset = RT_OFFSET_NONE;

	if (surface_format & RSE::ARRAY_FORMAT_COLOR) {
		geom.color_byte_offset = attrib_offset;
		attrib_offset += sizeof(uint32_t);
	}
	if (surface_format & RSE::ARRAY_FORMAT_TEX_UV) {
		geom.uv_byte_offset = attrib_offset;
		attrib_offset += compressed ? sizeof(uint16_t) * 2 : sizeof(float) * 2;
	}
	if (surface_format & RSE::ARRAY_FORMAT_TEX_UV2) {
		attrib_offset += compressed ? sizeof(uint16_t) * 2 : sizeof(float) * 2;
	}
	for (int ci = 0; ci < RSE::ARRAY_CUSTOM_COUNT; ci++) {
		const uint32_t fmt_shift[RSE::ARRAY_CUSTOM_COUNT] = { RSE::ARRAY_FORMAT_CUSTOM0_SHIFT, RSE::ARRAY_FORMAT_CUSTOM1_SHIFT, RSE::ARRAY_FORMAT_CUSTOM2_SHIFT, RSE::ARRAY_FORMAT_CUSTOM3_SHIFT };
		if (surface_format & (1ULL << (RSE::ARRAY_CUSTOM0 + ci))) {
			uint32_t fmt = (surface_format >> fmt_shift[ci]) & RSE::ARRAY_FORMAT_CUSTOM_MASK;
			const uint32_t fmtsize[RSE::ARRAY_CUSTOM_MAX] = { 4, 4, 4, 8, 4, 8, 12, 16 };
			attrib_offset += fmtsize[fmt];
		}
	}
	geom.attribute_stride = attrib_offset;

	// UV scale (fp16 packed, matches GLSL unpackHalf2x16)
	Vector4 uv_scale = mesh_storage->mesh_surface_get_uv_scale(p_mesh_surface);
	geom.uv_scale_packed = (uint32_t(Math::make_half_float(uv_scale.y)) << 16) | Math::make_half_float(uv_scale.x);

	// Index format (no device address — caller fills those in)
	if (index_buffer.is_valid() && index_count > 0) {
		bool is_16bit = vertex_count <= 65536 && vertex_count > 0;
		geom.index_format = is_16bit ? RT_INDEX_FORMAT_UINT16 : RT_INDEX_FORMAT_UINT32;
		geom.primitive_count = index_count / 3;
	} else {
		geom.index_format = RT_INDEX_FORMAT_NONE;
		geom.primitive_count = vertex_count / 3;
	}
}

void RendererSceneRenderImplementation::_fill_surface_triangle_geometry(RID p_vertex_buffer, RID p_index_buffer, const RT_GeometryData &p_geometry, bool p_2d, bool p_compressed, RD::AccelerationStructureGeometry &r_geometry) {
	RD::DataFormat pos_format;
	if (p_2d) {
		pos_format = RD::DATA_FORMAT_R32G32_SFLOAT;
	} else if (p_compressed) {
		pos_format = RD::DATA_FORMAT_R16G16B16A16_UNORM;
	} else {
		pos_format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
	}

	// Type defaults to TYPE_TRIANGLES; set explicitly for clarity.
	r_geometry.type = RD::AccelerationStructureGeometry::TYPE_TRIANGLES;
	r_geometry.geometry.triangles.vertex_buffer = p_vertex_buffer;
	r_geometry.geometry.triangles.vertex_stride = p_geometry.position_stride;
	r_geometry.geometry.triangles.vertex_count = p_geometry.vertex_count;
	r_geometry.geometry.triangles.vertex_format = pos_format;

	const uint32_t index_count = (p_geometry.index_format != RT_INDEX_FORMAT_NONE) ? p_geometry.primitive_count * 3 : 0;
	if (p_index_buffer.is_valid() && index_count > 0) {
		r_geometry.geometry.triangles.index_buffer = p_index_buffer;
		r_geometry.geometry.triangles.index_count = index_count;
	}
}

void RenderRaytracing::_populate_surface_blas(
		void *p_mesh_surface,
		RID p_vertex_buffer_override,
		bool p_force_uncompressed,
		bool p_prefer_fast_build,
		bool p_allow_update,
		uint32_t p_cache_key,
		RTSurfaceData *r_surf_data,
		LocalVector<RID> &r_dirty_blas_list,
		const RTOmmEntry *p_omm,
		bool *r_omm_attached) {
	RID vertex_buffer, attribute_buffer, index_buffer;
	_fill_surface_geometry_data(p_mesh_surface, p_force_uncompressed, r_surf_data,
			&vertex_buffer, &attribute_buffer, &index_buffer);

	if (p_vertex_buffer_override.is_valid()) {
		vertex_buffer = p_vertex_buffer_override;
	}

	RD *rd = RD::get_singleton();
	RT_GeometryData &geom = r_surf_data->geometry;

	// Geometry residency (D93d): a surface whose geometry the mesh storage streams has its fallback
	// level in its buffers, which the trace never reads. Its blobs carry its hits, with no buffer
	// addressed; where they cannot (no stream, a failed population, an allocation), the surface's
	// arrays are uploaded from the RAM copy for the population's life and it builds from those.
	r_surf_data->geometry_streamed = p_vertex_buffer_override.is_null() && RendererRD::MeshStorage::get_singleton()->mesh_surface_is_geometry_streamed(p_mesh_surface);
	if (r_surf_data->geometry_streamed) {
		vertex_buffer = RID();
		attribute_buffer = RID();
		index_buffer = RID();
		if (cluster_blas_enabled && !p_allow_update && geom.primitive_count > 0 && _populate_surface_cluster_blas(p_mesh_surface, RID(), RID(), p_cache_key, r_surf_data, true)) {
			return;
		}
		_release_cluster_geometry(r_surf_data);
		if (!_materialize_surface(r_surf_data, p_mesh_surface, true)) {
			WARN_PRINT_ONCE(vformat("RT: surface [%d] streams its geometry, and neither its blobs nor an upload of its arrays could be made; it is not traced.", p_cache_key));
			return;
		}
		vertex_buffer = r_surf_data->materialized_vertex_buffer;
		attribute_buffer = r_surf_data->materialized_attribute_buffer;
		index_buffer = r_surf_data->materialized_index_buffer;
	}

	// Hit shaders read these through their addresses; an in-place update must wait for an earlier frame's trace.
	// An override is a per-frame copy, declared on the trace instead (register_raytracing_buffer_dependencies).
	if (vertex_buffer.is_valid()) {
		geom.vertex_buffer_address = rd->buffer_get_device_address(vertex_buffer);
		if (p_vertex_buffer_override.is_null()) {
			rd->buffer_add_device_address_read_stages(vertex_buffer, RDD::PIPELINE_STAGE_RAY_TRACING_SHADER_BIT);
		}
	}
	if (attribute_buffer.is_valid()) {
		geom.attribute_buffer_address = rd->buffer_get_device_address(attribute_buffer);
		rd->buffer_add_device_address_read_stages(attribute_buffer, RDD::PIPELINE_STAGE_RAY_TRACING_SHADER_BIT);
	}
	if (index_buffer.is_valid() && geom.index_format != RT_INDEX_FORMAT_NONE) {
		geom.index_buffer_address = rd->buffer_get_device_address(index_buffer);
		rd->buffer_add_device_address_read_stages(index_buffer, RDD::PIPELINE_STAGE_RAY_TRACING_SHADER_BIT);
	}

	uint32_t index_count = (geom.index_format != RT_INDEX_FORMAT_NONE) ? geom.primitive_count * 3 : 0;

	bool is_2d = RendererRD::MeshStorage::get_singleton()->mesh_surface_get_format(p_mesh_surface) & RSE::ARRAY_FLAG_USE_2D_VERTICES;
	bool compressed = r_surf_data->is_compressed;

	// A static indexed 3D surface builds from clusters (D79); a deformed surface reaching this
	// function is one the cluster path declined or the path is off (D82), and refits a triangle
	// BLAS in place, as do 2D and non-indexed surfaces.
	if (cluster_blas_enabled) {
		if (p_vertex_buffer_override.is_null() && !p_allow_update && !is_2d && index_buffer.is_valid() && index_count > 0) {
			if (_populate_surface_cluster_blas(p_mesh_surface, vertex_buffer, index_buffer, p_cache_key, r_surf_data)) {
				_count_unstreamed_geometry(p_mesh_surface, p_cache_key, r_surf_data);
				return;
			}
			_release_cluster_geometry(r_surf_data); // A triangle BLAS reads the surface's own buffers.
		} else {
			print_verbose(vformat("RT: surface [%d] on the triangle BLAS path (deformed %d, 2D %d, indexed %d).", p_cache_key, int(p_vertex_buffer_override.is_valid() || p_allow_update), int(is_2d), int(index_buffer.is_valid() && index_count > 0)));
		}
	}
	_count_unstreamed_geometry(p_mesh_surface, p_cache_key, r_surf_data);

	// Create BLAS using the new geometry-based API.
	{
		RD::AccelerationStructureGeometry as_geom;
		_fill_surface_triangle_geometry(vertex_buffer, index_buffer, geom, is_2d, compressed, as_geom);
		const bool omm_attached = p_omm != nullptr && _omm_deformed_geometry(p_omm, geom, as_geom);

		BitField<RD::AccelerationStructureFlagBits> as_flags = p_prefer_fast_build
				? RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT
				: RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT;
		if (p_allow_update) {
			as_flags.set_flag(RD::ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT);
		}

		r_surf_data->blas = RD::get_singleton()->blas_create({ &as_geom, 1 }, as_flags);
		if (!r_surf_data->blas.is_valid()) {
			return;
		}
		RD::get_singleton()->set_resource_name(r_surf_data->blas,
				String(p_vertex_buffer_override.is_valid() ? "RT BLAS deformed [" : "RT BLAS [") + itos(p_cache_key) + "]");
		r_dirty_blas_list.push_back(r_surf_data->blas);
		if (r_omm_attached) {
			*r_omm_attached = omm_attached;
		}
	}
}

// Stage 2's materialization worklist (D93d): a surface the mesh storage streams the geometry of,
// whose trace nonetheless reads the surface's own buffers.
void RenderRaytracing::_count_unstreamed_geometry(void *p_mesh_surface, uint32_t p_cache_key, const RTSurfaceData *p_surf_data) {
	if (!RendererRD::MeshStorage::get_singleton()->mesh_surface_is_geometry_streamed(p_mesh_surface) || (p_surf_data->geometry.flags & RT_GEOM_FLAG_CLUSTER_GEOMETRY)) {
		return;
	}
	cluster_geometry_unstreamed_populations++;
	print_verbose(vformat("RT: surface [%d] streams its geometry in the mesh storage, but the trace reads an upload of its arrays (%d such populations over the run, D93d).", p_cache_key, cluster_geometry_unstreamed_populations));
}

// Uploads of a streamed surface's arrays (D93d), as RTSurfaceData describes them: the RAM copy's
// vertex (padded as the mesh storage pads it), attribute and index data, the coarse tail included,
// in buffers the surface data owns, which `geometry` addresses. A transient upload is touched by each
// frame that needs it and released past the minimum age; a held one lives with the population. A
// capped upload (an emissive material's, a merge's bake) counts against cluster_lod_stream_upload_cap
// in the walk's order, so which frame it lands in never depends on timing; past the cap it waits for
// a later frame (the merge takes the expanded path, the emissive sampling skips the surface), and
// the first of a frame always goes. A population's upload, the fallback when blobs fail, is not capped.
bool RenderRaytracing::_materialize_surface(RTSurfaceData *p_surf_data, void *p_mesh_surface, bool p_held, bool p_capped) {
	RD *rd = RD::get_singleton();
	p_surf_data->materialized_frame = rd->get_frames_drawn();
	if (p_surf_data->materialized_vertex_buffer.is_valid()) {
		if (p_held && !p_surf_data->materialized_held) {
			p_surf_data->materialized_held = true;
			materialized_transient.erase(p_surf_data);
		}
		return true;
	}
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();
	const Vector<uint8_t> &ram_vertices = mesh_storage->mesh_surface_get_ram_vertex_data(p_mesh_surface);
	const Vector<uint8_t> &ram_attributes = mesh_storage->mesh_surface_get_ram_attribute_data(p_mesh_surface);
	const Vector<uint8_t> &ram_indices = mesh_storage->mesh_surface_get_ram_index_data(p_mesh_surface);
	RT_GeometryData &geom = p_surf_data->geometry;
	if (ram_vertices.is_empty() || ram_indices.is_empty() || geom.index_format == RT_INDEX_FORMAT_NONE) {
		return false;
	}
	const BitField<RD::BufferCreationBits> flags = RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT | RD::BUFFER_CREATION_AS_STORAGE_BIT;
	const uint64_t format = mesh_storage->mesh_surface_get_format(p_mesh_surface);
	Vector<uint8_t> vertices = ram_vertices;
	if (!(format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES) && (format & RSE::ARRAY_FORMAT_NORMAL) && !(format & RSE::ARRAY_FORMAT_TANGENT)) {
		vertices.resize_initialized(vertices.size() + sizeof(uint16_t) * 2);
	}
	const uint64_t bytes = uint64_t(vertices.size()) + ram_attributes.size() + ram_indices.size();
	if (p_capped && materialized_frame_uploads > 0 && materialized_frame_bytes + bytes > cluster_lod_stream_upload_cap) {
		materialized_deferred++;
		return false;
	}
	const uint32_t index_size = geom.index_format == RT_INDEX_FORMAT_UINT16 ? sizeof(uint16_t) : sizeof(uint32_t);
	RID vertex_buffer = rd->vertex_buffer_create(vertices.size(), vertices, flags);
	RID attribute_buffer = ram_attributes.is_empty() ? RID() : rd->vertex_buffer_create(ram_attributes.size(), ram_attributes, flags);
	RID index_buffer = rd->index_buffer_create(ram_indices.size() / index_size, geom.index_format == RT_INDEX_FORMAT_UINT16 ? RD::INDEX_BUFFER_FORMAT_UINT16 : RD::INDEX_BUFFER_FORMAT_UINT32, ram_indices, false, flags);
	if (vertex_buffer.is_null() || index_buffer.is_null() || (attribute_buffer.is_null() && !ram_attributes.is_empty())) {
		for (const RID &buffer : { vertex_buffer, attribute_buffer, index_buffer }) {
			if (buffer.is_valid()) {
				rd->free_rid(buffer);
			}
		}
		return false;
	}
	rd->set_resource_name(vertex_buffer, "RT uploaded vertices");
	rd->set_resource_name(index_buffer, "RT uploaded indices");
	p_surf_data->materialized_vertex_buffer = vertex_buffer;
	p_surf_data->materialized_attribute_buffer = attribute_buffer;
	p_surf_data->materialized_index_buffer = index_buffer;
	geom.vertex_buffer_address = rd->buffer_get_device_address(vertex_buffer);
	rd->buffer_add_device_address_read_stages(vertex_buffer, RDD::PIPELINE_STAGE_RAY_TRACING_SHADER_BIT);
	if (attribute_buffer.is_valid()) {
		rd->set_resource_name(attribute_buffer, "RT uploaded attributes");
		geom.attribute_buffer_address = rd->buffer_get_device_address(attribute_buffer);
		rd->buffer_add_device_address_read_stages(attribute_buffer, RDD::PIPELINE_STAGE_RAY_TRACING_SHADER_BIT);
	}
	geom.index_buffer_address = rd->buffer_get_device_address(index_buffer);
	rd->buffer_add_device_address_read_stages(index_buffer, RDD::PIPELINE_STAGE_RAY_TRACING_SHADER_BIT);
	p_surf_data->materialized_bytes = bytes;
	if (p_capped) {
		materialized_frame_bytes += bytes;
		materialized_frame_uploads++;
	}
	p_surf_data->materialized_held = p_held;
	if (!p_held) {
		materialized_transient.push_back(p_surf_data);
	}
	materialized_live_bytes += p_surf_data->materialized_bytes;
	materialized_peak_bytes = MAX(materialized_peak_bytes, materialized_live_bytes);
	materialized_uploads++;
	return true;
}

// The upload goes, and `geometry` addresses no buffer again. RenderingDevice keeps the buffers past
// the frames that may still read them.
void RenderRaytracing::_release_materialized(RTSurfaceData *p_surf_data) {
	if (p_surf_data->materialized_vertex_buffer.is_null()) {
		return;
	}
	RD *rd = RD::get_singleton();
	for (RID *buffer : { &p_surf_data->materialized_vertex_buffer, &p_surf_data->materialized_attribute_buffer, &p_surf_data->materialized_index_buffer }) {
		if (buffer->is_valid()) {
			rd->free_rid(*buffer);
			*buffer = RID();
		}
	}
	p_surf_data->geometry.vertex_buffer_address = 0;
	p_surf_data->geometry.attribute_buffer_address = 0;
	p_surf_data->geometry.index_buffer_address = 0;
	materialized_live_bytes -= p_surf_data->materialized_bytes;
	p_surf_data->materialized_bytes = 0;
	if (!p_surf_data->materialized_held) {
		materialized_transient.erase(p_surf_data);
	}
	p_surf_data->materialized_held = false;
}

// Once a frame: transient uploads no frame needed within the minimum age go.
void RenderRaytracing::_sweep_materialized() {
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	if (materialized_swept_frame == frames_drawn) {
		return;
	}
	materialized_swept_frame = frames_drawn;
	materialized_frame_bytes = 0;
	materialized_frame_uploads = 0;
	for (uint32_t i = materialized_transient.size(); i-- > 0;) {
		RTSurfaceData *surf_data = materialized_transient[i];
		if (frames_drawn - surf_data->materialized_frame > cluster_lod_stream_min_age) {
			_release_materialized(surf_data);
		}
	}
}

bool RenderRaytracing::_populate_surface_cluster_blas(void *p_mesh_surface, RID p_vertex_buffer, RID p_index_buffer, uint32_t p_cache_key, RTSurfaceData *r_surf_data, bool p_blobs_only, const RTOmmEntry *p_omm) {
	RD *rd = RD::get_singleton();
	const RT_GeometryData &geom = r_surf_data->geometry;
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();

	const uint32_t primitive_count = geom.primitive_count;
	if (primitive_count == 0 || (!p_blobs_only && (geom.vertex_buffer_address == 0 || geom.index_buffer_address == 0)) || caps.max_triangles_per_cluster == 0 || caps.max_vertices_per_cluster < 3) {
		print_verbose(vformat("RT: surface [%d] on the triangle BLAS path: %d triangles, addresses %d/%d, clusters of %d/%d.", p_cache_key, primitive_count, int(geom.vertex_buffer_address != 0), int(geom.index_buffer_address != 0), caps.max_triangles_per_cluster, caps.max_vertices_per_cluster));
		return false;
	}

	// The clusters (_cluster_layout_for_surface): from the surface's table (D80) or runs of the
	// device's cluster size without one.
	RTClusterLayout layout;
	_cluster_layout_for_surface(p_mesh_surface, primitive_count, caps.max_triangles_per_cluster, caps.max_vertices_per_cluster, p_cache_key, layout);
	const LocalVector<uint32_t> &cluster_triangles = layout.triangles;
	const LocalVector<uint32_t> &cluster_vertices = layout.vertices;
	const uint32_t max_cluster_triangles = layout.max_triangles;
	const uint32_t max_cluster_vertices = layout.max_vertices;
	const bool from_table = layout.from_table;
	// With a cluster LOD hierarchy (D85) the CLAS cover every level and the BLAS the LOD 0
	// clusters, the first lod0_cluster_count of them, so the image is the flat clustering's.
	// A surface's micromap CLAS (D96 OMMd) are its LOD 0 clusters alone: the micromap is baked over
	// the LOD 0 triangles, and an instance traces them only where it traces LOD 0.
	const uint32_t lod0_cluster_count = layout.lod0_clusters;
	const uint32_t cluster_count = p_omm ? lod0_cluster_count : cluster_triangles.size();
	const uint32_t total_cluster_vertices = p_omm ? layout.lod0_vertices : layout.total_vertices;
	const uint32_t total_cluster_triangles = p_omm ? primitive_count : layout.total_triangles;
	uint64_t omm_address = 0, omm_index_address = 0;
	if (p_omm && !_omm_cluster_addresses(p_omm, primitive_count, omm_address, omm_index_address)) {
		return false;
	}
	const uint32_t index_size = geom.index_format == RT_INDEX_FORMAT_UINT16 ? sizeof(uint16_t) : sizeof(uint32_t);
	const RD::DataFormat pos_format = r_surf_data->is_compressed ? RD::DATA_FORMAT_R16G16B16A16_UNORM : RD::DATA_FORMAT_R32G32B32_SFLOAT;
	// A streamed surface (D92) builds its persistent groups' CLAS alone, in runs of clusters, and its
	// BLAS over them: the coarsest cut, which stands wherever LOD 0's BLAS would. Otherwise every
	// cluster is built and the BLAS spans LOD 0.
	RTLodStream *stream = nullptr;
	if (!p_omm && cluster_lod_stream_enabled && cluster_lod_select_enabled && layout.levels > 1) { // As the hierarchy below is attached.
		stream = _create_lod_stream(p_mesh_surface, layout, p_vertex_buffer, p_index_buffer, p_cache_key);
	}
	const uint32_t build_count = stream ? stream->persistent_clusters : cluster_count;
	const uint32_t blas_count = stream ? stream->persistent_clusters : lod0_cluster_count;
	// Geometry residency (D93b): a streamed surface whose records and blobs are in place gives every
	// CLAS its cluster's index as its ID, here and in each load, and its hits read through them. Its
	// builds read the blobs' indices and staged positions, not its buffers (D93c).
	RTClusterGeometryStage geometry_stage;
	const bool cluster_geometry = stream && cluster_lod_stream_geometry_enabled && _populate_cluster_geometry(p_mesh_surface, layout, stream, p_cache_key, r_surf_data, geometry_stage);
	if (p_blobs_only && !cluster_geometry) {
		// A surface whose buffers hold its fallback level builds from blobs or not at all; the caller
		// uploads its arrays and builds again (D93d).
		if (stream) {
			memdelete(stream);
		}
		return false;
	}

	// Sizes (D81): the CLAS are built with implicit destinations into a staging buffer, so their
	// size query returns one region for all of them and the build writes each one's size back;
	// the BLAS is built at an explicit address, the staging base, so its handle exists this frame
	// (explicit builds write no sizes, so the BLAS keeps its worst case). With
	// rendering/pathtracing/cluster_blas_compaction on, once the CLAS sizes are read they move to
	// the arena at their compacted size and the BLAS is rebuilt over them
	// (_process_cluster_compactions); off, the staging is the surface's for life. The argument
	// sections the move needs (the sizes, info B, the compact addresses) are laid out either way.
	RD::ClusterAccelerationStructureInput clas_input;
	clas_input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_TRIANGLE_CLUSTER;
	clas_input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_IMPLICIT_DESTINATIONS;
	clas_input.max_acceleration_structure_count = build_count;
	clas_input.flags = _cluster_build_flags_omm(RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT, p_omm != nullptr);
	clas_input.vertex_format = pos_format;
	clas_input.max_cluster_triangle_count = max_cluster_triangles;
	clas_input.max_cluster_vertex_count = max_cluster_vertices;
	clas_input.max_total_triangle_count = stream ? stream->persistent_triangles : total_cluster_triangles;
	clas_input.max_total_vertex_count = stream ? stream->persistent_vertices : total_cluster_vertices;

	RD::ClusterAccelerationStructureInput blas_input;
	blas_input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_CLUSTERS_BOTTOM_LEVEL;
	blas_input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
	blas_input.max_acceleration_structure_count = 1;
	blas_input.flags = _cluster_build_flags_omm(RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT, p_omm != nullptr);
	blas_input.max_total_cluster_count = blas_count;
	blas_input.max_cluster_count_per_acceleration_structure = blas_count;

	uint64_t clas_region_size = 0, clas_scratch = 0, blas_size = 0, blas_scratch = 0;
	rd->cluster_acceleration_structure_get_build_sizes(clas_input, clas_region_size, clas_scratch);
	rd->cluster_acceleration_structure_get_build_sizes(blas_input, blas_size, blas_scratch);
	if (clas_region_size == 0 || blas_size == 0) {
		print_verbose(vformat("RT: surface [%d] on the triangle BLAS path: the driver sized the CLAS region at %d and the BLAS at %d bytes.", p_cache_key, clas_region_size, blas_size));
		if (stream) {
			memdelete(stream);
		}
		return false;
	}

	// Staging: the BLAS first, then the implicit CLAS region at its alignment. The buffer's own
	// address is aligned only as the allocator saw fit, so the base is aligned inside a padded buffer.
	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	const uint64_t clas_region = _align_up(blas_size, base_alignment);
	const uint64_t staging_size = base_alignment + clas_region + clas_region_size;
	if (staging_size > UINT32_MAX) {
		if (stream) {
			memdelete(stream);
		}
		ERR_FAIL_V_MSG(false, "RT: a cluster BLAS's staging exceeds 4 GB.");
	}

	RID staging = rd->storage_buffer_create(uint32_t(staging_size), {}, 0,
			RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_STORAGE_BIT);
	if (staging.is_null()) {
		if (stream) {
			memdelete(stream);
		}
		return false;
	}
	const uint64_t buffer_address = rd->buffer_get_device_address(staging);
	const uint64_t staging_address = _align_up(buffer_address, base_alignment);
	const uint64_t blas_offset = staging_address - buffer_address;
	RID blas = rd->cluster_blas_create(staging, blas_offset);
	if (blas.is_null()) {
		rd->free_rid(staging);
		if (stream) {
			memdelete(stream);
		}
		return false;
	}

	// Arguments, 16-byte sections: the CLAS infos; BLAS info A, over the implicit addresses (this
	// frame's build); BLAS info B, over the compact addresses (the rebuild after the move); the
	// implicit addresses (cluster_count CLAS the driver writes, then the BLAS's staging address);
	// the CLAS sizes; the compact addresses, filled at the move.
	const uint32_t clas_info_stride = caps.triangle_build_info_size;
	const uint32_t blas_info_stride = caps.bottom_level_build_info_size;
	const uint64_t clas_infos_offset = 0;
	const uint64_t blas_info_a_offset = _align_up(clas_infos_offset + uint64_t(clas_info_stride) * build_count, 16);
	const uint64_t blas_info_b_offset = _align_up(blas_info_a_offset + blas_info_stride, 16);
	const uint64_t implicit_addresses_offset = _align_up(blas_info_b_offset + blas_info_stride, 16);
	const uint64_t sizes_offset = _align_up(implicit_addresses_offset + sizeof(uint64_t) * (build_count + 1), 16);
	const uint64_t compact_addresses_offset = _align_up(sizes_offset + sizeof(uint32_t) * build_count, 16);
	const uint64_t args_size = _align_up(compact_addresses_offset + sizeof(uint64_t) * (build_count + 1), 16);

	LocalVector<uint8_t> args;
	args.resize(args_size);
	memset(args.ptr(), 0, args_size);

	uint32_t first_triangle = 0;
	uint32_t built = 0;
	uint32_t run = 0; // A streamed surface's next persistent run.
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t triangle_count = cluster_triangles[c];
		if (stream) {
			while (run < stream->persistent_runs.size() && c >= stream->persistent_runs[run] + stream->persistent_runs[run + 1]) {
				run += 2;
			}
			if (run >= stream->persistent_runs.size() || c < stream->persistent_runs[run]) {
				first_triangle += triangle_count;
				continue;
			}
		}
		RD::ClusterTriangleBuildInfo info;
		info.cluster_id = cluster_geometry ? c : first_triangle;
		info.triangle_count = triangle_count;
		if (cluster_geometry) {
			info.vertex_count = stream->cluster_packed[c] >> 16;
			info.index_type = RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT8;
			info.index_address = geometry_stage.index_addresses[built];
			info.index_stride = sizeof(uint8_t);
			info.vertex_address = geometry_stage.positions_address + geometry_stage.position_offsets[built];
		} else {
			info.vertex_count = stream ? stream->cluster_packed[c] >> 16 : cluster_vertices[c]; // As persistent_vertices counts them.
			info.index_type = geom.index_format == RT_INDEX_FORMAT_UINT16 ? RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT16 : RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT32;
			info.index_address = geom.index_buffer_address + uint64_t(first_triangle) * 3u * index_size;
			info.index_stride = index_size;
			info.vertex_address = geom.vertex_buffer_address;
		}
		info.vertex_stride = geom.position_stride;
		if (p_omm) {
			// The micromap's index runs in the surface's primitive order, which the clusters split.
			info.opacity_micromap_address = omm_address;
			info.opacity_micromap_index_address = omm_index_address + uint64_t(first_triangle) * sizeof(int32_t);
			info.opacity_micromap_index_type = RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT32;
			info.opacity_micromap_index_stride = sizeof(int32_t);
		}
		rd->cluster_triangle_build_info_write(args.ptr() + clas_infos_offset + uint64_t(built) * clas_info_stride, info);
		built++;
		first_triangle += triangle_count;
	}
	reinterpret_cast<uint64_t *>(args.ptr() + implicit_addresses_offset)[build_count] = staging_address;

	RID args_buffer = rd->storage_buffer_create(uint32_t(args_size), {}, 0,
			RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
	if (args_buffer.is_null()) {
		rd->free_rid(blas);
		rd->free_rid(staging);
		if (stream) {
			memdelete(stream);
		}
		return false;
	}
	const uint64_t args_address = rd->buffer_get_device_address(args_buffer);
	{
		RD::ClusterBottomLevelBuildInfo blas_info;
		blas_info.cluster_references_count = blas_count;
		blas_info.cluster_references_stride = sizeof(uint64_t);
		blas_info.cluster_references_address = args_address + implicit_addresses_offset;
		rd->cluster_bottom_level_build_info_write(args.ptr() + blas_info_a_offset, blas_info);
		blas_info.cluster_references_address = args_address + compact_addresses_offset;
		rd->cluster_bottom_level_build_info_write(args.ptr() + blas_info_b_offset, blas_info);
	}
	rd->buffer_update(args_buffer, 0, uint32_t(args_size), args.ptr());

	// The two builds this frame: every CLAS into the staging region, then the BLAS over the
	// addresses the driver wrote.
	{
		RD::ClusterAccelerationStructureBuild build;
		build.input = clas_input;
		build.dst_implicit_buffer = staging;
		build.dst_implicit_offset = blas_offset + clas_region;
		build.dst_addresses_buffer = args_buffer;
		build.dst_addresses_offset = implicit_addresses_offset;
		build.dst_addresses_stride = sizeof(uint64_t);
		build.dst_sizes_buffer = args_buffer;
		build.dst_sizes_offset = sizes_offset;
		build.dst_sizes_stride = sizeof(uint32_t);
		build.src_infos_buffer = args_buffer;
		build.src_infos_offset = clas_infos_offset;
		build.src_infos_stride = clas_info_stride;
		if (cluster_geometry) {
			build.source_buffers = geometry_stage.blob_buffers;
			build.source_buffers.push_back(geometry_stage.positions_buffer);
		} else {
			build.source_buffers.push_back(p_vertex_buffer);
			build.source_buffers.push_back(p_index_buffer);
		}
		if (p_omm) {
			build.source_buffers.push_back(p_omm->index_buffer);
			build.source_micromaps.push_back(p_omm->micromap);
		}
		if (rd->cluster_acceleration_structure_build(build) != OK) {
			rd->free_rid(blas);
			rd->free_rid(args_buffer);
			rd->free_rid(staging);
			if (stream) {
				memdelete(stream);
			}
			return false;
		}
	}
	{
		RD::ClusterAccelerationStructureBuild build;
		build.input = blas_input;
		build.dst_storage_buffer = staging;
		build.dst_addresses_buffer = args_buffer;
		build.dst_addresses_offset = implicit_addresses_offset + sizeof(uint64_t) * build_count;
		build.dst_addresses_stride = sizeof(uint64_t);
		build.src_infos_buffer = args_buffer;
		build.src_infos_offset = blas_info_a_offset;
		build.src_infos_stride = blas_info_stride;
		build.source_acceleration_structure_buffers.push_back(staging); // The CLAS it references, read through its info.
		build.dst_acceleration_structures.push_back(blas);
		if (rd->cluster_acceleration_structure_build(build) != OK) {
			rd->free_rid(blas);
			rd->free_rid(args_buffer);
			rd->free_rid(staging);
			if (stream) {
				memdelete(stream);
			}
			return false;
		}
	}

	r_surf_data->blas = blas;
	const String name_suffix = p_omm ? " OMM [" + itos(p_omm->serial) + "]" : " [" + itos(p_cache_key) + "]";
	rd->set_resource_name(r_surf_data->blas, "RT cluster BLAS" + name_suffix);
	rd->set_resource_name(staging, "RT cluster staging" + name_suffix);
	rd->set_resource_name(args_buffer, "RT cluster args" + name_suffix);
	r_surf_data->cluster_storage = staging;
	r_surf_data->cluster_args = args_buffer;
	r_surf_data->blas_size = staging_size + args_size;
	r_surf_data->cluster_count = cluster_count;
	r_surf_data->cluster_lod0_count = lod0_cluster_count;
	r_surf_data->cluster_built_count = build_count;
	r_surf_data->cluster_blas_count = blas_count;
	r_surf_data->cluster_omm = p_omm != nullptr;
	r_surf_data->cluster_cache_key = p_cache_key;
	r_surf_data->cluster_blas_worst_size = blas_size;
	r_surf_data->cluster_clas_region_size = clas_region_size;
	r_surf_data->cluster_args_implicit_offset = implicit_addresses_offset;
	r_surf_data->cluster_args_compact_offset = compact_addresses_offset;
	r_surf_data->cluster_args_blas_info_offset = blas_info_b_offset;

	// The sizes come back on the render thread a frame queue later; the compaction runs then.
	// Without it (rendering/pathtracing/cluster_blas_compaction off) the surface keeps its staging
	// buffers and BLAS for life, and nothing is read back.
	if (cluster_compaction_enabled) {
		const uint64_t ticket = ++cluster_compaction_next_ticket;
		const Error readback = rd->buffer_get_data_async(args_buffer, callable_mp_static(&RenderRaytracing::_on_cluster_sizes_read).bind(ticket), uint32_t(sizes_offset), uint32_t(sizeof(uint32_t) * build_count));
		if (readback == OK) {
			r_surf_data->compaction_ticket = ticket;
			cluster_compaction_pending.insert(ticket, r_surf_data);
		} else {
			WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: the sizes could not be read back; the surface stays in its staging buffer.", p_cache_key));
		}
	}

	// The hierarchy for the selection (D86), from the same table and the addresses this build writes.
	// A streamed surface's residency goes with it (D92); without a hierarchy it traces its coarsest cut.
	if (!p_omm && cluster_lod_select_enabled && layout.levels > 1) {
		r_surf_data->cluster_lod_stream = stream;
		_populate_surface_cluster_lod(p_mesh_surface, layout, args_buffer, implicit_addresses_offset, p_cache_key, r_surf_data);
		if (stream && r_surf_data->cluster_lod_table_offset == UINT32_MAX) {
			WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: a streamed surface's hierarchy could not be placed; it traces its coarsest level.", p_cache_key));
			r_surf_data->cluster_lod_stream = nullptr;
			memdelete(stream);
		}
	}

	cluster_blas_builds++;
	cluster_triangles_built += total_cluster_triangles;
	if (p_omm) {
		print_verbose(vformat("RT: cluster BLAS [%d] with opacity micromap %d (D96 OMMd): %d triangles in %d LOD 0 clusters, %d KB staging, %d KB arguments.", p_cache_key, p_omm->serial, primitive_count, cluster_count, staging_size / 1024, args_size / 1024));
	} else if (layout.levels > 1) {
		print_verbose(vformat("RT: cluster BLAS [%d]: %d triangles in %d clusters over %d LOD levels (%d clusters, %d triangles at LOD 0) of at most %d triangles and %d vertices, %d KB staging, %d KB arguments.", p_cache_key, layout.total_triangles, cluster_count, layout.levels, lod0_cluster_count, primitive_count, max_cluster_triangles, max_cluster_vertices, staging_size / 1024, args_size / 1024));
	} else {
		print_verbose(vformat("RT: cluster BLAS [%d]: %d triangles in %d %s of at most %d triangles and %d vertices, %d KB staging, %d KB arguments.", p_cache_key, primitive_count, cluster_count, from_table ? "clusters" : "runs", max_cluster_triangles, max_cluster_vertices, staging_size / 1024, args_size / 1024));
	}
	return true;
}

// The clusters of a surface: each a contiguous range of the index buffer, its ID its first
// triangle, which is what the hit shaders add to gl_PrimitiveID (rt_resolve_triangle). A surface
// created with the setting on carries its table (D80): spatial clusters with the count of
// distinct vertices each reaches. Without one, or with one that does not describe this index
// buffer, the clusters are runs of T triangles (D79), which reach at most 3T distinct vertices,
// T chosen so that bound fits p_max_vertices. Shared by the static and the deformed path (D82).
void RenderRaytracing::_cluster_layout_for_surface(void *p_mesh_surface, uint32_t p_primitive_count, uint32_t p_max_triangles, uint32_t p_max_vertices, uint32_t p_cache_key, RTClusterLayout &r_layout) {
	r_layout.triangles.clear();
	r_layout.vertices.clear();
	r_layout.max_triangles = 0;
	r_layout.max_vertices = 0;
	r_layout.total_vertices = 0;
	r_layout.total_triangles = 0;
	r_layout.lod0_clusters = 0;
	r_layout.lod0_vertices = 0;
	r_layout.levels = 1;
	r_layout.from_table = false;
	{
		const Vector<uint8_t> &table_bytes = RendererRD::MeshStorage::get_singleton()->mesh_surface_get_cluster_data(p_mesh_surface);
		const uint32_t words = table_bytes.size() / sizeof(uint32_t);
		if (words >= 2 && table_bytes.size() % sizeof(uint32_t) == 0) {
			const uint32_t *table = reinterpret_cast<const uint32_t *>(table_bytes.ptr());
			if (table[0] == RenderingServerTypes::MESH_CLUSTER_DATA_VERSION) {
				r_layout.triangles.reserve(words - 1);
				r_layout.vertices.reserve(words - 1);
				uint32_t sum = 0;
				bool valid = true;
				for (uint32_t c = 1; c < words && valid; c++) {
					const uint32_t triangles = table[c] & 0xFFFFu;
					const uint32_t vertices = table[c] >> 16;
					valid = triangles > 0 && triangles <= p_max_triangles && vertices > 0 && vertices <= p_max_vertices;
					r_layout.triangles.push_back(triangles);
					r_layout.vertices.push_back(vertices);
					r_layout.max_triangles = MAX(r_layout.max_triangles, triangles);
					r_layout.max_vertices = MAX(r_layout.max_vertices, vertices);
					r_layout.total_vertices += vertices;
					sum += triangles;
				}
				r_layout.from_table = valid && sum == p_primitive_count;
				r_layout.total_triangles = sum;
				r_layout.lod0_clusters = r_layout.triangles.size();
				r_layout.lod0_vertices = r_layout.total_vertices;
			} else if (table[0] == RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION && words >= RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS) {
				// A cluster LOD hierarchy (D85): the LOD 0 clusters first, summing to the drawn
				// triangles, then the coarse levels, summing to the header's coarse count. The
				// coarse levels are taken only when the mesh storage appended them to the index
				// buffer (the buffer holds every level's indices); otherwise the LOD 0 clusters
				// are the surface's flat clustering.
				const uint32_t cluster_count = table[1];
				const uint32_t group_count = table[2];
				const uint32_t lod0_clusters = table[4];
				const uint32_t coarse_triangles = table[6];
				const uint64_t expected_words = RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS + uint64_t(RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS) * cluster_count + uint64_t(RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS) * group_count + uint64_t(coarse_triangles) * 3;
				bool valid = words == expected_words && lod0_clusters > 0 && lod0_clusters <= cluster_count && table[5] == p_primitive_count;
				// A table marked stale (its positions updated in place since it was built, D86) is
				// taken as its LOD 0 clusters: the coarse levels no longer describe the surface.
				const bool stale = (table[7] & RenderingServerTypes::MESH_CLUSTER_LOD_FLAG_STALE) != 0;
				const bool tailed = valid && !stale && coarse_triangles > 0 && uint64_t(RendererRD::MeshStorage::get_singleton()->mesh_surface_get_rt_index_total(p_mesh_surface)) >= (uint64_t(table[5]) + coarse_triangles) * 3;
				const uint32_t taken = tailed ? cluster_count : lod0_clusters;
				if (valid) {
					r_layout.triangles.reserve(taken);
					r_layout.vertices.reserve(taken);
					const uint32_t *cluster_words = table + RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS;
					uint32_t lod0_sum = 0;
					uint32_t coarse_sum = 0;
					for (uint32_t c = 0; c < taken && valid; c++) {
						const uint32_t packed = cluster_words[uint64_t(c) * RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS];
						const uint32_t triangles = packed & 0xFFFFu;
						const uint32_t vertices = packed >> 16;
						valid = triangles > 0 && triangles <= p_max_triangles && vertices > 0 && vertices <= p_max_vertices;
						r_layout.triangles.push_back(triangles);
						r_layout.vertices.push_back(vertices);
						r_layout.max_triangles = MAX(r_layout.max_triangles, triangles);
						r_layout.max_vertices = MAX(r_layout.max_vertices, vertices);
						r_layout.total_vertices += vertices;
						if (c < lod0_clusters) {
							lod0_sum += triangles;
							r_layout.lod0_vertices += vertices;
						} else {
							coarse_sum += triangles;
						}
					}
					valid = valid && lod0_sum == p_primitive_count && (!tailed || coarse_sum == coarse_triangles);
					r_layout.total_triangles = lod0_sum + coarse_sum;
					r_layout.lod0_clusters = lod0_clusters;
					r_layout.levels = tailed ? MAX(1u, table[3]) : 1u;
				}
				r_layout.from_table = valid;
			}
			if (!r_layout.from_table) {
				WARN_PRINT_ONCE(vformat("RT: surface [%d] carries a cluster table that does not describe its index buffer; clustered as runs instead.", p_cache_key));
			}
		}
	}
	if (!r_layout.from_table) {
		const uint32_t triangles_per_cluster = MAX(1u, MIN(p_max_triangles, p_max_vertices / 3u));
		const uint32_t run_count = (p_primitive_count + triangles_per_cluster - 1) / triangles_per_cluster;
		r_layout.triangles.clear();
		r_layout.vertices.clear();
		r_layout.triangles.reserve(run_count);
		r_layout.vertices.reserve(run_count);
		r_layout.max_triangles = 0;
		r_layout.max_vertices = 0;
		r_layout.total_vertices = 0;
		for (uint32_t c = 0; c < run_count; c++) {
			const uint32_t triangles = MIN(triangles_per_cluster, p_primitive_count - c * triangles_per_cluster);
			const uint32_t vertices = MIN(triangles * 3u, p_max_vertices);
			r_layout.triangles.push_back(triangles);
			r_layout.vertices.push_back(vertices);
			r_layout.max_triangles = MAX(r_layout.max_triangles, triangles);
			r_layout.max_vertices = MAX(r_layout.max_vertices, vertices);
			r_layout.total_vertices += vertices;
		}
		r_layout.total_triangles = p_primitive_count;
		r_layout.lod0_clusters = run_count;
		r_layout.lod0_vertices = r_layout.total_vertices;
		r_layout.levels = 1;
	}
}

// ---------------------------------------------------------------------------
// The deformed cluster path (D82)
// ---------------------------------------------------------------------------

void RenderRaytracing::_release_deformed_cluster_storage(RTDeformedCacheEntry &p_entry) {
	_release_deformed_blas(p_entry);
	if (p_entry.cluster_storage.is_valid()) {
		RD::get_singleton()->free_rid(p_entry.cluster_storage);
		p_entry.cluster_storage = RID();
	}
	p_entry.cluster_storage_capacity = 0;
	p_entry.cluster_index_buffer = RID();
}

// The templates of a deformed mesh surface (D107): one a LOD 0 cluster of the layout, keyed by the
// surface's index buffer, whose RID no other surface ever takes. They hold the clusters' triangles
// and indices; positions come with each instantiation. Built at explicit addresses, so that every
// entry's instantiation infos are written once, by the next flush, which reads the first entry's
// positions as the build's hint. No instantiation bounding box is given: the CLAS are exact for any
// pose, and a box's tighter traversal is an open item.
RTDeformedClusterTemplate *RenderRaytracing::_deformed_cluster_template_acquire(RID p_index_buffer, uint32_t p_surface_counter, RID p_vertex_buffer, const RTClusterLayout &p_layout, uint32_t p_cluster_count, uint64_t p_index_address, uint32_t p_index_format, uint64_t p_vertex_address, uint32_t p_vertex_stride, uint64_t &r_key) {
	// One set a version of the surface: an index-region update keeps the buffer and bumps the
	// counter, and the old set leaves with its last holder.
	const uint64_t key = (uint64_t(hash_murmur3_one_64(p_index_buffer.get_id())) << 32) | p_surface_counter;
	r_key = key;
	RTDeformedClusterTemplate *found = deformed_cluster_templates.getptr(key);
	if (found) {
		// A request for fewer clusters is the same version's LOD 0, the first of a per-level set (D108),
		// as after a failed selection flush took the hierarchies off.
		if (found->index_buffer != p_index_buffer || found->surface_counter != p_surface_counter || found->cluster_count < p_cluster_count) {
			return nullptr; // A colliding key, or another layout of the same version: built per frame, as D82.
		}
		found->refs++;
		return found;
	}

	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	uint32_t total_triangles = 0, total_vertices = 0;
	for (uint32_t c = 0; c < p_cluster_count; c++) {
		total_triangles += p_layout.triangles[c];
		total_vertices += p_layout.vertices[c];
	}
	RD::ClusterAccelerationStructureInput input;
	input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_TRIANGLE_CLUSTER_TEMPLATE;
	input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
	input.max_acceleration_structure_count = p_cluster_count;
	input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT); // Built once, traced through every instantiation.
	input.vertex_format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
	input.max_cluster_triangle_count = p_layout.max_triangles;
	input.max_cluster_vertex_count = p_layout.max_vertices;
	input.max_total_triangle_count = total_triangles;
	input.max_total_vertex_count = total_vertices;
	uint64_t template_size = 0, scratch = 0;
	rd->cluster_acceleration_structure_get_build_sizes(input, template_size, scratch);
	// The worst case of one CLAS instantiated from them, the stride of an entry's CLAS.
	RD::ClusterAccelerationStructureInput instantiate_input = input;
	instantiate_input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_INSTANTIATE_TRIANGLE_CLUSTER;
	instantiate_input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT);
	uint64_t clas_size = 0;
	rd->cluster_acceleration_structure_get_build_sizes(instantiate_input, clas_size, scratch);
	// The worst case under the templates' own flags too, the larger taken: an instantiation's size
	// follows its template, whose build flags differ from the instantiation's.
	uint64_t clas_size_template_flags = 0;
	instantiate_input.flags = input.flags;
	rd->cluster_acceleration_structure_get_build_sizes(instantiate_input, clas_size_template_flags, scratch);
	if (clas_size_template_flags != clas_size) {
		print_verbose(vformat("RT: deformed CLAS instantiated from templates: %d bytes at worst under the instantiation's flags, %d under the templates'; the larger taken (D107).", clas_size, clas_size_template_flags));
	}
	clas_size = MAX(clas_size, clas_size_template_flags);
	if (template_size == 0 || clas_size == 0) {
		WARN_PRINT_ONCE("RT: the driver sized a deformed surface's cluster templates at 0 bytes; deformed surfaces build their CLAS each frame (D82).");
		deformed_cluster_templates_enabled = false;
		return nullptr;
	}
	const uint64_t alignment = caps.cluster_template_byte_alignment;
	const uint64_t stride = _align_up(template_size, alignment);
	const uint64_t storage_size = alignment + stride * p_cluster_count;
	ERR_FAIL_COND_V_MSG(storage_size > UINT32_MAX, nullptr, "RT: a deformed surface's cluster templates exceed 4 GB.");
	RTDeformedClusterTemplate t;
	t.storage = rd->storage_buffer_create(uint32_t(storage_size), {}, 0, RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_STORAGE_BIT);
	ERR_FAIL_COND_V_MSG(t.storage.is_null(), nullptr, "RT: a deformed surface's cluster template storage could not be created.");
	rd->set_resource_name(t.storage, "RT deformed cluster templates [" + itos(key) + "]");
	t.base_address = _align_up(rd->buffer_get_device_address(t.storage), alignment);
	t.stride = uint32_t(stride);
	t.cluster_count = p_cluster_count;
	t.refs = 1;
	t.index_buffer = p_index_buffer;
	t.surface_counter = p_surface_counter;
	t.vertex_buffer = p_vertex_buffer;
	t.max_triangles = p_layout.max_triangles;
	t.max_vertices = p_layout.max_vertices;
	t.total_triangles = total_triangles;
	t.total_vertices = total_vertices;
	t.clas_size = clas_size;
	// The build infos, as the D82 layout's CLAS infos: each cluster's ID its first triangle.
	const uint32_t index_size = p_index_format == RT_INDEX_FORMAT_UINT16 ? sizeof(uint16_t) : sizeof(uint32_t);
	const uint32_t info_stride = caps.triangle_template_build_info_size;
	t.build_infos.resize(uint64_t(info_stride) * p_cluster_count);
	uint32_t first_triangle = 0;
	for (uint32_t c = 0; c < p_cluster_count; c++) {
		RD::ClusterTriangleBuildInfo info;
		info.cluster_id = first_triangle;
		info.triangle_count = p_layout.triangles[c];
		info.vertex_count = p_layout.vertices[c];
		info.index_type = p_index_format == RT_INDEX_FORMAT_UINT16 ? RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT16 : RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT32;
		info.index_address = p_index_address + uint64_t(first_triangle) * 3u * index_size;
		info.index_stride = index_size;
		info.vertex_address = p_vertex_address;
		info.vertex_stride = p_vertex_stride;
		rd->cluster_triangle_template_build_info_write(t.build_infos.ptr() + uint64_t(c) * info_stride, info);
		first_triangle += p_layout.triangles[c];
	}
	deformed_cluster_batch.pending_templates.push_back(key);
	return &deformed_cluster_templates.insert(key, t)->value;
}

void RenderRaytracing::_deformed_cluster_template_release(uint64_t p_key) {
	RTDeformedClusterTemplate *t = deformed_cluster_templates.getptr(p_key);
	ERR_FAIL_NULL(t);
	if (--t->refs > 0) {
		return;
	}
	if (t->storage.is_valid()) {
		RD::get_singleton()->free_rid(t->storage); // Deferred past the fence of every frame in flight.
	}
	deformed_cluster_batch.pending_templates.erase(p_key);
	deformed_cluster_templates.erase(p_key);
}

// The templates made since the last flush (D107), each its own build, recorded before the frame's
// instantiations that read them. Returns false when one could not be recorded.
bool RenderRaytracing::_build_deformed_cluster_templates() {
	RTDeformedClusterBatch &batch = deformed_cluster_batch;
	if (batch.pending_templates.is_empty()) {
		return true;
	}
	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	const uint32_t info_stride = caps.triangle_template_build_info_size;
	for (const uint64_t key : batch.pending_templates) {
		RTDeformedClusterTemplate *t = deformed_cluster_templates.getptr(key);
		if (t == nullptr || t->built) {
			continue;
		}
		// Its arguments, used once: the infos, then the explicit addresses.
		const uint64_t addresses_offset = _align_up(uint64_t(info_stride) * t->cluster_count, 16);
		const uint64_t args_size = addresses_offset + sizeof(uint64_t) * t->cluster_count;
		ERR_FAIL_COND_V(args_size > UINT32_MAX, false);
		Vector<uint8_t> args_data;
		args_data.resize(args_size);
		uint8_t *args = args_data.ptrw();
		memset(args, 0, args_size);
		memcpy(args, t->build_infos.ptr(), t->build_infos.size());
		for (uint32_t c = 0; c < t->cluster_count; c++) {
			reinterpret_cast<uint64_t *>(args + addresses_offset)[c] = t->base_address + uint64_t(c) * t->stride;
		}
		const RID args_buffer = rd->storage_buffer_create(uint32_t(args_size), args_data, 0, RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
		ERR_FAIL_COND_V_MSG(args_buffer.is_null(), false, "RT: a deformed surface's cluster template arguments could not be created.");

		RD::ClusterAccelerationStructureBuild build;
		build.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_TRIANGLE_CLUSTER_TEMPLATE;
		build.input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
		build.input.max_acceleration_structure_count = t->cluster_count;
		build.input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
		build.input.vertex_format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
		build.input.max_cluster_triangle_count = t->max_triangles;
		build.input.max_cluster_vertex_count = t->max_vertices;
		build.input.max_total_triangle_count = t->total_triangles;
		build.input.max_total_vertex_count = t->total_vertices;
		build.dst_storage_buffer = t->storage;
		build.dst_addresses_buffer = args_buffer;
		build.dst_addresses_offset = addresses_offset;
		build.dst_addresses_stride = sizeof(uint64_t);
		build.src_infos_buffer = args_buffer;
		build.src_infos_stride = info_stride;
		build.source_buffers.push_back(t->index_buffer);
		build.source_buffers.push_back(t->vertex_buffer);
		const Error err = rd->cluster_acceleration_structure_build(build);
		rd->free_rid(args_buffer); // Deferred past the fence of every frame in flight.
		if (err != OK) {
			return false;
		}
		t->built = true;
		t->build_infos.clear();
		t->vertex_buffer = RID();
		deformed_cluster_templates_built_total++;
		frame_extra_builds++; // A build in the counters (D97), as the other builds outside the walk.
		frame_extra_triangles += t->total_triangles;
		print_verbose(vformat("RT: deformed cluster templates [%s]: %d clusters, %d triangles, %d KB, %d bytes a CLAS instantiated at worst (D107); %d template sets built so far.", String::num_uint64(key, 16), t->cluster_count, t->total_triangles, (uint64_t(t->stride) * t->cluster_count) / 1024, t->clas_size, deformed_cluster_templates_built_total));
	}
	batch.pending_templates.clear();
	return true;
}

bool RenderRaytracing::_populate_deformed_cluster_layout(void *p_mesh_surface, uint32_t p_cache_key, uint32_t p_surface_counter, RTDeformedCacheEntry &r_entry, const RTOmmEntry *p_omm, bool *r_omm_attached) {
	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	RTSurfaceData *surf_data = r_entry.ptr;

	RID vertex_buffer, attribute_buffer, index_buffer;
	_fill_surface_geometry_data(p_mesh_surface, true, surf_data, &vertex_buffer, &attribute_buffer, &index_buffer);
	RT_GeometryData &geom = surf_data->geometry;
	geom.vertex_buffer_address = rd->buffer_get_device_address(r_entry.owned_vb_full);
	// The mesh's own attribute and index buffers are read in place, as for a static surface.
	if (attribute_buffer.is_valid()) {
		geom.attribute_buffer_address = rd->buffer_get_device_address(attribute_buffer);
		rd->buffer_add_device_address_read_stages(attribute_buffer, RDD::PIPELINE_STAGE_RAY_TRACING_SHADER_BIT);
	}
	if (index_buffer.is_valid() && geom.index_format != RT_INDEX_FORMAT_NONE) {
		geom.index_buffer_address = rd->buffer_get_device_address(index_buffer);
		rd->buffer_add_device_address_read_stages(index_buffer, RDD::PIPELINE_STAGE_RAY_TRACING_SHADER_BIT);
	}

	const bool is_2d = RendererRD::MeshStorage::get_singleton()->mesh_surface_get_format(p_mesh_surface) & RSE::ARRAY_FLAG_USE_2D_VERTICES;
	const uint32_t primitive_count = geom.primitive_count;
	uint32_t max_triangles = 0, max_vertices = 0;
	_deformed_cluster_maxima(caps, max_triangles, max_vertices);
	if (is_2d || geom.index_format == RT_INDEX_FORMAT_NONE || primitive_count == 0 || geom.vertex_buffer_address == 0 || geom.index_buffer_address == 0 || max_triangles == 0 || max_vertices < 3) {
		print_verbose(vformat("RT: deformed surface [%d] on the triangle BLAS path (2D %d, indexed %d, %d triangles).", p_cache_key, int(is_2d), int(geom.index_format != RT_INDEX_FORMAT_NONE), primitive_count));
		return false;
	}

	RTClusterLayout layout;
	_cluster_layout_for_surface(p_mesh_surface, primitive_count, max_triangles, max_vertices, p_cache_key, layout);
	// The batch rebuilds the LOD 0 clusters alone (D82); with the hierarchy on the GPU (D87) the
	// selection builds the clusters it picks from every level, so the BLAS storage is sized for
	// the count over every level, the bound of any selection.
	const uint32_t cluster_count = layout.lod0_clusters;
	// A micromap (D96 OMMd) is baked over the LOD 0 triangles: its CLAS are LOD 0's alone, as the batch builds them.
	uint64_t omm_address = 0, omm_index_address = 0;
	const bool omm = p_omm && _omm_cluster_addresses(p_omm, primitive_count, omm_address, omm_index_address);
	bool lod = !omm && layout.levels > 1 && cluster_lod_deformed_enabled && cluster_lod_clas_slot_size > 0;
	_release_deformed_cluster_lod(r_entry);
	r_entry.cluster_omm = false; // Set with the layout below.

	// The surface's templates where the entry needs nothing they cannot carry, a micromap being the
	// entry's own (OMMd): one a LOD 0 cluster for the batch (D107), or with the hierarchy one a cluster
	// of every level (D108), whose first LOD 0 count serve the batch too. They belong to this version
	// of the surface (an index-region update keeps the buffer and bumps the counter), and are held
	// from here, so that every failure below releases them with the layout. A hierarchy whose
	// templates cannot be had is traced at LOD 0, since the selection instantiates every entry's CLAS
	// when the device has templates.
	RTDeformedClusterTemplate *cluster_template = nullptr;
	if (!omm && index_buffer.is_valid() && (lod ? cluster_lod_deformed_templated : deformed_cluster_templates_enabled)) {
		uint64_t template_key = 0;
		cluster_template = _deformed_cluster_template_acquire(index_buffer, p_surface_counter, r_entry.owned_vb_full, layout, lod ? layout.triangles.size() : cluster_count, geom.index_buffer_address, geom.index_format, geom.vertex_buffer_address, geom.position_stride, template_key);
		if (cluster_template) {
			r_entry.cluster_templated = true;
			r_entry.cluster_template_key = template_key;
		}
	}
	if (lod && cluster_lod_deformed_templated && !cluster_template) {
		lod = false; // The selection instantiates every hierarchy's CLAS: LOD 0 through the batch instead.
	}
	const uint32_t infos_count = lod ? layout.triangles.size() : cluster_count;

	// The entry's storage holds the BLAS, at an explicit address the size query bounds for this
	// cluster count, and, templated, its CLAS slots after it: the batch's LOD 0 (D107) and the
	// selection's picks (D109). Untemplated, its CLAS are built into the frame's region, by the batch
	// with implicit destinations (_flush_deformed_cluster_batch, as the static path builds its own,
	// D81) or by the selection into explicit slots (D87).
	RD::ClusterAccelerationStructureInput blas_input;
	blas_input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_CLUSTERS_BOTTOM_LEVEL;
	blas_input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
	blas_input.max_acceleration_structure_count = 1;
	blas_input.flags = _cluster_build_flags_omm(RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT, omm); // As the batch builds it.
	blas_input.max_total_cluster_count = infos_count;
	blas_input.max_cluster_count_per_acceleration_structure = infos_count;
	uint64_t blas_size = 0, blas_scratch = 0;
	rd->cluster_acceleration_structure_get_build_sizes(blas_input, blas_size, blas_scratch);
	if (blas_size == 0) {
		print_verbose(vformat("RT: deformed surface [%d] on the triangle BLAS path: the driver sized the BLAS at 0 bytes.", p_cache_key));
		_release_deformed_blas(r_entry); // Its templates, if it took them; it has no BLAS yet.
		return false;
	}

	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	// A templated entry's CLAS follow its BLAS, one worst-case instantiation apart (D107): its LOD 0
	// clusters, or with the hierarchy the slots its selection keeps there (D109), as many as any cut.
	const uint64_t clas_stride = cluster_template ? _align_up(cluster_template->clas_size, uint64_t(caps.cluster_byte_alignment)) : 0;
	const uint32_t slot_capacity = cluster_template && lod ? _deformed_cluster_cut_capacity(p_mesh_surface, cluster_count, infos_count) : cluster_count;
	const uint64_t storage_size = base_alignment + _align_up(blas_size, base_alignment) + clas_stride * slot_capacity;
	if (storage_size > UINT32_MAX) {
		_release_deformed_cluster_storage(r_entry);
		ERR_FAIL_V_MSG(false, "RT: a deformed cluster BLAS's storage exceeds 4 GB.");
	}
	if (r_entry.cluster_storage_capacity < storage_size) {
		if (r_entry.cluster_storage.is_valid()) {
			rd->free_rid(r_entry.cluster_storage);
			r_entry.cluster_storage = RID();
		}
		r_entry.cluster_storage_capacity = 0;
		r_entry.cluster_storage = rd->storage_buffer_create(uint32_t(storage_size), {}, 0,
				RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_STORAGE_BIT);
		if (r_entry.cluster_storage.is_null()) {
			_release_deformed_cluster_storage(r_entry);
			return false;
		}
		r_entry.cluster_storage_capacity = storage_size;
		rd->set_resource_name(r_entry.cluster_storage, "RT deformed cluster BLAS storage [" + itos(p_cache_key) + "]");
		// The trace and the ray queries read it through the TLAS: a later layout's first write under
		// a new handle, which no tracing list declared, waits for them too (D107).
		rd->buffer_add_device_address_read_stages(r_entry.cluster_storage, RDD::PIPELINE_STAGE_RAY_TRACING_SHADER_BIT | RDD::PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	}
	const uint64_t buffer_address = rd->buffer_get_device_address(r_entry.cluster_storage);
	const uint64_t base_address = _align_up(buffer_address, base_alignment);
	RID blas = rd->cluster_blas_create(r_entry.cluster_storage, base_address - buffer_address);
	if (blas.is_null()) {
		_release_deformed_cluster_storage(r_entry);
		return false;
	}

	// The CLAS infos, written once: the batch keeps LOD 0's, the table every level's. Templated, the
	// instantiation of each cluster's template from the entry's positions (D107, D108); else its build
	// info, the positions from the entry's owned vertex buffer, the indices from the mesh's shared
	// index buffer, each cluster's ID its first triangle (the coarse levels' past the drawn range, D85),
	// which the templates carry too.
	const uint32_t index_size = geom.index_format == RT_INDEX_FORMAT_UINT16 ? sizeof(uint16_t) : sizeof(uint32_t);
	const uint32_t clas_info_stride = cluster_template ? caps.instantiate_info_size : caps.triangle_build_info_size;
	LocalVector<uint8_t> all_infos;
	all_infos.resize(uint64_t(clas_info_stride) * infos_count);
	uint32_t first_triangle = 0;
	if (cluster_template) {
		for (uint32_t c = 0; c < infos_count; c++) {
			RD::ClusterInstantiateInfo info;
			info.template_address = cluster_template->base_address + uint64_t(c) * cluster_template->stride;
			info.vertex_address = geom.vertex_buffer_address;
			info.vertex_stride = geom.position_stride;
			rd->cluster_instantiate_info_write(all_infos.ptr() + uint64_t(c) * clas_info_stride, info);
		}
	} else {
		for (uint32_t c = 0; c < infos_count; c++) {
			const uint32_t triangle_count = layout.triangles[c];
			RD::ClusterTriangleBuildInfo info;
			info.cluster_id = first_triangle;
			info.triangle_count = triangle_count;
			info.vertex_count = layout.vertices[c];
			info.index_type = geom.index_format == RT_INDEX_FORMAT_UINT16 ? RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT16 : RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT32;
			info.index_address = geom.index_buffer_address + uint64_t(first_triangle) * 3u * index_size;
			info.index_stride = index_size;
			info.vertex_address = geom.vertex_buffer_address;
			info.vertex_stride = geom.position_stride;
			if (omm) {
				info.opacity_micromap_address = omm_address;
				info.opacity_micromap_index_address = omm_index_address + uint64_t(first_triangle) * sizeof(int32_t);
				info.opacity_micromap_index_type = RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT32;
				info.opacity_micromap_index_stride = sizeof(int32_t);
			}
			rd->cluster_triangle_build_info_write(all_infos.ptr() + uint64_t(c) * clas_info_stride, info);
			first_triangle += triangle_count;
		}
	}
	r_entry.cluster_infos.resize(uint64_t(clas_info_stride) * cluster_count);
	memcpy(r_entry.cluster_infos.ptr(), all_infos.ptr(), r_entry.cluster_infos.size());
	if (cluster_template) {
		// The batch instantiates its LOD 0 CLAS into the entry's own storage (D107), and the
		// selection its picks into the same slots (D109).
		r_entry.cluster_clas_address = base_address + _align_up(blas_size, base_alignment);
		r_entry.cluster_clas_stride = uint32_t(clas_stride);
	}
	r_entry.cluster_lod_slot_capacity = slot_capacity;

	surf_data->blas = blas;
	surf_data->blas_size = storage_size;
	rd->set_resource_name(blas, "RT deformed cluster BLAS [" + itos(p_cache_key) + "]");
	r_entry.cluster_blas_address = base_address;
	r_entry.cluster_count = cluster_count;
	r_entry.cluster_max_triangles = layout.max_triangles;
	r_entry.cluster_max_vertices = layout.max_vertices;
	r_entry.cluster_total_vertices = layout.lod0_vertices;
	r_entry.cluster_index_buffer = index_buffer;
	r_entry.cluster_omm = omm;
	r_entry.cluster_built_frame = 0;
	r_entry.cluster_lod_last = false;
	if (lod) {
		_populate_deformed_cluster_lod(p_mesh_surface, layout, all_infos, clas_info_stride, p_cache_key, r_entry);
	}
	if (r_omm_attached) {
		*r_omm_attached = omm;
	}
	if (layout.levels > 1) {
		print_verbose(vformat("RT: deformed cluster BLAS [%d]: %d triangles in %d clusters at LOD 0 of %d over %d levels, of at most %d triangles and %d vertices, %d KB BLAS storage, %s.", p_cache_key, primitive_count, cluster_count, layout.triangles.size(), layout.levels, layout.max_triangles, layout.max_vertices, storage_size / 1024, r_entry.cluster_lod_table_offset != UINT32_MAX ? vformat("selecting from %d words of the LOD table%s", r_entry.cluster_lod_table_words, cluster_template ? vformat(", instantiated from templates (D108) into %d slots of its own (D109, %s)", slot_capacity, slot_capacity == cluster_count ? "every cut within LOD 0" : "a cut can pass LOD 0") : "") : "LOD 0 traced"));
	} else {
		print_verbose(vformat("RT: deformed cluster BLAS [%d]: %d triangles in %d %s of at most %d triangles and %d vertices, %d KB BLAS storage%s.", p_cache_key, primitive_count, cluster_count, layout.from_table ? "clusters" : "runs", layout.max_triangles, layout.max_vertices, storage_size / 1024, cluster_template ? vformat(" with its CLAS, instantiated from templates (D107, %d bytes a CLAS at worst)", clas_stride) : String()));
	}
	return true;
}

// A cut of a hierarchy (D85) replaces whole groups: a coarser cluster is taken only where the group it
// was simplified from is not, so a cut can be turned into LOD 0 by putting back, coarsest first, the
// group each taken cluster replaces, and that never shrinks it where no group is replaced by more
// clusters than it has. Where that holds, LOD 0's count bounds every cut (D109).
uint32_t RenderRaytracing::_deformed_cluster_cut_capacity(void *p_mesh_surface, uint32_t p_lod0_clusters, uint32_t p_all_clusters) {
	const Vector<uint8_t> &table_bytes = RendererRD::MeshStorage::get_singleton()->mesh_surface_get_cluster_data(p_mesh_surface);
	if (table_bytes.size() < int64_t(RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS * sizeof(uint32_t))) {
		return p_all_clusters;
	}
	const uint32_t *table = reinterpret_cast<const uint32_t *>(table_bytes.ptr());
	const uint32_t cluster_count = table[1];
	const uint32_t group_count = table[2];
	if (table[0] != RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION || cluster_count != p_all_clusters || group_count == 0 ||
			uint64_t(table_bytes.size()) < sizeof(uint32_t) * (RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS + uint64_t(RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS) * cluster_count)) {
		return p_all_clusters;
	}
	const uint32_t *cluster_words = table + RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS;
	LocalVector<uint32_t> members;
	LocalVector<uint32_t> replacements;
	members.resize_initialized(group_count);
	replacements.resize_initialized(group_count);
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t *w = cluster_words + uint64_t(c) * RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
		const int32_t refined = int32_t(w[2]);
		if (w[1] >= group_count || refined >= int32_t(group_count)) {
			return p_all_clusters;
		}
		members[w[1]]++;
		if (refined >= 0) {
			replacements[refined]++;
		}
	}
	for (uint32_t g = 0; g < group_count; g++) {
		if (replacements[g] > members[g]) {
			deformed_cut_bound_failures_total++;
			return p_all_clusters;
		}
	}
	return p_lod0_clusters;
}

// The entry's hierarchy as the selection reads it (D87): its region of the LOD table, laid out
// as a static surface's (the address words zero, since its CLAS are made each frame; each
// cluster's group and refined group; the groups' spheres and errors) with every level's CLAS
// info after them, a build info or, templated, an instantiation info (D108), which the selection
// copies to the frame's build for the clusters it picks. On failure the entry keeps no region
// and the batch traces its LOD 0 clusters.
bool RenderRaytracing::_populate_deformed_cluster_lod(void *p_mesh_surface, const RTClusterLayout &p_layout, const LocalVector<uint8_t> &p_infos, uint32_t p_info_size, uint32_t p_cache_key, RTDeformedCacheEntry &r_entry) {
	RD *rd = RD::get_singleton();
	const uint32_t info_size = p_info_size; // A build info, or an instantiation info when templated (D108).
	const uint32_t info_words = info_size / sizeof(uint32_t);
	const Vector<uint8_t> &table_bytes = RendererRD::MeshStorage::get_singleton()->mesh_surface_get_cluster_data(p_mesh_surface);
	const uint32_t cluster_count = p_layout.triangles.size();
	if (table_bytes.size() < int64_t(RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS * sizeof(uint32_t)) || info_words * sizeof(uint32_t) != info_size || p_infos.size() != info_size * cluster_count) {
		return false;
	}
	const uint32_t *table = reinterpret_cast<const uint32_t *>(table_bytes.ptr());
	const uint32_t group_count = table[2];
	if (table[0] != RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION || table[1] != cluster_count || group_count == 0) {
		return false; // The layout took every level of this very table (levels > 1), so the counts agree.
	}
	const uint32_t *cluster_words = table + RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS;
	const uint32_t *group_words = cluster_words + uint64_t(RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS) * cluster_count;
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t *w = cluster_words + uint64_t(c) * RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
		const int32_t refined = int32_t(w[2]);
		if (w[1] >= group_count || refined >= int32_t(group_count)) {
			WARN_PRINT_ONCE(vformat("RT: deformed cluster BLAS [%d]: the hierarchy's links leave its groups; traced at LOD 0.", p_cache_key));
			return false;
		}
	}
	const uint32_t GROUP_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS;
	// The traversal's tree (D91), after the build infos; without one the selection tests every cluster.
	LocalVector<uint32_t> tree_words;
	LocalVector<uint32_t> tree_levels;
	uint32_t tree_nodes = 0;
	if (cluster_lod_traversal_enabled && !_build_cluster_lod_tree(cluster_words, cluster_count, group_words, group_count, tree_words, tree_levels, tree_nodes)) {
		tree_words.clear();
		WARN_PRINT_ONCE(vformat("RT: deformed cluster BLAS [%d]: the hierarchy's groups do not own contiguous runs of its clusters; its selection tests every cluster.", p_cache_key));
	}
	// Then the last selection's digest (D109): its count and two hashes, zero until the first selection.
	const uint32_t DIGEST_WORDS = 3;
	const uint64_t words64 = uint64_t(cluster_count) * 4 + uint64_t(group_count) * GROUP_WORDS + uint64_t(cluster_count) * info_words + tree_words.size() + DIGEST_WORDS;
	ERR_FAIL_COND_V_MSG(words64 > UINT32_MAX / 4, false, "RT: a deformed cluster LOD hierarchy's region exceeds the table's reach.");
	const uint32_t words = uint32_t(words64);
	uint32_t offset = 0;
	if (!_cluster_lod_table_allocate(words, offset)) {
		WARN_PRINT_ONCE(vformat("RT: deformed cluster BLAS [%d]: no room in the cluster LOD table; traced at LOD 0.", p_cache_key));
		return false;
	}
	LocalVector<uint32_t> host;
	host.resize(words);
	memset(host.ptr(), 0, cluster_count * 2 * sizeof(uint32_t));
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t *w = cluster_words + uint64_t(c) * RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
		host[cluster_count * 2 + c * 2] = w[1];
		host[cluster_count * 2 + c * 2 + 1] = w[2];
	}
	for (uint32_t g = 0; g < group_count; g++) {
		const uint32_t *w = group_words + uint64_t(g) * GROUP_WORDS;
		uint32_t *out = host.ptr() + cluster_count * 4 + uint64_t(g) * GROUP_WORDS;
		// The table's (depth, first cluster, cluster count, center x y z, radius, error) as
		// (center x y z, radius, error, depth, first cluster, cluster count), as the static region.
		out[0] = w[3];
		out[1] = w[4];
		out[2] = w[5];
		out[3] = w[6];
		out[4] = w[7];
		out[5] = w[0];
		out[6] = w[1];
		out[7] = w[2];
	}
	const uint32_t infos_offset = cluster_count * 4 + group_count * GROUP_WORDS;
	memcpy(host.ptr() + infos_offset, p_infos.ptr(), p_infos.size());
	const uint32_t tree_offset = infos_offset + cluster_count * info_words;
	if (!tree_words.is_empty()) {
		memcpy(host.ptr() + tree_offset, tree_words.ptr(), tree_words.size() * sizeof(uint32_t));
		r_entry.cluster_lod_tree.nodes_offset = offset + tree_offset;
		r_entry.cluster_lod_tree.refs_offset = offset + tree_offset + tree_nodes * 8;
		r_entry.cluster_lod_tree.group_count = group_count;
		r_entry.cluster_lod_tree.level_nodes = tree_levels;
	}
	const uint32_t digest_offset = tree_offset + tree_words.size();
	memset(host.ptr() + digest_offset, 0, DIGEST_WORDS * sizeof(uint32_t));
	rd->buffer_update(cluster_lod_table, offset * sizeof(uint32_t), uint32_t(words * sizeof(uint32_t)), host.ptr());
	r_entry.cluster_lod_table_offset = offset;
	r_entry.cluster_lod_table_words = words;
	r_entry.cluster_lod_infos_offset = offset + infos_offset;
	r_entry.cluster_lod_digest_offset = offset + digest_offset;
	r_entry.cluster_lod_count = cluster_count;
	r_entry.cluster_lod_total_triangles = p_layout.total_triangles;
	r_entry.cluster_lod_total_vertices = p_layout.total_vertices;
	r_entry.cluster_lod_pushed_frame = 0;
	return true;
}

void RenderRaytracing::_release_deformed_cluster_lod(RTDeformedCacheEntry &p_entry) {
	if (p_entry.cluster_lod_table_offset != UINT32_MAX) {
		// As _release_surface_cluster_lod: a frame in flight may still select from the region, and
		// the next content written there is a transfer recorded behind that selection.
		cluster_lod_table_live_words -= p_entry.cluster_lod_table_words;
		_cluster_lod_table_free(p_entry.cluster_lod_table_offset, p_entry.cluster_lod_table_words);
	}
	p_entry.cluster_lod_table_offset = UINT32_MAX;
	p_entry.cluster_lod_table_words = 0;
	p_entry.cluster_lod_infos_offset = 0;
	p_entry.cluster_lod_digest_offset = UINT32_MAX;
	p_entry.cluster_lod_tree = RTLodTree();
	p_entry.cluster_lod_count = 0;
	p_entry.cluster_lod_total_triangles = 0;
	p_entry.cluster_lod_total_vertices = 0;
	p_entry.cluster_lod_pushed_frame = 0;
	p_entry.cluster_lod_pending = false;
	p_entry.cluster_lod_pending_kind = RTDeformedCacheEntry::CLUSTER_QUEUED_NONE;
	p_entry.cluster_lod_last = false;
}

void RenderRaytracing::_queue_deformed_cluster_rebuild(RTDeformedCacheEntry &p_entry, uint8_t p_kind) {
	RTDeformedClusterBatch &batch = deformed_cluster_batch;
	const uint32_t cluster_count = p_entry.cluster_count;
	const uint32_t info_bytes = p_entry.cluster_infos.size();
	const uint32_t info_base = batch.clas_infos.size();
	batch.clas_infos.resize(info_base + info_bytes);
	memcpy(batch.clas_infos.ptr() + info_base, p_entry.cluster_infos.ptr(), info_bytes);
	batch.entries.push_back(&p_entry);
	batch.blas_addresses.push_back(p_entry.cluster_blas_address);
	batch.entry_cluster_counts.push_back(cluster_count);
	batch.storage_buffers.push_back(p_entry.cluster_storage);
	batch.blas_handles.push_back(p_entry.ptr->blas);
	batch.source_buffers.push_back(p_entry.owned_vb_full); // One per entry.
	if (p_entry.cluster_index_buffer.is_valid() && !batch.index_buffers_seen.has(p_entry.cluster_index_buffer)) {
		batch.index_buffers_seen.push_back(p_entry.cluster_index_buffer); // A few meshes, many instances.
		batch.source_buffers.push_back(p_entry.cluster_index_buffer);
	}
	batch.cluster_count += cluster_count;
	batch.total_triangles += p_entry.ptr->geometry.primitive_count;
	batch.total_vertices += p_entry.cluster_total_vertices;
	batch.max_cluster_triangles = MAX(batch.max_cluster_triangles, p_entry.cluster_max_triangles);
	batch.max_cluster_vertices = MAX(batch.max_cluster_vertices, p_entry.cluster_max_vertices);
	batch.max_clusters_per_blas = MAX(batch.max_clusters_per_blas, cluster_count);
	p_entry.cluster_frame_kind = p_kind;
}

bool RenderRaytracing::_unqueue_deformed_cluster_rebuild(RTDeformedCacheEntry &p_entry) {
	// The reverse of the queueing above, for the entry queued last; the maxima stay, as bounds.
	RTDeformedClusterBatch &batch = deformed_cluster_batch;
	const uint32_t n = batch.entries.size();
	if (n == 0 || batch.entries[n - 1] != &p_entry) {
		return false;
	}
	batch.entries.resize(n - 1);
	batch.clas_infos.resize(batch.clas_infos.size() - p_entry.cluster_infos.size());
	batch.blas_addresses.resize(n - 1);
	batch.entry_cluster_counts.resize(n - 1);
	batch.storage_buffers.resize(n - 1);
	batch.blas_handles.resize(n - 1);
	const RID index_buffer = p_entry.cluster_index_buffer;
	if (index_buffer.is_valid() && batch.source_buffers[batch.source_buffers.size() - 1] == index_buffer) {
		// Listed by this entry, first of the batch to read it.
		batch.source_buffers.resize(batch.source_buffers.size() - 1);
		batch.index_buffers_seen.erase(index_buffer);
	}
	batch.source_buffers.resize(batch.source_buffers.size() - 1); // Its owned vertex buffer.
	batch.cluster_count -= p_entry.cluster_count;
	batch.total_triangles -= p_entry.ptr->geometry.primitive_count;
	batch.total_vertices -= p_entry.cluster_total_vertices;
	p_entry.cluster_frame_kind = RTDeformedCacheEntry::CLUSTER_QUEUED_NONE;
	return true;
}

// A deformed entry's CLAS live in the region of the flush that built them. The trace reads that
// region through the BLAS, which the graph does not track, so a region is taken again only once
// the last frame that could trace it has been waited on: an entry is rebuilt on its next visible
// frame when its build is frame_delay frames old, whether its data changed or not, so a region
// written at frame F is last traced by frame F + D - 1, whose fence RenderingDevice waits on at
// frame F + 2D - 1; the pool reuses a region from that frame on. Both stamps are RenderingDevice's
// frames_drawn, the counter its fences follow.
bool RenderRaytracing::_deformed_cluster_region_expired(const RTDeformedCacheEntry &p_entry, uint64_t p_frames_drawn) const {
	return p_entry.cluster_built_frame == 0 || p_frames_drawn - p_entry.cluster_built_frame >= uint64_t(RD::get_singleton()->get_frame_delay());
}

// A region from the pool: one whose last trace has been waited on (see
// _deformed_cluster_region_expired), the smallest that fits, else the largest grown, else a
// new one: the selection's regions (D87) are a hundred times the batch's, and a small flush
// taking a large region first fit would have every region grow to the largest.
int32_t RenderRaytracing::_deformed_cluster_region_acquire(uint64_t p_alloc, uint64_t p_alignment) {
	RTDeformedClusterBatch &batch = deformed_cluster_batch;
	RD *rd = RD::get_singleton();
	const uint64_t frames_drawn = rd->get_frames_drawn();
	const uint64_t frame_delay = rd->get_frame_delay();
	int32_t region_index = -1;
	int32_t largest_index = -1;
	for (uint32_t i = 0; i < batch.regions.size(); i++) {
		const RTDeformedClusterRegion &candidate = batch.regions[i];
		if (frames_drawn - candidate.last_used_frame < 2 * frame_delay - 1) {
			continue;
		}
		if (candidate.capacity >= p_alloc) {
			if (region_index < 0 || candidate.capacity < batch.regions[region_index].capacity) {
				region_index = int32_t(i);
			}
		} else if (largest_index < 0 || candidate.capacity > batch.regions[largest_index].capacity) {
			largest_index = int32_t(i);
		}
	}
	if (region_index < 0) {
		region_index = largest_index;
	}
	if (region_index < 0) {
		RTDeformedClusterRegion region;
		batch.regions.push_back(region);
		region_index = int32_t(batch.regions.size()) - 1;
	}
	RTDeformedClusterRegion &region = batch.regions[region_index];
	if (region.capacity < p_alloc) {
		if (region.buffer.is_valid()) {
			rd->free_rid(region.buffer);
			region.buffer = RID();
		}
		const uint64_t capacity = MIN(MAX(p_alloc, region.capacity * 2), uint64_t(UINT32_MAX));
		region.capacity = 0;
		region.buffer = rd->storage_buffer_create(uint32_t(capacity), {}, 0,
				RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_STORAGE_BIT);
		if (region.buffer.is_null()) {
			batch.regions.remove_at(region_index);
			return -1;
		}
		region.capacity = capacity;
		region.base_offset = _align_up(rd->buffer_get_device_address(region.buffer), p_alignment) - rd->buffer_get_device_address(region.buffer);
		rd->set_resource_name(region.buffer, "RT deformed cluster region " + itos(region_index));
	}
	region.last_used_frame = frames_drawn;
	return region_index;
}

// Records the frame's builds, two a group. Returns false when one could not be recorded, the
// builds before it recorded; the caller puts the path in its fallback either way.
bool RenderRaytracing::_record_deformed_cluster_batch() {
	RTDeformedClusterBatch &batch = deformed_cluster_batch;
	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	const uint32_t entry_count = batch.entries.size();
	const uint32_t cluster_count = batch.cluster_count;
	const uint32_t clas_info_stride = caps.triangle_build_info_size;
	const uint32_t instantiate_info_stride = caps.instantiate_info_size;
	const uint32_t blas_info_stride = caps.bottom_level_build_info_size;

	// The templates made since the last flush (D107), before the instantiations that read them.
	if (!_build_deformed_cluster_templates()) {
		return false;
	}

	// Two groups, each its own CLAS builds and BLAS build: the entries whose CLAS infos reference an
	// opacity micromap (D96 OMMd), flagged for it, then the rest, not, since the flag sizes every
	// structure of a build for a micromap and each entry's BLAS storage was sized with its own
	// group's flags. Within a group the entries whose CLAS are built (D82) come first, then the
	// templated ones (D107), whose CLAS are instantiated in place; the arguments follow that order.
	struct Group {
		bool omm = false;
		uint32_t entry_begin = 0;
		uint32_t entry_count = 0;
		uint32_t cluster_begin = 0;
		uint32_t cluster_count = 0;
		// The built part (D82).
		uint32_t built_entries = 0;
		uint32_t built_clusters = 0;
		uint32_t built_info_begin = 0; // In built clusters, over the groups.
		uint32_t triangles = 0;
		uint32_t vertices = 0;
		uint64_t region_offset = 0;
		RD::ClusterAccelerationStructureInput clas_input;
		// The templated part (D107).
		uint32_t templated_entries = 0;
		uint32_t templated_clusters = 0;
		uint32_t templated_info_begin = 0; // In templated clusters, over the groups.
		uint32_t templated_triangles = 0;
		uint32_t templated_vertices = 0;
		uint32_t templated_max_triangles = 0;
		uint32_t templated_max_vertices = 0;
	};
	Group groups[2];
	groups[0].omm = true;
	batch.order.clear();
	batch.info_offsets.resize(entry_count);
	{
		uint32_t info_offset = 0;
		for (uint32_t e = 0; e < entry_count; e++) {
			batch.info_offsets[e] = info_offset;
			info_offset += batch.entries[e]->cluster_infos.size(); // Build or instantiation infos.
		}
	}
	Vector<RID> source_micromaps;
	LocalVector<RID> micromap_index_buffers;
	uint32_t built_info_total = 0;
	uint32_t templated_info_total = 0;
	uint32_t cluster_total = 0;
	for (Group &group : groups) {
		group.entry_begin = batch.order.size();
		group.cluster_begin = cluster_total;
		group.built_info_begin = built_info_total;
		group.templated_info_begin = templated_info_total;
		for (uint32_t pass = 0; pass < 2; pass++) {
			const bool templated = pass == 1;
			for (uint32_t e = 0; e < entry_count; e++) {
				const RTDeformedCacheEntry *entry = batch.entries[e];
				if (entry->cluster_omm != group.omm || entry->cluster_templated != templated) {
					continue;
				}
				if (group.omm) {
					// The micromap its infos reference, with its index buffer, which the build reads. Always
					// found: an entry's micromap is released only with its BLAS (_omm_release_entry), which
					// leaves the batch empty of it; a miss fails the flush, which turns the path off.
					RTOmmEntry *const *found = omm_by_serial.getptr(entry->omm_serial);
					ERR_FAIL_NULL_V_MSG(found, false, "RT: a deformed cluster entry's CLAS reference a micromap that is gone.");
					if (!source_micromaps.has((*found)->micromap)) {
						source_micromaps.push_back((*found)->micromap);
						micromap_index_buffers.push_back((*found)->index_buffer);
					}
				}
				batch.order.push_back(e);
				const uint32_t clusters = batch.entry_cluster_counts[e];
				if (templated) {
					group.templated_entries++;
					group.templated_clusters += clusters;
					group.templated_triangles += entry->ptr->geometry.primitive_count;
					group.templated_vertices += entry->cluster_total_vertices;
					group.templated_max_triangles = MAX(group.templated_max_triangles, entry->cluster_max_triangles);
					group.templated_max_vertices = MAX(group.templated_max_vertices, entry->cluster_max_vertices);
				} else {
					group.built_entries++;
					group.built_clusters += clusters;
					group.triangles += entry->ptr->geometry.primitive_count;
					group.vertices += entry->cluster_total_vertices;
				}
			}
		}
		group.entry_count = batch.order.size() - group.entry_begin;
		group.cluster_count = group.built_clusters + group.templated_clusters;
		built_info_total += group.built_clusters;
		templated_info_total += group.templated_clusters;
		cluster_total += group.cluster_count;
	}

	// Each group's built CLAS, with implicit destinations into its part of one region the driver
	// sub-allocates under the batch's own maxima, and the size query's one part for all of them.
	const uint64_t region_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	uint64_t region_size = 0;
	for (Group &group : groups) {
		if (group.built_entries == 0) {
			continue;
		}
		RD::ClusterAccelerationStructureInput &clas_input = group.clas_input;
		clas_input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_TRIANGLE_CLUSTER;
		clas_input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_IMPLICIT_DESTINATIONS;
		clas_input.max_acceleration_structure_count = group.built_clusters;
		clas_input.flags = _cluster_build_flags_omm(RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT, group.omm);
		clas_input.vertex_format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
		clas_input.max_cluster_triangle_count = batch.max_cluster_triangles;
		clas_input.max_cluster_vertex_count = batch.max_cluster_vertices;
		clas_input.max_total_triangle_count = group.triangles;
		clas_input.max_total_vertex_count = group.vertices;
		uint64_t group_size = 0, clas_scratch = 0;
		rd->cluster_acceleration_structure_get_build_sizes(clas_input, group_size, clas_scratch);
		ERR_FAIL_COND_V_MSG(group_size == 0, false, "RT: the driver sized the deformed cluster batch's region at 0 bytes.");
		group.region_offset = _align_up(region_size, region_alignment);
		region_size = group.region_offset + group_size;
	}

	// A region from the pool (a frame with several flushes takes several), where any CLAS are built.
	RTDeformedClusterRegion *region = nullptr;
	int32_t region_index = -1;
	if (region_size > 0) {
		const uint64_t region_alloc = region_alignment + region_size;
		ERR_FAIL_COND_V_MSG(region_alloc > UINT32_MAX, false, "RT: the deformed cluster batch's region exceeds 4 GB.");
		region_index = _deformed_cluster_region_acquire(region_alloc, region_alignment);
		ERR_FAIL_COND_V_MSG(region_index < 0, false, "RT: the deformed cluster batch's region could not be created.");
		region = &batch.regions[region_index];
	}

	// The frame's arguments, 16-byte sections: the CLAS build infos, the instantiation infos, the
	// BLAS infos, the CLAS addresses (written by the CLAS build, or here for templated entries,
	// whose CLAS have fixed addresses), the BLAS addresses. One grow-only buffer and one grow-only
	// host copy: the graph orders the upload after the previous frame's builds through the
	// buffer's tracker (each build leaves it at the cluster input read-write usage, and the copy
	// takes its barrier), so a ring would buy nothing.
	const uint64_t clas_infos_offset = 0;
	const uint64_t instantiate_infos_offset = _align_up(clas_infos_offset + uint64_t(clas_info_stride) * built_info_total, 16);
	const uint64_t blas_infos_offset = _align_up(instantiate_infos_offset + uint64_t(instantiate_info_stride) * templated_info_total, 16);
	const uint64_t clas_addresses_offset = _align_up(blas_infos_offset + uint64_t(blas_info_stride) * entry_count, 16);
	const uint64_t blas_addresses_offset = _align_up(clas_addresses_offset + sizeof(uint64_t) * cluster_count, 16);
	const uint64_t args_size = _align_up(blas_addresses_offset + sizeof(uint64_t) * entry_count, 16);
	ERR_FAIL_COND_V_MSG(args_size > UINT32_MAX, false, "RT: the deformed cluster batch's arguments exceed 4 GB.");
	if (batch.args_capacity < args_size) {
		if (batch.args_buffer.is_valid()) {
			rd->free_rid(batch.args_buffer); // Deferred past the fence of every frame in flight.
			batch.args_buffer = RID();
		}
		const uint32_t capacity = uint32_t(MIN(MAX(args_size, uint64_t(batch.args_capacity) * 2), uint64_t(UINT32_MAX)));
		batch.args_capacity = 0;
		batch.args_buffer = rd->storage_buffer_create(capacity, {}, 0,
				RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
		ERR_FAIL_COND_V_MSG(batch.args_buffer.is_null(), false, "RT: the deformed cluster batch's argument buffer could not be created.");
		batch.args_capacity = capacity;
		rd->set_resource_name(batch.args_buffer, "RT deformed cluster batch args");
	}
	const RID args_buffer = batch.args_buffer;
	const uint64_t args_address = rd->buffer_get_device_address(args_buffer);

	if (batch.args_host.size() < args_size) {
		batch.args_host.resize(args_size);
	}
	uint8_t *args = batch.args_host.ptr();
	memset(args, 0, args_size);
	// The build takes its lists as Vectors: these, the groups' slices of them below, and
	// RenderingDevice's own per-build map, lists and scratch are the flush's allocations.
	Vector<RID> storage_buffers, source_buffers, blas_handles, template_buffers;
	storage_buffers.resize(entry_count);
	blas_handles.resize(entry_count);
	for (const Group &group : groups) {
		uint32_t first_cluster = group.cluster_begin;
		uint32_t built_info = group.built_info_begin;
		uint32_t templated_info = group.templated_info_begin;
		for (uint32_t i = group.entry_begin; i < group.entry_begin + group.entry_count; i++) {
			const uint32_t e = batch.order[i];
			const RTDeformedCacheEntry *entry = batch.entries[e];
			const uint32_t entry_clusters = batch.entry_cluster_counts[e];
			if (entry->cluster_templated) {
				memcpy(args + instantiate_infos_offset + uint64_t(templated_info) * instantiate_info_stride, batch.clas_infos.ptr() + batch.info_offsets[e], uint64_t(entry_clusters) * instantiate_info_stride);
				templated_info += entry_clusters;
				uint64_t *clas_addresses = reinterpret_cast<uint64_t *>(args + clas_addresses_offset) + first_cluster;
				for (uint32_t c = 0; c < entry_clusters; c++) {
					clas_addresses[c] = entry->cluster_clas_address + uint64_t(c) * entry->cluster_clas_stride;
				}
				const RTDeformedClusterTemplate *cluster_template = deformed_cluster_templates.getptr(entry->cluster_template_key);
				ERR_FAIL_COND_V_MSG(cluster_template == nullptr || !cluster_template->built, false, "RT: a templated deformed cluster entry's templates are gone or unbuilt.");
				if (!template_buffers.has(cluster_template->storage)) {
					template_buffers.push_back(cluster_template->storage); // A few meshes, many instances.
				}
			} else {
				memcpy(args + clas_infos_offset + uint64_t(built_info) * clas_info_stride, batch.clas_infos.ptr() + batch.info_offsets[e], uint64_t(entry_clusters) * clas_info_stride);
				built_info += entry_clusters;
			}
			reinterpret_cast<uint64_t *>(args + blas_addresses_offset)[i] = batch.blas_addresses[e];
			RD::ClusterBottomLevelBuildInfo blas_info;
			blas_info.cluster_references_count = entry_clusters;
			blas_info.cluster_references_stride = sizeof(uint64_t);
			blas_info.cluster_references_address = args_address + clas_addresses_offset + sizeof(uint64_t) * first_cluster;
			rd->cluster_bottom_level_build_info_write(args + blas_infos_offset + uint64_t(i) * blas_info_stride, blas_info);
			storage_buffers.write[i] = batch.storage_buffers[e];
			blas_handles.write[i] = batch.blas_handles[e];
			first_cluster += entry_clusters;
		}
	}
	rd->buffer_update(args_buffer, 0, uint32_t(args_size), args);

	source_buffers.resize(batch.source_buffers.size());
	memcpy(source_buffers.ptrw(), batch.source_buffers.ptr(), sizeof(RID) * batch.source_buffers.size());
	for (const RID &index_buffer : micromap_index_buffers) {
		source_buffers.push_back(index_buffer);
	}

	// Per group, the built CLAS in one build into its part of the region, the driver writing
	// their addresses; the templated CLAS instantiated in one build into the entries' own storage;
	// then every cluster BLAS over its slice of the addresses.
	for (const Group &group : groups) {
		if (group.entry_count == 0) {
			continue;
		}
		if (group.built_entries > 0) {
			RD::ClusterAccelerationStructureBuild build;
			build.input = group.clas_input;
			build.dst_implicit_buffer = region->buffer;
			build.dst_implicit_offset = region->base_offset + group.region_offset;
			build.dst_addresses_buffer = args_buffer;
			build.dst_addresses_offset = clas_addresses_offset + sizeof(uint64_t) * group.cluster_begin;
			build.dst_addresses_stride = sizeof(uint64_t);
			build.src_infos_buffer = args_buffer;
			build.src_infos_offset = clas_infos_offset + uint64_t(clas_info_stride) * group.built_info_begin;
			build.src_infos_stride = clas_info_stride;
			build.source_buffers = source_buffers;
			if (group.omm) {
				build.source_micromaps = source_micromaps;
			}
			if (rd->cluster_acceleration_structure_build(build) != OK) {
				return false;
			}
		}
		const uint32_t templated_begin = group.entry_begin + group.built_entries;
		const uint32_t group_end = group.entry_begin + group.entry_count;
		if (group.templated_entries > 0) {
			// In place, where the last frame's trace read them through the BLAS: the BLAS handles are
			// listed, so their trackers, which the lists that traced them declared, order this rewrite
			// after those reads (RenderingDevice::ClusterAccelerationStructureBuild::dst_acceleration_structures).
			RD::ClusterAccelerationStructureBuild build;
			build.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_INSTANTIATE_TRIANGLE_CLUSTER;
			build.input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
			build.input.max_acceleration_structure_count = group.templated_clusters;
			build.input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT);
			build.input.vertex_format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
			build.input.max_cluster_triangle_count = group.templated_max_triangles;
			build.input.max_cluster_vertex_count = group.templated_max_vertices;
			build.input.max_total_triangle_count = group.templated_triangles;
			build.input.max_total_vertex_count = group.templated_vertices;
			build.dst_storage_buffers = storage_buffers.slice(templated_begin, group_end);
			build.dst_addresses_buffer = args_buffer;
			build.dst_addresses_offset = clas_addresses_offset + sizeof(uint64_t) * (group.cluster_begin + group.built_clusters);
			build.dst_addresses_stride = sizeof(uint64_t);
			build.src_infos_buffer = args_buffer;
			build.src_infos_offset = instantiate_infos_offset + uint64_t(instantiate_info_stride) * group.templated_info_begin;
			build.src_infos_stride = instantiate_info_stride;
			build.source_buffers = source_buffers;
			build.source_acceleration_structure_buffers = template_buffers;
			build.dst_acceleration_structures = blas_handles.slice(templated_begin, group_end);
			if (rd->cluster_acceleration_structure_build(build) != OK) {
				return false;
			}
		}
		{
			RD::ClusterAccelerationStructureBuild build;
			build.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_CLUSTERS_BOTTOM_LEVEL;
			build.input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
			build.input.max_acceleration_structure_count = group.entry_count;
			build.input.flags = _cluster_build_flags_omm(RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT, group.omm);
			build.input.max_total_cluster_count = group.cluster_count;
			build.input.max_cluster_count_per_acceleration_structure = batch.max_clusters_per_blas;
			build.dst_storage_buffers = storage_buffers.slice(group.entry_begin, group_end);
			build.dst_addresses_buffer = args_buffer;
			build.dst_addresses_offset = blas_addresses_offset + sizeof(uint64_t) * group.entry_begin;
			build.dst_addresses_stride = sizeof(uint64_t);
			build.src_infos_buffer = args_buffer;
			build.src_infos_offset = blas_infos_offset + uint64_t(blas_info_stride) * group.entry_begin;
			build.src_infos_stride = blas_info_stride;
			// The CLAS each BLAS references, read through its info: the region's, and the templated
			// entries' own, in the storage each BLAS is written to.
			if (group.built_entries > 0) {
				build.source_acceleration_structure_buffers.push_back(region->buffer);
			}
			if (group.templated_entries > 0) {
				build.source_acceleration_structure_buffers.append_array(storage_buffers.slice(templated_begin, group_end));
			}
			build.dst_acceleration_structures = blas_handles.slice(group.entry_begin, group_end);
			if (rd->cluster_acceleration_structure_build(build) != OK) {
				return false;
			}
		}
	}

	batch.frames_flushed++;
	if (batch.frames_flushed == 1 || batch.frames_flushed % 300 == 0) {
		print_verbose(vformat("RT: deformed cluster batch: %d surfaces (%d with opacity micromaps, %d instantiated from templates), %d clusters, %d triangles, %d KB region %d of %d, %d KB of arguments uploaded, %d expiry rebuilds, %d template sets and %d clusters instantiated so far.", entry_count, groups[0].entry_count, groups[0].templated_entries + groups[1].templated_entries, cluster_count, batch.total_triangles, region_size / 1024, region_index, batch.regions.size(), args_size / 1024, deformed_cluster_expiry_rebuilds_total, deformed_cluster_templates_built_total, deformed_cluster_clusters_instantiated_total));
	}
	return true;
}

void RenderRaytracing::_flush_deformed_cluster_batch() {
	RTDeformedClusterBatch &batch = deformed_cluster_batch;
	if (batch.entries.is_empty()) {
		return;
	}
	RD *rd = RD::get_singleton();
	const uint64_t frames_drawn = rd->get_frames_drawn();

	if (_record_deformed_cluster_batch()) {
		// The builds are recorded: the entries' CLAS live in this flush's region from now, and the
		// render-info counters take the surfaces that entered the TLAS this frame, by what queued
		// them, as the triangle path counts its builds and refits at the TLAS push. A rebuild of an
		// unchanged surface for its region's expiry is neither, and is counted on its own.
		for (RTDeformedCacheEntry *entry : batch.entries) {
			entry->cluster_built_frame = frames_drawn;
			entry->cluster_lod_last = false; // LOD 0 again (D87).
			if (entry->cluster_pushed_frame == frames_drawn) {
				const uint32_t triangles = entry->ptr ? entry->ptr->geometry.primitive_count : 0;
				switch (entry->cluster_frame_kind) {
					case RTDeformedCacheEntry::CLUSTER_QUEUED_CHANGED:
						if (entry->cluster_templated) {
							// Instantiated in place and the BLAS rebuilt over them (D107): the per-frame
							// update a triangle BLAS's refit is, counted as one.
							deformed_cluster_updates++;
							deformed_cluster_triangles_updated += triangles;
							deformed_cluster_clusters_instantiated_total += entry->cluster_count;
							break;
						}
						[[fallthrough]];
					case RTDeformedCacheEntry::CLUSTER_QUEUED_FIRST_BUILD:
						// A cluster BLAS is rebuilt, never refit: a change is a build (D97).
						deformed_cluster_builds++;
						deformed_cluster_triangles_built += triangles;
						if (entry->cluster_templated) {
							deformed_cluster_clusters_instantiated_total += entry->cluster_count;
						}
						break;
					case RTDeformedCacheEntry::CLUSTER_QUEUED_EXPIRY:
						if (entry->cluster_templated) {
							// LOD 0 again after a selection (D108): instantiated in place, an update.
							deformed_cluster_updates++;
							deformed_cluster_triangles_updated += triangles;
							deformed_cluster_clusters_instantiated_total += entry->cluster_count;
							break;
						}
						deformed_cluster_expiry_rebuilds_total++;
						deformed_cluster_builds++; // And so is a rebuild for the region's reuse (D97).
						deformed_cluster_triangles_built += triangles;
						break;
					default:
						break;
				}
			}
			entry->cluster_frame_kind = RTDeformedCacheEntry::CLUSTER_QUEUED_NONE;
		}
	} else {
		// Nothing of this frame's batch may be traced: the path goes off for the rest of the run
		// (every cluster entry moves to the triangle path on its next visible frame, see
		// process_deformed_surface, so no flush runs again and the batch's buffers can go), a
		// surface whose last build is recent keeps that build for this frame, and one with no
		// build or an expired region (its region may be taken by this very flush) leaves this
		// frame's TLAS: build_tlas drops its handle from the instance list before the build.
		cluster_deformed_enabled = false;
		uint32_t untraceable = 0;
		for (RTDeformedCacheEntry *entry : batch.entries) {
			entry->blas_built_once = false;
			entry->cluster_frame_kind = RTDeformedCacheEntry::CLUSTER_QUEUED_NONE;
			// A templated entry's CLAS (D107) may be half rewritten under its BLAS: it leaves too.
			if (entry->ptr && entry->ptr->blas.is_valid() && (entry->cluster_templated || _deformed_cluster_region_expired(*entry, frames_drawn))) {
				batch.dropped_handles.push_back(entry->ptr->blas);
				untraceable++;
			}
		}
		for (RTDeformedClusterRegion &region : batch.regions) {
			if (region.buffer.is_valid()) {
				rd->free_rid(region.buffer); // Deferred past the fence of every frame in flight.
			}
		}
		batch.regions.clear();
		if (batch.args_buffer.is_valid()) {
			rd->free_rid(batch.args_buffer);
			batch.args_buffer = RID();
			batch.args_capacity = 0;
		}
		WARN_PRINT(vformat("RT: the deformed cluster batch could not be recorded; the deformed cluster path is off from the next frame, %d of its %d surfaces have no build to trace and leave this frame's TLAS, the rest trace their last build.", untraceable, batch.entries.size()));
	}

	batch.entries.clear();
	batch.clas_infos.clear();
	batch.blas_addresses.clear();
	batch.entry_cluster_counts.clear();
	batch.storage_buffers.clear();
	batch.source_buffers.clear();
	batch.blas_handles.clear();
	batch.index_buffers_seen.clear();
	batch.cluster_count = 0;
	batch.total_triangles = 0;
	batch.total_vertices = 0;
	batch.max_cluster_triangles = 0;
	batch.max_cluster_vertices = 0;
	batch.max_clusters_per_blas = 0;
}

void RenderRaytracing::_on_cluster_sizes_read(const PackedByteArray &p_sizes, uint64_t p_ticket) {
	// Called by RenderingDevice on the render thread as it begins a later frame: only queue
	// the result, the compaction records its commands from the next RT frame.
	if (singleton == nullptr) {
		return;
	}
	RTClusterCompactionResult result;
	result.ticket = p_ticket;
	result.sizes = p_sizes;
	singleton->cluster_compaction_results.push_back(result);
}

void RenderRaytracing::_process_cluster_compactions() {
	RD *rd = RD::get_singleton();

	// Regions retired a frame queue ago are out of every frame in flight and go back to their chunk.
	_arena_sweep(cluster_arena, cluster_arena_retired, cluster_arena_reserved_bytes, cluster_arena_live_bytes);
	_arena_sweep(cluster_geometry_arena, cluster_geometry_arena_retired, cluster_geometry_arena_reserved_bytes, cluster_geometry_arena_live_bytes);

	if (cluster_compaction_results.is_empty()) {
		return;
	}
	const uint64_t frames_drawn = rd->get_frames_drawn();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);

	// Taken off the member first: a buffer update below can stall on the frame queue, which
	// delivers further readbacks into the member while these are processed.
	LocalVector<RTClusterCompactionResult> results;
	results = cluster_compaction_results;
	cluster_compaction_results.clear();
	LocalVector<RTClusterCompactionResult> held; // By the build budget (D97), ahead of what arrives meanwhile.
	uint32_t newly_held = 0;
	for (RTClusterCompactionResult &result : results) {
		RTSurfaceData **found = cluster_compaction_pending.getptr(result.ticket);
		if (found == nullptr) {
			continue; // The surface was released while its sizes were in flight.
		}
		// The build budget (D97): the move and the BLAS over the moved CLAS, the surface tracing its
		// staging meanwhile. Not a surface with a hierarchy: a held compaction would hold its cached
		// levels (D90) and a streamed surface's level-0 pin (D92), so it is forced.
		const bool budgeted = build_budget.enabled && (*found)->cluster_lod_table_offset == UINT32_MAX;
		if (budgeted) {
			if (result.held_frame == frames_drawn) {
				held.push_back(result); // Held by an earlier viewport's step this frame: decided once a frame.
				continue;
			}
			// One overdue item a frame goes past the limits.
			const uint64_t since = result.held_since != 0 ? result.held_since : frames_drawn;
			const BuildBudgetTake take = _build_budget_take(uint64_t((*found)->cluster_built_count + 1) * RT_BUILD_BUDGET_CLUSTER_TRIANGLES, true, !build_budget.compaction_overdue_taken && frames_drawn - since >= build_budget.max_wait_frames);
			if (take == BUILD_BUDGET_REFUSED) {
				result.held_since = since;
				result.held_frame = frames_drawn;
				held.push_back(result);
				newly_held++;
				continue;
			}
			build_budget.compaction_overdue_taken = build_budget.compaction_overdue_taken || take == BUILD_BUDGET_ADMITTED_PAST;
			_build_budget_note_wait(result.held_since);
		}
		RTSurfaceData *surf = *found;
		cluster_compaction_pending.erase(result.ticket);
		surf->compaction_ticket = 0;

		const uint32_t cluster_count = surf->cluster_built_count; // A streamed surface's persistent clusters alone (D92).
		if (result.sizes.size() < int64_t(sizeof(uint32_t) * cluster_count)) {
			WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: the sizes read back are short; the surface stays in its staging buffer.", surf->cluster_cache_key));
			continue;
		}
		const uint32_t *sizes = reinterpret_cast<const uint32_t *>(result.sizes.ptr());
		// The region: the BLAS at its base, sized as the query's worst case since an explicit
		// build writes no size, then each CLAS at the cluster alignment.
		const uint64_t clas_offset = _align_up(_align_up(surf->cluster_blas_worst_size, caps.cluster_bottom_level_byte_alignment), caps.cluster_byte_alignment);
		uint64_t clas_total = 0;
		bool sized = true;
		for (uint32_t c = 0; c < cluster_count && sized; c++) {
			sized = sizes[c] > 0;
			clas_total += _align_up(sizes[c], caps.cluster_byte_alignment);
		}
		if (!sized) {
			WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: a built size read back as zero; the surface stays in its staging buffer.", surf->cluster_cache_key));
			continue;
		}
		const uint64_t region_size = _align_up(clas_offset + clas_total, base_alignment);

		uint32_t chunk_index = 0;
		uint64_t region_offset = 0;
		if (!_cluster_arena_allocate(region_size, chunk_index, region_offset)) {
			WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: no arena chunk could be created; the surface stays in its staging buffer.", surf->cluster_cache_key));
			continue;
		}
		const RTClusterArenaChunk &chunk = cluster_arena[chunk_index];
		const uint64_t region_address = chunk.address + region_offset;
		RID blas = rd->cluster_blas_create(chunk.buffer, chunk.base_offset + region_offset);
		if (blas.is_null()) {
			_cluster_arena_retire(chunk_index, region_offset, region_size);
			WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: no handle for the compacted BLAS; the surface stays in its staging buffer.", surf->cluster_cache_key));
			continue;
		}

		LocalVector<uint64_t> compact_addresses;
		compact_addresses.resize(cluster_count + 1);
		uint64_t cursor = region_address + clas_offset;
		for (uint32_t c = 0; c < cluster_count; c++) {
			compact_addresses[c] = cursor;
			cursor += _align_up(sizes[c], caps.cluster_byte_alignment);
		}
		compact_addresses[cluster_count] = region_address;
		rd->buffer_update(surf->cluster_args, uint32_t(surf->cluster_args_compact_offset), uint32_t(sizeof(uint64_t) * (cluster_count + 1)), compact_addresses.ptr());

		// The move: every CLAS from its implicit staging address (the address array the build
		// wrote is, one address per entry, exactly the move's source info) to its compact one. The
		// byte budget is the implicit build's whole region, as NVIDIA's sample passes it, not the
		// sum of the read-back sizes: what the driver counts as moved is its own.
		bool ok = true;
		{
			RD::ClusterAccelerationStructureBuild move;
			move.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MOVE_OBJECTS;
			move.input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
			move.input.max_acceleration_structure_count = cluster_count;
			move.input.move_type = RD::ClusterAccelerationStructureInput::CLUSTER_MOVE_TRIANGLE_CLUSTER;
			move.input.move_no_overlap = true;
			move.input.move_max_bytes = surf->cluster_clas_region_size;
			move.dst_storage_buffer = chunk.buffer;
			move.dst_addresses_buffer = surf->cluster_args;
			move.dst_addresses_offset = surf->cluster_args_compact_offset;
			move.dst_addresses_stride = sizeof(uint64_t);
			move.src_infos_buffer = surf->cluster_args;
			move.src_infos_offset = surf->cluster_args_implicit_offset;
			move.src_infos_stride = sizeof(uint64_t);
			move.source_acceleration_structure_buffers.push_back(surf->cluster_storage);
			ok = rd->cluster_acceleration_structure_build(move) == OK;
		}
		// The BLAS again, over the compact addresses, at the region's base.
		if (ok) {
			RD::ClusterAccelerationStructureBuild build;
			build.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_CLUSTERS_BOTTOM_LEVEL;
			build.input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
			build.input.max_acceleration_structure_count = 1;
			build.input.flags = _cluster_build_flags_omm(RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT, surf->cluster_omm);
			build.input.max_total_cluster_count = surf->cluster_blas_count; // The BLAS spans the LOD 0 clusters (D85), or a streamed surface's coarsest cut (D92); the move spanned every CLAS built.
			build.input.max_cluster_count_per_acceleration_structure = surf->cluster_blas_count;
			build.dst_storage_buffer = chunk.buffer;
			build.dst_addresses_buffer = surf->cluster_args;
			build.dst_addresses_offset = surf->cluster_args_compact_offset + sizeof(uint64_t) * cluster_count;
			build.dst_addresses_stride = sizeof(uint64_t);
			build.src_infos_buffer = surf->cluster_args;
			build.src_infos_offset = surf->cluster_args_blas_info_offset;
			build.src_infos_stride = caps.bottom_level_build_info_size;
			build.source_acceleration_structure_buffers.push_back(chunk.buffer); // The moved CLAS it references, read through its info.
			build.dst_acceleration_structures.push_back(blas);
			ok = rd->cluster_acceleration_structure_build(build) == OK;
		}
		if (!ok) {
			// Nothing references the region, but a recorded move may still write it: retired, not freed.
			rd->free_rid(blas);
			_cluster_arena_retire(chunk_index, region_offset, region_size);
			WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: the compaction could not be recorded; the surface stays in its staging buffer.", surf->cluster_cache_key));
			continue;
		}
		rd->set_resource_name(blas, "RT cluster BLAS" + String(surf->cluster_omm ? " OMM" : "") + " [" + itos(surf->cluster_cache_key) + "]");
		if (surf->cluster_lod_table_offset != UINT32_MAX) {
			// A cached level (D90) would reference the staged CLAS, which go with this frame. None is
			// placed while the compaction is pending, so this only guards that rule.
			_release_lod_cached_levels(surf);
			// The selection reads the CLAS addresses from the LOD table (D86): the compact ones from
			// this frame on, the upload ordered ahead of the frame's selection by the table's tracker
			// and the instances' BLAS builds behind the move by the chunk's.
			if (surf->cluster_lod_stream) {
				// A streamed surface's persistent runs (D92), the loaded groups' addresses untouched.
				const RTLodStream *stream = surf->cluster_lod_stream;
				uint32_t moved = 0;
				for (uint32_t r = 0; r < stream->persistent_runs.size(); r += 2) {
					rd->buffer_update(cluster_lod_table, (surf->cluster_lod_table_offset + stream->persistent_runs[r] * 2) * sizeof(uint32_t), uint32_t(sizeof(uint64_t) * stream->persistent_runs[r + 1]), compact_addresses.ptr() + moved);
					moved += stream->persistent_runs[r + 1];
				}
				surf->cluster_lod_stream->persistent_bytes = region_size;
			} else {
				rd->buffer_update(cluster_lod_table, surf->cluster_lod_table_offset * sizeof(uint32_t), uint32_t(sizeof(uint64_t) * cluster_count), compact_addresses.ptr());
			}
		}

		// The swap: the TLAS is built from the surface's handle every frame, and the buffers the
		// move reads are disposed of with this frame.
		const uint64_t staged_bytes = surf->blas_size;
		if (surf->blas.is_valid() && rd->acceleration_structure_is_valid(surf->blas)) {
			rd->free_rid(surf->blas);
		}
		surf->blas = blas;
		rd->free_rid(surf->cluster_storage);
		surf->cluster_storage = RID();
		rd->free_rid(surf->cluster_args);
		surf->cluster_args = RID();
		surf->arena_chunk = int32_t(chunk_index);
		surf->arena_offset = region_offset;
		surf->arena_size = region_size;
		surf->blas_size = region_size;

		cluster_compactions_total++;
		frame_extra_builds++; // A move and a BLAS over existing CLAS: a build, no new triangles (D97).
		build_budget.built += budgeted ? 1 : 0;
		cluster_bytes_staged_total += staged_bytes;
		cluster_bytes_compacted_total += region_size;
		print_verbose(vformat("RT: cluster BLAS [%d] compacted: %d clusters, %d KB staged -> %d KB (chunk %d at %d KB; arena %d chunks, %d / %d KB live).", surf->cluster_cache_key, cluster_count, staged_bytes / 1024, region_size / 1024, chunk_index, region_offset / 1024, cluster_arena.size(), cluster_arena_live_bytes / 1024, cluster_arena_reserved_bytes / 1024));
	}
	if (!held.is_empty()) {
		build_budget.held += newly_held;
		for (const RTClusterCompactionResult &result : cluster_compaction_results) {
			held.push_back(result);
		}
		cluster_compaction_results = held;
	}
}

// A cluster arena (D81): chunks sub-allocated first fit, each region size a multiple of the
// arena's alignment and each chunk base aligned to it, so every offset handed out is aligned. The
// static CLAS arena and the cluster geometry arena (D93b) are two instances.
static bool _arena_allocate(LocalVector<RTClusterArenaChunk> &r_arena, uint64_t &r_reserved_bytes, uint64_t &r_live_bytes, uint64_t p_alignment, BitField<RD::BufferCreationBits> p_usage, BitField<RDD::PipelineStageBits> p_address_read_stages, const String &p_name, uint64_t p_size, uint32_t &r_chunk, uint64_t &r_offset) {
	// First fit over the live chunks.
	for (uint32_t i = 0; i < r_arena.size(); i++) {
		RTClusterArenaChunk &chunk = r_arena[i];
		if (chunk.buffer.is_null()) {
			continue;
		}
		for (uint32_t j = 0; j < chunk.free_list.size(); j++) {
			RTClusterArenaChunk::FreeRange &range = chunk.free_list[j];
			if (range.size < p_size) {
				continue;
			}
			r_chunk = i;
			r_offset = range.offset;
			range.offset += p_size;
			range.size -= p_size;
			if (range.size == 0) {
				chunk.free_list.remove_at(j);
			}
			chunk.live_regions++;
			r_live_bytes += p_size;
			return true;
		}
	}

	// A new chunk, of the standard size or, for a region larger than that, its own.
	RD *rd = RD::get_singleton();
	const uint64_t chunk_size = MAX(uint64_t(RT_CLUSTER_ARENA_CHUNK_BYTES), p_size);
	const uint64_t buffer_size = chunk_size + p_alignment;
	ERR_FAIL_COND_V_MSG(buffer_size > UINT32_MAX, false, "RT: a cluster arena chunk exceeds 4 GB.");
	RID buffer = rd->storage_buffer_create(uint32_t(buffer_size), {}, 0, p_usage);
	if (buffer.is_null()) {
		return false;
	}
	if (p_address_read_stages != 0) {
		// Read through device addresses: a write to a region reused after retirement waits for those reads.
		rd->buffer_add_device_address_read_stages(buffer, p_address_read_stages);
	}
	RTClusterArenaChunk chunk;
	chunk.buffer = buffer;
	const uint64_t buffer_address = rd->buffer_get_device_address(buffer);
	chunk.address = _align_up(buffer_address, p_alignment);
	chunk.base_offset = chunk.address - buffer_address;
	chunk.size = chunk_size;
	chunk.live_regions = 1;
	if (chunk_size > p_size) {
		chunk.free_list.push_back({ p_size, chunk_size - p_size });
	}

	// A dead slot is reused so the chunk indices surfaces hold stay valid.
	uint32_t index = r_arena.size();
	for (uint32_t i = 0; i < r_arena.size(); i++) {
		if (r_arena[i].buffer.is_null()) {
			index = i;
			break;
		}
	}
	if (index == r_arena.size()) {
		r_arena.push_back(chunk);
	} else {
		r_arena[index] = chunk;
	}
	rd->set_resource_name(buffer, p_name + " [" + itos(index) + "]");
	r_reserved_bytes += chunk_size;
	r_live_bytes += p_size;
	r_chunk = index;
	r_offset = 0;
	return true;
}

static void _arena_free(LocalVector<RTClusterArenaChunk> &r_arena, uint64_t &r_reserved_bytes, uint64_t &r_live_bytes, uint32_t p_chunk, uint64_t p_offset, uint64_t p_size) {
	ERR_FAIL_UNSIGNED_INDEX(p_chunk, r_arena.size());
	RTClusterArenaChunk &chunk = r_arena[p_chunk];
	ERR_FAIL_COND(chunk.buffer.is_null() || chunk.live_regions == 0);

	// Insert in offset order, then merge with either neighbour it touches.
	uint32_t at = 0;
	while (at < chunk.free_list.size() && chunk.free_list[at].offset < p_offset) {
		at++;
	}
	chunk.free_list.insert(at, { p_offset, p_size });
	if (at + 1 < chunk.free_list.size() && chunk.free_list[at].offset + chunk.free_list[at].size == chunk.free_list[at + 1].offset) {
		chunk.free_list[at].size += chunk.free_list[at + 1].size;
		chunk.free_list.remove_at(at + 1);
	}
	if (at > 0 && chunk.free_list[at - 1].offset + chunk.free_list[at - 1].size == chunk.free_list[at].offset) {
		chunk.free_list[at - 1].size += chunk.free_list[at].size;
		chunk.free_list.remove_at(at);
	}
	chunk.live_regions--;
	r_live_bytes -= p_size;

	// A chunk with nothing live in it goes, so the arena's memory follows its contents.
	if (chunk.live_regions == 0) {
		RD::get_singleton()->free_rid(chunk.buffer);
		r_reserved_bytes -= chunk.size;
		chunk = RTClusterArenaChunk();
	}
}

// Regions retired a frame queue ago are out of every frame in flight and go back to their chunk.
static void _arena_sweep(LocalVector<RTClusterArenaChunk> &r_arena, LocalVector<RTClusterRetiredRegion> &r_retired, uint64_t &r_reserved_bytes, uint64_t &r_live_bytes) {
	if (r_retired.is_empty()) {
		return;
	}
	RD *rd = RD::get_singleton();
	const uint64_t frames_drawn = rd->get_frames_drawn();
	const uint64_t frame_delay = rd->get_frame_delay();
	for (uint32_t i = 0; i < r_retired.size();) {
		const RTClusterRetiredRegion &retired = r_retired[i];
		if (frames_drawn - retired.frame >= frame_delay) {
			_arena_free(r_arena, r_reserved_bytes, r_live_bytes, retired.chunk, retired.offset, retired.size);
			r_retired.remove_at_unordered(i);
		} else {
			i++;
		}
	}
}

bool RenderRaytracing::_cluster_arena_allocate(uint64_t p_size, uint32_t &r_chunk, uint64_t &r_offset) {
	const RD::ClusterAccelerationStructureCapabilities &caps = RD::get_singleton()->get_cluster_acceleration_structure_capabilities();
	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	return _arena_allocate(cluster_arena, cluster_arena_reserved_bytes, cluster_arena_live_bytes, base_alignment,
			RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_STORAGE_BIT, 0, "RT cluster arena", p_size, r_chunk, r_offset);
}

void RenderRaytracing::_cluster_arena_retire(uint32_t p_chunk, uint64_t p_offset, uint64_t p_size) {
	// The chunk outlives the region and the trace's reads of it are not on the chunk's tracker,
	// so the region is handed out again only once the frames that may read it have retired.
	RTClusterRetiredRegion retired;
	retired.chunk = p_chunk;
	retired.offset = p_offset;
	retired.size = p_size;
	retired.frame = RD::get_singleton()->get_frames_drawn();
	cluster_arena_retired.push_back(retired);
}

void RenderRaytracing::_cluster_arena_free(uint32_t p_chunk, uint64_t p_offset, uint64_t p_size) {
	_arena_free(cluster_arena, cluster_arena_reserved_bytes, cluster_arena_live_bytes, p_chunk, p_offset, p_size);
}

void RenderRaytracing::release_mesh_surface(RID p_mesh, uint32_t p_surface) {
	// Its opacity micromaps and their BLASes (D96).
	_omm_release_mesh_surface(p_mesh, p_surface);
	// The RT cache entry of a surface its mesh is dropping: its BLAS, staging or arena region go
	// now rather than when the cache slot is next reused.
	const uint32_t cache_key = (p_mesh.get_local_index() << 8) | (p_surface & 0xFF);
	const uint32_t chunk_idx = cache_key / RT_CACHE_CHUNK_SIZE;
	if (chunk_idx >= (uint32_t)surface_chunks.size() || surface_chunks[chunk_idx] == nullptr) {
		return;
	}
	RTCacheEntry *entry = &surface_chunks[chunk_idx][cache_key % RT_CACHE_CHUNK_SIZE];
	if (entry->ptr) {
		_release_surface_data(entry->ptr);
	}
	// Its wait for the build budget (D97) goes with it: the slot's next surface starts afresh.
	entry->budget_frame = UINT64_MAX;
	entry->budget_wait_since = 0;
	entry->budget_admitted = false;
}

void RenderRaytracing::_mesh_surface_freed(RID p_mesh, uint32_t p_surface) {
	if (singleton) {
		singleton->release_mesh_surface(p_mesh, p_surface);
	}
}

void RenderRaytracing::_release_surface_data(RTSurfaceData *p_surf_data) {
	RD *rd = RD::get_singleton();
	// A triangle BLAS may already be gone with its mesh's buffers; a cluster BLAS handle is freed here.
	if (p_surf_data->blas.is_valid() && rd->acceleration_structure_is_valid(p_surf_data->blas)) {
		rd->free_rid(p_surf_data->blas);
	}
	p_surf_data->blas = RID();
	p_surf_data->blas_size = 0;
	if (p_surf_data->compaction_ticket != 0) {
		cluster_compaction_pending.erase(p_surf_data->compaction_ticket);
		p_surf_data->compaction_ticket = 0;
	}
	if (p_surf_data->cluster_storage.is_valid()) {
		rd->free_rid(p_surf_data->cluster_storage);
		p_surf_data->cluster_storage = RID();
	}
	if (p_surf_data->cluster_args.is_valid()) {
		rd->free_rid(p_surf_data->cluster_args);
		p_surf_data->cluster_args = RID();
	}
	if (p_surf_data->arena_chunk >= 0) {
		_cluster_arena_retire(uint32_t(p_surf_data->arena_chunk), p_surf_data->arena_offset, p_surf_data->arena_size);
		p_surf_data->arena_chunk = -1;
		p_surf_data->arena_offset = 0;
		p_surf_data->arena_size = 0;
	}
	_release_surface_cluster_lod(p_surf_data);
	_release_cluster_geometry(p_surf_data);
	_release_materialized(p_surf_data);
	p_surf_data->geometry_streamed = false;
}

// ---------------------------------------------------------------------------
// Geometry residency (D93b, D93c)
// ---------------------------------------------------------------------------

// A blob's per-vertex strides, as the shader derives them (rt_resolve_triangle): the compressed
// format's normal entry is widened to 8 bytes.
static _FORCE_INLINE_ void _cluster_geometry_strides(const RT_GeometryData &p_geom, uint32_t &r_normal_stride, uint32_t &r_attribute_stride) {
	const bool compressed = (p_geom.flags & RT_GEOM_FLAG_COMPRESSED) != 0;
	r_normal_stride = p_geom.normal_byte_offset != RT_OFFSET_NONE ? (compressed ? 8u : p_geom.normal_stride) : 0u;
	r_attribute_stride = p_geom.attribute_stride; // Not the address: a surface whose geometry streams has none (D93d).
}

// One cluster's section of its blob, and where its attribute block and its indices start in it.
static _FORCE_INLINE_ uint64_t _cluster_geometry_section(uint32_t p_vertices, uint32_t p_triangles, uint32_t p_normal_stride, uint32_t p_attribute_stride, uint64_t &r_attributes_at, uint64_t &r_indices_at) {
	const uint64_t alignment = RT_CLUSTER_GEOMETRY_ALIGNMENT;
	r_attributes_at = _align_up(uint64_t(p_vertices) * p_normal_stride, alignment);
	r_indices_at = r_attributes_at + _align_up(uint64_t(p_vertices) * p_attribute_stride, alignment);
	return r_indices_at + _align_up(uint64_t(p_triangles) * 3, alignment);
}

// Every index of the surface, LOD 0 and the coarse tail, as the stream's clusters cover them.
static _FORCE_INLINE_ uint64_t _lod_stream_index_count(const RTLodStream *p_stream) {
	const uint32_t last = p_stream->cluster_first_triangle.size() - 1;
	return (uint64_t(p_stream->cluster_first_triangle[last]) + (p_stream->cluster_packed[last] & 0xFFFFu)) * 3;
}

// Whether the surface's RAM copy holds what its buffers do: the positions and the normal block,
// every attribute, every level's indices. An update to the surface's vertices or indices drops the
// copy and repopulates the surface, so a load checks it again.
static bool _cluster_geometry_ram_valid(void *p_mesh_surface, const RT_GeometryData &p_geom, uint64_t p_index_count) {
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();
	const Vector<uint8_t> &vertex_data = mesh_storage->mesh_surface_get_ram_vertex_data(p_mesh_surface);
	const Vector<uint8_t> &attribute_data = mesh_storage->mesh_surface_get_ram_attribute_data(p_mesh_surface);
	const Vector<uint8_t> &index_data = mesh_storage->mesh_surface_get_ram_index_data(p_mesh_surface);
	uint32_t normal_stride = 0, attribute_stride = 0;
	_cluster_geometry_strides(p_geom, normal_stride, attribute_stride);
	const uint32_t index_size = p_geom.index_format == RT_INDEX_FORMAT_UINT16 ? sizeof(uint16_t) : sizeof(uint32_t);
	uint64_t vertex_bytes_needed = uint64_t(p_geom.vertex_count) * p_geom.position_stride;
	if (p_geom.normal_byte_offset != RT_OFFSET_NONE) {
		vertex_bytes_needed = MAX(vertex_bytes_needed, uint64_t(p_geom.normal_byte_offset) + uint64_t(p_geom.vertex_count) * p_geom.normal_stride);
	}
	return !vertex_data.is_empty() && !index_data.is_empty() && p_geom.index_format != RT_INDEX_FORMAT_NONE && uint64_t(vertex_data.size()) >= vertex_bytes_needed && uint64_t(attribute_data.size()) >= uint64_t(p_geom.vertex_count) * attribute_stride && uint64_t(index_data.size()) >= p_index_count * index_size;
}

// A group's records: per cluster its address in the group's blob at p_address, its distinct
// vertices | triangles << 16, and its first triangle in [LOD 0 | coarse tail].
static void _cluster_geometry_group_records(const RT_GeometryData &p_geom, const RTLodStream *p_stream, uint32_t p_group, uint64_t p_address, uint32_t *r_records) {
	const RTLodStreamGroup &group = p_stream->groups[p_group];
	uint32_t normal_stride = 0, attribute_stride = 0;
	_cluster_geometry_strides(p_geom, normal_stride, attribute_stride);
	uint64_t at = 0;
	for (uint32_t k = 0; k < group.cluster_count; k++) {
		const uint32_t c = group.first_cluster + k;
		const uint32_t triangles = p_stream->cluster_packed[c] & 0xFFFFu;
		const uint32_t vertices = p_stream->cluster_packed[c] >> 16;
		const uint64_t address = p_address + at;
		uint32_t *record = r_records + uint64_t(k) * RT_CLUSTER_GEOMETRY_RECORD_WORDS;
		record[0] = uint32_t(address);
		record[1] = uint32_t(address >> 32);
		record[2] = vertices | (triangles << 16);
		record[3] = p_stream->cluster_first_triangle[c];
		uint64_t attributes_at = 0, indices_at = 0;
		at += _cluster_geometry_section(vertices, triangles, normal_stride, attribute_stride, attributes_at, indices_at);
	}
}

RTClusterGeometryStage::~RTClusterGeometryStage() {
	if (positions_buffer.is_valid()) {
		RD::get_singleton()->free_rid(positions_buffer);
	}
}

// A streamed surface's cluster geometry. Each group's blob holds its clusters in order, each one
// self-contained: its distinct vertices' normal block (the surface's own encoding, the compressed
// format's entries widened to 8 bytes, the oct16 normal and then the position word whose high half
// carries the tangent angle), their attribute block (the surface's interleaved attributes, byte
// for byte), then its triangles as 8-bit indices into those vertices, every section at
// RT_CLUSTER_GEOMETRY_ALIGNMENT. Positions are absent: a hit takes them from the CLAS, and a build
// reads them staged in the blob's vertex order (D93c). A record per cluster in the LOD table gives a
// hit its cluster's blob, counts and first triangle in [LOD 0 | coarse tail]. The population sizes
// every group's blob and staged positions, gathers the persistent groups' blobs and leaves the other
// records zero: those blobs load and unload with their group's CLAS (D93c). The gathers read the
// surface's RAM copy, which MeshStorage keeps for a surface with a hierarchy while geometry
// streaming is on: the bytes its buffers were made from.
bool RenderRaytracing::_populate_cluster_geometry(void *p_mesh_surface, const RTClusterLayout &p_layout, RTLodStream *p_stream, uint32_t p_cache_key, RTSurfaceData *r_surf_data, RTClusterGeometryStage &r_stage) {
	RD *rd = RD::get_singleton();
	const Vector<uint8_t> &index_data = RendererRD::MeshStorage::get_singleton()->mesh_surface_get_ram_index_data(p_mesh_surface);
	RT_GeometryData &geom = r_surf_data->geometry;
	const uint64_t gather_start = OS::get_singleton()->get_ticks_usec();
	const uint32_t vertex_count = geom.vertex_count;
	const uint32_t cluster_count = p_layout.triangles.size();
	const bool index_16 = geom.index_format == RT_INDEX_FORMAT_UINT16;
	if (p_stream->cluster_first_triangle.size() != cluster_count || !_cluster_geometry_ram_valid(p_mesh_surface, geom, uint64_t(p_layout.total_triangles) * 3)) {
		print_verbose(vformat("RT: cluster BLAS [%d] streams its CLAS but reads its own buffers: no RAM copy of its arrays matches them (D93b).", p_cache_key));
		return false;
	}
	uint32_t normal_stride = 0, attribute_stride = 0;
	_cluster_geometry_strides(geom, normal_stride, attribute_stride);

	// Each cluster's distinct vertices, the count its blob section and its build use: within the
	// 8-bit indices and the bound the builds are sized with, its triangles in order. The test is the
	// one the mesh storage took (D93d).
	LocalVector<uint32_t> distinct;
	if (!mesh_cluster_lod_local_vertices(RendererRD::MeshStorage::get_singleton()->mesh_surface_get_cluster_data(p_mesh_surface), index_data.ptr(), index_data.size(), index_16, vertex_count, cluster_geometry_local, distinct) || distinct.size() != cluster_count) {
		print_verbose(vformat("RT: cluster BLAS [%d] streams its CLAS but reads its own buffers: a cluster indexes past the surface or past %d vertices (D93b).", p_cache_key, MIN(RT_CLUSTER_GEOMETRY_MAX_VERTICES, p_stream->max_vertices)));
		return false;
	}
	bool sized = true;
	for (RTLodStreamGroup &group : p_stream->groups) {
		uint64_t blob = 0, positions = 0;
		for (uint32_t c = group.first_cluster; c < group.first_cluster + group.cluster_count; c++) {
			uint64_t attributes_at = 0, indices_at = 0;
			blob += _cluster_geometry_section(distinct[c], p_layout.triangles[c], normal_stride, attribute_stride, attributes_at, indices_at);
			positions += _align_up(uint64_t(distinct[c]) * geom.position_stride, RT_CLUSTER_GEOMETRY_ALIGNMENT);
		}
		sized = sized && blob <= UINT32_MAX && positions <= UINT32_MAX;
		group.blob_bytes = sized ? uint32_t(blob) : 0;
		group.position_bytes = sized ? uint32_t(positions) : 0;
	}
	uint32_t record_offset = 0;
	const uint32_t record_words = cluster_count * RT_CLUSTER_GEOMETRY_RECORD_WORDS;
	if (!sized || !_cluster_lod_table_allocate(record_words, record_offset)) {
		for (RTLodStreamGroup &group : p_stream->groups) {
			group.blob_bytes = 0;
			group.position_bytes = 0;
		}
		return false;
	}
	// The builds read the clusters' distinct vertices from here on, so the counts become theirs.
	for (uint32_t c = 0; c < cluster_count; c++) {
		p_stream->cluster_packed[c] = p_layout.triangles[c] | (distinct[c] << 16);
	}
	p_stream->persistent_vertices = 0;
	for (uint32_t r = 0; r < p_stream->persistent_runs.size(); r += 2) {
		for (uint32_t c = p_stream->persistent_runs[r]; c < p_stream->persistent_runs[r] + p_stream->persistent_runs[r + 1]; c++) {
			p_stream->persistent_vertices += distinct[c];
		}
	}

	// The records, in the LOD table, zero but for the persistent groups'.
	r_surf_data->cluster_geometry_offset = record_offset;
	r_surf_data->cluster_geometry_words = record_words;
	r_surf_data->cluster_geometry_blobs.resize(p_stream->groups.size());
	for (RTClusterArenaRegion &region : r_surf_data->cluster_geometry_blobs) {
		region = RTClusterArenaRegion();
	}
	geom.flags |= RT_GEOM_FLAG_CLUSTER_GEOMETRY;
	geom.cluster_record_base = record_offset;

	// The persistent groups' blobs in cluster order, the population's build order.
	LocalVector<uint64_t> persistent; // first cluster << 32 | group.
	for (uint32_t g = 0; g < p_stream->groups.size(); g++) {
		if (p_stream->groups[g].state == RTLodStreamGroup::PERSISTENT) {
			persistent.push_back((uint64_t(p_stream->groups[g].first_cluster) << 32) | g);
		}
	}
	persistent.sort();
	LocalVector<RTLodStreamLoad::Item> items;
	for (const uint64_t key : persistent) {
		RTLodStreamLoad::Item item;
		item.surface = r_surf_data;
		item.group = uint32_t(key);
		items.push_back(item);
	}
	// The gather reads a group through its surface, which the population attaches its stream to only
	// once the build is recorded. The persistent groups are few and coarse: gathered here.
	RTLodStream *attached = r_surf_data->cluster_lod_stream;
	r_surf_data->cluster_lod_stream = p_stream;
	RTClusterGeometryGather gather;
	bool ok = _prepare_cluster_geometry_gather(items, gather);
	for (uint32_t i = 0; i < gather.items.size() && ok; i++) {
		gather.gather_item(i, nullptr);
		ok = gather.item_ok[i] != 0;
	}
	ok = ok && _place_cluster_geometry(items, gather, r_stage);
	r_surf_data->cluster_lod_stream = attached;
	if (ok) {
		LocalVector<uint32_t> records;
		records.resize(record_words);
		memset(records.ptr(), 0, sizeof(uint32_t) * record_words);
		for (const RTLodStreamLoad::Item &item : items) {
			const RTClusterArenaRegion &region = r_surf_data->cluster_geometry_blobs[item.group];
			_cluster_geometry_group_records(geom, p_stream, item.group, cluster_geometry_arena[region.chunk].address + region.offset, records.ptr() + uint64_t(p_stream->groups[item.group].first_cluster) * RT_CLUSTER_GEOMETRY_RECORD_WORDS);
		}
		ok = rd->buffer_update(cluster_lod_table, record_offset * sizeof(uint32_t), record_words * sizeof(uint32_t), records.ptr()) == OK;
	}
	if (!ok) {
		_release_cluster_geometry(r_surf_data);
		for (RTLodStreamGroup &group : p_stream->groups) {
			group.blob_bytes = 0;
			group.position_bytes = 0;
		}
		return false;
	}
	p_stream->persistent_blob_bytes = r_stage.blob_bytes;
	uint64_t blob_bytes_total = 0;
	for (const RTLodStreamGroup &group : p_stream->groups) {
		blob_bytes_total += group.blob_bytes;
	}
	print_verbose(vformat("RT: cluster BLAS [%d] reads its geometry from blobs (D93c): %d clusters in %d groups, %d KB of persistent blobs and %d KB of their staged positions, %d KB of blobs in all, %.1f bytes a triangle, gathered in %d us.", p_cache_key, cluster_count, p_stream->groups.size(), r_stage.blob_bytes / 1024, gather.positions.size() / 1024, blob_bytes_total / 1024, double(blob_bytes_total) / double(MAX(p_layout.total_triangles, 1u)), OS::get_singleton()->get_ticks_usec() - gather_start));
	return true;
}

// A gather's layout (D93c): each item's blob and positions slices from its clusters' counts, and its
// surface's RAM arrays shared by refcount, so the gather reads what the render thread may replace or
// free meanwhile without a lock. An item whose surface reads its own buffers keeps no gather. False
// when the slices would not fit.
bool RenderRaytracing::_prepare_cluster_geometry_gather(LocalVector<RTLodStreamLoad::Item> &r_items, RTClusterGeometryGather &r_gather) {
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();
	LocalVector<void *> source_surfaces;
	uint64_t blob_cursor = 0;
	uint64_t positions_cursor = 0;
	for (RTLodStreamLoad::Item &item : r_items) {
		const RT_GeometryData &geom = item.surface->geometry;
		if ((geom.flags & RT_GEOM_FLAG_CLUSTER_GEOMETRY) == 0) {
			item.gather = UINT32_MAX;
			continue;
		}
		const RTLodStream *stream = item.surface->cluster_lod_stream;
		const RTLodStreamGroup &group = stream->groups[item.group];
		int64_t source = source_surfaces.find(stream->mesh_surface);
		if (source < 0) {
			source = source_surfaces.size();
			source_surfaces.push_back(stream->mesh_surface);
			RTClusterGeometryGather::Source added;
			added.vertex_data = mesh_storage->mesh_surface_get_ram_vertex_data(stream->mesh_surface);
			added.attribute_data = mesh_storage->mesh_surface_get_ram_attribute_data(stream->mesh_surface);
			added.index_data = mesh_storage->mesh_surface_get_ram_index_data(stream->mesh_surface);
			added.vertex_count = geom.vertex_count;
			added.position_stride = geom.position_stride;
			added.has_normal = geom.normal_byte_offset != RT_OFFSET_NONE;
			added.normal_byte_offset = added.has_normal ? geom.normal_byte_offset : 0;
			added.normal_source_stride = geom.normal_stride;
			_cluster_geometry_strides(geom, added.normal_stride, added.attribute_stride);
			added.compressed = (geom.flags & RT_GEOM_FLAG_COMPRESSED) != 0;
			added.index_16 = geom.index_format == RT_INDEX_FORMAT_UINT16;
			r_gather.sources.push_back(added);
		}
		const RTClusterGeometryGather::Source &from = r_gather.sources[source];
		RTClusterGeometryGather::Item gathered;
		gathered.source = uint32_t(source);
		gathered.first_cluster = r_gather.cluster_packed.size();
		gathered.cluster_count = group.cluster_count;
		gathered.blob_offset = blob_cursor;
		gathered.blob_size = group.blob_bytes;
		for (uint32_t c = group.first_cluster; c < group.first_cluster + group.cluster_count; c++) {
			const uint32_t triangles = stream->cluster_packed[c] & 0xFFFFu;
			const uint32_t vertices = stream->cluster_packed[c] >> 16;
			r_gather.cluster_packed.push_back(stream->cluster_packed[c]);
			r_gather.cluster_first_triangle.push_back(stream->cluster_first_triangle[c]);
			r_gather.cluster_at.push_back(blob_cursor);
			r_gather.positions_at.push_back(positions_cursor);
			uint64_t attributes_at = 0, indices_at = 0;
			blob_cursor += _cluster_geometry_section(vertices, triangles, from.normal_stride, from.attribute_stride, attributes_at, indices_at);
			positions_cursor += _align_up(uint64_t(vertices) * from.position_stride, RT_CLUSTER_GEOMETRY_ALIGNMENT);
		}
		if (blob_cursor - gathered.blob_offset != gathered.blob_size) {
			return false; // The slices would overlap: the sizing and the layout disagree.
		}
		item.gather = r_gather.items.size();
		r_gather.items.push_back(gathered);
	}
	if (blob_cursor > UINT32_MAX || positions_cursor > UINT32_MAX) {
		return false;
	}
	r_gather.blobs.resize(blob_cursor);
	r_gather.positions.resize(positions_cursor);
	r_gather.item_ok.resize(r_gather.items.size());
	return true;
}

// One item of a gather, on any thread: its blob's and its positions' slices, each cluster's distinct
// vertices in first-use order through a small open-addressed map (at most RT_CLUSTER_GEOMETRY_MAX_VERTICES
// a cluster), so the output is the population's layout byte for byte.
void RTClusterGeometryGather::gather_item(uint32_t p_item, void *p_unused) {
	static constexpr uint32_t MAP_BITS = 9;
	static constexpr uint32_t MAP_SIZE = 1u << MAP_BITS;
	static_assert(MAP_SIZE >= 2 * RT_CLUSTER_GEOMETRY_MAX_VERTICES && RT_CLUSTER_GEOMETRY_MAX_VERTICES <= 256);
	const Item &item = items[p_item];
	const Source &from = sources[item.source];
	const uint8_t *vertex_bytes = from.vertex_data.ptr();
	const uint8_t *attribute_bytes = from.attribute_data.ptr();
	const uint8_t *index_bytes = from.index_data.ptr();
	memset(blobs.ptr() + item.blob_offset, 0, item.blob_size);
	const uint32_t last = item.first_cluster + item.cluster_count - 1;
	const uint64_t positions_end = positions_at[last] + _align_up(uint64_t(cluster_packed[last] >> 16) * from.position_stride, RT_CLUSTER_GEOMETRY_ALIGNMENT);
	memset(positions.ptr() + positions_at[item.first_cluster], 0, positions_end - positions_at[item.first_cluster]);
	uint32_t keys[MAP_SIZE];
	uint8_t values[MAP_SIZE];
	bool ok = true;
	for (uint32_t k = 0; k < item.cluster_count && ok; k++) {
		const uint32_t c = item.first_cluster + k;
		const uint32_t triangles = cluster_packed[c] & 0xFFFFu;
		const uint32_t vertices = cluster_packed[c] >> 16;
		uint64_t attributes_at = 0, indices_at = 0;
		_cluster_geometry_section(vertices, triangles, from.normal_stride, from.attribute_stride, attributes_at, indices_at);
		uint8_t *normals = blobs.ptr() + cluster_at[c];
		uint8_t *attributes = normals + attributes_at;
		uint8_t *local_indices = normals + indices_at;
		uint8_t *cluster_positions = positions.ptr() + positions_at[c];
		memset(keys, 0xFF, sizeof(keys));
		uint32_t count = 0;
		const uint64_t first = uint64_t(cluster_first_triangle[c]) * 3;
		for (uint64_t i = 0; i < uint64_t(triangles) * 3; i++) {
			const uint32_t vertex = from.index_16 ? reinterpret_cast<const uint16_t *>(index_bytes)[first + i] : reinterpret_cast<const uint32_t *>(index_bytes)[first + i];
			ok = vertex < from.vertex_count; // The population's sizing checked it, and the copy is the one it read.
			if (!ok) {
				break;
			}
			uint32_t slot = (vertex * 2654435761u) >> (32 - MAP_BITS);
			while (keys[slot] != UINT32_MAX && keys[slot] != vertex) {
				slot = (slot + 1) & (MAP_SIZE - 1);
			}
			if (keys[slot] == UINT32_MAX) {
				ok = count < vertices;
				if (!ok) {
					break;
				}
				keys[slot] = vertex;
				values[slot] = uint8_t(count);
				if (from.compressed && from.has_normal) {
					memcpy(normals + count * 8, vertex_bytes + from.normal_byte_offset + uint64_t(vertex) * from.normal_source_stride, sizeof(uint32_t));
					memcpy(normals + count * 8 + 4, vertex_bytes + uint64_t(vertex) * from.position_stride + 4, sizeof(uint32_t));
				} else if (from.has_normal) {
					memcpy(normals + uint64_t(count) * from.normal_stride, vertex_bytes + from.normal_byte_offset + uint64_t(vertex) * from.normal_source_stride, from.normal_stride);
				}
				if (from.attribute_stride > 0) {
					memcpy(attributes + uint64_t(count) * from.attribute_stride, attribute_bytes + uint64_t(vertex) * from.attribute_stride, from.attribute_stride);
				}
				memcpy(cluster_positions + uint64_t(count) * from.position_stride, vertex_bytes + uint64_t(vertex) * from.position_stride, from.position_stride);
				count++;
			}
			local_indices[i] = values[slot];
		}
		ok = ok && count == vertices;
	}
	item_ok[p_item] = ok ? 1 : 0;
}

// A gather's placement (D93c), on the render thread: each gathered item's blob into its own region of
// the cluster geometry arena, uploaded from the gather (a region that follows the previous one in its
// chunk and in the gather goes up in the same update: one transfer, not a barrier per group), the
// positions into a buffer of the build's own, and per cluster of the build its indices' address and
// its positions' offset; zeros for an item whose surface reads its own buffers.
bool RenderRaytracing::_place_cluster_geometry(const LocalVector<RTLodStreamLoad::Item> &p_items, const RTClusterGeometryGather &p_gather, RTClusterGeometryStage &r_stage) {
	RD *rd = RD::get_singleton();
	int32_t run_chunk = -1;
	uint64_t run_offset = 0;
	uint64_t run_size = 0;
	const uint8_t *run_data = nullptr;
	bool ok = true;
	uint32_t placed = 0;
	for (; placed < p_items.size(); placed++) {
		const RTLodStreamLoad::Item &item = p_items[placed];
		const RTLodStreamGroup &group = item.surface->cluster_lod_stream->groups[item.group];
		if (item.gather == UINT32_MAX) {
			for (uint32_t k = 0; k < group.cluster_count; k++) {
				r_stage.index_addresses.push_back(0);
				r_stage.position_offsets.push_back(0);
			}
			continue;
		}
		const RTClusterGeometryGather::Item &gathered = p_gather.items[item.gather];
		const RTClusterGeometryGather::Source &from = p_gather.sources[gathered.source];
		uint32_t chunk_index = 0;
		uint64_t offset = 0;
		ok = _arena_allocate(cluster_geometry_arena, cluster_geometry_arena_reserved_bytes, cluster_geometry_arena_live_bytes, RT_CLUSTER_GEOMETRY_ALIGNMENT,
				RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT, RDD::PIPELINE_STAGE_RAY_TRACING_SHADER_BIT, "RT cluster geometry arena", gathered.blob_size, chunk_index, offset);
		if (!ok) {
			break;
		}
		item.surface->cluster_geometry_blobs[item.group] = { int32_t(chunk_index), offset, gathered.blob_size };
		const uint8_t *data = p_gather.blobs.ptr() + gathered.blob_offset;
		if (run_chunk == int32_t(chunk_index) && run_offset + run_size == offset && run_data + run_size == data) {
			run_size += gathered.blob_size;
		} else {
			if (run_chunk >= 0) {
				const RTClusterArenaChunk &run_target = cluster_geometry_arena[run_chunk];
				rd->buffer_update(run_target.buffer, run_target.base_offset + run_offset, run_size, run_data);
			}
			run_chunk = int32_t(chunk_index);
			run_offset = offset;
			run_size = gathered.blob_size;
			run_data = data;
		}
		const RTClusterArenaChunk &chunk = cluster_geometry_arena[chunk_index];
		if (!r_stage.blob_buffers.has(chunk.buffer)) {
			r_stage.blob_buffers.push_back(chunk.buffer);
		}
		r_stage.blob_bytes += gathered.blob_size;
		const uint64_t base = chunk.address + offset;
		for (uint32_t k = 0; k < gathered.cluster_count; k++) {
			const uint32_t c = gathered.first_cluster + k;
			uint64_t attributes_at = 0, indices_at = 0;
			_cluster_geometry_section(p_gather.cluster_packed[c] >> 16, p_gather.cluster_packed[c] & 0xFFFFu, from.normal_stride, from.attribute_stride, attributes_at, indices_at);
			r_stage.index_addresses.push_back(base + (p_gather.cluster_at[c] - gathered.blob_offset) + indices_at);
			r_stage.position_offsets.push_back(p_gather.positions_at[c]);
		}
	}
	if (ok && run_chunk >= 0) {
		const RTClusterArenaChunk &run_target = cluster_geometry_arena[run_chunk];
		rd->buffer_update(run_target.buffer, run_target.base_offset + run_offset, run_size, run_data);
	}
	if (ok && !p_gather.positions.is_empty()) {
		r_stage.positions_buffer = rd->storage_buffer_create(p_gather.positions.size(), Span<uint8_t>(p_gather.positions.ptr(), p_gather.positions.size()), 0,
				RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
		ok = r_stage.positions_buffer.is_valid();
		if (ok) {
			rd->set_resource_name(r_stage.positions_buffer, "RT cluster geometry positions");
			r_stage.positions_address = rd->buffer_get_device_address(r_stage.positions_buffer);
		}
	}
	if (!ok) {
		for (uint32_t i = 0; i < placed && i < p_items.size(); i++) {
			_retire_cluster_geometry_blob(p_items[i].surface, p_items[i].group);
		}
		if (placed < p_items.size()) {
			_retire_cluster_geometry_blob(p_items[placed].surface, p_items[placed].group);
		}
		WARN_PRINT_ONCE("RT: streamed cluster geometry could not be placed; its groups stay at a coarser level.");
		return false;
	}
	return true;
}

// A loaded group's records, in place before the group turns resident (D93c).
void RenderRaytracing::_write_cluster_geometry_records(const RTSurfaceData *p_surf_data, uint32_t p_group) {
	const RTLodStream *stream = p_surf_data->cluster_lod_stream;
	const RTLodStreamGroup &group = stream->groups[p_group];
	const RTClusterArenaRegion &region = p_surf_data->cluster_geometry_blobs[p_group];
	LocalVector<uint32_t> records;
	records.resize(group.cluster_count * RT_CLUSTER_GEOMETRY_RECORD_WORDS);
	_cluster_geometry_group_records(p_surf_data->geometry, stream, p_group, cluster_geometry_arena[region.chunk].address + region.offset, records.ptr());
	RD::get_singleton()->buffer_update(cluster_lod_table, (p_surf_data->cluster_geometry_offset + group.first_cluster * RT_CLUSTER_GEOMETRY_RECORD_WORDS) * sizeof(uint32_t), sizeof(uint32_t) * records.size(), records.ptr());
}

// A group's blob back to the arena past the frames that may trace it or a recorded build may read it.
void RenderRaytracing::_retire_cluster_geometry_blob(RTSurfaceData *p_surf_data, uint32_t p_group) {
	if (p_group >= p_surf_data->cluster_geometry_blobs.size() || p_surf_data->cluster_geometry_blobs[p_group].chunk < 0) {
		return;
	}
	RTClusterArenaRegion &region = p_surf_data->cluster_geometry_blobs[p_group];
	RTClusterRetiredRegion retired;
	retired.chunk = uint32_t(region.chunk);
	retired.offset = region.offset;
	retired.size = region.size;
	retired.frame = RD::get_singleton()->get_frames_drawn();
	cluster_geometry_arena_retired.push_back(retired);
	region = RTClusterArenaRegion();
}

// The surface's records go back to the LOD table (a later write there is ordered behind the trace
// that binds the table) and its blobs back to the arena past the frames that may trace them.
void RenderRaytracing::_release_cluster_geometry(RTSurfaceData *p_surf_data) {
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	for (const RTClusterArenaRegion &region : p_surf_data->cluster_geometry_blobs) {
		if (region.chunk >= 0) {
			RTClusterRetiredRegion retired;
			retired.chunk = uint32_t(region.chunk);
			retired.offset = region.offset;
			retired.size = region.size;
			retired.frame = frames_drawn;
			cluster_geometry_arena_retired.push_back(retired);
		}
	}
	p_surf_data->cluster_geometry_blobs.clear();
	if (p_surf_data->cluster_geometry_offset != UINT32_MAX) {
		cluster_lod_table_live_words -= p_surf_data->cluster_geometry_words;
		_cluster_lod_table_free(p_surf_data->cluster_geometry_offset, p_surf_data->cluster_geometry_words);
	}
	p_surf_data->cluster_geometry_offset = UINT32_MAX;
	p_surf_data->cluster_geometry_words = 0;
	p_surf_data->geometry.flags &= ~uint32_t(RT_GEOM_FLAG_CLUSTER_GEOMETRY);
	p_surf_data->geometry.cluster_record_base = 0;
}

// ---------------------------------------------------------------------------
// Cluster LOD selection (D86)
// ---------------------------------------------------------------------------

static _FORCE_INLINE_ uint32_t _float_bits(float p_value) {
	uint32_t bits;
	memcpy(&bits, &p_value, sizeof(bits));
	return bits;
}

static _FORCE_INLINE_ float _bits_float(uint32_t p_bits) {
	float value;
	memcpy(&value, &p_bits, sizeof(value));
	return value;
}

// A region of the LOD table, first fit over its free ranges; the table grows by a copy on the
// GPU when none fits, the old buffer freed once the frames in flight are done with it.
bool RenderRaytracing::_cluster_lod_table_allocate(uint32_t p_words, uint32_t &r_offset) {
	ERR_FAIL_COND_V(p_words == 0, false);
	for (uint32_t i = 0; i < cluster_lod_table_free.size(); i++) {
		RTClusterArenaChunk::FreeRange &range = cluster_lod_table_free[i];
		if (range.size < p_words) {
			continue;
		}
		r_offset = uint32_t(range.offset);
		range.offset += p_words;
		range.size -= p_words;
		if (range.size == 0) {
			cluster_lod_table_free.remove_at(i);
		}
		cluster_lod_table_live_words += p_words;
		return true;
	}

	RD *rd = RD::get_singleton();
	const uint64_t needed = uint64_t(cluster_lod_table_capacity) + p_words;
	const uint64_t capacity = MAX(MAX(uint64_t(cluster_lod_table_capacity) * 2, needed), uint64_t(1u << 16));
	ERR_FAIL_COND_V_MSG(capacity * sizeof(uint32_t) > UINT32_MAX, false, "RT: the cluster LOD table exceeds 4 GB.");
	RID buffer = rd->storage_buffer_create(uint32_t(capacity * sizeof(uint32_t)));
	if (buffer.is_null()) {
		return false;
	}
	rd->set_resource_name(buffer, "RT cluster LOD table");
	if (cluster_lod_table.is_valid()) {
		rd->buffer_copy(cluster_lod_table, buffer, 0, 0, cluster_lod_table_capacity * sizeof(uint32_t));
		rd->free_rid(cluster_lod_table); // Deferred past the fence of every frame in flight.
	}
	const uint32_t old_capacity = cluster_lod_table_capacity;
	cluster_lod_table = buffer;
	cluster_lod_table_capacity = uint32_t(capacity);
	// The new tail joins the free ranges (merged with the last one if it reaches the old end), then
	// the request is taken from it.
	_cluster_lod_table_free(old_capacity, cluster_lod_table_capacity - old_capacity);
	RTClusterArenaChunk::FreeRange &tail = cluster_lod_table_free[cluster_lod_table_free.size() - 1];
	r_offset = uint32_t(tail.offset);
	tail.offset += p_words;
	tail.size -= p_words;
	if (tail.size == 0) {
		cluster_lod_table_free.remove_at(cluster_lod_table_free.size() - 1);
	}
	cluster_lod_table_live_words += p_words;
	return true;
}

// Returns a region to the free ranges, kept in offset order and merged with either neighbour.
void RenderRaytracing::_cluster_lod_table_free(uint32_t p_offset, uint32_t p_words) {
	if (p_words == 0) {
		return;
	}
	uint32_t at = 0;
	while (at < cluster_lod_table_free.size() && cluster_lod_table_free[at].offset < p_offset) {
		at++;
	}
	cluster_lod_table_free.insert(at, { p_offset, p_words });
	if (at + 1 < cluster_lod_table_free.size() && cluster_lod_table_free[at].offset + cluster_lod_table_free[at].size == cluster_lod_table_free[at + 1].offset) {
		cluster_lod_table_free[at].size += cluster_lod_table_free[at + 1].size;
		cluster_lod_table_free.remove_at(at + 1);
	}
	if (at > 0 && cluster_lod_table_free[at - 1].offset + cluster_lod_table_free[at - 1].size == cluster_lod_table_free[at].offset) {
		cluster_lod_table_free[at - 1].size += cluster_lod_table_free[at].size;
		cluster_lod_table_free.remove_at(at);
	}
}

// The traversal's tree over a hierarchy's groups (D91), after vk_lod_clusters' spatial hierarchy:
// per depth, the groups sorted along a Morton curve over their centers and packed 32 to a node,
// nodes packed 32 to a parent until one is left, the depths' roots under one root the same way;
// then laid out breadth first, so a node's children are contiguous, the refs giving each leaf's
// groups. A node's sphere encloses its children's stored spheres, rounded up, and its error is
// their largest, the order the selection's node test needs to prune nothing the exhaustive test
// would select. False when a group's clusters are not exactly the contiguous run its record names,
// which the cluster pass assumes, or when a group's sphere or error is not a number, which the
// node bounds' MIN and MAX would carry or drop by position; the caller then discards the outputs
// and the surface keeps the exhaustive test.
bool RenderRaytracing::_build_cluster_lod_tree(const uint32_t *p_cluster_words, uint32_t p_cluster_count, const uint32_t *p_group_words, uint32_t p_group_count, LocalVector<uint32_t> &r_words, LocalVector<uint32_t> &r_level_nodes, uint32_t &r_node_count) {
	const uint32_t GROUP_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS;
	const uint32_t CLUSTER_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
	const uint32_t FANOUT = 32; // The lanes of half a workgroup, one per child.
	r_words.clear();
	r_level_nodes.clear();
	r_node_count = 0;
	uint32_t depth_count = 0;
	for (uint32_t g = 0; g < p_group_count; g++) {
		const uint32_t *w = p_group_words + uint64_t(g) * GROUP_WORDS;
		if (uint64_t(w[1]) + w[2] > p_cluster_count || w[0] >= p_group_count) {
			return false;
		}
		for (int k = 3; k < 8; k++) {
			if (Math::is_nan(_bits_float(w[k]))) {
				return false;
			}
		}
		for (uint32_t c = w[1]; c < w[1] + w[2]; c++) {
			if (p_cluster_words[uint64_t(c) * CLUSTER_WORDS + 1] != g) {
				return false;
			}
		}
		depth_count = MAX(depth_count, w[0] + 1);
	}
	for (uint32_t c = 0; c < p_cluster_count; c++) {
		const uint32_t *w = p_group_words + uint64_t(p_cluster_words[uint64_t(c) * CLUSTER_WORDS + 1]) * GROUP_WORDS;
		if (c < w[1] || c - w[1] >= w[2]) {
			return false; // Its group's run does not hold it.
		}
	}

	struct Node {
		float center[3] = {};
		float radius = 0.0f;
		float error = 0.0f;
		bool leaf = false; // Its children are groups.
		uint32_t first = 0; // Into the children scratch.
		uint32_t count = 0;
	};
	LocalVector<Node> nodes;
	LocalVector<uint32_t> children; // Per node, its children (nodes, or groups for a leaf) contiguous.
	// A node over children[p_first, p_first + p_count): the box of their spheres' centers, the
	// radius in double from the stored float center, rounded up, the largest error.
	auto make_node = [&](uint32_t p_first, uint32_t p_count, bool p_leaf) -> uint32_t {
		double lo[3] = { 0.0, 0.0, 0.0 }, hi[3] = { 0.0, 0.0, 0.0 };
		float error = 0.0f;
		for (uint32_t i = 0; i < p_count; i++) {
			const uint32_t child = children[p_first + i];
			float center[3], radius, child_error;
			if (p_leaf) {
				const uint32_t *w = p_group_words + uint64_t(child) * GROUP_WORDS;
				center[0] = _bits_float(w[3]);
				center[1] = _bits_float(w[4]);
				center[2] = _bits_float(w[5]);
				radius = _bits_float(w[6]);
				child_error = _bits_float(w[7]);
			} else {
				const Node &n = nodes[child];
				center[0] = n.center[0];
				center[1] = n.center[1];
				center[2] = n.center[2];
				radius = n.radius;
				child_error = n.error;
			}
			for (int a = 0; a < 3; a++) {
				const double l = double(center[a]) - double(radius);
				const double h = double(center[a]) + double(radius);
				lo[a] = i == 0 ? l : MIN(lo[a], l);
				hi[a] = i == 0 ? h : MAX(hi[a], h);
			}
			error = i == 0 ? child_error : MAX(error, child_error);
		}
		Node node;
		for (int a = 0; a < 3; a++) {
			node.center[a] = float((lo[a] + hi[a]) * 0.5);
		}
		double radius = 0.0;
		for (uint32_t i = 0; i < p_count; i++) {
			const uint32_t child = children[p_first + i];
			double d[3];
			double child_radius;
			if (p_leaf) {
				const uint32_t *w = p_group_words + uint64_t(child) * GROUP_WORDS;
				for (int a = 0; a < 3; a++) {
					d[a] = double(_bits_float(w[3 + a])) - double(node.center[a]);
				}
				child_radius = double(_bits_float(w[6]));
			} else {
				for (int a = 0; a < 3; a++) {
					d[a] = double(nodes[child].center[a]) - double(node.center[a]);
				}
				child_radius = double(nodes[child].radius);
			}
			radius = MAX(radius, Math::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) + child_radius);
		}
		node.radius = float(radius * (1.0 + 1.0 / 65536.0)); // Above the double value after rounding to float.
		node.error = error;
		node.leaf = p_leaf;
		node.first = p_first;
		node.count = p_count;
		nodes.push_back(node);
		return nodes.size() - 1;
	};
	// Packs a level of nodes (or groups, for the leaves) 32 at a time, in their order, into parents.
	auto pack = [&](const LocalVector<uint32_t> &p_items, bool p_leaf) -> LocalVector<uint32_t> {
		LocalVector<uint32_t> parents;
		for (uint32_t i = 0; i < p_items.size(); i += FANOUT) {
			const uint32_t count = MIN(FANOUT, p_items.size() - i);
			const uint32_t first = children.size();
			for (uint32_t k = 0; k < count; k++) {
				children.push_back(p_items[i + k]);
			}
			parents.push_back(make_node(first, count, p_leaf));
		}
		return parents;
	};

	struct Keyed {
		uint32_t key = 0;
		uint32_t group = 0;
		bool operator<(const Keyed &p_other) const {
			return key != p_other.key ? key < p_other.key : group < p_other.group;
		}
	};
	LocalVector<uint32_t> tops; // Each depth's root, depth order.
	LocalVector<Keyed> keyed;
	for (uint32_t depth = 0; depth < depth_count; depth++) {
		keyed.clear();
		double lo[3] = { 0.0, 0.0, 0.0 }, hi[3] = { 0.0, 0.0, 0.0 };
		for (uint32_t g = 0; g < p_group_count; g++) {
			const uint32_t *w = p_group_words + uint64_t(g) * GROUP_WORDS;
			if (w[0] != depth) {
				continue;
			}
			for (int a = 0; a < 3; a++) {
				const double v = double(_bits_float(w[3 + a]));
				lo[a] = keyed.is_empty() ? v : MIN(lo[a], v);
				hi[a] = keyed.is_empty() ? v : MAX(hi[a], v);
			}
			Keyed k;
			k.group = g;
			keyed.push_back(k);
		}
		if (keyed.is_empty()) {
			continue;
		}
		for (Keyed &k : keyed) {
			const uint32_t *w = p_group_words + uint64_t(k.group) * GROUP_WORDS;
			uint32_t key = 0;
			for (int a = 0; a < 3; a++) {
				const double extent = hi[a] - lo[a];
				const double t = extent > 0.0 ? (double(_bits_float(w[3 + a])) - lo[a]) / extent : 0.0;
				uint32_t q = uint32_t(CLAMP(t * 1023.0 + 0.5, 0.0, 1023.0));
				// Spread ten bits three apart.
				q = (q | (q << 16)) & 0x030000FF;
				q = (q | (q << 8)) & 0x0300F00F;
				q = (q | (q << 4)) & 0x030C30C3;
				q = (q | (q << 2)) & 0x09249249;
				key |= q << a;
			}
			k.key = key;
		}
		keyed.sort();
		LocalVector<uint32_t> level;
		for (const Keyed &k : keyed) {
			level.push_back(k.group);
		}
		level = pack(level, true);
		while (level.size() > 1) {
			level = pack(level, false);
		}
		tops.push_back(level[0]);
	}
	while (tops.size() > 1) {
		tops = pack(tops, false);
	}
	ERR_FAIL_COND_V(tops.is_empty(), false);

	// Breadth first from the root: a node's children are enqueued together, so they are
	// contiguous; a leaf's groups take the next run of the refs.
	LocalVector<uint32_t> order;
	LocalVector<uint32_t> depth_of;
	order.push_back(tops[0]);
	depth_of.push_back(0);
	for (uint32_t i = 0; i < order.size(); i++) {
		const Node &n = nodes[order[i]];
		if (r_level_nodes.size() <= depth_of[i]) {
			r_level_nodes.push_back(0);
		}
		r_level_nodes[depth_of[i]]++;
		if (!n.leaf) {
			for (uint32_t k = 0; k < n.count; k++) {
				order.push_back(children[n.first + k]);
				depth_of.push_back(depth_of[i] + 1);
			}
		}
	}
	const uint32_t node_count = order.size();
	r_words.resize(uint64_t(node_count) * 8 + p_group_count);
	uint32_t next_child = 1;
	uint32_t next_ref = 0;
	for (uint32_t i = 0; i < node_count; i++) {
		const Node &n = nodes[order[i]];
		uint32_t *w = r_words.ptr() + uint64_t(i) * 8;
		w[0] = _float_bits(n.center[0]);
		w[1] = _float_bits(n.center[1]);
		w[2] = _float_bits(n.center[2]);
		w[3] = _float_bits(n.radius);
		w[4] = _float_bits(n.error);
		w[6] = n.count | (n.leaf ? 0x80000000u : 0u);
		w[7] = 0;
		if (n.leaf) {
			w[5] = next_ref;
			for (uint32_t k = 0; k < n.count; k++) {
				r_words[uint64_t(node_count) * 8 + next_ref++] = children[n.first + k];
			}
		} else {
			w[5] = next_child; // Its children's breadth-first places, as enqueued above.
			next_child += n.count;
		}
	}
	ERR_FAIL_COND_V(next_ref != p_group_count || next_child != node_count, false);
	r_node_count = node_count;
	return true;
}

// The surface's hierarchy as the selection reads it: its region of the LOD table (the CLAS
// addresses copied on the GPU from the argument buffer the implicit build writes them into, each
// cluster's group and refined group, the groups' spheres and errors), and the worst case of a BLAS
// over any selection (no selection takes more clusters than the hierarchy holds), each instance's
// slot. On failure nothing is kept and the surface's LOD 0 BLAS is traced everywhere.
void RenderRaytracing::_populate_surface_cluster_lod(void *p_mesh_surface, const RTClusterLayout &p_layout, RID p_args_buffer, uint64_t p_implicit_addresses_offset, uint32_t p_cache_key, RTSurfaceData *r_surf_data) {
	RD *rd = RD::get_singleton();
	// What a failure leaves traced: LOD 0's BLAS, or a streamed surface's coarsest cut (D92).
	const char *fallback = r_surf_data->cluster_lod_stream ? "traced at its coarsest level" : "traced at LOD 0";
	const Vector<uint8_t> &table_bytes = RendererRD::MeshStorage::get_singleton()->mesh_surface_get_cluster_data(p_mesh_surface);
	const uint32_t cluster_count = p_layout.triangles.size();
	if (table_bytes.size() < int64_t(RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS * sizeof(uint32_t))) {
		return;
	}
	const uint32_t *table = reinterpret_cast<const uint32_t *>(table_bytes.ptr());
	const uint32_t group_count = table[2];
	if (table[0] != RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION || table[1] != cluster_count || group_count == 0) {
		return; // The layout took every level of this very table (levels > 1), so the counts agree.
	}
	const uint32_t *cluster_words = table + RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS;
	const uint32_t *group_words = cluster_words + uint64_t(RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS) * cluster_count;
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t *w = cluster_words + uint64_t(c) * RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
		const int32_t refined = int32_t(w[2]);
		if (w[1] >= group_count || refined >= int32_t(group_count)) {
			WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: the hierarchy's links leave its groups; %s.", p_cache_key, fallback));
			return;
		}
	}

	// The classification's bounds per depth (D89): the groups' centers' box gives the sphere's
	// center, then the radius enclosing every group's sphere, the farthest center, the smallest
	// radius and the error range, all over the very floats the selection reads. The sharing rests
	// on every cluster of depth d refining a group of depth d - 1 (meshoptimizer's levels are
	// global); a table that breaks it keeps its selection and is never shared.
	const uint32_t GROUP_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS;
	uint32_t level_count = table[3];
	bool levels_hold = level_count > 0 && level_count <= group_count;
	for (uint32_t g = 0; g < group_count && levels_hold; g++) {
		levels_hold = group_words[uint64_t(g) * GROUP_WORDS] < level_count;
	}
	for (uint32_t c = 0; c < cluster_count && levels_hold; c++) {
		const uint32_t *w = cluster_words + uint64_t(c) * RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
		const uint32_t depth = group_words[uint64_t(w[1]) * GROUP_WORDS];
		const int32_t refined = int32_t(w[2]);
		levels_hold = refined < 0 ? depth == 0 : depth > 0 && group_words[uint64_t(refined) * GROUP_WORDS] == depth - 1;
	}
	if (!levels_hold) {
		WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: the hierarchy's depths are not one level per refinement; its instances never share a build.", p_cache_key));
		level_count = 0;
	}
	LocalVector<RTLodLevelBound> levels;
	levels.resize(level_count);
	LocalVector<AABB> level_boxes;
	level_boxes.resize(level_count);
	for (uint32_t g = 0; g < group_count && level_count > 0; g++) {
		const uint32_t *w = group_words + uint64_t(g) * GROUP_WORDS;
		const Vector3 center(_bits_float(w[3]), _bits_float(w[4]), _bits_float(w[5]));
		const float radius = _bits_float(w[6]);
		const float error = _bits_float(w[7]);
		RTLodLevelBound &level = levels[w[0]];
		if (level.empty) {
			level.empty = false;
			level_boxes[w[0]] = AABB(center, Vector3());
			level.min_radius = radius;
			level.max_error = error;
			level.min_error = error;
		} else {
			level_boxes[w[0]].expand_to(center);
			level.min_radius = MIN(level.min_radius, radius);
			level.max_error = MAX(level.max_error, error);
			level.min_error = MIN(level.min_error, error);
		}
	}
	for (uint32_t l = 0; l < level_count; l++) {
		levels[l].center = level_boxes[l].get_center();
	}
	for (uint32_t g = 0; g < group_count && level_count > 0; g++) {
		const uint32_t *w = group_words + uint64_t(g) * GROUP_WORDS;
		RTLodLevelBound &level = levels[w[0]];
		const float spread = float(Vector3(_bits_float(w[3]), _bits_float(w[4]), _bits_float(w[5])).distance_to(level.center));
		level.center_spread = MAX(level.center_spread, spread);
		level.enclosing_radius = MAX(level.enclosing_radius, spread + _bits_float(w[6]));
	}
	// Each depth's discrete level (D90), counted by the selection shader's level rule: a cluster
	// is in the levels from just under the group it was simplified from (0 at LOD 0) down to its
	// own group's depth, or to the last depth when its group is terminal.
	for (uint32_t c = 0; c < cluster_count && level_count > 0; c++) {
		const uint32_t *w = cluster_words + uint64_t(c) * RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
		const uint32_t *group = group_words + uint64_t(w[1]) * GROUP_WORDS;
		const int32_t refined = int32_t(w[2]);
		const uint32_t first = refined < 0 ? 0 : group_words[uint64_t(refined) * GROUP_WORDS] + 1;
		const uint32_t last = _bits_float(group[7]) >= 3.0e38f ? level_count - 1 : MIN(group[0], level_count - 1);
		for (uint32_t l = first; l <= last; l++) {
			levels[l].cut_clusters++;
		}
	}

	RD::ClusterAccelerationStructureInput blas_input;
	blas_input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_CLUSTERS_BOTTOM_LEVEL;
	blas_input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
	blas_input.max_acceleration_structure_count = 1;
	blas_input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT); // As the per-frame builds are flagged.
	blas_input.max_total_cluster_count = cluster_count;
	blas_input.max_cluster_count_per_acceleration_structure = cluster_count;
	uint64_t blas_size = 0, blas_scratch = 0;
	rd->cluster_acceleration_structure_get_build_sizes(blas_input, blas_size, blas_scratch);
	if (blas_size == 0) {
		WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: the driver sized a BLAS over its selection at 0 bytes; %s.", p_cache_key, fallback));
		return;
	}

	// The traversal's tree (D91), after the group records; without one the selection tests every cluster.
	LocalVector<uint32_t> tree_words;
	LocalVector<uint32_t> tree_levels;
	uint32_t tree_nodes = 0;
	if (cluster_lod_traversal_enabled && !_build_cluster_lod_tree(cluster_words, cluster_count, group_words, group_count, tree_words, tree_levels, tree_nodes)) {
		tree_words.clear();
		WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: the hierarchy's groups do not own contiguous runs of its clusters; its selection tests every cluster.", p_cache_key));
	}

	// The region: the addresses (lo, hi), the cluster records, the group records as the shader
	// reads them, then the tree's nodes and refs, and for a streamed surface (D92)
	// RT_LOD_STREAM_STATE_WORDS words of state per group: zero, but for the roots' flag.
	RTLodStream *stream = r_surf_data->cluster_lod_stream;
	const uint32_t state_words = stream ? group_count * RT_LOD_STREAM_STATE_WORDS : 0;
	const uint64_t words64 = uint64_t(cluster_count) * 4 + uint64_t(group_count) * GROUP_WORDS + tree_words.size() + state_words;
	ERR_FAIL_COND_MSG(words64 > UINT32_MAX / 4, "RT: a cluster LOD hierarchy's region exceeds the table's reach.");
	const uint32_t words = uint32_t(words64);
	uint32_t offset = 0;
	if (!_cluster_lod_table_allocate(words, offset)) {
		WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: no room in the cluster LOD table; %s.", p_cache_key, fallback));
		return;
	}
	// The host copy starts after the addresses, which the GPU copies below; a streamed surface's
	// starts at the region's start, its addresses zero but for the persistent runs copied over them.
	const uint32_t host_first = stream ? 0 : cluster_count * 2;
	LocalVector<uint32_t> host;
	host.resize(words - host_first);
	if (stream) {
		memset(host.ptr(), 0, host.size() * sizeof(uint32_t));
		uint32_t *state = host.ptr() + (words - state_words - host_first);
		for (uint32_t g = 0; g < group_count; g++) {
			const RTLodStreamGroup &group = stream->groups[g];
			if (group.state != RTLodStreamGroup::PERSISTENT && group.parents_count == 0) {
				state[g * RT_LOD_STREAM_STATE_WORDS + 2] = RT_LOD_STREAM_STATE_FLAG_ROOT;
			}
		}
	}
	uint32_t *records = host.ptr() + (cluster_count * 2 - host_first);
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t *w = cluster_words + uint64_t(c) * RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
		records[c * 2] = w[1];
		records[c * 2 + 1] = w[2];
	}
	const uint32_t tree_offset = cluster_count * 4 + group_count * GROUP_WORDS;
	if (!tree_words.is_empty()) {
		memcpy(host.ptr() + tree_offset - host_first, tree_words.ptr(), tree_words.size() * sizeof(uint32_t));
	}
	for (uint32_t g = 0; g < group_count; g++) {
		const uint32_t *w = group_words + uint64_t(g) * GROUP_WORDS;
		uint32_t *out = host.ptr() + (cluster_count * 4 - host_first) + uint64_t(g) * GROUP_WORDS;
		// The table's (depth, first cluster, cluster count, center x y z, radius, error) as
		// (center x y z, radius, error, depth, first cluster, cluster count).
		out[0] = w[3];
		out[1] = w[4];
		out[2] = w[5];
		out[3] = w[6];
		out[4] = w[7];
		out[5] = w[0];
		out[6] = w[1];
		out[7] = w[2];
	}
	rd->buffer_update(cluster_lod_table, (offset + host_first) * sizeof(uint32_t), uint32_t(host.size() * sizeof(uint32_t)), host.ptr());
	bool copied = true;
	if (stream) {
		// The persistent runs' addresses, in the order the population built them.
		uint32_t built = 0;
		for (uint32_t r = 0; r < stream->persistent_runs.size() && copied; r += 2) {
			const uint32_t first = stream->persistent_runs[r];
			const uint32_t count = stream->persistent_runs[r + 1];
			copied = rd->buffer_copy(p_args_buffer, cluster_lod_table, uint32_t(p_implicit_addresses_offset + sizeof(uint64_t) * built), (offset + first * 2) * sizeof(uint32_t), uint32_t(sizeof(uint64_t) * count)) == OK;
			built += count;
		}
	} else {
		copied = rd->buffer_copy(p_args_buffer, cluster_lod_table, uint32_t(p_implicit_addresses_offset), offset * sizeof(uint32_t), uint32_t(sizeof(uint64_t) * cluster_count)) == OK;
	}
	if (!copied) {
		cluster_lod_table_live_words -= words;
		_cluster_lod_table_free(offset, words);
		WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: the CLAS addresses could not be copied to the cluster LOD table; %s.", p_cache_key, fallback));
		return;
	}
	r_surf_data->cluster_lod_table_offset = offset;
	r_surf_data->cluster_lod_table_words = words;
	r_surf_data->cluster_lod_blas_worst_size = blas_size;
	r_surf_data->cluster_lod_generation = ++cluster_lod_generation_counter;
	r_surf_data->cluster_lod_levels = levels;
	if (!tree_words.is_empty()) {
		r_surf_data->cluster_lod_tree.nodes_offset = offset + tree_offset;
		r_surf_data->cluster_lod_tree.refs_offset = offset + tree_offset + tree_nodes * 8;
		r_surf_data->cluster_lod_tree.group_count = group_count;
		r_surf_data->cluster_lod_tree.level_nodes = tree_levels;
	}
	if (stream) {
		stream->state_offset = offset + words - state_words;
		cluster_lod_stream_surfaces.insert(offset, r_surf_data);
	}
}

void RenderRaytracing::_release_surface_cluster_lod(RTSurfaceData *p_surf_data) {
	_release_lod_cached_levels(p_surf_data);
	_release_lod_stream(p_surf_data);
	if (p_surf_data->cluster_lod_table_offset == UINT32_MAX) {
		return;
	}
	// A frame in flight may still select from the region; the next content written there is a
	// transfer this frame records, behind that selection in queue order.
	cluster_lod_table_live_words -= p_surf_data->cluster_lod_table_words;
	_cluster_lod_table_free(p_surf_data->cluster_lod_table_offset, p_surf_data->cluster_lod_table_words);
	p_surf_data->cluster_lod_table_offset = UINT32_MAX;
	p_surf_data->cluster_lod_table_words = 0;
	p_surf_data->cluster_lod_blas_worst_size = 0;
	p_surf_data->cluster_lod_levels.clear();
	p_surf_data->cluster_lod_tree = RTLodTree();
}

// ---------------------------------------------------------------------------
// CLAS residency (D92)
// ---------------------------------------------------------------------------

// A surface is pinned (D92) for the minimum age after an emissive instance or an expanded
// MultiMesh last referenced it.
static _FORCE_INLINE_ bool _lod_stream_pinned(const RTLodStream *p_stream, uint64_t p_frames_drawn, uint64_t p_min_age) {
	return p_stream->pinned_frame != 0 && p_frames_drawn - p_stream->pinned_frame <= p_min_age;
}

// A streamed surface's residency, from its table, or null when the surface cannot stream and keeps
// every CLAS built: its groups must own contiguous runs of clusters (the cluster pass and a load's
// build infos assume it), its links must stay inside the table, every refinement must go one depth
// finer (its finest level is a cached level, D90, which needs that) and its bounds and errors must
// be numbers (the tree's test). The persistent groups are the terminal ones, the coarsest of their
// part of the surface; their clusters, in runs in cluster order, are the population's build. The
// test is mesh_cluster_lod_streamable(), which the mesh storage takes too when it decides whether a
// surface streams its geometry (D93d).
RTLodStream *RenderRaytracing::_create_lod_stream(void *p_mesh_surface, const RTClusterLayout &p_layout, RID p_vertex_buffer, RID p_index_buffer, uint32_t p_cache_key) {
	const uint32_t HEADER_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS;
	const uint32_t CLUSTER_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
	const uint32_t GROUP_WORDS = RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS;
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();
	const Vector<uint8_t> &table_bytes = mesh_storage->mesh_surface_get_cluster_data(p_mesh_surface);
	const RD::ClusterAccelerationStructureCapabilities &caps = RD::get_singleton()->get_cluster_acceleration_structure_capabilities();
	const uint32_t cluster_count = p_layout.triangles.size();
	if (!mesh_cluster_lod_streamable(table_bytes, mesh_storage->mesh_surface_get_rt_index_count(p_mesh_surface), mesh_storage->mesh_surface_get_rt_index_total(p_mesh_surface), caps.max_triangles_per_cluster, caps.max_vertices_per_cluster)) {
		WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: the hierarchy cannot stream (its groups' runs, links or depths are not as the residency needs); every level stays built.", p_cache_key));
		return nullptr;
	}
	const uint32_t *table = reinterpret_cast<const uint32_t *>(table_bytes.ptr());
	const uint32_t group_count = table[2];
	const uint32_t level_count = table[3];
	ERR_FAIL_COND_V(table[1] != cluster_count, nullptr); // The layout took every level of this very table.
	const uint32_t *cluster_words = table + HEADER_WORDS;
	const uint32_t *group_words = cluster_words + uint64_t(CLUSTER_WORDS) * cluster_count;

	RTLodStream *stream = memnew(RTLodStream);
	stream->vertex_buffer = p_vertex_buffer;
	stream->index_buffer = p_index_buffer;
	stream->mesh_surface = p_mesh_surface;
	stream->max_triangles = p_layout.max_triangles;
	stream->max_vertices = p_layout.max_vertices;
	stream->groups.resize(group_count);
	stream->depth_streamed.resize(level_count);
	stream->depth_resident.resize(level_count);
	stream->level_last_used.resize(level_count);
	for (uint32_t l = 0; l < level_count; l++) {
		stream->depth_streamed[l] = 0;
		stream->depth_resident[l] = 0;
		stream->level_last_used[l] = 0;
	}
	for (uint32_t g = 0; g < group_count; g++) {
		const uint32_t *w = group_words + uint64_t(g) * GROUP_WORDS;
		RTLodStreamGroup &group = stream->groups[g];
		group.depth = w[0];
		group.first_cluster = w[1];
		group.cluster_count = w[2];
		if (_bits_float(w[7]) >= 3.0e38f) {
			group.state = RTLodStreamGroup::PERSISTENT;
		} else {
			stream->depth_streamed[group.depth]++;
			stream->absent_groups++;
		}
	}
	// Each group's children (the groups its clusters were simplified from, each once) and from them
	// its parents, as two runs of the links.
	{
		for (uint32_t g = 0; g < group_count; g++) {
			RTLodStreamGroup &group = stream->groups[g];
			group.children_first = stream->links.size();
			for (uint32_t c = group.first_cluster; c < group.first_cluster + group.cluster_count; c++) {
				const int32_t refined = int32_t(cluster_words[uint64_t(c) * CLUSTER_WORDS + 2]);
				if (refined < 0) {
					continue;
				}
				bool seen = false;
				for (uint32_t k = group.children_first; k < stream->links.size() && !seen; k++) {
					seen = stream->links[k] == uint32_t(refined);
				}
				if (!seen) {
					stream->links.push_back(uint32_t(refined));
					stream->groups[refined].parents_count++;
				}
			}
			group.children_count = stream->links.size() - group.children_first;
		}
		const uint32_t parents_base = stream->links.size();
		uint32_t cursor = parents_base;
		for (uint32_t g = 0; g < group_count; g++) {
			stream->groups[g].parents_first = cursor;
			cursor += stream->groups[g].parents_count;
			stream->groups[g].parents_count = 0;
		}
		stream->links.resize(cursor);
		for (uint32_t g = 0; g < group_count; g++) {
			const RTLodStreamGroup &group = stream->groups[g];
			for (uint32_t k = group.children_first; k < group.children_first + group.children_count; k++) {
				RTLodStreamGroup &child = stream->groups[stream->links[k]];
				stream->links[child.parents_first + child.parents_count++] = g;
			}
		}
		// A group the sloppy fallback simplified to nothing keeps a finite error but no cluster refines
		// from it: no parent, so no resident cluster ever asks for it, and the selection's own test of
		// it does (debris, leaf cards). The closure takes it as a root.
		for (const RTLodStreamGroup &group : stream->groups) {
			stream->root_groups += group.state != RTLodStreamGroup::PERSISTENT && group.parents_count == 0 ? 1 : 0;
		}
		// The persistent clusters as runs in cluster order, and each cluster's first triangle and counts.
		stream->cluster_first_triangle.resize(cluster_count);
		stream->cluster_packed.resize(cluster_count);
		uint32_t first_triangle = 0;
		for (uint32_t c = 0; c < cluster_count; c++) {
			stream->cluster_first_triangle[c] = first_triangle;
			stream->cluster_packed[c] = p_layout.triangles[c] | (p_layout.vertices[c] << 16);
			first_triangle += p_layout.triangles[c];
			const bool persistent = stream->groups[cluster_words[uint64_t(c) * CLUSTER_WORDS + 1]].state == RTLodStreamGroup::PERSISTENT;
			if (!persistent) {
				continue;
			}
			const uint32_t runs = stream->persistent_runs.size();
			if (runs > 0 && stream->persistent_runs[runs - 2] + stream->persistent_runs[runs - 1] == c) {
				stream->persistent_runs[runs - 1]++;
			} else {
				stream->persistent_runs.push_back(c);
				stream->persistent_runs.push_back(1);
			}
			stream->persistent_clusters++;
			stream->persistent_triangles += p_layout.triangles[c];
			stream->persistent_vertices += p_layout.vertices[c];
		}
	}
	print_verbose(vformat("RT: cluster BLAS [%d] streams (D92): %d groups, %d persistent in %d clusters, %d refined into nothing (requested by their own test).", p_cache_key, group_count, group_count - stream->absent_groups, stream->persistent_clusters, stream->root_groups));
	return stream;
}

// The surface's loaded groups go back to the arena past the frames that may trace them, and a load
// in flight forgets it.
void RenderRaytracing::_release_lod_stream(RTSurfaceData *p_surf_data) {
	RTLodStream *stream = p_surf_data->cluster_lod_stream;
	if (stream == nullptr) {
		return;
	}
	for (const RTLodStreamGroup &group : stream->groups) {
		if (group.state == RTLodStreamGroup::RESIDENT && group.arena_chunk >= 0) {
			_cluster_arena_retire(uint32_t(group.arena_chunk), group.arena_offset, group.bytes);
		}
	}
	for (RTLodStreamLoad &load : cluster_lod_stream_loads) {
		for (RTLodStreamLoad::Item &item : load.items) {
			if (item.surface == p_surf_data) {
				item.surface = nullptr;
			}
		}
	}
	for (RTLodStreamGatherJob *job : cluster_lod_stream_gathers) {
		for (RTLodStreamLoad::Item &item : job->items) {
			if (item.surface == p_surf_data) {
				item.surface = nullptr; // Its group has no blob yet, and the gather reads its own copies.
			}
		}
	}
	RTSurfaceData **registered = cluster_lod_stream_surfaces.getptr(p_surf_data->cluster_lod_table_offset);
	if (registered && *registered == p_surf_data) {
		cluster_lod_stream_surfaces.erase(p_surf_data->cluster_lod_table_offset);
	}
	memdelete(stream);
	p_surf_data->cluster_lod_stream = nullptr;
}

void RenderRaytracing::_on_lod_stream_read(const PackedByteArray &p_bytes, uint64_t p_frame, uint32_t p_part) {
	// Called by RenderingDevice on the render thread as it begins a later frame: only queued.
	if (singleton == nullptr) {
		return;
	}
	RTLodStreamRead read;
	read.frame = p_frame;
	read.part = p_part;
	read.bytes = p_bytes;
	singleton->cluster_lod_stream_reads.push_back(read);
}

void RenderRaytracing::_on_lod_stream_sizes_read(const PackedByteArray &p_sizes, uint64_t p_ticket) {
	if (singleton == nullptr) {
		return;
	}
	RTClusterCompactionResult result;
	result.ticket = p_ticket;
	result.sizes = p_sizes;
	singleton->cluster_lod_stream_load_results.push_back(result);
}

// Level d of a streamed surface holds the clusters of its depth-d groups (and of persistent ones),
// whose parents the closure keeps resident with them: it may be built once every group of depth d
// and coarser is resident.
bool RenderRaytracing::_lod_stream_level_resident(const RTSurfaceData *p_surf_data, uint32_t p_depth) const {
	const RTLodStream *stream = p_surf_data->cluster_lod_stream;
	for (uint32_t d = p_depth; d < stream->depth_streamed.size(); d++) {
		if (stream->depth_resident[d] != stream->depth_streamed[d]) {
			return false;
		}
	}
	return true;
}

// What an instance that needs a streamed surface's finest level traces (an emissive mesh light,
// whose sampling covers the finest triangles, D50, or an expanded MultiMesh, which has no slot of
// its own): its cached level 0 once built, else the coarsest cut, which the pin then loads up from.
RID RenderRaytracing::_lod_stream_pinned_blas(RTSurfaceData *p_surf_data) {
	RTLodStream *stream = p_surf_data->cluster_lod_stream;
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	stream->pinned_frame = frames_drawn;
	if (!p_surf_data->cluster_lod_cache.is_empty() && p_surf_data->cluster_lod_cache[0].blas.is_valid()) {
		p_surf_data->cluster_lod_cache[0].last_used_frame = RSG::rasterizer->get_frame_number();
		stream->level_last_used[0] = frames_drawn;
		return p_surf_data->cluster_lod_cache[0].blas;
	}
	return p_surf_data->blas;
}

// The pinned surfaces' cached level 0, built in this frame's batch once every group is resident.
void RenderRaytracing::_prepare_lod_stream_pins() {
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	for (const KeyValue<uint32_t, RTSurfaceData *> &E : cluster_lod_stream_surfaces) {
		RTSurfaceData *surface = E.value;
		if (surface->cluster_lod_stream->pinned_frame != frames_drawn || surface->compaction_ticket != 0 || surface->cluster_lod_levels.is_empty()) {
			continue;
		}
		if (!surface->cluster_lod_cache.is_empty() && surface->cluster_lod_cache[0].blas.is_valid()) {
			continue;
		}
		if (_prepare_lod_cached_level(surface, 0)) {
			surface->cluster_lod_stream->level_last_used[0] = frames_drawn;
		}
	}
}

// One group's unload: its addresses and its geometry records (D93c) zeroed in the table (ordered
// before the next selection by the table's tracker), its regions retired past the frames that may
// still trace them, and the surface's cached levels that hold it or rest on it released.
void RenderRaytracing::_evict_lod_stream_group(RTSurfaceData *p_surf_data, uint32_t p_group) {
	RTLodStream *stream = p_surf_data->cluster_lod_stream;
	RTLodStreamGroup &group = stream->groups[p_group];
	RD::get_singleton()->buffer_clear(cluster_lod_table, (p_surf_data->cluster_lod_table_offset + group.first_cluster * 2) * sizeof(uint32_t), sizeof(uint64_t) * group.cluster_count);
	_cluster_arena_retire(uint32_t(group.arena_chunk), group.arena_offset, group.bytes);
	if (group.blob_bytes > 0) {
		RD::get_singleton()->buffer_clear(cluster_lod_table, (p_surf_data->cluster_geometry_offset + group.first_cluster * RT_CLUSTER_GEOMETRY_RECORD_WORDS) * sizeof(uint32_t), sizeof(uint32_t) * RT_CLUSTER_GEOMETRY_RECORD_WORDS * group.cluster_count);
		_retire_cluster_geometry_blob(p_surf_data, p_group);
	}
	for (uint32_t d = 0; d <= group.depth && d < p_surf_data->cluster_lod_cache.size(); d++) {
		_release_lod_cached_level(p_surf_data->cluster_lod_cache[d]);
	}
	for (uint32_t k = group.parents_first; k < group.parents_first + group.parents_count; k++) {
		stream->groups[stream->links[k]].live_children--;
	}
	group.state = RTLodStreamGroup::ABSENT;
	group.arena_chunk = -1;
	group.arena_offset = 0;
	group.requested = 0;
	stream->depth_resident[group.depth]--;
	stream->resident_bytes -= group.bytes + group.blob_bytes;
	stream->resident_blob_bytes -= group.blob_bytes;
	stream->absent_groups++;
	lod_stream_stats.evictions++;
}

// Whether a group may be unloaded now: resident, no child resident or loading, unused for the
// minimum age, under no cached level placed within it, its surface not pinned.
bool RenderRaytracing::_lod_stream_evictable(const RTLodStream *p_stream, uint32_t p_group, uint64_t p_frames_drawn) const {
	const RTLodStreamGroup &group = p_stream->groups[p_group];
	const uint64_t min_age = cluster_lod_stream_min_age;
	if (group.state != RTLodStreamGroup::RESIDENT || group.live_children > 0 || p_frames_drawn - group.last_used < min_age || _lod_stream_pinned(p_stream, p_frames_drawn, min_age)) {
		return false;
	}
	for (uint32_t d = 0; d <= group.depth && d < p_stream->level_last_used.size(); d++) {
		if (p_stream->level_last_used[d] != 0 && p_frames_drawn - p_stream->level_last_used[d] < min_age) {
			return false;
		}
	}
	return true;
}

// Unloads least recently used groups until p_needed bytes are free. The candidates are gathered and
// sorted once a step, on its first need, and consumed across its wants: a parent whose last child
// goes joins them in order, and a candidate that took a child since is passed over. A group being
// admitted holds its parents (their live children counted first), so its room never comes from them.
bool RenderRaytracing::_evict_lod_stream_groups(uint64_t p_needed, uint64_t &r_used) {
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	LocalVector<LodStreamCandidate> &candidates = lod_stream_candidates;
	if (!lod_stream_candidates_built) {
		lod_stream_candidates_built = true;
		candidates.clear();
		lod_stream_candidate_next = 0;
		for (const KeyValue<uint32_t, RTSurfaceData *> &E : cluster_lod_stream_surfaces) {
			const RTLodStream *stream = E.value->cluster_lod_stream;
			for (uint32_t g = 0; g < stream->groups.size(); g++) {
				if (_lod_stream_evictable(stream, g, frames_drawn)) {
					LodStreamCandidate candidate;
					candidate.surface = E.value;
					candidate.group = g;
					candidate.last_used = stream->groups[g].last_used;
					candidates.push_back(candidate);
				}
			}
		}
		candidates.sort();
	}
	uint64_t freed = 0;
	while (freed < p_needed && lod_stream_candidate_next < candidates.size()) {
		const LodStreamCandidate candidate = candidates[lod_stream_candidate_next++];
		RTLodStream *stream = candidate.surface->cluster_lod_stream;
		if (!_lod_stream_evictable(stream, candidate.group, frames_drawn)) {
			continue;
		}
		const RTLodStreamGroup &group = stream->groups[candidate.group];
		const uint64_t bytes = group.bytes + group.blob_bytes;
		const uint32_t parents_first = group.parents_first;
		const uint32_t parents_count = group.parents_count;
		_evict_lod_stream_group(candidate.surface, candidate.group);
		freed += bytes;
		r_used -= MIN(r_used, bytes);
		for (uint32_t k = parents_first; k < parents_first + parents_count; k++) {
			_join_lod_stream_candidate(candidate.surface, stream->links[k], frames_drawn);
		}
	}
	return freed >= p_needed;
}

// A group that became evictable during the step (a parent whose last child went, or one a deferred
// want held) joins the candidates not yet consumed, in least-recently-used order. A duplicate is
// harmless: the second is passed over once the first is evicted.
void RenderRaytracing::_join_lod_stream_candidate(RTSurfaceData *p_surf_data, uint32_t p_group, uint64_t p_frames_drawn) {
	const RTLodStream *stream = p_surf_data->cluster_lod_stream;
	if (!lod_stream_candidates_built || !_lod_stream_evictable(stream, p_group, p_frames_drawn)) {
		return; // Not gathered yet, the gathering finds it; or not evictable.
	}
	LocalVector<LodStreamCandidate> &candidates = lod_stream_candidates;
	LodStreamCandidate joined;
	joined.surface = p_surf_data;
	joined.group = p_group;
	joined.last_used = stream->groups[p_group].last_used;
	uint32_t at = lod_stream_candidate_next;
	uint32_t to = candidates.size();
	while (at < to) {
		const uint32_t mid = (at + to) / 2;
		if (candidates[mid].last_used <= joined.last_used) {
			at = mid + 1;
		} else {
			to = mid;
		}
	}
	candidates.insert(at, joined);
}

// A group admitted or loading that will not become resident: back to absent.
static void _lod_stream_group_unload_pending(RTLodStream *p_stream, uint32_t p_group) {
	RTLodStreamGroup &group = p_stream->groups[p_group];
	for (uint32_t k = group.parents_first; k < group.parents_first + group.parents_count; k++) {
		p_stream->groups[p_stream->links[k]].live_children--;
	}
	p_stream->loading_bytes -= group.loading_bytes;
	group.loading_bytes = 0;
	group.state = RTLodStreamGroup::ABSENT;
	group.requested = 0; // A retry's latency starts at its own request.
	p_stream->absent_groups++;
}

// A load's first step: the admitted groups' CLAS built with implicit destinations into a staging
// buffer of their own, in one build per position format, and their sizes read back. False with
// the groups back to absent.
bool RenderRaytracing::_issue_lod_stream_load(const LocalVector<RTLodStreamLoad::Item> &p_items, bool p_compressed, const RTClusterGeometryGather *p_gather) {
	if (p_items.is_empty()) {
		return true;
	}
	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	RTLodStreamLoad load;
	load.items = p_items;
	uint32_t total_triangles = 0, total_vertices = 0, max_triangles = 0, max_vertices = 0;
	Vector<RID> source_buffers;
	for (RTLodStreamLoad::Item &item : load.items) {
		const RTLodStream *stream = item.surface->cluster_lod_stream;
		const RTLodStreamGroup &group = stream->groups[item.group];
		item.first = load.cluster_count;
		load.cluster_count += group.cluster_count;
		for (uint32_t c = group.first_cluster; c < group.first_cluster + group.cluster_count; c++) {
			total_triangles += stream->cluster_packed[c] & 0xFFFFu;
			total_vertices += stream->cluster_packed[c] >> 16;
		}
		max_triangles = MAX(max_triangles, stream->max_triangles);
		max_vertices = MAX(max_vertices, stream->max_vertices);
		if ((item.surface->geometry.flags & RT_GEOM_FLAG_CLUSTER_GEOMETRY) != 0) {
			continue; // Its build reads its blob and its staged positions (D93c).
		}
		if (!source_buffers.has(stream->vertex_buffer)) {
			source_buffers.push_back(stream->vertex_buffer);
		}
		if (!source_buffers.has(stream->index_buffer)) {
			source_buffers.push_back(stream->index_buffer);
		}
	}
	// The groups' gathered blobs placed in their regions and their positions staged (D93c).
	RTClusterGeometryStage geometry_stage;
	bool ok = p_gather == nullptr || _place_cluster_geometry(load.items, *p_gather, geometry_stage);
	for (const RID &buffer : geometry_stage.blob_buffers) {
		source_buffers.push_back(buffer);
	}
	if (geometry_stage.positions_buffer.is_valid()) {
		source_buffers.push_back(geometry_stage.positions_buffer);
	}

	RD::ClusterAccelerationStructureInput input;
	input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_TRIANGLE_CLUSTER;
	input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_IMPLICIT_DESTINATIONS;
	input.max_acceleration_structure_count = load.cluster_count;
	input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT); // As the population builds them.
	input.vertex_format = p_compressed ? RD::DATA_FORMAT_R16G16B16A16_UNORM : RD::DATA_FORMAT_R32G32B32_SFLOAT;
	input.max_cluster_triangle_count = max_triangles;
	input.max_cluster_vertex_count = max_vertices;
	input.max_total_triangle_count = total_triangles;
	input.max_total_vertex_count = total_vertices;
	uint64_t region_size = 0, scratch = 0;
	rd->cluster_acceleration_structure_get_build_sizes(input, region_size, scratch);

	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	const uint32_t info_stride = caps.triangle_build_info_size;
	const uint64_t infos_offset = 0;
	const uint64_t implicit_offset = _align_up(infos_offset + uint64_t(info_stride) * load.cluster_count, 16);
	const uint64_t sizes_offset = _align_up(implicit_offset + sizeof(uint64_t) * load.cluster_count, 16);
	const uint64_t compact_offset = _align_up(sizes_offset + sizeof(uint32_t) * load.cluster_count, 16);
	const uint64_t args_size = _align_up(compact_offset + sizeof(uint64_t) * load.cluster_count, 16);
	const uint64_t staging_size = base_alignment + region_size;
	ok = ok && region_size > 0 && staging_size <= UINT32_MAX && args_size <= UINT32_MAX;
	if (ok) {
		load.staging = rd->storage_buffer_create(uint32_t(staging_size), {}, 0,
				RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_STORAGE_BIT);
		load.args = rd->storage_buffer_create(uint32_t(args_size), {}, 0,
				RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
		ok = load.staging.is_valid() && load.args.is_valid();
	}
	if (ok) {
		LocalVector<uint8_t> args;
		args.resize(args_size);
		memset(args.ptr(), 0, args_size);
		uint32_t k = 0;
		for (const RTLodStreamLoad::Item &item : load.items) {
			const RTSurfaceData *surface = item.surface;
			const RTLodStream *stream = surface->cluster_lod_stream;
			const RTLodStreamGroup &group = stream->groups[item.group];
			const RT_GeometryData &geom = surface->geometry;
			const uint32_t index_size = geom.index_format == RT_INDEX_FORMAT_UINT16 ? sizeof(uint16_t) : sizeof(uint32_t);
			const bool cluster_geometry = (geom.flags & RT_GEOM_FLAG_CLUSTER_GEOMETRY) != 0; // As the population built the surface's CLAS (D93b).
			for (uint32_t c = group.first_cluster; c < group.first_cluster + group.cluster_count; c++) {
				RD::ClusterTriangleBuildInfo info;
				info.cluster_id = cluster_geometry ? c : stream->cluster_first_triangle[c];
				info.triangle_count = stream->cluster_packed[c] & 0xFFFFu;
				info.vertex_count = stream->cluster_packed[c] >> 16;
				if (cluster_geometry) {
					info.index_type = RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT8;
					info.index_address = geometry_stage.index_addresses[k];
					info.index_stride = sizeof(uint8_t);
					info.vertex_address = geometry_stage.positions_address + geometry_stage.position_offsets[k];
				} else {
					info.index_type = geom.index_format == RT_INDEX_FORMAT_UINT16 ? RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT16 : RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT32;
					info.index_address = geom.index_buffer_address + uint64_t(stream->cluster_first_triangle[c]) * 3u * index_size;
					info.index_stride = index_size;
					info.vertex_address = geom.vertex_buffer_address;
				}
				info.vertex_stride = geom.position_stride;
				rd->cluster_triangle_build_info_write(args.ptr() + infos_offset + uint64_t(k) * info_stride, info);
				k++;
			}
		}
		rd->buffer_update(load.args, 0, uint32_t(args_size), args.ptr());
		const uint64_t buffer_address = rd->buffer_get_device_address(load.staging);
		RD::ClusterAccelerationStructureBuild build;
		build.input = input;
		build.dst_implicit_buffer = load.staging;
		build.dst_implicit_offset = _align_up(buffer_address, base_alignment) - buffer_address;
		build.dst_addresses_buffer = load.args;
		build.dst_addresses_offset = implicit_offset;
		build.dst_addresses_stride = sizeof(uint64_t);
		build.dst_sizes_buffer = load.args;
		build.dst_sizes_offset = sizes_offset;
		build.dst_sizes_stride = sizeof(uint32_t);
		build.src_infos_buffer = load.args;
		build.src_infos_offset = infos_offset;
		build.src_infos_stride = info_stride;
		build.source_buffers = source_buffers;
		ok = rd->cluster_acceleration_structure_build(build) == OK;
	}
	if (ok) {
		load.ticket = ++cluster_lod_stream_next_ticket;
		ok = rd->buffer_get_data_async(load.args, callable_mp_static(&RenderRaytracing::_on_lod_stream_sizes_read).bind(load.ticket), uint32_t(sizes_offset), uint32_t(sizeof(uint32_t) * load.cluster_count)) == OK;
	}
	if (!ok) {
		if (load.staging.is_valid()) {
			rd->free_rid(load.staging);
		}
		if (load.args.is_valid()) {
			rd->free_rid(load.args);
		}
		for (const RTLodStreamLoad::Item &item : load.items) {
			_retire_cluster_geometry_blob(item.surface, item.group);
			_lod_stream_group_unload_pending(item.surface->cluster_lod_stream, item.group);
		}
		lod_stream_stats.failed += load.items.size();
		WARN_PRINT_ONCE("RT: a load of streamed cluster acceleration structures could not be recorded; its groups stay at a coarser level.");
		return false;
	}
	rd->set_resource_name(load.staging, "RT cluster LOD stream staging");
	rd->set_resource_name(load.args, "RT cluster LOD stream args");
	load.region_size = region_size;
	load.implicit_offset = implicit_offset;
	load.compact_offset = compact_offset;
	cluster_lod_stream_loads.push_back(load);
	return true;
}

// A load's start (D93c). The groups of a surface that streams its geometry gather it on the worker
// threads first, and their build is issued once the gather is complete (_issue_lod_stream_gathers);
// the others are issued now. A surface whose RAM copy went since its population (an update drops it
// and repopulates the surface) cannot gather: its groups stay absent until the repopulation.
void RenderRaytracing::_start_lod_stream_load(LocalVector<RTLodStreamLoad::Item> &r_items, bool p_compressed) {
	if (!cluster_lod_stream_geometry_enabled) {
		_issue_lod_stream_load(r_items, p_compressed, nullptr);
		return;
	}
	LocalVector<RTLodStreamLoad::Item> now;
	RTLodStreamGatherJob *job = nullptr;
	for (const RTLodStreamLoad::Item &item : r_items) {
		RTLodStream *stream = item.surface->cluster_lod_stream;
		if ((item.surface->geometry.flags & RT_GEOM_FLAG_CLUSTER_GEOMETRY) == 0) {
			now.push_back(item);
			continue;
		}
		if (!_cluster_geometry_ram_valid(stream->mesh_surface, item.surface->geometry, _lod_stream_index_count(stream))) {
			_lod_stream_group_unload_pending(stream, item.group);
			lod_stream_stats.failed++;
			continue;
		}
		if (job == nullptr) {
			job = memnew(RTLodStreamGatherJob);
			job->compressed = p_compressed;
			job->frame = RD::get_singleton()->get_frames_drawn();
		}
		job->items.push_back(item);
	}
	_issue_lod_stream_load(now, p_compressed, nullptr);
	if (job == nullptr) {
		return;
	}
	if (!_prepare_cluster_geometry_gather(job->items, job->gather)) {
		for (const RTLodStreamLoad::Item &item : job->items) {
			_lod_stream_group_unload_pending(item.surface->cluster_lod_stream, item.group);
		}
		lod_stream_stats.failed += job->items.size();
		memdelete(job);
		return;
	}
	// Low priority, as every job that outlives its tick (docs/workflow.md): a gather may wait behind
	// a terrain build, never a tick's fan-out behind a gather.
	WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
	const int elements = int(job->gather.items.size());
	job->group = pool->add_template_group_task(&job->gather, &RTClusterGeometryGather::gather_item, (void *)nullptr, elements, MIN(elements, MAX(1, pool->get_thread_count())), false, SNAME("RT cluster geometry gather"));
	cluster_lod_stream_gathers.push_back(job);
}

// The gathers run on the worker threads and are issued a fixed RT_LOD_STREAM_GATHER_LATENCY frames
// after their admission, waited for then if still running, the way the terrain's job pipeline does
// (D88): the frame a load lands never depends on the threads' scheduling, so the residency history
// and every image after it are the same from run to run, and each frame uploads exactly what one
// frame admitted within the cap. In admission order, so a group's build is never issued before a
// parent's admitted earlier. An item whose surface went has been forgotten; one whose gather failed
// goes back to absent.
void RenderRaytracing::_issue_lod_stream_gathers() {
	WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	uint32_t issued = 0;
	uint64_t uploaded = 0;
	for (; issued < cluster_lod_stream_gathers.size(); issued++) {
		RTLodStreamGatherJob *job = cluster_lod_stream_gathers[issued];
		if (frames_drawn - job->frame < RT_LOD_STREAM_GATHER_LATENCY) {
			break;
		}
		if (!pool->is_group_task_completed(job->group)) {
			lod_stream_stats.gather_waits++;
		}
		pool->wait_for_group_task_completion(job->group);
		LocalVector<RTLodStreamLoad::Item> items;
		for (const RTLodStreamLoad::Item &item : job->items) {
			if (item.surface == nullptr) {
				continue;
			}
			if (!job->gather.item_ok[item.gather]) {
				_lod_stream_group_unload_pending(item.surface->cluster_lod_stream, item.group);
				lod_stream_stats.failed++;
				continue;
			}
			items.push_back(item);
		}
		uploaded += job->gather.blobs.size() + job->gather.positions.size();
		_issue_lod_stream_load(items, job->compressed, &job->gather);
		memdelete(job);
	}
	if (issued > 0) {
		for (uint32_t i = issued; i < cluster_lod_stream_gathers.size(); i++) {
			cluster_lod_stream_gathers[i - issued] = cluster_lod_stream_gathers[i];
		}
		cluster_lod_stream_gathers.resize(cluster_lod_stream_gathers.size() - issued);
		lod_stream_stats.uploaded_bytes += uploaded;
		if (lod_stream_stats.upload_frames.size() < 65536) {
			lod_stream_stats.upload_frames.push_back(uint32_t(MIN(uploaded / 1024, uint64_t(UINT32_MAX))));
		}
	}
}

// A load's second step, its sizes back: each group's exact region in the arena, every CLAS of the
// load moved there in one move, and each group's addresses written to the table, after which the
// group is resident. A group whose surface went, or whose parent did not become resident, still has
// its CLAS moved (the move covers the load) into a region retired at once. With any region missing
// the whole load is dropped and its groups are absent again.
void RenderRaytracing::_complete_lod_stream_loads() {
	if (cluster_lod_stream_load_results.is_empty()) {
		return;
	}
	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	const uint64_t frames_drawn = rd->get_frames_drawn();
	// Taken off the member first: a buffer update below can stall on the frame queue, which
	// delivers further readbacks into the member while these are processed.
	LocalVector<RTClusterCompactionResult> results;
	results = cluster_lod_stream_load_results;
	cluster_lod_stream_load_results.clear();
	for (const RTClusterCompactionResult &result : results) {
		uint32_t index = 0;
		while (index < cluster_lod_stream_loads.size() && cluster_lod_stream_loads[index].ticket != result.ticket) {
			index++;
		}
		if (index == cluster_lod_stream_loads.size()) {
			continue;
		}
		RTLodStreamLoad load = cluster_lod_stream_loads[index];
		cluster_lod_stream_loads.remove_at(index);

		struct Region {
			int32_t chunk = -1;
			uint64_t offset = 0;
			uint64_t size = 0;
		};
		LocalVector<Region> regions;
		regions.resize(load.items.size());
		LocalVector<uint64_t> compact;
		compact.resize(load.cluster_count);
		Vector<RID> chunks;
		bool ok = result.sizes.size() >= int64_t(sizeof(uint32_t) * load.cluster_count);
		const uint32_t *sizes = ok ? reinterpret_cast<const uint32_t *>(result.sizes.ptr()) : nullptr;
		for (uint32_t i = 0; i < load.items.size() && ok; i++) {
			const RTLodStreamLoad::Item &item = load.items[i];
			const uint32_t count = (i + 1 < load.items.size() ? load.items[i + 1].first : load.cluster_count) - item.first;
			uint64_t size = 0;
			for (uint32_t c = 0; c < count && ok; c++) {
				ok = sizes[item.first + c] > 0;
				size += _align_up(sizes[item.first + c], caps.cluster_byte_alignment);
			}
			size = _align_up(size, base_alignment);
			uint32_t chunk = 0;
			uint64_t offset = 0;
			ok = ok && _cluster_arena_allocate(size, chunk, offset);
			if (!ok) {
				break;
			}
			regions[i].chunk = int32_t(chunk);
			regions[i].offset = offset;
			regions[i].size = size;
			uint64_t cursor = cluster_arena[chunk].address + offset;
			for (uint32_t c = 0; c < count; c++) {
				compact[item.first + c] = cursor;
				cursor += _align_up(sizes[item.first + c], caps.cluster_byte_alignment);
			}
			if (!chunks.has(cluster_arena[chunk].buffer)) {
				chunks.push_back(cluster_arena[chunk].buffer);
			}
		}
		if (ok) {
			rd->buffer_update(load.args, uint32_t(load.compact_offset), uint32_t(sizeof(uint64_t) * load.cluster_count), compact.ptr());
			RD::ClusterAccelerationStructureBuild move;
			move.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MOVE_OBJECTS;
			move.input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
			move.input.max_acceleration_structure_count = load.cluster_count;
			move.input.move_type = RD::ClusterAccelerationStructureInput::CLUSTER_MOVE_TRIANGLE_CLUSTER;
			move.input.move_no_overlap = true;
			move.input.move_max_bytes = load.region_size;
			move.dst_storage_buffers = chunks;
			move.dst_addresses_buffer = load.args;
			move.dst_addresses_offset = load.compact_offset;
			move.dst_addresses_stride = sizeof(uint64_t);
			move.src_infos_buffer = load.args;
			move.src_infos_offset = load.implicit_offset;
			move.src_infos_stride = sizeof(uint64_t);
			move.source_acceleration_structure_buffers.push_back(load.staging);
			ok = rd->cluster_acceleration_structure_build(move) == OK;
		}
		for (uint32_t i = 0; i < load.items.size(); i++) {
			const RTLodStreamLoad::Item &item = load.items[i];
			RTLodStream *stream = item.surface ? item.surface->cluster_lod_stream : nullptr;
			bool landed = ok && stream != nullptr;
			if (landed) {
				// The closure: a parent's load failed or was dropped, so this one cannot be traced.
				const RTLodStreamGroup &group = stream->groups[item.group];
				for (uint32_t k = group.parents_first; k < group.parents_first + group.parents_count && landed; k++) {
					const RTLodStreamGroup::State parent = stream->groups[stream->links[k]].state;
					landed = parent == RTLodStreamGroup::RESIDENT || parent == RTLodStreamGroup::PERSISTENT;
				}
			}
			if (!landed) {
				if (regions[i].chunk >= 0) {
					_cluster_arena_retire(uint32_t(regions[i].chunk), regions[i].offset, regions[i].size); // A recorded move may still write it.
				}
				if (stream != nullptr) {
					_retire_cluster_geometry_blob(item.surface, item.group);
					_lod_stream_group_unload_pending(stream, item.group);
					lod_stream_stats.failed++;
				}
				continue;
			}
			RTLodStreamGroup &group = stream->groups[item.group];
			rd->buffer_update(cluster_lod_table, (item.surface->cluster_lod_table_offset + group.first_cluster * 2) * sizeof(uint32_t), uint32_t(sizeof(uint64_t) * group.cluster_count), compact.ptr() + item.first);
			if (group.blob_bytes > 0) {
				_write_cluster_geometry_records(item.surface, item.group); // A group is resident only with its blob in place (D93c).
			}
			stream->loading_bytes -= group.loading_bytes;
			group.loading_bytes = 0;
			group.state = RTLodStreamGroup::RESIDENT;
			group.arena_chunk = regions[i].chunk;
			group.arena_offset = regions[i].offset;
			group.bytes = regions[i].size;
			group.last_used = frames_drawn;
			stream->resident_bytes += group.bytes + group.blob_bytes;
			stream->resident_blob_bytes += group.blob_bytes;
			stream->depth_resident[group.depth]++;
			if (group.requested != 0 && lod_stream_stats.latency.size() < 65536) {
				lod_stream_stats.latency.push_back(uint32_t(MIN(frames_drawn - group.requested, uint64_t(UINT32_MAX))));
			}
			group.requested = 0;
			lod_stream_stats.loads++;
			lod_stream_stats.loaded_clusters += group.cluster_count;
		}
		if (!ok) {
			WARN_PRINT_ONCE("RT: a load of streamed cluster acceleration structures could not be placed; its groups stay at a coarser level.");
		}
		rd->free_rid(load.staging);
		rd->free_rid(load.args);
	}
}

// The residency step of a frame (D92), before its selection: the loads whose sizes are back land;
// then, once a frame, the requests and uses the selection read back since. Wanted groups (requested,
// every group of a pinned surface, a cached level's groups softly) are admitted coarse first, each
// only with its parents resident or loading, within the build cap and, for an unpinned surface, the
// budget, unloading least recently used groups to make room; what does not fit waits for a later
// frame's request. Nothing here waits on the GPU.
void RenderRaytracing::_process_lod_stream() {
	_complete_lod_stream_loads();
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	if (frames_drawn == cluster_lod_stream_step_frame) {
		return; // A second viewport's frame: its readbacks wait for the next step.
	}
	cluster_lod_stream_step_frame = frames_drawn;
	lod_stream_candidates_built = false;
	uint64_t load_usec = OS::get_singleton()->get_ticks_usec();
	_issue_lod_stream_gathers();
	load_usec = OS::get_singleton()->get_ticks_usec() - load_usec;
	const uint64_t min_age = cluster_lod_stream_min_age;

	struct Want {
		RTSurfaceData *surface = nullptr;
		uint32_t group = 0;
		uint32_t depth = 0;
		uint32_t order = 0;
		bool soft = false; // A cached level's ask: after the selection's requests, into free room only.
		bool operator<(const Want &p_other) const {
			if (soft != p_other.soft) {
				return !soft;
			}
			if (depth != p_other.depth) {
				return depth > p_other.depth; // Coarse first.
			}
			return order < p_other.order;
		}
	};
	LocalVector<Want> wants;
	LocalVector<RTLodStreamRead> reads;
	reads = cluster_lod_stream_reads;
	cluster_lod_stream_reads.clear();
	for (const RTLodStreamRead &read : reads) {
		const uint32_t words = uint32_t(read.bytes.size() / sizeof(uint32_t));
		if (words < 4) {
			continue;
		}
		const uint32_t *w = reinterpret_cast<const uint32_t *>(read.bytes.ptr());
		const bool requests = read.part == 0;
		const uint32_t cap = requests ? RT_LOD_STREAM_REQUEST_CAP : RT_LOD_STREAM_TOUCH_CAP;
		const uint32_t count = w[0];
		uint32_t &peak = requests ? cluster_lod_stream_request_peak : cluster_lod_stream_touch_peak;
		peak = MAX(MIN(count, cap), peak - peak / 32); // A decaying maximum, the next reads' size.
		const uint32_t listed = MIN(count, cap);
		const uint32_t records = MIN(listed, (words - 4) / 2);
		(requests ? lod_stream_stats.request_overflow : lod_stream_stats.touch_overflow) += count - listed;
		(requests ? lod_stream_stats.request_unread : lod_stream_stats.touch_unread) += listed - records;
		for (uint32_t i = 0; i < records; i++) {
			RTSurfaceData **found = cluster_lod_stream_surfaces.getptr(w[4 + i * 2]);
			const uint32_t g = w[4 + i * 2 + 1];
			if (!found || g >= (*found)->cluster_lod_stream->groups.size()) {
				continue; // A surface released since, or a region reused: harmless, and dropped.
			}
			RTLodStreamGroup &group = (*found)->cluster_lod_stream->groups[g];
			if (!requests) {
				group.last_used = MAX(group.last_used, read.frame);
				continue;
			}
			lod_stream_stats.requests++;
			lod_stream_stats.root_requests += group.parents_count == 0 ? 1 : 0;
			if (group.state != RTLodStreamGroup::ABSENT) {
				continue;
			}
			if (group.requested == 0) {
				group.requested = read.frame;
			}
			Want want;
			want.surface = *found;
			want.group = g;
			want.depth = group.depth;
			want.order = wants.size();
			wants.push_back(want);
		}
	}
	// A pinned surface wants every group, and a cached level asked for and not resident the groups
	// of its depth and coarser (the D90 trade: a level's memory for an instance's per-frame build);
	// the closure orders them.
	for (const KeyValue<uint32_t, RTSurfaceData *> &E : cluster_lod_stream_surfaces) {
		RTLodStream *stream = E.value->cluster_lod_stream;
		if (stream->absent_groups == 0) {
			continue;
		}
		uint32_t from_depth = UINT32_MAX;
		bool soft = false;
		if (_lod_stream_pinned(stream, frames_drawn, min_age)) {
			from_depth = 0;
		} else if (stream->level_wanted_frame != 0 && frames_drawn - stream->level_wanted_frame <= 2 && !(cluster_lod_stream_pressure_frame != 0 && frames_drawn - cluster_lod_stream_pressure_frame < min_age)) {
			// Not while the budget turns the selection's own requests away: under pressure the
			// memory goes to the detail the selection asks for, and the instance builds its own.
			from_depth = stream->level_wanted;
			soft = true;
		}
		if (from_depth == UINT32_MAX) {
			continue;
		}
		for (uint32_t g = 0; g < stream->groups.size(); g++) {
			RTLodStreamGroup &group = stream->groups[g];
			if (group.state == RTLodStreamGroup::ABSENT && group.depth >= from_depth) {
				if (group.requested == 0 && !soft) {
					group.requested = frames_drawn; // The latency counts what the trace waits for, not a level's fill.
				}
				Want want;
				want.surface = E.value;
				want.group = g;
				want.depth = group.depth;
				want.order = wants.size();
				want.soft = soft;
				wants.push_back(want);
			}
		}
	}

	uint64_t used = 0;
	uint64_t resident_total = 0;
	for (const KeyValue<uint32_t, RTSurfaceData *> &E : cluster_lod_stream_surfaces) {
		const RTLodStream *stream = E.value->cluster_lod_stream;
		resident_total += stream->resident_bytes;
		if (!_lod_stream_pinned(stream, frames_drawn, min_age)) {
			used += stream->resident_bytes + stream->loading_bytes;
		}
	}
	// A pin that lapsed puts its surface's groups under the budget again: back down to it first.
	if (used > cluster_lod_stream_budget) {
		_evict_lod_stream_groups(used - cluster_lod_stream_budget, used);
	}

	if (!wants.is_empty()) {
		wants.sort();
		const RD::ClusterAccelerationStructureCapabilities &caps = RD::get_singleton()->get_cluster_acceleration_structure_capabilities();
		const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
		uint32_t builds_left = cluster_lod_stream_builds_per_frame;
		// A frame's admitted gathers are issued together, at a fixed latency (D93c): what is admitted
		// here is what one frame uploads.
		uint64_t uploaded = 0;
		bool room_exhausted = false;
		bool overdue_taken = false;
		LocalVector<RTLodStreamLoad::Item> items[2]; // Per position format: float, then 16-bit.
		for (const Want &want : wants) {
			RTLodStream *stream = want.surface->cluster_lod_stream;
			RTLodStreamGroup &group = stream->groups[want.group];
			if (group.state != RTLodStreamGroup::ABSENT) {
				continue; // Wanted twice, or admitted above.
			}
			bool parents_ready = true;
			for (uint32_t k = group.parents_first; k < group.parents_first + group.parents_count && parents_ready; k++) {
				parents_ready = stream->groups[stream->links[k]].state != RTLodStreamGroup::ABSENT;
			}
			if (!parents_ready) {
				continue; // Its parents first: they are wanted too, and it is asked for again.
			}
			if (group.cluster_count > builds_left) {
				lod_stream_stats.deferred_cap++;
				continue;
			}
			// The frame's geometry uploads (D93c): a group past the cap waits, unless it is the first.
			const uint64_t upload = uint64_t(group.blob_bytes) + group.position_bytes;
			if (upload > 0 && uploaded > 0 && uploaded + upload > cluster_lod_stream_upload_cap) {
				lod_stream_stats.deferred_upload++;
				continue;
			}
			// The build budget (D97): the group's CLAS from the last frame's remainder, checked here
			// and spent below once the memory check has passed. The coarser resident cut stands in,
			// and the group is asked for again.
			uint32_t group_triangles = 0;
			bool budget_past = false;
			if (build_budget.enabled) {
				for (uint32_t c = group.first_cluster; c < group.first_cluster + group.cluster_count; c++) {
					group_triangles += stream->cluster_packed[c] & 0xFFFF;
				}
				if (!_build_budget_fits(group_triangles, true)) {
					// One overdue group a frame goes past the limits.
					budget_past = !overdue_taken && group.requested != 0 && frames_drawn - group.requested >= build_budget.max_wait_frames;
					if (!budget_past) {
						build_budget.deferred++;
						build_budget.deferred_total++;
						lod_stream_stats.deferred_build_budget++;
						continue;
					}
				}
			}
			const uint64_t estimate = (group.bytes > 0 ? group.bytes : _align_up(uint64_t(group.cluster_count) * cluster_lod_stream_cluster_worst, base_alignment)) + group.blob_bytes;
			const bool pinned = _lod_stream_pinned(stream, frames_drawn, min_age);
			// The parents hold their child from here, so the room it needs never comes from them.
			for (uint32_t k = group.parents_first; k < group.parents_first + group.parents_count; k++) {
				stream->groups[stream->links[k]].live_children++;
			}
			if (!pinned && used + estimate > cluster_lod_stream_budget) {
				const bool fits = !want.soft && !room_exhausted && _evict_lod_stream_groups(used + estimate - cluster_lod_stream_budget, used);
				if (!fits) {
					// The hold ends: a parent the eviction passed over while held is a candidate again.
					for (uint32_t k = group.parents_first; k < group.parents_first + group.parents_count; k++) {
						stream->groups[stream->links[k]].live_children--;
						_join_lod_stream_candidate(want.surface, stream->links[k], frames_drawn);
					}
					if (want.soft) {
						lod_stream_stats.deferred_soft++;
					} else {
						cluster_lod_stream_pressure_frame = frames_drawn;
						room_exhausted = true;
						lod_stream_stats.deferred_budget++;
					}
					continue;
				}
			}
			group.state = RTLodStreamGroup::LOADING;
			group.loading_bytes = estimate;
			stream->loading_bytes += estimate;
			stream->absent_groups--;
			if (!pinned) {
				used += estimate;
			}
			builds_left -= group.cluster_count;
			uploaded += upload;
			if (build_budget.enabled) {
				_build_budget_spend(group_triangles, true);
				overdue_taken = overdue_taken || budget_past;
				build_budget.built++;
			}
#ifdef TOOLS_ENABLED
			if (!build_budget.enabled) {
				for (uint32_t c = group.first_cluster; c < group.first_cluster + group.cluster_count; c++) {
					group_triangles += stream->cluster_packed[c] & 0xFFFF;
				}
			}
			frame_extra_builds++; // Its CLAS, counted in the frame that admits it (D97).
			frame_extra_triangles += group_triangles;
#endif
			RTLodStreamLoad::Item item;
			item.surface = want.surface;
			item.group = want.group;
			items[want.surface->is_compressed ? 1 : 0].push_back(item);
		}
		const uint64_t start_usec = OS::get_singleton()->get_ticks_usec();
		_start_lod_stream_load(items[0], false);
		_start_lod_stream_load(items[1], true);
		load_usec += OS::get_singleton()->get_ticks_usec() - start_usec;
	}
	lod_stream_stats.peak_bytes = MAX(lod_stream_stats.peak_bytes, resident_total);
	if (cluster_lod_stream_geometry_enabled) {
		lod_stream_stats.gather_usec += load_usec;
		lod_stream_stats.gather_max_usec = MAX(lod_stream_stats.gather_max_usec, load_usec);
	}

	if (frames_drawn % 300 == 0 && OS::get_singleton()->is_stdout_verbose()) {
		uint32_t resident_groups = 0, streamed_groups = 0, pinned_surfaces = 0, root_groups = 0, resident_roots = 0, open = 0;
		uint64_t persistent_bytes = 0, pinned_bytes = 0, loading_bytes = 0;
		for (const KeyValue<uint32_t, RTSurfaceData *> &E : cluster_lod_stream_surfaces) {
			const RTLodStream *stream = E.value->cluster_lod_stream;
			// The closure the cut's hole-freedom rests on: a resident group's parents are resident.
			for (const RTLodStreamGroup &group : stream->groups) {
				if (group.state != RTLodStreamGroup::RESIDENT) {
					continue;
				}
				resident_roots += group.parents_count == 0 ? 1 : 0;
				for (uint32_t k = group.parents_first; k < group.parents_first + group.parents_count; k++) {
					const RTLodStreamGroup::State parent = stream->groups[stream->links[k]].state;
					open += (parent == RTLodStreamGroup::RESIDENT || parent == RTLodStreamGroup::PERSISTENT) ? 0 : 1;
				}
			}
			for (uint32_t d = 0; d < stream->depth_streamed.size(); d++) {
				resident_groups += stream->depth_resident[d];
				streamed_groups += stream->depth_streamed[d];
			}
			root_groups += stream->root_groups;
			persistent_bytes += stream->persistent_bytes;
			loading_bytes += stream->loading_bytes;
			if (_lod_stream_pinned(stream, frames_drawn, min_age)) {
				pinned_surfaces++;
				pinned_bytes += stream->resident_bytes;
			}
		}
		uint32_t p50 = 0, latency_max = 0;
		if (!lod_stream_stats.latency.is_empty()) {
			lod_stream_stats.latency.sort();
			p50 = lod_stream_stats.latency[lod_stream_stats.latency.size() / 2];
			latency_max = lod_stream_stats.latency[lod_stream_stats.latency.size() - 1];
		}
		print_verbose(vformat("RT: CLAS streaming (D92): %d surfaces (%d pinned), %d of %d streamed groups resident (%d of %d roots, groups refined into nothing), %d KB resident (%d KB pinned, peak %d KB, budget %d KB), %d KB persistent, %d KB loading; over the run %d requests (%d for roots), %d loads (%d clusters), deferred %d for the build cap, %d for the build budget (D97), %d for the budget and %d cached-level asks without room, %d evictions, %d failed, %d requests and %d use reports past their lists, %d and %d past the frame's read; request to resident p50 %d, max %d frames over %d loads since the last report; %d resident groups with a parent not resident.", cluster_lod_stream_surfaces.size(), pinned_surfaces, resident_groups, streamed_groups, resident_roots, root_groups, resident_total / 1024, pinned_bytes / 1024, lod_stream_stats.peak_bytes / 1024, cluster_lod_stream_budget / 1024, persistent_bytes / 1024, loading_bytes / 1024, lod_stream_stats.requests, lod_stream_stats.root_requests, lod_stream_stats.loads, lod_stream_stats.loaded_clusters, lod_stream_stats.deferred_cap, lod_stream_stats.deferred_build_budget, lod_stream_stats.deferred_budget, lod_stream_stats.deferred_soft, lod_stream_stats.evictions, lod_stream_stats.failed, lod_stream_stats.request_overflow, lod_stream_stats.touch_overflow, lod_stream_stats.request_unread, lod_stream_stats.touch_unread, p50, latency_max, lod_stream_stats.latency.size(), open));
		if (open > 0) {
			ERR_PRINT(vformat("RT: CLAS streaming (D92): %d resident groups have a parent that is not; the traced cut may have holes.", open));
		}
		if (cluster_lod_stream_geometry_enabled) {
			uint64_t resident_blob_bytes = 0, persistent_blob_bytes = 0;
			for (const KeyValue<uint32_t, RTSurfaceData *> &E : cluster_lod_stream_surfaces) {
				resident_blob_bytes += E.value->cluster_lod_stream->resident_blob_bytes;
				persistent_blob_bytes += E.value->cluster_lod_stream->persistent_blob_bytes;
			}
			uint32_t upload_p50 = 0, upload_max = 0;
			if (!lod_stream_stats.upload_frames.is_empty()) {
				lod_stream_stats.upload_frames.sort();
				upload_p50 = lod_stream_stats.upload_frames[lod_stream_stats.upload_frames.size() / 2];
				upload_max = lod_stream_stats.upload_frames[lod_stream_stats.upload_frames.size() - 1];
			}
			print_verbose(vformat("RT: cluster geometry (D93c): %d KB of blobs resident with their groups (in the figures above) and %d KB persistent, %d KB in %d KB of arena chunks; uploads (blobs and staged positions) p50 %d KB, max %d KB a frame over %d frames with loads since the last report, %d KB over the run (cap %d KB a frame), deferred %d for the upload cap; loads on the render thread (gathers issued and placed, builds recorded) %d us at most in a frame since the last report, %d us over the run, gathers issued %d frames after admission, %d waited for over the run, %d in flight; %d populations over the run read an upload of their arrays (D93d); uploads %d KB held, %d KB at most, %d over the run, %d deferred for the cap; RAM copies %d KB, fallback levels %d KB.", resident_blob_bytes / 1024, persistent_blob_bytes / 1024, cluster_geometry_arena_live_bytes / 1024, cluster_geometry_arena_reserved_bytes / 1024, upload_p50, upload_max, lod_stream_stats.upload_frames.size(), lod_stream_stats.uploaded_bytes / 1024, cluster_lod_stream_upload_cap / 1024, lod_stream_stats.deferred_upload, lod_stream_stats.gather_max_usec, lod_stream_stats.gather_usec, RT_LOD_STREAM_GATHER_LATENCY, lod_stream_stats.gather_waits, cluster_lod_stream_gathers.size(), cluster_geometry_unstreamed_populations, materialized_live_bytes / 1024, materialized_peak_bytes / 1024, materialized_uploads, materialized_deferred, RendererRD::MeshStorage::get_singleton()->get_cluster_geometry_ram_bytes() / 1024, RendererRD::MeshStorage::get_singleton()->get_cluster_geometry_fallback_bytes() / 1024));
			lod_stream_stats.gather_max_usec = 0;
			lod_stream_stats.upload_frames.clear();
		}
		lod_stream_stats.latency.clear();
	}
}

RTLodInstanceEntry *RenderRaytracing::_access_lod_slot(RID &r_handle) {
	RTLodInstanceEntry *entry = lod_pool.get_or_null(r_handle);
	if (!entry) {
		r_handle = lod_pool.make_rid();
		entry = lod_pool.get_or_null(r_handle);
	}
	return entry;
}

void RenderRaytracing::_release_lod_entry(RTLodInstanceEntry &p_entry) {
	RD *rd = RD::get_singleton();
	if (p_entry.blas.is_valid() && rd->acceleration_structure_is_valid(p_entry.blas)) {
		rd->free_rid(p_entry.blas);
	}
	p_entry.blas = RID();
	p_entry.blas_address = 0;
	p_entry.pushed_frame = 0;
	p_entry.frame_blas = RID();
	if (p_entry.arena_chunk >= 0) {
		_cluster_arena_retire(uint32_t(p_entry.arena_chunk), p_entry.arena_offset, p_entry.arena_size);
		p_entry.arena_chunk = -1;
		p_entry.arena_offset = 0;
		p_entry.arena_size = 0;
	}
	p_entry.ptr = nullptr;
	p_entry.generation = 0;
}

// The instance's slot for this surface's hierarchy: kept while the surface's population is the
// one it was sized for, else a fresh worst-case slot in the arena and a handle at it.
bool RenderRaytracing::_prepare_lod_entry(RTLodInstanceEntry &p_entry, RTSurfaceData *p_surf_data) {
	if (p_entry.ptr == p_surf_data && p_entry.generation == p_surf_data->cluster_lod_generation && p_entry.blas.is_valid()) {
		return true;
	}
	_release_lod_entry(p_entry);
	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	const uint64_t size = _align_up(p_surf_data->cluster_lod_blas_worst_size, base_alignment);
	uint32_t chunk_index = 0;
	uint64_t offset = 0;
	if (size == 0 || !_cluster_arena_allocate(size, chunk_index, offset)) {
		return false;
	}
	const RTClusterArenaChunk &chunk = cluster_arena[chunk_index];
	p_entry.blas = rd->cluster_blas_create(chunk.buffer, chunk.base_offset + offset);
	if (p_entry.blas.is_null()) {
		_cluster_arena_retire(chunk_index, offset, size);
		return false;
	}
	p_entry.blas_address = chunk.address + offset;
	p_entry.arena_chunk = int32_t(chunk_index);
	p_entry.arena_offset = offset;
	p_entry.arena_size = size;
	p_entry.ptr = p_surf_data;
	p_entry.generation = p_surf_data->cluster_lod_generation;
	return true;
}

// Appends the instance to the frame's batch: its record for the shader (the object-to-trace
// transform as rows, the largest axis scale the metric scales the spheres and errors by, the
// counts and offsets), and the buffers its build touches, each once.
void RenderRaytracing::_queue_lod_instance(RTLodInstanceEntry &p_entry, const Transform3D &p_transform, uint32_t p_slot) {
	RTLodSelectBatch &batch = lod_batch;
	const RTSurfaceData *surf = p_entry.ptr;
	const uint32_t cluster_count = surf->cluster_count;
	batch.entries.push_back(&p_entry);
	batch.deformed_entries.push_back(nullptr);
	batch.slots.push_back(p_slot);
	batch.blas_handles.push_back(p_entry.blas);
	batch.blas_addresses.push_back(p_entry.blas_address);
	batch.lod0_clusters.push_back(surf->cluster_lod0_count);

	uint32_t *w = _lod_instance_record(p_transform, cluster_count, surf->cluster_lod_table_offset);
	w[16] = 0;
	w[17] = 0;
	w[18] = 0;
	w[19] = 0;
	if (surf->cluster_lod_stream) {
		w[23] = surf->cluster_lod_stream->state_offset; // Its groups' state (D92).
		batch.stream_items++;
	}
	_lod_record_tree(p_transform, surf->cluster_lod_tree);

	const RID storage = cluster_arena[p_entry.arena_chunk].buffer;
	if (!batch.storage_buffers.has(storage)) {
		batch.storage_buffers.push_back(storage);
	}
	const RID source = surf->arena_chunk >= 0 ? cluster_arena[surf->arena_chunk].buffer : surf->cluster_storage;
	if (source.is_valid() && !batch.source_buffers.has(source)) {
		batch.source_buffers.push_back(source);
	}
	p_entry.pushed_frame = RD::get_singleton()->get_frames_drawn();
	p_entry.frame_blas = p_entry.blas;
}

// The record's first sixteen words, shared by both kinds of item: the object-to-trace transform
// as rows, the largest axis scale the metric scales the spheres and errors by, the cluster count
// over every level, the hierarchy's word offset in the table, and the list's first cluster, which
// the flush turns into a word offset. Returns the record for the caller's remaining words.
uint32_t *RenderRaytracing::_lod_instance_record(const Transform3D &p_transform, uint32_t p_cluster_count, uint32_t p_table_offset) {
	RTLodSelectBatch &batch = lod_batch;
	const Basis &basis = p_transform.basis;
	float scale = MAX(float(basis.get_column(0).length()), MAX(float(basis.get_column(1).length()), float(basis.get_column(2).length())));
	scale = MAX(scale, 1e-6f);
	const uint32_t base = batch.instance_words.size();
	batch.instance_words.resize(base + RT_LOD_INSTANCE_WORDS);
	uint32_t *w = batch.instance_words.ptr() + base;
	for (int row = 0; row < 3; row++) {
		w[row * 4 + 0] = _float_bits(float(basis.rows[row].x));
		w[row * 4 + 1] = _float_bits(float(basis.rows[row].y));
		w[row * 4 + 2] = _float_bits(float(basis.rows[row].z));
		w[row * 4 + 3] = _float_bits(float(p_transform.origin[row]));
	}
	w[12] = _float_bits(scale);
	w[13] = p_cluster_count;
	w[14] = p_table_offset;
	w[15] = batch.cluster_total;
	for (uint32_t i = 16; i < RT_LOD_INSTANCE_WORDS; i++) {
		w[i] = 0;
	}
	batch.cluster_total += p_cluster_count;
	batch.max_clusters = MAX(batch.max_clusters, p_cluster_count);
	return w;
}

// The last queued item's traversal (D91): with a tree, its record flags it and names the tree's
// nodes and refs and the basis's Frobenius norm (the node test's scale, a bound on the transform's
// stretch under shear, rounded up), and the batch's queues grow by its node counts; without one it
// widens the exhaustive dispatch.
void RenderRaytracing::_lod_record_tree(const Transform3D &p_transform, const RTLodTree &p_tree) {
	RTLodSelectBatch &batch = lod_batch;
	uint32_t *w = batch.instance_words.ptr() + batch.instance_words.size() - RT_LOD_INSTANCE_WORDS;
	if (!cluster_lod_traversal_enabled || p_tree.nodes_offset == UINT32_MAX) {
		batch.flat_max_clusters = MAX(batch.flat_max_clusters, w[13]);
		return;
	}
	const Basis &basis = p_transform.basis;
	const double frobenius = Math::sqrt(double(basis.get_column(0).length_squared() + basis.get_column(1).length_squared() + basis.get_column(2).length_squared()));
	const float stretch = MAX(float(frobenius * (1.0 + 1.0 / 65536.0)), _bits_float(w[12]));
	w[16] |= 4u; // INSTANCE_FLAG_TREE.
	w[20] = p_tree.nodes_offset;
	w[21] = p_tree.refs_offset;
	w[22] = _float_bits(stretch);
	if (batch.tree_level_capacity.size() < p_tree.level_nodes.size()) {
		const uint32_t from = batch.tree_level_capacity.size();
		batch.tree_level_capacity.resize(p_tree.level_nodes.size());
		for (uint32_t l = from; l < batch.tree_level_capacity.size(); l++) {
			batch.tree_level_capacity[l] = 0;
		}
	}
	for (uint32_t l = 0; l < p_tree.level_nodes.size(); l++) {
		batch.tree_level_capacity[l] += p_tree.level_nodes[l];
	}
	batch.tree_group_capacity += p_tree.group_count;
	batch.tree_count++;
}

// A deformed entry on the selection (D87): its record flags the CLAS build and names its CLAS infos
// in the table (build infos, or instantiation infos when templated, D108). Templated, it is a
// persistent item (D109): its own slots (base, stride, count), its cluster list, its kept digest,
// its triangles and its BLAS address, forced where its slots may not hold its last selection over
// the current positions; else its first slot of the frame's region (the flush turns that into the
// destination base). The builds take its storage, its owned vertex buffer, its index buffer and
// its templates.
void RenderRaytracing::_queue_lod_deformed_instance(RTDeformedCacheEntry &p_entry, const Transform3D &p_transform, uint32_t p_slot) {
	RTLodSelectBatch &batch = lod_batch;
	const uint32_t cluster_count = p_entry.cluster_lod_count;
	batch.entries.push_back(nullptr);
	batch.deformed_entries.push_back(&p_entry);
	batch.slots.push_back(p_slot);
	batch.blas_handles.push_back(p_entry.ptr->blas);
	batch.blas_addresses.push_back(p_entry.cluster_blas_address);
	batch.lod0_clusters.push_back(p_entry.cluster_count);

	uint32_t *w = _lod_instance_record(p_transform, cluster_count, p_entry.cluster_lod_table_offset);
	w[16] = 1; // INSTANCE_FLAG_DEFORMED.
	w[17] = p_entry.cluster_lod_infos_offset;
	w[18] = batch.clas_bound; // The first slot; the flush writes the destination base over 18 and 19.
	w[19] = 0;
	const bool persistent = cluster_lod_deformed_templated && p_entry.cluster_templated && p_entry.cluster_lod_digest_offset != UINT32_MAX;
	uint32_t slots = cluster_count;
	if (persistent) {
		// Its own slots (D109): forced where its slots do not hold its last selection over the
		// current positions, that is a new layout, a skinning change, or the batch's LOD 0 rebuild
		// since; a first build counted by the CPU, the updates by the device.
		const bool first = p_entry.cluster_lod_pending_kind == RTDeformedCacheEntry::CLUSTER_QUEUED_FIRST_BUILD;
		const bool force = first || p_entry.cluster_lod_pending_kind == RTDeformedCacheEntry::CLUSTER_QUEUED_CHANGED || !p_entry.cluster_lod_last || p_entry.cluster_lod_slots_stale;
		slots = p_entry.cluster_lod_slot_capacity;
		w[16] = 1 | 8 | (force ? 16 : 0) | (first ? 32 : 0); // DEFORMED, PERSISTENT, FORCE, FIRST.
		w[18] = uint32_t(p_entry.cluster_clas_address & 0xFFFFFFFFULL);
		w[19] = uint32_t(p_entry.cluster_clas_address >> 32);
		w[24] = p_entry.cluster_clas_stride;
		w[25] = slots;
		w[28] = batch.clas_bound; // Its cluster list's first word; the flush adds the section's offset.
		w[29] = p_entry.cluster_lod_digest_offset;
		w[30] = p_entry.ptr ? p_entry.ptr->geometry.primitive_count : 0;
		w[32] = uint32_t(p_entry.cluster_blas_address & 0xFFFFFFFFULL);
		w[33] = uint32_t(p_entry.cluster_blas_address >> 32);
		batch.deformed_storage_buffers.push_back(p_entry.cluster_storage); // Its own build's, not the static one's.
		batch.deformed_max_capacity = MAX(batch.deformed_max_capacity, slots);
	} else {
		batch.storage_buffers.push_back(p_entry.cluster_storage); // Its own; nothing to dedupe.
	}
	_lod_record_tree(p_transform, p_entry.cluster_lod_tree);

	batch.clas_source_buffers.push_back(p_entry.owned_vb_full);
	if (p_entry.cluster_templated) {
		// The templates its instantiations read (D108); a few meshes, many instances.
		const RTDeformedClusterTemplate *cluster_template = deformed_cluster_templates.getptr(p_entry.cluster_template_key);
		if (cluster_template && !batch.clas_template_buffers.has(cluster_template->storage)) {
			batch.clas_template_buffers.push_back(cluster_template->storage);
		}
	}
	if (p_entry.cluster_index_buffer.is_valid() && !batch.clas_index_buffers_seen.has(p_entry.cluster_index_buffer)) {
		batch.clas_index_buffers_seen.push_back(p_entry.cluster_index_buffer); // A few meshes, many instances.
		batch.clas_source_buffers.push_back(p_entry.cluster_index_buffer);
	}
	batch.deformed_count++;
	batch.clas_bound += slots;
	batch.clas_total_triangles += p_entry.cluster_lod_total_triangles;
	batch.clas_total_vertices += p_entry.cluster_lod_total_vertices;
	p_entry.cluster_frame_kind = p_entry.cluster_lod_pending_kind;
	p_entry.cluster_lod_pushed_frame = RD::get_singleton()->get_frames_drawn();
}

// The view the selection's metric reads (D86): an error of cluster_lod_pixel_error output pixels
// at the group's distance, as the reference reads it (error / distance >= 2 tan(fov / 2) x pixels
// / height, the projection's second diagonal being 1 / tan(fov / 2); orthographic, the same size
// per pixel with no distance), the camera in trace space and the near clamp. The selection and
// the sharing's classification (D89) read the same one.
RenderRaytracing::LodView RenderRaytracing::_lod_view(const RenderDataRD *p_render_data, const RTViewportState *p_state, float p_pixel_error) {
	LodView view;
	const RenderSceneDataRD *scene_data = p_render_data->scene_data;
	const float projection_scale = float(scene_data->cam_projection.columns[1][1]);
	uint32_t target_height = 0;
	if (p_render_data->render_buffers.is_valid()) {
		target_height = uint32_t(MAX(0, p_render_data->render_buffers->get_target_size().height));
	}
	if (projection_scale > 0.0f && target_height > 0) {
		view.threshold = 2.0f * p_pixel_error / (projection_scale * float(target_height));
	}
	view.camera = scene_data->cam_transform.origin - p_state->rt_origin;
	view.near_distance = MAX(scene_data->z_near, 1e-3f);
	view.orthogonal = scene_data->cam_orthogonal;
	return view;
}

// The depths an instance's selection can fall in (D89), from each depth's bounds instead of its
// groups, and conservative both ways. A cluster of depth d is selected only if its group (depth
// d) passes the metric and the group it refines (depth d - 1) does not. No group of a depth can
// pass if the depth's largest error does not at the nearest any of its spheres can come, so
// level_min is the first depth that could; every group of a depth passes if the smallest error
// does at the farthest a group's near side can be, so once every depth from j on is certain to,
// no cluster past depth j can be selected, and level_max is the first such j. The bounds move a
// point at most the basis's Frobenius norm times its distance, which holds under shear, where the
// selection's largest-axis scale would not; the margins cover the shader's float arithmetic.
void RenderRaytracing::_classify_lod_instance(const RTSurfaceData *p_surf_data, const Transform3D &p_transform, const LodView &p_view, uint32_t &r_level_min, uint32_t &r_level_max) {
	const LocalVector<RTLodLevelBound> &levels = p_surf_data->cluster_lod_levels;
	const uint32_t level_count = levels.size();
	r_level_min = 0;
	r_level_max = UINT32_MAX; // Without bounds the range is unknown, and nothing is shared with it.
	if (level_count == 0) {
		return;
	}
	const Basis &basis = p_transform.basis;
	const Vector3 column_0 = basis.get_column(0);
	const Vector3 column_1 = basis.get_column(1);
	const Vector3 column_2 = basis.get_column(2);
	const float scale = MAX(MAX(float(column_0.length()), MAX(float(column_1.length()), float(column_2.length()))), 1e-6f); // As the record's.
	const double stretch = MAX(Math::sqrt(double(column_0.length_squared() + column_1.length_squared() + column_2.length_squared())), double(scale));
	const double threshold = p_view.threshold;
	const double near_distance = p_view.orthogonal ? 0.0 : double(p_view.near_distance); // The shader clamps a perspective distance alone.
	const double error_up = 1.0 + 1.0 / 1024.0;
	const double error_down = 1.0 - 1.0 / 1024.0;

	bool min_found = false;
	uint32_t certain_from = level_count; // The first depth from which every depth surely passes.
	bool certain_run = true;
	for (uint32_t l = level_count; l-- > 0;) {
		const RTLodLevelBound &level = levels[l];
		bool may_pass = false;
		bool must_pass = true;
		if (!level.empty) {
			const Vector3 center = p_transform.xform(level.center);
			const double distance = double((center - p_view.camera).length());
			const double magnitude = double(p_transform.origin.length()) + stretch * (double(level.center.length()) + double(level.enclosing_radius)) + double(p_view.camera.length());
			const double slack = magnitude * (1.0 / 65536.0);
			double nearest = distance - stretch * double(level.enclosing_radius) - slack;
			double farthest = distance + stretch * double(level.center_spread) - double(scale) * double(level.min_radius) + slack;
			if (p_view.orthogonal) {
				nearest = 1.0;
				farthest = 1.0;
			}
			may_pass = double(level.max_error) * error_up * double(scale) >= threshold * MAX(near_distance, nearest);
			must_pass = double(level.min_error) * error_down * double(scale) >= threshold * MAX(near_distance, farthest);
		}
		if (may_pass) {
			r_level_min = l; // The scan runs coarse to fine, so the last found is the finest.
			min_found = true;
		}
		if (certain_run && must_pass) {
			certain_from = l;
		} else {
			certain_run = false;
		}
	}
	if (!min_found) {
		r_level_min = 0;
	}
	r_level_max = MAX(MIN(certain_from, level_count - 1), r_level_min);
}

// A static instance's own build: its slot prepared, its record queued and its slot in the TLAS on
// it. False with the slot left on the surface's LOD 0 BLAS, past the dispatch's instance limit or
// with no room in the arena.
bool RenderRaytracing::_queue_lod_candidate(const RTLodSelectBatch::Candidate &p_candidate) {
	if (lod_batch.entries.size() + lod_batch.cache_builds.size() >= cluster_lod_max_instances || !_prepare_lod_entry(*p_candidate.entry, p_candidate.surface)) {
		return false;
	}
	_queue_lod_instance(*p_candidate.entry, p_candidate.transform, p_candidate.slot);
	p_candidate.entry->last_built_frame = RSG::rasterizer->get_frame_number();
	blass[p_candidate.slot] = p_candidate.entry->blas;
	return true;
}

// Cached levels (D90), before the stand-ins. A candidate whose selection spans no more than the
// tolerance (its coarsest depth at most that far below its finest) is covered by any one depth
// from its finest down to the tolerance below its coarsest, and traces such a depth's cached
// level: the coarsest one kept, else one it asks for. A surface's asks are settled together in
// the order of their finest depths, each taking the depth last asked for when that is within its
// range and else asking for its own finest, which gives the fewest levels for the set, each as
// coarse as it can be. A candidate whose finest depth is LOD 0, whose range is wider than the
// tolerance, whose surface's CLAS are about to move or whose level finds no slot is left to the
// sharing.
void RenderRaytracing::_place_lod_cached_levels() {
	RTLodSelectBatch &batch = lod_batch;
	const uint32_t candidate_count = batch.candidates.size();
	const uint32_t frame = RSG::rasterizer->get_frame_number();
	const uint32_t NO_DEPTH = UINT32_MAX;
	batch.cached_depths.resize(candidate_count);
	batch.cache_requests.clear();
	for (uint32_t i = 0; i < candidate_count; i++) {
		const RTLodSelectBatch::Candidate &candidate = batch.candidates[i];
		RTSurfaceData *surface = candidate.surface;
		// A streamed surface's finest level is a cached level too (D92): it has no LOD 0 BLAS.
		const bool streamed = surface->cluster_lod_stream != nullptr;
		if (surface->cluster_lod_levels.is_empty() || (candidate.level_min == 0 && !streamed) || candidate.level_max - candidate.level_min > cluster_lod_share_tolerance || surface->compaction_ticket != 0) {
			continue;
		}
		const uint32_t highest = candidate.level_min;
		const uint32_t lowest = MAX(candidate.level_max > cluster_lod_share_tolerance ? candidate.level_max - cluster_lod_share_tolerance : 0u, streamed ? 0u : 1u);
		uint32_t depth = NO_DEPTH;
		for (uint32_t d = highest + 1; d-- > lowest;) {
			if (d < surface->cluster_lod_cache.size() && surface->cluster_lod_cache[d].blas.is_valid()) {
				depth = d;
				break;
			}
		}
		if (depth != NO_DEPTH) {
			batch.targets[i] = RTLodSelectBatch::TARGET_CACHED;
			batch.cached_depths[i] = depth;
			surface->cluster_lod_cache[depth].last_used_frame = frame;
			if (streamed) {
				surface->cluster_lod_stream->level_last_used[depth] = RD::get_singleton()->get_frames_drawn();
			}
			continue;
		}
		RTLodSelectBatch::CacheRequest request;
		request.generation = surface->cluster_lod_generation;
		request.highest = highest;
		request.lowest = lowest;
		request.candidate = i;
		batch.cache_requests.push_back(request);
	}
	batch.cache_requests.sort();
	RTSurfaceData *surface = nullptr;
	uint32_t point = NO_DEPTH;
	for (const RTLodSelectBatch::CacheRequest &request : batch.cache_requests) {
		RTSurfaceData *candidate_surface = batch.candidates[request.candidate].surface;
		if (candidate_surface != surface) {
			surface = candidate_surface;
			point = NO_DEPTH;
		}
		if (point == NO_DEPTH || request.lowest > point) {
			point = _prepare_lod_cached_level(surface, request.highest) ? request.highest : NO_DEPTH;
		}
		if (point != NO_DEPTH) {
			batch.targets[request.candidate] = RTLodSelectBatch::TARGET_CACHED;
			batch.cached_depths[request.candidate] = point;
			surface->cluster_lod_cache[point].last_used_frame = frame;
			if (surface->cluster_lod_stream) {
				surface->cluster_lod_stream->level_last_used[point] = RD::get_singleton()->get_frames_drawn();
			}
		}
	}
}

// A cached level's slot (D90): the driver's size for one fast-trace BLAS over the level's exact
// count, in the arena, and a handle at it; the level is built in this frame's batch, after every
// other item. False past the dispatch's instance limit or with no room.
bool RenderRaytracing::_prepare_lod_cached_level(RTSurfaceData *p_surf_data, uint32_t p_depth) {
	RTLodSelectBatch &batch = lod_batch;
	const uint32_t count = p_surf_data->cluster_lod_levels[p_depth].cut_clusters;
	if (count == 0 || batch.entries.size() + batch.cache_builds.size() >= cluster_lod_max_instances) {
		return false;
	}
	if (p_surf_data->cluster_lod_stream && !_lod_stream_level_resident(p_surf_data, p_depth)) {
		// Built once every group it holds is resident (D92), which the next residency step asks
		// for; the selection stands in meanwhile.
		RTLodStream *stream = p_surf_data->cluster_lod_stream;
		const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
		stream->level_wanted = stream->level_wanted_frame == frames_drawn ? MIN(stream->level_wanted, p_depth) : p_depth;
		stream->level_wanted_frame = frames_drawn;
		return false;
	}
	if (p_depth < p_surf_data->cluster_lod_cache.size() && p_surf_data->cluster_lod_cache[p_depth].blas.is_valid()) {
		return true; // Already kept, or prepared earlier this frame.
	}
	if (build_budget.enabled) {
		// The build budget (D97), the wait kept on the level's depth: the instance's own selection
		// stands in meanwhile.
		if (p_surf_data->cluster_lod_cache.size() < p_surf_data->cluster_lod_levels.size()) {
			p_surf_data->cluster_lod_cache.resize(p_surf_data->cluster_lod_levels.size());
		}
		RTLodCachedLevel &waiting = p_surf_data->cluster_lod_cache[p_depth];
		if (!_build_budget_take_walk(uint64_t(count) * RT_BUILD_BUDGET_CLUSTER_TRIANGLES, waiting.budget_frame, waiting.budget_wait_since)) {
			return false;
		}
	}
	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	RD::ClusterAccelerationStructureInput input;
	input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_CLUSTERS_BOTTOM_LEVEL;
	input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
	input.max_acceleration_structure_count = 1;
	input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
	input.max_total_cluster_count = count;
	input.max_cluster_count_per_acceleration_structure = count;
	uint64_t size = 0, scratch = 0;
	rd->cluster_acceleration_structure_get_build_sizes(input, size, scratch);
	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	size = _align_up(size, base_alignment);
	uint32_t chunk_index = 0;
	uint64_t offset = 0;
	if (size == 0 || !_cluster_arena_allocate(size, chunk_index, offset)) {
		return false;
	}
	const RTClusterArenaChunk &chunk = cluster_arena[chunk_index];
	const RID blas = rd->cluster_blas_create(chunk.buffer, chunk.base_offset + offset);
	if (blas.is_null()) {
		_cluster_arena_retire(chunk_index, offset, size);
		return false;
	}
	if (p_surf_data->cluster_lod_cache.size() < p_surf_data->cluster_lod_levels.size()) {
		p_surf_data->cluster_lod_cache.resize(p_surf_data->cluster_lod_levels.size());
	}
	RTLodCachedLevel &level = p_surf_data->cluster_lod_cache[p_depth];
	level.blas = blas;
	level.blas_address = chunk.address + offset;
	level.arena_chunk = int32_t(chunk_index);
	level.arena_offset = offset;
	level.arena_size = size;
	cluster_lod_cached_surfaces.insert(p_surf_data);
	cluster_lod_cache_live++;
	cluster_lod_cache_bytes += size;
	RTLodSelectBatch::CacheBuild build;
	build.surface = p_surf_data;
	build.depth = p_depth;
	batch.cache_builds.push_back(build);
	build_budget.built += build_budget.enabled ? 1 : 0;
	return true;
}

// The level's item: no instance and no slot in the TLAS, the record's flag and depth in place of
// a transform, the selection's list over every cluster as any item's, its exact count for the
// build's bounds.
void RenderRaytracing::_queue_lod_cached_level(RTSurfaceData *p_surf_data, uint32_t p_depth) {
	RTLodSelectBatch &batch = lod_batch;
	const RTLodCachedLevel &level = p_surf_data->cluster_lod_cache[p_depth];
	const uint32_t count = p_surf_data->cluster_lod_levels[p_depth].cut_clusters;
	if (batch.cache_max_clusters == 0) {
		batch.cache_first = batch.entries.size();
	}
	batch.entries.push_back(nullptr);
	batch.deformed_entries.push_back(nullptr);
	batch.slots.push_back(UINT32_MAX);
	batch.blas_handles.push_back(level.blas);
	batch.blas_addresses.push_back(level.blas_address);
	batch.lod0_clusters.push_back(0);

	uint32_t *w = _lod_instance_record(Transform3D(), p_surf_data->cluster_count, p_surf_data->cluster_lod_table_offset);
	w[16] = 2; // INSTANCE_FLAG_LEVEL.
	w[17] = p_depth;
	w[18] = count; // The slot's count: the shader takes no more, and counts any past it as overflow.
	_lod_record_tree(Transform3D(), RTLodTree()); // A level's rule is by depth: tested exhaustively.

	const RID storage = cluster_arena[level.arena_chunk].buffer;
	if (!batch.storage_buffers.has(storage)) {
		batch.storage_buffers.push_back(storage);
	}
	const RID source = p_surf_data->arena_chunk >= 0 ? cluster_arena[p_surf_data->arena_chunk].buffer : p_surf_data->cluster_storage;
	if (source.is_valid() && !batch.source_buffers.has(source)) {
		batch.source_buffers.push_back(source);
	}
	batch.cache_list_clusters += p_surf_data->cluster_count;
	batch.cache_total_clusters += count;
	batch.cache_max_clusters = MAX(batch.cache_max_clusters, count);
	cluster_lod_cache_builds_total++;
}

void RenderRaytracing::_release_lod_cached_level(RTLodCachedLevel &p_level) {
	if (p_level.blas.is_null()) {
		return;
	}
	RD *rd = RD::get_singleton();
	if (rd->acceleration_structure_is_valid(p_level.blas)) {
		rd->free_rid(p_level.blas);
	}
	if (p_level.arena_chunk >= 0) {
		_cluster_arena_retire(uint32_t(p_level.arena_chunk), p_level.arena_offset, p_level.arena_size);
	}
	cluster_lod_cache_live--;
	cluster_lod_cache_bytes -= p_level.arena_size;
	p_level = RTLodCachedLevel();
}

void RenderRaytracing::_release_lod_cached_levels(RTSurfaceData *p_surf_data) {
	for (RTLodCachedLevel &level : p_surf_data->cluster_lod_cache) {
		_release_lod_cached_level(level);
	}
	p_surf_data->cluster_lod_cache.clear();
	cluster_lod_cached_surfaces.erase(p_surf_data);
}

// BLAS sharing (D89), after the walk and before the flush. Each static instance on the selection
// is classified into the depths its selection can fall in. A build whose coarsest depth is at or
// under an instance's finest is at least as fine over every part of the surface, so the instance
// may trace it; it does when the build's finest depth is no more than cluster_lod_share_tolerance
// below the instance's coarsest, which bounds the detail it adds. The surface's LOD 0 BLAS, built
// once, is such a build with depth 0 at both ends. Per surface and per coarsest depth one instance
// stands for the others, the one starting coarsest (ties to the walk's order), and builds when
// anyone takes it; an instance takes the one starting coarsest among those it may trace, the LOD 0
// BLAS unless a stand-in starts coarser, and builds its own when neither is within the tolerance.
// With cached levels on (D90), the candidates a single depth covers are placed on one first, and
// the stand-ins settle the rest.
// With sharing off, every instance builds its own in the walk's order, D86's frame.
void RenderRaytracing::_resolve_lod_candidates(const RenderDataRD *p_render_data, const RTViewportState *p_state) {
	RTLodSelectBatch &batch = lod_batch;
	batch.shared_count = 0;
	batch.lod0_count = 0;
	batch.cached_count = 0;
	batch.shared_slots.clear(); // The flush clears them only when it records; a frame on cached levels alone does not (D90).
	batch.shared_surfaces.clear();
	const uint32_t candidate_count = batch.candidates.size();
	if (cluster_lod_stream_enabled) {
		_prepare_lod_stream_pins(); // The pinned surfaces' cached level 0 (D92), queued with the others below.
	}
	if (candidate_count == 0) {
		for (const RTLodSelectBatch::CacheBuild &build : batch.cache_builds) {
			_queue_lod_cached_level(build.surface, build.depth);
		}
		return;
	}
	if (!cluster_lod_share_enabled) {
		for (uint32_t i = 0; i < candidate_count; i++) {
			_queue_lod_candidate(batch.candidates[i]); // Past the limit, the slot keeps LOD 0 this frame.
		}
		batch.candidates.clear();
		return;
	}

	// Per surface and coarsest depth, the candidate standing for it.
	const uint32_t NONE = UINT32_MAX;
	const uint32_t LOD0 = UINT32_MAX - 1;
	const uint32_t CACHED = RTLodSelectBatch::TARGET_CACHED;
	const LodView view = _lod_view(p_render_data, p_state, lod_frame_pixel_error);
	batch.targets.resize(candidate_count);
	for (uint32_t i = 0; i < candidate_count; i++) {
		RTLodSelectBatch::Candidate &candidate = batch.candidates[i];
		_classify_lod_instance(candidate.surface, candidate.transform, view, candidate.level_min, candidate.level_max);
		batch.targets[i] = NONE;
	}
	if (cluster_lod_cache_enabled) {
		_place_lod_cached_levels(); // The cached levels first (D90): what they cover builds nothing.
	}
	batch.stand_in_bases.clear();
	batch.stand_ins.clear();
	for (uint32_t i = 0; i < candidate_count; i++) {
		const RTLodSelectBatch::Candidate &candidate = batch.candidates[i];
		const uint32_t depths = candidate.surface->cluster_lod_levels.size();
		if (depths == 0 || batch.targets[i] == CACHED) {
			continue;
		}
		AHashMap<RTSurfaceData *, uint32_t>::Iterator it = batch.stand_in_bases.find(candidate.surface);
		if (!it) {
			it = batch.stand_in_bases.insert(candidate.surface, batch.stand_ins.size());
			for (uint32_t d = 0; d < depths; d++) {
				batch.stand_ins.push_back(NONE);
			}
		}
		uint32_t &stand_in = batch.stand_ins[it->value + MIN(candidate.level_max, depths - 1)];
		if (stand_in == NONE || candidate.level_min > batch.candidates[stand_in].level_min) {
			stand_in = i;
		}
	}

	// What each candidate would trace: the LOD 0 BLAS, a stand-in's build, or its own (NONE).
	batch.builds.resize(candidate_count);
	memset(batch.builds.ptr(), 0, candidate_count);
	for (uint32_t i = 0; i < candidate_count; i++) {
		const RTLodSelectBatch::Candidate &candidate = batch.candidates[i];
		if (batch.targets[i] == CACHED) {
			continue;
		}
		// A streamed surface (D92) has no LOD 0 BLAS: its finest level is a cached level, placed above
		// when resident, so its candidates take a stand-in from depth 0 on or build their own.
		const bool streamed = candidate.surface->cluster_lod_stream != nullptr;
		uint32_t pick = streamed ? NONE : LOD0;
		uint32_t pick_min = 0;
		AHashMap<RTSurfaceData *, uint32_t>::Iterator it = batch.stand_in_bases.find(candidate.surface);
		if (it) {
			const uint32_t depths = candidate.surface->cluster_lod_levels.size();
			for (uint32_t d = streamed ? 0 : 1; d <= candidate.level_min && d < depths; d++) {
				const uint32_t stand_in = batch.stand_ins[it->value + d];
				if (stand_in == NONE) {
					continue;
				}
				// Coarser over a tie, and the LOD 0 BLAS, which costs no build, over a stand-in starting at 0.
				const uint32_t start = batch.candidates[stand_in].level_min;
				if (pick == NONE || (pick == LOD0 ? start > 0 : start >= pick_min)) {
					pick = stand_in;
					pick_min = start;
				}
			}
		}
		batch.targets[i] = (pick == NONE || pick == i || candidate.level_max - pick_min > cluster_lod_share_tolerance) ? NONE : pick;
	}
	// Who builds. A stand-in only ever takes one of a finer depth, so marking the takes of the
	// candidates that stand for nothing, then each surface's stand-ins from the coarsest depth
	// down, settles it in one pass: a stand-in taken builds and traces its own, its own pick
	// dropped; one not taken passes its pick on.
	for (uint32_t i = 0; i < candidate_count; i++) {
		const RTLodSelectBatch::Candidate &candidate = batch.candidates[i];
		AHashMap<RTSurfaceData *, uint32_t>::Iterator it = batch.stand_in_bases.find(candidate.surface);
		if (it && batch.stand_ins[it->value + MIN(candidate.level_max, candidate.surface->cluster_lod_levels.size() - 1)] == i) {
			continue;
		}
		const uint32_t target = batch.targets[i];
		if (target == NONE) {
			batch.builds[i] = 1;
		} else if (target != LOD0 && target != CACHED) {
			batch.builds[target] = 1;
		}
	}
	for (const KeyValue<RTSurfaceData *, uint32_t> &E : batch.stand_in_bases) {
		for (uint32_t d = E.key->cluster_lod_levels.size(); d-- > 0;) {
			const uint32_t stand_in = batch.stand_ins[E.value + d];
			if (stand_in == NONE || batch.builds[stand_in]) {
				continue;
			}
			const uint32_t target = batch.targets[stand_in];
			if (target == NONE) {
				batch.builds[stand_in] = 1;
			} else if (target != LOD0) {
				batch.builds[target] = 1;
			}
		}
	}

	// The builds queue in the walk's order; then every other candidate takes what it traces.
	for (uint32_t i = 0; i < candidate_count; i++) {
		if (batch.builds[i] && !_queue_lod_candidate(batch.candidates[i])) {
			batch.builds[i] = 0; // Not built this frame: it and the candidates taking it keep LOD 0.
		}
	}
	for (const RTLodSelectBatch::CacheBuild &build : batch.cache_builds) {
		_queue_lod_cached_level(build.surface, build.depth); // After every other item, a build of their own (D90).
	}
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	for (uint32_t i = 0; i < candidate_count; i++) {
		const RTLodSelectBatch::Candidate &candidate = batch.candidates[i];
		const uint32_t target = batch.targets[i];
		if (target == CACHED) {
			const RID blas = candidate.surface->cluster_lod_cache[batch.cached_depths[i]].blas;
			blass[candidate.slot] = blas;
			candidate.entry->pushed_frame = frames_drawn;
			candidate.entry->frame_blas = blas;
			batch.shared_slots.push_back(candidate.slot);
			batch.shared_surfaces.push_back(candidate.surface);
			batch.cached_count++;
			continue;
		}
		if (target == NONE || batch.builds[i]) {
			continue; // Its own build, queued above, or none and the LOD 0 BLAS it holds.
		}
		if (target == LOD0) {
			candidate.entry->pushed_frame = frames_drawn;
			candidate.entry->frame_blas = candidate.surface->blas;
			batch.lod0_count++;
			continue;
		}
		if (!batch.builds[target]) {
			continue;
		}
		const RTLodSelectBatch::Candidate &shared = batch.candidates[target];
		blass[candidate.slot] = shared.entry->blas;
		candidate.entry->pushed_frame = frames_drawn;
		candidate.entry->frame_blas = shared.entry->blas;
		batch.shared_slots.push_back(candidate.slot);
		batch.shared_surfaces.push_back(candidate.surface);
		batch.shared_count++;
	}
	batch.candidates.clear();
}

// Records the frame's selection and builds. Returns false with nothing the TLAS may take: the
// caller puts every queued instance on its surface's LOD 0 BLAS.
bool RenderRaytracing::_record_lod_batch(const RenderDataRD *p_render_data, const RTViewportState *p_state) {
	RTLodSelectBatch &batch = lod_batch;
	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	const uint32_t entry_count = batch.entries.size();
	const uint32_t cluster_total = batch.cluster_total;
	const uint32_t blas_info_stride = caps.bottom_level_build_info_size;
	ERR_FAIL_COND_V_MSG(blas_info_stride % sizeof(uint32_t) != 0 || blas_info_stride < 16, false, "RT: the driver's bottom-level build info is not a whole number of words.");

	// The deformed items' infos as the table holds them: instantiations of their templates (D108), else builds (D87).
	const uint32_t clas_info_stride = cluster_lod_deformed_templated ? caps.instantiate_info_size : caps.triangle_build_info_size;
	ERR_FAIL_COND_V_MSG(clas_info_stride % sizeof(uint32_t) != 0 || clas_info_stride == 0, false, "RT: the driver's cluster build or instantiation info is not a whole number of words.");
	const uint32_t clas_bound = batch.clas_bound;
	// The persistent deformed items (D109): templated, the first items of the frame (queued in the
	// walk, before the static ones and the cached levels), with their own instantiation and BLAS build.
	const uint32_t persistent_count = cluster_lod_deformed_templated ? batch.deformed_count : 0;
	for (uint32_t e = 0; e < persistent_count; e++) {
		ERR_FAIL_COND_V_MSG(e >= entry_count || batch.entries[e] != nullptr || batch.deformed_entries[e] == nullptr || (batch.instance_words[e * RT_LOD_INSTANCE_WORDS + 16] & 8u) == 0, false, "RT: the cluster LOD batch's deformed items are not its first, all persistent.");
	}

	// The frame's arguments, 16-byte sections: the build infos (their counts the selection's
	// atomics), the slot addresses, the instance records, the CLAS count (D87), the reference
	// lists, then the deformed items' CLAS destinations and build infos, which the selection
	// writes. One grow-only buffer and host copy, as the deformed batch keeps (D82).
	const uint64_t blas_infos_offset = 0;
	const uint64_t blas_dst_offset = _align_up(blas_infos_offset + uint64_t(blas_info_stride) * entry_count, 16);
	const uint64_t instances_offset = _align_up(blas_dst_offset + sizeof(uint64_t) * entry_count, 16);
	const uint64_t clas_count_offset = _align_up(instances_offset + uint64_t(RT_LOD_INSTANCE_WORDS) * sizeof(uint32_t) * entry_count, 16);
	// The traversal (D91): a queue length per breadth-first level, one past the last and the group
	// queue's, the verbose check's count per item, the first level's queue (each tree item at its
	// root), all written here; the deeper levels' queues, each at its level's capacity, and the group
	// queue, after everything else. A level or the group queue needing more workgroups than the
	// device dispatches sends the frame's tree items to the exhaustive test.
	uint32_t tree_levels = batch.tree_count > 0 ? batch.tree_level_capacity.size() : 0;
	const uint64_t group_limit = rd->limit_get(RD::LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X);
	uint64_t queue_rest_items = 0;
	for (uint32_t l = 0; l < tree_levels; l++) {
		if ((uint64_t(batch.tree_level_capacity[l]) + 1) / 2 > group_limit) {
			tree_levels = 0;
			break;
		}
		queue_rest_items += l > 0 ? batch.tree_level_capacity[l] : 0;
	}
	if ((uint64_t(batch.tree_group_capacity) + 1) / 2 > group_limit) {
		tree_levels = 0;
	}
	if (tree_levels == 0) {
		queue_rest_items = 0;
	}
	const uint32_t tree_items = tree_levels > 0 ? batch.tree_count : 0;
	const uint32_t group_items = tree_levels > 0 ? batch.tree_group_capacity : 0;
	const uint64_t tree_counts_offset = _align_up(clas_count_offset + 16, 16);
	const uint64_t group_count_offset = tree_counts_offset + sizeof(uint32_t) * (tree_levels + 1);
	const uint64_t check_offset = _align_up(group_count_offset + sizeof(uint32_t), 16);
	const uint64_t queue_offset = _align_up(check_offset + sizeof(uint32_t) * entry_count, 16);
	const uint64_t lists_offset = _align_up(queue_offset + sizeof(uint32_t) * 2 * tree_items, 16);
	const uint64_t clas_dst_offset = _align_up(lists_offset + sizeof(uint64_t) * cluster_total, 16);
	const uint64_t clas_infos_offset = _align_up(clas_dst_offset + sizeof(uint64_t) * clas_bound, 16);
	// The persistent items' cluster lists, the compacted BLAS count with the updates and their
	// triangles, the compacted build infos and BLAS addresses (D109).
	const uint64_t cluster_lists_offset = _align_up(clas_infos_offset + uint64_t(clas_info_stride) * clas_bound, 16);
	const uint64_t compact_counts_offset = _align_up(cluster_lists_offset + sizeof(uint32_t) * (persistent_count > 0 ? clas_bound : 0), 16);
	const uint64_t compact_infos_offset = compact_counts_offset + 16;
	const uint64_t compact_dst_offset = _align_up(compact_infos_offset + uint64_t(blas_info_stride) * persistent_count, 16);
	const uint64_t queue_rest_offset = _align_up(compact_dst_offset + sizeof(uint64_t) * persistent_count, 16);
	const uint64_t group_queue_offset = _align_up(queue_rest_offset + sizeof(uint32_t) * 2 * queue_rest_items, 16);
	// CLAS residency (D92): the requests and the use reports, each a count (zeroed apart from the
	// upload) and its capped list, each read back as a prefix sized from the recent counts.
	const bool stream = batch.stream_items > 0;
	const uint64_t stream_requests_offset = _align_up(group_queue_offset + sizeof(uint32_t) * 2 * group_items, 16);
	const uint64_t stream_touches_offset = stream_requests_offset + 16 + sizeof(uint32_t) * 2 * RT_LOD_STREAM_REQUEST_CAP;
	const uint64_t stream_size = stream ? 32 + sizeof(uint32_t) * 2 * (RT_LOD_STREAM_REQUEST_CAP + RT_LOD_STREAM_TOUCH_CAP) : 0;
	const uint64_t args_size = _align_up(stream_requests_offset + stream_size, 16);
	ERR_FAIL_COND_V_MSG(args_size > UINT32_MAX, false, "RT: the cluster LOD batch's arguments exceed 4 GB.");
	if (batch.args_capacity < args_size) {
		if (batch.args_buffer.is_valid()) {
			rd->free_rid(batch.args_buffer); // Deferred past the fence of every frame in flight.
			batch.args_buffer = RID();
		}
		const uint32_t capacity = uint32_t(MIN(MAX(args_size, uint64_t(batch.args_capacity) * 2), uint64_t(UINT32_MAX)));
		batch.args_capacity = 0;
		batch.args_buffer = rd->storage_buffer_create(capacity, {}, 0,
				RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
		ERR_FAIL_COND_V_MSG(batch.args_buffer.is_null(), false, "RT: the cluster LOD batch's argument buffer could not be created.");
		batch.args_capacity = capacity;
		rd->set_resource_name(batch.args_buffer, "RT cluster LOD batch args");
	}
	const RID args_buffer = batch.args_buffer;
	const uint64_t args_address = rd->buffer_get_device_address(args_buffer);
	if (batch.dispatch_buffer.is_null()) {
		batch.dispatch_buffer = rd->storage_buffer_create(16, {}, RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
		ERR_FAIL_COND_V_MSG(batch.dispatch_buffer.is_null(), false, "RT: the cluster LOD traversal's dispatch buffer could not be created.");
		rd->set_resource_name(batch.dispatch_buffer, "RT cluster LOD traversal dispatch");
	}
	// The untemplated deformed items' CLAS go to a region of the deformed batch's pool (D82's reuse
	// rule), each item's clusters in a contiguous range of slots; persistent items (D109) take none.
	int32_t region_index = -1;
	uint64_t region_base_address = 0;
	if (clas_bound > 0 && persistent_count == 0) {
		const uint64_t region_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
		const uint64_t region_alloc = region_alignment + uint64_t(clas_bound) * cluster_lod_clas_slot_size;
		ERR_FAIL_COND_V_MSG(region_alloc > UINT32_MAX, false, "RT: the cluster LOD selection's CLAS region exceeds 4 GB.");
		region_index = _deformed_cluster_region_acquire(region_alloc, region_alignment);
		ERR_FAIL_COND_V_MSG(region_index < 0, false, "RT: the cluster LOD selection's CLAS region could not be created.");
		const RTDeformedClusterRegion &region = deformed_cluster_batch.regions[region_index];
		region_base_address = rd->buffer_get_device_address(region.buffer) + region.base_offset;
	}

	// The host copy and the upload cover the sections the CPU writes; the lists the selection fills.
	if (batch.args_host.size() < lists_offset) {
		batch.args_host.resize(lists_offset);
	}
	uint8_t *args = batch.args_host.ptr();
	memset(args, 0, lists_offset);
	uint32_t flat_max_clusters = batch.flat_max_clusters;
	// The exhaustive dispatch's rows (D91): the items it tests, and the tree items the verbose check
	// tests, each as the range from its first item to its last. With the traversal on, the flat
	// items are the cached levels, contiguous from cache_first; off, or on a frame the tree items
	// fall back, every item. Only a surface whose tree could not be built mixes the two kinds, and
	// then the range takes the tree items between, which skip themselves.
	uint32_t flat_first = entry_count, flat_end = 0;
	uint32_t tree_first = entry_count, tree_end = 0;
	{
		uint64_t *dst = reinterpret_cast<uint64_t *>(args + blas_dst_offset);
		uint32_t *instances = reinterpret_cast<uint32_t *>(args + instances_offset);
		uint32_t *queue = reinterpret_cast<uint32_t *>(args + queue_offset);
		uint32_t queued = 0;
		memcpy(instances, batch.instance_words.ptr(), sizeof(uint32_t) * RT_LOD_INSTANCE_WORDS * entry_count);
		for (uint32_t e = 0; e < entry_count; e++) {
			uint32_t *record = instances + e * RT_LOD_INSTANCE_WORDS;
			if ((record[16] & 4u) != 0) { // INSTANCE_FLAG_TREE.
				if (tree_levels > 0) {
					queue[queued * 2] = e;
					queue[queued * 2 + 1] = 0; // The root.
					queued++;
				} else {
					record[16] &= ~4u;
					flat_max_clusters = MAX(flat_max_clusters, record[13]);
				}
			}
			if ((record[16] & 4u) != 0) {
				tree_first = MIN(tree_first, e);
				tree_end = e + 1;
			} else {
				flat_first = MIN(flat_first, e);
				flat_end = e + 1;
			}
			const uint32_t first_cluster = record[15];
			RD::ClusterBottomLevelBuildInfo blas_info;
			blas_info.cluster_references_count = 0;
			blas_info.cluster_references_stride = sizeof(uint64_t);
			blas_info.cluster_references_address = args_address + lists_offset + sizeof(uint64_t) * first_cluster;
			rd->cluster_bottom_level_build_info_write(args + blas_infos_offset + uint64_t(e) * blas_info_stride, blas_info);
			dst[e] = batch.blas_addresses[e];
			record[15] = uint32_t(lists_offset / sizeof(uint32_t)) + first_cluster * 2;
			if ((record[16] & 8u) != 0) { // INSTANCE_FLAG_PERSISTENT (D109): its own slots; its cluster list.
				record[28] += uint32_t(cluster_lists_offset / sizeof(uint32_t));
			} else if ((record[16] & 1u) != 0) { // INSTANCE_FLAG_DEFORMED.
				const uint64_t dst_base = region_base_address + uint64_t(record[18]) * cluster_lod_clas_slot_size;
				record[18] = uint32_t(dst_base & 0xFFFFFFFFULL);
				record[19] = uint32_t(dst_base >> 32);
			}
		}
		reinterpret_cast<uint32_t *>(args + tree_counts_offset)[0] = queued;
	}
	rd->buffer_update(args_buffer, 0, uint32_t(lists_offset), args);
	if (persistent_count > 0) {
		const uint32_t compact_init[4] = { 0, 0, 0, 0 };
		rd->buffer_update(args_buffer, uint32_t(compact_counts_offset), sizeof(compact_init), compact_init);
	}
	if (stream) {
		const uint32_t stream_init[4] = { 0, 0, 0, 0 };
		rd->buffer_update(args_buffer, uint32_t(stream_requests_offset), sizeof(stream_init), stream_init);
		rd->buffer_update(args_buffer, uint32_t(stream_touches_offset), sizeof(stream_init), stream_init);
	}
	if (tree_levels > 0) {
		const uint32_t dispatch_init[4] = { 0, 1, 1, 0 };
		rd->buffer_update(batch.dispatch_buffer, 0, sizeof(dispatch_init), dispatch_init);
	}

	const LodView view = _lod_view(p_render_data, p_state, lod_frame_pixel_error);
	const float threshold = view.threshold;
	struct SelectPC {
		float camera[3];
		float threshold;
		float near_distance;
		uint32_t entry_count;
		uint32_t instances_offset;
		uint32_t blas_infos_offset;
		uint32_t blas_info_stride;
		uint32_t flags;
		uint32_t clas_count_offset;
		uint32_t clas_dst_offset;
		uint32_t clas_infos_offset;
		uint32_t clas_info_words;
		uint32_t clas_slot_size;
		uint32_t queue_in;
		uint32_t queue_out;
		uint32_t count_in;
		uint32_t count_out;
		uint32_t check_offset;
		uint32_t group_queue;
		uint32_t group_count;
		uint32_t entry_first;
		uint32_t frame_index;
		uint32_t stream_requests;
		uint32_t stream_touches;
		uint32_t persistent_count; // D109.
		uint32_t compact_counts;
		uint32_t compact_infos;
		uint32_t compact_dst;
	} pc;
	pc.camera[0] = float(view.camera.x);
	pc.camera[1] = float(view.camera.y);
	pc.camera[2] = float(view.camera.z);
	pc.threshold = threshold;
	pc.near_distance = view.near_distance;
	pc.entry_count = entry_count;
	pc.instances_offset = uint32_t(instances_offset / sizeof(uint32_t));
	pc.blas_infos_offset = uint32_t(blas_infos_offset / sizeof(uint32_t));
	pc.blas_info_stride = blas_info_stride / sizeof(uint32_t);
	pc.flags = view.orthogonal ? 1u : 0u;
	pc.clas_count_offset = uint32_t(clas_count_offset / sizeof(uint32_t)); // The word after it counts the cached levels' overflow (D90).
	pc.clas_dst_offset = uint32_t(clas_dst_offset / sizeof(uint32_t));
	pc.clas_infos_offset = uint32_t(clas_infos_offset / sizeof(uint32_t));
	pc.clas_info_words = clas_info_stride / sizeof(uint32_t);
	pc.clas_slot_size = uint32_t(cluster_lod_clas_slot_size);
	pc.queue_in = 0;
	pc.queue_out = 0;
	pc.count_in = 0;
	pc.count_out = 0;
	pc.check_offset = uint32_t(check_offset / sizeof(uint32_t));
	pc.group_queue = uint32_t(group_queue_offset / sizeof(uint32_t));
	pc.group_count = uint32_t(group_count_offset / sizeof(uint32_t));
	pc.entry_first = 0;
	pc.frame_index = uint32_t(rd->get_frames_drawn());
	pc.stream_requests = uint32_t(stream_requests_offset / sizeof(uint32_t));
	pc.stream_touches = uint32_t(stream_touches_offset / sizeof(uint32_t));
	pc.persistent_count = persistent_count;
	pc.compact_counts = uint32_t(compact_counts_offset / sizeof(uint32_t));
	pc.compact_infos = uint32_t(compact_infos_offset / sizeof(uint32_t));
	pc.compact_dst = uint32_t(compact_dst_offset / sizeof(uint32_t));
	// The verbose check (D91): the exhaustive test counts the tree items' clusters in the same frame,
	// for the readback to compare with what the traversal appended; on the first flushes and every
	// thirtieth.
	const bool tree_check = tree_items > 0 && OS::get_singleton()->is_stdout_verbose() && (batch.frames_flushed < 8 || batch.frames_flushed % 30 == 0);
	// A verbose run brackets the selection with two timestamps, which the graph keeps in order with
	// every command around them, and reads the span a finished frame captured. The span is not the
	// selection's cost: it includes any wait the traversal's barriers make on other work still in
	// flight (about 0.2 ms on a 4000-rock field whatever the passes did, while the frame's GPU time
	// stayed level), so it is read beside the frame's time.
	const bool select_timer = OS::get_singleton()->is_stdout_verbose();
	if (select_timer) {
		const uint64_t timer_frame = rd->get_captured_timestamps_frame();
		if (timer_frame != lod_select_timer_frame) {
			lod_select_timer_frame = timer_frame;
			uint64_t begin = 0;
			bool open = false;
			for (uint32_t i = 0; i < rd->get_captured_timestamps_count(); i++) {
				const String name = rd->get_captured_timestamp_name(i);
				if (name == "RT cluster LOD selection") {
					begin = rd->get_captured_timestamp_gpu_time(i);
					open = true;
				} else if (open && name == "RT cluster LOD selection end" && lod_select_gpu_us.size() < 65536) {
					lod_select_gpu_us.push_back(float(double(rd->get_captured_timestamp_gpu_time(i) - begin) / 1000.0));
					open = false;
				}
			}
		}
		rd->capture_timestamp("RT cluster LOD selection");
	}

	// The selection, in a compute list of its own (RenderingDevice records no cluster build
	// while one is open); the graph orders it after the upload and before the build through the
	// argument buffer's tracker, and after every table upload through the table's. The traversal's
	// cluster pass reads the dispatch buffer as its indirect arguments, so it runs in a list of its
	// own with the arguments bound in its place (the graph takes one usage of a buffer per list).
	{
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 0;
			u.append_id(args_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 1;
			u.append_id(cluster_lod_table);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 2;
			u.append_id(batch.dispatch_buffer);
			uniforms.push_back(u);
		}
		RID uniform_set = rd->uniform_set_create(uniforms, lod_select_shader.version_shader, 0, /*p_linear_pool=*/true);
		ERR_FAIL_COND_V(!uniform_set.is_valid(), false);
		RID cluster_set;
		if (tree_levels > 0) {
			uniforms.write[2].set_id(0, args_buffer);
			cluster_set = rd->uniform_set_create(uniforms, lod_select_shader.version_shader, 0, /*p_linear_pool=*/true);
			if (!cluster_set.is_valid()) {
				rd->free_rid(uniform_set);
				ERR_FAIL_V(false);
			}
		}
		RD::ComputeListID compute_list = rd->compute_list_begin();
		rd->compute_list_bind_compute_pipeline(compute_list, lod_select_shader.pipeline);
		rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
		if (flat_max_clusters > 0 && flat_first < flat_end) {
			SelectPC flat_pc = pc;
			flat_pc.entry_first = flat_first;
			rd->compute_list_set_push_constant(compute_list, &flat_pc, sizeof(SelectPC));
			rd->compute_list_dispatch_threads(compute_list, flat_max_clusters, flat_end - flat_first, 1);
		}
		if (tree_check && tree_first < tree_end) {
			SelectPC check_pc = pc;
			check_pc.flags |= 2u; // FLAG_CHECK.
			check_pc.entry_first = tree_first;
			rd->compute_list_set_push_constant(compute_list, &check_pc, sizeof(SelectPC));
			rd->compute_list_dispatch_threads(compute_list, batch.max_clusters, tree_end - tree_first, 1);
		}
		// The traversal (D91): a dispatch per breadth-first level at that level's capacity, two
		// queue items per workgroup, each after a barrier on the previous level's queue.
		uint64_t level_queue = queue_offset;
		uint64_t next_queue = queue_rest_offset;
		for (uint32_t l = 0; l < tree_levels; l++) {
			if (l > 0) {
				rd->compute_list_add_barrier(compute_list);
			}
			SelectPC node_pc = pc;
			node_pc.flags |= 4u; // FLAG_NODES.
			node_pc.queue_in = uint32_t(level_queue / sizeof(uint32_t));
			node_pc.queue_out = uint32_t(next_queue / sizeof(uint32_t));
			node_pc.count_in = uint32_t(tree_counts_offset / sizeof(uint32_t)) + l;
			node_pc.count_out = node_pc.count_in + 1;
			rd->compute_list_set_push_constant(compute_list, &node_pc, sizeof(SelectPC));
			rd->compute_list_dispatch(compute_list, (batch.tree_level_capacity[l] + 1) / 2, 1, 1);
			level_queue = next_queue;
			next_queue += sizeof(uint32_t) * 2 * (l + 1 < tree_levels ? batch.tree_level_capacity[l + 1] : 0);
		}
		rd->compute_list_end();
		if (tree_levels > 0) {
			SelectPC cluster_pc = pc;
			cluster_pc.flags |= 8u; // FLAG_CLUSTERS.
			compute_list = rd->compute_list_begin();
			rd->compute_list_bind_compute_pipeline(compute_list, lod_select_shader.pipeline);
			rd->compute_list_bind_uniform_set(compute_list, cluster_set, 0);
			rd->compute_list_set_push_constant(compute_list, &cluster_pc, sizeof(SelectPC));
			rd->compute_list_dispatch_indirect(compute_list, batch.dispatch_buffer, 0);
			rd->compute_list_end();
			rd->free_rid(cluster_set);
		}
		if (persistent_count > 0) {
			// The commit, then the digests it compared against (D109), after every selection pass.
			compute_list = rd->compute_list_begin();
			rd->compute_list_bind_compute_pipeline(compute_list, lod_select_shader.pipeline);
			rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
			SelectPC commit_pc = pc;
			commit_pc.flags |= 16u; // FLAG_COMMIT.
			rd->compute_list_set_push_constant(compute_list, &commit_pc, sizeof(SelectPC));
			rd->compute_list_dispatch_threads(compute_list, MAX(batch.deformed_max_capacity, 1u), persistent_count, 1);
			rd->compute_list_add_barrier(compute_list);
			SelectPC digest_pc = pc;
			digest_pc.flags |= 32u; // FLAG_DIGEST.
			rd->compute_list_set_push_constant(compute_list, &digest_pc, sizeof(SelectPC));
			rd->compute_list_dispatch_threads(compute_list, persistent_count, 1, 1);
			rd->compute_list_end();
		}
		rd->free_rid(uniform_set);
	}
	if (select_timer) {
		rd->capture_timestamp("RT cluster LOD selection end");
	}
	// The frame's requests and uses (D92), for the residency step a frame queue later: each list's
	// count and a prefix twice its recent peak (at least 256 records, at most the list). A request
	// past the prefix is made again the next frame, the per-frame guard being per frame; a use report
	// past it waits for its group's next period.
	if (stream) {
		const uint32_t request_read = CLAMP(Math::next_power_of_2(cluster_lod_stream_request_peak * 2 + 64), 256u, RT_LOD_STREAM_REQUEST_CAP);
		const uint32_t touch_read = CLAMP(Math::next_power_of_2(cluster_lod_stream_touch_peak * 2 + 64), 256u, RT_LOD_STREAM_TOUCH_CAP);
		rd->buffer_get_data_async(args_buffer, callable_mp_static(&RenderRaytracing::_on_lod_stream_read).bind(rd->get_frames_drawn(), 0u), uint32_t(stream_requests_offset), 16 + uint32_t(sizeof(uint32_t)) * 2 * request_read);
		rd->buffer_get_data_async(args_buffer, callable_mp_static(&RenderRaytracing::_on_lod_stream_read).bind(rd->get_frames_drawn(), 1u), uint32_t(stream_touches_offset), 16 + uint32_t(sizeof(uint32_t)) * 2 * touch_read);
	}

	// The persistent items' CLAS (D109): the commit's instantiations, as many as it counted, into the
	// items' own slots, rewritten in place where the last frame's trace read them through the BLAS:
	// their handles are listed, so their trackers, which the tracing lists declared, order the rewrite
	// after those reads (as D107's batch); then their BLAS over the compacted list, as many as it
	// counted. A skipped item keeps both.
	if (persistent_count > 0) {
		if (!_build_deformed_cluster_templates()) {
			return false;
		}
		uint32_t max_triangles = 0, max_vertices = 0;
		_deformed_cluster_maxima(caps, max_triangles, max_vertices);
		Vector<RID> source_buffers, storage_buffers, template_buffers, blas_handles;
		source_buffers.resize(batch.clas_source_buffers.size());
		memcpy(source_buffers.ptrw(), batch.clas_source_buffers.ptr(), sizeof(RID) * batch.clas_source_buffers.size());
		storage_buffers.resize(batch.deformed_storage_buffers.size());
		memcpy(storage_buffers.ptrw(), batch.deformed_storage_buffers.ptr(), sizeof(RID) * batch.deformed_storage_buffers.size());
		template_buffers.resize(batch.clas_template_buffers.size());
		memcpy(template_buffers.ptrw(), batch.clas_template_buffers.ptr(), sizeof(RID) * batch.clas_template_buffers.size());
		blas_handles.resize(persistent_count);
		memcpy(blas_handles.ptrw(), batch.blas_handles.ptr(), sizeof(RID) * persistent_count);
		{
			RD::ClusterAccelerationStructureBuild build;
			build.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_INSTANTIATE_TRIANGLE_CLUSTER;
			build.input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
			build.input.max_acceleration_structure_count = clas_bound;
			build.input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT);
			build.input.vertex_format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
			build.input.max_cluster_triangle_count = max_triangles;
			build.input.max_cluster_vertex_count = max_vertices;
			build.input.max_total_triangle_count = batch.clas_total_triangles;
			build.input.max_total_vertex_count = batch.clas_total_vertices;
			build.dst_storage_buffers = storage_buffers;
			build.dst_addresses_buffer = args_buffer;
			build.dst_addresses_offset = clas_dst_offset;
			build.dst_addresses_stride = sizeof(uint64_t);
			build.src_infos_buffer = args_buffer;
			build.src_infos_offset = clas_infos_offset;
			build.src_infos_stride = clas_info_stride;
			build.src_infos_count_buffer = args_buffer;
			build.src_infos_count_offset = clas_count_offset;
			build.source_buffers = source_buffers;
			build.source_acceleration_structure_buffers = template_buffers;
			build.dst_acceleration_structures = blas_handles;
			if (rd->cluster_acceleration_structure_build(build) != OK) {
				return false;
			}
		}
		{
			RD::ClusterAccelerationStructureBuild build;
			build.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_CLUSTERS_BOTTOM_LEVEL;
			build.input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
			build.input.max_acceleration_structure_count = persistent_count;
			build.input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT);
			build.input.max_total_cluster_count = clas_bound;
			build.input.max_cluster_count_per_acceleration_structure = MAX(batch.deformed_max_capacity, 1u);
			build.dst_storage_buffers = storage_buffers;
			build.dst_addresses_buffer = args_buffer;
			build.dst_addresses_offset = compact_dst_offset;
			build.dst_addresses_stride = sizeof(uint64_t);
			build.src_infos_buffer = args_buffer;
			build.src_infos_offset = compact_infos_offset;
			build.src_infos_stride = blas_info_stride;
			build.src_infos_count_buffer = args_buffer;
			build.src_infos_count_offset = compact_counts_offset;
			build.source_acceleration_structure_buffers = storage_buffers; // Their CLAS, in their own storage.
			build.dst_acceleration_structures = blas_handles;
			if (rd->cluster_acceleration_structure_build(build) != OK) {
				return false;
			}
		}
#ifdef TOOLS_ENABLED
		// The updates the device counted, for the counters a frame queue later (they are published in
		// tools builds only), and the clusters past an item's slots, which a sound cut bound never gives.
		rd->buffer_get_data_async(args_buffer, callable_mp_static(&RenderRaytracing::_on_lod_deformed_counts_read).bind(persistent_count), uint32_t(compact_counts_offset), 16);
		rd->buffer_get_data_async(args_buffer, callable_mp_static(&RenderRaytracing::_on_lod_cache_overflow_read), uint32_t(clas_count_offset + sizeof(uint32_t)), 2 * sizeof(uint32_t));
#endif
	}

	// The deformed items' CLAS (D87): every cluster the selection picked for them, built into the
	// slots the selection assigned, as many as it counted; instantiated from the clusters' templates
	// (D108), built first if new. The slots are the pool's region, never rewritten in place.
	if (clas_bound > 0 && persistent_count == 0) {
		uint32_t max_triangles = 0, max_vertices = 0;
		_deformed_cluster_maxima(caps, max_triangles, max_vertices);
		Vector<RID> source_buffers;
		source_buffers.resize(batch.clas_source_buffers.size());
		memcpy(source_buffers.ptrw(), batch.clas_source_buffers.ptr(), sizeof(RID) * batch.clas_source_buffers.size());
		RD::ClusterAccelerationStructureBuild build;
		if (cluster_lod_deformed_templated) {
			if (!_build_deformed_cluster_templates()) {
				return false;
			}
			build.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_INSTANTIATE_TRIANGLE_CLUSTER;
			for (const RID &template_buffer : batch.clas_template_buffers) {
				build.source_acceleration_structure_buffers.push_back(template_buffer);
			}
		} else {
			build.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_TRIANGLE_CLUSTER;
		}
		build.input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
		build.input.max_acceleration_structure_count = clas_bound;
		build.input.flags = _cluster_build_flags(RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT);
		build.input.vertex_format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
		build.input.max_cluster_triangle_count = max_triangles;
		build.input.max_cluster_vertex_count = max_vertices;
		build.input.max_total_triangle_count = batch.clas_total_triangles;
		build.input.max_total_vertex_count = batch.clas_total_vertices;
		build.dst_storage_buffers.push_back(deformed_cluster_batch.regions[region_index].buffer);
		build.dst_addresses_buffer = args_buffer;
		build.dst_addresses_offset = clas_dst_offset;
		build.dst_addresses_stride = sizeof(uint64_t);
		build.src_infos_buffer = args_buffer;
		build.src_infos_offset = clas_infos_offset;
		build.src_infos_stride = clas_info_stride;
		build.src_infos_count_buffer = args_buffer;
		build.src_infos_count_offset = clas_count_offset;
		build.source_buffers = source_buffers;
		if (rd->cluster_acceleration_structure_build(build) != OK) {
			return false;
		}
	}

	// Every instance's cluster BLAS over its list, into its slot, fast-build as it is rebuilt every
	// frame; then the cached levels built this frame (D90), the items after them, fast-trace as
	// they are traced from then on, each over its level's exact count.
	const uint32_t cache_first = batch.cache_max_clusters > 0 ? batch.cache_first : entry_count;
	{
		if (cluster_lod_stream_enabled) {
			// A streamed surface's loaded groups (D92) live in any chunk: every live chunk is a source.
			for (const RTClusterArenaChunk &chunk : cluster_arena) {
				if (chunk.buffer.is_valid() && !batch.source_buffers.has(chunk.buffer)) {
					batch.source_buffers.push_back(chunk.buffer);
				}
			}
		}
		Vector<RID> storage_buffers, source_buffers;
		storage_buffers.resize(batch.storage_buffers.size());
		source_buffers.resize(batch.source_buffers.size() + (region_index >= 0 ? 1 : 0));
		memcpy(storage_buffers.ptrw(), batch.storage_buffers.ptr(), sizeof(RID) * batch.storage_buffers.size());
		memcpy(source_buffers.ptrw(), batch.source_buffers.ptr(), sizeof(RID) * batch.source_buffers.size());
		if (region_index >= 0) {
			source_buffers.write[batch.source_buffers.size()] = deformed_cluster_batch.regions[region_index].buffer;
		}
		const uint32_t frame_total = cluster_total - batch.cache_list_clusters;
		for (int pass = 0; pass < 2; pass++) {
			// The persistent items' BLAS were built above (D109); the rest as before.
			const uint32_t first = pass == 0 ? persistent_count : cache_first;
			const uint32_t count = pass == 0 ? cache_first - persistent_count : entry_count - cache_first;
			if (count == 0) {
				continue;
			}
			Vector<RID> blas_handles;
			blas_handles.resize(count);
			memcpy(blas_handles.ptrw(), batch.blas_handles.ptr() + first, sizeof(RID) * count);

			RD::ClusterAccelerationStructureBuild build;
			build.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_CLUSTERS_BOTTOM_LEVEL;
			build.input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
			build.input.max_acceleration_structure_count = count;
			build.input.flags = _cluster_build_flags(pass == 0 ? RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT : RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
			build.input.max_total_cluster_count = pass == 0 ? frame_total : batch.cache_total_clusters;
			build.input.max_cluster_count_per_acceleration_structure = pass == 0 ? MIN(batch.max_clusters, frame_total) : batch.cache_max_clusters;
			build.dst_storage_buffers = storage_buffers;
			build.dst_addresses_buffer = args_buffer;
			build.dst_addresses_offset = blas_dst_offset + sizeof(uint64_t) * first;
			build.dst_addresses_stride = sizeof(uint64_t);
			build.src_infos_buffer = args_buffer;
			build.src_infos_offset = blas_infos_offset + uint64_t(blas_info_stride) * first;
			build.src_infos_stride = blas_info_stride;
			build.source_acceleration_structure_buffers = source_buffers; // The CLAS each BLAS references, read through its list.
			build.dst_acceleration_structures = blas_handles;
			if (rd->cluster_acceleration_structure_build(build) != OK) {
				return false;
			}
		}
	}

	// A cached level's slot is sized for the count the CPU made of its level (D90), which the
	// shader holds to; on a verbose run, what it counted and what it left out are read back.
	if (cache_first < entry_count && OS::get_singleton()->is_stdout_verbose()) {
		PackedInt32Array expected;
		expected.resize(entry_count - cache_first);
		for (uint32_t i = 0; i < batch.cache_builds.size() && i < uint32_t(expected.size()); i++) {
			expected.write[i] = int32_t(batch.cache_builds[i].surface->cluster_lod_levels[batch.cache_builds[i].depth].cut_clusters);
		}
		rd->buffer_get_data_async(args_buffer, callable_mp_static(&RenderRaytracing::_on_lod_cache_counts_read).bind(blas_info_stride, expected), uint32_t(blas_infos_offset + uint64_t(blas_info_stride) * cache_first), uint32_t(blas_info_stride * (entry_count - cache_first)));
		if (persistent_count == 0) { // Else read with the persistent items' counts, above.
			rd->buffer_get_data_async(args_buffer, callable_mp_static(&RenderRaytracing::_on_lod_cache_overflow_read), uint32_t(clas_count_offset + sizeof(uint32_t)), 2 * sizeof(uint32_t));
		}
	}

	// The traversal's check (D91): the items' records, their appended counts and the exhaustive
	// counts, from the start of the arguments to the end of the counts, in one readback.
	if (tree_check) {
		rd->buffer_get_data_async(args_buffer, callable_mp_static(&RenderRaytracing::_on_lod_tree_check_read).bind(entry_count, uint32_t(blas_infos_offset / sizeof(uint32_t)), blas_info_stride, uint32_t(instances_offset / sizeof(uint32_t)), uint32_t(check_offset / sizeof(uint32_t))), 0, uint32_t(check_offset + sizeof(uint32_t) * entry_count));
	}

	// The counts for the report, read back a frame queue later, so requested on the frames just
	// before it; only when the run is verbose.
	const uint64_t report_phase = batch.frames_flushed % 300;
	if (report_phase >= 295 && OS::get_singleton()->is_stdout_verbose()) {
		const uint64_t ticket = ++lod_counts_next_ticket;
		if (cache_first > 0) {
			rd->buffer_get_data_async(args_buffer, callable_mp_static(&RenderRaytracing::_on_lod_counts_read).bind(ticket, cache_first, blas_info_stride), uint32_t(blas_infos_offset), uint32_t(blas_info_stride * cache_first));
		}
		rd->buffer_get_data_async(args_buffer, callable_mp_static(&RenderRaytracing::_on_lod_clas_count_read).bind(ticket), uint32_t(clas_count_offset), sizeof(uint32_t));
	}

	batch.frames_flushed++;
	if (batch.frames_flushed == 1 || batch.frames_flushed % 300 == 0) {
		uint32_t lod0_total = 0;
		for (uint32_t e = 0; e < entry_count; e++) {
			lod0_total += batch.lod0_clusters[e];
		}
		print_verbose(vformat("RT: cluster LOD selection: %d instances (%d deformed, %d past the region ceiling; %d more tracing a shared build, %d the LOD 0 BLAS, %d a cached level), %d cached levels built (%d live, %d KB, %d over the run), %d clusters at LOD 0, %d over every level, %d selected a frame queue ago (%d instances then, %d CLAS %s of %d, counted on the device), %.2f px, threshold %s, %d KB of arguments, %d KB of the LOD table live, %d KB CLAS region (none for persistent items, D109).", cache_first, batch.deformed_count, batch.deformed_routed, batch.shared_count, batch.lod0_count, batch.cached_count, entry_count - cache_first, cluster_lod_cache_live, cluster_lod_cache_bytes / 1024, cluster_lod_cache_builds_total, lod0_total, cluster_total - batch.cache_list_clusters, lod_counts_selected, lod_counts_entries, lod_counts_clas, cluster_lod_deformed_templated ? "instantiated" : "built", clas_bound, lod_frame_pixel_error, String::num_scientific(threshold), args_size / 1024, uint64_t(cluster_lod_table_live_words) * sizeof(uint32_t) / 1024, region_index >= 0 ? uint64_t(clas_bound) * cluster_lod_clas_slot_size / 1024 : 0));
		if (cluster_lod_traversal_enabled) {
			uint64_t queue_capacity = 0;
			for (uint32_t l = 0; l < tree_levels; l++) {
				queue_capacity += batch.tree_level_capacity[l];
			}
			print_verbose(vformat("RT: cluster LOD traversal (D91): %d of %d items walked their tree in %d levels (queues for %d nodes), the others tested every cluster over %d threads each; %d frames checked against the exhaustive test, %d items mismatched.", tree_items, entry_count, tree_levels, queue_capacity, flat_max_clusters, lod_tree_checks, lod_tree_mismatches));
		}
		if (cluster_lod_deformed_templated) {
			print_verbose(vformat("RT: persistent deformed selection (D109): %d of %d selections re-instantiated over the run (the device's counts, a frame queue late), %d hierarchies whose cuts LOD 0 does not bound.", lod_deformed_updates_total, lod_deformed_instances_total, deformed_cut_bound_failures_total));
		}
		if (!lod_select_gpu_us.is_empty()) {
			LocalVector<float> sorted(lod_select_gpu_us);
			sorted.sort();
			print_verbose(vformat("RT: cluster LOD selection, GPU span bracketing it (D91; includes waits on work in flight): p50 %.1f us, p90 %.1f us, max %.1f us over %d frames.", sorted[sorted.size() / 2], sorted[sorted.size() * 9 / 10], sorted[sorted.size() - 1], sorted.size()));
		}
	}
	return true;
}

void RenderRaytracing::_on_lod_deformed_counts_read(const PackedByteArray &p_bytes, uint32_t p_instances) {
	// Called by RenderingDevice on the render thread as it begins a later frame, a frame queue after
	// the selection (D109): the compacted count, the updates (first builds excluded) and their
	// triangles, which the next flush publishes.
	if (singleton == nullptr || p_bytes.size() < int64_t(3 * sizeof(uint32_t))) {
		return;
	}
	const uint32_t *words = reinterpret_cast<const uint32_t *>(p_bytes.ptr());
	singleton->lod_deformed_updates_read += words[1];
	singleton->lod_deformed_triangles_read += words[2];
	singleton->lod_deformed_updates_total += words[1];
	singleton->lod_deformed_instances_total += p_instances;
}

void RenderRaytracing::_on_lod_tree_check_read(const PackedByteArray &p_bytes, uint32_t p_entry_count, uint32_t p_blas_infos_offset, uint32_t p_stride, uint32_t p_instances_offset, uint32_t p_check_offset) {
	// Called by RenderingDevice on the render thread as it begins a later frame. The first word of
	// an item's build info is the count the traversal appended.
	if (singleton == nullptr || p_bytes.size() < int64_t(sizeof(uint32_t)) * (p_check_offset + p_entry_count)) {
		return;
	}
	const uint32_t *words = reinterpret_cast<const uint32_t *>(p_bytes.ptr());
	uint32_t mismatched = 0;
	for (uint32_t e = 0; e < p_entry_count; e++) {
		if ((words[p_instances_offset + e * RT_LOD_INSTANCE_WORDS + 16] & 4u) == 0) {
			continue;
		}
		const uint32_t walked = words[p_blas_infos_offset + e * (p_stride / sizeof(uint32_t))];
		const uint32_t tested = words[p_check_offset + e];
		if (walked != tested) {
			if (mismatched < 4) {
				ERR_PRINT(vformat("RT: cluster LOD traversal (D91): item %d selected %d clusters, the exhaustive test %d.", e, walked, tested));
			}
			mismatched++;
		}
	}
	singleton->lod_tree_checks++;
	singleton->lod_tree_mismatches += mismatched;
}

void RenderRaytracing::_flush_lod_batch(const RenderDataRD *p_render_data, const RTViewportState *p_state) {
	RTLodSelectBatch &batch = lod_batch;
	if (batch.entries.is_empty()) {
		if (batch.lod0_count + batch.cached_count > 0) {
			// Every instance on the LOD 0 BLAS (D89) or a cached level (D90): nothing to record, but
			// the report still says so.
			batch.frames_flushed++;
			if (batch.frames_flushed == 1 || batch.frames_flushed % 300 == 0) {
				print_verbose(vformat("RT: cluster LOD selection: 0 instances (%d on the LOD 0 BLAS, %d on a cached level; %d cached levels live, %d KB, %d built over the run), nothing built.", batch.lod0_count, batch.cached_count, cluster_lod_cache_live, cluster_lod_cache_bytes / 1024, cluster_lod_cache_builds_total));
			}
		}
		return;
	}
	const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
	// The persistent selection's updates the device counted a frame queue ago (D109), published now.
	deformed_cluster_updates += lod_deformed_updates_read;
	deformed_cluster_triangles_updated += uint32_t(MIN(lod_deformed_triangles_read, uint64_t(UINT32_MAX)));
	lod_deformed_updates_read = 0;
	lod_deformed_triangles_read = 0;
	if (_record_lod_batch(p_render_data, p_state)) {
		// Every static instance's BLAS over its selection and every cached level is a build (D97).
		for (uint32_t e = 0; e < batch.entries.size(); e++) {
			frame_extra_builds += batch.entries[e] != nullptr ? 1 : 0;
		}
		frame_extra_builds += batch.cache_builds.size();
		// The deformed items' CLAS live in this flush's region from now (D87), or a persistent item's in
		// its own slots (D109); the counters take what the batch would have counted, as
		// _flush_deformed_cluster_batch does, a persistent item's updates being the device's.
		for (uint32_t e = 0; e < batch.deformed_entries.size(); e++) {
			RTDeformedCacheEntry *entry = batch.deformed_entries[e];
			if (entry == nullptr) {
				continue;
			}
			entry->cluster_built_frame = frames_drawn;
			entry->cluster_lod_last = true;
			entry->cluster_lod_slots_stale = false; // Its slots hold this frame's positions now (D109).
			const uint32_t triangles = entry->ptr ? entry->ptr->geometry.primitive_count : 0;
			switch (entry->cluster_frame_kind) {
				case RTDeformedCacheEntry::CLUSTER_QUEUED_FIRST_BUILD:
				case RTDeformedCacheEntry::CLUSTER_QUEUED_CHANGED:
				case RTDeformedCacheEntry::CLUSTER_QUEUED_EXPIRY:
					if (entry->cluster_templated && entry->cluster_frame_kind != RTDeformedCacheEntry::CLUSTER_QUEUED_FIRST_BUILD) {
						// Instantiated, or kept, on the device (D109): its updates are the device's count,
						// read back and published above.
						break;
					}
					// A cluster BLAS is rebuilt, never refit: every kind is a build (D97).
					deformed_cluster_builds++;
					deformed_cluster_triangles_built += triangles;
					break;
				default:
					break; // Unchanged: on the device's count too, if templated (D109).
			}
			entry->cluster_frame_kind = RTDeformedCacheEntry::CLUSTER_QUEUED_NONE;
		}
	} else {
		// Nothing of this frame's batch may be traced: every queued static instance takes its
		// surface's LOD 0 BLAS, which stands built, a deformed one has no build to trace and
		// leaves this frame's TLAS through the deformed batch's dropped handles (D87), and the
		// path goes off for the rest of the run.
		cluster_lod_select_enabled = false;
		cluster_lod_deformed_enabled = false;
		cluster_lod_stream_enabled = false; // A streamed surface traces its coarsest cut for the rest of the run (D92).
		// Loads in flight are held, never completed, until cleanup_caches frees their buffers. The
		// gathers in flight (D93c) would never be issued: finished and dropped now.
		for (RTLodStreamGatherJob *job : cluster_lod_stream_gathers) {
			WorkerThreadPool::get_singleton()->wait_for_group_task_completion(job->group);
			memdelete(job);
		}
		cluster_lod_stream_gathers.clear();
		uint32_t untraceable = 0;
		for (uint32_t e = 0; e < batch.entries.size(); e++) {
			const uint32_t slot = batch.slots[e];
			if (batch.entries[e] != nullptr) {
				if (slot < blass.size() && batch.entries[e]->ptr) {
					blass[slot] = batch.entries[e]->ptr->blas;
				}
			} else if (batch.deformed_entries[e] != nullptr) {
				RTDeformedCacheEntry *entry = batch.deformed_entries[e];
				entry->blas_built_once = false;
				entry->cluster_frame_kind = RTDeformedCacheEntry::CLUSTER_QUEUED_NONE;
				if (entry->ptr && entry->ptr->blas.is_valid()) {
					deformed_cluster_batch.dropped_handles.push_back(entry->ptr->blas);
					untraceable++;
				}
			}
		}
		for (uint32_t i = 0; i < batch.shared_slots.size(); i++) {
			if (batch.shared_slots[i] < blass.size()) {
				blass[batch.shared_slots[i]] = batch.shared_surfaces[i]->blas; // The instances that shared a build (D89) or a cached level (D90).
			}
		}
		for (const RTLodSelectBatch::CacheBuild &build : batch.cache_builds) {
			if (build.depth < build.surface->cluster_lod_cache.size()) {
				_release_lod_cached_level(build.surface->cluster_lod_cache[build.depth]); // Never built.
			}
		}
		if (batch.args_buffer.is_valid()) {
			RD::get_singleton()->free_rid(batch.args_buffer);
			batch.args_buffer = RID();
			batch.args_capacity = 0;
		}
		WARN_PRINT(vformat("RT: the cluster LOD selection could not be recorded; its %d instances trace LOD 0 (%d deformed ones leave this frame's TLAS) and the selection is off from the next frame.", batch.entries.size(), untraceable));
	}
	batch.entries.clear();
	batch.deformed_entries.clear();
	batch.shared_slots.clear();
	batch.shared_surfaces.clear();
	batch.shared_count = 0;
	batch.lod0_count = 0;
	batch.cached_count = 0;
	batch.cache_builds.clear();
	batch.cache_first = 0;
	batch.cache_list_clusters = 0;
	batch.cache_total_clusters = 0;
	batch.cache_max_clusters = 0;
	batch.slots.clear();
	batch.blas_handles.clear();
	batch.blas_addresses.clear();
	batch.lod0_clusters.clear();
	batch.instance_words.clear();
	batch.storage_buffers.clear();
	batch.source_buffers.clear();
	batch.cluster_total = 0;
	batch.max_clusters = 0;
	batch.tree_count = 0;
	batch.flat_max_clusters = 0;
	batch.tree_level_capacity.clear();
	batch.tree_group_capacity = 0;
	batch.stream_items = 0;
	batch.deformed_count = 0;
	batch.deformed_routed = 0;
	batch.clas_bound = 0;
	batch.clas_total_triangles = 0;
	batch.clas_total_vertices = 0;
	batch.clas_source_buffers.clear();
	batch.clas_index_buffers_seen.clear();
	batch.clas_template_buffers.clear();
	batch.deformed_storage_buffers.clear();
	batch.deformed_max_capacity = 0;
}

void RenderRaytracing::_on_position_fetch_check_read(const PackedByteArray &p_bytes) {
	if (p_bytes.size() < int64_t(sizeof(uint32_t) * RT_POSITION_FETCH_CHECK_WORDS)) {
		return;
	}
	uint32_t words[RT_POSITION_FETCH_CHECK_WORDS];
	memcpy(words, p_bytes.ptr(), sizeof(words));
	float max_difference[3];
	for (uint32_t i = 0; i < 3; i++) {
		memcpy(&max_difference[i], &words[i * 3 + 2], sizeof(float));
	}
	print_line(vformat("RT: position fetch check (D93a), the last 300 frames: float %d hits, %d differ, max %s; compressed %d hits, %d differ, max %s; deformed %d hits, %d differ, max %s.", words[0], words[1], String::num_scientific(max_difference[0]), words[3], words[4], String::num_scientific(max_difference[1]), words[6], words[7], String::num_scientific(max_difference[2])));
	print_line(vformat("RT: cluster geometry check (D93b), the last 300 frames: %d hits read through a blob, %d with a normal or attribute word that differs from the surface's buffers.", words[9], words[10]));
}

void RenderRaytracing::_on_lod_counts_read(const PackedByteArray &p_counts, uint64_t p_ticket, uint32_t p_entry_count, uint32_t p_stride) {
	// Called by RenderingDevice on the render thread as it begins a later frame; the report reads
	// the latest whole batch.
	if (singleton == nullptr || p_ticket < singleton->lod_counts_ticket || p_counts.size() < int64_t(uint64_t(p_stride) * p_entry_count)) {
		return;
	}
	const uint8_t *bytes = p_counts.ptr();
	uint32_t selected = 0;
	for (uint32_t e = 0; e < p_entry_count; e++) {
		uint32_t count;
		memcpy(&count, bytes + uint64_t(e) * p_stride, sizeof(count));
		selected += count;
	}
	singleton->lod_counts_ticket = p_ticket;
	singleton->lod_counts_selected = selected;
	singleton->lod_counts_entries = p_entry_count;
}

void RenderRaytracing::_on_lod_cache_counts_read(const PackedByteArray &p_counts, uint32_t p_stride, const PackedInt32Array &p_expected) {
	if (p_counts.size() < int64_t(uint64_t(p_stride) * p_expected.size())) {
		return;
	}
	uint32_t mismatched = 0;
	for (int i = 0; i < p_expected.size(); i++) {
		uint32_t count;
		memcpy(&count, p_counts.ptr() + uint64_t(i) * p_stride, sizeof(count));
		mismatched += count != uint32_t(p_expected[i]) ? 1 : 0;
	}
	if (mismatched > 0) {
		ERR_PRINT(vformat("RT: %d of %d cached levels selected a count other than their slot's (D90); the shader's level rule and the CPU's count differ.", mismatched, p_expected.size()));
	} else {
		print_verbose(vformat("RT: %d cached levels selected the counts their slots hold (D90).", p_expected.size()));
	}
}

void RenderRaytracing::_on_lod_cache_overflow_read(const PackedByteArray &p_count) {
	if (p_count.size() < int64_t(sizeof(uint32_t))) {
		return;
	}
	uint32_t overflow;
	memcpy(&overflow, p_count.ptr(), sizeof(overflow));
	if (overflow > 0) {
		ERR_PRINT(vformat("RT: the selection took %d clusters past the cached levels' slots (D90); the shader's level rule and the CPU's count differ, and those clusters were left out.", overflow));
	}
	// The word after it (D109): clusters past a persistent deformed item's slots.
	uint32_t persistent_overflow = 0;
	if (p_count.size() >= int64_t(2 * sizeof(uint32_t))) {
		memcpy(&persistent_overflow, p_count.ptr() + sizeof(uint32_t), sizeof(persistent_overflow));
	}
	if (persistent_overflow > 0) {
		ERR_PRINT(vformat("RT: the selection took %d clusters past a persistent deformed item's slots (D109); its cut bound does not hold, and those clusters were left out.", persistent_overflow));
	}
}

void RenderRaytracing::_on_lod_clas_count_read(const PackedByteArray &p_count, uint64_t p_ticket) {
	if (singleton == nullptr || p_ticket < singleton->lod_counts_ticket || p_count.size() < int64_t(sizeof(uint32_t))) {
		return;
	}
	uint32_t count;
	memcpy(&count, p_count.ptr(), sizeof(count));
	singleton->lod_counts_clas = count;
}

// ---------------------------------------------------------------------------
// Procedural geometry processing
// ---------------------------------------------------------------------------

void RenderRaytracing::update_procedural_blas(RTProceduralState *p_state, LocalVector<RID> &r_dirty_blas_list) {
	// Pack AABB data into a byte buffer.
	Vector<uint8_t> aabb_bytes;
	uint32_t aabb_count = 1;

	if (p_state->aabb_data.size() >= 6 && (p_state->aabb_data.size() % 6) == 0) {
		aabb_count = p_state->aabb_data.size() / 6;
		aabb_bytes.resize(p_state->aabb_data.size() * sizeof(float));
		memcpy(aabb_bytes.ptrw(), p_state->aabb_data.ptr(), aabb_bytes.size());
	} else {
		const AABB &a = p_state->culling_aabb;
		float single[6] = {
			(float)a.position.x, (float)a.position.y, (float)a.position.z,
			(float)(a.position.x + a.size.x), (float)(a.position.y + a.size.y), (float)(a.position.z + a.size.z)
		};
		aabb_bytes.resize(sizeof(single));
		memcpy(aabb_bytes.ptrw(), single, sizeof(single));
	}

	uint32_t required_bytes = aabb_bytes.size();
	bool needs_new_blas = false;

	// Grow-only: only recreate the buffer when capacity is exceeded or count changed.
	if (required_bytes > p_state->gpu_buffer_capacity || aabb_count != p_state->aabb_count) {
		if (p_state->blas.is_valid()) {
			RD::get_singleton()->free_rid(p_state->blas);
			p_state->blas = RID();
		}
		if (p_state->gpu_buffer.is_valid()) {
			RD::get_singleton()->free_rid(p_state->gpu_buffer);
		}
		p_state->gpu_buffer = RD::get_singleton()->storage_buffer_create(required_bytes, aabb_bytes,
				0, RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
		p_state->gpu_buffer_capacity = required_bytes;
		p_state->aabb_count = aabb_count;
		needs_new_blas = true;
	} else {
		// Buffer is large enough -- just update contents.
		RD::get_singleton()->buffer_update(p_state->gpu_buffer, 0, required_bytes, aabb_bytes.ptr());
		needs_new_blas = !p_state->blas.is_valid();
	}

	if (needs_new_blas) {
		ERR_FAIL_COND(!p_state->gpu_buffer.is_valid());

		RD::AccelerationStructureGeometry geom;
		geom.type = RD::AccelerationStructureGeometry::TYPE_AABBS;
		geom.geometry.aabbs.buffer = p_state->gpu_buffer;
		geom.geometry.aabbs.count = aabb_count;
		geom.geometry.aabbs.stride = 24; // VkAabbPositionsKHR: two float3 (min, max).
		p_state->blas = RD::get_singleton()->blas_create({ &geom, 1 }, RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
	}

	// BDA for shader access.
	if (p_state->expose_bounds && p_state->gpu_buffer.is_valid()) {
		p_state->gpu_buffer_address = RD::get_singleton()->buffer_get_device_address(p_state->gpu_buffer);
		// Intersection shaders read the bounds through the address; the in-place update above must wait for them.
		RD::get_singleton()->buffer_add_device_address_read_stages(p_state->gpu_buffer, RDD::PIPELINE_STAGE_RAY_TRACING_SHADER_BIT);
	} else {
		p_state->gpu_buffer_address = 0;
	}

	if (p_state->blas.is_valid()) {
		r_dirty_blas_list.push_back(p_state->blas);
	}
}

// ---------------------------------------------------------------------------
// Material processing
// ---------------------------------------------------------------------------

RTMaterialData *RenderRaytracing::process_material(RID p_material_rid, uint16_t p_material_invalidation_counter) {
	// Static default material for invalid/null materials
	static RTMaterialData s_default_mat;
	static bool s_default_mat_initialized = false;
	if (!s_default_mat_initialized) {
		s_default_mat.data.albedo_color[0] = 1.0f;
		s_default_mat.data.albedo_color[1] = 1.0f;
		s_default_mat.data.albedo_color[2] = 1.0f;
		s_default_mat.data.albedo_color[3] = 1.0f;
		s_default_mat.data.emission_color[0] = 0.0f;
		s_default_mat.data.emission_color[1] = 0.0f;
		s_default_mat.data.emission_color[2] = 0.0f;
		s_default_mat.data.emission_strength = 0.0f;
		s_default_mat.data.roughness = 1.0f;
		s_default_mat.data.specular = 0.5f;
		s_default_mat.data.ao_strength = 1.0f;
		s_default_mat.data.uv1_scale[0] = 1.0f;
		s_default_mat.data.uv1_scale[1] = 1.0f;
		s_default_mat.data.uv1_offset[0] = 0.0f;
		s_default_mat.data.uv1_offset[1] = 0.0f;
		s_default_mat.data.alpha_threshold = 0.5f;
		s_default_mat_initialized = true;
	}

	if (!p_material_rid.is_valid()) {
		return &s_default_mat;
	}

	// Cache lookup
	uint32_t mat_idx = get_rid_index(p_material_rid);
	uint32_t mat_version = get_rid_version(p_material_rid);
	RTMaterialCacheEntry *entry = get_material_cache_entry(mat_idx);

	uint32_t current_frame = RSG::rasterizer->get_frame_number();
	bool needs_refresh = !entry->ptr ||
			entry->cached_rid_version != mat_version ||
			entry->cached_counter != p_material_invalidation_counter;

	if (!needs_refresh) {
		entry->last_used_frame = current_frame;
		if (entry->ptr->is_custom_shader) {
			uint32_t shader_id = RendererRD::MaterialStorage::get_singleton()->material_get_shader_id(p_material_rid);
			uint32_t old_sbt = entry->ptr->rt_sbt_offset;
			uint32_t new_sbt = SceneShaderRaytracing::get_singleton()->register_custom_shader(shader_id, p_material_rid);
			entry->ptr->rt_sbt_offset = new_sbt;
			// HG slot change invalidates cached UBO layout / BDA.
			if (old_sbt != new_sbt) {
				needs_refresh = true;
			}
		}
		if (!needs_refresh) {
			return entry->ptr;
		}
	}

	// Cache miss - need to rebuild material
	if (!entry->ptr) {
		entry->ptr = memnew(RTMaterialData);
	}

	RTMaterialData *mat_data = entry->ptr;
	RT_MaterialData &mat = mat_data->data;

	// Initialize defaults
	mat.albedo_color[0] = 1.0f;
	mat.albedo_color[1] = 1.0f;
	mat.albedo_color[2] = 1.0f;
	mat.albedo_color[3] = 1.0f;
	mat.emission_color[0] = 0.0f;
	mat.emission_color[1] = 0.0f;
	mat.emission_color[2] = 0.0f;
	mat.emission_strength = 0.0f;
	mat.metallic = 0.0f;
	mat.roughness = 1.0f;
	mat.specular = 0.5f;
	mat.ao_strength = 1.0f;
	mat.flags = 0;
	mat.albedo_texture_idx = 0;
	mat.normal_texture_idx = 0;
	mat.orm_texture_idx = 0;
	mat.emission_texture_idx = 0;
	mat.uv1_scale[0] = 1.0f;
	mat.uv1_scale[1] = 1.0f;
	mat.uv1_offset[0] = 0.0f;
	mat.uv1_offset[1] = 0.0f;
	mat.normal_map_depth = 1.0f;
	mat.uniform_address = 0;
	mat.alpha_threshold = 0.5f;
	mat.backlight_rgba8 = 0;
	mat.backlight_texture_idx = 0;
	bool omm_custom = false;
	RID omm_custom_alpha_rd; // An opted-in custom shader's hint_alpha texture, with a real bindless slot.

	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();
	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();

	// A parameter is read only when the material's shader declares its uniform: BaseMaterial3D keeps every
	// parameter it was ever given, but the raster evaluates a feature (normal map, emission) only while it
	// is enabled, and the two evaluations of a material must agree.
	const RendererRD::MaterialStorage::ShaderData *shader_data = material_storage->material_get_shader_data(p_material_rid);
	auto shader_declares = [&](const StringName &p_uniform) -> bool {
		return shader_data && shader_data->uniforms.has(p_uniform);
	};

	// Helper lambda to get texture from material parameter
	// p_srgb should be true for color textures (albedo, emission) that need sRGB->linear conversion
	auto get_material_texture = [&](const StringName &p_param, bool p_srgb = false) -> RID {
		Variant tex_var = material_storage->material_get_param(p_material_rid, p_param);
		if (tex_var.get_type() == Variant::OBJECT || tex_var.get_type() == Variant::RID) {
			RID tex_rid = tex_var;
			if (tex_rid.is_valid()) {
				return texture_storage->texture_get_rd_texture(tex_rid, p_srgb);
			}
		}
		return RID();
	};

	// Textures
	// Albedo is a color texture - needs sRGB->linear conversion
	RID albedo_rd = get_material_texture("texture_albedo", true);
	if (albedo_rd.is_valid()) {
		mat.albedo_texture_idx = bindless_block->add_texture(albedo_rd);
	}
	// An opacity micromap's alpha source (D96): only the texture the hit shaders read, not one a full
	// bindless table replaced with the default.
	mat_data->albedo_texture_rd = mat.albedo_texture_idx != 0 ? albedo_rd : RID();

	RID normal_rd = get_material_texture("texture_normal");
	if (normal_rd.is_valid() && shader_declares("texture_normal")) {
		mat.normal_texture_idx = bindless_block->add_texture(normal_rd);
		mat.flags |= RT_MAT_FLAG_HAS_NORMAL_MAP;

		Variant normal_scale_var = material_storage->material_get_param(p_material_rid, "normal_scale");
		if (normal_scale_var.get_type() == Variant::FLOAT) {
			mat.normal_map_depth = normal_scale_var;
		}
	}

	RID orm_rd = get_material_texture("texture_orm");
	if (orm_rd.is_valid()) {
		mat.orm_texture_idx = bindless_block->add_texture(orm_rd);
	} else {
		RID roughness_rd = get_material_texture("texture_roughness");
		if (roughness_rd.is_valid()) {
			mat.orm_texture_idx = bindless_block->add_texture(roughness_rd);
		}
	}

	// Emission is a color texture - needs sRGB->linear conversion
	RID emission_rd = get_material_texture("texture_emission", true);
	if (emission_rd.is_valid() && shader_declares("texture_emission")) {
		mat.emission_texture_idx = bindless_block->add_texture(emission_rd);
		mat.flags |= RT_MAT_FLAG_HAS_EMISSION_TEX;
	}

	// Material properties
	// Colors declared with source_color in Godot shaders are stored in sRGB;
	// material_get_param returns the raw sRGB value, so we convert to linear here.
	Variant albedo_var = material_storage->material_get_param(p_material_rid, "albedo");
	if (albedo_var.get_type() == Variant::COLOR) {
		Color c = ((Color)albedo_var).srgb_to_linear();
		mat.albedo_color[0] = c.r;
		mat.albedo_color[1] = c.g;
		mat.albedo_color[2] = c.b;
		mat.albedo_color[3] = c.a;
		mat_data->rt_sbt_offset = 0;
		mat_data->is_custom_shader = false;
	} else {
		mat_data->is_custom_shader = true;
		uint32_t shader_id = material_storage->material_get_shader_id(p_material_rid);
		mat_data->rt_sbt_offset = SceneShaderRaytracing::get_singleton()->register_custom_shader(shader_id, p_material_rid);

		const SceneShaderRaytracing::CustomShaderEntry *cse =
				SceneShaderRaytracing::get_singleton()->get_custom_shader_entry(mat_data->rt_sbt_offset);
		if (cse && cse->uniform_total_size > 0) {
			Vector<uint8_t> ubo_data;
			ubo_data.resize(cse->uniform_total_size);
			memset(ubo_data.ptrw(), 0, cse->uniform_total_size);

			for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &kv : cse->uniforms) {
				const ShaderLanguage::ShaderNode::Uniform &u = kv.value;
				if (ShaderLanguage::is_sampler_type(u.type)) {
					continue;
				}
				if (u.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
					continue; // Not in the block, and its order is not an offset.
				}
				if (u.order < 0 || u.order >= (int)cse->uniform_offsets.size()) {
					continue;
				}

				uint32_t offset = cse->uniform_offsets[u.order];
				uint32_t size = SceneShaderRaytracing::uniform_std140_size(u);
				if (offset + size > cse->uniform_total_size) {
					continue;
				}

				uint8_t *dst = ubo_data.ptrw() + offset;

				if (u.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL) {
					int32_t idx = material_storage->global_shader_uniform_get_buffer_index(kv.key);
					uint32_t uidx = (idx >= 0) ? (uint32_t)idx : 0;
					memcpy(dst, &uidx, sizeof(uint32_t));
				} else {
					// The same std140 packer as the raster's material buffer, so a
					// source_color default, an unhinted color and an array read the
					// same bytes in the hit shader as in the fragment shader.
					Variant val = material_storage->material_get_param(p_material_rid, kv.key);
					RendererRD::MaterialStorage::fill_std140_uniform(u, val, dst, true);
				}
			}

			RendererRD::TextureStorage *ts = RendererRD::TextureStorage::get_singleton();
			for (int ti = 0; ti < cse->texture_uniforms.size(); ti++) {
				const SceneShaderRaytracing::TextureUniformInfo &tui = cse->texture_uniforms[ti];
				uint32_t bindless_idx = 0;

				if (tui.is_global) {
					RID tex_rid = material_storage->global_shader_uniform_get_texture(tui.name);
					if (tex_rid.is_valid()) {
						RID rd_tex = ts->texture_get_rd_texture(tex_rid, tui.use_color);
						if (rd_tex.is_valid()) {
							bindless_idx = bindless_block->add_texture(rd_tex);
							if (tui.buffer_offset == cse->alpha_texture_buffer_offset && bindless_idx != 0) {
								omm_custom_alpha_rd = rd_tex;
							}
						}
					}
				} else {
					Variant tex_var = material_storage->material_get_param(p_material_rid, tui.name);
					if (tex_var.get_type() == Variant::OBJECT || tex_var.get_type() == Variant::RID) {
						RID tex_rid = tex_var;
						if (tex_rid.is_valid()) {
							RID rd_tex = ts->texture_get_rd_texture(tex_rid, tui.use_color);
							if (rd_tex.is_valid()) {
								bindless_idx = bindless_block->add_texture(rd_tex);
								if (tui.buffer_offset == cse->alpha_texture_buffer_offset && bindless_idx != 0) {
									omm_custom_alpha_rd = rd_tex;
								}
							}
						}
					}
				}

				if (bindless_idx == 0 && tui.hint != ShaderLanguage::ShaderNode::Uniform::HINT_NONE) {
					using Hint = ShaderLanguage::ShaderNode::Uniform::Hint;
					RID default_tex;
					switch (tui.hint) {
						case Hint::HINT_DEFAULT_BLACK:
							default_tex = ts->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
							break;
						case Hint::HINT_DEFAULT_TRANSPARENT:
							default_tex = ts->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_TRANSPARENT);
							break;
						case Hint::HINT_NORMAL:
							default_tex = ts->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_NORMAL);
							break;
						case Hint::HINT_ANISOTROPY:
							default_tex = ts->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_ANISO);
							break;
						default:
							default_tex = ts->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_WHITE);
							break;
					}
					if (default_tex.is_valid()) {
						bindless_idx = bindless_block->add_texture(default_tex);
					}
				}

				if (tui.buffer_offset + 4 <= cse->uniform_total_size) {
					memcpy(ubo_data.ptrw() + tui.buffer_offset, &bindless_idx, 4);
				}
			}

			// If the shader declared hint_alpha, mirror that bindless index into
			// mat.albedo_texture_idx so ray_query_alpha_test() can sample it cheaply.
			if (cse->alpha_texture_buffer_offset != UINT32_MAX &&
					cse->alpha_texture_buffer_offset + 4 <= cse->uniform_total_size) {
				uint32_t alpha_idx = 0;
				memcpy(&alpha_idx, ubo_data.ptr() + cse->alpha_texture_buffer_offset, 4);
				mat.albedo_texture_idx = alpha_idx;
			}
			omm_custom = cse->opacity_micromap;

			// Try the suballoc pool first. Common materials (UBO <= slot size)
			// just buffer_update an existing slot - O(1), no driver allocation,
			// no per-frame storage_buffer_create cost.
			bool used_pool = false;
			if (cse->uniform_total_size <= MAT_UBO_POOL_SLOT_SIZE) {
				if (mat_data->uniform_pool_slot == UINT32_MAX) {
					mat_data->uniform_pool_slot = mat_ubo_pool_allocate();
				}
				if (mat_data->uniform_pool_slot != UINT32_MAX) {
					// Transitioning from a dedicated buffer back into the pool.
					if (mat_data->uniform_buffer.is_valid()) {
						RD::get_singleton()->free_rid(mat_data->uniform_buffer);
						mat_data->uniform_buffer = RID();
					}
					mat_ubo_pool_update(mat_data->uniform_pool_slot, ubo_data.ptr(), cse->uniform_total_size);
					mat.uniform_address = mat_ubo_pool_get_address(mat_data->uniform_pool_slot);
					used_pool = true;
				}
			}

			if (!used_pool) {
				// Oversized or pool exhausted: dedicated per-material buffer.
				// This is the slow path: a per-material storage_buffer_create on
				// every rebuild. Warn once so it's visible in the log; the fix is
				// either to shrink the material's uniform footprint below
				// MAT_UBO_POOL_SLOT_SIZE or to grow the pool slot/capacity.
				const char *reason = (cse->uniform_total_size > MAT_UBO_POOL_SLOT_SIZE)
						? "uniform size exceeds slot"
						: "pool exhausted";
				WARN_PRINT_ONCE(vformat(
						"RT Material UBO falling back to dedicated buffer (%s): "
						"sbt_offset=%d, uniform_total_size=%d, slot_size=%d.",
						String(reason), mat_data->rt_sbt_offset,
						cse->uniform_total_size, MAT_UBO_POOL_SLOT_SIZE));

				if (mat_data->uniform_pool_slot != UINT32_MAX) {
					mat_ubo_pool_release(mat_data->uniform_pool_slot);
					mat_data->uniform_pool_slot = UINT32_MAX;
				}
				if (mat_data->uniform_buffer.is_valid()) {
					RD::get_singleton()->free_rid(mat_data->uniform_buffer);
				}
				mat_data->uniform_buffer = RD::get_singleton()->storage_buffer_create(cse->uniform_total_size, ubo_data, 0, RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT);
				RD::get_singleton()->set_resource_name(mat_data->uniform_buffer, String("RT Material UBO [sbt=") + itos(mat_data->rt_sbt_offset) + "]");
				mat.uniform_address = RD::get_singleton()->buffer_get_device_address(mat_data->uniform_buffer);
			}
		}
	}

	Variant metallic_var = material_storage->material_get_param(p_material_rid, "metallic");
	if (metallic_var.get_type() == Variant::FLOAT) {
		mat.metallic = metallic_var;
	}

	Variant roughness_var = material_storage->material_get_param(p_material_rid, "roughness");
	if (roughness_var.get_type() == Variant::FLOAT) {
		mat.roughness = roughness_var;
	}

	Variant specular_var = material_storage->material_get_param(p_material_rid, "specular");
	if (specular_var.get_type() == Variant::FLOAT) {
		mat.specular = specular_var;
	}

	Variant emission_var = shader_declares("emission") ? material_storage->material_get_param(p_material_rid, "emission") : Variant();
	if (emission_var.get_type() == Variant::COLOR) {
		Color c = ((Color)emission_var).srgb_to_linear();
		mat.emission_color[0] = c.r;
		mat.emission_color[1] = c.g;
		mat.emission_color[2] = c.b;
	}

	Variant emission_energy_var = material_storage->material_get_param(p_material_rid, "emission_energy");
	if (emission_energy_var.get_type() == Variant::FLOAT) {
		mat.emission_strength = emission_energy_var;
	}

	// The operator is not a uniform; BaseMaterial3D publishes it as a parameter beside the ones its shader bakes it into.
	Variant emission_operator_var = material_storage->material_get_param(p_material_rid, "emission_operator");
	if (emission_operator_var.get_type() == Variant::INT && (int)emission_operator_var == 1) { // 1 = BaseMaterial3D::EMISSION_OP_MULTIPLY.
		mat.flags |= RT_MAT_FLAG_EMISSION_MULTIPLY;
	}

	// An emissive mesh light (D50): a constant, untextured, non-zero emission the trace can sample
	// without evaluating the material. A textured emission varies over the surface and a custom
	// shader's is unknown until it runs, so neither is listed; nor is the Multiply operator
	// without a texture, whose hit-side emission is the colour times a black default.
	if (!mat_data->is_custom_shader && mat.emission_strength > 0.0f && !(mat.flags & (RT_MAT_FLAG_HAS_EMISSION_TEX | RT_MAT_FLAG_EMISSION_MULTIPLY)) &&
			(mat.emission_color[0] > 0.0f || mat.emission_color[1] > 0.0f || mat.emission_color[2] > 0.0f)) {
		mat.flags |= RT_MAT_FLAG_EMISSIVE_CONSTANT;
	}

	// UV1 scale and offset (vec3 in Godot, we only use xy).
	Variant uv1_scale_var = material_storage->material_get_param(p_material_rid, "uv1_scale");
	if (uv1_scale_var.get_type() == Variant::VECTOR3) {
		Vector3 s = uv1_scale_var;
		mat.uv1_scale[0] = s.x;
		mat.uv1_scale[1] = s.y;
	}

	Variant uv1_offset_var = material_storage->material_get_param(p_material_rid, "uv1_offset");
	if (uv1_offset_var.get_type() == Variant::VECTOR3) {
		Vector3 o = uv1_offset_var;
		mat.uv1_offset[0] = o.x;
		mat.uv1_offset[1] = o.y;
	}

	// Back lighting (D103): BaseMaterial3D declares backlight only while the feature is enabled, and its
	// fragment writes BACKLIGHT = backlight.rgb + texture_backlight.rgb, the colour a source_color (kept
	// sRGB here, decoded in the hit shader) and the texture hinted black but not as a colour, so read
	// linear. The raster's light pass weighs it by the diffuse albedo; the trace takes the same product
	// as the albedo of a thin diffuse-transmission lobe. A custom shader's BACKLIGHT is not carried.
	if (!mat_data->is_custom_shader && shader_declares("backlight")) {
		Variant backlight_var = material_storage->material_get_param(p_material_rid, "backlight");
		Color backlight = backlight_var.get_type() == Variant::COLOR ? (Color)backlight_var : Color(0, 0, 0);
		RID backlight_rd = get_material_texture("texture_backlight");
		if (backlight_rd.is_valid()) {
			mat.backlight_texture_idx = bindless_block->add_texture(backlight_rd);
		}
		if (mat.backlight_texture_idx != 0 || backlight.r > 0.0f || backlight.g > 0.0f || backlight.b > 0.0f) {
			mat.backlight_rgba8 = backlight.clamp().to_abgr32();
			mat.flags |= RT_MAT_FLAG_DIFFUSE_TRANSMISSION;
		}
	}

	// The alpha test's cutoff, from the material's own uniform when its shader declares one
	// (BaseMaterial3D does in alpha scissor mode); 0.5 otherwise.
	if (shader_declares("alpha_scissor_threshold")) {
		Variant threshold_var = material_storage->material_get_param(p_material_rid, "alpha_scissor_threshold");
		if (threshold_var.get_type() == Variant::NIL) {
			// A ShaderMaterial leaves a uniform at its default out of its parameters; the raster packs the default.
			threshold_var = shader_data->get_default_parameter("alpha_scissor_threshold");
		}
		if (threshold_var.get_type() == Variant::FLOAT) {
			mat.alpha_threshold = threshold_var;
		}
	}

	// A custom shader with render_mode opacity_micromap (D96) traces the standard alpha test over its
	// hint_alpha texture at the raw UV, and its micromap is baked from the same row.
	mat_data->omm_custom = omm_custom;
	if (omm_custom) {
		mat.uv1_scale[0] = 1.0f;
		mat.uv1_scale[1] = 1.0f;
		mat.uv1_offset[0] = 0.0f;
		mat.uv1_offset[1] = 0.0f;
		mat.albedo_color[3] = 1.0f;
		mat_data->albedo_texture_rd = omm_custom_alpha_rd;
	}

	// Point filtering: check if material requests nearest filtering (e.g. pixel art).
	// BaseMaterial3D exposes this as "texture_filter" int param (0=nearest, 1=linear, etc.).
	Variant filter_var = material_storage->material_get_param(p_material_rid, "texture_filter");
	if (filter_var.get_type() == Variant::INT) {
		int filter_mode = filter_var;
		// 0 = TEXTURE_FILTER_NEAREST, 2 = TEXTURE_FILTER_NEAREST_WITH_MIPMAPS,
		// 4 = TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC
		if (filter_mode == 0 || filter_mode == 2 || filter_mode == 4) {
			mat.flags |= RT_MAT_FLAG_POINT_FILTER;
		}
	}

	// Update cache entry
	entry->cached_counter = p_material_invalidation_counter;
	entry->cached_rid_version = mat_version;
	entry->last_used_frame = current_frame;

	return mat_data;
}

// ---------------------------------------------------------------------------
// Acceleration structure building
// ---------------------------------------------------------------------------

bool RenderRaytracing::build_acceleration_structures(RTViewportState *p_state, const LocalVector<RID> &p_dirty_blas_list, const LocalVector<RID> &p_dirty_blas_update_list) {
	RENDER_TIMESTAMP("BLAS Build");

	for (const RID &blas_rid : p_dirty_blas_list) {
		if (blas_rid.is_valid()) {
			RD::get_singleton()->blas_build(blas_rid);
		}
	}

	for (const RID &blas_rid : p_dirty_blas_update_list) {
		if (blas_rid.is_valid()) {
			RD::get_singleton()->blas_update(blas_rid);
		}
	}

	RENDER_TIMESTAMP("TLAS Build");

	if (ptlas_slots_enabled) {
		// Slot-resident instances (D94b): the writes, the builds, the rows' scatter. False when the
		// structure could not be brought up to date, and the frame does not trace.
		return _commit_slots(p_state);
	}

	uint32_t needed = MAX(blass.size(), (uint32_t)1);
	if (!p_state->tlas.is_valid() || needed > p_state->tlas_max_instances) {
		if (p_state->tlas.is_valid()) {
			RD::get_singleton()->free_rid(p_state->tlas);
		}
		p_state->tlas_max_instances = needed * 2;
		p_state->tlas = RD::get_singleton()->tlas_create(p_state->tlas_max_instances, RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
		RD::get_singleton()->set_resource_name(p_state->tlas, "RT TLAS");
	}

	LocalVector<RD::AccelerationStructureInstance> instances;
	instances.resize(blass.size());
	for (uint32_t i = 0; i < blass.size(); i++) {
		RD::AccelerationStructureInstance &inst = instances[i];
		inst.id = i;
		inst.transform = blas_transforms[i];
		inst.transform.origin -= p_state->rt_origin;
		inst.blas = blass[i];
		inst.flags = BitField<RD::AccelerationStructureInstanceFlagBits>(instance_flags[i]);
		inst.mask = (i < instance_masks.size()) ? instance_masks[i] : (RT_INSTANCE_MASK_GEOMETRY | RT_INSTANCE_MASK_SHADOW);
		uint32_t sbt_off = (i < sbt_offsets.size()) ? sbt_offsets[i] : 0;
		inst.hit_sbt_range = RD::HitShaderBindingTableRange((1ULL << 32) | uint64_t(sbt_off));
	}

	RD::get_singleton()->tlas_build(p_state->tlas, instances);
	return true;
}

void RenderRaytracing::finalize_buffers(RTViewportState *p_state) {
	// Grow-only uploads. Callers must not free these in prepare_frame().
	auto update_or_grow = [](RID &p_buffer, uint32_t &p_capacity, const void *p_data, uint32_t p_size) {
		if (p_size == 0) {
			return;
		}
		if (p_size > p_capacity) {
			if (p_buffer.is_valid()) {
				RD::get_singleton()->free_rid(p_buffer);
			}
			p_capacity = p_size;
			Vector<uint8_t> init;
			init.resize(p_size);
			memcpy(init.ptrw(), p_data, p_size);
			p_buffer = RD::get_singleton()->storage_buffer_create(p_size, init);
		} else {
			RD::get_singleton()->buffer_update(p_buffer, 0, p_size, p_data);
		}
	};

	if (ptlas_slots_enabled) {
		// The per-instance rows are the slots' (D94b), scattered by the commit; the motion
		// transforms stay the frame's compact list.
		update_or_grow(p_state->motion_transform_buffer, p_state->motion_transform_buffer_capacity,
				motion_transforms.ptr(), motion_transforms.size() * sizeof(RT_InstanceMotionData));
		return;
	}

	update_or_grow(p_state->geometry_buffer, p_state->geometry_buffer_capacity,
			geometry_data.ptr(), geometry_data.size() * sizeof(RT_GeometryData));
	update_or_grow(p_state->material_buffer, p_state->material_buffer_capacity,
			material_data.ptr(), material_data.size() * sizeof(RT_MaterialData));
	update_or_grow(p_state->motion_index_buffer, p_state->motion_index_buffer_capacity,
			motion_indices.ptr(), motion_indices.size() * sizeof(int32_t));
	update_or_grow(p_state->motion_transform_buffer, p_state->motion_transform_buffer_capacity,
			motion_transforms.ptr(), motion_transforms.size() * sizeof(RT_InstanceMotionData));
}

// ---------------------------------------------------------------------------
// Merged MultiMesh BLAS builder
// ---------------------------------------------------------------------------

// A MultiMesh-surface's merge, planned without side effects (D93d), so the TLAS walk (which
// uploads a streamed surface's arrays for a bake) and _build_merged_mm_blas() cannot disagree.
// False when the surface does not merge: no instance buffer, a compressed format (positions are
// UNORM16x4, not float3), no triangles, or more merged triangles than the cap. r_bakes when the
// call would bake: on the frame's first touch, with no entry yet, a structure or surface change,
// moved transforms or a BLAS never built. r_first, when asked, when the merge has no entry yet: a
// first build the build budget (D97) may hold back while the expanded path stands in.
bool RenderRaytracing::_plan_merged_mm(RID p_mm_rid, RID p_mm_gpu_buffer, void *p_mesh_surface, uint32_t p_mm_count, uint32_t p_surface_index, uint32_t p_surface_counter, bool &r_bakes, bool *r_first) {
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();
	r_bakes = false;
	if (r_first) {
		*r_first = false;
	}
	if (p_mm_gpu_buffer.is_null() || (mesh_storage->mesh_surface_get_format(p_mesh_surface) & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES)) {
		return false;
	}
	const uint32_t vertex_count = mesh_storage->mesh_surface_get_rt_vertex_count(p_mesh_surface);
	const uint32_t index_count = mesh_storage->mesh_surface_get_rt_index_count(p_mesh_surface);
	const uint32_t prim_count = index_count > 0 ? index_count / 3 : vertex_count / 3;
	static const uint32_t MM_MERGED_BLAS_MAX_TRIANGLES = (uint32_t)GLOBAL_GET("rendering/pathtracing/multimesh_merged_blas_max_triangles");
	if (prim_count == 0 || vertex_count == 0 || (uint64_t)p_mm_count * prim_count > MM_MERGED_BLAS_MAX_TRIANGLES) {
		return false;
	}
	const uint32_t li = p_mm_rid.get_local_index();
	const RTMergedMMEntry *entry = nullptr;
	if (li < mm_handles.size() && mm_handles[li].mm_validator == static_cast<uint32_t>(p_mm_rid.get_id() >> 32) && p_surface_index < mm_handles[li].per_surface.size()) {
		entry = merged_mm_pool.get_or_null(mm_handles[li].per_surface[p_surface_index]);
	}
	if (!entry) {
		r_bakes = true;
		if (r_first) {
			*r_first = true;
		}
		return true;
	}
	const bool first_touch = entry->last_used_frame != RSG::rasterizer->get_frame_number();
	const bool built = entry->blas_built_once && entry->last_mm_count == p_mm_count && entry->last_surface_counter == p_surface_counter;
	r_bakes = first_touch && (!built || entry->cached_mm_last_change != mesh_storage->multimesh_get_last_change(p_mm_rid));
	return true; // An existing entry's rebake is forced (D97): r_first stays false.
}

bool RenderRaytracing::_build_merged_mm_blas(
		RID p_mm_rid,
		RID p_mm_gpu_buffer,
		void *p_mesh_surface,
		uint32_t p_mm_count,
		uint32_t p_surface_index,
		uint32_t p_surface_counter,
		const AABB &p_mm_aabb,
		const RTSurfaceData *p_source,
		RD::ComputeListID p_compute_list,
		LocalVector<RID> &r_dirty_blas_list,
		LocalVector<RID> &r_dirty_blas_update_list,
		RTSurfaceData *r_surf_data) {
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();

	// Whether the surface merges, and whether this call bakes (D93d): the plan the walk took.
	bool needs_rebake = false;
	if (!_plan_merged_mm(p_mm_rid, p_mm_gpu_buffer, p_mesh_surface, p_mm_count, p_surface_index, p_surface_counter, needs_rebake)) {
		return false;
	}

	// The surface's own counts and buffers, switched together: a surface whose geometry streams has
	// its fallback level in its buffers, and bakes from the upload of its arrays (D93d), which only a
	// bake needs; without one, the expanded path traces it through its blobs that frame.
	uint32_t vertex_count = mesh_storage->mesh_surface_get_rt_vertex_count(p_mesh_surface);
	uint32_t index_count = mesh_storage->mesh_surface_get_rt_index_count(p_mesh_surface);
	RID index_buffer = mesh_storage->mesh_surface_get_index_buffer(p_mesh_surface, 0);
	RID source_vertex_buffer = mesh_storage->mesh_surface_get_vertex_buffer(p_mesh_surface);
	RID source_attribute_buffer = mesh_storage->mesh_surface_get_attribute_buffer(p_mesh_surface);
	const bool geometry_streamed = mesh_storage->mesh_surface_is_geometry_streamed(p_mesh_surface);
	if (geometry_streamed) {
		const bool uploaded = p_source && p_source->materialized_vertex_buffer.is_valid();
		index_buffer = uploaded ? p_source->materialized_index_buffer : RID();
		source_vertex_buffer = uploaded ? p_source->materialized_vertex_buffer : RID();
		source_attribute_buffer = uploaded ? p_source->materialized_attribute_buffer : RID();
	}
	if (geometry_streamed && needs_rebake && source_vertex_buffer.is_null()) {
		return false; // No upload for the bake, before anything changes (D93d): the expanded path this frame.
	}
	bool indexed = (index_buffer.is_valid() || geometry_streamed) && index_count > 0;
	uint32_t prim_count = indexed ? (index_count / 3) : (vertex_count / 3);

	// Compressed meshes, an empty surface and the merged triangle cap are the plan's (_plan_merged_mm).
	uint64_t surface_format = mesh_storage->mesh_surface_get_format(p_mesh_surface);

	// Side table -> pool: dense direct index keyed by RID local_index, with a
	// validator stamp to detect RID local-index recycling for a different MM.
	const uint32_t li = p_mm_rid.get_local_index();
	const uint32_t mm_validator = static_cast<uint32_t>(p_mm_rid.get_id() >> 32);
	if (li >= mm_handles.size()) {
		mm_handles.resize(li + 1);
	}
	MMSurfaceHandles &mm_entry = mm_handles[li];
	if (mm_entry.mm_validator != mm_validator) {
		// Local index has been recycled (or first-ever access). Release any slots
		// the previous owner left behind so we don't leak until TTL.
		RD *rd_local = RD::get_singleton();
		for (uint32_t s = 0; s < mm_entry.per_surface.size(); s++) {
			RID &h = mm_entry.per_surface[s];
			RTMergedMMEntry *old = merged_mm_pool.get_or_null(h);
			if (!old) {
				h = RID();
				continue;
			}
			if (old->blas.is_valid()) {
				rd_local->free_rid(old->blas);
			}
			if (old->merged_vtx_buffer.is_valid()) {
				rd_local->free_rid(old->merged_vtx_buffer);
			}
			if (old->merged_attr_buffer.is_valid()) {
				rd_local->free_rid(old->merged_attr_buffer);
			}
			if (old->replicated_idx_buffer.is_valid()) {
				rd_local->free_rid(old->replicated_idx_buffer);
			}
			_omm_release_merged(*old);
			merged_mm_pool.free(h);
			h = RID();
		}
		mm_entry.per_surface.clear();
		mm_entry.mm_validator = mm_validator;
	}

	RTMergedMMEntry *entry_ptr = _access_merged_mm_slot(mm_entry.surface_handle(p_surface_index));
	ERR_FAIL_NULL_V(entry_ptr, false);
	RTMergedMMEntry &entry = *entry_ptr;
	const uint32_t cache_key = (li << 8) | (p_surface_index & 0xFF); // Used only for resource naming below.

	const uint32_t current_frame = RSG::rasterizer->get_frame_number();

	// First viewport this frame does the merge dispatch + BLAS refit; later
	// viewports referencing the same MultiMesh-surface reuse the result.
	// `last_used_frame` doubles as the per-frame guard, so capture the gate
	// BEFORE bumping it.
	const bool first_touch_this_frame = entry.last_used_frame != current_frame;
	entry.last_used_frame = current_frame;

	// Detect whether instance transforms moved since our last bake. Without this
	// we'd re-merge + refit every frame even for fully-static MultiMeshes, which
	// is significant wasted compute for high-instance-count crowds/foliage.
	const uint64_t mm_last_change = mesh_storage->multimesh_get_last_change(p_mm_rid);
	const bool transforms_changed = (entry.cached_mm_last_change != mm_last_change);

	bool structure_changed = (entry.last_mm_count != p_mm_count ||
			entry.last_surface_counter != p_surface_counter);

	entry.indexed = indexed;

	if (structure_changed) {
		RD *rd = RD::get_singleton();
		if (entry.blas.is_valid()) {
			rd->free_rid(entry.blas);
			entry.blas = RID();
		}
		entry.blas_built_once = false;
		entry.last_mm_count = p_mm_count;
		entry.last_surface_counter = p_surface_counter;
	}

	bool has_normal = surface_format & RSE::ARRAY_FORMAT_NORMAL;
	bool has_tangent = surface_format & RSE::ARRAY_FORMAT_TANGENT;
	bool has_tbn = has_normal;

	// Layout of uncompressed vertex buffer: [float3 positions × V] + [packed TBN × V].
	// normal_stride = 8 when both normal+tangent present (two uint16x2 packed), 4 with normal only.
	uint32_t tbn_stride = 0;
	if (has_normal && has_tangent) {
		tbn_stride = 8; // 2 × uint32 (normal oct, tangent oct+sign)
	} else if (has_normal) {
		tbn_stride = 4; // 1 × uint32 (normal oct only)
	}
	// Byte offset of the TBN block in the source vertex buffer.
	uint32_t src_tbn_byte_offset = vertex_count * 12; // after all float3 positions

	// Merged vertex buffer: [float3 pos × N*V] + [packed TBN × N*V] (if TBN present).
	uint32_t merged_vtx_bytes = p_mm_count * vertex_count * 12 + (has_tbn ? p_mm_count * vertex_count * tbn_stride : 0);

	// Attribute buffer: attribute_stride bytes × V, replicated N times.
	RTSurfaceData meta_sd;
	_fill_surface_geometry_data(p_mesh_surface, false, &meta_sd);
	uint32_t attrib_stride = meta_sd.geometry.attribute_stride;
	bool has_attr = attrib_stride > 0;
	const uint32_t MIN_ATTR_BYTES = 16;
	uint32_t merged_attr_bytes = has_attr ? (p_mm_count * vertex_count * attrib_stride) : MIN_ATTR_BYTES;

	RD *rd = RD::get_singleton();
	BitField<RD::BufferCreationBits> gpu_buf_flags =
			RD::BUFFER_CREATION_AS_STORAGE_BIT |
			RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT |
			RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT;

	// --- Grow / allocate merged vertex buffer ---
	if (!entry.merged_vtx_buffer.is_valid() || entry.vtx_capacity_bytes < merged_vtx_bytes) {
		if (entry.merged_vtx_buffer.is_valid()) {
			rd->free_rid(entry.merged_vtx_buffer);
			entry.merged_vtx_buffer = RID();
		}
		entry.vtx_capacity_bytes = merged_vtx_bytes;
		entry.merged_vtx_buffer = rd->vertex_buffer_create(merged_vtx_bytes, {}, gpu_buf_flags);
		ERR_FAIL_COND_V(!entry.merged_vtx_buffer.is_valid(), false);
		rd->set_resource_name(entry.merged_vtx_buffer, "RT MM merged vtx [" + itos(cache_key) + "]");
		entry.blas_built_once = false;
	}

	// --- Grow / allocate merged attribute buffer ---
	if (!entry.merged_attr_buffer.is_valid() || entry.attr_capacity_bytes < merged_attr_bytes) {
		if (entry.merged_attr_buffer.is_valid()) {
			rd->free_rid(entry.merged_attr_buffer);
			entry.merged_attr_buffer = RID();
		}
		entry.attr_capacity_bytes = merged_attr_bytes;
		entry.merged_attr_buffer = rd->storage_buffer_create(merged_attr_bytes, {}, 0, gpu_buf_flags);
		ERR_FAIL_COND_V(!entry.merged_attr_buffer.is_valid(), false);
		rd->set_resource_name(entry.merged_attr_buffer, "RT MM merged attr [" + itos(cache_key) + "]");
	}

	// --- Grow / allocate replicated index buffer ---
	if (indexed) {
		uint32_t needed_idx = p_mm_count * index_count;
		if (!entry.replicated_idx_buffer.is_valid() || entry.idx_capacity < needed_idx) {
			if (entry.replicated_idx_buffer.is_valid()) {
				rd->free_rid(entry.replicated_idx_buffer);
				entry.replicated_idx_buffer = RID();
			}
			entry.idx_capacity = needed_idx;
			entry.replicated_idx_buffer = rd->index_buffer_create(
					needed_idx, RD::INDEX_BUFFER_FORMAT_UINT32, {}, false, gpu_buf_flags);
			ERR_FAIL_COND_V(!entry.replicated_idx_buffer.is_valid(), false);
			rd->set_resource_name(entry.replicated_idx_buffer, "RT MM replicated idx [" + itos(cache_key) + "]");
			entry.blas_built_once = false;
		}
	}

	// The plan's bake (_plan_merged_mm) is this one, from the state before the changes above; were they
	// to differ, the walk would have uploaded for the wrong frame.
	ERR_FAIL_COND_V_MSG(needs_rebake != (first_touch_this_frame && (transforms_changed || !entry.blas_built_once)), false, "RT: a merged MultiMesh's plan and its bake disagree (D93d).");

	RID src_attr_buf;
	if (has_attr) {
		src_attr_buf = source_attribute_buffer;
		ERR_FAIL_COND_V(needs_rebake && !src_attr_buf.is_valid(), false);
	} else {
		src_attr_buf = entry.merged_attr_buffer;
	}

	MergeShader::Mode merge_mode = indexed ? MergeShader::MODE_INDEXED : MergeShader::MODE_NON_INDEXED;

	// Pre-validate source vertex buffer before opening the compute list. A streamed surface's merge
	// reads it only when it bakes (D93d).
	RID vtx_buf = source_vertex_buffer;
	ERR_FAIL_COND_V(needs_rebake && !vtx_buf.is_valid(), false);
	uint64_t vtx_bda = vtx_buf.is_valid() ? rd->buffer_get_device_address(vtx_buf) : 0;
	// The merge reads the source mesh through addresses; an in-place update must wait for an earlier frame's merge.
	if (vtx_buf.is_valid()) {
		rd->buffer_add_device_address_read_stages(vtx_buf, RDD::PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	}

	// --- Single merged dispatch: bake vertices + TBN + attributes + (optional) indices ---
	if (needs_rebake) {
		Vector<RD::Uniform> uniforms;
		{
			auto push_buf = [&](uint32_t binding, RID buf) {
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = binding;
				u.append_id(buf);
				uniforms.push_back(u);
			};
			push_buf(0, entry.merged_vtx_buffer);
			push_buf(1, p_mm_gpu_buffer);
			push_buf(2, entry.merged_attr_buffer);
			push_buf(3, src_attr_buf);
			if (indexed) {
				push_buf(4, entry.replicated_idx_buffer);
			}
		}
		RID merge_uniform_set = rd->uniform_set_create(uniforms, mm_merge_shader.version_shader[merge_mode], 0, /*p_linear_pool=*/true);
		ERR_FAIL_COND_V(!merge_uniform_set.is_valid(), false);
		uint32_t mm_stride = mesh_storage->multimesh_get_stride(p_mm_rid);
		uint32_t mm_cur_offset = mesh_storage->multimesh_get_current_instance_offset(p_mm_rid);
		uint32_t tbn_stride_words = tbn_stride / 4;
		// In the merged vertex buffer the TBN block starts after all N*V float3 positions.
		uint32_t dst_tbn_base_words = p_mm_count * vertex_count * 3;

		struct MergePC {
			uint32_t src_vtx_lo, src_vtx_hi;
			// MODE_INDEXED only -- present in struct (uploaded only when indexed):
			uint32_t src_idx_lo, src_idx_hi;
			uint32_t index_count, src_is_16bit;
			// Common tail:
			uint32_t vertex_count, instance_count;
			uint32_t pos_stride_words;
			uint32_t src_tbn_base_words;
			uint32_t src_tbn_stride_words;
			uint32_t dst_tbn_base_words;
			uint32_t mm_stride, mm_offset;
			uint32_t has_tbn;
			uint32_t attr_stride_words;
		} pc;

		pc.src_vtx_lo = uint32_t(vtx_bda);
		pc.src_vtx_hi = uint32_t(vtx_bda >> 32);

		if (indexed) {
			uint64_t src_idx_bda = rd->buffer_get_device_address(index_buffer);
			rd->buffer_add_device_address_read_stages(index_buffer, RDD::PIPELINE_STAGE_COMPUTE_SHADER_BIT);
			pc.src_idx_lo = uint32_t(src_idx_bda);
			pc.src_idx_hi = uint32_t(src_idx_bda >> 32);
			pc.index_count = index_count;
			pc.src_is_16bit = (vertex_count <= 65536) ? 1u : 0u;
		}

		pc.vertex_count = vertex_count;
		pc.instance_count = p_mm_count;
		pc.pos_stride_words = 3; // always float3 (uncompressed check at top)
		pc.src_tbn_base_words = src_tbn_byte_offset / 4;
		pc.src_tbn_stride_words = tbn_stride_words;
		pc.dst_tbn_base_words = dst_tbn_base_words;
		pc.mm_stride = mm_stride;
		pc.mm_offset = mm_cur_offset;
		pc.has_tbn = has_tbn ? 1u : 0u;
		pc.attr_stride_words = has_attr ? (attrib_stride / 4) : 0u;

		MergeShader::Mode mode = indexed ? MergeShader::MODE_INDEXED : MergeShader::MODE_NON_INDEXED;
		rd->compute_list_bind_compute_pipeline(p_compute_list, mm_merge_shader.pipeline[mode]);
		rd->compute_list_bind_uniform_set(p_compute_list, merge_uniform_set, 0);
		if (indexed) {
			rd->compute_list_set_push_constant(p_compute_list, &pc, sizeof(MergePC));
		} else {
			// Re-pack the non-indexed PC so that the common tail follows
			// src_vtx_lo/hi without the index gap.
			struct MergePCNonIndexed {
				uint32_t src_vtx_lo, src_vtx_hi;
				uint32_t vertex_count, instance_count;
				uint32_t pos_stride_words;
				uint32_t src_tbn_base_words;
				uint32_t src_tbn_stride_words;
				uint32_t dst_tbn_base_words;
				uint32_t mm_stride, mm_offset;
				uint32_t has_tbn;
				uint32_t attr_stride_words;
			} pc_ni;
			pc_ni.src_vtx_lo = pc.src_vtx_lo;
			pc_ni.src_vtx_hi = pc.src_vtx_hi;
			pc_ni.vertex_count = pc.vertex_count;
			pc_ni.instance_count = pc.instance_count;
			pc_ni.pos_stride_words = pc.pos_stride_words;
			pc_ni.src_tbn_base_words = pc.src_tbn_base_words;
			pc_ni.src_tbn_stride_words = pc.src_tbn_stride_words;
			pc_ni.dst_tbn_base_words = pc.dst_tbn_base_words;
			pc_ni.mm_stride = pc.mm_stride;
			pc_ni.mm_offset = pc.mm_offset;
			pc_ni.has_tbn = pc.has_tbn;
			pc_ni.attr_stride_words = pc.attr_stride_words;
			rd->compute_list_set_push_constant(p_compute_list, &pc_ni, sizeof(MergePCNonIndexed));
		}

		// Single thread count: every thread processes one vertex (idx < N*V)
		// and -- for MODE_INDEXED -- one output index (idx < N*I) using disjoint
		// destination buffers, so no in-shader barrier is required.
		uint32_t thread_count = p_mm_count * vertex_count;
		if (indexed) {
			thread_count = MAX(thread_count, p_mm_count * index_count);
		}
		rd->compute_list_dispatch_threads(p_compute_list, thread_count, 1, 1);

		// Drop the RID now that the dispatch is recorded; the underlying Vulkan
		// descriptor set lives in the linear pool until frame end.
		rd->free_rid(merge_uniform_set);
	}

	// --- Build or refit the merged BLAS (uses merged_vtx_buffer for positions) ---
	if (!entry.blas.is_valid()) {
		RD::AccelerationStructureGeometry as_geom;
		as_geom.type = RD::AccelerationStructureGeometry::TYPE_TRIANGLES;
		as_geom.geometry.triangles.vertex_buffer = entry.merged_vtx_buffer;
		as_geom.geometry.triangles.vertex_stride = 12; // float3, positions section only
		as_geom.geometry.triangles.vertex_count = p_mm_count * vertex_count;
		as_geom.geometry.triangles.vertex_format = RD::DATA_FORMAT_R32G32B32_SFLOAT;

		if (indexed) {
			as_geom.geometry.triangles.index_buffer = entry.replicated_idx_buffer;
			as_geom.geometry.triangles.index_count = p_mm_count * index_count;
		}
		_omm_merged_geometry(entry, p_mm_count, prim_count, as_geom); // Its micromap, when it carries one (D96).

		BitField<RD::AccelerationStructureFlagBits> as_flags =
				RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT |
				RD::ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT;
		entry.blas = rd->blas_create({ &as_geom, 1 }, as_flags);
		ERR_FAIL_COND_V(!entry.blas.is_valid(), false);
		rd->set_resource_name(entry.blas, "RT MM merged BLAS [" + itos(cache_key) + "]");
	}

	if (needs_rebake) {
		if (!entry.blas_built_once) {
			r_dirty_blas_list.push_back(entry.blas);
			entry.blas_built_once = true;
		} else {
			r_dirty_blas_update_list.push_back(entry.blas);
		}
		entry.cached_mm_last_change = mm_last_change;
	}

	if (!entry.blas_built_once) {
		return false; // Never built (an earlier failure this frame): the expanded path, not an unbuilt BLAS.
	}

	// --- Populate r_surf_data from the metadata already computed above, then override merged buffer addresses.
	*r_surf_data = meta_sd;
	r_surf_data->blas = entry.blas;
	r_surf_data->is_compressed = false;
	r_surf_data->aabb_transform = Transform3D();

	RT_GeometryData &geom = r_surf_data->geometry;

	// Point vertex address at merged buffer (positions + TBN all baked world-space).
	geom.vertex_buffer_address = rd->buffer_get_device_address(entry.merged_vtx_buffer);
	geom.vertex_count = p_mm_count * vertex_count;
	geom.position_stride = 12; // float3, uncompressed
	geom.flags &= ~RT_GEOM_FLAG_COMPRESSED;
	// The merged buffer spans every instance, so its box is the MultiMesh's own (the emissive gather's area proxy, D50),
	// read in Phase 1 because fetching it may flush dirty multimesh buffers, which the open compute list forbids.
	geom.aabb_size_x = p_mm_aabb.size.x;
	geom.aabb_size_y = p_mm_aabb.size.y;
	geom.aabb_size_z = p_mm_aabb.size.z;
	geom.aabb_pos_x = p_mm_aabb.position.x;
	geom.aabb_pos_y = p_mm_aabb.position.y;
	geom.aabb_pos_z = p_mm_aabb.position.z;

	// TBN section starts after all positions in the merged vertex buffer.
	if (has_tbn) {
		uint32_t tbn_base = p_mm_count * vertex_count * 12;
		geom.normal_byte_offset = tbn_base;
		geom.normal_stride = tbn_stride;
		if (has_tangent) {
			geom.tangent_byte_offset = tbn_base; // tangent is at +4 from normal within the same stride pair
			geom.tangent_stride = tbn_stride;
		} else {
			geom.tangent_byte_offset = RT_OFFSET_NONE;
			geom.tangent_stride = 0;
		}
	}

	// Point attribute address at fully replicated attribute buffer.
	if (has_attr && entry.merged_attr_buffer.is_valid()) {
		geom.attribute_buffer_address = rd->buffer_get_device_address(entry.merged_attr_buffer);
	}

	// Point index address at the replicated (uint32) index buffer.
	if (indexed && entry.replicated_idx_buffer.is_valid()) {
		geom.index_buffer_address = rd->buffer_get_device_address(entry.replicated_idx_buffer);
		geom.index_format = RT_INDEX_FORMAT_UINT32; // always uint32 in replicated buffer
		geom.primitive_count = p_mm_count * prim_count;
	} else {
		geom.index_buffer_address = 0;
		geom.index_format = RT_INDEX_FORMAT_NONE;
		geom.primitive_count = p_mm_count * prim_count;
	}

	return true;
}

// ---------------------------------------------------------------------------
// TLAS creation (main entry point per frame)
// ---------------------------------------------------------------------------

_FORCE_INLINE_ static uint32_t _rt_indices_to_primitives(RSE::PrimitiveType p_primitive, uint32_t p_indices) {
	static const uint32_t divisor[RSE::PRIMITIVE_MAX] = { 1, 2, 1, 3, 1 };
	static const uint32_t subtractor[RSE::PRIMITIVE_MAX] = { 0, 0, 1, 0, 2 };
	return (p_indices - subtractor[p_primitive]) / divisor[p_primitive];
}

RTViewportState *RenderRaytracing::build_tlas(const RenderDataRD *p_render_data, uint32_t p_rt_flags, bool p_primary_from_gbuffer) {
	if (!p_render_data || !p_render_data->rt_instances) {
		return nullptr;
	}

	RTViewportState *state = _get_or_create_viewport_state(p_render_data);
	if (!state) {
		return nullptr;
	}

	frame_extra_builds = 0; // prepare_frame's compactions, loads and bakes count into them (D97).
	frame_extra_triangles = 0;
	prepare_frame();
	_sweep_materialized();
	state->has_procedural_instances = false;
	if (ptlas_slots_enabled) {
		state->slot_build++; // The walk stamps the ranges it reaches with it (D94b).
	}

	// The cluster LOD selection (D86) belongs to the traced primary ray (D84): under the G-buffer
	// primary the raster's surface must agree with the traced geometry it casts its shadow and
	// bounce rays from, and a coarser level above it shadows it, so that frame traces LOD 0. With
	// streaming (D92) a surface may have no LOD 0 resident, so that frame selects at 0 pixels
	// instead, the finest resident cut; deformed meshes still trace LOD 0 there.
	const bool lod_select_frame = cluster_lod_select_enabled && (!p_primary_from_gbuffer || cluster_lod_stream_enabled);
	const bool lod_deformed_frame = cluster_lod_select_enabled && !p_primary_from_gbuffer;
	lod_frame_pixel_error = p_primary_from_gbuffer ? 0.0f : cluster_lod_pixel_error;

	// Camera-relative tracing: the trace runs in world space minus an anchor
	// snapped to a grid on x and z, so float32 stays precise kilometers from
	// the origin. The anchor moves only when the camera leaves its cell, and
	// every instance, light and camera upload below subtracts it in double
	// before narrowing to float. Its y stays 0 so world height (fog height,
	// user shaders) is exact. A grid of 0 disables it (absolute world).
	{
		static const double RT_ORIGIN_GRID = (double)GLOBAL_GET("rendering/pathtracing/relative_origin_grid");
		const Vector3 cam_pos = p_render_data->scene_data->cam_transform.origin;
		if (RT_ORIGIN_GRID <= 0.0) {
			state->rt_origin = Vector3();
			state->rt_origin_valid = true;
		} else if (!state->rt_origin_valid || Math::abs(cam_pos.x - state->rt_origin.x) > RT_ORIGIN_GRID || Math::abs(cam_pos.z - state->rt_origin.z) > RT_ORIGIN_GRID) {
			state->rt_origin = Vector3(Math::floor(cam_pos.x / RT_ORIGIN_GRID) * RT_ORIGIN_GRID, 0.0, Math::floor(cam_pos.z / RT_ORIGIN_GRID) * RT_ORIGIN_GRID);
			state->rt_origin_valid = true;
		}
	}

	// Builds bundle if needed; live_ready_mask drives TLAS inclusion below.
	SceneShaderRaytracing *rt_shader_singleton = SceneShaderRaytracing::get_singleton();
	rt_shader_singleton->ensure_pipeline_bundle(p_rt_flags);

	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();
	LocalVector<RID> dirty_blas_list;
	LocalVector<RID> dirty_blas_update_list;

#ifdef TOOLS_ENABLED
	uint32_t tlas_instance_count = 0;
	uint32_t tlas_primitive_count = 0;
	uint32_t rt_blas_builds = 0;
	uint32_t rt_blas_refits = 0;
	uint32_t rt_triangles_built = 0;
	uint32_t rt_triangles_refit = 0;
	cluster_blas_builds = 0;
	cluster_triangles_built = 0;
	deformed_cluster_builds = 0;
	deformed_cluster_triangles_built = 0;
	deformed_cluster_updates = 0;
	deformed_cluster_triangles_updated = 0;
	// Instances left out of this TLAS because their custom hit group is still compiling.
	uint32_t rt_instances_compiling = 0;
	const bool collect_render_info = (p_render_data->render_info != nullptr);
#endif

	// -----------------------------------------------------------------------
	// Phase 1: CPU / buffer-update work
	// -----------------------------------------------------------------------
	struct PendingMMSurface {
		RID mm_rid;
		RID mm_gpu_buffer;
		const RenderForwardClustered::GeometryInstanceSurfaceDataCache *mm_surf;
		void *mesh_surface;
		uint32_t mm_count;
		uint32_t surface_index;
		uint32_t surface_counter;
		Transform3D instance_transform;
		Transform3D prev_instance_transform;
		bool transform_moved;
		RTMaterialData *mat_data;
		uint32_t inst_flags;
		uint8_t inst_mask;
		AABB mm_aabb;
		RTSurfaceData *source = nullptr; // A surface whose geometry streams: its population, holding the upload of its arrays (D93d).
		bool omm_eligible = false; // An alpha-tested standard material, or a custom shader with render_mode opacity_micromap, on a surface whose geometry is resident (D96).
	};
	LocalVector<PendingMMSurface> pending_mm_surfaces;

	// The build budget (D97): this frame's new surfaces and merges admitted or left waiting, once,
	// before the first viewport's walk, which publishes the frame's budget counters.
	const bool build_budget_publishes = build_budget.enabled && build_budget.prepass_frame != RD::get_singleton()->get_frames_drawn();
	_build_budget_prepass(p_render_data, p_rt_flags);

	const PagedArray<RenderGeometryInstance *> &rt_instances = *p_render_data->rt_instances;
	for (uint32_t i = 0; i < (uint32_t)rt_instances.size(); i++) {
		const RenderForwardClustered::GeometryInstanceForwardClustered *inst =
				static_cast<const RenderForwardClustered::GeometryInstanceForwardClustered *>(rt_instances[i]);
		if (!inst || !inst->data) {
			continue;
		}
		const Transform3D &instance_transform = inst->transform;

		// Determine previous-frame transform for motion vectors.
		const Transform3D &prev_instance_transform =
				(inst->transform_status == RenderForwardClustered::GeometryInstanceForwardClustered::TransformStatus::TELEPORTED)
				? inst->transform
				: inst->prev_transform;

		// Handle procedural RT instances (intersection shaders).
		if (inst->rt_procedural) {
			SceneShaderRaytracing *rt_shader = SceneShaderRaytracing::get_singleton();
			RTProceduralState *ps = inst->rt_procedural;

			// Intersection code comes from ShaderMaterial on material_override.
			if (!inst->data || !inst->data->material_override.is_valid()) {
				continue;
			}
			RID proc_material_rid = inst->data->material_override;
			uint32_t shader_id = material_storage->material_get_shader_id(proc_material_rid);
			if (shader_id == 0) {
				continue;
			}

			uint32_t hg_index = rt_shader->register_procedural_shader(shader_id, proc_material_rid);
			if (hg_index == 0) {
				continue;
			}
			if (!rt_shader->is_hg_ready_in_bundle(hg_index, p_rt_flags)) {
#ifdef TOOLS_ENABLED
				if (collect_render_info && rt_shader->is_hg_compiling_in_bundle(hg_index, p_rt_flags)) {
					rt_instances_compiling++;
				}
#endif
				continue;
			}

			const uint32_t proc_builds_before = dirty_blas_list.size();
			if (ps->dirty) {
#ifdef TOOLS_ENABLED
				uint32_t pre_proc_build_size = dirty_blas_list.size();
#endif
				update_procedural_blas(ps, dirty_blas_list);
				ps->dirty = false;
#ifdef TOOLS_ENABLED
				if (collect_render_info) {
					rt_blas_builds += dirty_blas_list.size() - pre_proc_build_size;
				}
#endif
			}

			if (ps->blas.is_valid()) {
				RT_GeometryData geom = {};
				geom.flags = RT_GEOM_FLAG_PROCEDURAL;
				geom.vertex_buffer_address = ps->gpu_buffer_address;
				geom.aabb_size_x = (float)ps->culling_aabb.size.x;
				geom.aabb_size_y = (float)ps->culling_aabb.size.y;
				geom.aabb_size_z = (float)ps->culling_aabb.size.z;
				// Material for procedural geometry (already validated above).
				uint16_t proc_mat_counter = material_storage->material_get_rt_invalidation_counter(proc_material_rid);
				RTMaterialData *proc_mat_data = process_material(proc_material_rid, proc_mat_counter);
				// Procedural instances disable triangle culling and are opaque.
				uint32_t inst_flags = RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT |
						RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT;
				const uint8_t inst_mask = RT_INSTANCE_MASK_GEOMETRY | RT_INSTANCE_MASK_PROCEDURAL | (inst->data->cast_shadows ? RT_INSTANCE_MASK_SHADOW : 0);
				state->has_procedural_instances = true;
				geom._pad[RT_GEOMETRY_LAYERS_WORD] = inst->layer_mask & RT_GEOMETRY_LAYERS_MASK; // The decals' cull mask test (D104).

				if (ptlas_slots_enabled) {
					// Its slot keeps what it holds unless the instance or its rows changed (D94b).
					const uint32_t range = _slot_range_acquire(state, inst->rt_slots, RT_SLOT_KEY_PROCEDURAL, 1);
					const uint32_t slot = slot_ranges[range].first;
					const uint64_t geometry_generation = _local_row_generation(range, geom);
					const uint64_t material_generation = _material_row_generation(proc_mat_data);
					const bool rebuilt = dirty_blas_list.size() != proc_builds_before;
					if (!rebuilt && !_slot_moved(inst->transform_status == RenderForwardClustered::GeometryInstanceForwardClustered::TransformStatus::MOVED, prev_instance_transform, instance_transform) && _slot_unchanged(state, slot, ps->blas, instance_transform, inst_flags, inst_mask, hg_index, geometry_generation, material_generation)) {
						continue;
					}
					_slot_emit(slot, geometry_generation, material_generation);
				}

				blass.push_back(ps->blas);
				blas_transforms.push_back(instance_transform);
				sbt_offsets.push_back(hg_index);
				geometry_data.push_back(geom);

				if (inst->transform_status == RenderForwardClustered::GeometryInstanceForwardClustered::TransformStatus::MOVED) {
					motion_indices.push_back((int32_t)motion_transforms.size());
					RT_InstanceMotionData motion = {};
					Transform3D prev_relative = prev_instance_transform;
					prev_relative.origin -= state->rt_origin;
					RendererRD::MaterialStorage::store_transform_transposed_3x4(prev_relative, motion.prev_object_to_world);
					motion_transforms.push_back(motion);
				} else {
					motion_indices.push_back(-1);
				}

				material_data.push_back(proc_mat_data->data);
				instance_flags.push_back(inst_flags);
				instance_masks.push_back(inst_mask);
			}
			continue;
		}

		// MultiMesh: resolve materials and warm data cache now.
		// Compute dispatches and TLAS assembly are deferred to Phase 2.
		if (inst->data->base_type == RSE::INSTANCE_MULTIMESH) {
			RID mm_rid = inst->data->base;

			if (mesh_storage->multimesh_get_transform_format(mm_rid) != RSE::MULTIMESH_TRANSFORM_3D) {
				continue;
			}

			uint32_t mm_count = mesh_storage->multimesh_get_instances_to_draw(mm_rid);
			if (mm_count == 0) {
				continue;
			}

			RID mm_gpu_buffer = mesh_storage->multimesh_get_gpu_buffer(mm_rid);
			// Populate data cache now — first access triggers GPU readback, safe here.
			mesh_storage->multimesh_get_local_data_ptr(mm_rid);

			bool transform_moved = (inst->transform_status ==
					RenderForwardClustered::GeometryInstanceForwardClustered::TransformStatus::MOVED);

			const RenderForwardClustered::GeometryInstanceSurfaceDataCache *mm_surf = inst->surface_caches;
			while (mm_surf) {
				if (mm_surf->rt_pass_flags & RenderForwardClustered::GeometryInstanceSurfaceDataCache::FLAG_PASS_ALPHA) {
					mm_surf = mm_surf->next;
					continue;
				}

				void *mesh_surface = mm_surf->surface;
				uint32_t surface_counter = mesh_storage->mesh_surface_get_rt_invalidation_counter(mesh_surface);

				RID material_rid;
				if (mm_surf->owner->data->material_override.is_valid()) {
					material_rid = mm_surf->owner->data->material_override;
				} else if (mm_surf->surface_index < mm_surf->owner->data->surface_materials.size() &&
						mm_surf->owner->data->surface_materials[mm_surf->surface_index].is_valid()) {
					material_rid = mm_surf->owner->data->surface_materials[mm_surf->surface_index];
				} else {
					RID mesh_rid = mesh_storage->multimesh_get_mesh(mm_rid);
					if (mesh_rid.is_valid() && mesh_storage->owns_mesh(mesh_rid)) {
						material_rid = mesh_storage->mesh_surface_get_material(mesh_rid, mm_surf->surface_index);
					}
				}

				uint16_t material_counter = material_storage->material_get_rt_invalidation_counter(material_rid);
				RTMaterialData *mat_data = process_material(material_rid, material_counter);

				if (mat_data->rt_sbt_offset > 0 &&
						!rt_shader_singleton->is_hg_ready_in_bundle(mat_data->rt_sbt_offset, p_rt_flags)) {
#ifdef TOOLS_ENABLED
					if (collect_render_info && rt_shader_singleton->is_hg_compiling_in_bundle(mat_data->rt_sbt_offset, p_rt_flags)) {
						rt_instances_compiling += mm_count;
					}
#endif
					mm_surf = mm_surf->next;
					continue;
				}

				uint32_t inst_flags = 0;
				if (mm_surf->shader) {
					switch (mm_surf->shader->rt_cull_mode()) {
						case RSE::CULL_MODE_DISABLED:
							inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT;
							inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
							break;
						case RSE::CULL_MODE_FRONT:
							break;
						case RSE::CULL_MODE_BACK:
						default:
							inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
							break;
					}
				} else {
					inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
				}
				if (mat_data->rt_sbt_offset > 0) {
					const SceneShaderRaytracing::CustomShaderEntry *cse =
							rt_shader_singleton->get_custom_shader_entry(mat_data->rt_sbt_offset);
					if (!cse || (!cse->uses_alpha_clip && cse->alpha_texture_buffer_offset == UINT32_MAX)) {
						inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT;
					}
					if (cse && cse->uses_transmission) {
						// A transmissive instance is traced as cull_disabled: the refracted ray needs its exit interface.
						inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT;
						inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
					}
				} else {
					bool is_alpha = mm_surf->shader &&
							(mm_surf->shader->uses_alpha_clip || mm_surf->shader->uses_blend_alpha || mm_surf->shader->uses_alpha);
					if (!is_alpha) {
						inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT;
					}
					if (mat_data->data.flags & RT_MAT_FLAG_DIFFUSE_TRANSMISSION) {
						// A thin transmissive instance (D103) is traced as cull_disabled: its transmitted rays leave by its back.
						inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT;
						inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
					}
				}

				PendingMMSurface pending;
				// A surface whose buffers hold its fallback level reaches the merge's bake, and an
				// emissive material's sampling, only through an upload of its arrays (D93d), made here
				// before the merge's compute list opens and hosted by the surface's population, which
				// the expanded path traces anyway. A merge that does not bake this frame reads nothing
				// of the surface, so its upload goes past the minimum age.
				if (mesh_storage->mesh_surface_is_geometry_streamed(mesh_surface)) {
					if (p_primary_from_gbuffer) {
						WARN_PRINT_ONCE("RT: the G-buffer primary rasterizes a MultiMesh of a surface whose geometry streams at its fallback level (D93d), coarser than the traced surface; the traced primary is the look.");
					}
					bool bakes = false;
					const bool merges = !_build_budget_merge_waits(mm_rid, mm_surf->surface_index) && _plan_merged_mm(mm_rid, mm_gpu_buffer, mesh_surface, mm_count, mm_surf->surface_index, surface_counter, bakes);
					pending.source = process_surface(mm_surf, mesh_surface, surface_counter, instance_transform, dirty_blas_list);
					if (pending.source && pending.source->geometry_streamed && ((merges && bakes) || (mat_data->data.flags & RT_MAT_FLAG_EMISSIVE_CONSTANT))) {
						_materialize_surface(pending.source, mesh_surface, false, true);
					}
				}
				pending.mm_rid = mm_rid;
				pending.mm_gpu_buffer = mm_gpu_buffer;
				pending.mm_surf = mm_surf;
				pending.mesh_surface = mesh_surface;
				pending.mm_count = mm_count;
				pending.surface_index = mm_surf->surface_index;
				pending.surface_counter = surface_counter;
				pending.instance_transform = instance_transform;
				pending.prev_instance_transform = prev_instance_transform;
				pending.transform_moved = transform_moved;
				pending.mat_data = mat_data;
				pending.inst_flags = inst_flags;
				pending.inst_mask = RT_INSTANCE_MASK_GEOMETRY | (inst->data->cast_shadows ? RT_INSTANCE_MASK_SHADOW : 0);
				pending.mm_aabb = mesh_storage->_multimesh_get_aabb(mm_rid);
				pending.omm_eligible = omm_enabled && !(inst_flags & RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT) && (mat_data->rt_sbt_offset == 0 || mat_data->omm_custom) && !mesh_storage->mesh_surface_is_geometry_streamed(mesh_surface);
				pending_mm_surfaces.push_back(pending);

				mm_surf = mm_surf->next;
			}
			continue;
		}

		// Walk the surface cache linked list.
		const RenderForwardClustered::GeometryInstanceSurfaceDataCache *surf = inst->surface_caches;
		bool instance_static = inst->transform_status == RenderForwardClustered::GeometryInstanceForwardClustered::TransformStatus::NONE;
		while (surf) {
			// Skip surfaces routed to the raster alpha overlay
			if (surf->rt_pass_flags & RenderForwardClustered::GeometryInstanceSurfaceDataCache::FLAG_PASS_ALPHA) {
				surf = surf->next;
				continue;
			}

			void *mesh_surface = surf->surface;
			uint32_t surface_counter = mesh_storage->mesh_surface_get_rt_invalidation_counter(mesh_surface);
			const uint32_t pre_slot_build_size = dirty_blas_list.size();
			const uint32_t pre_slot_refit_size = dirty_blas_update_list.size();

#ifdef TOOLS_ENABLED
			uint32_t pre_build_size = dirty_blas_list.size();
			uint32_t pre_refit_size = dirty_blas_update_list.size();
#endif

			// MeshInstance skinning/blend shapes provide a deformed vertex buffer.
			RTSurfaceData *surf_data = nullptr;
			uint32_t deformed_cache_key = 0;
			if (inst->mesh_instance.is_valid()) {
				RID curr_vb = mesh_storage->mesh_instance_get_vertex_buffer(inst->mesh_instance, surf->surface_index);
				if (curr_vb.is_valid()) {
					RTDeformedGeometrySource src;
					src.current_vb = curr_vb;
					src.prev_vb = mesh_storage->mesh_instance_get_prev_vertex_buffer(inst->mesh_instance, surf->surface_index);
					src.change_stamp = mesh_storage->mesh_instance_get_last_change(inst->mesh_instance, surf->surface_index);
					uint64_t mi_id = inst->mesh_instance.get_id();
					uint32_t mi_index = static_cast<uint32_t>(mi_id & 0xFFFFFFFFULL);
					src.cache_version = static_cast<uint32_t>(mi_id >> 32);
					src.cache_key = (static_cast<uint64_t>(mi_index) << 16) | (surf->surface_index & 0xFFFFu);
					src.surface_counter = surface_counter;
					deformed_cache_key = static_cast<uint32_t>(src.cache_key);
					surf_data = process_deformed_surface(surf, mesh_surface, src, dirty_blas_list, dirty_blas_update_list, lod_deformed_frame && cluster_lod_deformed_enabled);
				}
			}
			if (!surf_data) {
				surf_data = process_surface(surf, mesh_surface, surface_counter, instance_transform, dirty_blas_list);
			}
			if (!surf_data || !surf_data->blas.is_valid()) {
				surf = surf->next;
				continue;
			}

			// Resolve material before TLAS so we can skip surfaces whose HG is not live yet (override > surface > mesh).
			RID material_rid;
			if (surf->owner->data->material_override.is_valid()) {
				material_rid = surf->owner->data->material_override;
			} else if (surf->surface_index < surf->owner->data->surface_materials.size() &&
					surf->owner->data->surface_materials[surf->surface_index].is_valid()) {
				material_rid = surf->owner->data->surface_materials[surf->surface_index];
			} else {
				RID mesh_rid = surf->owner->data->base;
				if (mesh_rid.is_valid() && mesh_storage->owns_mesh(mesh_rid)) {
					material_rid = mesh_storage->mesh_surface_get_material(mesh_rid, surf->surface_index);
				}
			}

			uint16_t material_counter = material_storage->material_get_rt_invalidation_counter(material_rid);
			RTMaterialData *mat_data = process_material(material_rid, material_counter);

			if (mat_data->rt_sbt_offset > 0 &&
					!rt_shader_singleton->is_hg_ready_in_bundle(mat_data->rt_sbt_offset, p_rt_flags)) {
#ifdef TOOLS_ENABLED
				if (collect_render_info && rt_shader_singleton->is_hg_compiling_in_bundle(mat_data->rt_sbt_offset, p_rt_flags)) {
					rt_instances_compiling++;
				}
#endif
				surf = surf->next;
				continue;
			}

			// Compute or reuse cached final transform (instance * aabb_transform for compressed meshes).
			Transform3D final_transform;
			if (instance_static && surf->cached_final_transform_valid) {
				final_transform = surf->cached_final_transform;
			} else {
				final_transform = instance_transform;
				if (surf_data->is_compressed) {
					final_transform = instance_transform * surf_data->aabb_transform;
				}
				surf->cached_final_transform = final_transform;
				surf->cached_final_transform_valid = true;
			}

			// Determine per-instance TLAS flags from material properties.
			uint32_t inst_flags = 0;
			if (surf->shader) {
				switch (surf->shader->rt_cull_mode()) {
					case RSE::CULL_MODE_DISABLED:
						inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT;
						inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
						break;
					case RSE::CULL_MODE_FRONT:
						break;
					case RSE::CULL_MODE_BACK:
					default:
						inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
						break;
				}
			} else {
				inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
			}

			if (mat_data->rt_sbt_offset > 0) {
				// Custom shader: enable any-hit if it uses alpha clip or has a hint_albedo texture.
				const SceneShaderRaytracing::CustomShaderEntry *cse =
						rt_shader_singleton->get_custom_shader_entry(mat_data->rt_sbt_offset);
				if (!cse || (!cse->uses_alpha_clip && cse->alpha_texture_buffer_offset == UINT32_MAX)) {
					inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT;
				}
				if (cse && cse->uses_transmission) {
					// A transmissive instance is traced as cull_disabled: the refracted ray needs its exit interface.
					inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT;
					inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
				}
			} else {
				// Standard material: FORCE_OPAQUE if no alpha usage.
				bool is_alpha = surf->shader && (surf->shader->uses_alpha_clip || surf->shader->uses_blend_alpha || surf->shader->uses_alpha);
				if (!is_alpha) {
					inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT;
				}
				if (mat_data->data.flags & RT_MAT_FLAG_DIFFUSE_TRANSMISSION) {
					// A thin transmissive instance (D103) is traced as cull_disabled: its transmitted rays leave by its back.
					inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT;
					inst_flags |= RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
				}
			}
			const uint8_t inst_mask = RT_INSTANCE_MASK_GEOMETRY | (inst->data->cast_shadows ? RT_INSTANCE_MASK_SHADOW : 0);

			// The cluster LOD selection (D86): an instance of a surface whose hierarchy is on the GPU
			// traces its own BLAS over the clusters this frame's selection picks for it, built at
			// the flush below, unless the material is an emissive mesh light (its sampling covers
			// the LOD 0 triangles alone, D50) or the frame's flush fails, when the surface's LOD 0
			// BLAS stands in. The metric transforms the mesh-space spheres, so the instance's own
			// transform, not the compressed surface's final one.
			// Which build the instance traces is settled after the walk, where the instances of one
			// surface can share one (D89); until then the slot holds the LOD 0 BLAS.
			RID blas_handle = surf_data->blas;
			// Opacity micromaps (D96): an alpha-tested standard material, or a custom shader with
			// render_mode opacity_micromap, traces its surface's micromap, once baked. Only such an
			// instance: a micromap's transparent states cull even where FORCE_OPAQUE is set.
			if (omm_enabled) {
				const bool omm_material = !(inst_flags & RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT) && (mat_data->rt_sbt_offset == 0 || mat_data->omm_custom) &&
						surf->owner->data->base_type == RSE::INSTANCE_MESH;
				if (!surf->rt_deformed_handle.is_valid() && !(surf_data->geometry.flags & RT_GEOM_FLAG_DEFORMED)) {
					// A static BLAS is shared by every instance of the surface: such an instance
					// traces the surface's second BLAS, which carries the micromap. On the cluster
					// path (OMMd) only where the instance traces LOD 0, which the micromap is baked
					// over: not a streamed surface (D92), and not where the selection below picks its
					// clusters (D86), whose coarser levels would need bakes of their own.
					const bool lod0_traced = surf_data->cluster_count == 0 ||
							(!surf_data->cluster_lod_stream && !(lod_select_frame && surf_data->cluster_lod_table_offset != UINT32_MAX && !(mat_data->data.flags & RT_MAT_FLAG_EMISSIVE_CONSTANT)));
					if (omm_material && lod0_traced && !surf_data->geometry_streamed) {
						RTOmmEntry *omm = _omm_request(surf->owner->data->base, surf->surface_index, surface_counter, mat_data);
						const RID omm_blas = surf_data->cluster_count > 0 ? _omm_surface_cluster_blas(omm, mesh_surface, surf_data) : _omm_surface_blas(omm, mesh_surface, surf_data, dirty_blas_list);
						if (omm_blas.is_valid()) {
							blas_handle = omm_blas;
						}
					}
				} else if (surf->rt_deformed_handle.is_valid()) {
					// A deformed triangle BLAS is the instance's own (OMMc), and so are a deformed
					// cluster BLAS's CLAS (OMMd): it carries the micromap itself. Not a compressed
					// surface: the deformed row reads its attributes as uncompressed
					// (force_uncompressed), the bake as they are, so the two would not agree. Not an
					// entry with a hierarchy (D87): its selection builds coarser levels the micromap
					// is not baked over. A cluster entry built for an earlier viewport of this frame
					// keeps that build (nothing is rebuilt in place twice in a frame, D82).
					RTDeformedCacheEntry *deformed = deformed_pool.get_or_null(surf->rt_deformed_handle);
					const bool omm_cluster_settles = deformed != nullptr && deformed->cluster_count > 0 && deformed->cluster_lod_table_offset == UINT32_MAX && deformed->cluster_built_frame != RD::get_singleton()->get_frames_drawn();
					if (deformed != nullptr && deformed->ptr == surf_data && (deformed->cluster_count == 0 || omm_cluster_settles)) {
						uint64_t serial = 0;
						if (omm_material && !(mesh_storage->mesh_surface_get_format(mesh_surface) & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES)) {
							const RTOmmEntry *omm = _omm_request(surf->owner->data->base, surf->surface_index, surface_counter, mat_data);
							if (omm != nullptr && omm->state == RTOmmEntry::STATE_BUILT && omm->serial != deformed->omm_failed_serial) {
								serial = omm->serial;
							}
						}
						deformed->omm_wanted_serial = serial;
						if (serial != deformed->omm_serial) {
							// Settled in this walk, as a merge is before its build: the BLAS traced this
							// frame never carries a micromap the material no longer asks for, whose
							// transparent states would cull even under FORCE_OPAQUE. The build, refit or
							// cluster rebuild the processing queued for it is dropped, and the BLAS, or
							// the cluster layout whose CLAS infos carry the micromap, is built again here.
							if (deformed->cluster_count > 0) {
								if (deformed->cluster_frame_kind != RTDeformedCacheEntry::CLUSTER_QUEUED_NONE && !_unqueue_deformed_cluster_rebuild(*deformed)) {
									ERR_PRINT_ONCE("RT: a deformed cluster entry queued out of order cannot take another micromap this frame.");
									surf = surf->next;
									continue;
								}
								_release_deformed_blas(*deformed);
								if (_build_deformed_cluster_layout(surf->rt_deformed_handle, *deformed, mesh_surface, deformed_cache_key, surface_counter)) {
									_queue_deformed_cluster_rebuild(*deformed, RTDeformedCacheEntry::CLUSTER_QUEUED_FIRST_BUILD);
								}
							} else {
								dirty_blas_list.resize(pre_slot_build_size);
								dirty_blas_update_list.resize(pre_slot_refit_size);
								_release_deformed_blas(*deformed);
								_build_deformed_triangle_blas(surf->rt_deformed_handle, *deformed, mesh_surface, deformed_cache_key, surface_counter, dirty_blas_list);
							}
							deformed->blas_built_once = surf_data->blas.is_valid();
							_mark_deformed_geometry(*deformed);
							if (!surf_data->blas.is_valid()) {
								surf = surf->next;
								continue;
							}
							blas_handle = surf_data->blas;
						}
					}
				}
			}
			// A slot-resident instance (D94b) whose BLAS this frame builds or may build after the walk
			// is never proven unchanged here: the commit sees the BLAS it ends on.
			bool slot_rebuilt = dirty_blas_list.size() != pre_slot_build_size || dirty_blas_update_list.size() != pre_slot_refit_size;
			if (p_primary_from_gbuffer && surf_data->geometry_streamed) {
				WARN_PRINT_ONCE("RT: the G-buffer primary rasterizes a surface whose geometry streams at its fallback level (D93d), coarser than the traced surface; the traced primary is the look.");
			}
			// An emissive material samples the surface's own triangles (D50), which a surface whose
			// buffers hold its fallback level has only as an upload (D93d).
			if (surf_data->geometry_streamed && (mat_data->data.flags & RT_MAT_FLAG_EMISSIVE_CONSTANT)) {
				_materialize_surface(surf_data, mesh_surface, false, true);
			}
			if (lod_select_frame && surf_data->cluster_lod_stream && (mat_data->data.flags & RT_MAT_FLAG_EMISSIVE_CONSTANT)) {
				blas_handle = _lod_stream_pinned_blas(surf_data); // A streamed surface's LOD 0 is its cached level 0 (D92).
			} else if (lod_select_frame && surf_data->cluster_lod_table_offset != UINT32_MAX && !(mat_data->data.flags & RT_MAT_FLAG_EMISSIVE_CONSTANT)) {
				slot_rebuilt = true;
				RTLodInstanceEntry *lod_entry = _access_lod_slot(surf->rt_lod_handle);
				lod_entry->last_used_frame = RSG::rasterizer->get_frame_number();
				if (lod_entry->pushed_frame == RD::get_singleton()->get_frames_drawn()) {
					blas_handle = lod_entry->frame_blas; // Settled for an earlier viewport of this frame.
				} else {
					RTLodSelectBatch::Candidate candidate;
					candidate.entry = lod_entry;
					candidate.surface = surf_data;
					candidate.transform = instance_transform;
					candidate.transform.origin -= state->rt_origin;
					candidate.slot = blass.size();
					lod_batch.candidates.push_back(candidate);
				}
			}
			if (surf->rt_deformed_handle.is_valid()) {
				// A deformed surface in the TLAS this frame: the flush counts it (D82). One with a
				// hierarchy on a selecting frame (D87) joins the selection here, where its slot and
				// material are known: the batch's rebuild stands in for an emissive mesh light or
				// past the dispatch's instance limit, and a second viewport reuses the frame's build.
				RTDeformedCacheEntry *deformed = deformed_pool.get_or_null(surf->rt_deformed_handle);
				if (deformed && deformed->cluster_count > 0) {
					const uint64_t frames_drawn = RD::get_singleton()->get_frames_drawn();
					// A templated entry (D107) is rewritten only on the frames something rebuilds it: queued
					// for this frame's batch, pending this frame's selection (D108, or its LOD 0 rebuild
					// through the batch, queued below), or built by an earlier viewport; the others are
					// rebuilt every frame (D82). The slot is re-emitted on those frames, so a failed flush
					// can drop the instance.
					if (!deformed->cluster_templated || deformed->cluster_lod_pending || deformed->cluster_frame_kind != RTDeformedCacheEntry::CLUSTER_QUEUED_NONE || deformed->cluster_built_frame == frames_drawn) {
						slot_rebuilt = true;
					}
					deformed->cluster_pushed_frame = frames_drawn;
					if (surf_data != deformed->ptr) {
						deformed->cluster_lod_pending = false; // Not the surface traced here.
					} else if (deformed->cluster_built_frame == frames_drawn) {
						// Built for an earlier viewport of this frame, by either path: the first
						// viewport of a frame decides, so a G-buffer viewport rendered after a
						// traced one traces the selection for deformed meshes, and a traced one
						// after a G-buffer viewport traces LOD 0. Nothing is rebuilt in place
						// twice in one frame.
						deformed->cluster_lod_pending = false;
					} else if (deformed->cluster_lod_pending) {
						deformed->cluster_lod_pending = false;
						const bool selects = !(mat_data->data.flags & RT_MAT_FLAG_EMISSIVE_CONSTANT);
						// Under the frame's CLAS region ceiling, or for a persistent item (D109), which takes no
						// region, the arguments' ceiling over its slots.
						const bool persistent = cluster_lod_deformed_templated && deformed->cluster_templated;
						const bool fits = persistent ? uint64_t(lod_batch.clas_bound) + deformed->cluster_lod_slot_capacity <= cluster_lod_persistent_ceiling_slots : uint64_t(lod_batch.clas_bound) + deformed->cluster_lod_count <= cluster_lod_clas_ceiling_clusters;
						if (selects && !fits) {
							lod_batch.deformed_routed++; // Past the frame's ceiling: LOD 0 through the batch.
						}
						if (selects && fits && lod_batch.entries.size() < cluster_lod_max_instances) {
							Transform3D lod_transform = instance_transform;
							lod_transform.origin -= state->rt_origin;
							_queue_lod_deformed_instance(*deformed, lod_transform, blass.size());
						} else {
							uint8_t kind = deformed->cluster_lod_pending_kind;
							if (kind == RTDeformedCacheEntry::CLUSTER_QUEUED_NONE && deformed->cluster_lod_last) {
								kind = RTDeformedCacheEntry::CLUSTER_QUEUED_EXPIRY; // The last build was a selection: LOD 0 again first.
							}
							if (kind != RTDeformedCacheEntry::CLUSTER_QUEUED_NONE) {
								_queue_deformed_cluster_rebuild(*deformed, kind);
							}
						}
					}
				}
			}

#ifdef TOOLS_ENABLED
			if (collect_render_info) {
				tlas_instance_count++;
				uint32_t vertices = mesh_storage->mesh_surface_get_vertices_drawn_count(mesh_surface);
				// A surface whose geometry streams draws its fallback level; the trace holds the surface (D93d).
				uint32_t prim_count = surf_data->geometry_streamed ? surf_data->geometry.primitive_count : _rt_indices_to_primitives(surf->primitive, vertices);
				tlas_primitive_count += prim_count;
				uint32_t build_delta = dirty_blas_list.size() - pre_build_size;
				uint32_t refit_delta = dirty_blas_update_list.size() - pre_refit_size;
				rt_blas_builds += build_delta;
				rt_blas_refits += refit_delta;
				rt_triangles_built += prim_count * build_delta;
				rt_triangles_refit += prim_count * refit_delta;
			}
#endif

			if (build_budget.enabled) {
				_build_budget_stamp(inst, surf->surface_index, RD::get_singleton()->get_frames_drawn()); // Its next miss is forced (D97).
			}
			if (ptlas_slots_enabled) {
				// Its slot keeps what it holds unless the instance or its rows changed (D94b).
				const uint32_t range = _slot_range_acquire(state, inst->rt_slots, surf->surface_index, 1);
				const uint32_t slot = slot_ranges[range].first;
				const uint64_t geometry_generation = _layer_row_generation(_surface_row_generation(surf_data), inst->layer_mask);
				const uint64_t material_generation = _material_row_generation(mat_data);
				if (!slot_rebuilt && !_slot_moved(inst->transform_status == RenderForwardClustered::GeometryInstanceForwardClustered::TransformStatus::MOVED, prev_instance_transform, instance_transform) && _slot_unchanged(state, slot, blas_handle, final_transform, inst_flags, inst_mask, mat_data->rt_sbt_offset, geometry_generation, material_generation)) {
					surf = surf->next;
					continue;
				}
				_slot_emit(slot, geometry_generation, material_generation);
			}

			blas_transforms.push_back(final_transform);
			blass.push_back(blas_handle);
			geometry_data.push_back(surf_data->geometry);
			geometry_data[geometry_data.size() - 1]._pad[RT_GEOMETRY_LAYERS_WORD] = inst->layer_mask & RT_GEOMETRY_LAYERS_MASK; // D104.

			if (inst->transform_status == RenderForwardClustered::GeometryInstanceForwardClustered::TransformStatus::MOVED) {
				motion_indices.push_back((int32_t)motion_transforms.size());
				RT_InstanceMotionData motion = {};
				Transform3D prev_final = prev_instance_transform;
				if (surf_data->is_compressed) {
					prev_final = prev_instance_transform * surf_data->aabb_transform;
				}
				prev_final.origin -= state->rt_origin;
				RendererRD::MaterialStorage::store_transform_transposed_3x4(prev_final, motion.prev_object_to_world);
				motion_transforms.push_back(motion);
			} else {
				motion_indices.push_back(-1);
			}

			sbt_offsets.push_back(mat_data->rt_sbt_offset);
			material_data.push_back(mat_data->data);
			instance_flags.push_back(inst_flags);
			instance_masks.push_back(inst_mask);

			surf = surf->next;
		}
	}

	// -----------------------------------------------------------------------
	// Phase 2: GPU compute — merged MultiMesh BLAS dispatches.
	// -----------------------------------------------------------------------
	// Opacity micromaps (D96): a merge is shared by every instance of its MultiMesh, and a micromap's
	// transparent states cull even where FORCE_OPAQUE is set, so the merge carries the base surface's
	// micromap, once baked, only when every instance of it this frame asks for the same one. Otherwise
	// it stays plain and the any-hit decides for all of them.
	HashMap<uint64_t, const RTOmmEntry *> mm_omm_agreed;
	if (omm_enabled) {
		for (const PendingMMSurface &pending : pending_mm_surfaces) {
			const RTOmmEntry *omm = nullptr;
			if (pending.omm_eligible) {
				RTOmmEntry *requested = _omm_request(mesh_storage->multimesh_get_mesh(pending.mm_rid), pending.surface_index, pending.surface_counter, pending.mat_data);
				if (requested != nullptr && requested->state == RTOmmEntry::STATE_BUILT) {
					omm = requested;
				}
			}
			const uint64_t key = (uint64_t(pending.mm_rid.get_local_index()) << 32) | pending.surface_index;
			const RTOmmEntry **agreed = mm_omm_agreed.getptr(key);
			if (agreed == nullptr) {
				mm_omm_agreed.insert(key, omm);
			} else if (*agreed != omm) {
				*agreed = nullptr;
			}
		}
	}

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	for (const PendingMMSurface &pending : pending_mm_surfaces) {
		const uint32_t mm_slot_build_size = dirty_blas_list.size();
		const uint32_t mm_slot_refit_size = dirty_blas_update_list.size();
#ifdef TOOLS_ENABLED
		uint32_t mm_pre_build_size = dirty_blas_list.size();
		uint32_t mm_pre_refit_size = dirty_blas_update_list.size();
#endif
		if (omm_enabled) {
			_omm_attach_merged(pending.mm_rid, pending.surface_index, mm_omm_agreed[(uint64_t(pending.mm_rid.get_local_index()) << 32) | pending.surface_index]);
		}
		RTSurfaceData merged_sd;
		// A merge the build budget (D97) left waiting traces its expanded instances meanwhile.
		bool use_merged = pending.mm_gpu_buffer.is_valid() && !_build_budget_merge_waits(pending.mm_rid, pending.surface_index) &&
				_build_merged_mm_blas(pending.mm_rid, pending.mm_gpu_buffer, pending.mesh_surface,
						pending.mm_count, pending.surface_index, pending.surface_counter, pending.mm_aabb, pending.source,
						compute_list, dirty_blas_list, dirty_blas_update_list, &merged_sd);

		if (use_merged) {
			merged_sd.geometry._pad[RT_GEOMETRY_LAYERS_WORD] = pending.mm_surf->owner->layer_mask & RT_GEOMETRY_LAYERS_MASK; // D104.
			if (build_budget.enabled) {
				// The merge's surface: expanded the next frame (past its cap), its base surface is forced (D97).
				_build_budget_stamp(pending.mm_surf->owner, pending.surface_index, RD::get_singleton()->get_frames_drawn());
			}
			if (_build_budget_merge_admitted(pending.mm_rid, pending.surface_index)) {
				build_budget.built += dirty_blas_list.size() - mm_slot_build_size;
			}
#ifdef TOOLS_ENABLED
			if (collect_render_info) {
				tlas_instance_count++;
				uint32_t prim_count = merged_sd.geometry.primitive_count;
				tlas_primitive_count += prim_count;
				uint32_t build_delta = dirty_blas_list.size() - mm_pre_build_size;
				uint32_t refit_delta = dirty_blas_update_list.size() - mm_pre_refit_size;
				rt_blas_builds += build_delta;
				rt_blas_refits += refit_delta;
				rt_triangles_built += prim_count * build_delta;
				rt_triangles_refit += prim_count * refit_delta;
			}
#endif
			if (ptlas_slots_enabled) {
				// One slot for the merge (D94b), its frame-local row compared by value.
				const uint32_t range = _slot_range_acquire(state, pending.mm_surf->owner->rt_slots, pending.surface_index, 1);
				const uint32_t slot = slot_ranges[range].first;
				state->slot_mm_keys.erase(range); // An expanded key does not describe the merge.
				const uint64_t geometry_generation = _local_row_generation(range, merged_sd.geometry);
				const uint64_t material_generation = _material_row_generation(pending.mat_data);
				const bool rebuilt = dirty_blas_list.size() != mm_slot_build_size || dirty_blas_update_list.size() != mm_slot_refit_size;
				if (!rebuilt && _slot_unchanged(state, slot, merged_sd.blas, pending.instance_transform, pending.inst_flags, pending.inst_mask, pending.mat_data->rt_sbt_offset, geometry_generation, material_generation)) {
					continue;
				}
				_slot_emit(slot, geometry_generation, material_generation);
			}
			blass.push_back(merged_sd.blas);
			blas_transforms.push_back(pending.instance_transform);
			geometry_data.push_back(merged_sd.geometry);
			sbt_offsets.push_back(pending.mat_data->rt_sbt_offset);
			material_data.push_back(pending.mat_data->data);
			motion_indices.push_back(-1);
			instance_flags.push_back(pending.inst_flags);
			instance_masks.push_back(pending.inst_mask);
		} else {
			// Fallback: expanded TLAS — one entry per instance, shared BLAS.
			const float *mm_data = mesh_storage->multimesh_get_local_data_ptr(pending.mm_rid);
			if (!mm_data) {
				continue;
			}

			const uint32_t mm_stride = mesh_storage->multimesh_get_stride(pending.mm_rid);
			const uint32_t mm_cur_offset = mesh_storage->multimesh_get_current_instance_offset(pending.mm_rid);

			RTSurfaceData *surf_data = process_surface(pending.mm_surf, pending.mesh_surface,
					pending.surface_counter, pending.instance_transform, dirty_blas_list);
			if (!surf_data || !surf_data->blas.is_valid()) {
				continue;
			}
			if (build_budget.enabled) {
				_build_budget_stamp(pending.mm_surf->owner, pending.surface_index, RD::get_singleton()->get_frames_drawn()); // D97.
			}

			// Slot-resident (D94b): one slot per element. A range whose key is unchanged is skipped
			// whole; otherwise each element is tested against its slot's shadow.
			uint32_t slot_first = 0;
			uint64_t slot_geometry_generation = 0;
			uint64_t slot_material_generation = 0;
			RID slot_blas;
			bool slot_rebuilt = false;
			bool slot_moved = false;
			bool skip_range = false;
			if (ptlas_slots_enabled) {
				const uint32_t range = _slot_range_acquire(state, pending.mm_surf->owner->rt_slots, pending.surface_index, pending.mm_count);
				slot_first = slot_ranges[range].first;
				slot_geometry_generation = _layer_row_generation(_surface_row_generation(surf_data), pending.mm_surf->owner->layer_mask);
				slot_material_generation = _material_row_generation(pending.mat_data);
				// A streamed surface's LOD 0 is its cached level 0 (D92), which the reference pins.
				slot_blas = surf_data->cluster_lod_stream && cluster_lod_select_enabled ? _lod_stream_pinned_blas(surf_data) : surf_data->blas;
				slot_rebuilt = dirty_blas_list.size() != mm_slot_build_size || dirty_blas_update_list.size() != mm_slot_refit_size;
				slot_moved = _slot_moved(pending.transform_moved, pending.prev_instance_transform, pending.instance_transform);
				RTSlotMultiMeshKey key;
				key.instance_transform = pending.instance_transform;
				key.blas = slot_blas;
				key.buffer_generation = mesh_storage->multimesh_get_buffer_generation(pending.mm_rid);
				key.geometry_generation = slot_geometry_generation;
				key.material_generation = slot_material_generation;
				key.count = pending.mm_count;
				key.sbt_offset = pending.mat_data->rt_sbt_offset;
				key.flags = pending.inst_flags;
				key.mask = pending.inst_mask;
				// The key is stored whenever it changed, so a rebuilt or moved range is walked whole now
				// and may be skipped from the next build on.
				const bool key_unchanged = _slot_multimesh_key_matches(state, range, key);
				skip_range = !slot_rebuilt && !slot_moved && key_unchanged;
			}

			for (uint32_t mi = 0; mi < (skip_range ? 0 : pending.mm_count); mi++) {
				const float *d = mm_data + (mm_cur_offset + mi) * mm_stride;
				Transform3D mm_xform;
				mm_xform.basis.rows[0][0] = d[0];
				mm_xform.basis.rows[0][1] = d[1];
				mm_xform.basis.rows[0][2] = d[2];
				mm_xform.origin.x = d[3];
				mm_xform.basis.rows[1][0] = d[4];
				mm_xform.basis.rows[1][1] = d[5];
				mm_xform.basis.rows[1][2] = d[6];
				mm_xform.origin.y = d[7];
				mm_xform.basis.rows[2][0] = d[8];
				mm_xform.basis.rows[2][1] = d[9];
				mm_xform.basis.rows[2][2] = d[10];
				mm_xform.origin.z = d[11];

				Transform3D final_transform = pending.instance_transform * mm_xform;
				if (surf_data->is_compressed) {
					final_transform = final_transform * surf_data->aabb_transform;
				}

				if (ptlas_slots_enabled) {
					if (!slot_rebuilt && !slot_moved && _slot_unchanged(state, slot_first + mi, slot_blas, final_transform, pending.inst_flags, pending.inst_mask, pending.mat_data->rt_sbt_offset, slot_geometry_generation, slot_material_generation)) {
						continue;
					}
					_slot_emit(slot_first + mi, slot_geometry_generation, slot_material_generation);
					blass.push_back(slot_blas);
				} else {
					// A streamed surface's LOD 0 is its cached level 0 (D92), which the reference pins.
					blass.push_back(surf_data->cluster_lod_stream && cluster_lod_select_enabled ? _lod_stream_pinned_blas(surf_data) : surf_data->blas);
				}
				blas_transforms.push_back(final_transform);
				geometry_data.push_back(surf_data->geometry);
				geometry_data[geometry_data.size() - 1]._pad[RT_GEOMETRY_LAYERS_WORD] = pending.mm_surf->owner->layer_mask & RT_GEOMETRY_LAYERS_MASK; // D104.
				sbt_offsets.push_back(pending.mat_data->rt_sbt_offset);
				material_data.push_back(pending.mat_data->data);

				if (pending.transform_moved) {
					Transform3D prev_final = pending.prev_instance_transform * mm_xform;
					if (surf_data->is_compressed) {
						prev_final = prev_final * surf_data->aabb_transform;
					}
					motion_indices.push_back((int32_t)motion_transforms.size());
					RT_InstanceMotionData motion = {};
					prev_final.origin -= state->rt_origin;
					RendererRD::MaterialStorage::store_transform_transposed_3x4(prev_final, motion.prev_object_to_world);
					motion_transforms.push_back(motion);
				} else {
					motion_indices.push_back(-1);
				}

				instance_flags.push_back(pending.inst_flags);
				instance_masks.push_back(pending.inst_mask);
			}

#ifdef TOOLS_ENABLED
			if (collect_render_info) {
				tlas_instance_count += pending.mm_count;
				uint32_t vertices = mesh_storage->mesh_surface_get_vertices_drawn_count(pending.mesh_surface);
				uint32_t prim_count = surf_data->geometry_streamed ? surf_data->geometry.primitive_count : _rt_indices_to_primitives(pending.mm_surf->primitive, vertices); // As above (D93d).
				tlas_primitive_count += prim_count * pending.mm_count;
				uint32_t build_delta = dirty_blas_list.size() - mm_pre_build_size;
				uint32_t refit_delta = dirty_blas_update_list.size() - mm_pre_refit_size;
				rt_blas_builds += build_delta;
				rt_blas_refits += refit_delta;
				rt_triangles_built += prim_count * build_delta;
				rt_triangles_refit += prim_count * refit_delta;
			}
#endif
		}
	}

	RD::get_singleton()->compute_list_end();

	// The cluster LOD selection and every instance's BLAS over it (D86), the shared builds settled
	// first (D89), recorded here so its fallback still indexes the instance arrays as they were
	// pushed (the deformed flush below may compact them).
	_resolve_lod_candidates(p_render_data, state);
	_flush_lod_batch(p_render_data, state);

	// The deformed surfaces' clusters and cluster BLAS of the frame (D82), recorded after the
	// merge dispatches' compute list and before the counters are published and the TLAS takes
	// them; a failed flush names the handles that never got a build, and their instances leave
	// every per-instance array here.
	_flush_deformed_cluster_batch();
	if (!deformed_cluster_batch.dropped_handles.is_empty() && ptlas_slots_enabled) {
		// Slot-resident (D94b): the arrays stay beside their slots, and the commit disables these.
		for (uint32_t i = 0; i < blass.size(); i++) {
			if (deformed_cluster_batch.dropped_handles.has(blass[i])) {
#ifdef TOOLS_ENABLED
				if (collect_render_info) {
					tlas_instance_count--;
					tlas_primitive_count -= geometry_data[i].primitive_count;
				}
#endif
				blass[i] = RID();
			}
		}
		deformed_cluster_batch.dropped_handles.clear();
	} else if (!deformed_cluster_batch.dropped_handles.is_empty()) {
		uint32_t kept = 0;
		for (uint32_t i = 0; i < blass.size(); i++) {
			if (deformed_cluster_batch.dropped_handles.has(blass[i])) {
#ifdef TOOLS_ENABLED
				if (collect_render_info) {
					tlas_instance_count--;
					tlas_primitive_count -= geometry_data[i].primitive_count;
				}
#endif
				continue;
			}
			if (kept != i) {
				blass[kept] = blass[i];
				blas_transforms[kept] = blas_transforms[i];
				instance_flags[kept] = instance_flags[i];
				instance_masks[kept] = instance_masks[i];
				sbt_offsets[kept] = sbt_offsets[i];
				geometry_data[kept] = geometry_data[i];
				material_data[kept] = material_data[i];
				motion_indices[kept] = motion_indices[i];
			}
			kept++;
		}
		blass.resize(kept);
		blas_transforms.resize(kept);
		instance_flags.resize(kept);
		instance_masks.resize(kept);
		sbt_offsets.resize(kept);
		geometry_data.resize(kept);
		material_data.resize(kept);
		motion_indices.resize(kept);
		deformed_cluster_batch.dropped_handles.clear();
	}

	// Phase 3: BLAS / TLAS build.
#ifdef TOOLS_ENABLED
	if (collect_render_info) {
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_OBJECTS_IN_FRAME] += tlas_instance_count;
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += tlas_primitive_count;
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_RT_TLAS_INSTANCES] += tlas_instance_count;
		const uint32_t frame_builds = rt_blas_builds + cluster_blas_builds + deformed_cluster_builds + frame_extra_builds;
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_RT_BLAS_BUILDS] += frame_builds;
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_RT_BLAS_REFITS] += rt_blas_refits + deformed_cluster_updates;
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_RT_TRIANGLES_BUILT] += rt_triangles_built + cluster_triangles_built + deformed_cluster_triangles_built + frame_extra_triangles;
		if (build_budget_publishes) {
			// The build budget's frame (D97): what it admitted is deferrable work, the rest forced.
			int *info = p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE];
			info[RSE::VIEWPORT_RENDER_INFO_RT_BUILDS_FORCED] += frame_builds > build_budget.built ? int(frame_builds - build_budget.built) : 0;
			info[RSE::VIEWPORT_RENDER_INFO_RT_BUILDS_DEFERRED] += int(build_budget.deferred);
			info[RSE::VIEWPORT_RENDER_INFO_RT_BUILD_QUEUE_DEPTH] += int(build_budget.waiting + build_budget.held);
			info[RSE::VIEWPORT_RENDER_INFO_RT_BUILD_BUDGET_TRIANGLES] += int(MIN(build_budget.spent_triangles, uint64_t(INT32_MAX)));
		}
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_RT_TRIANGLES_REFIT] += rt_triangles_refit + deformed_cluster_triangles_updated;
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_RT_INSTANCES_COMPILING] += rt_instances_compiling;
	}
#endif

	SceneShaderRaytracing::get_singleton()->finalize_custom_shaders();

	const bool built = build_acceleration_structures(state, dirty_blas_list, dirty_blas_update_list);
#ifdef TOOLS_ENABLED
	if (built && collect_render_info && ptlas_slots_enabled) {
		// The slot commit's counters (D94b), read after it ran. A failed commit leaves them partly
		// this frame's and partly the last success's, so that frame, which does not trace, reads 0
		// and the next reads a build from scratch.
		const RTViewportState::SlotCounters &counters = state->slot_counters;
		int *info = p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE];
		info[RSE::VIEWPORT_RENDER_INFO_RT_SLOTS_ENABLED] += counters.enabled;
		info[RSE::VIEWPORT_RENDER_INFO_RT_SLOT_WRITES] += counters.writes;
		info[RSE::VIEWPORT_RENDER_INFO_RT_SLOT_DISABLES] += counters.disables;
		info[RSE::VIEWPORT_RENDER_INFO_RT_SLOT_STALE_REWRITES] += counters.stale_rewrites;
		info[RSE::VIEWPORT_RENDER_INFO_RT_SLOT_ROWS] += counters.rows;
		info[RSE::VIEWPORT_RENDER_INFO_RT_PARTITIONS_LIVE] += counters.partitions_live;
		info[RSE::VIEWPORT_RENDER_INFO_RT_PARTITIONS_TOUCHED] += counters.partitions_touched;
		info[RSE::VIEWPORT_RENDER_INFO_RT_PARTITION_TRANSLATIONS] += counters.translations;
		info[RSE::VIEWPORT_RENDER_INFO_RT_SLOTS_GLOBAL] += counters.global;
		info[RSE::VIEWPORT_RENDER_INFO_RT_SLOTS_GLOBAL_OVERFLOW] += counters.global_overflow;
		info[RSE::VIEWPORT_RENDER_INFO_RT_PTLAS_FROM_SCRATCH] += counters.from_scratch_last;
		info[RSE::VIEWPORT_RENDER_INFO_RT_PTLAS_DANGLING_DISABLES] += counters.dangling_last;
	}
#endif
	if (!built) {
		return nullptr; // The frame skips the fog and the trace (D94b).
	}
	finalize_buffers(state);

	return state;
}

// ---------------------------------------------------------------------------
// Light gathering
// ---------------------------------------------------------------------------

uint32_t RenderRaytracing::gather_lights(const RenderDataRD *p_render_data, RT_LightData *r_light_data, uint32_t p_max_lights, const Vector3 &p_origin, float &r_score_sum, uint32_t &r_directional_count) {
	r_score_sum = 0.0f;
	r_directional_count = 0;
	uint32_t rt_light_count = 0;

	if (!p_render_data || !p_render_data->lights) {
		return rt_light_count;
	}

	RendererRD::LightStorage *ls = RendererRD::LightStorage::get_singleton();
	const Transform3D &cam_xform = p_render_data->scene_data->cam_transform;
	const Vector3 cam_pos = cam_xform.origin;
	const Vector3 cam_forward = -cam_xform.basis.get_column(2).normalized(); // -Z is forward in Godot.

	// Compute light energy matching rasterizer conventions
	auto compute_light_energy = [&](RID p_base, RSE::LightType p_type) -> float {
		float sign = ls->light_is_negative(p_base) ? -1.0f : 1.0f;
		float e = sign * ls->light_get_param(p_base, RSE::LIGHT_PARAM_ENERGY);
		if (owner->is_using_physical_light_units()) {
			e *= ls->light_get_param(p_base, RSE::LIGHT_PARAM_INTENSITY);
			if (p_type == RSE::LIGHT_OMNI) {
				e *= 1.0f / (Math::PI * 4.0f);
			} else if (p_type == RSE::LIGHT_SPOT) {
				e *= 1.0f / Math::PI;
			}
		} else {
			e *= Math::PI;
		}
		if (p_render_data->camera_attributes.is_valid()) {
			e *= RSG::camera_attributes->camera_attributes_get_exposure_normalization_factor(p_render_data->camera_attributes);
		}
		return e;
	};

	// Scoring helper: approximate power/solid-angle contribution.
	struct LightScore {
		RID light_instance;
		float score;
		float share; // The light's weight in the split between the analytic and the emissive list (D50).
	};

	LocalVector<LightScore> positional_lights;

	// Lights behind the camera are deprioritized. The penalty ramps in linearly
	// over BEHIND_FRUSTUM_BUFFER_M so nearby behind-camera lights aren't punished harshly.
	// At or in front of the camera plane: no penalty. Beyond the buffer: full penalty.
	static const float BEHIND_FRUSTUM_PENALTY = 0.05f;
	static const float BEHIND_FRUSTUM_BUFFER_M = 20.0f;

	// Helper: score a positional light and add to candidates.
	auto score_positional_light = [&](RID light_instance) {
		RID base = ls->light_instance_get_base_light(light_instance);
		Transform3D xform = ls->light_instance_get_base_transform(light_instance);
		Vector3 light_pos = xform.origin;
		Vector3 to_light = light_pos - cam_pos;
		float dist_sq = MAX(to_light.dot(to_light), 0.01f);
		Color color = ls->light_get_color(base);
		float energy = ls->light_get_param(base, RSE::LIGHT_PARAM_ENERGY);
		float lum = color.r * 0.2126f + color.g * 0.7152f + color.b * 0.0722f;

		// Range factor: strongly prefer lights with a large influence radius.
		// Lights with no range limit (range == 0) are treated as infinite.
		float range = ls->light_get_param(base, RSE::LIGHT_PARAM_RANGE);
		float range_factor = (range > 0.0f) ? (range * range) : 1.0e12f;

		// Signed depth of the light along the camera forward axis (world units).
		// Positive = in front, negative = behind. Ramp penalty in over the buffer zone.
		float forward_depth = to_light.dot(cam_forward);
		float t = CLAMP((forward_depth + BEHIND_FRUSTUM_BUFFER_M) / BEHIND_FRUSTUM_BUFFER_M, 0.0f, 1.0f);
		float frustum_factor = Math::lerp(BEHIND_FRUSTUM_PENALTY, 1.0f, t);

		float score = (energy * lum * range_factor * frustum_factor) / dist_sq;

		LightScore ls_entry = {};
		ls_entry.light_instance = light_instance;
		ls_entry.score = score;
		// The share in the split with the emissive list (D50): the irradiance at the camera in the
		// shader's units, unlike the score above, which is a ranking with a range preference.
		Color linear_col = color.srgb_to_linear();
		float lum_linear = linear_col.r * 0.2126f + linear_col.g * 0.7152f + linear_col.b * 0.0722f;
		ls_entry.share = compute_light_energy(base, ls->light_get_type(base)) * lum_linear / dist_sq;
		positional_lights.push_back(ls_entry);
	};

	// Directional lights from the frustum-culled list (they're global, always included).
	const PagedArray<RID> &lights = *p_render_data->lights;
	for (uint32_t li = 0; li < (uint32_t)lights.size(); li++) {
		RID light_instance = lights[li];
		RID base = ls->light_instance_get_base_light(light_instance);
		RSE::LightType type = ls->light_get_type(base);

		if (type != RSE::LIGHT_DIRECTIONAL) {
			continue;
		}
		if (rt_light_count >= p_max_lights) {
			break;
		}
		RT_LightData &ld = r_light_data[rt_light_count];
		Transform3D xform = ls->light_instance_get_base_transform(light_instance);
		Vector3 dir = -xform.basis.get_column(2).normalized();
		ld.position[0] = dir.x;
		ld.position[1] = dir.y;
		ld.position[2] = dir.z;
		ld.type = RT_LIGHT_TYPE_DIRECTIONAL;
		Color linear_col = ls->light_get_color(base).srgb_to_linear();
		float energy = compute_light_energy(base, RSE::LIGHT_DIRECTIONAL);
		ld.emission[0] = linear_col.r * energy;
		ld.emission[1] = linear_col.g * energy;
		ld.emission[2] = linear_col.b * energy;
		// Its share of the light samples (D50): the irradiance it delivers at the camera, in the
		// units the emissive list scores in (the same energy and exposure the shader multiplies).
		r_score_sum += (linear_col.r * 0.2126f + linear_col.g * 0.7152f + linear_col.b * 0.0722f) * energy;
		ld.radius = Math::deg_to_rad(ls->light_get_param(base, RSE::LIGHT_PARAM_SIZE) * 0.5f); // Half-angle in radians.
		ld.attenuation = 0.0f; // No distance attenuation.
		ld.inv_max_range = -1.0f; // Infinite range.
		ld.max_range_squared = 0.0f;
		ld.specular_amount = ls->light_get_param(base, RSE::LIGHT_PARAM_SPECULAR);
		ld.indirect_energy = ls->light_get_param(base, RSE::LIGHT_PARAM_INDIRECT_ENERGY);
		ld.inv_spot_attenuation = 0.0f;
		ld.cos_spot_angle = 0.0f;
		ld.spot_direction[0] = 0.0f;
		ld.spot_direction[1] = 0.0f;
		ld.spot_direction[2] = 0.0f;
		rt_light_count++;
	}

	r_directional_count = rt_light_count;

	// Positional lights from the AABB-culled RT list (superset of frustum).
	if (p_render_data->rt_lights) {
		const PagedArray<RID> &rt_lights = *p_render_data->rt_lights;
		for (uint32_t li = 0; li < (uint32_t)rt_lights.size(); li++) {
			score_positional_light(rt_lights[li]);
		}
	}

	// Sort all positional lights by score descending.
	struct LightScoreComparator {
		bool operator()(const LightScore &a, const LightScore &b) const {
			return a.score > b.score;
		}
	};
	positional_lights.sort_custom<LightScoreComparator>();

	// Fill remaining slots with top positional lights.
	for (uint32_t i = 0; i < positional_lights.size() && rt_light_count < p_max_lights; i++) {
		RID light_instance = positional_lights[i].light_instance;
		r_score_sum += positional_lights[i].share;
		RID base = ls->light_instance_get_base_light(light_instance);
		RSE::LightType type = ls->light_get_type(base);

		RT_LightData &ld = r_light_data[rt_light_count];
		Transform3D xform = ls->light_instance_get_base_transform(light_instance);
		const Vector3 relative_pos = xform.origin - p_origin;
		ld.position[0] = relative_pos.x;
		ld.position[1] = relative_pos.y;
		ld.position[2] = relative_pos.z;
		ld.type = (type == RSE::LIGHT_SPOT) ? RT_LIGHT_TYPE_SPOT : RT_LIGHT_TYPE_OMNI;

		Color linear_col = ls->light_get_color(base).srgb_to_linear();
		float energy = compute_light_energy(base, type);
		ld.emission[0] = linear_col.r * energy;
		ld.emission[1] = linear_col.g * energy;
		ld.emission[2] = linear_col.b * energy;
		ld.radius = ls->light_get_param(base, RSE::LIGHT_PARAM_SIZE);
		ld.attenuation = ls->light_get_param(base, RSE::LIGHT_PARAM_ATTENUATION);
		float range = ls->light_get_param(base, RSE::LIGHT_PARAM_RANGE);
		if (range > 0.0f) {
			ld.inv_max_range = 1.0f / range;
			ld.max_range_squared = range * range;
		} else {
			ld.inv_max_range = -1.0f;
			ld.max_range_squared = 0.0f;
		}
		ld.specular_amount = ls->light_get_param(base, RSE::LIGHT_PARAM_SPECULAR) * 2.0f; // Matches rasterizer convention (light_storage.cpp), normalizes 0.5 default to 1.0.
		ld.indirect_energy = ls->light_get_param(base, RSE::LIGHT_PARAM_INDIRECT_ENERGY);

		if (type == RSE::LIGHT_SPOT) {
			ld.inv_spot_attenuation = 1.0f / MAX(0.001f, ls->light_get_param(base, RSE::LIGHT_PARAM_SPOT_ATTENUATION));
			float spot_angle_deg = ls->light_get_param(base, RSE::LIGHT_PARAM_SPOT_ANGLE);
			ld.cos_spot_angle = Math::cos(Math::deg_to_rad(spot_angle_deg));
			Vector3 spot_dir = -xform.basis.get_column(2).normalized();
			ld.spot_direction[0] = spot_dir.x;
			ld.spot_direction[1] = spot_dir.y;
			ld.spot_direction[2] = spot_dir.z;
		} else {
			ld.inv_spot_attenuation = 0.0f;
			ld.cos_spot_angle = 0.0f;
			ld.spot_direction[0] = 0.0f;
			ld.spot_direction[1] = 0.0f;
			ld.spot_direction[2] = 0.0f;
		}
		rt_light_count++;
	}

	return rt_light_count;
}

// ---------------------------------------------------------------------------
// Emissive mesh lights (D50)
// ---------------------------------------------------------------------------

/// Lists the frame's emissive mesh lights from the TLAS instances build_tlas just pushed: every
/// triangle instance on the standard hit group whose material carries RT_MAT_FLAG_EMISSIVE_CONSTANT.
/// The selection weight is the emission's luminance times an area proxy, half the surface area of
/// the instance's box in the trace (a quad reads about its area, a sphere within a factor of two);
/// it only shapes the importance, since the shader divides by the probability the CDF assigns.
/// Over the cap the entries weighted nearest the camera are kept; the list is sorted by instance
/// index so a hit can find its entry by binary search.
uint32_t RenderRaytracing::gather_emissive(const RenderDataRD *p_render_data, const RTViewportState *p_state, RT_EmissiveInstance *r_emissive_data, uint32_t p_max_emissive, float &r_score_sum) {
	r_score_sum = 0.0f;
	if (!p_render_data || !p_render_data->scene_data) {
		return 0;
	}

	LocalVector<EmissiveCandidate> &candidates = emissive_candidates;
	candidates.clear();
	const Vector3 cam_pos = p_render_data->scene_data->cam_transform.origin - p_state->rt_origin;
	// The shader scales emission by the exposure normalisation; the split's shares must too.
	float exposure = p_render_data->camera_attributes.is_valid() ? RSG::camera_attributes->camera_attributes_get_exposure_normalization_factor(p_render_data->camera_attributes) : 1.0f;

	// Slot-resident instances (D94b) are read from the emissive slots the commit keeps, in slot
	// order (the order the walk's arrays would have them in), the index a slot.
	const bool from_slots = ptlas_slots_enabled;
	emissive_slot_records.clear();
	if (from_slots) {
		for (const KeyValue<uint32_t, RTSlotEmissive> &E : p_state->emissive_slots) {
			emissive_slot_records.push_back(Pair<uint32_t, const RTSlotEmissive *>(E.key, &E.value));
		}
		struct SlotComparator {
			bool operator()(const Pair<uint32_t, const RTSlotEmissive *> &a, const Pair<uint32_t, const RTSlotEmissive *> &b) const {
				return a.first < b.first;
			}
		};
		emissive_slot_records.sort_custom<SlotComparator>();
	}
	const uint32_t source_count = from_slots ? emissive_slot_records.size() : blass.size();
	for (uint32_t n = 0; n < source_count; n++) {
		const uint32_t i = from_slots ? emissive_slot_records[n].first : n;
		const RTSlotEmissive *record = from_slots ? emissive_slot_records[n].second : nullptr;
		if (!from_slots && (i >= material_data.size() || i >= geometry_data.size() || i >= sbt_offsets.size() || i >= instance_flags.size())) {
			break;
		}
		const RT_MaterialData &mat = record ? record->material : material_data[i];
		const RT_GeometryData &geom = record ? record->geometry : geometry_data[i];
		const uint32_t sbt_offset = record ? record->sbt_offset : sbt_offsets[i];
		const uint32_t flags = record ? record->flags : instance_flags[i];
		if (!(mat.flags & RT_MAT_FLAG_EMISSIVE_CONSTANT) || sbt_offset != 0 || (geom.flags & RT_GEOM_FLAG_PROCEDURAL) || geom.primitive_count == 0) {
			continue;
		}
		if (geom.vertex_buffer_address == 0 || (geom.index_format != RT_INDEX_FORMAT_NONE && geom.index_buffer_address == 0)) {
			continue; // A surface whose geometry streams, not uploaded (D93d): its triangles cannot be sampled.
		}
		if (!(flags & RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT)) {
			// An alpha-tested or blended emitter's any-hit skips texels a sampled point would credit; it keeps chance-hit emission.
			continue;
		}
		// The instance transform already folds a compressed surface's box in (its positions are then a unit cube).
		AABB local_box = (geom.flags & RT_GEOM_FLAG_COMPRESSED) ? AABB(Vector3(), Vector3(1, 1, 1)) : AABB(Vector3(geom.aabb_pos_x, geom.aabb_pos_y, geom.aabb_pos_z), Vector3(geom.aabb_size_x, geom.aabb_size_y, geom.aabb_size_z));
		Transform3D relative = record ? record->transform : blas_transforms[i];
		relative.origin -= p_state->rt_origin;
		AABB box = relative.xform(local_box);
		const Vector3 &sz = box.size;
		float area_proxy = (float)(sz.x * sz.y + sz.y * sz.z + sz.z * sz.x); // Half the box's surface area.
		float lum = (0.2126f * mat.emission_color[0] + 0.7152f * mat.emission_color[1] + 0.0722f * mat.emission_color[2]) * mat.emission_strength;
		float weight = lum * area_proxy;
		if (!(weight > 0.0f)) {
			continue;
		}
		Vector3 center = box.get_center();
		float dist_sq = (float)(center - cam_pos).length_squared();
		EmissiveCandidate c;
		c.index = i;
		c.record = record;
		c.weight = weight;
		c.score = weight * exposure / (dist_sq + area_proxy);
		c.center = center;
		c.radius = (float)(sz.length() * 0.5);
		candidates.push_back(c);
	}

	if (candidates.size() > p_max_emissive) {
		struct ScoreComparator {
			bool operator()(const EmissiveCandidate &a, const EmissiveCandidate &b) const {
				return a.score > b.score;
			}
		};
		candidates.sort_custom<ScoreComparator>();
		candidates.resize(p_max_emissive);
	}
	struct IndexComparator {
		bool operator()(const EmissiveCandidate &a, const EmissiveCandidate &b) const {
			return a.index < b.index;
		}
	};
	candidates.sort_custom<IndexComparator>();

	double total = 0.0;
	for (const EmissiveCandidate &c : candidates) {
		total += c.weight;
		r_score_sum += c.score;
	}
	if (!(total > 0.0)) {
		return 0;
	}

	double running = 0.0;
	uint32_t count = candidates.size();
	for (uint32_t n = 0; n < count; n++) {
		const EmissiveCandidate &c = candidates[n];
		RT_EmissiveInstance &e = r_emissive_data[n];
		// A slot's sampled triangles sit where the trace finds them: its cell-relative rows under its
		// partition's translation, rounded as the build rounds them (D94c).
		Transform3D relative;
		if (c.record) {
			relative = _slot_traced_transform(p_state, p_state->slots[c.index]);
		} else {
			relative = blas_transforms[c.index];
			relative.origin -= p_state->rt_origin;
		}
		RendererRD::MaterialStorage::store_transform_transposed_3x4(relative, e.xform);
		const RT_MaterialData &mat = c.record ? c.record->material : material_data[c.index];
		const uint32_t flags = c.record ? c.record->flags : instance_flags[c.index];
		e.radiance[0] = mat.emission_color[0] * mat.emission_strength;
		e.radiance[1] = mat.emission_color[1] * mat.emission_strength;
		e.radiance[2] = mat.emission_color[2] * mat.emission_strength;
		running += c.weight;
		e.cdf = (n + 1 == count) ? 1.0f : (float)(running / total);
		e.instance_index = c.index;
		e.primitive_count = c.record ? c.record->geometry.primitive_count : geometry_data[c.index].primitive_count;
		e.flags = 0;
		if (flags & RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT) {
			e.flags |= RT_EMISSIVE_FLAG_TWO_SIDED;
		}
		if (flags & RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT) {
			e.flags |= RT_EMISSIVE_FLAG_FLIP_FACING;
		}
		if (relative.basis.determinant() < 0.0) {
			// The hardware decides the facing from the object-space winding; the sampler's normal is
			// world-space, which a mirroring transform flips.
			e.flags ^= RT_EMISSIVE_FLAG_FLIP_FACING;
		}
		e.pdf = (float)(c.weight / total);
		e.center[0] = (float)c.center.x;
		e.center[1] = (float)c.center.y;
		e.center[2] = (float)c.center.z;
		e.radius = c.radius;
	}
	return count;
}

// ---------------------------------------------------------------------------
// The light tree (D55)
// ---------------------------------------------------------------------------

static float _rt_luminance(const float *p_rgb) {
	return 0.2126f * p_rgb[0] + 0.7152f * p_rgb[1] + 0.0722f * p_rgb[2];
}

/// The smallest sphere holding two spheres.
static void _rt_sphere_union(const Vector3 &p_c1, float p_r1, const Vector3 &p_c2, float p_r2, Vector3 &r_center, float &r_radius) {
	Vector3 d = p_c2 - p_c1;
	float dist = d.length();
	if (dist + p_r2 <= p_r1) {
		r_center = p_c1;
		r_radius = p_r1;
		return;
	}
	if (dist + p_r1 <= p_r2) {
		r_center = p_c2;
		r_radius = p_r2;
		return;
	}
	float r = 0.5f * (dist + p_r1 + p_r2);
	r_center = p_c1 + d * ((r - p_r1) / dist);
	r_radius = r;
}

/// The smallest cone holding two orientation cones (Conty Estevez and Kulla 2018, without the
/// emission falloff angle, which the shader bounds at a right angle). A cosine of -1 is every
/// direction and absorbs anything.
static void _rt_cone_union(const Vector3 &p_a1, float p_cos1, const Vector3 &p_a2, float p_cos2, Vector3 &r_axis, float &r_cos) {
	if (p_cos1 <= -1.0f || p_cos2 <= -1.0f) {
		r_axis = p_a1;
		r_cos = -1.0f;
		return;
	}
	Vector3 a1 = p_a1;
	Vector3 a2 = p_a2;
	float t1 = Math::acos(CLAMP(p_cos1, -1.0f, 1.0f));
	float t2 = Math::acos(CLAMP(p_cos2, -1.0f, 1.0f));
	if (t2 > t1) {
		SWAP(a1, a2);
		SWAP(t1, t2);
	}
	float td = Math::acos(CLAMP(a1.dot(a2), -1.0f, 1.0f));
	if (MIN(td + t2, (float)Math::PI) <= t1) {
		r_axis = a1;
		r_cos = Math::cos(t1);
		return;
	}
	float to = 0.5f * (t1 + td + t2);
	if (to >= (float)Math::PI) {
		r_axis = a1;
		r_cos = -1.0f;
		return;
	}
	float tr = to - t1;
	Vector3 rot_axis = a1.cross(a2);
	if (rot_axis.length_squared() > 1.0e-12f) {
		a1 = Basis(rot_axis.normalized(), tr).xform(a1).normalized();
	}
	r_axis = a1;
	r_cos = Math::cos(to);
}

/// Lists the frame's leaves, one per positional light and per emissive entry (the directional
/// lights and the sky are the tree's top-level companions in the shader), and builds a binary
/// tree over them balanced by count, split at the median of the leaf centres along the longest
/// axis, in depth-first order so a node's left child follows it and a node's leaves are contiguous
/// in the order. Each node carries the union of its children's spheres and cones, their summed
/// power, their largest range and its leaves' range, and the falloff its analytic lights and its
/// emitters each take (D63): their shares of the power, the analytic lights' slowest decay and
/// widest range. Then the cuts: the tree's shallowest
/// nodes, at most p_cut_size of them, which the shader scores once per hit and picks a leaf under
/// by power or uniformly (D56, D60), and a second, coarser cut of at most p_cut_indirect_size for
/// the surfaces past the path's first diffuse bounce (D59). The nodes, the cuts, the leaf list,
/// each emissive entry's cut slots and leaf position, and each cut's alias tables (only with
/// p_power_pick, the one reader) land in the scratch vectors for the upload. Returns the node count, 0 when there is no leaf.
uint32_t RenderRaytracing::build_light_tree(const RT_LightData *p_lights, uint32_t p_light_count, uint32_t p_directional_count, const RT_EmissiveInstance *p_emissive, uint32_t p_emissive_count, float p_exposure, uint32_t p_cut_size, uint32_t p_cut_indirect_size, bool p_power_pick) {
	light_tree_leaves.clear();
	light_tree_nodes.clear();
	light_tree_cut.clear();
	light_tree_cut_indirect.clear();
	if (light_tree_emissive_cut.size() < RT_EMISSIVE_MAX) {
		light_tree_emissive_cut.resize(RT_EMISSIVE_MAX);
	}
	if (light_tree_emissive_leaf.size() < RT_EMISSIVE_MAX) {
		light_tree_emissive_leaf.resize(RT_EMISSIVE_MAX);
	}
	for (uint32_t n = 0; n < p_emissive_count; n++) {
		light_tree_emissive_cut[n] = RT_LIGHT_TREE_NO_NODE;
		light_tree_emissive_leaf[n] = RT_LIGHT_TREE_NO_NODE;
	}

	for (uint32_t i = p_directional_count; i < p_light_count; i++) {
		const RT_LightData &l = p_lights[i];
		// A negative light subtracts; its importance is its magnitude.
		float power = Math::abs(_rt_luminance(l.emission));
		if (!(power > 0.0f)) {
			continue;
		}
		LightTreeLeaf leaf;
		leaf.center = Vector3(l.position[0], l.position[1], l.position[2]);
		leaf.radius = MAX(l.radius, 0.0f);
		// Keyed on the cone the shader keys on: a spot opened to a half turn has cos -1 and is an omni here.
		if (l.type == RT_LIGHT_TYPE_SPOT && l.cos_spot_angle > -1.0f) {
			leaf.axis = Vector3(l.spot_direction[0], l.spot_direction[1], l.spot_direction[2]);
			leaf.cos_theta_o = l.cos_spot_angle;
		} else {
			leaf.axis = Vector3(0, 0, 1);
			leaf.cos_theta_o = -1.0f;
		}
		leaf.power = power;
		leaf.max_range = l.max_range_squared > 0.0f ? Math::sqrt(l.max_range_squared) : 0.0f;
		// The node importance and a leaf slot's target (D59) fall off by it (D63).
		leaf.decay = l.attenuation;
		leaf.encoded = RT_LIGHT_TREE_LEAF_BIT | i;
		light_tree_leaves.push_back(leaf);
	}
	// The emissive gather's ranking is in the entries' order; its weight is the luminance times
	// the area proxy, which under the exposure is the entry's power at unit distance.
	ERR_FAIL_COND_V(p_emissive_count > 0 && emissive_candidates.size() != p_emissive_count, 0);
	for (uint32_t n = 0; n < p_emissive_count; n++) {
		const RT_EmissiveInstance &e = p_emissive[n];
		float power = emissive_candidates[n].weight * p_exposure;
		if (!(power > 0.0f)) {
			continue;
		}
		LightTreeLeaf leaf;
		leaf.center = Vector3(e.center[0], e.center[1], e.center[2]);
		leaf.radius = MAX(e.radius, 0.0f);
		leaf.axis = Vector3(0, 0, 1);
		leaf.cos_theta_o = -1.0f;
		leaf.power = power;
		leaf.max_range = 0.0f;
		leaf.decay = 2.0f;
		leaf.encoded = RT_LIGHT_TREE_LEAF_BIT | RT_LIGHT_TREE_EMISSIVE_BIT | n;
		light_tree_leaves.push_back(leaf);
	}

	uint32_t leaf_count = light_tree_leaves.size();
	if (leaf_count == 0) {
		return 0;
	}
	light_tree_order.resize(leaf_count);
	light_tree_leaf_list.resize(leaf_count);
	if (light_tree_leaf_alias.size() < 2 * RT_LIGHT_TREE_LEAVES_MAX) {
		light_tree_leaf_alias.resize(2 * RT_LIGHT_TREE_LEAVES_MAX);
		light_tree_alias_scaled.resize(RT_LIGHT_TREE_LEAVES_MAX);
		light_tree_alias_small.reserve(RT_LIGHT_TREE_LEAVES_MAX);
		light_tree_alias_large.reserve(RT_LIGHT_TREE_LEAVES_MAX);
	}
	for (uint32_t i = 0; i < leaf_count; i++) {
		light_tree_order[i] = i;
	}
	light_tree_nodes.reserve(2 * leaf_count - 1);
	_build_light_tree_node(0, leaf_count);
	_build_light_tree_cut(p_cut_size, light_tree_cut, 0, 0, p_power_pick);
	// The indirect cut is never wider than the primary: equal sizes (the default) score one cut everywhere.
	_build_light_tree_cut(MIN(p_cut_indirect_size, p_cut_size), light_tree_cut_indirect, 16, RT_LIGHT_TREE_LEAVES_MAX, p_power_pick);
	return light_tree_nodes.size();
}

/// A cut: the root, then breadth first the shallowest node in it replaced by its two children
/// while the size allows, so the cut is the tree's shallowest full level plus part of the next and
/// a tree with no more leaves than the size is cut at its leaves, where the score is each light's
/// own. Each cut node is copied into r_cut for the upload and the emissive entries under it map to
/// its slot in the 16 bits of their entry at p_slot_shift (the primary cut's slot in the low half,
/// the indirect cut's in the high half, D59). With p_power_pick the leaves under each cut node get an alias table over
/// the node's range at p_alias_base plus their leaf-list positions (Vose's construction, in
/// double), so the shader's power-weighted pick is one read at the position a uniform pick lands on
/// (D61; D60 searched cumulative shares), and every entry carries its own share and its alias's, the
/// same floats on both sides of the estimator. A share is the leaf's power share mixed with a
/// uniform floor (RT_LIGHT_TREE_PICK_FLOOR of the node spread evenly), so no leaf's share rounds to
/// zero in float: an analytic light the pick never reached would be light no strategy delivers.
static constexpr double RT_LIGHT_TREE_PICK_FLOOR = 1.0 / 128.0; // The uniform share of the power-weighted pick (D60): every leaf keeps at least this over the node's count.

void RenderRaytracing::_build_light_tree_cut(uint32_t p_cut_size, LocalVector<RT_LightTreeNode> &r_cut, uint32_t p_slot_shift, uint32_t p_alias_base, bool p_power_pick) {
	p_cut_size = CLAMP(p_cut_size, 1u, (uint32_t)RT_LIGHT_TREE_CUT_MAX);
	r_cut.clear();
	light_tree_cut_queue.clear();
	light_tree_cut_queue.push_back(0);
	uint32_t head = 0;
	while (head < light_tree_cut_queue.size()) {
		uint32_t node = light_tree_cut_queue[head];
		const RT_LightTreeNode &n = light_tree_nodes[node];
		// Splitting turns one pending node into two: allowed while the cut plus the queue's rest fits.
		uint32_t size_now = r_cut.size() + (light_tree_cut_queue.size() - head);
		if ((n.child & RT_LIGHT_TREE_LEAF_BIT) == 0 && size_now + 1 <= p_cut_size) {
			light_tree_cut_queue.push_back(node + 1);
			light_tree_cut_queue.push_back(n.child);
		} else {
			uint32_t slot = r_cut.size();
			r_cut.push_back(n);
			uint32_t first = n.leaf_range >> 16;
			uint32_t count = n.leaf_range & 0xFFFFu;
			// The alias table only the power pick reads (D61).
			if (p_power_pick) {
				// Summed in double, stored as float; the floor keeps every share far above float's resolution.
				double node_power = 0.0;
				for (uint32_t i = first; i < first + count; i++) {
					node_power += light_tree_nodes[light_tree_leaf_list[i]].power;
				}
				const double floor_share = node_power > 0.0 ? RT_LIGHT_TREE_PICK_FLOOR / count : 1.0 / count;
				const double power_scale = node_power > 0.0 ? (1.0 - RT_LIGHT_TREE_PICK_FLOOR) / node_power : 0.0;
				light_tree_alias_small.clear();
				light_tree_alias_large.clear();
				for (uint32_t i = first; i < first + count; i++) {
					const double share = floor_share + light_tree_nodes[light_tree_leaf_list[i]].power * power_scale;
					RT_LightTreeAlias &e = light_tree_leaf_alias[p_alias_base + i];
					e.threshold = 1.0f;
					e.alias = i;
					e.share = (float)share;
					light_tree_alias_scaled[i] = share * count;
					if (light_tree_alias_scaled[i] < 1.0) {
						light_tree_alias_small.push_back(i);
					} else {
						light_tree_alias_large.push_back(i);
					}
				}
				// Each position under 1 is filled up to 1 by one position over it, which gives away the difference.
				while (!light_tree_alias_small.is_empty() && !light_tree_alias_large.is_empty()) {
					const uint32_t s = light_tree_alias_small[light_tree_alias_small.size() - 1];
					light_tree_alias_small.resize(light_tree_alias_small.size() - 1);
					const uint32_t l = light_tree_alias_large[light_tree_alias_large.size() - 1];
					RT_LightTreeAlias &e = light_tree_leaf_alias[p_alias_base + s];
					e.threshold = (float)light_tree_alias_scaled[s];
					e.alias = l;
					light_tree_alias_scaled[l] -= 1.0 - light_tree_alias_scaled[s];
					if (light_tree_alias_scaled[l] < 1.0) {
						light_tree_alias_large.resize(light_tree_alias_large.size() - 1);
						light_tree_alias_small.push_back(l);
					}
				}
				// What is left on either list is 1 up to rounding and keeps every pick (threshold 1, its own alias).
				for (uint32_t i = first; i < first + count; i++) {
					RT_LightTreeAlias &e = light_tree_leaf_alias[p_alias_base + i];
					e.alias_share = light_tree_leaf_alias[p_alias_base + e.alias].share;
				}
			}
			for (uint32_t i = first; i < first + count; i++) {
				uint32_t encoded = light_tree_nodes[light_tree_leaf_list[i]].child;
				if (encoded & RT_LIGHT_TREE_EMISSIVE_BIT) {
					uint32_t entry = encoded & ~(RT_LIGHT_TREE_LEAF_BIT | RT_LIGHT_TREE_EMISSIVE_BIT);
					light_tree_emissive_cut[entry] = (light_tree_emissive_cut[entry] & ~(0xFFFFu << p_slot_shift)) | (slot << p_slot_shift);
				}
			}
		}
		head++;
	}
	DEV_ASSERT(r_cut.size() <= p_cut_size);
}

uint32_t RenderRaytracing::_build_light_tree_node(uint32_t p_begin, uint32_t p_end) {
	uint32_t node_index = light_tree_nodes.size();
	light_tree_nodes.push_back(RT_LightTreeNode());
	DEV_ASSERT(p_end - p_begin < 0x10000 && p_begin < 0x10000);
	if (p_end - p_begin == 1) {
		const LightTreeLeaf &leaf = light_tree_leaves[light_tree_order[p_begin]];
		RT_LightTreeNode &n = light_tree_nodes[node_index];
		n.center[0] = leaf.center.x;
		n.center[1] = leaf.center.y;
		n.center[2] = leaf.center.z;
		n.radius = leaf.radius;
		n.axis[0] = leaf.axis.x;
		n.axis[1] = leaf.axis.y;
		n.axis[2] = leaf.axis.z;
		n.cos_theta_o = leaf.cos_theta_o;
		n.power = leaf.power;
		n.max_range = leaf.max_range;
		n.child = leaf.encoded;
		n.leaf_range = (p_begin << 16) | 1u;
		const bool emissive = (leaf.encoded & RT_LIGHT_TREE_EMISSIVE_BIT) != 0;
		n.analytic_share = emissive ? 0.0f : 1.0f;
		n.decay = leaf.decay;
		n.analytic_range = emissive ? 0.0f : leaf.max_range;
		n.emissive_share = emissive ? 1.0f : 0.0f;
		light_tree_leaf_list[p_begin] = node_index;
		if (emissive) {
			// Where a chance hit on the entry finds its share under its cut node (D60).
			light_tree_emissive_leaf[leaf.encoded & ~(RT_LIGHT_TREE_LEAF_BIT | RT_LIGHT_TREE_EMISSIVE_BIT)] = p_begin;
		}
		return node_index;
	}

	// The median along the longest axis of the leaf centres.
	AABB bounds(light_tree_leaves[light_tree_order[p_begin]].center, Vector3());
	for (uint32_t i = p_begin + 1; i < p_end; i++) {
		bounds.expand_to(light_tree_leaves[light_tree_order[i]].center);
	}
	int axis = bounds.get_longest_axis_index();
	struct AxisComparator {
		const LightTreeLeaf *leaves = nullptr;
		int axis = 0;
		bool operator()(uint32_t a, uint32_t b) const {
			return leaves[a].center[axis] < leaves[b].center[axis];
		}
	};
	SortArray<uint32_t, AxisComparator> sorter;
	sorter.compare.leaves = light_tree_leaves.ptr();
	sorter.compare.axis = axis;
	sorter.sort(light_tree_order.ptr() + p_begin, p_end - p_begin);
	uint32_t mid = (p_begin + p_end) / 2;

	uint32_t left = _build_light_tree_node(p_begin, mid);
	uint32_t right = _build_light_tree_node(mid, p_end);
	DEV_ASSERT(left == node_index + 1);

	// The vector may have moved: take the references after the children are built.
	const RT_LightTreeNode &l = light_tree_nodes[left];
	const RT_LightTreeNode &r = light_tree_nodes[right];
	Vector3 center;
	float radius;
	_rt_sphere_union(Vector3(l.center[0], l.center[1], l.center[2]), l.radius, Vector3(r.center[0], r.center[1], r.center[2]), r.radius, center, radius);
	Vector3 cone_axis;
	float cone_cos;
	_rt_cone_union(Vector3(l.axis[0], l.axis[1], l.axis[2]), l.cos_theta_o, Vector3(r.axis[0], r.axis[1], r.axis[2]), r.cos_theta_o, cone_axis, cone_cos);
	RT_LightTreeNode &n = light_tree_nodes[node_index];
	n.center[0] = center.x;
	n.center[1] = center.y;
	n.center[2] = center.z;
	n.radius = radius;
	n.axis[0] = cone_axis.x;
	n.axis[1] = cone_axis.y;
	n.axis[2] = cone_axis.z;
	n.cos_theta_o = cone_cos;
	n.power = l.power + r.power;
	n.max_range = (l.max_range > 0.0f && r.max_range > 0.0f) ? MAX(l.max_range, r.max_range) : 0.0f;
	n.child = right;
	n.leaf_range = (p_begin << 16) | (p_end - p_begin);
	// The two kinds' shares of the power (D63), in double so neither kind's share is lost to the
	// other's: the shader weighs their falloffs by them with no division and no subtraction.
	const double power_sum = (double)l.power + (double)r.power;
	const double analytic_power = (double)l.analytic_share * l.power + (double)r.analytic_share * r.power;
	const double emissive_power = (double)l.emissive_share * l.power + (double)r.emissive_share * r.power;
	n.analytic_share = power_sum > 0.0 ? (float)(analytic_power / power_sum) : 0.0f;
	n.emissive_share = power_sum > 0.0 ? (float)(emissive_power / power_sum) : 0.0f;
	// The analytic lights' falloff over the children that carry them: the slowest decay, so no such
	// light falls off faster than its node reads past a metre, and the widest range.
	if (l.analytic_share > 0.0f && r.analytic_share > 0.0f) {
		n.decay = MIN(l.decay, r.decay);
		n.analytic_range = (l.analytic_range > 0.0f && r.analytic_range > 0.0f) ? MAX(l.analytic_range, r.analytic_range) : 0.0f;
	} else {
		const RT_LightTreeNode &a = l.analytic_share > 0.0f ? l : r;
		n.decay = a.decay;
		n.analytic_range = a.analytic_range;
	}
	// A node is read at its centre's distance and a light inside it can be far nearer: never steeper
	// than the inverse square, and a negative decay (capped at 1 by the light) reads flat. A leaf's
	// distance is its light's own, so a leaf keeps its decay.
	n.decay = CLAMP(n.decay, 0.0f, 2.0f);
	return node_index;
}

// ---------------------------------------------------------------------------
// Uniform set update
// ---------------------------------------------------------------------------

void RenderRaytracing::_update_scene_uniform_buffer(RTViewportState *p_state, const RenderDataRD *p_render_data) {
	// A copy of the raster scene UBO whose cameras are relative to rt_origin:
	// the trace's world is world minus the anchor, so inv_view and view carry
	// the relative origin as plain floats (no precision split) for the current
	// and the previous frame. Everything else in the UBO is copied as is.
	const uint32_t size = RenderSceneDataRD::get_uniform_buffer_size_bytes();
	if (!p_state->scene_uniform_buffer.is_valid()) {
		p_state->scene_uniform_buffer = RD::get_singleton()->uniform_buffer_create(size);
		RD::get_singleton()->set_resource_name(p_state->scene_uniform_buffer, "RT Scene Data Buffer");
	}

	struct CameraBlock {
		float inv_view_matrix[12];
		float view_matrix[12];
#ifdef REAL_T_IS_DOUBLE
		float inv_view_precision[4];
#endif
	};
	static_assert(offsetof(RenderSceneDataRD::UBO, view_matrix) == offsetof(RenderSceneDataRD::UBO, inv_view_matrix) + 12 * sizeof(float));
#ifdef REAL_T_IS_DOUBLE
	static_assert(offsetof(RenderSceneDataRD::UBO, inv_view_precision) == offsetof(RenderSceneDataRD::UBO, view_matrix) + 12 * sizeof(float));
#endif
	RenderSceneDataRD::UBODATA ubo_data = p_render_data->scene_data->ubo_data;
	RenderSceneDataRD::UBO *ubos[2] = { &ubo_data.ubo, &ubo_data.prev_ubo };
	const Transform3D *cameras[2] = { &p_render_data->scene_data->cam_transform, &p_render_data->scene_data->prev_cam_transform };
	for (uint32_t i = 0; i < 2; i++) {
		Transform3D relative = *cameras[i];
		relative.origin -= p_state->rt_origin;
		CameraBlock block = {};
		RendererRD::MaterialStorage::store_transform_transposed_3x4(relative, block.inv_view_matrix);
		RendererRD::MaterialStorage::store_transform_transposed_3x4(relative.affine_inverse(), block.view_matrix);
		memcpy(ubos[i]->inv_view_matrix, &block, sizeof(CameraBlock));
	}
	RD::get_singleton()->buffer_update(p_state->scene_uniform_buffer, 0, size, &ubo_data);

	// What sky_radiance reads from this copy, for the sky share's weights (D65): a later pass of the
	// frame can refill the raster UBO (a sky's first render moves its baked exposure).
	p_state->sky_light_scene[0] = ubo_data.ubo.IBL_exposure_normalization;
	p_state->sky_light_scene[1] = ubo_data.ubo.fog_sky_affect;
	p_state->sky_light_scene[2] = ubo_data.ubo.fog_aerial_perspective;
	p_state->sky_light_scene[3] = 0.2126f * ubo_data.ubo.fog_light_color[0] + 0.7152f * ubo_data.ubo.fog_light_color[1] + 0.0722f * ubo_data.ubo.fog_light_color[2];
}

// ---------------------------------------------------------------------------
// Decals in the hit shaders (D104): the raster's decal records with trace-space transforms, over
// every decal in the trace's cull volume, and a bounding volume hierarchy a hit walks for the
// decals around it, since the raster's froxel clusters only cover what the camera sees.
// ---------------------------------------------------------------------------

namespace {
struct DecalCentroidCompare {
	const AABB *bounds = nullptr;
	int axis = 0;
	_FORCE_INLINE_ bool operator()(uint32_t p_a, uint32_t p_b) const {
		return bounds[p_a].get_center()[axis] < bounds[p_b].get_center()[axis];
	}
};
} // namespace

uint32_t RenderRaytracing::_build_decal_node(uint32_t p_first, uint32_t p_count) {
	const uint32_t node_index = decal_nodes.size();
	decal_nodes.push_back(RT_DecalNode());

	AABB bounds = decal_bounds[decal_order[p_first]];
	AABB centroids(bounds.get_center(), Vector3());
	for (uint32_t i = 1; i < p_count; i++) {
		const AABB &b = decal_bounds[decal_order[p_first + i]];
		bounds.merge_with(b);
		centroids.expand_to(b.get_center());
	}

	uint32_t child = 0;
	uint32_t count = 0;
	if (p_count <= RT_DECAL_LEAF_SIZE) {
		// A leaf's records are stored contiguously in leaf order; each keeps its rank in `order`.
		child = RT_DECAL_LEAF_BIT | decal_leaf_records.size();
		count = p_count;
		for (uint32_t i = 0; i < p_count; i++) {
			decal_leaf_records.push_back(decal_records[decal_order[p_first + i]]);
		}
	} else {
		// A median split by centroid along the longest axis of the centroids' box.
		SortArray<uint32_t, DecalCentroidCompare> sorter;
		sorter.compare.bounds = decal_bounds.ptr();
		sorter.compare.axis = centroids.get_longest_axis_index();
		sorter.sort(decal_order.ptr() + p_first, p_count);
		const uint32_t half = p_count / 2;
		_build_decal_node(p_first, half);
		child = _build_decal_node(p_first + half, p_count - half);
	}

	RT_DecalNode &node = decal_nodes[node_index];
	node.aabb_min[0] = (float)bounds.position.x;
	node.aabb_min[1] = (float)bounds.position.y;
	node.aabb_min[2] = (float)bounds.position.z;
	node.aabb_max[0] = (float)(bounds.position.x + bounds.size.x);
	node.aabb_max[1] = (float)(bounds.position.y + bounds.size.y);
	node.aabb_max[2] = (float)(bounds.position.z + bounds.size.z);
	node.child = child;
	node.count = count;
	return node_index;
}

uint32_t RenderRaytracing::gather_decals(RTViewportState *p_state, const RenderDataRD *p_render_data) {
	decal_candidates.clear();
	if (!p_render_data || !p_render_data->rt_decals || !p_render_data->scene_data) {
		return 0;
	}
	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	const Vector3 camera_origin = p_render_data->scene_data->cam_transform.origin;

	// The raster's selection and sort (TextureStorage::update_decal_buffer): a decal past its distance
	// fade, or with neither an albedo nor an emission texture, adds nothing; the rest are ranked by
	// camera distance minus the sorting offset. Unlike the raster's buffer, a skipped decal takes no slot.
	const PagedArray<RID> &decals = *p_render_data->rt_decals;
	for (uint64_t i = 0; i < decals.size(); i++) {
		const RID instance = decals[i];
		if (!texture_storage->owns_decal_instance(instance)) {
			continue;
		}
		const RID decal = texture_storage->decal_instance_get_base(instance);
		if (!texture_storage->owns_decal(decal)) {
			continue;
		}
		if (!texture_storage->decal_get_texture(decal, RSE::DECAL_TEXTURE_ALBEDO).is_valid() && !texture_storage->decal_get_texture(decal, RSE::DECAL_TEXTURE_EMISSION).is_valid()) {
			continue;
		}
		const real_t distance = camera_origin.distance_to(texture_storage->decal_instance_get_transform(instance).origin);
		if (texture_storage->decal_is_distance_fade_enabled(decal) && distance > texture_storage->decal_get_distance_fade_begin(decal) + texture_storage->decal_get_distance_fade_length(decal)) {
			continue;
		}
		DecalCandidate candidate;
		candidate.instance = instance;
		candidate.depth = float(distance - texture_storage->decal_instance_get_sorting_offset(instance));
		decal_candidates.push_back(candidate);
	}
	if (decal_candidates.is_empty()) {
		return 0;
	}
	decal_candidates.sort();

	uint32_t count = decal_candidates.size();
	if (count > RT_DECALS_MAX) {
		WARN_PRINT_ONCE(vformat("RT: %d decals in the trace's range, past its capacity of %d; the farthest are not traced.", count, int(RT_DECALS_MAX)));
		count = RT_DECALS_MAX;
	}

	Transform3D uv_xform;
	uv_xform.basis.scale(Vector3(2.0, 1.0, 2.0));
	uv_xform.origin = Vector3(-1.0, 0.0, -1.0);

	decal_records.resize(count);
	decal_bounds.resize(count);
	decal_order.resize(count);
	for (uint32_t i = 0; i < count; i++) {
		const RID instance = decal_candidates[i].instance;
		const RID decal = texture_storage->decal_instance_get_base(instance);
		const Transform3D xform = texture_storage->decal_instance_get_transform(instance);
		const Vector3 decal_extents = texture_storage->decal_get_size(decal) / 2;

		float fade = 1.0f;
		if (texture_storage->decal_is_distance_fade_enabled(decal)) {
			const real_t distance = camera_origin.distance_to(xform.origin);
			const float fade_begin = texture_storage->decal_get_distance_fade_begin(decal);
			const float fade_length = texture_storage->decal_get_distance_fade_length(decal);
			if (distance > fade_begin) {
				fade = Math::smoothstep(0.0f, 1.0f, 1.0f - float(distance - fade_begin) / fade_length);
			}
		}

		// Trace space is world space relative to this frame's anchor, taken in double before the cast.
		Transform3D relative = xform;
		relative.origin -= p_state->rt_origin;
		Transform3D scale_xform;
		scale_xform.basis.scale(decal_extents);

		RT_DecalData &r = decal_records[i];
		r = RT_DecalData();
		RendererRD::MaterialStorage::store_transform_transposed_3x4((relative * scale_xform * uv_xform).affine_inverse(), r.xform);

		const RID textures[4] = {
			texture_storage->decal_get_texture(decal, RSE::DECAL_TEXTURE_ALBEDO),
			texture_storage->decal_get_texture(decal, RSE::DECAL_TEXTURE_NORMAL),
			texture_storage->decal_get_texture(decal, RSE::DECAL_TEXTURE_ORM),
			texture_storage->decal_get_texture(decal, RSE::DECAL_TEXTURE_EMISSION),
		};
		float *rects[4] = { r.albedo_rect, r.normal_rect, r.orm_rect, r.emission_rect };
		for (int t = 0; t < 4; t++) {
			if (textures[t].is_valid()) {
				const Rect2 rect = texture_storage->decal_atlas_get_texture_rect(textures[t]);
				rects[t][0] = rect.position.x;
				rects[t][1] = rect.position.y;
				rects[t][2] = rect.size.x;
				rects[t][3] = rect.size.y;
			}
		}

		const Basis normal_xform = xform.basis.orthonormalized();
		for (int row = 0; row < 3; row++) {
			r.normal_xform[row * 4 + 0] = normal_xform.rows[row][0];
			r.normal_xform[row * 4 + 1] = normal_xform.rows[row][1];
			r.normal_xform[row * 4 + 2] = normal_xform.rows[row][2];
		}
		const Vector3 normal = xform.basis.get_column(Vector3::AXIS_Y).normalized();
		r.normal[0] = normal.x;
		r.normal[1] = normal.y;
		r.normal[2] = normal.z;

		const Color modulate = texture_storage->decal_get_modulate(decal).srgb_to_linear();
		r.modulate[0] = modulate.r;
		r.modulate[1] = modulate.g;
		r.modulate[2] = modulate.b;
		r.modulate[3] = modulate.a * fade;
		r.mask = texture_storage->decal_get_cull_mask(decal);
		r.emission_energy = texture_storage->decal_get_emission_energy(decal) * fade;
		r.albedo_mix = texture_storage->decal_get_albedo_mix(decal);
		r.upper_fade = texture_storage->decal_get_upper_fade(decal);
		r.lower_fade = texture_storage->decal_get_lower_fade(decal);
		r.normal_fade = texture_storage->decal_get_normal_fade(decal);
		r.order = i;

		// The decal's box in trace space, grown a little so a point on its face passes the float test.
		decal_bounds[i] = relative.xform(AABB(-decal_extents, decal_extents * 2.0)).grow(0.001);
		decal_order[i] = i;
	}

	decal_leaf_records.clear();
	decal_nodes.clear();
	_build_decal_node(0, count);

	// Made with the first decal; a frame whose buffers cannot be made traces no decals.
	if (!p_state->decal_buffer.is_valid()) {
		p_state->decal_buffer = RD::get_singleton()->storage_buffer_create(RT_DECALS_MAX * sizeof(RT_DecalData));
		ERR_FAIL_COND_V_MSG(!p_state->decal_buffer.is_valid(), 0, "RT: the decal buffer could not be created; decals are not traced.");
		RD::get_singleton()->set_resource_name(p_state->decal_buffer, "RT Decal Buffer");
	}
	if (!p_state->decal_node_buffer.is_valid()) {
		p_state->decal_node_buffer = RD::get_singleton()->storage_buffer_create(RT_DECAL_NODES_MAX * sizeof(RT_DecalNode));
		ERR_FAIL_COND_V_MSG(!p_state->decal_node_buffer.is_valid(), 0, "RT: the decal hierarchy buffer could not be created; decals are not traced.");
		RD::get_singleton()->set_resource_name(p_state->decal_node_buffer, "RT Decal Node Buffer");
	}
	RD::get_singleton()->buffer_update(p_state->decal_buffer, 0, decal_leaf_records.size() * sizeof(RT_DecalData), decal_leaf_records.ptr());
	RD::get_singleton()->buffer_update(p_state->decal_node_buffer, 0, decal_nodes.size() * sizeof(RT_DecalNode), decal_nodes.ptr());
	return count;
}

RID RenderRaytracing::update_uniform_set(RTViewportState *p_state, const RenderDataRD *p_render_data, uint32_t p_rt_flags, bool p_primary_from_gbuffer) {
	ERR_FAIL_NULL_V(p_state, RID());

	Ref<RenderForwardClustered::RenderBufferDataForwardClustered> rb_data;
	if (p_render_data && p_render_data->render_buffers.is_valid()) {
		if (p_render_data->render_buffers->has_custom_data(RB_SCOPE_FORWARD_CLUSTERED)) {
			rb_data = p_render_data->render_buffers->get_custom_data(RB_SCOPE_FORWARD_CLUSTERED);
		}
	}

	if (rb_data.is_null()) {
		return RID();
	}

	RenderSceneBuffersRD *rb = p_render_data->render_buffers.ptr();

	// SET 0 indices must match raytracing_common_inc.glsl / scene_raytracing_raygen.glsl / samplers includes.
	Vector<RD::Uniform> uniforms;

	{
		RD::Uniform u;
		u.binding = 0;
		u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
		rt_ensure_textures(rb);
		u.append_id(rt_get_texture(rb));
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.binding = 1;
		u.uniform_type = RD::UNIFORM_TYPE_ACCELERATION_STRUCTURE;
		ERR_FAIL_COND_V(p_state->tlas == RID(), RID());
		u.append_id(p_state->tlas);
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.binding = 2;
		u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
		_update_scene_uniform_buffer(p_state, p_render_data);
		u.append_id(p_state->scene_uniform_buffer);
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.binding = 3;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		if (p_state->geometry_buffer.is_valid()) {
			u.append_id(p_state->geometry_buffer);
		} else {
			// Use a default buffer if no geometry
			u.append_id(RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer());
		}
		uniforms.push_back(u);
	}

	// Binding 4: Per-instance motion index buffer (int32 per TLAS instance, -1 = no motion).
	{
		RD::Uniform u;
		u.binding = 4;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		if (p_state->motion_index_buffer.is_valid()) {
			u.append_id(p_state->motion_index_buffer);
		} else {
			u.append_id(RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer());
		}
		uniforms.push_back(u);
	}

	// Motion transforms past sampler block growth reservation (bindings 28-31).
	{
		RD::Uniform u;
		u.binding = 32;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		if (p_state->motion_transform_buffer.is_valid()) {
			u.append_id(p_state->motion_transform_buffer);
		} else {
			u.append_id(RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer());
		}
		uniforms.push_back(u);
	}

	// Binding 5: Material buffer.
	{
		RD::Uniform u;
		u.binding = 5;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		if (p_state->material_buffer.is_valid()) {
			u.append_id(p_state->material_buffer);
		} else {
			u.append_id(RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer());
		}
		uniforms.push_back(u);
	}

	// Binding 6: Raytracing params + unjittered VP matrices.
	{
		struct {
			//TODO: I see no reason why we need to pass this as a float array
			float params[SceneShaderRaytracing::RT_PARAM_COUNT];
			float prev_vp_unjittered[16];
			float curr_vp_unjittered[16];
			float rt_origin[4];
			float camera_medium_absorption[4];
			float camera_medium_scatter[4];
		} rt_ubo = {};
		static_assert(sizeof(rt_ubo) == 68 * sizeof(float));
		// Without an Environment the trace keeps the diffuse bounce limit it had before it was a knob (D74); a 0 would end every path at its first hit.
		rt_ubo.params[SceneShaderRaytracing::RT_PARAM_MAX_DIFFUSE_BOUNCES] = 2.0f;

		if (p_render_data && p_render_data->environment.is_valid()) {
			RendererEnvironmentStorage *env_storage = RendererEnvironmentStorage::get_singleton();
			RID env = p_render_data->environment;
			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_VIS_MODE] = (float)env_storage->environment_get_pathtracing_debug_mode(env);
			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_SAMPLE_COUNT] = (float)env_storage->environment_get_pathtracing_samples_per_pixel(env);
			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_MAX_BOUNCES] = (float)env_storage->environment_get_pathtracing_max_bounces(env);
			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_MAX_DIFFUSE_BOUNCES] = (float)env_storage->environment_get_pathtracing_max_diffuse_bounces(env);
			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_DENOISER] = (float)(int)env_storage->environment_get_pathtracing_denoiser(env);
			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_CLAMP_DIRECT] = env_storage->environment_get_pathtracing_clamp_direct(env);
			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_CLAMP_INDIRECT] = env_storage->environment_get_pathtracing_clamp_indirect(env);

			// The medium the camera is inside of (Environment.camera_medium_*, D49): the raygen seeds
			// every path with it and composes no fog over the camera segment. Its top is handed over
			// in trace space, relative to this frame's anchor, in double before the cast.
			if (env_storage->environment_get_camera_medium_enabled(env)) {
				// The absorption is a coefficient and goes up raw; the scattering is a colour and follows
				// the Environment's convention (sRGB authored, linear uploaded), as the shader's
				// source_color scatter_color does through the material packer.
				Color absorption = env_storage->environment_get_camera_medium_absorption(env);
				Color scattering = env_storage->environment_get_camera_medium_scattering(env).srgb_to_linear();
				rt_ubo.camera_medium_absorption[0] = MAX(absorption.r, 0.0f);
				rt_ubo.camera_medium_absorption[1] = MAX(absorption.g, 0.0f);
				rt_ubo.camera_medium_absorption[2] = MAX(absorption.b, 0.0f);
				rt_ubo.camera_medium_absorption[3] = 1.0f;
				rt_ubo.camera_medium_scatter[0] = CLAMP(scattering.r, 0.0f, 1.0f);
				rt_ubo.camera_medium_scatter[1] = CLAMP(scattering.g, 0.0f, 1.0f);
				rt_ubo.camera_medium_scatter[2] = CLAMP(scattering.b, 0.0f, 1.0f);
				rt_ubo.camera_medium_scatter[3] = (float)((real_t)env_storage->environment_get_camera_medium_height(env) - p_state->rt_origin.y);
			}
		}
		// The frame decides (the Environment's choice, unless MSAA forced the traced primary ray).
		rt_ubo.params[SceneShaderRaytracing::RT_PARAM_PRIMARY_SURFACE] = p_primary_from_gbuffer ? (float)RSE::PT_PRIMARY_SURFACE_GBUFFER : (float)RSE::PT_PRIMARY_SURFACE_TRACED;
		rt_ubo.params[SceneShaderRaytracing::RT_PARAM_HAS_PROCEDURAL_INSTANCES] = p_state->has_procedural_instances ? 1.0f : 0.0f;

		// The froxel volumetric fog the raygen composes over the camera segment (D37): the froxel's
		// depth mapping as the raster's fragment reads it, and the Environment's sky affect.
		{
			Ref<RendererRD::Fog::VolumetricFog> vfog;
			if (rb->has_custom_data(RB_SCOPE_FOG)) {
				vfog = rb->get_custom_data(RB_SCOPE_FOG);
			}
			// Gate on the integrated map, not on the object: a VolumetricFog whose map failed to
			// allocate would otherwise have the raygen compose the stand-in texture over the frame.
			const bool has_vfog = vfog.is_valid() && vfog->fog_map.is_valid();
			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_VFOG_ENABLED] = has_vfog ? 1.0f : 0.0f;
			if (has_vfog) {
				rt_ubo.params[SceneShaderRaytracing::RT_PARAM_VFOG_INV_LENGTH] = vfog->length > 0.0f ? 1.0f / vfog->length : 1.0f;
				rt_ubo.params[SceneShaderRaytracing::RT_PARAM_VFOG_INV_SPREAD] = vfog->spread > 0.0f ? 1.0f / vfog->spread : 1.0f;
				rt_ubo.params[SceneShaderRaytracing::RT_PARAM_VFOG_SKY_AFFECT] = (p_render_data && p_render_data->environment.is_valid()) ? RendererEnvironmentStorage::get_singleton()->environment_get_volumetric_fog_sky_affect(p_render_data->environment) : 0.0f;
			}
		}

		// rt_params layout (see RaytracingParamIndex enum):
		// [0] = VIS_MODE, [1] = SAMPLE_COUNT, [2] = MAX_BOUNCES,
		// [3] = DLSS_RR_ENABLED, [14] = LIGHT_COUNT, [15] = FRAME_INDEX
		rt_ubo.params[SceneShaderRaytracing::RT_PARAM_FRAME_INDEX] = float(p_state->frame_counter++);

		// Unjittered VP for motion vectors (matches raster convention).
		{
			Projection correction;
			correction.set_depth_correction(true);

			// Both cameras relative to this frame's anchor, so the previous
			// frame reprojects correctly across an anchor change.
			Transform3D prev_cam = p_render_data->scene_data->prev_cam_transform;
			prev_cam.origin -= p_state->rt_origin;
			Transform3D curr_cam = p_render_data->scene_data->cam_transform;
			curr_cam.origin -= p_state->rt_origin;

			Projection prev_vp = (correction * p_render_data->scene_data->prev_cam_projection) * Projection(prev_cam.affine_inverse());
			RendererRD::MaterialStorage::store_camera(prev_vp, rt_ubo.prev_vp_unjittered);

			Projection curr_vp = (correction * p_render_data->scene_data->cam_projection) * Projection(curr_cam.affine_inverse());
			RendererRD::MaterialStorage::store_camera(curr_vp, rt_ubo.curr_vp_unjittered);

			rt_ubo.rt_origin[0] = (float)p_state->rt_origin.x;
			rt_ubo.rt_origin[1] = (float)p_state->rt_origin.y;
			rt_ubo.rt_origin[2] = (float)p_state->rt_origin.z;
			rt_ubo.rt_origin[3] = 0.0f;
		}

		// --- Light gathering ---
		// The light tree (D55) selects over the whole buffer; the uniform pick keeps its cap.
		const bool light_tree = (p_rt_flags & SceneShaderRaytracing::RT_FLAG_LIGHT_TREE) != 0;
		const uint32_t light_cap = light_tree ? RT_LIGHTS_MAX : RT_LIGHTS_UNIFORM_MAX;
		if (light_scratch.size() < RT_LIGHTS_MAX) {
			light_scratch.resize(RT_LIGHTS_MAX);
		}
		RT_LightData *rt_light_data = light_scratch.ptr();
		memset(rt_light_data, 0, light_cap * sizeof(RT_LightData));

		float light_score_sum = 0.0f;
		uint32_t rt_directional_count = 0;
		uint32_t rt_light_count = gather_lights(p_render_data, rt_light_data, light_cap, p_state->rt_origin, light_score_sum, rt_directional_count);

		rt_ubo.params[SceneShaderRaytracing::RT_PARAM_LIGHT_COUNT] = float(rt_light_count);
		rt_ubo.params[SceneShaderRaytracing::RT_PARAM_DIRECTIONAL_COUNT] = float(rt_directional_count);

		// The decals the hit shaders apply (D104), with their hierarchy; none skips the walk.
		rt_ubo.params[SceneShaderRaytracing::RT_PARAM_DECAL_COUNT] = float(gather_decals(p_state, p_render_data));

		// Upload light buffer: the cap's entries, as the uniform pick always had (zeroed past the count).
		{
			uint32_t buf_size = RT_LIGHTS_MAX * sizeof(RT_LightData);
			if (!p_state->light_buffer.is_valid()) {
				p_state->light_buffer = RD::get_singleton()->storage_buffer_create(buf_size);
				RD::get_singleton()->set_resource_name(p_state->light_buffer, "RT Light Buffer");
			}
			RD::get_singleton()->buffer_update(p_state->light_buffer, 0, light_cap * sizeof(RT_LightData), rt_light_data);
		}

		// --- Emissive mesh lights (D50) ---
		{
			bool emissive_enabled = true;
			int emissive_candidate_count = 8;
			int light_tree_cut_size = 32;
			int light_tree_cut_indirect_size = 32;
			if (p_render_data && p_render_data->environment.is_valid()) {
				emissive_enabled = RendererEnvironmentStorage::get_singleton()->environment_get_pathtracing_emissive_lights(p_render_data->environment);
				emissive_candidate_count = RendererEnvironmentStorage::get_singleton()->environment_get_pathtracing_emissive_candidates(p_render_data->environment);
				light_tree_cut_size = RendererEnvironmentStorage::get_singleton()->environment_get_pathtracing_light_tree_cut(p_render_data->environment);
				light_tree_cut_indirect_size = RendererEnvironmentStorage::get_singleton()->environment_get_pathtracing_light_tree_cut_indirect(p_render_data->environment);
			}
			uint32_t buf_size = RT_EMISSIVE_MAX * sizeof(RT_EmissiveInstance);
			if (!p_state->emissive_buffer.is_valid()) {
				p_state->emissive_buffer = RD::get_singleton()->storage_buffer_create(buf_size);
				RD::get_singleton()->set_resource_name(p_state->emissive_buffer, "RT Emissive Buffer");
			}
			uint32_t rt_emissive_count = 0;
			float emissive_score_sum = 0.0f;
			if (emissive_enabled) {
				if (emissive_scratch.size() < RT_EMISSIVE_MAX) {
					emissive_scratch.resize(RT_EMISSIVE_MAX);
				}
				rt_emissive_count = gather_emissive(p_render_data, p_state, emissive_scratch.ptr(), RT_EMISSIVE_MAX, emissive_score_sum);
				if (rt_emissive_count > 0) {
					RD::get_singleton()->buffer_update(p_state->emissive_buffer, 0, rt_emissive_count * sizeof(RT_EmissiveInstance), emissive_scratch.ptr());
				}
			}

			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_EMISSIVE_COUNT] = float(rt_emissive_count);
			// One light sample per hit (D50): the share that goes to the emissive list follows the two
			// lists' summed camera-weighted scores, clamped so neither strategy starves. Any share in
			// (0, 1) is unbiased; the split only moves the noise between the two kinds of light.
			float emissive_probability = 0.0f;
			if (rt_emissive_count > 0) {
				emissive_probability = (rt_light_count > 0) ? CLAMP(emissive_score_sum / MAX(emissive_score_sum + light_score_sum, 1.0e-20f), 0.1f, 0.9f) : 1.0f;
			}
			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_EMISSIVE_PROBABILITY] = emissive_probability;
			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_EMISSIVE_CANDIDATES] = float(CLAMP(emissive_candidate_count, 1, 16));
			// The sky's share (D54) is judged in the shader against both lists' summed scores, since
			// the sky's own total lives in the table the GPU builds.
			rt_ubo.params[SceneShaderRaytracing::RT_PARAM_SKY_SURFACE_SCORE] = light_score_sum + emissive_score_sum;
			if (rt_emissive_count != p_state->emissive_count_logged) {
				// Once per change, so a run's log shows whether the list was ever non-empty and its share.
				print_line(vformat("Path tracer: %d emissive instance(s) sampled as lights, share %.2f of the light samples.", rt_emissive_count, emissive_probability));
				p_state->emissive_count_logged = rt_emissive_count;
			}

			// --- The light tree (D55) ---
			// Rebuilt every frame the Environment asks for it (the lights move with the anchor); a
			// cleared buffer reads as no tree, which the shader treats as no positional light.
			{
				if (!p_state->light_tree_buffer.is_valid()) {
					p_state->light_tree_buffer = RD::get_singleton()->storage_buffer_create(RT_LIGHT_TREE_BUFFER_SIZE);
					RD::get_singleton()->set_resource_name(p_state->light_tree_buffer, "RT Light Tree Buffer");
					RD::get_singleton()->buffer_clear(p_state->light_tree_buffer, 0, RT_LIGHT_TREE_BUFFER_SIZE);
				}
				if (light_tree) {
					const bool power_pick = (p_rt_flags & SceneShaderRaytracing::RT_FLAG_LIGHT_TREE_POWER_PICK) != 0;
					float exposure = (p_render_data && p_render_data->camera_attributes.is_valid()) ? RSG::camera_attributes->camera_attributes_get_exposure_normalization_factor(p_render_data->camera_attributes) : 1.0f;
					uint32_t node_count = build_light_tree(rt_light_data, rt_light_count, rt_directional_count, emissive_scratch.ptr(), rt_emissive_count, exposure, (uint32_t)MAX(light_tree_cut_size, 1), (uint32_t)MAX(light_tree_cut_indirect_size, 1), power_pick);
					uint32_t cut_count = node_count > 0 ? light_tree_cut.size() : 0;
					uint32_t cut_indirect_count = node_count > 0 ? light_tree_cut_indirect.size() : 0;
					uint32_t leaf_count = node_count > 0 ? light_tree_leaf_list.size() : 0;
					uint32_t header[4] = { node_count, cut_count, leaf_count, cut_indirect_count };
					RD::get_singleton()->buffer_update(p_state->light_tree_buffer, 0, sizeof(header), header);
					if (node_count > 0) {
						RD::get_singleton()->buffer_update(p_state->light_tree_buffer, RT_LIGHT_TREE_HEADER_SIZE, node_count * sizeof(RT_LightTreeNode), light_tree_nodes.ptr());
						RD::get_singleton()->buffer_update(p_state->light_tree_buffer, RT_LIGHT_TREE_CUT_OFFSET, cut_count * sizeof(RT_LightTreeNode), light_tree_cut.ptr());
						RD::get_singleton()->buffer_update(p_state->light_tree_buffer, RT_LIGHT_TREE_CUT_INDIRECT_OFFSET, cut_indirect_count * sizeof(RT_LightTreeNode), light_tree_cut_indirect.ptr());
						RD::get_singleton()->buffer_update(p_state->light_tree_buffer, RT_LIGHT_TREE_LEAF_LIST_OFFSET, leaf_count * sizeof(uint32_t), light_tree_leaf_list.ptr());
					}
					if (node_count > 0 && power_pick) {
						RD::get_singleton()->buffer_update(p_state->light_tree_buffer, RT_LIGHT_TREE_LEAF_ALIAS_OFFSET, leaf_count * sizeof(RT_LightTreeAlias), light_tree_leaf_alias.ptr());
						RD::get_singleton()->buffer_update(p_state->light_tree_buffer, RT_LIGHT_TREE_LEAF_ALIAS_OFFSET + RT_LIGHT_TREE_LEAVES_MAX * sizeof(RT_LightTreeAlias), leaf_count * sizeof(RT_LightTreeAlias), light_tree_leaf_alias.ptr() + RT_LIGHT_TREE_LEAVES_MAX);
					}
					if (rt_emissive_count > 0) {
						RD::get_singleton()->buffer_update(p_state->light_tree_buffer, RT_LIGHT_TREE_EMISSIVE_CUT_OFFSET, rt_emissive_count * sizeof(uint32_t), light_tree_emissive_cut.ptr());
						RD::get_singleton()->buffer_update(p_state->light_tree_buffer, RT_LIGHT_TREE_EMISSIVE_LEAF_OFFSET, rt_emissive_count * sizeof(uint32_t), light_tree_emissive_leaf.ptr());
					}
				}
			}
		}

		if (!p_state->params_buffer.is_valid()) {
			p_state->params_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(rt_ubo));
			RD::get_singleton()->set_resource_name(p_state->params_buffer, "RT Params Buffer");
		}
		RD::get_singleton()->buffer_update(p_state->params_buffer, 0, sizeof(rt_ubo), &rt_ubo);

		RD::Uniform u;
		u.binding = 6;
		u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
		u.append_id(p_state->params_buffer);
		uniforms.push_back(u);
	}

	// Binding 7: Sky radiance octahedral map (for pathtracing sky sampling).
	{
		RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
		const bool use_octmap_array = owner->is_using_radiance_octmap_array();
		RID radiance_texture;

		if (p_render_data && p_render_data->environment.is_valid()) {
			RID sky_rid = owner->environment_get_sky(p_render_data->environment);
			if (sky_rid.is_valid()) {
				radiance_texture = use_octmap_array
						? owner->sky.sky_get_radiance_texture_rd(sky_rid)
						: owner->sky.sky_get_radiance_2d_texture_rd(sky_rid);
			}
		}

		// Fall back to a default whose type matches the shader's declared binding.
		if (!radiance_texture.is_valid()) {
			radiance_texture = texture_storage->texture_rd_get_default(use_octmap_array
							? RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_2D_ARRAY_BLACK
							: RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
		}

		RD::Uniform u;
		u.binding = 7;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.append_id(radiance_texture);
		uniforms.push_back(u);
	}

	// Binding 8: Sampler for radiance texture (linear filtering with mipmaps and clamp).
	{
		RD::Uniform u;
		u.binding = 8;
		u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
		u.append_id(RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(
				RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
		uniforms.push_back(u);
	}

	// Bindings 9-12: DLSS Ray Reconstruction output buffers (only in DLSS RR shader variant).
	bool dlss_rr_enabled = dlss_rr_has_buffers(rb);
	if (dlss_rr_enabled) {
		// Binding 9: DLSS RR Diffuse Albedo
		{
			RD::Uniform u;
			u.binding = 9;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.append_id(dlss_rr_get_diffuse_albedo(rb));
			uniforms.push_back(u);
		}

		// Binding 10: DLSS RR Specular Albedo
		{
			RD::Uniform u;
			u.binding = 10;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.append_id(dlss_rr_get_specular_albedo(rb));
			uniforms.push_back(u);
		}

		// Binding 11: DLSS RR Normal + Roughness
		{
			RD::Uniform u;
			u.binding = 11;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.append_id(dlss_rr_get_normal_roughness(rb));
			uniforms.push_back(u);
		}

		// Binding 12: DLSS RR Specular Hit Distance
		{
			RD::Uniform u;
			u.binding = 12;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.append_id(dlss_rr_get_specular_hit_dist(rb));
			uniforms.push_back(u);
		}
	}

	// Binding 13: Light buffer (SSBO).
	{
		RD::Uniform u;
		u.binding = 13;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		if (p_state->light_buffer.is_valid()) {
			u.append_id(p_state->light_buffer);
		} else {
			u.append_id(RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer());
		}
		uniforms.push_back(u);
	}

	// Binding 14: Global shader uniforms SSBO.
	{
		RD::Uniform u;
		u.binding = 14;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		RID buf = RendererRD::MaterialStorage::get_singleton()->global_shader_uniforms_get_storage_buffer();
		if (buf.is_valid()) {
			u.append_id(buf);
		} else {
			u.append_id(RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer());
		}
		uniforms.push_back(u);
	}

	// Binding 15: RT depth output (R32F storage image for writing depth from closest_hit/miss).
	{
		RD::Uniform u;
		u.binding = 15;
		u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
		u.append_id(rt_get_depth_texture(rb));
		uniforms.push_back(u);
	}

	// Bindings 16-27: Material samplers (12 filter/repeat combinations for custom shaders).
	RendererRD::MaterialStorage::get_singleton()->samplers_rd_get_default().append_uniforms(uniforms, 16);

	// Binding 28: Velocity output (RG16F). Past the 16-27 sampler range.
	{
		rb->ensure_velocity();
		RD::Uniform u;
		u.binding = 28;
		u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
		u.append_id(rb->get_velocity_buffer(false));
		uniforms.push_back(u);
	}

	// Bindings 33-38 and 40-41: the G-buffer primary surface and the depth it was rasterized with,
	// read by the raygen when the frame shades from the G-buffer (the medium's two targets sit
	// past the froxel at 39, D49). The raygen declares them in every variant, so a traced frame
	// binds black stand-ins.
	{
		bool has_gbuffer = p_primary_from_gbuffer && rt_has_gbuffer_textures(rb);
		RID black = RendererRD::TextureStorage::get_singleton()->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
		for (int i = 0; i < RT_GBUFFER_TARGETS; i++) {
			RD::Uniform u;
			u.binding = i < 5 ? 33 + i : 40 + (i - 5);
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.append_id(has_gbuffer ? rb->get_texture(RB_SCOPE_RT_GBUFFER, rt_gbuffer_texture_name(i)) : black);
			uniforms.push_back(u);
		}
		RD::Uniform u;
		u.binding = 38;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.append_id(has_gbuffer ? rb->get_depth_texture() : black);
		uniforms.push_back(u);
	}

	// Binding 39: the froxel volumetric fog (in-scatter and transmittance per froxel) the raygen
	// composes over the camera segment (D37); a transparent stand-in when the Environment has none.
	{
		RID fog_map;
		if (rb->has_custom_data(RB_SCOPE_FOG)) {
			Ref<RendererRD::Fog::VolumetricFog> vfog = rb->get_custom_data(RB_SCOPE_FOG);
			fog_map = vfog->fog_map;
		}
		RD::Uniform u;
		u.binding = 39;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		// The same stand-in the raster binds when a frame carries no froxel volume; neither samples
		// it, since both gate on the volume existing first.
		u.append_id(fog_map.is_valid() ? fog_map : RendererRD::TextureStorage::get_singleton()->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_3D_WHITE));
		uniforms.push_back(u);
	}

	// Binding 42: the emissive mesh lights (D50).
	{
		RD::Uniform u;
		u.binding = 42;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		if (p_state->emissive_buffer.is_valid()) {
			u.append_id(p_state->emissive_buffer);
		} else {
			u.append_id(RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer());
		}
		uniforms.push_back(u);
	}

	// Binding 43: the sky's sampling table (D54), filled by update_sky_light once the octmap is
	// rendered; zero until then, which the shader reads as no sky light.
	{
		if (!p_state->sky_light_buffer.is_valid()) {
			p_state->sky_light_buffer = RD::get_singleton()->storage_buffer_create(RT_SKY_LIGHT_BUFFER_SIZE);
			RD::get_singleton()->set_resource_name(p_state->sky_light_buffer, "RT Sky Light Buffer");
			RD::get_singleton()->buffer_clear(p_state->sky_light_buffer, 0, RT_SKY_LIGHT_BUFFER_SIZE);
		}
		RD::Uniform u;
		u.binding = 43;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.append_id(p_state->sky_light_buffer);
		uniforms.push_back(u);
	}

	// Binding 44: the light tree (D55), rebuilt above while the Environment asks for it.
	{
		RD::Uniform u;
		u.binding = 44;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.append_id(p_state->light_tree_buffer);
		uniforms.push_back(u);
	}

	// Binding 45: the cluster geometry records (D93b), which sit in the LOD table; a table the
	// population grows this frame is bound here as the new buffer, since the set is made after it.
	// Bound only while geometry streams: a bound table is a declared read, which orders the
	// selection's writes to it behind the trace.
	{
		RD::Uniform u;
		u.binding = 45;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		if (cluster_lod_stream_geometry_enabled && cluster_lod_table.is_valid()) {
			u.append_id(cluster_lod_table);
		} else {
			u.append_id(RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer());
		}
		uniforms.push_back(u);
	}
	// Bindings 47-51: the decals (D104), their hierarchy, the raster's decal atlas in its two views and
	// the decal filter's sampler. Bound whether or not the frame has decals; the count gates the reads.
	{
		RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
		const RID default_buffer = RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer();
		const RID white = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_WHITE);
		const RID atlas = texture_storage->decal_atlas_get_texture();
		const RID atlas_srgb = texture_storage->decal_atlas_get_texture_srgb();

		RD::Uniform records;
		records.binding = 47;
		records.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		records.append_id(p_state->decal_buffer.is_valid() ? p_state->decal_buffer : default_buffer);
		uniforms.push_back(records);

		RD::Uniform nodes;
		nodes.binding = 48;
		nodes.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		nodes.append_id(p_state->decal_node_buffer.is_valid() ? p_state->decal_node_buffer : default_buffer);
		uniforms.push_back(nodes);

		RD::Uniform atlas_unorm;
		atlas_unorm.binding = 49;
		atlas_unorm.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		atlas_unorm.append_id(atlas.is_valid() ? atlas : white);
		uniforms.push_back(atlas_unorm);

		RD::Uniform atlas_srgb_view;
		atlas_srgb_view.binding = 50;
		atlas_srgb_view.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		atlas_srgb_view.append_id(atlas_srgb.is_valid() ? atlas_srgb : white);
		uniforms.push_back(atlas_srgb_view);

		// The raster's decal sampler (render_forward_clustered.cpp, the render pass set's binding 10).
		RSE::CanvasItemTextureFilter filter = RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS;
		switch (owner->decals_get_filter()) {
			case RSE::DECAL_FILTER_NEAREST:
				filter = RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST;
				break;
			case RSE::DECAL_FILTER_LINEAR:
				filter = RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR;
				break;
			case RSE::DECAL_FILTER_NEAREST_MIPMAPS:
				filter = RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS;
				break;
			case RSE::DECAL_FILTER_LINEAR_MIPMAPS:
				filter = RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS;
				break;
			case RSE::DECAL_FILTER_NEAREST_MIPMAPS_ANISOTROPIC:
				filter = RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC;
				break;
			case RSE::DECAL_FILTER_LINEAR_MIPMAPS_ANISOTROPIC:
				filter = RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC;
				break;
		}
		RD::Uniform sampler;
		sampler.binding = 51;
		sampler.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
		sampler.append_id(RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(filter, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
		uniforms.push_back(sampler);
	}

	// Binding 46: the position-fetch check (D93a), declared only by a verbose run's shaders.
	if (position_fetch_check_buffer.is_valid()) {
		RD::Uniform u;
		u.binding = 46;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.append_id(position_fetch_check_buffer);
		uniforms.push_back(u);
	}

	RID shader_rd = shader ? shader->get_pipeline_shader_rd(p_rt_flags) : RID();

	RID result;
	if (shader_rd.is_valid()) {
		// Release the previous frame's scene uniform set now (before allocating
		// this frame's). Disposal is deferred by RenderingDevice until the frame
		// that used it has finished on the GPU, so this is safe and prevents the
		// uniform_set_owner pool from leaking one RID per accumulated frame.
		//
		// The set references render-buffer resources, so RenderingDevice may have
		// already auto-freed it via its dependency cascade (e.g. a texture/buffer
		// it referenced was recreated). RID::is_valid() only checks for non-null,
		// so we must query uniform_set_is_valid() to avoid a double free.
		if (p_state->scene_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(p_state->scene_uniform_set)) {
			RD::get_singleton()->free_rid(p_state->scene_uniform_set);
		}
		p_state->scene_uniform_set = RID();

		result = RD::get_singleton()->uniform_set_create(
				uniforms,
				shader_rd,
				RenderForwardClustered::SCENE_UNIFORM_SET,
				/*p_linear_pool=*/true);
		p_state->scene_uniform_set = result;

		// === SET 1: Bindless textures ===
		if (bindless_block && bindless_block->is_initialized()) {
			bindless_block->finalize(shader_rd, 1);
			bindless_uniform_set = bindless_block->get_uniform_set();
		}
	}

	return result;
}

// ---------------------------------------------------------------------------
// Trace-time buffer dependencies
// ---------------------------------------------------------------------------

// The sky's sampling table (D54): rebuilt when the Environment's sky or the count of its
// renders into the octmap changes, as a pyramid over the octmap's sharpest layer (D69): its
// leaves, one thread per texel, the levels above one dispatch each while they hold more nodes
// than one workgroup, then the rest and the header in one workgroup; cleared when the sky goes,
// so the shader's share reads 0 and nothing samples a table that is stale.
// Under the share by variance (D62) its moments follow (D65): for every receiver normal of the
// grid, one workgroup per normal, built with the frame's weights of the delivered radiance over
// the exposure normalisation once per table and again when the weights' shape moves (D72), so the
// trace reads each normal's clipped variance and means ready made.
void RenderRaytracing::update_sky_light(RTViewportState *p_state, const RenderDataRD *p_render_data, uint32_t p_rt_flags) {
	ERR_FAIL_NULL(p_state);
	if (!p_state->sky_light_buffer.is_valid() || (p_rt_flags & SceneShaderRaytracing::RT_FLAG_SKY_LIGHT) == 0) {
		return;
	}
	RD *rd = RD::get_singleton();
	RID sky_rid;
	if (p_render_data && p_render_data->environment.is_valid()) {
		sky_rid = owner->environment_get_sky(p_render_data->environment);
	}
	if (!sky_rid.is_valid()) {
		if (p_state->sky_light_sky.is_valid()) {
			rd->buffer_clear(p_state->sky_light_buffer, 0, RT_SKY_LIGHT_BUFFER_SIZE);
			p_state->sky_light_sky = RID();
			p_state->sky_light_version = 0;
			p_state->sky_light_moments_built = false;
			p_state->sky_light_mix_valid = false;
		}
		return;
	}
	uint64_t version = owner->sky.sky_get_radiance_version(sky_rid);
	const bool build_table = sky_rid != p_state->sky_light_sky || version != p_state->sky_light_version;
	const bool variance_share = (p_rt_flags & SceneShaderRaytracing::RT_FLAG_SKY_LIGHT_VARIANCE_SHARE) != 0;

	// The frame's weights of L, L_f and 1 in the luminance sky_radiance delivers (D64): the sharpest
	// level under the exposure mixed toward the fog by its sky affect, the fog's colour being the
	// Environment's fog light mixed with the exposed fog level by the aerial perspective; without
	// the fog, the exposure times L. Read from the trace's scene data copy, as update_uniform_set made
	// it earlier in the frame (_update_scene_uniform_buffer).
	float mix[3];
	const float exposure = p_state->sky_light_scene[0];
	if ((p_rt_flags & SceneShaderRaytracing::RT_FLAG_FOG_ENABLED) != 0) {
		const float affect = p_state->sky_light_scene[1];
		const float aerial = p_state->sky_light_scene[2] > 0.0f ? p_state->sky_light_scene[2] : 0.0f;
		const float fog = p_state->sky_light_scene[3];
		mix[0] = exposure * (1.0f - affect);
		mix[1] = exposure * affect * aerial;
		mix[2] = affect * (1.0f - aerial) * fog;
	} else {
		mix[0] = exposure;
		mix[1] = 0.0f;
		mix[2] = 0.0f;
	}
	// The moments are built with the weights over the exposure normalisation (D72), which only scales
	// them and which the trace folds into its S, so a move of the camera exposure or the
	// background's intensity costs no rebuild, while the weights' shape (the fog's sky affect, its
	// aerial perspective, the fog light's fixed term against the exposed levels) does once it moves
	// by a hundredth. The normalisation is the unit, so the fog's sky affect at 1 (the default) still
	// leaves the fog's level and colour in the moments.
	bool mix_changed = !p_state->sky_light_mix_valid;
	if (!mix_changed) {
		for (int i = 0; i < 3; i++) {
			const float now = mix[i] / exposure;
			const float built = p_state->sky_light_mix[i] / p_state->sky_light_scene_exposure;
			if (Math::abs(now - built) > 0.01f * MAX(Math::abs(built), 1e-6f)) {
				mix_changed = true;
			}
		}
	}
	bool build_moments = variance_share && exposure > 0.0f && (build_table || !p_state->sky_light_moments_built || mix_changed);
	if (!build_table && !build_moments) {
		return;
	}
	RID radiance = owner->sky.sky_get_radiance_2d_texture_rd(sky_rid);
	if (!radiance.is_valid() || version == 0) {
		return; // Not rendered yet; the next frame's version says when.
	}

	RD::Uniform u_octmap;
	u_octmap.binding = 0;
	u_octmap.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
	u_octmap.append_id(radiance);
	RD::Uniform u_sampler;
	u_sampler.binding = 1;
	u_sampler.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
	u_sampler.append_id(RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(
			RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
	RD::Uniform u_table;
	u_table.binding = 2;
	u_table.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	u_table.append_id(p_state->sky_light_buffer);
	// Through the cache: a sky that re-renders every frame (a sun that moves) rebuilds every frame.
	// In array mode the fog's level is a roughness layer, which the sharpest layer's own texture does
	// not reach, so the array joins the moments' set (D64).
	const bool use_octmap_array = owner->is_using_radiance_octmap_array();
	const int moments_mode = use_octmap_array ? SkyLightShader::MODE_MOMENTS_OCTMAP_ARRAY : SkyLightShader::MODE_MOMENTS_OCTMAP_2D;
	RID moments_set;
	if (build_moments) {
		if (use_octmap_array) {
			RID layers = owner->sky.sky_get_radiance_texture_rd(sky_rid);
			if (layers.is_valid()) {
				RD::Uniform u_layers;
				u_layers.binding = 3;
				u_layers.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u_layers.append_id(layers);
				moments_set = UniformSetCacheRD::get_singleton()->get_cache(sky_light_shader.version_shader[moments_mode], 0, u_octmap, u_sampler, u_table, u_layers);
			}
		} else {
			moments_set = UniformSetCacheRD::get_singleton()->get_cache(sky_light_shader.version_shader[moments_mode], 0, u_octmap, u_sampler, u_table);
		}
		if (!moments_set.is_valid()) {
			// No layers yet: the table alone, and the moments with a later frame.
			build_moments = false;
			if (!build_table) {
				return;
			}
		}
	}

	struct SkyLightPC {
		float border_pad;
		float border_scale;
		float lod_cell;
		float floor_fraction;
		float mix_level;
		float mix_fog_level;
		float mix_constant;
		float lod_leaf;
		float tap_offset;
		uint32_t leaf_level;
		uint32_t reduce_level;
		uint32_t pad;
	} pc;
	pc.border_pad = owner->sky.sky_get_uv_border_size(sky_rid);
	pc.border_scale = 1.0f - 2.0f * pc.border_pad;
	// The octmap's texels inside its padding, which the pyramid's leaves are (D69): the largest power
	// of two that fits, held to the moments' grid at the least and RT_SKY_LIGHT_MAX_LEVELS at the most.
	const RD::TextureFormat radiance_format = rd->texture_get_format(radiance);
	const uint32_t padding = (uint32_t)Math::round(pc.border_pad * (float)radiance_format.width);
	const uint32_t interior = radiance_format.width > 2 * padding ? radiance_format.width - 2 * padding : radiance_format.width;
	uint32_t leaf_level = RT_SKY_LIGHT_CELL_LEVEL;
	while (leaf_level < RT_SKY_LIGHT_MAX_LEVELS && (2u << leaf_level) <= interior) {
		leaf_level++;
	}
	const uint32_t leaf_n = 1u << leaf_level;
	// In array mode the layer's mips are box averages: a cell of the moments' grid spans
	// radiance_size * 2 / 256 texels of the sharpest mip and the build taps it four times, so the mip
	// whose texel is half a cell wide averages the cell (the sharpest when the octmap is no finer than
	// the grid). In 2D mode the mips are roughness levels, so the sharpest is read. A leaf is read the
	// same way at its own mip; where it is one texel or less, one tap at its centre reads that texel.
	pc.lod_cell = use_octmap_array ? MAX(0.0f, Math::log2((float)owner->sky.sky_get_radiance_size(sky_rid) / (float)(1u << RT_SKY_LIGHT_CELL_LEVEL))) : 0.0f;
	const bool leaf_is_texel = interior <= leaf_n;
	pc.lod_leaf = (use_octmap_array && !leaf_is_texel) ? MAX(0.0f, Math::log2((float)interior / (float)leaf_n) - 1.0f) : 0.0f;
	pc.tap_offset = leaf_is_texel ? 0.0f : 0.25f;
	pc.leaf_level = leaf_level;
	pc.reduce_level = 0;
	pc.floor_fraction = 1.0e-3f;
	// The weights over the exposure normalisation (D72), which the trace reads its S over; 0 where no
	// moments are built.
	pc.mix_level = exposure > 0.0f ? mix[0] / exposure : 0.0f;
	pc.mix_fog_level = exposure > 0.0f ? mix[1] / exposure : 0.0f;
	pc.mix_constant = exposure > 0.0f ? mix[2] / exposure : 0.0f;
	pc.pad = 0;

	RD::ComputeListID compute_list = rd->compute_list_begin();
	if (build_table) {
		rd->compute_list_bind_compute_pipeline(compute_list, sky_light_shader.pipeline[SkyLightShader::MODE_LEAF]);
		rd->compute_list_bind_uniform_set(compute_list, UniformSetCacheRD::get_singleton()->get_cache(sky_light_shader.version_shader[SkyLightShader::MODE_LEAF], 0, u_octmap, u_sampler, u_table), 0);
		rd->compute_list_set_push_constant(compute_list, &pc, sizeof(SkyLightPC));
		rd->compute_list_dispatch(compute_list, (leaf_n + 15) / 16, (leaf_n + 15) / 16, 1);
		// Each level from the one below, down to the first the header pass reduces itself.
		RID table_set = UniformSetCacheRD::get_singleton()->get_cache(sky_light_shader.version_shader[SkyLightShader::MODE_REDUCE], 0, u_table);
		rd->compute_list_bind_compute_pipeline(compute_list, sky_light_shader.pipeline[SkyLightShader::MODE_REDUCE]);
		rd->compute_list_bind_uniform_set(compute_list, table_set, 0);
		for (uint32_t level = leaf_level - 1; level > RT_SKY_LIGHT_HEADER_LEVELS; level--) {
			rd->compute_list_add_barrier(compute_list); // The level reads the one below.
			pc.reduce_level = level;
			rd->compute_list_set_push_constant(compute_list, &pc, sizeof(SkyLightPC));
			rd->compute_list_dispatch(compute_list, ((1u << (2 * level)) + 255) / 256, 1, 1);
		}
		rd->compute_list_add_barrier(compute_list); // The header reads the last reduced level and the cells' level.
		rd->compute_list_bind_compute_pipeline(compute_list, sky_light_shader.pipeline[SkyLightShader::MODE_HEADER]);
		rd->compute_list_bind_uniform_set(compute_list, UniformSetCacheRD::get_singleton()->get_cache(sky_light_shader.version_shader[SkyLightShader::MODE_HEADER], 0, u_table), 0);
		rd->compute_list_set_push_constant(compute_list, &pc, sizeof(SkyLightPC));
		rd->compute_list_dispatch(compute_list, 1, 1, 1);
	}
	if (build_moments) {
		if (build_table) {
			rd->compute_list_add_barrier(compute_list); // The moments read the pyramid's cells and the header.
		}
		rd->compute_list_bind_compute_pipeline(compute_list, sky_light_shader.pipeline[moments_mode]);
		rd->compute_list_bind_uniform_set(compute_list, moments_set, 0);
		rd->compute_list_set_push_constant(compute_list, &pc, sizeof(SkyLightPC));
		rd->compute_list_dispatch(compute_list, RT_SKY_LIGHT_NORMALS, 1, 1);
	}
	rd->compute_list_end();

	if (build_table) {
		p_state->sky_light_sky = sky_rid;
		p_state->sky_light_version = version;
		p_state->sky_light_moments_built = false;
	}
	if (build_moments) {
		p_state->sky_light_moments_built = true;
		p_state->sky_light_mix_valid = true;
		p_state->sky_light_mix[0] = mix[0];
		p_state->sky_light_mix[1] = mix[1];
		p_state->sky_light_mix[2] = mix[2];
		p_state->sky_light_scene_exposure = exposure;
	}
}

void RenderRaytracing::register_raytracing_buffer_dependencies(RD::RaytracingListID p_list) {
	RD *rd = RD::get_singleton();

	// The hit shader reads vertex / attribute / index data via BDA stored in
	// RT_GeometryData. The draw graph cannot infer those reads from the BDA, so
	// every per-frame-written buffer that the trace will touch via BDA must be
	// declared here -- otherwise the producing copy/dispatch may not be visible
	// to the trace. (Static mesh VB/AB/IB are not declared: an in-place update of
	// one waits for earlier traces through buffer_add_device_address_read_stages.)

	// Mat UBO pool buffer dependency.
	if (mat_ubo_pool_buffer.is_valid()) {
		rd->raytracing_list_add_buffer_dependency(p_list, mat_ubo_pool_buffer, /*p_writable=*/false);
	}

	// Deformed surface buffer dependencies.
	for (uint32_t i = 0; i < deformed_active_this_frame.size(); i++) {
		RTDeformedCacheEntry *e = deformed_pool.get_or_null(deformed_active_this_frame[i]);
		if (!e) {
			continue;
		}
		if (e->owned_vb_full.is_valid()) {
			rd->raytracing_list_add_buffer_dependency(p_list, e->owned_vb_full, /*p_writable=*/false);
		}
		if (e->prev_pos_vb.is_valid()) {
			rd->raytracing_list_add_buffer_dependency(p_list, e->prev_pos_vb, /*p_writable=*/false);
		}
	}

	// Merged MultiMesh buffer dependencies.
	for (uint32_t i = 0; i < merged_mm_active_this_frame.size(); i++) {
		RTMergedMMEntry *e = merged_mm_pool.get_or_null(merged_mm_active_this_frame[i]);
		if (!e) {
			continue;
		}
		if (e->merged_vtx_buffer.is_valid()) {
			rd->raytracing_list_add_buffer_dependency(p_list, e->merged_vtx_buffer, /*p_writable=*/false);
		}
		if (e->merged_attr_buffer.is_valid()) {
			rd->raytracing_list_add_buffer_dependency(p_list, e->merged_attr_buffer, /*p_writable=*/false);
		}
		if (e->replicated_idx_buffer.is_valid()) {
			rd->raytracing_list_add_buffer_dependency(p_list, e->replicated_idx_buffer, /*p_writable=*/false);
		}
	}

	// The cluster geometry blobs (D93b), written by a population or a load in this frame: the chunks, a few
	// 32 MB buffers. Their in-place reuse past retirement is ordered by the read stages each declares.
	for (const RTClusterArenaChunk &chunk : cluster_geometry_arena) {
		if (chunk.buffer.is_valid()) {
			rd->raytracing_list_add_buffer_dependency(p_list, chunk.buffer, /*p_writable=*/false);
		}
	}
}

// ---------------------------------------------------------------------------
// Output copy
// ---------------------------------------------------------------------------

void RenderRaytracing::copy_output_texture(const RenderDataRD *p_render_data) {
	Ref<RenderSceneBuffersRD> rb = p_render_data->render_buffers;
	ERR_FAIL_COND(rb.is_null());

	if (!rt_has_texture(rb.ptr())) {
		return;
	}

	// Copy raytracing output to main color buffer
	for (uint32_t v = 0; v < rb->get_view_count(); v++) {
		RID src = rt_get_texture(rb.ptr());
		RID dst = rb->get_internal_texture(v);
		owner->copy_effects->copy_to_rect(src, dst, Rect2i(0, 0, rb->get_internal_size().x, rb->get_internal_size().y), false, false, false, false, false, true);
	}
}
