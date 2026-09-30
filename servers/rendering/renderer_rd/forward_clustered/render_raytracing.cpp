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

void RenderRaytracing::initialize(RenderForwardClustered *p_owner) {
	owner = p_owner;
	singleton = this;
	bindless_block = memnew(BindlessBlock);
	// A surface its mesh drops releases its RT cache entry at once (D81), not at the slot's reuse.
	RendererRD::MeshStorage::get_singleton()->set_surface_freed_callback(&RenderRaytracing::_mesh_surface_freed);

	cluster_blas_enabled = RD::get_singleton()->has_feature(RD::SUPPORTS_CLUSTER_ACCELERATION_STRUCTURE);
	if (cluster_blas_enabled) {
		cluster_compaction_enabled = GLOBAL_GET("rendering/pathtracing/cluster_blas_compaction");
		cluster_deformed_enabled = GLOBAL_GET("rendering/pathtracing/cluster_blas_deformed");
		const RD::ClusterAccelerationStructureCapabilities &caps = RD::get_singleton()->get_cluster_acceleration_structure_capabilities();
		// One line on the console (the setting is off by default): the harness reads it as proof the path is active.
		print_line(vformat("RT: static BLAS built from clusters of up to %d triangles and %d vertices (D79), compaction %s, deformed meshes %s.", caps.max_triangles_per_cluster, caps.max_vertices_per_cluster, cluster_compaction_enabled ? "on" : "off", cluster_deformed_enabled ? "rebuilt from clusters (D82)" : "refit as triangle BLAS"));
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
	sky_light_shader.shader.version_free(sky_light_shader.version);
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

	RD *rd = RD::get_singleton();

	// The cluster arena (D81): every region went with its surface above; a result still in
	// flight has no surface to land on.
	cluster_compaction_pending.clear();
	cluster_compaction_results.clear();
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

	// Free all cached deformed surface data.
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
	if (!p_entry.ptr || !p_entry.ptr->blas.is_valid()) {
		return;
	}
	RD *rd = RD::get_singleton();
	if (rd->acceleration_structure_is_valid(p_entry.ptr->blas)) {
		rd->free_rid(p_entry.ptr->blas);
	}
	p_entry.ptr->blas = RID();
	p_entry.blas_built_once = false;
	// The cluster layout (D82) is tied to the handle's slot; the storage stays for the next layout.
	p_entry.cluster_count = 0;
	p_entry.cluster_built_frame = 0;
	p_entry.cluster_frame_kind = RTDeformedCacheEntry::CLUSTER_QUEUED_NONE;
	p_entry.cluster_infos.clear();
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
	// Sizes read back since the last frame move their CLAS into the arena before the TLAS
	// instances of this frame take each surface's handle (D81).
	_process_cluster_compactions();

	blass.clear();
	blas_transforms.clear();
	instance_flags.clear();
	instance_masks.clear();
	sbt_offsets.clear();
	geometry_data.clear();
	material_data.clear();
	motion_indices.clear();
	motion_transforms.clear();
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
		LocalVector<RID> &r_dirty_blas_update_list) {
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

	bool layout_changed = entry.cached_vertex_count != vertex_count || entry.cached_full_size != full_size;
	bool data_changed = layout_changed ||
			entry.cached_change_stamp != p_source.change_stamp ||
			entry.cached_buffer_id != buffer_id;
	bool needs_full_rebuild = !entry.ptr || !entry.blas_built_once || layout_changed ||
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

	// Per-frame copies run exactly once per frame, even when the same surface is visible from multiple viewports.
	const bool first_touch_this_frame = entry.last_used_frame != current_frame;
	if (first_touch_this_frame) {
		if (entry.prev_pos_seeded) {
			rd->buffer_copy(entry.owned_vb_full, entry.prev_pos_vb, 0, 0, pos_size);
		}
		if (data_changed || !entry.prev_pos_seeded) {
			rd->buffer_copy(p_source.current_vb, entry.owned_vb_full, 0, 0, full_size);
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
			clustered = _populate_deformed_cluster_layout(p_mesh_surface, static_cast<uint32_t>(p_source.cache_key), entry);
		}
		if (clustered) {
			_queue_deformed_cluster_rebuild(entry, RTDeformedCacheEntry::CLUSTER_QUEUED_FIRST_BUILD);
		} else {
			_populate_surface_blas(p_mesh_surface, entry.owned_vb_full, true, true, true,
					static_cast<uint32_t>(p_source.cache_key), entry.ptr, r_dirty_blas_list);
		}
		entry.blas_built_once = entry.ptr->blas.is_valid();
	} else if (data_changed || (entry.cluster_count > 0 && _deformed_cluster_region_expired(entry, rd->get_frames_drawn()))) {
		cache_misses++;
		if (entry.cluster_count > 0) {
			_queue_deformed_cluster_rebuild(entry, data_changed ? RTDeformedCacheEntry::CLUSTER_QUEUED_CHANGED : RTDeformedCacheEntry::CLUSTER_QUEUED_EXPIRY);
		} else {
			r_dirty_blas_update_list.push_back(entry.ptr->blas);
		}
	} else {
		cache_hits++;
	}

	if (entry.ptr && entry.ptr->blas.is_valid()) {
		entry.ptr->geometry.flags |= RT_GEOM_FLAG_DEFORMED;
		uint64_t prev_addr = rd->buffer_get_device_address(entry.prev_pos_vb);
		entry.ptr->geometry.prev_vertex_buffer_address_lo = static_cast<uint32_t>(prev_addr & 0xFFFFFFFFULL);
		entry.ptr->geometry.prev_vertex_buffer_address_hi = static_cast<uint32_t>(prev_addr >> 32);
	}

	surf->cached_final_transform_valid = false;

	entry.cached_change_stamp = p_source.change_stamp;
	entry.cached_key_version = p_source.cache_version;
	entry.cached_surface_counter = p_source.surface_counter;
	entry.cached_buffer_id = buffer_id;
	entry.cached_vertex_count = vertex_count;
	entry.cached_full_size = full_size;
	entry.last_used_frame = current_frame;

	return entry.ptr;
}

// ---------------------------------------------------------------------------
// Surface BLAS helper (shared between static and deformed paths)
// ---------------------------------------------------------------------------

// Fills RTSurfaceData geometry metadata from the surface format.
// Returns the RIDs needed for BLAS creation so _populate_surface_blas can use them.
static void _fill_surface_geometry_data(
		void *p_mesh_surface,
		bool p_force_uncompressed,
		RTSurfaceData *r_surf_data,
		RID *r_vertex_buffer = nullptr,
		RID *r_attribute_buffer = nullptr,
		RID *r_index_buffer = nullptr) {
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

	uint32_t vertex_count = mesh_storage->mesh_surface_get_vertex_count(p_mesh_surface);
	uint32_t index_count = mesh_storage->mesh_surface_get_index_count(p_mesh_surface, 0);

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

void RenderRaytracing::_populate_surface_blas(
		void *p_mesh_surface,
		RID p_vertex_buffer_override,
		bool p_force_uncompressed,
		bool p_prefer_fast_build,
		bool p_allow_update,
		uint32_t p_cache_key,
		RTSurfaceData *r_surf_data,
		LocalVector<RID> &r_dirty_blas_list) {
	RID vertex_buffer, attribute_buffer, index_buffer;
	_fill_surface_geometry_data(p_mesh_surface, p_force_uncompressed, r_surf_data,
			&vertex_buffer, &attribute_buffer, &index_buffer);

	if (p_vertex_buffer_override.is_valid()) {
		vertex_buffer = p_vertex_buffer_override;
	}

	RD *rd = RD::get_singleton();
	RT_GeometryData &geom = r_surf_data->geometry;

	if (vertex_buffer.is_valid()) {
		geom.vertex_buffer_address = rd->buffer_get_device_address(vertex_buffer);
	}
	if (attribute_buffer.is_valid()) {
		geom.attribute_buffer_address = rd->buffer_get_device_address(attribute_buffer);
	}
	if (index_buffer.is_valid() && geom.index_format != RT_INDEX_FORMAT_NONE) {
		geom.index_buffer_address = rd->buffer_get_device_address(index_buffer);
	}

	uint32_t vertex_count = geom.vertex_count;
	uint32_t index_count = (geom.index_format != RT_INDEX_FORMAT_NONE) ? geom.primitive_count * 3 : 0;
	uint32_t position_stride = geom.position_stride;

	bool is_2d = RendererRD::MeshStorage::get_singleton()->mesh_surface_get_format(p_mesh_surface) & RSE::ARRAY_FLAG_USE_2D_VERTICES;
	bool compressed = r_surf_data->is_compressed;

	// A static indexed 3D surface builds from clusters (D79); a deformed surface reaching this
	// function is one the cluster path declined or the path is off (D82), and refits a triangle
	// BLAS in place, as do 2D and non-indexed surfaces.
	if (cluster_blas_enabled) {
		if (p_vertex_buffer_override.is_null() && !p_allow_update && !is_2d && index_buffer.is_valid() && index_count > 0) {
			if (_populate_surface_cluster_blas(p_mesh_surface, vertex_buffer, index_buffer, p_cache_key, r_surf_data)) {
				return;
			}
		} else {
			print_verbose(vformat("RT: surface [%d] on the triangle BLAS path (deformed %d, 2D %d, indexed %d).", p_cache_key, int(p_vertex_buffer_override.is_valid() || p_allow_update), int(is_2d), int(index_buffer.is_valid() && index_count > 0)));
		}
	}

	// Create BLAS using the new geometry-based API.
	{
		RD::DataFormat pos_format;
		if (is_2d) {
			pos_format = RD::DATA_FORMAT_R32G32_SFLOAT;
		} else if (compressed) {
			pos_format = RD::DATA_FORMAT_R16G16B16A16_UNORM;
		} else {
			pos_format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
		}

		RD::AccelerationStructureGeometry as_geom;
		// Type defaults to TYPE_TRIANGLES; set explicitly for clarity.
		as_geom.type = RD::AccelerationStructureGeometry::TYPE_TRIANGLES;
		as_geom.geometry.triangles.vertex_buffer = vertex_buffer;
		as_geom.geometry.triangles.vertex_stride = position_stride;
		as_geom.geometry.triangles.vertex_count = vertex_count;
		as_geom.geometry.triangles.vertex_format = pos_format;

		if (index_buffer.is_valid() && index_count > 0) {
			as_geom.geometry.triangles.index_buffer = index_buffer;
			as_geom.geometry.triangles.index_count = index_count;
		}

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
	}
}

bool RenderRaytracing::_populate_surface_cluster_blas(void *p_mesh_surface, RID p_vertex_buffer, RID p_index_buffer, uint32_t p_cache_key, RTSurfaceData *r_surf_data) {
	RD *rd = RD::get_singleton();
	const RT_GeometryData &geom = r_surf_data->geometry;
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();

	const uint32_t primitive_count = geom.primitive_count;
	if (primitive_count == 0 || geom.vertex_buffer_address == 0 || geom.index_buffer_address == 0 || caps.max_triangles_per_cluster == 0 || caps.max_vertices_per_cluster < 3) {
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
	const uint32_t total_cluster_vertices = layout.total_vertices;
	const bool from_table = layout.from_table;
	const uint32_t cluster_count = cluster_triangles.size();
	const uint32_t index_size = geom.index_format == RT_INDEX_FORMAT_UINT16 ? sizeof(uint16_t) : sizeof(uint32_t);
	const RD::DataFormat pos_format = r_surf_data->is_compressed ? RD::DATA_FORMAT_R16G16B16A16_UNORM : RD::DATA_FORMAT_R32G32B32_SFLOAT;

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
	clas_input.max_acceleration_structure_count = cluster_count;
	clas_input.flags = RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT;
	clas_input.vertex_format = pos_format;
	clas_input.max_cluster_triangle_count = max_cluster_triangles;
	clas_input.max_cluster_vertex_count = max_cluster_vertices;
	clas_input.max_total_triangle_count = primitive_count;
	clas_input.max_total_vertex_count = total_cluster_vertices;

	RD::ClusterAccelerationStructureInput blas_input;
	blas_input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_CLUSTERS_BOTTOM_LEVEL;
	blas_input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
	blas_input.max_acceleration_structure_count = 1;
	blas_input.flags = RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT;
	blas_input.max_total_cluster_count = cluster_count;
	blas_input.max_cluster_count_per_acceleration_structure = cluster_count;

	uint64_t clas_region_size = 0, clas_scratch = 0, blas_size = 0, blas_scratch = 0;
	rd->cluster_acceleration_structure_get_build_sizes(clas_input, clas_region_size, clas_scratch);
	rd->cluster_acceleration_structure_get_build_sizes(blas_input, blas_size, blas_scratch);
	if (clas_region_size == 0 || blas_size == 0) {
		print_verbose(vformat("RT: surface [%d] on the triangle BLAS path: the driver sized the CLAS region at %d and the BLAS at %d bytes.", p_cache_key, clas_region_size, blas_size));
		return false;
	}

	// Staging: the BLAS first, then the implicit CLAS region at its alignment. The buffer's own
	// address is aligned only as the allocator saw fit, so the base is aligned inside a padded buffer.
	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	const uint64_t clas_region = _align_up(blas_size, base_alignment);
	const uint64_t staging_size = base_alignment + clas_region + clas_region_size;
	ERR_FAIL_COND_V_MSG(staging_size > UINT32_MAX, false, "RT: a cluster BLAS's staging exceeds 4 GB.");

	RID staging = rd->storage_buffer_create(uint32_t(staging_size), {}, 0,
			RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_STORAGE_BIT);
	if (staging.is_null()) {
		return false;
	}
	const uint64_t buffer_address = rd->buffer_get_device_address(staging);
	const uint64_t staging_address = _align_up(buffer_address, base_alignment);
	const uint64_t blas_offset = staging_address - buffer_address;
	RID blas = rd->cluster_blas_create(staging, blas_offset);
	if (blas.is_null()) {
		rd->free_rid(staging);
		return false;
	}

	// Arguments, 16-byte sections: the CLAS infos; BLAS info A, over the implicit addresses (this
	// frame's build); BLAS info B, over the compact addresses (the rebuild after the move); the
	// implicit addresses (cluster_count CLAS the driver writes, then the BLAS's staging address);
	// the CLAS sizes; the compact addresses, filled at the move.
	const uint32_t clas_info_stride = caps.triangle_build_info_size;
	const uint32_t blas_info_stride = caps.bottom_level_build_info_size;
	const uint64_t clas_infos_offset = 0;
	const uint64_t blas_info_a_offset = _align_up(clas_infos_offset + uint64_t(clas_info_stride) * cluster_count, 16);
	const uint64_t blas_info_b_offset = _align_up(blas_info_a_offset + blas_info_stride, 16);
	const uint64_t implicit_addresses_offset = _align_up(blas_info_b_offset + blas_info_stride, 16);
	const uint64_t sizes_offset = _align_up(implicit_addresses_offset + sizeof(uint64_t) * (cluster_count + 1), 16);
	const uint64_t compact_addresses_offset = _align_up(sizes_offset + sizeof(uint32_t) * cluster_count, 16);
	const uint64_t args_size = _align_up(compact_addresses_offset + sizeof(uint64_t) * (cluster_count + 1), 16);

	LocalVector<uint8_t> args;
	args.resize(args_size);
	memset(args.ptr(), 0, args_size);

	uint32_t first_triangle = 0;
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t triangle_count = cluster_triangles[c];
		RD::ClusterTriangleBuildInfo info;
		info.cluster_id = first_triangle;
		info.triangle_count = triangle_count;
		info.vertex_count = cluster_vertices[c];
		info.index_type = geom.index_format == RT_INDEX_FORMAT_UINT16 ? RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT16 : RD::ClusterTriangleBuildInfo::CLUSTER_INDEX_TYPE_UINT32;
		info.index_address = geom.index_buffer_address + uint64_t(first_triangle) * 3u * index_size;
		info.index_stride = index_size;
		info.vertex_address = geom.vertex_buffer_address;
		info.vertex_stride = geom.position_stride;
		rd->cluster_triangle_build_info_write(args.ptr() + clas_infos_offset + uint64_t(c) * clas_info_stride, info);
		first_triangle += triangle_count;
	}
	reinterpret_cast<uint64_t *>(args.ptr() + implicit_addresses_offset)[cluster_count] = staging_address;

	RID args_buffer = rd->storage_buffer_create(uint32_t(args_size), {}, 0,
			RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
	if (args_buffer.is_null()) {
		rd->free_rid(blas);
		rd->free_rid(staging);
		return false;
	}
	const uint64_t args_address = rd->buffer_get_device_address(args_buffer);
	{
		RD::ClusterBottomLevelBuildInfo blas_info;
		blas_info.cluster_references_count = cluster_count;
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
		build.source_buffers.push_back(p_vertex_buffer);
		build.source_buffers.push_back(p_index_buffer);
		if (rd->cluster_acceleration_structure_build(build) != OK) {
			rd->free_rid(blas);
			rd->free_rid(args_buffer);
			rd->free_rid(staging);
			return false;
		}
	}
	{
		RD::ClusterAccelerationStructureBuild build;
		build.input = blas_input;
		build.dst_storage_buffer = staging;
		build.dst_addresses_buffer = args_buffer;
		build.dst_addresses_offset = implicit_addresses_offset + sizeof(uint64_t) * cluster_count;
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
			return false;
		}
	}

	r_surf_data->blas = blas;
	rd->set_resource_name(r_surf_data->blas, "RT cluster BLAS [" + itos(p_cache_key) + "]");
	rd->set_resource_name(staging, "RT cluster staging [" + itos(p_cache_key) + "]");
	rd->set_resource_name(args_buffer, "RT cluster args [" + itos(p_cache_key) + "]");
	r_surf_data->cluster_storage = staging;
	r_surf_data->cluster_args = args_buffer;
	r_surf_data->blas_size = staging_size + args_size;
	r_surf_data->cluster_count = cluster_count;
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
		const Error readback = rd->buffer_get_data_async(args_buffer, callable_mp_static(&RenderRaytracing::_on_cluster_sizes_read).bind(ticket), uint32_t(sizes_offset), uint32_t(sizeof(uint32_t) * cluster_count));
		if (readback == OK) {
			r_surf_data->compaction_ticket = ticket;
			cluster_compaction_pending.insert(ticket, r_surf_data);
		} else {
			WARN_PRINT_ONCE(vformat("RT: cluster BLAS [%d]: the sizes could not be read back; the surface stays in its staging buffer.", p_cache_key));
		}
	}

	cluster_blas_builds++;
	cluster_triangles_built += primitive_count;
	print_verbose(vformat("RT: cluster BLAS [%d]: %d triangles in %d %s of at most %d triangles and %d vertices, %d KB staging, %d KB arguments.", p_cache_key, primitive_count, cluster_count, from_table ? "clusters" : "runs", max_cluster_triangles, max_cluster_vertices, staging_size / 1024, args_size / 1024));
	return true;
}

// The clusters of a surface: each a contiguous range of the index buffer, its ID its first
// triangle, which is what the hit shaders add to gl_PrimitiveID (rt_primitive_index). A surface
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
	}
}

// ---------------------------------------------------------------------------
// The deformed cluster path (D82)
// ---------------------------------------------------------------------------

// The cluster limits a deformed layout is built under: the mesh table's limits clamped to the
// device, so a table's clusters are taken as they are and the runs fallback stays within them.
static _FORCE_INLINE_ void _deformed_cluster_maxima(const RD::ClusterAccelerationStructureCapabilities &p_caps, uint32_t &r_max_triangles, uint32_t &r_max_vertices) {
	r_max_triangles = MIN(uint32_t(RenderingServerTypes::MESH_CLUSTER_MAX_TRIANGLES), p_caps.max_triangles_per_cluster);
	r_max_vertices = MIN(uint32_t(RenderingServerTypes::MESH_CLUSTER_MAX_VERTICES), p_caps.max_vertices_per_cluster);
}

void RenderRaytracing::_release_deformed_cluster_storage(RTDeformedCacheEntry &p_entry) {
	_release_deformed_blas(p_entry);
	if (p_entry.cluster_storage.is_valid()) {
		RD::get_singleton()->free_rid(p_entry.cluster_storage);
		p_entry.cluster_storage = RID();
	}
	p_entry.cluster_storage_capacity = 0;
	p_entry.cluster_index_buffer = RID();
}

bool RenderRaytracing::_populate_deformed_cluster_layout(void *p_mesh_surface, uint32_t p_cache_key, RTDeformedCacheEntry &r_entry) {
	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	RTSurfaceData *surf_data = r_entry.ptr;

	RID vertex_buffer, attribute_buffer, index_buffer;
	_fill_surface_geometry_data(p_mesh_surface, true, surf_data, &vertex_buffer, &attribute_buffer, &index_buffer);
	RT_GeometryData &geom = surf_data->geometry;
	geom.vertex_buffer_address = rd->buffer_get_device_address(r_entry.owned_vb_full);
	if (attribute_buffer.is_valid()) {
		geom.attribute_buffer_address = rd->buffer_get_device_address(attribute_buffer);
	}
	if (index_buffer.is_valid() && geom.index_format != RT_INDEX_FORMAT_NONE) {
		geom.index_buffer_address = rd->buffer_get_device_address(index_buffer);
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
	const uint32_t cluster_count = layout.triangles.size();

	// The entry's storage holds the BLAS alone, at an explicit address the size query bounds for
	// this cluster count; its CLAS are built with implicit destinations into the frame's region
	// (_flush_deformed_cluster_batch), as the static path builds its own (D81).
	RD::ClusterAccelerationStructureInput blas_input;
	blas_input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_CLUSTERS_BOTTOM_LEVEL;
	blas_input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
	blas_input.max_acceleration_structure_count = 1;
	blas_input.flags = RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT;
	blas_input.max_total_cluster_count = cluster_count;
	blas_input.max_cluster_count_per_acceleration_structure = cluster_count;
	uint64_t blas_size = 0, blas_scratch = 0;
	rd->cluster_acceleration_structure_get_build_sizes(blas_input, blas_size, blas_scratch);
	if (blas_size == 0) {
		print_verbose(vformat("RT: deformed surface [%d] on the triangle BLAS path: the driver sized the BLAS at 0 bytes.", p_cache_key));
		return false;
	}

	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	const uint64_t storage_size = base_alignment + _align_up(blas_size, base_alignment);
	ERR_FAIL_COND_V_MSG(storage_size > UINT32_MAX, false, "RT: a deformed cluster BLAS's storage exceeds 4 GB.");
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
	}
	const uint64_t buffer_address = rd->buffer_get_device_address(r_entry.cluster_storage);
	const uint64_t base_address = _align_up(buffer_address, base_alignment);
	RID blas = rd->cluster_blas_create(r_entry.cluster_storage, base_address - buffer_address);
	if (blas.is_null()) {
		_release_deformed_cluster_storage(r_entry);
		return false;
	}

	// The CLAS build infos, written once: the positions from the entry's owned vertex buffer, the
	// indices from the mesh's shared index buffer, each cluster's ID its first triangle.
	const uint32_t index_size = geom.index_format == RT_INDEX_FORMAT_UINT16 ? sizeof(uint16_t) : sizeof(uint32_t);
	const uint32_t clas_info_stride = caps.triangle_build_info_size;
	r_entry.cluster_infos.resize(uint64_t(clas_info_stride) * cluster_count);
	uint32_t first_triangle = 0;
	for (uint32_t c = 0; c < cluster_count; c++) {
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
		rd->cluster_triangle_build_info_write(r_entry.cluster_infos.ptr() + uint64_t(c) * clas_info_stride, info);
		first_triangle += triangle_count;
	}

	surf_data->blas = blas;
	surf_data->blas_size = storage_size;
	rd->set_resource_name(blas, "RT deformed cluster BLAS [" + itos(p_cache_key) + "]");
	r_entry.cluster_blas_address = base_address;
	r_entry.cluster_count = cluster_count;
	r_entry.cluster_max_triangles = layout.max_triangles;
	r_entry.cluster_max_vertices = layout.max_vertices;
	r_entry.cluster_total_vertices = layout.total_vertices;
	r_entry.cluster_index_buffer = index_buffer;
	r_entry.cluster_built_frame = 0;
	print_verbose(vformat("RT: deformed cluster BLAS [%d]: %d triangles in %d %s of at most %d triangles and %d vertices, %d KB BLAS storage.", p_cache_key, primitive_count, cluster_count, layout.from_table ? "clusters" : "runs", layout.max_triangles, layout.max_vertices, storage_size / 1024));
	return true;
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

// Records the frame's two builds. Returns false with nothing recorded, or with the CLAS build
// recorded and the BLAS build not; the caller puts the path in its fallback either way.
bool RenderRaytracing::_record_deformed_cluster_batch() {
	RTDeformedClusterBatch &batch = deformed_cluster_batch;
	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	const uint64_t frames_drawn = rd->get_frames_drawn();
	const uint64_t frame_delay = rd->get_frame_delay();
	const uint32_t entry_count = batch.entries.size();
	const uint32_t cluster_count = batch.cluster_count;

	// The CLAS of the frame, with implicit destinations into one region the driver sub-allocates
	// under the batch's own maxima, and the size query's one region for all of them.
	RD::ClusterAccelerationStructureInput clas_input;
	clas_input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_TRIANGLE_CLUSTER;
	clas_input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_IMPLICIT_DESTINATIONS;
	clas_input.max_acceleration_structure_count = cluster_count;
	clas_input.flags = RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT;
	clas_input.vertex_format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
	clas_input.max_cluster_triangle_count = batch.max_cluster_triangles;
	clas_input.max_cluster_vertex_count = batch.max_cluster_vertices;
	clas_input.max_total_triangle_count = batch.total_triangles;
	clas_input.max_total_vertex_count = batch.total_vertices;
	uint64_t region_size = 0, clas_scratch = 0;
	rd->cluster_acceleration_structure_get_build_sizes(clas_input, region_size, clas_scratch);
	ERR_FAIL_COND_V_MSG(region_size == 0, false, "RT: the driver sized the deformed cluster batch's region at 0 bytes.");
	const uint64_t region_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	const uint64_t region_alloc = region_alignment + region_size;
	ERR_FAIL_COND_V_MSG(region_alloc > UINT32_MAX, false, "RT: the deformed cluster batch's region exceeds 4 GB.");

	// A region from the pool: one whose last trace has been waited on (see
	// _deformed_cluster_region_expired), grown if it is too small, or a new one. A frame with
	// several flushes takes several.
	int32_t region_index = -1;
	for (uint32_t i = 0; i < batch.regions.size(); i++) {
		if (frames_drawn - batch.regions[i].last_used_frame >= 2 * frame_delay - 1) {
			region_index = int32_t(i);
			break;
		}
	}
	if (region_index < 0) {
		RTDeformedClusterRegion region;
		batch.regions.push_back(region);
		region_index = int32_t(batch.regions.size()) - 1;
	}
	RTDeformedClusterRegion &region = batch.regions[region_index];
	if (region.capacity < region_alloc) {
		if (region.buffer.is_valid()) {
			rd->free_rid(region.buffer);
			region.buffer = RID();
		}
		const uint64_t capacity = MIN(MAX(region_alloc, region.capacity * 2), uint64_t(UINT32_MAX));
		region.capacity = 0;
		region.buffer = rd->storage_buffer_create(uint32_t(capacity), {}, 0,
				RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_STORAGE_BIT);
		ERR_FAIL_COND_V_MSG(region.buffer.is_null(), false, "RT: the deformed cluster batch's region could not be created.");
		region.capacity = capacity;
		region.base_offset = _align_up(rd->buffer_get_device_address(region.buffer), region_alignment) - rd->buffer_get_device_address(region.buffer);
		rd->set_resource_name(region.buffer, "RT deformed cluster region " + itos(region_index));
	}
	region.last_used_frame = frames_drawn;

	// The frame's arguments, 16-byte sections: the CLAS infos, the BLAS infos, the CLAS addresses
	// the build writes, the BLAS addresses. One grow-only buffer and one grow-only host copy:
	// the graph orders the upload after the previous frame's builds through the buffer's tracker
	// (each build leaves it at the cluster input read-write usage, and the copy takes its
	// barrier), so a ring would buy nothing.
	const uint32_t clas_info_stride = caps.triangle_build_info_size;
	const uint32_t blas_info_stride = caps.bottom_level_build_info_size;
	const uint64_t clas_infos_offset = 0;
	const uint64_t blas_infos_offset = _align_up(clas_infos_offset + uint64_t(clas_info_stride) * cluster_count, 16);
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
	memcpy(args + clas_infos_offset, batch.clas_infos.ptr(), batch.clas_infos.size());
	memcpy(args + blas_addresses_offset, batch.blas_addresses.ptr(), sizeof(uint64_t) * entry_count);
	{
		uint32_t first_cluster = 0;
		for (uint32_t e = 0; e < entry_count; e++) {
			RD::ClusterBottomLevelBuildInfo blas_info;
			blas_info.cluster_references_count = batch.entry_cluster_counts[e];
			blas_info.cluster_references_stride = sizeof(uint64_t);
			blas_info.cluster_references_address = args_address + clas_addresses_offset + sizeof(uint64_t) * first_cluster;
			rd->cluster_bottom_level_build_info_write(args + blas_infos_offset + uint64_t(e) * blas_info_stride, blas_info);
			first_cluster += batch.entry_cluster_counts[e];
		}
	}
	rd->buffer_update(args_buffer, 0, uint32_t(args_size), args);

	// The build takes its lists as Vectors: these three, the one-element source list below, and
	// RenderingDevice's own per-build map, lists and scratch are the flush's allocations.
	Vector<RID> storage_buffers, source_buffers, blas_handles;
	storage_buffers.resize(entry_count);
	blas_handles.resize(entry_count);
	source_buffers.resize(batch.source_buffers.size());
	memcpy(storage_buffers.ptrw(), batch.storage_buffers.ptr(), sizeof(RID) * entry_count);
	memcpy(blas_handles.ptrw(), batch.blas_handles.ptr(), sizeof(RID) * entry_count);
	memcpy(source_buffers.ptrw(), batch.source_buffers.ptr(), sizeof(RID) * batch.source_buffers.size());

	// Every CLAS of the frame in one build into the region, the driver writing their addresses,
	// then every cluster BLAS over its slice of them.
	{
		RD::ClusterAccelerationStructureBuild build;
		build.input = clas_input;
		build.dst_implicit_buffer = region.buffer;
		build.dst_implicit_offset = region.base_offset;
		build.dst_addresses_buffer = args_buffer;
		build.dst_addresses_offset = clas_addresses_offset;
		build.dst_addresses_stride = sizeof(uint64_t);
		build.src_infos_buffer = args_buffer;
		build.src_infos_offset = clas_infos_offset;
		build.src_infos_stride = clas_info_stride;
		build.source_buffers = source_buffers;
		if (rd->cluster_acceleration_structure_build(build) != OK) {
			return false;
		}
	}
	{
		RD::ClusterAccelerationStructureBuild build;
		build.input.op_type = RD::ClusterAccelerationStructureInput::CLUSTER_OP_BUILD_CLUSTERS_BOTTOM_LEVEL;
		build.input.op_mode = RD::ClusterAccelerationStructureInput::CLUSTER_OP_MODE_EXPLICIT_DESTINATIONS;
		build.input.max_acceleration_structure_count = entry_count;
		build.input.flags = RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT;
		build.input.max_total_cluster_count = cluster_count;
		build.input.max_cluster_count_per_acceleration_structure = batch.max_clusters_per_blas;
		build.dst_storage_buffers = storage_buffers;
		build.dst_addresses_buffer = args_buffer;
		build.dst_addresses_offset = blas_addresses_offset;
		build.dst_addresses_stride = sizeof(uint64_t);
		build.src_infos_buffer = args_buffer;
		build.src_infos_offset = blas_infos_offset;
		build.src_infos_stride = blas_info_stride;
		build.source_acceleration_structure_buffers.push_back(region.buffer); // The CLAS each BLAS references, read through its info.
		build.dst_acceleration_structures = blas_handles;
		if (rd->cluster_acceleration_structure_build(build) != OK) {
			return false;
		}
	}

	batch.frames_flushed++;
	if (batch.frames_flushed == 1 || batch.frames_flushed % 300 == 0) {
		print_verbose(vformat("RT: deformed cluster batch: %d surfaces, %d clusters, %d triangles, %d KB region %d of %d, %d KB of arguments uploaded, %d expiry rebuilds so far.", entry_count, cluster_count, batch.total_triangles, region_size / 1024, region_index, batch.regions.size(), args_size / 1024, deformed_cluster_expiry_rebuilds_total));
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
			if (entry->cluster_pushed_frame == frames_drawn) {
				const uint32_t triangles = entry->ptr ? entry->ptr->geometry.primitive_count : 0;
				switch (entry->cluster_frame_kind) {
					case RTDeformedCacheEntry::CLUSTER_QUEUED_FIRST_BUILD:
						deformed_cluster_builds++;
						deformed_cluster_triangles_built += triangles;
						break;
					case RTDeformedCacheEntry::CLUSTER_QUEUED_CHANGED:
						deformed_cluster_refits++;
						deformed_cluster_triangles_refit += triangles;
						break;
					case RTDeformedCacheEntry::CLUSTER_QUEUED_EXPIRY:
						deformed_cluster_expiry_rebuilds_total++;
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
			if (entry->ptr && entry->ptr->blas.is_valid() && _deformed_cluster_region_expired(*entry, frames_drawn)) {
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
	if (!cluster_arena_retired.is_empty()) {
		const uint64_t frames_drawn = rd->get_frames_drawn();
		const uint64_t frame_delay = rd->get_frame_delay();
		for (uint32_t i = 0; i < cluster_arena_retired.size();) {
			const RTClusterRetiredRegion &retired = cluster_arena_retired[i];
			if (frames_drawn - retired.frame >= frame_delay) {
				_cluster_arena_free(retired.chunk, retired.offset, retired.size);
				cluster_arena_retired.remove_at_unordered(i);
			} else {
				i++;
			}
		}
	}

	if (cluster_compaction_results.is_empty()) {
		return;
	}
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);

	// Taken off the member first: a buffer update below can stall on the frame queue, which
	// delivers further readbacks into the member while these are processed.
	LocalVector<RTClusterCompactionResult> results;
	results = cluster_compaction_results;
	cluster_compaction_results.clear();
	for (const RTClusterCompactionResult &result : results) {
		RTSurfaceData **found = cluster_compaction_pending.getptr(result.ticket);
		if (found == nullptr) {
			continue; // The surface was released while its sizes were in flight.
		}
		RTSurfaceData *surf = *found;
		cluster_compaction_pending.erase(result.ticket);
		surf->compaction_ticket = 0;

		const uint32_t cluster_count = surf->cluster_count;
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
			build.input.flags = RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT;
			build.input.max_total_cluster_count = cluster_count;
			build.input.max_cluster_count_per_acceleration_structure = cluster_count;
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
		rd->set_resource_name(blas, "RT cluster BLAS [" + itos(surf->cluster_cache_key) + "]");

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
		cluster_bytes_staged_total += staged_bytes;
		cluster_bytes_compacted_total += region_size;
		print_verbose(vformat("RT: cluster BLAS [%d] compacted: %d clusters, %d KB staged -> %d KB (chunk %d at %d KB; arena %d chunks, %d / %d KB live).", surf->cluster_cache_key, cluster_count, staged_bytes / 1024, region_size / 1024, chunk_index, region_offset / 1024, cluster_arena.size(), cluster_arena_live_bytes / 1024, cluster_arena_reserved_bytes / 1024));
	}
}

bool RenderRaytracing::_cluster_arena_allocate(uint64_t p_size, uint32_t &r_chunk, uint64_t &r_offset) {
	// First fit over the live chunks. Every region size is a multiple of the base alignment and
	// every chunk base is aligned to it, so every offset handed out is aligned.
	for (uint32_t i = 0; i < cluster_arena.size(); i++) {
		RTClusterArenaChunk &chunk = cluster_arena[i];
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
			cluster_arena_live_bytes += p_size;
			return true;
		}
	}

	// A new chunk, of the standard size or, for a region larger than that, its own.
	RD *rd = RD::get_singleton();
	const RD::ClusterAccelerationStructureCapabilities &caps = rd->get_cluster_acceleration_structure_capabilities();
	const uint64_t base_alignment = MAX(caps.cluster_byte_alignment, caps.cluster_bottom_level_byte_alignment);
	const uint64_t chunk_size = MAX(uint64_t(RT_CLUSTER_ARENA_CHUNK_BYTES), p_size);
	const uint64_t buffer_size = chunk_size + base_alignment;
	ERR_FAIL_COND_V_MSG(buffer_size > UINT32_MAX, false, "RT: a cluster arena chunk exceeds 4 GB.");
	RID buffer = rd->storage_buffer_create(uint32_t(buffer_size), {}, 0,
			RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_STORAGE_BIT);
	if (buffer.is_null()) {
		return false;
	}
	RTClusterArenaChunk chunk;
	chunk.buffer = buffer;
	const uint64_t buffer_address = rd->buffer_get_device_address(buffer);
	chunk.address = _align_up(buffer_address, base_alignment);
	chunk.base_offset = chunk.address - buffer_address;
	chunk.size = chunk_size;
	chunk.live_regions = 1;
	if (chunk_size > p_size) {
		chunk.free_list.push_back({ p_size, chunk_size - p_size });
	}

	// A dead slot is reused so the chunk indices surfaces hold stay valid.
	uint32_t index = cluster_arena.size();
	for (uint32_t i = 0; i < cluster_arena.size(); i++) {
		if (cluster_arena[i].buffer.is_null()) {
			index = i;
			break;
		}
	}
	if (index == cluster_arena.size()) {
		cluster_arena.push_back(chunk);
	} else {
		cluster_arena[index] = chunk;
	}
	rd->set_resource_name(buffer, "RT cluster arena [" + itos(index) + "]");
	cluster_arena_reserved_bytes += chunk_size;
	cluster_arena_live_bytes += p_size;
	r_chunk = index;
	r_offset = 0;
	return true;
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
	ERR_FAIL_UNSIGNED_INDEX(p_chunk, cluster_arena.size());
	RTClusterArenaChunk &chunk = cluster_arena[p_chunk];
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
	cluster_arena_live_bytes -= p_size;

	// A chunk with nothing live in it goes, so the arena's memory follows its contents.
	if (chunk.live_regions == 0) {
		RD::get_singleton()->free_rid(chunk.buffer);
		cluster_arena_reserved_bytes -= chunk.size;
		chunk = RTClusterArenaChunk();
	}
}

void RenderRaytracing::release_mesh_surface(RID p_mesh, uint32_t p_surface) {
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

void RenderRaytracing::build_acceleration_structures(RTViewportState *p_state, const LocalVector<RID> &p_dirty_blas_list, const LocalVector<RID> &p_dirty_blas_update_list) {
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

bool RenderRaytracing::_build_merged_mm_blas(
		RID p_mm_rid,
		RID p_mm_gpu_buffer,
		void *p_mesh_surface,
		uint32_t p_mm_count,
		uint32_t p_surface_index,
		uint32_t p_surface_counter,
		const AABB &p_mm_aabb,
		RD::ComputeListID p_compute_list,
		LocalVector<RID> &r_dirty_blas_list,
		LocalVector<RID> &r_dirty_blas_update_list,
		RTSurfaceData *r_surf_data) {
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();

	uint32_t vertex_count = mesh_storage->mesh_surface_get_vertex_count(p_mesh_surface);
	RID index_buffer = mesh_storage->mesh_surface_get_index_buffer(p_mesh_surface, 0);
	uint32_t index_count = mesh_storage->mesh_surface_get_index_count(p_mesh_surface, 0);
	bool indexed = index_buffer.is_valid() && index_count > 0;
	uint32_t prim_count = indexed ? (index_count / 3) : (vertex_count / 3);

	// Skip compressed meshes: their positions are UNORM16x4, not float3.
	uint64_t surface_format = mesh_storage->mesh_surface_get_format(p_mesh_surface);
	if (surface_format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES) {
		return false;
	}

	if (prim_count == 0 || vertex_count == 0) {
		return false;
	}
	static const uint32_t MM_MERGED_BLAS_MAX_TRIANGLES = (uint32_t)GLOBAL_GET("rendering/pathtracing/multimesh_merged_blas_max_triangles");
	if ((uint64_t)p_mm_count * prim_count > MM_MERGED_BLAS_MAX_TRIANGLES) {
		return false; // Too large; fall back to expanded TLAS.
	}

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

	RID src_attr_buf;
	if (has_attr) {
		src_attr_buf = mesh_storage->mesh_surface_get_attribute_buffer(p_mesh_surface);
		ERR_FAIL_COND_V(!src_attr_buf.is_valid(), false);
	} else {
		src_attr_buf = entry.merged_attr_buffer;
	}

	MergeShader::Mode merge_mode = indexed ? MergeShader::MODE_INDEXED : MergeShader::MODE_NON_INDEXED;

	// Pre-validate source vertex buffer before opening the compute list.
	RID vtx_buf = mesh_storage->mesh_surface_get_vertex_buffer(p_mesh_surface);
	ERR_FAIL_COND_V(!vtx_buf.is_valid(), false);
	uint64_t vtx_bda = rd->buffer_get_device_address(vtx_buf);

	const bool needs_rebake = first_touch_this_frame &&
			(transforms_changed || !entry.blas_built_once);

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

RTViewportState *RenderRaytracing::build_tlas(const RenderDataRD *p_render_data, uint32_t p_rt_flags) {
	if (!p_render_data || !p_render_data->rt_instances) {
		return nullptr;
	}

	RTViewportState *state = _get_or_create_viewport_state(p_render_data);
	if (!state) {
		return nullptr;
	}

	prepare_frame();
	state->has_procedural_instances = false;

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
	deformed_cluster_refits = 0;
	deformed_cluster_triangles_refit = 0;
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
	};
	LocalVector<PendingMMSurface> pending_mm_surfaces;

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
				blass.push_back(ps->blas);
				blas_transforms.push_back(instance_transform);
				sbt_offsets.push_back(hg_index);

				RT_GeometryData geom = {};
				geom.flags = RT_GEOM_FLAG_PROCEDURAL;
				geom.vertex_buffer_address = ps->gpu_buffer_address;
				geom.aabb_size_x = (float)ps->culling_aabb.size.x;
				geom.aabb_size_y = (float)ps->culling_aabb.size.y;
				geom.aabb_size_z = (float)ps->culling_aabb.size.z;
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

				// Material for procedural geometry (already validated above).
				uint16_t proc_mat_counter = material_storage->material_get_rt_invalidation_counter(proc_material_rid);
				RTMaterialData *proc_mat_data = process_material(proc_material_rid, proc_mat_counter);
				material_data.push_back(proc_mat_data->data);

				// Procedural instances disable triangle culling and are opaque.
				uint32_t inst_flags = RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT |
						RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT;
				instance_flags.push_back(inst_flags);
				instance_masks.push_back(RT_INSTANCE_MASK_GEOMETRY | RT_INSTANCE_MASK_PROCEDURAL | (inst->data->cast_shadows ? RT_INSTANCE_MASK_SHADOW : 0));
				state->has_procedural_instances = true;
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
				}

				PendingMMSurface pending;
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

#ifdef TOOLS_ENABLED
			uint32_t pre_build_size = dirty_blas_list.size();
			uint32_t pre_refit_size = dirty_blas_update_list.size();
#endif

			// MeshInstance skinning/blend shapes provide a deformed vertex buffer.
			RTSurfaceData *surf_data = nullptr;
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
					surf_data = process_deformed_surface(surf, mesh_surface, src, dirty_blas_list, dirty_blas_update_list);
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
			blas_transforms.push_back(final_transform);

			blass.push_back(surf_data->blas);
			geometry_data.push_back(surf_data->geometry);
			if (surf->rt_deformed_handle.is_valid()) {
				// A deformed surface in the TLAS this frame: the flush counts it (D82).
				RTDeformedCacheEntry *deformed = deformed_pool.get_or_null(surf->rt_deformed_handle);
				if (deformed && deformed->cluster_count > 0) {
					deformed->cluster_pushed_frame = RD::get_singleton()->get_frames_drawn();
				}
			}

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

#ifdef TOOLS_ENABLED
			if (collect_render_info) {
				tlas_instance_count++;
				uint32_t vertices = mesh_storage->mesh_surface_get_vertices_drawn_count(mesh_surface);
				uint32_t prim_count = _rt_indices_to_primitives(surf->primitive, vertices);
				tlas_primitive_count += prim_count;
				uint32_t build_delta = dirty_blas_list.size() - pre_build_size;
				uint32_t refit_delta = dirty_blas_update_list.size() - pre_refit_size;
				rt_blas_builds += build_delta;
				rt_blas_refits += refit_delta;
				rt_triangles_built += prim_count * build_delta;
				rt_triangles_refit += prim_count * refit_delta;
			}
#endif

			sbt_offsets.push_back(mat_data->rt_sbt_offset);
			material_data.push_back(mat_data->data);

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
			}
			instance_flags.push_back(inst_flags);
			instance_masks.push_back(RT_INSTANCE_MASK_GEOMETRY | (inst->data->cast_shadows ? RT_INSTANCE_MASK_SHADOW : 0));

			surf = surf->next;
		}
	}

	// -----------------------------------------------------------------------
	// Phase 2: GPU compute — merged MultiMesh BLAS dispatches.
	// -----------------------------------------------------------------------
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	for (const PendingMMSurface &pending : pending_mm_surfaces) {
#ifdef TOOLS_ENABLED
		uint32_t mm_pre_build_size = dirty_blas_list.size();
		uint32_t mm_pre_refit_size = dirty_blas_update_list.size();
#endif
		RTSurfaceData merged_sd;
		bool use_merged = pending.mm_gpu_buffer.is_valid() &&
				_build_merged_mm_blas(pending.mm_rid, pending.mm_gpu_buffer, pending.mesh_surface,
						pending.mm_count, pending.surface_index, pending.surface_counter, pending.mm_aabb,
						compute_list, dirty_blas_list, dirty_blas_update_list, &merged_sd);

		if (use_merged) {
			blass.push_back(merged_sd.blas);
			blas_transforms.push_back(pending.instance_transform);
			geometry_data.push_back(merged_sd.geometry);
			sbt_offsets.push_back(pending.mat_data->rt_sbt_offset);
			material_data.push_back(pending.mat_data->data);
			motion_indices.push_back(-1);
			instance_flags.push_back(pending.inst_flags);
			instance_masks.push_back(pending.inst_mask);
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

			for (uint32_t mi = 0; mi < pending.mm_count; mi++) {
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

				blass.push_back(surf_data->blas);
				blas_transforms.push_back(final_transform);
				geometry_data.push_back(surf_data->geometry);
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
				uint32_t prim_count = _rt_indices_to_primitives(pending.mm_surf->primitive, vertices);
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

	// The deformed surfaces' clusters and cluster BLAS of the frame (D82), recorded after the
	// merge dispatches' compute list and before the counters are published and the TLAS takes
	// them; a failed flush names the handles that never got a build, and their instances leave
	// every per-instance array here.
	_flush_deformed_cluster_batch();
	if (!deformed_cluster_batch.dropped_handles.is_empty()) {
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
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_RT_BLAS_BUILDS] += rt_blas_builds + cluster_blas_builds + deformed_cluster_builds;
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_RT_BLAS_REFITS] += rt_blas_refits + deformed_cluster_refits;
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_RT_TRIANGLES_BUILT] += rt_triangles_built + cluster_triangles_built + deformed_cluster_triangles_built;
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_RT_TRIANGLES_REFIT] += rt_triangles_refit + deformed_cluster_triangles_refit;
		p_render_data->render_info->info[RSE::VIEWPORT_RENDER_INFO_TYPE_VISIBLE][RSE::VIEWPORT_RENDER_INFO_RT_INSTANCES_COMPILING] += rt_instances_compiling;
	}
#endif

	SceneShaderRaytracing::get_singleton()->finalize_custom_shaders();

	build_acceleration_structures(state, dirty_blas_list, dirty_blas_update_list);
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

	for (uint32_t i = 0; i < blass.size(); i++) {
		if (i >= material_data.size() || i >= geometry_data.size() || i >= sbt_offsets.size() || i >= instance_flags.size()) {
			break;
		}
		const RT_MaterialData &mat = material_data[i];
		const RT_GeometryData &geom = geometry_data[i];
		if (!(mat.flags & RT_MAT_FLAG_EMISSIVE_CONSTANT) || sbt_offsets[i] != 0 || (geom.flags & RT_GEOM_FLAG_PROCEDURAL) || geom.primitive_count == 0) {
			continue;
		}
		if (!(instance_flags[i] & RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT)) {
			// An alpha-tested or blended emitter's any-hit skips texels a sampled point would credit; it keeps chance-hit emission.
			continue;
		}
		// The instance transform already folds a compressed surface's box in (its positions are then a unit cube).
		AABB local_box = (geom.flags & RT_GEOM_FLAG_COMPRESSED) ? AABB(Vector3(), Vector3(1, 1, 1)) : AABB(Vector3(geom.aabb_pos_x, geom.aabb_pos_y, geom.aabb_pos_z), Vector3(geom.aabb_size_x, geom.aabb_size_y, geom.aabb_size_z));
		Transform3D relative = blas_transforms[i];
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
		Transform3D relative = blas_transforms[c.index];
		relative.origin -= p_state->rt_origin;
		RendererRD::MaterialStorage::store_transform_transposed_3x4(relative, e.xform);
		const RT_MaterialData &mat = material_data[c.index];
		e.radiance[0] = mat.emission_color[0] * mat.emission_strength;
		e.radiance[1] = mat.emission_color[1] * mat.emission_strength;
		e.radiance[2] = mat.emission_color[2] * mat.emission_strength;
		running += c.weight;
		e.cdf = (n + 1 == count) ? 1.0f : (float)(running / total);
		e.instance_index = c.index;
		e.primitive_count = geometry_data[c.index].primitive_count;
		e.flags = 0;
		if (instance_flags[c.index] & RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT) {
			e.flags |= RT_EMISSIVE_FLAG_TWO_SIDED;
		}
		if (instance_flags[c.index] & RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT) {
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
		static_assert(sizeof(rt_ubo) == 64 * sizeof(float));
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
	// to the trace. (Static mesh VB/AB/IB are uploaded once via the transfer
	// worker and don't need per-frame barriers.)

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
