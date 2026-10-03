/**************************************************************************/
/*  render_raytracing_omm.cpp                                             */
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

// Opacity micromaps (D96, docs/r1-opacity-micromaps.md). A surface traced with an alpha-tested
// standard material gets a micromap baked from the material's alpha texture: a setup pass gives
// each primitive its UVs as the trace computes them, its subdivision level and whether the whole
// triangle is transparent or opaque; the CPU reads that back, gives uniform primitives a special
// index and deduplicates the rest by their UVs; the bake pass then writes every micro-triangle's
// state and the micromap is built. The micromap is exact for the any-hit's alpha test, so a BLAS
// carrying it traces the same image with fewer any-hit invocations. Until the bake lands, a
// surface traces its plain BLAS as before.

#include "core/config/project_settings.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "servers/rendering/renderer_rd/forward_clustered/render_forward_clustered.h"
#include "servers/rendering/renderer_rd/storage_rd/mesh_storage.h"
#include "servers/rendering/rendering_server_globals.h"

using namespace RendererSceneRenderImplementation;

// Setup passes dispatched a frame; each is one dispatch and a readback.
static constexpr uint32_t RT_OMM_SETUPS_PER_FRAME = 16;
// The most bytes of states one surface's micromap may take; past it, every level drops by one.
static constexpr uint64_t RT_OMM_MAX_STATE_BYTES = 64ull * 1024 * 1024;
// Must match opacity_micromap.glsl.
static constexpr uint32_t RT_OMM_PYRAMID_MAX_LEVELS = 15;
static constexpr uint32_t RT_OMM_PYRAMID_HEADER_WORDS = 48;
static constexpr uint32_t RT_OMM_SETUP_WORDS = 8;
static constexpr uint32_t RT_OMM_STATE_UNKNOWN_TRANSPARENT = 2;
static constexpr uint32_t RT_OMM_STATE_OPAQUE = 1;
static constexpr uint32_t RT_OMM_STATE_TRANSPARENT = 0;

struct RTOmmPushConstant {
	uint32_t tex_size[2];
	uint32_t level_count;
	uint32_t pass_level;
	float uv_scale[2];
	float uv_offset[2];
	float albedo_alpha;
	float threshold;
	float scale_texels;
	uint32_t max_level;
	uint32_t count;
	uint32_t base;
	uint32_t word_stride;
	uint32_t pad;
};
static_assert(sizeof(RTOmmPushConstant) == 64, "RTOmmPushConstant must match opacity_micromap.glsl");

// A descriptor's identity for the deduplication: its three UVs' bits and its level.
struct RTOmmDescriptorKey {
	uint32_t words[7] = {};

	bool operator==(const RTOmmDescriptorKey &p_other) const {
		return memcmp(words, p_other.words, sizeof(words)) == 0;
	}
	static uint32_t hash(const RTOmmDescriptorKey &p_key) {
		uint32_t h = HASH_MURMUR3_SEED;
		for (uint32_t i = 0; i < 7; i++) {
			h = hash_murmur3_one_32(p_key.words[i], h);
		}
		return hash_fmix32(h);
	}
};

static uint32_t _omm_float_bits(float p_value) {
	uint32_t bits;
	memcpy(&bits, &p_value, sizeof(bits));
	return bits;
}

static float _omm_bits_float(uint32_t p_bits) {
	float value;
	memcpy(&value, &p_bits, sizeof(value));
	return value;
}

// Words of states a descriptor of p_level takes: two bits a micro-triangle, at least one word.
static uint32_t _omm_level_words(uint32_t p_level) {
	return MAX((1u << (2 * p_level)) / 16u, 1u);
}

static RTOmmPushConstant _omm_push_constant(const RTOmmKey &p_key, uint32_t p_width, uint32_t p_height, uint32_t p_level_count, float p_scale, uint32_t p_max_level) {
	RTOmmPushConstant pc = {};
	pc.tex_size[0] = p_width;
	pc.tex_size[1] = p_height;
	pc.level_count = p_level_count;
	pc.uv_scale[0] = _omm_bits_float(p_key.uv_scale[0]);
	pc.uv_scale[1] = _omm_bits_float(p_key.uv_scale[1]);
	pc.uv_offset[0] = _omm_bits_float(p_key.uv_offset[0]);
	pc.uv_offset[1] = _omm_bits_float(p_key.uv_offset[1]);
	pc.albedo_alpha = _omm_bits_float(p_key.albedo_alpha);
	pc.threshold = _omm_bits_float(p_key.threshold);
	pc.scale_texels = p_scale;
	pc.max_level = p_max_level;
	return pc;
}

void RenderRaytracing::_omm_initialize() {
	RD *rd = RD::get_singleton();
	const RD::OpacityMicromapCapabilities &caps = rd->get_opacity_micromap_capabilities();
	omm_enabled = caps.supported && GLOBAL_GET("rendering/pathtracing/opacity_micromaps");
	if (!omm_enabled) {
		return;
	}
	omm_scale = CLAMP(float(GLOBAL_GET("rendering/pathtracing/opacity_micromap_scale")), 0.25f, 64.0f);
	omm_max_level = uint32_t(CLAMP(int(GLOBAL_GET("rendering/pathtracing/opacity_micromap_max_level")), 1, 12));
	omm_max_level = MIN(omm_max_level, caps.max_4_state_subdivision_level);
	if (omm_max_level == 0 || caps.triangle_entry_size != 8) {
		WARN_PRINT("RT: the device's opacity micromaps take no subdivision or an unexpected triangle entry; opacity micromaps are off.");
		omm_enabled = false;
		return;
	}

	Vector<String> modes;
	modes.push_back("\n#define MODE_PYRAMID_BASE\n");
	modes.push_back("\n#define MODE_PYRAMID_REDUCE\n");
	modes.push_back("\n#define MODE_SETUP\n");
	modes.push_back("\n#define MODE_BAKE\n");
	omm_shader.shader.initialize(modes);
	omm_shader.version = omm_shader.shader.version_create();
	for (int i = 0; i < OmmShader::MODE_MAX; i++) {
		omm_shader.version_shader[i] = omm_shader.shader.version_get_shader(omm_shader.version, i);
		omm_shader.pipeline[i] = rd->compute_pipeline_create(omm_shader.version_shader[i]);
	}
	// texelFetch alone reads it; any sampler state serves.
	omm_sampler = rd->sampler_create(RD::SamplerState());

	print_line(vformat("RT: opacity micromaps (D96): exact 4-state micromaps baked from alpha-tested standard materials, a micro-triangle about %.2f texels, levels up to %d.", omm_scale, omm_max_level));
}

void RenderRaytracing::_omm_finalize() {
	RD *rd = RD::get_singleton();
	LocalVector<RTOmmEntry *> entries;
	for (const KeyValue<RTOmmKey, RTOmmEntry *> &kv : omm_entries) {
		entries.push_back(kv.value);
	}
	for (RTOmmEntry *entry : entries) {
		_omm_release_entry(entry);
	}
	for (const KeyValue<RID, RTOmmPyramid> &kv : omm_pyramids) {
		if (kv.value.buffer.is_valid()) {
			rd->free_rid(kv.value.buffer);
		}
	}
	omm_pyramids.clear();
	omm_readbacks.clear();
}

RTOmmEntry *RenderRaytracing::_omm_request(RID p_mesh, uint32_t p_surface_index, uint32_t p_surface_counter, const RTMaterialData *p_material) {
	// The standard hit group's alpha test alone: its texture's alpha times the color's against the threshold.
	if (!omm_enabled || p_material == nullptr || p_material->is_custom_shader || p_material->rt_sbt_offset != 0 || p_material->albedo_texture_rd.is_null()) {
		return nullptr;
	}
	if (!(p_material->data.albedo_color[3] > 0.0f) || !(p_material->data.alpha_threshold > 0.0f)) {
		return nullptr; // Everything cut, or nothing: the any-hit is all there is to it.
	}

	RTOmmKey key;
	key.mesh_id = p_mesh.get_id();
	key.surface_index = p_surface_index;
	key.surface_counter = p_surface_counter;
	key.texture_id = p_material->albedo_texture_rd.get_id();
	key.uv_scale[0] = _omm_float_bits(p_material->data.uv1_scale[0]);
	key.uv_scale[1] = _omm_float_bits(p_material->data.uv1_scale[1]);
	key.uv_offset[0] = _omm_float_bits(p_material->data.uv1_offset[0]);
	key.uv_offset[1] = _omm_float_bits(p_material->data.uv1_offset[1]);
	key.albedo_alpha = _omm_float_bits(p_material->data.albedo_color[3]);
	key.threshold = _omm_float_bits(p_material->data.alpha_threshold);

	const uint32_t frame = RSG::rasterizer->get_frame_number();
	RTOmmEntry **found = omm_entries.getptr(key);
	if (found != nullptr) {
		(*found)->last_used_frame = frame;
		return *found;
	}

	// The surface's contents changed: micromaps of its earlier contents can never be traced again.
	LocalVector<RTOmmEntry *> superseded;
	for (const KeyValue<RTOmmKey, RTOmmEntry *> &kv : omm_entries) {
		if (kv.key.mesh_id == key.mesh_id && kv.key.surface_index == key.surface_index && kv.key.surface_counter != key.surface_counter) {
			superseded.push_back(kv.value);
		}
	}
	for (RTOmmEntry *stale : superseded) {
		_omm_release_entry(stale);
	}

	RTOmmEntry *entry = memnew(RTOmmEntry);
	entry->key = key;
	entry->serial = omm_next_serial++;
	entry->last_used_frame = frame;
	omm_entries.insert(key, entry);
	omm_by_serial.insert(entry->serial, entry);
	omm_queue.push_back(entry);
	return entry;
}

void RenderRaytracing::_on_omm_read(const PackedByteArray &p_data, uint64_t p_ticket, bool p_stats) {
	// Called by RenderingDevice as it begins a later frame: queued, processed before the next walk.
	if (singleton == nullptr) {
		return;
	}
	RTOmmReadback readback;
	readback.ticket = p_ticket;
	readback.stats = p_stats;
	readback.data = p_data;
	singleton->omm_readbacks.push_back(readback);
}

void RenderRaytracing::_omm_process() {
	if (!omm_enabled) {
		return;
	}
	RD *rd = RD::get_singleton();
	const uint64_t frames_drawn = rd->get_frames_drawn();
	if (omm_processed_frame == frames_drawn) {
		return; // A second viewport's frame.
	}
	omm_processed_frame = frames_drawn;
	const uint32_t frame = RSG::rasterizer->get_frame_number();

	// Readbacks: a setup's becomes the bake and the micromap, a bake's counts become statistics.
	// Taken off the member first, since a buffer operation below can deliver more into it.
	LocalVector<RTOmmReadback> readbacks;
	readbacks = omm_readbacks;
	omm_readbacks.clear();
	for (const RTOmmReadback &readback : readbacks) {
		RTOmmEntry **found = omm_pending.getptr(readback.ticket);
		if (found == nullptr) {
			continue; // Released while in flight.
		}
		RTOmmEntry *entry = *found;
		omm_pending.erase(readback.ticket);
		entry->ticket = 0;
		if (readback.stats) {
			if (readback.data.size() >= int64_t(sizeof(uint32_t) * 2)) {
				const uint32_t *counts = reinterpret_cast<const uint32_t *>(readback.data.ptr());
				entry->known = counts[0];
				entry->unknown = counts[1];
				omm_stats.known += counts[0];
				omm_stats.unknown += counts[1];
				print_verbose(vformat("RT: opacity micromap [%d]: %d primitives, %d special, %d descriptors, %d KB, micro-triangles %d known and %d unknown.", entry->serial, entry->primitive_count, entry->special_primitives, entry->descriptors, entry->micromap_bytes / 1024, counts[0], counts[1]));
			}
		} else {
			_omm_bake(entry, readback.data);
		}
	}

	// Micromaps unused for the MultiMesh cache's time to live go, with the BLASes that carry them.
	static const uint32_t ttl = uint32_t(MAX(int(GLOBAL_GET("rendering/pathtracing/multimesh_blas_cache_ttl_frames")), 1));
	LocalVector<RTOmmEntry *> expired;
	for (const KeyValue<RTOmmKey, RTOmmEntry *> &kv : omm_entries) {
		if (frame - kv.value->last_used_frame > ttl) {
			expired.push_back(kv.value);
		}
	}
	for (RTOmmEntry *entry : expired) {
		_omm_release_entry(entry);
	}

	for (uint32_t i = 0; i < RT_OMM_SETUPS_PER_FRAME && !omm_queue.is_empty(); i++) {
		RTOmmEntry *entry = omm_queue[0];
		omm_queue.remove_at(0);
		_omm_setup(entry);
	}

	// A pyramid outlives the bakes that read it by the same time to live.
	LocalVector<RID> stale_pyramids;
	for (const KeyValue<RID, RTOmmPyramid> &kv : omm_pyramids) {
		if (frame - kv.value.last_used_frame > ttl) {
			stale_pyramids.push_back(kv.key);
		}
	}
	for (const RID &texture : stale_pyramids) {
		rd->free_rid(omm_pyramids[texture].buffer);
		omm_pyramids.erase(texture);
	}

	if (OS::get_singleton()->is_stdout_verbose() && frames_drawn % 300 == 0 && !omm_entries.is_empty()) {
		const double coverage = omm_stats.known + omm_stats.unknown > 0 ? double(omm_stats.known) / double(omm_stats.known + omm_stats.unknown) : 0.0;
		print_line(vformat("RT: opacity micromaps (D96): %d live, %d baked and %d failed over the run, %d queued, %d in flight; %d KB of micromaps, %d descriptors for %d primitives (%d special); %d BLASes made; micro-triangles %d known and %d unknown (coverage %.4f).", omm_entries.size(), omm_stats.baked, omm_stats.failed, omm_queue.size(), omm_pending.size(), omm_stats.micromap_bytes / 1024, omm_stats.descriptors, omm_stats.primitives, omm_stats.special_primitives, omm_stats.blases, omm_stats.known, omm_stats.unknown, coverage));
	}
}

bool RenderRaytracing::_omm_pyramid(RID p_texture, uint32_t p_width, uint32_t p_height, RTOmmPyramid *&r_pyramid) {
	RD *rd = RD::get_singleton();
	const uint32_t frame = RSG::rasterizer->get_frame_number();
	r_pyramid = omm_pyramids.getptr(p_texture);
	if (r_pyramid != nullptr) {
		r_pyramid->last_used_frame = frame;
		return true;
	}

	// Level k halves level k - 1, rounding up, until one cell covers the texture.
	LocalVector<uint32_t> header;
	header.resize_initialized(RT_OMM_PYRAMID_HEADER_WORDS);
	uint32_t level_count = 0;
	uint64_t words = RT_OMM_PYRAMID_HEADER_WORDS;
	uint32_t width = p_width;
	uint32_t height = p_height;
	while (width > 1 || height > 1) {
		ERR_FAIL_COND_V_MSG(level_count == RT_OMM_PYRAMID_MAX_LEVELS, false, "RT: an alpha texture too large for the opacity micromap pyramid.");
		width = (width + 1) / 2;
		height = (height + 1) / 2;
		header[level_count * 3] = uint32_t(words);
		header[level_count * 3 + 1] = width;
		header[level_count * 3 + 2] = height;
		words += uint64_t(width) * height;
		level_count++;
	}
	ERR_FAIL_COND_V(words * sizeof(uint32_t) > UINT32_MAX, false);

	RTOmmPyramid pyramid;
	pyramid.level_count = level_count;
	pyramid.last_used_frame = frame;
	// The header alone is uploaded: the passes below write every cell of every level.
	pyramid.buffer = rd->storage_buffer_create(uint32_t(words * sizeof(uint32_t)));
	ERR_FAIL_COND_V(pyramid.buffer.is_null(), false);
	rd->buffer_update(pyramid.buffer, 0, header.size() * sizeof(uint32_t), header.ptr());
	rd->set_resource_name(pyramid.buffer, "RT OMM pyramid");

	if (level_count > 0) {
		RD::Uniform texture_uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ omm_sampler, p_texture }));
		RD::Uniform pyramid_uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, pyramid.buffer);
		RID base_set = rd->uniform_set_create(Vector<RD::Uniform>({ texture_uniform, pyramid_uniform }), omm_shader.version_shader[OmmShader::MODE_PYRAMID_BASE], 0, true);
		RID reduce_set = rd->uniform_set_create(Vector<RD::Uniform>({ pyramid_uniform }), omm_shader.version_shader[OmmShader::MODE_PYRAMID_REDUCE], 0, true);
		RTOmmPushConstant pc = {};
		pc.tex_size[0] = p_width;
		pc.tex_size[1] = p_height;
		pc.level_count = level_count;
		RD::ComputeListID list = rd->compute_list_begin();
		pc.pass_level = 1;
		rd->compute_list_bind_compute_pipeline(list, omm_shader.pipeline[OmmShader::MODE_PYRAMID_BASE]);
		rd->compute_list_bind_uniform_set(list, base_set, 0);
		rd->compute_list_set_push_constant(list, &pc, sizeof(pc));
		rd->compute_list_dispatch_threads(list, header[1] * header[2], 1, 1);
		for (uint32_t k = 2; k <= level_count; k++) {
			rd->compute_list_add_barrier(list);
			pc.pass_level = k;
			rd->compute_list_bind_compute_pipeline(list, omm_shader.pipeline[OmmShader::MODE_PYRAMID_REDUCE]);
			rd->compute_list_bind_uniform_set(list, reduce_set, 0);
			rd->compute_list_set_push_constant(list, &pc, sizeof(pc));
			rd->compute_list_dispatch_threads(list, header[(k - 1) * 3 + 1] * header[(k - 1) * 3 + 2], 1, 1);
		}
		rd->compute_list_end();
		rd->free_rid(base_set);
		rd->free_rid(reduce_set);
	}

	omm_pyramids.insert(p_texture, pyramid);
	r_pyramid = omm_pyramids.getptr(p_texture);
	return true;
}

void RenderRaytracing::_omm_setup(RTOmmEntry *p_entry) {
	RD *rd = RD::get_singleton();
	RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();
	auto fail = [&](const char *p_why) {
		print_verbose(vformat("RT: opacity micromap [%d] not baked: %s.", p_entry->serial, p_why));
		p_entry->state = RTOmmEntry::STATE_FAILED;
		omm_stats.failed++;
	};

	// The surface as it was asked for: the same mesh, surface and contents.
	const RID mesh = RID::from_uint64(p_entry->key.mesh_id);
	void *mesh_surface = mesh_storage->owns_mesh(mesh) ? mesh_storage->mesh_get_surface(mesh, p_entry->key.surface_index) : nullptr;
	if (mesh_surface == nullptr || mesh_storage->mesh_surface_get_rt_invalidation_counter(mesh_surface) != p_entry->key.surface_counter) {
		fail("its surface changed or went away");
		return;
	}
	if (mesh_storage->mesh_surface_is_geometry_streamed(mesh_surface) || (mesh_storage->mesh_surface_get_format(mesh_surface) & RSE::ARRAY_FLAG_USE_2D_VERTICES)) {
		fail("its geometry streams or is 2D");
		return;
	}
	const RID texture = RID::from_uint64(p_entry->key.texture_id);
	if (!rd->texture_is_valid(texture)) {
		fail("its alpha texture went away");
		return;
	}

	// The trace's own geometry row, so the setup fetches the UVs as the any-hit does.
	RTSurfaceData meta;
	RID vertex_buffer, attribute_buffer, index_buffer;
	_fill_surface_geometry_data(mesh_surface, false, &meta, &vertex_buffer, &attribute_buffer, &index_buffer);
	RT_GeometryData &geom = meta.geometry;
	if (geom.primitive_count == 0 || geom.uv_byte_offset == RT_OFFSET_NONE || attribute_buffer.is_null()) {
		fail("it has no triangles or no UVs");
		return;
	}
	geom.vertex_buffer_address = vertex_buffer.is_valid() ? rd->buffer_get_device_address(vertex_buffer) : 0;
	geom.attribute_buffer_address = rd->buffer_get_device_address(attribute_buffer);
	rd->buffer_add_device_address_read_stages(attribute_buffer, RDD::PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	if (index_buffer.is_valid() && geom.index_format != RT_INDEX_FORMAT_NONE) {
		geom.index_buffer_address = rd->buffer_get_device_address(index_buffer);
		rd->buffer_add_device_address_read_stages(index_buffer, RDD::PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	}
	p_entry->primitive_count = geom.primitive_count;

	const RD::TextureFormat format = rd->texture_get_format(texture);
	RTOmmPyramid *pyramid = nullptr;
	if (!_omm_pyramid(texture, format.width, format.height, pyramid)) {
		fail("no pyramid for its alpha texture");
		return;
	}

	Vector<uint8_t> row;
	row.resize(sizeof(RT_GeometryData));
	memcpy(row.ptrw(), &geom, sizeof(RT_GeometryData));
	RID row_buffer = rd->storage_buffer_create(row.size(), row);
	const uint64_t setup_bytes = uint64_t(p_entry->primitive_count) * RT_OMM_SETUP_WORDS * sizeof(uint32_t);
	if (row_buffer.is_null() || setup_bytes > UINT32_MAX) {
		if (row_buffer.is_valid()) {
			rd->free_rid(row_buffer);
		}
		fail("its setup buffer could not be made");
		return;
	}
	p_entry->setup_buffer = rd->storage_buffer_create(uint32_t(setup_bytes));
	if (p_entry->setup_buffer.is_null()) {
		rd->free_rid(row_buffer);
		fail("its setup buffer could not be made");
		return;
	}
	rd->set_resource_name(p_entry->setup_buffer, "RT OMM setup [" + itos(p_entry->serial) + "]");

	RD::Uniform texture_uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ omm_sampler, texture }));
	RD::Uniform pyramid_uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, pyramid->buffer);
	RD::Uniform row_uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, row_buffer);
	RD::Uniform out_uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, p_entry->setup_buffer);
	RID set = rd->uniform_set_create(Vector<RD::Uniform>({ texture_uniform, pyramid_uniform, row_uniform, out_uniform }), omm_shader.version_shader[OmmShader::MODE_SETUP], 0, true);
	RTOmmPushConstant pc = _omm_push_constant(p_entry->key, format.width, format.height, pyramid->level_count, omm_scale, omm_max_level);
	pc.count = p_entry->primitive_count;
	RD::ComputeListID list = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(list, omm_shader.pipeline[OmmShader::MODE_SETUP]);
	rd->compute_list_bind_uniform_set(list, set, 0);
	rd->compute_list_set_push_constant(list, &pc, sizeof(pc));
	rd->compute_list_dispatch_threads(list, p_entry->primitive_count, 1, 1);
	rd->compute_list_end();
	rd->free_rid(set);
	rd->free_rid(row_buffer);

	const uint64_t ticket = omm_next_ticket++;
	if (rd->buffer_get_data_async(p_entry->setup_buffer, callable_mp_static(&RenderRaytracing::_on_omm_read).bind(ticket, false)) != OK) {
		rd->free_rid(p_entry->setup_buffer);
		p_entry->setup_buffer = RID();
		fail("its setup could not be read back");
		return;
	}
	p_entry->ticket = ticket;
	p_entry->state = RTOmmEntry::STATE_SETUP;
	omm_pending.insert(ticket, p_entry);
}

void RenderRaytracing::_omm_bake(RTOmmEntry *p_entry, const PackedByteArray &p_setup) {
	RD *rd = RD::get_singleton();
	if (p_entry->setup_buffer.is_valid()) {
		rd->free_rid(p_entry->setup_buffer);
		p_entry->setup_buffer = RID();
	}
	auto fail = [&](const char *p_why) {
		print_verbose(vformat("RT: opacity micromap [%d] not baked: %s.", p_entry->serial, p_why));
		p_entry->state = RTOmmEntry::STATE_FAILED;
		omm_stats.failed++;
	};
	const uint32_t primitive_count = p_entry->primitive_count;
	if (p_setup.size() < int64_t(primitive_count) * RT_OMM_SETUP_WORDS * int64_t(sizeof(uint32_t))) {
		fail("its setup read back short");
		return;
	}
	const RID texture = RID::from_uint64(p_entry->key.texture_id);
	RTOmmPyramid *pyramid = omm_pyramids.getptr(texture);
	if (!rd->texture_is_valid(texture) || pyramid == nullptr) {
		fail("its alpha texture went away");
		return;
	}
	pyramid->last_used_frame = RSG::rasterizer->get_frame_number();
	const uint32_t *setup = reinterpret_cast<const uint32_t *>(p_setup.ptr());

	// Uniform primitives take a special index; the rest one descriptor per distinct UVs and level.
	p_entry->indices.resize(primitive_count);
	HashMap<RTOmmDescriptorKey, uint32_t, RTOmmDescriptorKey> descriptor_of;
	LocalVector<uint32_t> descriptor_source; // The first primitive of each descriptor.
	LocalVector<uint32_t> descriptor_level;
	uint32_t special = 0;
	for (uint32_t p = 0; p < primitive_count; p++) {
		const uint32_t *words = setup + p * RT_OMM_SETUP_WORDS;
		const uint32_t state = words[7];
		if (state == RT_OMM_STATE_TRANSPARENT) {
			p_entry->indices[p] = RD::MICROMAP_SPECIAL_INDEX_FULLY_TRANSPARENT;
			special++;
			continue;
		}
		if (state == RT_OMM_STATE_OPAQUE) {
			p_entry->indices[p] = RD::MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE;
			special++;
			continue;
		}
		RTOmmDescriptorKey key;
		memcpy(key.words, words, sizeof(key.words));
		const uint32_t *existing = descriptor_of.getptr(key);
		if (existing != nullptr) {
			p_entry->indices[p] = int32_t(*existing);
			continue;
		}
		const uint32_t descriptor = descriptor_source.size();
		descriptor_of.insert(key, descriptor);
		descriptor_source.push_back(p);
		descriptor_level.push_back(CLAMP(words[6], 1u, omm_max_level));
		p_entry->indices[p] = int32_t(descriptor);
	}

	// A micromap needs one entry; one no primitive references stands in when every primitive is special.
	const bool placeholder = descriptor_source.is_empty();
	if (placeholder) {
		descriptor_source.push_back(0);
		descriptor_level.push_back(1);
	}

	// Within the byte budget: every level drops by one until the states fit.
	uint64_t state_words = 0;
	for (;;) {
		state_words = 0;
		uint32_t finest = 1;
		for (uint32_t level : descriptor_level) {
			state_words += _omm_level_words(level);
			finest = MAX(finest, level);
		}
		if (state_words * sizeof(uint32_t) <= RT_OMM_MAX_STATE_BYTES || finest == 1) {
			break;
		}
		for (uint32_t &level : descriptor_level) {
			level = MAX(level - 1, 1u);
		}
	}
	if (state_words * sizeof(uint32_t) > RT_OMM_MAX_STATE_BYTES) {
		fail("its states exceed the budget at level 1");
		return;
	}

	// The descriptors (VkMicromapTriangleEXT: byte offset, level, format), the bake's list and the usages.
	const uint32_t descriptor_count = descriptor_source.size();
	Vector<uint8_t> triangles;
	triangles.resize(uint64_t(descriptor_count) * 8);
	Vector<uint8_t> bake_list;
	bake_list.resize(uint64_t(descriptor_count) * 8 * sizeof(uint32_t));
	uint32_t *bake_words = reinterpret_cast<uint32_t *>(bake_list.ptrw());
	uint8_t *triangle_bytes = triangles.ptrw();
	uint32_t array_counts[13] = {};
	uint32_t index_counts[13] = {};
	uint32_t finest = 1;
	uint32_t first_word = 0;
	for (uint32_t d = 0; d < descriptor_count; d++) {
		const uint32_t level = descriptor_level[d];
		const uint32_t data_offset = first_word * sizeof(uint32_t);
		const uint16_t level16 = uint16_t(level);
		const uint16_t format16 = uint16_t(RD::MICROMAP_FORMAT_OPACITY_4_STATE);
		memcpy(triangle_bytes + d * 8, &data_offset, 4);
		memcpy(triangle_bytes + d * 8 + 4, &level16, 2);
		memcpy(triangle_bytes + d * 8 + 6, &format16, 2);
		const uint32_t *source = setup + descriptor_source[d] * RT_OMM_SETUP_WORDS;
		memcpy(bake_words + d * 8, source, 6 * sizeof(uint32_t));
		bake_words[d * 8 + 6] = level;
		bake_words[d * 8 + 7] = first_word;
		first_word += _omm_level_words(level);
		array_counts[level]++;
		finest = MAX(finest, level);
	}
	for (uint32_t p = 0; p < primitive_count; p++) {
		if (p_entry->indices[p] >= 0) {
			index_counts[descriptor_level[p_entry->indices[p]]]++;
		}
	}
	LocalVector<RD::MicromapUsage> array_usages;
	p_entry->index_usages.clear();
	for (uint32_t level = 1; level <= 12; level++) {
		if (array_counts[level] > 0) {
			RD::MicromapUsage usage;
			usage.count = array_counts[level];
			usage.subdivision_level = level;
			array_usages.push_back(usage);
		}
		if (index_counts[level] > 0) {
			RD::MicromapUsage usage;
			usage.count = index_counts[level];
			usage.subdivision_level = level;
			p_entry->index_usages.push_back(usage);
		}
	}

	const BitField<RD::BufferCreationBits> micromap_input = RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_MICROMAP_BUILD_INPUT_READ_ONLY_BIT;
	Vector<uint8_t> index_bytes;
	index_bytes.resize(uint64_t(primitive_count) * sizeof(int32_t));
	memcpy(index_bytes.ptrw(), p_entry->indices.ptr(), index_bytes.size());
	p_entry->index_buffer = rd->storage_buffer_create(index_bytes.size(), index_bytes, 0, RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
	RID triangle_buffer = rd->storage_buffer_create(triangles.size(), triangles, 0, micromap_input);
	RID state_buffer = rd->storage_buffer_create(uint32_t(state_words * sizeof(uint32_t)), {}, 0, micromap_input);
	RID bake_buffer = rd->storage_buffer_create(bake_list.size(), bake_list);
	Vector<uint8_t> zeros;
	zeros.resize_initialized(sizeof(uint32_t) * 2);
	RID stats_buffer = rd->storage_buffer_create(zeros.size(), zeros);
	auto free_inputs = [&]() {
		for (RID buffer : { triangle_buffer, state_buffer, bake_buffer, stats_buffer }) {
			if (buffer.is_valid()) {
				rd->free_rid(buffer);
			}
		}
	};
	if (p_entry->index_buffer.is_null() || triangle_buffer.is_null() || state_buffer.is_null() || bake_buffer.is_null() || stats_buffer.is_null()) {
		free_inputs();
		if (p_entry->index_buffer.is_valid()) {
			rd->free_rid(p_entry->index_buffer);
			p_entry->index_buffer = RID();
		}
		fail("its buffers could not be made");
		return;
	}
	rd->set_resource_name(p_entry->index_buffer, "RT OMM index [" + itos(p_entry->serial) + "]");

	// The bake: a descriptor a row of workgroups, a word of 16 micro-triangles a thread.
	const RD::TextureFormat format = rd->texture_get_format(texture);
	RD::Uniform texture_uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ omm_sampler, texture }));
	RD::Uniform pyramid_uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, pyramid->buffer);
	RD::Uniform list_uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, bake_buffer);
	RD::Uniform state_uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, state_buffer);
	RD::Uniform stats_uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, stats_buffer);
	RID set = rd->uniform_set_create(Vector<RD::Uniform>({ texture_uniform, pyramid_uniform, list_uniform, state_uniform, stats_uniform }), omm_shader.version_shader[OmmShader::MODE_BAKE], 0, true);
	RTOmmPushConstant pc = _omm_push_constant(p_entry->key, format.width, format.height, pyramid->level_count, omm_scale, omm_max_level);
	const uint32_t max_words = _omm_level_words(finest);
	const uint32_t groups_x = MIN((max_words + 63) / 64, 256u);
	pc.word_stride = groups_x * 64;
	const uint32_t max_rows = uint32_t(MIN(rd->limit_get(RD::LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Y), uint64_t(65535)));
	RD::ComputeListID list = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(list, omm_shader.pipeline[OmmShader::MODE_BAKE]);
	rd->compute_list_bind_uniform_set(list, set, 0);
	for (uint32_t base = 0; base < descriptor_count; base += max_rows) {
		pc.base = base;
		pc.count = MIN(descriptor_count - base, max_rows);
		rd->compute_list_set_push_constant(list, &pc, sizeof(pc));
		rd->compute_list_dispatch(list, groups_x, pc.count, 1);
	}
	rd->compute_list_end();
	rd->free_rid(set);

	p_entry->micromap = rd->micromap_create(array_usages);
	if (p_entry->micromap.is_null() || rd->micromap_build(p_entry->micromap, state_buffer, 0, triangle_buffer, 0) != OK) {
		free_inputs();
		if (p_entry->micromap.is_valid()) {
			rd->free_rid(p_entry->micromap);
			p_entry->micromap = RID();
		}
		rd->free_rid(p_entry->index_buffer);
		p_entry->index_buffer = RID();
		fail("its micromap could not be made");
		return;
	}
	rd->set_resource_name(p_entry->micromap, "RT OMM [" + itos(p_entry->serial) + "]");

	const uint64_t ticket = omm_next_ticket++;
	if (rd->buffer_get_data_async(stats_buffer, callable_mp_static(&RenderRaytracing::_on_omm_read).bind(ticket, true)) == OK) {
		p_entry->ticket = ticket;
		omm_pending.insert(ticket, p_entry);
	}
	// The inputs are read by the commands recorded above; their frees wait for the frame.
	free_inputs();

	p_entry->descriptors = placeholder ? 0 : descriptor_count;
	p_entry->special_primitives = special;
	p_entry->micromap_bytes = rd->micromap_get_size(p_entry->micromap);
	p_entry->state = RTOmmEntry::STATE_BUILT;
	omm_stats.baked++;
	omm_stats.micromap_bytes += p_entry->micromap_bytes;
	omm_stats.descriptors += p_entry->descriptors;
	omm_stats.primitives += primitive_count;
	omm_stats.special_primitives += special;
}

static void _omm_geometry_attach(const RTOmmEntry *p_entry, RID p_index_buffer, const LocalVector<RD::MicromapUsage> &p_usages, RD::AccelerationStructureGeometry &r_geometry) {
	r_geometry.geometry.triangles.micromap = p_entry->micromap;
	r_geometry.geometry.triangles.micromap_index_buffer = p_index_buffer;
	r_geometry.geometry.triangles.micromap_index_offset = 0;
	r_geometry.geometry.triangles.micromap_usages.resize(p_usages.size());
	for (uint32_t i = 0; i < p_usages.size(); i++) {
		r_geometry.geometry.triangles.micromap_usages.write[i] = p_usages[i];
	}
}

RID RenderRaytracing::_omm_surface_blas(RTOmmEntry *p_entry, void *p_mesh_surface, const RTSurfaceData *p_surf_data, LocalVector<RID> &r_dirty_blas_list) {
	if (p_entry == nullptr || p_entry->state != RTOmmEntry::STATE_BUILT) {
		return RID();
	}
	RD *rd = RD::get_singleton();
	if (p_entry->static_blas.is_valid()) {
		if (rd->acceleration_structure_is_valid(p_entry->static_blas)) {
			return p_entry->static_blas;
		}
		p_entry->static_blas = RID(); // Freed with its mesh's buffers.
	}

	// A second BLAS over the surface's own buffers, as its plain one is built, carrying the micromap.
	RTSurfaceData meta;
	RID vertex_buffer, attribute_buffer, index_buffer;
	_fill_surface_geometry_data(p_mesh_surface, false, &meta, &vertex_buffer, &attribute_buffer, &index_buffer);
	if (meta.geometry.primitive_count != p_entry->primitive_count || vertex_buffer.is_null() || meta.is_compressed != p_surf_data->is_compressed) {
		return RID();
	}
	RD::AccelerationStructureGeometry as_geom;
	_fill_surface_triangle_geometry(vertex_buffer, index_buffer, meta.geometry, false, meta.is_compressed, as_geom);
	_omm_geometry_attach(p_entry, p_entry->index_buffer, p_entry->index_usages, as_geom);
	p_entry->static_blas = rd->blas_create({ &as_geom, 1 }, RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
	if (p_entry->static_blas.is_null()) {
		return RID();
	}
	rd->set_resource_name(p_entry->static_blas, "RT BLAS OMM [" + itos(p_entry->serial) + "]");
	r_dirty_blas_list.push_back(p_entry->static_blas);
	omm_stats.blases++;
	return p_entry->static_blas;
}

void RenderRaytracing::_omm_attach_merged(RID p_mm_rid, uint32_t p_surface_index, const RTOmmEntry *p_entry) {
	const uint32_t li = p_mm_rid.get_local_index();
	if (li >= mm_handles.size() || mm_handles[li].mm_validator != uint32_t(p_mm_rid.get_id() >> 32) || p_surface_index >= mm_handles[li].per_surface.size()) {
		return; // No merge yet: its first build is plain, and the next frame attaches.
	}
	const RID handle = mm_handles[li].per_surface[p_surface_index];
	RTMergedMMEntry *merged = merged_mm_pool.get_or_null(handle);
	if (merged == nullptr) {
		return;
	}
	uint64_t serial = p_entry != nullptr && p_entry->serial != merged->omm_failed_serial ? p_entry->serial : 0;
	// Viewports that settle the merge differently in one frame would remake it in each walk: once
	// they disagree, it stays plain for the cache's time to live.
	static const uint64_t ttl = uint64_t(MAX(int(GLOBAL_GET("rendering/pathtracing/multimesh_blas_cache_ttl_frames")), 1));
	const uint64_t frame = RD::get_singleton()->get_frames_drawn();
	if (merged->omm_wanted_frame == frame && merged->omm_wanted_serial != serial) {
		merged->omm_conflict_frame = frame;
	}
	merged->omm_wanted_frame = frame;
	merged->omm_wanted_serial = serial;
	if (merged->omm_conflict_frame != UINT64_MAX && frame - merged->omm_conflict_frame <= ttl) {
		serial = 0;
	}
	if (merged->omm_serial == serial) {
		return;
	}
	// Another micromap, or none: the merge makes its BLAS again, built in full this frame.
	if (merged->blas.is_valid()) {
		RD::get_singleton()->free_rid(merged->blas);
		merged->blas = RID();
	}
	merged->blas_built_once = false;
	_omm_release_merged(*merged);
	merged->omm_serial = serial;
	if (serial != 0) {
		RTOmmEntry *entry = omm_by_serial[serial];
		if (entry->merged_handles.find(handle) < 0) {
			entry->merged_handles.push_back(handle);
		}
	}
}

bool RenderRaytracing::_omm_merged_geometry(RTMergedMMEntry &r_entry, uint32_t p_mm_count, uint32_t p_primitives_per_element, RD::AccelerationStructureGeometry &r_geometry) {
	if (r_entry.omm_serial == 0) {
		return false;
	}
	RTOmmEntry **found = omm_by_serial.getptr(r_entry.omm_serial);
	if (found == nullptr || (*found)->state != RTOmmEntry::STATE_BUILT || (*found)->primitive_count != p_primitives_per_element || p_mm_count == 0) {
		r_entry.omm_failed_serial = r_entry.omm_serial;
		r_entry.omm_serial = 0;
		return false;
	}
	const RTOmmEntry *entry = *found;
	RD *rd = RD::get_singleton();

	// The merge repeats the surface element by element, so element e's primitive p is e * P + p.
	const uint64_t bytes = uint64_t(p_mm_count) * p_primitives_per_element * sizeof(int32_t);
	if (bytes > UINT32_MAX) {
		r_entry.omm_failed_serial = r_entry.omm_serial;
		r_entry.omm_serial = 0;
		return false;
	}
	Vector<uint8_t> repeated;
	repeated.resize(bytes);
	uint8_t *dst = repeated.ptrw();
	const uint64_t element_bytes = uint64_t(p_primitives_per_element) * sizeof(int32_t);
	for (uint32_t e = 0; e < p_mm_count; e++) {
		memcpy(dst + e * element_bytes, entry->indices.ptr(), element_bytes);
	}
	if (r_entry.omm_index_buffer.is_valid()) {
		rd->free_rid(r_entry.omm_index_buffer);
	}
	r_entry.omm_index_buffer = rd->storage_buffer_create(repeated.size(), repeated, 0, RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT | RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
	if (r_entry.omm_index_buffer.is_null()) {
		r_entry.omm_failed_serial = r_entry.omm_serial;
		r_entry.omm_serial = 0;
		return false;
	}
	rd->set_resource_name(r_entry.omm_index_buffer, "RT OMM merged index [" + itos(entry->serial) + "]");
	LocalVector<RD::MicromapUsage> usages(entry->index_usages);
	for (RD::MicromapUsage &usage : usages) {
		usage.count *= p_mm_count;
	}
	_omm_geometry_attach(entry, r_entry.omm_index_buffer, usages, r_geometry);
	omm_stats.blases++;
	return true;
}

void RenderRaytracing::_omm_release_merged(RTMergedMMEntry &r_entry) {
	if (r_entry.omm_index_buffer.is_valid()) {
		RD::get_singleton()->free_rid(r_entry.omm_index_buffer);
		r_entry.omm_index_buffer = RID();
	}
	r_entry.omm_serial = 0;
}

void RenderRaytracing::_omm_release_entry(RTOmmEntry *p_entry) {
	RD *rd = RD::get_singleton();
	// The BLASes that carry the micromap go first: a merge makes its BLAS again, plain.
	for (const RID &handle : p_entry->merged_handles) {
		RTMergedMMEntry *merged = merged_mm_pool.get_or_null(handle);
		if (merged == nullptr || merged->omm_serial != p_entry->serial) {
			continue;
		}
		if (merged->blas.is_valid()) {
			rd->free_rid(merged->blas);
			merged->blas = RID();
		}
		merged->blas_built_once = false;
		_omm_release_merged(*merged);
	}
	if (p_entry->static_blas.is_valid() && rd->acceleration_structure_is_valid(p_entry->static_blas)) {
		rd->free_rid(p_entry->static_blas);
	}
	if (p_entry->micromap.is_valid()) {
		rd->free_rid(p_entry->micromap);
	}
	if (p_entry->index_buffer.is_valid()) {
		rd->free_rid(p_entry->index_buffer);
	}
	if (p_entry->setup_buffer.is_valid()) {
		rd->free_rid(p_entry->setup_buffer);
	}
	if (p_entry->ticket != 0) {
		omm_pending.erase(p_entry->ticket);
	}
	int64_t queued = omm_queue.find(p_entry);
	if (queued >= 0) {
		omm_queue.remove_at(queued);
	}
	omm_entries.erase(p_entry->key);
	omm_by_serial.erase(p_entry->serial);
	memdelete(p_entry);
}

void RenderRaytracing::_omm_release_mesh_surface(RID p_mesh, uint32_t p_surface) {
	if (omm_entries.is_empty()) {
		return;
	}
	LocalVector<RTOmmEntry *> released;
	for (const KeyValue<RTOmmKey, RTOmmEntry *> &kv : omm_entries) {
		if (kv.key.mesh_id == p_mesh.get_id() && kv.key.surface_index == p_surface) {
			released.push_back(kv.value);
		}
	}
	for (RTOmmEntry *entry : released) {
		_omm_release_entry(entry);
	}
}
