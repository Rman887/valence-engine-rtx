/**************************************************************************/
/*  mesh_storage.cpp                                                      */
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

#include "mesh_storage.h"

#include "core/config/project_settings.h"
#include "servers/rendering/mesh_cluster_lod.h"
#include "servers/rendering/renderer_viewport.h"
#include "servers/rendering/rendering_server.h"
#include "servers/rendering/rendering_server_types.h"

using namespace RendererRD;

MeshStorage *MeshStorage::singleton = nullptr;

MeshStorage *MeshStorage::get_singleton() {
	return singleton;
}

bool MeshStorage::is_cluster_geometry_streaming_wanted() {
	// Read once: every setting involved needs a restart.
	static int wanted = -1;
	if (wanted < 0) {
		RD *rd = RD::get_singleton();
		const bool supported = rd->has_feature(RD::SUPPORTS_CLUSTER_ACCELERATION_STRUCTURE) && rd->get_cluster_acceleration_structure_capabilities().position_fetch;
		// CLAS streaming (D92) needs the selection, its traversal, the cached levels (with the sharing
		// they belong to) and the compaction.
		const bool stream = GLOBAL_GET("rendering/pathtracing/cluster_lod") && GLOBAL_GET("rendering/pathtracing/cluster_lod_select") && GLOBAL_GET("rendering/pathtracing/cluster_lod_traversal") &&
				GLOBAL_GET("rendering/pathtracing/cluster_lod_share") && GLOBAL_GET("rendering/pathtracing/cluster_lod_cache") && GLOBAL_GET("rendering/pathtracing/cluster_blas_compaction") &&
				GLOBAL_GET("rendering/pathtracing/cluster_lod_stream");
		wanted = supported && stream && GLOBAL_GET("rendering/pathtracing/cluster_lod_stream_geometry") ? 1 : 0;
	}
	return wanted == 1;
}

float MeshStorage::_cluster_geometry_fallback_ratio() {
	static float ratio = -1.0f;
	if (ratio < 0.0f) {
		ratio = CLAMP(float(GLOBAL_GET("rendering/pathtracing/cluster_lod_stream_fallback_ratio")), 0.0f, 1.0f);
	}
	return ratio;
}

bool MeshStorage::_is_cluster_geometry_ram_check_wanted() {
	// Read once, like the switches above. A check of its own rather than --verbose: it reads every
	// streamed surface's buffers back as the surface is added, which stalls RenderingDevice and so
	// moves the frames D92's readbacks land in.
	static int wanted = -1;
	if (wanted < 0) {
		wanted = GLOBAL_GET("rendering/pathtracing/cluster_lod_stream_geometry_ram_check") ? 1 : 0;
	}
	return wanted == 1;
}

MeshStorage::MeshStorage() {
	singleton = this;

	default_rd_storage_buffer = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t) * 4);

	//default rd buffers
	{
		Vector<uint8_t> buffer;
		{
			buffer.resize(sizeof(float) * 3);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = reinterpret_cast<float *>(w);
				fptr[0] = 0.0;
				fptr[1] = 0.0;
				fptr[2] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_VERTEX] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //normal
			buffer.resize(sizeof(float) * 3);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = reinterpret_cast<float *>(w);
				fptr[0] = 1.0;
				fptr[1] = 0.0;
				fptr[2] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_NORMAL] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //tangent
			buffer.resize(sizeof(float) * 4);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = reinterpret_cast<float *>(w);
				fptr[0] = 1.0;
				fptr[1] = 0.0;
				fptr[2] = 0.0;
				fptr[3] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_TANGENT] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //color
			buffer.resize(sizeof(float) * 4);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = reinterpret_cast<float *>(w);
				fptr[0] = 1.0;
				fptr[1] = 1.0;
				fptr[2] = 1.0;
				fptr[3] = 1.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_COLOR] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //tex uv 1
			buffer.resize(sizeof(float) * 2);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = reinterpret_cast<float *>(w);
				fptr[0] = 0.0;
				fptr[1] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_TEX_UV] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}
		{ //tex uv 2
			buffer.resize(sizeof(float) * 2);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = reinterpret_cast<float *>(w);
				fptr[0] = 0.0;
				fptr[1] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_TEX_UV2] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		for (int i = 0; i < RSE::ARRAY_CUSTOM_COUNT; i++) {
			buffer.resize(sizeof(float) * 4);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = reinterpret_cast<float *>(w);
				fptr[0] = 0.0;
				fptr[1] = 0.0;
				fptr[2] = 0.0;
				fptr[3] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_CUSTOM0 + i] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //bones
			buffer.resize(sizeof(uint32_t) * 4);
			{
				uint8_t *w = buffer.ptrw();
				uint32_t *fptr = reinterpret_cast<uint32_t *>(w);
				fptr[0] = 0;
				fptr[1] = 0;
				fptr[2] = 0;
				fptr[3] = 0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_BONES] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //weights
			buffer.resize(sizeof(float) * 4);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = reinterpret_cast<float *>(w);
				fptr[0] = 0.0;
				fptr[1] = 0.0;
				fptr[2] = 0.0;
				fptr[3] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_WEIGHTS] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}
	}

	{
		Vector<String> skeleton_modes;
		skeleton_modes.push_back("\n#define MODE_2D\n");
		skeleton_modes.push_back("");

		skeleton_shader.shader.initialize(skeleton_modes);
		skeleton_shader.version = skeleton_shader.shader.version_create();
		for (int i = 0; i < SkeletonShader::SHADER_MODE_MAX; i++) {
			skeleton_shader.version_shader[i] = skeleton_shader.shader.version_get_shader(skeleton_shader.version, i);
			skeleton_shader.pipeline[i] = RD::get_singleton()->compute_pipeline_create(skeleton_shader.version_shader[i]);
		}

		{
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.binding = 0;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.append_id(default_rd_storage_buffer);
				uniforms.push_back(u);
			}
			skeleton_shader.default_skeleton_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, skeleton_shader.version_shader[0], SkeletonShader::UNIFORM_SET_SKELETON);
		}
	}
}

MeshStorage::~MeshStorage() {
	//def buffers
	for (int i = 0; i < DEFAULT_RD_BUFFER_MAX; i++) {
		RD::get_singleton()->free_rid(mesh_default_rd_buffers[i]);
	}

	skeleton_shader.shader.version_free(skeleton_shader.version);

	RD::get_singleton()->free_rid(default_rd_storage_buffer);

	singleton = nullptr;
}

bool MeshStorage::free(RID p_rid) {
	if (owns_mesh(p_rid)) {
		mesh_free(p_rid);
		return true;
	} else if (owns_mesh_instance(p_rid)) {
		mesh_instance_free(p_rid);
		return true;
	} else if (owns_multimesh(p_rid)) {
		multimesh_free(p_rid);
		return true;
	} else if (owns_skeleton(p_rid)) {
		skeleton_free(p_rid);
		return true;
	}

	return false;
}

/* MESH API */

RID MeshStorage::mesh_allocate() {
	return mesh_owner.allocate_rid();
}

void MeshStorage::mesh_initialize(RID p_rid) {
	mesh_owner.initialize_rid(p_rid, Mesh());
}

void MeshStorage::mesh_free(RID p_rid) {
	mesh_clear(p_rid);
	mesh_set_shadow_mesh(p_rid, RID());
	Mesh *mesh = mesh_owner.get_or_null(p_rid);
	ERR_FAIL_NULL(mesh);

	mesh->dependency.deleted_notify(p_rid);
	if (mesh->instances.size()) {
		ERR_PRINT("deleting mesh with active instances");
	}
	if (mesh->shadow_owners.size()) {
		for (Mesh *E : mesh->shadow_owners) {
			Mesh *shadow_owner = E;
			shadow_owner->shadow_mesh = RID();
			shadow_owner->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
		}
	}
	mesh_owner.free(p_rid);
}

void MeshStorage::mesh_set_blend_shape_count(RID p_mesh, int p_blend_shape_count) {
	ERR_FAIL_COND(p_blend_shape_count < 0);

	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);

	ERR_FAIL_COND(mesh->surface_count > 0); //surfaces already exist

	mesh->blend_shape_count = p_blend_shape_count;
}

// Triangle clusters at load (D80): a 3D triangle-list surface arriving without its cluster table
// (saved or imported before the setting was on, or handed over as packed data) is clustered here
// from its packed positions and indices, so the cluster BLAS builds from spatial clusters and not
// from runs. Positions are read as RenderingServer::_get_array_from_surface reads them. The table
// is not written back to the resource; a mesh re-saved or re-imported with the setting on carries
// it and skips this step.
// Skinned and blend-shape surfaces are clustered too (D82): only the index data is reordered, so
// the skin and blend-shape data, both in vertex order, are untouched.
static void _cluster_surface_at_load(RenderingServerTypes::SurfaceData &r_surface) {
	if (r_surface.cluster_data.size() || r_surface.index_count == 0 || r_surface.index_count % 3 != 0 || r_surface.vertex_count == 0 || r_surface.primitive != RSE::PRIMITIVE_TRIANGLES || !(r_surface.format & RSE::ARRAY_FORMAT_VERTEX) || (r_surface.format & RSE::ARRAY_FLAG_USE_2D_VERTICES)) {
		return;
	}
	if (RenderingServer::build_clusters_func == nullptr || !GLOBAL_GET("rendering/pathtracing/cluster_blas")) {
		return;
	}

	uint32_t offsets[RSE::ARRAY_MAX];
	uint32_t vertex_element_size = 0, normal_element_size = 0, attrib_element_size = 0, skin_element_size = 0;
	RS::get_singleton()->mesh_surface_make_offsets_from_format(r_surface.format, r_surface.vertex_count, r_surface.index_count, offsets, vertex_element_size, normal_element_size, attrib_element_size, skin_element_size);
	ERR_FAIL_COND(uint64_t(offsets[RSE::ARRAY_VERTEX]) + uint64_t(vertex_element_size) * r_surface.vertex_count > uint64_t(r_surface.vertex_data.size()));

	Vector<Vector3> positions;
	positions.resize(r_surface.vertex_count);
	{
		Vector3 *w = positions.ptrw();
		const uint8_t *r = r_surface.vertex_data.ptr();
		const bool compressed = r_surface.format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
		for (uint32_t j = 0; j < r_surface.vertex_count; j++) {
			const uint8_t *v = &r[j * vertex_element_size + offsets[RSE::ARRAY_VERTEX]];
			if (compressed) {
				const uint16_t *q = reinterpret_cast<const uint16_t *>(v);
				w[j] = Vector3(float(q[0]) / 65535.0, float(q[1]) / 65535.0, float(q[2]) / 65535.0) * r_surface.aabb.size + r_surface.aabb.position;
			} else {
				const float *f = reinterpret_cast<const float *>(v);
				w[j] = Vector3(f[0], f[1], f[2]);
			}
		}
	}

	const bool index_16 = r_surface.vertex_count <= 65536;
	const uint32_t index_size = index_16 ? sizeof(uint16_t) : sizeof(uint32_t);
	ERR_FAIL_COND(uint64_t(r_surface.index_count) * index_size != uint64_t(r_surface.index_data.size()));
	Vector<int> indices;
	indices.resize(r_surface.index_count);
	{
		int *w = indices.ptrw();
		const uint8_t *r = r_surface.index_data.ptr();
		for (uint32_t i = 0; i < r_surface.index_count; i++) {
			w[i] = index_16 ? int(reinterpret_cast<const uint16_t *>(r)[i]) : int(reinterpret_cast<const uint32_t *>(r)[i]);
		}
	}

	Vector<uint8_t> cluster_data = RenderingServer::mesh_surface_build_clusters(positions, indices);
	if (cluster_data.is_empty()) {
		return;
	}
	{
		uint8_t *w = r_surface.index_data.ptrw();
		const int *r = indices.ptr();
		for (uint32_t i = 0; i < r_surface.index_count; i++) {
			if (index_16) {
				reinterpret_cast<uint16_t *>(w)[i] = uint16_t(r[i]);
			} else {
				reinterpret_cast<uint32_t *>(w)[i] = uint32_t(r[i]);
			}
		}
	}
	r_surface.cluster_data = cluster_data;
}

// The coarse levels of a cluster LOD hierarchy (D85): while rendering/pathtracing/cluster_blas
// and cluster_lod are both on, and the surface's table is version 2 and describes its index
// data, the coarse clusters' triangles are packed to the surface's index format and appended to
// r_index_data, so the index buffer holds every level. Otherwise the buffer stays the drawn
// range and the RT cache, which reads the buffer's length (Surface::index_total), traces LOD 0
// from the table's first clusters. A coarse index outside the surface leaves the buffer at the
// drawn range too.
static void _cluster_lod_index_tail(const RenderingServerTypes::SurfaceData &p_surface, bool p_is_index_16, Vector<uint8_t> &r_index_data, uint32_t &r_index_total) {
	const Vector<uint8_t> &table_bytes = p_surface.cluster_data;
	const uint64_t words = table_bytes.size() / sizeof(uint32_t);
	if (words < RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS || table_bytes.size() % sizeof(uint32_t) != 0) {
		return;
	}
	const uint32_t *table = reinterpret_cast<const uint32_t *>(table_bytes.ptr());
	if (table[0] != RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION) {
		return;
	}
	if (!GLOBAL_GET("rendering/pathtracing/cluster_blas") || !GLOBAL_GET("rendering/pathtracing/cluster_lod")) {
		return;
	}
	if (table[7] & RenderingServerTypes::MESH_CLUSTER_LOD_FLAG_STALE) {
		return; // The coarse levels are traced no more (D86); the tail would sit unused.
	}
	const uint32_t cluster_count = table[1];
	const uint32_t group_count = table[2];
	const uint32_t lod0_triangle_count = table[5];
	const uint32_t coarse_triangle_count = table[6];
	const uint64_t expected_words = RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS + uint64_t(RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS) * cluster_count + uint64_t(RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS) * group_count + uint64_t(coarse_triangle_count) * 3;
	if (words != expected_words || coarse_triangle_count == 0 || uint64_t(lod0_triangle_count) * 3 != p_surface.index_count) {
		return;
	}
	const uint32_t index_size = p_is_index_16 ? 2 : 4;
	if (uint64_t(p_surface.index_count) * index_size != uint64_t(p_surface.index_data.size())) {
		return;
	}
	const uint32_t *coarse = table + (expected_words - uint64_t(coarse_triangle_count) * 3);
	const uint32_t coarse_index_count = coarse_triangle_count * 3;
	const uint64_t tail_bytes = uint64_t(coarse_index_count) * index_size;
	const uint64_t old_bytes = r_index_data.size();
	r_index_data.resize(old_bytes + tail_bytes);
	uint8_t *tail = r_index_data.ptrw() + old_bytes;
	const uint32_t vertex_count = p_surface.vertex_count;
	for (uint32_t i = 0; i < coarse_index_count; i++) {
		const uint32_t index = coarse[i];
		if (index >= vertex_count) {
			WARN_PRINT_ONCE("A surface's cluster LOD table indexes outside the surface; its coarse levels are left out.");
			r_index_data.resize(old_bytes);
			return;
		}
		if (p_is_index_16) {
			reinterpret_cast<uint16_t *>(tail)[i] = uint16_t(index);
		} else {
			reinterpret_cast<uint32_t *>(tail)[i] = index;
		}
	}
	r_index_total = p_surface.index_count + coarse_index_count;
}

// The fallback level of a surface that streams its geometry (D93d), the arrays the raster's buffers
// are made from: mesh_cluster_lod_fallback_level()'s triangles, read from the index data with its
// coarse tail, their vertices compacted in first-use order (r_vertices maps each back to the
// surface's own), the indices remapped and packed to 16 bits when the level's vertices allow. The
// vertex data keeps the surface's layout, the positions and then the normal block, and the attribute
// data its interleaving. False when the arrays do not have the size the format gives them.
static bool _cluster_fallback_arrays(const RenderingServerTypes::SurfaceData &p_surface, const Vector<uint8_t> &p_index_data, bool p_index_16, float p_ratio, LocalVector<uint32_t> &r_vertices, Vector<uint8_t> &r_vertex_data, Vector<uint8_t> &r_attribute_data, Vector<uint8_t> &r_index_data, uint32_t &r_vertex_count, uint32_t &r_index_count, bool &r_index_16) {
	uint32_t offsets[RSE::ARRAY_MAX];
	uint32_t vertex_element_size = 0, normal_element_size = 0, attrib_element_size = 0, skin_element_size = 0;
	RS::get_singleton()->mesh_surface_make_offsets_from_format(p_surface.format, p_surface.vertex_count, p_surface.index_count, offsets, vertex_element_size, normal_element_size, attrib_element_size, skin_element_size);
	const uint64_t vertex_count = p_surface.vertex_count;
	if (uint64_t(p_surface.vertex_data.size()) != (uint64_t(vertex_element_size) + normal_element_size) * vertex_count || uint64_t(p_surface.attribute_data.size()) != uint64_t(attrib_element_size) * vertex_count) {
		return false;
	}

	LocalVector<uint32_t> runs;
	uint32_t depth = 0;
	uint64_t triangles = 0;
	mesh_cluster_lod_fallback_level(p_surface.cluster_data, p_ratio, runs, depth, triangles);
	if (triangles == 0 || triangles * 3 > UINT32_MAX) {
		return false;
	}

	// The level's indices into the surface, then into its compacted vertices.
	LocalVector<uint32_t> remap;
	remap.resize(p_surface.vertex_count);
	memset(remap.ptr(), 0xFF, sizeof(uint32_t) * remap.size());
	LocalVector<uint32_t> indices;
	indices.reserve(uint32_t(triangles * 3));
	r_vertices.clear();
	const uint8_t *index_bytes = p_index_data.ptr();
	for (uint32_t r = 0; r < runs.size(); r += 2) {
		const uint64_t first = uint64_t(runs[r]) * 3;
		const uint64_t end = first + uint64_t(runs[r + 1]) * 3;
		for (uint64_t i = first; i < end; i++) {
			const uint32_t index = p_index_16 ? reinterpret_cast<const uint16_t *>(index_bytes)[i] : reinterpret_cast<const uint32_t *>(index_bytes)[i];
			if (remap[index] == UINT32_MAX) {
				remap[index] = r_vertices.size();
				r_vertices.push_back(index);
			}
			indices.push_back(remap[index]);
		}
	}

	r_vertex_count = r_vertices.size();
	r_index_count = indices.size();
	r_index_16 = r_vertex_count <= 65536;
	r_vertex_data.resize(uint64_t(vertex_element_size + normal_element_size) * r_vertex_count);
	r_attribute_data.resize(uint64_t(attrib_element_size) * r_vertex_count);
	r_index_data.resize(uint64_t(r_index_count) * (r_index_16 ? sizeof(uint16_t) : sizeof(uint32_t)));
	const uint8_t *vertex_from = p_surface.vertex_data.ptr();
	const uint8_t *attribute_from = p_surface.attribute_data.ptr();
	uint8_t *vertex_to = r_vertex_data.ptrw();
	uint8_t *attribute_to = r_attribute_data.ptrw();
	for (uint32_t v = 0; v < r_vertex_count; v++) {
		const uint64_t o = r_vertices[v];
		memcpy(vertex_to + uint64_t(v) * vertex_element_size, vertex_from + o * vertex_element_size, vertex_element_size);
		if (normal_element_size) {
			memcpy(vertex_to + uint64_t(vertex_element_size) * r_vertex_count + uint64_t(v) * normal_element_size, vertex_from + uint64_t(vertex_element_size) * vertex_count + o * normal_element_size, normal_element_size);
		}
		if (attrib_element_size) {
			memcpy(attribute_to + uint64_t(v) * attrib_element_size, attribute_from + o * attrib_element_size, attrib_element_size);
		}
	}
	uint8_t *index_to = r_index_data.ptrw();
	for (uint32_t i = 0; i < r_index_count; i++) {
		if (r_index_16) {
			reinterpret_cast<uint16_t *>(index_to)[i] = uint16_t(indices[i]);
		} else {
			reinterpret_cast<uint32_t *>(index_to)[i] = indices[i];
		}
	}
	return true;
}

// The attribute bytes of a surface's fallback level, from its RAM copy (D93d).
static void _cluster_fallback_attributes(const Vector<uint8_t> &p_attribute_data, uint32_t p_vertex_count, const LocalVector<uint32_t> &p_vertices, Vector<uint8_t> &r_attribute_data) {
	const uint64_t stride = p_vertex_count ? p_attribute_data.size() / p_vertex_count : 0;
	r_attribute_data.resize(stride * p_vertices.size());
	uint8_t *to = r_attribute_data.ptrw();
	for (uint32_t v = 0; v < p_vertices.size(); v++) {
		memcpy(to + uint64_t(v) * stride, p_attribute_data.ptr() + uint64_t(p_vertices[v]) * stride, stride);
	}
}

/// Returns stride
void MeshStorage::mesh_add_surface(RID p_mesh, const RenderingServerTypes::SurfaceData &p_surface) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);

	ERR_FAIL_COND(mesh->surface_count == RSE::MAX_MESH_SURFACES);

#ifdef DEBUG_ENABLED
	//do a validation, to catch errors first
	{
		uint32_t stride = 0;
		uint32_t attrib_stride = 0;
		uint32_t skin_stride = 0;

		for (int i = 0; i < RSE::ARRAY_WEIGHTS; i++) {
			if ((p_surface.format & (1ULL << i))) {
				switch (i) {
					case RSE::ARRAY_VERTEX: {
						if ((p_surface.format & RSE::ARRAY_FLAG_USE_2D_VERTICES) || (p_surface.format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES)) {
							stride += sizeof(float) * 2;
						} else {
							stride += sizeof(float) * 3;
						}

					} break;
					case RSE::ARRAY_NORMAL: {
						stride += sizeof(uint16_t) * 2;

					} break;
					case RSE::ARRAY_TANGENT: {
						if (!(p_surface.format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES)) {
							stride += sizeof(uint16_t) * 2;
						}
					} break;
					case RSE::ARRAY_COLOR: {
						attrib_stride += sizeof(uint32_t);
					} break;
					case RSE::ARRAY_TEX_UV: {
						if (p_surface.format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES) {
							attrib_stride += sizeof(uint16_t) * 2;
						} else {
							attrib_stride += sizeof(float) * 2;
						}

					} break;
					case RSE::ARRAY_TEX_UV2: {
						if (p_surface.format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES) {
							attrib_stride += sizeof(uint16_t) * 2;
						} else {
							attrib_stride += sizeof(float) * 2;
						}

					} break;
					case RSE::ARRAY_CUSTOM0:
					case RSE::ARRAY_CUSTOM1:
					case RSE::ARRAY_CUSTOM2:
					case RSE::ARRAY_CUSTOM3: {
						int idx = i - RSE::ARRAY_CUSTOM0;
						const uint32_t fmt_shift[RSE::ARRAY_CUSTOM_COUNT] = { RSE::ARRAY_FORMAT_CUSTOM0_SHIFT, RSE::ARRAY_FORMAT_CUSTOM1_SHIFT, RSE::ARRAY_FORMAT_CUSTOM2_SHIFT, RSE::ARRAY_FORMAT_CUSTOM3_SHIFT };
						uint32_t fmt = (p_surface.format >> fmt_shift[idx]) & RSE::ARRAY_FORMAT_CUSTOM_MASK;
						const uint32_t fmtsize[RSE::ARRAY_CUSTOM_MAX] = { 4, 4, 4, 8, 4, 8, 12, 16 };
						attrib_stride += fmtsize[fmt];

					} break;
					case RSE::ARRAY_WEIGHTS:
					case RSE::ARRAY_BONES: {
						//uses a separate array
						bool use_8 = p_surface.format & RSE::ARRAY_FLAG_USE_8_BONE_WEIGHTS;
						skin_stride += sizeof(int16_t) * (use_8 ? 16 : 8);
					} break;
				}
			}
		}

		int expected_size = stride * p_surface.vertex_count;
		ERR_FAIL_COND_MSG(expected_size != p_surface.vertex_data.size(), "Size of vertex data provided (" + itos(p_surface.vertex_data.size()) + ") does not match expected (" + itos(expected_size) + ")");

		int bs_expected_size = expected_size * mesh->blend_shape_count;

		ERR_FAIL_COND_MSG(bs_expected_size != p_surface.blend_shape_data.size(), "Size of blend shape data provided (" + itos(p_surface.blend_shape_data.size()) + ") does not match expected (" + itos(bs_expected_size) + ")");

		int expected_attrib_size = attrib_stride * p_surface.vertex_count;
		ERR_FAIL_COND_MSG(expected_attrib_size != p_surface.attribute_data.size(), "Size of attribute data provided (" + itos(p_surface.attribute_data.size()) + ") does not match expected (" + itos(expected_attrib_size) + ")");

		if ((p_surface.format & RSE::ARRAY_FORMAT_WEIGHTS) && (p_surface.format & RSE::ARRAY_FORMAT_BONES)) {
			expected_size = skin_stride * p_surface.vertex_count;
			ERR_FAIL_COND_MSG(expected_size != p_surface.skin_data.size(), "Size of skin data provided (" + itos(p_surface.skin_data.size()) + ") does not match expected (" + itos(expected_size) + ")");
		}
	}

#endif

	uint64_t surface_version = p_surface.format & (uint64_t(RSE::ARRAY_FLAG_FORMAT_VERSION_MASK) << RSE::ARRAY_FLAG_FORMAT_VERSION_SHIFT);
	RenderingServerTypes::SurfaceData new_surface = p_surface;
#ifdef DISABLE_DEPRECATED

	ERR_FAIL_COND_MSG(surface_version != RSE::ARRAY_FLAG_FORMAT_CURRENT_VERSION, "Surface version provided (" + itos(int(surface_version >> RSE::ARRAY_FLAG_FORMAT_VERSION_SHIFT)) + ") does not match current version (" + itos(RSE::ARRAY_FLAG_FORMAT_CURRENT_VERSION >> RSE::ARRAY_FLAG_FORMAT_VERSION_SHIFT) + ")");

#else

	if (surface_version != uint64_t(RSE::ARRAY_FLAG_FORMAT_CURRENT_VERSION)) {
		RS::get_singleton()->fix_surface_compatibility(new_surface);
		surface_version = new_surface.format & (RSE::ARRAY_FLAG_FORMAT_VERSION_MASK << RSE::ARRAY_FLAG_FORMAT_VERSION_SHIFT);
		ERR_FAIL_COND_MSG(surface_version != RSE::ARRAY_FLAG_FORMAT_CURRENT_VERSION,
				vformat("Surface version provided (%d) does not match current version (%d).",
						(surface_version >> RSE::ARRAY_FLAG_FORMAT_VERSION_SHIFT) & RSE::ARRAY_FLAG_FORMAT_VERSION_MASK,
						(RSE::ARRAY_FLAG_FORMAT_CURRENT_VERSION >> RSE::ARRAY_FLAG_FORMAT_VERSION_SHIFT) & RSE::ARRAY_FLAG_FORMAT_VERSION_MASK));
	}
#endif

	_cluster_surface_at_load(new_surface);

	static uint32_t s_next_rt_invalidation_counter = 1; // Monotonic counter is required so that new surfaces are also invalidated.
	Mesh::Surface *s = memnew(Mesh::Surface);
	s->rt_invalidation_counter = s_next_rt_invalidation_counter++;

	s->format = new_surface.format;
	s->primitive = new_surface.primitive;

	const bool use_as_storage = (new_surface.skin_data.size() || mesh->blend_shape_count > 0);
	const bool requested_storage_buffer = (new_surface.format & RSE::ARRAY_FLAG_USE_STORAGE_BUFFER);
	BitField<RD::BufferCreationBits> as_storage_flag = (use_as_storage || requested_storage_buffer) ? RD::BUFFER_CREATION_AS_STORAGE_BIT : 0;
	BitField<RD::BufferCreationBits> requested_storage_flag = requested_storage_buffer ? RD::BUFFER_CREATION_AS_STORAGE_BIT : 0;
	// Raytracing needs all geometry buffers usable as acceleration-structure build
	// inputs and addressable via buffer device address for bindless closest-hit access.
	if (RD::get_singleton()->has_feature(RD::SUPPORTS_RAYTRACING_PIPELINE) || RD::get_singleton()->has_feature(RD::SUPPORTS_RAY_QUERY)) {
		as_storage_flag.set_flag(RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT);
		as_storage_flag.set_flag(RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
		as_storage_flag.set_flag(RD::BUFFER_CREATION_AS_STORAGE_BIT);
		requested_storage_flag.set_flag(RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT);
		requested_storage_flag.set_flag(RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
		requested_storage_flag.set_flag(RD::BUFFER_CREATION_AS_STORAGE_BIT);
	}

	// A cluster LOD hierarchy (D85) appends its coarse levels' triangles to the index buffer
	// past the drawn range: the raster's index array covers index_count as before, the trace
	// addresses every cluster through the one buffer.
	const bool is_index_16 = new_surface.vertex_count <= 65536 && new_surface.vertex_count > 0;
	Vector<uint8_t> index_data = new_surface.index_data;
	uint32_t index_total = new_surface.index_count;
	if (new_surface.index_count) {
		_cluster_lod_index_tail(new_surface, is_index_16, index_data, index_total);
	}

	// Geometry residency (D93): a surface that streams its geometry keeps in RAM the arrays it was
	// added with (the index data with its coarse tail), for the trace's blobs and for
	// mesh_get_surface(); the vectors share the caller's data. Its buffers hold its fallback level
	// (D93d), which every raster consumer draws, and the trace reads the whole surface through
	// blobs or an upload of its own. A skinned or blend-shaped surface is deformed, and deformed
	// surfaces do not stream; skin data marks one even without the bones flag, as for
	// use_as_storage above. The mesh storage decides, and is the authority: the tables' tests are
	// the ones the trace takes before it streams a surface, but the trace streams a subset of these
	// surfaces (not the ones no instance traces, not while streaming is off or after a failed
	// flush, not where its population fails) and uploads the rest from this copy.
	Vector<uint8_t> buffer_vertex_data = new_surface.vertex_data;
	Vector<uint8_t> buffer_attribute_data = new_surface.attribute_data;
	Vector<uint8_t> buffer_index_data = index_data;
	uint32_t buffer_vertex_count = new_surface.vertex_count;
	uint32_t buffer_index_count = new_surface.index_count;
	uint32_t buffer_index_total = index_total;
	bool buffer_index_16 = is_index_16;
	if (index_total > new_surface.index_count && !(new_surface.format & RSE::ARRAY_FORMAT_BONES) && new_surface.skin_data.is_empty() && mesh->blend_shape_count == 0 && is_cluster_geometry_streaming_wanted()) {
		const RD::ClusterAccelerationStructureCapabilities &caps = RD::get_singleton()->get_cluster_acceleration_structure_capabilities();
		LocalVector<uint16_t> local;
		LocalVector<uint32_t> distinct;
		LocalVector<uint32_t> fallback_vertices;
		if (mesh_cluster_lod_streamable(new_surface.cluster_data, new_surface.index_count, index_total, caps.max_triangles_per_cluster, caps.max_vertices_per_cluster) &&
				mesh_cluster_lod_local_vertices(new_surface.cluster_data, index_data.ptr(), index_data.size(), is_index_16, new_surface.vertex_count, local, distinct) &&
				_cluster_fallback_arrays(new_surface, index_data, is_index_16, _cluster_geometry_fallback_ratio(), fallback_vertices, buffer_vertex_data, buffer_attribute_data, buffer_index_data, buffer_vertex_count, buffer_index_count, buffer_index_16)) {
			buffer_index_total = buffer_index_count;
			s->fallback_in_buffers = true;
			s->ram_vertex_data = new_surface.vertex_data;
			s->ram_attribute_data = new_surface.attribute_data;
			s->ram_index_data = index_data;
			s->ram_lods = new_surface.lods;
			s->ram_fallback_vertices = fallback_vertices;
		} else {
			buffer_vertex_data = new_surface.vertex_data;
			buffer_attribute_data = new_surface.attribute_data;
			buffer_index_data = index_data;
			buffer_vertex_count = new_surface.vertex_count;
			buffer_index_count = new_surface.index_count;
			buffer_index_16 = is_index_16;
		}
	}

	if (buffer_vertex_data.size()) {
		// If we have an uncompressed surface that contains normals, but not tangents, we need to differentiate the array
		// from a compressed array in the shader. To do so, we allow the normal to read 4 components out of the buffer
		// But only give it 2 components per normal. So essentially, each vertex reads the next normal in normal.zw.
		// This allows us to avoid adding a shader permutation, and avoid passing dummy tangents. Since the stride is kept small
		// this should still be a net win for bandwidth.
		// If we do this, then the last normal will read past the end of the array. So we need to pad the array with dummy data.
		if (!(new_surface.format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES) && (new_surface.format & RSE::ARRAY_FORMAT_NORMAL) && !(new_surface.format & RSE::ARRAY_FORMAT_TANGENT)) {
			// Unfortunately, we need to copy the buffer, which is fine as doing a resize triggers a CoW anyway.
			Vector<uint8_t> new_vertex_data;
			new_vertex_data.resize_initialized(buffer_vertex_data.size() + sizeof(uint16_t) * 2);
			memcpy(new_vertex_data.ptrw(), buffer_vertex_data.ptr(), buffer_vertex_data.size());
			s->vertex_buffer = RD::get_singleton()->vertex_buffer_create(new_vertex_data.size(), new_vertex_data, as_storage_flag);
			s->vertex_buffer_size = new_vertex_data.size();
		} else {
			s->vertex_buffer = RD::get_singleton()->vertex_buffer_create(buffer_vertex_data.size(), buffer_vertex_data, as_storage_flag);
			s->vertex_buffer_size = buffer_vertex_data.size();
		}
	}

	if (buffer_attribute_data.size()) {
		s->attribute_buffer = RD::get_singleton()->vertex_buffer_create(buffer_attribute_data.size(), buffer_attribute_data, requested_storage_flag);
		s->attribute_buffer_size = buffer_attribute_data.size();
	}
	if (new_surface.skin_data.size()) {
		s->skin_buffer = RD::get_singleton()->vertex_buffer_create(new_surface.skin_data.size(), new_surface.skin_data, as_storage_flag);
		s->skin_buffer_size = new_surface.skin_data.size();
	}

	s->vertex_count = buffer_vertex_count;
	s->rt_vertex_count = new_surface.vertex_count;
	s->rt_index_count = new_surface.index_count;
	s->rt_index_total = index_total;

	if (new_surface.format & RSE::ARRAY_FORMAT_BONES) {
		mesh->has_bone_weights = true;
	}

	if (new_surface.index_count) {
		s->index_buffer = RD::get_singleton()->index_buffer_create(buffer_index_total, buffer_index_16 ? RD::INDEX_BUFFER_FORMAT_UINT16 : RD::INDEX_BUFFER_FORMAT_UINT32, buffer_index_data, false, requested_storage_flag);
		s->index_buffer_size = buffer_index_data.size();
		s->index_count = buffer_index_count;
		s->index_total = buffer_index_total;
		s->index_array = RD::get_singleton()->index_array_create(s->index_buffer, 0, s->index_count);
		// Godot's LODs index the surface's own vertices; a fallback level is drawn without them.
		if (new_surface.lods.size() && !mesh_surface_is_geometry_streamed(s)) {
			s->lods = memnew_arr(Mesh::Surface::LOD, new_surface.lods.size());
			s->lod_count = new_surface.lods.size();

			for (int i = 0; i < new_surface.lods.size(); i++) {
				uint32_t indices = new_surface.lods[i].index_data.size() / (is_index_16 ? 2 : 4);
				s->lods[i].index_buffer = RD::get_singleton()->index_buffer_create(indices, is_index_16 ? RD::INDEX_BUFFER_FORMAT_UINT16 : RD::INDEX_BUFFER_FORMAT_UINT32, new_surface.lods[i].index_data, false, requested_storage_flag);
				s->lods[i].index_buffer_size = new_surface.lods[i].index_data.size();
				s->lods[i].index_array = RD::get_singleton()->index_array_create(s->lods[i].index_buffer, 0, indices);
				s->lods[i].edge_length = new_surface.lods[i].edge_length;
				s->lods[i].index_count = indices;
			}
		}
	}

	ERR_FAIL_COND_MSG(!new_surface.index_count && !new_surface.vertex_count, "Meshes must contain a vertex array, an index array, or both");

	s->aabb = new_surface.aabb;
	s->bone_aabbs = new_surface.bone_aabbs; //only really useful for returning them.
	s->mesh_to_skeleton_xform = p_surface.mesh_to_skeleton_xform;

	s->uv_scale = new_surface.uv_scale;
	s->cluster_data = new_surface.cluster_data;
	if (mesh_surface_is_geometry_streamed(s)) {
		cluster_geometry_ram_bytes += _mesh_surface_ram_bytes(*s);
		cluster_geometry_fallback_bytes += uint64_t(s->vertex_buffer_size) + s->attribute_buffer_size + s->index_buffer_size;
		if (_is_cluster_geometry_ram_check_wanted()) {
			_mesh_surface_check_ram(*s, new_surface, buffer_vertex_data, buffer_attribute_data, buffer_index_data);
		}
	}

	if (mesh->blend_shape_count > 0) {
		s->blend_shape_buffer = RD::get_singleton()->storage_buffer_create(new_surface.blend_shape_data.size(), new_surface.blend_shape_data);
		s->blend_shape_buffer_size = new_surface.blend_shape_data.size();
	}

	if (use_as_storage) {
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.binding = 0;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			if (s->vertex_buffer.is_valid()) {
				u.append_id(s->vertex_buffer);
			} else {
				u.append_id(default_rd_storage_buffer);
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 1;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			if (s->skin_buffer.is_valid()) {
				u.append_id(s->skin_buffer);
			} else {
				u.append_id(default_rd_storage_buffer);
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 2;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			if (s->blend_shape_buffer.is_valid()) {
				u.append_id(s->blend_shape_buffer);
			} else {
				u.append_id(default_rd_storage_buffer);
			}
			uniforms.push_back(u);
		}

		s->uniform_set = RD::get_singleton()->uniform_set_create(uniforms, skeleton_shader.version_shader[0], SkeletonShader::UNIFORM_SET_SURFACE);
	}

	if (mesh->surface_count == 0) {
		mesh->aabb = new_surface.aabb;
	} else {
		mesh->aabb.merge_with(new_surface.aabb);
	}
	mesh->skeleton_aabb_version = 0;

	s->material = new_surface.material;

	mesh->surfaces = (Mesh::Surface **)memrealloc(mesh->surfaces, sizeof(Mesh::Surface *) * (mesh->surface_count + 1));
	mesh->surfaces[mesh->surface_count] = s;
	mesh->surface_count++;

	for (MeshInstance *mi : mesh->instances) {
		_mesh_instance_add_surface(mi, mesh, mesh->surface_count - 1);
	}

	mesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);

	for (Mesh *E : mesh->shadow_owners) {
		Mesh *shadow_owner = E;
		shadow_owner->shadow_mesh = RID();
		shadow_owner->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
	}

	mesh->material_cache.clear();
}

void MeshStorage::_mesh_surface_clear(Mesh *p_mesh, int p_surface) {
	Mesh::Surface &s = *p_mesh->surfaces[p_surface];
	_mesh_surface_clear_ram(s);

	if (s.vertex_buffer.is_valid()) {
		RD::get_singleton()->free_rid(s.vertex_buffer); // Clears arrays as dependency automatically, including all versions.
	}
	if (s.attribute_buffer.is_valid()) {
		RD::get_singleton()->free_rid(s.attribute_buffer);
	}
	if (s.skin_buffer.is_valid()) {
		RD::get_singleton()->free_rid(s.skin_buffer);
	}
	if (s.versions) {
		memfree(s.versions); // reallocs, so free with memfree.
	}

	if (s.index_buffer.is_valid()) {
		RD::get_singleton()->free_rid(s.index_buffer);
	}

	if (s.lod_count) {
		for (uint32_t j = 0; j < s.lod_count; j++) {
			RD::get_singleton()->free_rid(s.lods[j].index_buffer);
		}
		memdelete_arr(s.lods);
	}

	if (s.blend_shape_buffer.is_valid()) {
		RD::get_singleton()->free_rid(s.blend_shape_buffer);
	}

	memdelete(p_mesh->surfaces[p_surface]);
}

int MeshStorage::mesh_get_blend_shape_count(RID p_mesh) const {
	const Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, -1);
	return mesh->blend_shape_count;
}

void MeshStorage::mesh_set_blend_shape_mode(RID p_mesh, RSE::BlendShapeMode p_mode) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);
	ERR_FAIL_INDEX((int)p_mode, 2);

	mesh->blend_shape_mode = p_mode;
}

RSE::BlendShapeMode MeshStorage::mesh_get_blend_shape_mode(RID p_mesh) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, RSE::BLEND_SHAPE_MODE_NORMALIZED);
	return mesh->blend_shape_mode;
}

void MeshStorage::mesh_surface_update_vertex_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
	ERR_FAIL_COND(p_data.is_empty());
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	ERR_FAIL_COND(mesh->surfaces[p_surface]->vertex_buffer.is_null());

	uint64_t data_size = p_data.size();
	const uint8_t *r = p_data.ptr();

	// A surface that streams its geometry (D93d) gets its own arrays back in its buffers first, so the
	// region lands at the offsets it names; its RAM copy goes below.
	if (mesh_surface_is_geometry_streamed(mesh->surfaces[p_surface])) {
		_mesh_surface_restore_buffers(p_mesh, mesh, p_surface);
	}

	RD::get_singleton()->buffer_update(mesh->surfaces[p_surface]->vertex_buffer, p_offset, data_size, r);

	// A cluster LOD hierarchy (D85) was simplified from the previous positions: its coarse
	// levels, bounds and errors no longer describe the surface, so the rebuilt cluster BLAS traces
	// LOD 0 alone until a re-creation builds a new one (D86). The clusters themselves are
	// index-only and stand. Without a hierarchy the surface no longer streams, so the RAM copy (D93)
	// goes.
	Vector<uint8_t> &table = mesh->surfaces[p_surface]->cluster_data;
	if (table.size() >= int64_t(RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS * sizeof(uint32_t))) {
		uint32_t *words = reinterpret_cast<uint32_t *>(table.ptrw());
		if (words[0] == RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION) {
			words[7] |= RenderingServerTypes::MESH_CLUSTER_LOD_FLAG_STALE;
		}
	}
	_mesh_surface_clear_ram(*mesh->surfaces[p_surface]);

	// Invalidate RT cache for this surface.
	mesh->surfaces[p_surface]->rt_invalidation_counter++;
}

void MeshStorage::mesh_surface_update_attribute_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
	ERR_FAIL_COND(p_data.is_empty());
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	ERR_FAIL_COND(mesh->surfaces[p_surface]->attribute_buffer.is_null());

	uint64_t data_size = p_data.size();
	const uint8_t *r = p_data.ptr();

	// A surface that streams its geometry (D93d): the region goes into the RAM copy, which the blobs
	// gathered at the repopulation below read, and the fallback level's attributes are made again
	// from it. A region past the copy gets the surface's own arrays back in its buffers first, and
	// the copy goes below.
	Mesh::Surface *surface = mesh->surfaces[p_surface];
	if (mesh_surface_is_geometry_streamed(surface)) {
		Vector<uint8_t> &attributes = surface->ram_attribute_data;
		if (p_offset >= 0 && uint64_t(p_offset) + data_size <= uint64_t(attributes.size())) {
			memcpy(attributes.ptrw() + p_offset, r, data_size);
			Vector<uint8_t> fallback;
			_cluster_fallback_attributes(attributes, surface->rt_vertex_count, surface->ram_fallback_vertices, fallback);
			RD::get_singleton()->buffer_update(surface->attribute_buffer, 0, fallback.size(), fallback.ptr());
			surface->rt_invalidation_counter++;
			return;
		}
		_mesh_surface_restore_buffers(p_mesh, mesh, p_surface);
	}

	RD::get_singleton()->buffer_update(mesh->surfaces[p_surface]->attribute_buffer, p_offset, data_size, r);

	// Past the RAM copy: it goes, and the surface no longer streams.
	if (mesh_surface_is_geometry_streamed(mesh->surfaces[p_surface])) {
		_mesh_surface_clear_ram(*mesh->surfaces[p_surface]);
	}

	// Invalidate RT cache for this surface (attributes include UVs).
	mesh->surfaces[p_surface]->rt_invalidation_counter++;
}

void MeshStorage::mesh_surface_update_skin_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
	ERR_FAIL_COND(p_data.is_empty());
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	ERR_FAIL_COND(mesh->surfaces[p_surface]->skin_buffer.is_null());

	uint64_t data_size = p_data.size();
	const uint8_t *r = p_data.ptr();

	RD::get_singleton()->buffer_update(mesh->surfaces[p_surface]->skin_buffer, p_offset, data_size, r);

	// Invalidate RT cache for this surface (skinned meshes need BLAS rebuild).
	mesh->surfaces[p_surface]->rt_invalidation_counter++;
}

void RendererRD::MeshStorage::mesh_surface_update_index_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
	ERR_FAIL_COND(p_data.is_empty());
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	ERR_FAIL_COND(mesh->surfaces[p_surface]->index_buffer.is_null());

	uint64_t data_size = p_data.size();
	const uint8_t *r = p_data.ptr();

	// A surface that streams its geometry (D93d) gets its own arrays back in its buffers first, so the
	// region lands at the offsets it names; its RAM copy goes below.
	if (mesh_surface_is_geometry_streamed(mesh->surfaces[p_surface])) {
		_mesh_surface_restore_buffers(p_mesh, mesh, p_surface);
	}

	RD::get_singleton()->buffer_update(mesh->surfaces[p_surface]->index_buffer, p_offset, data_size, r);

	// Invalidate RT cache for this surface. The cluster table described the previous indices;
	// without it the rebuilt cluster BLAS falls back to runs, which hold for any index contents,
	// and the surface no longer streams (D93).
	mesh->surfaces[p_surface]->cluster_data.clear();
	_mesh_surface_clear_ram(*mesh->surfaces[p_surface]);
	mesh->surfaces[p_surface]->rt_invalidation_counter++;
}

RID MeshStorage::mesh_surface_get_vertex_buffer_rd_rid(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, RID());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RID());
	return mesh->surfaces[p_surface]->vertex_buffer;
}

RID MeshStorage::mesh_surface_get_attribute_buffer_rd_rid(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, RID());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RID());
	return mesh->surfaces[p_surface]->attribute_buffer;
}

RID MeshStorage::mesh_surface_get_skin_buffer_rd_rid(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, RID());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RID());
	return mesh->surfaces[p_surface]->skin_buffer;
}

RID MeshStorage::mesh_surface_get_index_buffer_rd_rid(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, RID());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RID());
	return mesh->surfaces[p_surface]->index_buffer;
}

void MeshStorage::mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	mesh->surfaces[p_surface]->material = p_material;

	mesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MATERIAL);
	mesh->material_cache.clear();
}

RID MeshStorage::mesh_surface_get_material(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, RID());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RID());

	return mesh->surfaces[p_surface]->material;
}

// A surface that streams its geometry gets its own arrays back in its buffers (D93d), from its RAM
// copy, before a region write lands at the offsets the arrays have: the fallback level's buffers go,
// with the vertex arrays cached over them here and in the mesh's instances, and the whole vertex,
// attribute and index buffers (the coarse tail included) and Godot's LODs are made as
// mesh_add_surface() makes them, and an indirect MultiMesh of the mesh draws the surface's count
// again. The caller writes the region, then drops the RAM copy.
void MeshStorage::_mesh_surface_restore_buffers(RID p_mesh_rid, Mesh *p_mesh, int p_surface) {
	Mesh::Surface &s = *p_mesh->surfaces[p_surface];
	RD *rd = RD::get_singleton();
	if (s.fallback_in_buffers) {
		cluster_geometry_fallback_bytes -= uint64_t(s.vertex_buffer_size) + s.attribute_buffer_size + s.index_buffer_size;
		s.fallback_in_buffers = false;
	}
	// The buffers' dependents (vertex and index arrays) go with them.
	if (s.vertex_buffer.is_valid()) {
		rd->free_rid(s.vertex_buffer);
		s.vertex_buffer = RID();
	}
	if (s.attribute_buffer.is_valid()) {
		rd->free_rid(s.attribute_buffer);
		s.attribute_buffer = RID();
	}
	if (s.index_buffer.is_valid()) {
		rd->free_rid(s.index_buffer);
		s.index_buffer = RID();
		s.index_array = RID();
	}
	s.version_lock.lock();
	if (s.versions) {
		memfree(s.versions); // reallocs, so free with memfree.
		s.versions = nullptr;
		s.version_count = 0;
	}
	s.version_lock.unlock();
	for (MeshInstance *mi : p_mesh->instances) {
		if (uint32_t(p_surface) < mi->surfaces.size() && mi->surfaces[p_surface].versions) {
			memfree(mi->surfaces[p_surface].versions);
			mi->surfaces[p_surface].versions = nullptr;
			mi->surfaces[p_surface].version_count = 0;
		}
	}

	// A surface that streams its geometry is neither skinned nor blend-shaped (mesh_add_surface).
	BitField<RD::BufferCreationBits> flags = (s.format & RSE::ARRAY_FLAG_USE_STORAGE_BUFFER) ? RD::BUFFER_CREATION_AS_STORAGE_BIT : 0;
	if (rd->has_feature(RD::SUPPORTS_RAYTRACING_PIPELINE) || rd->has_feature(RD::SUPPORTS_RAY_QUERY)) {
		flags.set_flag(RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT);
		flags.set_flag(RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
		flags.set_flag(RD::BUFFER_CREATION_AS_STORAGE_BIT);
	}
	if (s.ram_vertex_data.size()) {
		Vector<uint8_t> vertex_data = s.ram_vertex_data;
		if (!(s.format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES) && (s.format & RSE::ARRAY_FORMAT_NORMAL) && !(s.format & RSE::ARRAY_FORMAT_TANGENT)) {
			vertex_data.resize_initialized(vertex_data.size() + sizeof(uint16_t) * 2); // The pad mesh_add_surface() gives the normal block.
		}
		s.vertex_buffer = rd->vertex_buffer_create(vertex_data.size(), vertex_data, flags);
		s.vertex_buffer_size = vertex_data.size();
	}
	if (s.ram_attribute_data.size()) {
		s.attribute_buffer = rd->vertex_buffer_create(s.ram_attribute_data.size(), s.ram_attribute_data, flags);
		s.attribute_buffer_size = s.ram_attribute_data.size();
	}
	s.vertex_count = s.rt_vertex_count;
	s.index_count = s.rt_index_count;
	s.index_total = s.rt_index_total;
	const bool is_index_16 = s.vertex_count <= 65536 && s.vertex_count > 0;
	s.index_buffer = rd->index_buffer_create(s.index_total, is_index_16 ? RD::INDEX_BUFFER_FORMAT_UINT16 : RD::INDEX_BUFFER_FORMAT_UINT32, s.ram_index_data, false, flags);
	s.index_buffer_size = s.ram_index_data.size();
	s.index_array = rd->index_array_create(s.index_buffer, 0, s.index_count);
	if (s.ram_lods.size()) {
		s.lods = memnew_arr(Mesh::Surface::LOD, s.ram_lods.size());
		s.lod_count = s.ram_lods.size();
		for (int i = 0; i < s.ram_lods.size(); i++) {
			const uint32_t indices = s.ram_lods[i].index_data.size() / (is_index_16 ? 2 : 4);
			s.lods[i].index_buffer = rd->index_buffer_create(indices, is_index_16 ? RD::INDEX_BUFFER_FORMAT_UINT16 : RD::INDEX_BUFFER_FORMAT_UINT32, s.ram_lods[i].index_data, false, flags);
			s.lods[i].index_buffer_size = s.ram_lods[i].index_data.size();
			s.lods[i].index_array = rd->index_array_create(s.lods[i].index_buffer, 0, indices);
			s.lods[i].edge_length = s.ram_lods[i].edge_length;
			s.lods[i].index_count = indices;
		}
	}
	// An indirect MultiMesh's command buffer holds each surface's draw count, written when its mesh
	// was set (_multimesh_set_mesh); the instance count beside it stays. Rare enough to walk them all.
	const uint32_t count = mesh_surface_get_vertices_drawn_count(&s);
	for (const RID &multimesh_rid : multimesh_owner.get_owned_list()) {
		MultiMesh *multimesh = multimesh_owner.get_or_null(multimesh_rid);
		if (multimesh && multimesh->indirect && multimesh->mesh == p_mesh_rid && multimesh->command_buffer.is_valid() && uint32_t(p_surface) < multimesh->command_buffer_surfaces) {
			rd->buffer_update(multimesh->command_buffer, p_surface * sizeof(uint32_t) * INDIRECT_MULTIMESH_COMMAND_STRIDE, sizeof(uint32_t), &count);
		}
	}
	p_mesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
}

RenderingServerTypes::SurfaceData MeshStorage::mesh_get_surface(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, RenderingServerTypes::SurfaceData());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RenderingServerTypes::SurfaceData());

	const Mesh::Surface &s = *mesh->surfaces[p_surface];
	// A surface that streams its geometry (D93d) answers from its RAM copy, the arrays it was added
	// with, without reading its buffers back.
	if (mesh_surface_is_geometry_streamed(&s)) {
		return _mesh_surface_get_from_ram(s);
	}
	return _mesh_surface_get_from_buffers(s);
}

RenderingServerTypes::SurfaceData MeshStorage::_mesh_surface_get_from_buffers(const Mesh::Surface &p_surface) const {
	const Mesh::Surface &s = p_surface;
	RenderingServerTypes::SurfaceData sd;
	sd.format = s.format;
	if (s.vertex_buffer.is_valid()) {
		sd.vertex_data = RD::get_singleton()->buffer_get_data(s.vertex_buffer);
		// When using an uncompressed buffer with normals, but without tangents, we have to trim the padding.
		if (!(s.format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES) && (s.format & RSE::ARRAY_FORMAT_NORMAL) && !(s.format & RSE::ARRAY_FORMAT_TANGENT)) {
			sd.vertex_data.resize(sd.vertex_data.size() - sizeof(uint16_t) * 2);
		}
	}
	if (s.attribute_buffer.is_valid()) {
		sd.attribute_data = RD::get_singleton()->buffer_get_data(s.attribute_buffer);
	}
	if (s.skin_buffer.is_valid()) {
		sd.skin_data = RD::get_singleton()->buffer_get_data(s.skin_buffer);
	}
	sd.vertex_count = s.vertex_count;
	sd.index_count = s.index_count;
	sd.primitive = s.primitive;

	if (sd.index_count) {
		// The drawn range alone: a cluster LOD tail (D85) is the table's, re-appended at upload.
		const uint32_t index_size = (s.vertex_count <= 65536 && s.vertex_count > 0) ? sizeof(uint16_t) : sizeof(uint32_t);
		sd.index_data = RD::get_singleton()->buffer_get_data(s.index_buffer, 0, s.index_count * index_size);
	}
	sd.aabb = s.aabb;
	sd.uv_scale = s.uv_scale;
	sd.cluster_data = s.cluster_data;
	for (uint32_t i = 0; i < s.lod_count; i++) {
		RenderingServerTypes::SurfaceData::LOD lod;
		lod.edge_length = s.lods[i].edge_length;
		lod.index_data = RD::get_singleton()->buffer_get_data(s.lods[i].index_buffer);
		sd.lods.push_back(lod);
	}

	sd.bone_aabbs = s.bone_aabbs;
	sd.mesh_to_skeleton_xform = s.mesh_to_skeleton_xform;

	if (s.blend_shape_buffer.is_valid()) {
		sd.blend_shape_data = RD::get_singleton()->buffer_get_data(s.blend_shape_buffer);
	}

	return sd;
}

// The surface's arrays as it was added (D93d): its counts are the trace's, its index data the RAM
// copy's drawn range in the index size of the whole surface. A surface that streams its geometry is
// neither skinned nor blend-shaped, so it has no skin or blend shape data.
RenderingServerTypes::SurfaceData MeshStorage::_mesh_surface_get_from_ram(const Mesh::Surface &p_surface) {
	const Mesh::Surface &s = p_surface;
	RenderingServerTypes::SurfaceData sd;
	sd.format = s.format;
	sd.primitive = s.primitive;
	sd.vertex_data = s.ram_vertex_data;
	sd.attribute_data = s.ram_attribute_data;
	sd.vertex_count = s.rt_vertex_count;
	sd.index_count = s.rt_index_count;
	if (sd.index_count) {
		const uint32_t index_size = (s.rt_vertex_count <= 65536 && s.rt_vertex_count > 0) ? sizeof(uint16_t) : sizeof(uint32_t);
		sd.index_data = s.ram_index_data.slice(0, int64_t(sd.index_count) * index_size);
	}
	sd.aabb = s.aabb;
	sd.uv_scale = s.uv_scale;
	sd.cluster_data = s.cluster_data;
	sd.lods = s.ram_lods;
	sd.bone_aabbs = s.bone_aabbs;
	sd.mesh_to_skeleton_xform = s.mesh_to_skeleton_xform;
	return sd;
}

// The RAM check (D93d, rendering/pathtracing/cluster_lod_stream_geometry_ram_check): at a surface's
// creation, mesh_get_surface()'s answer from the RAM copy against the arrays the surface was added
// with, and the buffers read back against the fallback level's arrays they were made from, byte for
// byte. The table, AABB, uv_scale and bone fields come from the same Surface fields on every path,
// so only the counts, the arrays and the LODs are evidence.
void MeshStorage::_mesh_surface_check_ram(const Mesh::Surface &p_surface, const RenderingServerTypes::SurfaceData &p_given, const Vector<uint8_t> &p_vertex_data, const Vector<uint8_t> &p_attribute_data, const Vector<uint8_t> &p_index_data) {
	const RenderingServerTypes::SurfaceData ram = _mesh_surface_get_from_ram(p_surface);
	const RenderingServerTypes::SurfaceData gpu = _mesh_surface_get_from_buffers(p_surface);
	String differs;
	if (ram.format != p_given.format || ram.primitive != p_given.primitive || ram.vertex_count != p_given.vertex_count || ram.index_count != p_given.index_count) {
		differs += " counts";
	}
	if (ram.vertex_data != p_given.vertex_data) {
		differs += " vertex";
	}
	if (ram.attribute_data != p_given.attribute_data) {
		differs += " attribute";
	}
	if (ram.index_data != p_given.index_data) {
		differs += " index";
	}
	bool lods_equal = ram.lods.size() == p_given.lods.size();
	for (int i = 0; i < ram.lods.size() && lods_equal; i++) {
		lods_equal = ram.lods[i].edge_length == p_given.lods[i].edge_length && ram.lods[i].index_data == p_given.lods[i].index_data;
	}
	if (!lods_equal) {
		differs += " lods";
	}
	if (gpu.vertex_data != p_vertex_data || gpu.attribute_data != p_attribute_data || gpu.index_data != p_index_data || gpu.vertex_count != p_surface.vertex_count || !gpu.lods.is_empty()) {
		differs += " fallback";
	}
	ram_check_surfaces++;
	if (!differs.is_empty()) {
		ram_check_differing++;
		print_line(vformat("MeshStorage: a surface's RAM copy or fallback level differs (D93d):%s; %d of %d checked differ.", differs, ram_check_differing, ram_check_surfaces));
	} else {
		print_line(vformat("MeshStorage: a surface's RAM copy equals its arrays and its buffers its fallback level (D93d), %d vertices, %d indices and %d LODs, the fallback %d vertices and %d indices; %d of %d checked differ.", ram.vertex_count, ram.index_count, ram.lods.size(), p_surface.vertex_count, p_surface.index_count, ram_check_differing, ram_check_surfaces));
	}
}

int MeshStorage::mesh_get_surface_count(RID p_mesh) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, 0);
	return mesh->surface_count;
}

void MeshStorage::mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);
	mesh->custom_aabb = p_aabb;

	mesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_AABB);
}

AABB MeshStorage::mesh_get_custom_aabb(RID p_mesh) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, AABB());
	return mesh->custom_aabb;
}

AABB MeshStorage::mesh_get_aabb(RID p_mesh, RID p_skeleton) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, AABB());

	if (mesh->custom_aabb != AABB()) {
		return mesh->custom_aabb;
	}

	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	// A mesh can be shared by multiple skeletons and we need to avoid using the AABB from a different skeleton.
	if (!skeleton || skeleton->size == 0 || (mesh->skeleton_aabb_version == skeleton->version && mesh->skeleton_aabb_rid == p_skeleton)) {
		return mesh->aabb;
	}

	AABB aabb;

	for (uint32_t i = 0; i < mesh->surface_count; i++) {
		AABB laabb;
		const Mesh::Surface &surface = *mesh->surfaces[i];
		if ((surface.format & RSE::ARRAY_FORMAT_BONES) && surface.bone_aabbs.size()) {
			int bs = surface.bone_aabbs.size();
			const AABB *skbones = surface.bone_aabbs.ptr();

			int sbs = skeleton->size;
			ERR_CONTINUE(bs > sbs);
			const float *baseptr = skeleton->data.ptr();

			bool found_bone_aabb = false;

			if (skeleton->use_2d) {
				for (int j = 0; j < bs; j++) {
					if (skbones[j].size == Vector3(-1, -1, -1)) {
						continue; //bone is unused
					}

					const float *dataptr = baseptr + j * 8;

					Transform3D mtx;

					mtx.basis.rows[0][0] = dataptr[0];
					mtx.basis.rows[0][1] = dataptr[1];
					mtx.origin.x = dataptr[3];

					mtx.basis.rows[1][0] = dataptr[4];
					mtx.basis.rows[1][1] = dataptr[5];
					mtx.origin.y = dataptr[7];

					// Transform bounds to skeleton's space before applying animation data.
					AABB baabb = surface.mesh_to_skeleton_xform.xform(skbones[j]);
					baabb = mtx.xform(baabb);

					if (!found_bone_aabb) {
						laabb = baabb;
						found_bone_aabb = true;
					} else {
						laabb.merge_with(baabb);
					}
				}
			} else {
				for (int j = 0; j < bs; j++) {
					if (skbones[j].size == Vector3(-1, -1, -1)) {
						continue; //bone is unused
					}

					const float *dataptr = baseptr + j * 12;

					Transform3D mtx;

					mtx.basis.rows[0][0] = dataptr[0];
					mtx.basis.rows[0][1] = dataptr[1];
					mtx.basis.rows[0][2] = dataptr[2];
					mtx.origin.x = dataptr[3];
					mtx.basis.rows[1][0] = dataptr[4];
					mtx.basis.rows[1][1] = dataptr[5];
					mtx.basis.rows[1][2] = dataptr[6];
					mtx.origin.y = dataptr[7];
					mtx.basis.rows[2][0] = dataptr[8];
					mtx.basis.rows[2][1] = dataptr[9];
					mtx.basis.rows[2][2] = dataptr[10];
					mtx.origin.z = dataptr[11];

					// Transform bounds to skeleton's space before applying animation data.
					AABB baabb = surface.mesh_to_skeleton_xform.xform(skbones[j]);
					baabb = mtx.xform(baabb);

					if (!found_bone_aabb) {
						laabb = baabb;
						found_bone_aabb = true;
					} else {
						laabb.merge_with(baabb);
					}
				}
			}

			if (found_bone_aabb) {
				// Transform skeleton bounds back to mesh's space if any animated AABB applied.
				laabb = surface.mesh_to_skeleton_xform.affine_inverse().xform(laabb);
			}

			if (laabb.size == Vector3()) {
				laabb = surface.aabb;
			}
		} else {
			laabb = surface.aabb;
		}

		if (i == 0) {
			aabb = laabb;
		} else {
			aabb.merge_with(laabb);
		}
	}

	mesh->aabb = aabb;

	mesh->skeleton_aabb_version = skeleton->version;
	mesh->skeleton_aabb_rid = p_skeleton;
	return aabb;
}

void MeshStorage::mesh_set_path(RID p_mesh, const String &p_path) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);

	mesh->path = p_path;
}

String MeshStorage::mesh_get_path(RID p_mesh) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, String());

	return mesh->path;
}

void MeshStorage::mesh_set_shadow_mesh(RID p_mesh, RID p_shadow_mesh) {
	ERR_FAIL_COND_MSG(p_mesh == p_shadow_mesh, "Cannot set a mesh as its own shadow mesh.");
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);

	Mesh *shadow_mesh = mesh_owner.get_or_null(mesh->shadow_mesh);
	if (shadow_mesh) {
		shadow_mesh->shadow_owners.erase(mesh);
	}
	mesh->shadow_mesh = p_shadow_mesh;

	shadow_mesh = mesh_owner.get_or_null(mesh->shadow_mesh);

	if (shadow_mesh) {
		shadow_mesh->shadow_owners.insert(mesh);
	}

	mesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
}

void MeshStorage::mesh_clear(RID p_mesh) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);

	// Clear instance data before mesh data.
	for (MeshInstance *mi : mesh->instances) {
		_mesh_instance_clear(mi);
	}

	for (uint32_t i = 0; i < mesh->surface_count; i++) {
		if (surface_freed_callback) {
			surface_freed_callback(p_mesh, i);
		}
		_mesh_surface_clear(mesh, i);
	}
	if (mesh->surfaces) {
		memfree(mesh->surfaces);
	}

	mesh->surfaces = nullptr;
	mesh->surface_count = 0;
	mesh->material_cache.clear();
	mesh->has_bone_weights = false;
	mesh->aabb = AABB();
	mesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);

	for (Mesh *E : mesh->shadow_owners) {
		Mesh *shadow_owner = E;
		shadow_owner->shadow_mesh = RID();
		shadow_owner->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
	}
}

void MeshStorage::mesh_surface_remove(RID p_mesh, int p_surface) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL(mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);

	// Clear instance data before mesh data.
	for (MeshInstance *mi : mesh->instances) {
		_mesh_instance_remove_surface(mi, p_surface);
	}

	// The surfaces above shift down a slot, so their cache keys change too; the old last key is
	// released here since re-adding a surface reuses it.
	if (surface_freed_callback) {
		for (uint32_t i = p_surface; i < mesh->surface_count; i++) {
			surface_freed_callback(p_mesh, i);
		}
	}
	_mesh_surface_clear(mesh, p_surface);

	if ((uint32_t)p_surface < mesh->surface_count - 1) {
		memmove(mesh->surfaces + p_surface, mesh->surfaces + p_surface + 1, sizeof(Mesh::Surface *) * (mesh->surface_count - (p_surface + 1)));
	}
	mesh->surfaces = (Mesh::Surface **)memrealloc(mesh->surfaces, sizeof(Mesh::Surface *) * (mesh->surface_count - 1));
	--mesh->surface_count;

	mesh->material_cache.clear();

	mesh->skeleton_aabb_version = 0;

	if (mesh->has_bone_weights) {
		mesh->has_bone_weights = false;
		for (uint32_t i = 0; i < mesh->surface_count; i++) {
			if (mesh->surfaces[i]->format & RSE::ARRAY_FORMAT_BONES) {
				mesh->has_bone_weights = true;
				break;
			}
		}
	}

	if (mesh->surface_count == 0) {
		mesh->aabb = AABB();
	} else {
		mesh->aabb = mesh->surfaces[0]->aabb;
		for (uint32_t i = 1; i < mesh->surface_count; i++) {
			mesh->aabb.merge_with(mesh->surfaces[i]->aabb);
		}
	}

	mesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);

	for (Mesh *E : mesh->shadow_owners) {
		Mesh *shadow_owner = E;
		shadow_owner->shadow_mesh = RID();
		shadow_owner->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
	}
}

void MeshStorage::mesh_debug_usage(List<RenderingServerTypes::MeshInfo> *r_info) {
	for (const RID &mesh_rid : mesh_owner.get_owned_list()) {
		Mesh *mesh = mesh_owner.get_or_null(mesh_rid);
		if (!mesh) {
			continue;
		}
		RenderingServerTypes::MeshInfo mesh_info;
		mesh_info.mesh = mesh_rid;
		mesh_info.path = mesh->path;

		for (uint32_t surface_index = 0; surface_index < mesh->surface_count; surface_index++) {
			MeshStorage::Mesh::Surface *surface = mesh->surfaces[surface_index];

			mesh_info.vertex_buffer_size += surface->vertex_buffer_size;
			mesh_info.attribute_buffer_size += surface->attribute_buffer_size;
			mesh_info.skin_buffer_size += surface->skin_buffer_size;
			mesh_info.index_buffer_size += surface->index_buffer_size;
			mesh_info.blend_shape_buffer_size += surface->blend_shape_buffer_size;
			mesh_info.vertex_count += surface->vertex_count;

			for (uint32_t lod_index = 0; lod_index < surface->lod_count; lod_index++) {
				mesh_info.lod_index_buffers_size += surface->lods[lod_index].index_buffer_size;
			}
		}

		r_info->push_back(mesh_info);
	}
}

bool MeshStorage::mesh_needs_instance(RID p_mesh, bool p_has_skeleton) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, false);

	return mesh->blend_shape_count > 0 || (mesh->has_bone_weights && p_has_skeleton);
}

Dependency *MeshStorage::mesh_get_dependency(RID p_mesh) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_NULL_V(mesh, nullptr);

	return &mesh->dependency;
}

/* MESH INSTANCE */

RID MeshStorage::mesh_instance_create(RID p_base) {
	Mesh *mesh = mesh_owner.get_or_null(p_base);
	ERR_FAIL_NULL_V(mesh, RID());

	RID rid = mesh_instance_owner.make_rid();
	MeshInstance *mi = mesh_instance_owner.get_or_null(rid);

	mi->mesh = mesh;

	for (uint32_t i = 0; i < mesh->surface_count; i++) {
		_mesh_instance_add_surface(mi, mesh, i);
	}

	mi->I = mesh->instances.push_back(mi);

	mi->dirty = true;

	return rid;
}

void MeshStorage::mesh_instance_free(RID p_rid) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_rid);
	_mesh_instance_clear(mi);
	mi->mesh->instances.erase(mi->I);
	mi->I = nullptr;

	mesh_instance_owner.free(p_rid);
}

void MeshStorage::mesh_instance_set_skeleton(RID p_mesh_instance, RID p_skeleton) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_mesh_instance);
	if (mi->skeleton == p_skeleton) {
		return;
	}
	mi->skeleton = p_skeleton;
	mi->skeleton_version = 0;
	mi->dirty = true;
}

void MeshStorage::mesh_instance_set_blend_shape_weight(RID p_mesh_instance, int p_shape, float p_weight) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_mesh_instance);
	ERR_FAIL_NULL(mi);
	ERR_FAIL_INDEX(p_shape, (int)mi->blend_weights.size());
	mi->blend_weights[p_shape] = p_weight;
	mi->weights_dirty = true;
	//will be eventually updated
}

void MeshStorage::_mesh_instance_clear(MeshInstance *mi) {
	while (mi->surfaces.size()) {
		_mesh_instance_remove_surface(mi, mi->surfaces.size() - 1);
	}
	mi->dirty = false;
}

void MeshStorage::_mesh_instance_add_surface(MeshInstance *mi, Mesh *mesh, uint32_t p_surface) {
	if (mesh->blend_shape_count > 0 && mi->blend_weights_buffer.is_null()) {
		mi->blend_weights.resize(mesh->blend_shape_count);
		for (float &weight : mi->blend_weights) {
			weight = 0;
		}
		mi->blend_weights_buffer = RD::get_singleton()->storage_buffer_create(sizeof(float) * mi->blend_weights.size(), mi->blend_weights.span().reinterpret<uint8_t>());
		mi->weights_dirty = true;
	}

	MeshInstance::Surface s;
	if ((mesh->blend_shape_count > 0 || (mesh->surfaces[p_surface]->format & RSE::ARRAY_FORMAT_BONES)) && mesh->surfaces[p_surface]->vertex_buffer_size > 0) {
		_mesh_instance_add_surface_buffer(mi, mesh, &s, p_surface, 0);
	}

	mi->surfaces.push_back(s);
	mi->dirty = true;
}

void MeshStorage::_mesh_instance_add_surface_buffer(MeshInstance *mi, Mesh *mesh, MeshInstance::Surface *s, uint32_t p_surface, uint32_t p_buffer_index) {
	BitField<RD::BufferCreationBits> buffer_flags = RD::BUFFER_CREATION_AS_STORAGE_BIT;
	// Allow this skinned/blend-shape output buffer to be consumed by the raytracing path as a BLAS triangle source.
	if (RD::get_singleton()->has_feature(RD::SUPPORTS_RAYTRACING_PIPELINE) || RD::get_singleton()->has_feature(RD::SUPPORTS_RAY_QUERY)) {
		buffer_flags.set_flag(RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT);
		buffer_flags.set_flag(RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
	}
	s->vertex_buffer[p_buffer_index] = RD::get_singleton()->vertex_buffer_create(mesh->surfaces[p_surface]->vertex_buffer_size, Vector<uint8_t>(), buffer_flags);

	Vector<RD::Uniform> uniforms;
	{
		RD::Uniform u;
		u.binding = 1;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.append_id(s->vertex_buffer[p_buffer_index]);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.binding = 2;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		if (mi->blend_weights_buffer.is_valid()) {
			u.append_id(mi->blend_weights_buffer);
		} else {
			u.append_id(default_rd_storage_buffer);
		}
		uniforms.push_back(u);
	}
	s->uniform_set[p_buffer_index] = RD::get_singleton()->uniform_set_create(uniforms, skeleton_shader.version_shader[0], SkeletonShader::UNIFORM_SET_INSTANCE);
}

void MeshStorage::_mesh_instance_remove_surface(MeshInstance *mi, int p_surface) {
	MeshInstance::Surface &surface = mi->surfaces[p_surface];

	if (surface.versions) {
		for (uint32_t j = 0; j < surface.version_count; j++) {
			RD::get_singleton()->free_rid(surface.versions[j].vertex_array);
		}
		memfree(surface.versions);
	}
	for (uint32_t i = 0; i < 2; i++) {
		if (surface.vertex_buffer[i].is_valid()) {
			RD::get_singleton()->free_rid(surface.vertex_buffer[i]);
		}
	}

	mi->surfaces.remove_at(p_surface);

	if (mi->surfaces.is_empty()) {
		if (mi->blend_weights_buffer.is_valid()) {
			RD::get_singleton()->free_rid(mi->blend_weights_buffer);
			mi->blend_weights_buffer = RID();
		}

		mi->blend_weights.clear();
		mi->weights_dirty = false;
		mi->skeleton_version = 0;
	}
	mi->dirty = true;
}

void MeshStorage::mesh_instance_check_for_update(RID p_mesh_instance) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_mesh_instance);

	bool needs_update = mi->dirty;

	if (mi->weights_dirty && !mi->weight_update_list.in_list()) {
		dirty_mesh_instance_weights.add(&mi->weight_update_list);
		needs_update = true;
	}

	if (mi->array_update_list.in_list()) {
		return;
	}

	if (!needs_update && mi->skeleton.is_valid()) {
		Skeleton *sk = skeleton_owner.get_or_null(mi->skeleton);
		if (sk && sk->version != mi->skeleton_version) {
			needs_update = true;
		}
	}

	if (needs_update) {
		dirty_mesh_instance_arrays.add(&mi->array_update_list);
	}
}

void MeshStorage::mesh_instance_set_canvas_item_transform(RID p_mesh_instance, const Transform2D &p_transform) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_mesh_instance);
	mi->canvas_item_transform_2d = p_transform;
}

void MeshStorage::update_mesh_instances() {
	while (dirty_mesh_instance_weights.first()) {
		MeshInstance *mi = dirty_mesh_instance_weights.first()->self();

		if (mi->blend_weights_buffer.is_valid()) {
			RD::get_singleton()->buffer_update(mi->blend_weights_buffer, 0, mi->blend_weights.size() * sizeof(float), mi->blend_weights.ptr());
		}
		dirty_mesh_instance_weights.remove(&mi->weight_update_list);
		mi->weights_dirty = false;
	}
	if (dirty_mesh_instance_arrays.first() == nullptr) {
		return; //nothing to do
	}

	//process skeletons and blend shapes
	uint64_t frame = RSG::rasterizer->get_frame_number();
	bool uses_motion_vectors = (RSG::viewport->get_num_viewports_with_motion_vectors() > 0) || (RendererCompositorStorage::get_singleton()->get_num_compositor_effects_with_motion_vectors() > 0);
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	while (dirty_mesh_instance_arrays.first()) {
		MeshInstance *mi = dirty_mesh_instance_arrays.first()->self();

		Skeleton *sk = skeleton_owner.get_or_null(mi->skeleton);

		for (uint32_t i = 0; i < mi->surfaces.size(); i++) {
			if (mi->surfaces[i].uniform_set[0].is_null() || mi->mesh->surfaces[i]->uniform_set.is_null()) {
				// Skip over mesh instances that don't require their own uniform buffers.
				continue;
			}

			mi->surfaces[i].previous_buffer = mi->surfaces[i].current_buffer;

			if (uses_motion_vectors && mi->surfaces[i].last_change && (frame - mi->surfaces[i].last_change) <= 2) {
				// Use a 2-frame tolerance so that stepped skeletal animations have correct motion vectors
				// (stepped animation is common for distant NPCs).
				uint32_t new_buffer_index = mi->surfaces[i].current_buffer ^ 1;

				if (mi->surfaces[i].uniform_set[new_buffer_index].is_null()) {
					// Create the new vertex buffer on demand where the result for the current frame will be stored.
					_mesh_instance_add_surface_buffer(mi, mi->mesh, &mi->surfaces[i], i, new_buffer_index);
				}

				mi->surfaces[i].current_buffer = new_buffer_index;
			}

			mi->surfaces[i].last_change = frame;

			RID mi_surface_uniform_set = mi->surfaces[i].uniform_set[mi->surfaces[i].current_buffer];
			if (mi_surface_uniform_set.is_null()) {
				continue;
			}

			bool array_is_2d = mi->mesh->surfaces[i]->format & RSE::ARRAY_FLAG_USE_2D_VERTICES;

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, skeleton_shader.pipeline[array_is_2d ? SkeletonShader::SHADER_MODE_2D : SkeletonShader::SHADER_MODE_3D]);

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, mi_surface_uniform_set, SkeletonShader::UNIFORM_SET_INSTANCE);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, mi->mesh->surfaces[i]->uniform_set, SkeletonShader::UNIFORM_SET_SURFACE);
			if (sk && sk->uniform_set_mi.is_valid()) {
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, sk->uniform_set_mi, SkeletonShader::UNIFORM_SET_SKELETON);
			} else {
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, skeleton_shader.default_skeleton_uniform_set, SkeletonShader::UNIFORM_SET_SKELETON);
			}

			SkeletonShader::PushConstant push_constant;

			push_constant.has_normal = mi->mesh->surfaces[i]->format & RSE::ARRAY_FORMAT_NORMAL;
			push_constant.has_tangent = mi->mesh->surfaces[i]->format & RSE::ARRAY_FORMAT_TANGENT;
			push_constant.has_skeleton = sk != nullptr && sk->use_2d == array_is_2d && (mi->mesh->surfaces[i]->format & RSE::ARRAY_FORMAT_BONES);
			push_constant.has_blend_shape = mi->mesh->blend_shape_count > 0;

			push_constant.normal_tangent_stride = (push_constant.has_normal ? 1 : 0) + (push_constant.has_tangent ? 1 : 0);

			push_constant.vertex_count = mi->mesh->surfaces[i]->vertex_count;
			push_constant.vertex_stride = ((mi->mesh->surfaces[i]->vertex_buffer_size / mi->mesh->surfaces[i]->vertex_count) / 4) - push_constant.normal_tangent_stride;
			push_constant.skin_stride = (mi->mesh->surfaces[i]->skin_buffer_size / mi->mesh->surfaces[i]->vertex_count) / 4;
			push_constant.skin_weight_offset = (mi->mesh->surfaces[i]->format & RSE::ARRAY_FLAG_USE_8_BONE_WEIGHTS) ? 4 : 2;

			Transform2D transform = Transform2D();
			if (sk && sk->use_2d) {
				transform = mi->canvas_item_transform_2d.affine_inverse() * sk->base_transform_2d;
			}
			push_constant.skeleton_transform_x[0] = transform.columns[0][0];
			push_constant.skeleton_transform_x[1] = transform.columns[0][1];
			push_constant.skeleton_transform_y[0] = transform.columns[1][0];
			push_constant.skeleton_transform_y[1] = transform.columns[1][1];
			push_constant.skeleton_transform_offset[0] = transform.columns[2][0];
			push_constant.skeleton_transform_offset[1] = transform.columns[2][1];

			Transform2D inverse_transform = transform.affine_inverse();
			push_constant.inverse_transform_x[0] = inverse_transform.columns[0][0];
			push_constant.inverse_transform_x[1] = inverse_transform.columns[0][1];
			push_constant.inverse_transform_y[0] = inverse_transform.columns[1][0];
			push_constant.inverse_transform_y[1] = inverse_transform.columns[1][1];
			push_constant.inverse_transform_offset[0] = inverse_transform.columns[2][0];
			push_constant.inverse_transform_offset[1] = inverse_transform.columns[2][1];

			push_constant.blend_shape_count = mi->mesh->blend_shape_count;
			push_constant.normalized_blend_shapes = mi->mesh->blend_shape_mode == RSE::BLEND_SHAPE_MODE_NORMALIZED;
			push_constant.pad1 = 0;

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SkeletonShader::PushConstant));

			//dispatch without barrier, so all is done at the same time
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.vertex_count, 1, 1);
		}

		mi->dirty = false;
		if (sk) {
			mi->skeleton_version = sk->version;
		}
		dirty_mesh_instance_arrays.remove(&mi->array_update_list);
	}

	RD::get_singleton()->compute_list_end();
}

RD::VertexFormatID MeshStorage::_mesh_surface_generate_vertex_format(uint64_t p_surface_format, uint64_t p_input_mask, bool p_instanced_surface, bool p_input_motion_vectors, bool p_point_size_emulated, uint32_t &r_position_stride) {
	Vector<RD::VertexAttribute> attributes;
	uint32_t normal_tangent_stride = 0;
	uint32_t attribute_stride = 0;
	uint32_t skin_stride = 0;

	r_position_stride = 0;

	for (int i = 0; i < RSE::ARRAY_INDEX; i++) {
		RD::VertexAttribute vd;
		vd.location = i;

		if (!(p_surface_format & (1ULL << i))) {
			vd.stride = 0;
			switch (i) {
				case RSE::ARRAY_VERTEX:
				case RSE::ARRAY_NORMAL:
					vd.format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
					break;
				case RSE::ARRAY_TEX_UV:
				case RSE::ARRAY_TEX_UV2:
					vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
					break;
				case RSE::ARRAY_BONES:
					vd.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
					break;
				case RSE::ARRAY_TANGENT:
				case RSE::ARRAY_COLOR:
				case RSE::ARRAY_CUSTOM0:
				case RSE::ARRAY_CUSTOM1:
				case RSE::ARRAY_CUSTOM2:
				case RSE::ARRAY_CUSTOM3:
				case RSE::ARRAY_WEIGHTS:
					vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
					break;
				default:
					DEV_ASSERT(false && "Unknown vertex format element.");
					break;
			}
		} else {
			// Mark that it needs a stride set (default uses 0).
			vd.stride = 1;

			switch (i) {
				case RSE::ARRAY_VERTEX: {
					vd.offset = r_position_stride;

					if (p_surface_format & RSE::ARRAY_FLAG_USE_2D_VERTICES) {
						vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
						r_position_stride = sizeof(float) * 2;
					} else {
						if (!p_instanced_surface && (p_surface_format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES)) {
							vd.format = RD::DATA_FORMAT_R16G16B16A16_UNORM;
							r_position_stride = sizeof(uint16_t) * 4;
						} else {
							vd.format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
							r_position_stride = sizeof(float) * 3;
						}
					}

				} break;
				case RSE::ARRAY_NORMAL: {
					vd.offset = 0;

					if (!p_instanced_surface && (p_surface_format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES)) {
						vd.format = RD::DATA_FORMAT_R16G16_UNORM;
						normal_tangent_stride += sizeof(uint16_t) * 2;
					} else {
						vd.format = RD::DATA_FORMAT_R16G16B16A16_UNORM;
						// A small trick here: if we are uncompressed and we have normals, but no tangents. We need
						// the shader to think there are 4 components to "axis_tangent_attrib". So we give a size of 4,
						// but a stride based on only having 2 elements.
						if (!(p_surface_format & RSE::ARRAY_FORMAT_TANGENT)) {
							normal_tangent_stride += sizeof(uint16_t) * 2;
						} else {
							normal_tangent_stride += sizeof(uint16_t) * 4;
						}
					}
				} break;
				case RSE::ARRAY_TANGENT: {
					vd.stride = 0;
					vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
				} break;
				case RSE::ARRAY_COLOR: {
					vd.offset = attribute_stride;

					vd.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
					attribute_stride += sizeof(int8_t) * 4;
				} break;
				case RSE::ARRAY_TEX_UV: {
					vd.offset = attribute_stride;
					if (p_surface_format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES) {
						vd.format = RD::DATA_FORMAT_R16G16_UNORM;
						attribute_stride += sizeof(uint16_t) * 2;
					} else {
						vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
						attribute_stride += sizeof(float) * 2;
					}

				} break;
				case RSE::ARRAY_TEX_UV2: {
					vd.offset = attribute_stride;
					if (p_surface_format & RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES) {
						vd.format = RD::DATA_FORMAT_R16G16_UNORM;
						attribute_stride += sizeof(uint16_t) * 2;
					} else {
						vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
						attribute_stride += sizeof(float) * 2;
					}
				} break;
				case RSE::ARRAY_CUSTOM0:
				case RSE::ARRAY_CUSTOM1:
				case RSE::ARRAY_CUSTOM2:
				case RSE::ARRAY_CUSTOM3: {
					vd.offset = attribute_stride;

					int idx = i - RSE::ARRAY_CUSTOM0;
					const uint32_t fmt_shift[RSE::ARRAY_CUSTOM_COUNT] = { RSE::ARRAY_FORMAT_CUSTOM0_SHIFT, RSE::ARRAY_FORMAT_CUSTOM1_SHIFT, RSE::ARRAY_FORMAT_CUSTOM2_SHIFT, RSE::ARRAY_FORMAT_CUSTOM3_SHIFT };
					uint32_t fmt = (p_surface_format >> fmt_shift[idx]) & RSE::ARRAY_FORMAT_CUSTOM_MASK;
					const uint32_t fmtsize[RSE::ARRAY_CUSTOM_MAX] = { 4, 4, 4, 8, 4, 8, 12, 16 };
					const RD::DataFormat fmtrd[RSE::ARRAY_CUSTOM_MAX] = { RD::DATA_FORMAT_R8G8B8A8_UNORM, RD::DATA_FORMAT_R8G8B8A8_SNORM, RD::DATA_FORMAT_R16G16_SFLOAT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, RD::DATA_FORMAT_R32_SFLOAT, RD::DATA_FORMAT_R32G32_SFLOAT, RD::DATA_FORMAT_R32G32B32_SFLOAT, RD::DATA_FORMAT_R32G32B32A32_SFLOAT };
					vd.format = fmtrd[fmt];
					attribute_stride += fmtsize[fmt];
				} break;
				case RSE::ARRAY_BONES: {
					vd.offset = skin_stride;

					vd.format = RD::DATA_FORMAT_R16G16B16A16_UINT;
					skin_stride += sizeof(int16_t) * 4;
				} break;
				case RSE::ARRAY_WEIGHTS: {
					vd.offset = skin_stride;

					vd.format = RD::DATA_FORMAT_R16G16B16A16_UNORM;
					skin_stride += sizeof(int16_t) * 4;
				} break;
			}

			if (p_point_size_emulated) {
				vd.frequency = RD::VERTEX_FREQUENCY_INSTANCE;
			}
		}

		if (!(p_input_mask & (1ULL << i))) {
			continue; // Shader does not need this, skip it (but computing stride was important anyway)
		}

		attributes.push_back(vd);

		if (p_input_motion_vectors) {
			// Since the previous vertex, normal and tangent can't be part of the vertex format but they are required when
			// motion vectors are enabled, we opt to push a copy of the vertex attribute with a different location.
			switch (i) {
				case RSE::ARRAY_VERTEX: {
					vd.location = ATTRIBUTE_LOCATION_PREV_VERTEX;
				} break;
				case RSE::ARRAY_NORMAL: {
					vd.location = ATTRIBUTE_LOCATION_PREV_NORMAL;
				} break;
				case RSE::ARRAY_TANGENT: {
					vd.location = ATTRIBUTE_LOCATION_PREV_TANGENT;
				} break;
			}

			if (int(vd.location) != i) {
				attributes.push_back(vd);
			}
		}
	}

	// Update final stride.
	for (int i = 0; i < attributes.size(); i++) {
		if (attributes[i].stride == 0) {
			// Default location.
			continue;
		}

		int loc = attributes[i].location;
		if (loc == RSE::ARRAY_VERTEX || loc == ATTRIBUTE_LOCATION_PREV_VERTEX) {
			attributes.write[i].stride = r_position_stride;
		} else if ((loc < RSE::ARRAY_COLOR) || ((loc >= ATTRIBUTE_LOCATION_PREV_NORMAL) && (loc <= ATTRIBUTE_LOCATION_PREV_TANGENT))) {
			attributes.write[i].stride = normal_tangent_stride;
		} else if (loc < RSE::ARRAY_BONES) {
			attributes.write[i].stride = attribute_stride;
		} else {
			attributes.write[i].stride = skin_stride;
		}
	}

	return RD::get_singleton()->vertex_format_create(attributes);
}

void MeshStorage::_mesh_surface_generate_version_for_input_mask(Mesh::Surface::Version &v, Mesh::Surface *s, uint64_t p_input_mask, bool p_input_motion_vectors, bool p_point_size_emulated, MeshInstance::Surface *mis, uint32_t p_current_buffer, uint32_t p_previous_buffer) {
	uint32_t position_stride = 0;
	v.vertex_format = _mesh_surface_generate_vertex_format(s->format, p_input_mask, mis != nullptr, p_input_motion_vectors, p_point_size_emulated, position_stride);

	Vector<RID> buffers;
	Vector<uint64_t> offsets;
	RID buffer;
	uint64_t offset = 0;
	for (int i = 0; i < RSE::ARRAY_INDEX; i++) {
		offset = 0;

		if (!(s->format & (1ULL << i))) {
			// Not supplied by surface, use default buffers.
			buffer = mesh_default_rd_buffers[i];
		} else {
			// Supplied by surface, use buffer.
			switch (i) {
				case RSE::ARRAY_VERTEX:
				case RSE::ARRAY_NORMAL:
					offset = i == RSE::ARRAY_NORMAL ? position_stride * s->vertex_count : 0;
					buffer = mis != nullptr ? mis->vertex_buffer[p_current_buffer] : s->vertex_buffer;
					break;
				case RSE::ARRAY_TANGENT:
					buffer = mesh_default_rd_buffers[i];
					break;
				case RSE::ARRAY_COLOR:
				case RSE::ARRAY_TEX_UV:
				case RSE::ARRAY_TEX_UV2:
				case RSE::ARRAY_CUSTOM0:
				case RSE::ARRAY_CUSTOM1:
				case RSE::ARRAY_CUSTOM2:
				case RSE::ARRAY_CUSTOM3:
					buffer = s->attribute_buffer;
					break;
				case RSE::ARRAY_BONES:
				case RSE::ARRAY_WEIGHTS:
					buffer = s->skin_buffer;
					break;
			}
		}

		if (!(p_input_mask & (1ULL << i))) {
			continue; // Shader does not need this, skip it (but computing stride was important anyway)
		}

		buffers.push_back(buffer);
		offsets.push_back(offset);

		if (p_input_motion_vectors) {
			// Push the buffer for motion vector inputs.
			if (i == RSE::ARRAY_VERTEX || i == RSE::ARRAY_NORMAL || i == RSE::ARRAY_TANGENT) {
				if (mis && buffer != mesh_default_rd_buffers[i]) {
					buffers.push_back(mis->vertex_buffer[p_previous_buffer]);
				} else {
					buffers.push_back(buffer);
				}

				offsets.push_back(offset);
			}
		}
	}

	v.input_mask = p_input_mask;
	v.current_buffer = p_current_buffer;
	v.previous_buffer = p_previous_buffer;
	v.input_motion_vectors = p_input_motion_vectors;
	v.point_size_emulated = p_point_size_emulated;
	v.vertex_array = RD::get_singleton()->vertex_array_create(p_point_size_emulated ? 0 : s->vertex_count, v.vertex_format, buffers, offsets);
}

////////////////// MULTIMESH

RID MeshStorage::_multimesh_allocate() {
	return multimesh_owner.allocate_rid();
}
void MeshStorage::_multimesh_initialize(RID p_rid) {
	multimesh_owner.initialize_rid(p_rid, MultiMesh());
}

void MeshStorage::_multimesh_free(RID p_rid) {
	// Remove from interpolator.
	_interpolation_data.notify_free_multimesh(p_rid);
	_update_dirty_multimeshes();
	multimesh_allocate_data(p_rid, 0, RSE::MULTIMESH_TRANSFORM_2D);
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_rid);
	multimesh->dependency.deleted_notify(p_rid);
	multimesh_owner.free(p_rid);
}

void MeshStorage::_multimesh_allocate_data(RID p_multimesh, int p_instances, RSE::MultimeshTransformFormat p_transform_format, bool p_use_colors, bool p_use_custom_data, bool p_use_indirect) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL(multimesh);

	if (multimesh->instances == p_instances && multimesh->xform_format == p_transform_format && multimesh->uses_colors == p_use_colors && multimesh->uses_custom_data == p_use_custom_data) {
		return;
	}

	if (multimesh->buffer.is_valid()) {
		RD::get_singleton()->free_rid(multimesh->buffer);
		multimesh->buffer = RID();
		multimesh->uniform_set_2d = RID(); //cleared by dependency
		multimesh->uniform_set_3d = RID(); //cleared by dependency
	}

	if (multimesh->data_cache_dirty_regions) {
		memdelete_arr(multimesh->data_cache_dirty_regions);
		multimesh->data_cache_dirty_regions = nullptr;
		multimesh->data_cache_dirty_region_count = 0;
	}

	if (multimesh->previous_data_cache_dirty_regions) {
		memdelete_arr(multimesh->previous_data_cache_dirty_regions);
		multimesh->previous_data_cache_dirty_regions = nullptr;
		multimesh->previous_data_cache_dirty_region_count = 0;
	}

	multimesh->instances = p_instances;
	multimesh->xform_format = p_transform_format;
	multimesh->uses_colors = p_use_colors;
	multimesh->color_offset_cache = p_transform_format == RSE::MULTIMESH_TRANSFORM_2D ? 8 : 12;
	multimesh->uses_custom_data = p_use_custom_data;
	multimesh->custom_data_offset_cache = multimesh->color_offset_cache + (p_use_colors ? 4 : 0);
	multimesh->stride_cache = multimesh->custom_data_offset_cache + (p_use_custom_data ? 4 : 0);
	multimesh->buffer_set = false;

	multimesh->indirect = p_use_indirect;
	multimesh->command_buffer = RID();

	//print_line("allocate, elements: " + itos(p_instances) + " 2D: " + itos(p_transform_format == RSE::MULTIMESH_TRANSFORM_2D) + " colors " + itos(multimesh->uses_colors) + " data " + itos(multimesh->uses_custom_data) + " stride " + itos(multimesh->stride_cache) + " total size " + itos(multimesh->stride_cache * multimesh->instances));
	multimesh->data_cache = Vector<float>();
	multimesh->aabb = AABB();
	multimesh->aabb_dirty = false;
	multimesh->visible_instances = MIN(multimesh->visible_instances, multimesh->instances);
	multimesh->motion_vectors_current_offset = 0;
	multimesh->motion_vectors_previous_offset = 0;
	multimesh->motion_vectors_last_change = -1;
	multimesh->motion_vectors_enabled = false;

	if (multimesh->instances) {
		uint32_t buffer_size = multimesh->instances * multimesh->stride_cache * sizeof(float);
		BitField<RD::BufferCreationBits> mm_flags = 0;
		if (RD::get_singleton()->has_feature(RD::SUPPORTS_RAYTRACING_PIPELINE) || RD::get_singleton()->has_feature(RD::SUPPORTS_RAY_QUERY)) {
			mm_flags.set_flag(RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT);
		}
		LocalVector<uint8_t> zeros;
		zeros.resize_initialized(buffer_size);
		multimesh->buffer = RD::get_singleton()->storage_buffer_create(buffer_size, zeros.span(), 0, mm_flags);
	}

	multimesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MULTIMESH);
}

void MeshStorage::_multimesh_enable_motion_vectors(MultiMesh *multimesh) {
	if (multimesh->motion_vectors_enabled) {
		return;
	}

	multimesh->motion_vectors_enabled = true;

	multimesh->motion_vectors_current_offset = 0;
	multimesh->motion_vectors_previous_offset = 0;
	multimesh->motion_vectors_last_change = -1;

	if (!multimesh->data_cache.is_empty()) {
		multimesh->data_cache.append_array(multimesh->data_cache);
	}

	uint32_t buffer_size = multimesh->instances * multimesh->stride_cache * sizeof(float);
	uint32_t new_buffer_size = buffer_size * 2;
	BitField<RD::BufferCreationBits> mm_mv_flags = 0;
	if (RD::get_singleton()->has_feature(RD::SUPPORTS_RAYTRACING_PIPELINE) || RD::get_singleton()->has_feature(RD::SUPPORTS_RAY_QUERY)) {
		mm_mv_flags.set_flag(RD::BUFFER_CREATION_DEVICE_ADDRESS_BIT);
	}
	RID new_buffer = RD::get_singleton()->storage_buffer_create(new_buffer_size, {}, 0, mm_mv_flags);

	if (multimesh->buffer_set && multimesh->data_cache.is_empty()) {
		// If the buffer was set but there's no data cached in the CPU, we copy the buffer directly on the GPU.
		RD::get_singleton()->buffer_copy(multimesh->buffer, new_buffer, 0, 0, buffer_size);
		RD::get_singleton()->buffer_copy(multimesh->buffer, new_buffer, 0, buffer_size, buffer_size);
	} else if (!multimesh->data_cache.is_empty()) {
		// Simply upload the data cached in the CPU, which should already be doubled in size.
		ERR_FAIL_COND(multimesh->data_cache.size() * sizeof(float) != size_t(new_buffer_size));
		RD::get_singleton()->buffer_update(new_buffer, 0, new_buffer_size, multimesh->data_cache.ptr());
	}

	if (multimesh->buffer.is_valid()) {
		RD::get_singleton()->free_rid(multimesh->buffer);
	}

	multimesh->buffer = new_buffer;
	multimesh->uniform_set_3d = RID(); // Cleared by dependency.

	// Invalidate any references to the buffer that was released and the uniform set that was pointing to it.
	multimesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MULTIMESH);
}

void MeshStorage::_multimesh_get_motion_vectors_offsets(RID p_multimesh, uint32_t &r_current_offset, uint32_t &r_prev_offset) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL(multimesh);
	r_current_offset = multimesh->motion_vectors_current_offset;
	if (!_multimesh_uses_motion_vectors(multimesh)) {
		multimesh->motion_vectors_previous_offset = multimesh->motion_vectors_current_offset;
	}
	r_prev_offset = multimesh->motion_vectors_previous_offset;
}

bool MeshStorage::_multimesh_uses_motion_vectors_offsets(RID p_multimesh) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, false);
	return _multimesh_uses_motion_vectors(multimesh);
}

int MeshStorage::_multimesh_get_instance_count(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, 0);
	return multimesh->instances;
}

void MeshStorage::_multimesh_set_mesh(RID p_multimesh, RID p_mesh) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL(multimesh);
	if (multimesh->mesh == p_mesh) {
		return;
	}
	multimesh->mesh = p_mesh;

	if (multimesh->indirect) {
		Mesh *mesh = mesh_owner.get_or_null(p_mesh);
		ERR_FAIL_NULL(mesh);
		if (mesh->surface_count > 0) {
			if (multimesh->command_buffer.is_valid()) {
				RD::get_singleton()->free_rid(multimesh->command_buffer);
			}

			Vector<uint8_t> newVector;
			newVector.resize_initialized(sizeof(uint32_t) * INDIRECT_MULTIMESH_COMMAND_STRIDE * mesh->surface_count);

			for (uint32_t i = 0; i < mesh->surface_count; i++) {
				uint32_t count = mesh_surface_get_vertices_drawn_count(mesh->surfaces[i]);
				newVector.set(i * sizeof(uint32_t) * INDIRECT_MULTIMESH_COMMAND_STRIDE, static_cast<uint8_t>(count));
				newVector.set(i * sizeof(uint32_t) * INDIRECT_MULTIMESH_COMMAND_STRIDE + 1, static_cast<uint8_t>(count >> 8));
				newVector.set(i * sizeof(uint32_t) * INDIRECT_MULTIMESH_COMMAND_STRIDE + 2, static_cast<uint8_t>(count >> 16));
				newVector.set(i * sizeof(uint32_t) * INDIRECT_MULTIMESH_COMMAND_STRIDE + 3, static_cast<uint8_t>(count >> 24));
			}

			RID newBuffer = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t) * INDIRECT_MULTIMESH_COMMAND_STRIDE * mesh->surface_count, newVector, RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
			multimesh->command_buffer = newBuffer;
			multimesh->command_buffer_surfaces = mesh->surface_count;
		}
	}

	if (multimesh->instances == 0) {
		return;
	}

	if (multimesh->data_cache.size()) {
		//we have a data cache, just mark it dirt
		_multimesh_mark_all_dirty(multimesh, false, true);
	} else if (multimesh->instances) {
		//need to re-create AABB unfortunately, calling this has a penalty
		if (multimesh->buffer_set) {
			Vector<uint8_t> buffer = RD::get_singleton()->buffer_get_data(multimesh->buffer);
			const uint8_t *r = buffer.ptr() + multimesh->motion_vectors_current_offset * multimesh->stride_cache * sizeof(float);
			const float *data = reinterpret_cast<const float *>(r);
			_multimesh_re_create_aabb(multimesh, data, multimesh->instances);
		}
	}

	multimesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
}

#define MULTIMESH_DIRTY_REGION_SIZE 512

void MeshStorage::_multimesh_make_local(MultiMesh *multimesh) const {
	if (multimesh->data_cache.size() > 0) {
		return; //already local
	}

	// this means that the user wants to load/save individual elements,
	// for this, the data must reside on CPU, so just copy it there.
	uint32_t buffer_size = multimesh->instances * multimesh->stride_cache;
	if (multimesh->motion_vectors_enabled) {
		buffer_size *= 2;
	}
	multimesh->data_cache.resize(buffer_size);
	{
		float *w = multimesh->data_cache.ptrw();

		if (multimesh->buffer_set) {
			Vector<uint8_t> buffer = RD::get_singleton()->buffer_get_data(multimesh->buffer);
			{
				const uint8_t *r = buffer.ptr();
				memcpy(w, r, buffer.size());
			}
		} else {
			memset(w, 0, buffer_size * sizeof(float));
		}
	}
	uint32_t data_cache_dirty_region_count = Math::division_round_up(multimesh->instances, MULTIMESH_DIRTY_REGION_SIZE);
	multimesh->data_cache_dirty_regions = memnew_arr(bool, data_cache_dirty_region_count);
	memset(multimesh->data_cache_dirty_regions, 0, data_cache_dirty_region_count * sizeof(bool));
	multimesh->data_cache_dirty_region_count = 0;

	multimesh->previous_data_cache_dirty_regions = memnew_arr(bool, data_cache_dirty_region_count);
	memset(multimesh->previous_data_cache_dirty_regions, 0, data_cache_dirty_region_count * sizeof(bool));
	multimesh->previous_data_cache_dirty_region_count = 0;
}

void MeshStorage::_multimesh_update_motion_vectors_data_cache(MultiMesh *multimesh) {
	ERR_FAIL_COND(multimesh->data_cache.is_empty());

	if (!multimesh->motion_vectors_enabled) {
		return;
	}

	uint32_t frame = RSG::rasterizer->get_frame_number();
	if (multimesh->motion_vectors_last_change != frame) {
		multimesh->motion_vectors_previous_offset = multimesh->motion_vectors_current_offset;
		multimesh->motion_vectors_current_offset = multimesh->instances - multimesh->motion_vectors_current_offset;
		multimesh->motion_vectors_last_change = frame;

		if (multimesh->previous_data_cache_dirty_region_count > 0) {
			uint8_t *data = (uint8_t *)multimesh->data_cache.ptrw();
			uint32_t current_ofs = multimesh->motion_vectors_current_offset * multimesh->stride_cache * sizeof(float);
			uint32_t previous_ofs = multimesh->motion_vectors_previous_offset * multimesh->stride_cache * sizeof(float);
			uint32_t visible_instances = multimesh->visible_instances >= 0 ? multimesh->visible_instances : multimesh->instances;
			uint32_t visible_region_count = visible_instances == 0 ? 0 : Math::division_round_up(visible_instances, (uint32_t)MULTIMESH_DIRTY_REGION_SIZE);
			uint32_t region_size = multimesh->stride_cache * MULTIMESH_DIRTY_REGION_SIZE * sizeof(float);
			uint32_t size = multimesh->stride_cache * (uint32_t)multimesh->instances * (uint32_t)sizeof(float);
			for (uint32_t i = 0; i < visible_region_count; i++) {
				if (multimesh->previous_data_cache_dirty_regions[i]) {
					uint32_t offset = i * region_size;
					memcpy(data + current_ofs + offset, data + previous_ofs + offset, MIN(region_size, size - offset));
				}
			}
		}
	}
}

bool MeshStorage::_multimesh_uses_motion_vectors(MultiMesh *multimesh) {
	return (RSG::rasterizer->get_frame_number() - multimesh->motion_vectors_last_change) < 2;
}

void MeshStorage::_multimesh_mark_dirty(MultiMesh *multimesh, int p_index, bool p_aabb) {
	uint32_t region_index = p_index / MULTIMESH_DIRTY_REGION_SIZE;
#ifdef DEBUG_ENABLED
	uint32_t data_cache_dirty_region_count = Math::division_round_up(multimesh->instances, MULTIMESH_DIRTY_REGION_SIZE);
	ERR_FAIL_UNSIGNED_INDEX(region_index, data_cache_dirty_region_count); //bug
#endif
	if (!multimesh->data_cache_dirty_regions[region_index]) {
		multimesh->data_cache_dirty_regions[region_index] = true;
		multimesh->data_cache_dirty_region_count++;
	}

	if (p_aabb) {
		multimesh->aabb_dirty = true;
	}

	if (!multimesh->dirty) {
		multimesh->dirty_list = multimesh_dirty_list;
		multimesh_dirty_list = multimesh;
		multimesh->dirty = true;
	}
}

void MeshStorage::_multimesh_mark_all_dirty(MultiMesh *multimesh, bool p_data, bool p_aabb) {
	if (p_data) {
		uint32_t data_cache_dirty_region_count = Math::division_round_up(multimesh->instances, MULTIMESH_DIRTY_REGION_SIZE);

		for (uint32_t i = 0; i < data_cache_dirty_region_count; i++) {
			if (!multimesh->data_cache_dirty_regions[i]) {
				multimesh->data_cache_dirty_regions[i] = true;
				multimesh->data_cache_dirty_region_count++;
			}
		}
	}

	if (p_aabb) {
		multimesh->aabb_dirty = true;
	}

	if (!multimesh->dirty) {
		multimesh->dirty_list = multimesh_dirty_list;
		multimesh_dirty_list = multimesh;
		multimesh->dirty = true;
	}
}

void MeshStorage::_multimesh_re_create_aabb(MultiMesh *multimesh, const float *p_data, int p_instances) {
	ERR_FAIL_COND(multimesh->mesh.is_null());
	if (multimesh->custom_aabb != AABB()) {
		return;
	}
	AABB aabb;
	AABB mesh_aabb = mesh_get_aabb(multimesh->mesh);
	for (int i = 0; i < p_instances; i++) {
		const float *data = p_data + multimesh->stride_cache * i;
		Transform3D t;

		if (multimesh->xform_format == RSE::MULTIMESH_TRANSFORM_3D) {
			t.basis.rows[0][0] = data[0];
			t.basis.rows[0][1] = data[1];
			t.basis.rows[0][2] = data[2];
			t.origin.x = data[3];
			t.basis.rows[1][0] = data[4];
			t.basis.rows[1][1] = data[5];
			t.basis.rows[1][2] = data[6];
			t.origin.y = data[7];
			t.basis.rows[2][0] = data[8];
			t.basis.rows[2][1] = data[9];
			t.basis.rows[2][2] = data[10];
			t.origin.z = data[11];

		} else {
			t.basis.rows[0][0] = data[0];
			t.basis.rows[0][1] = data[1];
			t.origin.x = data[3];

			t.basis.rows[1][0] = data[4];
			t.basis.rows[1][1] = data[5];
			t.origin.y = data[7];
		}

		if (i == 0) {
			aabb = t.xform(mesh_aabb);
		} else {
			aabb.merge_with(t.xform(mesh_aabb));
		}
	}

	multimesh->aabb = aabb;
}

void MeshStorage::_multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform3D &p_transform) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL(multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(multimesh->xform_format != RSE::MULTIMESH_TRANSFORM_3D);

	_multimesh_make_local(multimesh);

	bool uses_motion_vectors = (RSG::viewport->get_num_viewports_with_motion_vectors() > 0) || (RendererCompositorStorage::get_singleton()->get_num_compositor_effects_with_motion_vectors() > 0);
	if (uses_motion_vectors) {
		_multimesh_enable_motion_vectors(multimesh);
	}

	_multimesh_update_motion_vectors_data_cache(multimesh);

	{
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + (multimesh->motion_vectors_current_offset + p_index) * multimesh->stride_cache;

		dataptr[0] = p_transform.basis.rows[0][0];
		dataptr[1] = p_transform.basis.rows[0][1];
		dataptr[2] = p_transform.basis.rows[0][2];
		dataptr[3] = p_transform.origin.x;
		dataptr[4] = p_transform.basis.rows[1][0];
		dataptr[5] = p_transform.basis.rows[1][1];
		dataptr[6] = p_transform.basis.rows[1][2];
		dataptr[7] = p_transform.origin.y;
		dataptr[8] = p_transform.basis.rows[2][0];
		dataptr[9] = p_transform.basis.rows[2][1];
		dataptr[10] = p_transform.basis.rows[2][2];
		dataptr[11] = p_transform.origin.z;
	}

	_multimesh_mark_dirty(multimesh, p_index, true);
}

void MeshStorage::_multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL(multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(multimesh->xform_format != RSE::MULTIMESH_TRANSFORM_2D);

	_multimesh_make_local(multimesh);
	_multimesh_update_motion_vectors_data_cache(multimesh);

	{
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + (multimesh->motion_vectors_current_offset + p_index) * multimesh->stride_cache;

		dataptr[0] = p_transform.columns[0][0];
		dataptr[1] = p_transform.columns[1][0];
		dataptr[2] = 0;
		dataptr[3] = p_transform.columns[2][0];
		dataptr[4] = p_transform.columns[0][1];
		dataptr[5] = p_transform.columns[1][1];
		dataptr[6] = 0;
		dataptr[7] = p_transform.columns[2][1];
	}

	_multimesh_mark_dirty(multimesh, p_index, true);
}

void MeshStorage::_multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL(multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(!multimesh->uses_colors);

	_multimesh_make_local(multimesh);
	_multimesh_update_motion_vectors_data_cache(multimesh);

	{
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + (multimesh->motion_vectors_current_offset + p_index) * multimesh->stride_cache + multimesh->color_offset_cache;

		dataptr[0] = p_color.r;
		dataptr[1] = p_color.g;
		dataptr[2] = p_color.b;
		dataptr[3] = p_color.a;
	}

	_multimesh_mark_dirty(multimesh, p_index, false);
}

void MeshStorage::_multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL(multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(!multimesh->uses_custom_data);

	_multimesh_make_local(multimesh);
	_multimesh_update_motion_vectors_data_cache(multimesh);

	{
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + (multimesh->motion_vectors_current_offset + p_index) * multimesh->stride_cache + multimesh->custom_data_offset_cache;

		dataptr[0] = p_color.r;
		dataptr[1] = p_color.g;
		dataptr[2] = p_color.b;
		dataptr[3] = p_color.a;
	}

	_multimesh_mark_dirty(multimesh, p_index, false);
}

RID MeshStorage::_multimesh_get_mesh(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, RID());

	return multimesh->mesh;
}

Dependency *MeshStorage::multimesh_get_dependency(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, nullptr);

	return &multimesh->dependency;
}

Transform3D MeshStorage::_multimesh_instance_get_transform(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, Transform3D());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Transform3D());
	ERR_FAIL_COND_V(multimesh->xform_format != RSE::MULTIMESH_TRANSFORM_3D, Transform3D());

	_multimesh_make_local(multimesh);

	Transform3D t;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + (multimesh->motion_vectors_current_offset + p_index) * multimesh->stride_cache;

		t.basis.rows[0][0] = dataptr[0];
		t.basis.rows[0][1] = dataptr[1];
		t.basis.rows[0][2] = dataptr[2];
		t.origin.x = dataptr[3];
		t.basis.rows[1][0] = dataptr[4];
		t.basis.rows[1][1] = dataptr[5];
		t.basis.rows[1][2] = dataptr[6];
		t.origin.y = dataptr[7];
		t.basis.rows[2][0] = dataptr[8];
		t.basis.rows[2][1] = dataptr[9];
		t.basis.rows[2][2] = dataptr[10];
		t.origin.z = dataptr[11];
	}

	return t;
}

Transform2D MeshStorage::_multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, Transform2D());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Transform2D());
	ERR_FAIL_COND_V(multimesh->xform_format != RSE::MULTIMESH_TRANSFORM_2D, Transform2D());

	_multimesh_make_local(multimesh);

	Transform2D t;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + (multimesh->motion_vectors_current_offset + p_index) * multimesh->stride_cache;

		t.columns[0][0] = dataptr[0];
		t.columns[1][0] = dataptr[1];
		t.columns[2][0] = dataptr[3];
		t.columns[0][1] = dataptr[4];
		t.columns[1][1] = dataptr[5];
		t.columns[2][1] = dataptr[7];
	}

	return t;
}

Color MeshStorage::_multimesh_instance_get_color(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, Color());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Color());
	ERR_FAIL_COND_V(!multimesh->uses_colors, Color());

	_multimesh_make_local(multimesh);

	Color c;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + (multimesh->motion_vectors_current_offset + p_index) * multimesh->stride_cache + multimesh->color_offset_cache;

		c.r = dataptr[0];
		c.g = dataptr[1];
		c.b = dataptr[2];
		c.a = dataptr[3];
	}

	return c;
}

Color MeshStorage::_multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, Color());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Color());
	ERR_FAIL_COND_V(!multimesh->uses_custom_data, Color());

	_multimesh_make_local(multimesh);

	Color c;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + (multimesh->motion_vectors_current_offset + p_index) * multimesh->stride_cache + multimesh->custom_data_offset_cache;

		c.r = dataptr[0];
		c.g = dataptr[1];
		c.b = dataptr[2];
		c.a = dataptr[3];
	}

	return c;
}

void MeshStorage::_multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL(multimesh);
	ERR_FAIL_COND(p_buffer.size() != (multimesh->instances * (int)multimesh->stride_cache));

	bool used_motion_vectors = multimesh->motion_vectors_enabled;
	bool uses_motion_vectors = (RSG::viewport->get_num_viewports_with_motion_vectors() > 0) || (RendererCompositorStorage::get_singleton()->get_num_compositor_effects_with_motion_vectors() > 0);
	if (uses_motion_vectors) {
		_multimesh_enable_motion_vectors(multimesh);
	}

	if (multimesh->motion_vectors_enabled) {
		uint32_t frame = RSG::rasterizer->get_frame_number();

		if (multimesh->motion_vectors_last_change != frame) {
			multimesh->motion_vectors_previous_offset = multimesh->motion_vectors_current_offset;
			multimesh->motion_vectors_current_offset = multimesh->instances - multimesh->motion_vectors_current_offset;
			multimesh->motion_vectors_last_change = frame;
		}
	}

	{
		const float *r = p_buffer.ptr();
		RD::get_singleton()->buffer_update(multimesh->buffer, multimesh->motion_vectors_current_offset * multimesh->stride_cache * sizeof(float), p_buffer.size() * sizeof(float), r);
		if (multimesh->motion_vectors_enabled && !used_motion_vectors) {
			// Motion vectors were just enabled, and the other half of the buffer will be empty.
			// Need to ensure that both halves are filled for correct operation.
			RD::get_singleton()->buffer_update(multimesh->buffer, multimesh->motion_vectors_previous_offset * multimesh->stride_cache * sizeof(float), p_buffer.size() * sizeof(float), r);
		}
		multimesh->buffer_set = true;
	}

	// rendering/pathtracing/multimesh_cache_cpu_transforms: keep a CPU mirror of the
	// transform buffer so the path tracer never needs a GPU->CPU readback.
	// Trade-off: CPU memory usage doubles for every MultiMesh that calls set_buffer().
	static const bool keep_cpu_cache = GLOBAL_GET("rendering/pathtracing/multimesh_cache_cpu_transforms");
	if (keep_cpu_cache && multimesh->data_cache.size() == 0) {
		// Initialize the data cache from the data we already have on CPU.
		uint32_t cache_size = multimesh->instances * multimesh->stride_cache;
		if (multimesh->motion_vectors_enabled) {
			cache_size *= 2;
		}
		multimesh->data_cache.resize(cache_size);
		memset(multimesh->data_cache.ptrw(), 0, cache_size * sizeof(float));
		uint32_t region_count = Math::division_round_up((uint32_t)multimesh->instances, (uint32_t)MULTIMESH_DIRTY_REGION_SIZE);
		multimesh->data_cache_dirty_regions = memnew_arr(bool, region_count);
		memset(multimesh->data_cache_dirty_regions, 0, region_count * sizeof(bool));
		multimesh->data_cache_dirty_region_count = 0;
		multimesh->previous_data_cache_dirty_regions = memnew_arr(bool, region_count);
		memset(multimesh->previous_data_cache_dirty_regions, 0, region_count * sizeof(bool));
		multimesh->previous_data_cache_dirty_region_count = 0;
	}

	if (multimesh->data_cache.size()) {
		float *cache_data = multimesh->data_cache.ptrw();
		memcpy(cache_data + (multimesh->motion_vectors_current_offset * multimesh->stride_cache), p_buffer.ptr(), p_buffer.size() * sizeof(float));
		_multimesh_mark_all_dirty(multimesh, true, true); //update AABB
	} else if (multimesh->mesh.is_valid()) {
		//if we have a mesh set, we need to re-generate the AABB from the new data
		const float *data = p_buffer.ptr();

		if (multimesh->custom_aabb == AABB()) {
			_multimesh_re_create_aabb(multimesh, data, multimesh->instances);
			multimesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_AABB);
		}
	}
}

RID MeshStorage::_multimesh_get_command_buffer_rd_rid(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, RID());
	return multimesh->command_buffer;
}

RID MeshStorage::_multimesh_get_buffer_rd_rid(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, RID());
	return multimesh->buffer;
}

Vector<float> MeshStorage::_multimesh_get_buffer(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, Vector<float>());
	if (multimesh->buffer.is_null()) {
		return Vector<float>();
	} else {
		Vector<float> ret;
		ret.resize(multimesh->instances * multimesh->stride_cache);
		float *w = ret.ptrw();

		if (multimesh->data_cache.size()) {
			const uint8_t *r = (uint8_t *)multimesh->data_cache.ptr() + multimesh->motion_vectors_current_offset * multimesh->stride_cache * sizeof(float);
			memcpy(w, r, ret.size() * sizeof(float));
		} else {
			Vector<uint8_t> buffer = RD::get_singleton()->buffer_get_data(multimesh->buffer);
			const uint8_t *r = buffer.ptr() + multimesh->motion_vectors_current_offset * multimesh->stride_cache * sizeof(float);
			memcpy(w, r, ret.size() * sizeof(float));
		}
		return ret;
	}
}

void MeshStorage::_multimesh_set_visible_instances(RID p_multimesh, int p_visible) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL(multimesh);
	ERR_FAIL_COND(p_visible < -1 || p_visible > multimesh->instances);
	if (multimesh->visible_instances == p_visible) {
		return;
	}

	if (multimesh->data_cache.size()) {
		// There is a data cache, but we may need to update some sections.
		_multimesh_mark_all_dirty(multimesh, false, true);
		int start = multimesh->visible_instances >= 0 ? multimesh->visible_instances : multimesh->instances;
		for (int i = start; i < p_visible; i++) {
			_multimesh_mark_dirty(multimesh, i, true);
		}
	}

	multimesh->visible_instances = p_visible;

	if (multimesh->indirect) { //we have to update the command buffer for the instance counts, in each stride this will be the second integer.
		Mesh *mesh = mesh_owner.get_or_null(multimesh->mesh);
		if (mesh != nullptr) {
			for (uint32_t i = 0; i < mesh->surface_count; i++) {
				RD::get_singleton()->buffer_update(multimesh->command_buffer, (i * sizeof(uint32_t) * INDIRECT_MULTIMESH_COMMAND_STRIDE) + sizeof(uint32_t), sizeof(uint32_t), &p_visible);
			}
		}
	}

	multimesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MULTIMESH_VISIBLE_INSTANCES);
}

int MeshStorage::_multimesh_get_visible_instances(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, 0);
	return multimesh->visible_instances;
}

void MeshStorage::_multimesh_set_custom_aabb(RID p_multimesh, const AABB &p_aabb) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL(multimesh);
	multimesh->custom_aabb = p_aabb;
	multimesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_AABB);
}

AABB MeshStorage::_multimesh_get_custom_aabb(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, AABB());
	return multimesh->custom_aabb;
}

AABB MeshStorage::_multimesh_get_aabb(RID p_multimesh) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V(multimesh, AABB());
	if (multimesh->custom_aabb != AABB()) {
		return multimesh->custom_aabb;
	}

	if (multimesh->aabb_dirty) {
		_update_dirty_multimeshes();
	}
	return multimesh->aabb;
}

MeshStorage::MultiMeshInterpolator *MeshStorage::_multimesh_get_interpolator(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_NULL_V_MSG(multimesh, nullptr, "Multimesh not found: " + itos(p_multimesh.get_id()));

	return &multimesh->interpolator;
}

void MeshStorage::_update_dirty_multimeshes() {
	while (multimesh_dirty_list) {
		MultiMesh *multimesh = multimesh_dirty_list;

		if (multimesh->data_cache.size()) { //may have been cleared, so only process if it exists

			uint32_t visible_instances = multimesh->visible_instances >= 0 ? multimesh->visible_instances : multimesh->instances;
			uint32_t buffer_offset = multimesh->motion_vectors_current_offset * multimesh->stride_cache;
			const float *data = multimesh->data_cache.ptr() + buffer_offset;

			uint32_t total_dirty_regions = multimesh->data_cache_dirty_region_count + multimesh->previous_data_cache_dirty_region_count;
			if (total_dirty_regions != 0) {
				uint32_t data_cache_dirty_region_count = Math::division_round_up(multimesh->instances, (int)MULTIMESH_DIRTY_REGION_SIZE);
				uint32_t visible_region_count = visible_instances == 0 ? 0 : Math::division_round_up(visible_instances, (uint32_t)MULTIMESH_DIRTY_REGION_SIZE);

				uint32_t region_size = multimesh->stride_cache * MULTIMESH_DIRTY_REGION_SIZE * sizeof(float);
				if (total_dirty_regions > 32 || total_dirty_regions > visible_region_count / 2) {
					//if there too many dirty regions, or represent the majority of regions, just copy all, else transfer cost piles up too much
					RD::get_singleton()->buffer_update(multimesh->buffer, buffer_offset * sizeof(float), MIN(visible_region_count * region_size, multimesh->instances * (uint32_t)multimesh->stride_cache * (uint32_t)sizeof(float)), data);
				} else {
					//not that many regions? update them all
					for (uint32_t i = 0; i < visible_region_count; i++) {
						if (multimesh->data_cache_dirty_regions[i] || multimesh->previous_data_cache_dirty_regions[i]) {
							uint32_t offset = i * region_size;
							uint32_t size = multimesh->stride_cache * (uint32_t)multimesh->instances * (uint32_t)sizeof(float);
							uint32_t region_start_index = multimesh->stride_cache * MULTIMESH_DIRTY_REGION_SIZE * i;
							RD::get_singleton()->buffer_update(multimesh->buffer, buffer_offset * sizeof(float) + offset, MIN(region_size, size - offset), &data[region_start_index]);
						}
					}
				}

				memcpy(multimesh->previous_data_cache_dirty_regions, multimesh->data_cache_dirty_regions, data_cache_dirty_region_count * sizeof(bool));
				memset(multimesh->data_cache_dirty_regions, 0, data_cache_dirty_region_count * sizeof(bool));

				multimesh->previous_data_cache_dirty_region_count = multimesh->data_cache_dirty_region_count;
				multimesh->data_cache_dirty_region_count = 0;
			}

			if (multimesh->aabb_dirty) {
				//aabb is dirty..
				multimesh->aabb_dirty = false;
				if (multimesh->custom_aabb == AABB()) {
					_multimesh_re_create_aabb(multimesh, data, visible_instances);
					multimesh->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_AABB);
				}
			}
		}

		multimesh_dirty_list = multimesh->dirty_list;

		multimesh->dirty_list = nullptr;
		multimesh->dirty = false;
	}

	multimesh_dirty_list = nullptr;
}

/* SKELETON API */

RID MeshStorage::skeleton_allocate() {
	return skeleton_owner.allocate_rid();
}
void MeshStorage::skeleton_initialize(RID p_rid) {
	skeleton_owner.initialize_rid(p_rid, Skeleton());
}

void MeshStorage::skeleton_free(RID p_rid) {
	_update_dirty_skeletons();
	skeleton_allocate_data(p_rid, 0);
	Skeleton *skeleton = skeleton_owner.get_or_null(p_rid);
	skeleton->dependency.deleted_notify(p_rid);
	skeleton_owner.free(p_rid);
}

void MeshStorage::_skeleton_make_dirty(Skeleton *skeleton) {
	if (!skeleton->dirty) {
		skeleton->dirty = true;
		skeleton->dirty_list = skeleton_dirty_list;
		skeleton_dirty_list = skeleton;
	}
}

void MeshStorage::skeleton_allocate_data(RID p_skeleton, int p_bones, bool p_2d_skeleton) {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);
	ERR_FAIL_NULL(skeleton);
	ERR_FAIL_COND(p_bones < 0);

	if (skeleton->size == p_bones && skeleton->use_2d == p_2d_skeleton) {
		return;
	}

	skeleton->size = p_bones;
	skeleton->use_2d = p_2d_skeleton;
	skeleton->uniform_set_3d = RID();

	if (skeleton->buffer.is_valid()) {
		RD::get_singleton()->free_rid(skeleton->buffer);
		skeleton->buffer = RID();
		skeleton->data.clear();
		skeleton->uniform_set_mi = RID();
	}

	if (skeleton->size) {
		skeleton->data.resize(skeleton->size * (skeleton->use_2d ? 8 : 12));
		skeleton->buffer = RD::get_singleton()->storage_buffer_create(skeleton->data.size() * sizeof(float));
		memset(skeleton->data.ptr(), 0, skeleton->data.size() * sizeof(float));

		_skeleton_make_dirty(skeleton);

		{
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.binding = 0;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.append_id(skeleton->buffer);
				uniforms.push_back(u);
			}
			skeleton->uniform_set_mi = RD::get_singleton()->uniform_set_create(uniforms, skeleton_shader.version_shader[0], SkeletonShader::UNIFORM_SET_SKELETON);
		}
	}

	skeleton->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_SKELETON_DATA);
}

int MeshStorage::skeleton_get_bone_count(RID p_skeleton) const {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);
	ERR_FAIL_NULL_V(skeleton, 0);

	return skeleton->size;
}

void MeshStorage::skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform3D &p_transform) {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	ERR_FAIL_NULL(skeleton);
	ERR_FAIL_INDEX(p_bone, skeleton->size);
	ERR_FAIL_COND(skeleton->use_2d);

	float *dataptr = skeleton->data.ptr() + p_bone * 12;

	dataptr[0] = p_transform.basis.rows[0][0];
	dataptr[1] = p_transform.basis.rows[0][1];
	dataptr[2] = p_transform.basis.rows[0][2];
	dataptr[3] = p_transform.origin.x;
	dataptr[4] = p_transform.basis.rows[1][0];
	dataptr[5] = p_transform.basis.rows[1][1];
	dataptr[6] = p_transform.basis.rows[1][2];
	dataptr[7] = p_transform.origin.y;
	dataptr[8] = p_transform.basis.rows[2][0];
	dataptr[9] = p_transform.basis.rows[2][1];
	dataptr[10] = p_transform.basis.rows[2][2];
	dataptr[11] = p_transform.origin.z;

	_skeleton_make_dirty(skeleton);
}

Transform3D MeshStorage::skeleton_bone_get_transform(RID p_skeleton, int p_bone) const {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	ERR_FAIL_NULL_V(skeleton, Transform3D());
	ERR_FAIL_INDEX_V(p_bone, skeleton->size, Transform3D());
	ERR_FAIL_COND_V(skeleton->use_2d, Transform3D());

	const float *dataptr = skeleton->data.ptr() + p_bone * 12;

	Transform3D t;

	t.basis.rows[0][0] = dataptr[0];
	t.basis.rows[0][1] = dataptr[1];
	t.basis.rows[0][2] = dataptr[2];
	t.origin.x = dataptr[3];
	t.basis.rows[1][0] = dataptr[4];
	t.basis.rows[1][1] = dataptr[5];
	t.basis.rows[1][2] = dataptr[6];
	t.origin.y = dataptr[7];
	t.basis.rows[2][0] = dataptr[8];
	t.basis.rows[2][1] = dataptr[9];
	t.basis.rows[2][2] = dataptr[10];
	t.origin.z = dataptr[11];

	return t;
}

void MeshStorage::skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	ERR_FAIL_NULL(skeleton);
	ERR_FAIL_INDEX(p_bone, skeleton->size);
	ERR_FAIL_COND(!skeleton->use_2d);
	float *dataptr = skeleton->data.ptr() + p_bone * 8;

	dataptr[0] = p_transform.columns[0][0];
	dataptr[1] = p_transform.columns[1][0];
	dataptr[2] = 0;
	dataptr[3] = p_transform.columns[2][0];
	dataptr[4] = p_transform.columns[0][1];
	dataptr[5] = p_transform.columns[1][1];
	dataptr[6] = 0;
	dataptr[7] = p_transform.columns[2][1];

	_skeleton_make_dirty(skeleton);
}

Transform2D MeshStorage::skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	ERR_FAIL_NULL_V(skeleton, Transform2D());
	ERR_FAIL_INDEX_V(p_bone, skeleton->size, Transform2D());
	ERR_FAIL_COND_V(!skeleton->use_2d, Transform2D());

	const float *dataptr = skeleton->data.ptr() + p_bone * 8;

	Transform2D t;
	t.columns[0][0] = dataptr[0];
	t.columns[1][0] = dataptr[1];
	t.columns[2][0] = dataptr[3];
	t.columns[0][1] = dataptr[4];
	t.columns[1][1] = dataptr[5];
	t.columns[2][1] = dataptr[7];

	return t;
}

void MeshStorage::skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	ERR_FAIL_NULL(skeleton);
	ERR_FAIL_COND(!skeleton->use_2d);

	skeleton->base_transform_2d = p_base_transform;
}

void MeshStorage::_update_dirty_skeletons() {
	while (skeleton_dirty_list) {
		Skeleton *skeleton = skeleton_dirty_list;

		if (skeleton->size) {
			RD::get_singleton()->buffer_update(skeleton->buffer, 0, skeleton->data.size() * sizeof(float), skeleton->data.ptr());
		}

		skeleton_dirty_list = skeleton->dirty_list;

		skeleton->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_SKELETON_BONES);

		skeleton->version++;

		skeleton->dirty = false;
		skeleton->dirty_list = nullptr;
	}

	skeleton_dirty_list = nullptr;
}

void MeshStorage::skeleton_update_dependency(RID p_skeleton, DependencyTracker *p_instance) {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);
	ERR_FAIL_NULL(skeleton);

	p_instance->update_dependency(&skeleton->dependency);
}
