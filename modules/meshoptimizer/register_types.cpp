/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "scene/resources/surface_tool.h"
#include "servers/rendering/rendering_server.h"

#include <thirdparty/meshoptimizer/meshoptimizer.h>

// The cluster LOD hierarchy builder (D85); its implementation is clusterlod_impl.cpp.
#include <thirdparty/meshoptimizer/clusterlod.h>

#include <cstddef>

// Triangle clusters at surface creation (D80). RenderingServer::MeshCluster mirrors meshopt_Meshlet;
// the header overloads meshopt_buildMeshletsSpatial for other index types, so the C signature is
// reached through this wrapper.
static_assert(sizeof(RenderingServer::MeshCluster) == sizeof(meshopt_Meshlet) && offsetof(RenderingServer::MeshCluster, triangle_count) == offsetof(meshopt_Meshlet, triangle_count) && offsetof(RenderingServer::MeshCluster, vertex_offset) == offsetof(meshopt_Meshlet, vertex_offset), "RenderingServer::MeshCluster must lay out as meshopt_Meshlet.");

static size_t _build_clusters_spatial(RenderingServer::MeshCluster *r_clusters, unsigned int *r_cluster_vertices, unsigned char *r_cluster_triangles, const unsigned int *p_indices, size_t p_index_count, const float *p_vertex_positions, size_t p_vertex_count, size_t p_vertex_positions_stride, size_t p_max_vertices, size_t p_min_triangles, size_t p_max_triangles, float p_fill_weight) {
	return meshopt_buildMeshletsSpatial(reinterpret_cast<meshopt_Meshlet *>(r_clusters), r_cluster_vertices, r_cluster_triangles, p_indices, p_index_count, p_vertex_positions, p_vertex_count, p_vertex_positions_stride, p_max_vertices, p_min_triangles, p_max_triangles, p_fill_weight);
}

// Cluster LOD hierarchies at surface creation (D85): clodBuild's output callback, one call per
// group with the group's clusters, copied into the server's own structs; the group's index is
// what the builder records in the clusters simplified from it (clodCluster::refined).
struct ClusterLodOutput {
	LocalVector<RenderingServer::MeshClusterLodCluster> *clusters = nullptr;
	LocalVector<RenderingServer::MeshClusterLodGroup> *groups = nullptr;
	LocalVector<unsigned int> *indices = nullptr;
	bool ok = true;
};

static int _cluster_lod_output(void *p_context, clodGroup p_group, const clodCluster *p_clusters, size_t p_cluster_count) {
	ClusterLodOutput *out = static_cast<ClusterLodOutput *>(p_context);
	const int group_id = int(out->groups->size());
	RenderingServer::MeshClusterLodGroup group;
	group.depth = p_group.depth;
	group.first_cluster = out->clusters->size();
	group.cluster_count = uint32_t(p_cluster_count);
	group.center[0] = p_group.simplified.center[0];
	group.center[1] = p_group.simplified.center[1];
	group.center[2] = p_group.simplified.center[2];
	group.radius = p_group.simplified.radius;
	group.error = p_group.simplified.error;
	out->groups->push_back(group);
	for (size_t c = 0; c < p_cluster_count; c++) {
		const clodCluster &src = p_clusters[c];
		if (src.index_count == 0 || src.index_count % 3 != 0) {
			out->ok = false;
			continue;
		}
		RenderingServer::MeshClusterLodCluster cluster;
		cluster.triangle_count = uint32_t(src.index_count / 3);
		cluster.vertex_count = uint32_t(src.vertex_count);
		cluster.group = group_id;
		cluster.refined = src.refined;
		cluster.center[0] = src.bounds.center[0];
		cluster.center[1] = src.bounds.center[1];
		cluster.center[2] = src.bounds.center[2];
		cluster.radius = src.bounds.radius;
		cluster.error = src.bounds.error;
		cluster.first_index = out->indices->size();
		for (size_t i = 0; i < src.index_count; i++) {
			out->indices->push_back(src.indices[i]);
		}
		out->clusters->push_back(cluster);
	}
	return group_id;
}

static bool _build_cluster_lod(const float *p_vertex_positions, size_t p_vertex_count, size_t p_vertex_positions_stride, const float *p_vertex_attributes, size_t p_vertex_attributes_stride, const float *p_attribute_weights, size_t p_attribute_count, unsigned int p_attribute_protect_mask, const unsigned int *p_indices, size_t p_index_count, LocalVector<RenderingServer::MeshClusterLodCluster> &r_clusters, LocalVector<RenderingServer::MeshClusterLodGroup> &r_groups, LocalVector<unsigned int> &r_indices) {
	// The ray-tracing defaults, with the clusterizer's D80 limits so LOD 0 is the flat
	// clustering, the NVIDIA sample's group size and error merge, and clusters optimized for
	// locality within (a reorder of each cluster's triangles alone).
	clodConfig config = clodDefaultConfigRT(RenderingServerTypes::MESH_CLUSTER_MAX_TRIANGLES);
	config.max_vertices = RenderingServerTypes::MESH_CLUSTER_MAX_VERTICES;
	config.min_triangles = RenderingServerTypes::MESH_CLUSTER_MIN_TRIANGLES;
	config.cluster_fill_weight = RenderingServerTypes::MESH_CLUSTER_FILL_WEIGHT;
	config.partition_size = RenderingServerTypes::MESH_CLUSTER_LOD_PARTITION_SIZE;
	config.partition_sort = true;
	config.optimize_clusters = true;
	config.optimize_clusters_level = 1;
	config.simplify_error_merge_previous = RenderingServerTypes::MESH_CLUSTER_LOD_ERROR_MERGE_PREVIOUS;

	clodMesh mesh = {};
	mesh.indices = p_indices;
	mesh.index_count = p_index_count;
	mesh.vertex_count = p_vertex_count;
	mesh.vertex_positions = p_vertex_positions;
	mesh.vertex_positions_stride = p_vertex_positions_stride;
	mesh.vertex_attributes = p_vertex_attributes;
	mesh.vertex_attributes_stride = p_vertex_attributes_stride;
	mesh.attribute_weights = p_attribute_weights;
	mesh.attribute_count = p_attribute_count;
	mesh.attribute_protect_mask = p_attribute_protect_mask;

	r_clusters.clear();
	r_groups.clear();
	r_indices.clear();
	ClusterLodOutput out;
	out.clusters = &r_clusters;
	out.groups = &r_groups;
	out.indices = &r_indices;
	const size_t cluster_count = clodBuild(config, mesh, &out, _cluster_lod_output);
	return out.ok && cluster_count > 0 && cluster_count == r_clusters.size();
}

void initialize_meshoptimizer_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	SurfaceTool::optimize_vertex_cache_func = meshopt_optimizeVertexCache;
	SurfaceTool::optimize_vertex_fetch_remap_func = meshopt_optimizeVertexFetchRemap;
	SurfaceTool::simplify_func = meshopt_simplify;
	SurfaceTool::simplify_with_attrib_func = meshopt_simplifyWithAttributes;
	SurfaceTool::simplify_scale_func = meshopt_simplifyScale;
	SurfaceTool::generate_remap_func = meshopt_generateVertexRemap;
	SurfaceTool::remap_vertex_func = meshopt_remapVertexBuffer;
	SurfaceTool::remap_index_func = meshopt_remapIndexBuffer;

	RenderingServer::build_clusters_bound_func = meshopt_buildMeshletsBound;
	RenderingServer::build_clusters_func = _build_clusters_spatial;
	RenderingServer::build_cluster_lod_func = _build_cluster_lod;
}

void uninitialize_meshoptimizer_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	SurfaceTool::optimize_vertex_cache_func = nullptr;
	SurfaceTool::optimize_vertex_fetch_remap_func = nullptr;
	SurfaceTool::simplify_func = nullptr;
	SurfaceTool::simplify_scale_func = nullptr;
	SurfaceTool::generate_remap_func = nullptr;
	SurfaceTool::remap_vertex_func = nullptr;
	SurfaceTool::remap_index_func = nullptr;

	RenderingServer::build_clusters_bound_func = nullptr;
	RenderingServer::build_clusters_func = nullptr;
	RenderingServer::build_cluster_lod_func = nullptr;
}
