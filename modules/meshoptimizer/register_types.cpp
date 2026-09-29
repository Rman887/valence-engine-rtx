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

#include <cstddef>

// Triangle clusters at surface creation (D80). RenderingServer::MeshCluster mirrors meshopt_Meshlet;
// the header overloads meshopt_buildMeshletsSpatial for other index types, so the C signature is
// reached through this wrapper.
static_assert(sizeof(RenderingServer::MeshCluster) == sizeof(meshopt_Meshlet) && offsetof(RenderingServer::MeshCluster, triangle_count) == offsetof(meshopt_Meshlet, triangle_count) && offsetof(RenderingServer::MeshCluster, vertex_offset) == offsetof(meshopt_Meshlet, vertex_offset), "RenderingServer::MeshCluster must lay out as meshopt_Meshlet.");

static size_t _build_clusters_spatial(RenderingServer::MeshCluster *r_clusters, unsigned int *r_cluster_vertices, unsigned char *r_cluster_triangles, const unsigned int *p_indices, size_t p_index_count, const float *p_vertex_positions, size_t p_vertex_count, size_t p_vertex_positions_stride, size_t p_max_vertices, size_t p_min_triangles, size_t p_max_triangles, float p_fill_weight) {
	return meshopt_buildMeshletsSpatial(reinterpret_cast<meshopt_Meshlet *>(r_clusters), r_cluster_vertices, r_cluster_triangles, p_indices, p_index_count, p_vertex_positions, p_vertex_count, p_vertex_positions_stride, p_max_vertices, p_min_triangles, p_max_triangles, p_fill_weight);
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
}
