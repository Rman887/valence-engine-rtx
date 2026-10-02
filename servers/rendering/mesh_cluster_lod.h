/**************************************************************************/
/*  mesh_cluster_lod.h                                                    */
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

#pragma once

#include "core/templates/local_vector.h"
#include "core/templates/vector.h"

// Geometry residency (D93): a streamed cluster's blob indexes its own vertices in 8 bits.
constexpr uint32_t MESH_CLUSTER_GEOMETRY_MAX_VERTICES = 256;

// Whether a surface streams its geometry (D93d), decided alike by the RD mesh storage when the
// surface is added and by the path tracer when it is populated.
//
// The first test reads the table (RenderingServerTypes::SurfaceData::cluster_data): a version-2
// hierarchy that describes the index buffer (LOD 0 summing to p_index_count / 3, the coarse levels
// appended within p_index_total, every cluster within the device's limits) and whose residency
// holds (D92: groups own contiguous runs, links stay inside the table, every refinement goes one
// depth finer, bounds and errors are numbers, and some groups but not all are persistent).
bool mesh_cluster_lod_streamable(const Vector<uint8_t> &p_cluster_data, uint32_t p_index_count, uint32_t p_index_total, uint32_t p_max_triangles, uint32_t p_max_vertices);

// The second reads the index buffer's bytes, LOD 0 and the coarse tail: every cluster indexes
// inside the surface and reaches at most MESH_CLUSTER_GEOMETRY_MAX_VERTICES distinct vertices, and
// no more than the table's largest count. r_distinct takes each cluster's count. r_scratch is a
// vertex-to-local map at 0xFFFF, grown as needed and left so. Call it only on a table the first
// accepted.
bool mesh_cluster_lod_local_vertices(const Vector<uint8_t> &p_cluster_data, const uint8_t *p_index_data, uint64_t p_index_bytes, bool p_index_16, uint32_t p_vertex_count, LocalVector<uint16_t> &r_scratch, LocalVector<uint32_t> &r_distinct);
