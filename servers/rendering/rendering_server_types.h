/**************************************************************************/
/*  rendering_server_types.h                                              */
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

#include "core/io/image.h"
#include "core/math/aabb.h"
#include "core/math/rect2.h"
#include "core/math/rect2i.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2.h"
#include "core/math/vector4.h"
#include "core/string/ustring.h"
#include "core/templates/rid.h"
#include "core/templates/vector.h"
#include "servers/rendering/rendering_server_enums.h"

#include <cstdint>

template <typename T>
class Vector;

namespace RenderingServerTypes {

/* TEXTURE API */

typedef void (*TextureDetectCallback)(void *);
typedef void (*TextureDetectRoughnessCallback)(void *, const String &, RSE::TextureDetectRoughnessChannel);

struct TextureInfo {
	RID texture;
	uint32_t width;
	uint32_t height;
	uint32_t depth;
	Image::Format format;
	int64_t bytes;
	String path;
	RSE::TextureType type;
};

/* SHADER API */

struct ShaderNativeSourceCode {
	struct Version {
		struct Stage {
			String name;
			String code;
		};
		Vector<Stage> stages;
	};
	Vector<Version> versions;
};

/* MESH API */

struct SurfaceData {
	RSE::PrimitiveType primitive = RSE::PRIMITIVE_MAX;

	uint64_t format = RSE::ARRAY_FLAG_FORMAT_CURRENT_VERSION;
	Vector<uint8_t> vertex_data; // Vertex, Normal, Tangent (change with skinning, blendshape).
	Vector<uint8_t> attribute_data; // Color, UV, UV2, Custom0-3.
	Vector<uint8_t> skin_data; // Bone index, Bone weight.
	uint32_t vertex_count = 0;
	Vector<uint8_t> index_data;
	uint32_t index_count = 0;

	AABB aabb;
	struct LOD {
		float edge_length = 0.0f;
		Vector<uint8_t> index_data;
	};
	Vector<LOD> lods;
	Vector<AABB> bone_aabbs;

	// Transforms used in runtime bone AABBs compute.
	// Since bone AABBs is saved in Mesh space, but bones is in Skeleton space.
	Transform3D mesh_to_skeleton_xform;

	Vector<uint8_t> blend_shape_data;

	Vector4 uv_scale;

	// The surface's triangle clusters (D80), present when the surface was created with
	// rendering/pathtracing/cluster_blas on: a uint32 version, then one uint32 per cluster
	// holding its triangle count in the low 16 bits and the count of distinct vertices its
	// indices reach in the high 16 bits. The clusters partition the index buffer in order,
	// so a cluster's first triangle is the sum of the counts before it. Empty otherwise.
	// Version 2 (D85, created with rendering/pathtracing/cluster_lod on as well) is a cluster
	// LOD hierarchy, all words uint32 or float: an 8-word header (version, cluster count over
	// every level, group count, level count, LOD 0 cluster count, LOD 0 triangle count, coarse
	// triangle count, flags: MESH_CLUSTER_LOD_FLAG_STALE once the positions were updated in
	// place, D86); per cluster 8 words (triangles | vertices << 16 as in version
	// 1, group, refined group as int32 or -1, bounds center x y z, radius, error); per group 8
	// words (depth, first cluster, cluster count, simplified bounds center x y z, radius, error,
	// FLT_MAX for a terminal group); then the coarse clusters' triangles as uint32 vertex
	// indices, 3 per triangle. Clusters and groups are in the builder's output order, the LOD 0
	// clusters first (their triangles are index_data, in that order), so a cluster's first
	// triangle in the concatenation [index_data | coarse triangles] is again the sum of the
	// counts before it. The RD mesh storage appends the coarse triangles to the surface's index
	// buffer past index_count; the raster draws index_count, the trace reaches every cluster.
	Vector<uint8_t> cluster_data;

	// The surface's subdivision payload (D122), present when the surface is the refined level of
	// a subdivision surface, built at import. Its layout is MeshSubdivisionConstants'. A subdivided
	// surface carries no blend_shape_data whatever the mesh's blend shape count: its blend shapes
	// live on its cage, in the payload. Empty otherwise.
	Vector<uint8_t> subdivision_data;

	RID material;
};

// The subdivision payload (D122, docs/character-subdivision.md). All words are little-endian
// uint32 unless a section says otherwise. A header of MESH_SUBDIVISION_HEADER_WORDS words
// (version, flags, the surface holding the owner sections, section count), then per section
// MESH_SUBDIVISION_SECTION_WORDS words (id, byte offset from the payload's start, byte size);
// every section starts on a 16-byte boundary, and an id appears at most once. Every subdivided
// surface holds the surface sections; one surface of each mesh, MESH_SUBDIVISION_FLAG_OWNER in its
// flags, also holds the owner sections and names itself in word 2, and the others, added after it,
// name it in word 2. A subdivided surface carries no skin (its cage does) and, on a mesh with
// blend shapes, shares the mesh with subdivided surfaces only.
enum MeshSubdivisionConstants : uint32_t {
	// Version 2 (D122 S2): the cage's blend shapes as sparse per-vertex deltas (CAGE_BLEND_*), not dense ones.
	// Version 3 (D122 tangent frames): each R's binormal sign (TANGENT_SIGNS), from MikkTSpace's grouping.
	// Version 4 (D122 S6): the displacement shape (owner info word 10) and its heights (DISPLACEMENT_HEIGHTS).
	MESH_SUBDIVISION_DATA_VERSION = 4,
	MESH_SUBDIVISION_HEADER_WORDS = 4,
	MESH_SUBDIVISION_SECTION_WORDS = 3,
	MESH_SUBDIVISION_SECTION_ALIGN = 16,
	MESH_SUBDIVISION_FLAG_OWNER = 1,
	// Bones and weights per cage vertex, as Godot's 8-weight skin: 8 uint16 bones, then 8 uint16
	// unorm weights.
	MESH_SUBDIVISION_CAGE_BONES = 8,

	// Surface sections, on every subdivided surface. R is the surface's vertex, U the refined vertex
	// of the mesh's welded refinement (a vertex split at a UV seam is several R of one U).
	MESH_SUBDIVISION_SECTION_R_TO_U = 1, // uint32 per R.
	MESH_SUBDIVISION_SECTION_TANGENT_OFFSETS = 2, // uint32 per R plus one, into the entries.
	// uint32 triangle * 3 + corner, in the surface's index order. That order is frozen at build: no
	// storage may reorder a subdivided surface's indices (the load-time clustering skips it).
	MESH_SUBDIVISION_SECTION_TANGENT_ENTRIES = 3,
	// One bit per R (bit R % 32 of word R / 32), set where its binormal sign is +1; bits past the last R
	// are 0. The sign is MikkTSpace's for the R's group, so a vertex MikkTSpace splits is several R.
	MESH_SUBDIVISION_SECTION_TANGENT_SIGNS = 4,

	// Owner sections. The owner's cage is its own cage vertices, then its ghosts (another mesh's cage
	// vertices its stencils reach); its U are its own, then the ghost U the seam's normals need.
	MESH_SUBDIVISION_SECTION_OWNER_INFO = 16, // MESH_SUBDIVISION_OWNER_INFO_WORDS words, below.
	MESH_SUBDIVISION_SECTION_CAGE_POSITIONS = 17, // float x, y, z per cage vertex.
	MESH_SUBDIVISION_SECTION_CAGE_SKIN = 18, // 8 uint16 bones, 8 uint16 unorm weights per cage vertex; absent without skin.
	// Id 19 held version 1's dense blend deltas and is unknown since version 2.
	MESH_SUBDIVISION_SECTION_STENCIL_OFFSETS = 20, // uint32 per U plus one, into the indices and weights.
	MESH_SUBDIVISION_SECTION_STENCIL_INDICES = 21, // uint16 or uint32 (info word 8) cage vertex per weight.
	MESH_SUBDIVISION_SECTION_STENCIL_WEIGHTS = 22, // float per weight.
	MESH_SUBDIVISION_SECTION_QUADS = 23, // uint32 4 U per refined quad in the source's corner order: own quads, then ghost quads.
	MESH_SUBDIVISION_SECTION_NORMAL_OFFSETS = 24, // uint32 per own U plus one, into the entries.
	MESH_SUBDIVISION_SECTION_NORMAL_ENTRIES = 25, // uint32 quad * 4 + corner.
	MESH_SUBDIVISION_SECTION_CAGE_WELDED = 26, // uint32 per cage vertex: its index in the welded cage.
	MESH_SUBDIVISION_SECTION_U_WELDED = 27, // uint32 per U: its index in the welded refinement.
	// The cage's blend shapes, per cage vertex: its non-zero deltas, one 16-byte entry per shape that
	// moves it (uint32 shape index in the mesh's order, then float x, y, z), shapes strictly ascending
	// within a vertex. A runtime pass sums them in that order, which gives the same bits as a dense loop
	// over the weighted shapes: a zero delta adds exactly.
	MESH_SUBDIVISION_SECTION_CAGE_BLEND_OFFSETS = 28, // uint32 per cage vertex plus one, into the entries.
	MESH_SUBDIVISION_SECTION_CAGE_BLEND_ENTRIES = 29,
	MESH_SUBDIVISION_CAGE_BLEND_ENTRY_WORDS = 4,
	// float per own U: its displacement height along its undisplaced normal, scaled at run time by the
	// weight of the blend shape owner info word 10 names. Present exactly when that word names one.
	MESH_SUBDIVISION_SECTION_DISPLACEMENT_HEIGHTS = 30,

	// MESH_SUBDIVISION_SECTION_OWNER_INFO's words.
	MESH_SUBDIVISION_INFO_CAGE_COUNT = 0,
	MESH_SUBDIVISION_INFO_OWN_CAGE_COUNT = 1,
	MESH_SUBDIVISION_INFO_U_COUNT = 2,
	MESH_SUBDIVISION_INFO_OWN_U_COUNT = 3,
	MESH_SUBDIVISION_INFO_QUAD_COUNT = 4,
	MESH_SUBDIVISION_INFO_OWN_QUAD_COUNT = 5,
	MESH_SUBDIVISION_INFO_BLEND_SHAPE_COUNT = 6,
	MESH_SUBDIVISION_INFO_LEVEL = 7,
	MESH_SUBDIVISION_INFO_STENCIL_INDEX_BYTES = 8,
	MESH_SUBDIVISION_INFO_STENCIL_WEIGHT_COUNT = 9,
	MESH_SUBDIVISION_INFO_DISPLACEMENT_SHAPE = 10, // The blend shape whose weight scales the displacement, or MESH_SUBDIVISION_NO_DISPLACEMENT.
	MESH_SUBDIVISION_OWNER_INFO_WORDS = 11,
	MESH_SUBDIVISION_NO_DISPLACEMENT = 0xFFFFFFFF,
};

enum MeshClusterConstants : uint32_t {
	MESH_CLUSTER_DATA_VERSION = 1,
	// meshopt limits are 256 vertices and 512 triangles; every cluster acceleration structure
	// device allows at least 256 of each. 128 is the size RTX Mega Geometry recommends for
	// static geometry.
	MESH_CLUSTER_MAX_TRIANGLES = 128,
	MESH_CLUSTER_MIN_TRIANGLES = 32,
	MESH_CLUSTER_MAX_VERTICES = 128,
	// The cluster LOD hierarchy (D85): its table version and word counts, and the partition
	// target (meshopt_partitionClusters may exceed its target by a third, so 24 holds a group to
	// at most 32 clusters, the NVIDIA sample's group size).
	MESH_CLUSTER_LOD_DATA_VERSION = 2,
	MESH_CLUSTER_LOD_HEADER_WORDS = 8,
	MESH_CLUSTER_LOD_CLUSTER_WORDS = 8,
	MESH_CLUSTER_LOD_GROUP_WORDS = 8,
	MESH_CLUSTER_LOD_PARTITION_SIZE = 24,
	// The header's flags word (word 7): the hierarchy was simplified from positions since
	// updated in place, so its coarse levels are traced no more (D86).
	MESH_CLUSTER_LOD_FLAG_STALE = 1,
};
// The clusterizer's fill weight (D80) and the hierarchy's error merge across levels (D85):
// a level's error is max(previous * merge, own), 1.5 being the NVIDIA sample's choice.
constexpr float MESH_CLUSTER_FILL_WEIGHT = 0.5f;
constexpr float MESH_CLUSTER_LOD_ERROR_MERGE_PREVIOUS = 1.5f;

struct MeshInfo {
	RID mesh;
	String path;
	uint32_t vertex_buffer_size = 0;
	uint32_t attribute_buffer_size = 0;
	uint32_t skin_buffer_size = 0;
	uint32_t index_buffer_size = 0;
	uint32_t blend_shape_buffer_size = 0;
	uint32_t lod_index_buffers_size = 0;
	uint64_t vertex_count = 0;
};

/* STATUS INFORMATION */

struct FrameProfileArea {
	String name;
	double gpu_msec;
	double cpu_msec;
};

/* COMPOSITOR */

struct BlitToScreen {
	RID render_target;
	Rect2 src_rect = Rect2(0.0, 0.0, 1.0, 1.0);
	Rect2i dst_rect;

	struct {
		bool use_layer = false;
		uint32_t layer = 0;
	} multi_view;

	struct {
		//lens distorted parameters for VR
		bool apply = false;
		Vector2 eye_center;
		float k1 = 0.0;
		float k2 = 0.0;

		float upscale = 1.0;
		float aspect_ratio = 1.0;
	} lens_distortion;
};

/* BACKGROUND */

// Helper for RSE::SplashStretchMode, put here for convenience.
inline Rect2 get_splash_stretched_screen_rect(const Size2 &p_image_size, const Size2 &p_window_size, RSE::SplashStretchMode p_stretch_mode) {
	Size2 imgsize = p_image_size;
	Rect2 screenrect;
	switch (p_stretch_mode) {
		case RSE::SPLASH_STRETCH_MODE_DISABLED: {
			screenrect.size = imgsize;
			screenrect.position = ((p_window_size - screenrect.size) / 2.0).floor();
		} break;
		case RSE::SPLASH_STRETCH_MODE_KEEP: {
			if (p_window_size.width > p_window_size.height) {
				// Scale horizontally.
				screenrect.size.y = p_window_size.height;
				screenrect.size.x = imgsize.width * p_window_size.height / imgsize.height;
				screenrect.position.x = (p_window_size.width - screenrect.size.x) / 2;
			} else {
				// Scale vertically.
				screenrect.size.x = p_window_size.width;
				screenrect.size.y = imgsize.height * p_window_size.width / imgsize.width;
				screenrect.position.y = (p_window_size.height - screenrect.size.y) / 2;
			}
		} break;
		case RSE::SPLASH_STRETCH_MODE_KEEP_WIDTH: {
			// Scale vertically.
			screenrect.size.x = p_window_size.width;
			screenrect.size.y = imgsize.height * p_window_size.width / imgsize.width;
			screenrect.position.y = (p_window_size.height - screenrect.size.y) / 2;
		} break;
		case RSE::SPLASH_STRETCH_MODE_KEEP_HEIGHT: {
			// Scale horizontally.
			screenrect.size.y = p_window_size.height;
			screenrect.size.x = imgsize.width * p_window_size.height / imgsize.height;
			screenrect.position.x = (p_window_size.width - screenrect.size.x) / 2;
		} break;
		case RSE::SPLASH_STRETCH_MODE_COVER: {
			double window_aspect = (double)p_window_size.width / p_window_size.height;
			double img_aspect = imgsize.width / imgsize.height;

			if (window_aspect > img_aspect) {
				// Scale vertically.
				screenrect.size.x = p_window_size.width;
				screenrect.size.y = imgsize.height * p_window_size.width / imgsize.width;
				screenrect.position.y = (p_window_size.height - screenrect.size.y) / 2;
			} else {
				// Scale horizontally.
				screenrect.size.y = p_window_size.height;
				screenrect.size.x = imgsize.width * p_window_size.height / imgsize.height;
				screenrect.position.x = (p_window_size.width - screenrect.size.x) / 2;
			}
		} break;
		case RSE::SPLASH_STRETCH_MODE_IGNORE: {
			screenrect.size.x = p_window_size.width;
			screenrect.size.y = p_window_size.height;
		} break;
	}
	return screenrect;
}

/* RENDERING METHOD */

struct RenderInfo {
	int info[RSE::VIEWPORT_RENDER_INFO_TYPE_MAX][RSE::VIEWPORT_RENDER_INFO_MAX] = {};
};

} // namespace RenderingServerTypes
