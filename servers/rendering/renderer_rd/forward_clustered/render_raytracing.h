/**************************************************************************/
/*  render_raytracing.h                                                   */
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

#include "core/math/transform_3d.h"
#include "core/object/worker_thread_pool.h"
#include "core/string/string_name.h"
#include "core/templates/a_hash_map.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "core/templates/vector.h"
#include "servers/rendering/mesh_cluster_lod.h"
#include "servers/rendering/renderer_rd/bindless_block.h"
#include "servers/rendering/renderer_rd/shaders/raytracing/cluster_lod_select.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/raytracing/multimesh_merge.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/raytracing/opacity_micromap.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/raytracing/ptlas_rows.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/raytracing/sky_light_table.glsl.gen.h"
#include "servers/rendering/rendering_device.h"

#define RB_TEX_RAYTRACING SNAME("raytracing")
#define RB_TEX_RT_DEPTH SNAME("rt_depth")

#define RB_SCOPE_DLSS_RR SNAME("dlss_rr")
#define RB_TEX_DLSS_RR_DIFFUSE_ALBEDO SNAME("diffuse_albedo")
#define RB_TEX_DLSS_RR_SPECULAR_ALBEDO SNAME("specular_albedo")
#define RB_TEX_DLSS_RR_NORMAL_ROUGHNESS SNAME("normal_roughness")
#define RB_TEX_DLSS_RR_SPECULAR_HIT_DIST SNAME("specular_hit_dist")

// The G-buffer primary surface (Environment.pathtracing_primary_surface), one scope so it frees as a unit.
#define RB_SCOPE_RT_GBUFFER SNAME("rt_gbuffer")

class RenderDataRD;
class RenderSceneBuffersRD;

namespace RendererSceneRenderImplementation {

class RenderForwardClustered;
class SceneShaderRaytracing;

// Must match GLSL GeometryData (std430, 128 bytes).
struct alignas(16) RT_GeometryData {
	uint64_t vertex_buffer_address;
	uint64_t attribute_buffer_address;
	uint64_t index_buffer_address;
	uint32_t vertex_count;
	uint32_t position_stride;
	uint32_t normal_byte_offset;
	uint32_t normal_stride;
	uint32_t tangent_byte_offset;
	uint32_t tangent_stride;
	uint32_t attribute_stride;
	uint32_t uv_byte_offset;
	uint32_t uv_scale_packed;
	uint32_t index_format;
	uint32_t primitive_count;
	uint32_t flags;
	float aabb_size_x;
	float aabb_size_y;
	float aabb_size_z;
	uint32_t color_byte_offset;
	float aabb_pos_x;
	float aabb_pos_y;
	float aabb_pos_z;
	// For deformed geometry: previous-frame position buffer used for motion vectors.
	uint32_t prev_vertex_buffer_address_lo;
	uint32_t prev_vertex_buffer_address_hi;
	// With RT_GEOM_FLAG_CLUSTER_GEOMETRY (D93): the surface's first cluster geometry record, in words
	// into the LOD table.
	uint32_t cluster_record_base;
	uint32_t _pad[4];
};
static_assert(sizeof(RT_GeometryData) == 128, "RT_GeometryData must be 128 bytes for std430");

/// Per-instance motion data for velocity computation (matches GLSL InstanceMotionData, 48 bytes).
struct RT_InstanceMotionData {
	float prev_object_to_world[12]; // Previous object-to-world (mat3x4, transposed 3x4).
};
static_assert(sizeof(RT_InstanceMotionData) == 48, "RT_InstanceMotionData must be 48 bytes");

// Must match GLSL MaterialData (std430, 112 bytes).
struct alignas(16) RT_MaterialData {
	uint32_t albedo_texture_idx;
	uint32_t normal_texture_idx;
	uint32_t orm_texture_idx;
	uint32_t emission_texture_idx;
	float albedo_color[4];
	float emission_color[3];
	float emission_strength;
	float metallic;
	float roughness;
	float ao_strength;
	uint32_t flags;
	float uv1_scale[2];
	float uv1_offset[2];
	float normal_map_depth; // Strength [0..N], default 1.0 (not Z-depth).
	float specular; // Dielectric specular [0..1], default 0.5 -> F0 = 0.04.
	uint64_t uniform_address; // BDA for custom shader uniform buffer (0 = none).
	float alpha_threshold; // The alpha test's cutoff: opaque where alpha >= it (default 0.5).
	uint32_t _pad[3];
};
static_assert(sizeof(RT_MaterialData) == 112, "RT_MaterialData must be 112 bytes for std430");

// Light types for raytracing (matches GLSL RT_LIGHT_TYPE_* defines).
enum RTLightType : uint32_t {
	RT_LIGHT_TYPE_OMNI = 0,
	RT_LIGHT_TYPE_DIRECTIONAL = 1,
	RT_LIGHT_TYPE_SPOT = 3,
};

// Must match GLSL RTLightData (std430, 80 bytes).
struct alignas(16) RT_LightData {
	float position[3]; // World position (omni/spot) or direction (directional, normalized).
	uint32_t type;
	float emission[3];
	float radius;
	float attenuation;
	float inv_max_range; // 1.0/range, or -1.0 for infinite.
	float max_range_squared; // range*range, or 0.0 for infinite.
	float specular_amount;
	float indirect_energy;
	float inv_spot_attenuation;
	float cos_spot_angle;
	float _pad0;
	float spot_direction[3];
	float _pad1;
};
static_assert(sizeof(RT_LightData) == 80, "RT_LightData must be 80 bytes for std430");

enum {
	RT_LIGHTS_MAX = 1024, // The light buffer's capacity; the light tree (D55) selects over all of it.
	RT_LIGHTS_UNIFORM_MAX = 64, // The cap without the tree, where the pick is uniform over the list.
	RT_LIGHTS_FRUSTUM_BUDGET = 48,
	RT_LIGHTS_INDIRECT_BUDGET = RT_LIGHTS_UNIFORM_MAX - RT_LIGHTS_FRUSTUM_BUDGET,
};

// An emissive mesh light (D50): a TLAS instance whose material emits a constant radiance, listed
// for the trace's next-event estimation. Must match GLSL RTEmissiveInstance (std430, 96 bytes).
struct alignas(16) RT_EmissiveInstance {
	float xform[12]; // Object-to-world, trace space (mat3x4, transposed 3x4).
	float radiance[3]; // Linear emission before the exposure normalisation.
	float cdf; // Inclusive; the last entry reads 1.0.
	uint32_t instance_index; // The TLAS instance: its geometries[] entry.
	uint32_t primitive_count;
	uint32_t flags;
	float pdf; // This entry's selection probability.
	float center[3]; // The instance's bounding sphere in trace space (D51): the resampler's proximity proxy.
	float radius;
};
static_assert(sizeof(RT_EmissiveInstance) == 96, "RT_EmissiveInstance must be 96 bytes for std430");

enum {
	RT_EMISSIVE_MAX = 1024,
	RT_EMISSIVE_FLAG_TWO_SIDED = 1u,
	RT_EMISSIVE_FLAG_FLIP_FACING = 2u, // The instance flips the winding: its lit face winds clockwise as seen (Godot front faces).
};

// A node of the light tree (D55): a bounding sphere, an orientation cone, the summed power and the
// largest range of the lights under it, its right child and its leaves, and the falloff of its
// analytic lights (D63). Must match GLSL RTLightTreeNode (std430, 64 bytes) in raytracing_light_tree_inc.glsl.
struct alignas(16) RT_LightTreeNode {
	float center[3]; // Trace space.
	float radius;
	float axis[3]; // The orientation cone's axis; unused when cos_theta_o is -1.
	float cos_theta_o; // Cosine of the cone's half-angle, -1 for every direction.
	float power; // Luminance at unit distance, summed over the leaves.
	float max_range; // The largest light range under the node, 0 for unbounded.
	uint32_t child; // An inner node: the right child (the left is the next node); a leaf: RT_LIGHT_TREE_LEAF_BIT | flags | index.
	uint32_t leaf_range; // The node's leaves in the leaf list: first << 16 | count (D56).
	float analytic_share; // The omni and spot lights' share of power (D63): 1 without emitters, 0 without them.
	float decay; // The slowest attenuation exponent among those lights (Light3D's attenuation), within [0, 2] on an inner node; 2 without them.
	float analytic_range; // The largest range among those lights, 0 when one is unbounded or there are none.
	float emissive_share; // The emitters' share of power, which falls off as the inverse square.
};
static_assert(sizeof(RT_LightTreeNode) == 64, "RT_LightTreeNode must be 64 bytes for std430");

// One leaf-list position's entry in its cut node's alias table (D60, D61): a pick lands on the
// position by the leaf count, keeps it when the rescaled remainder is under the threshold and takes
// the alias otherwise, each with its share of the node, so one read gives the leaf and its
// probability. Must match the uvec4 of tree_leaf_alias in raytracing_light_tree_inc.glsl.
struct alignas(16) RT_LightTreeAlias {
	float threshold; // The position keeps a pick whose remainder is below this; 1 for a position with no alias.
	uint32_t alias; // The leaf-list position the pick goes to otherwise (absolute, inside the same cut node).
	float share; // The position's own share of its cut node, the power share with the uniform floor.
	float alias_share; // The alias's share, the same float as the alias position's own.
};
static_assert(sizeof(RT_LightTreeAlias) == 16, "RT_LightTreeAlias must be 16 bytes for std430");

enum : uint32_t {
	RT_LIGHT_TREE_LEAF_BIT = 0x80000000u,
	RT_LIGHT_TREE_EMISSIVE_BIT = 0x40000000u,
	RT_LIGHT_TREE_NO_NODE = 0xFFFFFFFFu,
	RT_LIGHT_TREE_LEAVES_MAX = RT_LIGHTS_MAX + RT_EMISSIVE_MAX,
	RT_LIGHT_TREE_NODES_MAX = 2 * RT_LIGHT_TREE_LEAVES_MAX - 1, // Must match RT_LIGHT_TREE_NODES_MAX in raytracing_light_tree_inc.glsl.
	RT_LIGHT_TREE_CUT_MAX = 64, // The most cut nodes a hit scores (D56); must match RT_LIGHT_TREE_CUT_MAX in raytracing_light_tree_inc.glsl.
	RT_LIGHT_TREE_HEADER_SIZE = 4 * sizeof(uint32_t),
	RT_LIGHT_TREE_EMISSIVE_CUT_OFFSET = RT_LIGHT_TREE_HEADER_SIZE + RT_LIGHT_TREE_NODES_MAX * sizeof(RT_LightTreeNode), // Each emissive entry's cut slots: the primary cut's in the low 16 bits, the indirect cut's in the high (D59).
	RT_LIGHT_TREE_CUT_OFFSET = RT_LIGHT_TREE_EMISSIVE_CUT_OFFSET + RT_EMISSIVE_MAX * sizeof(uint32_t), // The cut's nodes, copied.
	RT_LIGHT_TREE_CUT_INDIRECT_OFFSET = RT_LIGHT_TREE_CUT_OFFSET + RT_LIGHT_TREE_CUT_MAX * sizeof(RT_LightTreeNode), // The indirect bounces' coarser cut (D59), copied.
	RT_LIGHT_TREE_LEAF_LIST_OFFSET = RT_LIGHT_TREE_CUT_INDIRECT_OFFSET + RT_LIGHT_TREE_CUT_MAX * sizeof(RT_LightTreeNode), // The leaf nodes in the build's order.
	RT_LIGHT_TREE_EMISSIVE_LEAF_OFFSET = RT_LIGHT_TREE_LEAF_LIST_OFFSET + RT_LIGHT_TREE_LEAVES_MAX * sizeof(uint32_t), // Each emissive entry's position in the leaf list (D60).
	RT_LIGHT_TREE_LEAF_ALIAS_OFFSET = RT_LIGHT_TREE_EMISSIVE_LEAF_OFFSET + RT_EMISSIVE_MAX * sizeof(uint32_t), // Each leaf's alias-table entry inside its cut node, the primary cut's then the indirect's (D60, D61).
	RT_LIGHT_TREE_BUFFER_SIZE = RT_LIGHT_TREE_LEAF_ALIAS_OFFSET + 2 * RT_LIGHT_TREE_LEAVES_MAX * sizeof(RT_LightTreeAlias),
};
static_assert(RT_LIGHT_TREE_CUT_OFFSET % 16 == 0, "the light tree's cut must start on a 16-byte boundary for std430");
static_assert(RT_LIGHT_TREE_LEAF_ALIAS_OFFSET % 16 == 0, "the light tree's alias table must start on a 16-byte boundary for std430");

enum {
	RT_OFFSET_NONE = 0xFFFFFFFFu,
	RT_CACHE_CHUNK_SIZE = 256,
	RT_CACHE_CHUNK_SHIFT = 8,
	RT_CACHE_CHUNK_MASK = 255,
	// The static cluster arena's chunk size (D81): a constant to size on real geometry, not a setting.
	RT_CLUSTER_ARENA_CHUNK_BYTES = 32u * 1024u * 1024u,
};

// Material flags for RT (matches GLSL mat_flags bit layout).
enum {
	RT_MAT_FLAG_HAS_NORMAL_MAP = 1u,
	RT_MAT_FLAG_HAS_EMISSION_TEX = 2u,
	RT_MAT_FLAG_POINT_FILTER = 4u,
	RT_MAT_FLAG_EMISSION_MULTIPLY = 8u, // BaseMaterial3D emission_operator Multiply; Add otherwise.
	RT_MAT_FLAG_EMISSIVE_CONSTANT = 16u, // Emits a non-zero constant radiance with no texture: an emissive mesh light (D50).
};

// Index format for RT geometry (matches GLSL fetch_indices).
enum {
	RT_INDEX_FORMAT_UINT16 = 0,
	RT_INDEX_FORMAT_UINT32 = 1,
	RT_INDEX_FORMAT_NONE = 2,
};

enum {
	RT_GEOM_FLAG_COMPRESSED = 1u,
	RT_GEOM_FLAG_PROCEDURAL = 2u,
	// Set when the BLAS uses a per-frame-deformed vertex buffer.
	RT_GEOM_FLAG_DEFORMED = 4u,
	// Geometry residency (D93): the surface's CLAS carry their cluster's index as their ID, and a hit
	// reads its triangle from the cluster's blob through the records at cluster_record_base.
	RT_GEOM_FLAG_CLUSTER_GEOMETRY = 8u,
};

/// Per-instance state for procedural RT geometry. Heap-allocated, only exists for procedural instances.
struct RTProceduralState {
	AABB culling_aabb;
	PackedFloat32Array aabb_data; // N * 6 floats (min/max per AABB). Empty = single AABB.
	bool expose_bounds = false;
	bool dirty = true;
	RID blas;
	RID gpu_buffer;
	uint32_t gpu_buffer_capacity = 0; // Bytes, grow-only.
	uint64_t gpu_buffer_address = 0; // BDA (0 = not exposed).
	uint32_t aabb_count = 0;
};

// The selection's spatial hierarchy over a hierarchy's groups (D91), in its region of the LOD table:
// a tree of bounding spheres per depth under one root, each node enclosing its children's spheres
// and carrying their largest error, so a node whose test fails has no group under it that passes.
// Nodes are eight words (center, radius, error, first child, child count with bit 31 set when the
// children are groups, a reserved word) in breadth-first order, the root first, a node's children
// contiguous; a group child is reached through the refs, one word per group. The node count of each
// breadth-first level sizes the frame's queues.
struct RTLodTree {
	uint32_t nodes_offset = UINT32_MAX; // Words into the table; UINT32_MAX without a tree.
	uint32_t refs_offset = 0; // Words into the table.
	uint32_t group_count = 0; // The group queue's share of an item.
	LocalVector<uint32_t> level_nodes; // Nodes per breadth-first level, the root's level first.
};

// One depth of a cluster LOD hierarchy as the instance classification reads it (D89): over the
// depth's groups, a sphere around them all (its center, the radius enclosing every group's
// sphere, the farthest group center), the smallest group radius and the largest and smallest
// group error, in mesh space. A depth with no group has no bound.
struct RTLodLevelBound {
	Vector3 center;
	float enclosing_radius = 0.0f;
	float center_spread = 0.0f;
	float min_radius = 0.0f;
	float max_error = 0.0f;
	float min_error = 0.0f;
	bool empty = true;
	uint32_t cut_clusters = 0; // The clusters of the depth's discrete level, a cached build's exact count (D90).
};

// A discrete level's cluster BLAS (D90): every cluster of one depth's groups, with those of
// terminal groups above it, the whole surface at that depth. It is built once, fast-trace, into
// its own arena slot, and traced by any instance whose selection it covers within the tolerance;
// never rebuilt, so no frame's build writes what an earlier frame's trace may still read. Released
// on the slots' TTL unused or with the surface's hierarchy; none is made while the surface's
// compaction is pending.
struct RTLodCachedLevel {
	RID blas;
	uint64_t blas_address = 0;
	int32_t arena_chunk = -1;
	uint64_t arena_offset = 0;
	uint64_t arena_size = 0;
	uint32_t last_used_frame = 0;
};

// CLAS residency (D92): a streamed surface's groups, the unit its CLAS are loaded and unloaded in.
// A persistent group (a terminal one, the coarsest of its part of the surface) is built with the
// surface and never leaves; any other is absent, loading (its geometry gathered on the worker threads
// under geometry streaming, D93c, then its CLAS built into a load's staging, its sizes in flight) or
// resident at its own arena region. A group's parents are the groups holding a
// cluster simplified from it, its children the groups its clusters were simplified from; a group
// loads only with its parents resident or loading, and unloads only with no child resident or
// loading (live_children), so residency stays monotone along the DAG.
struct RTLodStreamGroup {
	enum State : uint8_t {
		ABSENT,
		LOADING,
		RESIDENT,
		PERSISTENT,
	};
	uint32_t first_cluster = 0;
	uint32_t cluster_count = 0;
	uint32_t depth = 0;
	uint32_t parents_first = 0; // Into RTLodStream::links.
	uint32_t parents_count = 0;
	uint32_t children_first = 0;
	uint32_t children_count = 0;
	uint32_t live_children = 0;
	State state = ABSENT;
	uint64_t last_used = 0; // frames_drawn of the last reported use, or of the load.
	uint64_t requested = 0; // frames_drawn of the first request since it was absent, 0 for none.
	uint64_t bytes = 0; // Its region's bytes once loaded; until then estimated at the worst case.
	uint64_t loading_bytes = 0; // What its load holds against the budget while in flight.
	// Geometry residency (D93c), 0 when the surface reads its own buffers: its blob's bytes, which
	// load and unload with its CLAS under the same budget, and its clusters' positions, staged for
	// their build only; both count against a frame's upload cap.
	uint32_t blob_bytes = 0;
	uint32_t position_bytes = 0;
	int32_t arena_chunk = -1;
	uint64_t arena_offset = 0;
};

// A streamed surface's residency (D92): what a load needs to build its groups' CLAS again (the
// buffers, each cluster's first triangle and counts), the groups and their links, the persistent
// clusters as runs in build order, and per depth the groups that stream and how many are resident,
// which says when a cached level (D90) may be built; a placed level stamps its depth's use, which
// keeps the groups it holds. A surface an emissive instance or an expanded MultiMesh references is
// pinned: loaded whole, outside the budget, and traced at its cached level 0 once that is built.
struct RTLodStream {
	RID vertex_buffer;
	RID index_buffer;
	void *mesh_surface = nullptr; // Its RAM copy is what a load gathers a group's geometry from (D93c).
	uint32_t state_offset = 0; // Words into the LOD table: RT_LOD_STREAM_STATE_WORDS per group, the selection's state.
	uint32_t max_triangles = 0;
	uint32_t max_vertices = 0;
	LocalVector<uint32_t> cluster_first_triangle;
	LocalVector<uint32_t> cluster_packed; // Triangles | vertices << 16.
	LocalVector<RTLodStreamGroup> groups;
	LocalVector<uint32_t> links;
	LocalVector<uint32_t> persistent_runs; // (first cluster, count) pairs, the population's build order.
	uint32_t persistent_clusters = 0;
	uint32_t persistent_triangles = 0;
	uint32_t persistent_vertices = 0;
	LocalVector<uint32_t> depth_streamed;
	LocalVector<uint32_t> depth_resident;
	LocalVector<uint64_t> level_last_used; // frames_drawn a cached level of the depth was placed.
	uint32_t absent_groups = 0;
	uint32_t root_groups = 0; // Not persistent, refined into nothing: requested by their own test.
	uint64_t resident_bytes = 0; // With the loaded groups' blobs (D93c).
	uint64_t loading_bytes = 0;
	uint64_t persistent_bytes = 0;
	uint64_t resident_blob_bytes = 0; // The part of resident_bytes that is blobs (D93c).
	uint64_t persistent_blob_bytes = 0;
	uint64_t pinned_frame = 0; // frames_drawn it was last referenced as pinned.
	// A cached level placement asked for and found not resident: its depth (the finest asked that
	// frame) and the frame, which the next residency step turns into wants for every absent group
	// of that depth and coarser.
	uint32_t level_wanted = UINT32_MAX;
	uint64_t level_wanted_frame = 0;
};

// A region of a cluster arena (D81), as a cluster geometry blob holds one (D93b).
struct RTClusterArenaRegion {
	int32_t chunk = -1;
	uint64_t offset = 0;
	uint64_t size = 0;
};

struct RTSurfaceData {
	RID blas;
	RT_GeometryData geometry = {};
	Transform3D aabb_transform;
	bool is_compressed = false;
	uint64_t blas_size = 0;
	// A cluster BLAS (D79) is first built into buffers this surface owns, the staging storage and
	// the build's arguments, and `blas` is a handle into the staging. With
	// rendering/pathtracing/cluster_blas_compaction on (D81), once its sizes are read back the CLAS
	// move into the arena's region below, the BLAS is rebuilt there, `blas` becomes a handle into
	// the chunk and the two buffers are freed; off, the staging is kept for life.
	RID cluster_storage;
	RID cluster_args;
	uint32_t cluster_count = 0;
	uint32_t cluster_lod0_count = 0; // The clusters the BLAS is built over (D85), all of them without a hierarchy.
	uint32_t cluster_cache_key = 0;
	uint64_t cluster_blas_worst_size = 0; // The size query's worst case for the one BLAS, its slot in the region.
	uint64_t cluster_clas_region_size = 0; // The implicit build's region for every CLAS, the move's byte budget.
	uint64_t cluster_args_implicit_offset = 0; // The addresses the implicit build wrote, the move's sources.
	uint64_t cluster_args_compact_offset = 0; // The addresses the move writes to, the rebuild's references.
	uint64_t cluster_args_blas_info_offset = 0; // The BLAS build info over the compact addresses.
	uint64_t compaction_ticket = 0; // The sizes readback in flight, 0 for none.
	int32_t arena_chunk = -1;
	uint64_t arena_offset = 0;
	uint64_t arena_size = 0;
	// The cluster LOD hierarchy on the GPU (D86): the surface's region of the shared LOD table
	// (its CLAS addresses over every level, kept current across the compaction; each cluster's
	// group and the group it refines; the groups' bounds and errors), which the selection reads,
	// and the worst case of a BLAS over any selection, the size of an instance's slot.
	uint32_t cluster_lod_table_offset = UINT32_MAX; // Words; UINT32_MAX without a hierarchy.
	uint32_t cluster_lod_table_words = 0;
	uint64_t cluster_lod_blas_worst_size = 0;
	uint32_t cluster_lod_generation = 0; // Unique per population; an instance's slot follows it.
	LocalVector<RTLodLevelBound> cluster_lod_levels; // Per depth, the classification's bounds (D89); empty, never shared.
	LocalVector<RTLodCachedLevel> cluster_lod_cache; // Per depth, its cached level if any (D90); empty until one is.
	RTLodTree cluster_lod_tree; // The traversal's tree over the groups (D91), in the same region.
	// Streamed (D92): the population built only the persistent groups' CLAS, `blas` is over them
	// (the coarsest cut, the surface's fallback), and the rest load and unload with the camera.
	RTLodStream *cluster_lod_stream = nullptr;
	uint32_t cluster_built_count = 0; // The CLAS the population built and the compaction moves.
	uint32_t cluster_blas_count = 0; // The clusters `blas` is over.
	// Geometry residency (D93b): a streamed surface whose CLAS carry their cluster's index as their
	// ID. Its hits read their triangle through RT_CLUSTER_GEOMETRY_RECORD_WORDS words a cluster in
	// the LOD table, from this offset, in the blob of the cluster's group, one region of the cluster
	// geometry arena per group. A group's blob exists while the group is persistent, resident, or
	// loading past its gather (D93c), and its records are zero but while it is persistent or resident.
	uint32_t cluster_geometry_offset = UINT32_MAX; // Words; UINT32_MAX when the hits read the surface's own buffers.
	uint32_t cluster_geometry_words = 0;
	LocalVector<RTClusterArenaRegion> cluster_geometry_blobs; // Per group; chunk -1 while it has none.
	// Geometry residency (D93d): the mesh storage streams this surface's geometry, so its own buffers
	// hold the fallback level the raster draws. The trace reads the surface through its blobs, and
	// `geometry` addresses no buffer; where it must read the whole surface's buffers (its population
	// could not stream it, an emissive material samples its triangles, a MultiMesh merge bakes from
	// it), they are uploaded from the mesh storage's RAM copy into buffers of its own, which
	// `geometry` then addresses. Held for the population's life, or released past the minimum age
	// without use.
	bool geometry_streamed = false;
	bool materialized_held = false;
	RID materialized_vertex_buffer;
	RID materialized_attribute_buffer;
	RID materialized_index_buffer;
	uint64_t materialized_bytes = 0;
	uint64_t materialized_frame = 0;
	// The row a slot-resident instance last saw (D94b): `geometry` as of `row_generation`, which
	// advances, unique over every row, whenever the walk finds `geometry` changed since.
	RT_GeometryData row_snapshot = {};
	uint64_t row_generation = 0;
};

// A surface's geometry metadata from its format, and the buffers a BLAS of it reads.
void _fill_surface_geometry_data(void *p_mesh_surface, bool p_force_uncompressed, RTSurfaceData *r_surf_data, RID *r_vertex_buffer = nullptr, RID *r_attribute_buffer = nullptr, RID *r_index_buffer = nullptr);
// The triangle geometry of a surface's own BLAS over its buffers, as _populate_surface_blas builds it.
void _fill_surface_triangle_geometry(RID p_vertex_buffer, RID p_index_buffer, const RT_GeometryData &p_geometry, bool p_2d, bool p_compressed, RD::AccelerationStructureGeometry &r_geometry);

// An instance's cluster BLAS over the clusters the LOD selection picks for it (D86): its own
// worst-case slot in the arena, rebuilt on every frame the instance enters the TLAS from the
// surface's current CLAS. Reached through the surface cache's rt_lod_handle and TTL-evicted with
// the deformed entries; re-sized when the surface it was made for is populated again.
struct RTLodInstanceEntry {
	RTSurfaceData *ptr = nullptr;
	uint32_t generation = 0;
	uint32_t last_used_frame = 0;
	uint64_t pushed_frame = 0; // frames_drawn when it last entered the TLAS; a second viewport in the frame traces frame_blas.
	RID frame_blas; // What it traced at pushed_frame: its own BLAS, or the one it shares (D89).
	uint32_t last_built_frame = 0; // An instance tracing another's build gives its slot back on the TTL (D89).
	RID blas;
	uint64_t blas_address = 0;
	int32_t arena_chunk = -1;
	uint64_t arena_offset = 0;
	uint64_t arena_size = 0;
};

struct RTDeformedCacheEntry;

// The selection's instance record, in words. Must match INSTANCE_WORDS in cluster_lod_select.glsl.
static constexpr uint32_t RT_LOD_INSTANCE_WORDS = 24;

// CLAS residency's lists (D92): the requests and the use reports one frame's selection may append,
// and the frames between two reports of one group's use. Must match STREAM_* in cluster_lod_select.glsl.
static constexpr uint32_t RT_LOD_STREAM_REQUEST_CAP = 8192;
static constexpr uint32_t RT_LOD_STREAM_TOUCH_CAP = 16384;
static constexpr uint32_t RT_LOD_STREAM_TOUCH_PERIOD = 8;
// A group's state words in the table: requested frame, reported frame (the selection's atomics),
// flags the CPU writes once. Must match STATE_WORDS and STATE_FLAG_ROOT.
static constexpr uint32_t RT_LOD_STREAM_STATE_WORDS = 3;
static constexpr uint32_t RT_LOD_STREAM_STATE_FLAG_ROOT = 1;
// The position-fetch check's words (D93a): hits compared, hits that differ and the largest absolute
// difference (float bits), for float positions, compressed positions and deformed surfaces; then
// (D93b) cluster geometry hits whose blob was compared with the surface's buffers and those with a
// word that differs.
static constexpr uint32_t RT_POSITION_FETCH_CHECK_WORDS = 11;
// A cluster geometry record (D93b): the cluster's blob address (two words), its vertex count |
// triangle count << 16, its first triangle in [LOD 0 | coarse tail]. Must match the GLSL.
static constexpr uint32_t RT_CLUSTER_GEOMETRY_RECORD_WORDS = 4;
// A blob's sections and its clusters start at this alignment. Must match the GLSL.
static constexpr uint32_t RT_CLUSTER_GEOMETRY_ALIGNMENT = 16;
// Blob indices are 8 bits, cluster-local.
static constexpr uint32_t RT_CLUSTER_GEOMETRY_MAX_VERTICES = MESH_CLUSTER_GEOMETRY_MAX_VERTICES;

// A load of streamed groups (D92), D81's two steps at group granularity: this frame the groups'
// CLAS are built with implicit destinations into a staging buffer, their sizes read back; when they
// are, each group gets its exact region of the arena, the CLAS move there and the group's addresses
// go to the LOD table. Items whose surface went in the meantime have a null surface.
struct RTLodStreamLoad {
	struct Item {
		RTSurfaceData *surface = nullptr;
		uint32_t group = 0;
		uint32_t first = 0; // Its first CLAS in the load.
		uint32_t gather = UINT32_MAX; // Its item in the gather that staged its geometry (D93c), if any.
	};
	uint64_t ticket = 0;
	RID staging;
	RID args;
	uint64_t region_size = 0; // The implicit build's region, the move's byte budget.
	uint64_t implicit_offset = 0;
	uint64_t compact_offset = 0;
	uint32_t cluster_count = 0;
	LocalVector<Item> items;
};

// A gather of streamed groups' geometry (D93c), self-contained so that its items can run on worker
// threads while the render thread goes on: each group's blob and its clusters' positions (in the
// blob's vertex order, at the surface's position stride, RT_CLUSTER_GEOMETRY_ALIGNMENT apart) go to
// slices of `blobs` and `positions` the render thread laid out from the clusters' counts, read from
// refcounted copies of the surfaces' RAM arrays, which an update or a release on the render thread
// copies or frees without touching. An item whose indices do not match its counts is marked failed.
struct RTClusterGeometryGather {
	struct Source {
		Vector<uint8_t> vertex_data;
		Vector<uint8_t> attribute_data;
		Vector<uint8_t> index_data;
		uint32_t vertex_count = 0;
		uint32_t position_stride = 0;
		uint32_t normal_byte_offset = 0;
		uint32_t normal_source_stride = 0;
		uint32_t normal_stride = 0; // In the blob.
		uint32_t attribute_stride = 0;
		bool has_normal = false;
		bool compressed = false;
		bool index_16 = false;
	};
	struct Item {
		uint32_t source = 0;
		uint32_t first_cluster = 0; // Into the per-cluster arrays below.
		uint32_t cluster_count = 0;
		uint64_t blob_offset = 0; // Its blob's slice of `blobs`.
		uint64_t blob_size = 0;
	};
	LocalVector<Source> sources;
	LocalVector<Item> items;
	LocalVector<uint32_t> cluster_packed; // Triangles | vertices << 16, as the stream's.
	LocalVector<uint32_t> cluster_first_triangle;
	LocalVector<uint64_t> cluster_at; // Its section in `blobs`.
	LocalVector<uint64_t> positions_at; // Its positions in `positions`.
	LocalVector<uint8_t> blobs;
	LocalVector<uint8_t> positions;
	LocalVector<uint8_t> item_ok;

	void gather_item(uint32_t p_item, void *p_unused);
};

// A load's gather in flight on the worker threads (D93c): its build is issued RT_LOD_STREAM_GATHER_LATENCY
// frames after its admission, the render thread waiting for it then only if it is still running.
// Items whose surface went in the meantime have a null surface.
struct RTLodStreamGatherJob {
	LocalVector<RTLodStreamLoad::Item> items;
	bool compressed = false;
	uint64_t frame = 0;
	RTClusterGeometryGather gather;
	WorkerThreadPool::GroupID group = -1;
};

// Frames from a load's admission to its build when its geometry is gathered on the worker threads
// (D93c): fixed, so the residency history never depends on the threads' scheduling; two leave a
// gather a whole frame of slack behind the other low-priority jobs.
static constexpr uint64_t RT_LOD_STREAM_GATHER_LATENCY = 2;

// What a build of streamed groups' CLAS reads in place of the surface's buffers (D93c): per cluster
// of the build, its 8-bit indices' address in its group's blob (0 for a surface that reads its own
// buffers) and its positions' offset in the gather's positions; the arena chunks the indices are
// in. The positions go up in a buffer of the build's own, freed with the stage once the build is
// recorded.
struct RTClusterGeometryStage {
	LocalVector<uint64_t> position_offsets;
	LocalVector<uint64_t> index_addresses;
	Vector<RID> blob_buffers;
	uint64_t blob_bytes = 0;
	RID positions_buffer;
	uint64_t positions_address = 0;

	RTClusterGeometryStage() = default;
	RTClusterGeometryStage(const RTClusterGeometryStage &) = delete;
	RTClusterGeometryStage &operator=(const RTClusterGeometryStage &) = delete;
	~RTClusterGeometryStage();
};

struct RTLodStreamRead {
	uint64_t frame = 0;
	uint32_t part = 0; // 0: the requests, 1: the use reports; each its count, then records from word 4.
	PackedByteArray bytes;
};

// The frame's CLAS region for the deformed items (D87) stays under this many bytes: the argument
// and region sizes are 32-bit, and a flush that cannot record turns the selection off for the
// run, so past it the rest of the frame's deformed instances rebuild at LOD 0 through the
// deformed batch. A VRAM budget (6d) replaces it.
static constexpr uint64_t RT_LOD_CLAS_REGION_CEILING = uint64_t(1) << 30;

// The cluster LOD selection of one frame (D86): every instance on the path appends itself; the
// flush, in build_tlas after the MultiMesh merge's compute list, uploads one argument buffer (the
// build infos with their counts at zero, the slot addresses, the instance records, room for the
// reference lists and, for the deformed items, the CLAS count, destinations and build infos, D87),
// runs the selection compute over it, records one CLAS build for the deformed items' selections
// and then one cluster BLAS build over every instance's list, and a second, fast-trace, over the
// cached levels built this frame (D90). Every list is grow-only across frames.
struct RTLodSelectBatch {
	// The static instances of the frame before the sharing resolves them (D89): each one's slot,
	// the surface its slot is for, its trace-space transform, and the depth range its selection
	// could fall in, classified at the resolve.
	struct Candidate {
		RTLodInstanceEntry *entry = nullptr;
		RTSurfaceData *surface = nullptr;
		Transform3D transform;
		uint32_t slot = 0;
		uint32_t level_min = 0;
		uint32_t level_max = 0;
	};
	LocalVector<Candidate> candidates;
	// The instances that trace another's build this frame (D89): their slots and surfaces, for the
	// fallback, and how many there were, for the report, with those on the surface's LOD 0 BLAS.
	LocalVector<uint32_t> shared_slots;
	LocalVector<RTSurfaceData *> shared_surfaces;
	uint32_t shared_count = 0;
	uint32_t lod0_count = 0;
	// The resolve's scratch, grow-only: each surface's first slot in the stand-in table (one slot
	// per depth), the table, and per candidate what it traces and whether it builds.
	AHashMap<RTSurfaceData *, uint32_t> stand_in_bases;
	LocalVector<uint32_t> stand_ins;
	LocalVector<uint32_t> targets;
	LocalVector<uint8_t> builds;
	// The cached levels (D90): per candidate the depth it traces, the frame's requests for a level
	// not yet cached (sorted per surface by their coarsest admissible depth), and the levels this
	// frame builds, appended after every other item so they form their own fast-trace build.
	static constexpr uint32_t TARGET_CACHED = UINT32_MAX - 2; // In targets, beside the resolve's own two.
	struct CacheRequest {
		uint32_t generation = 0; // Its surface's population, the sort's first key.
		uint32_t highest = 0; // Its finest depth, the coarsest it may trace.
		uint32_t lowest = 0; // The finest depth within the tolerance of its coarsest.
		uint32_t candidate = 0;
		bool operator<(const CacheRequest &p_other) const {
			if (generation != p_other.generation) {
				return generation < p_other.generation;
			}
			if (highest != p_other.highest) {
				return highest < p_other.highest;
			}
			return candidate < p_other.candidate;
		}
	};
	struct CacheBuild {
		RTSurfaceData *surface = nullptr;
		uint32_t depth = 0;
	};
	LocalVector<uint32_t> cached_depths;
	LocalVector<CacheRequest> cache_requests;
	LocalVector<CacheBuild> cache_builds;
	uint32_t cached_count = 0; // Instances tracing a cached level this frame.
	uint32_t cache_first = 0; // The first cached build's item.
	uint32_t cache_list_clusters = 0; // Their reference lists' clusters, outside the per-frame build.
	uint32_t cache_total_clusters = 0; // Their exact counts, summed.
	uint32_t cache_max_clusters = 0;
	// Per item, a static instance's slot or a deformed entry (D87), the other null.
	LocalVector<RTLodInstanceEntry *> entries;
	LocalVector<RTDeformedCacheEntry *> deformed_entries;
	LocalVector<uint32_t> slots; // Each entry's index in the per-frame instance arrays, for the fallback.
	LocalVector<uint32_t> instance_words; // RT_LOD_INSTANCE_WORDS words per entry, the shader's instance record; the list offset and a deformed entry's destination base patched at the flush.
	LocalVector<uint32_t> lod0_clusters; // Per entry, what LOD 0 would trace, for the report.
	LocalVector<uint64_t> blas_addresses;
	uint32_t cluster_total = 0;
	uint32_t max_clusters = 0;
	// The traversal (D91): the items that walk their tree rather than test every cluster, the
	// largest cluster count of the others (the exhaustive dispatch's width), and the frame's queue
	// capacity per breadth-first level, the sum of the tree items' node counts there.
	uint32_t tree_count = 0;
	uint32_t flat_max_clusters = 0;
	LocalVector<uint32_t> tree_level_capacity;
	uint32_t tree_group_capacity = 0; // The tree items' groups, the group queue's capacity.
	uint32_t stream_items = 0; // Items of streamed surfaces (D92): the frame reads its requests back.
	RID dispatch_buffer; // The cluster pass's indirect arguments, which the leaves write.
	LocalVector<RID> storage_buffers; // The arena chunks the slots live in, each once, and each deformed entry's storage.
	LocalVector<RID> source_buffers; // The buffers the surfaces' CLAS live in, each once.
	LocalVector<RID> blas_handles;
	// The deformed items' CLAS, built this frame from their lists (D87): the count bound (their
	// clusters over every level), the build's totals, the vertex and index buffers it reads.
	uint32_t deformed_count = 0;
	uint32_t deformed_routed = 0; // Deformed instances the ceiling sent to the deformed batch this frame.
	uint32_t clas_bound = 0;
	uint32_t clas_total_triangles = 0;
	uint32_t clas_total_vertices = 0;
	LocalVector<RID> clas_source_buffers;
	LocalVector<RID> clas_index_buffers_seen;
	RID args_buffer;
	uint32_t args_capacity = 0; // Bytes.
	LocalVector<uint8_t> args_host;
	uint64_t frames_flushed = 0;
};

// One chunk of the static cluster arena (D81): a storage buffer sub-allocated first fit from a
// free list kept sorted by offset and merged on free. A dead chunk keeps its slot with a null
// buffer so the indices surfaces hold stay valid.
struct RTClusterArenaChunk {
	struct FreeRange {
		uint64_t offset = 0;
		uint64_t size = 0;
	};
	RID buffer;
	uint64_t address = 0; // The aligned base inside the buffer.
	uint64_t base_offset = 0; // Its offset from the buffer's start.
	uint64_t size = 0; // Usable bytes from the base.
	uint32_t live_regions = 0;
	LocalVector<FreeRange> free_list;
};

struct RTClusterCompactionResult {
	uint64_t ticket = 0;
	PackedByteArray sizes;
};

// A region whose surface is gone but whose chunk the frames in flight may still trace: freed
// once the frame queue has turned over, as RenderingDevice defers its own frees.
struct RTClusterRetiredRegion {
	uint32_t chunk = 0;
	uint64_t offset = 0;
	uint64_t size = 0;
	uint64_t frame = 0;
};

/// Inputs for a surface backed by a per-frame-deformed vertex buffer.
struct RTDeformedGeometrySource {
	RID current_vb; ///< Vertex buffer (object-space positions) the BLAS is built / refit against.
	RID prev_vb; ///< Previous-frame positions for motion vectors. Optional.
	uint64_t change_stamp = 0; ///< Changes when `current_vb` contents change.
	uint64_t cache_key = 0; ///< Caller-defined cache identity.
	uint32_t cache_version = 0; ///< Changes when the resource behind `cache_key` is recycled.
	uint32_t surface_counter = 0; ///< Changes when the underlying mesh surface changes.
};

struct RTMaterialData {
	alignas(16) RT_MaterialData data = {};
	uint32_t global_buffer_index = UINT32_MAX;
	uint32_t rt_sbt_offset = 0;
	bool is_custom_shader = false;
	bool omm_custom = false; // A custom shader with render_mode opacity_micromap (D96).
	RID uniform_buffer; // Buffer pointer for mats > 512 bytes.
	uint32_t uniform_pool_slot = UINT32_MAX; // Index into the material UBO pool, or UINT32_MAX (unused)
	RID albedo_texture_rd;
	RID normal_texture_rd;
	RID orm_texture_rd;
	RID emission_texture_rd;
	// As RTSurfaceData's (D94b): `data` as of `row_generation`.
	RT_MaterialData row_snapshot = {};
	uint64_t row_generation = 0;
};

struct RTCacheEntry {
	RTSurfaceData *ptr = nullptr;
	uint32_t last_used_frame = 0;
	uint32_t cached_counter = 0;
	uint32_t cached_rid_version = 0;
	uint8_t failed_attempts = 0;
	uint64_t size_bytes = 0;
};

/// BLAS cache entry for a surface driven by a per-frame-deformed vertex buffer.
///
/// The skinned vertex buffer is owned by the engine's MeshInstance system and may
/// be freed/reallocated outside our control. To decouple BLAS lifetime from that
/// (and keep last-frame positions alive for motion vectors) we copy the skinned VB
/// into our own buffers each frame:
///   * owned_vb_full mirrors the skinned VB layout (positions + N + T) and is what
///     the BLAS and the hit shader read from this frame.
///   * prev_pos_vb stores the previous-frame positions only (float3 packed) for the
///     motion-vector path. Updated from owned_vb_full's position section each frame
///     before owned_vb_full is overwritten with the new skinned data.
struct RTDeformedCacheEntry {
	RTSurfaceData *ptr = nullptr;
	uint32_t last_used_frame = 0;
	uint64_t cached_change_stamp = 0;
	uint32_t cached_key_version = 0;
	uint32_t cached_surface_counter = 0;
	uint64_t cached_buffer_id = 0; // RID id of the deformed vertex buffer at the time of build.
	bool blas_built_once = false; // True once the BLAS has been fully built; subsequent ticks can refit.

	RID owned_vb_full;
	uint32_t owned_vb_full_capacity = 0; // Bytes; grow-only.
	RID prev_pos_vb;
	uint32_t prev_pos_vb_capacity = 0; // Bytes; grow-only.
	uint32_t cached_vertex_count = 0;
	uint32_t cached_full_size = 0;
	bool prev_pos_seeded = false; // False until the first owned->prev copy has run.
	// The opacity micromap the triangle BLAS carries (D96 OMMc): the surface's, as the instance's
	// walk wants it for its alpha-tested material. 0 for none; the walk builds the BLAS again when
	// it wants another, and any other rebuild takes the last one wanted. A micromap the BLAS could
	// not carry is not asked for again.
	uint64_t omm_serial = 0;
	uint64_t omm_wanted_serial = 0;
	uint64_t omm_failed_serial = 0;

	// The cluster path (D82): the entry owns the storage of its cluster BLAS alone (one worst-case
	// slot at the aligned base); its CLAS are built on every frame the data changes, from
	// owned_vb_full, with implicit destinations into the frame's region shared with every other
	// deformed surface of the frame (RTDeformedClusterBatch), and the BLAS is rebuilt over them.
	// The build infos are written once, when the layout is made, and appended to the batch each rebuild.
	enum ClusterQueuedKind : uint8_t {
		CLUSTER_QUEUED_NONE = 0,
		CLUSTER_QUEUED_FIRST_BUILD, // The layout's first build, a build in the counters.
		CLUSTER_QUEUED_CHANGED, // The data changed, the refit it replaces in the counters.
		CLUSTER_QUEUED_EXPIRY, // Unchanged, rebuilt because its region is due for reuse; counted on its own.
	};
	RID cluster_storage;
	uint64_t cluster_storage_capacity = 0; // Bytes; grow-only.
	uint64_t cluster_blas_address = 0; // The BLAS slot, the handle's address.
	uint32_t cluster_count = 0;
	uint32_t cluster_max_triangles = 0; // The layout's largest cluster, for the batch's maxima.
	uint32_t cluster_max_vertices = 0;
	uint32_t cluster_total_vertices = 0;
	uint64_t cluster_built_frame = 0; // RenderingDevice's frames_drawn at the flush whose region holds the CLAS; 0 for none.
	uint64_t cluster_pushed_frame = 0; // frames_drawn when the surface last entered the TLAS instance list.
	uint8_t cluster_frame_kind = CLUSTER_QUEUED_NONE; // Why it is in this frame's batch.
	LocalVector<uint8_t> cluster_infos; // cluster_count driver build infos.
	RID cluster_index_buffer; // The mesh's index buffer the infos read, listed as a source at each rebuild.
	// The cluster LOD hierarchy on the GPU (D87): the entry's region of the shared LOD table
	// (each cluster's group and the group it refines, the groups' bounds and errors, and every
	// level's CLAS build info, since a deformed hierarchy has no CLAS until a frame builds the
	// selected ones), the count over every level (a selection's bound, the BLAS storage's size)
	// and the build's totals. While the region exists the selection, not the batch above,
	// rebuilds the entry on every frame it is traced.
	uint32_t cluster_lod_table_offset = UINT32_MAX; // Words; UINT32_MAX without a hierarchy.
	uint32_t cluster_lod_table_words = 0;
	uint32_t cluster_lod_infos_offset = 0; // Words, into the table.
	RTLodTree cluster_lod_tree; // The traversal's tree over the groups (D91), in the same region.
	uint32_t cluster_lod_count = 0; // Clusters over every level.
	uint32_t cluster_lod_total_triangles = 0;
	uint32_t cluster_lod_total_vertices = 0;
	uint64_t cluster_lod_pushed_frame = 0; // frames_drawn when the selection last built it; a second viewport in the frame reuses that build.
	uint8_t cluster_lod_pending_kind = CLUSTER_QUEUED_NONE; // What the batch above would have queued this frame, decided at the TLAS push.
	bool cluster_lod_pending = false; // Set by process_deformed_surface on a selecting frame, taken at the TLAS push.
	bool cluster_lod_last = false; // The last build was a selection: the batch above rebuilds it before tracing it again.
};

// One region of the deformed cluster batch (D82): the CLAS of one flush, sub-allocated by the
// driver, reused once the last frame that could trace it has been waited on
// (RenderRaytracing::_deformed_cluster_region_expired).
struct RTDeformedClusterRegion {
	RID buffer;
	uint64_t capacity = 0; // Bytes; grow-only.
	uint64_t base_offset = 0; // The aligned base's offset from the buffer's start.
	uint64_t last_used_frame = 0; // RenderingDevice's frames_drawn at the flush that wrote it.
};

// The deformed cluster builds of one frame (D82): every deformed surface whose data changed
// appends its cached CLAS infos and one bottom-level info over its slice of the address array
// the CLAS build writes; the flush, in build_tlas after the MultiMesh merge's compute list and
// before the counters are published, records two builds, every CLAS into the frame's region
// then every cluster BLAS into its own storage, over one argument buffer uploaded once per
// frame. Every list is grow-only across frames; the flush's allocations are the three Vectors
// and the one-element source list the RenderingDevice build takes, and that build's own map,
// lists and scratch.
struct RTDeformedClusterBatch {
	LocalVector<RTDeformedCacheEntry *> entries; // The queued entries, stamped once the builds record.
	LocalVector<uint8_t> clas_infos;
	LocalVector<uint64_t> blas_addresses;
	uint32_t cluster_count = 0;
	LocalVector<uint32_t> entry_cluster_counts;
	LocalVector<RID> storage_buffers;
	LocalVector<RID> source_buffers; // Each entry's owned vertex buffer, and each distinct index buffer once.
	LocalVector<RID> index_buffers_seen;
	LocalVector<RID> blas_handles;
	uint32_t total_triangles = 0;
	uint32_t total_vertices = 0;
	uint32_t max_cluster_triangles = 0;
	uint32_t max_cluster_vertices = 0;
	uint32_t max_clusters_per_blas = 0;
	// One argument buffer, grow-only: the graph orders each frame's upload after the previous
	// frame's builds through the buffer's tracker.
	RID args_buffer;
	uint32_t args_capacity = 0; // Bytes.
	LocalVector<uint8_t> args_host; // The upload's host copy; grow-only.
	LocalVector<RTDeformedClusterRegion> regions;
	LocalVector<RID> dropped_handles; // Handles a failed flush left without a build, to leave this frame's TLAS.
	uint64_t frames_flushed = 0;
};

/// Cache entry for a per-(MultiMesh, surface) merged BLAS.
/// All vertex data (positions, normals, tangents, UVs, colors) is fully baked per-instance
/// so the hit shader uses the standard code path — no special per-instance lookups.
struct RTMergedMMEntry {
	// Merged vertex buffer: [float3 pos × N*V] + [packed TBN × N*V] (if mesh has normals).
	// The BLAS reads only the position section; the hit shader reads TBN via normal_byte_offset.
	RID merged_vtx_buffer;
	uint32_t vtx_capacity_bytes = 0;

	// Merged attribute buffer: [UV + color × N*V] replicated per instance.
	RID merged_attr_buffer;
	uint32_t attr_capacity_bytes = 0;

	// Replicated index buffer: uint32 N*I entries (invalid if non-indexed).
	RID replicated_idx_buffer;
	uint32_t idx_capacity = 0;

	RID blas;
	uint32_t last_mm_count = 0;
	uint32_t last_surface_counter = 0;
	uint32_t last_used_frame = 0;
	uint64_t cached_mm_last_change = 0;
	bool blas_built_once = false;
	bool indexed = false; // selects MODE_INDEXED vs MODE_NON_INDEXED variant

	// The opacity micromap the merge's BLAS carries (D96): the base surface's, its index repeated
	// per element. 0 for none; the BLAS is made again when it changes. A micromap the merge
	// could not carry is not asked for again.
	uint64_t omm_serial = 0;
	uint64_t omm_failed_serial = 0;
	RID omm_index_buffer;
	// The serial the frame's latest walk wanted, and the last frame two walks disagreed.
	uint64_t omm_wanted_serial = 0;
	uint64_t omm_wanted_frame = UINT64_MAX;
	uint64_t omm_conflict_frame = UINT64_MAX;
};

// Opacity micromaps (D96, docs/r1-opacity-micromaps.md): one per surface and the alpha test's
// inputs, baked on the GPU in two passes with a readback between them and attached to a BLAS of
// the surface's own. Exact for the standard hit group's alpha test, so the image does not change.
struct RTOmmKey {
	uint64_t mesh_id = 0;
	uint32_t surface_index = 0;
	uint32_t surface_counter = 0; // The surface's RT invalidation counter.
	uint64_t texture_id = 0; // The RD texture the trace samples for alpha.
	uint32_t uv_scale[2] = {}; // Float bits.
	uint32_t uv_offset[2] = {};
	uint32_t albedo_alpha = 0;
	uint32_t threshold = 0;

	bool operator==(const RTOmmKey &p_other) const {
		return mesh_id == p_other.mesh_id && surface_index == p_other.surface_index && surface_counter == p_other.surface_counter && texture_id == p_other.texture_id && uv_scale[0] == p_other.uv_scale[0] && uv_scale[1] == p_other.uv_scale[1] && uv_offset[0] == p_other.uv_offset[0] && uv_offset[1] == p_other.uv_offset[1] && albedo_alpha == p_other.albedo_alpha && threshold == p_other.threshold;
	}
	static uint32_t hash(const RTOmmKey &p_key) {
		uint32_t h = hash_murmur3_one_64(p_key.mesh_id);
		h = hash_murmur3_one_32(p_key.surface_index, h);
		h = hash_murmur3_one_32(p_key.surface_counter, h);
		h = hash_murmur3_one_64(p_key.texture_id, h);
		h = hash_murmur3_one_32(p_key.uv_scale[0], h);
		h = hash_murmur3_one_32(p_key.uv_scale[1], h);
		h = hash_murmur3_one_32(p_key.uv_offset[0], h);
		h = hash_murmur3_one_32(p_key.uv_offset[1], h);
		h = hash_murmur3_one_32(p_key.albedo_alpha, h);
		h = hash_murmur3_one_32(p_key.threshold, h);
		return hash_fmix32(h);
	}
};

struct RTOmmEntry {
	enum State {
		STATE_QUEUED, // Waits for its setup pass.
		STATE_SETUP, // The setup pass ran; its readback is in flight.
		STATE_BUILT, // The micromap is built; BLASes may carry it.
		STATE_FAILED, // Not baked (its surface or texture went away, or an allocation failed); the any-hit decides.
	};
	RTOmmKey key;
	State state = STATE_QUEUED;
	uint64_t serial = 0; // Unique over the run; what a merged MultiMesh entry refers to it by.
	uint64_t ticket = 0; // The readback in flight.
	uint32_t primitive_count = 0;
	uint32_t last_used_frame = 0;
	RID setup_buffer;
	RID micromap;
	RID index_buffer; // One int32 a primitive: a micromap entry or a special index.
	LocalVector<int32_t> indices; // Its host copy, which a merged MultiMesh repeats per element.
	LocalVector<RD::MicromapUsage> index_usages; // The primitives' references per level, special indices left out.
	RID static_blas; // The surface's BLAS with the micromap, for its static instances.
	LocalVector<RID> merged_handles; // The merged MultiMesh entries that carry it.
	LocalVector<RID> deformed_handles; // The deformed cache entries whose triangle BLAS carries it (OMMc).
	uint32_t descriptors = 0;
	uint32_t special_primitives = 0;
	uint64_t micromap_bytes = 0;
	uint64_t known = 0; // Micro-triangles, from the bake's counts.
	uint64_t unknown = 0;
};

struct RTOmmPyramid {
	RID buffer;
	uint32_t level_count = 0;
	uint32_t last_used_frame = 0;
};

struct RTOmmReadback {
	uint64_t ticket = 0;
	bool stats = false;
	PackedByteArray data;
};

struct RTMaterialCacheEntry {
	RTMaterialData *ptr = nullptr;
	uint32_t last_used_frame = 0;
	uint16_t cached_counter = 0;
	uint32_t cached_rid_version = 0;
};

// Slot-resident instances (D94b), with rendering/pathtracing/partitioned_tlas: an instance keeps
// one slot of the partitioned TLAS while it stays in the trace, written only when it changes. A
// slot is the PTLAS instance index, the custom index and the row of the per-instance shader
// buffers. Slots come in ranges, one per RT surface of a geometry instance (the instance owns
// them, not its surface caches, which _mark_dirty frees): one slot, or a MultiMesh surface's
// expanded elements. Indices are shared by every viewport; each viewport's PTLAS, rows and the
// shadows below are its own, since the RT cull and rt_origin are per viewport.
struct RTSlotRange {
	uint32_t first = 0;
	uint32_t count = 0;
	uint32_t serial = 0; // Advances each time the id is reused, so a viewport's view of it resets.
	uint32_t enabled_viewports = 0; // The viewports whose PTLAS may hold a slot of it enabled.
	bool live = false;
	bool released = false; // Its owner dropped it; its indices return once no viewport holds it.
};

// A frame-local row (a merged MultiMesh's, a procedural instance's), compared by value; kept beside
// the ranges, since only those ranges have one.
struct RTSlotLocalRow {
	RT_GeometryData row = {};
	uint64_t generation = 0;
};

// The ranges a geometry instance owns: (key << 32) | range id, the key its surface index.
struct RTSlotOwner {
	LocalVector<uint64_t> ranges;
};

// What a viewport's PTLAS holds in one slot, and the rows its buffers hold there: the change test.
struct RTSlotShadow {
	float transform[12] = {}; // Rows of the 3x4, relative to its cell's origin (D94c).
	RID blas; // Invalid: disabled.
	uint64_t geometry_generation = 0;
	uint64_t material_generation = 0;
	uint32_t sbt_offset = 0;
	uint32_t partition = 0;
	uint32_t write_frame = 0; // The viewport's build count at the last instance write.
	uint32_t motion_frame = 0; // The build count its motion row was last written >= 0 at.
	uint32_t global_index = UINT32_MAX; // Its position in the global partition's list (D94b part 2).
	Vector3i cell; // The world cell its last write placed it in, which its transform is relative to.
	uint8_t flags = 0;
	uint8_t mask = 0;
	bool rows_valid = false; // The rows hold this slot's geometry and material generations.
	bool motion = false; // Its motion row is >= 0.
};

// A viewport's view of one range: whether the walk reached it this build, and whether the viewport
// holds a claim on it. Kept dense for the sweep, which reads every live range's each build.
struct RTSlotRangeView {
	uint32_t serial = 0;
	uint32_t present_frame = 0;
	bool enabled = false; // Holds a claim on the range (RTSlotRange::enabled_viewports).
};

// An expanded MultiMesh range's key (decision 6), kept beside the views: unchanged, the walk skips
// the range whole.
struct RTSlotMultiMeshKey {
	uint32_t serial = 0; // The range's serial it was taken for; 0 for none.
	Transform3D instance_transform;
	RID blas;
	uint64_t buffer_generation = 0;
	uint64_t geometry_generation = 0;
	uint64_t material_generation = 0;
	uint32_t count = 0;
	uint32_t sbt_offset = 0;
	uint32_t flags = 0;
	uint8_t mask = 0;
};

// An enabled slot whose material emits (D50), for the emissive gather: what the walk's arrays
// held for it when it was last written.
struct RTSlotEmissive {
	Transform3D transform; // World, as walked.
	RT_GeometryData geometry = {};
	RT_MaterialData material = {};
	uint32_t sbt_offset = 0;
	uint32_t flags = 0;
};

// One slot's rows for the scatter pass (ptlas_rows.glsl): the geometry and material rows when
// `parts` has bit 0, and the motion index always.
struct alignas(16) RT_SlotRowWrite {
	uint32_t slot;
	uint32_t parts;
	int32_t motion_index;
	uint32_t _pad;
	RT_GeometryData geometry;
	RT_MaterialData material;
};
static_assert(sizeof(RT_SlotRowWrite) == 256, "RT_SlotRowWrite must match ptlas_rows.glsl");

/// Per-viewport raytracing state.
///
/// Each viewport has its own visibility set (frustum/LOD/visibility ranges), so
/// the TLAS instance composition and per-instance SSBO contents differ across
/// viewports. Sharing them caused the wrong `gl_InstanceCustomIndexEXT` to
/// resolve to the wrong `geometries[]` / `materials[]` entries, dereferencing
/// stale BDAs and faulting the GPU.
///
/// Lifetime is tied to a `RenderSceneBuffersRD`: created lazily on first
/// `build_tlas` for that viewport, freed via `RenderRaytracing::free_viewport_state`
/// from `RenderBufferDataForwardClustered::free_data()`.
struct RTViewportState {
	RID tlas;
	uint32_t tlas_max_instances = 0;
	bool has_procedural_instances = false;

	RID geometry_buffer;
	uint32_t geometry_buffer_capacity = 0;
	RID material_buffer;
	uint32_t material_buffer_capacity = 0;
	RID motion_index_buffer;
	uint32_t motion_index_buffer_capacity = 0;
	RID motion_transform_buffer;
	uint32_t motion_transform_buffer_capacity = 0;

	RID light_buffer;
	RID emissive_buffer; // The emissive mesh lights of the frame (D50), RT_EMISSIVE_MAX entries.
	RID light_tree_buffer; // The light tree (D55): a header, the nodes, each emissive entry's cut slot, the cut's nodes and the leaf list (D56), each emissive entry's leaf position and the leaves' alias tables per cut (D60, D61); rebuilt per frame while the Environment asks for it.
	uint32_t emissive_count_logged = 0; // The last count printed for this viewport.
	RID params_buffer;
	// The sky's sampling table (D54), rebuilt by `RenderRaytracing::update_sky_light` when the
	// sky or its rendered radiance changes; zero (no sky light) until then.
	RID sky_light_buffer;
	RID sky_light_sky;
	uint64_t sky_light_version = 0;
	// The share by variance's moments (D65): built for the table they follow with the frame's weights
	// of the delivered radiance, which are kept to tell when their shape changes (D72: a change of
	// their scale, the exposure normalisation, needs no rebuild). The weights come from what sky_radiance reads in
	// this frame's scene data copy (the exposure, the fog's sky affect and aerial perspective, the fog
	// light's luminance), taken where the copy is made.
	bool sky_light_moments_built = false;
	bool sky_light_mix_valid = false;
	float sky_light_mix[3] = {};
	float sky_light_scene_exposure = 0.0f; // The exposure normalisation the kept weights were divided by.
	float sky_light_scene[4] = {};
	RID scene_uniform_set;

	// Camera-relative tracing: the anchor every position uploaded to the trace
	// is relative to (world minus rt_origin), and the scene UBO copy whose
	// camera carries the relative origin. See `RenderRaytracing::build_tlas`.
	Vector3 rt_origin;
	bool rt_origin_valid = false;
	RID scene_uniform_buffer;

	uint32_t frame_counter = 0;

	// Slot-resident instances (D94b): `tlas` is this viewport's PTLAS, and the geometry, material and
	// motion index buffers are indexed by slot, slot_capacity rows each, fed by the scatter pass.
	LocalVector<RTSlotShadow> slots;
	LocalVector<RTSlotRangeView> slot_range_views;
	HashMap<uint32_t, RTSlotMultiMeshKey> slot_mm_keys; // By range id.
	HashMap<uint32_t, RTSlotEmissive> emissive_slots;
	LocalVector<uint32_t> motion_slots; // The slots whose motion row the last build wrote >= 0.
	LocalVector<uint32_t> motion_slots_next;
	uint32_t slot_capacity = 0;
	uint32_t slot_build = 0; // Builds of this viewport's slots, from 1.
	uint64_t slot_commit_frame = 0; // frames_drawn at its last commit; 0 when it holds nothing.
	bool slot_rebuild = false; // The next build writes every enabled slot from scratch.
	Vector3 slot_origin; // The rt_origin of the last commit.
	bool slot_origin_valid = false;
	RID slot_row_buffer; // The scatter's input, grow-only.
	uint32_t slot_row_capacity = 0; // Rows.
	// The partitions (D94b part 2): each world cell's buckets of at most ptlas_partition_max_instances
	// enabled slots, and the global partition, which holds the slots rewritten while enabled until
	// they go ptlas_stable_frames builds unwritten. Occupancy mirrors the PTLAS's: enabled slots.
	struct SlotCell {
		LocalVector<uint32_t> partitions;
	};
	HashMap<Vector3i, SlotCell> slot_cells;
	LocalVector<uint32_t> partition_occupancy;
	LocalVector<Vector3i> partition_cells;
	// The translation the structure holds for each partition (D94c): its cell's origin less rt_origin,
	// rounded to float. NaN where unknown (a new partition, a build from scratch), so it is written.
	LocalVector<Vector3> partition_translations;
	LocalVector<uint32_t> partition_touched; // The build that last changed it, as the structure reports.
	LocalVector<uint32_t> partition_free;
	uint32_t partition_high_water = 0;
	uint32_t partition_capacity = 0;
	uint32_t partitions_live = 0;
	LocalVector<uint32_t> global_slots;
	// The slots written into the global partition at a build, ptlas_stable_frames + 1 buckets by
	// build: the bucket of the build ptlas_stable_frames ago holds every candidate to return.
	LocalVector<LocalVector<uint32_t>> global_return_buckets;
	uint32_t global_capacity = 0;
	uint32_t global_touched = 0;
	// The last build's counters (D94b): instance writes by cause, row writes, and the report's tallies.
	struct SlotCounters {
		uint32_t enabled = 0;
		uint32_t writes = 0;
		uint32_t writes_new = 0;
		uint32_t writes_moved = 0;
		uint32_t writes_blas = 0;
		uint32_t writes_rebuilt = 0;
		uint32_t writes_return = 0;
		uint32_t writes_origin = 0; // Global slots rewritten for an rt_origin shift (D94c).
		uint32_t translations = 0; // Partition translations written (D94c).
		uint32_t disables = 0;
		uint32_t rows = 0;
		uint32_t motion_rows = 0;
		uint32_t partitions_live = 0;
		uint32_t partitions_touched = 0;
		uint32_t global = 0;
		uint32_t global_overflow = 0;
		uint32_t from_scratch = 0; // Over the run.
		uint32_t from_scratch_last = 0; // 1 when the last commit built from scratch.
		uint64_t returns_total = 0; // Over the run: the global partition's slots back in their cells.
		uint64_t global_entries_total = 0; // Over the run: slots placed in the global partition.
		uint32_t global_peak = 0;
		uint32_t stale_rewrites = 0; // The PTLAS's own rewrites of a rebuilt BLAS's slots.
		uint32_t dangling_disables = 0; // Over the run: the PTLAS's safety net.
		uint32_t dangling_last = 0;
		uint32_t builds = 0; // The build calls of the commit: one, or one per chunk.
		uint64_t commit_usec = 0;
		uint64_t build_usec = 0; // The PTLAS build call within the commit (its stale scan included).
		uint64_t scan_usec = 0; // The PTLAS builds' stale and freed scans.
		uint64_t commit_usec_sum = 0; // Over the run, for the means the report prints.
		uint64_t build_usec_sum = 0;
		uint64_t scan_usec_sum = 0;
	} slot_counters;
};

class RenderRaytracing {
	friend class RenderForwardClustered;
	friend class RenderForwardClusteredPT;

	RenderForwardClustered *owner = nullptr;
	SceneShaderRaytracing *shader = nullptr;
	BindlessBlock *bindless_block = nullptr;

	RID bindless_uniform_set;

	// Caching (chunked sparse caches indexed by RID low bits / 256).
	Vector<RTCacheEntry *> surface_chunks;
	Vector<RTMaterialCacheEntry *> material_chunks;

	// Static indexed surfaces build their BLAS from triangle clusters (D79): the device has
	// VK_NV_cluster_acceleration_structure and rendering/pathtracing/cluster_blas asked for it.
	bool cluster_blas_enabled = false;
	// The compaction of cluster BLAS into the arena (D81), rendering/pathtracing/cluster_blas_compaction; off, a
	// surface keeps its build-time buffers for life.
	bool cluster_compaction_enabled = false;
	// Deformed surfaces rebuilt from their clusters each frame (D82), rendering/pathtracing/cluster_blas_deformed.
	bool cluster_deformed_enabled = false;
	// Cluster builds recorded this frame, for the render-info counters (they never enter dirty_blas_list).
	uint32_t cluster_blas_builds = 0;
	uint32_t cluster_triangles_built = 0;
	// The deformed cluster path's counters this frame: a layout's first build is a build, every
	// later rebuild is counted as the refit it replaces.
	uint32_t deformed_cluster_builds = 0;
	uint32_t deformed_cluster_triangles_built = 0;
	uint32_t deformed_cluster_refits = 0;
	uint32_t deformed_cluster_triangles_refit = 0;
	uint32_t deformed_cluster_expiry_rebuilds_total = 0; // Unchanged surfaces rebuilt for their region's reuse, over the run.
	RTDeformedClusterBatch deformed_cluster_batch;
	// The static cluster arena (D81) and the compactions in flight: a surface's sizes readback
	// is keyed by a ticket so a result whose surface went in the meantime is dropped.
	static RenderRaytracing *singleton;
	LocalVector<RTClusterArenaChunk> cluster_arena;
	LocalVector<RTClusterRetiredRegion> cluster_arena_retired;
	uint64_t cluster_arena_reserved_bytes = 0;
	uint64_t cluster_arena_live_bytes = 0;
	// The cluster geometry arena (D93b): the blobs, D81's allocator instanced a second time at the
	// blobs' alignment, its chunks declared as read by the trace through their addresses.
	LocalVector<RTClusterArenaChunk> cluster_geometry_arena;
	LocalVector<RTClusterRetiredRegion> cluster_geometry_arena_retired;
	uint64_t cluster_geometry_arena_reserved_bytes = 0;
	uint64_t cluster_geometry_arena_live_bytes = 0;
	// The population's vertex-to-local map (D93c), grown to the largest surface and kept at 0xFFFF between clusters.
	LocalVector<uint16_t> cluster_geometry_local;
	// Populations of surfaces the mesh storage streams the geometry of that read their own buffers (D93d).
	uint32_t cluster_geometry_unstreamed_populations = 0;
	// Uploads of such surfaces' arrays (D93d): those released past the minimum age without use, and
	// the bytes held now, at most, and the uploads made over the run.
	LocalVector<RTSurfaceData *> materialized_transient;
	uint64_t materialized_live_bytes = 0;
	uint64_t materialized_peak_bytes = 0;
	uint32_t materialized_uploads = 0;
	uint64_t materialized_swept_frame = 0;
	// A frame's transient uploads (an emissive material's, a merge's bake), within
	// cluster_lod_stream_upload_cap in the walk's order; the first of a frame always goes.
	uint64_t materialized_frame_bytes = 0;
	uint32_t materialized_frame_uploads = 0;
	uint32_t materialized_deferred = 0; // Over the run.
	HashMap<uint64_t, RTSurfaceData *> cluster_compaction_pending;
	LocalVector<RTClusterCompactionResult> cluster_compaction_results;

	// Opacity micromaps (D96), with rendering/pathtracing/opacity_micromaps.
	bool omm_enabled = false;
	float omm_scale = 2.0f;
	uint32_t omm_max_level = 10;
	HashMap<RTOmmKey, RTOmmEntry *, RTOmmKey> omm_entries;
	HashMap<uint64_t, RTOmmEntry *> omm_by_serial;
	HashMap<uint64_t, RTOmmEntry *> omm_pending; // By readback ticket.
	LocalVector<RTOmmEntry *> omm_queue;
	LocalVector<RTOmmReadback> omm_readbacks;
	HashMap<RID, RTOmmPyramid> omm_pyramids; // By texture.
	uint64_t omm_next_serial = 1;
	uint64_t omm_next_ticket = 1;
	uint64_t omm_processed_frame = UINT64_MAX;
	RID omm_sampler;
	struct OmmShader {
		enum Mode {
			MODE_PYRAMID_BASE,
			MODE_PYRAMID_REDUCE,
			MODE_SETUP,
			MODE_BAKE,
			MODE_MAX,
		};
		OpacityMicromapShaderRD shader;
		RID version;
		RID version_shader[MODE_MAX];
		RID pipeline[MODE_MAX];
	} omm_shader;
	struct OmmStats {
		uint32_t baked = 0;
		uint32_t failed = 0;
		uint64_t micromap_bytes = 0;
		uint64_t descriptors = 0;
		uint64_t primitives = 0;
		uint64_t special_primitives = 0;
		uint64_t known = 0;
		uint64_t unknown = 0;
		uint32_t blases = 0;
	} omm_stats;
	uint64_t cluster_compaction_next_ticket = 0;
	uint32_t cluster_compactions_total = 0;
	uint64_t cluster_bytes_staged_total = 0;
	uint64_t cluster_bytes_compacted_total = 0;
	// Cluster LOD selection (D86), rendering/pathtracing/cluster_lod_select with cluster_lod: the
	// selection compute, the shared LOD table every hierarchy's region is sub-allocated from (in
	// words, first fit, merged on free, grown by copy), the instance slots and the frame's batch.
	bool cluster_lod_select_enabled = false;
	// Deformed surfaces select too (D87), rendering/pathtracing/cluster_lod_deformed with the two
	// above and cluster_blas_deformed; their selected CLAS are built each frame into explicit slots
	// of this size (the worst case of one cluster under the D80 limits) in a region of the deformed
	// batch's pool.
	bool cluster_lod_deformed_enabled = false;
	uint64_t cluster_lod_clas_slot_size = 0;
	uint32_t cluster_lod_clas_ceiling_clusters = 0; // RT_LOD_CLAS_REGION_CEILING in slots, less the region's alignment.
	float cluster_lod_pixel_error = 1.0f;
	uint32_t cluster_lod_max_instances = 0; // The selection dispatch's instance axis, the device's workgroup count limit.
	// BLAS sharing (D89), rendering/pathtracing/cluster_lod_share with the selection: instances of
	// one surface whose selection can be no finer than another's trace that one's build, or the
	// surface's LOD 0 BLAS, when that adds no more than cluster_lod_share_tolerance levels of detail.
	bool cluster_lod_share_enabled = false;
	uint32_t cluster_lod_share_tolerance = 2;
	// Cached levels (D90), rendering/pathtracing/cluster_lod_cache with the sharing: the surfaces
	// holding one, for the TTL, and the totals for the report.
	bool cluster_lod_cache_enabled = false;
	HashSet<RTSurfaceData *> cluster_lod_cached_surfaces;
	uint32_t cluster_lod_cache_live = 0;
	uint64_t cluster_lod_cache_bytes = 0;
	uint64_t cluster_lod_cache_builds_total = 0;
	// The traversal (D91), rendering/pathtracing/cluster_lod_traversal with the selection; a verbose
	// run checks it against the exhaustive test of the same frame, these its tallies.
	bool cluster_lod_traversal_enabled = false;
	uint64_t lod_tree_checks = 0;
	uint64_t lod_tree_mismatches = 0;
	uint64_t lod_select_timer_frame = 0; // The captured frame last read for the selection's timing.
	LocalVector<float> lod_select_gpu_us; // The GPU span bracketing the selection per frame on a verbose run.
	struct LodSelectShader {
		ClusterLodSelectShaderRD shader;
		RID version;
		RID version_shader;
		RID pipeline;
	} lod_select_shader;
	RID cluster_lod_table;
	uint32_t cluster_lod_table_capacity = 0; // Words.
	uint32_t cluster_lod_table_live_words = 0;
	LocalVector<RTClusterArenaChunk::FreeRange> cluster_lod_table_free; // Offsets and sizes in words.
	uint32_t cluster_lod_generation_counter = 0;
	RID_Owner<RTLodInstanceEntry> lod_pool;
	RTLodSelectBatch lod_batch;
	// The selected counts read back a frame queue later, the latest whole batch, for the report.
	uint64_t lod_counts_next_ticket = 0;
	uint64_t lod_counts_ticket = 0;
	uint32_t lod_counts_selected = 0;
	uint32_t lod_counts_entries = 0;
	uint32_t lod_counts_clas = 0; // The CLAS the deformed items built in that batch.
	// The frame's selection error in pixels: the setting's, or 0 under the G-buffer primary when
	// streaming (D92), which traces the finest resident cut there.
	float lod_frame_pixel_error = 1.0f;

	// CLAS residency (D92), rendering/pathtracing/cluster_lod_stream with the selection, its
	// traversal, the cached levels and the compaction: the streamed surfaces by their LOD table
	// offset (which the selection's records carry), the readbacks and loads in flight, and the
	// totals for the report.
	bool cluster_lod_stream_enabled = false;
	uint64_t cluster_lod_stream_budget = 0; // Bytes.
	uint32_t cluster_lod_stream_builds_per_frame = 0; // Clusters.
	uint32_t cluster_lod_stream_min_age = 0; // Frames.
	uint64_t cluster_lod_stream_cluster_worst = 0; // One CLAS's worst-case bytes, a load's estimate.
	uint64_t cluster_lod_stream_upload_cap = 0; // Bytes of blobs and staged positions a frame's loads may upload (D93c).
	// Geometry residency (D93), rendering/pathtracing/cluster_lod_stream_geometry on a device with
	// position fetch: every cluster build allows data access, and hits read CLAS positions.
	bool cluster_build_data_access = false;
	bool cluster_lod_stream_geometry_enabled = false;
	// A verbose run's position-fetch check (D93a): the closest hit's counts of cluster hits whose
	// fetched positions it compared with the vertex buffer's, of those that differ and the largest
	// difference, by kind (RT_POSITION_FETCH_CHECK_WORDS words, binding 46), cleared after each read.
	RID position_fetch_check_buffer;
	HashMap<uint32_t, RTSurfaceData *> cluster_lod_stream_surfaces;
	LocalVector<RTLodStreamRead> cluster_lod_stream_reads;
	LocalVector<RTLodStreamLoad> cluster_lod_stream_loads;
	LocalVector<RTLodStreamGatherJob *> cluster_lod_stream_gathers; // In admission order (D93c).
	LocalVector<RTClusterCompactionResult> cluster_lod_stream_load_results;
	uint64_t cluster_lod_stream_next_ticket = 0;
	uint64_t cluster_lod_stream_pressure_frame = 0; // The last frame the budget deferred one of the selection's requests.
	uint64_t cluster_lod_stream_step_frame = 0; // Admission runs once a frame, whatever the viewports.
	// The lists' decaying peaks, which size the next frames' readbacks (records past the read are
	// asked for again the next frame).
	uint32_t cluster_lod_stream_request_peak = 0;
	uint32_t cluster_lod_stream_touch_peak = 0;
	// The step's eviction candidates, gathered and sorted on its first need, consumed across its wants.
	struct LodStreamCandidate {
		RTSurfaceData *surface = nullptr;
		uint32_t group = 0;
		uint64_t last_used = 0;
		bool operator<(const LodStreamCandidate &p_other) const {
			return last_used < p_other.last_used;
		}
	};
	LocalVector<LodStreamCandidate> lod_stream_candidates;
	uint32_t lod_stream_candidate_next = 0;
	bool lod_stream_candidates_built = false;
	struct LodStreamStats {
		uint64_t requests = 0;
		uint64_t loads = 0;
		uint64_t loaded_clusters = 0;
		uint64_t deferred_cap = 0;
		uint64_t deferred_budget = 0;
		uint64_t deferred_soft = 0; // A cached level's asks that found no free room.
		uint64_t root_requests = 0; // Requests for groups no cluster refines from, made by their own test.
		uint64_t evictions = 0;
		uint64_t request_overflow = 0;
		uint64_t touch_overflow = 0;
		uint64_t request_unread = 0; // Listed but past the frame's readback, asked for again later.
		uint64_t touch_unread = 0;
		uint64_t failed = 0;
		uint64_t peak_bytes = 0;
		LocalVector<uint32_t> latency; // Frames from request to resident, since the last report.
		// Geometry residency (D93c): the wants the upload cap deferred, the bytes of blobs and staged
		// positions each step with a load uploaded since the last report, and the gathers' time.
		uint64_t deferred_upload = 0;
		uint64_t uploaded_bytes = 0;
		LocalVector<uint32_t> upload_frames; // KB.
		uint64_t gather_usec = 0; // Gathers' preparation and placement on the render thread, over the run.
		uint64_t gather_max_usec = 0; // The largest in a frame, since the last report.
		uint64_t gather_waits = 0; // Gathers still running at their issue frame, waited for, over the run.
	} lod_stream_stats;

	// Merged MultiMesh BLAS cache and compute shader.
	struct MergeShader {
		enum Mode {
			MODE_NON_INDEXED = 0,
			MODE_INDEXED = 1,
			MODE_MAX,
		};
		MultimeshMergeShaderRD shader;
		RID version;
		RID version_shader[MODE_MAX];
		RID pipeline[MODE_MAX];
	} mm_merge_shader;

	// The sky's sampling table (D54): the compute shader that builds it from the radiance octmap as a
	// pyramid over the octmap's texels (D69: the leaves, the levels above them one dispatch each while
	// they are too large for one workgroup, then the rest with the header in one), then the share by
	// variance's moments per receiver normal (D65), in the form the trace reads the octmap in (the
	// roughness layers as an array, or one texture whose mips are the roughness levels), since the
	// fog's level is read as the trace reads it (D64); the moments carry the frame's weights' shape and
	// are clipped at multiples of the receiver's irradiance (D72), so there is no combine pass.
	struct SkyLightShader {
		enum Mode {
			MODE_LEAF = 0,
			MODE_REDUCE = 1,
			MODE_HEADER = 2,
			MODE_MOMENTS_OCTMAP_2D = 3,
			MODE_MOMENTS_OCTMAP_ARRAY = 4,
			MODE_MAX,
		};
		SkyLightTableShaderRD shader;
		RID version;
		RID version_shader[MODE_MAX];
		RID pipeline[MODE_MAX];
	} sky_light_shader;
	// The pyramid's level the moments and the concentrated light are taken at, 256 cells a side; the
	// leaves are never coarser. Must match SKY_CELL_LEVEL in sky_light_table.glsl.
	static constexpr uint32_t RT_SKY_LIGHT_CELL_LEVEL = 8;
	// The finest leaves: 1024 a side, 5.6 MB of pyramid, which the buffer is sized for.
	static constexpr uint32_t RT_SKY_LIGHT_MAX_LEVELS = 10;
	// The header pass reduces this level and the ones above it; must match SKY_HEADER_LEVELS in
	// sky_light_table.glsl.
	static constexpr uint32_t RT_SKY_LIGHT_HEADER_LEVELS = 5;
	static_assert(RT_SKY_LIGHT_HEADER_LEVELS == 5 && RT_SKY_LIGHT_CELL_LEVEL == 8 && RT_SKY_LIGHT_CELL_LEVEL <= RT_SKY_LIGHT_MAX_LEVELS, "MODE_HEADER in sky_light_table.glsl is written for these: level 5 from level 6 in shared memory, 64 level-8 quads per thread for the concentrated light.");
	// Levels 1 to RT_SKY_LIGHT_MAX_LEVELS, 4^l floats each: (4^(max + 1) - 4) / 3.
	static constexpr uint32_t RT_SKY_LIGHT_PYRAMID_FLOATS = ((1u << (2 * (RT_SKY_LIGHT_MAX_LEVELS + 1))) - 4) / 3;
	static constexpr uint32_t RT_SKY_LIGHT_HEADER_WORDS = 8; // The total, the concentrated part, the density scale, the leaf floor, the levels, the leaves a side, two pads.
	// The candidate shares the table judges the sky's variance at (D62); must match SKY_SHARE_COUNT in
	// sky_light_table.glsl and RT_SKY_SHARE_COUNT in raytracing_sky_inc.glsl.
	static constexpr uint32_t RT_SKY_LIGHT_SHARES = 8;
	// The clip levels the moments are taken at (D72); must match SKY_CLIP_LEVELS in sky_light_table.glsl
	// and RT_SKY_CLIP_LEVELS in raytracing_sky_inc.glsl.
	static constexpr uint32_t RT_SKY_LIGHT_CLIP_LEVELS = 6;
	// The cells a side of the receiver normals' grid over the octahedral square, a normal at every
	// vertex (D65); must match SKY_NORMAL_CELLS in sky_light_table.glsl and RT_SKY_NORMAL_CELLS in
	// raytracing_sky_inc.glsl.
	static constexpr uint32_t RT_SKY_LIGHT_NORMAL_CELLS = 8;
	static constexpr uint32_t RT_SKY_LIGHT_NORMALS = (RT_SKY_LIGHT_NORMAL_CELLS + 1) * (RT_SKY_LIGHT_NORMAL_CELLS + 1);
	// The header, the shares, per normal, clip level and share the four moments, then the pyramid at
	// its largest.
	static constexpr uint32_t RT_SKY_LIGHT_MOMENT_FLOATS = RT_SKY_LIGHT_NORMALS * RT_SKY_LIGHT_CLIP_LEVELS * RT_SKY_LIGHT_SHARES * 4;
	static constexpr uint32_t RT_SKY_LIGHT_BUFFER_SIZE = (RT_SKY_LIGHT_HEADER_WORDS + RT_SKY_LIGHT_SHARES + RT_SKY_LIGHT_MOMENT_FLOATS + RT_SKY_LIGHT_PYRAMID_FLOATS) * sizeof(float);
	static_assert((RT_SKY_LIGHT_HEADER_WORDS + RT_SKY_LIGHT_SHARES) % 4 == 0, "The moments' vec4 array must start 16-byte aligned in std430, or the shaders pad where the size does not.");
	static_assert((RT_SKY_LIGHT_HEADER_WORDS + RT_SKY_LIGHT_SHARES + RT_SKY_LIGHT_MOMENT_FLOATS) % 4 == 0, "The pyramid's vec4 array must start 16-byte aligned in std430, or the shaders pad where the size does not.");

	struct MMSurfaceHandles {
		LocalVector<RID> per_surface; // Grown on demand; entry per touched surface index.
		uint32_t mm_validator = 0; // High 32 bits of MM RID id at last access.

		// Resolve / grow storage for `p_surface_index`.
		RID &surface_handle(uint32_t p_surface_index) {
			if (p_surface_index >= per_surface.size()) {
				per_surface.resize(p_surface_index + 1);
			}
			return per_surface[p_surface_index];
		}
	};
	LocalVector<MMSurfaceHandles> mm_handles;

	RID_Owner<RTDeformedCacheEntry> deformed_pool;
	LocalVector<RID> deformed_active_this_frame;

	RID_Owner<RTMergedMMEntry> merged_mm_pool;
	LocalVector<RID> merged_mm_active_this_frame;

	RTDeformedCacheEntry *_access_deformed_slot(RID &r_handle);
	uint32_t _build_light_tree_node(uint32_t p_begin, uint32_t p_end); // The light tree's recursive build (D55).
	void _build_light_tree_cut(uint32_t p_cut_size, LocalVector<RT_LightTreeNode> &r_cut, uint32_t p_slot_shift, uint32_t p_alias_base, bool p_power_pick); // A cut the shader scores per hit (D56), the primary or the indirect one (D59), with its leaves' alias tables (D60, D61).
	void _release_deformed_blas(RTDeformedCacheEntry &p_entry);
	// The deformed triangle BLAS over owned_vb_full, carrying the micromap the walk wants (D96 OMMc).
	void _build_deformed_triangle_blas(RID p_handle, RTDeformedCacheEntry &r_entry, void *p_mesh_surface, uint32_t p_cache_key, uint32_t p_surface_counter, LocalVector<RID> &r_dirty_blas_list);
	// The hit row's deformed flag and previous-position address, which a build's row fill clears.
	void _mark_deformed_geometry(RTDeformedCacheEntry &r_entry);
	RTMergedMMEntry *_access_merged_mm_slot(RID &r_handle);

	LocalVector<uint32_t> material_free_slots;
	uint32_t next_material_slot = 0;
	uint64_t vram_used = 0;
	uint32_t cache_hits = 0;
	uint32_t cache_misses = 0;

	// Per-frame scratch arrays.
	LocalVector<RT_GeometryData> geometry_data;
	LocalVector<RT_MaterialData> material_data;
	LocalVector<int32_t> motion_indices; ///< Per-instance: index into motion_transforms[], or -1.
	LocalVector<RT_InstanceMotionData> motion_transforms; ///< Compact: only moving instances.
	LocalVector<RID> blass;
	LocalVector<Transform3D> blas_transforms;
	LocalVector<uint32_t> instance_flags;
	LocalVector<uint8_t> instance_masks;
	struct EmissiveCandidate {
		uint32_t index;
		const RTSlotEmissive *record; // A slot-resident instance's (D94b), else nullptr.
		float weight;
		float score;
		Vector3 center; // The box's centre and half diagonal in trace space, carried to the entry (D51).
		float radius;
	};
	LocalVector<EmissiveCandidate> emissive_candidates; ///< The emissive gather's ranking (D50).
	LocalVector<Pair<uint32_t, const RTSlotEmissive *>> emissive_slot_records; ///< The emissive slots in slot order (D94b).
	LocalVector<RT_EmissiveInstance> emissive_scratch;
	LocalVector<RT_LightData> light_scratch; ///< The frame's light list before upload.
	// The light tree's build scratch (D55): a leaf per positional light and per emissive entry,
	// the nodes in depth-first order, each emissive entry's cut slot, the cut's nodes and the leaf
	// list (D56), uploaded as one buffer.
	struct LightTreeLeaf {
		Vector3 center;
		float radius;
		Vector3 axis;
		float cos_theta_o;
		float power;
		float max_range;
		float decay; // The light's attenuation exponent (D63); 2 for an emissive entry.
		uint32_t encoded; // RT_LIGHT_TREE_LEAF_BIT | flags | index.
	};
	LocalVector<LightTreeLeaf> light_tree_leaves;
	LocalVector<uint32_t> light_tree_order; ///< Leaf indices, partitioned in place by the build.
	LocalVector<RT_LightTreeNode> light_tree_nodes;
	LocalVector<uint32_t> light_tree_leaf_list; ///< The leaf node at each position of the order: a node's leaves are contiguous.
	LocalVector<RT_LightTreeNode> light_tree_cut; ///< The cut's nodes, copied for the upload.
	LocalVector<RT_LightTreeNode> light_tree_cut_indirect; ///< The indirect bounces' coarser cut (D59), copied for the upload.
	LocalVector<uint32_t> light_tree_cut_queue; ///< The cut's breadth-first work queue.
	LocalVector<uint32_t> light_tree_emissive_cut; ///< Each emissive entry's cut slots (the primary's low, the indirect's high, 0xFFFF for none), RT_LIGHT_TREE_NO_NODE without a leaf.
	LocalVector<uint32_t> light_tree_emissive_leaf; ///< Each emissive entry's position in the leaf list (D60), RT_LIGHT_TREE_NO_NODE without a leaf.
	LocalVector<RT_LightTreeAlias> light_tree_leaf_alias; ///< Each leaf-list position's alias-table entry inside its cut node (D60, D61): the primary cut's, then the indirect's at RT_LIGHT_TREE_LEAVES_MAX.
	LocalVector<double> light_tree_alias_scaled; ///< The alias build's scratch: each leaf's share times its node's count, indexed by leaf-list position.
	LocalVector<uint32_t> light_tree_alias_small; ///< The alias build's scratch: positions whose scaled share is below 1.
	LocalVector<uint32_t> light_tree_alias_large; ///< The alias build's scratch: positions whose scaled share is 1 or more.
	// Camera and bounce rays select all geometry. Shadow rays select bit 1;
	// the G-buffer primary probe selects only procedural geometry (bit 2).
	// Keep these in sync with RT_MASK_* in raytracing_common_inc.glsl.
	static constexpr uint8_t RT_INSTANCE_MASK_GEOMETRY = 0x01;
	static constexpr uint8_t RT_INSTANCE_MASK_SHADOW = 0x02;
	static constexpr uint8_t RT_INSTANCE_MASK_PROCEDURAL = 0x04;
	LocalVector<uint32_t> sbt_offsets; // 0 = default material hit group

	// Slot-resident instances (D94b): on with every TLAS a PTLAS (D94a). The arrays above then hold
	// only the instances the walk could not prove unchanged, with their slots and row generations
	// beside them; the slots of every other instance keep what they hold.
	bool ptlas_slots_enabled = false;
	LocalVector<uint32_t> frame_slots;
	LocalVector<uint64_t> frame_geometry_generations;
	LocalVector<uint64_t> frame_material_generations;
	LocalVector<RTSlotRange> slot_ranges;
	LocalVector<uint32_t> slot_range_free_ids;
	struct SlotInterval {
		uint32_t first;
		uint32_t count;
	};
	LocalVector<SlotInterval> slot_free_intervals; // Sorted by first, coalesced.
	uint32_t slot_high_water = 0; // Slots from here on were never allocated.
	uint32_t slot_live = 0;
	uint64_t row_generation_counter = 0;
	HashMap<uint32_t, RTSlotLocalRow> slot_local_rows; // By range id.
	uint32_t slot_counters_reclaimed = 0; // Idle viewports whose slots were taken back, over the run.
	// A row the scatter writes: the frame's arrays at `source` (UINT32_MAX: the motion index alone).
	struct SlotRowRef {
		uint32_t slot = 0;
		uint32_t source = UINT32_MAX;
		int32_t motion_index = -1;
	};
	LocalVector<SlotRowRef> slot_row_refs;
	struct SlotRowsShader {
		PtlasRowsShaderRD shader;
		RID version;
		RID version_shader;
		RID pipeline;
	} slot_rows_shader;
	static constexpr uint32_t RT_SLOT_KEY_PROCEDURAL = UINT32_MAX;
	// rendering/pathtracing/ptlas_* (D94b part 2).
	double ptlas_cell_size = 128.0;
	uint32_t ptlas_partition_max_instances = 8192;
	uint32_t ptlas_global_max_instances = 65536;
	uint32_t ptlas_stable_frames = 16;
	static constexpr uint32_t RT_SLOT_MIN_CAPACITY = 1024;
	// The writes a build call takes and the rows a scatter dispatch takes, at most.
	static constexpr uint32_t RT_SLOT_CHUNK = 65536;
	// A viewport not committed for this many frames gives its slots back (longer than any frame in
	// flight, and long enough that one updated now and then does not rebuild each time).
	static constexpr uint64_t RT_SLOT_IDLE_FRAMES = 60;
	// Thread-free scratch of the commit.
	LocalVector<RD::PartitionedInstanceWrite> slot_writes;
	LocalVector<RT_SlotRowWrite> slot_row_writes;
	LocalVector<RD::PartitionTranslationWrite> slot_translation_writes;

	uint32_t _slot_range_acquire(RTViewportState *p_state, RTSlotOwner &r_owner, uint32_t p_key, uint32_t p_count);
	void _slot_range_release(uint32_t p_range);
	void _slot_range_try_free(uint32_t p_range);
	uint32_t _slot_interval_alloc(uint32_t p_count);
	void _slot_interval_free(uint32_t p_first, uint32_t p_count);
	uint64_t _surface_row_generation(RTSurfaceData *p_surf_data);
	uint64_t _material_row_generation(RTMaterialData *p_mat_data);
	uint64_t _local_row_generation(uint32_t p_range, const RT_GeometryData &p_row);
	static void _slot_relative_transform(const Transform3D &p_transform, const Vector3 &p_origin, float *r_rows);
	// A cell's origin (D94c): its corner on x and z, at y = 0, computed in double.
	_FORCE_INLINE_ Vector3 _slot_cell_origin(const Vector3i &p_cell) const {
		return Vector3(real_t(double(p_cell.x) * ptlas_cell_size), real_t(0), real_t(double(p_cell.z) * ptlas_cell_size));
	}
	// The walk's change test. The transform is narrowed as the shadow holds it, relative to the
	// shadow's cell (D94c), so an rt_origin shift alone changes nothing here.
	_FORCE_INLINE_ bool _slot_unchanged(const RTViewportState *p_state, uint32_t p_slot, RID p_blas, const Transform3D &p_transform, uint32_t p_flags, uint8_t p_mask, uint32_t p_sbt_offset, uint64_t p_geometry_generation, uint64_t p_material_generation) const {
		const RTSlotShadow &shadow = p_state->slots[p_slot];
		if (!(shadow.blas == p_blas && shadow.rows_valid && shadow.geometry_generation == p_geometry_generation && shadow.material_generation == p_material_generation && shadow.flags == uint8_t(p_flags) && shadow.mask == p_mask && shadow.sbt_offset == p_sbt_offset)) {
			return false;
		}
		float rows[12];
		_slot_relative_transform(p_transform, _slot_cell_origin(shadow.cell), rows);
		return memcmp(shadow.transform, rows, sizeof(rows)) == 0;
	}
	Vector3 _slot_cell_translation(const RTViewportState *p_state, const Vector3i &p_cell) const;
	Transform3D _slot_traced_transform(const RTViewportState *p_state, const RTSlotShadow &p_shadow) const;
	Transform3D _slot_record_transform(const RTViewportState *p_state, const RTSlotShadow &p_shadow) const;
	// Moved this frame for its motion: MOVED can be set without a change (the motion row of an
	// unchanged transform is the identity, which -1 gives too).
	static _FORCE_INLINE_ bool _slot_moved(bool p_status_moved, const Transform3D &p_prev, const Transform3D &p_current) {
		return p_status_moved && p_prev != p_current;
	}
	void _slot_emit(uint32_t p_slot, uint64_t p_geometry_generation, uint64_t p_material_generation);
	bool _commit_slots(RTViewportState *p_state);
	bool _flush_slot_writes(RTViewportState *p_state, bool p_from_scratch, bool p_translations = false);
	void _reset_viewport_slots(RTViewportState *p_state);
	void _reclaim_idle_viewports(RTViewportState *p_current);
	bool _slot_multimesh_key_matches(RTViewportState *p_state, uint32_t p_range, const RTSlotMultiMeshKey &p_key);
	Vector3i _slot_cell(const Transform3D &p_transform, const RT_GeometryData &p_geometry) const;
	uint32_t _slot_cell_partition(RTViewportState *p_state, const Vector3i &p_cell);
	void _slot_unplace(RTViewportState *p_state, RTSlotShadow &r_shadow);
	void _slot_place(RTViewportState *p_state, uint32_t p_slot, RTSlotShadow &r_shadow, bool p_was_enabled, const Vector3i &p_cell, bool p_prefer_global);
	bool _ensure_slot_capacity(RTViewportState *p_state);
	bool _scatter_slot_rows(RTViewportState *p_state);
	void _release_viewport_slots(RTViewportState *p_state);

	HashMap<RenderSceneBuffersRD *, RTViewportState *> viewport_states;

	RTViewportState *_get_or_create_viewport_state(const RenderDataRD *p_render_data);
	RTViewportState *_get_viewport_state(const RenderDataRD *p_render_data) const;
	void _free_viewport_state_internal(RTViewportState *p_state);

	// Material UBO sub-allocation pool. One large device-address buffer divided
	// into fixed-size slots for performance reasons, and easier to debug.
	RID mat_ubo_pool_buffer;
	uint64_t mat_ubo_pool_bda = 0;
	LocalVector<uint32_t> mat_ubo_pool_free_slots;
	uint32_t mat_ubo_pool_free_count = 0;
	uint32_t mat_ubo_pool_next_slot = 0;

	void mat_ubo_pool_ensure_initialized();
	uint32_t mat_ubo_pool_allocate(); // Returns slot index or UINT32_MAX if pool is exhausted.
	void mat_ubo_pool_release(uint32_t p_slot);
	void mat_ubo_pool_update(uint32_t p_slot, const void *p_data, uint32_t p_size);
	uint64_t mat_ubo_pool_get_address(uint32_t p_slot) const;

	// Cache helpers.
	static uint32_t get_rid_index(RID p_rid);
	static uint32_t get_rid_version(RID p_rid);
	RTCacheEntry *get_surface_cache_entry(uint32_t p_index);
	RTMaterialCacheEntry *get_material_cache_entry(uint32_t p_index);
	uint32_t allocate_material_slot();

	// Internal methods.
	RTSurfaceData *process_surface(
			const void *p_surf,
			void *p_mesh_surface,
			uint32_t p_surface_invalidation_counter,
			const Transform3D &p_transform,
			LocalVector<RID> &r_dirty_blas_list);
	RTSurfaceData *process_deformed_surface(
			const void *p_surf,
			void *p_mesh_surface,
			const struct RTDeformedGeometrySource &p_source,
			LocalVector<RID> &r_dirty_blas_list,
			LocalVector<RID> &r_dirty_blas_update_list,
			bool p_lod_select); // The frame selects (D87): an entry with a hierarchy defers its rebuild to the TLAS push.
	void _populate_surface_blas(
			void *p_mesh_surface,
			RID p_vertex_buffer_override,
			bool p_force_uncompressed,
			bool p_prefer_fast_build,
			bool p_allow_update,
			uint32_t p_cache_key,
			RTSurfaceData *r_surf_data,
			LocalVector<RID> &r_dirty_blas_list,
			const RTOmmEntry *p_omm = nullptr, // A triangle BLAS's micromap (D96 OMMc); r_omm_attached says whether it carries it.
			bool *r_omm_attached = nullptr);
	// The cluster path of _populate_surface_blas (D79): the surface's triangles as the clusters its
	// cluster table names (D80) or, without one, in runs of at most the device's cluster size; each a
	// CLAS, one cluster BLAS over them, both built this frame into the surface's own storage. Returns
	// false, with nothing created, if the surface does not fit.
	void _count_unstreamed_geometry(void *p_mesh_surface, uint32_t p_cache_key, const RTSurfaceData *p_surf_data);
	bool _materialize_surface(RTSurfaceData *p_surf_data, void *p_mesh_surface, bool p_held, bool p_capped = false);
	void _release_materialized(RTSurfaceData *p_surf_data);
	void _sweep_materialized();
	bool _plan_merged_mm(RID p_mm_rid, RID p_mm_gpu_buffer, void *p_mesh_surface, uint32_t p_mm_count, uint32_t p_surface_index, uint32_t p_surface_counter, bool &r_bakes);
	bool _populate_surface_cluster_blas(void *p_mesh_surface, RID p_vertex_buffer, RID p_index_buffer, uint32_t p_cache_key, RTSurfaceData *r_surf_data, bool p_blobs_only = false);
	// The clusters of a surface as the static and deformed paths both lay them out: from the
	// surface's table (D80), or runs of at most p_max_vertices / 3 triangles without one.
	// A cluster LOD table (D85) lists every level's clusters, the LOD 0 clusters first: the
	// BLAS is built over those, the CLAS over all; the totals count every level.
	struct RTClusterLayout {
		LocalVector<uint32_t> triangles;
		LocalVector<uint32_t> vertices;
		uint32_t max_triangles = 0;
		uint32_t max_vertices = 0;
		uint32_t total_vertices = 0;
		uint32_t total_triangles = 0;
		uint32_t lod0_clusters = 0;
		uint32_t lod0_vertices = 0;
		uint32_t levels = 1;
		bool from_table = false;
	};
	static void _cluster_layout_for_surface(void *p_mesh_surface, uint32_t p_primitive_count, uint32_t p_max_triangles, uint32_t p_max_vertices, uint32_t p_cache_key, RTClusterLayout &r_layout);
	// The deformed cluster path (D82): the entry's storage, handle, infos and slot addresses for
	// the surface's current layout. Returns false, with the entry's cluster fields cleared, if the
	// surface does not fit, and the caller builds a triangle BLAS instead.
	bool _populate_deformed_cluster_layout(void *p_mesh_surface, uint32_t p_cache_key, RTDeformedCacheEntry &r_entry);
	void _queue_deformed_cluster_rebuild(RTDeformedCacheEntry &p_entry, uint8_t p_kind);
	bool _deformed_cluster_region_expired(const RTDeformedCacheEntry &p_entry, uint64_t p_frames_drawn) const;
	// A region of the pool for this flush's CLAS: one whose last trace has been waited on, grown
	// to p_alloc if too small, or a new one; -1 if none could be made. The batch and the
	// selection (D87) share the pool.
	int32_t _deformed_cluster_region_acquire(uint64_t p_alloc, uint64_t p_alignment);
	bool _record_deformed_cluster_batch();
	void _flush_deformed_cluster_batch();
	void _release_deformed_cluster_storage(RTDeformedCacheEntry &p_entry);
	// The compaction (D81): RenderingDevice hands the built sizes to the static callback on the
	// render thread as a later frame begins; the next RT frame moves those surfaces' CLAS into
	// the arena, rebuilds the BLAS there and frees the staging.
	static void _on_cluster_sizes_read(const PackedByteArray &p_sizes, uint64_t p_ticket);
	void _process_cluster_compactions();

	// Opacity micromaps (D96), in render_raytracing_omm.cpp.
	void _omm_initialize();
	void _omm_finalize();
	// Once a frame, before the walk: readbacks become bakes and micromaps, queued surfaces get their setup pass.
	void _omm_process();
	// The entry for a surface traced with an alpha-tested standard material, queued for a bake the
	// first time it is asked for; nullptr when the material does not qualify.
	RTOmmEntry *_omm_request(RID p_mesh, uint32_t p_surface_index, uint32_t p_surface_counter, const RTMaterialData *p_material);
	// The surface's BLAS with its built micromap, made and queued for a build the first time; RID() until the micromap is built.
	RID _omm_surface_blas(RTOmmEntry *p_entry, void *p_mesh_surface, const RTSurfaceData *p_surf_data, LocalVector<RID> &r_dirty_blas_list);
	// Sets which micromap a merged MultiMesh's BLAS carries; a change frees the BLAS so the merge makes it again.
	void _omm_attach_merged(RID p_mm_rid, uint32_t p_surface_index, const RTOmmEntry *p_entry);
	// The merged BLAS's micromap fields, the base surface's index repeated p_mm_count times; false without one.
	bool _omm_merged_geometry(RTMergedMMEntry &r_entry, uint32_t p_mm_count, uint32_t p_primitives_per_element, RD::AccelerationStructureGeometry &r_geometry);
	void _omm_release_merged(RTMergedMMEntry &r_entry);
	// A deformed triangle BLAS's micromap (OMMc): the built entry of p_serial over the surface's current
	// contents, or nullptr; then its fields on the BLAS's geometry, false where the surface does not match.
	RTOmmEntry *_omm_deformed_entry(uint64_t p_serial, uint32_t p_surface_counter);
	bool _omm_deformed_geometry(const RTOmmEntry *p_entry, const RT_GeometryData &p_geometry, RD::AccelerationStructureGeometry &r_geometry);
	// Records that the deformed entry's BLAS carries the micromap, so that its release takes the BLAS along.
	void _omm_deformed_attached(RID p_handle, RTDeformedCacheEntry &r_entry, RTOmmEntry *p_omm);
	void _omm_release_entry(RTOmmEntry *p_entry);
	void _omm_release_mesh_surface(RID p_mesh, uint32_t p_surface);
	void _omm_setup(RTOmmEntry *p_entry);
	void _omm_bake(RTOmmEntry *p_entry, const PackedByteArray &p_setup);
	bool _omm_pyramid(RID p_texture, uint32_t p_width, uint32_t p_height, RTOmmPyramid *&r_pyramid);
	static void _on_omm_read(const PackedByteArray &p_data, uint64_t p_ticket, bool p_stats);
	// The cluster LOD selection (D86): the surface's region of the LOD table, an instance's slot,
	// the frame's batch, and the flush that runs the selection and builds every instance's BLAS.
	bool _cluster_lod_table_allocate(uint32_t p_words, uint32_t &r_offset);
	void _cluster_lod_table_free(uint32_t p_offset, uint32_t p_words);
	void _populate_surface_cluster_lod(void *p_mesh_surface, const RTClusterLayout &p_layout, RID p_args_buffer, uint64_t p_implicit_addresses_offset, uint32_t p_cache_key, RTSurfaceData *r_surf_data);
	void _release_surface_cluster_lod(RTSurfaceData *p_surf_data);
	RTLodInstanceEntry *_access_lod_slot(RID &r_handle);
	void _release_lod_entry(RTLodInstanceEntry &p_entry);
	bool _prepare_lod_entry(RTLodInstanceEntry &p_entry, RTSurfaceData *p_surf_data);
	void _queue_lod_instance(RTLodInstanceEntry &p_entry, const Transform3D &p_transform, uint32_t p_slot);
	// BLAS sharing (D89): the view the selection's metric reads, the depth range an instance's
	// selection could fall in, and the resolve that queues each surface's builds before the flush.
	struct LodView {
		Vector3 camera;
		float threshold = 0.0f;
		float near_distance = 1e-3f;
		bool orthogonal = false;
	};
	static LodView _lod_view(const RenderDataRD *p_render_data, const RTViewportState *p_state, float p_pixel_error);
	static void _classify_lod_instance(const RTSurfaceData *p_surf_data, const Transform3D &p_transform, const LodView &p_view, uint32_t &r_level_min, uint32_t &r_level_max);
	void _resolve_lod_candidates(const RenderDataRD *p_render_data, const RTViewportState *p_state);
	bool _queue_lod_candidate(const RTLodSelectBatch::Candidate &p_candidate);
	// Cached levels (D90): placing the candidates on them, a level's slot, its item in the batch
	// after every other, and the release of one or of all a surface holds.
	void _place_lod_cached_levels();
	bool _prepare_lod_cached_level(RTSurfaceData *p_surf_data, uint32_t p_depth);
	void _queue_lod_cached_level(RTSurfaceData *p_surf_data, uint32_t p_depth);
	void _release_lod_cached_level(RTLodCachedLevel &p_level);
	void _release_lod_cached_levels(RTSurfaceData *p_surf_data);
	static void _on_lod_cache_counts_read(const PackedByteArray &p_counts, uint32_t p_stride, const PackedInt32Array &p_expected);
	static void _on_lod_cache_overflow_read(const PackedByteArray &p_count);
	static void _on_lod_tree_check_read(const PackedByteArray &p_bytes, uint32_t p_entry_count, uint32_t p_blas_infos_offset, uint32_t p_stride, uint32_t p_instances_offset, uint32_t p_check_offset);
	// The deformed entries on the selection (D87): the entry's region of the LOD table, made with
	// its layout and released with its storage, and its record in the frame's batch.
	bool _populate_deformed_cluster_lod(void *p_mesh_surface, const RTClusterLayout &p_layout, const LocalVector<uint8_t> &p_infos, uint32_t p_cache_key, RTDeformedCacheEntry &r_entry);
	void _release_deformed_cluster_lod(RTDeformedCacheEntry &p_entry);
	void _queue_lod_deformed_instance(RTDeformedCacheEntry &p_entry, const Transform3D &p_transform, uint32_t p_slot);
	uint32_t *_lod_instance_record(const Transform3D &p_transform, uint32_t p_cluster_count, uint32_t p_table_offset);
	void _lod_record_tree(const Transform3D &p_transform, const RTLodTree &p_tree);
	static bool _build_cluster_lod_tree(const uint32_t *p_cluster_words, uint32_t p_cluster_count, const uint32_t *p_group_words, uint32_t p_group_count, LocalVector<uint32_t> &r_words, LocalVector<uint32_t> &r_level_nodes, uint32_t &r_node_count);
	bool _record_lod_batch(const RenderDataRD *p_render_data, const RTViewportState *p_state);
	void _flush_lod_batch(const RenderDataRD *p_render_data, const RTViewportState *p_state);
	static void _on_lod_counts_read(const PackedByteArray &p_counts, uint64_t p_ticket, uint32_t p_entry_count, uint32_t p_stride);
	static void _on_lod_clas_count_read(const PackedByteArray &p_count, uint64_t p_ticket);
	bool _cluster_arena_allocate(uint64_t p_size, uint32_t &r_chunk, uint64_t &r_offset);
	void _cluster_arena_retire(uint32_t p_chunk, uint64_t p_offset, uint64_t p_size);
	void _cluster_arena_free(uint32_t p_chunk, uint64_t p_offset, uint64_t p_size);
	// Geometry residency (D93b, D93c): a streamed surface's records and its persistent groups'
	// blobs, gathered from the surface's RAM copy, with what the population's build reads; false with
	// nothing allocated when the surface cannot read through them.
	bool _populate_cluster_geometry(void *p_mesh_surface, const RTClusterLayout &p_layout, RTLodStream *p_stream, uint32_t p_cache_key, RTSurfaceData *r_surf_data, RTClusterGeometryStage &r_stage);
	void _release_cluster_geometry(RTSurfaceData *p_surf_data);
	// A gather's layout and sources for the items whose surface streams its geometry (D93c), each
	// item's `gather` set; false when its slices would not fit.
	bool _prepare_cluster_geometry_gather(LocalVector<RTLodStreamLoad::Item> &r_items, RTClusterGeometryGather &r_gather);
	// Each gathered item's blob into its own arena region, uploaded, its records not yet written, the
	// positions into their buffer and every cluster's build inputs into r_stage; false with every
	// region of the call retired.
	bool _place_cluster_geometry(const LocalVector<RTLodStreamLoad::Item> &p_items, const RTClusterGeometryGather &p_gather, RTClusterGeometryStage &r_stage);
	void _write_cluster_geometry_records(const RTSurfaceData *p_surf_data, uint32_t p_group);
	void _retire_cluster_geometry_blob(RTSurfaceData *p_surf_data, uint32_t p_group);
	void _start_lod_stream_load(LocalVector<RTLodStreamLoad::Item> &r_items, bool p_compressed);
	void _issue_lod_stream_gathers();
	// CLAS residency (D92).
	RTLodStream *_create_lod_stream(void *p_mesh_surface, const RTClusterLayout &p_layout, RID p_vertex_buffer, RID p_index_buffer, uint32_t p_cache_key);
	void _release_lod_stream(RTSurfaceData *p_surf_data);
	static void _on_lod_stream_read(const PackedByteArray &p_bytes, uint64_t p_frame, uint32_t p_part);
	static void _on_lod_stream_sizes_read(const PackedByteArray &p_sizes, uint64_t p_ticket);
	void _process_lod_stream();
	void _complete_lod_stream_loads();
	bool _issue_lod_stream_load(const LocalVector<RTLodStreamLoad::Item> &p_items, bool p_compressed, const RTClusterGeometryGather *p_gather);
	bool _evict_lod_stream_groups(uint64_t p_needed, uint64_t &r_used);
	bool _lod_stream_evictable(const RTLodStream *p_stream, uint32_t p_group, uint64_t p_frames_drawn) const;
	void _join_lod_stream_candidate(RTSurfaceData *p_surf_data, uint32_t p_group, uint64_t p_frames_drawn);
	void _evict_lod_stream_group(RTSurfaceData *p_surf_data, uint32_t p_group);
	bool _lod_stream_level_resident(const RTSurfaceData *p_surf_data, uint32_t p_depth) const;
	RID _lod_stream_pinned_blas(RTSurfaceData *p_surf_data);
	void _prepare_lod_stream_pins();
	// Geometry residency (D93): a cluster build's flags, with data access while it is on.
	_FORCE_INLINE_ BitField<RD::AccelerationStructureFlagBits> _cluster_build_flags(RD::AccelerationStructureFlagBits p_base) const {
		BitField<RD::AccelerationStructureFlagBits> flags = p_base;
		if (cluster_build_data_access) {
			flags.set_flag(RD::ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_BIT);
		}
		return flags;
	}
	static void _on_position_fetch_check_read(const PackedByteArray &p_bytes);
	static void _mesh_surface_freed(RID p_mesh, uint32_t p_surface);
	// Frees the surface's BLAS and, on the cluster path, the staging buffers or the arena region it lives in.
	void _release_surface_data(RTSurfaceData *p_surf_data);
	RTMaterialData *process_material(RID p_material_rid, uint16_t p_material_invalidation_counter);
	bool _build_merged_mm_blas(
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
			RTSurfaceData *r_surf_data);
	void update_procedural_blas(RTProceduralState *p_state, LocalVector<RID> &r_dirty_blas_list);
	bool build_acceleration_structures(RTViewportState *p_state, const LocalVector<RID> &p_dirty_blas_list, const LocalVector<RID> &p_dirty_blas_update_list);
	void finalize_buffers(RTViewportState *p_state);
	void prepare_frame();
	void _update_scene_uniform_buffer(RTViewportState *p_state, const RenderDataRD *p_render_data);

public:
	void initialize(RenderForwardClustered *p_owner);
	// Geometry residency (D93) is on: the hit stages compile with position fetch.
	static bool is_position_fetch_enabled() { return singleton != nullptr && singleton->cluster_lod_stream_geometry_enabled; }

	void cleanup_caches();
	// Releases a geometry instance's PTLAS slot ranges (D94b) as it is freed.
	void release_slot_ranges(RTSlotOwner &r_owner);
	// Releases the ranges of surface indices from p_surface_count on, as a mesh swapped in place drops them (D95).
	void release_slot_ranges_from(RTSlotOwner &r_owner, uint32_t p_surface_count);
	// Releases the RT cache entry of a surface its mesh is dropping (MeshStorage's surface-freed callback).
	void release_mesh_surface(RID p_mesh, uint32_t p_surface);

	// p_primary_from_gbuffer: the frame's visible surface is the raster's, so the cluster LOD selection stands down (D86).
	RTViewportState *build_tlas(const RenderDataRD *p_render_data, uint32_t p_rt_flags, bool p_primary_from_gbuffer = false);
	uint32_t gather_lights(const RenderDataRD *p_render_data, RT_LightData *r_light_data, uint32_t p_max_lights, const Vector3 &p_origin, float &r_score_sum, uint32_t &r_directional_count);
	uint32_t gather_emissive(const RenderDataRD *p_render_data, const RTViewportState *p_state, RT_EmissiveInstance *r_emissive_data, uint32_t p_max_emissive, float &r_score_sum);
	uint32_t build_light_tree(const RT_LightData *p_lights, uint32_t p_light_count, uint32_t p_directional_count, const RT_EmissiveInstance *p_emissive, uint32_t p_emissive_count, float p_exposure, uint32_t p_cut_size, uint32_t p_cut_indirect_size, bool p_power_pick);
	RID update_uniform_set(RTViewportState *p_state, const RenderDataRD *p_render_data, uint32_t p_rt_flags, bool p_primary_from_gbuffer);
	void update_sky_light(RTViewportState *p_state, const RenderDataRD *p_render_data, uint32_t p_rt_flags);

	void copy_output_texture(const RenderDataRD *p_render_data);
	void free_viewport_state(RenderSceneBuffersRD *p_render_buffers);

	// Raytracing output textures (stored on the render buffers via named scopes).
	void rt_ensure_textures(RenderSceneBuffersRD *p_render_buffers);
	bool rt_has_texture(RenderSceneBuffersRD *p_render_buffers) const;
	RID rt_get_texture(RenderSceneBuffersRD *p_render_buffers) const;
	bool rt_has_depth_texture(RenderSceneBuffersRD *p_render_buffers) const;
	RID rt_get_depth_texture(RenderSceneBuffersRD *p_render_buffers) const;

	// The G-buffer primary surface: seven RGBA16F targets the scene shader's G-buffer variant writes
	// (albedo + specular, normal + roughness, geometry normal + metallic, emission + IOR, transmission + flags,
	// the medium's absorption and scattering), rasterized with the render buffers' velocity and depth,
	// read by the raygen.
	static constexpr int RT_GBUFFER_TARGETS = 7; // Pinned to RenderBufferDataForwardClustered::GBUFFER_TARGETS in the source.
	static StringName rt_gbuffer_texture_name(int p_target);
	void rt_ensure_gbuffer_textures(RenderSceneBuffersRD *p_render_buffers);
	void rt_free_gbuffer_textures(RenderSceneBuffersRD *p_render_buffers);
	bool rt_has_gbuffer_textures(RenderSceneBuffersRD *p_render_buffers) const;
	RID rt_get_gbuffer_framebuffer(RenderSceneBuffersRD *p_render_buffers);

	// DLSS Ray Reconstruction guide buffers (stored on the render buffers).
	void dlss_rr_ensure_buffers(RenderSceneBuffersRD *p_render_buffers);
	void dlss_rr_free_buffers(RenderSceneBuffersRD *p_render_buffers);
	bool dlss_rr_has_buffers(RenderSceneBuffersRD *p_render_buffers) const;
	RID dlss_rr_get_diffuse_albedo(RenderSceneBuffersRD *p_render_buffers) const;
	RID dlss_rr_get_specular_albedo(RenderSceneBuffersRD *p_render_buffers) const;
	RID dlss_rr_get_normal_roughness(RenderSceneBuffersRD *p_render_buffers) const;
	RID dlss_rr_get_specular_hit_dist(RenderSceneBuffersRD *p_render_buffers) const;

	void register_raytracing_buffer_dependencies(RD::RaytracingListID p_list);

	SceneShaderRaytracing *get_shader() const { return shader; }

	RID get_bindless_uniform_set() const { return bindless_uniform_set; }
	RID get_mat_ubo_pool_buffer() const { return mat_ubo_pool_buffer; }

	~RenderRaytracing();
};

} // namespace RendererSceneRenderImplementation
