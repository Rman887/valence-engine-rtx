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
#include "core/string/string_name.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "core/templates/vector.h"
#include "servers/rendering/renderer_rd/bindless_block.h"
#include "servers/rendering/renderer_rd/shaders/raytracing/multimesh_merge.glsl.gen.h"
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
	uint32_t _pad[5];
};
static_assert(sizeof(RT_GeometryData) == 128, "RT_GeometryData must be 128 bytes for std430");

/// Per-instance motion data for velocity computation (matches GLSL InstanceMotionData, 48 bytes).
struct RT_InstanceMotionData {
	float prev_object_to_world[12]; // Previous object-to-world (mat3x4, transposed 3x4).
};
static_assert(sizeof(RT_InstanceMotionData) == 48, "RT_InstanceMotionData must be 48 bytes");

// Must match GLSL MaterialData (std430, 96 bytes).
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
};
static_assert(sizeof(RT_MaterialData) == 96, "RT_MaterialData must be 96 bytes for std430");

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

struct RTSurfaceData {
	RID blas;
	RT_GeometryData geometry = {};
	Transform3D aabb_transform;
	bool is_compressed = false;
	uint64_t blas_size = 0;
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
	RID uniform_buffer; // Buffer pointer for mats > 512 bytes.
	uint32_t uniform_pool_slot = UINT32_MAX; // Index into the material UBO pool, or UINT32_MAX (unused)
	RID albedo_texture_rd;
	RID normal_texture_rd;
	RID orm_texture_rd;
	RID emission_texture_rd;
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
};

struct RTMaterialCacheEntry {
	RTMaterialData *ptr = nullptr;
	uint32_t last_used_frame = 0;
	uint16_t cached_counter = 0;
	uint32_t cached_rid_version = 0;
};

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
		float weight;
		float score;
		Vector3 center; // The box's centre and half diagonal in trace space, carried to the entry (D51).
		float radius;
	};
	LocalVector<EmissiveCandidate> emissive_candidates; ///< The emissive gather's ranking (D50).
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
			LocalVector<RID> &r_dirty_blas_update_list);
	void _populate_surface_blas(
			void *p_mesh_surface,
			RID p_vertex_buffer_override,
			bool p_force_uncompressed,
			bool p_prefer_fast_build,
			bool p_allow_update,
			uint32_t p_cache_key,
			RTSurfaceData *r_surf_data,
			LocalVector<RID> &r_dirty_blas_list);
	RTMaterialData *process_material(RID p_material_rid, uint16_t p_material_invalidation_counter);
	bool _build_merged_mm_blas(
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
			RTSurfaceData *r_surf_data);
	void update_procedural_blas(RTProceduralState *p_state, LocalVector<RID> &r_dirty_blas_list);
	void build_acceleration_structures(RTViewportState *p_state, const LocalVector<RID> &p_dirty_blas_list, const LocalVector<RID> &p_dirty_blas_update_list);
	void finalize_buffers(RTViewportState *p_state);
	void prepare_frame();
	void _update_scene_uniform_buffer(RTViewportState *p_state, const RenderDataRD *p_render_data);

public:
	void initialize(RenderForwardClustered *p_owner);

	void cleanup_caches();

	RTViewportState *build_tlas(const RenderDataRD *p_render_data, uint32_t p_rt_flags);
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
