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
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
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

class FileAccess;
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
	// _pad[RT_GEOMETRY_LAYERS_WORD]: the instance's visual layers, stamped on the frame's copy of the
	// row, never the cached one (D104); the hit shaders test a decal's cull mask against them.
	uint32_t _pad[2];
	// The second UV set's byte offset inside `attribute_stride`, RT_OFFSET_NONE without one, and its
	// decompression scale (fp16 x, fp16 y) for a compressed surface (D118).
	uint32_t uv2_byte_offset;
	uint32_t uv2_scale_packed;
};
static_assert(sizeof(RT_GeometryData) == 128, "RT_GeometryData must be 128 bytes for std430");

/// Per-instance motion data for velocity computation (matches GLSL InstanceMotionData, 48 bytes).
struct RT_InstanceMotionData {
	float prev_object_to_world[12]; // Previous object-to-world (mat3x4, transposed 3x4).
};
static_assert(sizeof(RT_InstanceMotionData) == 48, "RT_InstanceMotionData must be 48 bytes");

enum : uint32_t {
	RT_GEOMETRY_LAYERS_WORD = 0, // Must match the word raytracing_decals_inc.glsl reads.
	RT_GEOMETRY_LAYERS_MASK = (1u << 20) - 1, // The scene's layers; the editor's (20 and up) never enter the trace.
};

// Must match GLSL MaterialData (std430, 176 bytes).
struct alignas(16) RT_MaterialData {
	uint32_t albedo_texture_idx;
	uint32_t normal_texture_idx;
	uint32_t orm_texture_idx; // The ORM texture, or StandardMaterial3D's roughness texture (D118).
	uint32_t emission_texture_idx;
	float albedo_color[4];
	float emission_color[3];
	float emission_strength;
	float metallic;
	float roughness;
	uint32_t hair_ior_tilt; // With RT_MAT_FLAG_HAIR: packHalf2x16(hair_ior, the cuticle tilt in radians) (D126 H2).
	uint32_t flags;
	float uv1_scale[2];
	float uv1_offset[2];
	float normal_map_depth; // Strength [0..N], default 1.0 (not Z-depth).
	float specular; // Dielectric specular [0..1], default 0.5 -> F0 = 0.04.
	uint64_t uniform_address; // BDA for custom shader uniform buffer (0 = none).
	float alpha_threshold; // The alpha test's cutoff: opaque where alpha >= it (default 0.5).
	uint32_t backlight_rgba8; // With RT_MAT_FLAG_DIFFUSE_TRANSMISSION: BaseMaterial3D's backlight colour, sRGB, RGBA8.
	uint32_t backlight_texture_idx; // Its backlight texture, read linear as the raster reads it (0 = none).
	uint32_t metallic_texture_idx; // The ORM texture again, or StandardMaterial3D's metallic texture (D118).
	float anisotropy; // BaseMaterial3D's anisotropy_ratio (D118 MM1); 0 = an isotropic specular lobe.
	uint32_t anisotropy_flowmap_idx; // Its flow map, read linear (0 = none: along the tangent, at full strength).
	uint32_t clearcoat_half2; // packHalf2x16(clearcoat, clearcoat_roughness), BaseMaterial3D's coat (D118 MM2); 0 = no coat.
	uint32_t clearcoat_texture_idx; // Its texture, read linear: red scales the strength, green the roughness (0 = none).
	uint32_t rim_half2; // packHalf2x16(rim, rim_tint), BaseMaterial3D's rim read as a sheen layer (D118 MM3); 0 = no rim.
	uint32_t rim_texture_idx; // Its texture, read linear: red scales the rim, green the tint (0 = none).
	float uv2_scale[2]; // BaseMaterial3D's uv2_scale and uv2_offset, read for a detail layer on UV2 (D118 MM4).
	float uv2_offset[2];
	uint32_t detail_albedo_texture_idx; // With RT_MAT_FLAG_DETAIL: the detail albedo, sRGB (0 = none: the white default).
	uint32_t detail_normal_texture_idx; // Its normal map, read linear (0 = none: flat).
	uint32_t detail_mask_texture_idx; // Its mask, red read linear on UV1 (0 = none: the white default, everywhere).
	float subsurface_strength; // BaseMaterial3D's subsurf_scatter_strength, the diffuse share that walks under the surface (D120); 0 = none.
	uint32_t subsurface_texture_idx; // Its texture, red read linear on UV1, scaling the strength (0 = none).
	uint32_t hair_melanin_half2; // With RT_MAT_FLAG_HAIR: packHalf2x16(hair_melanin, hair_melanin_redness) (D126 H2).
};
static_assert(sizeof(RT_MaterialData) == 176, "RT_MaterialData must be 176 bytes for std430");

// Light types for raytracing (matches GLSL RT_LIGHT_TYPE_* defines).
enum RTLightType : uint32_t {
	RT_LIGHT_TYPE_OMNI = 0,
	RT_LIGHT_TYPE_DIRECTIONAL = 1,
	RT_LIGHT_TYPE_RECT = 2, // VAL (D124): AreaLight3D, a one-sided rectangle.
	RT_LIGHT_TYPE_SPOT = 3,
};

// Must match GLSL RTLightData (std430, 80 bytes).
struct alignas(16) RT_LightData {
	float position[3]; // World position (omni/spot) or direction (directional, normalized).
	uint32_t type;
	float emission[3]; // RECT: the rectangle's radiance.
	float radius; // RECT: the half diagonal.
	float attenuation;
	float inv_max_range; // 1.0/range, or -1.0 for infinite.
	float max_range_squared; // range*range, or 0.0 for infinite.
	float specular_amount;
	float indirect_energy;
	float inv_spot_attenuation; // RECT: the half height axis, x.
	float cos_spot_angle; // RECT: y.
	float _pad0; // RECT: z.
	float spot_direction[3]; // RECT: the half width axis (the light's local X times half its width).
	float _pad1; // RECT: the sign that turns cross(width axis, height axis) into the emitting normal.
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
	float alias_threshold; // VAL (P1i): Vose's alias table over the pdfs: a pick of this entry's position keeps it below this.
	uint32_t instance_index; // The TLAS instance: its geometries[] entry.
	uint32_t primitive_count;
	uint32_t flags; // RT_EMISSIVE_FLAG_*, and from RT_EMISSIVE_ALIAS_SHIFT on the alias a pick above the threshold goes to (P1i).
	float pdf; // This entry's selection probability.
	float center[3]; // The instance's bounding sphere in trace space (D51): the resampler's proximity proxy.
	float radius;
};
static_assert(sizeof(RT_EmissiveInstance) == 96, "RT_EmissiveInstance must be 96 bytes for std430");

enum {
	RT_EMISSIVE_MAX = 1024,
	RT_EMISSIVE_FLAG_TWO_SIDED = 1u,
	RT_EMISSIVE_FLAG_FLIP_FACING = 2u, // The instance flips the winding: its lit face winds clockwise as seen (Godot front faces).
	RT_EMISSIVE_ALIAS_SHIFT = 8u, // The alias's entry in a flags word (P1i); RT_EMISSIVE_MAX fits above it.
};

// A decal as the hit shaders apply it (D104): the raster's DecalData with its transforms in trace
// space instead of view space. Must match GLSL RTDecalData (std430, 224 bytes) in raytracing_decals_inc.glsl.
struct alignas(16) RT_DecalData {
	float xform[12]; // Trace space to the decal's UV space (x and z in [0, 1], y in [-1, 1]), three rows.
	float albedo_rect[4]; // Decal atlas rects, position then size; all zero for a missing texture.
	float normal_rect[4];
	float orm_rect[4];
	float emission_rect[4];
	float modulate[4]; // Linear, alpha times the distance fade.
	float normal_xform[12]; // The decal's world basis, orthonormalized, three rows (xyz used).
	float normal[3]; // The decal's up axis in world space, for the normal fade.
	uint32_t mask; // The decal's cull mask, against the instance's layers.
	float emission_energy; // Times the distance fade.
	float albedo_mix;
	float upper_fade;
	float lower_fade;
	float normal_fade;
	uint32_t order; // The raster's sort rank (camera distance minus the sorting offset): lower is nearer, applied later.
	float _pad[2];
};
static_assert(sizeof(RT_DecalData) == 224, "RT_DecalData must be 224 bytes for std430");

// A node of the decals' bounding volume hierarchy (D104), built per frame on the CPU over the decals'
// trace-space boxes. Must match GLSL RTDecalNode (std430, 32 bytes) in raytracing_decals_inc.glsl.
struct alignas(16) RT_DecalNode {
	float aabb_min[3];
	uint32_t child; // An inner node: the right child (the left is the next node); a leaf: RT_DECAL_LEAF_BIT | first record.
	float aabb_max[3];
	uint32_t count; // A leaf's record count, 0 for an inner node.
};
static_assert(sizeof(RT_DecalNode) == 32, "RT_DecalNode must be 32 bytes for std430");

enum : uint32_t {
	RT_DECALS_MAX = 4096, // The decal buffer's capacity; past it the farthest decals are dropped.
	RT_DECAL_LEAF_SIZE = 4, // Must not exceed what RT_DECAL_STACK_SIZE in raytracing_decals_inc.glsl allows for the depth.
	RT_DECAL_NODES_MAX = 2 * RT_DECALS_MAX - 1,
	RT_DECAL_LEAF_BIT = 0x80000000u, // Must match RT_DECAL_LEAF_BIT in raytracing_decals_inc.glsl.
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

// A leaf of the light tree (D55): a positional light or an emissive entry of the frame.
struct RTLightTreeLeaf {
	Vector3 center;
	float radius;
	Vector3 axis;
	float cos_theta_o;
	float power;
	float max_range;
	float decay; // The light's attenuation exponent (D63); 2 for an emissive entry.
	uint32_t encoded; // RT_LIGHT_TREE_LEAF_BIT | flags | index.
	uint64_t key; // Which light it is from frame to frame (P2i): a light's instance RID, an emitter's instance index.
};

// The light tree as built for one viewport and uploaded to its buffer (D55), kept so the next frame can
// refit it (P2i). Its structure (the leaves' order, the nodes' topology, the cuts, each emissive entry's
// cut slots and leaf position) follows the set of leaves and their centres alone, so a frame with the
// same lights at the same places keeps it: the leaves whose power, colour, range, cone, radius, decay or
// index changed are rewritten, their ancestors recomputed, the cut copies and alias tables over them
// rebuilt, and only what changed is uploaded.
struct RTLightTree {
	LocalVector<RT_LightTreeNode> nodes; // Depth first: a node's left child follows it.
	LocalVector<uint32_t> leaf_list; // The leaf node at each position of the build's order.
	AHashMap<uint64_t, uint32_t> light_positions; // Each analytic leaf's position by its key.
	AHashMap<uint64_t, uint32_t> emissive_positions; // Each emissive leaf's position by its key.
	LocalVector<uint32_t> cut_nodes[2]; // The node at each slot of the primary cut and of the indirect one (D59).
	LocalVector<RT_LightTreeNode> cuts[2]; // Those nodes, copied for the upload.
	LocalVector<uint32_t> emissive_cut; // Each emissive entry's cut slots (the primary's low, the indirect's high), RT_LIGHT_TREE_NO_NODE without a leaf.
	LocalVector<uint32_t> emissive_leaf; // Each emissive entry's position in the leaf list (D60), RT_LIGHT_TREE_NO_NODE without a leaf.
	LocalVector<RT_LightTreeAlias> leaf_alias; // Each position's alias-table entry inside its cut node (D60, D61): the primary cut's, then the indirect's at RT_LIGHT_TREE_LEAVES_MAX.
	uint32_t emissive_count = 0;
	uint32_t cut_size = 0;
	uint32_t cut_indirect_size = 0;
	bool power_pick = false;
	bool uploaded = false; // The viewport's buffer holds it.
};

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
	RT_MAT_FLAG_POINT_FILTER = 4u, // The textures' filter is nearest (pixel art), from the albedo texture's filter hint.
	RT_MAT_FLAG_EMISSION_MULTIPLY = 8u, // BaseMaterial3D emission_operator Multiply; Add otherwise.
	RT_MAT_FLAG_EMISSIVE_CONSTANT = 16u, // Emits a non-zero constant radiance with no texture: an emissive mesh light (D50).
	RT_MAT_FLAG_DIFFUSE_TRANSMISSION = 32u, // BaseMaterial3D backlight: a thin diffuse-transmission lobe (D103).
	RT_MAT_FLAG_ANISOTROPIC = 64u, // The albedo texture's filter is linear, mipmapped and anisotropic (D116).
	RT_MAT_FLAG_NO_MIPMAPS = 128u, // The albedo texture's filter has no mipmaps: every fetch reads the base level.
	// The channel the roughness (bits 8-10) and metallic (bits 11-13) textures are read from: BaseMaterial3D's
	// TextureChannel, 0 red to 3 alpha, 4 the grey average (D118).
	RT_MAT_ROUGHNESS_CHANNEL_SHIFT = 8u,
	RT_MAT_METALLIC_CHANNEL_SHIFT = 11u,
	// BaseMaterial3D's detail layer (D118 MM4): bit 14 on, bits 15-16 its blend mode (BlendMode Mix, Add, Subtract,
	// Multiply), bit 17 its textures on UV2.
	RT_MAT_FLAG_DETAIL = 1u << 14,
	RT_MAT_DETAIL_BLEND_SHIFT = 15u,
	RT_MAT_FLAG_DETAIL_UV2 = 1u << 17,
	RT_MAT_FLAG_EMISSION_UV2 = 1u << 18, // BaseMaterial3D's emission_on_uv2: the emission texture read on UV2 (D118 MM4).
	RT_MAT_FLAG_SUBSURFACE_SKIN = 1u << 19, // BaseMaterial3D's subsurf_scatter_skin_mode: the walk takes skin's channel ratios (D120).
	// BaseMaterial3D's hair feature (D126 H2): a strand hit (FLAG_LSS) shades with the hair BCSDF; bit 21 takes the
	// absorption from the albedo instead of the melanin.
	RT_MAT_FLAG_HAIR = 1u << 20,
	RT_MAT_FLAG_HAIR_COLOR = 1u << 21,
	// BaseMaterial3D's vertex_color_use_as_albedo (D126 H3): the albedo and its alpha multiplied by the vertex colour;
	// bit 23 converts the colour from sRGB first (vertex_color_is_srgb).
	RT_MAT_FLAG_VERTEX_COLOR = 1u << 22,
	RT_MAT_FLAG_VERTEX_COLOR_SRGB = 1u << 23,
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
	// The attribute buffer holds compressed UVs (UNORM16). The surface's own flag: a deformed row lays out
	// its vertex side uncompressed but reads the mesh's attribute buffer in place.
	RT_GEOM_FLAG_ATTRIBUTES_COMPRESSED = 16u,
	// A line list traced as linear swept spheres (D126): its BLAS's primitives are segments, two indices each, with the
	// radius per vertex from CUSTOM0, and its hits build their frame from the segment.
	RT_GEOM_FLAG_LSS = 32u,
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
// never rebuilt in place, so no frame's build writes what an earlier frame's trace may still read. Released
// by the memory budget (D112), least recently traced first, or with the surface's hierarchy; none
// is made while the surface's compaction is pending.
struct RTLodCachedLevel {
	RID blas;
	uint64_t blas_address = 0;
	int32_t arena_chunk = -1;
	uint64_t arena_offset = 0;
	uint64_t arena_size = 0;
	uint32_t last_used_frame = 0;
	bool evicted = false; // Freed by the memory budget (D112): its next build is a rebuild.
	// The build budget (D97): the frame its build was last refused and the frame it began waiting.
	uint64_t budget_frame = UINT64_MAX;
	uint64_t budget_wait_since = 0;
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

struct RTClusterAsyncPopulation;

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
	uint64_t compaction_ticket = 0; // The sizes readback in flight, or the compaction on the compute queue (D98b); 0 for none.
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
	bool cluster_omm = false; // Its CLAS carry an opacity micromap (D96 OMMd): its builds are flagged for one.
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
	// An admitted new surface whose BLAS the compute queue builds (D98b, D98c): frames_drawn from
	// which the main queue may trace it, stamped where its build is recorded, and UINT64_MAX from
	// the walk that queued it until then. Until then it is a waiting surface to every reader.
	uint64_t async_ready_frame = 0;
	// An admitted new cluster surface (D98c): its population's commands, held from the walk for the
	// compute queue, and its LOD table writes for the next frame. Null once those landed.
	RTClusterAsyncPopulation *cluster_async = nullptr;
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
	uint32_t last_built_frame = 0; // An instance tracing another's build keeps its slot under the slot budget (D113).
	RID blas;
	uint64_t blas_address = 0;
	int32_t arena_chunk = -1;
	uint64_t arena_offset = 0;
	uint64_t arena_size = 0;
	// Its last selection's digest in the LOD table (D113): the count and two hashes of the selected
	// CLAS addresses the BLAS in its slot was built over, held with the slot. A slot not yet built
	// over a selection (new, or last built where the frame could not keep digests) is forced.
	uint32_t digest_offset = UINT32_MAX;
	bool built = false;
};

struct RTDeformedCacheEntry;

// The selection's instance record, in words. Must match INSTANCE_WORDS in cluster_lod_select.glsl.
// Words 24 to 35 are a persistent deformed item's (D109): its slot stride and count, its digest, its
// cluster list, its kept digest in the table, its triangles and its BLAS address. A persistent static
// item (D113) takes the digest, the kept digest and the BLAS address.
static constexpr uint32_t RT_LOD_INSTANCE_WORDS = 36;
// A static instance's kept digest (D113), in words, and the slots one table allocation holds.
static constexpr uint32_t RT_LOD_DIGEST_WORDS = 3;
static constexpr uint32_t RT_LOD_DIGEST_BLOCK = 256;

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
static constexpr uint32_t RT_SUBSURFACE_STATS_WORDS = 7; // The walk's counters (D120), binding 52.
static constexpr uint32_t RT_SHADOW_STATS_WORDS = 3; // The inline shadow ray's counters (P1k), binding 53.
// The surface record's codec check (P3a), binding 55, asserts builds only: surfaces compared (one pixel in 64), surfaces with a field outside
// its encoding's rounding, those fields' bits (raytracing_surface_inc.glsl), shading normals the record replaces by the
// geometry normal, geometry normals it replaces by the normal facing the ray, surfaces with a non-finite colour or scalar.
static constexpr uint32_t RT_SURFACE_CODEC_CHECK_WORDS = 6;
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
// deformed batch. A VRAM budget (6d) replaces it. Persistent items (D109) use no region: the same
// bound applies to the arguments their slots take (an instantiation info, a destination and a
// cluster index each).
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
		bool rebuild = false; // Of a level the memory budget freed (D112), counted once the batch records it.
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
	LocalVector<RID> clas_template_buffers; // The templates the deformed items' instantiations read (D108).
	// The persistent deformed items (D109): their storage, each once, and their largest slot count.
	LocalVector<RID> deformed_storage_buffers;
	uint32_t deformed_max_capacity = 0;
	uint32_t static_first_builds = 0; // The static items queued with no build over a selection in their slot (D113).
	bool static_persistent = false; // The record kept the static items' digests (D113).
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
	uint64_t held_since = 0; // frames_drawn the build budget (D97) first held it back, 0 when never.
	uint64_t held_frame = UINT64_MAX; // frames_drawn it was last held: a later viewport's step passes it on.
};

// A compaction recorded on the compute queue (D98b): the move and the BLAS over the moved CLAS run
// beside frame `frame`, and the swap (the LOD table's compact addresses, the handle, the staging's
// frees) waits for the next frame, the surface tracing its staging meanwhile. The surface keeps its
// ticket in cluster_compaction_pending until then; a surface released in between leaves the region
// and the handle to be retired.
struct RTClusterCompactionInFlight {
	RTSurfaceData *surf = nullptr;
	uint64_t ticket = 0;
	uint64_t frame = 0;
	RID blas;
	uint32_t chunk = 0;
	uint64_t region_offset = 0;
	uint64_t region_size = 0;
	LocalVector<uint64_t> compact_addresses; // Per CLAS, then the BLAS.
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
	// The build budget (D97): the frame (frames_drawn) its population was last decided, the
	// answer, and the frame it began waiting, 0 while it is not.
	uint64_t budget_frame = UINT64_MAX;
	uint64_t budget_wait_since = 0;
	bool budget_admitted = false;
};

/// BLAS cache entry for a surface driven by a per-frame-deformed vertex buffer.
///
/// The skinned vertex buffer is owned by the engine's MeshInstance system and may
/// be freed/reallocated outside our control. To decouple BLAS lifetime from that
/// we copy the skinned VB into our own buffer on every frame it changed:
///   * owned_vb_full mirrors the skinned VB layout (positions + N + T) and is what
///     the BLAS and the hit shader read from this frame.
///   * prev_positions is the buffer whose leading positions are the previous frame's,
///     for the motion-vector path (P2f): owned_vb_full on a frame the data did not
///     change, the mesh instance's previous skinned buffer on one it did where the
///     instance double-buffers, else prev_pos_vb, which takes owned_vb_full's
///     positions before owned_vb_full is overwritten.
struct RTDeformedCacheEntry {
	RTSurfaceData *ptr = nullptr;
	uint32_t last_used_frame = 0;
	uint64_t cached_change_stamp = 0;
	uint32_t cached_key_version = 0;
	uint32_t cached_surface_counter = 0;
	uint64_t cached_buffer_id = 0; // RID id of the deformed vertex buffer at the time of build.
	void *cached_mesh_surface = nullptr; // The surface it was built for (D110: its handle can outlive the instance's surface cache).
	bool blas_built_once = false; // True once the BLAS has been fully built; subsequent ticks can refit.

	RID owned_vb_full;
	uint32_t owned_vb_full_capacity = 0; // Bytes; grow-only.
	RID prev_pos_vb; // Created on the first frame it is needed.
	uint32_t prev_pos_vb_capacity = 0; // Bytes; grow-only.
	RID prev_positions; // This frame's previous positions (see above); not owned unless prev_pos_vb.
	uint32_t cached_vertex_count = 0;
	uint32_t cached_full_size = 0;
	bool prev_pos_seeded = false; // False until owned_vb_full holds positions of this surface.
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
	LocalVector<uint8_t> cluster_infos; // cluster_count driver build infos, or instantiation infos when templated (D107).
	// Templated (D107): the CLAS are instantiated in place from the mesh surface's shared templates
	// (RTDeformedClusterTemplate, keyed by its index buffer and counter, and by the micromap's serial
	// where the entry carries one, GPU audit P3b) into the entry's own storage after the
	// BLAS, cluster_clas_stride bytes apart from cluster_clas_address, on the frames the data
	// changes; an unchanged surface does no work, and no region of the batch is involved.
	bool cluster_templated = false;
	uint64_t cluster_template_key = 0;
	uint64_t cluster_clas_address = 0;
	uint32_t cluster_clas_stride = 0;
	// The infos reference an opacity micromap (D96 OMMd): the BLAS storage was sized, and the batch
	// builds the entry, with the micromap flag, in a build of its own. Templated (GPU audit P3b), the
	// templates reference the micromap and the batch instantiates the entry, flagged, with the other
	// micromapped entries.
	bool cluster_omm = false;
	RID cluster_index_buffer; // The mesh's index buffer the infos read, listed as a source at each rebuild.
	// The cluster LOD hierarchy on the GPU (D87): the entry's region of the shared LOD table
	// (each cluster's group and the group it refines, the groups' bounds and errors, and every
	// level's CLAS info, a build info or, templated, an instantiation info (D108), since a deformed
	// hierarchy has no CLAS until a frame makes the selected ones), the count over every level (a selection's bound, the BLAS storage's size)
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
	// Persistent selection slots (D109), templated: the CLAS the selection instantiates live in the
	// entry's own storage after its BLAS, this many (its LOD 0 count where every cut is bounded by it,
	// else the count over every level), and its region of the table keeps the last selection's digest
	// at this word, so an unchanged selection of an unchanged surface keeps its CLAS and its BLAS.
	uint32_t cluster_lod_slot_capacity = 0;
	uint32_t cluster_lod_digest_offset = UINT32_MAX;
	// The positions changed since its slots were last instantiated (D109): set where owned_vb_full is
	// written, cleared by a selection that instantiated them, and a reason to force the next one. The
	// frame's pending kind alone is not enough: an instance can leave the walk after the copy (its hit
	// group still compiling, a viewport without it) and the next frame's kind no longer says CHANGED.
	bool cluster_lod_slots_stale = true;
	uint8_t cluster_lod_pending_kind = CLUSTER_QUEUED_NONE; // What the batch above would have queued this frame, decided at the TLAS push.
	bool cluster_lod_pending = false; // Set by process_deformed_surface on a selecting frame, taken at the TLAS push.
	bool cluster_lod_last = false; // The last build was a selection: the batch above rebuilds it before tracing it again.
};

// The cluster templates of one version of a deformed mesh surface (D107), shared by every deformed
// entry of it that carries the same micromap, or none (GPU audit P3b: a micromap's templates are a set of
// their own): one template a LOD 0 cluster, built once at explicit addresses into its own storage (the
// first flush after it is made), the build infos kept until then. Freed when no entry holds it.
struct RTDeformedClusterTemplate {
	RID storage;
	uint64_t base_address = 0;
	uint32_t stride = 0; // Bytes between templates.
	uint32_t cluster_count = 0;
	uint32_t refs = 0;
	bool built = false;
	LocalVector<uint8_t> build_infos; // Until built.
	RID index_buffer; // Read by the build.
	uint32_t surface_counter = 0; // The version of the surface they were built from (its RT invalidation counter).
	RID vertex_buffer; // The first entry's positions, the build's hint.
	uint32_t max_triangles = 0;
	uint32_t max_vertices = 0;
	uint32_t total_triangles = 0;
	uint32_t total_vertices = 0;
	uint64_t clas_size = 0; // The worst-case size of one CLAS instantiated from these templates.
	// The micromap they reference (D96 OMMd, GPU audit P3b): every CLAS instantiated from them carries it, so the set is
	// keyed by its serial too, built and instantiated with the micromap flag. 0 and null without one; the RIDs are
	// read by the build and dropped once built.
	uint64_t omm_serial = 0;
	RID micromap;
	RID micromap_index_buffer;
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
	LocalVector<uint32_t> order; // The flush's entry order, those with a micromap first (D96 OMMd); grow-only.
	LocalVector<uint32_t> info_offsets; // Each entry's first byte in clas_infos; grow-only.
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
	LocalVector<uint64_t> pending_templates; // Templates made since the last flush, built first by it (D107).
	// The flush's distinct micromaps, their index buffers and the templates its instantiations read; grow-only (P2c).
	LocalVector<RID> build_micromaps;
	LocalVector<RID> build_micromap_index_buffers;
	LocalVector<RID> build_templates;
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
	// The build budget (D97): the frame the static BLAS was last refused and the frame it began waiting.
	uint64_t budget_frame = UINT64_MAX;
	uint64_t budget_wait_since = 0;
	// On the cluster path (OMMd): the surface's LOD 0 CLAS again, each carrying the micromap, and a
	// cluster BLAS over them, staged then compacted as the surface's own; `blas` is the handle.
	RTSurfaceData *cluster_surface = nullptr;
	bool cluster_failed = false; // Its population failed; not asked for again.
	LocalVector<RID> merged_handles; // The merged MultiMesh entries that carry it.
	LocalVector<RID> deformed_handles; // The deformed cache entries whose triangle BLAS (OMMc) or CLAS (OMMd) carry it.
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
	uint64_t held_since = 0; // frames_drawn the build budget (D97) first held its bake back, 0 when never.
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
	// The viewport's build count at which `transform` was last written or compared equal to its
	// instance's; 0 for never.
	uint32_t verified_build = 0;
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
static_assert(sizeof(RT_SlotRowWrite) == 320, "RT_SlotRowWrite must match ptlas_rows.glsl");

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
	LocalVector<RT_LightData> light_buffer_held; // What light_buffer holds (P2i): the entries past it are zero.
	LocalVector<uint8_t> light_buffer_image; // Builds with asserts (P2i): light_buffer's bytes as its updates left them.
	RID emissive_buffer; // The emissive mesh lights of the frame (D50), RT_EMISSIVE_MAX entries.
	RID light_tree_buffer; // The light tree (D55): a header, the nodes, each emissive entry's cut slot, the cut's nodes and the leaf list (D56), each emissive entry's leaf position and the leaves' alias tables per cut (D60, D61); built or refit per frame while the Environment asks for it.
	RTLightTree light_tree; // What light_tree_buffer holds (P2i).
	LocalVector<uint8_t> light_tree_image; // Builds with asserts (P2i): light_tree_buffer's bytes as its updates left them, checked against a full build.
	uint32_t emissive_count_logged = 0; // The last count printed for this viewport.
	RID decal_buffer; // The decals the hit shaders apply (D104), RT_DECALS_MAX entries; made with the first decal.
	RID decal_node_buffer; // Their bounding volume hierarchy, RT_DECAL_NODES_MAX nodes.
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
	// The material samplers the hit shaders read through (D116): the viewport's anisotropy, as the raster's, but no
	// mipmap bias, which the trace adds to its own levels and gradients.
	RendererRD::MaterialStorage::Samplers samplers;
	// Binding 56: the same twelve as one array (GPU audit hot 5), built when the samplers are made.
	RD::Uniform sampler_array_uniform;

	// Camera-relative tracing: the anchor every position uploaded to the trace
	// is relative to (world minus rt_origin), and the scene UBO copy whose
	// camera carries the relative origin. See `RenderRaytracing::build_tlas`.
	Vector3 rt_origin;
	// The coarse anchor DLSS's camera constants are relative to (1024 m grid).
	Vector3 upscaler_origin;
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
	uint64_t slot_build_frame = 0; // The rasterizer's frame of the last build.
	bool slot_build_follows = false; // The last build came the frame after the one before it.
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
	friend struct RTClusterAsyncPopulation; // Holds an RTClusterLayout (D98c).

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
	// Line surfaces traced as linear swept spheres (D126): the device has SUPPORTS_LINEAR_SWEPT_SPHERES.
	bool lss_enabled = false;
	// The compaction of cluster BLAS into the arena (D81), rendering/pathtracing/cluster_blas_compaction; off, a
	// surface keeps its build-time buffers for life.
	bool cluster_compaction_enabled = false;
	// Deformed surfaces rebuilt from their clusters each frame (D82), rendering/pathtracing/cluster_blas_deformed.
	bool cluster_deformed_enabled = false;
	// Cluster builds recorded this frame, for the render-info counters (they never enter dirty_blas_list).
	uint32_t cluster_blas_builds = 0;
	uint32_t cluster_triangles_built = 0;
	// The deformed cluster path's counters this frame: every rebuild is a build (D97), a cluster BLAS
	// is never refit.
	uint32_t deformed_cluster_builds = 0;
	uint32_t deformed_cluster_triangles_built = 0;
	uint32_t deformed_cluster_expiry_rebuilds_total = 0; // Unchanged surfaces rebuilt for their region's reuse, over the run.
	// Templated entries (D107): a change instantiates the CLAS and rebuilds the BLAS over them, the
	// per-frame update a triangle BLAS's refit is, and is counted as one; templates built over the run.
	uint32_t deformed_cluster_updates = 0;
	uint32_t deformed_cluster_triangles_updated = 0;
	uint32_t deformed_cluster_templates_built_total = 0;
	uint64_t deformed_cluster_clusters_instantiated_total = 0;
	// The templates themselves (D107), by the index buffer's RID and the surface's counter. New
	// templates stop for the run when the driver sizes them at 0 (layouts made after that build
	// their CLAS per frame, as D82); a flush that cannot record turns the whole deformed cluster
	// path off instead, every entry moving to the triangle path, as D82's failure does.
	HashMap<uint64_t, RTDeformedClusterTemplate> deformed_cluster_templates;
	bool deformed_cluster_templates_enabled = false;
	// Micromapped templates (GPU audit P3b) stop for the run when the driver sizes them at 0, alone:
	// micromapped surfaces laid out after that build their CLAS per frame (D82), plain ones keep theirs.
	bool deformed_cluster_templates_omm_enabled = true;
	// Builds the counters above miss (D97): compactions, opacity micromap bakes, the
	// selection's per-instance BLAS and cached levels. Reset with them, before prepare_frame.
	uint32_t frame_extra_builds = 0;
	uint32_t frame_extra_triangles = 0;
	// The CLAS residency loads started this frame (D92), counted on their own (D111): groups,
	// clusters and triangles. The build budget still admits them, so a forced build is a build
	// the budget did not admit among the rest.
	uint32_t frame_stream_loads = 0;
	uint32_t frame_stream_load_clusters = 0;
	uint32_t frame_stream_load_triangles = 0;

	// The acceleration-structure build budget (D97, docs/r1-build-queue.md). A frame's deferrable
	// builds are admitted within a triangle and a build count: prepare_frame's stand-ins (stream
	// loads, opacity micromap bakes, compactions) spend what the previous frame left, the pre-pass
	// admits new surfaces and merges' first bakes by priority from the rest, and the walk's and the
	// flush's stand-ins (an opacity micromap BLAS, a cached level) take what remains. Costs are
	// triangles; a reference to an existing CLAS counts as RT_BUILD_BUDGET_CLUSTER_TRIANGLES.
	static constexpr uint64_t RT_BUILD_BUDGET_CLUSTER_TRIANGLES = 8;
	struct BuildBudget {
		bool enabled = false;
		uint64_t triangles_per_frame = 0;
		uint32_t builds_per_frame = 0;
		uint32_t max_wait_frames = 0;
		uint64_t frame = UINT64_MAX; // frames_drawn of the frame being accounted.
		uint64_t prepass_frame = UINT64_MAX;
		uint64_t carry_triangles = 0;
		uint32_t carry_builds = 0;
		uint64_t left_triangles = 0;
		uint32_t left_builds = 0;
		bool walk_overdue_taken = false; // The walk's one overdue item past the limits this frame.
		bool compaction_overdue_taken = false; // The compactions' one overdue item past the limits this frame.
		// This frame's figures, for the render-info counters.
		uint64_t spent_triangles = 0;
		uint32_t spent_builds = 0;
		uint32_t built = 0; // Builds recorded for admitted items: the rest of the frame's are forced.
		uint32_t deferred = 0; // Admissions refused.
		uint32_t waiting = 0; // New surfaces and merges the frame's decisions left waiting.
		uint32_t held = 0; // Compactions and micromap bakes put back for a later frame.
		// Over the run, and since the last verbose report.
		uint64_t admitted_total = 0;
		uint64_t deferred_total = 0;
		LocalVector<uint32_t> waits; // Frames each item admitted after a wait had waited, at most RT_BUILD_BUDGET_WAITS_CAP.
		uint64_t prepass_usec_total = 0;
		uint64_t prepass_usec_max = 0;
		uint32_t merges_admitted_after_wait = 0; // Since the last report, with the longest of their waits.
		uint32_t merge_wait_max = 0;
		enum CandidateKind : uint8_t {
			CANDIDATE_SURFACE,
			CANDIDATE_MERGE,
		};
		struct Candidate {
			RTCacheEntry *entry = nullptr; // A surface's.
			uint64_t merge_key = 0; // A merge's: MultiMesh local index << 32 | surface index.
			uint64_t merge_id = 0; // The MultiMesh's RID id.
			uint64_t cost = 0;
			uint64_t wait_since = 0;
			float priority = 0.0f;
			uint32_t order = 0;
			CandidateKind kind = CANDIDATE_SURFACE;
			bool overdue = false; // Waited build_budget_max_wait_frames: admitted regardless.
			// Overdue first, oldest first; surfaces before merges; the larger projected size; walk order.
			bool operator<(const Candidate &p_other) const {
				if (overdue != p_other.overdue) {
					return overdue;
				}
				if (overdue && wait_since != p_other.wait_since) {
					return wait_since < p_other.wait_since;
				}
				if (kind != p_other.kind) {
					return kind < p_other.kind;
				}
				if (priority != p_other.priority) {
					return priority > p_other.priority;
				}
				return order < p_other.order;
			}
		};
		// Frame-local, cleared and refilled by each pre-pass with their capacity kept.
		LocalVector<Candidate> candidates;
		AHashMap<RTCacheEntry *, uint32_t> candidate_surfaces; // Into candidates.
		AHashMap<uint64_t, uint32_t> candidate_merges;
		AHashMap<uint64_t, bool> merges_waiting; // This frame's; true when the merge waits, false when admitted.
		struct MergeWait {
			uint64_t mm_id = 0; // The MultiMesh's RID id: a recycled local index starts a new wait.
			uint64_t since = 0;
			uint64_t seen = 0;
		};
		AHashMap<uint64_t, MergeWait> merge_waits;
		LocalVector<uint64_t> stale_merges;
	} build_budget;
	static constexpr uint32_t RT_BUILD_BUDGET_WAITS_CAP = 65536;
	// An admission: refused, within the limits, or past them (a forced item).
	enum BuildBudgetTake : uint8_t {
		BUILD_BUDGET_REFUSED,
		BUILD_BUDGET_ADMITTED,
		BUILD_BUDGET_ADMITTED_PAST,
	};
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
	LocalVector<RTClusterCompactionInFlight> cluster_compactions_in_flight; // D98b.
	void _finish_cluster_compaction(const RTClusterCompactionInFlight &p_compaction);
	void _finish_cluster_compactions_in_flight();

	// The async build queue (D98b): the admitted new surfaces' triangle BLAS the walk queued for the
	// compute queue, recorded in an async build scope after the walk, and the frame's builds recorded
	// there (compactions included), counted as RT_ASYNC_BUILDS.
	struct AsyncBlas {
		RID blas;
		uint32_t triangles = 0;
		RTSurfaceData *surf = nullptr; // Stamped ready once recorded; a release takes it off the list.
	};
	LocalVector<AsyncBlas> async_blas_list;
	uint32_t frame_async_builds = 0;
	uint64_t async_builds_total = 0;
	uint64_t async_ready_next_frame = 0; // In-flight surfaces the walk reached in the frame after their build.
	uint64_t async_ready_later = 0;
	// A traced instance reaching an in-flight surface (D97's rule): its build taken back to the main
	// queue, or, recorded on the compute queue by an earlier viewport, the instance out for a frame.
	uint64_t async_pulled_back = 0;
	uint64_t async_forced_drops = 0;
	void _record_async_builds();

	// Surfaces the walk leaves out of the trace: its BLAS takes triangle lists, and line lists with a float
	// radius in CUSTOM0 as linear swept spheres where the device has them (D126); points, line strips,
	// triangle strips and other lines would read as other primitives. Run total, printed at exit.
	uint64_t non_triangle_surfaces_skipped = 0;
	bool _surface_traced(RSE::PrimitiveType p_primitive, void *p_mesh_surface) const;
	bool _skip_untraced_surface(RSE::PrimitiveType p_primitive, void *p_mesh_surface);

	// The cluster path on the async build queue (D98c). An admitted new surface's population runs in
	// the walk with cluster_capture set: its CPU work and its handles are made there, and its commands
	// are held (RTClusterAsyncPopulation) and recorded after the walk, in the scope, or on the main
	// queue where the main graph wrote a source this frame. Its LOD table writes land on the main queue
	// in the next frame's prepare_frame, before the walk that may trace it. A stream load's issue
	// records in a scope of its own.
	RTClusterAsyncPopulation *cluster_capture = nullptr;
	LocalVector<RTClusterAsyncPopulation *> cluster_async_populations; // In the walk's order.
	bool walk_in_compute_list = false; // Phase 2's merges: no build records here, a pull-back included.
	uint64_t async_triangle_builds_total = 0;
	uint64_t async_compactions_total = 0;
	uint64_t async_cluster_populations_total = 0;
	uint64_t async_cluster_fallbacks = 0;
	uint64_t async_stream_issues_total = 0;
	uint64_t async_stream_fallbacks = 0;
	Error _cluster_buffer_update(RID p_buffer, uint32_t p_offset, uint32_t p_size, const void *p_data);
	Error _cluster_table_update(uint32_t p_offset, uint32_t p_size, const void *p_data);
	Error _cluster_build(const RD::ClusterAccelerationStructureBuild &p_build);
	Error _cluster_sizes_readback(RID p_buffer, const Callable &p_callback, uint32_t p_offset, uint32_t p_size);
	bool _record_cluster_population(RTClusterAsyncPopulation *p_population);
	bool _finish_cluster_population(RTClusterAsyncPopulation *p_population);
	void _finish_cluster_populations_in_flight();
	void _drop_cluster_population(RTSurfaceData *p_surf_data);

	// D98a's concurrency probe (the debug argument --async-probe): one synthetic BLAS of about a
	// million triangles, rebuilt in an async build scope every frame and never traced, so the compute
	// queue builds beside the trace as D98's moved builds will.
	struct AsyncProbe {
		bool enabled = false;
		RID vertices;
		RID indices;
		RID blas;
		uint64_t frame = 0;
		uint64_t builds = 0;
		uint64_t async_builds = 0;
	} async_probe;
	void _async_probe_step();
	void _async_probe_free();

	// The build timer (the debug argument --rt-build-timer=<path>, D97's budget sizing and D98d's
	// overlap): timestamps on the main queue between the phases of build_tlas, the compute submit's span,
	// the deformation passes' spans (D122 S3, marks written by MeshStorage::update_mesh_instances()) and
	// the frame's build counters, one CSV row per frame, the timestamps read once per swap
	// (frame_ended()). Each mark orders every command recorded before it ahead of every command after
	// it, behind a wait for all of them, so a run with marks is compared only with another run with
	// marks; --rt-build-timer-marks=off drops them and keeps the viewport span, the compute span and
	// the counters, for a frame-time comparison.
	struct BuildTimerRow {
		uint64_t frame = UINT64_MAX;
		uint64_t cpu_usec = 0;
		int64_t cpu_frame_usec = -1; // From the row created before it; -1 for the first.
		bool written = false;
		uint32_t builds = 0;
		uint32_t triangles = 0;
		uint32_t async_builds = 0;
		uint32_t forced = 0;
		uint32_t budget_triangles = 0;
		uint32_t refits = 0;
		uint32_t stream_loads = 0;
		uint32_t stream_load_triangles = 0;
		uint64_t async_populations_total = 0;
		uint64_t async_stream_issues_total = 0;
		uint64_t async_compactions_total = 0;
	};
	static constexpr uint32_t BUILD_TIMER_ROWS = 16;
	struct BuildTimer {
		Ref<FileAccess> file;
		bool marks = true;
		bool compute = false; // The compute submits are timed too.
		BuildTimerRow rows[BUILD_TIMER_ROWS];
		uint64_t last_row_cpu_usec = 0;
		uint64_t rows_written = 0;
		uint64_t sets_lost = 0; // Frames that ran build_tlas whose timestamps a flush discarded before a swap read them.
	} build_timer;
	void _build_timer_mark(uint32_t p_phase);
	void _build_timer_read();
	void _build_timer_write_row(BuildTimerRow &r_row, const String &p_timing);
	BuildTimerRow &_build_timer_row();

	// Opacity micromaps (D96), with rendering/pathtracing/opacity_micromaps.
	bool omm_enabled = false;
	bool omm_indirect_2_state = false;
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
	// above and cluster_blas_deformed; their selected CLAS are built, or instantiated from templates
	// (D108), each frame into explicit slots of this size (the worst case of one cluster under the D80
	// limits, of a build or an instantiation) in a region of the deformed batch's pool.
	bool cluster_lod_deformed_enabled = false;
	// The deformed selection instantiates its CLAS from per-level templates (D108), decided once: the
	// table holds every hierarchy's instantiation infos then, and a hierarchy without templates is
	// traced at LOD 0. Off where the device has no templates or sizes an instantiation at 0 bytes.
	bool cluster_lod_deformed_templated = false;
	// The persistent selection (D109): the updates and their triangles the device counted, read back a
	// frame queue later (RenderingDevice's frame delay, 2 frames by default) and published by the next
	// flush; the hierarchies whose cuts the LOD 0 count does not bound, over the run.
	uint32_t cluster_lod_persistent_ceiling_slots = 0; // The slots one frame's persistent items may take (D109).
	uint32_t lod_deformed_updates_read = 0;
	uint64_t lod_deformed_triangles_read = 0;
	uint64_t lod_deformed_updates_total = 0;
	uint64_t lod_deformed_instances_total = 0;
	uint32_t deformed_cut_bound_failures_total = 0;
	// The persistent static selection (D113): the rebuilds the device counted (first builds excluded,
	// the CPU counts those), read back a frame queue later and published by the next flush as builds;
	// the free digest slots, taken from the LOD table RT_LOD_DIGEST_BLOCK at a time and never given back.
	uint32_t lod_static_rebuilds_read = 0;
	uint64_t lod_static_rebuilds_total = 0;
	uint64_t lod_static_instances_total = 0;
	LocalVector<uint32_t> lod_digest_free;
	// The static instances' own slots (D113), kept while an instance traces another's build under
	// rendering/pathtracing/cluster_lod_slot_budget_mb, least recently built freed first: the bytes
	// live, the budget, the evictions over the run and the eviction's grow-only scratch.
	uint64_t cluster_lod_slot_bytes = 0;
	uint64_t cluster_lod_slot_budget = 0;
	uint64_t cluster_lod_slot_evictions_total = 0;
	struct RTLodSlotEvictCandidate {
		uint32_t last_built = 0;
		RTLodInstanceEntry *entry = nullptr;
		bool operator<(const RTLodSlotEvictCandidate &p_other) const { return last_built < p_other.last_built; }
	};
	LocalVector<RTLodSlotEvictCandidate> cluster_lod_slot_evict_scratch;
	LocalVector<RID> pool_owned_scratch; // prepare_frame's TTL sweeps: a pool's RIDs, grow-only (P2c).
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
	// that have held a level, which the memory budget's eviction walks (D112), and the totals for the report.
	bool cluster_lod_cache_enabled = false;
	HashSet<RTSurfaceData *> cluster_lod_cached_surfaces;
	uint32_t cluster_lod_cache_live = 0;
	uint64_t cluster_lod_cache_bytes = 0;
	uint64_t cluster_lod_cache_builds_total = 0;
	// The cached levels' memory budget (D112), rendering/pathtracing/cluster_lod_cache_budget_mb: kept
	// until together they pass it, then freed least recently traced first; the evictions and the
	// rebuilds of evicted levels over the run.
	uint64_t cluster_lod_cache_budget = 0;
	uint64_t cluster_lod_cache_evictions_total = 0;
	uint64_t cluster_lod_cache_rebuilds_total = 0;
	// The eviction's scratch, grow-only: the levels outside the working set, oldest first, and the
	// surfaces left with none.
	struct RTLodCacheEvictCandidate {
		uint32_t last_used = 0;
		RTSurfaceData *surface = nullptr;
		uint32_t depth = 0;
		bool operator<(const RTLodCacheEvictCandidate &p_other) const { return last_used < p_other.last_used; }
	};
	LocalVector<RTLodCacheEvictCandidate> cluster_lod_cache_evict_scratch;
	LocalVector<RTSurfaceData *> cluster_lod_cache_emptied_scratch;
	// The traversal (D91), rendering/pathtracing/cluster_lod_traversal with the selection; a verbose
	// run checks it against the exhaustive test of the same frame, these its tallies.
	bool cluster_lod_traversal_enabled = false;
	uint64_t lod_tree_checks = 0;
	uint64_t lod_tree_mismatches = 0;
	uint64_t lod_select_timer_frame = 0; // The captured frame last read for the selection's timing.
	// A verbose run's GPU spans bracketing the sky table's build and its moments (GPU audit 5), per frame that built them.
	LocalVector<float> sky_table_gpu_us;
	LocalVector<float> sky_moments_gpu_us;
	uint64_t sky_timer_frame = 0; // The captured frame last read for them.
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
	// A device with position fetch: every triangle and cluster build allows data access, so the hit stages take the
	// hit triangle's corners from the acceleration structure (P1d); geometry residency (D93) relies on it.
	bool build_data_access = false;
	bool cluster_lod_stream_geometry_enabled = false;
	// A verbose run's position-fetch check (D93a): the closest hit's counts of cluster hits whose
	// fetched positions it compared with the vertex buffer's, of those that differ and the largest
	// difference, by kind (RT_POSITION_FETCH_CHECK_WORDS words, binding 46), cleared after each read.
	RID position_fetch_check_buffer;
	// The subsurface walk's counters (D120, RT_SUBSURFACE_STATS_WORDS words, binding 52), made by the first verbose
	// frame that compiles the walk in, read and cleared on the verbose report's cadence.
	RID subsurface_stats_buffer;
	// The inline shadow ray's counters (P1k, RT_SHADOW_STATS_WORDS words, binding 53), made by the first frame of a verbose
	// process that traces shadow rays as queries.
	RID shadow_stats_buffer;
	// The surface record's codec check (P3a, RT_SURFACE_CODEC_CHECK_WORDS words, binding 55), made by an asserts build's
	// first frame, read and cleared on the verbose report's cadence.
	RID surface_codec_check_buffer;
	// What the share by variance's moments read of each cell of the 256-grid (GPU audit 5): two floats a cell, written
	// and read within one rebuild, so one for every viewport; made by the first rebuild of the moments.
	RID sky_light_cells_buffer;
	// Binding 43 while a frame has no sky light (GPU audit 4e): the table's fixed part alone, cleared, which the shaders
	// read as no sky, shared by every viewport in place of a 5.6 MB table each.
	RID sky_light_placeholder;
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
		uint64_t deferred_build_budget = 0; // Wants the build budget (D97) held back.
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
	// variance's moments: the cells of the 256-grid read once, in the form the trace reads the octmap in
	// (the roughness layers as an array, or one texture whose mips are the roughness levels), since the
	// fog's level is read as the trace reads it (D64), then the moments per receiver normal (D65) and
	// candidate share (GPU audit 5); they carry the frame's weights' shape and are clipped at multiples
	// of the receiver's irradiance (D72), so there is no combine pass.
	struct SkyLightShader {
		enum Mode {
			MODE_LEAF = 0,
			MODE_REDUCE = 1,
			MODE_HEADER = 2,
			MODE_CELLS_OCTMAP_2D = 3, // The moments' cells (GPU audit 5), in the form the trace reads the octmap in.
			MODE_CELLS_OCTMAP_ARRAY = 4,
			MODE_MOMENTS = 5,
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
	// The moments' cells (GPU audit 5): the 256-grid, two floats each.
	static constexpr uint32_t RT_SKY_LIGHT_CELLS = (1u << RT_SKY_LIGHT_CELL_LEVEL) * (1u << RT_SKY_LIGHT_CELL_LEVEL);
	static constexpr uint32_t RT_SKY_LIGHT_CELLS_SIZE = RT_SKY_LIGHT_CELLS * 2 * sizeof(float);
	// Everything before the pyramid's runtime array: the least a binding of the table's block may hold.
	static constexpr uint32_t RT_SKY_LIGHT_FIXED_SIZE = (RT_SKY_LIGHT_HEADER_WORDS + RT_SKY_LIGHT_SHARES + RT_SKY_LIGHT_MOMENT_FLOATS) * sizeof(float);
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
	void _gather_light_tree_leaves(const RT_LightData *p_lights, uint32_t p_light_count, uint32_t p_directional_count, const RT_EmissiveInstance *p_emissive, uint32_t p_emissive_count, float p_exposure); // The frame's leaves into light_tree_leaves (D55).
	uint32_t _build_light_tree_node(RTLightTree &r_tree, uint32_t p_begin, uint32_t p_end); // The light tree's recursive build (D55).
	void _build_light_tree_cut(RTLightTree &r_tree, uint32_t p_cut, uint32_t p_cut_size); // A cut the shader scores per hit (D56), the primary (0) or the indirect one (1, D59).
	void _light_tree_cut_alias(RTLightTree &r_tree, uint32_t p_first, uint32_t p_count, uint32_t p_alias_base); // The alias table over one cut node's leaves (D60, D61).
	void _light_tree_map_emissive(RTLightTree &r_tree, uint32_t p_emissive_count); // Each emissive entry's cut slots and leaf position.
	bool _refit_light_tree(RTLightTree &r_tree, uint32_t p_emissive_count, uint32_t p_cut_size, uint32_t p_cut_indirect_size, bool p_power_pick); // P2i: false when the structure changed.
	void _release_deformed_blas(RTDeformedCacheEntry &p_entry);
	void _free_deformed_entry(RTDeformedCacheEntry &p_entry);
	// The deformed triangle BLAS over owned_vb_full, carrying the micromap the walk wants (D96 OMMc).
	void _build_deformed_triangle_blas(RID p_handle, RTDeformedCacheEntry &r_entry, void *p_mesh_surface, uint32_t p_cache_key, uint32_t p_surface_counter, LocalVector<RID> &r_dirty_blas_list);
	// The deformed cluster layout, its CLAS carrying the micromap the walk wants (D96 OMMd); as _populate_deformed_cluster_layout.
	bool _build_deformed_cluster_layout(RID p_handle, RTDeformedCacheEntry &r_entry, void *p_mesh_surface, uint32_t p_cache_key, uint32_t p_surface_counter);
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
	// The decals' gather and hierarchy (D104): each decal's record in rank order, its trace-space box,
	// the build's index permutation, and the records and nodes in leaf order as uploaded.
	struct DecalCandidate {
		RID instance;
		float depth; // Camera distance minus the sorting offset, the raster's sort key.
		bool operator<(const DecalCandidate &p_other) const { return depth < p_other.depth; }
	};
	LocalVector<DecalCandidate> decal_candidates;
	LocalVector<RT_DecalData> decal_records;
	LocalVector<AABB> decal_bounds;
	LocalVector<uint32_t> decal_order;
	LocalVector<RT_DecalData> decal_leaf_records;
	LocalVector<RT_DecalNode> decal_nodes;
	uint32_t _build_decal_node(uint32_t p_first, uint32_t p_count); // One node of the decals' hierarchy over decal_order[p_first, p_first + p_count), its index.
	// The light tree's scratch (D55): the frame's leaves, a leaf per positional light and per emissive
	// entry, and the build's and the refit's working lists; the tree itself is the viewport's (P2i).
	LocalVector<RTLightTreeLeaf> light_tree_leaves;
	LocalVector<uint64_t> light_scratch_keys; ///< Each gathered light's instance RID (P2i), parallel to light_scratch.
	LocalVector<uint8_t> light_upload_marks; ///< The light upload's marks (P2i): an entry that differs from what the buffer holds.
	// The light buffer's uploads over the frames since the last verbose report (P2i).
	struct LightBufferStats {
		uint64_t bytes = 0;
		uint32_t updates = 0;
		uint32_t entries = 0; // Entries written.
	} light_buffer_stats;
	LocalVector<uint32_t> light_tree_order; ///< Leaf indices, partitioned in place by the build.
	LocalVector<uint32_t> light_tree_cut_queue; ///< The cut's breadth-first work queue.
	LocalVector<uint32_t> light_tree_leaf_positions; ///< The refit's match (P2i): each leaf's position in the tree's order.
	LocalVector<uint8_t> light_tree_node_changed; ///< The refit's marks (P2i): a node rewritten this frame.
	LocalVector<uint8_t> light_tree_power_changed; ///< The refit's marks (P2i): a position whose leaf's power changed.
	LocalVector<uint8_t> light_tree_alias_changed; ///< The refit's marks (P2i): an alias entry rebuilt, the primary cut's then the indirect's.
	bool light_tree_emissive_moved = false; ///< The refit's mark (P2i): an emitter's index moved, so the emissive maps were redone.
	RTLightTree light_tree_check; ///< A full build the refit is compared with in builds with asserts (P2i).
	// The light tree's work over the frames since the last verbose report (P2i).
	struct LightTreeStats {
		uint32_t builds = 0;
		uint32_t refits = 0;
		uint32_t unchanged = 0; // Refits that changed nothing.
		uint64_t nodes_written = 0; // Nodes and cut copies uploaded.
		uint64_t bytes = 0;
		uint32_t updates = 0; // Buffer updates recorded.
		uint64_t build_usec = 0;
		uint64_t refit_usec = 0;
		uint64_t usec_max = 0;
	} light_tree_stats;
	LocalVector<double> light_tree_alias_scaled; ///< The alias build's scratch: each leaf's share times its node's count, indexed by leaf-list position.
	LocalVector<double> emissive_alias_scaled; ///< The emissive alias table's scratch (P1i): each entry's pdf times the count; its lists are the tree's.
	LocalVector<uint32_t> light_tree_alias_small; ///< The alias build's scratch: positions whose scaled share is below 1.
	LocalVector<uint32_t> light_tree_alias_large; ///< The alias build's scratch: positions whose scaled share is 1 or more.
	// Camera and bounce rays select all geometry. Shadow rays select bit 1;
	// the G-buffer primary probe selects only procedural geometry (bit 2).
	// Keep these in sync with RT_MASK_* in raytracing_common_inc.glsl.
	static constexpr uint8_t RT_INSTANCE_MASK_GEOMETRY = 0x01;
	static constexpr uint8_t RT_INSTANCE_MASK_SHADOW = 0x02;
	static constexpr uint8_t RT_INSTANCE_MASK_PROCEDURAL = 0x04;
	static constexpr uint8_t RT_INSTANCE_MASK_SUBSURFACE = 0x08; // A material whose diffuse lobe walks (D120); the walk's rays select it alone.
	LocalVector<uint32_t> sbt_offsets; // 0 = default material hit group, SceneShaderRaytracing::HIT_SBT_WALK_RECORD its walking variant

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
	// A cached row's generation with the instance's layers folded in (D104): its slot's row carries
	// them, so a change of layers alone re-emits it. Injective while generations stay under 2^44.
	static _FORCE_INLINE_ uint64_t _layer_row_generation(uint64_t p_generation, uint32_t p_layers) {
		return (p_generation << 20) | (p_layers & RT_GEOMETRY_LAYERS_MASK);
	}
	static void _slot_relative_transform(const Transform3D &p_transform, const Vector3 &p_origin, float *r_rows);
	// A cell's origin (D94c): its corner on x and z, at y = 0, computed in double.
	_FORCE_INLINE_ Vector3 _slot_cell_origin(const Vector3i &p_cell) const {
		return Vector3(real_t(double(p_cell.x) * ptlas_cell_size), real_t(0), real_t(double(p_cell.z) * ptlas_cell_size));
	}
	// The walk's change test. The transform is narrowed as the shadow holds it, relative to the
	// shadow's cell (D94c), so an rt_origin shift alone changes nothing here. `p_static`: the
	// instance's transform has not been set since the frame before (TransformStatus::NONE), so where
	// the viewport built that frame too and the slot's transform was verified then, it still holds.
	// The procedural and MultiMesh call sites leave it false on purpose: the skip covers the mesh walk alone.
	_FORCE_INLINE_ bool _slot_unchanged(RTViewportState *p_state, uint32_t p_slot, RID p_blas, const Transform3D &p_transform, uint32_t p_flags, uint8_t p_mask, uint32_t p_sbt_offset, uint64_t p_geometry_generation, uint64_t p_material_generation, bool p_static = false) const {
		RTSlotShadow &shadow = p_state->slots[p_slot];
		if (!(shadow.blas == p_blas && shadow.rows_valid && shadow.geometry_generation == p_geometry_generation && shadow.material_generation == p_material_generation && shadow.flags == uint8_t(p_flags) && shadow.mask == p_mask && shadow.sbt_offset == p_sbt_offset)) {
			return false;
		}
		if (!(p_static && p_state->slot_build_follows && shadow.verified_build != 0 && shadow.verified_build + 1 == p_state->slot_build)) {
			float rows[12];
			_slot_relative_transform(p_transform, _slot_cell_origin(shadow.cell), rows);
			if (memcmp(shadow.transform, rows, sizeof(rows)) != 0) {
				return false;
			}
		}
#ifdef DEV_ENABLED
		float dev_rows[12];
		_slot_relative_transform(p_transform, _slot_cell_origin(shadow.cell), dev_rows);
		DEV_ASSERT(memcmp(shadow.transform, dev_rows, sizeof(dev_rows)) == 0);
#endif
		shadow.verified_build = p_state->slot_build;
		return true;
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
	bool _plan_merged_mm(RID p_mm_rid, RID p_mm_gpu_buffer, void *p_mesh_surface, uint32_t p_mm_count, uint32_t p_surface_index, uint32_t p_surface_counter, bool &r_bakes, bool *r_first = nullptr);
	// The build budget (D97).
	void _build_budget_begin_frame();
	// A GeometryInstanceForwardClustered's traced surfaces (D97), passed untyped as process_surface's surface is.
	static void _build_budget_stamp(const void *p_instance, uint32_t p_surface_index, uint64_t p_frames_drawn);
	static bool _build_budget_traced(const void *p_instance, uint32_t p_surface_index, uint64_t p_frames_drawn);
	bool _build_budget_fits(uint64_t p_cost, bool p_carry) const;
	void _build_budget_spend(uint64_t p_cost, bool p_carry);
	BuildBudgetTake _build_budget_take(uint64_t p_cost, bool p_carry, bool p_force);
	bool _build_budget_take_walk(uint64_t p_cost, uint64_t &r_budget_frame, uint64_t &r_wait_since);
	void _build_budget_note_wait(uint64_t p_since);
	void _build_budget_consider_surface(RID p_mesh, uint32_t p_surface_index, void *p_mesh_surface, float p_priority);
	void _build_budget_prepass(const RenderDataRD *p_render_data, uint32_t p_rt_flags);
	bool _build_budget_admit_surface(RTCacheEntry *p_entry, void *p_mesh_surface);
	_FORCE_INLINE_ bool _build_budget_merge_waits(RID p_mm_rid, uint32_t p_surface_index) const {
		if (!build_budget.enabled) {
			return false;
		}
		const bool *waits = build_budget.merges_waiting.getptr((uint64_t(p_mm_rid.get_local_index()) << 32) | p_surface_index);
		return waits != nullptr && *waits;
	}
	_FORCE_INLINE_ bool _build_budget_merge_admitted(RID p_mm_rid, uint32_t p_surface_index) const {
		if (!build_budget.enabled) {
			return false;
		}
		const bool *waits = build_budget.merges_waiting.getptr((uint64_t(p_mm_rid.get_local_index()) << 32) | p_surface_index);
		return waits != nullptr && !*waits;
	}
	// With p_omm (D96 OMMd), the LOD 0 clusters alone, each CLAS carrying the micromap: no hierarchy, no streaming.
	bool _populate_surface_cluster_blas(void *p_mesh_surface, RID p_vertex_buffer, RID p_index_buffer, uint32_t p_cache_key, RTSurfaceData *r_surf_data, bool p_blobs_only = false, const RTOmmEntry *p_omm = nullptr);
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
	// With p_omm (D96 OMMd) every CLAS carries the micromap, and the entry takes no hierarchy (D87).
	bool _populate_deformed_cluster_layout(void *p_mesh_surface, uint32_t p_cache_key, uint32_t p_surface_counter, RTDeformedCacheEntry &r_entry, const RTOmmEntry *p_omm = nullptr, bool *r_omm_attached = nullptr);
	void _queue_deformed_cluster_rebuild(RTDeformedCacheEntry &p_entry, uint8_t p_kind);
	// Takes the entry back out of this frame's batch when it is the last one queued; false if it is not.
	bool _unqueue_deformed_cluster_rebuild(RTDeformedCacheEntry &p_entry);
	bool _deformed_cluster_region_expired(const RTDeformedCacheEntry &p_entry, uint64_t p_frames_drawn) const;
	// A region of the pool for this flush's CLAS: one whose last trace has been waited on, grown
	// to p_alloc if too small, or a new one; -1 if none could be made. The batch and the
	// selection (D87) share the pool.
	int32_t _deformed_cluster_region_acquire(uint64_t p_alloc, uint64_t p_alignment);
	bool _record_deformed_cluster_batch();
	// The RID lists the frame's cluster builds take, kept from frame to frame (P2c).
	enum ClusterBuildList {
		CLUSTER_BUILD_LIST_SOURCES,
		CLUSTER_BUILD_LIST_STORAGE,
		CLUSTER_BUILD_LIST_TEMPLATES,
		CLUSTER_BUILD_LIST_HANDLES,
		CLUSTER_BUILD_LIST_MICROMAPS,
		CLUSTER_BUILD_LIST_SLICE_STORAGE,
		CLUSTER_BUILD_LIST_SLICE_HANDLES,
		CLUSTER_BUILD_LIST_SLICE_SOURCES,
		CLUSTER_BUILD_LIST_MAX,
	};
	Vector<RID> cluster_build_lists[CLUSTER_BUILD_LIST_MAX];
	const Vector<RID> &_cluster_build_list(ClusterBuildList p_list, const RID *p_rids, uint32_t p_count, const RID *p_more = nullptr, uint32_t p_more_count = 0);
	void _flush_deformed_cluster_batch();
	void _release_deformed_cluster_storage(RTDeformedCacheEntry &p_entry);
	// The templates of this version of the mesh surface (D107), made on first use and queued for
	// the next flush's build; nullptr if they cannot be. With p_omm, those of its micromap (GPU audit P3b),
	// p_omm_address and p_omm_index_address its addresses. r_key is what the entry releases them by, once.
	RTDeformedClusterTemplate *_deformed_cluster_template_acquire(RID p_index_buffer, uint32_t p_surface_counter, RID p_vertex_buffer, const RTClusterLayout &p_layout, uint32_t p_cluster_count, uint64_t p_index_address, uint32_t p_index_format, uint64_t p_vertex_address, uint32_t p_vertex_stride, const RTOmmEntry *p_omm, uint64_t p_omm_address, uint64_t p_omm_index_address, uint64_t &r_key);
	void _deformed_cluster_template_release(uint64_t p_key);
	bool _build_deformed_cluster_templates();
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
	// A cluster surface's (OMMd): the LOD 0 CLAS again with the micromap, a cluster BLAS over them.
	RID _omm_surface_cluster_blas(RTOmmEntry *p_entry, void *p_mesh_surface, const RTSurfaceData *p_surf_data);
	// A CLAS info's micromap fields for a surface of p_primitive_count primitives (OMMd): the micromap's
	// address and the address of the index of primitive 0; false where the micromap does not match.
	bool _omm_cluster_addresses(const RTOmmEntry *p_entry, uint32_t p_primitive_count, uint64_t &r_micromap_address, uint64_t &r_index_address) const;
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
	void _populate_surface_cluster_lod(const Vector<uint8_t> &p_cluster_data, const RTClusterLayout &p_layout, RID p_args_buffer, uint64_t p_implicit_addresses_offset, uint32_t p_cache_key, RTSurfaceData *r_surf_data);
	// The hierarchy with a streamed surface's residency (D92), or its coarsest cut where it cannot be placed.
	void _attach_surface_cluster_lod(const Vector<uint8_t> &p_cluster_data, const RTClusterLayout &p_layout, RTLodStream *p_stream, RID p_args_buffer, uint64_t p_implicit_addresses_offset, uint32_t p_cache_key, RTSurfaceData *r_surf_data);
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
	// The slots a persistent selection needs (D109): the LOD 0 count where no group is replaced by more
	// clusters than it has, so that no cut is larger than LOD 0, else the count over every level.
	uint32_t _deformed_cluster_cut_capacity(void *p_mesh_surface, uint32_t p_lod0_clusters, uint32_t p_all_clusters);
	// The persistent selection's counts (D109), on the render thread a frame queue after the frame:
	// the instances it re-instantiated (first builds excluded) and their triangles, for the next
	// flush to publish (tools builds, where the counters are), and the clusters past an item's slots,
	// which a sound cut bound never gives.
	static void _on_lod_deformed_counts_read(const PackedByteArray &p_bytes, uint32_t p_instances);
	static void _on_lod_static_counts_read(const PackedByteArray &p_bytes, uint32_t p_instances);
	// The deformed entries on the selection (D87): the entry's region of the LOD table, made with
	// its layout and released with its storage, and its record in the frame's batch.
	bool _populate_deformed_cluster_lod(void *p_mesh_surface, const RTClusterLayout &p_layout, const LocalVector<uint8_t> &p_infos, uint32_t p_info_size, uint32_t p_cache_key, RTDeformedCacheEntry &r_entry);
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
	// Any build's flags, triangle or cluster (the name is D93's): with data access where the device has position
	// fetch (P1d), which geometry residency (D93) needs.
	_FORCE_INLINE_ BitField<RD::AccelerationStructureFlagBits> _cluster_build_flags(BitField<RD::AccelerationStructureFlagBits> p_base) const {
		BitField<RD::AccelerationStructureFlagBits> flags = p_base;
		if (build_data_access) {
			flags.set_flag(RD::ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_BIT);
		}
		return flags;
	}
	// A cluster build whose CLAS reference opacity micromaps (D96 OMMd): only those builds are flagged,
	// since the flag sizes every structure of the build for a micromap.
	_FORCE_INLINE_ BitField<RD::AccelerationStructureFlagBits> _cluster_build_flags_omm(RD::AccelerationStructureFlagBits p_base, bool p_omm) const {
		BitField<RD::AccelerationStructureFlagBits> flags = _cluster_build_flags(p_base);
		if (p_omm) {
			flags.set_flag(RD::ACCELERATION_STRUCTURE_ALLOW_CLUSTER_OPACITY_MICROMAPS_BIT);
		}
		return flags;
	}
	static void _on_position_fetch_check_read(const PackedByteArray &p_bytes);
	static void _on_subsurface_stats_read(const PackedByteArray &p_bytes);
	static void _on_shadow_stats_read(const PackedByteArray &p_bytes);
	static void _on_surface_codec_check_read(const PackedByteArray &p_bytes);
	static void _mesh_surface_freed(RID p_mesh, uint32_t p_surface);
	// Frees the surface's BLAS and, on the cluster path, the staging buffers or the arena region it lives in.
	void _release_surface_data(RTSurfaceData *p_surf_data);
	RTMaterialData *process_material(RID p_material_rid, uint16_t p_material_invalidation_counter);
	// The material a surface cache (`p_surf`, a GeometryInstanceSurfaceDataCache) traces with: the
	// instance's override, else its surface material, else its mesh's (a MultiMesh's mesh's) surface material.
	static RID _resolve_surface_material(const void *p_surf);
	// Its RT data, through process_material once and then from the surface cache while the material
	// lives with the same RT counter; a custom shader's material is processed every frame still, which
	// keeps its hit group slot current.
	RTMaterialData *_surface_material(const void *p_surf);
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
	// The build timer reads the timestamps a swap brought back (D98d); RendererCompositorRD calls it.
	static void frame_ended();
	// The device has position fetch: every build allows data access and the hit stages compile with it (P1d).
	static bool is_position_fetch_enabled() { return singleton != nullptr && singleton->build_data_access; }
	// Geometry residency (D93) is on: a verbose run's closest hit checks fetched positions against the vertex buffer.
	static bool is_position_fetch_check_enabled() { return singleton != nullptr && singleton->cluster_lod_stream_geometry_enabled; }
	// Opacity micromaps are on and rays past the first diffuse bounce trace them in 2-state (D96 OMMe).
	static bool is_omm_indirect_2_state_enabled() { return singleton != nullptr && singleton->omm_enabled && singleton->omm_indirect_2_state; }

	void cleanup_caches();
	// Releases a geometry instance's PTLAS slot ranges (D94b) as it is freed.
	void release_slot_ranges(RTSlotOwner &r_owner);
	// Releases the ranges of surface indices from p_surface_count on, as a mesh swapped in place drops them (D95).
	void release_slot_ranges_from(RTSlotOwner &r_owner, uint32_t p_surface_count);
	// Releases a deformed entry no surface cache holds any more (its mesh swapped out, its instance freed) at once,
	// as the expiry would deformed_mesh_cache_ttl_frames later (D122 S5).
	void release_deformed_handle(RID p_handle);
	// Releases the RT cache entry of a surface its mesh is dropping (MeshStorage's surface-freed callback).
	void release_mesh_surface(RID p_mesh, uint32_t p_surface);

	// p_primary_from_gbuffer: the frame's visible surface is the raster's, so the cluster LOD selection stands down (D86).
	RTViewportState *build_tlas(const RenderDataRD *p_render_data, uint32_t p_rt_flags, bool p_primary_from_gbuffer = false);
	uint32_t gather_lights(const RenderDataRD *p_render_data, RT_LightData *r_light_data, uint32_t p_max_lights, const Vector3 &p_origin, float &r_score_sum, uint32_t &r_directional_count);
	uint32_t gather_emissive(const RenderDataRD *p_render_data, const RTViewportState *p_state, RT_EmissiveInstance *r_emissive_data, uint32_t p_max_emissive, float &r_score_sum);
	// The decals the hit shaders apply (D104): gathered from the trace's cull, uploaded with their hierarchy; returns the count.
	uint32_t gather_decals(RTViewportState *p_state, const RenderDataRD *p_render_data);
	uint32_t build_light_tree(RTLightTree &r_tree, uint32_t p_emissive_count, uint32_t p_cut_size, uint32_t p_cut_indirect_size, bool p_power_pick); // A full build over light_tree_leaves (D55).
	// The frame's light tree into the viewport's buffer: refit in place when the leaves are the last
	// frame's lights at the same places, built and uploaded whole otherwise (P2i).
	void update_light_tree(RTViewportState *p_state, const RT_LightData *p_lights, uint32_t p_light_count, uint32_t p_directional_count, const RT_EmissiveInstance *p_emissive, uint32_t p_emissive_count, float p_exposure, uint32_t p_cut_size, uint32_t p_cut_indirect_size, bool p_power_pick);
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
