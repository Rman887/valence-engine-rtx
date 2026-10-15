/**************************************************************************/
/*  subdivision_builder.h                                                 */
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
#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "scene/resources/mesh.h"

// Builds the refined meshes of a subdivision surface at import (D122, docs/character-subdivision.md
// in the Valence repository). The input is a welded cage split over several meshes ("owners", such
// as a character's head and body) and their glTF primitives: each primitive's arrays, its vertices'
// cage vertex, its faces as cage faces of its vertices, and its face-varying UV weld. The cage is
// refined once, welded, with OpenSubdiv's Catmull-Clark at Blender's settings; each owner's mesh then
// holds the refined faces of its own primitives, its rest pose as static vertex data, and the
// subdivision payload (RenderingServerTypes::SurfaceData::subdivision_data) a runtime pass deforms it
// from: the owner's cage with its ghosts (the other owners' cage vertices its stencils reach), the
// stencils, and the normal and tangent adjacency.
class SubdivisionBuilder : public RefCounted {
	GDCLASS(SubdivisionBuilder, RefCounted);

public:
	static constexpr int MAX_FACE_VARYING_LAYERS = 3;
	static constexpr int MAX_OWNERS = 32;

private:
	struct FaceVaryingLayer {
		Vector<Vector2> values; // Per weld group, in the source's (Blender's) UV space, v up.
		Mesh::ArrayType slot = Mesh::ARRAY_TEX_UV;
	};

	struct SourceSurface {
		int owner = 0;
		Array arrays;
		Array blend_arrays;
		Vector<String> blend_shape_names;
		Vector<int32_t> cage_vertex;
		Vector<int32_t> faces;
		Vector<Vector<int32_t>> uv_weld;
		Ref<Material> material;
	};

	int cage_vertex_count = 0;
	Mesh::BlendShapeMode blend_shape_mode = Mesh::BLEND_SHAPE_MODE_NORMALIZED;
	LocalVector<FaceVaryingLayer> layers;
	LocalVector<SourceSurface> surfaces;
	HashMap<int, Vector<String>> owner_bone_names; // Per owner, its skin's joint names by bind index.

	// An owner's displacement (D122 S6): the surfaces whose faces displace, their faces' source polygon
	// indices (empty: add order is the source's), the face-varying layer the maps read, the maps, the height
	// range and the blend shape whose weight scales it.
	struct OwnerDisplacement {
		Vector<int> surfaces;
		Vector<Vector<int32_t>> source_faces;
		Mesh::ArrayType uv_slot = Mesh::ARRAY_TEX_UV2;
		Ref<Image> height_map;
		Ref<Image> mask;
		float min_height = 0.0f;
		float max_height = 0.0f;
		String shape_name;
	};
	HashMap<int, OwnerDisplacement> owner_displacements;

	HashMap<int, Ref<ArrayMesh>> owner_meshes;
	struct TriangleSources {
		int source_surface = 0;
		PackedInt32Array source_faces;
	};
	HashMap<int, Vector<TriangleSources>> owner_triangle_sources;
	Dictionary report;
	// Set only while build_fingerprint_probe() builds: every owner mesh's blend shapes and every built
	// surface's data are appended to it.
	LocalVector<uint8_t> *fingerprint_input = nullptr;
	void _fingerprint_append(const void *p_data, uint64_t p_size);
	// Set only while build_recording_quads() builds.
	LocalVector<LocalVector<int32_t>> *quad_render_output = nullptr;

	struct Work;
	Error _gather(Work &r_work);
	Error _gather_skin(Work &r_work);
	Error _refine(Work &r_work, int p_level);
	// Per owner, in three phases: which quads, refined vertices and cage vertices it holds (own, then
	// ghosts); its owner sections; its surfaces. _check_owners runs between the first two, over every
	// owner's partition.
	Error _partition_owner(Work &r_work, uint32_t p_part);
	Error _check_owners(Work &r_work);
	Error _owner_displacement(Work &r_work, uint32_t p_part);
	Error _owner_payload(Work &r_work, uint32_t p_part);
	Error _owner_surfaces(Work &r_work, uint32_t p_part);

protected:
	static void _bind_methods();

public:
	void set_cage_vertex_count(int p_count);
	int get_cage_vertex_count() const;
	void set_blend_shape_mode(Mesh::BlendShapeMode p_mode);
	Mesh::BlendShapeMode get_blend_shape_mode() const;
	int add_face_varying_layer(const PackedVector2Array &p_values, Mesh::ArrayType p_array_slot);
	int add_surface(int p_owner, const Array &p_arrays, const Array &p_blend_arrays, const PackedStringArray &p_blend_shape_names, const PackedInt32Array &p_cage_vertex, const PackedInt32Array &p_faces, const Array &p_uv_weld, const Ref<Material> &p_material);
	void set_owner_bone_names(int p_owner, const PackedStringArray &p_bone_names);
	PackedStringArray get_owner_bone_names(int p_owner) const;
	// D122 S6: displaces the owner's refined surface along its undisplaced normal, as Cycles' BOTH method moves
	// vertices: each refined vertex its height at build, scaled at run time by the weight of a blend shape named
	// p_shape_name, appended to the owner's mesh without deltas. A vertex takes its height from the first face of
	// p_surfaces in the source's face order that holds it (p_source_faces: one array per surface of its faces'
	// source polygon indices, or empty for add order), at that face's corner value in the face-varying layer of
	// p_uv_slot; the height is displacement_height() of the maps there. A vertex another owner also refines must get
	// 0, or build() fails, as it does for a shape name any owner uses or a source face named twice. The maps are
	// 8-bit L, RGB or opaque RGBA (Cycles associates a straight alpha, which is not modelled); p_mask may be null (1).
	Error set_owner_displacement(int p_owner, const PackedInt32Array &p_surfaces, const Array &p_source_faces, Mesh::ArrayType p_uv_slot, const Ref<Image> &p_height_map, const Ref<Image> &p_mask, float p_min_height, float p_max_height, const String &p_shape_name);
	void clear_owner_displacement(int p_owner);
	Error build(int p_level = 2);
	Ref<ArrayMesh> get_owner_mesh(int p_owner) const;
	// Per output surface, its add_surface() id and one source-face ordinal per final triangle.
	Array get_owner_triangle_sources(int p_owner) const;
	Dictionary get_report() const;
	void clear();

	// A hash of everything a fixed probe builds to, computed once per process. The probe is two owners across
	// a seam (ghost rings), one of them with two surfaces, quads and a triangle face, three UV layers in the A1
	// slots (UV, UV2, CUSTOM0 as RG float), colours, a blend shape and fractional skin weights, and a third owner
	// apart without blend shapes, whose surface takes a cluster LOD hierarchy and has a face with folded UVs, whose
	// render vertices MikkTSpace's grouping splits; one surface displaces (S6). The hash takes each owner mesh's
	// blend shape names and mode and every field of every surface as built, so it changes with anything that
	// changes those bytes: this builder, the rendering server's packing, skin quantization, clusterizer and
	// hierarchy builder, the payload layout and the vendored libraries. It depends on neither the renderer nor
	// the cluster settings (the builder clusters every surface itself). Not covered: 32-bit indices (the probe
	// has fewer than 65,536 vertices), levels other than 2, and other slots or layer counts.
	static String get_fingerprint();
	// The probe get_fingerprint() hashes, built (null after an error), its bytes appended to r_input when given;
	// and the hash of such bytes. Public for the doctest.
	static Ref<SubdivisionBuilder> build_fingerprint_probe(LocalVector<uint8_t> *r_input = nullptr);
	static String fingerprint_of(const LocalVector<uint8_t> &p_input);

	// A height as Cycles evaluates the source's displacement chain at a UV (v up), in float as its CPU kernel
	// does: each map's linear lookup with repeat in its stored 8-bit values (rows from the bottom, as Blender
	// stores an image), decoded from sRGB, reduced to its luminance; then (height luma * (max - min) + min) times
	// the mask's luma. NaN for a non-finite UV or a map it does not accept. Public for the doctest.
	static float displacement_height(const Ref<Image> &p_height_map, const Ref<Image> &p_mask, const Vector2 &p_uv, float p_min_height, float p_max_height);

	// MikkTSpace's frames on quads, as Blender computes a subdivided mesh's tangents: per corner, 4 corners a quad in
	// the source's (Blender's) order, counter-clockwise from the front, its position, normal and UV with v up, as
	// Blender feeds MikkTSpace, so a quad's diagonal and a badly mapped quad's orientation tie break as Blender's
	// do when Blender's faces start on OpenSubdiv's corner 0. The sign is +1 for an orientation-preserving group, the
	// convention SurfaceTool::generate_tangents uses on Godot's faces (clockwise, v down); reversing the winding and
	// flipping v together leave a UV triangle's orientation alone, so the two agree wherever a quad's two triangles
	// agree, not on a badly mapped quad, which MikkTSpace gives its larger triangle's orientation. r_frames gets each
	// corner's tangent and sign (w); the corners of one MikkTSpace group get the same bits. The builder groups
	// render vertices by it. Public for the doctest.
	static bool mikktspace_quad_frames(const LocalVector<Vector3> &p_positions, const LocalVector<Vector3> &p_normals, const LocalVector<Vector2> &p_uvs, LocalVector<Vector4> &r_frames);
	// build(), appending each surface's quads as built to r_quads: 4 render vertices a quad, in the source's corner
	// order, after MikkTSpace's split. For the doctest.
	Error build_recording_quads(int p_level, LocalVector<LocalVector<int32_t>> &r_quads);
};
