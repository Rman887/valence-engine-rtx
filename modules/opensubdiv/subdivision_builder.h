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

	HashMap<int, Ref<ArrayMesh>> owner_meshes;
	Dictionary report;

	struct Work;
	Error _gather(Work &r_work);
	Error _gather_skin(Work &r_work);
	Error _refine(Work &r_work, int p_level);
	// Per owner, in three phases: which quads, refined vertices and cage vertices it holds (own, then
	// ghosts); its owner sections; its surfaces. _check_owners runs between the first two, over every
	// owner's partition.
	Error _partition_owner(Work &r_work, uint32_t p_part);
	Error _check_owners(Work &r_work);
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
	Error build(int p_level = 2);
	Ref<ArrayMesh> get_owner_mesh(int p_owner) const;
	Dictionary get_report() const;
	void clear();
};
