/**************************************************************************/
/*  test_subdivision_builder.h                                            */
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

#include "../subdivision_builder.h"

#include "core/config/project_settings.h"
#include "scene/resources/3d/importer_mesh.h"
#include "scene/resources/material.h"
#include "scene/resources/surface_tool.h"
#include "servers/rendering/rendering_server.h"
#include "tests/test_macros.h"

namespace TestSubdivisionBuilder {

// A cage as glTF delivers it: each face's corners are their own glTF vertices (split at every UV
// seam), so the cage vertex map welds them back.
struct TestCage {
	PackedVector3Array positions; // Per cage vertex.
	LocalVector<int> faces; // 4 cage vertices per face.
	LocalVector<int> owner; // Per face.
};

struct TestPrimitive {
	Array arrays;
	PackedInt32Array cage_vertex;
	PackedInt32Array faces;
	PackedInt32Array uv_weld;
};

// One primitive per owner: every face of the owner, each corner its own glTF vertex and its own
// UV weld group (a UV island per face), the corner's UV the unit square's.
static TestPrimitive make_primitive(const TestCage &p_cage, int p_owner, PackedVector2Array &r_uv_values) {
	TestPrimitive primitive;
	PackedVector3Array vertices;
	PackedInt32Array bones;
	PackedFloat32Array weights;
	const Vector2 square[4] = { Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1) };
	for (uint32_t f = 0; f < p_cage.owner.size(); f++) {
		if (p_cage.owner[f] != p_owner) {
			continue;
		}
		for (int c = 0; c < 4; c++) {
			const int cage = p_cage.faces[4 * f + c];
			primitive.faces.push_back(vertices.size());
			primitive.cage_vertex.push_back(cage);
			vertices.push_back(p_cage.positions[cage]);
			for (int b = 0; b < 4; b++) {
				bones.push_back(b == 0 ? cage % 3 : 0);
				weights.push_back(b == 0 ? 1.0f : 0.0f);
			}
			primitive.uv_weld.push_back(r_uv_values.size());
			r_uv_values.push_back(square[c]);
		}
	}
	primitive.arrays.resize(Mesh::ARRAY_MAX);
	primitive.arrays[Mesh::ARRAY_VERTEX] = vertices;
	primitive.arrays[Mesh::ARRAY_BONES] = bones;
	primitive.arrays[Mesh::ARRAY_WEIGHTS] = weights;
	return primitive;
}

// The blend arrays of a primitive: per shape, the base positions plus the shape's delta at the
// cage vertices it moves.
static Array make_blend_arrays(const TestPrimitive &p_primitive, const LocalVector<HashMap<int, Vector3>> &p_shapes) {
	Array blend_arrays;
	const PackedVector3Array base = p_primitive.arrays[Mesh::ARRAY_VERTEX];
	for (const HashMap<int, Vector3> &shape : p_shapes) {
		PackedVector3Array targets = base;
		for (int v = 0; v < base.size(); v++) {
			const Vector3 *delta = shape.getptr(p_primitive.cage_vertex[v]);
			if (delta) {
				targets.write[v] = base[v] + *delta;
			}
		}
		Array blend;
		blend.resize(Mesh::ARRAY_MAX);
		blend[Mesh::ARRAY_VERTEX] = targets;
		blend_arrays.push_back(blend);
	}
	return blend_arrays;
}

// The joints the test primitives weight: bind index i is joint "B<i>".
static PackedStringArray test_bone_names() {
	PackedStringArray names;
	names.push_back("B0");
	names.push_back("B1");
	names.push_back("B2");
	return names;
}

static TestCage make_cube() {
	TestCage cube;
	for (int i = 0; i < 8; i++) {
		cube.positions.push_back(Vector3((i & 1) ? 1 : -1, (i & 2) ? 1 : -1, (i & 4) ? 1 : -1));
	}
	// Counter-clockwise seen from outside.
	const int faces[24] = { 0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5 };
	for (int i = 0; i < 24; i++) {
		cube.faces.push_back(faces[i]);
	}
	for (int f = 0; f < 6; f++) {
		cube.owner.push_back(0);
	}
	return cube;
}

// A strip of four quads along x, in the y = 0 plane, the first two owner 0's, the last two owner
// 1's: the seam is the column x = 2.
static TestCage make_strip() {
	TestCage strip;
	for (int x = 0; x <= 4; x++) {
		strip.positions.push_back(Vector3(x, 0, 0));
		strip.positions.push_back(Vector3(x, 0, 1));
	}
	for (int q = 0; q < 4; q++) {
		const int a = 2 * q;
		const int quad[4] = { a, a + 1, a + 3, a + 2 };
		for (int c = 0; c < 4; c++) {
			strip.faces.push_back(quad[c]);
		}
		strip.owner.push_back(q < 2 ? 0 : 1);
	}
	return strip;
}

static Vector<uint8_t> surface_payload(const Ref<ArrayMesh> &p_mesh, int p_surface) {
	return RS::get_singleton()->mesh_get_surface(p_mesh->get_rid(), p_surface).subdivision_data;
}

template <typename T>
static const T *payload_section(const Vector<uint8_t> &p_payload, uint32_t p_id, uint32_t &r_count) {
	uint32_t bytes = 0;
	const int64_t offset = RenderingServer::mesh_subdivision_data_find_section(p_payload, p_id, &bytes);
	r_count = offset < 0 ? 0 : bytes / sizeof(T);
	return offset < 0 ? nullptr : reinterpret_cast<const T *>(p_payload.ptr() + offset);
}

// A cage vertex's delta for a shape from the payload's sparse sections (D122 S2), zero where it has none.
static Vector3 payload_cage_delta(const Vector<uint8_t> &p_payload, uint32_t p_cage, uint32_t p_shape) {
	uint32_t count = 0;
	const uint32_t *offsets = payload_section<uint32_t>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_OFFSETS, count);
	const uint32_t *entries = payload_section<uint32_t>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_ENTRIES, count);
	if (offsets == nullptr || entries == nullptr) {
		return Vector3();
	}
	for (uint32_t e = offsets[p_cage]; e < offsets[p_cage + 1]; e++) {
		if (entries[4 * e] == p_shape) {
			float delta[3];
			memcpy(delta, &entries[4 * e + 1], sizeof(delta));
			return Vector3(delta[0], delta[1], delta[2]);
		}
	}
	return Vector3();
}

static uint32_t payload_blend_entry_count(const Vector<uint8_t> &p_payload) {
	uint32_t count = 0;
	payload_section<uint32_t>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_ENTRIES, count);
	return count / RenderingServerTypes::MESH_SUBDIVISION_CAGE_BLEND_ENTRY_WORDS;
}

// What a runtime pass would compute from the payload alone: each render vertex's position as its
// stencil sum over the owner's cage (ghosts included), and its normal from the owner's quads (ghost
// quads included) through the normal adjacency, against the surface's rest pose.
static void check_payload_reproduces_surface(const Ref<ArrayMesh> &p_mesh, int p_surface) {
	const Vector<uint8_t> owner_payload = surface_payload(p_mesh, 0);
	const Vector<uint8_t> payload = surface_payload(p_mesh, p_surface);
	uint32_t count = 0;
	const uint32_t *info = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, count);
	const uint32_t *offsets = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_OFFSETS, count);
	const uint16_t *indices = payload_section<uint16_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_INDICES, count);
	const float *weights = payload_section<float>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_WEIGHTS, count);
	const float *cage = payload_section<float>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_POSITIONS, count);
	const uint32_t *quads = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_QUADS, count);
	const uint32_t *normal_offsets = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_NORMAL_OFFSETS, count);
	const uint32_t *normal_entries = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_NORMAL_ENTRIES, count);
	const uint32_t *r_to_u = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, count);
	REQUIRE((info && offsets && indices && weights && cage && quads && normal_offsets && normal_entries && r_to_u));
	REQUIRE(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_STENCIL_INDEX_BYTES] == 2);
	const uint32_t u_count = info[RenderingServerTypes::MESH_SUBDIVISION_INFO_U_COUNT];
	LocalVector<Vector3> u_positions;
	u_positions.resize(u_count);
	for (uint32_t u = 0; u < u_count; u++) {
		Vector3 sum;
		for (uint32_t i = offsets[u]; i < offsets[u + 1]; i++) {
			CHECK(indices[i] < info[RenderingServerTypes::MESH_SUBDIVISION_INFO_CAGE_COUNT]);
			sum += Vector3(cage[3 * indices[i]], cage[3 * indices[i] + 1], cage[3 * indices[i] + 2]) * weights[i];
		}
		u_positions[u] = sum;
	}
	const Array arrays = p_mesh->surface_get_arrays(p_surface);
	const PackedVector3Array positions = arrays[Mesh::ARRAY_VERTEX];
	const PackedVector3Array normals = arrays[Mesh::ARRAY_NORMAL];
	CHECK(uint32_t(positions.size()) == count);
	for (int r = 0; r < positions.size(); r++) {
		const uint32_t u = r_to_u[r];
		REQUIRE(u < info[RenderingServerTypes::MESH_SUBDIVISION_INFO_OWN_U_COUNT]);
		CHECK(u_positions[u].is_equal_approx(positions[r]));
		Vector3 normal;
		for (uint32_t e = normal_offsets[u]; e < normal_offsets[u + 1]; e++) {
			const uint32_t *quad = &quads[4 * (normal_entries[e] / 4)];
			const uint32_t corner = normal_entries[e] % 4;
			CHECK(quad[corner] == u);
			const Vector3 face = (u_positions[quad[0]] - u_positions[quad[2]]).cross(u_positions[quad[1]] - u_positions[quad[3]]).normalized();
			const Vector3 to_prev = (u_positions[quad[(corner + 3) % 4]] - u_positions[u]).normalized();
			const Vector3 to_next = (u_positions[quad[(corner + 1) % 4]] - u_positions[u]).normalized();
			normal += face * Math::acos(CLAMP(to_prev.dot(to_next), -1.0f, 1.0f));
		}
		// The surface's normals are octahedron-encoded.
		CHECK(normal.normalized().dot(normals[r]) > 0.9999f);
	}
}

TEST_CASE("[SceneTree][SubdivisionBuilder] A cube refines to Catmull-Clark's level 2 with a valid payload") {
	const TestCage cube = make_cube();
	PackedVector2Array uv_values;
	const TestPrimitive primitive = make_primitive(cube, 0, uv_values);
	Ref<StandardMaterial3D> material;
	material.instantiate();

	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(8);
	CHECK(builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV) == 0);
	Array weld;
	weld.push_back(primitive.uv_weld);
	CHECK(builder->add_surface(0, primitive.arrays, Array(), PackedStringArray(), primitive.cage_vertex, primitive.faces, weld, material) == 0);
	REQUIRE(builder->build(2) == OK);

	const Dictionary report = builder->get_report();
	CHECK(int(report["cage_vertices"]) == 8);
	CHECK(int(report["refined_vertices"]) == 98); // 8 + 12 + 6, then 26 + 48 + 24.
	CHECK(int(report["refined_quads"]) == 96);

	const Ref<ArrayMesh> mesh = builder->get_owner_mesh(0);
	REQUIRE(mesh.is_valid());
	REQUIRE(mesh->get_surface_count() == 1);
	CHECK(mesh->surface_is_subdivided(0));
	CHECK(mesh->surface_get_material(0).ptr() == material.ptr());
	// A UV island per face: each face's 25 refined vertices are their own render vertices.
	CHECK(mesh->surface_get_array_len(0) == 6 * 25);
	CHECK(mesh->surface_get_array_index_len(0) == 96 * 6);

	// The refined surface lies inside the cage and, by symmetry, every refined vertex on a face's
	// centre line sits on that face's axis.
	const Array arrays = mesh->surface_get_arrays(0);
	const PackedVector3Array positions = arrays[Mesh::ARRAY_VERTEX];
	const PackedVector3Array normals = arrays[Mesh::ARRAY_NORMAL];
	for (int i = 0; i < positions.size(); i++) {
		CHECK(positions[i].abs().x <= 1.0f);
		CHECK(positions[i].abs().y <= 1.0f);
		CHECK(positions[i].abs().z <= 1.0f);
		// Outward normals on a convex surface.
		CHECK(normals[i].dot(positions[i]) > 0.0f);
	}
	CHECK(arrays[Mesh::ARRAY_TANGENT].get_type() == Variant::PACKED_FLOAT32_ARRAY);
	check_payload_reproduces_surface(mesh, 0);

	const Vector<uint8_t> payload = surface_payload(mesh, 0);
	String error;
	CHECK_MESSAGE(RenderingServer::mesh_subdivision_data_is_valid(payload, &error), error);
	const uint32_t *header = reinterpret_cast<const uint32_t *>(payload.ptr());
	CHECK(header[0] == RenderingServerTypes::MESH_SUBDIVISION_DATA_VERSION);
	CHECK((header[1] & RenderingServerTypes::MESH_SUBDIVISION_FLAG_OWNER) != 0);
	uint32_t count = 0;
	const uint32_t *info = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, count);
	REQUIRE(info != nullptr);
	CHECK(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_CAGE_COUNT] == 8);
	CHECK(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_OWN_CAGE_COUNT] == 8);
	CHECK(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_U_COUNT] == 98);
	CHECK(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_OWN_U_COUNT] == 98);
	CHECK(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_QUAD_COUNT] == 96);
	CHECK(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_STENCIL_INDEX_BYTES] == 2);
	const uint32_t *r_to_u = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, count);
	CHECK(count == 150);
	// Every render vertex's position is its refined vertex's stencil sum over the cage.
	const uint32_t *offsets = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_OFFSETS, count);
	CHECK(count == 99);
	const uint16_t *indices = payload_section<uint16_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_INDICES, count);
	const float *weights = payload_section<float>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_WEIGHTS, count);
	const float *cage = payload_section<float>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_POSITIONS, count);
	CHECK(count == 24);
	REQUIRE((r_to_u && offsets && indices && weights && cage));
	for (int r = 0; r < positions.size(); r++) {
		const uint32_t u = r_to_u[r];
		Vector3 sum;
		float weight_sum = 0.0f;
		for (uint32_t i = offsets[u]; i < offsets[u + 1]; i++) {
			sum += Vector3(cage[3 * indices[i]], cage[3 * indices[i] + 1], cage[3 * indices[i] + 2]) * weights[i];
			weight_sum += weights[i];
		}
		CHECK(sum.is_equal_approx(positions[r]));
		CHECK(Math::is_equal_approx(weight_sum, 1.0f));
	}
	// The skin travels with the cage: bone (cage vertex % 3) at full weight.
	const uint16_t *skin = payload_section<uint16_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_SKIN, count);
	REQUIRE(skin != nullptr);
	CHECK(count == 8 * 16);
	const uint32_t *welded = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_WELDED, count);
	for (int c = 0; c < 8; c++) {
		CHECK(skin[16 * c] == welded[c] % 3);
		CHECK(skin[16 * c + 8] == 65535);
	}
}

TEST_CASE("[SceneTree][SubdivisionBuilder] Two owners: ghosts make the seam identical on both sides") {
	const TestCage strip = make_strip();
	PackedVector2Array uv_values;
	const TestPrimitive left = make_primitive(strip, 0, uv_values);
	const TestPrimitive right = make_primitive(strip, 1, uv_values);
	// A shape on both owners that moves only a vertex past the seam (x = 3, owner 1's): owner 0's
	// glTF vertices never reach it, yet its ghost must carry it.
	LocalVector<HashMap<int, Vector3>> shapes;
	shapes.resize(1);
	shapes[0].insert(6, Vector3(0, 0.5f, 0));
	PackedStringArray names;
	names.push_back("Lift");

	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(strip.positions.size());
	builder->set_owner_bone_names(0, test_bone_names());
	builder->set_owner_bone_names(1, test_bone_names());
	builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
	Array left_weld;
	left_weld.push_back(left.uv_weld);
	Array right_weld;
	right_weld.push_back(right.uv_weld);
	builder->add_surface(0, left.arrays, make_blend_arrays(left, shapes), names, left.cage_vertex, left.faces, left_weld, Ref<Material>());
	builder->add_surface(1, right.arrays, make_blend_arrays(right, shapes), names, right.cage_vertex, right.faces, right_weld, Ref<Material>());
	REQUIRE(builder->build(2) == OK);

	const Dictionary owners = builder->get_report()["owners"];
	const Dictionary report_left = owners[0];
	CHECK(int(report_left["cage_vertices"]) == 6);
	CHECK(int(report_left["ghost_cage_vertices"]) > 0);
	CHECK(int(report_left["ghost_quads"]) > 0);

	const Ref<ArrayMesh> mesh_left = builder->get_owner_mesh(0);
	const Ref<ArrayMesh> mesh_right = builder->get_owner_mesh(1);
	REQUIRE((mesh_left.is_valid() && mesh_right.is_valid()));
	// The blend shape stays the mesh's; its surfaces carry no blend arrays.
	CHECK(mesh_left->get_blend_shape_count() == 1);
	CHECK(mesh_left->get_blend_shape_name(0) == StringName("Lift"));
	CHECK(mesh_left->surface_is_subdivided(0));

	// Owner 0's ghost cage vertex 6 carries the shape's delta.
	const Vector<uint8_t> payload = surface_payload(mesh_left, 0);
	uint32_t count = 0;
	const uint32_t *welded = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_WELDED, count);
	const uint32_t cage_count = count;
	REQUIRE(welded);
	CHECK(payload_blend_entry_count(payload) == 1); // Sparse: the one cage vertex the shape moves.
	bool found = false;
	for (uint32_t c = 0; c < cage_count; c++) {
		if (welded[c] == 6) {
			found = true;
			CHECK(payload_cage_delta(payload, c, 0) == Vector3(0, 0.5f, 0));
		}
	}
	CHECK(found);

	// The seam's refined vertices: the same position and normal bits on both sides.
	const Array arrays_left = mesh_left->surface_get_arrays(0);
	const Array arrays_right = mesh_right->surface_get_arrays(0);
	const PackedVector3Array positions_left = arrays_left[Mesh::ARRAY_VERTEX];
	const PackedVector3Array normals_left = arrays_left[Mesh::ARRAY_NORMAL];
	const PackedVector3Array positions_right = arrays_right[Mesh::ARRAY_VERTEX];
	const PackedVector3Array normals_right = arrays_right[Mesh::ARRAY_NORMAL];
	int seam = 0;
	for (int i = 0; i < positions_left.size(); i++) {
		for (int j = 0; j < positions_right.size(); j++) {
			if (memcmp(&positions_left[i], &positions_right[j], sizeof(Vector3)) == 0) {
				seam++;
				CHECK(memcmp(&normals_left[i], &normals_right[j], sizeof(Vector3)) == 0);
			}
		}
	}
	CHECK(seam > 0);
	check_payload_reproduces_surface(mesh_left, 0);
	check_payload_reproduces_surface(mesh_right, 0);
	// The seam lies where Catmull-Clark puts the column x = 2 of the welded strip, not on a
	// boundary curve: its refined vertices keep x = 2.
	for (int i = 0; i < positions_left.size(); i++) {
		CHECK(positions_left[i].x <= 2.0f + CMP_EPSILON);
	}
}

TEST_CASE("[SceneTree][SubdivisionBuilder] A shape that moves an owner's ghosts must exist on that owner") {
	const TestCage strip = make_strip();
	PackedVector2Array uv_values;
	const TestPrimitive left = make_primitive(strip, 0, uv_values);
	const TestPrimitive right = make_primitive(strip, 1, uv_values);
	LocalVector<HashMap<int, Vector3>> shapes;
	shapes.resize(1);
	shapes[0].insert(6, Vector3(0, 0.5f, 0));
	PackedStringArray names;
	names.push_back("RightOnly");

	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(strip.positions.size());
	builder->set_owner_bone_names(0, test_bone_names());
	builder->set_owner_bone_names(1, test_bone_names());
	builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
	Array left_weld;
	left_weld.push_back(left.uv_weld);
	Array right_weld;
	right_weld.push_back(right.uv_weld);
	builder->add_surface(0, left.arrays, Array(), PackedStringArray(), left.cage_vertex, left.faces, left_weld, Ref<Material>());
	builder->add_surface(1, right.arrays, make_blend_arrays(right, shapes), names, right.cage_vertex, right.faces, right_weld, Ref<Material>());
	ERR_PRINT_OFF;
	CHECK(builder->build(2) == ERR_INVALID_DATA);
	CHECK(builder->get_owner_mesh(0).is_null());
	ERR_PRINT_ON;
}

TEST_CASE("[SceneTree][SubdivisionBuilder] Disagreeing glTF vertices of one cage vertex fail the build") {
	const TestCage cube = make_cube();
	PackedVector2Array uv_values;
	TestPrimitive primitive = make_primitive(cube, 0, uv_values);
	PackedVector3Array vertices = primitive.arrays[Mesh::ARRAY_VERTEX];
	vertices.write[5] += Vector3(0, 1e-3f, 0);
	primitive.arrays[Mesh::ARRAY_VERTEX] = vertices;

	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(8);
	builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
	Array weld;
	weld.push_back(primitive.uv_weld);
	builder->add_surface(0, primitive.arrays, Array(), PackedStringArray(), primitive.cage_vertex, primitive.faces, weld, Ref<Material>());
	ERR_PRINT_OFF;
	CHECK(builder->build(2) == ERR_INVALID_DATA);
	ERR_PRINT_ON;
}

TEST_CASE("[SceneTree][SubdivisionBuilder] The payload survives the mesh's surfaces round trip and refuses blend arrays") {
	const TestCage cube = make_cube();
	PackedVector2Array uv_values;
	const TestPrimitive primitive = make_primitive(cube, 0, uv_values);
	LocalVector<HashMap<int, Vector3>> shapes;
	shapes.resize(2);
	shapes[0].insert(0, Vector3(-0.25f, 0, 0));
	shapes[1].insert(7, Vector3(0, 0.25f, 0));
	PackedStringArray names;
	names.push_back("A");
	names.push_back("B");

	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(8);
	builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV2);
	Array weld;
	weld.push_back(primitive.uv_weld);
	builder->add_surface(0, primitive.arrays, make_blend_arrays(primitive, shapes), names, primitive.cage_vertex, primitive.faces, weld, Ref<Material>());
	REQUIRE(builder->build(2) == OK);
	const Ref<ArrayMesh> mesh = builder->get_owner_mesh(0);
	REQUIRE(mesh.is_valid());
	CHECK(mesh->get_blend_shape_count() == 2);
	CHECK((mesh->surface_get_format(0) & Mesh::ARRAY_FORMAT_TEX_UV2) != 0);
	CHECK((mesh->surface_get_format(0) & Mesh::ARRAY_FORMAT_TEX_UV) == 0);

	const Vector<uint8_t> payload = surface_payload(mesh, 0);
	Ref<ArrayMesh> copy = mesh->duplicate();
	REQUIRE(copy.is_valid());
	const Ref<ArrayMesh> both[2] = { mesh, copy };
	for (const Ref<ArrayMesh> &each : both) {
		const Array surfaces = each->get("_surfaces");
		REQUIRE(surfaces.size() == 1);
		const Dictionary d = surfaces[0];
		CHECK(d.has("subdivision_data"));
		CHECK_FALSE(d.has("blend_shapes"));
		CHECK(Vector<uint8_t>(d.get("subdivision_data", Vector<uint8_t>())) == payload);
		CHECK(each->surface_is_subdivided(0));
		CHECK(each->get_blend_shape_count() == 2);
	}

	// The cage's deltas, sparse per cage vertex, by shape in the mesh's order.
	uint32_t count = 0;
	const uint32_t *welded = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_WELDED, count);
	REQUIRE(welded);
	CHECK(payload_blend_entry_count(payload) == 2);
	for (uint32_t c = 0; c < 8; c++) {
		CHECK(payload_cage_delta(payload, c, 0) == (welded[c] == 0 ? Vector3(-0.25f, 0, 0) : Vector3()));
		CHECK(payload_cage_delta(payload, c, 1) == (welded[c] == 7 ? Vector3(0, 0.25f, 0) : Vector3()));
	}

	// Per-vertex blend arrays do not exist for a subdivided surface: readers fail loudly.
	ERR_PRINT_OFF;
	CHECK(mesh->surface_get_blend_shape_arrays(0).is_empty());
	const Ref<ImporterMesh> importer_mesh = ImporterMesh::from_mesh(mesh);
	CHECK(importer_mesh->get_surface_count() == 0);
	ERR_PRINT_ON;
}

// A grid of quads in the xz plane facing +y, its faces in the source's (Blender's) counter-clockwise
// order, columns from p_split_column on owner 1's; bent, it is y = 0.2 x^2 + 0.15 z^2.
static TestCage make_grid(int p_columns, int p_rows, int p_split_column, bool p_bend) {
	TestCage grid;
	for (int x = 0; x <= p_columns; x++) {
		for (int z = 0; z <= p_rows; z++) {
			grid.positions.push_back(Vector3(x, p_bend ? 0.2f * x * x + 0.15f * z * z : 0.0f, z));
		}
	}
	for (int x = 0; x < p_columns; x++) {
		for (int z = 0; z < p_rows; z++) {
			const int a = x * (p_rows + 1) + z;
			const int quad[4] = { a, a + 1, a + p_rows + 2, a + p_rows + 1 };
			for (int c = 0; c < 4; c++) {
				grid.faces.push_back(quad[c]);
			}
			grid.owner.push_back(p_split_column >= 0 && x >= p_split_column ? 1 : 0);
		}
	}
	return grid;
}

// One primitive of a group's faces whose glTF vertices are the cage vertices themselves, so its UVs
// are continuous: the weld group of a corner is its cage vertex.
static TestPrimitive make_shared_primitive(const TestCage &p_cage, int p_group) {
	TestPrimitive primitive;
	HashMap<int, int> vertex_of;
	PackedVector3Array vertices;
	for (uint32_t f = 0; f < p_cage.owner.size(); f++) {
		if (p_cage.owner[f] != p_group) {
			continue;
		}
		for (int c = 0; c < 4; c++) {
			const int cage = p_cage.faces[4 * f + c];
			if (!vertex_of.has(cage)) {
				vertex_of.insert(cage, vertices.size());
				primitive.cage_vertex.push_back(cage);
				vertices.push_back(p_cage.positions[cage]);
			}
			primitive.faces.push_back(vertex_of[cage]);
			primitive.uv_weld.push_back(cage);
		}
	}
	primitive.arrays.resize(Mesh::ARRAY_MAX);
	primitive.arrays[Mesh::ARRAY_VERTEX] = vertices;
	return primitive;
}

// UV values per cage vertex: u and v as the dot products of the vertex's (x, z) with two axes.
static PackedVector2Array grid_uvs(const TestCage &p_cage, const Vector2 &p_u_axis, const Vector2 &p_v_axis, float p_scale = 1.0f) {
	PackedVector2Array values;
	for (const Vector3 &position : p_cage.positions) {
		const Vector2 xz(position.x, position.z);
		values.push_back(Vector2(xz.dot(p_u_axis), xz.dot(p_v_axis)) * p_scale);
	}
	return values;
}

static Ref<ArrayMesh> build_grid(const TestCage &p_grid, const PackedVector2Array &p_uv_values) {
	const TestPrimitive primitive = make_shared_primitive(p_grid, 0);
	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(p_grid.positions.size());
	builder->add_face_varying_layer(p_uv_values, Mesh::ARRAY_TEX_UV);
	Array weld;
	weld.push_back(primitive.uv_weld);
	builder->add_surface(0, primitive.arrays, Array(), PackedStringArray(), primitive.cage_vertex, primitive.faces, weld, Ref<Material>());
	REQUIRE(builder->build(2) == OK);
	return builder->get_owner_mesh(0);
}

TEST_CASE("[SceneTree][SubdivisionBuilder] An owner with blend shapes gets flat clusters, not a cluster LOD hierarchy") {
	if (RenderingServer::build_clusters_func == nullptr || RenderingServer::build_cluster_lod_func == nullptr) {
		return; // No cluster builders without the meshoptimizer module.
	}
	const Variant cluster_blas = GLOBAL_GET("rendering/pathtracing/cluster_blas");
	const Variant cluster_lod = GLOBAL_GET("rendering/pathtracing/cluster_lod");
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", true);
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_lod", true);
	const TestCage grid = make_grid(6, 6, -1, true); // 1,152 refined triangles, past one cluster.
	const PackedVector2Array uv_values = grid_uvs(grid, Vector2(1, 0), Vector2(0, 1));
	auto cluster_version = [&](bool p_shapes) -> uint32_t {
		const TestPrimitive primitive = make_shared_primitive(grid, 0);
		Array blend_arrays;
		PackedStringArray names;
		if (p_shapes) {
			LocalVector<HashMap<int, Vector3>> shapes;
			shapes.resize(1);
			shapes[0].insert(grid.faces[0], Vector3(0, 0.1f, 0));
			blend_arrays = make_blend_arrays(primitive, shapes);
			names.push_back("A");
		}
		Ref<SubdivisionBuilder> builder;
		builder.instantiate();
		builder->set_cage_vertex_count(grid.positions.size());
		builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
		Array weld;
		weld.push_back(primitive.uv_weld);
		builder->add_surface(0, primitive.arrays, blend_arrays, names, primitive.cage_vertex, primitive.faces, weld, Ref<Material>());
		REQUIRE(builder->build(2) == OK);
		const Ref<ArrayMesh> mesh = builder->get_owner_mesh(0);
		REQUIRE(mesh.is_valid());
		const Dictionary surface = Array(mesh->get("_surfaces"))[0];
		const Vector<uint8_t> clusters = surface.get("cluster_data", Vector<uint8_t>());
		REQUIRE(clusters.size() >= int(sizeof(uint32_t)));
		uint32_t version;
		memcpy(&version, clusters.ptr(), sizeof(uint32_t));
		return version;
	};
	// Without shapes the hierarchy is built: the witness that the switches above reach the builder.
	const uint32_t without_shapes = cluster_version(false);
	const uint32_t with_shapes = cluster_version(true);
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", cluster_blas);
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_lod", cluster_lod);
	CHECK(without_shapes == RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION);
	CHECK(with_shapes == RenderingServerTypes::MESH_CLUSTER_DATA_VERSION);
}

TEST_CASE("[SceneTree][SubdivisionBuilder] Tangents have Godot's handedness, as SurfaceTool's MikkTSpace gives") {
	const TestCage grid = make_grid(2, 2, -1, false);
	struct Mapping {
		Vector2 u_axis;
		Vector2 v_axis;
	};
	const Mapping mappings[4] = {
		{ Vector2(1, 0), Vector2(0, 1) },
		{ Vector2(-1, 0), Vector2(0, 1) }, // Mirrored u.
		{ Vector2(1, 0), Vector2(0, -1) }, // Mirrored v.
		{ Vector2(0, 1), Vector2(1, 0) }, // Rotated.
	};
	for (const Mapping &mapping : mappings) {
		const Ref<ArrayMesh> mesh = build_grid(grid, grid_uvs(grid, mapping.u_axis, mapping.v_axis));
		REQUIRE(mesh.is_valid());
		const Array arrays = mesh->surface_get_arrays(0);
		Array plain = arrays.duplicate();
		plain[Mesh::ARRAY_TANGENT] = Variant();
		Ref<SurfaceTool> surface_tool;
		surface_tool.instantiate();
		surface_tool->create_from_arrays(plain);
		surface_tool->generate_tangents();
		const Array generated = surface_tool->commit_to_arrays();
		const PackedFloat32Array ours = arrays[Mesh::ARRAY_TANGENT];
		const PackedFloat32Array theirs = generated[Mesh::ARRAY_TANGENT];
		REQUIRE(ours.size() == theirs.size());
		for (int v = 0; v < ours.size() / 4; v++) {
			CHECK(ours[4 * v + 3] == theirs[4 * v + 3]);
			CHECK(Vector3(ours[4 * v], ours[4 * v + 1], ours[4 * v + 2]).dot(Vector3(theirs[4 * v], theirs[4 * v + 1], theirs[4 * v + 2])) > 0.99f);
		}
	}
}

TEST_CASE("[SceneTree][SubdivisionBuilder] Tiny UV triangles still orient the tangents") {
	const TestCage grid = make_grid(2, 2, -1, false);
	const Ref<ArrayMesh> unit = build_grid(grid, grid_uvs(grid, Vector2(1, 0), Vector2(0, 1)));
	// A thousandth: refined UV triangles of about 1e-8 in area, which an epsilon on the area dropped.
	const Ref<ArrayMesh> tiny = build_grid(grid, grid_uvs(grid, Vector2(1, 0), Vector2(0, 1), 1e-3f));
	REQUIRE((unit.is_valid() && tiny.is_valid()));
	const Array tiny_arrays = tiny->surface_get_arrays(0);
	const PackedFloat32Array expected = unit->surface_get_arrays(0)[Mesh::ARRAY_TANGENT];
	const PackedFloat32Array tangents = tiny_arrays[Mesh::ARRAY_TANGENT];
	REQUIRE(expected.size() == tangents.size());
	for (int v = 0; v < tangents.size() / 4; v++) {
		const Vector3 tangent(tangents[4 * v], tangents[4 * v + 1], tangents[4 * v + 2]);
		CHECK(tangents[4 * v + 3] == expected[4 * v + 3]);
		CHECK(tangent.dot(Vector3(expected[4 * v], expected[4 * v + 1], expected[4 * v + 2])) > 0.999f);
	}
	// Every triangle's UV area is below the epsilon (1e-5 on twice the area) that used to drop it.
	const PackedVector2Array uvs = tiny_arrays[Mesh::ARRAY_TEX_UV];
	const PackedInt32Array indices = tiny_arrays[Mesh::ARRAY_INDEX];
	float largest = 0.0f;
	for (int t = 0; t < indices.size() / 3; t++) {
		const Vector2 d1 = uvs[indices[3 * t + 1]] - uvs[indices[3 * t]];
		const Vector2 d2 = uvs[indices[3 * t + 2]] - uvs[indices[3 * t]];
		largest = MAX(largest, Math::abs(d1.x * d2.y - d2.x * d1.y));
	}
	CHECK(largest > 0.0f);
	CHECK(largest < 1e-5f);
}

TEST_CASE("[SceneTree][SubdivisionBuilder] UV values keep their range, stored as 1 - v") {
	const TestCage cube = make_cube();
	PackedVector2Array uv_values;
	const TestPrimitive primitive = make_primitive(cube, 0, uv_values);
	// Each face's island spans v from 0 to 0.5 in the source's space: 0.5 to 1 once stored.
	for (int i = 0; i < uv_values.size(); i++) {
		uv_values.write[i].y *= 0.5f;
	}
	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(8);
	builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
	Array weld;
	weld.push_back(primitive.uv_weld);
	builder->add_surface(0, primitive.arrays, Array(), PackedStringArray(), primitive.cage_vertex, primitive.faces, weld, Ref<Material>());
	REQUIRE(builder->build(2) == OK);
	const PackedVector2Array uvs = builder->get_owner_mesh(0)->surface_get_arrays(0)[Mesh::ARRAY_TEX_UV];
	REQUIRE(!uvs.is_empty());
	Vector2 low(1e9, 1e9);
	Vector2 high(-1e9, -1e9);
	for (const Vector2 &uv : uvs) {
		low = low.min(uv);
		high = high.max(uv);
	}
	// The islands' corners are kept exactly.
	CHECK(low == Vector2(0.0f, 0.5f));
	CHECK(high == Vector2(1.0f, 1.0f));
}

TEST_CASE("[SceneTree][SubdivisionBuilder] Colours refine within their faces") {
	const TestCage cube = make_cube();
	PackedVector2Array uv_values;
	TestPrimitive primitive = make_primitive(cube, 0, uv_values);
	// A colour per face, on every glTF vertex of the face (each face's corners are its own).
	PackedColorArray colors;
	const PackedVector3Array vertices = primitive.arrays[Mesh::ARRAY_VERTEX];
	for (int v = 0; v < vertices.size(); v++) {
		const float f = v / 4;
		colors.push_back(Color(f * 0.125f, 1.0f - f * 0.125f, 0.5f, 1.0f));
	}
	primitive.arrays[Mesh::ARRAY_COLOR] = colors;
	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(8);
	builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
	Array weld;
	weld.push_back(primitive.uv_weld);
	builder->add_surface(0, primitive.arrays, Array(), PackedStringArray(), primitive.cage_vertex, primitive.faces, weld, Ref<Material>());
	REQUIRE(builder->build(2) == OK);
	const Array arrays = builder->get_owner_mesh(0)->surface_get_arrays(0);
	const PackedColorArray refined = arrays[Mesh::ARRAY_COLOR];
	const PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
	REQUIRE(!refined.is_empty());
	// Every triangle lies in one face: its three corners carry that face's colour (as stored).
	for (int t = 0; t < indices.size() / 3; t++) {
		CHECK(refined[indices[3 * t]] == refined[indices[3 * t + 1]]);
		CHECK(refined[indices[3 * t]] == refined[indices[3 * t + 2]]);
	}
	for (const Color &color : refined) {
		bool found = false;
		for (const Color &source : colors) {
			found = found || (Math::abs(color.r - source.r) <= 1.0f / 255.0f && Math::abs(color.g - source.g) <= 1.0f / 255.0f && Math::abs(color.b - source.b) <= 1.0f / 255.0f);
		}
		CHECK(found);
	}
}

TEST_CASE("[SceneTree][SubdivisionBuilder] Triangle faces refine as Catmull-Clark triangles") {
	const Vector3 corners[4] = { Vector3(1, 1, 1), Vector3(1, -1, -1), Vector3(-1, 1, -1), Vector3(-1, -1, 1) };
	const int faces[12] = { 0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2 }; // Counter-clockwise from outside.
	const Vector2 square[3] = { Vector2(0, 0), Vector2(1, 0), Vector2(0, 1) };
	PackedVector3Array vertices;
	PackedInt32Array cage_vertex;
	PackedInt32Array glTF_faces;
	PackedInt32Array uv_weld;
	PackedVector2Array uv_values;
	for (int f = 0; f < 4; f++) {
		for (int c = 0; c < 3; c++) {
			glTF_faces.push_back(vertices.size());
			cage_vertex.push_back(faces[3 * f + c]);
			vertices.push_back(corners[faces[3 * f + c]]);
			uv_weld.push_back(uv_values.size());
			uv_values.push_back(square[c]);
		}
		glTF_faces.push_back(-1);
		uv_weld.push_back(-1);
	}
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	Array weld;
	weld.push_back(uv_weld);

	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(4);
	builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
	builder->add_surface(0, arrays, Array(), PackedStringArray(), cage_vertex, glTF_faces, weld, Ref<Material>());
	REQUIRE(builder->build(2) == OK);
	// Level 1: 4 + 6 + 4 vertices, 12 quads; level 2: 14 + 24 + 12 vertices, 48 quads.
	CHECK(int(builder->get_report()["refined_vertices"]) == 50);
	CHECK(int(builder->get_report()["refined_quads"]) == 48);
	const Ref<ArrayMesh> mesh = builder->get_owner_mesh(0);
	REQUIRE(mesh.is_valid());
	const Array refined = mesh->surface_get_arrays(0);
	const PackedVector3Array positions = refined[Mesh::ARRAY_VERTEX];
	const PackedVector3Array normals = refined[Mesh::ARRAY_NORMAL];
	for (int i = 0; i < positions.size(); i++) {
		CHECK(normals[i].dot(positions[i]) > 0.0f);
	}
	check_payload_reproduces_surface(mesh, 0);

	// A fourth index below -1 is no triangle marker.
	glTF_faces.write[3] = -2;
	builder->clear();
	builder->set_cage_vertex_count(4);
	builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
	builder->add_surface(0, arrays, Array(), PackedStringArray(), cage_vertex, glTF_faces, weld, Ref<Material>());
	ERR_PRINT_OFF;
	CHECK(builder->build(2) == ERR_INVALID_PARAMETER);
	ERR_PRINT_ON;
}

TEST_CASE("[SceneTree][SubdivisionBuilder] An owner of several surfaces: the owner sections ride on its first") {
	TestCage cube = make_cube();
	for (int f = 0; f < 6; f++) {
		cube.owner[f] = f < 3 ? 0 : 1; // Two primitives of one owner.
	}
	PackedVector2Array uv_values;
	const TestPrimitive first = make_primitive(cube, 0, uv_values);
	const TestPrimitive second = make_primitive(cube, 1, uv_values);
	Ref<StandardMaterial3D> material_a;
	material_a.instantiate();
	material_a->set_name("A");
	Ref<StandardMaterial3D> material_b;
	material_b.instantiate();
	material_b->set_name("B");

	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(8);
	builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
	Array first_weld;
	first_weld.push_back(first.uv_weld);
	Array second_weld;
	second_weld.push_back(second.uv_weld);
	builder->add_surface(0, first.arrays, Array(), PackedStringArray(), first.cage_vertex, first.faces, first_weld, material_a);
	builder->add_surface(0, second.arrays, Array(), PackedStringArray(), second.cage_vertex, second.faces, second_weld, material_b);
	REQUIRE(builder->build(2) == OK);
	const Ref<ArrayMesh> mesh = builder->get_owner_mesh(0);
	REQUIRE(mesh.is_valid());
	REQUIRE(mesh->get_surface_count() == 2);
	CHECK(mesh->surface_get_name(0) == "A");
	CHECK(mesh->surface_get_name(1) == "B");
	const Vector<uint8_t> payload = surface_payload(mesh, 1);
	const uint32_t *header = reinterpret_cast<const uint32_t *>(payload.ptr());
	CHECK((header[1] & RenderingServerTypes::MESH_SUBDIVISION_FLAG_OWNER) == 0);
	CHECK(header[2] == 0);
	uint32_t count = 0;
	CHECK(payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, count) == nullptr);
	check_payload_reproduces_surface(mesh, 0);
	check_payload_reproduces_surface(mesh, 1);
}

// What a runtime pass computes from an owner's payload alone, per welded refined vertex: the
// position as its stencil sum, the normal from the owner's quads through the normal adjacency, in
// float and in the payload's order.
struct PayloadNormals {
	HashMap<uint32_t, Vector3> normal; // Per welded refined vertex the owner holds as its own.
	HashMap<uint32_t, Vector<uint32_t>> entries; // Its quads' welded corners and the corner, in order.
};

static PayloadNormals payload_normals(const Vector<uint8_t> &p_payload) {
	PayloadNormals result;
	uint32_t count = 0;
	const uint32_t *info = payload_section<uint32_t>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, count);
	const uint32_t *offsets = payload_section<uint32_t>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_OFFSETS, count);
	const uint16_t *indices = payload_section<uint16_t>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_INDICES, count);
	const float *weights = payload_section<float>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_WEIGHTS, count);
	const float *cage = payload_section<float>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_POSITIONS, count);
	const uint32_t *quads = payload_section<uint32_t>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_QUADS, count);
	const uint32_t *normal_offsets = payload_section<uint32_t>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_NORMAL_OFFSETS, count);
	const uint32_t *normal_entries = payload_section<uint32_t>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_NORMAL_ENTRIES, count);
	const uint32_t *u_welded = payload_section<uint32_t>(p_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_U_WELDED, count);
	REQUIRE((info && offsets && indices && weights && cage && quads && normal_offsets && normal_entries && u_welded));
	const uint32_t u_count = info[RenderingServerTypes::MESH_SUBDIVISION_INFO_U_COUNT];
	LocalVector<Vector3> positions;
	positions.resize(u_count);
	for (uint32_t u = 0; u < u_count; u++) {
		Vector3 sum;
		for (uint32_t i = offsets[u]; i < offsets[u + 1]; i++) {
			sum += Vector3(cage[3 * indices[i]], cage[3 * indices[i] + 1], cage[3 * indices[i] + 2]) * weights[i];
		}
		positions[u] = sum;
	}
	for (uint32_t u = 0; u < info[RenderingServerTypes::MESH_SUBDIVISION_INFO_OWN_U_COUNT]; u++) {
		Vector3 normal;
		Vector<uint32_t> sequence;
		for (uint32_t e = normal_offsets[u]; e < normal_offsets[u + 1]; e++) {
			const uint32_t *quad = &quads[4 * (normal_entries[e] / 4)];
			const uint32_t corner = normal_entries[e] % 4;
			const Vector3 face = (positions[quad[0]] - positions[quad[2]]).cross(positions[quad[1]] - positions[quad[3]]).normalized();
			const Vector3 to_prev = (positions[quad[(corner + 3) % 4]] - positions[u]).normalized();
			const Vector3 to_next = (positions[quad[(corner + 1) % 4]] - positions[u]).normalized();
			normal += face * Math::acos(CLAMP(to_prev.dot(to_next), -1.0f, 1.0f));
			for (int c = 0; c < 4; c++) {
				sequence.push_back(u_welded[quad[c]]);
			}
			sequence.push_back(corner);
		}
		result.normal.insert(u_welded[u], normal.normalized());
		result.entries.insert(u_welded[u], sequence);
	}
	return result;
}

TEST_CASE("[SceneTree][SubdivisionBuilder] A bent seam: both owners' payloads give its normals the same bits") {
	const TestCage grid = make_grid(4, 2, 2, true);
	const TestPrimitive left = make_shared_primitive(grid, 0);
	const TestPrimitive right = make_shared_primitive(grid, 1);
	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(grid.positions.size());
	builder->add_face_varying_layer(grid_uvs(grid, Vector2(0.25f, 0), Vector2(0, 0.5f)), Mesh::ARRAY_TEX_UV);
	Array left_weld;
	left_weld.push_back(left.uv_weld);
	Array right_weld;
	right_weld.push_back(right.uv_weld);
	builder->add_surface(0, left.arrays, Array(), PackedStringArray(), left.cage_vertex, left.faces, left_weld, Ref<Material>());
	builder->add_surface(1, right.arrays, Array(), PackedStringArray(), right.cage_vertex, right.faces, right_weld, Ref<Material>());
	REQUIRE(builder->build(2) == OK);
	const Ref<ArrayMesh> meshes[2] = { builder->get_owner_mesh(0), builder->get_owner_mesh(1) };
	REQUIRE((meshes[0].is_valid() && meshes[1].is_valid()));

	// The rest pose: render vertices of one welded refined vertex carry the same position and normal
	// bits on both sides.
	HashMap<uint32_t, int> left_vertex;
	const Vector<uint8_t> payloads[2] = { surface_payload(meshes[0], 0), surface_payload(meshes[1], 0) };
	uint32_t count = 0;
	const Array left_arrays = meshes[0]->surface_get_arrays(0);
	const Array right_arrays = meshes[1]->surface_get_arrays(0);
	const PackedVector3Array left_positions = left_arrays[Mesh::ARRAY_VERTEX];
	const PackedVector3Array left_normals = left_arrays[Mesh::ARRAY_NORMAL];
	const PackedVector3Array right_positions = right_arrays[Mesh::ARRAY_VERTEX];
	const PackedVector3Array right_normals = right_arrays[Mesh::ARRAY_NORMAL];
	const uint32_t *left_r_to_u = payload_section<uint32_t>(payloads[0], RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, count);
	const uint32_t *left_welded = payload_section<uint32_t>(payloads[0], RenderingServerTypes::MESH_SUBDIVISION_SECTION_U_WELDED, count);
	const uint32_t *right_r_to_u = payload_section<uint32_t>(payloads[1], RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, count);
	const uint32_t *right_welded = payload_section<uint32_t>(payloads[1], RenderingServerTypes::MESH_SUBDIVISION_SECTION_U_WELDED, count);
	REQUIRE((left_r_to_u && left_welded && right_r_to_u && right_welded));
	for (int r = 0; r < left_positions.size(); r++) {
		left_vertex.insert(left_welded[left_r_to_u[r]], r);
	}
	int seam = 0;
	for (int r = 0; r < right_positions.size(); r++) {
		const int *l = left_vertex.getptr(right_welded[right_r_to_u[r]]);
		if (l) {
			seam++;
			CHECK(memcmp(&left_positions[*l], &right_positions[r], sizeof(Vector3)) == 0);
			CHECK(memcmp(&left_normals[*l], &right_normals[r], sizeof(Vector3)) == 0);
		}
	}
	CHECK(seam == 9); // The column x = 2 at level 2: 2 rows of quads, 4 refined edges each, 9 vertices.

	// The runtime's view: each seam vertex's normal from each owner's payload alone, the same quads in
	// the same order, the same bits.
	const PayloadNormals sides[2] = { payload_normals(payloads[0]), payload_normals(payloads[1]) };
	int shared = 0;
	for (const KeyValue<uint32_t, Vector3> &E : sides[0].normal) {
		const Vector3 *other = sides[1].normal.getptr(E.key);
		if (other == nullptr) {
			continue;
		}
		shared++;
		CHECK(memcmp(&E.value, other, sizeof(Vector3)) == 0);
		CHECK(sides[0].entries[E.key] == sides[1].entries[E.key]);
		CHECK(Math::abs(E.value.y) < 0.999f); // Bent: the seam's normals are not the plane's.
	}
	CHECK(shared == 9);
}

// The runtime passes on the CPU (D122 S2), as subdivision.glsl computes them from the payload alone,
// in real_t and in the shader's order of operations: the cage morphed by every weight over 1e-4 and skinned by the bones (a skeleton
// buffer's 12 floats a bone), the stencil sums, the own refined vertices' normals, and a surface's
// render vertices with their tangents. Both owners' passes give a seam vertex the bits computed here.
struct PassResult {
	LocalVector<Vector3> positions; // Per render vertex.
	LocalVector<Vector3> normals;
	LocalVector<Vector3> tangents;
	LocalVector<float> signs;
};

static float pass_corner_angle(const Vector3 &p_prev, const Vector3 &p_corner, const Vector3 &p_next) {
	const Vector3 to_prev = (p_prev - p_corner).normalized();
	const Vector3 to_next = (p_next - p_corner).normalized();
	return Math::acos(CLAMP(to_prev.dot(to_next), -1.0f, 1.0f));
}

static PassResult run_subdivision_passes(const Ref<ArrayMesh> &p_mesh, int p_surface, const LocalVector<float> &p_bones, const LocalVector<float> &p_weights) {
	PassResult result;
	const Vector<uint8_t> payload = surface_payload(p_mesh, p_surface);
	const Vector<uint8_t> owner_payload = surface_payload(p_mesh, int(RenderingServer::mesh_subdivision_data_owner_surface(payload)));
	uint32_t count = 0;
	const uint32_t *info = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, count);
	const float *rest = payload_section<float>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_POSITIONS, count);
	const uint16_t *skin = payload_section<uint16_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_SKIN, count);
	const uint32_t *blend_offsets = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_OFFSETS, count);
	const uint32_t *blend_entries = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_ENTRIES, count);
	const uint32_t *stencil_offsets = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_OFFSETS, count);
	const uint8_t *stencil_indices = payload_section<uint8_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_INDICES, count);
	const float *stencil_weights = payload_section<float>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_WEIGHTS, count);
	const uint32_t *quads = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_QUADS, count);
	const uint32_t *normal_offsets = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_NORMAL_OFFSETS, count);
	const uint32_t *normal_entries = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_NORMAL_ENTRIES, count);
	const uint32_t *r_to_u = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, count);
	const uint32_t *tangent_offsets = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_OFFSETS, count);
	const uint32_t *tangent_entries = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_ENTRIES, count);
	REQUIRE((info && rest && blend_offsets && stencil_offsets && stencil_indices && stencil_weights && quads && normal_offsets && normal_entries && r_to_u && tangent_offsets && tangent_entries));
	const uint32_t cage_count = info[RenderingServerTypes::MESH_SUBDIVISION_INFO_CAGE_COUNT];
	const uint32_t u_count = info[RenderingServerTypes::MESH_SUBDIVISION_INFO_U_COUNT];
	const bool index_16 = info[RenderingServerTypes::MESH_SUBDIVISION_INFO_STENCIL_INDEX_BYTES] == 2;

	// The cage pass.
	LocalVector<Vector3> cage;
	cage.resize(cage_count);
	for (uint32_t c = 0; c < cage_count; c++) {
		Vector3 vertex(rest[3 * c], rest[3 * c + 1], rest[3 * c + 2]);
		if (!p_weights.is_empty()) {
			Vector3 blend;
			for (uint32_t e = blend_offsets[c]; e < blend_offsets[c + 1]; e++) {
				const float w = p_weights[blend_entries[4 * e]];
				if (Math::abs(w) > 0.0001f) {
					float delta[3];
					memcpy(delta, &blend_entries[4 * e + 1], sizeof(delta));
					blend += Vector3(delta[0], delta[1], delta[2]) * w;
				}
			}
			vertex += blend;
		}
		if (skin && !p_bones.is_empty()) {
			float m[12] = {};
			for (int b = 0; b < 8; b++) {
				const uint32_t bone = skin[16 * c + b];
				const float w = float(skin[16 * c + 8 + b]) / 65535.0f;
				REQUIRE(bone * 12 < p_bones.size());
				for (int k = 0; k < 12; k++) {
					m[k] += p_bones[bone * 12 + k] * w;
				}
			}
			vertex = Vector3(m[0] * vertex.x + m[1] * vertex.y + m[2] * vertex.z + m[3], m[4] * vertex.x + m[5] * vertex.y + m[6] * vertex.z + m[7], m[8] * vertex.x + m[9] * vertex.y + m[10] * vertex.z + m[11]);
		}
		cage[c] = vertex;
	}
	// The stencil pass.
	LocalVector<Vector3> positions;
	positions.resize(u_count);
	for (uint32_t u = 0; u < u_count; u++) {
		Vector3 sum;
		for (uint32_t i = stencil_offsets[u]; i < stencil_offsets[u + 1]; i++) {
			uint32_t c;
			if (index_16) {
				uint16_t narrow;
				memcpy(&narrow, stencil_indices + 2 * i, 2);
				c = narrow;
			} else {
				memcpy(&c, stencil_indices + 4 * i, 4);
			}
			sum += cage[c] * stencil_weights[i];
		}
		positions[u] = sum;
	}
	// The normal pass.
	LocalVector<Vector3> normals;
	normals.resize(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_OWN_U_COUNT]);
	for (uint32_t u = 0; u < normals.size(); u++) {
		Vector3 sum;
		for (uint32_t e = normal_offsets[u]; e < normal_offsets[u + 1]; e++) {
			const uint32_t *quad = &quads[4 * (normal_entries[e] / 4)];
			const uint32_t corner = normal_entries[e] % 4;
			Vector3 p[4];
			for (int c = 0; c < 4; c++) {
				p[c] = positions[quad[c]];
			}
			const Vector3 face = (p[0] - p[2]).cross(p[1] - p[3]).normalized();
			sum += face * pass_corner_angle(p[(corner + 3) % 4], p[corner], p[(corner + 1) % 4]);
		}
		normals[u] = sum.is_zero_approx() ? Vector3(0, 1, 0) : sum.normalized();
	}
	// The write pass.
	const RenderingServerTypes::SurfaceData surface = RS::get_singleton()->mesh_get_surface(p_mesh->get_rid(), p_surface);
	const bool surface_index_16 = surface.vertex_count <= 65536;
	auto index = [&surface, surface_index_16](uint32_t p_i) -> uint32_t {
		if (surface_index_16) {
			uint16_t narrow;
			memcpy(&narrow, surface.index_data.ptr() + 2 * p_i, 2);
			return narrow;
		}
		uint32_t wide;
		memcpy(&wide, surface.index_data.ptr() + 4 * p_i, 4);
		return wide;
	};
	const PackedVector2Array uvs = p_mesh->surface_get_arrays(p_surface)[Mesh::ARRAY_TEX_UV];
	REQUIRE(uint32_t(uvs.size()) == surface.vertex_count);
	for (uint32_t r = 0; r < surface.vertex_count; r++) {
		const uint32_t u = r_to_u[r];
		const Vector3 normal = normals[u];
		result.positions.push_back(positions[u]);
		result.normals.push_back(normal);
		Vector3 tangent_sum;
		Vector3 bitangent_sum;
		for (uint32_t e = tangent_offsets[r]; e < tangent_offsets[r + 1]; e++) {
			const uint32_t triangle = tangent_entries[e] / 3;
			const uint32_t corner = tangent_entries[e] % 3;
			const uint32_t v[3] = { index(3 * triangle), index(3 * triangle + 1), index(3 * triangle + 2) };
			const Vector3 p[3] = { positions[r_to_u[v[0]]], positions[r_to_u[v[1]]], positions[r_to_u[v[2]]] };
			const Vector3 e1 = p[1] - p[0];
			const Vector3 e2 = p[2] - p[0];
			const Vector2 d1 = uvs[v[1]] - uvs[v[0]];
			const Vector2 d2 = uvs[v[2]] - uvs[v[0]];
			const float det = d1.x * d2.y - d2.x * d1.y;
			if (det == 0.0f) {
				continue;
			}
			const float orientation = det < 0.0f ? -1.0f : 1.0f;
			const Vector3 tangent = ((e1 * d2.y - e2 * d1.y) * orientation).normalized();
			const Vector3 bitangent = ((e2 * d1.x - e1 * d2.x) * orientation).normalized();
			const float angle = pass_corner_angle(p[(corner + 2) % 3], p[corner], p[(corner + 1) % 3]);
			tangent_sum += tangent * angle;
			bitangent_sum += bitangent * angle;
		}
		Vector3 t = tangent_sum - normal * normal.dot(tangent_sum);
		if (t.is_zero_approx()) {
			t = normal.get_any_perpendicular();
		}
		t.normalize();
		result.tangents.push_back(t);
		result.signs.push_back(normal.cross(t).dot(bitangent_sum) < 0.0f ? 1.0f : -1.0f);
	}
	return result;
}

// A skeleton buffer's 12 floats a bone: the basis rows, each followed by its origin component.
static void push_bone(LocalVector<float> &r_bones, const Transform3D &p_transform) {
	for (int row = 0; row < 3; row++) {
		for (int column = 0; column < 3; column++) {
			r_bones.push_back(p_transform.basis.rows[row][column]);
		}
		r_bones.push_back(p_transform.origin[row]);
	}
}

TEST_CASE("[SceneTree][SubdivisionBuilder] The runtime passes reproduce the rest pose, keep a posed seam's bits, and stay in the bone bounds") {
	// The bent grid of two owners, skinned to two bones per cage vertex, with two shapes that move
	// cage vertices on and beside the seam column x = 2.
	const TestCage grid = make_grid(4, 2, 2, true);
	TestPrimitive sides[2] = { make_shared_primitive(grid, 0), make_shared_primitive(grid, 1) };
	for (TestPrimitive &side : sides) {
		PackedInt32Array bones;
		PackedFloat32Array weights;
		for (int v = 0; v < side.cage_vertex.size(); v++) {
			const int cage = side.cage_vertex[v];
			const int slots[4] = { cage % 3, (cage + 1) % 3, 0, 0 };
			const float slot_weights[4] = { 0.75f, 0.25f, 0.0f, 0.0f };
			for (int b = 0; b < 4; b++) {
				bones.push_back(slots[b]);
				weights.push_back(slot_weights[b]);
			}
		}
		side.arrays[Mesh::ARRAY_BONES] = bones;
		side.arrays[Mesh::ARRAY_WEIGHTS] = weights;
	}
	LocalVector<HashMap<int, Vector3>> shapes;
	shapes.resize(2);
	shapes[0].insert(7, Vector3(0, 0.3f, 0)); // (2, 1): on the seam.
	shapes[1].insert(7, Vector3(0.05f, 0, 0));
	shapes[1].insert(4, Vector3(0, 0, 0.1f)); // (1, 1): owner 0's, a ghost of owner 1's.
	PackedStringArray names;
	names.push_back("A");
	names.push_back("B");
	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(grid.positions.size());
	builder->set_owner_bone_names(0, test_bone_names());
	builder->set_owner_bone_names(1, test_bone_names());
	builder->add_face_varying_layer(grid_uvs(grid, Vector2(0.25f, 0), Vector2(0, 0.5f)), Mesh::ARRAY_TEX_UV);
	for (int owner = 0; owner < 2; owner++) {
		Array weld;
		weld.push_back(sides[owner].uv_weld);
		builder->add_surface(owner, sides[owner].arrays, make_blend_arrays(sides[owner], shapes), names, sides[owner].cage_vertex, sides[owner].faces, weld, Ref<Material>());
	}
	REQUIRE(builder->build(2) == OK);
	const Ref<ArrayMesh> meshes[2] = { builder->get_owner_mesh(0), builder->get_owner_mesh(1) };
	REQUIRE((meshes[0].is_valid() && meshes[1].is_valid()));

	const Transform3D pose[3] = {
		Transform3D(),
		Transform3D(Basis(Vector3(1, 0, 0), 0.5f), Vector3(0, 0.2f, 0)),
		Transform3D(Basis(Vector3(0, 0, 1), -0.4f), Vector3(0.1f, 0, 0)),
	};
	LocalVector<float> posed;
	for (const Transform3D &bone : pose) {
		push_bone(posed, bone);
	}
	const LocalVector<float> no_weights = { 0.0f, 0.0f };
	const LocalVector<float> weights = { 0.7f, -0.3f };

	PassResult rest[2];
	PassResult moved[2];
	for (int owner = 0; owner < 2; owner++) {
		// At rest, from the payload alone, the surface the builder made: positions, normals, tangents.
		// Unskinned: two quantized skin weights do not sum to exactly 1.
		rest[owner] = run_subdivision_passes(meshes[owner], 0, LocalVector<float>(), no_weights);
		const Array arrays = meshes[owner]->surface_get_arrays(0);
		const PackedVector3Array positions = arrays[Mesh::ARRAY_VERTEX];
		const PackedVector3Array normals = arrays[Mesh::ARRAY_NORMAL];
		const PackedFloat32Array tangents = arrays[Mesh::ARRAY_TANGENT];
		REQUIRE(rest[owner].positions.size() == uint32_t(positions.size()));
		REQUIRE(tangents.size() == 4 * positions.size());
		for (int r = 0; r < positions.size(); r++) {
			CHECK(rest[owner].positions[r].is_equal_approx(positions[r]));
			CHECK(rest[owner].normals[r].dot(normals[r]) > 0.9999f); // The surface's are octahedron-encoded.
			CHECK(rest[owner].tangents[r].dot(Vector3(tangents[4 * r], tangents[4 * r + 1], tangents[4 * r + 2])) > 0.999f);
			CHECK(rest[owner].signs[r] == tangents[4 * r + 3]);
		}
		moved[owner] = run_subdivision_passes(meshes[owner], 0, posed, weights);
	}

	// Posed and morphed, a seam vertex has the same position and normal bits on both owners.
	uint32_t count = 0;
	const Vector<uint8_t> payloads[2] = { surface_payload(meshes[0], 0), surface_payload(meshes[1], 0) };
	const uint32_t *r_to_u[2];
	const uint32_t *welded[2];
	for (int owner = 0; owner < 2; owner++) {
		r_to_u[owner] = payload_section<uint32_t>(payloads[owner], RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, count);
		welded[owner] = payload_section<uint32_t>(payloads[owner], RenderingServerTypes::MESH_SUBDIVISION_SECTION_U_WELDED, count);
		REQUIRE((r_to_u[owner] && welded[owner]));
	}
	HashMap<uint32_t, uint32_t> left_vertex;
	for (uint32_t r = 0; r < moved[0].positions.size(); r++) {
		left_vertex.insert(welded[0][r_to_u[0][r]], r);
	}
	int seam = 0;
	float largest_move = 0.0f;
	for (uint32_t r = 0; r < moved[1].positions.size(); r++) {
		largest_move = MAX(largest_move, moved[1].positions[r].distance_to(rest[1].positions[r]));
		const uint32_t *l = left_vertex.getptr(welded[1][r_to_u[1][r]]);
		if (l == nullptr) {
			continue;
		}
		seam++;
		CHECK(memcmp(&moved[0].positions[*l], &moved[1].positions[r], sizeof(Vector3)) == 0);
		CHECK(memcmp(&moved[0].normals[*l], &moved[1].normals[r], sizeof(Vector3)) == 0);
	}
	CHECK(seam == 9);
	CHECK(largest_move > 0.05f);

	// The bone bounds (the cage vertices weighted to each bone), transformed by the pose as
	// mesh_get_aabb() transforms them, hold every posed render vertex. Morphs are outside them, as on
	// every skinned Godot mesh, so this pose has no weights.
	for (int owner = 0; owner < 2; owner++) {
		const Vector<AABB> bounds = RS::get_singleton()->mesh_get_surface(meshes[owner]->get_rid(), 0).bone_aabbs;
		REQUIRE(bounds.size() == 3);
		AABB merged;
		for (int b = 0; b < bounds.size(); b++) {
			REQUIRE(bounds[b].size.x >= 0.0f); // Every bone is used.
			const AABB transformed = pose[b].xform(bounds[b]);
			merged = b == 0 ? transformed : merged.merge(transformed);
		}
		const PassResult skinned = run_subdivision_passes(meshes[owner], 0, posed, no_weights);
		const AABB grown = merged.grow(1e-4f);
		for (const Vector3 &position : skinned.positions) {
			CHECK(grown.has_point(position));
		}
	}
	// An unskinned cage gives no bone bounds.
	const TestCage plain = make_grid(2, 2, -1, false);
	const Ref<ArrayMesh> unskinned = build_grid(plain, grid_uvs(plain, Vector2(1, 0), Vector2(0, 1)));
	CHECK(RS::get_singleton()->mesh_get_surface(unskinned->get_rid(), 0).bone_aabbs.is_empty());
}

TEST_CASE("[SceneTree][SubdivisionBuilder] The same input builds the same bytes") {
	const TestCage strip = make_strip();
	LocalVector<HashMap<int, Vector3>> shapes;
	shapes.resize(2);
	shapes[0].insert(6, Vector3(0, 0.5f, 0));
	shapes[1].insert(3, Vector3(0.25f, 0, 0));
	PackedStringArray names;
	names.push_back("Lift");
	names.push_back("Push");
	Ref<ArrayMesh> built[2][2];
	for (int run = 0; run < 2; run++) {
		PackedVector2Array uv_values;
		const TestPrimitive left = make_primitive(strip, 0, uv_values);
		const TestPrimitive right = make_primitive(strip, 1, uv_values);
		Ref<SubdivisionBuilder> builder;
		builder.instantiate();
		builder->set_cage_vertex_count(strip.positions.size());
		builder->set_owner_bone_names(0, test_bone_names());
		builder->set_owner_bone_names(1, test_bone_names());
		builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
		Array left_weld;
		left_weld.push_back(left.uv_weld);
		Array right_weld;
		right_weld.push_back(right.uv_weld);
		builder->add_surface(0, left.arrays, make_blend_arrays(left, shapes), names, left.cage_vertex, left.faces, left_weld, Ref<Material>());
		builder->add_surface(1, right.arrays, make_blend_arrays(right, shapes), names, right.cage_vertex, right.faces, right_weld, Ref<Material>());
		REQUIRE(builder->build(2) == OK);
		built[run][0] = builder->get_owner_mesh(0);
		built[run][1] = builder->get_owner_mesh(1);
	}
	for (int owner = 0; owner < 2; owner++) {
		REQUIRE(built[0][owner]->get_surface_count() == built[1][owner]->get_surface_count());
		for (int s = 0; s < built[0][owner]->get_surface_count(); s++) {
			const RenderingServerTypes::SurfaceData a = RS::get_singleton()->mesh_get_surface(built[0][owner]->get_rid(), s);
			const RenderingServerTypes::SurfaceData b = RS::get_singleton()->mesh_get_surface(built[1][owner]->get_rid(), s);
			CHECK(a.vertex_data == b.vertex_data);
			CHECK(a.attribute_data == b.attribute_data);
			CHECK(a.index_data == b.index_data);
			CHECK(a.cluster_data == b.cluster_data);
			CHECK(a.subdivision_data == b.subdivision_data);
		}
	}
}

TEST_CASE("[SceneTree][SubdivisionBuilder] Shapes that move a shared cage vertex keep one order in both owners") {
	const TestCage strip = make_strip();
	LocalVector<HashMap<int, Vector3>> shapes;
	shapes.resize(2);
	shapes[0].insert(4, Vector3(0, 0.5f, 0)); // Vertex 4 is on the seam, held by both owners.
	shapes[1].insert(4, Vector3(0.25f, 0, 0));
	PackedStringArray forward;
	forward.push_back("A");
	forward.push_back("B");
	PackedStringArray backward;
	backward.push_back("B");
	backward.push_back("A");
	LocalVector<HashMap<int, Vector3>> swapped;
	swapped.resize(2);
	swapped[0].insert(4, Vector3(0.25f, 0, 0));
	swapped[1].insert(4, Vector3(0, 0.5f, 0));
	for (int reversed = 0; reversed < 2; reversed++) {
		PackedVector2Array uv_values;
		const TestPrimitive left = make_primitive(strip, 0, uv_values);
		const TestPrimitive right = make_primitive(strip, 1, uv_values);
		Ref<SubdivisionBuilder> builder;
		builder.instantiate();
		builder->set_cage_vertex_count(strip.positions.size());
		builder->set_owner_bone_names(0, test_bone_names());
		builder->set_owner_bone_names(1, test_bone_names());
		builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
		Array left_weld;
		left_weld.push_back(left.uv_weld);
		Array right_weld;
		right_weld.push_back(right.uv_weld);
		builder->add_surface(0, left.arrays, make_blend_arrays(left, shapes), forward, left.cage_vertex, left.faces, left_weld, Ref<Material>());
		if (reversed) {
			builder->add_surface(1, right.arrays, make_blend_arrays(right, swapped), backward, right.cage_vertex, right.faces, right_weld, Ref<Material>());
			ERR_PRINT_OFF;
			CHECK(builder->build(2) == ERR_INVALID_DATA);
			ERR_PRINT_ON;
		} else {
			builder->add_surface(1, right.arrays, make_blend_arrays(right, shapes), forward, right.cage_vertex, right.faces, right_weld, Ref<Material>());
			CHECK(builder->build(2) == OK);
		}
	}
}

TEST_CASE("[SceneTree][SubdivisionBuilder] A ghost's bones are its receiving owner's bind indices, found by name") {
	const TestCage strip = make_strip();
	PackedStringArray reversed_names;
	reversed_names.push_back("B2");
	reversed_names.push_back("B1");
	reversed_names.push_back("B0");
	// Owner 1 binds the same joints in reverse: its bind index of joint B(c % 3) is 2 - c % 3.
	auto reverse_bones = [](TestPrimitive &r_primitive) {
		PackedInt32Array bones = r_primitive.arrays[Mesh::ARRAY_BONES];
		for (int v = 0; v < r_primitive.cage_vertex.size(); v++) {
			bones.write[4 * v] = 2 - r_primitive.cage_vertex[v] % 3;
		}
		r_primitive.arrays[Mesh::ARRAY_BONES] = bones;
	};
	{
		PackedVector2Array uv_values;
		const TestPrimitive left = make_primitive(strip, 0, uv_values);
		TestPrimitive right = make_primitive(strip, 1, uv_values);
		reverse_bones(right);
		Ref<SubdivisionBuilder> builder;
		builder.instantiate();
		builder->set_cage_vertex_count(strip.positions.size());
		builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
		Array left_weld;
		left_weld.push_back(left.uv_weld);
		Array right_weld;
		right_weld.push_back(right.uv_weld);
		builder->add_surface(0, left.arrays, Array(), PackedStringArray(), left.cage_vertex, left.faces, left_weld, Ref<Material>());
		builder->add_surface(1, right.arrays, Array(), PackedStringArray(), right.cage_vertex, right.faces, right_weld, Ref<Material>());
		// Two skinned owners without names: their bind indices cannot be matched.
		ERR_PRINT_OFF;
		CHECK(builder->build(2) == ERR_INVALID_PARAMETER);
		ERR_PRINT_ON;
		builder->set_owner_bone_names(0, test_bone_names());
		builder->set_owner_bone_names(1, reversed_names);
		REQUIRE(builder->build(2) == OK);
		for (int owner = 0; owner < 2; owner++) {
			const Vector<uint8_t> payload = surface_payload(builder->get_owner_mesh(owner), 0);
			uint32_t count = 0;
			const uint32_t *welded = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_WELDED, count);
			const uint32_t cage_count = count;
			const uint16_t *skin = payload_section<uint16_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_SKIN, count);
			REQUIRE((welded && skin));
			for (uint32_t c = 0; c < cage_count; c++) {
				CHECK(skin[16 * c] == (owner == 0 ? welded[c] % 3 : 2 - welded[c] % 3));
				CHECK(skin[16 * c + 8] == 65535);
				CHECK(skin[16 * c + 1] == 0); // An unweighted slot.
				CHECK(skin[16 * c + 9] == 0);
			}
		}
	}
	{
		// Owner 1 weights its vertices at x = 3 (6 and 7, owner 0's ghosts) to a joint owner 0 lacks.
		PackedVector2Array uv_values;
		const TestPrimitive left = make_primitive(strip, 0, uv_values);
		TestPrimitive right = make_primitive(strip, 1, uv_values);
		PackedInt32Array bones = right.arrays[Mesh::ARRAY_BONES];
		for (int v = 0; v < right.cage_vertex.size(); v++) {
			if (right.cage_vertex[v] == 6 || right.cage_vertex[v] == 7) {
				bones.write[4 * v] = 3;
			}
		}
		right.arrays[Mesh::ARRAY_BONES] = bones;
		PackedStringArray right_names = test_bone_names();
		right_names.push_back("B3");
		Ref<SubdivisionBuilder> builder;
		builder.instantiate();
		builder->set_cage_vertex_count(strip.positions.size());
		builder->set_owner_bone_names(0, test_bone_names());
		builder->set_owner_bone_names(1, right_names);
		builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
		Array left_weld;
		left_weld.push_back(left.uv_weld);
		Array right_weld;
		right_weld.push_back(right.uv_weld);
		builder->add_surface(0, left.arrays, Array(), PackedStringArray(), left.cage_vertex, left.faces, left_weld, Ref<Material>());
		builder->add_surface(1, right.arrays, Array(), PackedStringArray(), right.cage_vertex, right.faces, right_weld, Ref<Material>());
		ERR_PRINT_OFF;
		CHECK(builder->build(2) == ERR_INVALID_DATA);
		ERR_PRINT_ON;
	}
	{
		// A joint name twice in one owner's list would fold two bind indices into one: refused.
		PackedVector2Array uv_values;
		const TestPrimitive left = make_primitive(strip, 0, uv_values);
		const TestPrimitive right = make_primitive(strip, 1, uv_values);
		PackedStringArray repeated = test_bone_names();
		repeated.set(2, repeated[0]);
		Ref<SubdivisionBuilder> builder;
		builder.instantiate();
		builder->set_cage_vertex_count(strip.positions.size());
		builder->set_owner_bone_names(0, test_bone_names());
		builder->set_owner_bone_names(1, repeated);
		builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
		Array left_weld;
		left_weld.push_back(left.uv_weld);
		Array right_weld;
		right_weld.push_back(right.uv_weld);
		builder->add_surface(0, left.arrays, Array(), PackedStringArray(), left.cage_vertex, left.faces, left_weld, Ref<Material>());
		builder->add_surface(1, right.arrays, Array(), PackedStringArray(), right.cage_vertex, right.faces, right_weld, Ref<Material>());
		ERR_PRINT_OFF;
		CHECK(builder->build(2) == ERR_INVALID_PARAMETER);
		ERR_PRINT_ON;
	}
}

} // namespace TestSubdivisionBuilder
