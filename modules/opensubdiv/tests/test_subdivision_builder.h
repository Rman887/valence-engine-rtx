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

TEST_CASE("[SceneTree][SubdivisionBuilder] An owner without blend shapes gets a cluster LOD hierarchy, one with them flat clusters, whatever the settings") {
	if (RenderingServer::build_clusters_func == nullptr || RenderingServer::build_cluster_lod_func == nullptr) {
		return; // No cluster builders without the meshoptimizer module.
	}
	const Variant cluster_blas = GLOBAL_GET("rendering/pathtracing/cluster_blas");
	const Variant cluster_lod = GLOBAL_GET("rendering/pathtracing/cluster_lod");
	const TestCage grid = make_grid(6, 6, -1, true); // 1,152 refined triangles, past one cluster.
	const PackedVector2Array uv_values = grid_uvs(grid, Vector2(1, 0), Vector2(0, 1));
	auto built_clusters = [&](bool p_shapes, bool p_settings_on) -> Vector<uint8_t> {
		ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", p_settings_on);
		ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_lod", p_settings_on);
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
		return surface.get("cluster_data", Vector<uint8_t>());
	};
	// The builder clusters a subdivided surface itself (its index order is frozen at build), so the table and the
	// order do not follow the project's cluster settings: the hierarchy without blend shapes (D85), the flat
	// clusters with them (D87).
	Vector<uint8_t> clusters[2][2];
	for (int shapes = 0; shapes < 2; shapes++) {
		for (int on = 0; on < 2; on++) {
			clusters[shapes][on] = built_clusters(shapes, on);
		}
	}
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", cluster_blas);
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_lod", cluster_lod);
	for (int shapes = 0; shapes < 2; shapes++) {
		REQUIRE(clusters[shapes][0].size() >= int(sizeof(uint32_t)));
		uint32_t version;
		memcpy(&version, clusters[shapes][0].ptr(), sizeof(uint32_t));
		CHECK(version == (shapes ? RenderingServerTypes::MESH_CLUSTER_DATA_VERSION : RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION));
		CHECK(clusters[shapes][0] == clusters[shapes][1]);
	}
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

// SurfaceTool's MikkTSpace on triangles given per corner (deindexed), each corner's tangent and sign.
static LocalVector<Vector4> surface_tool_frames(const LocalVector<Vector3> &p_positions, const LocalVector<Vector3> &p_normals, const LocalVector<Vector2> &p_uvs) {
	Ref<SurfaceTool> surface_tool;
	surface_tool.instantiate();
	surface_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
	for (uint32_t i = 0; i < p_positions.size(); i++) {
		surface_tool->set_normal(p_normals[i]);
		surface_tool->set_uv(p_uvs[i]);
		surface_tool->add_vertex(p_positions[i]);
	}
	surface_tool->generate_tangents();
	const PackedFloat32Array tangents = surface_tool->commit_to_arrays()[Mesh::ARRAY_TANGENT];
	LocalVector<Vector4> frames;
	for (int i = 0; i < tangents.size() / 4; i++) {
		frames.push_back(Vector4(tangents[4 * i], tangents[4 * i + 1], tangents[4 * i + 2], tangents[4 * i + 3]));
	}
	return frames;
}

// SurfaceTool's view of quads given in the source's order with UVs v up: each quad as the two triangles along
// its 0-2 diagonal, in Godot's winding (clockwise) with UVs v down, deindexed. r_corner gets each triangle
// corner's quad corner.
static LocalVector<Vector4> surface_tool_quad_frames(const LocalVector<Vector3> &p_positions, const LocalVector<Vector3> &p_normals, const LocalVector<Vector2> &p_uvs, LocalVector<uint32_t> &r_corner) {
	LocalVector<Vector3> positions;
	LocalVector<Vector3> normals;
	LocalVector<Vector2> uvs;
	r_corner.clear();
	for (uint32_t q = 0; q < p_positions.size() / 4; q++) {
		const int order[6] = { 0, 2, 1, 0, 3, 2 }; // (0, 1, 2) and (0, 2, 3), reversed.
		for (int i : order) {
			positions.push_back(p_positions[4 * q + i]);
			normals.push_back(p_normals[4 * q + i]);
			uvs.push_back(Vector2(p_uvs[4 * q + i].x, 1.0f - p_uvs[4 * q + i].y));
			r_corner.push_back(4 * q + i);
		}
	}
	return surface_tool_frames(positions, normals, uvs);
}

TEST_CASE("[SceneTree][SubdivisionBuilder] MikkTSpace on quads: SurfaceTool's frames where a quad is two triangles, the larger triangle's orientation on a badly mapped one") {
	// A 2 x 1 strip of unit quads in the y = 0 plane, normal +y, its u mirrored at x = 1: the column there has
	// corners of both orientations, which MikkTSpace groups apart. Corners in the source's order, UVs v up.
	LocalVector<Vector3> positions;
	LocalVector<Vector3> normals;
	LocalVector<Vector2> uvs;
	auto add_quad = [&](const Vector3 *p_corners, const Vector2 *p_uvs) {
		for (int c = 0; c < 4; c++) {
			positions.push_back(p_corners[c]);
			normals.push_back(Vector3(0, 1, 0));
			uvs.push_back(p_uvs[c]);
		}
	};
	for (int x = 0; x < 2; x++) {
		const Vector3 corners[4] = { Vector3(x, 0, 0), Vector3(x + 1, 0, 0), Vector3(x + 1, 0, 1), Vector3(x, 0, 1) };
		const float u0 = x == 0 ? 0.0f : 1.0f;
		const float u1 = x == 0 ? 1.0f : 0.0f;
		const Vector2 quad_uvs[4] = { Vector2(u0, 0), Vector2(u1, 0), Vector2(u1, 1), Vector2(u0, 1) };
		add_quad(corners, quad_uvs);
	}
	LocalVector<Vector4> frames;
	REQUIRE(SubdivisionBuilder::mikktspace_quad_frames(positions, normals, uvs, frames));
	// The same quads as Godot's triangles along MikkTSpace's diagonal (0-2: both UV diagonals are equal, and so
	// are the position ones), through SurfaceTool: the same signs, and the same tangents.
	LocalVector<uint32_t> tri_corner;
	const LocalVector<Vector4> theirs = surface_tool_quad_frames(positions, normals, uvs, tri_corner);
	REQUIRE(theirs.size() == tri_corner.size());
	for (uint32_t i = 0; i < theirs.size(); i++) {
		const Vector4 &ours = frames[tri_corner[i]];
		CHECK(ours.w == theirs[i].w);
		CHECK(Vector3(ours.x, ours.y, ours.z).dot(Vector3(theirs[i].x, theirs[i].y, theirs[i].z)) > 0.999f);
	}
	// The mirror column's corners: one frame on each side, the two signs opposite.
	CHECK(frames[1].w == -frames[4].w);
	CHECK(frames[2].w == -frames[7].w);

	// A badly mapped quad: its UVs folded, so its triangles along the shorter UV diagonal (0-2) have opposite
	// orientations, the first the larger. MikkTSpace gives every corner the larger triangle's orientation;
	// SurfaceTool on the two triangles gives each its own.
	positions.clear();
	normals.clear();
	uvs.clear();
	const Vector3 corners[4] = { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 0, 1), Vector3(0, 0, 1) };
	const Vector2 folded[4] = { Vector2(0, 0), Vector2(2, 0), Vector2(0.5f, 0.5f), Vector2(0.4f, 0.1f) };
	add_quad(corners, folded);
	const float area_012 = (folded[1] - folded[0]).cross(folded[2] - folded[0]);
	const float area_023 = (folded[2] - folded[0]).cross(folded[3] - folded[0]);
	REQUIRE(area_012 * area_023 < 0.0f);
	REQUIRE(Math::abs(area_012) > Math::abs(area_023));
	REQUIRE((folded[2] - folded[0]).length_squared() < (folded[3] - folded[1]).length_squared());
	REQUIRE(SubdivisionBuilder::mikktspace_quad_frames(positions, normals, uvs, frames));
	const LocalVector<Vector4> per_triangle = surface_tool_quad_frames(positions, normals, uvs, tri_corner);
	REQUIRE(per_triangle.size() == 6);
	CHECK(per_triangle[0].w == -per_triangle[3].w); // SurfaceTool: each triangle its own orientation.
	for (int c = 0; c < 4; c++) {
		CHECK(frames[c].w == per_triangle[0].w); // MikkTSpace on the quad: the larger triangle's.
	}
}

// One owner's grid built with one UV layer, its welds as given, and the built quads recorded.
static Ref<ArrayMesh> build_recorded(const TestCage &p_grid, const TestPrimitive &p_primitive, const PackedVector2Array &p_uv_values, LocalVector<LocalVector<int32_t>> &r_quads, Ref<SubdivisionBuilder> &r_builder) {
	r_builder.instantiate();
	r_builder->set_cage_vertex_count(p_grid.positions.size());
	r_builder->add_face_varying_layer(p_uv_values, Mesh::ARRAY_TEX_UV);
	Array weld;
	weld.push_back(p_primitive.uv_weld);
	r_builder->add_surface(0, p_primitive.arrays, Array(), PackedStringArray(), p_primitive.cage_vertex, p_primitive.faces, weld, Ref<Material>());
	REQUIRE(r_builder->build_recording_quads(2, r_quads) == OK);
	return r_builder->get_owner_mesh(0);
}

// Every built quad corner against MikkTSpace on the built quads (their render vertices' rest positions, normals
// and UVs, v up). Every render vertex's sign is its corners' group's. Its tangent is compared, within 0.99, where
// MikkTSpace splits every quad around it along our diagonal (0-2): where its shorter UV diagonal is 1-3 it sums
// other triangles than ours, the per-frame triangulation the design leaves out of T2. Counts the badly mapped
// quads (their two triangles along MikkTSpace's diagonal disagree in orientation): all, and those whose corners'
// corner tangents were compared.
struct BuiltFrames {
	int bad = 0;
	int bad_compared = 0;
	int vertices_compared = 0;
	int vertices = 0;
};

static BuiltFrames check_built_frames(const Ref<ArrayMesh> &p_mesh, const LocalVector<int32_t> &p_quads) {
	const Array arrays = p_mesh->surface_get_arrays(0);
	const PackedVector3Array positions = arrays[Mesh::ARRAY_VERTEX];
	const PackedVector3Array normals = arrays[Mesh::ARRAY_NORMAL];
	const PackedVector2Array stored_uvs = arrays[Mesh::ARRAY_TEX_UV];
	const PackedFloat32Array tangents = arrays[Mesh::ARRAY_TANGENT];
	LocalVector<Vector3> corner_positions;
	LocalVector<Vector3> corner_normals;
	LocalVector<Vector2> corner_uvs;
	for (int32_t r : p_quads) {
		corner_positions.push_back(positions[r]);
		corner_normals.push_back(normals[r]);
		corner_uvs.push_back(Vector2(stored_uvs[r].x, 1.0f - stored_uvs[r].y));
	}
	LocalVector<Vector4> frames;
	REQUIRE(SubdivisionBuilder::mikktspace_quad_frames(corner_positions, corner_normals, corner_uvs, frames));
	const uint32_t quad_count = p_quads.size() / 4;
	LocalVector<uint8_t> diagonal_02;
	LocalVector<uint8_t> bad;
	diagonal_02.resize(quad_count);
	bad.resize(quad_count);
	LocalVector<uint8_t> other_diagonal; // Per render vertex: a quad around it splits along 1-3.
	other_diagonal.resize(positions.size());
	memset(other_diagonal.ptr(), 0, other_diagonal.size());
	BuiltFrames result;
	result.vertices = positions.size();
	for (uint32_t q = 0; q < quad_count; q++) {
		const Vector2 *uv = &corner_uvs[4 * q];
		// MikkTSpace's choice: the shorter UV diagonal, a tie broken by the shorter position diagonal, then 0-2.
		const float uv_02 = (uv[2] - uv[0]).length_squared();
		const float uv_13 = (uv[3] - uv[1]).length_squared();
		const Vector3 *p = &corner_positions[4 * q];
		diagonal_02[q] = uv_02 < uv_13 || (uv_02 == uv_13 && !(float((p[3] - p[1]).length_squared()) < float((p[2] - p[0]).length_squared())));
		const float a = diagonal_02[q] ? (uv[1] - uv[0]).cross(uv[2] - uv[0]) : (uv[1] - uv[0]).cross(uv[3] - uv[0]);
		const float b = diagonal_02[q] ? (uv[2] - uv[0]).cross(uv[3] - uv[0]) : (uv[2] - uv[1]).cross(uv[3] - uv[1]);
		bad[q] = a * b < 0.0f;
		result.bad += bad[q];
		for (int c = 0; c < 4; c++) {
			other_diagonal[p_quads[4 * q + c]] |= !diagonal_02[q];
		}
	}
	LocalVector<uint8_t> compared;
	compared.resize(positions.size());
	memset(compared.ptr(), 0, compared.size());
	for (uint32_t q = 0; q < quad_count; q++) {
		bool any_compared = false;
		for (int c = 0; c < 4; c++) {
			const int32_t r = p_quads[4 * q + c];
			const Vector4 &frame = frames[4 * q + c];
			CHECK(tangents[4 * r + 3] == frame.w);
			if (other_diagonal[r]) {
				continue;
			}
			any_compared = true;
			const float dot = Vector3(tangents[4 * r], tangents[4 * r + 1], tangents[4 * r + 2]).dot(Vector3(frame.x, frame.y, frame.z));
			CHECK_MESSAGE(dot > 0.99f, vformat("quad %d corner %d: tangent dot %f, bad %d", q, c, dot, int(bad[q])));
			compared[r] = 1;
		}
		result.bad_compared += bad[q] && any_compared ? 1 : 0;
	}
	for (uint8_t value : compared) {
		result.vertices_compared += value;
	}
	return result;
}

TEST_CASE("[SceneTree][SubdivisionBuilder] A UV mirror splits render vertices by MikkTSpace's groups, each with SurfaceTool's sign") {
	// A 2 x 2 grid whose u mirrors at x = 1, then one whose v mirrors at z = 1, welded at the mirror: the refined
	// vertices on the mirror line have corners of both orientations under one UV. The v mirror keeps the tangent
	// (1, 0, 0) on both sides and flips only the sign, so a group key that dropped the sign would merge them.
	const TestCage grid = make_grid(2, 2, -1, false);
	for (int mirror = 0; mirror < 2; mirror++) {
		PackedVector2Array values;
		for (const Vector3 &position : grid.positions) {
			values.push_back(mirror == 0 ? Vector2(Math::abs(position.x - 1.0f), position.z) : Vector2(position.x, Math::abs(position.z - 1.0f)));
		}
		LocalVector<LocalVector<int32_t>> quads;
		Ref<SubdivisionBuilder> builder;
		const Ref<ArrayMesh> mesh = build_recorded(grid, make_shared_primitive(grid, 0), values, quads, builder);
		REQUIRE(mesh.is_valid());
		REQUIRE(quads.size() == 1);
		// Level 2 of a 2 x 2 grid is 9 x 9 refined vertices, 9 of them on the mirror line: each is split once.
		const Dictionary owner = Dictionary(builder->get_report()["owners"])[0];
		const Array splits = owner["tangent_splits"];
		REQUIRE(splits.size() == 1);
		CHECK(int(splits[0]) == 9);
		const Array arrays = mesh->surface_get_arrays(0);
		const PackedVector3Array positions = arrays[Mesh::ARRAY_VERTEX];
		const PackedVector3Array normals = arrays[Mesh::ARRAY_NORMAL];
		const PackedVector2Array uvs = arrays[Mesh::ARRAY_TEX_UV];
		const PackedFloat32Array tangents = arrays[Mesh::ARRAY_TANGENT];
		const PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
		CHECK(positions.size() == 81 + 9);
		// SurfaceTool's MikkTSpace on the built triangles, deindexed: every corner's sign is its render vertex's.
		LocalVector<Vector3> corner_positions;
		LocalVector<Vector3> corner_normals;
		LocalVector<Vector2> corner_uvs;
		for (int i = 0; i < indices.size(); i++) {
			corner_positions.push_back(positions[indices[i]]);
			corner_normals.push_back(normals[indices[i]]);
			corner_uvs.push_back(uvs[indices[i]]);
		}
		const LocalVector<Vector4> theirs = surface_tool_frames(corner_positions, corner_normals, corner_uvs);
		REQUIRE(int(theirs.size()) == indices.size());
		for (int i = 0; i < indices.size(); i++) {
			const int r = indices[i];
			CHECK(tangents[4 * r + 3] == theirs[i].w);
			CHECK(Vector3(tangents[4 * r], tangents[4 * r + 1], tangents[4 * r + 2]).dot(Vector3(theirs[i].x, theirs[i].y, theirs[i].z)) > 0.99f);
		}
		// MikkTSpace on the built quads agrees with the built signs, and with the tangents where it splits the
		// quads as we do; a smooth mirror maps no quad badly.
		const BuiltFrames built = check_built_frames(mesh, quads[0]);
		CHECK(built.bad == 0);
		CHECK(built.vertices_compared > built.vertices / 2);
		// The copies keep their refined vertex: same position, normal and UV bits, opposite signs.
		const Vector<uint8_t> payload = surface_payload(mesh, 0);
		uint32_t count = 0;
		const uint32_t *r_to_u = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, count);
		REQUIRE(count == uint32_t(positions.size()));
		// Each copy comes right after the vertex it was split from.
		int pairs = 0;
		for (int r = 1; r < positions.size(); r++) {
			if (r_to_u[r] != r_to_u[r - 1]) {
				continue;
			}
			pairs++;
			CHECK(positions[r - 1] == positions[r]);
			CHECK(normals[r - 1] == normals[r]);
			CHECK(uvs[r - 1] == uvs[r]);
			CHECK(tangents[4 * (r - 1) + 3] == -tangents[4 * r + 3]);
		}
		CHECK(pairs == 9);
		// The payload's sign bits, read directly: bit r % 32 of word r / 32 set where w is +1, every vertex, copies
		// included, and none past the last.
		const uint32_t *signs = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_SIGNS, count);
		REQUIRE(signs != nullptr);
		REQUIRE(count == uint32_t(positions.size() + 31) / 32);
		int negative = 0;
		for (int r = 0; r < positions.size(); r++) {
			const bool bit = (signs[r / 32] >> (r % 32)) & 1;
			CHECK(bit == (tangents[4 * r + 3] > 0.0f));
			negative += bit ? 0 : 1;
		}
		CHECK(negative > 0);
		CHECK(negative < positions.size());
		REQUIRE(positions.size() % 32 != 0);
		CHECK((signs[count - 1] >> (positions.size() % 32)) == 0);
		check_payload_reproduces_surface(mesh, 0);
	}
}

TEST_CASE("[SceneTree][SubdivisionBuilder] Badly mapped refined quads keep MikkTSpace's tangents and signs") {
	// A 2 x 2 grid, each face its own UV island, the first face's UVs folded across itself (its corners 2 and 3
	// moved so the edges 1-2 and 3-0 cross, without collapsing any UV): its refined quads include badly mapped
	// ones, whose triangles disagree in orientation.
	const TestCage grid = make_grid(2, 2, -1, false);
	PackedVector2Array values;
	TestPrimitive primitive = make_primitive(grid, 0, values);
	values.write[2] = Vector2(0.1f, 0.8f);
	values.write[3] = Vector2(0.9f, 0.7f);
	LocalVector<LocalVector<int32_t>> quads;
	Ref<SubdivisionBuilder> builder;
	const Ref<ArrayMesh> mesh = build_recorded(grid, primitive, values, quads, builder);
	REQUIRE(mesh.is_valid());
	REQUIRE(quads.size() == 1);
	const Dictionary owner = Dictionary(builder->get_report()["owners"])[0];
	CHECK(int(Array(owner["tangent_splits"])[0]) > 0);
	const BuiltFrames built = check_built_frames(mesh, quads[0]);
	CHECK(built.bad > 0);
	CHECK_MESSAGE(built.bad_compared > 0, vformat("%d badly mapped quads, no corner of one compared", built.bad));
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
			uv_values.push_back(square[c] + Vector2(2 * f, 0));
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
	const Array ancestry = builder->get_owner_triangle_sources(0);
	REQUIRE(ancestry.size() == 1);
	const Dictionary triangle_source = ancestry[0];
	const PackedInt32Array parent_faces = triangle_source["source_faces"];
	REQUIRE(parent_faces.size() == 96);
	int parent_counts[4] = {};
	for (int32_t face : parent_faces) {
		REQUIRE(face >= 0);
		REQUIRE(face < 4);
		parent_counts[face]++;
	}
	for (int count : parent_counts) {
		CHECK(count == 24); // A coarse triangle becomes three quads, then twelve, then 24 triangles.
	}
	const Array refined = mesh->surface_get_arrays(0);
	const PackedVector3Array positions = refined[Mesh::ARRAY_VERTEX];
	const PackedVector3Array normals = refined[Mesh::ARRAY_NORMAL];
	const PackedVector2Array parent_uvs = refined[Mesh::ARRAY_TEX_UV];
	const PackedInt32Array parent_indices = refined[Mesh::ARRAY_INDEX];
	for (int t = 0; t < parent_faces.size(); t++) {
		const float u = (parent_uvs[parent_indices[3 * t]].x + parent_uvs[parent_indices[3 * t + 1]].x + parent_uvs[parent_indices[3 * t + 2]].x) / 3;
		CHECK(parent_faces[t] == int(Math::floor(u / 2)));
	}
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

TEST_CASE("[SceneTree][SubdivisionBuilder] Triangle ancestry follows the final indices and source surfaces") {
	TestCage cube = make_cube();
	for (int f = 0; f < 6; f++) {
		cube.owner[f] = f / 2; // Three input primitives, two of them assigned to output owner 0.
	}
	for (bool blend_shapes : { false, true }) {
		PackedVector2Array uv_values;
		TestPrimitive primitives[3];
		for (int s = 0; s < 3; s++) {
			primitives[s] = make_primitive(cube, s, uv_values);
		}
		// Each source face has a distinct UV island. The final triangle's UVs provide an
		// independent oracle for ancestry, unaffected by meshlet reordering or tangent splits.
		for (int s = 0; s < 3; s++) {
			for (int f = 0; f < 2; f++) {
				for (int c = 0; c < 4; c++) {
					uv_values.write[8 * s + 4 * f + c] += Vector2(2 * f, 3 * s);
				}
			}
		}
		Ref<SubdivisionBuilder> builder;
		builder.instantiate();
		builder->set_cage_vertex_count(8);
		builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
		builder->set_owner_bone_names(0, test_bone_names());
		builder->set_owner_bone_names(1, test_bone_names());
		PackedStringArray names;
		LocalVector<HashMap<int, Vector3>> shapes;
		if (blend_shapes) {
			names.push_back("shape");
			shapes.resize(1);
			shapes[0][0] = Vector3(0.01, 0, 0);
		}
		for (int s = 0; s < 3; s++) {
			Array weld;
			weld.push_back(primitives[s].uv_weld);
			builder->add_surface(s < 2 ? 0 : 1, primitives[s].arrays, make_blend_arrays(primitives[s], shapes), names, primitives[s].cage_vertex, primitives[s].faces, weld, Ref<Material>());
		}
		for (int level : { 1, 3 }) { // Level 3 also reaches the hierarchy builder without blend shapes.
			REQUIRE(builder->build(level) == OK);
			for (int owner = 0; owner < 2; owner++) {
				const Ref<ArrayMesh> mesh = builder->get_owner_mesh(owner);
				Array sources = builder->get_owner_triangle_sources(owner);
				REQUIRE(sources.size() == mesh->get_surface_count());
				for (int s = 0; s < sources.size(); s++) {
					const Dictionary entry = sources[s];
					const int source_surface = owner == 0 ? s : 2;
					CHECK(int(entry["source_surface"]) == source_surface);
					const PackedInt32Array faces = entry["source_faces"];
					const Array arrays = mesh->surface_get_arrays(s);
					const PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
					const PackedVector2Array uvs = arrays[Mesh::ARRAY_TEX_UV];
					if (level == 3 && RenderingServer::build_clusters_func != nullptr && RenderingServer::build_cluster_lod_func != nullptr) {
						const Dictionary stored = Array(mesh->get("_surfaces"))[s];
						const Vector<uint8_t> clusters = stored.get("cluster_data", Vector<uint8_t>());
						REQUIRE(clusters.size() >= int(sizeof(uint32_t)));
						uint32_t version;
						memcpy(&version, clusters.ptr(), sizeof(uint32_t));
						CHECK(version == (blend_shapes ? RenderingServerTypes::MESH_CLUSTER_DATA_VERSION : RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION));
					}
					REQUIRE(faces.size() == indices.size() / 3);
					int counts[2] = {};
					for (int t = 0; t < faces.size(); t++) {
						const Vector2 uv = (uvs[indices[3 * t]] + uvs[indices[3 * t + 1]] + uvs[indices[3 * t + 2]]) / 3;
						const int expected = int(Math::floor(uv.x / 2));
						REQUIRE(expected >= 0);
						REQUIRE(expected < 2);
						CHECK(faces[t] == expected);
						CHECK(int(Math::floor((1 - uv.y) / 3)) == source_surface);
						counts[expected]++;
					}
					CHECK(counts[0] == 2 * (1 << (2 * level)));
					CHECK(counts[1] == counts[0]);
				}
				// The outer dictionaries and packed arrays cannot mutate the builder's result.
				Dictionary changed = sources[0];
				PackedInt32Array changed_faces = changed["source_faces"];
				changed_faces.set(0, -1);
				changed["source_faces"] = changed_faces;
				sources.clear();
				const Dictionary unchanged = builder->get_owner_triangle_sources(owner)[0];
				CHECK(PackedInt32Array(unchanged["source_faces"])[0] >= 0);
			}
		}
		builder->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_NORMALIZED);
		ERR_PRINT_OFF;
		CHECK(builder->get_owner_triangle_sources(0).is_empty());
		ERR_PRINT_ON;
		REQUIRE(builder->build(2) == OK);
		ERR_PRINT_OFF;
		CHECK(builder->build(0) == ERR_INVALID_PARAMETER);
		CHECK(builder->get_owner_triangle_sources(0).is_empty());
		CHECK(builder->get_owner_mesh(0).is_null());
		ERR_PRINT_ON;
		REQUIRE(builder->build(2) == OK);
		builder->clear();
		ERR_PRINT_OFF;
		CHECK(builder->get_owner_triangle_sources(0).is_empty());
		ERR_PRINT_ON;
	}
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

TEST_CASE("[SceneTree][SubdivisionBuilder] A seam MikkTSpace splits keeps every copy's position and normal bits on both owners") {
	// The bent strip with its seam at x = 2 and v mirrored at z = 1, a line across both owners: each owner splits
	// its refined vertices on that line, the seam's one among them.
	const TestCage grid = make_grid(4, 2, 2, true);
	PackedVector2Array values;
	for (const Vector3 &position : grid.positions) {
		values.push_back(Vector2(position.x * 0.25f, Math::abs(position.z - 1.0f) * 0.5f));
	}
	const TestPrimitive left = make_shared_primitive(grid, 0);
	const TestPrimitive right = make_shared_primitive(grid, 1);
	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(grid.positions.size());
	builder->add_face_varying_layer(values, Mesh::ARRAY_TEX_UV);
	Array left_weld;
	left_weld.push_back(left.uv_weld);
	Array right_weld;
	right_weld.push_back(right.uv_weld);
	builder->add_surface(0, left.arrays, Array(), PackedStringArray(), left.cage_vertex, left.faces, left_weld, Ref<Material>());
	builder->add_surface(1, right.arrays, Array(), PackedStringArray(), right.cage_vertex, right.faces, right_weld, Ref<Material>());
	REQUIRE(builder->build(2) == OK);
	const Ref<ArrayMesh> meshes[2] = { builder->get_owner_mesh(0), builder->get_owner_mesh(1) };
	REQUIRE((meshes[0].is_valid() && meshes[1].is_valid()));
	// Each owner's half is 8 x 4 refined quads, 9 refined vertices on the line z = 1.
	for (int owner = 0; owner < 2; owner++) {
		const Dictionary report = Dictionary(builder->get_report()["owners"])[owner];
		CHECK(int(Array(report["tangent_splits"])[0]) == 9);
	}

	// Every copy of a seam vertex, on either side, carries the same position and normal bits.
	const Vector<uint8_t> payloads[2] = { surface_payload(meshes[0], 0), surface_payload(meshes[1], 0) };
	const Array arrays[2] = { meshes[0]->surface_get_arrays(0), meshes[1]->surface_get_arrays(0) };
	const PackedVector3Array positions[2] = { arrays[0][Mesh::ARRAY_VERTEX], arrays[1][Mesh::ARRAY_VERTEX] };
	const PackedVector3Array normals[2] = { arrays[0][Mesh::ARRAY_NORMAL], arrays[1][Mesh::ARRAY_NORMAL] };
	uint32_t count = 0;
	const uint32_t *r_to_u[2];
	const uint32_t *welded[2];
	for (int owner = 0; owner < 2; owner++) {
		r_to_u[owner] = payload_section<uint32_t>(payloads[owner], RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, count);
		welded[owner] = payload_section<uint32_t>(payloads[owner], RenderingServerTypes::MESH_SUBDIVISION_SECTION_U_WELDED, count);
		REQUIRE((r_to_u[owner] && welded[owner]));
	}
	HashMap<uint32_t, LocalVector<int>> left_copies;
	for (int r = 0; r < positions[0].size(); r++) {
		left_copies[welded[0][r_to_u[0][r]]].push_back(r);
	}
	int right_seam = 0;
	int pairs = 0;
	for (int r = 0; r < positions[1].size(); r++) {
		const LocalVector<int> *copies = left_copies.getptr(welded[1][r_to_u[1][r]]);
		if (copies == nullptr) {
			continue;
		}
		right_seam++;
		for (int l : *copies) {
			pairs++;
			CHECK(memcmp(&positions[0][l], &positions[1][r], sizeof(Vector3)) == 0);
			CHECK(memcmp(&normals[0][l], &normals[1][r], sizeof(Vector3)) == 0);
		}
	}
	// The seam's 9 refined vertices, the one on z = 1 split on both sides: 10 render vertices a side, 12 pairs.
	CHECK(right_seam == 10);
	CHECK(pairs == 12);
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
	const uint32_t *tangent_signs = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_SIGNS, count);
	REQUIRE((info && rest && blend_offsets && stencil_offsets && stencil_indices && stencil_weights && quads && normal_offsets && normal_entries && r_to_u && tangent_offsets && tangent_entries && tangent_signs));
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
		const float sign = ((tangent_signs[r / 32] >> (r % 32)) & 1) ? 1.0f : -1.0f;
		Vector3 tangent_sum;
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
			const float angle = pass_corner_angle(p[(corner + 2) % 3], p[corner], p[(corner + 1) % 3]);
			tangent_sum += tangent * angle;
		}
		Vector3 t = tangent_sum - normal * normal.dot(tangent_sum);
		if (t.is_zero_approx()) {
			t = normal.get_any_perpendicular();
		}
		t.normalize();
		result.tangents.push_back(t);
		result.signs.push_back(sign);
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
		// Unskinned: the passes' rest output with no bone matrices applied.
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

TEST_CASE("[SceneTree][SubdivisionBuilder] The fingerprint hashes a probe that reaches the A1 paths, the same on every call") {
	LocalVector<uint8_t> input;
	const Ref<SubdivisionBuilder> probe = SubdivisionBuilder::build_fingerprint_probe(&input);
	REQUIRE(probe.is_valid());
	REQUIRE(!input.is_empty());
	const String fingerprint = SubdivisionBuilder::get_fingerprint();
	CHECK(fingerprint.length() == 64);
	CHECK(fingerprint == SubdivisionBuilder::fingerprint_of(input));
	CHECK(SubdivisionBuilder::get_fingerprint() == fingerprint);

	for (int owner = 0; owner < 2; owner++) {
		const Ref<ArrayMesh> mesh = probe->get_owner_mesh(owner);
		REQUIRE(mesh.is_valid());
		CHECK(mesh->get_surface_count() == (owner == 0 ? 2 : 1));
		// Owner 0's displacement shape comes after its own (S6).
		CHECK(mesh->get_blend_shape_count() == (owner == 0 ? 2 : 1));
		const Vector<uint8_t> owner_payload = surface_payload(mesh, 0);
		uint32_t count = 0;
		const uint32_t *info = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, count);
		REQUIRE(info != nullptr);
		// Ghosts: cage vertices of the other owner its stencils reach.
		CHECK(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_CAGE_COUNT] > info[RenderingServerTypes::MESH_SUBDIVISION_INFO_OWN_CAGE_COUNT]);
		payload_section<uint16_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_SKIN, count);
		CHECK(count > 0);
		CHECK(payload_blend_entry_count(owner_payload) > 0);
		for (int s = 0; s < mesh->get_surface_count(); s++) {
			const RenderingServerTypes::SurfaceData data = RS::get_singleton()->mesh_get_surface(mesh->get_rid(), s);
			CHECK(!data.cluster_data.is_empty());
			// The three UV layers reach their A1 slots' packing: UV, UV2 and CUSTOM0 as two floats, and the colours.
			CHECK((data.format & RSE::ARRAY_FORMAT_TEX_UV) != 0);
			CHECK((data.format & RSE::ARRAY_FORMAT_TEX_UV2) != 0);
			CHECK((data.format & RSE::ARRAY_FORMAT_CUSTOM0) != 0);
			CHECK(((data.format >> RSE::ARRAY_FORMAT_CUSTOM0_SHIFT) & RSE::ARRAY_FORMAT_CUSTOM_MASK) == RSE::ARRAY_CUSTOM_RG_FLOAT);
			CHECK((data.format & RSE::ARRAY_FORMAT_COLOR) != 0);
		}
	}
	// Owner 2, apart and without blend shapes, takes the cluster LOD hierarchy.
	const Ref<ArrayMesh> apart = probe->get_owner_mesh(2);
	REQUIRE(apart.is_valid());
	CHECK(apart->get_blend_shape_count() == 0);
	const Vector<uint8_t> hierarchy = RS::get_singleton()->mesh_get_surface(apart->get_rid(), 0).cluster_data;
	REQUIRE(hierarchy.size() >= int(sizeof(uint32_t)));
	uint32_t version;
	memcpy(&version, hierarchy.ptr(), sizeof(uint32_t));
	CHECK(version == RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION);
	// Its folded face splits render vertices by MikkTSpace's groups.
	const Array apart_splits = Dictionary(Dictionary(probe->get_report()["owners"])[2])["tangent_splits"];
	REQUIRE(apart_splits.size() == 1);
	CHECK(int(apart_splits[0]) > 0);
	// The triangle face refines into owner 1's surface: Catmull-Clark gives it three quads at level 1, twelve at
	// level 2, beside the strip's 8 x 16 and owner 2's 16 x 16.
	CHECK(int(probe->get_report()["refined_quads"]) == 8 * 16 + 3 * 4 + 16 * 16);
	// Owner 0 displaces (S6): its shape and heights, some of them non-zero; owner 1 does not.
	{
		const Ref<ArrayMesh> mesh = probe->get_owner_mesh(0);
		CHECK(mesh->get_blend_shape_name(1) == StringName("ProbeDisplacement"));
		const Vector<uint8_t> owner_payload = surface_payload(mesh, 0);
		uint32_t count = 0;
		const uint32_t *info = payload_section<uint32_t>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, count);
		CHECK(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_DISPLACEMENT_SHAPE] == 1);
		const float *heights = payload_section<float>(owner_payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_DISPLACEMENT_HEIGHTS, count);
		REQUIRE(heights != nullptr);
		CHECK(count == info[RenderingServerTypes::MESH_SUBDIVISION_INFO_OWN_U_COUNT]);
		const Dictionary displacement = Dictionary(Dictionary(probe->get_report()["owners"])[0])["displacement"];
		CHECK(int(displacement["displaced_vertices"]) > 0);
		// The first surface does not displace: its own vertices are unreached, NaN.
		const PackedVector2Array sampled = displacement["displacement_uvs"];
		int unreached = 0;
		for (const Vector2 &uv : sampled) {
			unreached += Math::is_nan(uv.x) ? 1 : 0;
		}
		CHECK(unreached > 0);
		CHECK(unreached < sampled.size());
		const Vector<uint8_t> other = surface_payload(probe->get_owner_mesh(1), 0);
		const uint32_t *other_info = payload_section<uint32_t>(other, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, count);
		CHECK(other_info[RenderingServerTypes::MESH_SUBDIVISION_INFO_DISPLACEMENT_SHAPE] == RenderingServerTypes::MESH_SUBDIVISION_NO_DISPLACEMENT);
		CHECK(payload_section<float>(other, RenderingServerTypes::MESH_SUBDIVISION_SECTION_DISPLACEMENT_HEIGHTS, count) == nullptr);
	}
}

// D122 S6: an 8-bit RGB image from rows given top first, as Godot stores them.
static Ref<Image> rgb_image(int p_width, int p_height, const LocalVector<uint8_t> &p_rgb) {
	Vector<uint8_t> bytes;
	for (uint8_t value : p_rgb) {
		bytes.push_back(value);
	}
	return Image::create_from_data(p_width, p_height, false, Image::FORMAT_RGB8, bytes);
}

static float srgb_decoded(float p_encoded) {
	return p_encoded < 0.04045f ? p_encoded * (1.0f / 12.92f) : powf((p_encoded + 0.055f) * (1.0f / 1.055f), 2.4f);
}

static float decoded_luma(float p_r, float p_g, float p_b) {
	return srgb_decoded(p_r) * 0.2126390059f + srgb_decoded(p_g) * 0.7151686788f + srgb_decoded(p_b) * 0.0721923154f;
}

TEST_CASE("[SceneTree][SubdivisionBuilder] Displacement heights sample the maps as Cycles does") {
	// 2 x 2, top row first: (10, 20, 30) (200, 0, 0) / (0, 255, 0) (40, 40, 40).
	const Ref<Image> map = rgb_image(2, 2, { 10, 20, 30, 200, 0, 0, 0, 255, 0, 40, 40, 40 });
	const float f = 1.0f / 255.0f;
	// At a texel's centre, that texel alone; rows count from the bottom, as Blender stores an image.
	CHECK(SubdivisionBuilder::displacement_height(map, Ref<Image>(), Vector2(0.25, 0.75), 0.0f, 1.0f) == decoded_luma(10 * f, 20 * f, 30 * f));
	CHECK(SubdivisionBuilder::displacement_height(map, Ref<Image>(), Vector2(0.75, 0.25), 0.0f, 1.0f) == decoded_luma(40 * f, 40 * f, 40 * f));
	CHECK(SubdivisionBuilder::displacement_height(map, Ref<Image>(), Vector2(0.25, 0.25), 0.0f, 1.0f) == decoded_luma(0, 255 * f, 0));
	// At u = 0 the repeat mixes the first column with the last, half each, in the stored values before the decode.
	const float mixed = SubdivisionBuilder::displacement_height(map, Ref<Image>(), Vector2(0.0, 0.75), 0.0f, 1.0f);
	CHECK(mixed == doctest::Approx(decoded_luma(0.5f * 10 * f + 0.5f * 200 * f, 0.5f * 20 * f, 0.5f * 30 * f)).epsilon(1e-6));
	// The chain: (luma * (max - min) + min) * the mask's luma; a null mask is 1.
	const Ref<Image> mask = rgb_image(2, 2, { 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 });
	const float base = SubdivisionBuilder::displacement_height(map, Ref<Image>(), Vector2(0.25, 0.75), 0.001f, 0.003f);
	CHECK(base == decoded_luma(10 * f, 20 * f, 30 * f) * (0.003f - 0.001f) + 0.001f);
	CHECK(SubdivisionBuilder::displacement_height(map, mask, Vector2(0.25, 0.75), 0.001f, 0.003f) == base * decoded_luma(128 * f, 128 * f, 128 * f));
	// A grey image's value stands for every channel.
	Vector<uint8_t> grey_bytes = { 77, 77, 77, 77 };
	const Ref<Image> grey = Image::create_from_data(2, 2, false, Image::FORMAT_L8, grey_bytes);
	CHECK(SubdivisionBuilder::displacement_height(grey, Ref<Image>(), Vector2(0.3, 0.6), 0.0f, 1.0f) == doctest::Approx(decoded_luma(77 * f, 77 * f, 77 * f)).epsilon(1e-6));
	// A non-finite UV gives NaN; an opaque RGBA map samples as its RGB, one with any alpha below 255 is refused.
	CHECK(Math::is_nan(SubdivisionBuilder::displacement_height(map, Ref<Image>(), Vector2(NAN, 0.5), 0.0f, 1.0f)));
	Vector<uint8_t> rgba_bytes = { 10, 20, 30, 255, 200, 0, 0, 255, 0, 255, 0, 255, 40, 40, 40, 255 };
	const Ref<Image> opaque = Image::create_from_data(2, 2, false, Image::FORMAT_RGBA8, rgba_bytes);
	CHECK(SubdivisionBuilder::displacement_height(opaque, Ref<Image>(), Vector2(0.25, 0.75), 0.0f, 1.0f) == decoded_luma(10 * f, 20 * f, 30 * f));
	rgba_bytes.write[15] = 254;
	const Ref<Image> translucent = Image::create_from_data(2, 2, false, Image::FORMAT_RGBA8, rgba_bytes);
	ERR_PRINT_OFF;
	CHECK(Math::is_nan(SubdivisionBuilder::displacement_height(translucent, Ref<Image>(), Vector2(0.25, 0.75), 0.0f, 1.0f)));
	ERR_PRINT_ON;
}

// The strip with owner 0's two faces displacing: each face's own UV values (u along z, v along x), face 0's
// v from 0 to 0.25 and face 1's from 0.5 to 0.75, so the edge they share at x = 1 is a UV seam and the seam
// with owner 1 at x = 2 is face 1's v = 0.75. The mask's two top rows (v from 0.5 up) are 0 unless asked not.
struct DisplacedStrip {
	Ref<SubdivisionBuilder> builder;
	Ref<Image> height_map;
	Ref<Image> mask;
};

static DisplacedStrip make_displaced_strip(const Array &p_source_faces, bool p_mask_seam) {
	const TestCage strip = make_strip();
	PackedVector2Array ignored;
	const TestPrimitive left = make_primitive(strip, 0, ignored);
	const TestPrimitive right = make_primitive(strip, 1, ignored);
	const Vector2 square[4] = { Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1) };
	PackedVector2Array uv_values;
	for (int face = 0; face < 4; face++) {
		for (int c = 0; c < 4; c++) {
			uv_values.push_back(Vector2(0.1f + 0.8f * float(square[c].x), float(face % 2) * 0.5f + 0.25f * float(square[c].y)));
		}
	}
	DisplacedStrip result;
	result.height_map = rgb_image(2, 4, { 250, 250, 250, 30, 30, 30, 90, 90, 90, 200, 200, 200, 160, 160, 160, 60, 60, 60, 20, 20, 20, 120, 120, 120 });
	LocalVector<uint8_t> mask_rgb;
	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 2; x++) {
			const uint8_t value = (y < 2 && p_mask_seam) ? 0 : uint8_t(255 - 40 * y - 30 * x);
			for (int a = 0; a < 3; a++) {
				mask_rgb.push_back(value);
			}
		}
	}
	result.mask = rgb_image(2, 4, mask_rgb);
	result.builder.instantiate();
	result.builder->set_cage_vertex_count(strip.positions.size());
	result.builder->set_owner_bone_names(0, test_bone_names());
	result.builder->set_owner_bone_names(1, test_bone_names());
	result.builder->add_face_varying_layer(uv_values, Mesh::ARRAY_TEX_UV);
	Array left_weld;
	left_weld.push_back(left.uv_weld);
	Array right_weld;
	right_weld.push_back(right.uv_weld);
	result.builder->add_surface(0, left.arrays, Array(), PackedStringArray(), left.cage_vertex, left.faces, left_weld, Ref<Material>());
	result.builder->add_surface(1, right.arrays, Array(), PackedStringArray(), right.cage_vertex, right.faces, right_weld, Ref<Material>());
	CHECK(result.builder->set_owner_displacement(0, PackedInt32Array{ 0 }, p_source_faces, Mesh::ARRAY_TEX_UV, result.height_map, result.mask, 0.0f, 0.0008f, "Displace") == OK);
	return result;
}

TEST_CASE("[SceneTree][SubdivisionBuilder] A displacement takes each vertex's height from its first face in the source's order") {
	float shared_heights[2] = {};
	for (int order = 0; order < 2; order++) {
		// Add order puts face 0 first; the source order [1, 0] face 1.
		Array source_faces;
		if (order == 1) {
			source_faces.push_back(PackedInt32Array{ 1, 0 });
		}
		DisplacedStrip strip = make_displaced_strip(source_faces, true);
		REQUIRE(strip.builder->build(2) == OK);
		const Ref<ArrayMesh> mesh = strip.builder->get_owner_mesh(0);
		REQUIRE(mesh->get_blend_shape_count() == 1);
		CHECK(mesh->get_blend_shape_name(0) == StringName("Displace"));
		const Vector<uint8_t> payload = surface_payload(mesh, 0);
		uint32_t count = 0;
		const uint32_t *info = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, count);
		CHECK(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_DISPLACEMENT_SHAPE] == 0);
		CHECK(info[RenderingServerTypes::MESH_SUBDIVISION_INFO_BLEND_SHAPE_COUNT] == 1);
		const float *heights = payload_section<float>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_DISPLACEMENT_HEIGHTS, count);
		REQUIRE(heights != nullptr);
		const uint32_t *r_to_u = payload_section<uint32_t>(payload, RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, count);
		const Array arrays = mesh->surface_get_arrays(0);
		const PackedVector3Array positions = arrays[Mesh::ARRAY_VERTEX];
		const PackedVector2Array uvs = arrays[Mesh::ARRAY_TEX_UV];
		int shared = 0;
		int seam = 0;
		int interior = 0;
		int nonzero = 0;
		for (int r = 0; r < positions.size(); r++) {
			const float height = heights[r_to_u[r]];
			const Vector2 blender_uv(uvs[r].x, 1.0f - float(uvs[r].y)); // Stored as 1 - v.
			const bool face_1 = blender_uv.y >= 0.5f - 1e-6f;
			const float own = SubdivisionBuilder::displacement_height(strip.height_map, strip.mask, blender_uv, 0.0f, 0.0008f);
			nonzero += height != 0.0f ? 1 : 0;
			if (Math::is_equal_approx(positions[r].x, real_t(2.0), real_t(1e-5))) {
				seam++;
				CHECK(height == 0.0f); // The mask's top rows: the seam stays where owner 1 puts it.
			} else if (Math::is_equal_approx(positions[r].x, real_t(1.0), real_t(1e-5))) {
				// Two render vertices of one refined vertex, one per face: both carry the first face's height.
				shared++;
				if (face_1 == (order == 1)) {
					CHECK(height == own);
				}
				shared_heights[order] += height;
			} else {
				interior++;
				CHECK(height == own);
			}
		}
		CHECK(seam == 5);
		CHECK(shared == 10);
		CHECK(interior > 0);
		CHECK(nonzero > 0);
		const Dictionary report = Dictionary(Dictionary(strip.builder->get_report()["owners"])[0])["displacement"];
		CHECK(int(report["displaced_vertices"]) > 0);
		CHECK(int(report["seam_vertices_reached"]) == 5);
		// The value each refined vertex was sampled at: its own face's inside a face, the first face's on the edge the
		// order decides, as v up.
		const PackedVector2Array sampled = report["displacement_uvs"];
		REQUIRE(uint32_t(sampled.size()) == info[RenderingServerTypes::MESH_SUBDIVISION_INFO_OWN_U_COUNT]);
		for (int r = 0; r < positions.size(); r++) {
			const Vector2 blender_uv(uvs[r].x, 1.0f - float(uvs[r].y));
			const bool face_1 = blender_uv.y >= 0.5f - 1e-6f;
			if (!Math::is_equal_approx(positions[r].x, real_t(1.0), real_t(1e-5)) || face_1 == (order == 1)) {
				CHECK(sampled[r_to_u[r]] == blender_uv);
			} else {
				CHECK(sampled[r_to_u[r]] != blender_uv); // The other face's value on the edge: not the one sampled.
			}
		}
		CHECK(float(report["max_height"]) <= 0.0008f);
		// Owner 1 carries none.
		CHECK(strip.builder->get_owner_mesh(1)->get_blend_shape_count() == 0);
	}
	// The shared edge's faces read different values, so the order decides its heights.
	CHECK(shared_heights[0] != shared_heights[1]);
}

TEST_CASE("[SceneTree][SubdivisionBuilder] A displacement refuses to move the seam, and grows the bounds by its height") {
	{
		ERR_PRINT_OFF;
		DisplacedStrip strip = make_displaced_strip(Array(), false);
		CHECK(strip.builder->build(2) != OK);
		ERR_PRINT_ON;
	}
	DisplacedStrip strip = make_displaced_strip(Array(), true);
	REQUIRE(strip.builder->build(2) == OK);
	const RenderingServerTypes::SurfaceData displaced = RS::get_singleton()->mesh_get_surface(strip.builder->get_owner_mesh(0)->get_rid(), 0);
	const float bound = float(Dictionary(Dictionary(Dictionary(strip.builder->get_report()["owners"])[0])["displacement"])["max_height"]);
	REQUIRE(bound > 0.0f);
	strip.builder->clear_owner_displacement(0);
	REQUIRE(strip.builder->build(2) == OK);
	const RenderingServerTypes::SurfaceData plain = RS::get_singleton()->mesh_get_surface(strip.builder->get_owner_mesh(0)->get_rid(), 0);
	CHECK(strip.builder->get_owner_mesh(0)->get_blend_shape_count() == 0);
	CHECK(displaced.aabb.is_equal_approx(plain.aabb.grow(bound)));
	REQUIRE(displaced.bone_aabbs.size() == plain.bone_aabbs.size());
	for (int b = 0; b < plain.bone_aabbs.size(); b++) {
		CHECK(displaced.bone_aabbs[b].is_equal_approx(plain.bone_aabbs[b].grow(bound)));
	}
	// A surface of another owner and a source face named twice are refused at build, a missing map at once.
	ERR_PRINT_OFF;
	DisplacedStrip other = make_displaced_strip(Array(), true);
	other.builder->set_owner_displacement(0, PackedInt32Array{ 1 }, Array(), Mesh::ARRAY_TEX_UV, other.height_map, other.mask, 0.0f, 0.0008f, "Displace");
	CHECK(other.builder->build(2) != OK);
	Array repeated;
	repeated.push_back(PackedInt32Array{ 7, 7 });
	DisplacedStrip twice = make_displaced_strip(repeated, true);
	CHECK(twice.builder->build(2) != OK);
	CHECK(other.builder->set_owner_displacement(0, PackedInt32Array{ 0 }, Array(), Mesh::ARRAY_TEX_UV2, Ref<Image>(), Ref<Image>(), 0.0f, 1.0f, "D") != OK);
	ERR_PRINT_ON;
}

TEST_CASE("[SceneTree][SubdivisionBuilder] A displacement shape may not take a name another owner's blend shape uses") {
	// The strip with a shape named Displace on owner 1 only (without deltas, so owner 0 need not hold it); owner 0's
	// displacement shape takes the same name.
	const TestCage strip = make_strip();
	PackedVector2Array uv_values;
	const TestPrimitive left = make_primitive(strip, 0, uv_values);
	const TestPrimitive right = make_primitive(strip, 1, uv_values);
	LocalVector<HashMap<int, Vector3>> shapes;
	shapes.resize(1);
	PackedStringArray names;
	names.push_back("Displace");
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
	Vector<uint8_t> white = { 255, 255, 255, 255 };
	const Ref<Image> map = Image::create_from_data(2, 2, false, Image::FORMAT_L8, white);
	REQUIRE(builder->set_owner_displacement(0, PackedInt32Array{ 0 }, Array(), Mesh::ARRAY_TEX_UV, map, Ref<Image>(), 0.0f, 0.0f, "Displace") == OK);
	ERR_PRINT_OFF;
	CHECK(builder->build(2) != OK);
	ERR_PRINT_ON;
	REQUIRE(builder->set_owner_displacement(0, PackedInt32Array{ 0 }, Array(), Mesh::ARRAY_TEX_UV, map, Ref<Image>(), 0.0f, 0.0f, "Displace0") == OK);
	CHECK(builder->build(2) == OK);
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
