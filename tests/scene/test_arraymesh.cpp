/**************************************************************************/
/*  test_arraymesh.cpp                                                    */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_arraymesh)

#ifndef _3D_DISABLED

#include "core/config/project_settings.h"
#include "core/templates/hash_set.h"
#include "scene/resources/3d/importer_mesh.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/mesh.h"
#include "servers/rendering/rendering_server.h"

namespace TestArrayMesh {

TEST_CASE("[SceneTree][ArrayMesh] Adding and modifying blendshapes.") {
	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	StringName name_a{ "ShapeA" };
	StringName name_b{ "ShapeB" };

	SUBCASE("Adding a blend shape to the mesh before a surface is added.") {
		mesh->add_blend_shape(name_a);
		mesh->add_blend_shape(name_b);

		CHECK(mesh->get_blend_shape_name(0) == name_a);
		CHECK(mesh->get_blend_shape_name(1) == name_b);
	}

	SUBCASE("Add same blend shape multiple times appends name with number.") {
		mesh->add_blend_shape(name_a);
		mesh->add_blend_shape(name_a);
		mesh->add_blend_shape(name_a);

		CHECK(mesh->get_blend_shape_name(0) == "ShapeA");
		bool all_different = (static_cast<String>(mesh->get_blend_shape_name(0)) != static_cast<String>(mesh->get_blend_shape_name(1))) &&
				(static_cast<String>(mesh->get_blend_shape_name(1)) != static_cast<String>(mesh->get_blend_shape_name(2))) &&
				(static_cast<String>(mesh->get_blend_shape_name(0)) != static_cast<String>(mesh->get_blend_shape_name(2)));
		bool all_have_name = static_cast<String>(mesh->get_blend_shape_name(1)).contains("ShapeA") &&
				static_cast<String>(mesh->get_blend_shape_name(2)).contains("ShapeA");
		CHECK((all_different && all_have_name));
	}

	SUBCASE("ArrayMesh keeps correct count of number of blend shapes") {
		mesh->add_blend_shape(name_a);
		mesh->add_blend_shape(name_a);
		mesh->add_blend_shape(name_b);
		mesh->add_blend_shape(name_b);
		mesh->add_blend_shape(name_b);

		REQUIRE(mesh->get_blend_shape_count() == 5);
	}

	SUBCASE("Adding blend shape after surface is added causes error") {
		Ref<CylinderMesh> cylinder;
		cylinder.instantiate();
		Array cylinder_array;
		cylinder_array.resize(Mesh::ARRAY_MAX);
		cylinder->create_mesh_array(cylinder_array, 3.f, 3.f, 5.f);
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array);

		ERR_PRINT_OFF
		mesh->add_blend_shape(name_a);
		ERR_PRINT_ON
		CHECK(mesh->get_blend_shape_count() == 0);
	}

	SUBCASE("Adding blend shapes once all surfaces have been removed is allowed") {
		Ref<CylinderMesh> cylinder;
		cylinder.instantiate();
		Array cylinder_array;
		cylinder_array.resize(Mesh::ARRAY_MAX);
		cylinder->create_mesh_array(cylinder_array, 3.f, 3.f, 5.f);
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array);
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array);

		mesh->surface_remove(0);
		ERR_PRINT_OFF
		mesh->add_blend_shape(name_a);
		ERR_PRINT_ON
		CHECK(mesh->get_blend_shape_count() == 0);

		mesh->surface_remove(0);
		mesh->add_blend_shape(name_a);
		CHECK(mesh->get_blend_shape_count() == 1);
	}

	SUBCASE("Change blend shape name after adding.") {
		mesh->add_blend_shape(name_a);
		mesh->set_blend_shape_name(0, name_b);

		CHECK(mesh->get_blend_shape_name(0) == name_b);
	}

	SUBCASE("Change blend shape name to the name of one already there, should append number to end") {
		mesh->add_blend_shape(name_a);
		mesh->add_blend_shape(name_b);
		mesh->set_blend_shape_name(0, name_b);

		String name_string = mesh->get_blend_shape_name(0);
		CHECK(name_string.contains("ShapeB"));
		CHECK(name_string.length() > static_cast<String>(name_b).size());
	}

	SUBCASE("Clear all blend shapes before surface has been added.") {
		mesh->add_blend_shape(name_a);
		mesh->add_blend_shape(name_b);
		CHECK(mesh->get_blend_shape_count() == 2);

		mesh->clear_blend_shapes();
		CHECK(mesh->get_blend_shape_count() == 0);
	}

	SUBCASE("Clearing all blend shapes once all surfaces have been removed is allowed") {
		mesh->add_blend_shape(name_a);
		mesh->add_blend_shape(name_b);
		Ref<CylinderMesh> cylinder;
		cylinder.instantiate();
		Array cylinder_array;
		cylinder_array.resize(Mesh::ARRAY_MAX);
		cylinder->create_mesh_array(cylinder_array, 3.f, 3.f, 5.f);
		Array blend_shape;
		blend_shape.resize(Mesh::ARRAY_MAX);
		blend_shape[Mesh::ARRAY_VERTEX] = cylinder_array[Mesh::ARRAY_VERTEX];
		blend_shape[Mesh::ARRAY_NORMAL] = cylinder_array[Mesh::ARRAY_NORMAL];
		blend_shape[Mesh::ARRAY_TANGENT] = cylinder_array[Mesh::ARRAY_TANGENT];
		Array blend_shapes = { blend_shape, blend_shape };
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array, blend_shapes);
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array, blend_shapes);

		mesh->surface_remove(0);
		ERR_PRINT_OFF
		mesh->clear_blend_shapes();
		ERR_PRINT_ON
		CHECK(mesh->get_blend_shape_count() == 2);

		mesh->surface_remove(0);
		mesh->clear_blend_shapes();
		CHECK(mesh->get_blend_shape_count() == 0);
	}

	SUBCASE("Can't add surface with incorrect number of blend shapes.") {
		mesh->add_blend_shape(name_a);
		mesh->add_blend_shape(name_b);
		Array cylinder_array;
		ERR_PRINT_OFF
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array);
		ERR_PRINT_ON
		CHECK(mesh->get_surface_count() == 0);
	}

	SUBCASE("Can't clear blend shapes after surface had been added.") {
		mesh->add_blend_shape(name_a);
		mesh->add_blend_shape(name_b);
		Ref<CylinderMesh> cylinder;
		cylinder.instantiate();
		Array cylinder_array;
		cylinder_array.resize(Mesh::ARRAY_MAX);
		cylinder->create_mesh_array(cylinder_array, 3.f, 3.f, 5.f);
		Array blend_shape;
		blend_shape.resize(Mesh::ARRAY_MAX);
		blend_shape[Mesh::ARRAY_VERTEX] = cylinder_array[Mesh::ARRAY_VERTEX];
		blend_shape[Mesh::ARRAY_NORMAL] = cylinder_array[Mesh::ARRAY_NORMAL];
		blend_shape[Mesh::ARRAY_TANGENT] = cylinder_array[Mesh::ARRAY_TANGENT];
		Array blend_shapes = { blend_shape, blend_shape };
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array, blend_shapes);

		ERR_PRINT_OFF
		mesh->clear_blend_shapes();
		ERR_PRINT_ON
		CHECK(mesh->get_blend_shape_count() == 2);
	}

	SUBCASE("Set the blend shape mode of ArrayMesh and underlying mesh RID.") {
		mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_RELATIVE);
		CHECK(mesh->get_blend_shape_mode() == Mesh::BLEND_SHAPE_MODE_RELATIVE);
	}
}

TEST_CASE("[SceneTree][ArrayMesh] Surface metadata tests.") {
	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	Ref<CylinderMesh> cylinder;
	cylinder.instantiate();
	Array cylinder_array;
	cylinder_array.resize(Mesh::ARRAY_MAX);
	cylinder->create_mesh_array(cylinder_array, 3.f, 3.f, 5.f);
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array);

	Ref<BoxMesh> box;
	box.instantiate();
	Array box_array;
	box_array.resize(Mesh::ARRAY_MAX);
	box->create_mesh_array(box_array, Vector3(2.f, 1.2f, 1.6f));
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, box_array);

	SUBCASE("Add 2 surfaces and count the number of surfaces in the mesh.") {
		REQUIRE(mesh->get_surface_count() == 2);
	}

	SUBCASE("Get the surface array from mesh.") {
		REQUIRE(mesh->surface_get_arrays(0)[0] == cylinder_array[0]);
		REQUIRE(mesh->surface_get_arrays(1)[0] == box_array[0]);
	}

	SUBCASE("Get the array length of a particular surface.") {
		CHECK(mesh->surface_get_array_len(0) == static_cast<Vector<Vector3>>(cylinder_array[RSE::ARRAY_VERTEX]).size());
		CHECK(mesh->surface_get_array_len(1) == static_cast<Vector<Vector3>>(box_array[RSE::ARRAY_VERTEX]).size());
	}

	SUBCASE("Get the index array length of a particular surface.") {
		CHECK(mesh->surface_get_array_index_len(0) == static_cast<Vector<Vector3>>(cylinder_array[RSE::ARRAY_INDEX]).size());
		CHECK(mesh->surface_get_array_index_len(1) == static_cast<Vector<Vector3>>(box_array[RSE::ARRAY_INDEX]).size());
	}

	SUBCASE("Get correct primitive type") {
		CHECK(mesh->surface_get_primitive_type(0) == Mesh::PRIMITIVE_TRIANGLES);
		CHECK(mesh->surface_get_primitive_type(1) == Mesh::PRIMITIVE_TRIANGLES);
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLE_STRIP, box_array);
		CHECK(mesh->surface_get_primitive_type(2) == Mesh::PRIMITIVE_TRIANGLE_STRIP);
	}

	SUBCASE("Returns correct format for the mesh") {
		int format = RSE::ARRAY_FORMAT_BLEND_SHAPE_MASK | RSE::ARRAY_FORMAT_TEX_UV | RSE::ARRAY_FORMAT_INDEX;
		CHECK((mesh->surface_get_format(0) & format) != 0);
		CHECK((mesh->surface_get_format(1) & format) != 0);
	}

	SUBCASE("Set a surface name and retrieve it by name.") {
		mesh->surface_set_name(0, "surf1");
		CHECK(mesh->surface_find_by_name("surf1") == 0);
		CHECK(mesh->surface_get_name(0) == "surf1");
	}

	SUBCASE("Set material to two different surfaces.") {
		Ref<Material> mat;
		mat.instantiate();
		mesh->surface_set_material(0, mat);
		CHECK(mesh->surface_get_material(0) == mat);
		mesh->surface_set_material(1, mat);
		CHECK(mesh->surface_get_material(1) == mat);
	}

	SUBCASE("Set same material multiple times doesn't change material of surface.") {
		Ref<Material> mat;
		mat.instantiate();
		mesh->surface_set_material(0, mat);
		mesh->surface_set_material(0, mat);
		mesh->surface_set_material(0, mat);
		CHECK(mesh->surface_get_material(0) == mat);
	}

	SUBCASE("Set material of surface then change to different material.") {
		Ref<Material> mat1;
		mat1.instantiate();
		Ref<Material> mat2;
		mat2.instantiate();
		mesh->surface_set_material(1, mat1);
		CHECK(mesh->surface_get_material(1) == mat1);
		mesh->surface_set_material(1, mat2);
		CHECK(mesh->surface_get_material(1) == mat2);
	}

	SUBCASE("Get the LOD of the mesh.") {
		Dictionary lod;
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array, TypedArray<Array>(), lod);
		CHECK(mesh->surface_get_lods(2) == lod);
	}

	SUBCASE("Get the blend shape arrays from the mesh.") {
		TypedArray<Array> blend;
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array, blend);
		CHECK(mesh->surface_get_blend_shape_arrays(2) == blend);
	}
}

TEST_CASE("[SceneTree][ArrayMesh] Get/Set mesh metadata and actions") {
	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	Ref<CylinderMesh> cylinder;
	cylinder.instantiate();
	Array cylinder_array;
	cylinder_array.resize(Mesh::ARRAY_MAX);
	constexpr float cylinder_radius = 3.f;
	constexpr float cylinder_height = 5.f;
	cylinder->create_mesh_array(cylinder_array, cylinder_radius, cylinder_radius, cylinder_height);
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array);

	Ref<BoxMesh> box;
	box.instantiate();
	Array box_array;
	box_array.resize(Mesh::ARRAY_MAX);
	const Vector3 box_size = Vector3(2.f, 1.2f, 1.6f);
	box->create_mesh_array(box_array, box_size);
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, box_array);

	SUBCASE("Set the shadow mesh.") {
		Ref<ArrayMesh> shadow;
		shadow.instantiate();
		mesh->set_shadow_mesh(shadow);
		CHECK(mesh->get_shadow_mesh() == shadow);
	}

	SUBCASE("Set the shadow mesh multiple times.") {
		Ref<ArrayMesh> shadow;
		shadow.instantiate();
		mesh->set_shadow_mesh(shadow);
		mesh->set_shadow_mesh(shadow);
		mesh->set_shadow_mesh(shadow);
		mesh->set_shadow_mesh(shadow);
		CHECK(mesh->get_shadow_mesh() == shadow);
	}

	SUBCASE("Set the same shadow mesh on multiple meshes.") {
		Ref<ArrayMesh> shadow;
		shadow.instantiate();
		Ref<ArrayMesh> mesh2;
		mesh2.instantiate();
		mesh->set_shadow_mesh(shadow);
		mesh2->set_shadow_mesh(shadow);

		CHECK(mesh->get_shadow_mesh() == shadow);
		CHECK(mesh2->get_shadow_mesh() == shadow);
	}

	SUBCASE("Set the shadow mesh and then change it.") {
		Ref<ArrayMesh> shadow;
		shadow.instantiate();
		mesh->set_shadow_mesh(shadow);
		CHECK(mesh->get_shadow_mesh() == shadow);
		Ref<ArrayMesh> shadow2;
		shadow2.instantiate();
		mesh->set_shadow_mesh(shadow2);
		CHECK(mesh->get_shadow_mesh() == shadow2);
	}

	SUBCASE("Set custom AABB.") {
		AABB bound;
		mesh->set_custom_aabb(bound);
		CHECK(mesh->get_custom_aabb() == bound);
	}

	SUBCASE("Set custom AABB multiple times.") {
		AABB bound;
		mesh->set_custom_aabb(bound);
		mesh->set_custom_aabb(bound);
		mesh->set_custom_aabb(bound);
		mesh->set_custom_aabb(bound);
		CHECK(mesh->get_custom_aabb() == bound);
	}

	SUBCASE("Set custom AABB then change to another AABB.") {
		AABB bound;
		AABB bound2;
		mesh->set_custom_aabb(bound);
		CHECK(mesh->get_custom_aabb() == bound);
		mesh->set_custom_aabb(bound2);
		CHECK(mesh->get_custom_aabb() == bound2);
	}

	SUBCASE("Clear all surfaces should leave zero count.") {
		mesh->clear_surfaces();
		CHECK(mesh->get_surface_count() == 0);
	}

	SUBCASE("Able to get correct mesh RID.") {
		RID rid = mesh->get_rid();
		CHECK(RS::get_singleton()->mesh_get_surface_count(rid) == 2);
	}

	SUBCASE("Create surface from raw SurfaceData data.") {
		RID mesh_rid = mesh->get_rid();
		RenderingServerTypes::SurfaceData surface_data = RS::get_singleton()->mesh_get_surface(mesh_rid, 0);
		Ref<ArrayMesh> mesh2;
		mesh2.instantiate();
		mesh2->add_surface(surface_data.format, Mesh::PRIMITIVE_TRIANGLES, surface_data.vertex_data, surface_data.attribute_data,
				surface_data.skin_data, surface_data.vertex_count, surface_data.index_data, surface_data.index_count, surface_data.aabb);
		CHECK(mesh2->get_surface_count() == 1);
		CHECK(mesh2->surface_get_primitive_type(0) == Mesh::PRIMITIVE_TRIANGLES);
		CHECK((mesh2->surface_get_format(0) & surface_data.format) != 0);
		CHECK(mesh2->get_aabb().is_equal_approx(surface_data.aabb));
	}

	SUBCASE("Removing a surface decreases surface count.") {
		REQUIRE(mesh->get_surface_count() == 2);
		mesh->surface_remove(0);
		CHECK(mesh->get_surface_count() == 1);
		mesh->surface_remove(0);
		CHECK(mesh->get_surface_count() == 0);
	}

	SUBCASE("Remove the first surface and check the mesh's AABB.") {
		REQUIRE(mesh->get_surface_count() >= 1);
		mesh->surface_remove(0);
		const AABB box_aabb = AABB(-box_size / 2, box_size);
		CHECK(mesh->get_aabb().is_equal_approx(box_aabb));
	}

	SUBCASE("Remove the last surface and check the mesh's AABB.") {
		REQUIRE(mesh->get_surface_count() >= 1);
		mesh->surface_remove(mesh->get_surface_count() - 1);
		const AABB cylinder_aabb = AABB(Vector3(-cylinder_radius, -cylinder_height / 2, -cylinder_radius),
				Vector3(2 * cylinder_radius, cylinder_height, 2 * cylinder_radius));
		CHECK(mesh->get_aabb().is_equal_approx(cylinder_aabb));
	}

	SUBCASE("Remove all surfaces and check the mesh's AABB.") {
		while (mesh->get_surface_count()) {
			mesh->surface_remove(0);
		}
		CHECK(mesh->get_aabb() == AABB());
	}

	SUBCASE("Removing a non-existent surface causes error.") {
		ERR_PRINT_OFF
		mesh->surface_remove(42);
		ERR_PRINT_ON
		CHECK(mesh->get_surface_count() == 2);
	}
}

TEST_CASE("[SceneTree][ArrayMesh] Triangle clusters partition the index buffer (D80).") {
	// A 40 by 40 grid: 1681 vertices, 3200 triangles, so the builder must split it into many
	// clusters and no cluster can hold every vertex.
	const int n = 40;
	Vector<Vector3> positions;
	for (int y = 0; y <= n; y++) {
		for (int x = 0; x <= n; x++) {
			positions.push_back(Vector3(x, Math::sin(x * 0.3) * Math::cos(y * 0.2), y));
		}
	}
	Vector<int> indices;
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++) {
			const int i = y * (n + 1) + x;
			indices.push_back(i);
			indices.push_back(i + 1);
			indices.push_back(i + n + 1);
			indices.push_back(i + 1);
			indices.push_back(i + n + 2);
			indices.push_back(i + n + 1);
		}
	}
	const Vector<int> original = indices;

	Vector<uint8_t> table_bytes = RenderingServer::mesh_surface_build_clusters(positions, indices);
	if (RenderingServer::build_clusters_func == nullptr) {
		CHECK(table_bytes.is_empty());
		CHECK(indices == original);
		return;
	}
	REQUIRE(table_bytes.size() >= int(sizeof(uint32_t) * 2));
	REQUIRE(table_bytes.size() % sizeof(uint32_t) == 0);
	REQUIRE(indices.size() == original.size());
	const uint32_t *table = reinterpret_cast<const uint32_t *>(table_bytes.ptr());
	CHECK(table[0] == RenderingServerTypes::MESH_CLUSTER_DATA_VERSION);
	const uint32_t cluster_count = table_bytes.size() / sizeof(uint32_t) - 1;
	CHECK(cluster_count >= uint32_t(original.size() / 3 / RenderingServerTypes::MESH_CLUSTER_MAX_TRIANGLES));

	// Every cluster within its limits, with the distinct vertex count it declares, and the
	// clusters partitioning the triangles in order.
	uint32_t first_triangle = 0;
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t triangles = table[1 + c] & 0xFFFFu;
		const uint32_t vertices = table[1 + c] >> 16;
		CHECK(triangles > 0);
		CHECK(triangles <= RenderingServerTypes::MESH_CLUSTER_MAX_TRIANGLES);
		CHECK(vertices <= RenderingServerTypes::MESH_CLUSTER_MAX_VERTICES);
		REQUIRE(int((first_triangle + triangles) * 3) <= indices.size());
		HashSet<int> distinct;
		for (uint32_t t = first_triangle * 3; t < (first_triangle + triangles) * 3; t++) {
			distinct.insert(indices[t]);
		}
		CHECK(distinct.size() == vertices);
		first_triangle += triangles;
	}
	CHECK(int(first_triangle * 3) == indices.size());

	// The same triangles, as a set: each reordered triangle is one of the originals, with its
	// winding kept (the vertices rotated at most).
	HashSet<uint64_t> original_triangles;
	auto triangle_key = [](int a, int b, int c) -> uint64_t {
		// Rotate so the smallest index leads; the winding is then the order of the other two.
		while (!(a <= b && a <= c)) {
			const int t = a;
			a = b;
			b = c;
			c = t;
		}
		return (uint64_t(a) << 42) | (uint64_t(b) << 21) | uint64_t(c);
	};
	for (int t = 0; t < original.size(); t += 3) {
		original_triangles.insert(triangle_key(original[t], original[t + 1], original[t + 2]));
	}
	int matched = 0;
	for (int t = 0; t < indices.size(); t += 3) {
		matched += original_triangles.has(triangle_key(indices[t], indices[t + 1], indices[t + 2])) ? 1 : 0;
	}
	CHECK(matched == original.size() / 3);
	CHECK(original_triangles.size() == uint32_t(original.size() / 3));

	// A surface created from these arrays carries the table and the clustered index order with the
	// setting on, and neither with it off; the caller's arrays are untouched either way.
	Array arrays;
	arrays.resize(RSE::ARRAY_MAX);
	arrays[RSE::ARRAY_VERTEX] = positions;
	arrays[RSE::ARRAY_INDEX] = original;
	const Variant setting_before = GLOBAL_GET("rendering/pathtracing/cluster_blas");
	{
		ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", false);
		RenderingServerTypes::SurfaceData surface;
		REQUIRE(RS::get_singleton()->mesh_create_surface_data_from_arrays(&surface, RSE::PRIMITIVE_TRIANGLES, arrays) == OK);
		CHECK(surface.cluster_data.is_empty());
		CHECK(surface.index_count == uint32_t(original.size()));
	}
	{
		ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", true);
		RenderingServerTypes::SurfaceData surface;
		REQUIRE(RS::get_singleton()->mesh_create_surface_data_from_arrays(&surface, RSE::PRIMITIVE_TRIANGLES, arrays) == OK);
		CHECK(surface.cluster_data == table_bytes);
		REQUIRE(surface.index_count == uint32_t(original.size()));
		REQUIRE(surface.index_data.size() == int(sizeof(uint16_t)) * original.size()); // 1681 vertices: 16-bit indices.
		const uint16_t *packed = reinterpret_cast<const uint16_t *>(surface.index_data.ptr());
		int equal = 0;
		for (int i = 0; i < indices.size(); i++) {
			equal += int(packed[i]) == indices[i] ? 1 : 0;
		}
		CHECK(equal == indices.size());
		CHECK(Vector<int>(arrays[RSE::ARRAY_INDEX]) == original);

		// A pack that skips the clusterizer (D84) carries no table and the caller's index order,
		// and its vertex stream is the clustered pack's, byte for byte.
		RenderingServerTypes::SurfaceData skipped;
		REQUIRE(RS::get_singleton()->mesh_create_surface_data_from_arrays(&skipped, RSE::PRIMITIVE_TRIANGLES, arrays, Array(), Dictionary(), 0, false) == OK);
		CHECK(skipped.cluster_data.is_empty());
		CHECK(skipped.vertex_data == surface.vertex_data);
		REQUIRE(skipped.index_data.size() == surface.index_data.size());
		const uint16_t *kept = reinterpret_cast<const uint16_t *>(skipped.index_data.ptr());
		int same = 0;
		for (int i = 0; i < original.size(); i++) {
			same += int(kept[i]) == original[i] ? 1 : 0;
		}
		CHECK(same == original.size());
	}
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", setting_before);
}

TEST_CASE("[SceneTree][ArrayMesh] A skinned surface is clustered with its skin in vertex order (D82).") {
	// A 30 by 30 grid with four bones and weights per vertex and one blend shape, built into
	// surface data with the setting off and on: the clustered surface's vertex, attribute, skin and
	// blend-shape data are byte for byte the unclustered surface's, since the reorder touches the
	// index data alone, and the clustered index data is a winding-preserving permutation of the
	// triangles.
	const int n = 30;
	Vector<Vector3> positions;
	Vector<Vector3> normals;
	Vector<Vector2> uvs;
	Vector<int> bones;
	Vector<float> weights;
	Vector<Vector3> shape_positions;
	for (int y = 0; y <= n; y++) {
		for (int x = 0; x <= n; x++) {
			const Vector3 p(x, Math::cos(x * 0.25) * Math::sin(y * 0.35), y);
			positions.push_back(p);
			normals.push_back(Vector3(0, 1, 0));
			uvs.push_back(Vector2(float(x) / n, float(y) / n));
			const int bone = (x * 7 + y * 3) % 5;
			bones.push_back(bone);
			bones.push_back((bone + 1) % 5);
			bones.push_back(0);
			bones.push_back(0);
			const float w = 0.25f + 0.5f * float(x % 3) / 2.0f;
			weights.push_back(w);
			weights.push_back(1.0f - w);
			weights.push_back(0.0f);
			weights.push_back(0.0f);
			shape_positions.push_back(p + Vector3(0, 0.5 + 0.01 * x, 0));
		}
	}
	Vector<int> indices;
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++) {
			const int i = y * (n + 1) + x;
			indices.push_back(i);
			indices.push_back(i + 1);
			indices.push_back(i + n + 1);
			indices.push_back(i + 1);
			indices.push_back(i + n + 2);
			indices.push_back(i + n + 1);
		}
	}
	Array arrays;
	arrays.resize(RSE::ARRAY_MAX);
	arrays[RSE::ARRAY_VERTEX] = positions;
	arrays[RSE::ARRAY_NORMAL] = normals;
	arrays[RSE::ARRAY_TEX_UV] = uvs;
	arrays[RSE::ARRAY_BONES] = bones;
	arrays[RSE::ARRAY_WEIGHTS] = weights;
	arrays[RSE::ARRAY_INDEX] = indices;
	Array shape;
	shape.resize(RSE::ARRAY_MAX);
	shape[RSE::ARRAY_VERTEX] = shape_positions;
	shape[RSE::ARRAY_NORMAL] = normals;
	Array blend_shapes;
	blend_shapes.push_back(shape);

	const Variant setting_before = GLOBAL_GET("rendering/pathtracing/cluster_blas");
	RenderingServerTypes::SurfaceData plain;
	RenderingServerTypes::SurfaceData clustered;
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", false);
	REQUIRE(RS::get_singleton()->mesh_create_surface_data_from_arrays(&plain, RSE::PRIMITIVE_TRIANGLES, arrays, blend_shapes) == OK);
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", true);
	REQUIRE(RS::get_singleton()->mesh_create_surface_data_from_arrays(&clustered, RSE::PRIMITIVE_TRIANGLES, arrays, blend_shapes) == OK);
	// With the hierarchy on as well (D87): the skinned surface without its blend shape gets a
	// cluster LOD table whose coarse levels ride in the table, the skin and vertex data untouched
	// and the index data the drawn range alone, so the skinning output the coarse levels index is
	// the same buffer; with the blend shape it keeps the flat table (the hierarchy's errors are the
	// rest pose's, and a blend-shaped surface's shape is in its deformation).
	const Variant lod_before = GLOBAL_GET("rendering/pathtracing/cluster_lod");
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_lod", true);
	RenderingServerTypes::SurfaceData lod;
	RenderingServerTypes::SurfaceData lod_blend;
	REQUIRE(RS::get_singleton()->mesh_create_surface_data_from_arrays(&lod, RSE::PRIMITIVE_TRIANGLES, arrays) == OK);
	REQUIRE(RS::get_singleton()->mesh_create_surface_data_from_arrays(&lod_blend, RSE::PRIMITIVE_TRIANGLES, arrays, blend_shapes) == OK);
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_lod", lod_before);
	RenderingServerTypes::SurfaceData plain_skin;
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", false);
	REQUIRE(RS::get_singleton()->mesh_create_surface_data_from_arrays(&plain_skin, RSE::PRIMITIVE_TRIANGLES, arrays) == OK);
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", setting_before);

	CHECK(lod.vertex_count == plain_skin.vertex_count);
	CHECK(lod.index_count == plain_skin.index_count);
	CHECK(lod.index_data.size() == plain_skin.index_data.size());
	CHECK(lod.format == plain_skin.format);
	CHECK((lod.format & RSE::ARRAY_FORMAT_BONES) != 0);
	CHECK(lod.vertex_data == plain_skin.vertex_data);
	CHECK(lod.attribute_data == plain_skin.attribute_data);
	CHECK(lod.skin_data == plain_skin.skin_data);
	CHECK(lod_blend.blend_shape_data == plain.blend_shape_data);
	CHECK(lod_blend.skin_data == plain.skin_data);
	const bool lod_built = RenderingServer::build_cluster_lod_func != nullptr && RenderingServer::build_clusters_func != nullptr;
	if (lod_built) {
		// The blend-shaped surface's table is the flat one, whatever the setting.
		REQUIRE(lod_blend.cluster_data.size() >= int(sizeof(uint32_t) * 2));
		CHECK(reinterpret_cast<const uint32_t *>(lod_blend.cluster_data.ptr())[0] == RenderingServerTypes::MESH_CLUSTER_DATA_VERSION);
		REQUIRE(lod.cluster_data.size() >= int(sizeof(uint32_t) * RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS));
		const uint32_t *header = reinterpret_cast<const uint32_t *>(lod.cluster_data.ptr());
		CHECK(header[0] == RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION);
		CHECK(header[3] > 1); // Levels.
		CHECK(header[5] == uint32_t(indices.size() / 3)); // LOD 0 triangles, the drawn range.
		CHECK(header[6] > 0); // Coarse triangles, in the table.
		// The tail's indices, the coarse levels' triangles, reach the surface's own vertices only.
		const uint32_t cluster_count = header[1];
		const uint32_t group_count = header[2];
		const uint32_t coarse_triangles = header[6];
		const uint32_t tail_offset = RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS + RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS * cluster_count + RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS * group_count;
		REQUIRE(uint32_t(lod.cluster_data.size() / sizeof(uint32_t)) == tail_offset + coarse_triangles * 3);
		uint32_t tail_in_range = 0;
		for (uint32_t i = 0; i < coarse_triangles * 3; i++) {
			tail_in_range += header[tail_offset + i] < uint32_t(positions.size()) ? 1 : 0;
		}
		CHECK(tail_in_range == coarse_triangles * 3);
	}

	CHECK(plain.cluster_data.is_empty());
	CHECK((plain.format & RSE::ARRAY_FORMAT_BONES) != 0);
	CHECK(plain.blend_shape_data.size() > 0);
	REQUIRE(clustered.vertex_count == plain.vertex_count);
	REQUIRE(clustered.index_count == plain.index_count);
	REQUIRE(clustered.index_data.size() == plain.index_data.size());
	CHECK(clustered.format == plain.format);
	CHECK(clustered.vertex_data == plain.vertex_data);
	CHECK(clustered.attribute_data == plain.attribute_data);
	CHECK(clustered.skin_data == plain.skin_data);
	CHECK(clustered.blend_shape_data == plain.blend_shape_data);
	if (RenderingServer::build_clusters_func == nullptr) {
		CHECK(clustered.cluster_data.is_empty());
		CHECK(clustered.index_data == plain.index_data);
		return;
	}
	CHECK(clustered.cluster_data.size() >= int(sizeof(uint32_t) * 2));
	CHECK(clustered.index_data != plain.index_data);

	// The packed indices (16-bit here: 961 vertices), decoded and compared as triangle sets with
	// the winding kept (the smallest index leads, the order of the other two is the winding).
	auto decode = [](const RenderingServerTypes::SurfaceData &p_surface) -> Vector<int> {
		Vector<int> out;
		const bool wide = p_surface.vertex_count > 65536;
		const int count = p_surface.index_data.size() / int(wide ? sizeof(uint32_t) : sizeof(uint16_t));
		out.resize(count);
		for (int i = 0; i < count; i++) {
			out.write[i] = wide ? int(reinterpret_cast<const uint32_t *>(p_surface.index_data.ptr())[i]) : int(reinterpret_cast<const uint16_t *>(p_surface.index_data.ptr())[i]);
		}
		return out;
	};
	auto triangle_key = [](int a, int b, int c) -> uint64_t {
		while (!(a <= b && a <= c)) {
			const int t = a;
			a = b;
			b = c;
			c = t;
		}
		return (uint64_t(a) << 42) | (uint64_t(b) << 21) | uint64_t(c);
	};
	const Vector<int> plain_indices = decode(plain);
	const Vector<int> clustered_indices = decode(clustered);
	REQUIRE(plain_indices == indices);
	REQUIRE(clustered_indices.size() == indices.size());
	HashSet<uint64_t> original_triangles;
	for (int t = 0; t < indices.size(); t += 3) {
		original_triangles.insert(triangle_key(indices[t], indices[t + 1], indices[t + 2]));
	}
	HashSet<uint64_t> clustered_triangles;
	int matched = 0;
	int in_range = 0;
	for (int t = 0; t < clustered_indices.size(); t += 3) {
		const uint64_t key = triangle_key(clustered_indices[t], clustered_indices[t + 1], clustered_indices[t + 2]);
		clustered_triangles.insert(key);
		matched += original_triangles.has(key) ? 1 : 0;
		for (int k = 0; k < 3; k++) {
			in_range += (clustered_indices[t + k] >= 0 && clustered_indices[t + k] < positions.size()) ? 1 : 0;
		}
	}
	CHECK(matched == indices.size() / 3);
	CHECK(in_range == clustered_indices.size());
	CHECK(original_triangles.size() == uint32_t(indices.size() / 3));
	CHECK(clustered_triangles.size() == uint32_t(indices.size() / 3)); // A permutation: no triangle twice.

	// The LOD pack's index data is LOD 0 alone, the same permutation property (D87).
	if (lod_built) {
		const Vector<int> lod_indices = decode(lod);
		REQUIRE(lod_indices.size() == indices.size());
		HashSet<uint64_t> lod_triangles;
		int lod_matched = 0;
		int lod_in_range = 0;
		for (int t = 0; t < lod_indices.size(); t += 3) {
			const uint64_t key = triangle_key(lod_indices[t], lod_indices[t + 1], lod_indices[t + 2]);
			lod_triangles.insert(key);
			lod_matched += original_triangles.has(key) ? 1 : 0;
			for (int k = 0; k < 3; k++) {
				lod_in_range += (lod_indices[t + k] >= 0 && lod_indices[t + k] < positions.size()) ? 1 : 0;
			}
		}
		CHECK(lod_matched == indices.size() / 3);
		CHECK(lod_in_range == lod_indices.size());
		CHECK(lod_triangles.size() == uint32_t(indices.size() / 3));
	}
}

TEST_CASE("[SceneTree][ArrayMesh] A cluster LOD hierarchy keeps LOD 0 as the index array and its levels in the table (D85).") {
	// A 60 by 60 grid: 3721 vertices, 7200 triangles, enough clusters for several levels.
	const int n = 60;
	Vector<Vector3> positions;
	Vector<Vector3> normals;
	Vector<Vector2> uvs;
	for (int y = 0; y <= n; y++) {
		for (int x = 0; x <= n; x++) {
			positions.push_back(Vector3(x, Math::sin(x * 0.3) * Math::cos(y * 0.2), y));
			normals.push_back(Vector3(0, 1, 0));
			uvs.push_back(Vector2(x / float(n), y / float(n)));
		}
	}
	Vector<int> indices;
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++) {
			const int i = y * (n + 1) + x;
			indices.push_back(i);
			indices.push_back(i + 1);
			indices.push_back(i + n + 1);
			indices.push_back(i + 1);
			indices.push_back(i + n + 2);
			indices.push_back(i + n + 1);
		}
	}
	const Vector<int> original = indices;

	Vector<uint8_t> table_bytes = RenderingServer::mesh_surface_build_cluster_lod(positions, normals, uvs, indices);
	if (RenderingServer::build_cluster_lod_func == nullptr) {
		CHECK(table_bytes.is_empty());
		CHECK(indices == original);
		return;
	}
	REQUIRE(table_bytes.size() >= int(sizeof(uint32_t) * RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS));
	REQUIRE(table_bytes.size() % sizeof(uint32_t) == 0);
	const uint32_t *table = reinterpret_cast<const uint32_t *>(table_bytes.ptr());
	const uint32_t words = table_bytes.size() / sizeof(uint32_t);
	CHECK(table[0] == RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION);
	const uint32_t cluster_count = table[1];
	const uint32_t group_count = table[2];
	const uint32_t level_count = table[3];
	const uint32_t lod0_cluster_count = table[4];
	const uint32_t lod0_triangle_count = table[5];
	const uint32_t coarse_triangle_count = table[6];
	CHECK(table[7] == 0);
	CHECK(level_count > 1);
	CHECK(group_count > 1);
	CHECK(lod0_cluster_count > 1);
	CHECK(lod0_cluster_count < cluster_count);
	CHECK(lod0_triangle_count == uint32_t(original.size() / 3));
	CHECK(coarse_triangle_count > 0);
	CHECK(coarse_triangle_count < lod0_triangle_count * 2);
	REQUIRE(words == RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS + RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS * cluster_count + RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS * group_count + coarse_triangle_count * 3);
	const uint32_t *cluster_words = table + RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS;
	const uint32_t *group_words = cluster_words + RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS * cluster_count;
	const uint32_t *coarse = group_words + RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS * group_count;

	// The clusters: within the limits, LOD 0 first as original geometry, the rest refined from
	// an earlier group, the triangle sums those of the header.
	uint32_t lod0_sum = 0;
	uint32_t coarse_sum = 0;
	uint32_t in_limits = 0;
	uint32_t lod0_original = 0;
	uint32_t coarse_refined = 0;
	for (uint32_t c = 0; c < cluster_count; c++) {
		const uint32_t *w = cluster_words + c * RenderingServerTypes::MESH_CLUSTER_LOD_CLUSTER_WORDS;
		const uint32_t triangles = w[0] & 0xFFFFu;
		const uint32_t vertices = w[0] >> 16;
		const int32_t refined = int32_t(w[2]);
		in_limits += (triangles > 0 && triangles <= RenderingServerTypes::MESH_CLUSTER_MAX_TRIANGLES && vertices > 0 && vertices <= RenderingServerTypes::MESH_CLUSTER_MAX_VERTICES && w[1] < group_count) ? 1 : 0;
		if (c < lod0_cluster_count) {
			lod0_sum += triangles;
			lod0_original += refined == -1 ? 1 : 0;
		} else {
			coarse_sum += triangles;
			coarse_refined += (refined >= 0 && uint32_t(refined) < w[1]) ? 1 : 0;
		}
	}
	CHECK(in_limits == cluster_count);
	CHECK(lod0_sum == lod0_triangle_count);
	CHECK(coarse_sum == coarse_triangle_count);
	CHECK(lod0_original == lod0_cluster_count);
	CHECK(coarse_refined == cluster_count - lod0_cluster_count);

	// The groups: contiguous over the clusters in nondecreasing depth, the depth-0 groups
	// covering exactly the LOD 0 clusters.
	uint32_t next_cluster = 0;
	uint32_t depth0_clusters = 0;
	uint32_t max_depth = 0;
	bool ordered = true;
	int32_t last_depth = 0;
	for (uint32_t g = 0; g < group_count; g++) {
		const uint32_t *w = group_words + g * RenderingServerTypes::MESH_CLUSTER_LOD_GROUP_WORDS;
		const int32_t depth = int32_t(w[0]);
		ordered = ordered && depth >= last_depth && w[1] == next_cluster && w[2] > 0;
		if (depth == 0) {
			depth0_clusters += w[2];
		}
		next_cluster += w[2];
		max_depth = MAX(max_depth, uint32_t(depth));
		last_depth = depth;
	}
	CHECK(ordered);
	CHECK(next_cluster == cluster_count);
	CHECK(depth0_clusters == lod0_cluster_count);
	CHECK(max_depth + 1 == level_count);

	// Every coarse index inside the surface; the LOD 0 index array a permutation of the input.
	uint32_t coarse_in_range = 0;
	for (uint32_t i = 0; i < coarse_triangle_count * 3; i++) {
		coarse_in_range += coarse[i] < uint32_t(positions.size()) ? 1 : 0;
	}
	CHECK(coarse_in_range == coarse_triangle_count * 3);
	REQUIRE(indices.size() == original.size());
	auto triangle_key = [](int a, int b, int c) -> uint64_t {
		return (uint64_t(a) << 42) | (uint64_t(b) << 21) | uint64_t(c);
	};
	HashSet<uint64_t> original_triangles;
	HashSet<uint64_t> lod0_triangles;
	for (int t = 0; t < original.size(); t += 3) {
		original_triangles.insert(triangle_key(original[t], original[t + 1], original[t + 2]));
	}
	uint32_t lod0_found = 0;
	for (int t = 0; t < indices.size(); t += 3) {
		const uint64_t key = triangle_key(indices[t], indices[t + 1], indices[t + 2]);
		const uint64_t key_b = triangle_key(indices[t + 1], indices[t + 2], indices[t]);
		const uint64_t key_c = triangle_key(indices[t + 2], indices[t], indices[t + 1]);
		lod0_found += (original_triangles.has(key) || original_triangles.has(key_b) || original_triangles.has(key_c)) ? 1 : 0;
		lod0_triangles.insert(key);
	}
	CHECK(lod0_found == uint32_t(indices.size() / 3));
	CHECK(lod0_triangles.size() == uint32_t(original.size() / 3));

	// An ArrayMesh built from these arrays with both settings on carries the table, and a
	// duplicate() round trip (_get_surfaces then _set_surfaces) keeps the index data at the
	// drawn range and the table whole.
	const Variant blas_before = GLOBAL_GET("rendering/pathtracing/cluster_blas");
	const Variant lod_before = GLOBAL_GET("rendering/pathtracing/cluster_lod");
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", true);
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_lod", true);
	Array arrays;
	arrays.resize(RSE::ARRAY_MAX);
	arrays[RSE::ARRAY_VERTEX] = positions;
	arrays[RSE::ARRAY_NORMAL] = normals;
	arrays[RSE::ARRAY_TEX_UV] = uvs;
	arrays[RSE::ARRAY_INDEX] = original;
	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", blas_before);
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_lod", lod_before);
	REQUIRE(mesh->get_surface_count() == 1);
	Ref<ArrayMesh> copy = mesh->duplicate();
	REQUIRE(copy.is_valid());
	REQUIRE(copy->get_surface_count() == 1);
	auto check_surfaces = [&](const Ref<ArrayMesh> &p_mesh) {
		Array surfaces = p_mesh->get("_surfaces");
		CHECK(surfaces.size() == 1);
		if (surfaces.size() != 1) {
			return;
		}
		Dictionary d = surfaces[0];
		CHECK(d.has("index_count"));
		CHECK(d.has("index_data"));
		CHECK(d.has("cluster_data"));
		const int index_count = d.has("index_count") ? int(d["index_count"]) : 0;
		const Vector<uint8_t> index_data = d.has("index_data") ? Vector<uint8_t>(d["index_data"]) : Vector<uint8_t>();
		const Vector<uint8_t> cluster_data = d.has("cluster_data") ? Vector<uint8_t>(d["cluster_data"]) : Vector<uint8_t>();
		CHECK(index_count == original.size());
		CHECK(index_data.size() == index_count * int(sizeof(uint16_t))); // 3721 vertices: 16-bit indices, the drawn range alone.
		CHECK(cluster_data.size() == table_bytes.size());
		if (cluster_data.size() >= int(sizeof(uint32_t) * RenderingServerTypes::MESH_CLUSTER_LOD_HEADER_WORDS)) {
			CHECK(reinterpret_cast<const uint32_t *>(cluster_data.ptr())[0] == RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION);
			CHECK(reinterpret_cast<const uint32_t *>(cluster_data.ptr())[6] == coarse_triangle_count);
		}
	};
	check_surfaces(mesh);
	check_surfaces(copy);
}

// D122: a hand-built subdivision payload for a quad of four vertices, its cage the same four
// vertices, its refined level the quad itself; the storages' rules on it, without the builder.
struct SubdivisionSection {
	uint32_t id = 0;
	LocalVector<uint32_t> words;
};

static Vector<uint8_t> subdivision_payload_from(const LocalVector<SubdivisionSection> &p_sections, uint32_t p_flags, uint32_t p_owner_surface) {
	const uint32_t align = RenderingServerTypes::MESH_SUBDIVISION_SECTION_ALIGN;
	uint32_t offset = (RenderingServerTypes::MESH_SUBDIVISION_HEADER_WORDS + p_sections.size() * RenderingServerTypes::MESH_SUBDIVISION_SECTION_WORDS) * sizeof(uint32_t);
	LocalVector<uint32_t> table = { RenderingServerTypes::MESH_SUBDIVISION_DATA_VERSION, p_flags, p_owner_surface, p_sections.size() };
	LocalVector<uint32_t> offsets;
	for (const SubdivisionSection &section : p_sections) {
		offset = (offset + align - 1) / align * align;
		offsets.push_back(offset);
		table.push_back(section.id);
		table.push_back(offset);
		table.push_back(section.words.size() * sizeof(uint32_t));
		offset += section.words.size() * sizeof(uint32_t);
	}
	Vector<uint8_t> data;
	data.resize(offset);
	memset(data.ptrw(), 0, offset);
	memcpy(data.ptrw(), table.ptr(), table.size() * sizeof(uint32_t));
	for (uint32_t i = 0; i < p_sections.size(); i++) {
		if (p_sections[i].words.size()) {
			memcpy(data.ptrw() + offsets[i], p_sections[i].words.ptr(), p_sections[i].words.size() * sizeof(uint32_t));
		}
	}
	return data;
}

static LocalVector<SubdivisionSection> subdivision_quad_sections(bool p_owner, uint32_t p_blend_shapes) {
	LocalVector<SubdivisionSection> sections;
	auto add = [&sections](uint32_t p_id, const LocalVector<uint32_t> &p_words) {
		SubdivisionSection section;
		section.id = p_id;
		section.words = p_words;
		sections.push_back(section);
	};
	if (p_owner) {
		add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, { 4, 4, 4, 4, 1, 1, p_blend_shapes, 2, 4, 4, RenderingServerTypes::MESH_SUBDIVISION_NO_DISPLACEMENT });
		LocalVector<uint32_t> positions;
		const float corners[12] = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0 };
		for (float value : corners) {
			uint32_t bits;
			memcpy(&bits, &value, sizeof(bits));
			positions.push_back(bits);
		}
		add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_POSITIONS, positions);
		// With blend shapes, shape 0 moves cage vertex 1 by (0, 0, 0.5).
		if (p_blend_shapes > 0) {
			add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_OFFSETS, { 0, 0, 1, 1, 1 });
			add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_ENTRIES, { 0, 0, 0, 0x3f000000 });
		} else {
			add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_OFFSETS, { 0, 0, 0, 0, 0 });
			add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_ENTRIES, {});
		}
		add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_OFFSETS, { 0, 1, 2, 3, 4 });
		add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_INDICES, { 0, 1, 2, 3 }); // 32-bit indices.
		const uint32_t one = 0x3f800000;
		add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_WEIGHTS, { one, one, one, one });
		add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_QUADS, { 0, 1, 2, 3 });
		add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_NORMAL_OFFSETS, { 0, 1, 2, 3, 4 });
		add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_NORMAL_ENTRIES, { 0, 1, 2, 3 });
		add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_WELDED, { 0, 1, 2, 3 });
		add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_U_WELDED, { 0, 1, 2, 3 });
	}
	add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, { 0, 1, 2, 3 });
	// Triangles (0, 1, 2) and (0, 2, 3): vertex 0 in two, 1 in one, 2 in two, 3 in one.
	add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_OFFSETS, { 0, 2, 3, 5, 6 });
	add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_ENTRIES, { 0, 3, 1, 2, 4, 5 });
	add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_SIGNS, { 0b1011 }); // Vertex 2's sign is -1.
	return sections;
}

static Vector<uint8_t> subdivision_quad_payload(bool p_owner, uint32_t p_blend_shapes, uint32_t p_owner_surface = 0) {
	return subdivision_payload_from(subdivision_quad_sections(p_owner, p_blend_shapes), p_owner ? RenderingServerTypes::MESH_SUBDIVISION_FLAG_OWNER : 0, p_owner_surface);
}

// The quad as surface data: four vertices, two triangles, no blend arrays.
static RenderingServerTypes::SurfaceData subdivision_quad_surface(bool p_skin = false) {
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = PackedVector3Array{ Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0), Vector3(0, 1, 0) };
	arrays[Mesh::ARRAY_NORMAL] = PackedVector3Array{ Vector3(0, 0, 1), Vector3(0, 0, 1), Vector3(0, 0, 1), Vector3(0, 0, 1) };
	arrays[Mesh::ARRAY_INDEX] = PackedInt32Array{ 0, 1, 2, 0, 2, 3 };
	if (p_skin) {
		arrays[Mesh::ARRAY_BONES] = PackedInt32Array{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		arrays[Mesh::ARRAY_WEIGHTS] = PackedFloat32Array{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 };
	}
	RenderingServerTypes::SurfaceData surface;
	CHECK(RS::get_singleton()->mesh_create_surface_data_from_arrays(&surface, RSE::PRIMITIVE_TRIANGLES, arrays) == OK);
	return surface;
}

static void add_subdivision_quad(const Ref<ArrayMesh> &p_mesh, const RenderingServerTypes::SurfaceData &p_surface, const Vector<uint8_t> &p_payload, const Vector<uint8_t> &p_blend_data = Vector<uint8_t>()) {
	p_mesh->add_surface(p_surface.format, Mesh::PrimitiveType(p_surface.primitive), p_surface.vertex_data, p_surface.attribute_data, p_surface.skin_data, p_surface.vertex_count, p_surface.index_data, p_surface.index_count, p_surface.aabb, p_blend_data, p_surface.bone_aabbs, p_surface.lods, p_surface.uv_scale, p_surface.cluster_data, p_payload);
}

static int rendering_surface_count(const Ref<ArrayMesh> &p_mesh) {
	return RS::get_singleton()->mesh_get_surface_count(p_mesh->get_rid());
}

TEST_CASE("[SceneTree][ArrayMesh] A subdivided surface keeps its payload, carries no blend arrays, and refuses blend readers (D122).") {
	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	mesh->add_blend_shape("A");
	mesh->add_blend_shape("B");
	const RenderingServerTypes::SurfaceData surface = subdivision_quad_surface();
	const Vector<uint8_t> payload = subdivision_quad_payload(true, 2);
	String error;
	CHECK_MESSAGE(RenderingServer::mesh_subdivision_data_is_valid(payload, &error), error);
	add_subdivision_quad(mesh, surface, payload);
	REQUIRE(rendering_surface_count(mesh) == 1);
	CHECK(mesh->surface_is_subdivided(0));
	CHECK(mesh->get_blend_shape_count() == 2);
	CHECK(RS::get_singleton()->mesh_get_surface(mesh->get_rid(), 0).subdivision_data == payload);

	// A second surface names the first as its owner.
	add_subdivision_quad(mesh, surface, subdivision_quad_payload(false, 2, 0));
	CHECK(rendering_surface_count(mesh) == 2);

	Ref<ArrayMesh> copy = mesh->duplicate();
	REQUIRE(copy.is_valid());
	const Ref<ArrayMesh> both[2] = { mesh, copy };
	for (const Ref<ArrayMesh> &each : both) {
		const Array surfaces = each->get("_surfaces");
		REQUIRE(surfaces.size() == 2);
		const Dictionary d = surfaces[0];
		CHECK(Vector<uint8_t>(d.get("subdivision_data", Vector<uint8_t>())) == payload);
		CHECK_FALSE(d.has("blend_shapes"));
		CHECK(each->surface_is_subdivided(1));
		CHECK(rendering_surface_count(each) == 2);
	}

	ERR_PRINT_OFF;
	CHECK(mesh->surface_get_blend_shape_arrays(0).is_empty());
	CHECK(ImporterMesh::from_mesh(mesh)->get_surface_count() == 0);
	ERR_PRINT_ON;
}

TEST_CASE("[SceneTree][ArrayMesh] The storages refuse a subdivided surface against the mesh's rules (D122).") {
	const RenderingServerTypes::SurfaceData surface = subdivision_quad_surface();
	ERR_PRINT_OFF;
	SUBCASE("Blend data on a subdivided surface") {
		Ref<ArrayMesh> mesh;
		mesh.instantiate();
		mesh->add_blend_shape("A");
		Vector<uint8_t> blend_data;
		blend_data.resize(surface.vertex_data.size());
		add_subdivision_quad(mesh, surface, subdivision_quad_payload(true, 1), blend_data);
		CHECK(rendering_surface_count(mesh) == 0);
	}
	SUBCASE("Skin on a subdivided surface") {
		Ref<ArrayMesh> mesh;
		mesh.instantiate();
		add_subdivision_quad(mesh, subdivision_quad_surface(true), subdivision_quad_payload(true, 0));
		CHECK(rendering_surface_count(mesh) == 0);
	}
	SUBCASE("Subdivided and plain surfaces on a mesh with blend shapes") {
		Ref<ArrayMesh> mesh;
		mesh.instantiate();
		mesh->add_blend_shape("A");
		add_subdivision_quad(mesh, surface, subdivision_quad_payload(true, 1));
		REQUIRE(rendering_surface_count(mesh) == 1);
		Vector<uint8_t> blend_data;
		blend_data.resize(surface.vertex_data.size());
		memset(blend_data.ptrw(), 0, blend_data.size());
		add_subdivision_quad(mesh, surface, Vector<uint8_t>(), blend_data);
		CHECK(rendering_surface_count(mesh) == 1);
	}
	SUBCASE("A plain surface first, then a subdivided one, on a mesh with blend shapes") {
		Ref<ArrayMesh> mesh;
		mesh.instantiate();
		mesh->add_blend_shape("A");
		Vector<uint8_t> blend_data;
		blend_data.resize(surface.vertex_data.size());
		memset(blend_data.ptrw(), 0, blend_data.size());
		add_subdivision_quad(mesh, surface, Vector<uint8_t>(), blend_data);
		REQUIRE(rendering_surface_count(mesh) == 1);
		add_subdivision_quad(mesh, surface, subdivision_quad_payload(true, 1, 1));
		CHECK(rendering_surface_count(mesh) == 1);
	}
	SUBCASE("Without blend shapes, subdivided and plain surfaces mix") {
		Ref<ArrayMesh> mesh;
		mesh.instantiate();
		add_subdivision_quad(mesh, surface, subdivision_quad_payload(true, 0));
		add_subdivision_quad(mesh, surface, Vector<uint8_t>());
		CHECK(rendering_surface_count(mesh) == 2);
	}
	SUBCASE("A subdivided surface keeps uncompressed attributes") {
		Ref<ArrayMesh> mesh;
		mesh.instantiate();
		RenderingServerTypes::SurfaceData compressed = surface;
		compressed.format |= RSE::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
		compressed.subdivision_data = subdivision_quad_payload(true, 0);
		String error;
		CHECK_FALSE(RenderingServer::mesh_subdivision_surface_is_valid(compressed, 0, 0, -1, Vector<uint8_t>(), &error));
		add_subdivision_quad(mesh, compressed, subdivision_quad_payload(true, 0));
		CHECK(rendering_surface_count(mesh) == 0);
	}
	SUBCASE("A subdivided surface is whole refined quads of 3D vertices, its arrays matching its counts (D122 S2)") {
		// Each refusal by its own reason: the runtime passes read and write these arrays by the counts.
		auto refusal = [](const RenderingServerTypes::SurfaceData &p_candidate) {
			RenderingServerTypes::SurfaceData candidate = p_candidate;
			candidate.subdivision_data = subdivision_quad_payload(true, 0);
			String error;
			CHECK_FALSE(RenderingServer::mesh_subdivision_surface_is_valid(candidate, 0, 0, -1, Vector<uint8_t>(), &error));
			return error;
		};
		{
			RenderingServerTypes::SurfaceData candidate = surface;
			candidate.index_count = 3;
			candidate.index_data.resize(3 * sizeof(uint16_t));
			CHECK(refusal(candidate).contains("two triangles each"));
		}
		{
			RenderingServerTypes::SurfaceData candidate = surface;
			candidate.format |= RSE::ARRAY_FLAG_USE_2D_VERTICES;
			CHECK(refusal(candidate).contains("3D vertices"));
		}
		{
			RenderingServerTypes::SurfaceData candidate = surface;
			const uint16_t past = 4;
			memcpy(candidate.index_data.ptrw() + 5 * sizeof(uint16_t), &past, sizeof(past));
			CHECK(refusal(candidate).contains("index 5 is 4"));
		}
		{
			RenderingServerTypes::SurfaceData candidate = surface;
			candidate.index_data.resize(candidate.index_data.size() - sizeof(uint16_t));
			CHECK(refusal(candidate).contains("index data does not match"));
		}
		{
			RenderingServerTypes::SurfaceData candidate = surface;
			candidate.vertex_data.resize(candidate.vertex_data.size() - sizeof(float));
			CHECK(refusal(candidate).contains("vertex or attribute data"));
		}
		{
			RenderingServerTypes::SurfaceData candidate = surface;
			candidate.attribute_data.resize(candidate.attribute_data.size() + sizeof(float));
			CHECK(refusal(candidate).contains("vertex or attribute data"));
		}
		// The quad itself passes.
		RenderingServerTypes::SurfaceData good = surface;
		good.subdivision_data = subdivision_quad_payload(true, 0);
		CHECK(RenderingServer::mesh_subdivision_surface_is_valid(good, 0, 0, -1, Vector<uint8_t>()));
	}
	SUBCASE("A subdivided surface with blend shapes carries no cluster LOD hierarchy") {
		RenderingServerTypes::SurfaceData with_lod = surface;
		with_lod.subdivision_data = subdivision_quad_payload(true, 1);
		with_lod.cluster_data.resize(sizeof(uint32_t));
		const uint32_t lod_version = RenderingServerTypes::MESH_CLUSTER_LOD_DATA_VERSION;
		memcpy(with_lod.cluster_data.ptrw(), &lod_version, sizeof(uint32_t));
		String error;
		CHECK_FALSE(RenderingServer::mesh_subdivision_surface_is_valid(with_lod, 0, 1, -1, Vector<uint8_t>(), &error));
		CHECK(error.contains("cluster LOD"));
		Ref<ArrayMesh> mesh;
		mesh.instantiate();
		mesh->add_blend_shape("A");
		add_subdivision_quad(mesh, with_lod, with_lod.subdivision_data);
		CHECK(rendering_surface_count(mesh) == 0);
		// The flat table passes with the same payload and blend count: the version word is the reason.
		RenderingServerTypes::SurfaceData flat = with_lod;
		const uint32_t flat_version = RenderingServerTypes::MESH_CLUSTER_DATA_VERSION;
		memcpy(flat.cluster_data.ptrw(), &flat_version, sizeof(uint32_t));
		CHECK(RenderingServer::mesh_subdivision_surface_is_valid(flat, 0, 1, -1, Vector<uint8_t>(), &error));
		// Without blend shapes the hierarchy is allowed, as on D87's skinned surfaces.
		with_lod.subdivision_data = subdivision_quad_payload(true, 0);
		CHECK(RenderingServer::mesh_subdivision_surface_is_valid(with_lod, 0, 0, -1, Vector<uint8_t>(), &error));
	}
	SUBCASE("The owner's blend shape count is the mesh's") {
		Ref<ArrayMesh> mesh;
		mesh.instantiate();
		mesh->add_blend_shape("A");
		add_subdivision_quad(mesh, surface, subdivision_quad_payload(true, 2));
		CHECK(rendering_surface_count(mesh) == 0);
	}
	SUBCASE("A payload without owner sections names an earlier owner surface") {
		Ref<ArrayMesh> mesh;
		mesh.instantiate();
		add_subdivision_quad(mesh, surface, subdivision_quad_payload(false, 0, 0));
		CHECK(rendering_surface_count(mesh) == 0);
		add_subdivision_quad(mesh, surface, subdivision_quad_payload(true, 0, 0));
		REQUIRE(rendering_surface_count(mesh) == 1);
		add_subdivision_quad(mesh, surface, subdivision_quad_payload(false, 0, 1)); // Itself.
		CHECK(rendering_surface_count(mesh) == 1);
		add_subdivision_quad(mesh, surface, subdivision_quad_payload(true, 0, 0)); // An owner naming another surface.
		CHECK(rendering_surface_count(mesh) == 1);
	}
	ERR_PRINT_ON;
}

TEST_CASE("[SceneTree][ArrayMesh] Only surfaces after the last subdivided one can be removed (D122).") {
	// A payload names its owner surface by index, so a removal that shifts a subdivided surface is refused,
	// in the ArrayMesh and in the storage alike.
	const RenderingServerTypes::SurfaceData surface = subdivision_quad_surface();
	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	add_subdivision_quad(mesh, surface, Vector<uint8_t>()); // 0: plain.
	add_subdivision_quad(mesh, surface, subdivision_quad_payload(true, 0, 1)); // 1: owner.
	add_subdivision_quad(mesh, surface, subdivision_quad_payload(false, 0, 1)); // 2: names 1.
	add_subdivision_quad(mesh, surface, Vector<uint8_t>()); // 3: plain.
	REQUIRE(rendering_surface_count(mesh) == 4);
	ERR_PRINT_OFF;
	mesh->surface_remove(0);
	CHECK(mesh->get_surface_count() == 4);
	CHECK(rendering_surface_count(mesh) == 4);
	mesh->surface_remove(1);
	CHECK(mesh->get_surface_count() == 4);
	CHECK(rendering_surface_count(mesh) == 4);
	RS::get_singleton()->mesh_surface_remove(mesh->get_rid(), 1); // The storage refuses on its own too.
	CHECK(rendering_surface_count(mesh) == 4);
	RS::get_singleton()->mesh_surface_remove(mesh->get_rid(), 9); // Out of range.
	CHECK(rendering_surface_count(mesh) == 4);
	ERR_PRINT_ON;
	mesh->surface_remove(3);
	CHECK(rendering_surface_count(mesh) == 3);
	mesh->surface_remove(2);
	CHECK(rendering_surface_count(mesh) == 2);
	mesh->surface_remove(1);
	CHECK(rendering_surface_count(mesh) == 1);
	CHECK_FALSE(mesh->surface_is_subdivided(0));
}

TEST_CASE("[SceneTree][ArrayMesh] Malformed subdivision payloads are refused, also through _surfaces (D122).") {
	const RenderingServerTypes::SurfaceData surface = subdivision_quad_surface();
	auto valid = [&surface](const Vector<uint8_t> &p_payload) {
		RenderingServerTypes::SurfaceData candidate = surface;
		candidate.subdivision_data = p_payload;
		return RenderingServer::mesh_subdivision_surface_is_valid(candidate, 0, 1, -1, Vector<uint8_t>());
	};
	auto with_section = [](uint32_t p_id, const LocalVector<uint32_t> &p_words) {
		LocalVector<SubdivisionSection> sections = subdivision_quad_sections(true, 1);
		for (SubdivisionSection &section : sections) {
			if (section.id == p_id) {
				section.words = p_words;
			}
		}
		return subdivision_payload_from(sections, RenderingServerTypes::MESH_SUBDIVISION_FLAG_OWNER, 0);
	};
	auto set_word = [](Vector<uint8_t> p_payload, uint32_t p_word, uint32_t p_value) {
		memcpy(p_payload.ptrw() + p_word * sizeof(uint32_t), &p_value, sizeof(uint32_t));
		return p_payload;
	};
	const Vector<uint8_t> good = subdivision_quad_payload(true, 1);
	CHECK(valid(good));
	CHECK_FALSE(valid(Vector<uint8_t>{ 1, 2, 3 }));
	CHECK_FALSE(valid(set_word(good, 0, RenderingServerTypes::MESH_SUBDIVISION_DATA_VERSION + 1)));
	CHECK_FALSE(valid(set_word(good, 1, 3))); // An unknown flag.
	CHECK_FALSE(valid(set_word(good, 3, 1000))); // A table past the end.
	CHECK_FALSE(valid(set_word(good, 4, 99))); // An unknown section id.
	CHECK_FALSE(valid(set_word(good, 5, 4))); // A section inside the table.
	{
		uint32_t first_offset;
		memcpy(&first_offset, good.ptr() + 5 * sizeof(uint32_t), sizeof(uint32_t));
		CHECK_FALSE(valid(set_word(good, 5, first_offset + 4))); // Misaligned.
	}
	{
		// A duplicate id: the owner info's id given to the next section as well.
		const Vector<uint8_t> duplicate = set_word(good, 4 + 3, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO);
		CHECK_FALSE(valid(duplicate));
		uint32_t size = 0;
		CHECK(RenderingServer::mesh_subdivision_data_find_section(duplicate, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, &size) >= 0);
	}
	{
		// Owner sections without the owner flag, and an owner without them.
		CHECK_FALSE(valid(subdivision_payload_from(subdivision_quad_sections(true, 1), 0, 0)));
		CHECK_FALSE(valid(subdivision_payload_from(subdivision_quad_sections(false, 1), RenderingServerTypes::MESH_SUBDIVISION_FLAG_OWNER, 0)));
	}
	const uint32_t none = RenderingServerTypes::MESH_SUBDIVISION_NO_DISPLACEMENT;
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, { 4, 4, 4, 4, 1, 1, 1, 2, 4, 4 }))); // Short info: version 3's.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, { 4, 5, 4, 4, 1, 1, 1, 2, 4, 4, none }))); // Own past all.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, { 4, 4, 4, 4, 1, 1, 1, 2, 3, 4, none }))); // Index width.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, { 0, 1, 2, 4 }))); // Past the own U.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, { 0, 1, 2 }))); // Short.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_OFFSETS, { 0, 2, 1, 5, 6 }))); // Decreasing.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_OFFSETS, { 1, 2, 3, 5, 6 }))); // Not from 0.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_ENTRIES, { 0, 3, 1, 2, 4, 6 }))); // Past the indices.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_SIGNS, { 0b1011, 0 }))); // A word too many.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_SIGNS, {}))); // None.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_SIGNS, { 0b11011 }))); // A bit past the last vertex.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_INDICES, { 0, 1, 2, 4 }))); // Past the cage.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_OFFSETS, { 0, 1, 2, 3, 3 }))); // Short of the weights.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_QUADS, { 0, 1, 2, 4 }))); // Past the U.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_NORMAL_ENTRIES, { 0, 1, 2, 4 }))); // Past the corners.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_OFFSETS, { 0, 0, 1, 1 }))); // Short.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_OFFSETS, { 0, 0, 1, 1, 2 }))); // Past the entries.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_ENTRIES, { 0, 0, 0 }))); // Not a whole entry.
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_ENTRIES, { 1, 0, 0, 0x3f000000 }))); // Past the shapes.
	{
		// One vertex's entries not strictly ascending: shape 0 twice.
		LocalVector<SubdivisionSection> sections = subdivision_quad_sections(true, 1);
		for (SubdivisionSection &section : sections) {
			if (section.id == RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_OFFSETS) {
				section.words = LocalVector<uint32_t>{ 0, 0, 2, 2, 2 };
			} else if (section.id == RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_ENTRIES) {
				section.words = LocalVector<uint32_t>{ 0, 0, 0, 0x3f000000, 0, 0, 0, 0x3f000000 };
			}
		}
		CHECK_FALSE(valid(subdivision_payload_from(sections, RenderingServerTypes::MESH_SUBDIVISION_FLAG_OWNER, 0)));
	}
	CHECK_FALSE(valid(set_word(good, 0, 1))); // Version 1, whose deltas were dense.
	CHECK_FALSE(valid(set_word(good, 0, 2))); // Version 2, without tangent signs.
	{
		// Version 3 without its tangent signs.
		LocalVector<SubdivisionSection> sections = subdivision_quad_sections(true, 1);
		for (uint32_t i = 0; i < sections.size(); i++) {
			if (sections[i].id == RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_SIGNS) {
				sections.remove_at(i);
				break;
			}
		}
		CHECK_FALSE(valid(subdivision_payload_from(sections, RenderingServerTypes::MESH_SUBDIVISION_FLAG_OWNER, 0)));
	}
	CHECK_FALSE(valid(set_word(good, 4, 19))); // Version 1's dense delta section id.
	CHECK_FALSE(valid(set_word(good, 0, 3))); // Version 3, without the displacement word.
	{
		// The displacement (S6): a shape in range without cage deltas, with a finite height per own U, or neither. The
		// mesh has two shapes; shape 0 moves cage vertex 1, shape 1 nothing.
		auto valid_two = [&surface](const Vector<uint8_t> &p_payload) {
			RenderingServerTypes::SurfaceData candidate = surface;
			candidate.subdivision_data = p_payload;
			return RenderingServer::mesh_subdivision_surface_is_valid(candidate, 0, 2, -1, Vector<uint8_t>());
		};
		auto displaced = [](uint32_t p_shape, const LocalVector<uint32_t> &p_heights, bool p_with_heights) {
			LocalVector<SubdivisionSection> sections = subdivision_quad_sections(true, 2);
			for (SubdivisionSection &section : sections) {
				if (section.id == RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO) {
					section.words[RenderingServerTypes::MESH_SUBDIVISION_INFO_DISPLACEMENT_SHAPE] = p_shape;
				}
			}
			if (p_with_heights) {
				SubdivisionSection heights;
				heights.id = RenderingServerTypes::MESH_SUBDIVISION_SECTION_DISPLACEMENT_HEIGHTS;
				heights.words = p_heights;
				sections.push_back(heights);
			}
			return subdivision_payload_from(sections, RenderingServerTypes::MESH_SUBDIVISION_FLAG_OWNER, 0);
		};
		const uint32_t half = 0x3f000000;
		const LocalVector<uint32_t> four = { 0, half, 0, half };
		CHECK(valid_two(displaced(1, four, true)));
		CHECK_FALSE(valid_two(displaced(0, four, true))); // A shape the cage's deltas move.
		CHECK_FALSE(valid_two(displaced(none, four, true))); // Heights without a shape.
		CHECK_FALSE(valid_two(displaced(1, four, false))); // A shape without heights.
		CHECK_FALSE(valid_two(displaced(2, four, true))); // A shape past the mesh's.
		CHECK_FALSE(valid_two(displaced(1, { 0, half, 0 }, true))); // Short.
		CHECK_FALSE(valid_two(displaced(1, { 0, half, 0x7fc00000, half }, true))); // A NaN.
		CHECK_FALSE(valid_two(displaced(1, { 0, 0x7f800000, 0, half }, true))); // An infinity.
	}
	CHECK_FALSE(valid(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_U_WELDED, { 0, 1, 2 }))); // Short.

	// The .scn path: a surface Dictionary set back through _surfaces is checked by the storage.
	Ref<ArrayMesh> source;
	source.instantiate();
	source->add_blend_shape("A");
	add_subdivision_quad(source, surface, good);
	REQUIRE(rendering_surface_count(source) == 1);
	const Array surfaces = source->get("_surfaces");
	auto loads = [&surfaces](const Vector<uint8_t> &p_payload) {
		Dictionary d = Dictionary(surfaces[0]).duplicate();
		d["subdivision_data"] = p_payload;
		Array changed;
		changed.push_back(d);
		Ref<ArrayMesh> loaded;
		loaded.instantiate();
		loaded->set("_blend_shape_names", PackedStringArray{ "A" });
		loaded->set("_surfaces", changed);
		return rendering_surface_count(loaded);
	};
	CHECK(loads(good) == 1);
	ERR_PRINT_OFF;
	CHECK(loads(set_word(good, 0, RenderingServerTypes::MESH_SUBDIVISION_DATA_VERSION + 1)) == 0);
	CHECK(loads(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, { 0, 1, 2, 9 })) == 0);
	CHECK(loads(with_section(RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_INDICES, { 0, 1, 2, 70000 })) == 0);
	CHECK(loads(set_word(good, 4 + 3, RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO)) == 0);
	ERR_PRINT_ON;
}

TEST_CASE("[SceneTree][ArrayMesh] Skin weights are quantized to nearest with their sum kept (D122).") {
	const auto sum_of = [](const uint16_t *p_q, uint32_t p_count) {
		int64_t sum = 0;
		for (uint32_t i = 0; i < p_count; i++) {
			sum += p_q[i];
		}
		return sum;
	};
	uint16_t q[8];
	// Weights that truncation loses about one unit each on: the rounded sum is still 65535.
	const float spread[8] = { 0.3f, 0.21f, 0.17f, 0.13f, 0.09f, 0.05f, 0.03f, 0.02f };
	RS::mesh_quantize_skin_weights(spread, 8, q);
	CHECK(sum_of(q, 8) == 65535);
	for (uint32_t i = 1; i < 8; i++) {
		CHECK(q[i] == uint16_t(Math::round(spread[i] * 65535.0f))); // Only the largest moves.
	}
	CHECK(Math::abs(int(q[0]) - int(Math::round(spread[0] * 65535.0f))) <= 4);
	// Seven sevenths round to 65534 together: the remainder goes on the first of the equal largest.
	const double sevenths[8] = { 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 0 };
	RS::mesh_quantize_skin_weights(sevenths, 8, q);
	CHECK(q[0] == 9363);
	for (uint32_t i = 1; i < 7; i++) {
		CHECK(q[i] == 9362);
	}
	CHECK(q[7] == 0);
	// Weights that do not sum to 1 keep their own rounded sum; a NaN or a negative weight is 0.
	const float doubled[4] = { 1.0f, 1.0f, 0.0f, 0.0f };
	RS::mesh_quantize_skin_weights(doubled, 4, q);
	CHECK((q[0] == 65535 && q[1] == 65535 && q[2] == 0 && q[3] == 0));
	const float invalid[4] = { Math::NaN, -0.5f, 0.0f, 0.0f };
	RS::mesh_quantize_skin_weights(invalid, 4, q);
	CHECK(sum_of(q, 4) == 0);
	const float zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	RS::mesh_quantize_skin_weights(zero, 4, q);
	CHECK(sum_of(q, 4) == 0);
	// A weight above 1 counts as 1, before the sum.
	const float over[4] = { 1.5f, 0.25f, 0.0f, 0.0f };
	RS::mesh_quantize_skin_weights(over, 4, q);
	CHECK((q[0] == 65535 && q[1] == 16384 && q[2] == 0 && q[3] == 0));
	// The same values as float and as double pack the same.
	int float_double_differ = 0;
	for (int v = 0; v < 512; v++) {
		float as_float[4];
		double as_double[4];
		float total = 0.0f;
		for (int b = 0; b < 4; b++) {
			as_float[b] = 0.5f + float((v * 31 + b * 17) % 97);
			total += as_float[b];
		}
		for (int b = 0; b < 4; b++) {
			as_float[b] /= total;
			as_double[b] = as_float[b];
		}
		uint16_t from_double[4];
		RS::mesh_quantize_skin_weights(as_float, 4, q);
		RS::mesh_quantize_skin_weights(as_double, 4, from_double);
		float_double_differ += memcmp(q, from_double, sizeof(from_double)) != 0 ? 1 : 0;
	}
	CHECK(float_double_differ == 0);

	// Through surface data: every vertex's weights read back summing to 1 within float rounding.
	Vector<Vector3> positions;
	Vector<int> bones;
	Vector<float> weights;
	for (int v = 0; v < 64; v++) {
		positions.push_back(Vector3(v, 0, v % 3));
		float raw[8];
		float total = 0;
		for (int b = 0; b < 8; b++) {
			raw[b] = 1.0f + float((v * 13 + b * 7) % 11);
			total += raw[b];
			bones.push_back(b);
		}
		for (int b = 0; b < 8; b++) {
			weights.push_back(raw[b] / total);
		}
	}
	Array arrays;
	arrays.resize(RSE::ARRAY_MAX);
	arrays[RSE::ARRAY_VERTEX] = positions;
	arrays[RSE::ARRAY_BONES] = bones;
	arrays[RSE::ARRAY_WEIGHTS] = weights;
	RenderingServerTypes::SurfaceData surface;
	REQUIRE(RS::get_singleton()->mesh_create_surface_data_from_arrays(&surface, RSE::PRIMITIVE_POINTS, arrays, Array(), Dictionary(), RSE::ARRAY_FLAG_USE_8_BONE_WEIGHTS) == OK);
	const Array read = RS::get_singleton()->mesh_create_arrays_from_surface_data(surface);
	const PackedFloat32Array read_weights = read[RSE::ARRAY_WEIGHTS];
	REQUIRE(read_weights.size() == weights.size());
	int not_affine = 0;
	for (int v = 0; v < 64; v++) {
		double sum = 0;
		for (int b = 0; b < 8; b++) {
			sum += read_weights[v * 8 + b];
			CHECK(Math::abs(read_weights[v * 8 + b] - weights[v * 8 + b]) <= 4.5f / 65535.0f);
		}
		not_affine += Math::abs(sum - 1.0) > 1e-6 ? 1 : 0;
	}
	CHECK(not_affine == 0);
	// Decoded weights packed again give the same bytes: meshes edited through their arrays
	// (MeshDataTool, SurfaceTool::create_from) keep their skin exactly.
	Array repacked_arrays = read;
	RenderingServerTypes::SurfaceData repacked;
	REQUIRE(RS::get_singleton()->mesh_create_surface_data_from_arrays(&repacked, RSE::PRIMITIVE_POINTS, repacked_arrays, Array(), Dictionary(), RSE::ARRAY_FLAG_USE_8_BONE_WEIGHTS) == OK);
	CHECK(repacked.skin_data == surface.skin_data);
}

} // namespace TestArrayMesh

#endif // _3D_DISABLED
