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
	ProjectSettings::get_singleton()->set_setting("rendering/pathtracing/cluster_blas", setting_before);

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

} // namespace TestArrayMesh

#endif // _3D_DISABLED
