/**************************************************************************/
/*  test_surface_tool.cpp                                                 */
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

TEST_FORCE_LINK(test_surface_tool)

#include "scene/resources/surface_tool.h"

namespace TestSurfaceTool {

TEST_CASE("[SurfaceTool] index() merges vertices that compare equal, whatever the sign of a zero.") {
	Ref<SurfaceTool> st;
	st.instantiate();
	st->begin(Mesh::PRIMITIVE_TRIANGLES);
	// Two triangles over the same three corners, the second's normals with a negative zero: the
	// vertices compare equal, so they must also hash equal for index() to merge them.
	const Vector3 corners[3] = { Vector3(0, 0, 0), Vector3(0, 0, 1), Vector3(1, 0, 0) };
	for (int i = 0; i < 3; i++) {
		st->set_normal(Vector3(0, 1, 0));
		st->add_vertex(corners[i]);
	}
	for (int i = 0; i < 3; i++) {
		st->set_normal(Vector3(-0.0, 1, 0));
		st->add_vertex(corners[i]);
	}
	st->index();
	const Array arrays = st->commit_to_arrays();
	CHECK(PackedVector3Array(arrays[Mesh::ARRAY_VERTEX]).size() == 3);
	CHECK(PackedInt32Array(arrays[Mesh::ARRAY_INDEX]).size() == 6);
}

} // namespace TestSurfaceTool
