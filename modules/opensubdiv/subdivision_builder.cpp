/**************************************************************************/
/*  subdivision_builder.cpp                                               */
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

#include "subdivision_builder.h"

#include "core/crypto/crypto_core.h"
#include "core/object/class_db.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "servers/rendering/rendering_server.h"

#include <thirdparty/misc/mikktspace.h>

#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/stencilTable.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/far/topologyDescriptor.h>

namespace Far = OpenSubdiv::Far;
namespace Sdc = OpenSubdiv::Sdc;

namespace {

// Face-varying values are refined in double, as the S0 parity tool refines them (D122).
struct FVarValue {
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
	double w = 0.0;
	void Clear() { x = y = z = w = 0.0; }
	void AddWithWeight(const FVarValue &p_src, float p_weight) {
		x += p_src.x * p_weight;
		y += p_src.y * p_weight;
		z += p_src.z * p_weight;
		w += p_src.w * p_weight;
	}
};

void refine_face_varying(const Far::TopologyRefiner &p_refiner, int p_channel, int p_level, LocalVector<FVarValue> &r_values) {
	Far::PrimvarRefiner primvar(p_refiner);
	for (int level = 1; level <= p_level; level++) {
		LocalVector<FVarValue> next;
		next.resize(p_refiner.GetLevel(level).GetNumFVarValues(p_channel));
		const FVarValue *src = r_values.ptr();
		FVarValue *dst = next.ptr();
		primvar.InterpolateFaceVarying(level, src, dst, p_channel);
		r_values = next;
	}
}

// Blender's polygon normal of a quad (normal_quad_v3: the cross product of its diagonals).
Vector3 quad_normal(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, const Vector3 &p_d) {
	return (p_a - p_c).cross(p_b - p_d).normalized();
}

// The angle between a corner's two edges, the weight Blender gives a face's normal at a vertex.
float corner_angle(const Vector3 &p_prev, const Vector3 &p_corner, const Vector3 &p_next) {
	const Vector3 to_prev = (p_prev - p_corner).normalized();
	const Vector3 to_next = (p_next - p_corner).normalized();
	return Math::acos(CLAMP(to_prev.dot(to_next), -1.0f, 1.0f));
}

class PayloadWriter {
	struct Section {
		uint32_t id = 0;
		Vector<uint8_t> bytes;
	};
	LocalVector<Section> sections;

public:
	void add(uint32_t p_id, const void *p_data, size_t p_bytes) {
		Section section;
		section.id = p_id;
		section.bytes.resize(p_bytes);
		if (p_bytes) {
			memcpy(section.bytes.ptrw(), p_data, p_bytes);
		}
		sections.push_back(section);
	}

	template <typename T>
	void add(uint32_t p_id, const LocalVector<T> &p_values) {
		add(p_id, p_values.ptr(), p_values.size() * sizeof(T));
	}

	Vector<uint8_t> finish(uint32_t p_flags, uint32_t p_owner_surface) const {
		const uint64_t align = RenderingServerTypes::MESH_SUBDIVISION_SECTION_ALIGN;
		uint64_t offset = (RenderingServerTypes::MESH_SUBDIVISION_HEADER_WORDS + uint64_t(sections.size()) * RenderingServerTypes::MESH_SUBDIVISION_SECTION_WORDS) * sizeof(uint32_t);
		LocalVector<uint32_t> table;
		table.push_back(RenderingServerTypes::MESH_SUBDIVISION_DATA_VERSION);
		table.push_back(p_flags);
		table.push_back(p_owner_surface);
		table.push_back(sections.size());
		LocalVector<uint64_t> offsets;
		for (const Section &section : sections) {
			offset = (offset + align - 1) / align * align;
			offsets.push_back(offset);
			table.push_back(section.id);
			table.push_back(uint32_t(offset));
			table.push_back(uint32_t(section.bytes.size()));
			offset += section.bytes.size();
		}
		Vector<uint8_t> data;
		ERR_FAIL_COND_V_MSG(offset > UINT32_MAX, data, "Subdivision payload over 4 GB.");
		data.resize(offset);
		uint8_t *w = data.ptrw();
		memset(w, 0, offset);
		memcpy(w, table.ptr(), table.size() * sizeof(uint32_t));
		for (uint32_t i = 0; i < sections.size(); i++) {
			if (sections[i].bytes.size()) {
				memcpy(w + offsets[i], sections[i].bytes.ptr(), sections[i].bytes.size());
			}
		}
		return data;
	}
};

// A render vertex: its refined vertex and its value in every face-varying layer and the colours.
struct CornerKey {
	int32_t u = 0;
	int32_t value[SubdivisionBuilder::MAX_FACE_VARYING_LAYERS + 1] = {};

	bool operator==(const CornerKey &p_other) const {
		if (u != p_other.u) {
			return false;
		}
		for (int i = 0; i <= SubdivisionBuilder::MAX_FACE_VARYING_LAYERS; i++) {
			if (value[i] != p_other.value[i]) {
				return false;
			}
		}
		return true;
	}
};

struct CornerKeyHasher {
	static uint32_t hash(const CornerKey &p_key) {
		uint32_t h = hash_murmur3_one_32(uint32_t(p_key.u));
		for (int i = 0; i <= SubdivisionBuilder::MAX_FACE_VARYING_LAYERS; i++) {
			h = hash_murmur3_one_32(uint32_t(p_key.value[i]), h);
		}
		return hash_fmix32(h);
	}
};

// A render vertex and one MikkTSpace frame of its corners (tangent and sign, as the float bits MikkTSpace
// gives them): one group of it.
struct GroupKey {
	int32_t render = 0;
	float frame[4] = {};
};

struct GroupKeyHasher {
	static uint32_t hash(const GroupKey &p_key) {
		return hash_murmur3_buffer(&p_key, sizeof(GroupKey));
	}
};

struct GroupKeyComparator {
	static bool compare(const GroupKey &p_a, const GroupKey &p_b) {
		return memcmp(&p_a, &p_b, sizeof(GroupKey)) == 0;
	}
};

// MikkTSpace's callbacks over quads given per corner (SubdivisionBuilder::mikktspace_quad_frames).
struct QuadFrames {
	const LocalVector<Vector3> *positions = nullptr;
	const LocalVector<Vector3> *normals = nullptr;
	const LocalVector<Vector2> *uvs = nullptr;
	LocalVector<Vector4> *frames = nullptr;
};

const QuadFrames &quad_frames(const SMikkTSpaceContext *p_context) {
	return *static_cast<const QuadFrames *>(p_context->m_pUserData);
}

int quad_frames_faces(const SMikkTSpaceContext *p_context) {
	return quad_frames(p_context).positions->size() / 4;
}

int quad_frames_vertices(const SMikkTSpaceContext *p_context, const int p_face) {
	return 4;
}

void quad_frames_position(const SMikkTSpaceContext *p_context, float r_out[], const int p_face, const int p_vertex) {
	const Vector3 &v = (*quad_frames(p_context).positions)[4 * p_face + p_vertex];
	r_out[0] = v.x;
	r_out[1] = v.y;
	r_out[2] = v.z;
}

void quad_frames_normal(const SMikkTSpaceContext *p_context, float r_out[], const int p_face, const int p_vertex) {
	const Vector3 &v = (*quad_frames(p_context).normals)[4 * p_face + p_vertex];
	r_out[0] = v.x;
	r_out[1] = v.y;
	r_out[2] = v.z;
}

void quad_frames_uv(const SMikkTSpaceContext *p_context, float r_out[], const int p_face, const int p_vertex) {
	const Vector2 &v = (*quad_frames(p_context).uvs)[4 * p_face + p_vertex];
	r_out[0] = v.x;
	r_out[1] = v.y;
}

// The sign as SurfaceTool::mikktSetTSpaceDefault sets it: +1 for an orientation-preserving group.
void quad_frames_set(const SMikkTSpaceContext *p_context, const float p_tangent[], const float p_sign, const int p_face, const int p_vertex) {
	(*quad_frames(p_context).frames)[4 * p_face + p_vertex] = Vector4(p_tangent[0], p_tangent[1], p_tangent[2], p_sign < 0.0f ? -1.0f : 1.0f);
}

double ms_since(uint64_t p_usec) {
	return double(OS::get_singleton()->get_ticks_usec() - p_usec) / 1000.0;
}

// D122 S6: an 8-bit image as Cycles' CPU kernel samples it (intern/cycles/kernel/device/cpu/image.h, Blender
// 5.2.1), in float. Its stored values over 255, a grey image's in every colour channel; rows counted from the
// bottom, as Blender stores an image, where Godot's first row is the top.
struct CyclesImage {
	const uint8_t *data = nullptr;
	int width = 0;
	int height = 0;
	int channels = 0;

	void texel(int p_x, int p_y_from_bottom, float r_rgb[3]) const {
		const uint8_t *pixel = data + (int64_t(height - 1 - p_y_from_bottom) * width + p_x) * channels;
		const float f = 1.0f / 255.0f;
		for (int a = 0; a < 3; a++) {
			r_rgb[a] = pixel[channels >= 3 ? a : 0] * f;
		}
	}
};

bool cycles_image_of(const Ref<Image> &p_image, CyclesImage &r_image) {
	if (p_image.is_null() || p_image->is_empty() || p_image->is_compressed()) {
		return false;
	}
	switch (p_image->get_format()) {
		case Image::FORMAT_L8:
			r_image.channels = 1;
			break;
		case Image::FORMAT_RGB8:
			r_image.channels = 3;
			break;
		case Image::FORMAT_RGBA8:
			r_image.channels = 4;
			break;
		default:
			return false;
	}
	r_image.data = p_image->ptr();
	r_image.width = p_image->get_width();
	r_image.height = p_image->get_height();
	if (r_image.channels == 4) {
		// Cycles associates a straight alpha with the colour on load (ImageMetaData::conform_pixels), which the
		// lookup here does not model: an RGBA map must be opaque.
		const int64_t pixels = int64_t(r_image.width) * r_image.height;
		for (int64_t i = 0; i < pixels; i++) {
			if (r_image.data[i * 4 + 3] != 255) {
				return false;
			}
		}
	}
	return true;
}

// Cycles' frac(): the integer part rounds down, from a truncation.
float cycles_frac(float p_x, int *r_i) {
	const int i = int(p_x) - ((p_x < 0.0f) ? 1 : 0);
	*r_i = i;
	return p_x - float(i);
}

int cycles_wrap_periodic(int p_x, int p_width) {
	p_x %= p_width;
	return p_x < 0 ? p_x + p_width : p_x;
}

// interp_linear() with EXTENSION_REPEAT: texel centres at +0.5, the four texels weighted in its order.
void cycles_linear(const CyclesImage &p_image, float p_u, float p_v, float r_rgb[3]) {
	const float x = p_u * float(p_image.width);
	const float y = p_v * float(p_image.height);
	int ix, iy;
	const float tx = cycles_frac(x - 0.5f, &ix);
	const float ty = cycles_frac(y - 0.5f, &iy);
	ix = cycles_wrap_periodic(ix, p_image.width);
	const int nix = cycles_wrap_periodic(ix + 1, p_image.width);
	iy = cycles_wrap_periodic(iy, p_image.height);
	const int niy = cycles_wrap_periodic(iy + 1, p_image.height);
	float t00[3], t10[3], t01[3], t11[3];
	p_image.texel(ix, iy, t00);
	p_image.texel(nix, iy, t10);
	p_image.texel(ix, niy, t01);
	p_image.texel(nix, niy, t11);
	const float w00 = (1.0f - ty) * (1.0f - tx);
	const float w10 = (1.0f - ty) * tx;
	const float w01 = ty * (1.0f - tx);
	const float w11 = ty * tx;
	for (int a = 0; a < 3; a++) {
		r_rgb[a] = w00 * t00[a] + w10 * t10[a] + w01 * t01[a] + w11 * t11[a];
	}
}

// color_srgb_to_linear() (intern/cycles/util/color.h), the exact curve.
float cycles_srgb_to_linear(float p_c) {
	if (p_c < 0.04045f) {
		return (p_c < 0.0f) ? 0.0f : p_c * (1.0f / 12.92f);
	}
	return powf((p_c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

// An sRGB image's colour at a UV, decoded, reduced to its luminance as RGB to BW and a colour-to-float
// conversion do (linear_rgb_to_gray(): the Y row of the scene linear space, Linear Rec.709 in Blender's default
// configuration).
float cycles_luma(const CyclesImage &p_image, float p_u, float p_v) {
	float rgb[3];
	cycles_linear(p_image, p_u, p_v, rgb);
	return cycles_srgb_to_linear(rgb[0]) * 0.2126390059f + cycles_srgb_to_linear(rgb[1]) * 0.7151686788f + cycles_srgb_to_linear(rgb[2]) * 0.0721923154f;
}

// The source's displacement chain at a UV on maps cycles_image_of() accepted (SubdivisionBuilder::displacement_height()).
float cycles_displacement_height(const CyclesImage &p_height_map, const CyclesImage *p_mask, float p_u, float p_v, float p_min_height, float p_max_height) {
	if (!Math::is_finite(p_u) || !Math::is_finite(p_v)) {
		return NAN;
	}
	// The DAZ Displacement group's chain: Math (max - min), Math.001 (luma times it), Math.002 (plus min); the
	// Displacement node's midlevel 0 and scale 1 leave it as is.
	const float range = p_max_height - p_min_height;
	float height = cycles_luma(p_height_map, p_u, p_v) * range + p_min_height;
	if (p_mask) {
		height *= cycles_luma(*p_mask, p_u, p_v);
	}
	return height;
}

} // namespace

// A skin slot without weight: its bone is not compared and is written as bind index 0.
constexpr uint16_t SKIN_NO_BONE = 0xFFFF;

struct SubdivisionBuilder::Work {
	// The welded cage, compacted to the vertices the faces reference.
	LocalVector<int32_t> compact_of_welded;
	LocalVector<int32_t> welded_of_compact;
	LocalVector<Vector3> positions;
	bool has_skin = false;
	// Per compact vertex, 8 bones then 8 unorm weights. With named bones a bone is an index into
	// bone_names, the union of every owner's joint names; without (one owner), its bind index. A slot
	// of weight 0 holds SKIN_NO_BONE.
	LocalVector<uint16_t> skin;
	bool named_bones = false;
	Vector<String> bone_names;
	HashMap<String, int> bone_index;

	// The blend shapes of every owner together, by name, as their non-zero deltas per compact vertex.
	Vector<String> shape_names;
	HashMap<String, int> shape_index;
	LocalVector<HashMap<uint32_t, Vector3>> shape_deltas;
	HashMap<int, Vector<String>> owner_shapes;
	LocalVector<int> owners;

	// Coarse faces, every surface's in add order; corners flattened, without a triangle's fourth.
	LocalVector<int> face_size;
	LocalVector<int> face_surface;
	LocalVector<int> face_local;
	LocalVector<int> corners;
	LocalVector<int> corner_weld[MAX_FACE_VARYING_LAYERS];
	bool has_color = false;
	LocalVector<int> corner_color;
	LocalVector<FVarValue> color_values;

	// The refined level, welded.
	int level = 2;
	int u_count = 0;
	LocalVector<Vector3> u_positions;
	LocalVector<Vector3> u_normals;
	LocalVector<uint32_t> u_owner_mask;
	LocalVector<uint32_t> stencil_offsets;
	LocalVector<uint32_t> stencil_indices; // Compact cage vertices.
	LocalVector<float> stencil_weights;
	int quad_count = 0;
	LocalVector<int> quads; // 4 U per refined quad, in the cage face's corner order.
	LocalVector<int> quad_face; // Its coarse face.
	LocalVector<int> quad_surface;
	LocalVector<int> quad_fvar[MAX_FACE_VARYING_LAYERS]; // 4 values per refined quad.
	LocalVector<FVarValue> fvar_values[MAX_FACE_VARYING_LAYERS];
	LocalVector<int> quad_color;
	LocalVector<FVarValue> refined_colors;

	// One owner's share of the refinement, one per entry of owners, in its order.
	struct Part {
		int owner = 0;
		LocalVector<int> own_quads; // Welded quads of its own surfaces.
		LocalVector<int> ghost_quads; // Other owners' welded quads that touch one of its refined vertices.
		LocalVector<int32_t> local_u; // Per welded refined vertex, its local index, or -1.
		LocalVector<int32_t> welded_u; // Per local refined vertex: own, then ghost, each in welded order.
		uint32_t own_u_count = 0;
		LocalVector<int32_t> local_cage; // Per compact cage vertex, its local index, or -1.
		LocalVector<int32_t> cage_compact; // Per local cage vertex: own, then ghost, each in welded order.
		uint32_t own_cage_count = 0;
		// Per local cage vertex, its bones with a non-zero weight in the owner's bind indices, SKIN_NO_BONE
		// in the other slots; the surfaces' bone bounds read it.
		LocalVector<uint16_t> cage_binds;
		uint64_t blend_entries = 0;
		PayloadWriter owner_payload;
		// Its displacement (S6): the shape's name, and a height per own refined vertex.
		bool displaced = false;
		String displacement_shape;
		LocalVector<float> heights;
		float height_bound = 0.0f; // The largest |height|, which its bounds grow by.
		Dictionary displacement_report;
	};
	LocalVector<Part> parts;

	double gather_ms = 0.0;
	double refine_ms = 0.0;
	double stencil_ms = 0.0;
	double normal_ms = 0.0;
	Dictionary owner_reports;
};

void SubdivisionBuilder::set_cage_vertex_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);
	cage_vertex_count = p_count;
	owner_meshes.clear();
	owner_triangle_sources.clear();
}

int SubdivisionBuilder::get_cage_vertex_count() const {
	return cage_vertex_count;
}

void SubdivisionBuilder::set_blend_shape_mode(Mesh::BlendShapeMode p_mode) {
	blend_shape_mode = p_mode;
	owner_meshes.clear();
	owner_triangle_sources.clear();
}

Mesh::BlendShapeMode SubdivisionBuilder::get_blend_shape_mode() const {
	return blend_shape_mode;
}

int SubdivisionBuilder::add_face_varying_layer(const PackedVector2Array &p_values, Mesh::ArrayType p_array_slot) {
	ERR_FAIL_COND_V_MSG(layers.size() >= MAX_FACE_VARYING_LAYERS, -1, vformat("At most %d face-varying layers.", MAX_FACE_VARYING_LAYERS));
	ERR_FAIL_COND_V_MSG(p_array_slot != Mesh::ARRAY_TEX_UV && p_array_slot != Mesh::ARRAY_TEX_UV2 && p_array_slot != Mesh::ARRAY_CUSTOM0, -1, "A face-varying layer becomes ARRAY_TEX_UV, ARRAY_TEX_UV2 or ARRAY_CUSTOM0.");
	for (const FaceVaryingLayer &layer : layers) {
		ERR_FAIL_COND_V_MSG(layer.slot == p_array_slot, -1, "Two face-varying layers in one array slot.");
	}
	FaceVaryingLayer layer;
	layer.values = p_values;
	layer.slot = p_array_slot;
	layers.push_back(layer);
	owner_meshes.clear();
	owner_triangle_sources.clear();
	return layers.size() - 1;
}

int SubdivisionBuilder::add_surface(int p_owner, const Array &p_arrays, const Array &p_blend_arrays, const PackedStringArray &p_blend_shape_names, const PackedInt32Array &p_cage_vertex, const PackedInt32Array &p_faces, const Array &p_uv_weld, const Ref<Material> &p_material) {
	ERR_FAIL_INDEX_V_MSG(p_owner, MAX_OWNERS, -1, vformat("Owners are 0 to %d.", MAX_OWNERS - 1));
	ERR_FAIL_COND_V_MSG(p_arrays.size() != Mesh::ARRAY_MAX, -1, "The arrays must be a Mesh's ARRAY_MAX arrays.");
	SourceSurface surface;
	surface.owner = p_owner;
	surface.arrays = p_arrays;
	surface.blend_arrays = p_blend_arrays;
	for (const String &name : p_blend_shape_names) {
		surface.blend_shape_names.push_back(name);
	}
	surface.cage_vertex = p_cage_vertex;
	surface.faces = p_faces;
	for (int i = 0; i < p_uv_weld.size(); i++) {
		surface.uv_weld.push_back(PackedInt32Array(p_uv_weld[i]));
	}
	surface.material = p_material;
	surfaces.push_back(surface);
	owner_meshes.clear();
	owner_triangle_sources.clear();
	return surfaces.size() - 1;
}

void SubdivisionBuilder::set_owner_bone_names(int p_owner, const PackedStringArray &p_bone_names) {
	ERR_FAIL_INDEX_MSG(p_owner, MAX_OWNERS, vformat("Owners are 0 to %d.", MAX_OWNERS - 1));
	Vector<String> names;
	for (const String &name : p_bone_names) {
		names.push_back(name);
	}
	owner_bone_names[p_owner] = names;
	owner_meshes.clear();
	owner_triangle_sources.clear();
}

PackedStringArray SubdivisionBuilder::get_owner_bone_names(int p_owner) const {
	PackedStringArray names;
	const Vector<String> *found = owner_bone_names.getptr(p_owner);
	if (found) {
		for (const String &name : *found) {
			names.push_back(name);
		}
	}
	return names;
}

Error SubdivisionBuilder::set_owner_displacement(int p_owner, const PackedInt32Array &p_surfaces, const Array &p_source_faces, Mesh::ArrayType p_uv_slot, const Ref<Image> &p_height_map, const Ref<Image> &p_mask, float p_min_height, float p_max_height, const String &p_shape_name) {
	ERR_FAIL_INDEX_V_MSG(p_owner, MAX_OWNERS, ERR_INVALID_PARAMETER, vformat("Owners are 0 to %d.", MAX_OWNERS - 1));
	ERR_FAIL_COND_V_MSG(p_surfaces.is_empty(), ERR_INVALID_PARAMETER, "A displacement needs at least one surface.");
	ERR_FAIL_COND_V_MSG(!p_source_faces.is_empty() && p_source_faces.size() != p_surfaces.size(), ERR_INVALID_PARAMETER, "The source faces are one array per displacing surface, or none.");
	ERR_FAIL_COND_V_MSG(p_uv_slot != Mesh::ARRAY_TEX_UV && p_uv_slot != Mesh::ARRAY_TEX_UV2 && p_uv_slot != Mesh::ARRAY_CUSTOM0, ERR_INVALID_PARAMETER, "The maps read a face-varying layer: ARRAY_TEX_UV, ARRAY_TEX_UV2 or ARRAY_CUSTOM0.");
	CyclesImage image;
	ERR_FAIL_COND_V_MSG(!cycles_image_of(p_height_map, image), ERR_INVALID_PARAMETER, "The height map must be an uncompressed 8-bit L, RGB or opaque RGBA image.");
	ERR_FAIL_COND_V_MSG(p_mask.is_valid() && !cycles_image_of(p_mask, image), ERR_INVALID_PARAMETER, "The mask must be an uncompressed 8-bit L, RGB or opaque RGBA image.");
	ERR_FAIL_COND_V_MSG(!Math::is_finite(p_min_height) || !Math::is_finite(p_max_height), ERR_INVALID_PARAMETER, "The height range must be finite.");
	ERR_FAIL_COND_V_MSG(p_shape_name.is_empty(), ERR_INVALID_PARAMETER, "The displacement shape needs a name.");
	OwnerDisplacement displacement;
	for (int surface : p_surfaces) {
		ERR_FAIL_COND_V_MSG(displacement.surfaces.has(surface), ERR_INVALID_PARAMETER, vformat("Surface %d is named twice.", surface));
		displacement.surfaces.push_back(surface);
	}
	for (int i = 0; i < p_source_faces.size(); i++) {
		ERR_FAIL_COND_V_MSG(p_source_faces[i].get_type() != Variant::PACKED_INT32_ARRAY, ERR_INVALID_PARAMETER, "Each surface's source faces are a PackedInt32Array.");
		const PackedInt32Array faces = p_source_faces[i];
		Vector<int32_t> source;
		for (int32_t face : faces) {
			ERR_FAIL_COND_V_MSG(face < 0, ERR_INVALID_PARAMETER, "A source face index is negative.");
			source.push_back(face);
		}
		displacement.source_faces.push_back(source);
	}
	displacement.uv_slot = p_uv_slot;
	displacement.height_map = p_height_map;
	displacement.mask = p_mask;
	displacement.min_height = p_min_height;
	displacement.max_height = p_max_height;
	displacement.shape_name = p_shape_name;
	owner_displacements[p_owner] = displacement;
	owner_meshes.clear();
	owner_triangle_sources.clear();
	return OK;
}

void SubdivisionBuilder::clear_owner_displacement(int p_owner) {
	owner_displacements.erase(p_owner);
	owner_meshes.clear();
	owner_triangle_sources.clear();
}

float SubdivisionBuilder::displacement_height(const Ref<Image> &p_height_map, const Ref<Image> &p_mask, const Vector2 &p_uv, float p_min_height, float p_max_height) {
	if (!Math::is_finite(p_uv.x) || !Math::is_finite(p_uv.y)) {
		return NAN;
	}
	CyclesImage height_map;
	ERR_FAIL_COND_V_MSG(!cycles_image_of(p_height_map, height_map), NAN, "The height map must be an uncompressed 8-bit L, RGB or opaque RGBA image.");
	CyclesImage mask;
	ERR_FAIL_COND_V_MSG(p_mask.is_valid() && !cycles_image_of(p_mask, mask), NAN, "The mask must be an uncompressed 8-bit L, RGB or opaque RGBA image.");
	return cycles_displacement_height(height_map, p_mask.is_valid() ? &mask : nullptr, float(p_uv.x), float(p_uv.y), p_min_height, p_max_height);
}

Ref<ArrayMesh> SubdivisionBuilder::get_owner_mesh(int p_owner) const {
	const Ref<ArrayMesh> *mesh = owner_meshes.getptr(p_owner);
	ERR_FAIL_NULL_V_MSG(mesh, Ref<ArrayMesh>(), vformat("No mesh for owner %d: build() first.", p_owner));
	return *mesh;
}

Array SubdivisionBuilder::get_owner_triangle_sources(int p_owner) const {
	const Vector<TriangleSources> *sources = owner_triangle_sources.getptr(p_owner);
	ERR_FAIL_NULL_V_MSG(sources, Array(), vformat("No triangle sources for owner %d: build() first.", p_owner));
	Array result;
	for (const TriangleSources &source : *sources) {
		Dictionary entry;
		entry["source_surface"] = source.source_surface;
		entry["source_faces"] = source.source_faces;
		result.push_back(entry);
	}
	return result;
}

Dictionary SubdivisionBuilder::get_report() const {
	return report;
}

void SubdivisionBuilder::clear() {
	cage_vertex_count = 0;
	blend_shape_mode = Mesh::BLEND_SHAPE_MODE_NORMALIZED;
	layers.clear();
	surfaces.clear();
	owner_bone_names.clear();
	owner_displacements.clear();
	owner_meshes.clear();
	owner_triangle_sources.clear();
	report = Dictionary();
}

// Validates the input and gathers the welded cage: positions, skin and blend shape deltas per cage
// vertex, which every glTF vertex of one cage vertex must agree on bit for bit.
Error SubdivisionBuilder::_gather(Work &r_work) {
	const uint64_t begin = OS::get_singleton()->get_ticks_usec();
	ERR_FAIL_COND_V_MSG(cage_vertex_count <= 0, ERR_INVALID_PARAMETER, "set_cage_vertex_count() first.");
	ERR_FAIL_COND_V_MSG(surfaces.is_empty(), ERR_INVALID_PARAMETER, "No surface added.");

	// The faces and their corners, and the compact cage.
	r_work.compact_of_welded.resize(cage_vertex_count);
	for (int32_t &index : r_work.compact_of_welded) {
		index = -1;
	}
	for (uint32_t layer = 0; layer < layers.size(); layer++) {
		ERR_FAIL_COND_V_MSG(layers[layer].values.is_empty(), ERR_INVALID_PARAMETER, vformat("Face-varying layer %d has no values.", layer));
	}
	for (uint32_t s = 0; s < surfaces.size(); s++) {
		const SourceSurface &surface = surfaces[s];
		const PackedVector3Array vertices = surface.arrays[Mesh::ARRAY_VERTEX];
		const int vertex_count = vertices.size();
		ERR_FAIL_COND_V_MSG(vertex_count == 0, ERR_INVALID_PARAMETER, vformat("Surface %d has no vertices.", s));
		ERR_FAIL_COND_V_MSG(surface.cage_vertex.size() != vertex_count, ERR_INVALID_PARAMETER, vformat("Surface %d: %d cage vertex entries for %d vertices.", s, surface.cage_vertex.size(), vertex_count));
		ERR_FAIL_COND_V_MSG(surface.faces.is_empty() || surface.faces.size() % 4 != 0, ERR_INVALID_PARAMETER, vformat("Surface %d: faces must be 4 indices per face, and at least one face.", s));
		ERR_FAIL_COND_V_MSG(surface.uv_weld.size() != int(layers.size()), ERR_INVALID_PARAMETER, vformat("Surface %d: %d UV weld arrays for %d face-varying layers.", s, surface.uv_weld.size(), layers.size()));
		ERR_FAIL_COND_V_MSG(surface.blend_arrays.size() != surface.blend_shape_names.size(), ERR_INVALID_PARAMETER, vformat("Surface %d: %d blend arrays for %d names.", s, surface.blend_arrays.size(), surface.blend_shape_names.size()));
		for (int a = 0; a < Mesh::ARRAY_MAX; a++) {
			if (surface.arrays[a].get_type() == Variant::NIL || a == Mesh::ARRAY_VERTEX || a == Mesh::ARRAY_NORMAL || a == Mesh::ARRAY_TANGENT || a == Mesh::ARRAY_BONES || a == Mesh::ARRAY_WEIGHTS || a == Mesh::ARRAY_INDEX || a == Mesh::ARRAY_COLOR) {
				continue;
			}
			// A UV or custom array survives only as a face-varying layer in its slot.
			bool layered = false;
			for (const FaceVaryingLayer &layer : layers) {
				layered = layered || int(layer.slot) == a;
			}
			ERR_FAIL_COND_V_MSG(!layered, ERR_INVALID_PARAMETER, vformat("Surface %d has array %d, which no face-varying layer replaces; it would be lost.", s, a));
		}
		for (int l = 0; l < surface.uv_weld.size(); l++) {
			ERR_FAIL_COND_V_MSG(surface.uv_weld[l].size() != surface.faces.size(), ERR_INVALID_PARAMETER, vformat("Surface %d: UV weld %d has %d entries for %d face corners.", s, l, surface.uv_weld[l].size(), surface.faces.size()));
		}
		const int32_t *cage = surface.cage_vertex.ptr();
		for (int v = 0; v < vertex_count; v++) {
			ERR_FAIL_INDEX_V_MSG(cage[v], cage_vertex_count, ERR_INVALID_PARAMETER, vformat("Surface %d vertex %d: cage vertex %d out of range.", s, v, cage[v]));
		}
		const int32_t *faces = surface.faces.ptr();
		for (int f = 0; f < surface.faces.size() / 4; f++) {
			ERR_FAIL_COND_V_MSG(faces[4 * f + 3] < -1, ERR_INVALID_PARAMETER, vformat("Surface %d face %d: a triangle's fourth index must be -1.", s, f));
			const int size = faces[4 * f + 3] == -1 ? 3 : 4;
			for (int c = 0; c < size; c++) {
				const int32_t vertex = faces[4 * f + c];
				ERR_FAIL_INDEX_V_MSG(vertex, vertex_count, ERR_INVALID_PARAMETER, vformat("Surface %d face %d: vertex %d out of range.", s, f, vertex));
				const int32_t welded = cage[vertex];
				for (int d = 0; d < c; d++) {
					ERR_FAIL_COND_V_MSG(cage[faces[4 * f + d]] == welded, ERR_INVALID_PARAMETER, vformat("Surface %d face %d repeats cage vertex %d.", s, f, welded));
				}
				if (r_work.compact_of_welded[welded] < 0) {
					r_work.compact_of_welded[welded] = r_work.welded_of_compact.size();
					r_work.welded_of_compact.push_back(welded);
				}
				r_work.corners.push_back(r_work.compact_of_welded[welded]);
				for (uint32_t l = 0; l < layers.size(); l++) {
					const int32_t value = surface.uv_weld[l][4 * f + c];
					ERR_FAIL_INDEX_V_MSG(value, layers[l].values.size(), ERR_INVALID_PARAMETER, vformat("Surface %d face %d corner %d: UV weld group %d of layer %d out of range.", s, f, c, value, l));
					r_work.corner_weld[l].push_back(value);
				}
			}
			for (uint32_t l = 0; l < layers.size() && size == 3; l++) {
				ERR_FAIL_COND_V_MSG(surface.uv_weld[l][4 * f + 3] != -1, ERR_INVALID_PARAMETER, vformat("Surface %d face %d: a triangle's fourth weld entry must be -1.", s, f));
			}
			r_work.face_size.push_back(size);
			r_work.face_surface.push_back(s);
			r_work.face_local.push_back(f);
		}
	}
	// The faces' vertices in welded order, so that the compact cage does not depend on face order.
	{
		LocalVector<int32_t> sorted(r_work.welded_of_compact);
		sorted.sort();
		LocalVector<int32_t> remap;
		remap.resize(sorted.size());
		for (uint32_t i = 0; i < sorted.size(); i++) {
			remap[r_work.compact_of_welded[sorted[i]]] = i;
			r_work.compact_of_welded[sorted[i]] = i;
		}
		r_work.welded_of_compact = sorted;
		for (int &corner : r_work.corners) {
			corner = remap[corner];
		}
	}
	// Owners and their blend shape names, the same on every surface of an owner.
	for (uint32_t s = 0; s < surfaces.size(); s++) {
		const SourceSurface &surface = surfaces[s];
		Vector<String> *names = r_work.owner_shapes.getptr(surface.owner);
		if (names == nullptr) {
			r_work.owner_shapes.insert(surface.owner, surface.blend_shape_names);
			r_work.owners.push_back(surface.owner);
		} else {
			ERR_FAIL_COND_V_MSG(!(*names == surface.blend_shape_names), ERR_INVALID_PARAMETER, vformat("Surface %d's blend shapes differ from its owner %d's other surfaces'.", s, surface.owner));
		}
		for (const String &name : surface.blend_shape_names) {
			if (!r_work.shape_index.has(name)) {
				r_work.shape_index.insert(name, r_work.shape_names.size());
				r_work.shape_names.push_back(name);
			}
		}
	}
	r_work.owners.sort();
	r_work.shape_deltas.resize(r_work.shape_names.size());

	// Positions and skin per glTF vertex, gathered per cage vertex.
	for (const SourceSurface &surface : surfaces) {
		r_work.has_skin = r_work.has_skin || surface.arrays[Mesh::ARRAY_BONES].get_type() != Variant::NIL;
		r_work.has_color = r_work.has_color || surface.arrays[Mesh::ARRAY_COLOR].get_type() != Variant::NIL;
	}
	const Error skin_error = _gather_skin(r_work);
	if (skin_error != OK) {
		return skin_error;
	}
	uint32_t mismatches = 0;

	// Blend shape deltas: the target minus the base, non-zero ones kept per cage vertex. Every glTF
	// vertex of a cage vertex that carries a shape must carry the same delta.
	for (uint32_t s = 0; s < surfaces.size(); s++) {
		const SourceSurface &surface = surfaces[s];
		const PackedVector3Array vertices = surface.arrays[Mesh::ARRAY_VERTEX];
		for (int b = 0; b < surface.blend_shape_names.size(); b++) {
			const Array blend = surface.blend_arrays[b];
			ERR_FAIL_COND_V_MSG(blend.size() != Mesh::ARRAY_MAX, ERR_INVALID_PARAMETER, vformat("Surface %d blend shape %d is not ARRAY_MAX arrays.", s, b));
			const PackedVector3Array targets = blend[Mesh::ARRAY_VERTEX];
			ERR_FAIL_COND_V_MSG(targets.size() != vertices.size(), ERR_INVALID_PARAMETER, vformat("Surface %d blend shape %s has %d positions for %d vertices.", s, surface.blend_shape_names[b], targets.size(), vertices.size()));
			HashMap<uint32_t, Vector3> &deltas = r_work.shape_deltas[r_work.shape_index[surface.blend_shape_names[b]]];
			for (int v = 0; v < vertices.size(); v++) {
				const int32_t compact = r_work.compact_of_welded[surface.cage_vertex[v]];
				const Vector3 delta = targets[v] - vertices[v];
				if (compact < 0 || delta == Vector3()) {
					continue;
				}
				const Vector3 *existing = deltas.getptr(compact);
				if (existing == nullptr) {
					deltas.insert(compact, delta);
				} else if (memcmp(existing, &delta, sizeof(Vector3)) != 0) {
					mismatches++;
				}
			}
		}
	}
	// A zero delta where another vertex of the same cage vertex moves.
	for (uint32_t s = 0; s < surfaces.size(); s++) {
		const SourceSurface &surface = surfaces[s];
		const PackedVector3Array vertices = surface.arrays[Mesh::ARRAY_VERTEX];
		for (int b = 0; b < surface.blend_shape_names.size(); b++) {
			const PackedVector3Array targets = Array(surface.blend_arrays[b])[Mesh::ARRAY_VERTEX];
			const HashMap<uint32_t, Vector3> &deltas = r_work.shape_deltas[r_work.shape_index[surface.blend_shape_names[b]]];
			for (int v = 0; v < vertices.size(); v++) {
				const int32_t compact = r_work.compact_of_welded[surface.cage_vertex[v]];
				if (compact >= 0 && targets[v] == vertices[v] && deltas.has(compact)) {
					mismatches++;
				}
			}
		}
	}
	ERR_FAIL_COND_V_MSG(mismatches, ERR_INVALID_DATA, vformat("%d blend shape deltas disagree with another vertex of their cage vertex.", mismatches));

	// Colours as a face-varying channel with one value per glTF vertex (a surface without colours
	// is white), refined bilinearly as Blender interpolates point and corner attributes.
	if (r_work.has_color) {
		for (const SourceSurface &surface : surfaces) {
			const int base = r_work.color_values.size();
			const PackedVector3Array vertices = surface.arrays[Mesh::ARRAY_VERTEX];
			const PackedColorArray colors = surface.arrays[Mesh::ARRAY_COLOR];
			ERR_FAIL_COND_V_MSG(!colors.is_empty() && colors.size() != vertices.size(), ERR_INVALID_PARAMETER, "A colour array's size differs from its vertex count.");
			for (int v = 0; v < vertices.size(); v++) {
				const Color color = colors.is_empty() ? Color(1, 1, 1, 1) : colors[v];
				FVarValue value;
				value.x = color.r;
				value.y = color.g;
				value.z = color.b;
				value.w = color.a;
				r_work.color_values.push_back(value);
			}
			const int32_t *faces = surface.faces.ptr();
			for (int f = 0; f < surface.faces.size() / 4; f++) {
				const int size = faces[4 * f + 3] == -1 ? 3 : 4;
				for (int c = 0; c < size; c++) {
					r_work.corner_color.push_back(base + faces[4 * f + c]);
				}
			}
		}
	}
	r_work.gather_ms = ms_since(begin);
	return OK;
}

// Positions and skin per cage vertex, from every glTF vertex of it, which must agree bit for bit.
// With several owners the skin is compared and kept by joint name, so that a ghost's bones can be
// written in its receiving owner's bind indices.
Error SubdivisionBuilder::_gather_skin(Work &r_work) {
	const uint32_t compact_count = r_work.welded_of_compact.size();
	const uint32_t bones_max = RenderingServerTypes::MESH_SUBDIVISION_CAGE_BONES;
	r_work.positions.resize(compact_count);
	LocalVector<uint8_t> seen;
	seen.resize(compact_count);
	memset(seen.ptr(), 0, compact_count);
	if (r_work.has_skin) {
		uint32_t named = 0;
		for (int owner : r_work.owners) {
			named += owner_bone_names.has(owner) ? 1 : 0;
		}
		ERR_FAIL_COND_V_MSG(named != 0 && named != r_work.owners.size(), ERR_INVALID_PARAMETER, "Either every owner of a skinned cage names its bones (set_owner_bone_names()) or none does.");
		ERR_FAIL_COND_V_MSG(named == 0 && r_work.owners.size() > 1, ERR_INVALID_PARAMETER, "A skinned cage of several owners needs every owner's bone names (set_owner_bone_names()), so that a ghost's bones are written in its receiving owner's bind indices.");
		r_work.named_bones = named != 0;
		for (int owner : r_work.owners) {
			if (!r_work.named_bones) {
				break;
			}
			// A name twice in one owner's list would map both bind indices to one union entry, and the
			// owner's own skin would be rewritten to the first: joint names must be unique per owner.
			HashSet<String> owner_names;
			for (const String &name : owner_bone_names[owner]) {
				ERR_FAIL_COND_V_MSG(owner_names.has(name), ERR_INVALID_PARAMETER, vformat("Owner %d names bone %s twice; set_owner_bone_names() needs unique joint names.", owner, name));
				owner_names.insert(name);
				if (!r_work.bone_index.has(name)) {
					r_work.bone_index.insert(name, r_work.bone_names.size());
					r_work.bone_names.push_back(name);
				}
			}
		}
		ERR_FAIL_COND_V_MSG(r_work.bone_names.size() >= SKIN_NO_BONE, ERR_INVALID_PARAMETER, vformat("%d bone names, at most %d.", r_work.bone_names.size(), SKIN_NO_BONE - 1));
		r_work.skin.resize(compact_count * 2 * bones_max);
	}
	uint32_t mismatches = 0;
	String first_mismatch;
	for (uint32_t s = 0; s < surfaces.size(); s++) {
		const SourceSurface &surface = surfaces[s];
		const PackedVector3Array vertices = surface.arrays[Mesh::ARRAY_VERTEX];
		const PackedInt32Array bones = surface.arrays[Mesh::ARRAY_BONES];
		const PackedFloat32Array weights = surface.arrays[Mesh::ARRAY_WEIGHTS];
		const int vertex_count = vertices.size();
		const Vector<String> *bone_names = r_work.named_bones ? owner_bone_names.getptr(surface.owner) : nullptr;
		int bones_per_vertex = 0;
		if (r_work.has_skin) {
			ERR_FAIL_COND_V_MSG(bones.is_empty() || weights.size() != bones.size() || bones.size() % vertex_count != 0, ERR_INVALID_PARAMETER, vformat("Surface %d: every surface of a skinned cage needs bones and weights per vertex.", s));
			bones_per_vertex = bones.size() / vertex_count;
			ERR_FAIL_COND_V_MSG(bones_per_vertex != 4 && bones_per_vertex != 8, ERR_INVALID_PARAMETER, vformat("Surface %d: %d bones per vertex, expected 4 or 8.", s, bones_per_vertex));
		}
		for (int v = 0; v < vertex_count; v++) {
			const int32_t compact = r_work.compact_of_welded[surface.cage_vertex[v]];
			if (compact < 0) {
				continue; // A vertex no face of the cage reaches.
			}
			uint16_t skin[2 * RenderingServerTypes::MESH_SUBDIVISION_CAGE_BONES];
			for (uint32_t b = 0; b < bones_max; b++) {
				skin[b] = SKIN_NO_BONE;
				skin[bones_max + b] = 0;
			}
			uint16_t quantized[8];
			if (bones_per_vertex > 0) {
				RenderingServer::mesh_quantize_skin_weights(weights.ptr() + v * bones_per_vertex, bones_per_vertex, quantized); // As the skin buffer stores them.
			}
			for (int b = 0; b < bones_per_vertex; b++) {
				const uint16_t weight = quantized[b];
				if (weight == 0) {
					continue;
				}
				const int32_t bind = bones[v * bones_per_vertex + b];
				if (bone_names) {
					ERR_FAIL_INDEX_V_MSG(bind, bone_names->size(), ERR_INVALID_PARAMETER, vformat("Surface %d vertex %d: bone %d past its owner's %d bone names.", s, v, bind, bone_names->size()));
					skin[b] = uint16_t(r_work.bone_index[(*bone_names)[bind]]);
				} else {
					ERR_FAIL_INDEX_V_MSG(bind, SKIN_NO_BONE, ERR_INVALID_PARAMETER, vformat("Surface %d vertex %d: bone %d out of range.", s, v, bind));
					skin[b] = uint16_t(bind);
				}
				skin[bones_max + b] = weight;
			}
			if (!seen[compact]) {
				seen[compact] = 1;
				r_work.positions[compact] = vertices[v];
				if (r_work.has_skin) {
					memcpy(&r_work.skin[compact * 2 * bones_max], skin, sizeof(skin));
				}
				continue;
			}
			const bool same_position = memcmp(&r_work.positions[compact], &vertices[v], sizeof(Vector3)) == 0;
			const bool same_skin = !r_work.has_skin || memcmp(&r_work.skin[compact * 2 * bones_max], skin, sizeof(skin)) == 0;
			if (!same_position || !same_skin) {
				if (mismatches == 0) {
					first_mismatch = vformat("surface %d vertex %d, cage vertex %d (%s)", s, v, surface.cage_vertex[v], same_position ? "skin" : "position");
				}
				mismatches++;
			}
		}
	}
	ERR_FAIL_COND_V_MSG(mismatches, ERR_INVALID_DATA, vformat("%d glTF vertices disagree with another vertex of their cage vertex, first %s.", mismatches, first_mismatch));
	for (uint32_t c = 0; c < compact_count; c++) {
		ERR_FAIL_COND_V_MSG(!seen[c], ERR_INVALID_DATA, vformat("Cage vertex %d has no glTF vertex.", r_work.welded_of_compact[c]));
	}
	return OK;
}

// Refines the welded cage: positions through vertex stencils, the face-varying layers in double,
// the colours bilinearly, and every refined vertex's normal as Blender computes it.
Error SubdivisionBuilder::_refine(Work &r_work, int p_level) {
	uint64_t begin = OS::get_singleton()->get_ticks_usec();
	r_work.level = p_level;
	using Descriptor = Far::TopologyDescriptor;
	Descriptor::FVarChannel channels[MAX_FACE_VARYING_LAYERS];
	for (uint32_t l = 0; l < layers.size(); l++) {
		channels[l].numValues = layers[l].values.size();
		channels[l].valueIndices = r_work.corner_weld[l].ptr();
	}
	Descriptor descriptor;
	descriptor.numVertices = r_work.welded_of_compact.size();
	descriptor.numFaces = r_work.face_size.size();
	descriptor.numVertsPerFace = r_work.face_size.ptr();
	descriptor.vertIndicesPerFace = r_work.corners.ptr();
	descriptor.numFVarChannels = layers.size();
	descriptor.fvarChannels = channels;

	// Blender's settings (S0): VTX_BOUNDARY_EDGE_ONLY for boundary_smooth ALL, FVAR_LINEAR_BOUNDARIES
	// for uv_smooth PRESERVE_BOUNDARIES, no creases, Catmull-Clark on triangles.
	Sdc::Options options;
	options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);
	options.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_BOUNDARIES);
	options.SetCreasingMethod(Sdc::Options::CREASE_UNIFORM);
	options.SetTriangleSubdivision(Sdc::Options::TRI_SUB_CATMARK);
	Far::TopologyRefiner *refiner = Far::TopologyRefinerFactory<Descriptor>::Create(descriptor, Far::TopologyRefinerFactory<Descriptor>::Options(Sdc::SCHEME_CATMARK, options));
	ERR_FAIL_NULL_V_MSG(refiner, ERR_INVALID_DATA, "OpenSubdiv rejected the cage's topology (see its errors above).");
	Far::TopologyRefiner::UniformOptions uniform(p_level);
	uniform.fullTopologyInLastLevel = true;
	refiner->RefineUniform(uniform);
	const Far::TopologyLevel &last = refiner->GetLevel(p_level);
	r_work.u_count = last.GetNumVertices();
	r_work.quad_count = last.GetNumFaces();

	// The refined quads, their coarse face's surface, and their face-varying values.
	r_work.quads.resize(4 * r_work.quad_count);
	r_work.quad_face.resize(r_work.quad_count);
	r_work.quad_surface.resize(r_work.quad_count);
	for (uint32_t l = 0; l < layers.size(); l++) {
		r_work.quad_fvar[l].resize(4 * r_work.quad_count);
	}
	for (int q = 0; q < r_work.quad_count; q++) {
		const Far::ConstIndexArray vertices = last.GetFaceVertices(q);
		if (vertices.size() != 4) {
			delete refiner;
			ERR_FAIL_V_MSG(ERR_BUG, "A refined face is not a quad.");
		}
		int parent = q;
		for (int level = p_level; level > 0; level--) {
			parent = refiner->GetLevel(level).GetFaceParentFace(parent);
		}
		r_work.quad_face[q] = parent;
		r_work.quad_surface[q] = r_work.face_surface[parent];
		for (int c = 0; c < 4; c++) {
			r_work.quads[4 * q + c] = vertices[c];
		}
		for (uint32_t l = 0; l < layers.size(); l++) {
			const Far::ConstIndexArray values = last.GetFaceFVarValues(q, l);
			for (int c = 0; c < 4; c++) {
				r_work.quad_fvar[l][4 * q + c] = values[c];
			}
		}
	}
	for (uint32_t l = 0; l < layers.size(); l++) {
		r_work.fvar_values[l].resize(layers[l].values.size());
		for (int i = 0; i < layers[l].values.size(); i++) {
			r_work.fvar_values[l][i].x = layers[l].values[i].x;
			r_work.fvar_values[l][i].y = layers[l].values[i].y;
		}
		refine_face_varying(*refiner, l, p_level, r_work.fvar_values[l]);
	}
	r_work.refine_ms = ms_since(begin);

	// Vertex stencils: each refined vertex as a weighted sum of cage vertices (the runtime path).
	begin = OS::get_singleton()->get_ticks_usec();
	Far::StencilTableFactory::Options stencil_options;
	stencil_options.generateIntermediateLevels = false;
	stencil_options.generateOffsets = true;
	stencil_options.interpolationMode = Far::StencilTableFactory::INTERPOLATE_VERTEX;
	stencil_options.maxLevel = p_level;
	const Far::StencilTable *table = Far::StencilTableFactory::Create(*refiner, stencil_options);
	if (table == nullptr || table->GetNumStencils() != r_work.u_count) {
		delete table;
		delete refiner;
		ERR_FAIL_V_MSG(ERR_BUG, "The stencil table does not cover the refined level.");
	}
	r_work.stencil_offsets.resize(r_work.u_count + 1);
	r_work.stencil_indices.resize(table->GetControlIndices().size());
	r_work.stencil_weights.resize(table->GetWeights().size());
	r_work.u_positions.resize(r_work.u_count);
	for (int u = 0; u < r_work.u_count; u++) {
		const int offset = table->GetOffsets()[u];
		const int size = table->GetSizes()[u];
		r_work.stencil_offsets[u] = offset;
		double sum[3] = {};
		for (int i = offset; i < offset + size; i++) {
			const int index = table->GetControlIndices()[i];
			const float weight = table->GetWeights()[i];
			r_work.stencil_indices[i] = index;
			r_work.stencil_weights[i] = weight;
			for (int a = 0; a < 3; a++) {
				sum[a] += double(weight) * double(r_work.positions[index][a]);
			}
		}
		r_work.u_positions[u] = Vector3(sum[0], sum[1], sum[2]);
	}
	r_work.stencil_offsets[r_work.u_count] = r_work.stencil_indices.size();
	delete table;
	r_work.stencil_ms = ms_since(begin);

	// Colours: a bilinear refinement of the same topology, whose refined faces are the same faces.
	if (r_work.has_color) {
		Descriptor::FVarChannel color_channel;
		color_channel.numValues = r_work.color_values.size();
		color_channel.valueIndices = r_work.corner_color.ptr();
		Descriptor color_descriptor = descriptor;
		color_descriptor.numFVarChannels = 1;
		color_descriptor.fvarChannels = &color_channel;
		Sdc::Options linear;
		linear.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_ALL);
		Far::TopologyRefiner *bilinear = Far::TopologyRefinerFactory<Descriptor>::Create(color_descriptor, Far::TopologyRefinerFactory<Descriptor>::Options(Sdc::SCHEME_BILINEAR, linear));
		if (bilinear == nullptr) {
			delete refiner;
			ERR_FAIL_V_MSG(ERR_INVALID_DATA, "OpenSubdiv rejected the cage's topology for the colours.");
		}
		bilinear->RefineUniform(uniform);
		const Far::TopologyLevel &linear_last = bilinear->GetLevel(p_level);
		bool same = linear_last.GetNumFaces() == r_work.quad_count;
		r_work.quad_color.resize(4 * r_work.quad_count);
		for (int q = 0; q < r_work.quad_count && same; q++) {
			const Far::ConstIndexArray vertices = linear_last.GetFaceVertices(q);
			const Far::ConstIndexArray values = linear_last.GetFaceFVarValues(q, 0);
			same = vertices.size() == 4;
			for (int c = 0; c < 4 && same; c++) {
				same = vertices[c] == r_work.quads[4 * q + c];
				r_work.quad_color[4 * q + c] = values[c];
			}
		}
		r_work.refined_colors = r_work.color_values;
		refine_face_varying(*bilinear, 0, p_level, r_work.refined_colors);
		delete bilinear;
		if (!same) {
			delete refiner;
			ERR_FAIL_V_MSG(ERR_BUG, "The bilinear refinement's faces differ from Catmull-Clark's.");
		}
	}
	delete refiner;

	// Which owners' quads reach each refined vertex.
	r_work.u_owner_mask.resize(r_work.u_count);
	memset(r_work.u_owner_mask.ptr(), 0, r_work.u_count * sizeof(uint32_t));
	for (int q = 0; q < r_work.quad_count; q++) {
		const uint32_t bit = 1u << surfaces[r_work.quad_surface[q]].owner;
		for (int c = 0; c < 4; c++) {
			r_work.u_owner_mask[r_work.quads[4 * q + c]] |= bit;
		}
	}

	// Normals as Blender computes them on the refined mesh: each quad's normal weighted by its
	// corner angle, summed in quad order, so every owner sharing a vertex gets the same bits.
	begin = OS::get_singleton()->get_ticks_usec();
	r_work.u_normals.resize(r_work.u_count);
	for (Vector3 &normal : r_work.u_normals) {
		normal = Vector3();
	}
	for (int q = 0; q < r_work.quad_count; q++) {
		const int *quad = &r_work.quads[4 * q];
		const Vector3 normal = quad_normal(r_work.u_positions[quad[0]], r_work.u_positions[quad[1]], r_work.u_positions[quad[2]], r_work.u_positions[quad[3]]);
		for (int c = 0; c < 4; c++) {
			const Vector3 &prev = r_work.u_positions[quad[(c + 3) % 4]];
			const Vector3 &next = r_work.u_positions[quad[(c + 1) % 4]];
			r_work.u_normals[quad[c]] += normal * corner_angle(prev, r_work.u_positions[quad[c]], next);
		}
	}
	for (Vector3 &normal : r_work.u_normals) {
		normal = normal.is_zero_approx() ? Vector3(0, 1, 0) : normal.normalized();
	}
	r_work.normal_ms = ms_since(begin);
	return OK;
}

// An owner's partition: its own quads and the ghost quads (another owner's quads that touch one of
// its refined vertices, which its seam normals sum), its refined vertices (own, then the ghost
// quads' others), and its cage (the vertices of its own coarse faces, then the rest of its stencils'
// support), each part in welded order.
Error SubdivisionBuilder::_partition_owner(Work &r_work, uint32_t p_part) {
	Work::Part &part = r_work.parts[p_part];
	const uint32_t own_bit = 1u << part.owner;

	for (int q = 0; q < r_work.quad_count; q++) {
		if (surfaces[r_work.quad_surface[q]].owner == part.owner) {
			part.own_quads.push_back(q);
			continue;
		}
		for (int c = 0; c < 4; c++) {
			if (r_work.u_owner_mask[r_work.quads[4 * q + c]] & own_bit) {
				part.ghost_quads.push_back(q);
				break;
			}
		}
	}
	ERR_FAIL_COND_V_MSG(part.own_quads.is_empty(), ERR_INVALID_DATA, vformat("Owner %d has no refined faces.", part.owner));

	part.local_u.resize(r_work.u_count);
	for (int32_t &index : part.local_u) {
		index = -1;
	}
	for (int u = 0; u < r_work.u_count; u++) {
		if (r_work.u_owner_mask[u] & own_bit) {
			part.local_u[u] = part.welded_u.size();
			part.welded_u.push_back(u);
		}
	}
	part.own_u_count = part.welded_u.size();
	{
		LocalVector<int32_t> ghost_u;
		for (int q : part.ghost_quads) {
			for (int c = 0; c < 4; c++) {
				const int u = r_work.quads[4 * q + c];
				if (part.local_u[u] < 0) {
					part.local_u[u] = -2;
					ghost_u.push_back(u);
				}
			}
		}
		ghost_u.sort();
		for (int32_t u : ghost_u) {
			part.local_u[u] = part.welded_u.size();
			part.welded_u.push_back(u);
		}
	}

	const uint32_t compact_count = r_work.welded_of_compact.size();
	part.local_cage.resize(compact_count);
	for (int32_t &index : part.local_cage) {
		index = -1;
	}
	{
		uint32_t corner = 0;
		for (uint32_t f = 0; f < r_work.face_size.size(); f++) {
			const bool own = surfaces[r_work.face_surface[f]].owner == part.owner;
			for (int c = 0; c < r_work.face_size[f]; c++, corner++) {
				if (own) {
					part.local_cage[r_work.corners[corner]] = 0;
				}
			}
		}
	}
	for (uint32_t c = 0; c < compact_count; c++) {
		if (part.local_cage[c] == 0) {
			part.local_cage[c] = part.cage_compact.size();
			part.cage_compact.push_back(c);
		}
	}
	part.own_cage_count = part.cage_compact.size();
	LocalVector<uint8_t> ghost;
	ghost.resize(compact_count);
	memset(ghost.ptr(), 0, compact_count);
	for (int32_t u : part.welded_u) {
		for (uint32_t i = r_work.stencil_offsets[u]; i < r_work.stencil_offsets[u + 1]; i++) {
			const uint32_t c = r_work.stencil_indices[i];
			if (part.local_cage[c] < 0) {
				ghost[c] = 1;
			}
		}
	}
	for (uint32_t c = 0; c < compact_count; c++) {
		if (ghost[c]) {
			part.local_cage[c] = part.cage_compact.size();
			part.cage_compact.push_back(c);
		}
	}
	return OK;
}

// The rules between owners. A shape that moves an owner's cage or ghosts must exist on that owner,
// or its ghosts would not follow their originals. And on a cage vertex two owners hold, the shapes
// that move it must come in the same relative order in both owners' lists: a runtime pass sums the
// deltas in its mesh's order, and only the same order gives the same bits on both sides of a seam.
// Zero deltas add exactly, so shapes that do not move the vertex may come in any order.
Error SubdivisionBuilder::_check_owners(Work &r_work) {
	const uint32_t compact_count = r_work.welded_of_compact.size();
	const uint32_t part_count = r_work.parts.size();
	LocalVector<uint32_t> cage_parts; // Per compact cage vertex, a bit per part that holds it.
	cage_parts.resize(compact_count);
	memset(cage_parts.ptr(), 0, compact_count * sizeof(uint32_t));
	for (uint32_t p = 0; p < part_count; p++) {
		for (int32_t c : r_work.parts[p].cage_compact) {
			cage_parts[c] |= 1u << p;
		}
	}
	// Each shape's position in each part's list, -1 where the part lacks it.
	LocalVector<LocalVector<int32_t>> position;
	position.resize(part_count);
	for (uint32_t p = 0; p < part_count; p++) {
		position[p].resize(r_work.shape_names.size());
		for (int32_t &index : position[p]) {
			index = -1;
		}
		const Vector<String> &names = r_work.owner_shapes[r_work.parts[p].owner];
		for (int n = 0; n < names.size(); n++) {
			position[p][r_work.shape_index[names[n]]] = n;
		}
	}
	// The shapes that move each vertex held by two parts or more, in shape order.
	HashMap<uint32_t, LocalVector<int>> shared_moves;
	for (int shape = 0; shape < r_work.shape_names.size(); shape++) {
		uint32_t missing_parts = 0;
		for (const KeyValue<uint32_t, Vector3> &E : r_work.shape_deltas[shape]) {
			const uint32_t holders = cage_parts[E.key];
			for (uint32_t p = 0; p < part_count; p++) {
				if ((holders & (1u << p)) && position[p][shape] < 0) {
					missing_parts |= 1u << p;
				}
			}
			if (holders & (holders - 1)) {
				if (!shared_moves.has(E.key)) {
					shared_moves.insert(E.key, LocalVector<int>());
				}
				shared_moves[E.key].push_back(shape);
			}
		}
		for (uint32_t p = 0; p < part_count; p++) {
			ERR_FAIL_COND_V_MSG(missing_parts & (1u << p), ERR_INVALID_DATA, vformat("Owner %d lacks blend shape %s, which moves its cage or its ghosts.", r_work.parts[p].owner, r_work.shape_names[shape]));
		}
	}
	LocalVector<Vector3i> order; // Per shape: its position in part a's list, in part b's, its index.
	for (const KeyValue<uint32_t, LocalVector<int>> &E : shared_moves) {
		const uint32_t holders = cage_parts[E.key];
		for (uint32_t a = 0; a < part_count; a++) {
			for (uint32_t b = a + 1; b < part_count; b++) {
				if (!(holders & (1u << a)) || !(holders & (1u << b))) {
					continue;
				}
				order.clear();
				for (int shape : E.value) {
					order.push_back(Vector3i(position[a][shape], position[b][shape], shape));
				}
				order.sort();
				for (uint32_t i = 1; i < order.size(); i++) {
					ERR_FAIL_COND_V_MSG(order[i - 1].y > order[i].y, ERR_INVALID_DATA, vformat("Owners %d and %d order blend shapes %s and %s differently, and both move cage vertex %d, which both hold: the seam would differ in its last bits.", r_work.parts[a].owner, r_work.parts[b].owner, r_work.shape_names[order[i - 1].z], r_work.shape_names[order[i].z], r_work.welded_of_compact[E.key]));
				}
			}
		}
	}
	return OK;
}

// An owner's displacement (S6): each of its refined vertices takes its height where Cycles evaluates it, once
// per vertex at the first triangle that reaches it (scene/mesh_displace.cpp): the first face of a displacing
// surface in the source's face order, at that face's corner value. A refined vertex inside one coarse face has
// one value whichever of its refined faces reaches it, so the source's order is needed only between coarse
// faces. A vertex another owner also refines must stay at 0: both owners write it, from their own amounts.
Error SubdivisionBuilder::_owner_displacement(Work &r_work, uint32_t p_part) {
	Work::Part &part = r_work.parts[p_part];
	const OwnerDisplacement *displacement = owner_displacements.getptr(part.owner);
	if (!displacement) {
		return OK;
	}
	const uint64_t begin = OS::get_singleton()->get_ticks_usec();
	int layer = -1;
	for (uint32_t l = 0; l < layers.size(); l++) {
		if (layers[l].slot == displacement->uv_slot) {
			layer = l;
		}
	}
	ERR_FAIL_COND_V_MSG(layer < 0, ERR_INVALID_PARAMETER, vformat("Owner %d's displacement reads array slot %d, which no face-varying layer fills.", part.owner, displacement->uv_slot));
	ERR_FAIL_COND_V_MSG(r_work.shape_index.has(displacement->shape_name), ERR_INVALID_PARAMETER, vformat("Owner %d's displacement shape is named %s, which an owner's blend shape already uses.", part.owner, displacement->shape_name));
	CyclesImage height_map;
	CyclesImage mask;
	ERR_FAIL_COND_V_MSG(!cycles_image_of(displacement->height_map, height_map), ERR_INVALID_PARAMETER, vformat("Owner %d's height map is not an uncompressed 8-bit L, RGB or opaque RGBA image.", part.owner));
	ERR_FAIL_COND_V_MSG(displacement->mask.is_valid() && !cycles_image_of(displacement->mask, mask), ERR_INVALID_PARAMETER, vformat("Owner %d's mask is not an uncompressed 8-bit L, RGB or opaque RGBA image.", part.owner));
	const CyclesImage *mask_image = displacement->mask.is_valid() ? &mask : nullptr;

	// Each coarse face's place in the source's order, -1 for a face that does not displace.
	LocalVector<int64_t> face_key;
	face_key.resize(r_work.face_size.size());
	for (int64_t &key : face_key) {
		key = -1;
	}
	{
		HashSet<int32_t> source_keys;
		LocalVector<uint32_t> first_face; // Per surface, its first coarse face.
		first_face.resize(surfaces.size() + 1);
		first_face[0] = 0;
		for (uint32_t s = 0; s < surfaces.size(); s++) {
			first_face[s + 1] = first_face[s] + surfaces[s].faces.size() / 4;
		}
		for (int i = 0; i < displacement->surfaces.size(); i++) {
			const int s = displacement->surfaces[i];
			ERR_FAIL_INDEX_V_MSG(s, int(surfaces.size()), ERR_INVALID_PARAMETER, vformat("Owner %d's displacement names surface %d, which does not exist.", part.owner, s));
			ERR_FAIL_COND_V_MSG(surfaces[s].owner != part.owner, ERR_INVALID_PARAMETER, vformat("Owner %d's displacement names surface %d, of owner %d.", part.owner, s, surfaces[s].owner));
			const uint32_t face_count = first_face[s + 1] - first_face[s];
			if (displacement->source_faces.is_empty()) {
				for (uint32_t f = 0; f < face_count; f++) {
					face_key[first_face[s] + f] = first_face[s] + f;
				}
				continue;
			}
			const Vector<int32_t> &source = displacement->source_faces[i];
			ERR_FAIL_COND_V_MSG(uint32_t(source.size()) != face_count, ERR_INVALID_PARAMETER, vformat("Owner %d's displacement: surface %d has %d faces and %d source faces.", part.owner, s, face_count, source.size()));
			for (uint32_t f = 0; f < face_count; f++) {
				// Two faces at one place in the source's order would leave the first face to add order.
				ERR_FAIL_COND_V_MSG(source_keys.has(source[f]), ERR_INVALID_PARAMETER, vformat("Owner %d's displacement names source face %d twice (surface %d face %d).", part.owner, source[f], s, f));
				source_keys.insert(source[f]);
				face_key[first_face[s] + f] = source[f];
			}
		}
	}

	// Per refined vertex, its first displacing face's key and its corner value there. Faces of one coarse face
	// share a key and agree on the value; a disagreement would make the order within one matter, and is counted.
	LocalVector<int64_t> best_key;
	LocalVector<int32_t> best_value;
	best_key.resize(r_work.u_count);
	best_value.resize(r_work.u_count);
	for (int64_t &key : best_key) {
		key = INT64_MAX;
	}
	uint32_t value_conflicts = 0;
	for (int q = 0; q < r_work.quad_count; q++) {
		const int64_t key = face_key[r_work.quad_face[q]];
		if (key < 0) {
			continue;
		}
		for (int c = 0; c < 4; c++) {
			const int u = r_work.quads[4 * q + c];
			const int32_t value = r_work.quad_fvar[layer][4 * q + c];
			if (key < best_key[u]) {
				best_key[u] = key;
				best_value[u] = value;
			} else if (key == best_key[u] && value != best_value[u]) {
				value_conflicts++;
			}
		}
	}
	ERR_FAIL_COND_V_MSG(value_conflicts > 0, ERR_BUG, vformat("Owner %d's displacement: %d refined corners of one coarse face disagree on their value.", part.owner, value_conflicts));

	part.heights.resize(part.own_u_count);
	// The value each height was sampled at, per own U, NaN where no displacing face reaches it: the reference
	// check compares it with Blender's first face's corner value, which decides the face chosen.
	PackedVector2Array uvs;
	uvs.resize(part.own_u_count);
	Vector2 *uv_write = uvs.ptrw();
	uint32_t displaced = 0;
	uint32_t seam_reached = 0; // Seam vertices a displacing face reaches; each must have height 0.
	float lowest = 0.0f;
	float highest = 0.0f;
	for (uint32_t i = 0; i < part.own_u_count; i++) {
		const int u = part.welded_u[i];
		float height = 0.0f;
		if (best_key[u] != INT64_MAX) {
			// The value as the refined mesh stores it, in float, v up as Blender's.
			const FVarValue &value = r_work.fvar_values[layer][best_value[u]];
			uv_write[i] = Vector2(float(value.x), float(value.y));
			height = cycles_displacement_height(height_map, mask_image, float(value.x), float(value.y), displacement->min_height, displacement->max_height);
			ERR_FAIL_COND_V_MSG(!Math::is_finite(height), ERR_INVALID_DATA, vformat("Owner %d's displacement height at refined vertex %d is not finite.", part.owner, u));
			const uint32_t owners = r_work.u_owner_mask[u];
			seam_reached += (owners & (owners - 1)) ? 1 : 0;
			ERR_FAIL_COND_V_MSG((owners & (owners - 1)) && height != 0.0f, ERR_INVALID_DATA, vformat("Owner %d's displacement moves refined vertex %d by %f, which another owner also refines: the seam would open.", part.owner, u, height));
		}
		if (best_key[u] == INT64_MAX) {
			uv_write[i] = Vector2(NAN, NAN);
		}
		part.heights[i] = height;
		if (height != 0.0f) {
			displaced++;
		}
		lowest = MIN(lowest, height);
		highest = MAX(highest, height);
	}
	part.displaced = true;
	part.displacement_shape = displacement->shape_name;
	part.height_bound = MAX(-lowest, highest);
	part.displacement_report["shape"] = displacement->shape_name;
	part.displacement_report["displaced_vertices"] = displaced;
	part.displacement_report["min_height"] = lowest;
	part.displacement_report["max_height"] = highest;
	part.displacement_report["seam_vertices_reached"] = seam_reached; // Each at height 0: a non-zero one fails the build.
	part.displacement_report["displacement_uvs"] = uvs;
	part.displacement_report["ms"] = ms_since(begin);
	return OK;
}

// An owner's sections: its cage (positions, skin in its own bind indices, sparse deltas in its shape
// order), its stencils over the local cage, its quads, and the normal adjacency of its own refined
// vertices, each vertex's quads in welded quad order so that every owner sums a seam vertex's quads
// in the same order.
Error SubdivisionBuilder::_owner_payload(Work &r_work, uint32_t p_part) {
	Work::Part &part = r_work.parts[p_part];
	const uint32_t cage_count = part.cage_compact.size();
	const uint32_t bones_max = RenderingServerTypes::MESH_SUBDIVISION_CAGE_BONES;
	const Vector<String> &names = r_work.owner_shapes[part.owner];

	// The deltas per local cage vertex, in the owner's shape order: counted, then filled shape by shape,
	// so each vertex's entries come in ascending shape order.
	const uint32_t entry_words = RenderingServerTypes::MESH_SUBDIVISION_CAGE_BLEND_ENTRY_WORDS;
	LocalVector<uint32_t> blend_offsets;
	blend_offsets.resize(cage_count + 1);
	memset(blend_offsets.ptr(), 0, blend_offsets.size() * sizeof(uint32_t));
	for (int n = 0; n < names.size(); n++) {
		for (const KeyValue<uint32_t, Vector3> &E : r_work.shape_deltas[r_work.shape_index[names[n]]]) {
			const int32_t c = part.local_cage[E.key];
			if (c >= 0) {
				blend_offsets[c + 1]++;
			}
		}
	}
	for (uint32_t c = 0; c < cage_count; c++) {
		blend_offsets[c + 1] += blend_offsets[c];
	}
	const uint64_t entry_count = blend_offsets[cage_count];
	ERR_FAIL_COND_V_MSG(entry_count * entry_words * sizeof(uint32_t) > UINT32_MAX, ERR_OUT_OF_MEMORY, vformat("Owner %d's %d blend shape deltas exceed the payload's 4 GB.", part.owner, entry_count));
	LocalVector<uint32_t> blend_entries;
	blend_entries.resize(uint32_t(entry_count * entry_words));
	{
		LocalVector<uint32_t> fill(blend_offsets);
		for (int n = 0; n < names.size(); n++) {
			for (const KeyValue<uint32_t, Vector3> &E : r_work.shape_deltas[r_work.shape_index[names[n]]]) {
				const int32_t c = part.local_cage[E.key];
				if (c < 0) {
					continue;
				}
				uint32_t *entry = &blend_entries[fill[c]++ * entry_words];
				entry[0] = n;
				for (int a = 0; a < 3; a++) {
					const float value = E.value[a];
					memcpy(entry + 1 + a, &value, sizeof(float));
				}
			}
		}
	}
	part.blend_entries = entry_count;

	// The skin in the owner's bind indices: a ghost's joints are found in its receiving owner by name.
	LocalVector<uint16_t> cage_skin;
	part.cage_binds.clear();
	if (r_work.has_skin) {
		LocalVector<int32_t> to_bind;
		if (r_work.named_bones) {
			to_bind.resize(r_work.bone_names.size());
			for (int32_t &index : to_bind) {
				index = -1;
			}
			const Vector<String> &owner_names = owner_bone_names[part.owner];
			for (int i = owner_names.size() - 1; i >= 0; i--) {
				to_bind[r_work.bone_index[owner_names[i]]] = i; // The first bind index of a name.
			}
		}
		for (int32_t c : part.cage_compact) {
			const uint16_t *skin = &r_work.skin[c * 2 * bones_max];
			for (uint32_t b = 0; b < bones_max; b++) {
				uint16_t bind = 0;
				if (skin[b] != SKIN_NO_BONE) {
					if (r_work.named_bones) {
						ERR_FAIL_COND_V_MSG(to_bind[skin[b]] < 0, ERR_INVALID_DATA, vformat("Owner %d lacks bone %s, which its ghost cage vertex %d is weighted to.", part.owner, r_work.bone_names[skin[b]], r_work.welded_of_compact[c]));
						bind = uint16_t(to_bind[skin[b]]);
					} else {
						bind = skin[b];
					}
				}
				cage_skin.push_back(bind);
				part.cage_binds.push_back(skin[b] == SKIN_NO_BONE ? SKIN_NO_BONE : bind);
			}
			for (uint32_t b = 0; b < bones_max; b++) {
				cage_skin.push_back(skin[bones_max + b]);
			}
		}
	}

	const bool wide = cage_count > 65536;
	LocalVector<uint32_t> stencil_offsets;
	LocalVector<uint16_t> indices16;
	LocalVector<uint32_t> indices32;
	LocalVector<float> weights;
	stencil_offsets.push_back(0);
	for (int32_t u : part.welded_u) {
		for (uint32_t i = r_work.stencil_offsets[u]; i < r_work.stencil_offsets[u + 1]; i++) {
			const int32_t c = part.local_cage[r_work.stencil_indices[i]];
			if (wide) {
				indices32.push_back(c);
			} else {
				indices16.push_back(c);
			}
			weights.push_back(r_work.stencil_weights[i]);
		}
		stencil_offsets.push_back(weights.size());
	}

	// Quads: own, then ghost, as local refined vertices.
	LocalVector<uint32_t> quads;
	for (int q : part.own_quads) {
		for (int c = 0; c < 4; c++) {
			quads.push_back(part.local_u[r_work.quads[4 * q + c]]);
		}
	}
	for (int q : part.ghost_quads) {
		for (int c = 0; c < 4; c++) {
			quads.push_back(part.local_u[r_work.quads[4 * q + c]]);
		}
	}
	// Local quads in welded order: own and ghost quads are each in welded order, so merge them.
	LocalVector<uint32_t> by_welded;
	{
		uint32_t own = 0;
		uint32_t ghost = 0;
		while (own < part.own_quads.size() || ghost < part.ghost_quads.size()) {
			if (ghost == part.ghost_quads.size() || (own < part.own_quads.size() && part.own_quads[own] < part.ghost_quads[ghost])) {
				by_welded.push_back(own++);
			} else {
				by_welded.push_back(part.own_quads.size() + ghost++);
			}
		}
	}
	LocalVector<uint32_t> normal_offsets;
	normal_offsets.resize(part.own_u_count + 1);
	memset(normal_offsets.ptr(), 0, normal_offsets.size() * sizeof(uint32_t));
	for (uint32_t i : by_welded) {
		for (int c = 0; c < 4; c++) {
			if (quads[4 * i + c] < part.own_u_count) {
				normal_offsets[quads[4 * i + c] + 1]++;
			}
		}
	}
	for (uint32_t u = 0; u < part.own_u_count; u++) {
		normal_offsets[u + 1] += normal_offsets[u];
	}
	LocalVector<uint32_t> normal_entries;
	normal_entries.resize(normal_offsets[part.own_u_count]);
	LocalVector<uint32_t> fill(normal_offsets);
	for (uint32_t i : by_welded) {
		for (int c = 0; c < 4; c++) {
			const uint32_t u = quads[4 * i + c];
			if (u < part.own_u_count) {
				normal_entries[fill[u]++] = 4 * i + c;
			}
		}
	}

	LocalVector<float> cage_positions;
	LocalVector<uint32_t> cage_welded;
	for (int32_t c : part.cage_compact) {
		for (int a = 0; a < 3; a++) {
			cage_positions.push_back(r_work.positions[c][a]);
		}
		cage_welded.push_back(r_work.welded_of_compact[c]);
	}
	LocalVector<uint32_t> u_welded;
	for (int32_t u : part.welded_u) {
		u_welded.push_back(u);
	}
	LocalVector<uint32_t> info;
	info.resize(RenderingServerTypes::MESH_SUBDIVISION_OWNER_INFO_WORDS);
	info[RenderingServerTypes::MESH_SUBDIVISION_INFO_CAGE_COUNT] = cage_count;
	info[RenderingServerTypes::MESH_SUBDIVISION_INFO_OWN_CAGE_COUNT] = part.own_cage_count;
	info[RenderingServerTypes::MESH_SUBDIVISION_INFO_U_COUNT] = part.welded_u.size();
	info[RenderingServerTypes::MESH_SUBDIVISION_INFO_OWN_U_COUNT] = part.own_u_count;
	info[RenderingServerTypes::MESH_SUBDIVISION_INFO_QUAD_COUNT] = part.own_quads.size() + part.ghost_quads.size();
	info[RenderingServerTypes::MESH_SUBDIVISION_INFO_OWN_QUAD_COUNT] = part.own_quads.size();
	// The displacement's shape comes after the owner's own, without deltas.
	info[RenderingServerTypes::MESH_SUBDIVISION_INFO_BLEND_SHAPE_COUNT] = names.size() + (part.displaced ? 1 : 0);
	info[RenderingServerTypes::MESH_SUBDIVISION_INFO_LEVEL] = r_work.level;
	info[RenderingServerTypes::MESH_SUBDIVISION_INFO_STENCIL_INDEX_BYTES] = wide ? 4 : 2;
	info[RenderingServerTypes::MESH_SUBDIVISION_INFO_STENCIL_WEIGHT_COUNT] = weights.size();
	info[RenderingServerTypes::MESH_SUBDIVISION_INFO_DISPLACEMENT_SHAPE] = RenderingServerTypes::MESH_SUBDIVISION_NO_DISPLACEMENT;
	PayloadWriter &payload = part.owner_payload;
	if (part.displaced) {
		info[RenderingServerTypes::MESH_SUBDIVISION_INFO_DISPLACEMENT_SHAPE] = uint32_t(names.size()); // After the owner's own shapes.
	}
	payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_OWNER_INFO, info);
	payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_POSITIONS, cage_positions);
	if (r_work.has_skin) {
		payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_SKIN, cage_skin);
	}
	payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_OFFSETS, stencil_offsets);
	if (wide) {
		payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_INDICES, indices32);
	} else {
		payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_INDICES, indices16);
	}
	payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_STENCIL_WEIGHTS, weights);
	payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_QUADS, quads);
	payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_NORMAL_OFFSETS, normal_offsets);
	payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_NORMAL_ENTRIES, normal_entries);
	payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_WELDED, cage_welded);
	payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_U_WELDED, u_welded);
	payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_OFFSETS, blend_offsets);
	payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_CAGE_BLEND_ENTRIES, blend_entries);
	if (part.displaced) {
		payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_DISPLACEMENT_HEIGHTS, part.heights);
	}
	return OK;
}

// An owner's mesh: one surface per source surface of the owner, its own quads as two triangles
// each, with their rest pose, tangents, and the surface sections; the owner sections ride on the
// first surface.
Error SubdivisionBuilder::_owner_surfaces(Work &r_work, uint32_t p_part) {
	const uint64_t begin = OS::get_singleton()->get_ticks_usec();
	Work::Part &part = r_work.parts[p_part];
	Vector<String> names = r_work.owner_shapes[part.owner];
	if (part.displaced) {
		names.push_back(part.displacement_shape); // Its weight scales the displacement (S6).
	}
	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	for (const String &name : names) {
		mesh->add_blend_shape(name);
	}
	mesh->set_blend_shape_mode(blend_shape_mode);
	if (fingerprint_input) {
		const uint64_t header[3] = { uint64_t(part.owner), uint64_t(names.size()), uint64_t(blend_shape_mode) };
		_fingerprint_append(header, sizeof(header));
		for (const String &name : names) {
			const CharString utf8 = name.utf8();
			_fingerprint_append(utf8.get_data(), utf8.length());
		}
	}
	int tex_uv_layer = -1;
	for (uint32_t l = 0; l < layers.size(); l++) {
		if (layers[l].slot == Mesh::ARRAY_TEX_UV) {
			tex_uv_layer = l;
		}
	}
	Vector<uint8_t> owner_data;
	uint64_t payload_bytes = 0;
	uint64_t render_vertices = 0;
	uint64_t triangles_total = 0;
	Array surface_tangent_splits; // Per surface of the owner: the render vertices MikkTSpace's grouping added.
	Vector<TriangleSources> triangle_sources;

	// MikkTSpace's frames, as Blender gives the refined mesh its tangents: once over all the owner's own quads,
	// as Blender's calc_tangents runs over a whole mesh (several surfaces share a UV0 island, so they share
	// groups), on the rest surface, in the source's corner order with UVs v up, as Blender feeds it. Each quad is
	// split along its shorter UV diagonal and a badly mapped quad takes its larger triangle's orientation; every
	// corner of one group gets the same frame. 4 per own quad, in part.own_quads' order.
	LocalVector<Vector4> owner_frames;
	if (tex_uv_layer >= 0) {
		const uint32_t corner_count = part.own_quads.size() * 4;
		LocalVector<Vector3> corner_positions;
		LocalVector<Vector3> corner_normals;
		LocalVector<Vector2> corner_uvs;
		corner_positions.resize(corner_count);
		corner_normals.resize(corner_count);
		corner_uvs.resize(corner_count);
		for (uint32_t i = 0; i < corner_count; i++) {
			const int q = part.own_quads[i / 4];
			const int32_t welded = r_work.quads[4 * q + i % 4]; // An own quad's corners are own refined vertices.
			const FVarValue &value = r_work.fvar_values[tex_uv_layer][r_work.quad_fvar[tex_uv_layer][4 * q + i % 4]];
			corner_positions[i] = r_work.u_positions[welded];
			corner_normals[i] = r_work.u_normals[welded];
			corner_uvs[i] = Vector2(float(value.x), float(value.y));
		}
		ERR_FAIL_COND_V_MSG(!mikktspace_quad_frames(corner_positions, corner_normals, corner_uvs, owner_frames), ERR_OUT_OF_MEMORY, vformat("Owner %d: MikkTSpace failed (it fails only to allocate).", part.owner));
	}

	for (uint32_t s = 0; s < surfaces.size(); s++) {
		if (surfaces[s].owner != part.owner) {
			continue;
		}
		HashMap<CornerKey, int32_t, CornerKeyHasher> render_of;
		LocalVector<CornerKey> render_keys;
		LocalVector<int32_t> quad_render; // 4 render vertices per quad of the surface, in the source's corner order.
		LocalVector<uint32_t> quad_own; // Per quad of the surface: its index in part.own_quads.
		for (uint32_t own = 0; own < part.own_quads.size(); own++) {
			const int q = part.own_quads[own];
			if (r_work.quad_surface[q] != int(s)) {
				continue;
			}
			quad_own.push_back(own);
			int32_t render[4];
			for (int c = 0; c < 4; c++) {
				CornerKey key;
				key.u = part.local_u[r_work.quads[4 * q + c]];
				for (uint32_t l = 0; l < layers.size(); l++) {
					key.value[l] = r_work.quad_fvar[l][4 * q + c];
				}
				key.value[MAX_FACE_VARYING_LAYERS] = r_work.has_color ? r_work.quad_color[4 * q + c] : 0;
				const int32_t *existing = render_of.getptr(key);
				if (existing) {
					render[c] = *existing;
				} else {
					render[c] = render_keys.size();
					render_of.insert(key, render[c]);
					render_keys.push_back(key);
				}
			}
			for (int c = 0; c < 4; c++) {
				quad_render.push_back(render[c]);
			}
		}
		ERR_CONTINUE_MSG(quad_render.is_empty(), vformat("Surface %d has no refined faces.", s));

		// MikkTSpace's grouping (owner_frames, above): a render vertex whose corners fall in several groups
		// becomes one render vertex per group, in the order its corners come, and each render vertex keeps its
		// group's sign. Only the grouping and the sign are fixed here: a runtime pass sums each render vertex's
		// tangent from its own group's triangles on the posed surface.
		LocalVector<float> render_sign;
		uint32_t tangent_splits = 0;
		if (tex_uv_layer >= 0) {
			const uint32_t corner_count = quad_render.size();
			HashMap<GroupKey, int32_t, GroupKeyHasher, GroupKeyComparator> render_of_group;
			LocalVector<int32_t> copy_source; // Per copy, in the order they come: the render vertex it was split from.
			LocalVector<uint8_t> grouped;
			grouped.resize(render_keys.size());
			memset(grouped.ptr(), 0, grouped.size());
			render_sign.resize(render_keys.size());
			for (uint32_t i = 0; i < corner_count; i++) {
				const Vector4 &frame = owner_frames[4 * quad_own[i / 4] + i % 4];
				GroupKey key;
				key.render = quad_render[i];
				key.frame[0] = float(frame.x);
				key.frame[1] = float(frame.y);
				key.frame[2] = float(frame.z);
				key.frame[3] = float(frame.w);
				const int32_t *existing = render_of_group.getptr(key);
				int32_t render;
				if (existing) {
					render = *existing;
				} else if (!grouped[key.render]) {
					render = key.render;
					grouped[render] = 1;
				} else {
					render = render_keys.size();
					render_keys.push_back(CornerKey(render_keys[key.render]));
					render_sign.push_back(0.0f);
					copy_source.push_back(key.render);
					tangent_splits++;
				}
				if (!existing) {
					render_of_group.insert(key, render);
					render_sign[render] = frame.w < 0.0f ? -1.0f : 1.0f;
				}
				quad_render[i] = render;
			}
			// Each copy right after the vertex it was split from, in the order the copies came, so a copy stays near
			// its neighbours in the vertex buffer instead of at its end.
			if (tangent_splits > 0) {
				const uint32_t originals = render_keys.size() - tangent_splits;
				LocalVector<int32_t> first_copy;
				LocalVector<int32_t> next_copy;
				LocalVector<int32_t> last_copy;
				first_copy.resize(originals);
				last_copy.resize(originals);
				next_copy.resize(tangent_splits);
				for (uint32_t r = 0; r < originals; r++) {
					first_copy[r] = -1;
				}
				for (uint32_t c = 0; c < tangent_splits; c++) {
					const int32_t source = copy_source[c];
					next_copy[c] = -1;
					if (first_copy[source] < 0) {
						first_copy[source] = c;
					} else {
						next_copy[last_copy[source]] = c;
					}
					last_copy[source] = c;
				}
				LocalVector<int32_t> moved_to;
				moved_to.resize(render_keys.size());
				LocalVector<CornerKey> ordered_keys;
				LocalVector<float> ordered_signs;
				for (uint32_t r = 0; r < originals; r++) {
					moved_to[r] = ordered_keys.size();
					ordered_keys.push_back(render_keys[r]);
					ordered_signs.push_back(render_sign[r]);
					for (int32_t c = first_copy[r]; c >= 0; c = next_copy[c]) {
						moved_to[originals + c] = ordered_keys.size();
						ordered_keys.push_back(render_keys[originals + c]);
						ordered_signs.push_back(render_sign[originals + c]);
					}
				}
				render_keys = ordered_keys;
				render_sign = ordered_signs;
				for (int32_t &render : quad_render) {
					render = moved_to[render];
				}
			}
		}
		if (quad_render_output) {
			quad_render_output->push_back(LocalVector<int32_t>(quad_render));
		}
		LocalVector<int32_t> triangles;
		for (uint32_t q = 0; q < quad_render.size() / 4; q++) {
			const int32_t *render = &quad_render[4 * q];
			// The quad's corners run counter-clockwise from the front, as the source's (Blender's) do;
			// Godot's front faces are clockwise, so each triangle is reversed, as the glTF import does.
			const int32_t tris[6] = { render[0], render[2], render[1], render[0], render[3], render[2] };
			for (int i = 0; i < 6; i++) {
				triangles.push_back(tris[i]);
			}
		}

		const uint32_t count = render_keys.size();
		ERR_FAIL_COND_V_MSG(count >= (1u << 21), ERR_OUT_OF_MEMORY, vformat("Surface %d has %d render vertices, over the payload's 2 million.", s, count));
		PackedVector3Array positions;
		PackedVector3Array normals;
		positions.resize(count);
		normals.resize(count);
		PackedVector2Array uvs[MAX_FACE_VARYING_LAYERS];
		PackedColorArray colors;
		if (r_work.has_color) {
			colors.resize(count);
		}
		LocalVector<uint32_t> r_to_u;
		r_to_u.resize(count);
		for (uint32_t l = 0; l < layers.size(); l++) {
			uvs[l].resize(count);
		}
		for (uint32_t r = 0; r < count; r++) {
			const CornerKey &key = render_keys[r];
			const int32_t welded = part.welded_u[key.u];
			r_to_u[r] = key.u;
			positions.write[r] = r_work.u_positions[welded];
			normals.write[r] = r_work.u_normals[welded];
			for (uint32_t l = 0; l < layers.size(); l++) {
				// Refined in the source's UV space (v up), stored as glTF stores it: 1 - v in float.
				const FVarValue &value = r_work.fvar_values[l][key.value[l]];
				uvs[l].write[r] = Vector2(float(value.x), 1.0f - float(value.y));
			}
			if (r_work.has_color) {
				const FVarValue &value = r_work.refined_colors[key.value[MAX_FACE_VARYING_LAYERS]];
				colors.write[r] = Color(value.x, value.y, value.z, value.w);
			}
		}

		// Bone bounds, which a posed instance's culling bound is made from as RenderingServer's are for a
		// skinned surface (D122 S2): per bind index, the rest positions of the cage vertices weighted to it
		// that the surface's stencils reach. A posed refined vertex is a combination with positive weights
		// (Catmull-Clark's stencils, the skin's) of bone-transformed cage vertices, so it stays inside the
		// transformed bounds; bounds of the render vertices would not hold it. Unused bones are (-1, -1, -1).
		Vector<AABB> bone_aabbs;
		if (r_work.has_skin) {
			int max_bind = -1;
			for (uint16_t bind : part.cage_binds) {
				if (bind != SKIN_NO_BONE) {
					max_bind = MAX(max_bind, int(bind));
				}
			}
			bone_aabbs.resize(max_bind + 1);
			AABB *bounds = bone_aabbs.ptrw();
			for (int b = 0; b <= max_bind; b++) {
				bounds[b].size = Vector3(-1, -1, -1);
			}
			const uint32_t bones_max = RenderingServerTypes::MESH_SUBDIVISION_CAGE_BONES;
			for (uint32_t r = 0; r < count; r++) {
				const int32_t welded = part.welded_u[render_keys[r].u];
				for (uint32_t i = r_work.stencil_offsets[welded]; i < r_work.stencil_offsets[welded + 1]; i++) {
					const uint32_t compact = r_work.stencil_indices[i];
					const int32_t c = part.local_cage[compact];
					for (uint32_t b = 0; b < bones_max; b++) {
						const uint16_t bind = part.cage_binds[c * bones_max + b];
						if (bind == SKIN_NO_BONE) {
							continue;
						}
						if (bounds[bind].size.x < 0) {
							bounds[bind] = AABB(r_work.positions[compact], Vector3(CMP_EPSILON, CMP_EPSILON, CMP_EPSILON));
						} else {
							bounds[bind].expand_to(r_work.positions[compact]);
						}
					}
				}
			}
			int used = bone_aabbs.size();
			while (used > 0 && bounds[used - 1].size.x < 0) {
				used--;
			}
			bone_aabbs.resize(used);
			// A displaced vertex leaves its stencil's hull by up to its height (S6), at an amount up to 1.
			for (int b = 0; b < used && part.height_bound > 0.0f; b++) {
				if (bounds[b].size.x >= 0) {
					bounds[b] = bounds[b].grow(part.height_bound);
				}
			}
		}

		// Tangents from the triangles' UV gradients, each triangle's corner weighted by its angle,
		// orthogonalized against the normal. Only a triangle of zero UV area is skipped: the gradients
		// are taken without dividing by the area, its sign alone orienting them (as MikkTSpace orients each
		// triangle's gradient by its own UV area, whatever its group), so tiny UV triangles count fully. A
		// render vertex sums only its own MikkTSpace group's triangles (above). The sign is that group's,
		// Godot's as SurfaceTool's MikkTSpace gives it (clockwise triangles, v down).
		PackedFloat32Array tangents;
		if (tex_uv_layer >= 0) {
			LocalVector<Vector3> tangent_sum;
			tangent_sum.resize(count);
			for (uint32_t r = 0; r < count; r++) {
				tangent_sum[r] = Vector3();
			}
			const Vector2 *uv = uvs[tex_uv_layer].ptr();
			for (uint32_t t = 0; t < triangles.size() / 3; t++) {
				const int32_t *tri = &triangles[3 * t];
				const Vector3 e1 = positions[tri[1]] - positions[tri[0]];
				const Vector3 e2 = positions[tri[2]] - positions[tri[0]];
				const Vector2 d1 = uv[tri[1]] - uv[tri[0]];
				const Vector2 d2 = uv[tri[2]] - uv[tri[0]];
				const float det = d1.x * d2.y - d2.x * d1.y;
				if (det == 0.0f) {
					continue;
				}
				const float orientation = det < 0.0f ? -1.0f : 1.0f;
				const Vector3 tangent = ((e1 * d2.y - e2 * d1.y) * orientation).normalized();
				for (int c = 0; c < 3; c++) {
					const float angle = corner_angle(positions[tri[(c + 2) % 3]], positions[tri[c]], positions[tri[(c + 1) % 3]]);
					tangent_sum[tri[c]] += tangent * angle;
				}
			}
			tangents.resize(4 * count);
			for (uint32_t r = 0; r < count; r++) {
				const Vector3 n = normals[r];
				Vector3 t = tangent_sum[r] - n * n.dot(tangent_sum[r]);
				if (t.is_zero_approx()) {
					t = n.get_any_perpendicular();
				}
				t.normalize();
				tangents.write[4 * r + 0] = t.x;
				tangents.write[4 * r + 1] = t.y;
				tangents.write[4 * r + 2] = t.z;
				tangents.write[4 * r + 3] = render_sign[r];
			}
		}

		Array arrays;
		arrays.resize(Mesh::ARRAY_MAX);
		arrays[Mesh::ARRAY_VERTEX] = positions;
		arrays[Mesh::ARRAY_NORMAL] = normals;
		if (!tangents.is_empty()) {
			arrays[Mesh::ARRAY_TANGENT] = tangents;
		}
		uint64_t format_flags = 0;
		for (uint32_t l = 0; l < layers.size(); l++) {
			if (layers[l].slot == Mesh::ARRAY_CUSTOM0) {
				PackedFloat32Array custom;
				custom.resize(2 * count);
				for (uint32_t r = 0; r < count; r++) {
					custom.write[2 * r + 0] = uvs[l][r].x;
					custom.write[2 * r + 1] = uvs[l][r].y;
				}
				arrays[Mesh::ARRAY_CUSTOM0] = custom;
				format_flags |= uint64_t(Mesh::ARRAY_CUSTOM_RG_FLOAT) << Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT;
			} else {
				arrays[layers[l].slot] = uvs[l];
			}
		}
		if (r_work.has_color) {
			arrays[Mesh::ARRAY_COLOR] = colors;
		}
		PackedInt32Array indices;
		indices.resize(triangles.size());
		memcpy(indices.ptrw(), triangles.ptr(), triangles.size() * sizeof(int32_t));
		arrays[Mesh::ARRAY_INDEX] = indices;

		// The builder clusters the surface itself, as the rendering server would with
		// rendering/pathtracing/cluster_blas and cluster_lod on (the gate in
		// RenderingServer::mesh_create_surface_data_from_arrays, mirrored here without the settings; a change
		// to one is mirrored in the other or deliberately left out), whatever they say: a subdivided surface's index
		// order is frozen here (the payload's adjacency indexes it, and the storages never reorder it), so the
		// bytes built must not depend on a project setting. An owner without blend shapes gets a cluster LOD
		// hierarchy (D85) when it has more than one cluster's triangles, every other surface the flat clusters
		// (D80): the cage's morphs move a blend-shaped surface away from the rest pose a hierarchy's errors are
		// measured on (D87), and the storages refuse a hierarchy on one. With the settings off, or on a device
		// without cluster BLAS, the coarse levels or the whole table go unused over the same indices.
		Vector<int> cluster_indices;
		cluster_indices.resize(triangles.size());
		memcpy(cluster_indices.ptrw(), triangles.ptr(), triangles.size() * sizeof(int32_t));
		Vector<uint8_t> cluster_table;
		if (names.is_empty() && cluster_indices.size() > int(RenderingServerTypes::MESH_CLUSTER_MAX_TRIANGLES) * 3) {
			Vector<Vector3> normals;
			if (arrays[Mesh::ARRAY_NORMAL].get_type() == Variant::PACKED_VECTOR3_ARRAY) {
				normals = arrays[Mesh::ARRAY_NORMAL];
			}
			Vector<Vector2> hierarchy_uvs;
			if (arrays[Mesh::ARRAY_TEX_UV].get_type() == Variant::PACKED_VECTOR2_ARRAY) {
				hierarchy_uvs = arrays[Mesh::ARRAY_TEX_UV];
			}
			cluster_table = RenderingServer::mesh_surface_build_cluster_lod(arrays[Mesh::ARRAY_VERTEX], normals, hierarchy_uvs, cluster_indices);
		}
		if (cluster_table.is_empty()) {
			cluster_table = RenderingServer::mesh_surface_build_clusters(arrays[Mesh::ARRAY_VERTEX], cluster_indices);
		}
		if (cluster_table.size()) {
			arrays[Mesh::ARRAY_INDEX] = PackedInt32Array(cluster_indices);
		}
		RenderingServerTypes::SurfaceData surface_data;
		const Error err = RS::get_singleton()->mesh_create_surface_data_from_arrays(&surface_data, RSE::PRIMITIVE_TRIANGLES, arrays, Array(), Dictionary(), format_flags, false, false);
		ERR_FAIL_COND_V_MSG(err != OK, err, vformat("Surface %d: the refined arrays were rejected.", s));
		surface_data.cluster_data = cluster_table;
		surface_data.bone_aabbs = bone_aabbs;
		surface_data.aabb = surface_data.aabb.grow(part.height_bound);

		// Tangent adjacency over the surface's final index order (the cluster builder reorders
		// triangles), each vertex's entries in the order its tangent was summed above. That order is
		// frozen from here: the storages never reorder a subdivided surface's indices.
		// As RenderingServer packs indices: 16 bits up to 65536 vertices.
		const uint32_t index_size = surface_data.vertex_count <= 65536 ? 2 : 4;
		ERR_FAIL_COND_V(surface_data.index_count != triangles.size() || uint64_t(surface_data.index_data.size()) < uint64_t(surface_data.index_count) * index_size, ERR_BUG);
		auto final_index = [&surface_data, index_size](uint32_t p_i) -> uint32_t {
			const uint8_t *data = surface_data.index_data.ptr() + uint64_t(p_i) * index_size;
			return index_size == 2 ? uint32_t(*reinterpret_cast<const uint16_t *>(data)) : *reinterpret_cast<const uint32_t *>(data);
		};
		HashMap<uint64_t, uint32_t> final_triangle; // Sorted vertex triple to its final triangle.
		auto triple_key = [](uint32_t p_a, uint32_t p_b, uint32_t p_c) -> uint64_t {
			uint32_t v[3] = { p_a, p_b, p_c };
			if (v[1] < v[0]) {
				SWAP(v[0], v[1]);
			}
			if (v[2] < v[1]) {
				SWAP(v[1], v[2]);
			}
			if (v[1] < v[0]) {
				SWAP(v[0], v[1]);
			}
			// 21 bits a vertex: a surface's render vertices stay under two million.
			return (uint64_t(v[0]) << 42) | (uint64_t(v[1]) << 21) | uint64_t(v[2]);
		};
		for (uint32_t t = 0; t < surface_data.index_count / 3; t++) {
			final_triangle.insert(triple_key(final_index(3 * t), final_index(3 * t + 1), final_index(3 * t + 2)), t);
		}
		LocalVector<uint32_t> tangent_offsets;
		tangent_offsets.resize(count + 1);
		memset(tangent_offsets.ptr(), 0, tangent_offsets.size() * sizeof(uint32_t));
		for (uint32_t i = 0; i < triangles.size(); i++) {
			tangent_offsets[triangles[i] + 1]++;
		}
		for (uint32_t r = 0; r < count; r++) {
			tangent_offsets[r + 1] += tangent_offsets[r];
		}
		LocalVector<uint32_t> tangent_entries;
		tangent_entries.resize(triangles.size());
		PackedInt32Array source_faces;
		source_faces.resize(surface_data.index_count / 3);
		source_faces.fill(-1);
		LocalVector<uint32_t> fill(tangent_offsets);
		for (uint32_t t = 0; t < triangles.size() / 3; t++) {
			const uint32_t *found = final_triangle.getptr(triple_key(triangles[3 * t], triangles[3 * t + 1], triangles[3 * t + 2]));
			ERR_FAIL_NULL_V_MSG(found, ERR_BUG, "A refined triangle is missing from the surface's final indices.");
			const int coarse_face = r_work.quad_face[part.own_quads[quad_own[t / 2]]];
			ERR_FAIL_COND_V(r_work.face_surface[coarse_face] != int(s) || source_faces[*found] != -1, ERR_BUG);
			source_faces.set(*found, r_work.face_local[coarse_face]);
			for (int c = 0; c < 3; c++) {
				const uint32_t r = triangles[3 * t + c];
				uint32_t corner = 0;
				while (corner < 3 && final_index(3 * *found + corner) != r) {
					corner++;
				}
				ERR_FAIL_COND_V(corner == 3, ERR_BUG);
				tangent_entries[fill[r]++] = 3 * *found + corner;
			}
		}

		// Every final triangle must have exactly one source face, including after clustering.
		for (int32_t face : source_faces) {
			ERR_FAIL_COND_V(face < 0 || uint32_t(face) >= surfaces[s].faces.size() / 4, ERR_BUG);
		}

		// The payload: the surface sections, and the owner sections on the owner's first surface,
		// which every surface names in word 2.
		const bool first = mesh->get_surface_count() == 0;
		PayloadWriter payload = first ? part.owner_payload : PayloadWriter();
		payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_R_TO_U, r_to_u);
		payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_OFFSETS, tangent_offsets);
		payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_ENTRIES, tangent_entries);
		// One bit per render vertex, set where its sign is +1; without UVs every sign is +1 (unused).
		LocalVector<uint32_t> tangent_signs;
		tangent_signs.resize((count + 31) / 32);
		memset(tangent_signs.ptr(), 0, tangent_signs.size() * sizeof(uint32_t));
		for (uint32_t r = 0; r < count; r++) {
			if (render_sign.is_empty() || render_sign[r] > 0.0f) {
				tangent_signs[r / 32] |= 1u << (r % 32);
			}
		}
		payload.add(RenderingServerTypes::MESH_SUBDIVISION_SECTION_TANGENT_SIGNS, tangent_signs);
		surface_data.subdivision_data = payload.finish(first ? RenderingServerTypes::MESH_SUBDIVISION_FLAG_OWNER : 0, 0);
		ERR_FAIL_COND_V(surface_data.subdivision_data.is_empty(), ERR_OUT_OF_MEMORY);
		if (first) {
			owner_data = surface_data.subdivision_data;
		}
		// The check every storage runs, here, so that a rejected surface fails the build.
		const int surface_index = mesh->get_surface_count();
		String error;
		ERR_FAIL_COND_V_MSG(!RenderingServer::mesh_subdivision_surface_is_valid(surface_data, surface_index, names.size(), first ? -1 : 1, owner_data, &error), ERR_BUG, vformat("Surface %d's payload is invalid: %s.", s, error));
		payload_bytes += surface_data.subdivision_data.size();
		render_vertices += count;
		triangles_total += triangles.size() / 3;
		surface_tangent_splits.push_back(tangent_splits);

		if (fingerprint_input) {
			// Every field the surface carries, each length-prefixed, in build order.
			const uint64_t header[6] = { uint64_t(part.owner), uint64_t(s), surface_data.format, uint64_t(surface_data.primitive), surface_data.vertex_count, surface_data.index_count };
			_fingerprint_append(header, sizeof(header));
			for (const Vector<uint8_t> *bytes : { &surface_data.vertex_data, &surface_data.attribute_data, &surface_data.skin_data, &surface_data.index_data, &surface_data.blend_shape_data, &surface_data.cluster_data, &surface_data.subdivision_data }) {
				_fingerprint_append(bytes->ptr(), bytes->size());
			}
			_fingerprint_append(&surface_data.aabb, sizeof(AABB));
			_fingerprint_append(surface_data.bone_aabbs.ptr(), surface_data.bone_aabbs.size() * sizeof(AABB));
			for (const RenderingServerTypes::SurfaceData::LOD &lod : surface_data.lods) {
				_fingerprint_append(&lod.edge_length, sizeof(lod.edge_length));
				_fingerprint_append(lod.index_data.ptr(), lod.index_data.size());
			}
			_fingerprint_append(&surface_data.uv_scale, sizeof(Vector4));
		}

		mesh->add_surface(surface_data.format, Mesh::PrimitiveType(surface_data.primitive), surface_data.vertex_data, surface_data.attribute_data, surface_data.skin_data, surface_data.vertex_count, surface_data.index_data, surface_data.index_count, surface_data.aabb, Vector<uint8_t>(), surface_data.bone_aabbs, surface_data.lods, surface_data.uv_scale, surface_data.cluster_data, surface_data.subdivision_data);
		// ArrayMesh counts its own surface before the server accepts it, so the server's count is the one asked.
		ERR_FAIL_COND_V_MSG(RS::get_singleton()->mesh_get_surface_count(mesh->get_rid()) != surface_index + 1, ERR_BUG, vformat("Surface %d was not accepted by the rendering server.", s));
		TriangleSources source;
		source.source_surface = s;
		source.source_faces = source_faces;
		triangle_sources.push_back(source);
		if (surfaces[s].material.is_valid()) {
			mesh->surface_set_material(surface_index, surfaces[s].material);
			mesh->surface_set_name(surface_index, surfaces[s].material->get_name());
		}
	}
	ERR_FAIL_COND_V_MSG(mesh->get_surface_count() == 0, ERR_INVALID_DATA, vformat("Owner %d has no refined faces.", part.owner));
	owner_meshes.insert(part.owner, mesh);
	owner_triangle_sources.insert(part.owner, triangle_sources);

	Dictionary owner_report;
	owner_report["surfaces"] = mesh->get_surface_count();
	owner_report["cage_vertices"] = part.own_cage_count;
	owner_report["ghost_cage_vertices"] = part.cage_compact.size() - part.own_cage_count;
	owner_report["refined_vertices"] = part.own_u_count;
	owner_report["ghost_refined_vertices"] = part.welded_u.size() - part.own_u_count;
	owner_report["quads"] = part.own_quads.size();
	owner_report["ghost_quads"] = part.ghost_quads.size();
	owner_report["render_vertices"] = render_vertices;
	owner_report["triangles"] = triangles_total;
	owner_report["tangent_splits"] = surface_tangent_splits;
	owner_report["blend_shapes"] = names.size();
	owner_report["payload_bytes"] = payload_bytes;
	owner_report["blend_entries"] = part.blend_entries;
	if (part.displaced) {
		owner_report["displacement"] = part.displacement_report;
	}
	owner_report["ms"] = ms_since(begin);
	r_work.owner_reports[part.owner] = owner_report;
	return OK;
}

Error SubdivisionBuilder::build_recording_quads(int p_level, LocalVector<LocalVector<int32_t>> &r_quads) {
	quad_render_output = &r_quads;
	const Error err = build(p_level);
	quad_render_output = nullptr;
	return err;
}

bool SubdivisionBuilder::mikktspace_quad_frames(const LocalVector<Vector3> &p_positions, const LocalVector<Vector3> &p_normals, const LocalVector<Vector2> &p_uvs, LocalVector<Vector4> &r_frames) {
	ERR_FAIL_COND_V(p_positions.size() % 4 != 0 || p_normals.size() != p_positions.size() || p_uvs.size() != p_positions.size(), false);
	r_frames.resize(p_positions.size());
	for (Vector4 &frame : r_frames) {
		frame = Vector4();
	}
	QuadFrames data;
	data.positions = &p_positions;
	data.normals = &p_normals;
	data.uvs = &p_uvs;
	data.frames = &r_frames;
	SMikkTSpaceInterface callbacks;
	callbacks.m_getNumFaces = quad_frames_faces;
	callbacks.m_getNumVerticesOfFace = quad_frames_vertices;
	callbacks.m_getPosition = quad_frames_position;
	callbacks.m_getNormal = quad_frames_normal;
	callbacks.m_getTexCoord = quad_frames_uv;
	callbacks.m_setTSpaceBasic = quad_frames_set;
	callbacks.m_setTSpace = nullptr;
	SMikkTSpaceContext context;
	context.m_pInterface = &callbacks;
	context.m_pUserData = &data;
	return genTangSpaceDefault(&context);
}

Error SubdivisionBuilder::build(int p_level) {
	const uint64_t begin = OS::get_singleton()->get_ticks_usec();
	owner_meshes.clear();
	owner_triangle_sources.clear();
	report = Dictionary();
	ERR_FAIL_COND_V_MSG(p_level < 1 || p_level > 4, ERR_INVALID_PARAMETER, "The level is 1 to 4.");
	Work work;
	Error err = _gather(work);
	if (err == OK) {
		err = _refine(work, p_level);
	}
	if (err == OK) {
		work.parts.resize(work.owners.size());
		for (uint32_t i = 0; i < work.parts.size() && err == OK; i++) {
			work.parts[i].owner = work.owners[i];
			err = _partition_owner(work, i);
		}
	}
	if (err == OK) {
		err = _check_owners(work);
	}
	for (uint32_t i = 0; i < work.parts.size() && err == OK; i++) {
		err = _owner_displacement(work, i);
	}
	if (err == OK) {
		for (const KeyValue<int, OwnerDisplacement> &E : owner_displacements) {
			ERR_FAIL_COND_V_MSG(!work.owners.has(E.key), ERR_INVALID_PARAMETER, vformat("A displacement is set on owner %d, which has no surface.", E.key));
		}
	}
	for (uint32_t i = 0; i < work.parts.size() && err == OK; i++) {
		err = _owner_payload(work, i);
		if (err == OK) {
			err = _owner_surfaces(work, i);
		}
	}
	if (err != OK) {
		owner_meshes.clear();
		owner_triangle_sources.clear();
		return err;
	}
	report["level"] = p_level;
	report["cage_vertices"] = work.welded_of_compact.size();
	report["cage_faces"] = work.face_size.size();
	report["refined_vertices"] = work.u_count;
	report["refined_quads"] = work.quad_count;
	report["stencil_weights"] = work.stencil_weights.size();
	report["blend_shapes"] = work.shape_names.size();
	report["gather_ms"] = work.gather_ms;
	report["refine_ms"] = work.refine_ms;
	report["stencil_ms"] = work.stencil_ms;
	report["normal_ms"] = work.normal_ms;
	report["owners"] = work.owner_reports;
	report["total_ms"] = ms_since(begin);
	print_verbose(vformat("SubdivisionBuilder: %d cage vertices, %d faces to %d refined vertices at level %d, %d owners in %.1f ms.", work.welded_of_compact.size(), work.face_size.size(), work.u_count, p_level, work.owners.size(), double(report["total_ms"])));
	return OK;
}

void SubdivisionBuilder::_fingerprint_append(const void *p_data, uint64_t p_size) {
	const uint8_t *size = reinterpret_cast<const uint8_t *>(&p_size);
	for (uint32_t i = 0; i < sizeof(p_size); i++) {
		fingerprint_input->push_back(size[i]);
	}
	const uint8_t *data = reinterpret_cast<const uint8_t *>(p_data);
	for (uint64_t i = 0; i < p_size; i++) {
		fingerprint_input->push_back(data[i]);
	}
}

Ref<SubdivisionBuilder> SubdivisionBuilder::build_fingerprint_probe(LocalVector<uint8_t> *r_input) {
	// A 4 x 2 grid of quads and one triangle off its end. Owner 0 holds the first two columns as two surfaces
	// (its owner sections ride on the first), owner 1 the last two and the triangle; the seam is x = 2, so both
	// owners carry ghosts. Owner 2, apart, has no blend shapes and enough triangles for a cluster LOD hierarchy. The grid is off the plane so normals and tangents vary. Every corner is its own glTF
	// vertex with its own weld group, in three UV layers in the A1 slots (UV, UV2, CUSTOM0 as RG float), each
	// with its own values; colours vary by face; skin weights are fractional so the quantizer's rounding shows;
	// one blend shape moves a seam vertex and one vertex of each owner, so both carry sparse deltas. Owner 2's
	// first face has its UVs folded (two corners' values swapped), so its refined quads take both orientations
	// and MikkTSpace splits render vertices. Owner 0's second surface displaces (S6) through its UV2 values, its
	// source order putting its second face first, so the edge the two faces share (a UV seam: every face has its
	// own values) takes the second face's heights; its mask is zero where its faces meet the seam (UV2 v = 0.625).
	const int rows = 2;
	auto cage_index = [rows](int p_x, int p_z) { return p_x * (rows + 1) + p_z; };
	const int tip = cage_index(5, 0); // The triangle's own vertex, past the grid's last column.
	// Owner 2: a 4 x 4 grid of its own, apart from the strip, without blend shapes, so its 512 refined triangles
	// take the cluster LOD hierarchy.
	const int apart = tip + 1;
	const int apart_side = 5;
	auto apart_index = [apart, apart_side](int p_x, int p_z) { return apart + p_x * apart_side + p_z; };
	auto position_of = [rows, tip, apart, apart_side](int p_cage) {
		if (p_cage == tip) {
			return Vector3(5, 0.0625f, 0.5f);
		}
		if (p_cage >= apart) {
			const int x = (p_cage - apart) / apart_side;
			const int z = (p_cage - apart) % apart_side;
			return Vector3(10 + x, 0.25f * float((x * z) % 3), z);
		}
		const int x = p_cage / (rows + 1);
		const int z = p_cage % (rows + 1);
		return Vector3(x, 0.125f * float((x * 3 + z * 5) % 4), z);
	};
	PackedStringArray bone_names;
	bone_names.push_back("B0");
	bone_names.push_back("B1");
	bone_names.push_back("B2");
	PackedStringArray shape_names;
	shape_names.push_back("Probe");
	HashMap<int, Vector3> deltas;
	deltas.insert(cage_index(2, 1), Vector3(0, 0.25f, 0));
	deltas.insert(cage_index(1, 1), Vector3(0.125f, 0, 0));
	deltas.insert(cage_index(3, 1), Vector3(0, 0, 0.0625f));
	const Vector2 square[4] = { Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1) };

	// Faces, 4 cage vertices each (-1 the fourth of a triangle), and the source surface each goes to.
	LocalVector<int> faces;
	LocalVector<int> face_surface;
	for (int x = 0; x < 4; x++) {
		for (int z = 0; z < rows; z++) {
			const int quad[4] = { cage_index(x, z), cage_index(x, z + 1), cage_index(x + 1, z + 1), cage_index(x + 1, z) };
			for (int c = 0; c < 4; c++) {
				faces.push_back(quad[c]);
			}
			face_surface.push_back(x < 2 ? x : 2);
		}
	}
	// Its first edge runs opposite to the last quad's, so the cage stays a consistently oriented manifold.
	const int triangle[4] = { cage_index(4, 0), cage_index(4, 1), tip, -1 };
	for (int c = 0; c < 4; c++) {
		faces.push_back(triangle[c]);
	}
	face_surface.push_back(2);
	for (int x = 0; x < apart_side - 1; x++) {
		for (int z = 0; z < apart_side - 1; z++) {
			const int quad[4] = { apart_index(x, z), apart_index(x, z + 1), apart_index(x + 1, z + 1), apart_index(x + 1, z) };
			for (int c = 0; c < 4; c++) {
				faces.push_back(quad[c]);
			}
			face_surface.push_back(3);
		}
	}
	const int surface_owner[4] = { 0, 0, 1, 2 };

	const uint32_t folded_face = 8 + 1; // Owner 2's first: after the strip's eight quads and the triangle.
	const Vector2 folded[4] = { square[0], square[1], square[3], square[2] };
	PackedVector2Array uv_values[3];
	Array arrays[4];
	Array blend_arrays[4];
	PackedInt32Array cage_vertex[4];
	PackedInt32Array glTF_faces[4];
	PackedInt32Array uv_weld[4];
	for (int surface = 0; surface < 4; surface++) {
		PackedVector3Array vertices;
		PackedVector3Array targets;
		PackedInt32Array bones;
		PackedFloat32Array weights;
		PackedColorArray colors;
		for (uint32_t f = 0; f < face_surface.size(); f++) {
			if (face_surface[f] != surface) {
				continue;
			}
			for (int c = 0; c < 4; c++) {
				const int cage = faces[4 * f + c];
				if (cage < 0) {
					glTF_faces[surface].push_back(-1);
					uv_weld[surface].push_back(-1);
					continue;
				}
				const Vector3 position = position_of(cage);
				glTF_faces[surface].push_back(vertices.size());
				cage_vertex[surface].push_back(cage);
				vertices.push_back(position);
				const Vector3 *delta = deltas.getptr(cage);
				targets.push_back(delta ? position + *delta : position);
				const float even[4] = { 0.3f, 0.7f, 0.0f, 0.0f };
				const float odd[4] = { 0.15f, 0.35f, 0.5f, 0.0f };
				for (int b = 0; b < 4; b++) {
					bones.push_back(b < 3 ? b : 0);
					weights.push_back(cage % 2 == 0 ? even[b] : odd[b]);
				}
				colors.push_back(Color(0.125f * float(f % 8), 1.0f - 0.0625f * float(f % 16), 0.5f, 1.0f));
				const Vector2 uv = f == folded_face ? folded[c] : square[c];
				uv_weld[surface].push_back(uv_values[0].size());
				uv_values[0].push_back(uv);
				uv_values[1].push_back(uv * 0.5f + Vector2(0.25f, 0.125f));
				uv_values[2].push_back(Vector2(uv.y * 0.75f, uv.x * 0.125f + 0.5f));
			}
		}
		arrays[surface].resize(Mesh::ARRAY_MAX);
		arrays[surface][Mesh::ARRAY_VERTEX] = vertices;
		arrays[surface][Mesh::ARRAY_BONES] = bones;
		arrays[surface][Mesh::ARRAY_WEIGHTS] = weights;
		arrays[surface][Mesh::ARRAY_COLOR] = colors;
		if (surface_owner[surface] != 2) {
			Array blend;
			blend.resize(Mesh::ARRAY_MAX);
			blend[Mesh::ARRAY_VERTEX] = targets;
			blend_arrays[surface].push_back(blend);
		}
	}

	Ref<SubdivisionBuilder> builder;
	builder.instantiate();
	builder->set_cage_vertex_count(apart_index(apart_side - 1, apart_side - 1) + 1);
	const Mesh::ArrayType slots[3] = { Mesh::ARRAY_TEX_UV, Mesh::ARRAY_TEX_UV2, Mesh::ARRAY_CUSTOM0 };
	for (int l = 0; l < 3; l++) {
		ERR_FAIL_COND_V_MSG(builder->add_face_varying_layer(uv_values[l], slots[l]) != l, Ref<SubdivisionBuilder>(), "The fingerprint probe's UV layers were refused.");
	}
	for (int owner = 0; owner < 3; owner++) {
		builder->set_owner_bone_names(owner, bone_names);
	}
	for (int surface = 0; surface < 4; surface++) {
		Array weld;
		for (int l = 0; l < 3; l++) {
			weld.push_back(uv_weld[surface]);
		}
		ERR_FAIL_COND_V_MSG(builder->add_surface(surface_owner[surface], arrays[surface], blend_arrays[surface], surface_owner[surface] == 2 ? PackedStringArray() : shape_names, cage_vertex[surface], glTF_faces[surface], weld, Ref<Material>()) < 0, Ref<SubdivisionBuilder>(), "The fingerprint probe was refused.");
	}
	{
		// 4 x 4 maps, rows from the top as Godot stores them: the mask's third row from the bottom (row 1 from the
		// top) is zero, so the seam's vertices, at v = 0.625, read it alone.
		Vector<uint8_t> height_bytes;
		Vector<uint8_t> mask_bytes;
		for (int y = 0; y < 4; y++) {
			for (int x = 0; x < 4; x++) {
				height_bytes.push_back(uint8_t(16 + 48 * x + 8 * y));
				height_bytes.push_back(uint8_t(200 - 40 * x));
				height_bytes.push_back(uint8_t(32 * y + 7));
				mask_bytes.push_back(y == 1 ? 0 : uint8_t(255 - 50 * x - 20 * y));
			}
		}
		const Ref<Image> height_map = Image::create_from_data(4, 4, false, Image::FORMAT_RGB8, height_bytes);
		const Ref<Image> mask = Image::create_from_data(4, 4, false, Image::FORMAT_L8, mask_bytes);
		Array source_faces;
		source_faces.push_back(PackedInt32Array{ 5, 3 });
		ERR_FAIL_COND_V_MSG(builder->set_owner_displacement(0, PackedInt32Array{ 1 }, source_faces, Mesh::ARRAY_TEX_UV2, height_map, mask, 0.0f, 0.0008f, "ProbeDisplacement") != OK, Ref<SubdivisionBuilder>(), "The fingerprint probe's displacement was refused.");
	}
	builder->fingerprint_input = r_input;
	const Error err = builder->build(2);
	builder->fingerprint_input = nullptr;
	ERR_FAIL_COND_V_MSG(err != OK, Ref<SubdivisionBuilder>(), "The fingerprint probe did not build.");
	return builder;
}

String SubdivisionBuilder::fingerprint_of(const LocalVector<uint8_t> &p_input) {
	ERR_FAIL_COND_V(p_input.is_empty(), String());
	unsigned char hash[32];
	ERR_FAIL_COND_V(CryptoCore::sha256(p_input.ptr(), p_input.size(), hash) != OK, String());
	return String::hex_encode_buffer(hash, 32);
}

String SubdivisionBuilder::get_fingerprint() {
	static Mutex mutex;
	static String fingerprint;
	{
		MutexLock lock(mutex);
		if (!fingerprint.is_empty()) {
			return fingerprint;
		}
	}
	// Computed outside the lock: threads that both miss build the same bytes, and the first to store wins.
	// Nothing it hashes depends on a setting, so the value holds for the process.
	const uint64_t begin = OS::get_singleton()->get_ticks_usec();
	LocalVector<uint8_t> input;
	const String computed = build_fingerprint_probe(&input).is_valid() ? fingerprint_of(input) : String();
	print_verbose(vformat("SubdivisionBuilder: fingerprint %s from its probe in %.2f ms.", computed, double(OS::get_singleton()->get_ticks_usec() - begin) / 1000.0));
	MutexLock lock(mutex);
	if (fingerprint.is_empty()) {
		fingerprint = computed;
	}
	return fingerprint;
}

void SubdivisionBuilder::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_cage_vertex_count", "count"), &SubdivisionBuilder::set_cage_vertex_count);
	ClassDB::bind_method(D_METHOD("get_cage_vertex_count"), &SubdivisionBuilder::get_cage_vertex_count);
	ClassDB::bind_method(D_METHOD("set_blend_shape_mode", "mode"), &SubdivisionBuilder::set_blend_shape_mode);
	ClassDB::bind_method(D_METHOD("get_blend_shape_mode"), &SubdivisionBuilder::get_blend_shape_mode);
	ClassDB::bind_method(D_METHOD("add_face_varying_layer", "values", "array_slot"), &SubdivisionBuilder::add_face_varying_layer);
	ClassDB::bind_method(D_METHOD("add_surface", "owner", "arrays", "blend_arrays", "blend_shape_names", "cage_vertex", "faces", "uv_weld", "material"), &SubdivisionBuilder::add_surface);
	ClassDB::bind_method(D_METHOD("set_owner_bone_names", "owner", "bone_names"), &SubdivisionBuilder::set_owner_bone_names);
	ClassDB::bind_method(D_METHOD("get_owner_bone_names", "owner"), &SubdivisionBuilder::get_owner_bone_names);
	ClassDB::bind_method(D_METHOD("set_owner_displacement", "owner", "surfaces", "source_faces", "uv_slot", "height_map", "mask", "min_height", "max_height", "shape_name"), &SubdivisionBuilder::set_owner_displacement);
	ClassDB::bind_method(D_METHOD("clear_owner_displacement", "owner"), &SubdivisionBuilder::clear_owner_displacement);
	ClassDB::bind_method(D_METHOD("build", "level"), &SubdivisionBuilder::build, DEFVAL(2));
	ClassDB::bind_method(D_METHOD("get_owner_mesh", "owner"), &SubdivisionBuilder::get_owner_mesh);
	ClassDB::bind_method(D_METHOD("get_owner_triangle_sources", "owner"), &SubdivisionBuilder::get_owner_triangle_sources);
	ClassDB::bind_method(D_METHOD("get_report"), &SubdivisionBuilder::get_report);
	ClassDB::bind_method(D_METHOD("clear"), &SubdivisionBuilder::clear);
	ClassDB::bind_static_method("SubdivisionBuilder", D_METHOD("get_fingerprint"), &SubdivisionBuilder::get_fingerprint);
	ClassDB::bind_static_method("SubdivisionBuilder", D_METHOD("displacement_height", "height_map", "mask", "uv", "min_height", "max_height"), &SubdivisionBuilder::displacement_height);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "cage_vertex_count"), "set_cage_vertex_count", "get_cage_vertex_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "blend_shape_mode", PROPERTY_HINT_ENUM, "Normalized,Relative"), "set_blend_shape_mode", "get_blend_shape_mode");
}
