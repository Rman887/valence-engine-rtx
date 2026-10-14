// Shared data structures for raytracing shaders.
// Requires: GL_EXT_buffer_reference, GL_ARB_gpu_shader_int64, raytracing_inc.glsl (for OFFSET_NONE/FLAG_*)

// ============================================================================
// BUFFER REFERENCES
// ============================================================================
layout(buffer_reference, std430) readonly buffer FloatBuffer {
	float v[];
};
layout(buffer_reference, std430) readonly buffer Uint32Buffer {
	uint v[];
};

// ============================================================================
// GEOMETRY DATA (matches C++ RT_GeometryData, 128 bytes)
// ============================================================================
struct GeometryData {
	uint64_t vertex_address;
	uint64_t attribute_address;
	uint64_t index_address;

	uint vertex_count;
	uint position_stride;
	uint normal_byte_offset;
	uint normal_stride;
	uint tangent_byte_offset;
	uint tangent_stride;

	uint attribute_stride;
	uint uv_byte_offset;
	uint uv_scale_packed; // fp16 x, fp16 y
	uint index_format;
	uint primitive_count;
	uint flags;
	float aabb_size_x;
	float aabb_size_y;
	float aabb_size_z;

	// Byte offset of the vertex color attribute inside `attribute_stride`,
	// or OFFSET_NONE if the mesh has no vertex colors.
	uint color_byte_offset;

	float aabb_pos_x;
	float aabb_pos_y;
	float aabb_pos_z;

	// For deformed geometry: previous-frame position buffer used for motion vectors.
	uint prev_vertex_address_lo;
	uint prev_vertex_address_hi;

	// With FLAG_CLUSTER_GEOMETRY (D93b): the surface's first cluster geometry record, in words.
	uint cluster_record_base;
	uint _pad[2];
	// The second UV set (D118): its byte offset inside `attribute_stride` or OFFSET_NONE, and its scale (fp16 x, fp16 y).
	uint uv2_byte_offset;
	uint uv2_scale_packed;
};

void get_aabb_compression_xforms(GeometryData geom, out mat4 aabb_xform, out mat4 inv_aabb_xform) {
	if ((geom.flags & FLAG_COMPRESSED) == 0u) {
		aabb_xform = mat4(1.0);
		inv_aabb_xform = mat4(1.0);
		return;
	}
	vec3 aabb_sz = vec3(geom.aabb_size_x, geom.aabb_size_y, geom.aabb_size_z);
	vec3 aabb_po = vec3(geom.aabb_pos_x, geom.aabb_pos_y, geom.aabb_pos_z);
	aabb_xform = mat4(
			vec4(aabb_sz.x, 0.0, 0.0, 0.0),
			vec4(0.0, aabb_sz.y, 0.0, 0.0),
			vec4(0.0, 0.0, aabb_sz.z, 0.0),
			vec4(aabb_po, 1.0));
	vec3 inv_sz = 1.0 / aabb_sz;
	inv_aabb_xform = mat4(
			vec4(inv_sz.x, 0.0, 0.0, 0.0),
			vec4(0.0, inv_sz.y, 0.0, 0.0),
			vec4(0.0, 0.0, inv_sz.z, 0.0),
			vec4(-aabb_po * inv_sz, 1.0));
}

// ============================================================================
// PER-INSTANCE MOTION DATA (matches C++ RT_InstanceMotionData, 48 bytes)
// ============================================================================
struct InstanceMotionData {
	float prev_xform[12]; // Previous object-to-world (mat3x4, transposed 3x4)
};

// ============================================================================
// MATERIAL DATA (matches C++ layout, 176 bytes)
// ============================================================================
struct MaterialData {
	uint albedo_texture_idx;
	uint normal_texture_idx;
	uint orm_texture_idx; // The ORM texture, or StandardMaterial3D's roughness texture (D118).
	uint emission_texture_idx;

	vec4 albedo_color;
	vec3 emission_color;
	float emission_strength;

	float metallic;
	float roughness;
	uint hair_ior_tilt; // With flag bit 20: packHalf2x16(hair_ior, the cuticle tilt in radians) (D126 H2).
	uint flags; // Bit 0: has_normal_map, Bit 1: has_emission_tex, Bit 2: point_filter, Bit 3: emission_multiply, Bit 4: emissive_constant (an emissive mesh light, D50), Bit 5: diffuse_transmission (backlight, D103), Bits 8-10 and 11-13: the roughness and metallic textures' channels (D118), Bit 14: detail layer, Bits 15-16: its blend mode, Bit 17: its textures on UV2, Bit 18: the emission texture on UV2 (D118 MM4), Bit 19: subsurface skin mode (D120), Bit 20: hair (RT_MAT_FLAG_HAIR, D126 H2), Bit 21: its absorption from the albedo (RT_MAT_FLAG_HAIR_COLOR), Bit 22: the vertex colour as albedo (RT_MAT_FLAG_VERTEX_COLOR, D126 H3), Bit 23: that colour in sRGB, Bit 24: hashed alpha (RT_MAT_FLAG_ALPHA_HASH, D128)

	vec2 uv1_scale; // UV1 scale (default 1,1)
	vec2 uv1_offset; // UV1 offset (default 0,0)

	float normal_map_depth; // Normal map strength (default 1.0)
	float specular; // Dielectric specular [0..1], default 0.5 -> F0 = 0.04.
	uint64_t uniform_address; // BDA for custom shader uniform buffer (0 = none)
	float alpha_threshold; // The alpha test's cutoff: opaque where alpha >= it (default 0.5).
	uint backlight_rgba8; // With flag bit 5: the backlight colour, sRGB, RGBA8 (D103).
	uint backlight_texture_idx; // Its backlight texture, sampled linear (0 = none).
	uint metallic_texture_idx; // The ORM texture again, or StandardMaterial3D's metallic texture (D118).
	float anisotropy; // BaseMaterial3D's anisotropy_ratio (D118 MM1); 0 = an isotropic specular lobe.
	uint anisotropy_flowmap_idx; // Its flow map, read linear (0 = none: along the tangent, at full strength).
	uint clearcoat_half2; // packHalf2x16(clearcoat, clearcoat_roughness), BaseMaterial3D's coat (D118 MM2); 0 = no coat.
	uint clearcoat_texture_idx; // Its texture, read linear: red scales the strength, green the roughness (0 = none).
	uint rim_half2; // packHalf2x16(rim, rim_tint), BaseMaterial3D's rim read as a sheen layer (D118 MM3); 0 = no rim.
	uint rim_texture_idx; // Its texture, read linear: red scales the rim, green the tint (0 = none).
	vec2 uv2_scale; // BaseMaterial3D's uv2_scale and uv2_offset, read for a detail layer on UV2 (D118 MM4).
	vec2 uv2_offset;
	uint detail_albedo_texture_idx; // With flag bit 14: the detail albedo, sRGB (0 = none: the white default).
	uint detail_normal_texture_idx; // Its normal map, read linear (0 = none: flat).
	uint detail_mask_texture_idx; // Its mask, red read linear on UV1 (0 = none: the white default, everywhere).
	float subsurface_strength; // BaseMaterial3D's subsurf_scatter_strength, the diffuse share that walks under the surface (D120); 0 = none.
	uint subsurface_texture_idx; // Its texture, red read linear on UV1, scaling the strength (0 = none).
	uint hair_melanin_half2; // With flag bit 20: packHalf2x16(hair_melanin, hair_melanin_redness) (D126 H2).
};

#define RT_MAT_FLAG_HAIR (1u << 20) // BaseMaterial3D's hair feature (D126 H2): a strand hit shades with the hair BCSDF.
#define RT_MAT_FLAG_HAIR_COLOR (1u << 21) // Its absorption from the albedo instead of the melanin.
#define RT_MAT_FLAG_VERTEX_COLOR (1u << 22) // BaseMaterial3D's vertex_color_use_as_albedo (D126 H3): albedo and alpha times the vertex colour.
#define RT_MAT_FLAG_VERTEX_COLOR_SRGB (1u << 23) // Its vertex_color_is_srgb: the colour converted to linear first.
#define RT_MAT_FLAG_ALPHA_HASH (1u << 24) // BaseMaterial3D's TRANSPARENCY_ALPHA_HASH (D128): the alpha test is hashed, rt_alpha_hash_u.
