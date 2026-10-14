// The surface record (P3a, docs/p3a-raygen-shading.md): what shade_and_bounce reads of a hit and its
// material, packed into 23 words (92 bytes) so a hit shader can hand a surface back to the raygen
// instead of shading it. The colours, scalars and medium are fp16 (the G-buffer's own precision, so a
// G-buffer surface packs without loss), clamped to fp16's range, and a non-zero value that fp16 would
// flush to zero keeps fp16's smallest step, so no lobe's gate (a transmission, a coat, an emission)
// flips. The normals are 2 x 16-bit octahedral, as the payload's offset normal is. The hit position is
// not stored: the unpack rebuilds it as origin + direction x hit_t, the way compute_hit_data and the
// G-buffer primary build it.
//
// What the hit shader knows at compile time travels as data: the subsurface share is the effective
// one (0 outside a walking group, D120 SS2) and RT_SURFACE_CUSTOM_GROUP marks a custom hit group's
// surface, whose walk exit takes the entry's normal perturbation (subsurface_exit_shading_normal).
//
// A thin surface's diffuse transmission (D103) and a refracting interface (D24, D49) share five words:
// no hit group writes both (HG0 has no TRANSMISSION, a custom hit group and the G-buffer no BACKLIGHT),
// and RT_SURFACE_CODEC_CHECK reports a surface that would need them at once.
//
// A strand's hair (D126 H2) shares three words with the lobes a hair material clears: its melanin and
// redness take anisotropy_clearcoat, its IOR and cuticle tilt clearcoat_roughness_rim, and the strand's
// tangent, which the hair BCSDF needs and nothing else here keeps, anisotropy_tangent (under
// RT_SURFACE_HAIR_TANGENT).
//
// Defined for a degenerate surface, so every caller shades it the same way and emission, the media and
// the RR guides stay inside shade_and_bounce: a shading normal that is not finite or has no length
// becomes the geometry normal; a geometry normal that is not finite or has no length becomes the
// normal facing the ray (-ray_dir, RT_SURFACE_INVALID), and so does a shading normal that fell back to
// it. A colour or scalar that is not finite packs as 0.
//
// Required includes (before this file): raytracing_inc.glsl, raytracing_material_eval_inc.glsl,
// raytracing_shade_inc.glsl (HitData, rt_subsurface_walk_group).

#define RT_SURFACE_FRONT_FACE 1u
#define RT_SURFACE_THIN 2u // The interface words hold the diffuse transmission, not a refraction.
#define RT_SURFACE_SKIN 4u // MaterialResult.subsurface_skin.
#define RT_SURFACE_ANISOTROPIC 8u // anisotropy_tangent is valid.
#define RT_SURFACE_CLEARCOAT 16u // clearcoat_normal is valid.
#define RT_SURFACE_CUSTOM_GROUP 32u // HitData.custom_group: a custom hit group's surface.
#define RT_SURFACE_INVALID 64u // The geometry normal is not a direction: it unpacks as the normal facing the ray.
#define RT_SURFACE_SHADING_FROM_GEOMETRY 128u // The shading normal is not a direction: it unpacks as the geometry normal.
#define RT_SURFACE_HAIR 256u // MaterialResult.hair: three words hold the hair (above), not the anisotropy, coat and rim.
#define RT_SURFACE_HAIR_COLOR 512u // MaterialResult.hair_color.
#define RT_SURFACE_HAIR_TANGENT 1024u // Under RT_SURFACE_HAIR, anisotropy_tangent holds the strand's tangent.

struct SurfaceRecord {
	float hit_t;
	uint flags; // RT_SURFACE_*
	uint geometry_normal; // Octahedral 2 x 16.
	uint shading_normal; // Octahedral 2 x 16.
	uint albedo_rg;
	uint albedo_b_roughness;
	uint metalness_specular;
	uint emissive_rg;
	uint emissive_b_decal_r;
	uint decal_gb;
	// Transmission rgb, ior, absorption rgb, scattering rgb; or, on a thin surface, the diffuse transmission rgb.
	uint interface_0;
	uint interface_1;
	uint interface_2;
	uint interface_3;
	uint interface_4;
	uint anisotropy_clearcoat;
	uint clearcoat_roughness_rim;
	uint rim_tint_subsurface; // The effective subsurface share.
	// Octahedral 2 x 16, under RT_SURFACE_ANISOTROPIC. A direction only: both producers normalise it
	// (rt_anisotropy_tangent), so shade_and_bounce's projection decides its fallback as before; the check verifies it.
	uint anisotropy_tangent;
	uint clearcoat_normal; // Octahedral 2 x 16, under RT_SURFACE_CLEARCOAT.
	uint geometry_idx; // For the emission's weight (D50): the entry and the triangle the light sampling would have drawn.
	uint primitive_id;
	uint texture_lods; // texture_lod and texture_lod2 (D116), the walk's exit reads them (D120).
};

const float RT_SURFACE_HALF_MAX = 65504.0;
const float RT_SURFACE_HALF_MIN = 5.9604645e-8; // fp16's smallest step, 2^-24.

bool surface_is_direction(vec3 p_v) {
	return !any(isnan(p_v)) && !any(isinf(p_v)) && dot(p_v, p_v) > 1e-20;
}

bool surface_is_finite(float p_v) {
	return !isnan(p_v) && !isinf(p_v);
}

/// A value fp16 keeps on the same side of zero: 0 if it is not finite (clamp leaves a NaN undefined), clamped to fp16's
/// range, and a non-zero value at least its smallest step. That step is a subnormal, which the floor relies on fp16
/// keeping (packHalf2x16 does on NVIDIA; the check's gate test would report a flush).
float surface_half_value(float p_v) {
	float v = surface_is_finite(p_v) ? clamp(p_v, -RT_SURFACE_HALF_MAX, RT_SURFACE_HALF_MAX) : 0.0;
	return v > 0.0 ? max(v, RT_SURFACE_HALF_MIN) : (v < 0.0 ? min(v, -RT_SURFACE_HALF_MIN) : v);
}

uint surface_pack_half2(float p_a, float p_b) {
	return packHalf2x16(vec2(surface_half_value(p_a), surface_half_value(p_b)));
}

uint surface_pack_oct(vec3 p_direction) {
	return packUnorm2x16(vec3_to_oct(p_direction));
}

vec3 surface_unpack_oct(uint p_packed) {
	return oct_to_vec3(unpackUnorm2x16(p_packed) * 2.0 - 1.0);
}

/// The share of the diffuse lobe a surface walks with: its SSS_STRENGTH in a walking group, 0 elsewhere (D120 SS2).
float surface_subsurface_share(MaterialResult p_mat) {
#ifdef RT_SUBSURFACE_COMPILED
	return rt_subsurface_walk_group() ? p_mat.subsurface : 0.0;
#else
	return 0.0;
#endif
}

SurfaceRecord surface_pack(HitData p_hit, MaterialResult p_mat, float p_hit_t) {
	SurfaceRecord r;
	r.hit_t = p_hit_t;
	const bool geometry_valid = surface_is_direction(p_hit.geometry_normal);
	const bool shading_valid = surface_is_direction(p_mat.normal);
	// The unpack replaces what is not a direction (the flags say which); any valid encoding stands in until then.
	const vec3 geometry_normal = geometry_valid ? p_hit.geometry_normal : vec3(0.0, 0.0, 1.0);
	const vec3 shading_normal = shading_valid ? p_mat.normal : geometry_normal;
	const bool thin = any(greaterThan(p_mat.diffuse_transmission, vec3(0.0)));
	const bool anisotropic = p_mat.anisotropy != 0.0 && surface_is_direction(p_mat.anisotropy_tangent);
	const bool clearcoat = p_mat.clearcoat > 0.0 && surface_is_direction(p_mat.clearcoat_normal);
	const bool hair_tangent = p_mat.hair && surface_is_direction(p_hit.tangent);
	r.flags = (p_hit.is_front_face ? RT_SURFACE_FRONT_FACE : 0u) | (thin ? RT_SURFACE_THIN : 0u) | (p_mat.subsurface_skin ? RT_SURFACE_SKIN : 0u) | (anisotropic ? RT_SURFACE_ANISOTROPIC : 0u) | (clearcoat ? RT_SURFACE_CLEARCOAT : 0u) | (p_hit.custom_group ? RT_SURFACE_CUSTOM_GROUP : 0u) | (geometry_valid ? 0u : RT_SURFACE_INVALID) | (shading_valid ? 0u : RT_SURFACE_SHADING_FROM_GEOMETRY) | (p_mat.hair ? RT_SURFACE_HAIR : 0u) | (p_mat.hair_color ? RT_SURFACE_HAIR_COLOR : 0u) | (hair_tangent ? RT_SURFACE_HAIR_TANGENT : 0u);
	r.geometry_normal = surface_pack_oct(geometry_normal);
	r.shading_normal = surface_pack_oct(shading_normal);
	r.albedo_rg = surface_pack_half2(p_mat.albedo.r, p_mat.albedo.g);
	r.albedo_b_roughness = surface_pack_half2(p_mat.albedo.b, p_mat.roughness);
	r.metalness_specular = surface_pack_half2(p_mat.metalness, p_mat.specular);
	r.emissive_rg = surface_pack_half2(p_mat.emissive.r, p_mat.emissive.g);
	r.emissive_b_decal_r = surface_pack_half2(p_mat.emissive.b, p_mat.decal_emissive.r);
	r.decal_gb = surface_pack_half2(p_mat.decal_emissive.g, p_mat.decal_emissive.b);
	if (thin) {
		r.interface_0 = surface_pack_half2(p_mat.diffuse_transmission.r, p_mat.diffuse_transmission.g);
		r.interface_1 = surface_pack_half2(p_mat.diffuse_transmission.b, 0.0);
		r.interface_2 = 0u;
		r.interface_3 = 0u;
		r.interface_4 = 0u;
	} else {
		r.interface_0 = surface_pack_half2(p_mat.transmission.r, p_mat.transmission.g);
		r.interface_1 = surface_pack_half2(p_mat.transmission.b, p_mat.ior);
		r.interface_2 = surface_pack_half2(p_mat.absorption.r, p_mat.absorption.g);
		r.interface_3 = surface_pack_half2(p_mat.absorption.b, p_mat.scattering.r);
		r.interface_4 = surface_pack_half2(p_mat.scattering.g, p_mat.scattering.b);
	}
	r.anisotropy_clearcoat = surface_pack_half2(p_mat.anisotropy, p_mat.clearcoat);
	r.clearcoat_roughness_rim = surface_pack_half2(p_mat.clearcoat_roughness, p_mat.rim);
	r.rim_tint_subsurface = surface_pack_half2(p_mat.rim_tint, surface_subsurface_share(p_mat));
	r.anisotropy_tangent = anisotropic ? surface_pack_oct(p_mat.anisotropy_tangent) : 0u;
	if (p_mat.hair) {
		// A hair material's anisotropy, coat and rim are zero (D126 H2), so their words carry the hair.
		r.anisotropy_clearcoat = surface_pack_half2(p_mat.hair_melanin, p_mat.hair_redness);
		r.clearcoat_roughness_rim = surface_pack_half2(p_mat.hair_ior, p_mat.hair_tilt);
		r.anisotropy_tangent = hair_tangent ? surface_pack_oct(p_hit.tangent) : 0u;
	}
	r.clearcoat_normal = clearcoat ? surface_pack_oct(p_mat.clearcoat_normal) : 0u;
	r.geometry_idx = p_hit.geometry_idx;
	r.primitive_id = p_hit.primitive_id;
	r.texture_lods = surface_pack_half2(p_hit.texture_lod, p_hit.texture_lod2);
	return r;
}

/// The hit and material shade_and_bounce reads, from a record of the surface p_ray_origin + p_ray_dir x hit_t. The fields
/// only a hit shader reads (the tangent frame, the UVs, the vertex colour, the footprint and gradients) come back empty.
void surface_unpack(SurfaceRecord p_record, vec3 p_ray_origin, vec3 p_ray_dir, out HitData r_hit, out MaterialResult r_mat) {
	r_hit.hit_pos = p_ray_origin + p_ray_dir * p_record.hit_t;
	r_hit.geometry_normal = (p_record.flags & RT_SURFACE_INVALID) != 0u ? -p_ray_dir : surface_unpack_oct(p_record.geometry_normal);
	r_hit.tangent = vec3(0.0);
	r_hit.bitangent = vec3(0.0);
	r_hit.uv = vec2(0.0);
	r_hit.uv2 = vec2(0.0);
	r_hit.color = vec4(1.0);
	r_hit.is_front_face = (p_record.flags & RT_SURFACE_FRONT_FACE) != 0u;
	r_hit.custom_group = (p_record.flags & RT_SURFACE_CUSTOM_GROUP) != 0u;
	r_hit.geometry_idx = p_record.geometry_idx;
	r_hit.primitive_id = p_record.primitive_id;
	const vec2 lods = unpackHalf2x16(p_record.texture_lods);
	r_hit.texture_lod = lods.x;
	r_hit.texture_footprint = RT_TEXTURE_LOD_BASE;
	r_hit.texture_grad = vec4(0.0);
	r_hit.texture_lod2 = lods.y;
	r_hit.texture_grad2 = vec4(0.0);

	const vec2 albedo_rg = unpackHalf2x16(p_record.albedo_rg);
	const vec2 albedo_b_roughness = unpackHalf2x16(p_record.albedo_b_roughness);
	const vec2 metalness_specular = unpackHalf2x16(p_record.metalness_specular);
	const vec2 emissive_rg = unpackHalf2x16(p_record.emissive_rg);
	const vec2 emissive_b_decal_r = unpackHalf2x16(p_record.emissive_b_decal_r);
	const vec2 decal_gb = unpackHalf2x16(p_record.decal_gb);
	r_mat.albedo = vec3(albedo_rg, albedo_b_roughness.x);
	r_mat.alpha = 1.0;
	r_mat.roughness = albedo_b_roughness.y;
	r_mat.metalness = metalness_specular.x;
	r_mat.specular = metalness_specular.y;
	r_mat.emissive = vec3(emissive_rg, emissive_b_decal_r.x);
	r_mat.decal_emissive = vec3(emissive_b_decal_r.y, decal_gb);
	r_mat.normal = (p_record.flags & RT_SURFACE_SHADING_FROM_GEOMETRY) != 0u ? r_hit.geometry_normal : surface_unpack_oct(p_record.shading_normal);
	const vec2 i0 = unpackHalf2x16(p_record.interface_0);
	const vec2 i1 = unpackHalf2x16(p_record.interface_1);
	if ((p_record.flags & RT_SURFACE_THIN) != 0u) {
		r_mat.transmission = vec3(0.0);
		r_mat.ior = 1.5;
		r_mat.absorption = vec3(0.0);
		r_mat.scattering = vec3(0.0);
		r_mat.diffuse_transmission = vec3(i0, i1.x);
	} else {
		const vec2 i2 = unpackHalf2x16(p_record.interface_2);
		const vec2 i3 = unpackHalf2x16(p_record.interface_3);
		const vec2 i4 = unpackHalf2x16(p_record.interface_4);
		r_mat.transmission = vec3(i0, i1.x);
		r_mat.ior = i1.y;
		r_mat.absorption = vec3(i2, i3.x);
		r_mat.scattering = vec3(i3.y, i4);
		r_mat.diffuse_transmission = vec3(0.0);
	}
	const vec2 anisotropy_clearcoat = unpackHalf2x16(p_record.anisotropy_clearcoat);
	const vec2 clearcoat_roughness_rim = unpackHalf2x16(p_record.clearcoat_roughness_rim);
	const vec2 rim_tint_subsurface = unpackHalf2x16(p_record.rim_tint_subsurface);
	r_mat.anisotropy = anisotropy_clearcoat.x;
	r_mat.anisotropy_tangent = (p_record.flags & RT_SURFACE_ANISOTROPIC) != 0u ? surface_unpack_oct(p_record.anisotropy_tangent) : vec3(0.0);
	r_mat.clearcoat = anisotropy_clearcoat.y;
	r_mat.clearcoat_roughness = clearcoat_roughness_rim.x;
	r_mat.clearcoat_normal = (p_record.flags & RT_SURFACE_CLEARCOAT) != 0u ? surface_unpack_oct(p_record.clearcoat_normal) : r_hit.geometry_normal;
	r_mat.rim = clearcoat_roughness_rim.y;
	r_mat.rim_tint = rim_tint_subsurface.x;
	r_mat.subsurface = rim_tint_subsurface.y;
	r_mat.subsurface_skin = (p_record.flags & RT_SURFACE_SKIN) != 0u;
	material_clear_hair(r_mat);
	if ((p_record.flags & RT_SURFACE_HAIR) != 0u) {
		r_mat.hair = true;
		r_mat.hair_color = (p_record.flags & RT_SURFACE_HAIR_COLOR) != 0u;
		r_mat.hair_melanin = anisotropy_clearcoat.x;
		r_mat.hair_redness = anisotropy_clearcoat.y;
		r_mat.hair_ior = clearcoat_roughness_rim.x;
		r_mat.hair_tilt = clearcoat_roughness_rim.y;
		r_mat.anisotropy = 0.0;
		r_mat.clearcoat = 0.0;
		r_mat.clearcoat_roughness = 0.0;
		r_mat.rim = 0.0;
		r_hit.tangent = (p_record.flags & RT_SURFACE_HAIR_TANGENT) != 0u ? surface_unpack_oct(p_record.anisotropy_tangent) : vec3(0.0);
	}
}

#if defined(RT_SURFACE_CODEC_CHECK) && (defined(RT_STAGE_CLOSEST_HIT) || defined(RT_STAGE_RAYGEN))
// The codec against every surface an asserts build shades (P3a), the closest hits' and the G-buffer primary's: each is
// packed and unpacked beside the one that is shaded, and every field shade_and_bounce reads is compared within its
// encoding's rounding, a lobe's gate (whether it is there at all) exactly. Words: surfaces compared (one pixel in 64);
// then in every pixel, surfaces with a field outside it, the fields that were (the bits below, OR'd), shading normals
// the record replaces by the geometry normal, geometry normals it replaces by the normal facing the ray, and surfaces
// with a colour or scalar that is not finite (packed as 0). The host reports a mismatch or a non-finite input as an
// error and clears the words after each read.
layout(set = 0, binding = 55, std430) buffer SurfaceCodecCheck {
	uint words[];
}
surface_codec_check;

bool surface_check_half(float p_a, float p_b) {
	// fp16 keeps 11 significant bits; below its normal range, 2^-24 steps, and a non-zero value keeps one of them. A value
	// that is not finite packs as 0 (counted on its own).
	float a = surface_is_finite(p_a) ? clamp(p_a, -RT_SURFACE_HALF_MAX, RT_SURFACE_HALF_MAX) : 0.0;
	return abs(a - p_b) <= abs(a) * (1.0 / 1024.0) + 1e-7;
}

bool surface_check_half3(vec3 p_a, vec3 p_b) {
	return surface_check_half(p_a.x, p_b.x) && surface_check_half(p_a.y, p_b.y) && surface_check_half(p_a.z, p_b.z);
}

bool surface_check_direction(vec3 p_a, vec3 p_b) {
	// 2 x 16-bit octahedral rounds a direction by about 0.002 degrees; the bound is about 0.08.
	return dot(normalize(p_a), p_b) >= 0.999999;
}

bool surface_check_gate(float p_a, float p_b) {
	return (p_a > 0.0) == (p_b > 0.0);
}

bool surface_check_gate3(vec3 p_a, vec3 p_b) {
	// The lobes test their largest channel (transmit_max > 0) or any channel (a medium entered), the same gate.
	return any(greaterThan(p_a, vec3(0.0))) == any(greaterThan(p_b, vec3(0.0)));
}

void surface_codec_check_hit(HitData p_hit, MaterialResult p_mat, vec3 p_ray_origin, vec3 p_ray_dir, float p_hit_t) {
	const SurfaceRecord record = surface_pack(p_hit, p_mat, p_hit_t);
	HitData h;
	MaterialResult m;
	surface_unpack(record, p_ray_origin, p_ray_dir, h, m);
	const bool geometry_valid = surface_is_direction(p_hit.geometry_normal);
	const bool shading_valid = surface_is_direction(p_mat.normal);
	uint bad = 0u;
	bad |= length(h.hit_pos - p_hit.hit_pos) <= 1e-6 * max(1.0, length(p_hit.hit_pos)) ? 0u : 1u;
	// A degenerate geometry normal comes back as the normal facing the ray, a degenerate shading normal as the geometry
	// normal.
	const vec3 expected_geometry = geometry_valid ? p_hit.geometry_normal : -p_ray_dir;
	bad |= (surface_check_direction(expected_geometry, h.geometry_normal) && ((record.flags & RT_SURFACE_INVALID) != 0u) == !geometry_valid) ? 0u : 2u;
	bad |= surface_check_direction(shading_valid ? p_mat.normal : expected_geometry, m.normal) ? 0u : 4u;
	bad |= surface_check_half3(p_mat.albedo, m.albedo) ? 0u : 8u;
	bad |= (surface_check_half(p_mat.roughness, m.roughness) && surface_check_half(p_mat.metalness, m.metalness) && surface_check_half(p_mat.specular, m.specular)) ? 0u : 16u;
	bad |= surface_check_half3(p_mat.emissive, m.emissive) ? 0u : 32u;
	bad |= surface_check_half3(p_mat.decal_emissive, m.decal_emissive) ? 0u : 64u;
	const bool transmits = any(greaterThan(p_mat.transmission, vec3(0.0)));
	bad |= (surface_check_half3(p_mat.transmission, m.transmission) && (!transmits || surface_check_half(p_mat.ior, m.ior))) ? 0u : 128u;
	bad |= (surface_check_half3(p_mat.absorption, m.absorption) && surface_check_half3(p_mat.scattering, m.scattering)) ? 0u : 256u;
	bad |= surface_check_half3(p_mat.diffuse_transmission, m.diffuse_transmission) ? 0u : 512u;
	// The tangent is read only under the lobe and as a direction; both producers normalise it, which the record relies on.
	const bool tangent_read = p_mat.anisotropy != 0.0 && surface_is_direction(p_mat.anisotropy_tangent);
	bad |= (surface_check_half(p_mat.anisotropy, m.anisotropy) && (!tangent_read || (abs(length(p_mat.anisotropy_tangent) - 1.0) <= 1e-3 && surface_check_direction(p_mat.anisotropy_tangent, m.anisotropy_tangent)))) ? 0u : 1024u;
	const bool coat_normal_read = p_mat.clearcoat > 0.0 && surface_is_direction(p_mat.clearcoat_normal);
	bad |= (surface_check_half(p_mat.clearcoat, m.clearcoat) && surface_check_half(p_mat.clearcoat_roughness, m.clearcoat_roughness) && (!coat_normal_read || surface_check_direction(p_mat.clearcoat_normal, m.clearcoat_normal))) ? 0u : 2048u;
	bad |= (surface_check_half(p_mat.rim, m.rim) && surface_check_half(p_mat.rim_tint, m.rim_tint)) ? 0u : 4096u;
	const float share = surface_subsurface_share(p_mat);
	bad |= (surface_check_half(share, m.subsurface) && p_mat.subsurface_skin == m.subsurface_skin) ? 0u : 8192u;
	bad |= (p_hit.is_front_face == h.is_front_face && p_hit.custom_group == h.custom_group) ? 0u : 16384u;
	bad |= (p_hit.geometry_idx == h.geometry_idx && p_hit.primitive_id == h.primitive_id) ? 0u : 32768u;
	bad |= (surface_check_half(p_hit.texture_lod, h.texture_lod) && surface_check_half(p_hit.texture_lod2, h.texture_lod2)) ? 0u : 65536u;
	// Hair (D126 H2): the flags, its parameters and the strand's tangent (or none, when it had none).
	const bool hair_tangent_valid = surface_is_direction(p_hit.tangent);
	bad |= (p_mat.hair == m.hair && p_mat.hair_color == m.hair_color && (!p_mat.hair || (surface_check_half(p_mat.hair_melanin, m.hair_melanin) && surface_check_half(p_mat.hair_redness, m.hair_redness) && surface_check_half(p_mat.hair_ior, m.hair_ior) && surface_check_half(p_mat.hair_tilt, m.hair_tilt) && (hair_tangent_valid ? surface_check_direction(p_hit.tangent, h.tangent) : h.tangent == vec3(0.0))))) ? 0u : 262144u;
	// Every lobe's gate, exactly: a transmission, a thin transmission, a medium, an emission, a coat, a sheen, a walk, an
	// anisotropy present or not.
	const bool gates = surface_check_gate3(p_mat.transmission, m.transmission) && surface_check_gate3(p_mat.diffuse_transmission, m.diffuse_transmission) && surface_check_gate3(p_mat.absorption, m.absorption) && surface_check_gate3(p_mat.emissive, m.emissive) && surface_check_gate3(p_mat.decal_emissive, m.decal_emissive) && surface_check_gate(p_mat.clearcoat, m.clearcoat) && surface_check_gate(p_mat.rim, m.rim) && surface_check_gate(share, m.subsurface) && ((p_mat.anisotropy != 0.0) == (m.anisotropy != 0.0));
	bad |= gates ? 0u : 131072u;

	// Any colour or scalar the record carries that is not finite.
	const vec3 sums = p_mat.albedo + p_mat.emissive + p_mat.decal_emissive + p_mat.transmission + p_mat.absorption + p_mat.scattering + p_mat.diffuse_transmission;
	const float scalars = p_mat.roughness + p_mat.metalness + p_mat.specular + p_mat.ior + p_mat.anisotropy + p_mat.clearcoat + p_mat.clearcoat_roughness + p_mat.rim + p_mat.rim_tint + share + p_hit.texture_lod + p_hit.texture_lod2;
	const bool non_finite = !surface_is_finite(sums.x + sums.y + sums.z + scalars);
	const bool sampled = ((gl_LaunchIDEXT.x | gl_LaunchIDEXT.y) & 7u) == 0u;
	const bool fallback = !shading_valid;
#ifdef RT_STAGE_CLOSEST_HIT
	// Summed per subgroup: every hit of a frame passes here.
	const uint compared = subgroupAdd(sampled ? 1u : 0u);
	const uint mismatched = subgroupAdd(bad != 0u ? 1u : 0u);
	const uint fields = subgroupOr(bad);
	const uint fallbacks = subgroupAdd(fallback ? 1u : 0u);
	const uint invalid = subgroupAdd(geometry_valid ? 0u : 1u);
	const uint non_finites = subgroupAdd(non_finite ? 1u : 0u);
	const bool writer = subgroupElect();
#else
	// The G-buffer primary: one surface a pixel and sample. With subgroup operations here the asserts build lost the
	// device on the G-buffer runs (a driver issue, not characterised), so each invocation writes its own.
	const uint compared = sampled ? 1u : 0u;
	const uint mismatched = bad != 0u ? 1u : 0u;
	const uint fields = bad;
	const uint fallbacks = fallback ? 1u : 0u;
	const uint invalid = geometry_valid ? 0u : 1u;
	const uint non_finites = non_finite ? 1u : 0u;
	const bool writer = true;
#endif
	if (writer) {
		if (compared != 0u) {
			atomicAdd(surface_codec_check.words[0], compared);
		}
		if (mismatched != 0u) {
			atomicAdd(surface_codec_check.words[1], mismatched);
			atomicOr(surface_codec_check.words[2], fields);
		}
		if (fallbacks != 0u) {
			atomicAdd(surface_codec_check.words[3], fallbacks);
		}
		if (invalid != 0u) {
			atomicAdd(surface_codec_check.words[4], invalid);
		}
		if (non_finites != 0u) {
			atomicAdd(surface_codec_check.words[5], non_finites);
		}
	}
}
#endif
