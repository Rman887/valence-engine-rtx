// Material evaluation result - common output format for all hit groups.
// Standard materials and custom ShaderMaterials both produce this struct.

struct MaterialResult {
	vec3 albedo;
	float alpha;
	float roughness;
	float metalness;
	float specular; // Dielectric specular reflectance control [0..1], default 0.5 -> F0 = 0.04.
	vec3 emissive;
	vec3 decal_emissive; // Decals' emission (D104), which no light sample reaches, so it is added at full weight; with the exposure normalisation.
	vec3 normal; // Final shading normal (world space, after normal mapping).
	vec3 transmission; // Fraction of the unreflected light that continues through the surface, per channel; 0 = opaque.
	float ior; // Refractive index of the medium behind the surface, for the refracted direction and the interface's Fresnel.
	vec3 absorption; // Per-metre extinction of the medium behind a front face, per channel; 0 = no medium (D49).
	vec3 scattering; // The medium's single-scatter colour, 0..1: what a segment in it adds of the sun and sky (D49).
	vec3 diffuse_transmission; // Albedo of a thin diffuse-transmission lobe, 0..1: what a leaf passes to its other side, BACKLIGHT times the diffuse albedo (D103).
	float anisotropy; // VAL: the specular lobe's anisotropy as Godot's ANISOTROPY, at most 1; 0 = isotropic (D118 MM1).
	vec3 anisotropy_tangent; // World space, the direction its roughness stretches along: the tangent turned by ANISOTROPY_FLOW, not yet orthogonal to the normal.
	float clearcoat; // VAL: the coat's strength as Godot's CLEARCOAT, 0..1; 0 = no coat (D118 MM2).
	float clearcoat_roughness; // Godot's CLEARCOAT_ROUGHNESS, 0..1, mapped to the coat's alpha as the raster maps it.
	vec3 clearcoat_normal; // World space, the coat's normal: the vertex normal (a custom shader's NORMAL) without the normal map.
	float rim; // VAL: Godot's RIM, 0..1, read as the strength of a sheen layer; 0 = none (D118 MM3).
	float rim_tint; // Godot's RIM_TINT, 0..1: how far the sheen's colour moves from white to the albedo.
};

/// Sensible default for a mid-grey diffuse surface.
MaterialResult default_material_result(vec3 geometry_normal) {
	MaterialResult r;
	r.albedo = vec3(0.8);
	r.alpha = 1.0;
	r.roughness = 0.5;
	r.metalness = 0.0;
	r.specular = 0.5;
	r.emissive = vec3(0.0);
	r.decal_emissive = vec3(0.0);
	r.normal = geometry_normal;
	r.transmission = vec3(0.0);
	r.ior = 1.5;
	r.absorption = vec3(0.0);
	r.scattering = vec3(0.0);
	r.diffuse_transmission = vec3(0.0);
	r.anisotropy = 0.0;
	r.anisotropy_tangent = vec3(0.0);
	r.clearcoat = 0.0;
	r.clearcoat_roughness = 0.0;
	r.clearcoat_normal = geometry_normal;
	r.rim = 0.0;
	r.rim_tint = 0.0;
	return r;
}
