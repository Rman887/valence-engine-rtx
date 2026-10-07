// Texture LOD by ray cones (D116), after RTXPT's TexLODHelpers and the JCGT 2021 paper
// "Improved Shader and Texture Level of Detail Using Ray Cones". Every path ray carries a cone:
// its width at the ray's origin and its full spread angle (the payload's packed_cone). The
// width grows by spread x distance; the spread starts as the pixel's angle at render resolution
// and grows only where a non-delta lobe scatters, by the pdf heuristic RTXPT and vk_lod_clusters
// share. A hit turns the width into a texture-independent level from the triangle's UV and world
// areas; each fetch adds half the log2 of its texture's texel count.
//
// Required includes (before this file): raytracing_inc.glsl, raytracing_common_inc.glsl.

// The debug view of the level a 1024 x 1024 texture gets at the primary hit
// (Environment.RT_DEBUG_TEXTURE_LOD).
#define RT_VIS_MODE_TEXTURE_LOD 23

bool rt_texture_lod_enabled() {
	return get_rt_param(RT_PARAM_TEXTURE_LOD) > 0.5;
}

/// Whether the hit shaders sample over the cone's elliptical footprint with UV gradients, where they sample at
/// its level at all.
bool rt_texture_lod_anisotropic() {
	return rt_texture_lod_enabled() && get_rt_param(RT_PARAM_TEXTURE_LOD_ANISOTROPIC) > 0.5;
}

/// Whether a hit computes its level: the switch, or the debug view that shows it.
bool rt_texture_lod_wanted() {
#ifdef RT_DEBUG_ENABLED
	if (int(get_rt_param(RT_PARAM_VIS_MODE)) == RT_VIS_MODE_TEXTURE_LOD) {
		return true;
	}
#endif
	return rt_texture_lod_enabled();
}

/// The debug view's colour for a level: blue at 0 through cyan, green and yellow to red at 10.
vec3 rt_texture_lod_ramp(float p_level) {
	float x = clamp(p_level / 10.0, 0.0, 1.0) * 4.0;
	if (x < 1.0) {
		return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), x);
	} else if (x < 2.0) {
		return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), x - 1.0);
	} else if (x < 3.0) {
		return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), x - 2.0);
	}
	return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), x - 3.0);
}

/// The spread a camera ray starts with (its width is zero at the camera): the angle between the
/// ray the raygen traces through this pixel's NDC point and the one through the next pixel's
/// along y, both unprojected through the same inverse projection, so it is exact for whatever
/// projection the frame has, its jitter included.
float rt_cone_primary_spread(mat4 p_inv_projection, vec2 p_ndc, float p_launch_height) {
	vec4 t0 = p_inv_projection * vec4(p_ndc, 1.0, 1.0);
	vec4 t1 = p_inv_projection * vec4(p_ndc.x, p_ndc.y + 2.0 / p_launch_height, 1.0, 1.0);
	vec3 d0 = normalize(t0.xyz);
	vec3 d1 = normalize(t1.xyz);
	return atan(length(cross(d0, d1)), dot(d0, d1));
}

/// The spread after a non-delta scatter whose direction was drawn with density p_pdf: the lobe
/// counted as a spherical cap of solid angle 1 / p_pdf, its plane angle 2 acos(1 - omega / 2 pi)
/// scaled by 0.3 (RTXPT's deliberate underestimate, since the path's samples resolve the rest),
/// capped at 2 pi. A density of zero, which only a direction the lobe does not own gets, counts
/// as the whole sphere.
float rt_cone_scatter_spread(float p_spread, float p_pdf) {
	float solid_angle = p_pdf > 0.0 ? 1.0 / p_pdf : 4.0 * PI;
	float lobe = 2.0 * acos(clamp(1.0 - solid_angle / (2.0 * PI), -1.0, 1.0));
	return min(p_spread + 0.3 * lobe, 2.0 * PI);
}

/// The cone's footprint at a triangle hit as a level in metres: the log2 of the cone's width over
/// the square root of the cosine between the ray and the face normal (RTXPT's "more detail on
/// slopes", sharper than the paper's plain cosine at grazing angles), plus the raster's mipmap bias.
/// A texture mapped at d texels per metre reads at this plus log2(d). p_cross is the cross product of
/// the triangle's edges in trace space (the instance transform applied).
float rt_texture_footprint(vec3 p_cross, vec3 p_ray_dir, float p_width) {
	vec3 n = p_cross;
	float world_area = length(n);
	if (!(world_area > 1e-20) || !(p_width > 0.0)) {
		return RT_TEXTURE_LOD_BASE;
	}
	float cosine = max(abs(dot(n / world_area, p_ray_dir)), 1e-4);
	return max(log2(p_width) - 0.5 * log2(cosine) + get_rt_param(RT_PARAM_TEXTURE_LOD_BIAS), RT_TEXTURE_LOD_BASE);
}

/// The texture-independent level at a triangle hit: the footprint plus half the log2 of the
/// triangle's UV area over its world area, the mesh's UV density (UVs as the mesh holds them,
/// before any material scale).
float rt_texture_lod_triangle(float p_footprint, vec3 p_cross, vec2 p_uv0, vec2 p_uv1, vec2 p_uv2) {
	vec2 duv1 = p_uv1 - p_uv0;
	vec2 duv2 = p_uv2 - p_uv0;
	float uv_area = abs(duv1.x * duv2.y - duv1.y * duv2.x);
	if (p_footprint <= RT_TEXTURE_LOD_BASE || !(uv_area > 1e-20)) {
		return RT_TEXTURE_LOD_BASE;
	}
	return max(p_footprint + 0.5 * log2(uv_area / length(p_cross)), RT_TEXTURE_LOD_BASE);
}

/// The cone's elliptical footprint at a triangle hit as two UV gradients (JCGT 2021 section 5, Listing 1),
/// UVs as the mesh holds them, (g1, g2) in xy and zw; zero where there are none. The cone is taken as a
/// cylinder around the ray whose radius p_radius is the cone's full width, not the paper's half: a gradient is
/// the UV change per pixel step, and the spread is one pixel's angle, so one step spans the width, and along
/// the normal the ellipse reduces to the isotropic level exactly. The raster's mipmap bias comes in as a factor
/// on the radius. The section with the face's plane is an ellipse with the axis a1 along the ray's projection
/// onto the plane, stretched by 1 / abs(n.d) (Listing 1's similar-triangles length in closed form, the cosine
/// clamped as the isotropic footprint's is), and a2 across it at the radius. Each axis maps to UVs through the barycentric
/// coordinates of its end, the triangle's own UV-per-metre Jacobian. p_e1, p_e2 are the triangle's edges from
/// its first corner in trace space.
vec4 rt_texture_grad_triangle(vec3 p_e1, vec3 p_e2, vec3 p_ray_dir, float p_radius, vec2 p_uv0, vec2 p_uv1, vec2 p_uv2) {
	vec3 n = cross(p_e1, p_e2);
	float world_area = length(n);
	if (!(world_area > 1e-20) || !(p_radius > 0.0)) {
		return vec4(0.0);
	}
	vec3 f = n / world_area;
	vec3 d = normalize(p_ray_dir);
	float cosine = dot(f, d);
	vec3 h1 = d - cosine * f;
	float h1_length = length(h1);
	// Along the normal every direction in the plane is the ray's projection; the first edge stands for it.
	h1 = h1_length > 1e-4 ? h1 / h1_length : normalize(p_e1);
	vec3 a1 = h1 * (p_radius / max(abs(cosine), 1e-4));
	vec3 a2 = cross(f, h1) * p_radius;
	// The barycentric coordinates an axis's end moves by: twice the signed areas it spans with the edges
	// over twice the triangle's.
	float inv_area = 1.0 / world_area;
	vec2 b1 = vec2(dot(f, cross(a1, p_e2)), dot(f, cross(p_e1, a1))) * inv_area;
	vec2 b2 = vec2(dot(f, cross(a2, p_e2)), dot(f, cross(p_e1, a2))) * inv_area;
	vec2 duv1 = p_uv1 - p_uv0;
	vec2 duv2 = p_uv2 - p_uv0;
	vec4 grad = vec4(b1.x * duv1 + b1.y * duv2, b2.x * duv1 + b2.y * duv2);
	if (any(isnan(grad)) || any(isinf(grad))) {
		return vec4(0.0);
	}
	return grad;
}
