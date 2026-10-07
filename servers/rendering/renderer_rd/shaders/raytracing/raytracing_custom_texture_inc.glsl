// Texture LOD in custom hit groups (D116). The hit group's ShaderCompiler rewrites the user code's
// texture(s, uv) on a float 2D or 2D array sampler whose coordinate reads UV or UV2 directly into
// textureLod(s, uv, rt_texture_lod(textureSize(s, 0))), and texture(s, uv, bias) into
// textureLod(s, uv, (bias) [+ rt_texture_lod(...)]), which also makes the bias form compile in a
// ray-tracing stage. Every other call stays texture(), level 0 here. The level is rt_lod_t, the hit's
// texture-independent level, plus half the log2 of the texture's texel count, or level 0 while
// rt_lod_t is RT_TEXTURE_LOD_BASE. Only the closest hit sets rt_lod_t, after the vertex function and
// not for a shader that writes UV or UV2 (RT_CUSTOM_UV_WRITTEN), so vertex lookups, the any-hit's
// alpha test and the intersection stage read level 0. The sampler expression is written twice, into
// the lookup and into textureSize; a sampler array would index twice, but sampler arrays do not
// compile in custom hit groups (an older gap).
//
// Include under #ifdef RT_CUSTOM_HIT_GROUP, outside main(), before the user code.

float rt_lod_t = RT_TEXTURE_LOD_BASE;

float rt_texture_lod(ivec2 p_size) {
	if (rt_lod_t <= RT_TEXTURE_LOD_BASE) {
		return 0.0;
	}
	return rt_lod_t + 0.5 * log2(max(float(p_size.x) * float(p_size.y), 1.0));
}

float rt_texture_lod(ivec3 p_size) {
	return rt_texture_lod(p_size.xy);
}

// Such a lookup through a linear anisotropic sampler (the uniform's filter_linear_mipmap_anisotropic hint), without
// a bias, whose coordinate reads UV and not UV2, becomes
// (rt_grad_t.on ? textureGrad(s, uv, rt_grad_t.x, rt_grad_t.y) : <the level form above>): the cone's elliptical
// footprint as the mesh's UV gradients, with rendering/pathtracing/texture_lod_anisotropic. Other hints keep the
// isotropic level. A coordinate that scales UV gets the mesh's gradients, as it gets the mesh's level. rt_grad_t is
// set where rt_lod_t is, and only where the hit has gradients.
struct RtTextureGrad {
	bool on;
	vec2 x;
	vec2 y;
};
RtTextureGrad rt_grad_t = RtTextureGrad(false, vec2(0.0), vec2(0.0));

// The spatial stage function texture_projected(tex, uv, uv_per_metre) becomes
// textureLod(tex, uv, rt_texture_projected_lod(textureSize(tex, 0), uv_per_metre)): the cone's footprint in
// metres (rt_footprint_t, set as rt_lod_t is, but for any shader that calls the function) plus the log2 of
// the coordinate's units per metre and half the log2 of the texture's texel count; level 0 while the
// footprint is unknown, as in the any-hit and vertex().
float rt_footprint_t = RT_TEXTURE_LOD_BASE;

float rt_texture_projected_lod(ivec2 p_size, float p_uv_per_metre) {
	if (rt_footprint_t <= RT_TEXTURE_LOD_BASE) {
		return 0.0;
	}
	return rt_footprint_t + log2(max(p_uv_per_metre, 1e-20)) + 0.5 * log2(max(float(p_size.x) * float(p_size.y), 1.0));
}
