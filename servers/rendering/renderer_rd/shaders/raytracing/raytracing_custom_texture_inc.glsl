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

// A lookup whose coordinate reads UV2 and not UV reads at the second set's level, rt_lod2_t, set where rt_lod_t is
// (D118). It takes no gradients: such a lookup keeps the isotropic level.
float rt_lod2_t = RT_TEXTURE_LOD_BASE;

float rt_texture_lod2(ivec2 p_size) {
	if (rt_lod2_t <= RT_TEXTURE_LOD_BASE) {
		return 0.0;
	}
	return rt_lod2_t + 0.5 * log2(max(float(p_size.x) * float(p_size.y), 1.0));
}

float rt_texture_lod2(ivec3 p_size) {
	return rt_texture_lod2(p_size.xy);
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

// Stochastic texture filtering in custom hit groups (D117 F3). The level, gradient and projected forms above, on a
// float 2D uniform of material data (a source_color, hint_normal, hint_roughness_* or hint_anisotropy hint) through a
// linear filter, call rt_texture_filtered_lod(tex, s, nearest, uv, level, mipmaps) or rt_texture_filtered_grad(tex, s,
// nearest, uv, dx, dy) in their place: s the uniform's sampler, nearest the nearest-with-mipmaps sampler of its repeat
// mode, and level the expression the lookup above would read at. Any other sampler (a mask, a lookup table, noise, a
// plain sampler2D) keeps the deterministic lookup. A helper function's sampler argument counts as material data when
// every call passes it some (the parser's tex_argument_material), so a helper called with a colour map and a mask keeps
// the deterministic lookup for both. In the closest hit, with the filter on, one texel of the filter's footprint is
// read through nearest as HG0 reads it (rt_texture_filtering_position); otherwise, in the any-hit's alpha test, the
// intersection stage, the vertex function (before the hit draws its numbers) or with the filter off, each is exactly
// the lookup above. The texture arrives as nonuniformEXT: a uniform here is an index into the bindless table, which
// varies across a hit group's materials, and a texture passed through a parameter would otherwise lose the qualifier
// its index carries.
//
// rt_stf_u holds the hit's four numbers, shared by its lookups as HG0's are. The closest hit draws them after the
// vertex function, only for a shader with such a lookup (RT_CUSTOM_TEXTURE_FILTERING) and the filter on, the major
// axis's (z) where the hit has gradients. The filter's branches compile in only for a pipeline with the filter on
// (RT_TEXTURE_FILTERING_COMPILED, defined with RT_FLAG_TEXTURE_FILTERING): with it off, the driver compiles exactly the
// lookups above.
bool rt_stf_on = false;
vec4 rt_stf_u = vec4(0.0);

vec4 rt_texture_filtered_lod(nonuniformEXT texture2D p_texture, sampler p_sampler, sampler p_nearest, vec2 p_uv, float p_level, bool p_mipmaps) {
#if defined(RT_STAGE_CLOSEST_HIT) && defined(RT_TEXTURE_FILTERING_COMPILED)
	const uint stf_filter = rt_texture_filtering();
	if (stf_filter != 0u && rt_stf_on && !rt_texture_filtering_magnified(p_level)) {
		const uvec2 size = uvec2(textureSize(sampler2D(p_texture, p_nearest), 0));
		const uint levels = p_mipmaps ? uint(textureQueryLevels(sampler2D(p_texture, p_nearest))) : 1u;
		const vec3 pos = rt_texture_filtering_position(stf_filter, rt_stf_u, size, levels, p_mipmaps, p_uv, p_level, vec4(0.0), false);
		return textureLod(sampler2D(p_texture, p_nearest), pos.xy, pos.z);
	}
#endif
	return textureLod(sampler2D(p_texture, p_sampler), p_uv, p_level);
}

vec4 rt_texture_filtered_grad(nonuniformEXT texture2D p_texture, sampler p_sampler, sampler p_nearest, vec2 p_uv, vec2 p_dx, vec2 p_dy) {
#if defined(RT_STAGE_CLOSEST_HIT) && defined(RT_TEXTURE_FILTERING_COMPILED)
	const uint stf_filter = rt_texture_filtering();
	if (stf_filter != 0u && rt_stf_on) {
		const uvec2 size = uvec2(textureSize(sampler2D(p_texture, p_nearest), 0));
		const vec4 grad = vec4(p_dx, p_dy);
		const float level = rt_texture_filtering_level(size, grad);
		if (!rt_texture_filtering_magnified(level)) {
			const uint levels = uint(textureQueryLevels(sampler2D(p_texture, p_nearest)));
			const vec3 pos = rt_texture_filtering_position(stf_filter, rt_stf_u, size, levels, true, p_uv, level, grad, true);
			return textureLod(sampler2D(p_texture, p_nearest), pos.xy, pos.z);
		}
	}
#endif
	return textureGrad(sampler2D(p_texture, p_sampler), p_uv, p_dx, p_dy);
}
