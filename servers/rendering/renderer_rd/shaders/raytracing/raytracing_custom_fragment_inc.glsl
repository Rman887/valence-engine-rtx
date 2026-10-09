// Custom shader fragment setup for RT hit groups (closest-hit and any-hit).
// Include inside main() under #ifdef RT_CUSTOM_HIT_GROUP.
//
// Required variables before inclusion:
//   uint  rt_geometry_idx   -- geometry/material index
//   vec3  rt_hit_pos        -- world-space hit position
//   vec2  rt_uv             -- interpolated UV
//   vec2  rt_uv2            -- interpolated second UV set, zero when the mesh has none or the shader reads no UV2
//   vec3  rt_normal         -- world-space geometry normal (flipped for back-face)
//   vec3  rt_tangent        -- world-space tangent
//   vec3  rt_bitangent      -- world-space bitangent
//   bool  rt_front_face     -- true if front-face hit
//
// Required bindings/types:
//   materials[], CustomMaterialUniforms, scene_data_block

MaterialData rt_mat = materials[rt_geometry_idx];
material = CustomMaterialUniforms(rt_mat.uniform_address);

// Matrices.
mat4 rt_view_matrix = transpose(mat4(scene_data_block.data.view_matrix[0],
		scene_data_block.data.view_matrix[1],
		scene_data_block.data.view_matrix[2],
		vec4(0.0, 0.0, 0.0, 1.0)));

GeometryData rt_geom = geometries[rt_geometry_idx];
mat4 rt_aabb_xform;
mat4 rt_inv_aabb_xform;
get_aabb_compression_xforms(rt_geom, rt_aabb_xform, rt_inv_aabb_xform);

read_model_matrix = mat4(gl_ObjectToWorldEXT) * rt_inv_aabb_xform;
read_view_matrix = rt_view_matrix;
inv_view_matrix = transpose(mat4(scene_data_block.data.inv_view_matrix[0],
		scene_data_block.data.inv_view_matrix[1],
		scene_data_block.data.inv_view_matrix[2],
		vec4(0.0, 0.0, 0.0, 1.0)));
// The trace runs in world space minus rt_origin; user shader code gets the
// absolute matrices it expects (view_abs = view_rel * translate(-rt_origin)).
read_model_matrix[3].xyz += rt_origin.xyz;
// As the raster builds it: the basis itself under uniform scale, its inverse transpose otherwise.
model_normal_matrix = mat3(read_model_matrix);
if (!rt_basis_has_uniform_scale(model_normal_matrix)) {
	model_normal_matrix = transpose(inverse(model_normal_matrix));
}
inv_view_matrix[3].xyz += rt_origin.xyz;
read_view_matrix[3].xyz -= mat3(read_view_matrix) * rt_origin.xyz;
projection_matrix = scene_data_block.data.projection_matrix;
inv_projection_matrix = scene_data_block.data.inv_projection_matrix;
read_viewport_size = scene_data_block.data.viewport_size;
global_time = scene_data_block.data.time;

mat4 rt_world_to_object_decomp = rt_aabb_xform * mat4(gl_WorldToObjectEXT);

vertex = (rt_world_to_object_decomp * vec4(rt_hit_pos, 1.0)).xyz;
// The frame went to world space through the normal matrix, so it comes back through its inverse.
mat3 rt_model_basis = mat3(read_model_matrix);
normal = normalize(transpose(rt_model_basis) * rt_normal);
tangent = normalize(transpose(rt_model_basis) * rt_tangent);
binormal = normalize(transpose(rt_model_basis) * rt_bitangent);
uv_interp = rt_uv;
uv2_interp = rt_uv2;
color_interp = rt_color;
view = -gl_WorldRayDirectionEXT;
rt_front_facing = rt_front_face;
rt_screen_uv = vec2(gl_LaunchIDEXT.xy) / vec2(gl_LaunchSizeEXT.xy);
rt_frag_coord = vec4(gl_LaunchIDEXT.xy, 0.0, 1.0);

// Run vertex shader (computes varyings, may modify built-ins).
/* RT_CUSTOM_VERTEX_CALL */

#if defined(RT_STAGE_CLOSEST_HIT) && !defined(RT_CUSTOM_UV_WRITTEN)
// The fragment's texture() calls at the mesh's UVs read at the ray cone's level, or over its elliptical
// footprint (D116); the vertex function's, before this, and the any-hit's alpha test read level 0.
rt_lod_t = rt_hit_texture_lod;
rt_lod2_t = rt_hit_texture_lod2;
rt_grad_t = RtTextureGrad(rt_hit_texture_grad != vec4(0.0), rt_hit_texture_grad.xy, rt_hit_texture_grad.zw);
#endif
#ifdef RT_STAGE_CLOSEST_HIT
// And its texture_projected() calls at the cone's footprint (D116), whatever the shader does with UV.
rt_footprint_t = rt_hit_texture_footprint;
#endif
#if defined(RT_STAGE_CLOSEST_HIT) && defined(RT_CUSTOM_TEXTURE_FILTERING) && defined(RT_TEXTURE_FILTERING_COMPILED)
// The numbers its stochastic lookups share (D117 F3), drawn as HG0 draws them, the major axis's where the hit has
// gradients.
if (rt_texture_filtering() != 0u) {
	rt_stf_u = rt_texture_filtering_random(rt_grad_t.on);
	rt_stf_on = true;
}
#endif

// Post-vertex transform: object-space -> view-space (mirrors rasterizer post-vertex).
mat4 rt_modelview = read_view_matrix * read_model_matrix;
vertex = (rt_modelview * vec4(vertex, 1.0)).xyz;
mat3 rt_modelview_normal = mat3(read_view_matrix) * rt_normal_basis(rt_model_basis);
normal = normalize(rt_modelview_normal * normal);
tangent = normalize(rt_modelview_normal * tangent);
binormal = normalize(rt_modelview_normal * binormal);

// Fragment outputs with sensible defaults.
vec3 albedo = vec3(1.0);
float alpha = 1.0;
float metallic = 0.0;
float roughness = 0.5;
float specular = 0.5;
vec3 emission = vec3(0.0);
vec3 normal_map = vec3(0.5, 0.5, 1.0);
float normal_map_depth = 1.0;
float ao = 1.0;
float ao_light_affect = 0.0;
vec3 backlight = vec3(0.0);
vec3 transmission = vec3(0.0);
float ior = 1.5;
vec3 absorption = vec3(0.0);
vec3 scattering = vec3(0.0);
float sss_strength = 0.0;
// Built-ins the trace has no use for, at the raster's defaults: declared so a shader the raster runs
// still compiles its hit group (an undeclared built-in fails it, and the instance leaves the TLAS).
vec4 transmittance_color = vec4(0.0, 0.0, 0.0, 1.0);
float transmittance_depth = 0.0;
float transmittance_boost = 0.0;
float rt_depth = 0.0;
float premul_alpha = 1.0;
vec4 custom_radiance = vec4(0.0);
vec4 custom_irradiance = vec4(0.0);
vec3 light_vertex = vertex;
vec3 bent_normal_map = vec3(0.5);
vec2 rt_point_coord = vec2(0.5);
vec4 fog = vec4(0.0, 0.0, 0.0, 1.0);
float rim = 0.0;
float rim_tint = 0.0;
float clearcoat = 0.0;
float clearcoat_roughness = 0.0;
float anisotropy = 0.0;
vec2 anisotropy_flow = vec2(1.0, 0.0);
float alpha_scissor_threshold = 0.0;
float alpha_hash_scale = 1.0;
float alpha_antialiasing_edge = 0.0;
vec2 alpha_texture_coordinate = vec2(0.0);

{
	/* RT_CUSTOM_FRAGMENT_CODE */
}
