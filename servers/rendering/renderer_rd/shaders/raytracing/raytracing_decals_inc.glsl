// Decals in the hit shaders (D104): the raster's clustered decal loop (scene_forward_clustered.glsl)
// applied at every traced hit, so a decal reaches the traced primary, its reflections and its bounces.
// The raster finds a fragment's decals through its froxel clusters, which cover only what the camera
// sees; a hit here walks a bounding volume hierarchy over every decal in the trace's cull volume,
// built per frame on the CPU (RenderRaytracing::gather_decals), with the decals' transforms in trace
// space. The math is the raster's: the box test, the upper and lower fades, the normal fade, the
// albedo mix, the normal map into the shading normal, roughness and metallic from the ORM texture,
// additive emission. Differences: the textures are read at LOD 0 (no ray differentials; texture LOD
// in the trace is R2's), the ORM's occlusion has no field in MaterialResult, and the decals at a hit
// are applied farthest rank first, which the raster does only within one 32-decal cluster word.
//
// Requires: HitData and MaterialResult, geometries[] (the instance's layers in the row's first pad
// word, RenderRaytracing::build_tlas), get_rt_param and scene_data_block.

#define RT_DECAL_LEAF_BIT 0x80000000u // Must match RT_DECAL_LEAF_BIT in render_raytracing.h.
#define RT_GEOMETRY_LAYERS_WORD 0 // Must match RT_GEOMETRY_LAYERS_WORD in render_raytracing.h.
// The walk's stack: one entry per level, and the tree over RT_DECALS_MAX decals in leaves of
// RT_DECAL_LEAF_SIZE is a median split, ten levels deep.
#define RT_DECAL_STACK_SIZE 24
// The decals applied at one hit, the nearest by rank when more overlap.
#define RT_DECALS_PER_HIT 8

struct RTDecalData {
	vec4 xform[3]; // Trace space to the decal's UV space, three rows.
	vec4 albedo_rect;
	vec4 normal_rect;
	vec4 orm_rect;
	vec4 emission_rect;
	vec4 modulate;
	vec4 normal_xform[3]; // The decal's world basis, three rows (xyz).
	vec3 normal;
	uint mask;
	float emission_energy;
	float albedo_mix;
	float upper_fade;
	float lower_fade;
	float normal_fade;
	uint order; // The raster's sort rank: lower is nearer the camera, applied later.
	float pad0;
	float pad1;
};

struct RTDecalNode {
	vec3 aabb_min;
	uint child; // An inner node: the right child (the left is the next node); a leaf: RT_DECAL_LEAF_BIT | first record.
	vec3 aabb_max;
	uint count;
};

// clang-format off
layout(set = 0, binding = 47, std430) readonly buffer RTDecalBuffer {
	RTDecalData rt_decals[];
};

layout(set = 0, binding = 48, std430) readonly buffer RTDecalNodeBuffer {
	RTDecalNode rt_decal_nodes[];
};
// clang-format on

layout(set = 0, binding = 49) uniform texture2D rt_decal_atlas;
layout(set = 0, binding = 50) uniform texture2D rt_decal_atlas_srgb;
layout(set = 0, binding = 51) uniform sampler rt_decal_sampler;

vec3 rt_decal_uv(uint p_index, vec3 p_pos) {
	vec4 p = vec4(p_pos, 1.0);
	return vec3(dot(rt_decals[p_index].xform[0], p), dot(rt_decals[p_index].xform[1], p), dot(rt_decals[p_index].xform[2], p));
}

bool rt_decal_contains(vec3 p_uv) {
	return all(greaterThanEqual(p_uv, vec3(0.0, -1.0, 0.0))) && all(lessThanEqual(p_uv, vec3(1.0)));
}

void apply_decals(HitData p_h, inout MaterialResult r_m) {
	uint decal_count = uint(get_rt_param(RT_PARAM_DECAL_COUNT));
	if (decal_count == 0u) {
		return;
	}
	uint layers = geometries[p_h.geometry_idx]._pad[RT_GEOMETRY_LAYERS_WORD];
	if (layers == 0u) {
		return;
	}

	// The decals whose box holds the hit and whose cull mask meets the instance's layers, kept in
	// rank order, the nearest RT_DECALS_PER_HIT when more overlap.
	uint found[RT_DECALS_PER_HIT];
	uint found_order[RT_DECALS_PER_HIT];
	uint found_count = 0u;
	uint stack[RT_DECAL_STACK_SIZE];
	uint stack_size = 0u;
	uint node_index = 0u;
	while (true) {
		RTDecalNode node = rt_decal_nodes[node_index];
		if (all(greaterThanEqual(p_h.hit_pos, node.aabb_min)) && all(lessThanEqual(p_h.hit_pos, node.aabb_max))) {
			if ((node.child & RT_DECAL_LEAF_BIT) == 0u) {
				if (stack_size < RT_DECAL_STACK_SIZE) {
					stack[stack_size] = node.child;
					stack_size++;
				}
				node_index++;
				continue;
			}
			uint first = node.child & ~RT_DECAL_LEAF_BIT;
			for (uint i = first; i < first + node.count; i++) {
				if ((rt_decals[i].mask & layers) == 0u || !rt_decal_contains(rt_decal_uv(i, p_h.hit_pos))) {
					continue;
				}
				uint order = rt_decals[i].order;
				if (found_count == RT_DECALS_PER_HIT && order >= found_order[RT_DECALS_PER_HIT - 1]) {
					continue;
				}
				// Insert by rank; when full, the farthest is the one shifted out.
				uint j = min(found_count, uint(RT_DECALS_PER_HIT - 1));
				while (j > 0u && found_order[j - 1u] > order) {
					found[j] = found[j - 1u];
					found_order[j] = found_order[j - 1u];
					j--;
				}
				found[j] = i;
				found_order[j] = order;
				found_count = min(found_count + 1u, uint(RT_DECALS_PER_HIT));
			}
		}
		if (stack_size == 0u) {
			break;
		}
		stack_size--;
		node_index = stack[stack_size];
	}

	// Farthest rank first, so the nearest decal is mixed in last, on top, as the raster's order within a word.
	for (uint k = found_count; k > 0u; k--) {
		uint i = found[k - 1u];
		vec3 uv_local = rt_decal_uv(i, p_h.hit_pos);
		float fade = pow(1.0 - (uv_local.y > 0.0 ? uv_local.y : -uv_local.y), uv_local.y > 0.0 ? rt_decals[i].upper_fade : rt_decals[i].lower_fade);
		if (rt_decals[i].normal_fade > 0.0) {
			fade *= smoothstep(rt_decals[i].normal_fade, 1.0, dot(p_h.geometry_normal, rt_decals[i].normal) * 0.5 + 0.5);
		}

		if (rt_decals[i].albedo_rect != vec4(0.0)) {
			vec4 albedo_rect = rt_decals[i].albedo_rect;
			vec4 decal_albedo = textureLod(sampler2D(rt_decal_atlas_srgb, rt_decal_sampler), uv_local.xz * albedo_rect.zw + albedo_rect.xy, 0.0);
			decal_albedo *= rt_decals[i].modulate;
			decal_albedo.a *= fade;
			r_m.albedo = mix(r_m.albedo, decal_albedo.rgb, decal_albedo.a * rt_decals[i].albedo_mix);

			if (rt_decals[i].normal_rect != vec4(0.0)) {
				vec4 normal_rect = rt_decals[i].normal_rect;
				vec3 decal_normal = textureLod(sampler2D(rt_decal_atlas, rt_decal_sampler), uv_local.xz * normal_rect.zw + normal_rect.xy, 0.0).xyz;
				decal_normal.xy = decal_normal.xy * vec2(2.0, -2.0) - vec2(1.0, -1.0); // The raster's flipped y.
				decal_normal.z = sqrt(max(0.0, 1.0 - dot(decal_normal.xy, decal_normal.xy)));
				// Decal space to world space, xzy because y is up.
				vec3 n = decal_normal.xzy;
				vec3 world_normal = vec3(dot(rt_decals[i].normal_xform[0].xyz, n), dot(rt_decals[i].normal_xform[1].xyz, n), dot(rt_decals[i].normal_xform[2].xyz, n));
				r_m.normal = normalize(mix(r_m.normal, world_normal, decal_albedo.a));
			}

			if (rt_decals[i].orm_rect != vec4(0.0)) {
				vec4 orm_rect = rt_decals[i].orm_rect;
				vec3 decal_orm = textureLod(sampler2D(rt_decal_atlas, rt_decal_sampler), uv_local.xz * orm_rect.zw + orm_rect.xy, 0.0).xyz;
				r_m.roughness = mix(r_m.roughness, decal_orm.g, decal_albedo.a);
				r_m.metalness = mix(r_m.metalness, decal_orm.b, decal_albedo.a);
			}
		}

		if (rt_decals[i].emission_rect != vec4(0.0)) {
			// Additive and independent of the albedo, with the exposure normalisation MaterialResult carries,
			// apart from the material's own: the emissive light sampling never draws it, so it takes no MIS weight.
			vec4 emission_rect = rt_decals[i].emission_rect;
			vec3 decal_emission = textureLod(sampler2D(rt_decal_atlas_srgb, rt_decal_sampler), uv_local.xz * emission_rect.zw + emission_rect.xy, 0.0).rgb;
			r_m.decal_emissive += decal_emission * rt_decals[i].modulate.rgb * rt_decals[i].emission_energy * fade * scene_data_block.data.emissive_exposure_normalization;
		}
	}
}
