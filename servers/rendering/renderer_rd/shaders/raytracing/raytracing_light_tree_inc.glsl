// Light selection over a light tree (D55): every hit's one light sample is chosen among the sky,
// the directional lights and a binary tree over the positional lights and the emissive instances
// in proportion to how much each could contribute at the hit, judged per node from its bounding
// sphere, its power and its orientation cone (Conty Estevez and Kulla 2018), so a torch two
// metres away is drawn far more often than one across the field, a spot aimed elsewhere is not
// drawn at all, and a directional sun takes the share its irradiance earns at that surface. The
// tree is built on the CPU each frame (RenderRaytracing::build_light_tree) and walked top down
// on one random number rescaled per level; the probability of the leaf reached is the product of the
// ratios taken, exact, and a chance hit on an emitter recovers the same probability by walking
// from its leaf back to the root, so the MIS weights on both sides are written in the tree's
// density (D51 wrote them in the global one). The tree is the proposal: a few candidates are
// drawn through it and one is kept by its unshadowed contribution over its proposal probability
// (D51's resampling), which corrects the bounds where they are loose (a receiver inside a
// cluster's sphere, a cluster in range by its bound alone) and spends no sample on a light that
// delivers nothing. The off path (the per-frame shares, the uniform analytic pick, the resampled
// emissive candidates over the global CDF) compiles when the flag is clear.
//
// The normal the selection reads is the geometry normal as the payload carries it to the next
// vertex (the octahedral unorm16 of oct_offset_nrm), so the chance-hit side evaluates the same
// function on the same inputs. The receiver-side cosine bound is floored, never zero: a light
// under the geometric horizon can still light a normal-mapped surface, and a leaf whose
// probability is zero is a light next-event estimation can never deliver. The range window and
// the spot cone are exact zeros, since the light is zero there.
//
// Requires: raytracing_inc.glsl, raytracing_common_inc.glsl (RT_FLAGS, get_rt_param),
// raytracing_lights_inc.glsl (RTLightData, rt_lights), raytracing_sky_inc.glsl
// (sky_light_enabled, sky_total, sky_concentrated, and in the hit stages sky_sample_direct);
// in the hit stages raytracing_emissive_inc.glsl (rt_emissive, emissive_sample_instance) too.

#ifndef RT_LIGHT_TREE_BINDING
#define RT_LIGHT_TREE_BINDING 44
#endif

// Must match RT_LIGHT_TREE_NODES_MAX and RT_EMISSIVE_MAX in render_raytracing.h.
#define RT_LIGHT_TREE_NODES_MAX 4095u
#define RT_LIGHT_TREE_EMISSIVE_MAX 1024u

#define RT_LIGHT_TREE_LEAF_BIT 0x80000000u
#define RT_LIGHT_TREE_EMISSIVE_BIT 0x40000000u
#define RT_LIGHT_TREE_INDEX_MASK 0x3FFFFFFFu
#define RT_LIGHT_TREE_NO_NODE 0xFFFFFFFFu
// A tree balanced by count over 2048 leaves is 11 deep; the walks stop here whatever the buffer says.
#define RT_LIGHT_TREE_MAX_DEPTH 32 // A plain integer: the shade include tests it in a preprocessor expression.
// The most candidates one light sample resamples through the tree (Environment.pathtracing_emissive_candidates).
#define RT_LIGHT_TREE_MAX_CANDIDATES 16
// The receiver-side cosine bound's floor: a node wholly under the horizon is drawn at this rate.
#define RT_LIGHT_TREE_RECEIVER_FLOOR 0.02

// Must match C++ RT_LightTreeNode (std430, 48 bytes).
struct RTLightTreeNode {
	vec3 center; // Bounding sphere, trace space.
	float radius;
	vec3 axis; // Orientation cone: the spot directions under the node; unused when cos_theta_o is -1.
	float cos_theta_o; // Cosine of the cone's half-angle, -1 for every direction (omni and emissive).
	float power; // The luminance the node's lights deliver at unit distance, summed.
	float max_range; // The largest range under the node, 0 for unbounded.
	uint child; // An inner node: its right child (the left is the next node). A leaf: RT_LIGHT_TREE_LEAF_BIT | (RT_LIGHT_TREE_EMISSIVE_BIT) | index.
	uint parent; // RT_LIGHT_TREE_NO_NODE at the root.
};

layout(set = 0, binding = RT_LIGHT_TREE_BINDING, std430) readonly buffer LightTreeBuffer {
	uint tree_node_count; // 0 when the tree has no leaf (no positional light and no emitter).
	uint tree_pad[3];
	RTLightTreeNode tree_nodes[RT_LIGHT_TREE_NODES_MAX];
	uint tree_emissive_leaf[RT_LIGHT_TREE_EMISSIVE_MAX]; // The leaf of each emissive entry, RT_LIGHT_TREE_NO_NODE when it has none.
};

bool light_tree_enabled() {
	return (RT_FLAGS & RT_FLAG_LIGHT_TREE) != 0u;
}

/// How much the lights under a node could deliver at x facing N, from the node's bounds: its
/// power over the distance squared (floored by the sphere, so a receiver inside it reads the
/// power), the cosine at the receiver and the cosine at the light each bounded over the sphere's
/// angular extent. Zero beyond the node's range and outside its cone by more than a right angle,
/// where the lights are zero too; the receiver bound is floored.
float light_tree_node_importance(RTLightTreeNode n, vec3 x, vec3 N) {
	vec3 to_c = n.center - x;
	float d2 = dot(to_c, to_c);
	float d = sqrt(d2);
	if (n.max_range > 0.0 && d - n.radius > n.max_range) {
		return 0.0;
	}
	float inv_d2 = 1.0 / max(max(d2, n.radius * n.radius), 1e-8);
	if (d <= n.radius) {
		// Inside the bound no angle is bounded.
		return n.power * inv_d2;
	}
	float sin_u = n.radius / d;
	float cos_u = sqrt(max(1.0 - sin_u * sin_u, 0.0));
	vec3 dir = to_c / d;

	// cos(max(theta_i - theta_u, 0)) at the receiver, floored.
	float cos_i = clamp(dot(N, dir), -1.0, 1.0);
	float cos_bi = 1.0;
	if (cos_i < cos_u) {
		cos_bi = cos_i * cos_u + sqrt(max(1.0 - cos_i * cos_i, 0.0)) * sin_u;
	}
	cos_bi = max(cos_bi, RT_LIGHT_TREE_RECEIVER_FLOOR);

	// cos(max(theta - theta_o - theta_u, 0)) at the light, zero past a right angle beyond the cone.
	float cos_bl = 1.0;
	if (n.cos_theta_o > -1.0) {
		float theta = acos(clamp(dot(n.axis, -dir), -1.0, 1.0));
		float a = theta - acos(clamp(n.cos_theta_o, -1.0, 1.0)) - asin(clamp(sin_u, 0.0, 1.0));
		if (a > 0.0) {
			if (a >= 0.5 * PI) {
				return 0.0;
			}
			cos_bl = cos(a);
		}
	}
	return n.power * cos_bi * cos_bl * inv_d2;
}

/// A directional light's irradiance at a surface facing N, floored as the nodes' receiver bound
/// is: the leaf sampler judges the sample on the shading normal, which can see a sun the geometry
/// normal does not.
float light_tree_directional_importance(RTLightData light, vec3 N) {
	float lum = abs(dot(light.emission, vec3(0.2126, 0.7152, 0.0722)));
	return lum * max(dot(N, -light.position), RT_LIGHT_TREE_RECEIVER_FLOOR);
}

/// The sky's importance: its concentrated light as a horizontal-surface irradiance, D54's share
/// judged per hit; 0 without the sky light or a black sky.
float light_tree_sky_importance() {
	if (!sky_light_enabled() || !(sky_total > 0.0)) {
		return 0.0;
	}
	return max(0.25 * sky_concentrated * scene_data_block.data.IBL_exposure_normalization, 0.0);
}

/// The importances at the top of the selection: the sky, every directional light and the tree's
/// root, and their total.
struct LightTreeTop {
	float sky;
	float directional; // Summed.
	float root;
	float total;
	uint directional_count;
};

LightTreeTop light_tree_top(vec3 x, vec3 N) {
	LightTreeTop t;
	t.directional_count = uint(get_rt_param(RT_PARAM_DIRECTIONAL_COUNT));
	t.sky = light_tree_sky_importance();
	t.directional = 0.0;
	for (uint k = 0u; k < t.directional_count; k++) {
		t.directional += light_tree_directional_importance(rt_lights[k], N);
	}
	t.root = (tree_node_count > 0u) ? light_tree_node_importance(tree_nodes[0], x, N) : 0.0;
	t.total = t.sky + t.directional + t.root;
	return t;
}

/// With no surface light listed at all the sky is the only strategy, as D54 has it, whatever its
/// concentration; with lights listed but nothing reaching x, nothing is drawn.
bool light_tree_sky_only(LightTreeTop t) {
	return sky_light_enabled() && sky_total > 0.0 && get_rt_param(RT_PARAM_LIGHT_COUNT) == 0.0 && get_rt_param(RT_PARAM_EMISSIVE_COUNT) == 0.0;
}

/// The probability with which the sky was chosen at the previous vertex (x, N): what the miss
/// weighs the sky it found against.
float light_tree_sky_probability(vec3 x, vec3 N) {
	LightTreeTop t = light_tree_top(x, N);
	if (light_tree_sky_only(t)) {
		return 1.0;
	}
	if (!(t.total > 0.0)) {
		return 0.0;
	}
	return t.sky / t.total;
}

/// The probability with which the tree's descent from (x, N) reaches an emissive entry's leaf:
/// the root's share times, from the leaf up, the ratio taken at every level. Zero when the entry
/// has no leaf or a level on the way read nothing.
float light_tree_emissive_probability(uint p_entry, vec3 x, vec3 N) {
	if (p_entry >= RT_LIGHT_TREE_EMISSIVE_MAX) {
		return 0.0;
	}
	uint node = tree_emissive_leaf[p_entry];
	if (node == RT_LIGHT_TREE_NO_NODE || node >= tree_node_count) {
		return 0.0;
	}
	LightTreeTop t = light_tree_top(x, N);
	if (light_tree_sky_only(t) || !(t.total > 0.0)) {
		return 0.0;
	}
	float p = t.root / t.total;
	for (uint depth = 0u; depth < uint(RT_LIGHT_TREE_MAX_DEPTH); depth++) {
		uint parent = tree_nodes[node].parent;
		if (parent == RT_LIGHT_TREE_NO_NODE) {
			break;
		}
		uint left = parent + 1u;
		uint sibling = (node == left) ? tree_nodes[parent].child : left;
		float i_node = light_tree_node_importance(tree_nodes[node], x, N);
		float i_sibling = light_tree_node_importance(tree_nodes[sibling], x, N);
		float sum = i_node + i_sibling;
		if (!(sum > 0.0)) {
			return 0.0;
		}
		p *= i_node / sum;
		node = parent;
	}
	return p;
}

#ifndef RT_STAGE_MISS

/// What a candidate of the resampling is: the sky, a directional light, or a leaf of the tree
/// (an analytic light or an emissive instance), with the probability the proposal drew it with.
#define RT_LIGHT_TREE_KIND_SKY 0u
#define RT_LIGHT_TREE_KIND_DIRECTIONAL 1u
#define RT_LIGHT_TREE_KIND_LEAF 2u

struct LightTreeCandidate {
	uint kind;
	uint index; // The light or the emissive entry.
	uint node; // The leaf node (a leaf candidate).
	float p; // The proposal's probability of this candidate.
};

/// Draws one candidate from two random numbers: the strategy at the top by its importance from
/// the first, then the descent on the second, rescaled inside the interval taken at every level
/// so it stays uniform (the top split takes its own number: a small root share would otherwise
/// spend the bits a deep leaf's interval needs). False when a level on the way read nothing (the
/// bound passed, the children do not).
bool light_tree_draw(float u_top, float u, LightTreeTop t, vec3 x, vec3 N, out LightTreeCandidate r_c) {
	r_c.kind = RT_LIGHT_TREE_KIND_LEAF;
	r_c.index = 0u;
	r_c.node = 0u;
	r_c.p = 0.0;
	float pick = u_top * t.total;
	if (pick < t.sky) {
		r_c.kind = RT_LIGHT_TREE_KIND_SKY;
		r_c.p = t.sky / t.total;
		return true;
	}
	pick -= t.sky;
	if (pick < t.directional) {
		uint chosen = t.directional_count - 1u;
		float running = 0.0;
		for (uint k = 0u; k < t.directional_count; k++) {
			running += light_tree_directional_importance(rt_lights[k], N);
			if (pick < running) {
				chosen = k;
				break;
			}
		}
		r_c.kind = RT_LIGHT_TREE_KIND_DIRECTIONAL;
		r_c.index = chosen;
		r_c.p = light_tree_directional_importance(rt_lights[chosen], N) / t.total;
		return r_c.p > 0.0;
	}
	if (tree_node_count == 0u || !(t.root > 0.0)) {
		return false;
	}
	u = min(u, 0.99999994);
	float p = t.root / t.total;
	uint node = 0u;
	RTLightTreeNode n = tree_nodes[0];
	for (uint depth = 0u; depth < uint(RT_LIGHT_TREE_MAX_DEPTH) && (n.child & RT_LIGHT_TREE_LEAF_BIT) == 0u; depth++) {
		uint left = node + 1u;
		uint right = n.child;
		float i_left = light_tree_node_importance(tree_nodes[left], x, N);
		float i_right = light_tree_node_importance(tree_nodes[right], x, N);
		float sum = i_left + i_right;
		if (!(sum > 0.0)) {
			return false;
		}
		float ratio = i_left / sum;
		if (u < ratio) {
			node = left;
			p *= ratio;
			u = u / ratio;
		} else {
			node = right;
			p *= 1.0 - ratio;
			u = (u - ratio) / (1.0 - ratio);
		}
		u = min(u, 0.99999994);
		n = tree_nodes[node];
	}
	if ((n.child & RT_LIGHT_TREE_LEAF_BIT) == 0u || !(p > 0.0)) {
		return false;
	}
	r_c.index = n.child & RT_LIGHT_TREE_INDEX_MASK;
	r_c.node = node;
	r_c.p = p;
	return true;
}

/// An analytic light's unshadowed irradiance at hit_pos facing the shading normal N, in the
/// units the shader lights with: its luminance under the windowed attenuation and the spot cone,
/// times the cosine at the receiver raised by the light's angular size and floored (a sphere light
/// on the horizon still delivers). Exactly zero only beyond the range and outside the cone, where
/// the sample would carry nothing.
float light_tree_analytic_target(RTLightData light, vec3 hit_pos, vec3 N) {
	float lum = abs(dot(light.emission, vec3(0.2126, 0.7152, 0.0722)));
	if (light.type == RT_LIGHT_TYPE_DIRECTIONAL) {
		vec3 L = -normalize(light.position);
		return dot(N, L) > 0.0 ? lum * dot(N, L) : 0.0;
	}
	vec3 to_l = light.position - hit_pos;
	float dist_sq = dot(to_l, to_l);
	if (light.max_range_squared != 0.0 && dist_sq > light.max_range_squared) {
		return 0.0;
	}
	float dist = sqrt(dist_sq);
	vec3 L = to_l / max(dist, 1e-4);
	float spot_atten = 1.0;
	if (light.type == RT_LIGHT_TYPE_SPOT) {
		float scos = dot(-L, light.spot_direction);
		// The sampler tests the cone on a direction drawn within the light's angular size, so the
		// zero here is one angular size wider than the cone; inside it the falloff is floored.
		float cos_outer = cos(min(acos(clamp(light.cos_spot_angle, -1.0, 1.0)) + asin(clamp(light.radius / max(dist, light.radius), 0.0, 1.0)), PI));
		if (scos <= cos_outer) {
			return 0.0;
		}
		float spot_rim = clamp((1.0 - scos) / (1.0 - light.cos_spot_angle), 1e-4, 1.0);
		spot_atten = max(1.0 - pow(spot_rim, light.inv_spot_attenuation), RT_LIGHT_TREE_RECEIVER_FLOOR);
	}
	LightSample ls;
	ls.cone_axis = L;
	ls.cos_theta_max = 1.0;
	ls.emission = light.emission;
	ls.distance_sq = dist_sq;
	ls.max_distance = dist;
	float atten = lights_get_attenuation(ls, light.inv_max_range, light.attenuation) * spot_atten;
	float above = max(dot(N, L) + light.radius / max(dist, light.radius), RT_LIGHT_TREE_RECEIVER_FLOOR);
	return lum * atten * above;
}

/// The resampling target of a candidate: its unshadowed contribution estimate at the hit, in one
/// unit for every kind (an irradiance), against which the proposal's probability is corrected.
/// Any positive function is unbiased; the sky's is its own importance, so the proposal stands.
float light_tree_target(LightTreeCandidate c, LightTreeTop t, vec3 hit_pos, vec3 N) {
	if (c.kind == RT_LIGHT_TREE_KIND_SKY) {
		return t.sky;
	}
	if (c.kind == RT_LIGHT_TREE_KIND_DIRECTIONAL) {
		return light_tree_analytic_target(rt_lights[c.index], hit_pos, N);
	}
	uint encoded = tree_nodes[c.node].child;
	if ((encoded & RT_LIGHT_TREE_EMISSIVE_BIT) != 0u) {
		// The entry's power at unit distance is the leaf's; the proximity is D51's target.
		return tree_nodes[c.node].power * emissive_proximity(rt_emissive[c.index], hit_pos, N);
	}
	return light_tree_analytic_target(rt_lights[c.index], hit_pos, N);
}

/// One light sample by the tree: `candidates` draws through the proposal (the top-level split
/// and the descent, a random number each), each weighed by its target over its proposal
/// probability, one kept by weighted reservoir sampling on a single rescaled number (D51's
/// form), then the kept candidate's own sample (an analytic light's cone or point, an emissive
/// triangle, a sky direction). The resampling corrects the proposal where its bounds are loose (a
/// receiver inside a cluster's sphere, a cluster in range by its bound alone), so no sample is
/// spent on a light that delivers nothing when any candidate does. r_probability is what the
/// caller divides the contribution by: the resampled probability for an analytic light or the sky
/// (their weights written as before, the sky's MIS weight in the proposal's density the miss
/// evaluates too), 1 for an emissive leaf, whose weight already carries the proposal's density in
/// its MIS weight and the resampled density under it. Returns false when nothing was drawn or the
/// sample carries nothing. Draws: one for the reservoir, two per candidate and the leaf's pair.
bool light_tree_sample_direct(
		vec3 hit_pos,
		vec3 N_select,
		vec3 N,
		vec3 V,
		MaterialProperties material,
		inout uint rng_state,
		bool is_indirect_bounce,
		uint candidates,
		out LightSampleNEE r_sample,
		out float r_probability) {
	r_sample.direction = vec3(0.0, 1.0, 0.0);
	r_sample.shadow_distance = 0.0;
	r_sample.leg_distance = 0.0;
	r_sample.radiance = vec3(0.0);
	r_sample.weight = 0.0;
	r_sample.specular_multiplier = 1.0;
	r_probability = 1.0;

	LightTreeTop t = light_tree_top(hit_pos, N_select);
	if (light_tree_sky_only(t)) {
		return sky_sample_direct(N, V, material, rng_state, 1.0, r_sample);
	}
	if (!(t.total > 0.0)) {
		return false;
	}
	candidates = clamp(candidates, 1u, uint(RT_LIGHT_TREE_MAX_CANDIDATES));
	float u_res = min(rand(rng_state), 0.99999994);
	LightTreeCandidate selected;
	selected.kind = RT_LIGHT_TREE_KIND_LEAF;
	selected.index = 0u;
	selected.node = 0u;
	selected.p = 0.0;
	float w_sum = 0.0;
	float w_selected = 0.0;
	for (uint i = 0u; i < candidates; i++) {
		float u_top = min(rand(rng_state), 0.99999994);
		float u = rand(rng_state);
		LightTreeCandidate c;
		if (!light_tree_draw(u_top, u, t, hit_pos, N_select, c)) {
			continue;
		}
		float w = light_tree_target(c, t, hit_pos, N) / c.p;
		if (!(w > 0.0)) {
			continue;
		}
		w_sum += w;
		float accept = w / w_sum;
		if (u_res < accept) {
			selected = c;
			w_selected = w;
			u_res = u_res / accept;
		} else {
			u_res = (u_res - accept) / (1.0 - accept);
		}
		u_res = min(u_res, 0.99999994);
	}
	vec2 u2 = rand2(rng_state);
	if (!(w_selected > 0.0)) {
		return false;
	}
	// The kept candidate was really selected with candidates * w_selected / w_sum times its
	// proposal probability (D51).
	float resample_factor = float(candidates) * w_selected / w_sum;

	if (selected.kind == RT_LIGHT_TREE_KIND_SKY) {
		r_probability = selected.p * resample_factor;
		return sky_sample_direct(N, V, material, rng_state, selected.p, r_sample);
	}
	if (selected.kind == RT_LIGHT_TREE_KIND_DIRECTIONAL) {
		r_probability = selected.p * resample_factor;
		return lights_sample_light(rt_lights[selected.index], hit_pos, N, material, u2, is_indirect_bounce, r_sample);
	}
	if ((tree_nodes[selected.node].child & RT_LIGHT_TREE_EMISSIVE_BIT) != 0u) {
		r_probability = 1.0;
		return emissive_sample_instance(rt_emissive[selected.index], selected.p, resample_factor, hit_pos, N, V, material, u2, 1.0, r_sample);
	}
	r_probability = selected.p * resample_factor;
	return lights_sample_light(rt_lights[selected.index], hit_pos, N, material, u2, is_indirect_bounce, r_sample);
}

#endif // !RT_STAGE_MISS
