// Light selection over a light tree (D55): every hit's one light sample is chosen among the sky,
// the directional lights and a binary tree over the positional lights and the emissive instances
// in proportion to how much each could contribute at the hit, judged per node from its bounding
// sphere, its power and its orientation cone (Conty Estevez and Kulla 2018), so a torch two
// metres away is drawn far more often than one across the field, a spot aimed elsewhere is not
// drawn at all, and a directional sun takes the share its irradiance earns at that surface. The
// tree is built on the CPU, or refit in place while its lights stay where they are (P2i,
// RenderRaytracing::update_light_tree), which also publishes
// a cut of it: the nodes of its shallowest full levels, as many as Environment.pathtracing_light_tree_cut
// allows, with the leaves under each cut node contiguous in a leaf list. A hit scores the cut once
// (D56; the reads are independent, unlike a descent's) and every candidate is a pick over the cut
// by those scores, then a pick among the leaves under it: by their power (D60,
// Environment.pathtracing_light_tree_power_pick, one read of the cut node's alias table (D61), so a
// torch is drawn over its dim bulb and a bright lantern over a faint one before the resampling
// sees them) or uniformly with the switch off, so the probability of a leaf is its cut node's share
// times its share of the node (of the power, or one over the leaf count). A chance hit on an emitter
// recovers that probability from its entry's cut slot and leaf position, its node scored alone over
// the top's total the payload carries (GPU audit light 1a), so the MIS weights on both sides are
// written in the same density (D51 wrote them in the global one); they agree to the rounding of the
// sampler's running sum (light_tree_sample_direct names the bound).
// The cut is the proposal: a few candidates are drawn through it and one is kept by its unshadowed
// contribution over its proposal probability (D51's
// resampling), which corrects the bounds where they are loose (a receiver inside a cluster's sphere,
// a cluster in range by its bound alone) and the pick below the cut, and spends no sample on
// a light that delivers nothing; a cut node that is itself an omni or emissive leaf is targeted
// from the cut copy and reads no light (D59), a spot leaf keeps the target that skips it outside
// its cone. Past the path's first diffuse bounce a vertex scores a second, coarser cut of the same
// tree (D59, Environment.pathtracing_light_tree_cut_indirect): its light sample carries the least
// of the path's, so its proposal costs the least, and the payload carries which cut the sample was
// drawn from so the next vertex's densities read the same one. The off path (the per-frame shares,
// the uniform analytic pick, the resampled emissive candidates over the global CDF) compiles when
// the flag is clear.
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
// (sky_light_enabled, sky_total, sky_concentrated, sky_variance_share_enabled, sky_variance_share,
// and in the hit stages sky_sample_direct);
// in the hit stages raytracing_emissive_inc.glsl (rt_emissive, emissive_sample_instance) too.

#ifndef RT_LIGHT_TREE_BINDING
#define RT_LIGHT_TREE_BINDING 44
#endif

// Must match RT_LIGHT_TREE_NODES_MAX, RT_EMISSIVE_MAX, RT_LIGHT_TREE_LEAVES_MAX and
// RT_LIGHT_TREE_CUT_MAX in render_raytracing.h.
#define RT_LIGHT_TREE_NODES_MAX 4095u
#define RT_LIGHT_TREE_EMISSIVE_MAX 1024u
#define RT_LIGHT_TREE_LEAVES_MAX 2048u
#define RT_LIGHT_TREE_CUT_MAX 64 // A plain integer: it sizes the buffer's cut array below.

#define RT_LIGHT_TREE_LEAF_BIT 0x80000000u
#define RT_LIGHT_TREE_EMISSIVE_BIT 0x40000000u
#define RT_LIGHT_TREE_INDEX_MASK 0x3FFFFFFFu
#define RT_LIGHT_TREE_NO_NODE 0xFFFFFFFFu
// The most candidates one light sample resamples through the tree (Environment.pathtracing_light_tree_candidates).
#define RT_LIGHT_TREE_MAX_CANDIDATES 16
// The receiver-side cosine bound's floor: a node wholly under the horizon is drawn at this rate.
#define RT_LIGHT_TREE_RECEIVER_FLOOR 0.02

// Must match C++ RT_LightTreeNode (std430, 64 bytes).
struct RTLightTreeNode {
	vec3 center; // Bounding sphere, trace space.
	float radius;
	vec3 axis; // Orientation cone: the spot directions under the node; unused when cos_theta_o is -1.
	float cos_theta_o; // Cosine of the cone's half-angle, -1 for every direction (omni and emissive).
	float power; // The luminance the node's lights deliver at unit distance, summed.
	float max_range; // The largest range under the node, 0 for unbounded.
	uint child; // An inner node: its right child (the left is the next node). A leaf: RT_LIGHT_TREE_LEAF_BIT | (RT_LIGHT_TREE_EMISSIVE_BIT) | index.
	uint leaf_range; // The node's leaves in tree_leaf_list: first << 16 | count.
	float analytic_share; // The omni and spot lights' share of power (D63): 1 without emitters, 0 without them.
	float decay; // The slowest attenuation exponent among those lights, within [0, 2] on an inner node; 2 without them.
	float analytic_range; // The largest range among those lights, 0 when one is unbounded or there are none.
	float emissive_share; // The emitters' share of power.
};

layout(set = 0, binding = RT_LIGHT_TREE_BINDING, std430) readonly buffer LightTreeBuffer {
	uint tree_node_count; // 0 when the tree has no leaf (no positional light and no emitter).
	uint tree_cut_count; // The cut's nodes, at most RT_LIGHT_TREE_CUT_MAX; 0 with no leaf.
	uint tree_leaf_count;
	uint tree_cut_indirect_count; // The indirect cut's nodes (D59), at most RT_LIGHT_TREE_CUT_MAX; 0 with no leaf.
	RTLightTreeNode tree_nodes[RT_LIGHT_TREE_NODES_MAX];
	uint tree_emissive_cut[RT_LIGHT_TREE_EMISSIVE_MAX]; // The cut slots over each emissive entry's leaf (the primary cut's in the low 16 bits, the indirect's in the high, 0xFFFF for none), RT_LIGHT_TREE_NO_NODE when it has no leaf.
	RTLightTreeNode tree_cut[2 * RT_LIGHT_TREE_CUT_MAX]; // The cut's nodes, copied so a slot is one read; the indirect cut's (D59) follow at RT_LIGHT_TREE_CUT_MAX, one array so a slot is one indexed read.
	uint tree_leaf_list[RT_LIGHT_TREE_LEAVES_MAX]; // The leaf nodes in the build's order: a node's leaves are contiguous.
	uint tree_emissive_leaf[RT_LIGHT_TREE_EMISSIVE_MAX]; // Each emissive entry's position in tree_leaf_list (D60), RT_LIGHT_TREE_NO_NODE without a leaf.
	uvec4 tree_leaf_alias[2 * RT_LIGHT_TREE_LEAVES_MAX]; // Each leaf-list position's entry in its cut node's alias table (D60, D61): the threshold, the alias's position, the position's own share and the alias's, the floats as bits; the indirect cut's follow at RT_LIGHT_TREE_LEAVES_MAX, one array indexed by a base.
};

bool light_tree_enabled() {
	return (RT_FLAGS & RT_FLAG_LIGHT_TREE) != 0u;
}

/// The pick under a cut node by the lights' power (D60); clear, uniform.
bool light_tree_power_pick() {
	return (RT_FLAGS & RT_FLAG_LIGHT_TREE_POWER_PICK) != 0u;
}

/// A leaf's share of its cut node's power, from its alias-table entry in the cut `indirect` names
/// (D60, D61). The sampler reads the same float, as the entry's own share or as the alias share of
/// the entry that sent it there, which the CPU copies from this one.
float light_tree_leaf_share(uint at, bool indirect) {
	uint base = indirect ? RT_LIGHT_TREE_LEAVES_MAX : 0u;
	return uintBitsToFloat(tree_leaf_alias[base + at].z);
}

/// The cut a vertex scores (D59): the primary cut, or the coarser indirect cut past the path's first
/// diffuse bounce. Every reader of a slot takes the cut from the LightTreeTop it scored, so the
/// sampler and the densities never mix the two.
uint light_tree_cut_count(bool indirect) {
	return min(indirect ? tree_cut_indirect_count : tree_cut_count, uint(RT_LIGHT_TREE_CUT_MAX));
}

RTLightTreeNode light_tree_cut_node(uint slot, bool indirect) {
	return tree_cut[(indirect ? uint(RT_LIGHT_TREE_CUT_MAX) : 0u) + slot];
}

/// An emissive entry's slot in one cut, RT_LIGHT_TREE_NO_NODE without one.
uint light_tree_emissive_slot(uint entry, bool indirect) {
	uint slot = (tree_emissive_cut[entry] >> (indirect ? 16u : 0u)) & 0xFFFFu;
	return slot == 0xFFFFu ? RT_LIGHT_TREE_NO_NODE : slot;
}

/// The distance falloff of a node's power at d2, the squared distance from its centre (D63), its two
/// kinds weighed by their shares of the power. The emitters fall off as the inverse square floored
/// by the sphere, as they always have; the omni and spot lights by the curve lights_get_attenuation
/// charges them: their slowest decay exponent over the distance floored by the sphere and by 1 cm
/// (D123: uncapped), inside the quartic window of their widest range taken from the sphere's near
/// side. The inverse square read Godot's default decay of 1 as up to the distance times too dim past a
/// metre (and too bright inside one), so the proposal and the sky's share by variance (D62)
/// understated every such light in reach. The slowest exponent bounds a node's lights past a metre and
/// under-reads its faster ones inside one, by max(d, r) to the spread of their exponents: variance
/// only, since leaves and probabilities are exact (open: carry both extremes). A node of emitters
/// alone reads inv_d2 exactly, as before; the analytic curve costs a square root for the window, and
/// a reciprocal square root at a decay of 1, or a logarithm and an exponential at a decay but 1 or 2.
float light_tree_node_falloff(RTLightTreeNode n, float d2, float r2, float inv_d2) {
	if (!(n.analytic_share > 0.0)) {
		return inv_d2;
	}
	float dd = max(max(d2, r2), RT_LIGHT_MIN_DISTANCE * RT_LIGHT_MIN_DISTANCE);
	float analytic = (n.decay == 2.0) ? min(inv_d2, 1.0 / (RT_LIGHT_MIN_DISTANCE * RT_LIGHT_MIN_DISTANCE)) : ((n.decay == 1.0) ? inversesqrt(dd) : exp2(-0.5 * n.decay * log2(dd)));
	if (n.analytic_range > 0.0) {
		float nd = max(sqrt(d2) - n.radius, 0.0) / n.analytic_range;
		nd *= nd;
		nd *= nd;
		nd = max(1.0 - nd, 0.0);
		analytic *= nd * nd;
	}
	return n.analytic_share * analytic + n.emissive_share * inv_d2;
}

/// How much the lights under a node could deliver at x facing N, from the node's bounds: its
/// power times its lights' falloff with distance (light_tree_node_falloff, D63; the distance
/// floored by the sphere, so a receiver inside it reads the falloff at the radius), the cosine at
/// the receiver and the cosine at the light each bounded over the sphere's
/// angular extent. Zero beyond the node's range and outside its cone by more than a right angle,
/// where the lights are zero too; the receiver bound is floored. Every angle is kept as its
/// cosine and sine (the sum and difference identities), so a node costs one reciprocal square
/// root and at most four square roots (two of them for a spot's cone), no arc function: the cut
/// is scored on every hit. A strand's receiver (p_two_sided, D126 H2) bounds the cosine on either side of N.
float light_tree_node_importance(RTLightTreeNode n, vec3 x, vec3 N, bool p_two_sided) {
	vec3 to_c = n.center - x;
	float d2 = dot(to_c, to_c);
	float r2 = n.radius * n.radius;
	if (n.max_range > 0.0) {
		float reach = n.max_range + n.radius;
		if (d2 > reach * reach) {
			return 0.0;
		}
	}
	float inv_d2 = 1.0 / max(max(d2, r2), 1e-8);
	float falloff = light_tree_node_falloff(n, d2, r2, inv_d2);
	if (d2 <= r2) {
		// Inside the bound no angle is bounded.
		return n.power * falloff;
	}
	float inv_d = inversesqrt(d2);
	float sin_u = min(n.radius * inv_d, 1.0);
	float cos_u = sqrt(max(1.0 - sin_u * sin_u, 0.0));
	vec3 dir = to_c * inv_d;

	// cos(max(theta_i - theta_u, 0)) at the receiver, floored.
	float cos_i = clamp(dot(N, dir), -1.0, 1.0);
	cos_i = p_two_sided ? abs(cos_i) : cos_i;
	float cos_bi = 1.0;
	if (cos_i < cos_u) {
		cos_bi = cos_i * cos_u + sqrt(max(1.0 - cos_i * cos_i, 0.0)) * sin_u;
	}
	cos_bi = max(cos_bi, RT_LIGHT_TREE_RECEIVER_FLOOR);

	// cos(max(theta - theta_o - theta_u, 0)) at the light, zero past a right angle beyond the cone:
	// with phi = theta_o + theta_u, the bound applies when phi < pi and theta > phi (cos theta below
	// cos phi), and is then cos(theta - phi), zero or under when that angle reaches a right angle.
	float cos_bl = 1.0;
	if (n.cos_theta_o > -1.0) {
		float cos_t = clamp(dot(n.axis, -dir), -1.0, 1.0);
		float sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
		float cos_o = clamp(n.cos_theta_o, -1.0, 1.0);
		float sin_o = sqrt(max(1.0 - cos_o * cos_o, 0.0));
		float cos_phi = cos_o * cos_u - sin_o * sin_u;
		float sin_phi = sin_o * cos_u + cos_o * sin_u;
		bool phi_under_pi = sin_phi > 0.0 || cos_phi > 0.0;
		if (phi_under_pi && cos_t < cos_phi) {
			cos_bl = cos_t * cos_phi + sin_t * sin_phi;
			if (cos_bl <= 0.0) {
				return 0.0;
			}
		}
	}
	return n.power * cos_bi * cos_bl * falloff;
}

/// A directional light's irradiance at a surface facing N, floored as the nodes' receiver bound
/// is: the leaf sampler judges the sample on the shading normal, which can see a sun the geometry
/// normal does not. A strand's receiver (p_two_sided, D126 H2) takes a sun behind it as one in front.
float light_tree_directional_importance(RTLightData light, vec3 N, bool p_two_sided) {
	float lum = abs(dot(light.emission, vec3(0.2126, 0.7152, 0.0722)));
	float cos_r = dot(N, -light.position);
	return lum * max(p_two_sided ? abs(cos_r) : cos_r, RT_LIGHT_TREE_RECEIVER_FLOOR);
}

/// The sky's importance: its concentrated light as a horizontal-surface irradiance, D54's share
/// judged per hit; 0 without the sky light or a black sky. With the share by variance (D62) it
/// is the importance that gives the sky sky_variance_share's share against the directional
/// lights and the cut summed at this hit (p_surface), for a receiver facing N (D65):
/// s / (s + p_surface) = share.
float light_tree_sky_importance(float p_surface, vec3 N) {
	if (!sky_light_enabled() || !(sky_total > 0.0)) {
		return 0.0;
	}
	if (sky_variance_share_enabled()) {
		// The share by the clipped variance (D72) judges the image itself, so it takes no bound from
		// the concentration importance (D71's, which the display objective replaced).
		float share = sky_variance_share(p_surface, N);
		if (!(share > 0.0)) {
			return 0.0;
		}
		// The shares stop at 0.9; with nothing else in reach any positive importance is the whole.
		return (p_surface > 0.0) ? p_surface * share / (1.0 - share) : 1.0;
	}
	return max(0.25 * sky_concentrated * scene_data_block.data.IBL_exposure_normalization, 0.0);
}

/// The importances at the top of the selection: the sky, every directional light and the cut
/// (its nodes' importances summed, which stands for the tree's root), and their total.
struct LightTreeTop {
	float sky;
	float directional; // Summed.
	float root; // The cut's importances summed.
	float total;
	uint directional_count;
	uint cut_count;
	bool indirect; // Which cut was scored (D59).
};

/// Scores the top once for a hit: the cut's nodes are read independently and their importances
/// summed in slot order. Nothing is stored per slot (a per-hit array would live in local memory
/// and cost the trace its occupancy): a pick runs the same sum again to find its slot. A slot's
/// interval is its node's importance, which the next vertex's density evaluates alone, against
/// the total the payload carries there (GPU audit light 1a). `indirect` picks the cut (D59), p_two_sided a strand's
/// receiver (D126 H2), which the directional lights and the cut score on both sides of N; the sky's share stays the
/// hemisphere's (its table is per normal).
LightTreeTop light_tree_top(vec3 x, vec3 N, bool indirect, bool p_two_sided) {
	LightTreeTop t;
	t.indirect = indirect;
	t.directional_count = uint(get_rt_param(RT_PARAM_DIRECTIONAL_COUNT));
	t.directional = 0.0;
	for (uint k = 0u; k < t.directional_count; k++) {
		t.directional += light_tree_directional_importance(rt_lights[k], N, p_two_sided);
	}
	t.cut_count = light_tree_cut_count(indirect);
	float running = 0.0;
	for (uint s = 0u; s < t.cut_count; s++) {
		running += light_tree_node_importance(light_tree_cut_node(s, indirect), x, N, p_two_sided);
	}
	t.root = running;
	// The sky last: its share by variance (D62) is judged against the rest of the top.
	t.sky = light_tree_sky_importance(t.directional + t.root, N);
	t.total = t.sky + t.directional + t.root;
	return t;
}

/// The probability a pick from a top whose importances sum to `total` assigns one leaf of a cut node
/// with `count` leaves (the node's interval, its importance, over the total, over its leaves),
/// written so light_tree_draw_leaf (which holds the scored top) and the chance-hit side (which holds
/// the total the payload carried) evaluate one expression.
float light_tree_leaf_probability_count(float total, float interval, uint count) {
	return interval / (total * float(max(count, 1u)));
}

/// The probability under the power-weighted pick (D60): the node's interval over the total times
/// the leaf's share of the node's power (light_tree_leaf_share). A cut node that is a leaf takes
/// the count form on both sides.
float light_tree_leaf_probability_share(float total, float interval, float share) {
	return interval / total * share;
}

/// With no surface light listed at all the sky is the only strategy, as D54 has it, whatever its
/// concentration; with lights listed but nothing reaching x, nothing is drawn.
bool light_tree_sky_only(LightTreeTop t) {
	return sky_light_enabled() && sky_total > 0.0 && get_rt_param(RT_PARAM_LIGHT_COUNT) == 0.0 && get_rt_param(RT_PARAM_EMISSIVE_COUNT) == 0.0;
}

/// The probability with which a pick from the top t chooses the sky: what the miss weighs the sky
/// it finds against, carried there in the payload (GPU audit 4a).
float light_tree_sky_probability(LightTreeTop t) {
	if (light_tree_sky_only(t)) {
		return 1.0;
	}
	if (!(t.total > 0.0)) {
		return 0.0;
	}
	return t.sky / t.total;
}

/// The probability with which a pick from (x, N) reaches an emissive entry's leaf: its cut node's
/// importance there over the top's total (the sum the previous vertex scored, carried in the payload,
/// GPU audit light 1a) times the leaf's share of the node (by power, D60, or one over the leaf
/// count), the expression light_tree_draw_leaf evaluates. Only the entry's own node is scored.
/// Zero when the entry has no leaf, its cut node reads nothing there, or nothing was scored (a total of
/// 0; a sky-only frame lists no emitter, so no chance hit asks). `indirect` is the cut the previous
/// vertex scored, as the payload carries it (D59), and `two_sided` whether it scored a strand's receiver (D126 H2).
float light_tree_emissive_probability(uint p_entry, vec3 x, vec3 N, bool indirect, bool two_sided, float total) {
	if (p_entry >= RT_LIGHT_TREE_EMISSIVE_MAX || !(total > 0.0)) {
		return 0.0;
	}
	uint slot = light_tree_emissive_slot(p_entry, indirect);
	if (slot == RT_LIGHT_TREE_NO_NODE || slot >= light_tree_cut_count(indirect)) {
		return 0.0;
	}
	RTLightTreeNode n = light_tree_cut_node(slot, indirect);
	float interval = light_tree_node_importance(n, x, N, two_sided);
	if (!(interval > 0.0)) {
		return 0.0;
	}
	uint count = n.leaf_range & 0xFFFFu;
	if (light_tree_power_pick() && count > 1u) {
		uint at = tree_emissive_leaf[p_entry];
		uint first = n.leaf_range >> 16u;
		if (at < first || at >= first + count) {
			return 0.0;
		}
		return light_tree_leaf_probability_share(total, interval, light_tree_leaf_share(at, indirect));
	}
	return light_tree_leaf_probability_count(total, interval, count);
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
	uint encoded; // A leaf candidate's leaf word: RT_LIGHT_TREE_LEAF_BIT | (RT_LIGHT_TREE_EMISSIVE_BIT) | index.
	float power; // A leaf candidate's power at unit distance (the emissive target's scale).
	float p; // The proposal's probability of this candidate.
};

/// The top-level pick of one candidate on its own random number: the sky or a directional light
/// by its importance, written to r_c and true; or the cut, false, whose slot the sampler finds
/// with the candidate's second number in its pass over the cut (a small cut share would otherwise
/// spend the bits the leaf pick needs). A directional pick that read nothing leaves r_c.p at 0.
bool light_tree_draw_top(float u_top, LightTreeTop t, vec3 N, bool p_two_sided, out LightTreeCandidate r_c) {
	r_c.kind = RT_LIGHT_TREE_KIND_LEAF;
	r_c.index = 0u;
	r_c.encoded = 0u;
	r_c.power = 0.0;
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
			running += light_tree_directional_importance(rt_lights[k], N, p_two_sided);
			if (pick < running) {
				chosen = k;
				break;
			}
		}
		r_c.kind = RT_LIGHT_TREE_KIND_DIRECTIONAL;
		r_c.index = chosen;
		r_c.p = light_tree_directional_importance(rt_lights[chosen], N, p_two_sided) / t.total;
		return true;
	}
	return false;
}

/// The leaf a candidate lands on under the cut node `n`, whose width `drawn` of the running sum from
/// `below` holds its target, `interval` its importance: on the target rescaled inside the width, a uniform pick
/// among the node's leaves, which with the power pick (D60) is the position of an alias table whose
/// entry keeps it or sends it to its alias by the rest of the number (D61, one read where D60
/// searched cumulative shares). A cut node that is itself a leaf is the candidate, taken from the
/// cut copy with no read of the node list (D59). False when the leaf is not one.
bool light_tree_draw_leaf(float target, float below, float interval, float drawn, RTLightTreeNode n, LightTreeTop t, out LightTreeCandidate r_c) {
	r_c.kind = RT_LIGHT_TREE_KIND_LEAF;
	r_c.index = 0u;
	r_c.encoded = 0u;
	r_c.power = 0.0;
	r_c.p = 0.0;
	uint count = max(n.leaf_range & 0xFFFFu, 1u);
	uint encoded = n.child;
	float power = n.power;
	float p = light_tree_leaf_probability_count(t.total, interval, count);
	if ((encoded & RT_LIGHT_TREE_LEAF_BIT) == 0u) {
		uint first = n.leaf_range >> 16u;
		// The target's place inside the width the slot was drawn with (drawn, fl(below + interval) - below),
		// so the leaf pick is exact whatever rounding the slot's probability carries (GPU audit light 1a).
		float u_leaf = (drawn > 0.0) ? clamp((target - below) / drawn, 0.0, 0.99999994) : 0.0;
		uint last = min(first + count - 1u, RT_LIGHT_TREE_LEAVES_MAX - 1u);
		float u_count = u_leaf * float(count);
		uint at = min(first + uint(u_count), last);
		if (light_tree_power_pick()) {
			// The position keeps the pick below its threshold and sends it to its alias above: a leaf
			// is reached with its share, which the entry carries for both outcomes.
			uvec4 e = tree_leaf_alias[(t.indirect ? RT_LIGHT_TREE_LEAVES_MAX : 0u) + at];
			bool keep = clamp(u_count - float(at - first), 0.0, 0.99999994) < uintBitsToFloat(e.x);
			at = keep ? at : min(e.y, RT_LIGHT_TREE_LEAVES_MAX - 1u);
			p = light_tree_leaf_probability_share(t.total, interval, uintBitsToFloat(keep ? e.z : e.w));
		}
		uint leaf = tree_leaf_list[at];
		if (leaf >= tree_node_count) {
			return false;
		}
		encoded = tree_nodes[leaf].child;
		power = tree_nodes[leaf].power;
		if ((encoded & RT_LIGHT_TREE_LEAF_BIT) == 0u) {
			return false;
		}
	}
	r_c.index = encoded & RT_LIGHT_TREE_INDEX_MASK;
	r_c.encoded = encoded;
	r_c.power = power;
	r_c.p = p;
	return r_c.p > 0.0;
}

/// One offer to the weighted reservoir (D51's form) on a rescaled number.
void light_tree_reservoir_offer(LightTreeCandidate c, float w, inout float w_sum, inout float u_res, inout LightTreeCandidate selected, inout float w_selected) {
	if (!(w > 0.0)) {
		return;
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

/// An analytic light's unshadowed irradiance at hit_pos facing the shading normal N, in the
/// units the shader lights with: its luminance under the windowed attenuation and the spot cone,
/// times the cosine at the receiver raised by the light's angular size and floored (a sphere light
/// on the horizon still delivers). Exactly zero only beyond the range and outside the cone, where
/// the sample would carry nothing. A thin surface's receiver (p_two_sided, D103) takes every kind
/// from either side: the receiver's cosine is its absolute value. A rectangle (D124) is zero behind
/// its plane; in front, its radiance under the closest-point attenuation times its solid angle,
/// approximated as area times cosine over the squared distance and capped at a hemisphere.
float light_tree_analytic_target(RTLightData light, vec3 hit_pos, vec3 N, bool p_two_sided) {
	float lum = abs(dot(light.emission, vec3(0.2126, 0.7152, 0.0722)));
	if (light.type == RT_LIGHT_TYPE_DIRECTIONAL) {
		vec3 L = -normalize(light.position);
		float cos_r = p_two_sided ? abs(dot(N, L)) : dot(N, L);
		return cos_r > 0.0 ? lum * cos_r : 0.0;
	}
	vec3 to_l = light.position - hit_pos;
	float dist_sq = dot(to_l, to_l);
	if (light.type == RT_LIGHT_TYPE_RECT) {
		// D124: one-sided, its radiance over the solid angle the rectangle subtends (its area times the
		// cosine at the light over the distance squared, at most a hemisphere), under the raster's
		// attenuation; the receiver's cosine raised by the angular size and floored, as the other kinds.
		vec3 half_x = lights_rect_half_x(light);
		vec3 half_y = lights_rect_half_y(light);
		vec3 normal = lights_rect_normal(light);
		float above_plane = dot(normal, hit_pos - light.position);
		if (above_plane <= 0.0) {
			return 0.0;
		}
		float atten = lights_rect_attenuation(light, hit_pos, half_x, half_y, normal);
		float rect_dist = sqrt(dist_sq);
		float cos_l = above_plane / max(rect_dist, 1e-4);
		float solid = min(4.0 * length(half_x) * length(half_y) * max(cos_l, 1e-3) / max(dist_sq, light.radius * light.radius), 2.0 * PI);
		vec3 L_rect = to_l / max(rect_dist, 1e-4);
		float n_dot_rect = dot(N, L_rect);
		float above_rect = max((p_two_sided ? abs(n_dot_rect) : n_dot_rect) + light.radius / max(rect_dist, light.radius), RT_LIGHT_TREE_RECEIVER_FLOOR);
		return lum * solid * atten * above_rect;
	}
	if (light.max_range_squared != 0.0 && dist_sq > light.max_range_squared) {
		return 0.0;
	}
	float dist = sqrt(dist_sq);
	vec3 L = to_l / max(dist, 1e-4);
	float spot_atten = 1.0;
	if (light.type == RT_LIGHT_TYPE_SPOT) {
		float scos = dot(-L, light.spot_direction);
		// The sampler tests the cone on a direction drawn within the light's angular size, so the
		// zero here is one angular size wider than the cone; inside it the falloff is floored. The
		// widened cone's cosine by the sum identity, -1 once the two angles reach a half turn.
		float cos_s = clamp(light.cos_spot_angle, -1.0, 1.0);
		float sin_s = sqrt(max(1.0 - cos_s * cos_s, 0.0));
		float sin_u = clamp(light.radius / max(dist, light.radius), 0.0, 1.0);
		float cos_u = sqrt(max(1.0 - sin_u * sin_u, 0.0));
		float cos_outer = cos_s * cos_u - sin_s * sin_u;
		float sin_outer = sin_s * cos_u + cos_s * sin_u;
		if (sin_outer <= 0.0 && cos_outer <= 0.0) {
			cos_outer = -1.0;
		}
		if (scos <= cos_outer) {
			return 0.0;
		}
		float spot_rim = clamp((1.0 - scos) / (1.0 - light.cos_spot_angle), 1e-4, 1.0);
		spot_atten = max(1.0 - pow(spot_rim, light.inv_spot_attenuation), RT_LIGHT_TREE_RECEIVER_FLOOR);
	}
	float atten = lights_get_attenuation(dist_sq, light.inv_max_range, light.attenuation, light.radius) * spot_atten;
	float n_dot = dot(N, L);
	float above = max((p_two_sided ? abs(n_dot) : n_dot) + light.radius / max(dist, light.radius), RT_LIGHT_TREE_RECEIVER_FLOOR);
	return lum * atten * above;
}

/// The target of a candidate that is its own cut slot, an omni or emissive leaf (D59): the light's
/// contribution estimate from the cut copy alone, so the candidate reads no light. An emitter's is
/// D51's proximity from the same sphere (the receiver's cosine raised by the angular size, floored
/// just above zero, over the distance floored by the sphere). An omni light's is the analytic target
/// itself: its luminance under Godot's attenuation (the light's own decay exponent, the leaf's
/// decay, inside the quartic window to its range) times the cosine raised by the angular
/// size and floored. An inverse-square stand-in for that curve cost the matrix 2.4 dB: the studio's
/// lights decay linearly. A spot leaf keeps the full target: its cone's zero is not in the node.
/// A thin surface's receiver (p_two_sided, D103) takes light from either side.
float light_tree_leaf_target(RTLightTreeNode n, vec3 hit_pos, vec3 N, bool p_two_sided) {
	vec3 to_c = n.center - hit_pos;
	float d2 = dot(to_c, to_c);
	float r2 = n.radius * n.radius;
	float d = sqrt(d2);
	float n_dot = p_two_sided ? abs(dot(N, to_c)) : dot(N, to_c);
	if ((n.child & RT_LIGHT_TREE_EMISSIVE_BIT) != 0u) {
		float above = max(n_dot + n.radius, 1e-3 * n.radius);
		float denominator = max(d, n.radius) * max(d2, r2);
		return (above > 0.0 && denominator > 0.0) ? n.power * above / denominator : 0.0;
	}
	float dist = max(d, 1e-4);
	float above = max(n_dot / dist + n.radius / max(dist, n.radius), RT_LIGHT_TREE_RECEIVER_FLOOR);
	float atten = lights_falloff(max(dist, max(n.radius, RT_LIGHT_MIN_DISTANCE)), n.decay);
	if (n.max_range > 0.0) {
		float nd = dist / n.max_range;
		nd *= nd;
		nd *= nd;
		nd = max(1.0 - nd, 0.0);
		atten *= nd * nd;
	}
	return n.power * atten * above;
}

/// The resampling target of a candidate: its unshadowed contribution estimate at the hit, in one
/// unit for every kind (an irradiance), against which the proposal's probability is corrected.
/// Any positive function is unbiased; the sky's is its own importance, so the proposal stands.
float light_tree_target(LightTreeCandidate c, LightTreeTop t, vec3 hit_pos, vec3 N, bool p_two_sided) {
	if (c.kind == RT_LIGHT_TREE_KIND_SKY) {
		return t.sky;
	}
	if (c.kind == RT_LIGHT_TREE_KIND_DIRECTIONAL) {
		return light_tree_analytic_target(rt_lights[c.index], hit_pos, N, p_two_sided);
	}
	if ((c.encoded & RT_LIGHT_TREE_EMISSIVE_BIT) != 0u) {
		// The entry's power at unit distance is the leaf's; the proximity is D51's target.
		return c.power * emissive_proximity(rt_emissive[c.index], hit_pos, N, p_two_sided);
	}
	return light_tree_analytic_target(rt_lights[c.index], hit_pos, N, p_two_sided);
}

/// One light sample by the tree: `candidates` draws through the proposal (the top-level split
/// and the pick over the cut, a random number each), each weighed by its target over its proposal
/// probability (an omni or emissive leaf slot targeted from its cut copy, D59), one kept by weighted
/// reservoir sampling on a single rescaled number (D51's form; the cut's candidates are offered in
/// slot order, which changes nothing in the distribution), then the kept candidate's own sample (an analytic light's cone or point, an emissive
/// triangle, a sky direction). The resampling corrects the proposal where its bounds are loose (a
/// receiver inside a cluster's sphere, a cluster in range by its bound alone) and below the pick
/// under a cut node (by power, D60, or uniform), so no sample is spent on a light that delivers
/// nothing when any candidate does. r_probability is what the caller divides the contribution by: the resampled
/// probability for an analytic light or the sky (their weights written as before, the sky's MIS
/// weight in the proposal's density the miss evaluates too), 1 for an emissive leaf, whose weight
/// already carries the proposal's density in its MIS weight and the resampled density under it.
/// Returns false when nothing was drawn or the sample carries nothing. Draws: one for the
/// reservoir, at most two per candidate (the top split, and the cut's place) and the leaf's pair.
/// Past the first diffuse bounce (is_indirect_bounce) the proposal is the indirect cut (D59); the
/// caller marks the payload so the next vertex's densities read it too. r_sky_probability and
/// r_total are what the next vertex weighs against (GPU audit 4a, light 1a): the sky's share and the
/// top's total, 0 when nothing was scored.
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
		out float r_probability,
		out float r_sky_probability,
		out float r_total) {
	r_sample.direction = vec3(0.0, 1.0, 0.0);
	r_sample.shadow_distance = 0.0;
	r_sample.leg_distance = 0.0;
	r_sample.radiance = vec3(0.0);
	r_sample.weight = 0.0;
	r_sample.specular_multiplier = 1.0;
	r_sample.hair_value = vec3(-1.0);
	r_probability = 1.0;

	// A strand (D126 H2) is lit from behind as from the front, so its proposal scores both sides of N_select: the next
	// vertex's chance hit reads the same through the payload's strand flag (HAIR_RAY_FLAG). A thin surface and a coat
	// keep the hemisphere's proposal; their targets below are two-sided.
	const bool proposal_two_sided = hasHair(material);
	LightTreeTop t = light_tree_top(hit_pos, N_select, is_indirect_bounce, proposal_two_sided);
	r_sky_probability = light_tree_sky_probability(t);
	r_total = t.total;
	if (light_tree_sky_only(t)) {
		return sky_sample_direct(N, V, material, rng_state, 1.0, r_sample);
	}
	if (!(t.total > 0.0)) {
		return false;
	}
	candidates = clamp(candidates, 1u, uint(RT_LIGHT_TREE_MAX_CANDIDATES));
	// A coat (D118 MM2) reflects light from below the shading normal too, so its receiver is two-sided.
	bool two_sided = hasDiffuseTransmission(material) || hasClearcoat(material) || hasHair(material); // A strand takes light from behind (D126 H2).
	float u_res = min(rand(rng_state), 0.99999994);
	LightTreeCandidate selected;
	selected.kind = RT_LIGHT_TREE_KIND_LEAF;
	selected.index = 0u;
	selected.encoded = 0u;
	selected.power = 0.0;
	selected.p = 0.0;
	float w_sum = 0.0;
	float w_selected = 0.0;
	// Every candidate's top split is drawn first and a sky or directional candidate is offered at
	// once; the cut's candidates are placed in one pass over the same running sum light_tree_top
	// made, their targets generated in ascending order as the order statistics of as many uniform
	// numbers (the largest of k uniforms is u^(1/k), the next the same of the rest scaled under
	// it: a sequence with the joint distribution of k independent uniforms sorted), so the pass
	// advances one target at a time and stores nothing per candidate. The reservoir's choice does
	// not depend on the order of the offers, and the draw count per candidate stays two.
	uint tree_pending = 0u;
	for (uint i = 0u; i < candidates; i++) {
		float u_top = min(rand(rng_state), 0.99999994);
		LightTreeCandidate c;
		if (light_tree_draw_top(u_top, t, N_select, proposal_two_sided, c)) {
			if (c.p > 0.0) {
				light_tree_reservoir_offer(c, light_tree_target(c, t, hit_pos, N, two_sided) / c.p, w_sum, u_res, selected, w_selected);
			}
		} else if (t.cut_count > 0u && t.root > 0.0) {
			tree_pending++;
		}
	}
	float running = 0.0;
	float order = 1.0; // The largest uniform still to place, descending; the target is 1 - order.
	float target = -1.0;
	if (tree_pending > 0u) {
		order *= pow(rand(rng_state), 1.0 / float(tree_pending));
		target = min(1.0 - order, 0.99999994) * t.root;
		tree_pending--;
	}
	for (uint s = 0u; s < t.cut_count && target >= 0.0; s++) {
		float below = running;
		RTLightTreeNode n = light_tree_cut_node(s, t.indirect);
		// The slot's probability is its node's importance over the total, the form the chance hit
		// evaluates for the slot alone (GPU audit light 1a); the running sum places the targets.
		// Named limit: the width a slot is drawn with, fl(below + importance) - below, differs from
		// the importance by up to half an ulp of the running sum (relative about 2^-25 running /
		// importance), so the two sides of the MIS weight agree only to that rounding, where the
		// old `running - below` was exact on both; a slot whose importance is under half an ulp of
		// the sum before it is never drawn, yet the chance hit weighs it. No cheap exact form exists:
		// the hit cannot know the drawn width without the whole sum. The leaf within the slot is
		// placed on the drawn width, so the pick below the slot stays exact.
		float interval = light_tree_node_importance(n, hit_pos, N_select, proposal_two_sided);
		running += interval;
		if (!(interval > 0.0)) {
			continue;
		}
		bool last = (s + 1u == t.cut_count);
		// Defensive: the target is below the sum by construction (min(u, 0.99999994) * root is under
		// root in fp32, and the last positive slot's running sum is root bit for bit), so the last
		// slot's fallthrough never fires.
		while (target >= 0.0 && (target < running || last)) {
			LightTreeCandidate c;
			if (light_tree_draw_leaf(target, below, interval, running - below, n, t, c)) {
				// A slot that is an omni or emissive leaf is targeted from the cut copy (D59), so the
				// candidate reads no light (its bound alone as the target, the sky's rule, lost 1.2 dB
				// on the matrix: the correction is what the target buys). A spot leaf and a light under
				// an inner node keep the light's own contribution estimate.
				bool leaf_slot = (n.child & RT_LIGHT_TREE_LEAF_BIT) != 0u && n.cos_theta_o <= -1.0;
				float w = (leaf_slot ? light_tree_leaf_target(n, hit_pos, N, two_sided) : light_tree_target(c, t, hit_pos, N, two_sided)) / c.p;
				light_tree_reservoir_offer(c, w, w_sum, u_res, selected, w_selected);
			}
			if (tree_pending > 0u) {
				order *= pow(rand(rng_state), 1.0 / float(tree_pending));
				target = min(1.0 - order, 0.99999994) * t.root;
				tree_pending--;
			} else {
				target = -1.0;
			}
		}
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
	if ((selected.encoded & RT_LIGHT_TREE_EMISSIVE_BIT) != 0u) {
		r_probability = 1.0;
		return emissive_sample_instance(rt_emissive[selected.index], selected.p, resample_factor, hit_pos, N, V, material, u2, 1.0, r_sample);
	}
	r_probability = selected.p * resample_factor;
	return lights_sample_light(rt_lights[selected.index], hit_pos, N, material, u2, is_indirect_bounce, r_sample);
}

#endif // !RT_STAGE_MISS
