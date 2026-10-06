// The sky as a light (D54): a share of every hit's light samples is drawn from the sky's
// radiance octmap in proportion to its brightness, through the table sky_light_table.glsl builds
// once per sky change, shadowed like any light and weighed with the power heuristic against the
// bounce that reaches the sky on its own, which the miss weighs the same way, so the sky is
// counted once. The sun is a directional light and stays one; the sky's own picture of it is
// part of the sky. The table is a pyramid over the octahedral square (D69) whose leaves are the
// octmap's own texels (2^sky_levels a side), each level the sums of four children stored as one
// vec4 per parent: a hit draws a leaf by descending from the top, choosing a column then a row of
// each quad in proportion to their sums and rescaling its two random numbers inside the part
// chosen (one 16-byte read per level, a stratified pair kept stratified), so a sun painted in an
// HDRI is drawn inside its own texel. A leaf's density is its weight over the sum per unit
// octahedral area, and the density over solid angle follows from the octahedral map's Jacobian
// at the direction (4 |d|_1^3 of solid angle per unit of area at the unit direction d), which
// both sides evaluate through sky_leaf_pdf: the sample's density and the miss's (sky_pdf, the
// leaf found from the direction) are one function of the leaf's weight.
//
// Requires: raytracing_inc.glsl (oct_inc.glsl, rand2), scene_data_inc.glsl,
// raytracing_common_inc.glsl (RT_FLAGS, get_rt_param), radiance_octmap_sample and
// MAX_ROUGHNESS_LOD from the stage; in the hit stages brdf_inc.glsl (brdf_pdf) and
// raytracing_lights_inc.glsl (LightSampleNEE) too.

#ifndef RT_SKY_LIGHT_BINDING
#define RT_SKY_LIGHT_BINDING 43
#endif

#define RT_SKY_SHARE_COUNT 8u // Must match SKY_SHARE_COUNT in sky_light_table.glsl and RT_SKY_LIGHT_SHARES.
#define RT_SKY_CLIP_LEVELS 6u // Must match SKY_CLIP_LEVELS in sky_light_table.glsl and RT_SKY_LIGHT_CLIP_LEVELS; level j clips at 2^(j + 1) times the receiver's irradiance.
// The share's objective in the display domain (D72): a sample's value is held to RT_SKY_CLIP_RATIO
// times the vertex's mean (the lights' score plus the sky's irradiance), and the energy the clip
// loses counts RT_SKY_CLIP_ACCUMULATION times its square against the clipped variance, the frames
// the upscaler's accumulation averages a still over. Both fitted to a fixed-share sweep of the
// HDRI captures against their references (docs/performance.md, D72).
#define RT_SKY_CLIP_RATIO 2.0
#define RT_SKY_CLIP_ACCUMULATION 16.0
#define RT_SKY_NORMAL_CELLS 8u // Must match SKY_NORMAL_CELLS in sky_light_table.glsl and RT_SKY_LIGHT_NORMAL_CELLS.
#define RT_SKY_NORMALS ((RT_SKY_NORMAL_CELLS + 1u) * (RT_SKY_NORMAL_CELLS + 1u))

// Must match SkyLightBuffer in sky_light_table.glsl (std430, RT_SKY_LIGHT_BUFFER_SIZE bytes).
layout(set = 0, binding = RT_SKY_LIGHT_BINDING, std430) readonly buffer SkyLightBuffer {
	float sky_total; // The sum over the leaves of luminance times solid angle; 0 for a black sky or none built.
	float sky_concentrated; // The part of sky_total in the 256-grid's cells brighter than four times the sky's mean.
	float sky_pdf_scale; // A leaf's floored weight times this is its density per unit octahedral area.
	float sky_floor_leaf; // The floor every leaf's weight carries, not stored in the pyramid.
	uint sky_levels; // The pyramid's levels below the top; the leaves are level sky_levels.
	uint sky_leaf_n; // 2^sky_levels, the leaves a side.
	uint sky_pad[2];
	float sky_share[RT_SKY_SHARE_COUNT]; // The candidate shares (D62), ascending from 0, as the table writes them; the trace reads RT_SKY_SHARES.
	// Per receiver normal, clip level and share (D72), in the table's units (the delivered luminance
	// over the sharpest level's weight in it): the variance of the sky term with every sample clipped
	// at the level, the sky strategy's clipped mean, the two strategies' clipped means summed, the
	// receiver's irradiance, at the index (normal * RT_SKY_CLIP_LEVELS + level) * RT_SKY_SHARE_COUNT
	// + share.
	vec4 sky_share_moments[RT_SKY_NORMALS * RT_SKY_CLIP_LEVELS * RT_SKY_SHARE_COUNT];
	// The pyramid (D69): level l's quads, row-major over the level above's nodes, each a node's four
	// children in the order (2x, 2y), (2x + 1, 2y), (2x, 2y + 1), (2x + 1, 2y + 1), levels 1 to sky_levels.
	vec4 sky_pyramid[];
};

// The candidate shares the table's moments are built at, ascending from 0. Must match SKY_SHARES in
// sky_light_table.glsl (which writes them to sky_share). A copy at compile time so that
// sky_variance_share reads them by literal index and the chords' spans fold to constants.
const float RT_SKY_SHARES[RT_SKY_SHARE_COUNT] = float[](0.0, 1.0 / 32.0, 1.0 / 16.0, 0.125, 0.25, 0.5, 0.75, 0.9);

bool sky_light_enabled() {
	return (RT_FLAGS & RT_FLAG_SKY_LIGHT) != 0u;
}

bool sky_variance_share_enabled() {
	return (RT_FLAGS & RT_FLAG_SKY_LIGHT_VARIANCE_SHARE) != 0u;
}

/// World direction to the octmap's frame, as the miss maps its ray.
mat3 sky_world_to_sky() {
	return scene_data_block.data.radiance_inverse_xform * mat3(scene_data_block.data.inv_view_matrix);
}

/// The sky's part of the objective at one candidate share (D72): the clipped variance less the
/// covariance with the lights' sample, plus the accumulation's weight on the squared energy the
/// clip loses of the irradiance, from the table's moments in its units and S in them.
float sky_share_objective(vec4 p_moments, float p_surface, float p_irradiance) {
	float lost = p_irradiance - p_moments.z;
	return p_moments.x - 2.0 * p_surface * p_moments.y + RT_SKY_CLIP_ACCUMULATION * lost * lost;
}

/// The moments' unit: the exposure normalisation the frame's weights of the delivered luminance
/// are divided by in the table (the camera exposure, the background's energy and its intensity over
/// what the sky was baked with, so 1 at the bake), never 0 while the sky renders.
float sky_moments_unit() {
	return scene_data_block.data.IBL_exposure_normalization;
}

/// The sky's share by the variance it removes (D62): the share that minimizes the variance of the
/// hit's light sample and the bounce's sky together. With the surface lights' summed score S taken
/// as their sample's mean (a directional sun, a lamp in range: little variance of their own, 2 to
/// 4 % measured where 24 lanterns reach, D64), a share c raises their second moment from S^2 to
/// S^2 / (1 - c); the sky's term at c is the table's variance V(c), judged for a Lambertian
/// receiver facing p_normal with the radiance the trace delivers (D64); and since the one sample is
/// either a light or the sky, the two branches covary by -2 S mu(c), mu the sky strategy's
/// weighted mean: J(c) = S^2 c / (1 - c) + V(c) - 2 S mu(c). A smooth sky beside a bright sun
/// keeps a share near 0, a sky with its own small bright source a large one. The receiver is the
/// vertex's own (D65): the table holds V and mu at the vertices of a grid of normals over the
/// octahedral square in the sky's frame, and they are read bilinearly between the four around
/// p_normal, so both vary continuously with it. The table holds them at eight candidate shares,
/// and between them the sky's part g(c) = V(c) - 2 S mu(c) is taken as its lower convex hull over
/// the candidates (D67): J is then strictly convex in c for any S above 0 (the first term is, and
/// the hull is convex), so it has one minimum, which moves continuously with S and the normal
/// everywhere; at S = 0 it is the least candidate, as under D66. The share that took g linear
/// between neighbouring candidates instead (D66) left J with two minima wherever g was not convex
/// across them, and a real table is not: on sky_outdoor's physical sky at 10 and 13 hours 76 of
/// the 81 normals held a double well, and the share jumped from about 0.3 to 0 as S crossed 0.1
/// to 0.4 of the receiver's sky irradiance (`docs/performance.md`, D67). The hull's value at any c
/// is the least of the chords between two candidates that span c, so J's minimum over the hull is
/// either at a candidate, whose own J is compared first, or inside one of the 28 chords: with a the
/// chord's slope, J'(c) = S^2 / (1 - c)^2 + a on it, zero at 1 - S / sqrt(-a), and the chord is
/// valued only where that point lies inside it (D68; at an end it is a candidate's value). The
/// chords' minima are compared, not their hull membership, so the minimum found is exact.
/// Where the two sides of the weight read S or the normal a rounding apart (the tree recomputes S,
/// the 1e-7 of D55's origin; the stages transform the normal) the densities differ by that
/// rounding times the share's sensitivity, large only where J is flat (S under a hundredth of
/// the sky's irradiance): with one minimum there is no tie for them to fall on opposite sides of.
/// Where the share is not 0 the search is 7 candidate values, 28 chord slopes with two compares
/// each and a square root and a few operations on the chords whose stationary point is inside
/// them, and one inverse square root at the end for the share: no division (D68; the cost as
/// measured is in `docs/performance.md`); 7 slopes where the share is 0.
/// p_normal must be the normal the other side of the weight reads: the geometry normal as the
/// payload carries it on.
float sky_variance_share(float p_surface, vec3 p_normal) {
	// The grid's vertices are RT_SKY_NORMAL_CELLS cells a side over the octahedral square, so the
	// four around any normal lie inside it and the weights need no wrap.
	vec2 g = vec3_to_oct(sky_world_to_sky() * p_normal) * float(RT_SKY_NORMAL_CELLS);
	uvec2 lo = min(uvec2(g), uvec2(RT_SKY_NORMAL_CELLS - 1u));
	vec2 f = clamp(g - vec2(lo), 0.0, 1.0);
	const uint vertex_stride = RT_SKY_CLIP_LEVELS * RT_SKY_SHARE_COUNT;
	uint v00 = (lo.y * (RT_SKY_NORMAL_CELLS + 1u) + lo.x) * vertex_stride;
	uint v10 = v00 + vertex_stride;
	uint v01 = v00 + (RT_SKY_NORMAL_CELLS + 1u) * vertex_stride;
	uint v11 = v01 + vertex_stride;
	vec4 w = vec4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
	// The moments are per unit of the exposure normalisation, and every term of J scales with its
	// square when S is taken over it, so the search runs in the table's units at S over the frame's
	// unit (D72).
	float unit = sky_moments_unit();
	if (!(unit > 0.0)) {
		return 0.0;
	}
	float s = p_surface / unit;
	float s2 = s * s;
	// The receiver's irradiance, and the clip as a multiple of it: RT_SKY_CLIP_RATIO times the
	// vertex's mean, read between the two levels around it (level j clips at 2^(j + 1) times the
	// irradiance); past the last level the clip stays at the last, which errs toward the share 0,
	// the side the lights' own term already takes there.
	float irradiance = w.x * sky_share_moments[v00].w + w.y * sky_share_moments[v10].w + w.z * sky_share_moments[v01].w + w.w * sky_share_moments[v11].w;
	if (!(irradiance > 0.0)) {
		return 0.0;
	}
	float level = clamp(log2(RT_SKY_CLIP_RATIO * (s + irradiance) / irradiance) - 1.0, 0.0, float(RT_SKY_CLIP_LEVELS - 1u));
	uint j0 = min(uint(level), RT_SKY_CLIP_LEVELS - 2u);
	float fj = clamp(level - float(j0), 0.0, 1.0);
	uint o0 = j0 * RT_SKY_SHARE_COUNT;
	uint o1 = o0 + RT_SKY_SHARE_COUNT;
	// The sky's part of J at each candidate, g_k = V'(c_k) - 2 S B(c_k) + A (E - M(c_k))^2 with V' the
	// clipped variance, B the sky strategy's clipped mean, M both strategies' and A the accumulation
	// (RT_SKY_CLIP_ACCUMULATION), read bilinearly over the normal and linearly between the levels.
	// The candidate 0 is the share 0, which draws nothing from the sky: no covariance. Eight scalars,
	// and the candidates' shares read by literal index from a compile-time array: a ray-tracing stage
	// pays two to three times for a local array it indexes at run time (D56).
#define SKY_LEVEL(o, k) (w.x * sky_share_moments[v00 + o + k] + w.y * sky_share_moments[v10 + o + k] + w.z * sky_share_moments[v01 + o + k] + w.w * sky_share_moments[v11 + o + k])
#define SKY_MOMENTS(k) mix(SKY_LEVEL(o0, k), SKY_LEVEL(o1, k), fj)
#define SKY_G(k) sky_share_objective(SKY_MOMENTS(k), s, irradiance)
	float g0 = SKY_G(0u);
	float g1 = SKY_G(1u);
	float g2 = SKY_G(2u);
	float g3 = SKY_G(3u);
	float g4 = SKY_G(4u);
	float g5 = SKY_G(5u);
	float g6 = SKY_G(6u);
	float g7 = SKY_G(7u);
#undef SKY_G
#undef SKY_MOMENTS
#undef SKY_LEVEL
	// Most hits take the share 0 (a sun over a smooth sky, D62). J over the hull has its minimum at
	// 0 exactly when its right derivative there, S^2 plus the least slope from candidate 0, is not
	// negative, and being convex it then beats every chord: those hits skip the search.
	float a0 = min(min(min((g1 - g0) * (1.0 / RT_SKY_SHARES[1]), (g2 - g0) * (1.0 / RT_SKY_SHARES[2])), min((g3 - g0) * (1.0 / RT_SKY_SHARES[3]), (g4 - g0) * (1.0 / RT_SKY_SHARES[4]))),
			min(min((g5 - g0) * (1.0 / RT_SKY_SHARES[5]), (g6 - g0) * (1.0 / RT_SKY_SHARES[6])), (g7 - g0) * (1.0 / RT_SKY_SHARES[7])));
	if (a0 >= -s2) {
		return 0.0;
	}
	// The minimum of J over the hull lies at a hull vertex, which is a candidate, or inside a hull
	// edge, which is one of the 28 chords. A candidate's own J is g_k + S^2 c_k / (1 - c_k), the odds
	// a constant of the candidate; a candidate above the hull rates no better than the hull under
	// it, so all eight are compared and the least is the base. Candidate 0 is g0.
	float best = g0;
	float best_share = 0.0;
#define SKY_ODDS(k) (RT_SKY_SHARES[k] / (1.0 - RT_SKY_SHARES[k]))
#define SKY_CANDIDATE(k, gk)               \
	{                                      \
		float jk = gk + s2 * SKY_ODDS(k);  \
		if (jk < best) {                   \
			best = jk;                     \
			best_share = RT_SKY_SHARES[k]; \
		}                                  \
	}
	SKY_CANDIDATE(1, g1)
	SKY_CANDIDATE(2, g2)
	SKY_CANDIDATE(3, g3)
	SKY_CANDIDATE(4, g4)
	SKY_CANDIDATE(5, g5)
	SKY_CANDIDATE(6, g6)
	SKY_CANDIDATE(7, g7)
#undef SKY_CANDIDATE
	// On the chord from candidate i to j with slope a, J'(c) = S^2 / (1 - c)^2 + a is zero at
	// c = 1 - S / sqrt(-a), which lies strictly inside the chord exactly when -a lies between
	// S^2 times 1 / (1 - c_i)^2 and S^2 times 1 / (1 - c_j)^2, constants of the chord; at either
	// end the minimum is a candidate's, compared above, so only the interior case is valued. There
	// J is J(c_j) - (1 - c_j) (S / (1 - c_j) - sqrt(-a))^2: the candidate's own value less a square,
	// which is g_j + a (1 - c_j) - S^2 + 2 S sqrt(-a) written so that no term exceeds J itself (the
	// expanded form cancels terms of S^2 / (1 - c)^2 on the short chords and drifts five times as
	// far in float). Valued from the chord's right end: V(0) is the largest moment by far (3e4
	// against 4e2 on the HDRI), and a chord from candidate 0 anchored there would carry V(0)'s
	// float spacing to every interior minimum, enough for a rounding to snap the share to a
	// neighbouring candidate. The winning slope is kept and the share taken from it once at the end.
	float best_a = 0.0;
#define SKY_INV_SQ(k) (1.0 / ((1.0 - RT_SKY_SHARES[k]) * (1.0 - RT_SKY_SHARES[k])))
#define SKY_CHORD(i, j, gi, gj)                                                  \
	{                                                                            \
		float a = (gj - gi) * (1.0 / (RT_SKY_SHARES[j] - RT_SKY_SHARES[i]));     \
		if (-a > s2 * SKY_INV_SQ(i) && -a < s2 * SKY_INV_SQ(j)) {                \
			float d = s * (1.0 / (1.0 - RT_SKY_SHARES[j])) - sqrt(-a);           \
			float jc = gj + s2 * SKY_ODDS(j) - (1.0 - RT_SKY_SHARES[j]) * d * d; \
			if (jc < best) {                                                     \
				best = jc;                                                       \
				best_a = a;                                                      \
			}                                                                    \
		}                                                                        \
	}
	SKY_CHORD(0, 1, g0, g1)
	SKY_CHORD(0, 2, g0, g2)
	SKY_CHORD(0, 3, g0, g3)
	SKY_CHORD(0, 4, g0, g4)
	SKY_CHORD(0, 5, g0, g5)
	SKY_CHORD(0, 6, g0, g6)
	SKY_CHORD(0, 7, g0, g7)
	SKY_CHORD(1, 2, g1, g2)
	SKY_CHORD(1, 3, g1, g3)
	SKY_CHORD(1, 4, g1, g4)
	SKY_CHORD(1, 5, g1, g5)
	SKY_CHORD(1, 6, g1, g6)
	SKY_CHORD(1, 7, g1, g7)
	SKY_CHORD(2, 3, g2, g3)
	SKY_CHORD(2, 4, g2, g4)
	SKY_CHORD(2, 5, g2, g5)
	SKY_CHORD(2, 6, g2, g6)
	SKY_CHORD(2, 7, g2, g7)
	SKY_CHORD(3, 4, g3, g4)
	SKY_CHORD(3, 5, g3, g5)
	SKY_CHORD(3, 6, g3, g6)
	SKY_CHORD(3, 7, g3, g7)
	SKY_CHORD(4, 5, g4, g5)
	SKY_CHORD(4, 6, g4, g6)
	SKY_CHORD(4, 7, g4, g7)
	SKY_CHORD(5, 6, g5, g6)
	SKY_CHORD(5, 7, g5, g7)
	SKY_CHORD(6, 7, g6, g7)
#undef SKY_CHORD
#undef SKY_ODDS
#undef SKY_INV_SQ
	// An interior minimum's share, 1 - S / sqrt(-a), computed once; a chord's a is negative
	// wherever it won. The gate holds the point inside the chord in exact arithmetic; an inverse
	// square root a rounding high can put it an ulp past 0 or 0.9, and the readers of the share
	// (`light_tree_sky_importance`, the emissive density's 1 - share) hold to that range. The lights'
	// own sample, S / (1 - c), is below the clip up to the share 1 - S / clip, which caps the result:
	// past it the objective's form no longer holds and the sample would saturate (D72).
	float share = (best_a < 0.0) ? clamp(1.0 - s * inversesqrt(-best_a), 0.0, RT_SKY_SHARES[7]) : best_share;
	return min(share, 1.0 - s / (RT_SKY_CLIP_RATIO * (s + irradiance)));
}

/// The sky's radiance along a direction in the octmap's frame, as the miss reports it: the
/// sharpest prefiltered level under the exposure normalisation, mixed toward the Environment's
/// fog colour (and its aerial perspective) by the fog's sky affect. Both strategies read this.
vec3 sky_radiance(vec3 p_sky_dir) {
	vec2 border = vec2(scene_data_block.data.radiance_border_size,
			1.0 - scene_data_block.data.radiance_border_size * 2.0);
	vec2 sky_uv = vec3_to_oct_with_border(p_sky_dir, border);

	vec3 sky_color = radiance_octmap_sample(sky_uv, 0.0);
	sky_color *= scene_data_block.data.IBL_exposure_normalization;

	if ((RT_FLAGS & RT_FLAG_FOG_ENABLED) != 0u) {
		vec3 fog_color = scene_data_block.data.fog_light_color;

		if (scene_data_block.data.fog_aerial_perspective > 0.0) {
			vec3 sky_fog = radiance_octmap_sample(sky_uv, 1.0 / MAX_ROUGHNESS_LOD);
			sky_fog *= scene_data_block.data.IBL_exposure_normalization;
			fog_color = mix(fog_color, sky_fog, scene_data_block.data.fog_aerial_perspective);
		}

		sky_color = mix(sky_color, fog_color, scene_data_block.data.fog_sky_affect);
	}
	return sky_color;
}

/// The share of light samples the sky takes this frame. Every hit takes one light sample, so
/// a share given to the sky is taken from the sun and the lamps, whose samples carry almost no
/// variance, while a smooth sky is estimated as well by the BSDF's own directions as by a sky
/// sample: the share follows the sky's concentrated light (the table's bright cells, a sun or
/// a window in an HDRI, a bright patch under a cloud), as a horizontal-surface irradiance (a
/// quarter of the total) against the analytic and emissive lights' summed score, with no floor,
/// so a clear sky under a directional sun draws nothing and an HDRI that carries its own sun
/// draws often; 0 with the feature off or a black sky, 1 when the sky is the only light. Any
/// share in [0, 1] is unbiased (at 0 the miss keeps its whole weight). With the share by
/// variance (D62) the same summed score is the S sky_variance_share judges against, for the
/// receiver p_normal (D65), so the share is the vertex's: its geometry normal as the payload
/// carries it on, which the miss and the emitter found after it read back. Since D72 that share
/// judges the clipped estimator the display shows, not the linear irradiance.
float sky_light_probability(vec3 p_normal) {
	if (!sky_light_enabled()) {
		return 0.0;
	}
	if (!(sky_total > 0.0)) {
		return 0.0;
	}
	if (get_rt_param(RT_PARAM_LIGHT_COUNT) == 0.0 && get_rt_param(RT_PARAM_EMISSIVE_COUNT) == 0.0) {
		return 1.0;
	}
	// A listed light of no or negative score still takes its share: the gate is the counts.
	float surface_score = max(get_rt_param(RT_PARAM_SKY_SURFACE_SCORE), 0.0);
	if (sky_variance_share_enabled()) {
		// The share by the clipped variance (D72) judges the image itself, so it takes no bound from
		// the concentration share (D71's, which the display objective replaced).
		return sky_variance_share(surface_score, p_normal);
	}
	float sky_score = 0.25 * sky_concentrated * scene_data_block.data.IBL_exposure_normalization;
	if (!(sky_score > 0.0)) {
		return 0.0;
	}
	return min(sky_score / (sky_score + surface_score), 0.9);
}

/// The power-heuristic weight of the strategy with density p_this against the other's p_other
/// (the emissive lights' form).
float sky_mis_weight(float p_this, float p_other) {
	float a = p_this * p_this;
	float b = p_other * p_other;
	return a / max(a + b, 1e-20);
}

/// The pyramid's quads of level p_level start after the (4^(l - 1) - 1) / 3 quads of the levels
/// above. Must match sky_light_table.glsl.
uint sky_quad_base(uint p_level) {
	return ((1u << (2u * p_level - 2u)) - 1u) / 3u;
}

/// The sky strategy's density over solid angle at a unit direction in the octmap's frame, for a
/// leaf of floored weight p_weight that holds the direction: the density per unit octahedral area
/// over the map's Jacobian there. Both sides of the MIS weight evaluate it.
float sky_leaf_pdf(float p_weight, vec3 p_sky_dir) {
	float l1 = abs(p_sky_dir.x) + abs(p_sky_dir.y) + abs(p_sky_dir.z);
	return p_weight * sky_pdf_scale / (4.0 * l1 * l1 * l1);
}

/// The sky strategy's density at a direction: its leaf, read from the pyramid by its coordinates.
/// 0 without a table: the buffer is zero until the first build and cleared when the Environment
/// has no sky, and the miss asks whatever the share, so the header's levels must not index it.
float sky_pdf(vec3 p_sky_dir) {
	if (!(sky_total > 0.0)) {
		return 0.0;
	}
	vec2 uv = vec3_to_oct(p_sky_dir);
	uint n = sky_leaf_n;
	uvec2 leaf = min(uvec2(uv * float(n)), uvec2(n - 1u));
	uint quad = sky_quad_base(sky_levels) + (leaf.y >> 1u) * (n >> 1u) + (leaf.x >> 1u);
	float w = sky_pyramid[quad][((leaf.y & 1u) << 1u) | (leaf.x & 1u)] + sky_floor_leaf;
	return sky_leaf_pdf(w, p_sky_dir);
}

#ifndef RT_STAGE_MISS

/// Draws one sky direction for next-event estimation by warping two random numbers down the
/// pyramid: at each level the quad of the node so far is read, a column is chosen by the two
/// columns' sums and the first number rescaled inside it, then a row of that column by its two
/// weights and the second number rescaled, so the pair that arrives at the leaf is uniform inside
/// it and is its position there. The floor is added level by level (a level's node covers four
/// times the leaves of the one below), never read from memory. Returns the sample in the shape the
/// analytic and emissive strategies return: the weight carries the MIS weight against the BSDF
/// over the sample's density, and the caller divides by the strategy's share. False when the
/// direction lies under the surface, as a wasted light sample.
bool sky_sample_direct(vec3 N, vec3 V, MaterialProperties p_material, inout uint rng_state, float strategy_probability, out LightSampleNEE r_sample) {
	vec2 u = rand2(rng_state);
	uint levels = sky_levels;
	uint n = sky_leaf_n;
	float floor_node = sky_floor_leaf * float((n * n) >> 2u); // A level-1 node's: a quarter of the leaves.
	uvec2 node = uvec2(0u);
	uint quad = 0u;
	float weight = 0.0;
	for (uint level = 1u; level <= levels; level++) {
		// Below 1: the hash sampler's rand and a rescaling's rounding can both reach 1.0 exactly, and a
		// split whose other part holds only the floor rounds to 1.0 too; with u under 1 a number that
		// takes the upper part has the part's probability under 1 as well, so no rescaling is 0 / 0.
		u = min(u, vec2(0.99999994));
		vec4 q = sky_pyramid[quad] + vec4(floor_node);
		float left = q.x + q.z;
		float p_left = left / (left + (q.y + q.w));
		uvec2 child;
		vec2 column;
		if (u.x < p_left) {
			child.x = 0u;
			u.x /= p_left;
			column = q.xz;
		} else {
			child.x = 1u;
			u.x = (u.x - p_left) / (1.0 - p_left);
			column = q.yw;
		}
		float p_low = column.x / (column.x + column.y);
		if (u.y < p_low) {
			child.y = 0u;
			u.y /= p_low;
			weight = column.x;
		} else {
			child.y = 1u;
			u.y = (u.y - p_low) / (1.0 - p_low);
			weight = column.y;
		}
		node = (node << 1u) + child;
		quad = sky_quad_base(level + 1u) + node.y * (1u << level) + node.x;
		floor_node *= 0.25;
	}
	u = clamp(u, vec2(0.0), vec2(0.999999));
	vec2 uv = (vec2(node) + u) / float(n);
	vec3 sky_dir = oct_to_vec3(uv * 2.0 - 1.0);
	float pdf = sky_leaf_pdf(weight, sky_dir);
	vec3 L = sky_dir * sky_world_to_sky();
	if ((dot(N, L) <= 0.0 && !hasDiffuseTransmission(p_material)) || !(pdf > 0.0)) {
		return false;
	}
	r_sample.direction = L;
	r_sample.shadow_distance = 10000.0; // The trace's reach, as the directional lights use.
	r_sample.leg_distance = 1e30;
	r_sample.radiance = sky_radiance(sky_dir);
	r_sample.specular_multiplier = 1.0;
	float pdf_bsdf = brdf_pdf(N, L, V, p_material);
	r_sample.weight = sky_mis_weight(strategy_probability * pdf, pdf_bsdf) / pdf;
	return true;
}

#endif // !RT_STAGE_MISS
