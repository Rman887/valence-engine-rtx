// The sky as a light (D54): a share of every hit's light samples is drawn from the sky's
// radiance octmap in proportion to its brightness, through the table sky_light_cdf.glsl builds
// once per sky change, shadowed like any light and weighed with the power heuristic against the
// bounce that reaches the sky on its own, which the miss weighs the same way, so the sky is
// counted once. The sun is a directional light and stays one; the sky's own picture of it is
// part of the sky. The table is a 256 x 256 grid over the octahedral square with an inclusive CDF
// over the rows and one over each row's cells, and a density per unit octahedral area for each
// cell; the density over solid angle follows from the octahedral map's Jacobian at the
// direction (4 |d|_1^3 of solid angle per unit of area at the unit direction d), which both
// sides evaluate through sky_pdf, so the sample's density and the miss's are one function.
//
// Requires: raytracing_inc.glsl (oct_inc.glsl, rand2), scene_data_inc.glsl,
// raytracing_common_inc.glsl (RT_FLAGS, get_rt_param), radiance_octmap_sample and
// MAX_ROUGHNESS_LOD from the stage; in the hit stages brdf_inc.glsl (brdf_pdf) and
// raytracing_lights_inc.glsl (LightSampleNEE) too.

#ifndef RT_SKY_LIGHT_BINDING
#define RT_SKY_LIGHT_BINDING 43
#endif

#define RT_SKY_N 256u
#define RT_SKY_CELLS (RT_SKY_N * RT_SKY_N)
#define RT_SKY_SHARE_COUNT 8u // Must match SKY_SHARE_COUNT in sky_light_cdf.glsl and RT_SKY_LIGHT_SHARES.
#define RT_SKY_BASIS 12u // Must match SKY_BASIS in sky_light_cdf.glsl and RT_SKY_LIGHT_BASIS.
#define RT_SKY_NORMAL_CELLS 8u // Must match SKY_NORMAL_CELLS in sky_light_cdf.glsl and RT_SKY_LIGHT_NORMAL_CELLS.
#define RT_SKY_NORMALS ((RT_SKY_NORMAL_CELLS + 1u) * (RT_SKY_NORMAL_CELLS + 1u))

// Must match SkyLightBuffer in sky_light_cdf.glsl (std430, RT_SKY_LIGHT_BUFFER_SIZE bytes).
layout(set = 0, binding = RT_SKY_LIGHT_BINDING, std430) readonly buffer SkyLightBuffer {
	float sky_total; // The sum over cells of luminance times solid angle; 0 for a black sky or none built.
	float sky_concentrated; // The part of sky_total in cells brighter than four times the sky's mean.
	float sky_pad[2];
	float sky_share[RT_SKY_SHARE_COUNT]; // The candidate shares (D62), ascending from 0, as the table writes them; the trace reads RT_SKY_SHARES.
	vec2 sky_share_moments[RT_SKY_NORMALS * RT_SKY_SHARE_COUNT]; // Per receiver normal and share, the sky term's variance and its sky strategy's mean this frame (D65).
	float sky_basis[RT_SKY_NORMALS * RT_SKY_SHARE_COUNT * RT_SKY_BASIS]; // What the table's combine stage folds into sky_share_moments (D64); the trace reads none of it.
	float sky_marginal[RT_SKY_N]; // Inclusive CDF over the rows.
	float sky_conditional[RT_SKY_CELLS]; // Inclusive CDF over each row's cells.
	float sky_pdf_uv[RT_SKY_CELLS]; // The density per unit octahedral area inside the cell.
};

// The candidate shares the table's moments are built at, ascending from 0. Must match SKY_SHARES in
// sky_light_cdf.glsl (which writes them to sky_share). A copy at compile time so that
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
	uint v00 = (lo.y * (RT_SKY_NORMAL_CELLS + 1u) + lo.x) * RT_SKY_SHARE_COUNT;
	uint v10 = v00 + RT_SKY_SHARE_COUNT;
	uint v01 = v00 + (RT_SKY_NORMAL_CELLS + 1u) * RT_SKY_SHARE_COUNT;
	uint v11 = v01 + RT_SKY_SHARE_COUNT;
	vec4 w = vec4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
	float s2 = p_surface * p_surface;
	// The sky's part of J at each candidate, g_k = V(c_k) - 2 S mu(c_k), read bilinearly. The
	// candidate 0 is the share 0, which draws nothing from the sky: no covariance, V(0) alone. Eight
	// scalars, and the candidates' shares read by literal index from a compile-time array: a
	// ray-tracing stage pays two to three times for a local array it indexes at run time (D56).
#define SKY_MOMENTS(k) (w.x * sky_share_moments[v00 + k] + w.y * sky_share_moments[v10 + k] + w.z * sky_share_moments[v01 + k] + w.w * sky_share_moments[v11 + k])
#define SKY_G(k) dot(SKY_MOMENTS(k), vec2(1.0, -2.0 * p_surface))
	float g0 = SKY_MOMENTS(0u).x;
	float g1 = SKY_G(1u);
	float g2 = SKY_G(2u);
	float g3 = SKY_G(3u);
	float g4 = SKY_G(4u);
	float g5 = SKY_G(5u);
	float g6 = SKY_G(6u);
	float g7 = SKY_G(7u);
#undef SKY_G
#undef SKY_MOMENTS
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
			float d = p_surface * (1.0 / (1.0 - RT_SKY_SHARES[j])) - sqrt(-a);   \
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
	// (`light_tree_sky_importance`, the emissive density's 1 - share) hold to that range.
	return (best_a < 0.0) ? clamp(1.0 - p_surface * inversesqrt(-best_a), 0.0, RT_SKY_SHARES[7]) : best_share;
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
/// carries it on, which the miss and the emitter found after it read back.
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

/// The sky strategy's density over solid angle at a unit direction in the octmap's frame.
float sky_pdf(vec3 p_sky_dir) {
	vec2 uv = vec3_to_oct(p_sky_dir);
	uvec2 cell = min(uvec2(uv * float(RT_SKY_N)), uvec2(RT_SKY_N - 1u));
	float l1 = abs(p_sky_dir.x) + abs(p_sky_dir.y) + abs(p_sky_dir.z);
	return sky_pdf_uv[cell.y * RT_SKY_N + cell.x] / (4.0 * l1 * l1 * l1);
}

#ifndef RT_STAGE_MISS

/// The first row whose inclusive CDF reaches u.
uint sky_search_marginal(float u) {
	uint lo = 0u;
	uint hi = RT_SKY_N - 1u;
	while (lo < hi) {
		uint mid = (lo + hi) >> 1u;
		if (sky_marginal[mid] < u) {
			lo = mid + 1u;
		} else {
			hi = mid;
		}
	}
	return lo;
}

/// The first cell of a row whose inclusive CDF reaches u.
uint sky_search_conditional(uint p_row, float u) {
	uint base = p_row * RT_SKY_N;
	uint lo = 0u;
	uint hi = RT_SKY_N - 1u;
	while (lo < hi) {
		uint mid = (lo + hi) >> 1u;
		if (sky_conditional[base + mid] < u) {
			lo = mid + 1u;
		} else {
			hi = mid;
		}
	}
	return lo;
}

/// Draws one sky direction for next-event estimation from the table (two random numbers, the
/// remainders inside the chosen cell reused as the position within it), in the shape the
/// analytic and emissive strategies return: the weight carries the MIS weight against the BSDF
/// over the sample's density, and the caller divides by the strategy's share. False when the
/// direction lies under the surface, as a wasted light sample.
bool sky_sample_direct(vec3 N, vec3 V, MaterialProperties p_material, inout uint rng_state, float strategy_probability, out LightSampleNEE r_sample) {
	vec2 u = rand2(rng_state);
	uint row = sky_search_marginal(u.y);
	float row_lo = (row > 0u) ? sky_marginal[row - 1u] : 0.0;
	float row_in = clamp((u.y - row_lo) / max(sky_marginal[row] - row_lo, 1e-10), 0.0, 0.999999);
	uint col = sky_search_conditional(row, u.x);
	uint cell = row * RT_SKY_N + col;
	float col_lo = (col > 0u) ? sky_conditional[cell - 1u] : 0.0;
	float col_in = clamp((u.x - col_lo) / max(sky_conditional[cell] - col_lo, 1e-10), 0.0, 0.999999);
	vec2 uv = (vec2(col, row) + vec2(col_in, row_in)) / float(RT_SKY_N);
	vec3 sky_dir = oct_to_vec3(uv * 2.0 - 1.0);
	float pdf = sky_pdf(sky_dir);
	vec3 L = sky_dir * sky_world_to_sky();
	if (dot(N, L) <= 0.0 || pdf <= 0.0) {
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
