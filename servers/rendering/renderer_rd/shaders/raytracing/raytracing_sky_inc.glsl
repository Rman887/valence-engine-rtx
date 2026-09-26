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
	float sky_share[RT_SKY_SHARE_COUNT]; // The candidate shares (D62), ascending from 0; the table writes them.
	vec2 sky_share_moments[RT_SKY_NORMALS * RT_SKY_SHARE_COUNT]; // Per receiver normal and share, the sky term's variance and its sky strategy's mean this frame (D65).
	float sky_basis[RT_SKY_NORMALS * RT_SKY_SHARE_COUNT * RT_SKY_BASIS]; // What the table's combine stage folds into sky_share_moments (D64); the trace reads none of it.
	float sky_marginal[RT_SKY_N]; // Inclusive CDF over the rows.
	float sky_conditional[RT_SKY_CELLS]; // Inclusive CDF over each row's cells.
	float sky_pdf_uv[RT_SKY_CELLS]; // The density per unit octahedral area inside the cell.
};

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
/// p_normal, so both vary continuously with it. The table holds them at eight candidate shares;
/// between two neighbours both are taken as linear in c, so on that interval
/// J'(c) = S^2 / (1 - c)^2 + a, a the slope of the linear part, and J's minimum there is
/// 1 - S / sqrt(-a) clamped to it. The share is the best of the intervals' minima (D66), which
/// moves continuously with S and the normal wherever J has one minimum: a share that stepped
/// between candidates left seams where S crossed a step (D62's seam capture, still there under
/// D65's per-normal model). J has one minimum where the sky's part V - 2 S mu is convex across the
/// candidates, which the table does not guarantee: on a receiver that sees only a smooth sky mu
/// grows as c^2 at small c, so at a small S J can hold two minima, and the share still jumps where
/// they tie; as S goes to 0 each interval's minimum is an end, a candidate again. Where the two
/// sides of the weight read S or the normal a rounding apart (the tree recomputes S, the 1e-7 of
/// D55's origin; the stages transform the normal) the densities differ by that rounding, and by a
/// step only at such a tie, a set of that measure. p_normal must be the normal the other side of
/// the weight reads: the geometry normal as the payload carries it on.
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
	// The sky's part of J at each candidate, V(c) - 2 S mu(c). The first candidate, sky_share[0], is
	// the share 0, which draws nothing from the sky: no covariance, V(0) alone.
	float c0 = 0.0;
	float sky0 = w.x * sky_share_moments[v00].x + w.y * sky_share_moments[v10].x + w.z * sky_share_moments[v01].x + w.w * sky_share_moments[v11].x;
	float best = sky0;
	float best_share = 0.0;
	for (uint k = 1u; k < RT_SKY_SHARE_COUNT; k++) {
		float c1 = sky_share[k];
		vec2 moments = w.x * sky_share_moments[v00 + k] + w.y * sky_share_moments[v10 + k] + w.z * sky_share_moments[v01 + k] + w.w * sky_share_moments[v11 + k];
		float sky1 = moments.x - 2.0 * p_surface * moments.y;
		float a = (sky1 - sky0) / (c1 - c0);
		// Where J rises through the interval its minimum is the left end, already judged.
		float c = (a < 0.0) ? clamp(1.0 - p_surface * inversesqrt(-a), c0, c1) : c0;
		float j = s2 * c / (1.0 - c) + sky0 + a * (c - c0);
		if (j < best) {
			best = j;
			best_share = c;
		}
		c0 = c1;
		sky0 = sky1;
	}
	return best_share;
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
