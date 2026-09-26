#[compute]

#version 450

#VERSION_DEFINES

// The sky's sampling table for the path tracer (D54): a luminance distribution over the sky's
// radiance octmap, built once per sky change, from which a hit draws a sky direction in
// proportion to its brightness (raytracing_sky_inc.glsl). The table lives on a 256 x 256 grid over
// the octahedral square, so a direction's cell is found by the same mapping the miss reads the
// octmap with, and the density it carries is per unit of octahedral area: the solid-angle
// density at any direction follows from the octahedral map's Jacobian, which the sampling and
// the weighing sides evaluate alike, so no per-cell solid angle is ever needed for the density.
// MODE_TABLE builds it in one workgroup, one thread per row: each thread sums its row's cells (the
// cell weights parked in the table's own density slots until the total is known), the rows are
// reduced through shared memory, and every cell's density and both inclusive CDFs are written. The
// grid is fine enough that a sun painted in an HDRI (half a degree) fills a good part of its cell,
// so a sample drawn inside the cell is not a lottery between the disc and the sky beside it. The
// table also reports how much of the sky's light is concentrated (the cells brighter than
// SKY_CONCENTRATION_RATIO times the sky's mean radiance): a smooth sky is estimated well by the
// BSDF's own directions, and a light sample spent on it is one taken from a light whose sample
// had no variance, so the share the sky gets follows its concentrated light, not its total.
// For the share by variance (D62) MODE_MOMENTS then reports, at each of eight candidate shares, the
// integrals the variance of the sky's irradiance on a Lambertian receiver is built from, when a
// share c of the light samples draws from this table and the receiver's cosine lobe draws the
// bounce, the two combined by the power heuristic as the trace combines them: the second moment of
// the combined estimator, the integral of f^2 (q^3 + p^3) / (q^2 + p^2)^2 with p the lobe's
// density and q = c p_sky, and the two strategies' weighted means. The receiver is not only the one
// facing the sky's up (D65): one workgroup per normal of a grid over the octahedral square in the
// sky's frame, SKY_NORMAL_CELLS cells a side and a normal at every vertex, so the trace reads the
// four around its own normal bilinearly without a wrap (the vertices on the square's border that
// the octahedral fold identifies hold one direction, so the reading is continuous across it). The
// radiance the trace delivers is not the octmap's (D64): sky_radiance mixes the sharpest level L
// toward the fog, whose colour is the Environment's fog light mixed with a blurred level L_f by the
// aerial perspective, so along any direction it is alpha L + beta L_f + gamma, the weights set per
// frame by the fog and the exposure. The moments therefore hold, per normal and share, the
// second-moment integrals of every product of L, L_f and 1 and the two means' integrals of each,
// built only when the sky changes, and MODE_COMBINE folds them with the frame's weights into the
// variance and the sky strategy's mean the trace reads, run when the table or the weights change.
// A smooth sky reads nearly zero variance at c = 0 (the cosine lobe alone estimates it well) and
// rises with c; a sky that carries a small bright source reads high at c = 0 and falls as c grows.
// The Lambertian receiver, whose cosine lobe stands for the material's density, is a proxy for the
// trace's surfaces: it chooses the share, and any share is unbiased.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

#include "../oct_inc.glsl"

#define SKY_N 256u
#define SKY_CELLS (SKY_N * SKY_N)
#define SKY_CONCENTRATION_RATIO 4.0
#define SKY_FULL_SPHERE (4.0 * 3.14159265358979)
#define SKY_PI 3.14159265358979
#define SKY_SHARE_COUNT 8u
#define SKY_BASIS 12u // Must match RT_SKY_BASIS in raytracing_sky_inc.glsl and RT_SKY_LIGHT_BASIS.
#define SKY_NORMAL_CELLS 8u // Must match RT_SKY_NORMAL_CELLS in raytracing_sky_inc.glsl and RT_SKY_LIGHT_NORMAL_CELLS.
#define SKY_NORMALS ((SKY_NORMAL_CELLS + 1u) * (SKY_NORMAL_CELLS + 1u))

// The candidate shares, written to the table's sky_share; must match RT_SKY_SHARES, the trace's
// compile-time copy in raytracing_sky_inc.glsl (D67).
const float SKY_SHARES[SKY_SHARE_COUNT] = float[](0.0, 1.0 / 32.0, 1.0 / 16.0, 0.125, 0.25, 0.5, 0.75, 0.9);

#ifndef MODE_COMBINE
// The radiance octmap's sharpest layer as a 2D texture with its mip chain.
layout(set = 0, binding = 0) uniform texture2D radiance_octmap;
layout(set = 0, binding = 1) uniform sampler radiance_sampler;
#endif

#ifdef USE_RADIANCE_OCTMAP_ARRAY
// The octmap's roughness layers, for the fog's level, which is one of them.
layout(set = 0, binding = 3) uniform texture2DArray radiance_layers;
#endif

// Must match SkyLightBuffer in raytracing_sky_inc.glsl and RT_SKY_LIGHT_BUFFER_SIZE.
layout(set = 0, binding = 2, std430) restrict buffer SkyLightBuffer {
	float sky_total; // The sum over cells of luminance times solid angle; 0 for a black sky.
	float sky_concentrated; // The part of sky_total in cells brighter than the ratio times the mean.
	float sky_pad[2];
	float sky_share[SKY_SHARE_COUNT]; // The candidate shares, SKY_SHARES (D62).
	// Per receiver normal and share, the sky term's variance and the sky strategy's weighted mean
	// under the frame's weights (D65), what the trace reads.
	vec2 sky_share_moments[SKY_NORMALS * SKY_SHARE_COUNT];
	// Per receiver normal and share, SKY_BASIS integrals before the exposure (D64): the second
	// moment's over the products L L, L_f L_f, 1, L L_f, L, L_f, then the bounce's weighted mean over
	// L, L_f, 1, then the sky strategy's over L, L_f, 1.
	float sky_basis[SKY_NORMALS * SKY_SHARE_COUNT * SKY_BASIS];
	float sky_marginal[SKY_N]; // Inclusive CDF over the rows; the last entry reads 1.0.
	float sky_conditional[SKY_CELLS]; // Inclusive CDF over each row's cells, each row to 1.0.
	float sky_pdf_uv[SKY_CELLS]; // The density per unit octahedral area inside the cell.
};

layout(push_constant, std430) uniform Params {
	float border_pad; // The octmap's padding in UV space (radiance_border_size).
	float border_scale; // 1 - 2 * border_pad.
	float lod; // The octmap mip whose texel is about half a cell wide (0 when the mips are roughness levels, or the sharpest is already coarser).
	float floor_fraction; // The share of the sky's total spread evenly, so no cell has zero density.
	// The frame's weights of L, L_f and 1 in the delivered luminance (MODE_COMBINE), as the renderer
	// reads them from the scene data sky_radiance reads.
	float mix_level;
	float mix_fog_level;
	float mix_constant;
	float pad;
}
params;

#ifndef MODE_COMBINE

float sky_luminance(vec3 p_color) {
	return dot(p_color, vec3(0.2126, 0.7152, 0.0722));
}

// A direction inside a cell: tap t of the four the cell is read at.
vec3 sky_cell_tap(uint p_col, uint p_row, uint p_tap) {
	vec2 uv = (vec2(p_col, p_row) + vec2(0.25 + 0.5 * float(p_tap & 1u), 0.25 + 0.5 * float(p_tap >> 1u))) / float(SKY_N);
	return oct_to_vec3(uv * 2.0 - 1.0);
}

// Four taps inside the cell at the chosen mip: the cell's mean radiance, whatever the distortion
// of the octahedral map inside it.
float sky_cell_luminance(uint p_col, uint p_row) {
	float lum = 0.0;
	for (uint t = 0u; t < 4u; t++) {
		vec2 tex_uv = vec3_to_oct_with_border(sky_cell_tap(p_col, p_row, t), vec2(params.border_pad, params.border_scale));
		lum += sky_luminance(textureLod(sampler2D(radiance_octmap, radiance_sampler), tex_uv, params.lod).rgb);
	}
	return max(lum * 0.25, 0.0);
}

// The cell's centre direction and its solid angle from the map's Jacobian there: an octahedral area
// element d(uv) subtends 4 |d|_1^3 d(uv) of solid angle at the unit direction d it maps to.
vec3 sky_cell_centre(uint p_col, uint p_row, out float r_solid_angle) {
	vec2 uv_centre = (vec2(p_col, p_row) + 0.5) / float(SKY_N);
	vec3 dir_centre = oct_to_vec3(uv_centre * 2.0 - 1.0);
	float l1 = abs(dir_centre.x) + abs(dir_centre.y) + abs(dir_centre.z);
	r_solid_angle = 4.0 * l1 * l1 * l1 / float(SKY_CELLS);
	return dir_centre;
}

#endif // !MODE_COMBINE

#ifdef MODE_TABLE

shared float row_sum[SKY_N];
shared float row_concentrated[SKY_N];

void main() {
	uint row = gl_LocalInvocationID.x;
	uint row_base = row * SKY_N;
	float sum = 0.0;
	for (uint col = 0u; col < SKY_N; col++) {
		float lum = sky_cell_luminance(col, row);
		float solid_angle;
		sky_cell_centre(col, row, solid_angle);
		float w = lum * solid_angle;
		// Parked until the total is known: the weight in the density slot, the luminance in the
		// conditional slot; both are overwritten below by this same thread.
		sky_pdf_uv[row_base + col] = w;
		sky_conditional[row_base + col] = lum;
		sum += w;
	}
	row_sum[row] = sum;
	barrier();
	float total = 0.0;
	for (uint r = 0u; r < SKY_N; r++) {
		total += row_sum[r];
	}
	// The concentrated light: the cells above the ratio times the sphere's mean radiance.
	float threshold = SKY_CONCENTRATION_RATIO * total / SKY_FULL_SPHERE;
	float concentrated = 0.0;
	for (uint col = 0u; col < SKY_N; col++) {
		if (sky_conditional[row_base + col] > threshold) {
			concentrated += sky_pdf_uv[row_base + col];
		}
	}
	row_concentrated[row] = concentrated;
	barrier();
	// The floor keeps every cell reachable, so the density is positive wherever the octmap can
	// hold light; a black sky gets a uniform table that is never drawn from (its total is 0).
	float floor_w = (total > 0.0) ? params.floor_fraction * total / float(SKY_CELLS) : 1.0;
	float grand = total + floor_w * float(SKY_CELLS);
	float row_total = sum + floor_w * float(SKY_N);
	float acc = 0.0;
	for (uint col = 0u; col < SKY_N; col++) {
		uint cell = row_base + col;
		float w = sky_pdf_uv[cell] + floor_w;
		acc += w;
		sky_conditional[cell] = (col == SKY_N - 1u) ? 1.0 : acc / row_total;
		sky_pdf_uv[cell] = w / grand * float(SKY_CELLS);
	}
	float marginal = 0.0;
	for (uint r = 0u; r <= row; r++) {
		marginal += row_sum[r] + floor_w * float(SKY_N);
	}
	sky_marginal[row] = (row == SKY_N - 1u) ? 1.0 : marginal / grand;
	if (row == 0u) {
		float concentrated_total = 0.0;
		for (uint r = 0u; r < SKY_N; r++) {
			concentrated_total += row_concentrated[r];
		}
		sky_total = total;
		sky_concentrated = concentrated_total;
	}
	if (row < SKY_SHARE_COUNT) {
		sky_share[row] = SKY_SHARES[row];
	}
}

#endif // MODE_TABLE

#ifdef MODE_MOMENTS

shared float row_basis[SKY_BASIS][SKY_N + 1u]; // Padded: the twelve summing threads read one row each, a bank apart.

// The luminance of the level the fog's aerial perspective reads: sky_radiance's sample at a
// roughness of 1 / MAX_ROUGHNESS_LOD, which is the second roughness layer at its sharpest in array
// mode and the texture's second mip otherwise.
float sky_fog_luminance(vec2 p_tex_uv) {
#ifdef USE_RADIANCE_OCTMAP_ARRAY
	return sky_luminance(textureLod(sampler2DArray(radiance_layers, radiance_sampler), vec3(p_tex_uv, 1.0), 0.0).rgb);
#else
	return sky_luminance(textureLod(sampler2D(radiance_octmap, radiance_sampler), p_tex_uv, 1.0).rgb);
#endif
}

// The fog's level over the cell, from the same four taps as the cell's own luminance.
float sky_cell_fog_luminance(uint p_col, uint p_row) {
	float lum = 0.0;
	for (uint t = 0u; t < 4u; t++) {
		lum += sky_fog_luminance(vec3_to_oct_with_border(sky_cell_tap(p_col, p_row, t), vec2(params.border_pad, params.border_scale)));
	}
	return max(lum * 0.25, 0.0);
}

// One workgroup per receiver normal of the grid, one thread per row of the table MODE_TABLE built.
void main() {
	uint row = gl_LocalInvocationID.x;
	uint normal_index = gl_WorkGroupID.x;
	uvec2 vertex = uvec2(normal_index % (SKY_NORMAL_CELLS + 1u), normal_index / (SKY_NORMAL_CELLS + 1u));
	vec3 receiver = oct_to_vec3(vec2(vertex) / float(SKY_NORMAL_CELLS) * 2.0 - 1.0);
	// Per share, the integrals over this row's cells, in sky_basis's order.
	float basis[SKY_SHARE_COUNT][SKY_BASIS];
	for (uint k = 0u; k < SKY_SHARE_COUNT; k++) {
		for (uint i = 0u; i < SKY_BASIS; i++) {
			basis[k][i] = 0.0;
		}
	}
	for (uint col = 0u; col < SKY_N; col++) {
		// The cell at its centre as seen from the receiver: the cosine, the cosine lobe's density
		// and the table's, each over solid angle, and the power heuristic's weights.
		float solid_angle;
		vec3 dir_centre = sky_cell_centre(col, row, solid_angle);
		float cos_theta = dot(receiver, dir_centre);
		// Not merely positive: a receiver in the octahedron's equator at 45 degrees ((+-a, +-a, 0))
		// meets the cells whose centre has y = -x as the rounding residue of a*x - a*x, and at the
		// share 0 the second moment's (cos/pi)^4 then underflows to 0 / 0. Such a cell carries under
		// 1e-6 of a cell's cosine weight; at the floor every term stays a normal float.
		if (!(cos_theta > 1e-6)) {
			continue;
		}
		float lum = sky_cell_luminance(col, row);
		float lum_fog = sky_cell_fog_luminance(col, row);
		float p_bsdf = cos_theta / SKY_PI;
		float p2 = p_bsdf * p_bsdf;
		// The table's density over solid angle: per octahedral area over the Jacobian 4 |d|_1^3, which
		// is the cell's solid angle times the cell count, as sky_pdf reads it.
		float p_sky = sky_pdf_uv[row * SKY_N + col] / (solid_angle * float(SKY_CELLS));
		for (uint k = 0u; k < SKY_SHARE_COUNT; k++) {
			// With q = c p_sky the weights are q^2 / (q^2 + p^2) and p^2 / (q^2 + p^2), and the two
			// samples' second moments sum to f^2 (q^3 + p^3) / (q^2 + p^2)^2 over solid angle.
			float q = SKY_SHARES[k] * p_sky;
			float q2 = q * q;
			float sum2 = q2 + p2;
			float w_moment = solid_angle * cos_theta * cos_theta * (q2 * q + p2 * p_bsdf) / (sum2 * sum2);
			float w_bsdf = solid_angle * cos_theta * p2 / sum2;
			float w_sky = solid_angle * cos_theta * q2 / sum2;
			basis[k][0] += w_moment * lum * lum;
			basis[k][1] += w_moment * lum_fog * lum_fog;
			basis[k][2] += w_moment;
			basis[k][3] += w_moment * lum * lum_fog;
			basis[k][4] += w_moment * lum;
			basis[k][5] += w_moment * lum_fog;
			basis[k][6] += w_bsdf * lum;
			basis[k][7] += w_bsdf * lum_fog;
			basis[k][8] += w_bsdf;
			basis[k][9] += w_sky * lum;
			basis[k][10] += w_sky * lum_fog;
			basis[k][11] += w_sky;
		}
	}

	// The integrals at each share, the rows reduced through shared memory one share at a time, one
	// thread per integral.
	for (uint k = 0u; k < SKY_SHARE_COUNT; k++) {
		barrier(); // The previous share's reduction has read the rows.
		for (uint i = 0u; i < SKY_BASIS; i++) {
			row_basis[i][row] = basis[k][i];
		}
		barrier();
		if (row < SKY_BASIS) {
			float s = 0.0;
			for (uint r = 0u; r < SKY_N; r++) {
				s += row_basis[row][r];
			}
			sky_basis[(normal_index * SKY_SHARE_COUNT + k) * SKY_BASIS + row] = s;
		}
	}
}

#endif // MODE_MOMENTS

#ifdef MODE_COMBINE

// One thread per receiver normal and share: the moments' integrals of each product of L, L_f and 1
// combined with the frame's weights into the second moment, and of each level into the two
// strategies' means, then the variance (the two strategies' means both leave the second moment,
// since one sample of each is taken) and the sky strategy's mean the covariance reads.
void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= SKY_NORMALS * SKY_SHARE_COUNT) {
		return;
	}
	uint b = i * SKY_BASIS;
	vec3 m = vec3(params.mix_level, params.mix_fog_level, params.mix_constant);
	float moment = m.x * m.x * sky_basis[b] + m.y * m.y * sky_basis[b + 1u] + m.z * m.z * sky_basis[b + 2u] +
			2.0 * (m.x * m.y * sky_basis[b + 3u] + m.x * m.z * sky_basis[b + 4u] + m.y * m.z * sky_basis[b + 5u]);
	float mean_bsdf = dot(m, vec3(sky_basis[b + 6u], sky_basis[b + 7u], sky_basis[b + 8u]));
	float mean_sky = dot(m, vec3(sky_basis[b + 9u], sky_basis[b + 10u], sky_basis[b + 11u]));
	sky_share_moments[i] = vec2(max(moment - mean_bsdf * mean_bsdf - mean_sky * mean_sky, 0.0), mean_sky);
}

#endif // MODE_COMBINE
