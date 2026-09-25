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
// One workgroup, one thread per row: each thread sums its row's cells (the cell weights parked
// in the table's own density slots until the total is known), the rows are reduced through
// shared memory, and every cell's density and both inclusive CDFs are written. The grid is fine
// enough that a sun painted in an HDRI (half a degree) fills a good part of its cell, so a sample
// drawn inside the cell is not a lottery between the disc and the sky beside it. The
// table also reports how much of the sky's light is concentrated (the cells brighter than
// SKY_CONCENTRATION_RATIO times the sky's mean radiance): a smooth sky is estimated well by the
// BSDF's own directions, and a light sample spent on it is one taken from a light whose sample
// had no variance, so the share the sky gets follows its concentrated light, not its total.
// For the share by variance (D62) the table also reports, at each of eight candidate shares, the
// variance of the sky's irradiance on a surface facing the sky's up when a share c of the light
// samples draws from this table and the surface's cosine lobe draws the bounce, the two combined
// by the balance heuristic: the second moment of the combined estimator, the integral of
// f^2 / (p_bsdf + c p_sky), less the squares of the two strategies' weighted means, and the sky
// strategy's weighted mean itself, which the share's covariance with the other lights needs
// (raytracing_sky_inc.glsl). A smooth sky reads nearly zero at c = 0 (the cosine lobe alone
// estimates it well) and rises with c; a sky that carries a small bright source reads high at
// c = 0 and falls as c grows. The balance heuristic, the upward Lambertian receiver and the
// sky without the fog's mix are a proxy for the trace's power heuristic over any surface: they
// choose the share, and any share is unbiased.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

#include "../oct_inc.glsl"

#define SKY_N 256u
#define SKY_CELLS (SKY_N * SKY_N)
#define SKY_CONCENTRATION_RATIO 4.0
#define SKY_FULL_SPHERE (4.0 * 3.14159265358979)
#define SKY_PI 3.14159265358979
#define SKY_SHARE_COUNT 8u

// Written to the table's sky_share, which the trace reads, so the list lives here alone.
const float SKY_SHARES[SKY_SHARE_COUNT] = float[](0.0, 1.0 / 32.0, 1.0 / 16.0, 0.125, 0.25, 0.5, 0.75, 0.9);

// The radiance octmap's sharpest layer as a 2D texture with its mip chain.
layout(set = 0, binding = 0) uniform texture2D radiance_octmap;
layout(set = 0, binding = 1) uniform sampler radiance_sampler;

// Must match SkyLightBuffer in raytracing_sky_inc.glsl and RT_SKY_LIGHT_BUFFER_SIZE.
layout(set = 0, binding = 2, std430) restrict buffer SkyLightBuffer {
	float sky_total; // The sum over cells of luminance times solid angle; 0 for a black sky.
	float sky_concentrated; // The part of sky_total in cells brighter than the ratio times the mean.
	float sky_pad[2];
	float sky_share[SKY_SHARE_COUNT]; // The candidate shares, SKY_SHARES (D62).
	float sky_variance[SKY_SHARE_COUNT]; // The sky term's variance at each share, before the exposure.
	float sky_light_mean[SKY_SHARE_COUNT]; // The sky strategy's MIS-weighted mean at each share, before the exposure.
	float sky_marginal[SKY_N]; // Inclusive CDF over the rows; the last entry reads 1.0.
	float sky_conditional[SKY_CELLS]; // Inclusive CDF over each row's cells, each row to 1.0.
	float sky_pdf_uv[SKY_CELLS]; // The density per unit octahedral area inside the cell.
};

layout(push_constant, std430) uniform Params {
	float border_pad; // The octmap's padding in UV space (radiance_border_size).
	float border_scale; // 1 - 2 * border_pad.
	float lod; // The octmap mip whose texel is about half a cell wide (0 when the mips are roughness levels, or the sharpest is already coarser).
	float floor_fraction; // The share of the sky's total spread evenly, so no cell has zero density.
}
params;

shared float row_sum[SKY_N];
shared float row_concentrated[SKY_N];
shared float row_moment[SKY_N];
shared float row_mean_bsdf[SKY_N];
shared float row_mean_sky[SKY_N];

float sky_luminance(vec3 p_color) {
	return dot(p_color, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
	uint row = gl_LocalInvocationID.x;
	uint row_base = row * SKY_N;
	float sum = 0.0;
	for (uint col = 0u; col < SKY_N; col++) {
		// Four taps inside the cell at the chosen mip: the cell's mean radiance, whatever the
		// distortion of the octahedral map inside it.
		float lum = 0.0;
		for (uint t = 0u; t < 4u; t++) {
			vec2 uv = (vec2(col, row) + vec2(0.25 + 0.5 * float(t & 1u), 0.25 + 0.5 * float(t >> 1u))) / float(SKY_N);
			vec3 dir = oct_to_vec3(uv * 2.0 - 1.0);
			vec2 tex_uv = vec3_to_oct_with_border(dir, vec2(params.border_pad, params.border_scale));
			lum += sky_luminance(textureLod(sampler2D(radiance_octmap, radiance_sampler), tex_uv, params.lod).rgb);
		}
		lum = max(lum * 0.25, 0.0);
		// The cell's solid angle from the map's Jacobian at its centre: an octahedral area element
		// d(uv) subtends 4 |d|_1^3 d(uv) of solid angle at the unit direction d it maps to.
		vec2 uv_centre = (vec2(col, row) + 0.5) / float(SKY_N);
		vec3 dir_centre = oct_to_vec3(uv_centre * 2.0 - 1.0);
		float l1 = abs(dir_centre.x) + abs(dir_centre.y) + abs(dir_centre.z);
		float solid_angle = 4.0 * l1 * l1 * l1 / float(SKY_CELLS);
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
	// Per share: the second moment and the two strategies' weighted means over this row's cells.
	float moment[SKY_SHARE_COUNT];
	float mean_bsdf[SKY_SHARE_COUNT];
	float mean_sky[SKY_SHARE_COUNT];
	for (uint k = 0u; k < SKY_SHARE_COUNT; k++) {
		moment[k] = 0.0;
		mean_bsdf[k] = 0.0;
		mean_sky[k] = 0.0;
	}
	for (uint col = 0u; col < SKY_N; col++) {
		uint cell = row_base + col;
		float lum = sky_conditional[cell]; // The parked luminance, read before it is overwritten.
		float w = sky_pdf_uv[cell] + floor_w;
		acc += w;
		sky_conditional[cell] = (col == SKY_N - 1u) ? 1.0 : acc / row_total;
		sky_pdf_uv[cell] = w / grand * float(SKY_CELLS);

		// The cell at its centre as seen from a surface facing +Y: the irradiance integrand, the
		// cosine lobe's density and the table's, each over solid angle.
		vec2 uv_centre = (vec2(col, row) + 0.5) / float(SKY_N);
		vec3 dir_centre = oct_to_vec3(uv_centre * 2.0 - 1.0);
		if (dir_centre.y > 0.0) {
			float l1 = abs(dir_centre.x) + abs(dir_centre.y) + abs(dir_centre.z);
			float solid_angle = 4.0 * l1 * l1 * l1 / float(SKY_CELLS);
			float f = lum * dir_centre.y;
			float p_bsdf = dir_centre.y / SKY_PI;
			float p_sky = w / grand / solid_angle;
			for (uint k = 0u; k < SKY_SHARE_COUNT; k++) {
				float q = SKY_SHARES[k] * p_sky;
				float f_over = solid_angle * f / (p_bsdf + q);
				moment[k] += f_over * f;
				mean_bsdf[k] += f_over * p_bsdf;
				mean_sky[k] += f_over * q;
			}
		}
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

	// The variance at each share, the rows reduced through shared memory one share at a time.
	for (uint k = 0u; k < SKY_SHARE_COUNT; k++) {
		barrier(); // The previous share's reduction has read the rows.
		row_moment[row] = moment[k];
		row_mean_bsdf[row] = mean_bsdf[k];
		row_mean_sky[row] = mean_sky[k];
		barrier();
		if (row == 0u) {
			float m = 0.0;
			float b = 0.0;
			float s = 0.0;
			for (uint r = 0u; r < SKY_N; r++) {
				m += row_moment[r];
				b += row_mean_bsdf[r];
				s += row_mean_sky[r];
			}
			sky_share[k] = SKY_SHARES[k];
			sky_variance[k] = max(m - b * b - s * s, 0.0);
			sky_light_mean[k] = s;
		}
	}
}
