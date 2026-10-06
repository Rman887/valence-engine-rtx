#[compute]

#version 450

#VERSION_DEFINES

// Histogram auto exposure metering in three passes:
// BLOCK_PASS averages the image in linear radiance over 8x8 blocks, so the
// histogram is built from block means rather than from noisy single samples.
// HISTOGRAM_PASS bins the log2 scene luminance of each block mean.
// RESOLVE_PASS takes the mean log luminance between two percentiles, clamps it
// to the camera's bounds, adapts toward it from the previous frame and writes
// the metered luminance in the image's own units, the value the tonemap divides by.
// The image's own units are those before the frame's pre-exposure, so the stored value
// does not move when the pre-exposure does.

#define BLOCK_SIZE 8
#define HISTOGRAM_BINS 64

#ifdef RESOLVE_PASS
layout(local_size_x = HISTOGRAM_BINS, local_size_y = 1, local_size_z = 1) in;
#else
layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;
#endif

#ifdef BLOCK_PASS
layout(set = 0, binding = 0) uniform sampler2D source_texture;
layout(r32f, set = 1, binding = 0) uniform restrict writeonly image2D dest_blocks;

shared float block_sum[BLOCK_SIZE * BLOCK_SIZE];
shared uint block_count[BLOCK_SIZE * BLOCK_SIZE];
#endif

#ifdef HISTOGRAM_PASS
layout(r32f, set = 0, binding = 0) uniform restrict readonly image2D source_blocks;
#endif

#if defined(HISTOGRAM_PASS) || defined(RESOLVE_PASS)
#ifdef RESOLVE_PASS
#define HISTOGRAM_SET 0
#else
#define HISTOGRAM_SET 1
#endif
layout(set = HISTOGRAM_SET, binding = 0, std430) restrict buffer Histogram {
	uint bins[HISTOGRAM_BINS];
}
histogram;

shared uint local_bins[HISTOGRAM_BINS];
#endif

#ifdef RESOLVE_PASS
layout(r32f, set = 1, binding = 0) uniform restrict writeonly image2D dest_luminance;
layout(set = 2, binding = 0) uniform sampler2D prev_luminance;
#endif

layout(push_constant, std430) uniform Params {
	ivec2 source_size;
	float log_min; // Scene log2 luminance at the low edge of the first bin.
	float log_max; // Scene log2 luminance at the high edge of the last bin.
	float to_scene; // Image units to scene luminance (luminance multiplier / exposure normalization).
	float clamp_log_min; // The camera's bounds, scene log2 luminance.
	float clamp_log_max;
	float adjust_up; // Adaptation fraction this frame when the metered luminance rises.
	float adjust_down; // Same, when it falls (adapting to darkness).
	float low_fraction; // Window as fractions of the block count.
	float high_fraction;
	float source_scale; // 1 / the frame's pre-exposure: the metered image times this is in image units.
}
params;

void main() {
#ifdef BLOCK_PASS
	uint t = gl_LocalInvocationID.y * BLOCK_SIZE + gl_LocalInvocationID.x;
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

	block_sum[t] = 0.0;
	block_count[t] = 0u;
	if (all(lessThan(pos, params.source_size))) {
		vec3 color = texelFetch(source_texture, pos, 0).rgb;
		float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
		// NaN and Inf samples are left out, they would poison the block.
		if (!isnan(lum) && !isinf(lum)) {
			block_sum[t] = max(lum, 0.0);
			block_count[t] = 1u;
		}
	}
	groupMemoryBarrier();
	barrier();

	uint size = (BLOCK_SIZE * BLOCK_SIZE) >> 1;
	do {
		if (t < size) {
			block_sum[t] += block_sum[t + size];
			block_count[t] += block_count[t + size];
		}
		groupMemoryBarrier();
		barrier();
		size >>= 1;
	} while (size >= 1);

	if (t == 0) {
		// A block with no valid sample stores -1 and is not binned.
		float mean = block_count[0] > 0u ? block_sum[0] / float(block_count[0]) : -1.0;
		imageStore(dest_blocks, ivec2(gl_WorkGroupID.xy), vec4(mean));
	}
#endif

#ifdef HISTOGRAM_PASS
	uint t = gl_LocalInvocationID.y * BLOCK_SIZE + gl_LocalInvocationID.x;
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

	if (t < HISTOGRAM_BINS) {
		local_bins[t] = 0u;
	}
	groupMemoryBarrier();
	barrier();

	if (all(lessThan(pos, params.source_size))) {
		float mean = imageLoad(source_blocks, pos).r;
		if (mean >= 0.0) {
			float log_lum = log2(max(mean * params.source_scale * params.to_scene, 1e-30));
			float bin = (log_lum - params.log_min) / (params.log_max - params.log_min) * float(HISTOGRAM_BINS);
			atomicAdd(local_bins[uint(clamp(bin, 0.0, float(HISTOGRAM_BINS - 1)))], 1u);
		}
	}
	groupMemoryBarrier();
	barrier();

	if (t < HISTOGRAM_BINS && local_bins[t] > 0u) {
		atomicAdd(histogram.bins[t], local_bins[t]);
	}
#endif

#ifdef RESOLVE_PASS
	uint t = gl_LocalInvocationID.x;
	local_bins[t] = histogram.bins[t];
	groupMemoryBarrier();
	barrier();

	if (t == 0) {
		float total = 0.0;
		for (int i = 0; i < HISTOGRAM_BINS; i++) {
			total += float(local_bins[i]);
		}

		float prev = texelFetch(prev_luminance, ivec2(0, 0), 0).r;
		bool has_prev = prev > 0.0 && !isnan(prev) && !isinf(prev);
		float prev_log = has_prev ? log2(prev * params.to_scene) : 0.0;

		float target_log = prev_log;
		bool has_target = false;
		if (total > 0.0) {
			// Mean bin centre over the ranks [low, high), partial bins weighted by their overlap.
			// Capped below the total, so a histogram with any block in it always has a target.
			float low = min(total * params.low_fraction, total - 1.0);
			float high = max(total * params.high_fraction, low + 1.0);
			float bin_width = (params.log_max - params.log_min) / float(HISTOGRAM_BINS);
			float rank = 0.0;
			float weight_sum = 0.0;
			float log_sum = 0.0;
			for (int i = 0; i < HISTOGRAM_BINS; i++) {
				float count = float(local_bins[i]);
				float w = max(min(rank + count, high) - max(rank, low), 0.0);
				weight_sum += w;
				log_sum += w * (params.log_min + (float(i) + 0.5) * bin_width);
				rank += count;
			}
			if (weight_sum > 0.0) {
				target_log = log_sum / weight_sum;
				has_target = true;
			}
		}

		float result_log;
		if (!has_target && !has_prev) {
			// Nothing metered and nothing to keep: meter as a black frame, inside the bins.
			result_log = params.log_min + 0.5 * (params.log_max - params.log_min) / float(HISTOGRAM_BINS);
		} else if (!has_prev) {
			result_log = target_log;
		} else {
			target_log = clamp(target_log, params.clamp_log_min, params.clamp_log_max);
			float adjust = target_log > prev_log ? params.adjust_up : params.adjust_down;
			// Exponential in EV: the step is a fraction of the remaining distance in log2.
			result_log = mix(prev_log, target_log, adjust);
		}
		result_log = clamp(result_log, params.clamp_log_min, params.clamp_log_max);

		imageStore(dest_luminance, ivec2(0, 0), vec4(exp2(result_log) / params.to_scene));
	}
#endif
}
