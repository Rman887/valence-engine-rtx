#[compute]

#version 450

#VERSION_DEFINES

#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Opacity micromap bake (D96, docs/r1-opacity-micromaps.md). A micromap is exact for the standard
// hit group's alpha test, which keeps a hit where
//     textureLod(albedo, uv * uv1_scale + uv1_offset, 0.0).a * albedo_color.a >= alpha_threshold,
// sampled linearly with repeat: the sample at any point is a blend of four level-0 texels, so it lies
// between their smallest and largest alpha. A micro-triangle is transparent when the largest alpha
// over every texel a bilinear sample inside it can read fails the test, opaque when the smallest
// passes, and unknown (the any-hit decides) otherwise. A margin of ALPHA_MARGIN on each side covers
// the filter's precision. An unknown micro-triangle is unknown-opaque when the test passes at its
// centre, sampled as the any-hit samples; only a ray that forces two states reads which (D96 OMMe).
//
// MODE_PYRAMID_BASE and MODE_PYRAMID_REDUCE build a min/max pyramid of the texture's alpha, so a box
// of any size is a few reads. MODE_SETUP takes one primitive a thread: its UVs as the trace computes
// them, its level, and whether the whole triangle is transparent or opaque. MODE_BAKE takes one
// 32-bit word of the micromap a thread, 16 micro-triangles in the space-filling curve's order.

#define STATE_TRANSPARENT 0u
#define STATE_OPAQUE 1u
#define STATE_UNKNOWN_TRANSPARENT 2u
#define STATE_UNKNOWN_OPAQUE 3u

// The pyramid's header, in words before its cells: per level k (1 to PYRAMID_MAX_LEVELS), its
// cells' first word, width and height. A cell is one word, its smallest alpha in the low 16 bits
// rounded down and its largest in the high 16 bits rounded up; 0 and 0xFFFF stand for any value
// below and above (an alpha outside [0, 1], or one that rounds there).
#define PYRAMID_MAX_LEVELS 15u
#define PYRAMID_HEADER_WORDS 48u

#define ALPHA_MARGIN (1.0 / 512.0)
// Texels a UV box grows by on each side, over the rounding of the trace's barycentric blend of UVs.
#define TEXEL_EPSILON (1.0 / 64.0)
#define ALPHA_BELOW -1.0e30
#define ALPHA_ABOVE 1.0e30

layout(push_constant, std430) uniform Params {
	uvec2 tex_size; // Level 0.
	uint level_count; // Pyramid levels above level 0.
	uint pass_level; // MODE_PYRAMID_*: the level written.
	vec2 uv_scale;
	vec2 uv_offset;
	float albedo_alpha;
	float threshold;
	float scale_texels; // Texels a micro-triangle covers (opacity_micromap_scale).
	uint max_level;
	uint count; // Primitives, or descriptors in this dispatch.
	uint base; // MODE_BAKE: the first descriptor of this dispatch.
	uint word_stride; // MODE_BAKE: words between one thread's iterations.
	uint pad;
}
params;

#if defined(MODE_PYRAMID_BASE) || defined(MODE_SETUP) || defined(MODE_BAKE)
layout(set = 0, binding = 0) uniform sampler2D alpha_texture;
#endif

layout(set = 0, binding = 1, std430) restrict buffer Pyramid {
	uint words[];
}
pyramid;

#ifdef MODE_SETUP
#define RT_STAGE_RAYGEN // Leaves out the hit-only helpers.
#include "raytracing_inc.glsl"
#include "raytracing_hit_inc.glsl"

layout(set = 0, binding = 2, std430) restrict readonly buffer GeometryRow {
	GeometryData geometry;
}
geometry_row;

// Per primitive: its three UVs (uv1 transformed), its level and its class.
layout(set = 0, binding = 3, std430) restrict writeonly buffer SetupOut {
	uint words[];
}
setup_out;
#endif

#ifdef MODE_BAKE
// Per descriptor: its three UVs, its level and its first word in the states.
layout(set = 0, binding = 2, std430) restrict readonly buffer BakeList {
	uint words[];
}
bake_list;

layout(set = 0, binding = 3, std430) restrict writeonly buffer States {
	uint words[];
}
states;

// Micro-triangles found known and unknown.
layout(set = 0, binding = 4, std430) restrict buffer Stats {
	uint known;
	uint unknown;
}
stats;
#endif

uint pack_cell(float p_min, float p_max) {
	uint lo = p_min <= 0.0 ? 0u : uint(clamp(floor(p_min * 65535.0), 0.0, 65534.0));
	uint hi = p_max >= 1.0 ? 0xFFFFu : uint(clamp(ceil(p_max * 65535.0), 1.0, 65535.0));
	return lo | (hi << 16u);
}

#if defined(MODE_SETUP) || defined(MODE_BAKE)
vec2 unpack_cell(uint p_cell) {
	uint lo = p_cell & 0xFFFFu;
	uint hi = p_cell >> 16u;
	return vec2(lo == 0u ? ALPHA_BELOW : float(lo) / 65535.0, hi == 0xFFFFu ? ALPHA_ABOVE : float(hi) / 65535.0);
}

// The smallest and largest alpha over texels [p_lo, p_hi] (inclusive, inside the texture): a box of up to 3 x 3
// texels read directly, a larger one from the pyramid level whose cells cover it in at most 3 x 3 (GPU audit geo 8).
// A level's cells are aligned blocks of 2^k texels, so the cells read span up to about three times the box per axis,
// where covering it in at most 2 x 2 spanned up to about four; the level is the same or the one below, for at most
// nine reads instead of four.
vec2 rect_min_max(ivec2 p_lo, ivec2 p_hi) {
	if (p_hi.x - p_lo.x <= 2 && p_hi.y - p_lo.y <= 2) {
		vec2 r = vec2(ALPHA_ABOVE, ALPHA_BELOW);
		for (int y = p_lo.y; y <= p_hi.y; y++) {
			for (int x = p_lo.x; x <= p_hi.x; x++) {
				float a = texelFetch(alpha_texture, ivec2(x, y), 0).a;
				r = vec2(min(r.x, a), max(r.y, a));
			}
		}
		return r;
	}
	uint k = 1u;
	while (k < params.level_count && (((p_hi.x >> k) - (p_lo.x >> k)) > 2 || ((p_hi.y >> k) - (p_lo.y >> k)) > 2)) {
		k++;
	}
	uint first = pyramid.words[(k - 1u) * 3u];
	uint width = pyramid.words[(k - 1u) * 3u + 1u];
	ivec2 c_lo = p_lo >> int(k);
	ivec2 c_hi = p_hi >> int(k);
	vec2 r = vec2(ALPHA_ABOVE, ALPHA_BELOW);
	for (int y = c_lo.y; y <= c_hi.y; y++) {
		for (int x = c_lo.x; x <= c_hi.x; x++) {
			vec2 cell = unpack_cell(pyramid.words[first + uint(y) * width + uint(x)]);
			r = vec2(min(r.x, cell.x), max(r.y, cell.y));
		}
	}
	return r;
}

// One axis of a texel range under repeat: at most two intervals inside [0, size).
int axis_intervals(float p_first, float p_last, int p_size, out ivec2 r_a, out ivec2 r_b) {
	if (p_last - p_first + 1.0 >= float(p_size)) {
		r_a = ivec2(0, p_size - 1);
		return 1;
	}
	float wrapped = p_first - float(p_size) * floor(p_first / float(p_size));
	int start = clamp(int(wrapped), 0, p_size - 1);
	int end = start + int(p_last - p_first);
	if (end < p_size) {
		r_a = ivec2(start, end);
		return 1;
	}
	r_a = ivec2(start, p_size - 1);
	r_b = ivec2(0, end - p_size);
	return 2;
}

// The state of the triangle of texture-space UVs p_a, p_b, p_c (uv1 transformed).
uint classify(vec2 p_a, vec2 p_b, vec2 p_c) {
	vec2 size = vec2(params.tex_size);
	vec2 a = p_a * size;
	vec2 b = p_b * size;
	vec2 c = p_c * size;
	vec2 lo = min(min(a, b), c) - 0.5;
	vec2 hi = max(max(a, b), c) - 0.5;
	// Out of range or not a number: undecided, as a magnitude past 2^20 texels loses the texel's place.
	if (!(all(greaterThan(lo, vec2(-1048576.0))) && all(lessThan(hi, vec2(1048576.0))))) {
		return STATE_UNKNOWN_OPAQUE;
	}
	// The trace blends the UVs in another order, off by a few units in the last place of the magnitude.
	float magnitude = max(max(abs(lo.x), abs(hi.x)), max(abs(lo.y), abs(hi.y)));
	float epsilon = TEXEL_EPSILON + magnitude * (1.0 / 1048576.0);
	lo -= epsilon;
	hi += epsilon;
	// A bilinear sample at x reads texels floor(x) and floor(x) + 1.
	vec2 first = floor(lo);
	vec2 last = floor(hi) + 1.0;
	ivec2 xa, xb, ya, yb;
	int xn = axis_intervals(first.x, last.x, int(params.tex_size.x), xa, xb);
	int yn = axis_intervals(first.y, last.y, int(params.tex_size.y), ya, yb);
	vec2 r = vec2(ALPHA_ABOVE, ALPHA_BELOW);
	for (int i = 0; i < xn; i++) {
		ivec2 xs = i == 0 ? xa : xb;
		for (int j = 0; j < yn; j++) {
			ivec2 ys = j == 0 ? ya : yb;
			vec2 m = rect_min_max(ivec2(xs.x, ys.x), ivec2(xs.y, ys.y));
			r = vec2(min(r.x, m.x), max(r.y, m.y));
		}
	}
	float smallest = r.x * params.albedo_alpha;
	float largest = r.y * params.albedo_alpha;
	if (smallest >= params.threshold + ALPHA_MARGIN) {
		return STATE_OPAQUE;
	}
	if (largest < params.threshold - ALPHA_MARGIN) {
		return STATE_TRANSPARENT;
	}
	// Undecided: the side of the test at the centre, through the bake's linear, repeating sampler at
	// level 0 as the any-hit samples. Only a ray that forces two states reads which (OMMe).
	vec2 centre = (p_a + p_b + p_c) * (1.0 / 3.0);
	return textureLod(alpha_texture, centre, 0.0).a * params.albedo_alpha >= params.threshold ? STATE_UNKNOWN_OPAQUE : STATE_UNKNOWN_TRANSPARENT;
}
#endif

#ifdef MODE_BAKE
// The space-filling curve of VK_EXT_opacity_micromap: micro-triangle p_index of level p_level as
// barycentrics (u the weight of vertex 1, v of vertex 2) of its three corners. After the OptiX SDK's
// index2bary, checked against the specification's reference index at every level up to 8.
uint extract_even_bits(uint x) {
	x &= 0x55555555u;
	x = (x | (x >> 1u)) & 0x33333333u;
	x = (x | (x >> 2u)) & 0x0f0f0f0fu;
	x = (x | (x >> 4u)) & 0x00ff00ffu;
	x = (x | (x >> 8u)) & 0x0000ffffu;
	return x;
}

uint prefix_eor(uint x) {
	x ^= x >> 1u;
	x ^= x >> 2u;
	x ^= x >> 4u;
	x ^= x >> 8u;
	return x;
}

void micro_triangle(uint p_index, uint p_level, out vec2 r_0, out vec2 r_1, out vec2 r_2) {
	uint b0 = extract_even_bits(p_index);
	uint b1 = extract_even_bits(p_index >> 1u);
	uint fx = prefix_eor(b0);
	uint fy = prefix_eor(b0 & ~b1);
	uint t = fy ^ b1;
	uint mask = (1u << p_level) - 1u;
	uint iu = ((fx & ~t) | (b0 & ~t) | (~b0 & ~fx & t)) & mask;
	uint iv = (fy ^ b0) & mask;
	uint iw = ((~fx & ~t) | (b0 & ~t) | (~b0 & fx & t)) & mask;
	bool upright = ((iu & 1u) ^ (iv & 1u) ^ (iw & 1u)) != 0u;
	if (!upright) {
		iu++;
		iv++;
	}
	float step_size = 1.0 / float(1u << p_level);
	float d = upright ? step_size : -step_size;
	vec2 uv = vec2(float(iu), float(iv)) * step_size;
	r_0 = uv;
	r_1 = uv + vec2(d, 0.0);
	r_2 = uv + vec2(0.0, d);
}

vec2 at_bary(vec2 p_uv0, vec2 p_uv1, vec2 p_uv2, vec2 p_bary) {
	return (1.0 - p_bary.x - p_bary.y) * p_uv0 + p_bary.x * p_uv1 + p_bary.y * p_uv2;
}
#endif

void main() {
	uint index = gl_GlobalInvocationID.x;

#ifdef MODE_PYRAMID_BASE
	// Level 1 from the texture's level 0.
	uint width = pyramid.words[1];
	uint height = pyramid.words[2];
	if (index >= width * height) {
		return;
	}
	ivec2 cell = ivec2(index % width, index / width);
	ivec2 size = ivec2(params.tex_size);
	float smallest = 1.0e30;
	float largest = -1.0e30;
	for (int y = 0; y < 2; y++) {
		for (int x = 0; x < 2; x++) {
			ivec2 texel = cell * 2 + ivec2(x, y);
			if (texel.x < size.x && texel.y < size.y) {
				float a = texelFetch(alpha_texture, texel, 0).a;
				smallest = min(smallest, a);
				largest = max(largest, a);
			}
		}
	}
	pyramid.words[pyramid.words[0] + index] = pack_cell(smallest, largest);
#endif

#ifdef MODE_PYRAMID_REDUCE
	// Level pass_level from the one below.
	uint k = params.pass_level;
	uint first = pyramid.words[(k - 1u) * 3u];
	uint width = pyramid.words[(k - 1u) * 3u + 1u];
	uint height = pyramid.words[(k - 1u) * 3u + 2u];
	if (index >= width * height) {
		return;
	}
	uint below_first = pyramid.words[(k - 2u) * 3u];
	uint below_width = pyramid.words[(k - 2u) * 3u + 1u];
	uint below_height = pyramid.words[(k - 2u) * 3u + 2u];
	uvec2 cell = uvec2(index % width, index / width);
	uint lo = 0xFFFFu;
	uint hi = 0u;
	for (uint y = 0u; y < 2u; y++) {
		for (uint x = 0u; x < 2u; x++) {
			uvec2 below = cell * 2u + uvec2(x, y);
			if (below.x < below_width && below.y < below_height) {
				uint v = pyramid.words[below_first + below.y * below_width + below.x];
				lo = min(lo, v & 0xFFFFu);
				hi = max(hi, v >> 16u);
			}
		}
	}
	pyramid.words[first + index] = lo | (hi << 16u);
#endif

#ifdef MODE_SETUP
	if (index >= params.count) {
		return;
	}
	GeometryData geom = geometry_row.geometry;
	uint i0, i1, i2;
	get_triangle_indices_ex(geom, index, i0, i1, i2);
	// The trace's own fetch at each corner's barycentrics, transformed as the any-hit transforms it.
	vec2 uv0 = fetch_uv(geom, i0, i1, i2, vec3(1.0, 0.0, 0.0)) * params.uv_scale + params.uv_offset;
	vec2 uv1 = fetch_uv(geom, i0, i1, i2, vec3(0.0, 1.0, 0.0)) * params.uv_scale + params.uv_offset;
	vec2 uv2 = fetch_uv(geom, i0, i1, i2, vec3(0.0, 0.0, 1.0)) * params.uv_scale + params.uv_offset;
	uint state = classify(uv0, uv1, uv2);
	uint level = 0u;
	if (state >= STATE_UNKNOWN_TRANSPARENT) {
		vec2 size = vec2(params.tex_size);
		vec2 e1 = (uv1 - uv0) * size;
		vec2 e2 = (uv2 - uv0) * size;
		float texels = 0.5 * abs(e1.x * e2.y - e1.y * e2.x);
		float wanted = ceil(log2(max(texels / params.scale_texels, 1.0)) * 0.5);
		level = uint(clamp(wanted, 1.0, float(params.max_level)));
	}
	uint base = index * 8u;
	setup_out.words[base + 0u] = floatBitsToUint(uv0.x);
	setup_out.words[base + 1u] = floatBitsToUint(uv0.y);
	setup_out.words[base + 2u] = floatBitsToUint(uv1.x);
	setup_out.words[base + 3u] = floatBitsToUint(uv1.y);
	setup_out.words[base + 4u] = floatBitsToUint(uv2.x);
	setup_out.words[base + 5u] = floatBitsToUint(uv2.y);
	setup_out.words[base + 6u] = level;
	setup_out.words[base + 7u] = state;
#endif

#ifdef MODE_BAKE
	uint descriptor = params.base + gl_GlobalInvocationID.y;
	if (gl_GlobalInvocationID.y >= params.count) {
		return;
	}
	uint base = descriptor * 8u;
	vec2 uv0 = vec2(uintBitsToFloat(bake_list.words[base + 0u]), uintBitsToFloat(bake_list.words[base + 1u]));
	vec2 uv1 = vec2(uintBitsToFloat(bake_list.words[base + 2u]), uintBitsToFloat(bake_list.words[base + 3u]));
	vec2 uv2 = vec2(uintBitsToFloat(bake_list.words[base + 4u]), uintBitsToFloat(bake_list.words[base + 5u]));
	uint level = bake_list.words[base + 6u];
	uint first_word = bake_list.words[base + 7u];
	uint micro_count = 1u << (2u * level);
	uint word_count = max(micro_count / 16u, 1u);
	uint known = 0u;
	uint unknown = 0u;
	for (uint w = index; w < word_count; w += params.word_stride) {
		uint word = 0u;
		for (uint j = 0u; j < 16u; j++) {
			uint micro = w * 16u + j;
			if (micro >= micro_count) {
				break;
			}
			vec2 b0, b1, b2;
			micro_triangle(micro, level, b0, b1, b2);
			uint state = classify(at_bary(uv0, uv1, uv2, b0), at_bary(uv0, uv1, uv2, b1), at_bary(uv0, uv1, uv2, b2));
			word |= state << (2u * j);
			if (state >= STATE_UNKNOWN_TRANSPARENT) {
				unknown++;
			} else {
				known++;
			}
		}
		states.words[first_word + w] = word;
	}
	if (known > 0u) {
		atomicAdd(stats.known, known);
	}
	if (unknown > 0u) {
		atomicAdd(stats.unknown, unknown);
	}
#endif
}
