// Shared definitions for raytracing shaders (raygen, miss, closest_hit, any_hit).
// Must be included before scene_data_inc.glsl since it defines MAX_VIEWS.
#include "../oct_inc.glsl"

#include "../oct_inc.glsl"

#define MAX_VIEWS 2

#ifndef PI
#define PI 3.141592653589f
#endif

// ============================================================================
// RT_PARAMS INDICES - Must match RT_PARAM_* in scene_shader_raytracing.h
// ============================================================================
// rt_params is a vec4[5] uniform buffer (20 floats total)
// Access: rt_params[idx >> 2][idx & 3] or get_rt_param(idx)
#define RT_PARAM_VIS_MODE 0 // rt_params[0].x - Debug visualization mode (0 = disabled)
#define RT_PARAM_SAMPLE_COUNT 1 // rt_params[0].y - Samples per pixel
#define RT_PARAM_MAX_BOUNCES 2 // rt_params[0].z - Maximum ray bounces
#define RT_PARAM_DENOISER 3 // rt_params[0].w - Denoiser selection (0=none, 1=DLSS RR)
#define RT_PARAM_PRIMARY_SURFACE 4 // rt_params[1].x - Primary surface (0=G-buffer, 1=traced)
// The froxel volumetric fog composed over the camera segment (D37).
#define RT_PARAM_VFOG_ENABLED 5 // rt_params[1].y - 1 when the frame carries a froxel volume
#define RT_PARAM_VFOG_INV_LENGTH 6 // rt_params[1].z - 1 / volumetric_fog_length
#define RT_PARAM_VFOG_INV_SPREAD 7 // rt_params[1].w - 1 / volumetric_fog_detail_spread
#define RT_PARAM_VFOG_SKY_AFFECT 8 // rt_params[2].x - volumetric_fog_sky_affect
#define RT_PARAM_HAS_PROCEDURAL_INSTANCES 9 // rt_params[2].y - TLAS contains procedural geometry
// Emissive mesh lights (D50): the count of instances in the emissive buffer, 0 when the
// Environment turns the sampling off or no eligible instance was submitted.
#define RT_PARAM_EMISSIVE_COUNT 10 // rt_params[2].z
#define RT_PARAM_EMISSIVE_PROBABILITY 11 // rt_params[2].w - Share of light samples sent to the emissive list
#define RT_PARAM_EMISSIVE_CANDIDATES 12 // rt_params[3].x - Candidates resampled per emissive light sample (D51), 1 to 16
#define RT_PARAM_CLAMP_DIRECT 13 // rt_params[3].y - Luminance clamp on a direct light sample, 0 for none (D53)
#define RT_PARAM_LIGHT_COUNT 14 // rt_params[3].z - Number of active lights in light buffer
#define RT_PARAM_FRAME_INDEX 15 // rt_params[3].w - Frame counter for temporal variation
#define RT_PARAM_CLAMP_INDIRECT 16 // rt_params[4].x - Luminance clamp on an indirect contribution, 0 for none (D53)
#define RT_PARAM_SKY_SURFACE_SCORE 17 // rt_params[4].y - The analytic and emissive lights' summed score the sky's share is judged against (D54)
#define RT_PARAM_DIRECTIONAL_COUNT 18 // rt_params[4].z - The directional lights at the head of the light buffer, the light tree's top-level entries (D55)
#define RT_PARAM_MAX_DIFFUSE_BOUNCES 19 // rt_params[4].w - The diffuse bounce limit, Environment.pathtracing_max_diffuse_bounces (D74)

// ============================================================================
// PATHTRACING PAYLOAD (52 bytes, fp16/unorm-packed)
// ============================================================================
// Radiance (r) and throughput (T) interleaved as fp16 into 3 uints:
//   [0]=rg, [1]=bR, [2]=GB  (packHalf2x16 pairs).
// The next-bounce origin is not stored: raygen reconstructs it from the
// current ray and hit_t as `offset_ray_origin(ray_origin + ray_dir * hit_t,
// offset_normal, hit_t)`. The offset normal and next direction are stored as
// 16-bit unorm octahedral pairs (effective angular error <~0.001 deg,
// below the 8-bit normal quantization baked into offset_ray_origin).
// Use PathState + path_pack/path_unpack for fp32 working copies.

struct PathPayload {
	uint packed_rt[3]; // 12 bytes - radiance+throughput interleaved as fp16
	uint packed_bounces_flags; //  4 bytes - [flags:8][unused:7][back offset:1, D103][diffuse:8][total:8]
	uint rng_state; //  4 bytes - the sampler's state (D52): the PCG chain, or the Sobol index and dimension
	float hit_t; //  4 bytes - ray distance, used by raygen to rebuild origin
	uint oct_offset_nrm; //  4 bytes - packUnorm2x16(vec3_to_oct(offset normal))
	uint oct_next_dir; //  4 bytes - packUnorm2x16(vec3_to_oct(next direction))
	// The medium the path is inside of (D49): the per-metre extinction and the in-scattered
	// radiance a segment adds, as fp16 pairs, and the height (trace space) of the interface the
	// path entered through, which stands for the medium's top when a light's leg is charged.
	uint packed_medium[3]; // 12 bytes - absorption.rg | absorption.b, inscatter.r | inscatter.gb
	float medium_entry_height; //  4 bytes
	// The density with which the last scatter chose the ray's direction (D50), so the next hit can
	// weigh the emission it finds against the emissive light sampling that competes for it;
	// RT_BSDF_PDF_DELTA for a lobe light sampling can never hit (a mirror, a refraction).
	float bsdf_pdf; //  4 bytes
};

const float RT_BSDF_PDF_DELTA = -1.0;

/// Unpacked fp32 working copy of the payload.
struct PathState {
	vec3 radiance;
	vec3 throughput;
	uint packed_bounces_flags;
	uint rng_state;
	float hit_t;
	vec3 offset_normal;
	vec3 next_ray_dir;
	vec3 medium_absorption;
	vec3 medium_inscatter;
	float medium_entry_height;
	float bsdf_pdf;
};

PathState path_unpack(PathPayload p) {
	PathState s;
	vec2 rg = unpackHalf2x16(p.packed_rt[0]);
	vec2 bR = unpackHalf2x16(p.packed_rt[1]);
	vec2 GB = unpackHalf2x16(p.packed_rt[2]);
	s.radiance = max(vec3(rg.x, rg.y, bR.x), vec3(0.0));
	s.throughput = max(vec3(bR.y, GB.x, GB.y), vec3(0.0));
	s.packed_bounces_flags = p.packed_bounces_flags;
	s.rng_state = p.rng_state;
	s.hit_t = p.hit_t;
	s.offset_normal = oct_to_vec3(unpackUnorm2x16(p.oct_offset_nrm) * 2.0 - 1.0);
	s.next_ray_dir = oct_to_vec3(unpackUnorm2x16(p.oct_next_dir) * 2.0 - 1.0);
	vec2 ab = unpackHalf2x16(p.packed_medium[0]);
	vec2 bi = unpackHalf2x16(p.packed_medium[1]);
	vec2 gb = unpackHalf2x16(p.packed_medium[2]);
	s.medium_absorption = max(vec3(ab.x, ab.y, bi.x), vec3(0.0));
	s.medium_inscatter = max(vec3(bi.y, gb.x, gb.y), vec3(0.0));
	s.medium_entry_height = p.medium_entry_height;
	s.bsdf_pdf = p.bsdf_pdf;
	return s;
}

void path_pack(inout PathPayload p, PathState s) {
	const float FP16_MAX = 65504.0;
	vec3 r = clamp(s.radiance, vec3(0.0), vec3(FP16_MAX));
	vec3 t = clamp(s.throughput, vec3(0.0), vec3(FP16_MAX));
	p.packed_rt[0] = packHalf2x16(vec2(r.r, r.g));
	p.packed_rt[1] = packHalf2x16(vec2(r.b, t.r));
	p.packed_rt[2] = packHalf2x16(vec2(t.g, t.b));
	p.packed_bounces_flags = s.packed_bounces_flags;
	p.rng_state = s.rng_state;
	p.hit_t = s.hit_t;
	p.oct_offset_nrm = packUnorm2x16(vec3_to_oct(s.offset_normal));
	p.oct_next_dir = packUnorm2x16(vec3_to_oct(s.next_ray_dir));
	vec3 a = clamp(s.medium_absorption, vec3(0.0), vec3(FP16_MAX));
	vec3 i = clamp(s.medium_inscatter, vec3(0.0), vec3(FP16_MAX));
	p.packed_medium[0] = packHalf2x16(vec2(a.r, a.g));
	p.packed_medium[1] = packHalf2x16(vec2(a.b, i.r));
	p.packed_medium[2] = packHalf2x16(vec2(i.g, i.b));
	p.medium_entry_height = s.medium_entry_height;
	p.bsdf_pdf = s.bsdf_pdf;
}

/// A path state with no medium: the fields every fresh path (a sample's start, a shadow ray)
/// carries before a refracted bounce sets them.
void path_clear_medium(inout PathState s) {
	s.medium_absorption = vec3(0.0);
	s.medium_inscatter = vec3(0.0);
	s.medium_entry_height = 0.0;
}

// Bounce count helpers (bits 0-7: total, bits 8-15: diffuse)
uint get_total_bounces(uint packed) {
	return packed & 0xFFu;
}
uint get_diffuse_bounces(uint packed) {
	return (packed >> 8u) & 0xFFu;
}
uint pack_bounces(uint total, uint diffuse) {
	return total | (diffuse << 8u);
}
uint inc_total_bounce(uint packed) {
	return packed + 1u;
}
uint inc_diffuse_bounce(uint packed) {
	return packed + 0x101u;
} // +1 to both total and diffuse

// The next ray leaves from the back of the surface (bit 16, D103): a thin surface's diffuse
// transmission. The origin is offset along -offset_normal, while offset_normal stays the receiver
// the light selection judged, so the next vertex's chance-hit and miss densities read it unchanged.
const uint BACK_OFFSET_FLAG = (1u << 16);
uint set_back_offset(uint packed, bool back) {
	return back ? (packed | BACK_OFFSET_FLAG) : (packed & ~BACK_OFFSET_FLAG);
}
bool is_back_offset(uint packed) {
	return (packed & BACK_OFFSET_FLAG) != 0u;
}

// Sample 0 flag (bit 24) - only write DLSS RR outputs on first sample
const uint SAMPLE_ZERO_FLAG = (1u << 24);
uint set_sample_zero(uint packed) {
	return packed | SAMPLE_ZERO_FLAG;
}
bool is_sample_zero(uint packed) {
	return (packed & SAMPLE_ZERO_FLAG) != 0u;
}

// Shadow ray flag (bit 25) - indicates this ray is a shadow/occlusion test
const uint SHADOW_RAY_FLAG = (1u << 25);
uint set_shadow_ray(uint packed) {
	return packed | SHADOW_RAY_FLAG;
}
bool is_shadow_ray(uint packed) {
	return (packed & SHADOW_RAY_FLAG) != 0u;
}

const uint PATH_TERMINATED_FLAG = (1u << 26);
// The primary ray missed (bit 27): the raygen composes the camera segment's media over the sky
// as the sky pass does, not over a hit distance (D37).
const uint PRIMARY_MISS_FLAG = (1u << 27);
// A bounded procedural primary probe. A miss clears this flag without touching
// the path or its images, so the raygen can shade the raster surface instead.
const uint PRIMARY_PROBE_FLAG = (1u << 28);
// The path is inside a medium (bit 29, D49): the next segment is charged with the payload's
// absorption and in-scatter instead of the Environment's fog, and a miss takes the full column.
const uint IN_MEDIUM_FLAG = (1u << 29);
uint set_in_medium(uint packed, bool inside) {
	return inside ? (packed | IN_MEDIUM_FLAG) : (packed & ~IN_MEDIUM_FLAG);
}
bool is_in_medium(uint packed) {
	return (packed & IN_MEDIUM_FLAG) != 0u;
}
// The path has scattered off a non-delta lobe (bit 30, D53): what it finds from here on is
// indirect light under the clamp. A refraction or a mirror reflection carries the camera's view
// on deterministically, so a surface seen through water or glass still counts as direct.
const uint SCATTERED_FLAG = (1u << 30);
uint set_scattered(uint packed) {
	return packed | SCATTERED_FLAG;
}
bool is_scattered(uint packed) {
	return (packed & SCATTERED_FLAG) != 0u;
}
// The previous vertex selected its light sample over the light tree's indirect cut (bit 31, D59),
// so the density a chance hit or a miss recovers for its MIS weight reads that cut. Set at the
// first vertex past a diffuse bounce and kept: the diffuse count only grows.
const uint LIGHT_TREE_INDIRECT_FLAG = (1u << 31);
uint set_light_tree_indirect(uint packed) {
	return packed | LIGHT_TREE_INDIRECT_FLAG;
}
bool is_light_tree_indirect(uint packed) {
	return (packed & LIGHT_TREE_INDIRECT_FLAG) != 0u;
}
uint set_path_terminated(uint packed) {
	return packed | PATH_TERMINATED_FLAG;
}
uint clear_path_terminated(uint packed) {
	return packed & ~PATH_TERMINATED_FLAG;
}
bool is_path_terminated(uint packed) {
	return (packed & PATH_TERMINATED_FLAG) != 0u;
}

// Bounce limits. The whole-path limit is RT_GET_MAX_BOUNCES() (a specialization constant) and the
// diffuse limit RT_PARAM_MAX_DIFFUSE_BOUNCES (a uniform, so changing it rebuilds no hit group; D74).
#define MAX_DENOISER_SPECULAR_HIT_THRESHOLD 0.25

// ============================================================================
// CONSTANTS
// ============================================================================
const uint OFFSET_NONE = 0xFFFFFFFFu;
const uint FLAG_COMPRESSED = 1u;
const uint FLAG_PROCEDURAL = 2u;
// Set when the BLAS uses a per-frame-deformed vertex buffer.
const uint FLAG_DEFORMED = 4u;
// Geometry residency (D93b): the surface's CLAS carry their cluster's index, and a hit reads its
// triangle from the cluster's blob (rt_resolve_triangle).
const uint FLAG_CLUSTER_GEOMETRY = 8u;

// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================
// Two samplers behind the payload's one uint of state, chosen by the Environment through the
// pipeline's RT_FLAG_SOBOL_SAMPLER (D52), so the one not chosen compiles away. Hash: NVIDIA's
// PCG chain, re-seeded per sample from the pixel, the frame and the sample index and re-hashed by every draw, so every number is independent white
// noise. Sobol: an Owen-scrambled (0,2)-sequence (Burley 2020, "Practical Hash-based Owen
// Scrambling"): each pixel walks its own shuffle of the sequence, one index per sample and per
// frame; every draw takes the next dimension (a rand2 the stratified pair of one dimension);
// and each dimension re-shuffles the index with a hash of the pixel and the dimension, so the
// draws of one path stay decorrelated and any number of dimensions is available. The state
// packs the sample's index in its pixel's sequence (the frame count times the sample count plus
// the sample, 20 bits, wrapping) above the dimension counter (12 bits).

#define RT_SOBOL_DIM_BITS 12u
#define RT_SOBOL_DIM_MASK 0xFFFu
#define RT_SOBOL_INDEX_MASK 0xFFFFFu
// Each bounce owns a block of dimensions (worst case 40 draws: the two strategy choices, up to 16 light
// picks and 16 acceptances, the cone pair, the transmission and lobe choices, the direction pair, the
// roulette, a thin surface's transmission choice, D103),
// so a draw keeps its dimension across frames whatever the blocks before it consumed; 8 bounces
// use 512 of the 4096 available.
#define RT_SOBOL_DIMS_PER_BOUNCE 64u
#define RT_SOBOL_DIM_SCATTER 40u // The transmission, lobe and direction draws inside a bounce's block.
#define RT_SOBOL_DIM_ROULETTE 43u // The Russian roulette draw (D73), after the scatter's three (a fourth scatter draw must move it; the u suffix keeps it out of #if).
#define RT_SOBOL_DIM_DIFFUSE_TRANSMISSION 44u // A thin surface's transmission choice (D103), apart so the scatter's three keep their dimensions.

// Both in raytracing_common_inc.glsl, which every stage includes after this file, after its
// ray-tracing extensions are enabled and where the specialization constant is declared.
bool rng_is_sobol();
uint rt_pixel_seed();

/// PCG random number generator - improved XSH-RR variant with better avalanche
uint pcg_hash(uint seed) {
	uint state = seed * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

/// A Laine-Karras style permutation in the form Vegdahl derived ("Building a better LK hash",
/// 2021, the one Cycles ships): every bit of x is flipped by a hash of the bits below it, so on the
/// reversed digits of a radical inverse it is a nested uniform (Owen) scramble keyed by seed. Every
/// step is a bijection only while each `x ^= x * c` constant is even and the plain multiplier odd;
/// an odd constant there zeroes a bit for every input.
uint sobol_lk_permute(uint x, uint seed) {
	x ^= x * 0x3d20adeau;
	x += seed;
	x *= (seed >> 16u) | 1u;
	x ^= x * 0x05526c56u;
	x ^= x * 0x53a22864u;
	return x;
}

/// Owen-scrambles x, a fixed-point number in [0, 1) with its first digit at bit 31; on an index
/// (first digit at bit 0) the same function shuffles the sequence while every aligned block of
/// 2^k indices stays a block, which keeps the prefix stratification the frames rely on.
uint sobol_owen_scramble(uint x, uint seed) {
	return bitfieldReverse(sobol_lk_permute(bitfieldReverse(x), seed));
}

/// Sobol's second dimension at index i, fixed point: its direction numbers are the rows of
/// Pascal's triangle modulo 2 (the first dimension is the bit reversal of i).
uint sobol_dim1(uint i) {
	uint r = 0u;
	for (uint v = 1u << 31; i != 0u; i >>= 1) {
		if ((i & 1u) != 0u) {
			r ^= v;
		}
		v ^= v >> 1;
	}
	return r;
}

/// The seed of the next dimension's shuffle and scrambles for this pixel; advances the state.
uint sobol_next_dimension(inout uint state, out uint index) {
	uint dim = state & RT_SOBOL_DIM_MASK;
	index = state >> RT_SOBOL_DIM_BITS;
	state = (index << RT_SOBOL_DIM_BITS) | ((dim + 1u) & RT_SOBOL_DIM_MASK);
	return pcg_hash(rt_pixel_seed() ^ pcg_hash(dim));
}

/// A fixed-point sample's 24 most significant digits as a float in [0, 1), 1.0 never returned.
float sobol_to_float(uint x) {
	return float(x >> 8) * (1.0 / 16777216.0);
}

/// Pins the next draw's dimension (the Sobol sampler only; nothing in hash mode).
void rng_set_dimension(inout uint state, uint dim) {
	if (rng_is_sobol()) {
		state = (state & ~RT_SOBOL_DIM_MASK) | (dim & RT_SOBOL_DIM_MASK);
	}
}

/// Initialize RNG state: the hash chain's seed, or the sample's index in its pixel's sequence.
uint init_rng(uvec2 pixel, uint frame, uint sample_idx, uint samples_per_pixel) {
	if (rng_is_sobol()) {
		return ((frame * samples_per_pixel + sample_idx) & RT_SOBOL_INDEX_MASK) << RT_SOBOL_DIM_BITS;
	}
	// Use Wang hash for better mixing - eliminates linear patterns
	uint seed = pixel.x + pixel.y * 65536u + frame * 1000000u + sample_idx * 100000000u;
	return pcg_hash(seed);
}

/// Random float in [0, 1)
float rand(inout uint state) {
	if (rng_is_sobol()) {
		uint index;
		uint seed = sobol_next_dimension(state, index);
		index = sobol_owen_scramble(index, seed);
		return sobol_to_float(sobol_owen_scramble(bitfieldReverse(index), pcg_hash(seed ^ 0x68bc21ebu)));
	}
	state = pcg_hash(state);
	return float(state) / 4294967296.0;
}

/// Random vec2 in [0, 1): under the Sobol sampler a stratified pair of one dimension.
vec2 rand2(inout uint state) {
	if (rng_is_sobol()) {
		uint index;
		uint seed = sobol_next_dimension(state, index);
		index = sobol_owen_scramble(index, seed);
		uint x = sobol_owen_scramble(bitfieldReverse(index), pcg_hash(seed ^ 0x68bc21ebu));
		uint y = sobol_owen_scramble(sobol_dim1(index), pcg_hash(seed ^ 0x02e5be93u));
		return vec2(sobol_to_float(x), sobol_to_float(y));
	}
	return vec2(rand(state), rand(state));
}

// ============================================================================
// RAY ORIGIN OFFSET (prevents self-intersection)
// Wachter-Binder method: offsets in integer float representation so the
// displacement scales naturally with position magnitude.
// ============================================================================
float _offset_component(float p, float n_comp, int of_comp) {
	const float origin = 1.0 / 32.0;
	const float float_scale = 1.0 / 65536.0;
	int shifted = floatBitsToInt(p) + ((p >= 0.0) ? of_comp : -of_comp);
	float p_i = intBitsToFloat(shifted);
	return (abs(p) < origin) ? (p + float_scale * n_comp) : p_i;
}

// t is the distance along the ray that found p. A position's error scales with the magnitudes
// that went into it, not with each component's own: a component near zero (a water plane at
// trace-space y = 0) took 256 ULP of itself, or a fixed 1/65536 below 1/32, less than the error
// of a position rebuilt along a long ray or from raster depth (the G-buffer primary). Each
// component therefore moves at least a few ULP of the largest coordinate (the anchor-relative
// grid, up to 1024 m from the camera under D20) and 1/65536 of the ray's length along n.
vec3 offset_ray_origin(vec3 p, vec3 n, float t) {
	const float int_scale = 256.0;
	const float floor_scale = 1.0 / 65536.0;
	ivec3 of = ivec3(int_scale * n);
	vec3 p_wb = vec3(
			_offset_component(p.x, n.x, of.x),
			_offset_component(p.y, n.y, of.y),
			_offset_component(p.z, n.z, of.z));
	float scale = max(max(max(abs(p.x), abs(p.y)), abs(p.z)) * (1.0 / 32.0), t);
	vec3 floor_offset = n * (floor_scale * scale);
	return mix(p_wb, p + floor_offset, greaterThan(abs(floor_offset), abs(p_wb - p)));
}
