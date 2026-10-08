/***************************************************************************
 # Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

// This software contains source code provided by NVIDIA Corporation.
//
// VAL (D117): a GLSL port of RTX Texture Filtering's library, RTXTF-Library 771b42b26a1f (RTXTF 1.3.0), whose
// unmodified HLSL is kept in thirdparty/rtxtf/ (STFDefinitions.h, STFSamplerState.hlsli,
// STFSamplerStateImpl.hlsli). Stochastic texture filtering: one texel per lookup, drawn with the probability of
// its filter weight, and the level drawn between the two around the requested one; shading the texel filters the
// shading (Pharr et al., "Filtering After Shading With Stochastic Texture Filtering", I3D 2024).
//
// The translation keeps the library's names and arithmetic. The STF_SamplerState facade and its implementation
// struct are one struct here, and the members that mutate the state are free functions taking it inout
// (STF_<Member>(inout STF_SamplerState, ...)); HLSL's Texture2D and SamplerState are a GLSL sampler2D.
//
// Ported: the definitions; the 2D filter picks (point, bilinear, bicubic B-spline, Gaussian); the level
// computation (the STF paper's minor-axis anisotropic level); the addressing helpers; and for 2D textures the
// Sample, Load and GetSamplePos forms at an explicit level (Level) or explicit gradients (Grad).
//
// Not ported, with the reason:
// - The forms without an explicit level (Texture2DSample, Load, GetSamplePos) and the Bias forms. The level-less
//   forms and GetSamplePosBias take ddx and ddy, which ray tracing stages do not have; the Sample and Load Bias
//   wrappers pass zero gradients upstream, which collapses their level to the bias.
// - The magnification paths: wave-shared magnification (STF_MAGNIFICATION_METHOD_2x2_* to 4x4_FINE) and
//   Collaborative Texture Filtering (MIN_MAX*, MASK*), Impl 619-2040. They read neighbouring lanes as a 32-lane
//   wave laid out as 8x4 screen pixels; a closest hit's lanes are whatever rays the scheduler packed, reordered by
//   shader execution reordering, so the reads would mix unrelated pixels. The magnification method is therefore
//   always NONE here; any other value set on the state behaves as NONE.
// - Texture2DArray, Texture3D and TextureCube: no hit shader samples them (custom hit groups compile only
//   sampler2D uniforms).

#ifndef RTXTF_STF_INC_GLSL
#define RTXTF_STF_INC_GLSL

// ============================================================================
// STFDefinitions.h
// ============================================================================

#define STF_PI 3.14159265359

// Filter types
#define STF_FILTER_TYPE_POINT 0
#define STF_FILTER_TYPE_LINEAR 1
#define STF_FILTER_TYPE_CUBIC 2
#define STF_FILTER_TYPE_GAUSSIAN 3

// Mip-Level overrides and bias
#define STF_MIP_VALUE_MODE_NONE 0
#define STF_MIP_VALUE_MODE_MIP_LEVEL 1
#define STF_MIP_VALUE_MODE_MIP_BIAS 2

// Magnification strategies (only NONE is ported, see above)
#define STF_MAGNIFICATION_METHOD_NONE 0
#define STF_MAGNIFICATION_METHOD_2x2_QUAD 1
#define STF_MAGNIFICATION_METHOD_2x2_FINE 2
#define STF_MAGNIFICATION_METHOD_2x2_FINE_TEMPORAL 3
#define STF_MAGNIFICATION_METHOD_3x3_FINE_ALU 4
#define STF_MAGNIFICATION_METHOD_3x3_FINE_LUT 5
#define STF_MAGNIFICATION_METHOD_4x4_FINE 6
#define STF_MAGNIFICATION_METHOD_MIN_MAX 7
#define STF_MAGNIFICATION_METHOD_MIN_MAX_HELPER 8
#define STF_MAGNIFICATION_METHOD_MIN_MAX_V2 9
#define STF_MAGNIFICATION_METHOD_MIN_MAX_V2_HELPER 10
#define STF_MAGNIFICATION_METHOD_MASK 11
#define STF_MAGNIFICATION_METHOD_MASK2 12

// When magnification fails we (sometimes) need a matching fallback strategy
#define STF_MAGNIFICATION_FALLBACK_METHOD_BL1STFILTER_FAST 0
#define STF_MAGNIFICATION_FALLBACK_METHOD_DEBUG 1

// No anisotropy
#define STF_ANISO_LOD_METHOD_NONE 0
// The method described in the STF paper
#define STF_ANISO_LOD_METHOD_DEFAULT 1

// Texture addressing modes - the values match D3D12
#define STF_ADDRESS_MODE_WRAP 1
#define STF_ADDRESS_MODE_MIRROR 2
#define STF_ADDRESS_MODE_CLAMP 3
#define STF_ADDRESS_MODE_BORDER 4
#define STF_ADDRESS_MODE_MIRROR_ONCE 5

#define STF_FLT_MAX 3.402823466e+38

// ============================================================================
// STFSamplerStateImpl.hlsli: the filter picks
// ============================================================================

ivec2 STF_NearestNeighbor2D(vec2 coordinate) {
	// VAL: roundEven, as HLSL's round() compiles under DXC (RoundEven); GLSL's round() leaves the halfway case open.
	return ivec2(roundEven(coordinate));
}

// Takes the input coordinate and computes a probability based on the distance the float value is to the floor of itself
// The random value u is used to compare whether to increment the coordinate based on the probability to select a neighboring integer for sampling
int STF_StochasticLinear(float coordinate, inout float u) {
	int s = int(floor(coordinate));

	// Compute the float remainder from the floor on the input value,
	// probabilityS is used as the probability that s input coordinate will be incremented
	const float probabilityS = coordinate - float(s);

	// Increment the input coordinate s if the random value u [0,1] is less than the probability to select an incremented coordinate
	if (u < probabilityS) {
		++s;

		// Generates a new random number from u
		u /= probabilityS;
	} else {
		// Generates a new random number from u
		u = (u - probabilityS) / (1.0 - probabilityS);
	}

	return s;
}

// Takes a 2D coordinate and computes a texel coordinate based on a random input [0,1]
ivec2 STF_StochasticBilinear(vec2 st, inout vec2 u) {
	ivec2 res;
	res.x = STF_StochasticLinear(st.x, u.x);
	res.y = STF_StochasticLinear(st.y, u.y);
	return res;
}

// Helper function to compute Bicubic weights
vec4 STF_GetStochasticBicubicWeights(float t) {
	const float t2 = t * t;

	vec4 w;
	w.x = (1.0 / 6.0) * (-t * t2 + 3.0 * t2 - 3.0 * t + 1.0);
	w.y = (1.0 / 6.0) * (3.0 * t * t2 - 6.0 * t2 + 4.0);
	w.z = (1.0 / 6.0) * (-3.0 * t * t2 + 3.0 * t2 + 3.0 * t + 1.0);
	w.w = (1.0 / 6.0) * t * t2;
	return w;
}

// Takes in bicubic weights to determine which texel in the bicubic footprint to select.
// Bicubic is a 4x4 texel area footprint and so we increment between [0-3]
int STF_StochasticLinear(vec4 w, float u) {
	float w_sum = w.x;

	if (u < w_sum) {
		return 0;
	}

	w_sum += w.y;

	if (u < w_sum) {
		return 1;
	}

	w_sum += w.z;

	if (u < w_sum) {
		return 2;
	}

	return 3;
}

ivec2 STF_StochasticBicubic(vec2 st, float u, float u2) {
	// Gather the bicubic weights for each coordinate
	vec4 ws = STF_GetStochasticBicubicWeights(st.x - floor(st.x));
	vec4 wt = STF_GetStochasticBicubicWeights(st.y - floor(st.y));

	// Sets a beggining texel location to offset from for the bicubic footprint
	int s0 = int(floor(st.x)) - 1;
	int t0 = int(floor(st.y)) - 1;

	// For each coordinate determine the texel location based on separate probabilities
	int s = STF_StochasticLinear(ws, u);
	int t = STF_StochasticLinear(wt, u2);

	return ivec2(s0 + s, t0 + t);
}

vec2 STF_BoxMullerTransform(vec2 u) {
	// VAL: u.x is kept off 0, where log() is infinite and the offset with it (upstream takes u.x as given).
	float mag = sqrt(-2.0 * log(max(u.x, 1.0 / 16777216.0)));
	return mag * vec2(cos(2.0 * STF_PI * u.y), sin(2.0 * STF_PI * u.y));
}

// Generates a 2D Gaussian by computing the Box-Muller 2D transform
ivec2 STF_StochasticGaussian2D(float sigma, vec2 st, vec2 u) {
	vec2 offset = sigma * STF_BoxMullerTransform(u);
	// VAL: roundEven, as in STF_NearestNeighbor2D.
	return ivec2(roundEven(st + offset));
}

// Method described in the stf paper
float STF_ComputeTextureLod_STF(uvec2 dim,
		vec4 textureGrads,
		float minLod,
		float maxLod,
		float mipBias) {
	const float dudx = float(dim.x) * textureGrads.x;
	const float dvdx = float(dim.y) * textureGrads.y;
	const float dudy = float(dim.x) * textureGrads.z;
	const float dvdy = float(dim.y) * textureGrads.w;

	vec2 maxAxis = vec2(dudy, dvdy);
	vec2 minAxis = vec2(dudx, dvdx);

	if (dot(minAxis, minAxis) > dot(maxAxis, maxAxis)) {
		minAxis = vec2(dudy, dvdy);
		maxAxis = vec2(dudx, dvdx);
	}

	float minAxisLength = length(minAxis);
	const float maxAxisLength = length(maxAxis);

	float maxAnisotropy = 16.0;
	if (minAxisLength > 0.0 && (minAxisLength * maxAnisotropy) < maxAxisLength) {
		float scale = maxAxisLength / (minAxisLength * maxAnisotropy);
		minAxisLength *= scale;
	}

	const float log2MinAxis = minAxisLength > 0.00001 ? log2(minAxisLength) : minLod;
	const float mipValue = clamp(log2MinAxis + mipBias, minLod, maxLod);
	return mipValue;
}

// If not using the sampler from d3d then the various filter variations need to be implemented in SW:

// Applies the texture addressing mode specified by 'mode' (one of the STF_ADDRESS_MODE_... constants)
// to the texture coordinate 'u' and returns the sample position inside the (0, 1) interval.
// The 'size' parameter is the texture size in the corresponding dimension on the right mip level.
// The 'isBorder' out parameter will be set to 'true' if the sample lands on a border in BORDER mode.
// Should be used in combination with one of the GetSamplePos functions above when no hardware sampler is available.
float STF_ApplyAddressingMode1D(float u,
		uint size,
		uint mode,
		out bool isBorder) {
	float invSize = 1.0 / float(size);
	isBorder = false;
	switch (int(mode)) {
		case STF_ADDRESS_MODE_MIRROR:
			return 1.0 - abs(1.0 - fract(u * 0.5) * 2.0);

		case STF_ADDRESS_MODE_CLAMP:
			return clamp(u, invSize, 1.0 - invSize);

		case STF_ADDRESS_MODE_BORDER:
			isBorder = (u <= 0.0) || (u >= 1.0);
			return clamp(u, invSize, 1.0 - invSize);

		case STF_ADDRESS_MODE_MIRROR_ONCE:
			return clamp(abs(u), invSize, 1.0 - invSize);

		case STF_ADDRESS_MODE_WRAP:
		default:
			return fract(u);
	}
}

// 2D version of the ApplyAddressingMode function. See STF_ApplyAddressingMode1D for more info.
vec2 STF_ApplyAddressingMode2D(vec2 uv,
		uvec2 size,
		uvec2 modes,
		out bool isBorder) {
	bvec2 borders;
	vec2 results;
	results.x = STF_ApplyAddressingMode1D(uv.x, size.x, modes.x, borders.x);
	results.y = STF_ApplyAddressingMode1D(uv.y, size.y, modes.y, borders.y);
	isBorder = any(borders);
	return results;
}

// ============================================================================
// STFSamplerState.hlsli and STFSamplerStateImpl.hlsli: the state
// ============================================================================

struct STF_SamplerState {
	uint m_filterType; // STF_FILTER_TYPE_*
	uint m_frameIndex;
	uint m_anisoMethod; // STF_ANISO_LOD_METHOD_*
	uint m_magMethod; // STF_MAGNIFICATION_METHOD_* (only NONE is ported)
	uint m_fallbackMethod; // STF_MAGNIFICATION_FALLBACK_METHOD_*
	uvec3 m_addressingModes; // STF_ADDRESS_MODE_*
	float m_sigma; // used for Gaussian kernel
	vec4 m_u; // uniform random number(s)
	bool m_reseedOnSample;
	bool m_debugFallback; // return debug color to visualize mag 3.0 failures.
	uvec4 m_userData; // User data, can be used to store some application specific auxilary data in the sampler object
};

// u - Uniform random number(s) Format: 2D(Array) tex : [u, v, (slice), mip], 3D tex : [u, v, w, mip]
STF_SamplerState STF_Create(vec4 u) {
	STF_SamplerState p;
	p.m_filterType = STF_FILTER_TYPE_LINEAR;
	p.m_frameIndex = 0u;
	p.m_anisoMethod = STF_ANISO_LOD_METHOD_DEFAULT;
	// VAL: NONE, upstream 2x2_QUAD; the wave-shared methods are not ported (see the top of this file).
	p.m_magMethod = STF_MAGNIFICATION_METHOD_NONE;
	p.m_fallbackMethod = STF_MAGNIFICATION_FALLBACK_METHOD_BL1STFILTER_FAST;
	p.m_addressingModes = uvec3(STF_ADDRESS_MODE_WRAP, STF_ADDRESS_MODE_WRAP, STF_ADDRESS_MODE_WRAP);
	p.m_sigma = 0.7;
	p.m_u = u;
	// VAL: set to its documented default, false; upstream leaves it uninitialized.
	p.m_reseedOnSample = false;
	p.m_debugFallback = false;
	p.m_userData = uvec4(0u);
	return p;
}

// Uniform random numbers, 2D(Array) tex : [u, v, (slice), mip], 3D tex : [u, v, w, mip]
void STF_SetUniformRandom(inout STF_SamplerState s, vec4 u) {
	s.m_u = u;
}

// STF_FILTER_TYPE_*
void STF_SetFilterType(inout STF_SamplerState s, uint filterType) {
	s.m_filterType = filterType;
}

void STF_SetFrameIndex(inout STF_SamplerState s, uint frameIndex) {
	s.m_frameIndex = frameIndex;
}

// STF_ANISO_LOD_METHOD_*
void STF_SetAnisoMethod(inout STF_SamplerState s, uint anisoMethod) {
	s.m_anisoMethod = anisoMethod;
}

// STF_MAGNIFICATION_METHOD_* (only NONE is ported)
void STF_SetMagMethod(inout STF_SamplerState s, uint magMethod) {
	s.m_magMethod = magMethod;
}

// STF_MAGNIFICATION_FALLBACK_METHOD_*
void STF_SetFallbackMethod(inout STF_SamplerState s, uint fallbackMethod) {
	s.m_fallbackMethod = fallbackMethod;
}

// STF_ADDRESS_MODE_* - Only applied when using 'Load'
void STF_SetAddressingModes(inout STF_SamplerState s, uvec3 addressingModes) {
	s.m_addressingModes = addressingModes;
}

void STF_SetSigma(inout STF_SamplerState s, float sigma) {
	s.m_sigma = sigma;
}

// Each sample will update the random numbers default: false
void STF_SetReseedOnSample(inout STF_SamplerState s, bool reseedOnSample) {
	s.m_reseedOnSample = reseedOnSample;
}

void STF_SetUserData(inout STF_SamplerState s, uvec4 userData) {
	s.m_userData = userData;
}

void STF_SetDebugFailure(inout STF_SamplerState s, bool debugFailure) {
	s.m_debugFallback = debugFailure;
}

vec4 STF_GetUniformRandom(STF_SamplerState s) {
	return s.m_u;
}

// STF_FILTER_TYPE_*
uint STF_GetFilterType(STF_SamplerState s) {
	return s.m_filterType;
}

uint STF_GetFrameIndex(STF_SamplerState s) {
	return s.m_frameIndex;
}

// STF_ANISO_LOD_METHOD_*
uint STF_GetAnisoMethod(STF_SamplerState s) {
	return s.m_anisoMethod;
}

// STF_MAGNIFICATION_METHOD_*
uint STF_GetMagMethod(STF_SamplerState s) {
	return s.m_magMethod;
}

// STF_MAGNIFICATION_FALLBACK_METHOD_*
uint STF_GetFallbackMethod(STF_SamplerState s) {
	return s.m_fallbackMethod;
}

// STF_ADDRESS_MODE_* - Only applied when using 'Load'
uvec3 STF_GetAddressingModes(STF_SamplerState s) {
	return s.m_addressingModes;
}

float STF_GetSigma(STF_SamplerState s) {
	return s.m_sigma;
}

// Each sample will update the random numbers
bool STF_GetReseedOnSample(STF_SamplerState s) {
	return s.m_reseedOnSample;
}

uvec4 STF_GetUserData(STF_SamplerState s) {
	return s.m_userData;
}

bool STF_GetDebugFailure(STF_SamplerState s) {
	return s.m_debugFallback;
}

// ============================================================================
// STFSamplerStateImpl.hlsli: the sample position
// ============================================================================

float _STF_ComputeTextureLod(STF_SamplerState s,
		uvec2 dim,
		vec4 textureGrads,
		float minLod,
		float maxLod,
		float mipBias) {
	float lod = 0.0;
	if (s.m_anisoMethod == STF_ANISO_LOD_METHOD_DEFAULT) {
		lod = STF_ComputeTextureLod_STF(dim, textureGrads, minLod, maxLod, mipBias);
	}
	return lod;
}

vec3 _STF_GetTexture2DSamplePos(inout STF_SamplerState s,
		uint mipValueType,
		uint width,
		uint height,
		uint numberOfLevels,
		vec2 uv,
		vec2 ddxUV,
		vec2 ddyUV,
		float mipValue) {
	vec4 u = s.m_u.xyzw;
	int maxLevel = int(numberOfLevels) - 1;

	// Stochastically compute the texture mip level
	uint lod = 0u;
	if (mipValueType == STF_MIP_VALUE_MODE_NONE) {
		const float lodf = _STF_ComputeTextureLod(s, uvec2(width, height), vec4(ddxUV, ddyUV), 0.0, float(maxLevel), 0.0);

		int ilod = STF_StochasticLinear(lodf, u.w);
		lod = uint(clamp(ilod, 0, maxLevel));
	} else if (mipValueType == STF_MIP_VALUE_MODE_MIP_LEVEL) {
		int ilod = STF_StochasticLinear(mipValue, u.w);
		lod = uint(clamp(ilod, 0, maxLevel));
	} else if (mipValueType == STF_MIP_VALUE_MODE_MIP_BIAS) {
		const float lodf = _STF_ComputeTextureLod(s, uvec2(width, height), vec4(ddxUV, ddyUV), 0.0, float(maxLevel), mipValue);
		int ilod = STF_StochasticLinear(lodf, u.w);
		lod = uint(clamp(ilod, 0, maxLevel));
	}

	// Query the W/H for the specified mip level.
	width = max(1u, width >> lod);
	height = max(1u, height >> lod);

	// Convert the uv coordinate to a texel position
	const vec2 st = uv * vec2(width, height) - 0.5;
	ivec2 idx = ivec2(0);

	if (s.m_filterType == STF_FILTER_TYPE_POINT) {
		idx = STF_NearestNeighbor2D(st);
	} else if (s.m_filterType == STF_FILTER_TYPE_LINEAR) {
		idx = STF_StochasticBilinear(st, u.xy);
	} else if (s.m_filterType == STF_FILTER_TYPE_CUBIC) {
		idx = STF_StochasticBicubic(st, u.x, u.y);
	} else if (s.m_filterType == STF_FILTER_TYPE_GAUSSIAN) {
		idx = STF_StochasticGaussian2D(s.m_sigma, st, u.xy);
	}

	vec2 idxUV = (vec2(idx) + 0.5) / vec2(width, height);

	if (s.m_reseedOnSample) {
		s.m_u = u;
	}

	return vec3(idxUV, float(lod));
}

// ============================================================================
// STFSamplerStateImpl.hlsli: the lookups
// ============================================================================

vec4 _STF_Texture2DSampleImpl(inout STF_SamplerState s,
		uint mipValueType,
		sampler2D tex,
		vec2 uv,
		vec2 ddxUV,
		vec2 ddyUV,
		float mipValue) {
	uvec2 size = uvec2(textureSize(tex, 0));
	uint numberOfLevels = uint(textureQueryLevels(tex));

	vec3 samplePos = _STF_GetTexture2DSamplePos(s, mipValueType, size.x, size.y, numberOfLevels, uv, ddxUV, ddyUV, mipValue);

	// We use SampleLevel with the supplied sampler to make sure
	// we capture the right tiling mode (Wrap, Clamp and Mirror modes)
	// (The magnification step that follows upstream for filters other than POINT is not ported: method NONE.)
	return textureLod(tex, samplePos.xy, samplePos.z);
}

vec4 _STF_Texture2DLoadImpl(inout STF_SamplerState s,
		uint mipValueType,
		sampler2D tex,
		vec2 uv,
		vec2 ddxUV,
		vec2 ddyUV,
		float mipValue) {
	uvec2 size = uvec2(textureSize(tex, 0));
	uint numberOfLevels = uint(textureQueryLevels(tex));

	// (The collaborative magnification paths upstream takes first, MIN_MAX* and MASK*, are not ported.)
	vec3 samplePos = _STF_GetTexture2DSamplePos(s, mipValueType, size.x, size.y, numberOfLevels, uv, ddxUV, ddyUV, mipValue);
	uint lod = uint(samplePos.z);
	// VAL: at least 1, as _STF_GetTexture2DSamplePos clamps it. Upstream shifts without the clamp here, which gives 0
	// on the short side of a non-square texture past its last level there, and the addressing would divide by it.
	uint width = max(1u, size.x >> lod);
	uint height = max(1u, size.y >> lod);

	bool isBorder = false;
	const vec2 pixelIndex = vec2(width, height) * STF_ApplyAddressingMode2D(samplePos.xy, uvec2(width, height), s.m_addressingModes.xy, isBorder);
	return texelFetch(tex, ivec2(uvec2(pixelIndex)), int(lod));
}

// Texture2D with a sampler, will use textureLod internally.
vec4 STF_Texture2DSampleGrad(inout STF_SamplerState s, sampler2D tex, vec2 uv, vec2 ddxUV, vec2 ddyUV) {
	return _STF_Texture2DSampleImpl(s, STF_MIP_VALUE_MODE_NONE, tex, uv, ddxUV, ddyUV, 0.0);
}

vec4 STF_Texture2DSampleLevel(inout STF_SamplerState s, sampler2D tex, vec2 uv, float mipLevel) {
	return _STF_Texture2DSampleImpl(s, STF_MIP_VALUE_MODE_MIP_LEVEL, tex, uv, vec2(0.0), vec2(0.0), mipLevel);
}

// Texture2D with texelFetch and the state's addressing modes.
vec4 STF_Texture2DLoadGrad(inout STF_SamplerState s, sampler2D tex, vec2 uv, vec2 ddxUV, vec2 ddyUV) {
	return _STF_Texture2DLoadImpl(s, STF_MIP_VALUE_MODE_NONE, tex, uv, ddxUV, ddyUV, 0.0);
}

vec4 STF_Texture2DLoadLevel(inout STF_SamplerState s, sampler2D tex, vec2 uv, float mipLevel) {
	return _STF_Texture2DLoadImpl(s, STF_MIP_VALUE_MODE_MIP_LEVEL, tex, uv, vec2(0.0), vec2(0.0), mipLevel);
}

// Texture2D without texture objects.
// These functions return vec3(x, y, lod) where (x, y) point at texel centers in UV space, lod is integer.
// Note: use floor(f) to convert the sample positions to integer texel coordinates, not round(f).
vec3 STF_Texture2DGetSamplePosGrad(inout STF_SamplerState s, uint width, uint height, uint numberOfLevels, vec2 uv, vec2 ddxUV, vec2 ddyUV) {
	return _STF_GetTexture2DSamplePos(s, STF_MIP_VALUE_MODE_NONE, width, height, numberOfLevels, uv, ddxUV, ddyUV, 0.0);
}

vec3 STF_Texture2DGetSamplePosLevel(inout STF_SamplerState s, uint width, uint height, uint numberOfLevels, vec2 uv, float mipLevel) {
	return _STF_GetTexture2DSamplePos(s, STF_MIP_VALUE_MODE_MIP_LEVEL, width, height, numberOfLevels, uv, vec2(0.0), vec2(0.0), mipLevel);
}

#endif // RTXTF_STF_INC_GLSL
