// VAL: the hair BCSDFs of NVIDIA's RTX Character Rendering SDK 1.4 (D126 H2, docs/r6-strand-hair.md), ported from HLSL
// to GLSL: the far-field model [d'Eon 2022, elliptic-hair TRT fit] the path tracer shades strands with by default, and
// Chiang et al. 2016 for the A/B (rendering/pathtracing/hair_bcsdf).
//
// Source: https://github.com/NVIDIA-RTX/RTXCR-Material-Library, commit eb184f9637f81a58a620d0e3cb4356acbc45cbe6
// (2026-09-03), shaders/include/rtxcr/HairMaterial.hlsli, HairFarFieldBCSDF.hlsli, HairChiangBSDF.hlsli and
// utils/RtxcrMath.hlsli, RtxcrBsdf.hlsli, RtxcrHairBsdfHelper.hlsli.
//
// The port changes no arithmetic: the same constants, fits and branches. HLSL to GLSL only: float3 is vec3, saturate a
// clamp to [0, 1], atan2(y, x) atan(y, x), fmod HLSL's (truncating, rtxcr_fmod), a scalar's .rrr a vec3, [unroll]
// dropped, uint flags compared with 0. Left out: what the path tracer does not call (the separate-lobe Chiang variant,
// the normalized melanin model, the Lambert, Henyey-Greenstein and disk helpers, and RTXCR_CreateHairInteractionSurface,
// whose tangent the SDK's own sample does not use: like the sample, a caller builds the surface itself). Two
// departures, marked VAL: RTXCR_ChiangBsdfPdf, the density RTXCR_SampleChiangBsdf computes for its own direction, as a
// function of any direction, for multiple importance sampling; and that density tilts the outgoing angle by the
// cuticle, as the sampler draws and the eval evaluates (and as pbrt's HairBSDF::Pdf does), where RTXCR tilted the
// incoming one. RTXCR's form is not the sampler's density: in a white furnace (hair_bcsdf --mode=furnace, 64 spp) fibres
// read 1.004 to 1.062 of the sky with it under BSDF sampling alone and 1.006 to 1.065 with the sky's next-event target,
// and with this one 0.978 to 1.000 and 1.000 to 1.004, where the model's own integrals predict 0.974 to 0.998 and 1
// (D126 H2).
//
// Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#ifndef RTXCR_HAIR_GLSL
#define RTXCR_HAIR_GLSL

// ----------------------------------------------------------------------------
// utils/RtxcrMath.hlsli
// ----------------------------------------------------------------------------

#define RTXCR_PI (3.141592653589)
#define RTXCR_TWO_PI (2.0 * RTXCR_PI)
#define RTXCR_ONE_OVER_PI (1.0 / RTXCR_PI)
#define RTXCR_ONE_OVER_TWO_PI (1.0 / RTXCR_TWO_PI)
#define RTXCR_PI_OVER_EIGHT (0.626657069) // sqrt(pi / 8.0f);

// HLSL's fmod: the remainder takes the dividend's sign (GLSL's mod takes the divisor's).
float rtxcr_fmod(float x, float y) {
	return x - y * trunc(x / y);
}

// Safe sqrt for x between [0, 1]
float RTXCR_Sqrt01(float x) {
	return max(sqrt(clamp(x, 0.0, 1.0)), 1e-7);
}

// Safe sqrt for x
float RTXCR_Sqrt0(float x) {
	return sqrt(max(x, 1e-7));
}

float RTXCR_Atan2safe(float x, float y) {
	return abs(x) + abs(y) < 1e-7 ? 0.0 : atan(x, y);
}

float RTXCR_I0(float x) {
	float val = 0.0;
	float x2i = 1.0;
	float ifact = 1.0;
	uint i4 = 1u;
	for (uint i = 0u; i < 10u; i++) {
		if (i > 1u) {
			ifact *= float(i);
		}
		val += x2i / (ifact * ifact * float(i4));
		x2i *= x * x;
		i4 *= 4u;
	}
	return val;
}

float RTXCR_LogI0(float x) {
	if (x > 12.0) {
		return x + 0.5 * (-log(RTXCR_TWO_PI) + log(1.0 / x) + 0.125 / x);
	} else {
		return log(RTXCR_I0(x));
	}
}

float RTXCR_PhiFunction(int p, float gammaI, float gammaT) {
	return 2.0 * float(p) * gammaT - 2.0 * gammaI + float(p) * RTXCR_PI;
}

float RTXCR_Logistic(float x, float s) {
	x = abs(x);
	float tmp = exp(-x / s);
	return tmp / (s * (1.0 + tmp) * (1.0 + tmp));
}

float RTXCR_LogisticCDF(float x, float s) {
	return 1.0 / (1.0 + exp(-x / s));
}

float RTXCR_TrimmedLogistic(float x, float s, float a, float b) {
	return RTXCR_Logistic(x, s) / (RTXCR_LogisticCDF(b, s) - RTXCR_LogisticCDF(a, s));
}

float RTXCR_SampleTrimmedLogistic(float u, float s, float a, float b) {
	float k = RTXCR_LogisticCDF(b, s) - RTXCR_LogisticCDF(a, s);
	float x = -s * log(1.0 / (u * k + RTXCR_LogisticCDF(a, s)) - 1.0);
	return clamp(x, a, b);
}

// 1D Gaussian distribution normalized over [-inf,inf]
float RTXCR_Gaussian1D(const float x, const float stddev) {
	return exp(-x * x / (2.0 * stddev * stddev)) / (stddev * sqrt(2.0 * RTXCR_PI));
}

float RTXCR_PhiR(const float h) {
	return -2.0 * asin(h);
}

float RTXCR_PhiTT(const float h, const float a) { // a = 1.0 / eta_prime
	return RTXCR_PI - 2.0 * asin(h) + 2.0 * asin(h * a);
}

// sample from normal distribution (Box-Muller transform)
float RTXCR_RandomGaussian1D(const float xi1, const float xi2) {
	return sqrt(2.0) * cos(2.0 * RTXCR_PI * xi1) * RTXCR_Sqrt0(-log(1.0 - xi2));
}

// Samples a direction within a hemisphere oriented along +Z axis with a cosine-weighted distribution
vec3 RTXCR_SampleHemisphere(vec2 u, out float pdf) {
	const float a = sqrt(u.x);
	const float b = RTXCR_TWO_PI * u.y;

	const vec3 result = vec3(
			a * cos(b),
			a * sin(b),
			sqrt(1.0 - u.x));

	pdf = result.z * RTXCR_ONE_OVER_PI;

	return result;
}

// ----------------------------------------------------------------------------
// utils/RtxcrBsdf.hlsli
// ----------------------------------------------------------------------------

// Calculates the monochromatic base reflectivity from a given incident and transmitted IoR
float RTXCR_CalculateBaseReflectivity(const float incidentIoR, const float transmittedIoR) {
	const float tmp = (incidentIoR - transmittedIoR) / (incidentIoR + transmittedIoR);
	return tmp * tmp;
}

// Analytical fresnel equation for dielectric material.
float RTXCR_DielectricFresnel(
		const float eta, // refracted / reflected ior
		const float cosThetaI) { // cosine between of angle normal/half-vector and incident direction
	// cosThetaT^2 = 1 - sinThetaT^2 = 1 - (sinThetaI / eta)^2 = 1 - (1 - cosThetaI^2) / eta^2
	const float cosThetaT2 = 1.0 - (1.0 - cosThetaI * cosThetaI) / (eta * eta);
	// Handle TIR (Total internal reflection)
	if (cosThetaT2 < 0.0) {
		return 1.0;
	}

	const float cosThetaT = sqrt(cosThetaT2); // refracted angle cosine

	const float n1t1 = cosThetaI;
	const float n1t2 = cosThetaT;
	const float n2t1 = cosThetaI * eta;
	const float n2t2 = cosThetaT * eta;
	const float rs = (n1t2 - n2t1) / (n1t2 + n2t1);
	const float rp = (n1t1 - n2t2) / (n1t1 + n2t2);
	const float fres = 0.5 * (rs * rs + rp * rp);

	return clamp(fres, 0.0, 1.0);
}

// Schlick's approximation to Fresnel term
float RTXCR_EvalFresnelSchlick(float f0, float NdotS) {
	return f0 + (1.0 - f0) * pow(1.0 - NdotS, 5.0);
}

// Linear RGB -> perceptual luminance (Rec.709).
float RTXCR_Luminance(const vec3 rgb) {
	return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

// ----------------------------------------------------------------------------
// HairMaterial.hlsli
// ----------------------------------------------------------------------------

#define RTXCR_HairLobeType_R (0u)
#define RTXCR_HairLobeType_TT (1u)
#define RTXCR_HairLobeType_TRT (2u)
#define RTXCR_Hair_Max_Scattering_Events (3)

#define RTXCR_HairAbsorptionModel_Color (0u)
#define RTXCR_HairAbsorptionModel_Physics (1u)

struct RTXCR_HairInteractionSurface {
	vec3 incidentRayDirection;
	vec3 shadingNormal;
	vec3 tangent;
};

struct RTXCR_HairMaterialData {
	vec3 baseColor;
	float longitudinalRoughness; // beta_m

	float azimuthalRoughness; // beta_n
	float ior;
	float eta;
	uint fresnelApproximation;

	uint absorptionModel;
	float melanin;
	float melaninRedness;
	float cuticleAngleInDegrees; // alpha
};

struct RTXCR_HairMaterialInteraction {
	float h;
	float gammaI;
	vec3 absorptionCoefficient;

	float ior;
	float eta;
	uint fresnelApproximation;

	float logisticDistributionScalar; // s

	float v[RTXCR_Hair_Max_Scattering_Events + 1];

	float sin2kAlpha[RTXCR_Hair_Max_Scattering_Events];
	float cos2kAlpha[RTXCR_Hair_Max_Scattering_Events];
};

// Compute Longitudinal Roughness Variance
void RTXCR_ComputeRoughnessVariance(const float betaM, inout RTXCR_HairMaterialInteraction hairMaterialInteraction) {
	float tmp = 0.726 * betaM + 0.812 * betaM * betaM + 3.7 * pow(betaM, 20.0);
	hairMaterialInteraction.v[0] = max(tmp * tmp, 1e-7);
	hairMaterialInteraction.v[1] = 0.25 * hairMaterialInteraction.v[0];
	hairMaterialInteraction.v[2] = 4.0 * hairMaterialInteraction.v[0];
	for (int p = 3; p <= RTXCR_Hair_Max_Scattering_Events; ++p) {
		hairMaterialInteraction.v[p] = hairMaterialInteraction.v[2];
	}
}

// Compute azimuthally offset h
float RTXCR_CalculateAzimuthallyDistance(const RTXCR_HairInteractionSurface hairInteractionSurface) {
	// Project wi to the (B, N) plane
	vec3 wiProj = normalize(hairInteractionSurface.incidentRayDirection -
			dot(hairInteractionSurface.incidentRayDirection, hairInteractionSurface.tangent) * hairInteractionSurface.tangent);
	// Calculate the vector that perpendicular with projected wi on (B, N) plane
	vec3 wiProjPerpendicular = cross(wiProj, hairInteractionSurface.tangent);
	// h = sin(Gamma) = cos(pi/2 - Gamma) = dot(N, Wi_Proj_Prependicular)
	return dot(hairInteractionSurface.shadingNormal, wiProjPerpendicular);
}

// Mapping from color to absorption coefficient.
vec3 RTXCR_AbsorptionCoefficientFromColor(const vec3 color, const float betaN) {
	const float tmp = 5.969 - 0.215 * betaN + 2.532 * betaN * betaN - 10.73 * pow(betaN, 3.0) + 5.574 * pow(betaN, 4.0) + 0.245 * pow(betaN, 5.0);
	const vec3 sqrtAbsorptionCoefficient = log(max(color, vec3(1e-4))) / tmp;
	return sqrtAbsorptionCoefficient * sqrtAbsorptionCoefficient;
}

// Mapping from hair melanin to absorption coefficient
vec3 RTXCR_ComputeAbsorptionFromMelanin(float eumelanin, float pheomelanin) {
	return max(eumelanin * vec3(0.506, 0.841, 1.653) + pheomelanin * vec3(0.343, 0.733, 1.924), vec3(0.0, 0.0, 0.0));
}

vec3 RTXCR_AbsorptionCoefficientFromMelanin(const float melanin_concentration, const float melanin_redness) {
	float melanin_concentration_value = melanin_concentration;
	float melanin_gamma = 2.4;
	float melanin = melanin_concentration_value * melanin_concentration_value * melanin_gamma;
	float eumelanin = melanin * (1.0 - melanin_redness);
	float pheomelanin = melanin * melanin_redness;
	return RTXCR_ComputeAbsorptionFromMelanin(eumelanin, pheomelanin);
}

vec3 RTXCR_ComputeAbsorptionCoefficient(const RTXCR_HairMaterialData hairMaterialData) {
	switch (hairMaterialData.absorptionModel) {
		case RTXCR_HairAbsorptionModel_Color:
			return RTXCR_AbsorptionCoefficientFromColor(hairMaterialData.baseColor, hairMaterialData.azimuthalRoughness);
		case RTXCR_HairAbsorptionModel_Physics:
			return RTXCR_AbsorptionCoefficientFromMelanin(hairMaterialData.melanin, hairMaterialData.melaninRedness);
	}
	return vec3(0.0, 0.0, 0.0);
}

// Compute azimuthal logistic scale factor
float RTXCR_ComputelogisticDistributionScalar(const float betaN) {
	return max(RTXCR_PI_OVER_EIGHT * (0.265 * betaN + 1.194 * betaN * betaN + 5.372 * pow(betaN, 22.0)), 1e-7);
}

// Compute the scales that caused by the angle between hair cuticle and hair surface
void RTXCR_ComputeHairCuticleScales(const float cuticleAngleInDegrees, inout RTXCR_HairMaterialInteraction hairMaterialInteraction) {
	hairMaterialInteraction.sin2kAlpha[0] = sin(cuticleAngleInDegrees / 180.0 * RTXCR_PI);
	hairMaterialInteraction.cos2kAlpha[0] = sqrt(clamp(1.0 - hairMaterialInteraction.sin2kAlpha[0] * hairMaterialInteraction.sin2kAlpha[0], 0.0, 1.0));
	for (int i = 1; i < 3; i++) {
		// sin(2*Theta) = 2 * sin(Theta) * cos(Theta)
		hairMaterialInteraction.sin2kAlpha[i] =
				2.0 * hairMaterialInteraction.cos2kAlpha[i - 1] * hairMaterialInteraction.sin2kAlpha[i - 1];
		// cos(2*Theta) = (cos(Theta))^2 - (sin(Theta))^2
		hairMaterialInteraction.cos2kAlpha[i] =
				hairMaterialInteraction.cos2kAlpha[i - 1] * hairMaterialInteraction.cos2kAlpha[i - 1] -
				hairMaterialInteraction.sin2kAlpha[i - 1] * hairMaterialInteraction.sin2kAlpha[i - 1];
	}
}

RTXCR_HairMaterialInteraction RTXCR_CreateHairMaterialInteraction(
		const RTXCR_HairMaterialData hairMaterialData,
		const RTXCR_HairInteractionSurface hairInteractionSurface) {
	RTXCR_HairMaterialInteraction hairMaterialInteraction;
	hairMaterialInteraction.h = RTXCR_CalculateAzimuthallyDistance(hairInteractionSurface);
	hairMaterialInteraction.gammaI = asin(clamp(hairMaterialInteraction.h, -1.0, 1.0));
	hairMaterialInteraction.absorptionCoefficient = RTXCR_ComputeAbsorptionCoefficient(hairMaterialData);
	hairMaterialInteraction.fresnelApproximation = hairMaterialData.fresnelApproximation;
	hairMaterialInteraction.ior = hairMaterialData.ior;
	hairMaterialInteraction.eta = hairMaterialData.eta;
	hairMaterialInteraction.logisticDistributionScalar = RTXCR_ComputelogisticDistributionScalar(hairMaterialData.azimuthalRoughness);
	// Compute hairMaterialInteraction.v
	RTXCR_ComputeRoughnessVariance(hairMaterialData.longitudinalRoughness, hairMaterialInteraction);
	// Compute Hair Scales
	RTXCR_ComputeHairCuticleScales(hairMaterialData.cuticleAngleInDegrees, hairMaterialInteraction);
	return hairMaterialInteraction;
}

struct RTXCR_HairMaterialInteractionBcsdf {
	vec3 diffuseReflectionTint;
	float diffuseReflectionWeight;

	float roughness;
	vec3 absorptionCoefficient;

	float ior;
	float cuticleAngle;
};

RTXCR_HairMaterialInteractionBcsdf RTXCR_CreateHairMaterialInteractionBcsdf(
		const RTXCR_HairMaterialData hairMaterialData,
		const vec3 diffuseRefelctionTint,
		const float diffuseReflectionWeight,
		const float roughness) {
	RTXCR_HairMaterialInteractionBcsdf hairMaterialInteractionBcsdf;
	hairMaterialInteractionBcsdf.diffuseReflectionTint = diffuseRefelctionTint;
	hairMaterialInteractionBcsdf.diffuseReflectionWeight = diffuseReflectionWeight;
	hairMaterialInteractionBcsdf.roughness = roughness;
	hairMaterialInteractionBcsdf.absorptionCoefficient = RTXCR_ComputeAbsorptionCoefficient(hairMaterialData);
	hairMaterialInteractionBcsdf.ior = hairMaterialData.ior;
	hairMaterialInteractionBcsdf.cuticleAngle = radians(hairMaterialData.cuticleAngleInDegrees);
	return hairMaterialInteractionBcsdf;
}

// ----------------------------------------------------------------------------
// utils/RtxcrHairBsdfHelper.hlsli
// ----------------------------------------------------------------------------

// Longitudinal scattering function Mp.
float RTXCR_MP(const float cosThetaI, const float cosThetaO, const float sinThetaI, const float sinThetaO, const float v) {
	const float a = cosThetaI * cosThetaO / v;
	const float b = sinThetaI * sinThetaO / v;
	const float mp = (v <= 0.1) ? exp(RTXCR_LogI0(a) - b - 1.0 / v + 0.6931 + log(0.5 / v)) : (exp(-b) * RTXCR_I0(a)) / (sinh(1.0 / v) * 2.0 * v);
	return mp;
}

// Attenuation function Ap.
void RTXCR_AP(const RTXCR_HairMaterialInteraction hairMaterialInteraction, const float cosThetaI, const vec3 T, out vec3 ap[RTXCR_Hair_Max_Scattering_Events + 1]) {
	const float cosGammaI = RTXCR_Sqrt01(1.0 - hairMaterialInteraction.h * hairMaterialInteraction.h);
	const float cosTheta = cosThetaI * cosGammaI;
	const vec3 f = hairMaterialInteraction.fresnelApproximation != 0u ? vec3(RTXCR_EvalFresnelSchlick(RTXCR_CalculateBaseReflectivity(1.0, hairMaterialInteraction.ior), cosTheta)) : vec3(RTXCR_DielectricFresnel(hairMaterialInteraction.ior, cosTheta));

	ap[0] = f;
	ap[1] = T * (vec3(1.0, 1.0, 1.0) - f) * (vec3(1.0, 1.0, 1.0) - f);
	for (int p = 2; p < RTXCR_Hair_Max_Scattering_Events; ++p) {
		ap[p] = ap[p - 1] * T * f;
	}

	// Compute attenuation term accounting for remaining orders of scattering.
	ap[RTXCR_Hair_Max_Scattering_Events] = ap[RTXCR_Hair_Max_Scattering_Events - 1] * T * f / (vec3(1.0, 1.0, 1.0) - T * f);
}

// Azimuthal scattering function Np.
float RTXCR_NP(const float phi, const int p, const float s, const float gammaI, const float gammaT) {
	float dphi = phi - RTXCR_PhiFunction(p, gammaI, gammaT);

	// Remap dphi to [-pi, pi].
	dphi = rtxcr_fmod(dphi, RTXCR_TWO_PI);
	if (dphi > RTXCR_PI) {
		dphi -= RTXCR_TWO_PI;
	}
	if (dphi < -RTXCR_PI) {
		dphi += RTXCR_TWO_PI;
	}

	return RTXCR_TrimmedLogistic(dphi, s, -RTXCR_PI, RTXCR_PI);
}

// Compute a discrete pdf for sampling Ap (Lobe selection)
void RTXCR_ComputeApPdf(const RTXCR_HairMaterialInteraction hairMaterialInteraction, const float cosThetaO, out float apPdf[RTXCR_Hair_Max_Scattering_Events + 1]) {
	const float sinThetaO = RTXCR_Sqrt01(1.0 - cosThetaO * cosThetaO);

	// Compute refracted ray.
	const float sinThetaT = sinThetaO / hairMaterialInteraction.ior;
	const float cosThetaT = RTXCR_Sqrt01(1.0 - sinThetaT * sinThetaT);

	const float etap = RTXCR_Sqrt0(hairMaterialInteraction.ior * hairMaterialInteraction.ior - sinThetaO * sinThetaO) / cosThetaO;
	const float sinGammaT = hairMaterialInteraction.h / etap;
	const float cosGammaT = RTXCR_Sqrt01(1.0 - sinGammaT * sinGammaT);

	// Compute the transmittance T of a single path through the cylinder.
	const float tmp = -2.0 * cosGammaT / cosThetaT;
	const vec3 T = exp(hairMaterialInteraction.absorptionCoefficient * tmp);

	vec3 ap[RTXCR_Hair_Max_Scattering_Events + 1];
	RTXCR_AP(hairMaterialInteraction, cosThetaO, T, ap);

	// Compute apPdf from individal ap terms.
	float sumY = 0.0;
	for (int p = 0; p < RTXCR_Hair_Max_Scattering_Events; ++p) {
		apPdf[p] = RTXCR_Luminance(ap[p]);
		sumY += apPdf[p];
	}

	const float invSumY = 1.0 / sumY;
	for (int p2 = 0; p2 < RTXCR_Hair_Max_Scattering_Events; ++p2) {
		apPdf[p2] *= invSumY;
	}

	apPdf[RTXCR_Hair_Max_Scattering_Events] = 0.0;
}

// ----------------------------------------------------------------------------
// HairChiangBSDF.hlsli: Chiang16 Hair model, https://benedikt-bitterli.me/pchfm/, https://www.pbrt.org/hair.pdf
// wi and wo in the hair's local frame: x along the tangent, z the shading normal.
// ----------------------------------------------------------------------------

vec3 RTXCR_HairChiangBsdfEval(const RTXCR_HairMaterialInteraction hairMaterialInteraction,
		const vec3 wi,
		const vec3 wo) {
	const float sinThetaO = wo.x;
	const float cosThetaO = RTXCR_Sqrt01(1.0 - sinThetaO * sinThetaO);
	const float phiO = RTXCR_Atan2safe(wo.z, wo.y);

	const float sinThetaI = wi.x;
	const float cosThetaI = RTXCR_Sqrt01(1.0 - sinThetaI * sinThetaI);
	const float phiI = RTXCR_Atan2safe(wi.z, wi.y);

	// Compute refracted ray.
	const float sinThetaT = sinThetaO / hairMaterialInteraction.ior;
	const float cosThetaT = RTXCR_Sqrt01(1.0 - sinThetaT * sinThetaT);

	const float etap = RTXCR_Sqrt0(hairMaterialInteraction.ior * hairMaterialInteraction.ior - sinThetaO * sinThetaO) / cosThetaO;
	const float sinGammaT = hairMaterialInteraction.h / etap;
	const float cosGammaT = RTXCR_Sqrt01(1.0 - sinGammaT * sinGammaT);
	const float gammaT = asin(clamp(sinGammaT, -1.0, 1.0));

	// Compute the transmittance T of a single path through the cylinder
	const float tmp = -2.0 * cosGammaT / cosThetaT;
	const vec3 T = exp(hairMaterialInteraction.absorptionCoefficient * tmp);

	// Evaluate hair BCSDF for each lobe
	const float phi = phiI - phiO;
	vec3 ap[RTXCR_Hair_Max_Scattering_Events + 1];
	RTXCR_AP(hairMaterialInteraction, cosThetaO, T, ap);
	vec3 result = vec3(0.0);

	for (int p = 0; p < RTXCR_Hair_Max_Scattering_Events; ++p) {
		float sinThetaOp, cosThetaOp;
		if (p == 0) {
			sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[1] - cosThetaO * hairMaterialInteraction.sin2kAlpha[1];
			cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[1] + sinThetaO * hairMaterialInteraction.sin2kAlpha[1];
		} else if (p == 1) {
			sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[0] + cosThetaO * hairMaterialInteraction.sin2kAlpha[0];
			cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[0] - sinThetaO * hairMaterialInteraction.sin2kAlpha[0];
		} else if (p == 2) {
			sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[2] + cosThetaO * hairMaterialInteraction.sin2kAlpha[2];
			cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[2] - sinThetaO * hairMaterialInteraction.sin2kAlpha[2];
		} else {
			sinThetaOp = sinThetaO;
			cosThetaOp = cosThetaO;
		}

		cosThetaOp = abs(cosThetaOp);
		result += RTXCR_MP(cosThetaOp, cosThetaI, sinThetaOp, sinThetaI, hairMaterialInteraction.v[p]) *
				ap[p] *
				RTXCR_NP(phi, p, hairMaterialInteraction.logisticDistributionScalar, hairMaterialInteraction.gammaI, gammaT);
	}

	// Compute contribution of remaining terms after RTXCR_Hair_Max_Scattering_Events
	result += RTXCR_MP(cosThetaO, cosThetaI, sinThetaO, sinThetaI, hairMaterialInteraction.v[RTXCR_Hair_Max_Scattering_Events]) *
			ap[RTXCR_Hair_Max_Scattering_Events] *
			RTXCR_ONE_OVER_TWO_PI;

	// We omit this computation in BSDF, because the cosThetaI_N will be cancelled out when evaluate scattered radiance anyway
	// const float cosThetaI_N = wi.z; // The angle between wi and normal, which is (0, 0, 1) on local space
	// result = abs(cosThetaI_N) > 0.0f ? result / abs(cosThetaI_N) : 0.0f;

	return max(result, vec3(0.0));
}

bool RTXCR_SampleChiangBsdf(const RTXCR_HairMaterialInteraction hairMaterialInteraction,
		const vec3 wo,
		vec2 u[2],
		out vec3 wi,
		out float pdf,
		out vec3 weight,
		out uint lobeType) {
	const float sinThetaO = wo.x;
	const float cosThetaO = RTXCR_Sqrt01(1.0 - sinThetaO * sinThetaO);
	const float phiO = RTXCR_Atan2safe(wo.z, wo.y);

	// Determine which term p to sample for hair scattering.
	float apPdf[RTXCR_Hair_Max_Scattering_Events + 1];
	RTXCR_ComputeApPdf(hairMaterialInteraction, cosThetaO, apPdf);

	int p = 0;
	float vp = hairMaterialInteraction.v[0];
	{
		for (int i = 0; i < RTXCR_Hair_Max_Scattering_Events; ++i) {
			if (u[0].x >= apPdf[i]) {
				u[0].x -= apPdf[i];
				p = i + 1;
				vp = hairMaterialInteraction.v[i + 1];
			} else {
				break;
			}
		}
	}

	float sinThetaOp = sinThetaO;
	float cosThetaOp = cosThetaO;
	if (p == 0) {
		sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[1] - cosThetaO * hairMaterialInteraction.sin2kAlpha[1];
		cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[1] + sinThetaO * hairMaterialInteraction.sin2kAlpha[1];

		lobeType = RTXCR_HairLobeType_R;
	} else if (p == 1) {
		sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[0] + cosThetaO * hairMaterialInteraction.sin2kAlpha[0];
		cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[0] - sinThetaO * hairMaterialInteraction.sin2kAlpha[0];

		lobeType = RTXCR_HairLobeType_TT;
	} else if (p == 2) {
		sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[2] + cosThetaO * hairMaterialInteraction.sin2kAlpha[2];
		cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[2] - sinThetaO * hairMaterialInteraction.sin2kAlpha[2];

		lobeType = RTXCR_HairLobeType_TRT;
	} else {
		lobeType = RTXCR_HairLobeType_TRT;
	}

	// Sample Mp to compute thetaO
	u[1].x = max(u[1].x, 1e-5);
	const float cosTheta = 1.0 + vp * log(u[1].x + (1.0 - u[1].x) * exp(-2.0 / vp));
	const float sinTheta = RTXCR_Sqrt01(1.0 - cosTheta * cosTheta);
	const float cosPhi = cos(u[1].y * RTXCR_TWO_PI);
	const float sinThetaI = -cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp;
	const float cosThetaI = RTXCR_Sqrt01(1.0 - sinThetaI * sinThetaI);

	// Sample Np to compute dphi
	const float etap = RTXCR_Sqrt0(hairMaterialInteraction.ior * hairMaterialInteraction.ior - sinThetaO * sinThetaO) / cosThetaO;
	const float sinGammaT = hairMaterialInteraction.h / etap;
	const float gammaT = asin(clamp(sinGammaT, -1.0, 1.0));
	float dphi;
	if (p < RTXCR_Hair_Max_Scattering_Events) {
		dphi = RTXCR_PhiFunction(p, hairMaterialInteraction.gammaI, gammaT) +
				RTXCR_SampleTrimmedLogistic(u[0].y, hairMaterialInteraction.logisticDistributionScalar, -RTXCR_PI, RTXCR_PI);
	} else {
		dphi = u[0].y * RTXCR_TWO_PI;
	}

	const float phiI = phiO + dphi;
	wi = vec3(sinThetaI, cosThetaI * cos(phiI), cosThetaI * sin(phiI));

	pdf = 0.0;
	for (int i = 0; i < RTXCR_Hair_Max_Scattering_Events; ++i) {
		float sinThetaOq, cosThetaOq; // VAL: the outgoing angle tilted, as the sampler and the eval tilt it (pbrt).
		if (i == 0) {
			sinThetaOq = sinThetaO * hairMaterialInteraction.cos2kAlpha[1] - cosThetaO * hairMaterialInteraction.sin2kAlpha[1];
			cosThetaOq = cosThetaO * hairMaterialInteraction.cos2kAlpha[1] + sinThetaO * hairMaterialInteraction.sin2kAlpha[1];
		} else if (i == 1) {
			sinThetaOq = sinThetaO * hairMaterialInteraction.cos2kAlpha[0] + cosThetaO * hairMaterialInteraction.sin2kAlpha[0];
			cosThetaOq = cosThetaO * hairMaterialInteraction.cos2kAlpha[0] - sinThetaO * hairMaterialInteraction.sin2kAlpha[0];
		} else if (i == 2) {
			sinThetaOq = sinThetaO * hairMaterialInteraction.cos2kAlpha[2] + cosThetaO * hairMaterialInteraction.sin2kAlpha[2];
			cosThetaOq = cosThetaO * hairMaterialInteraction.cos2kAlpha[2] - sinThetaO * hairMaterialInteraction.sin2kAlpha[2];
		} else {
			sinThetaOq = sinThetaO;
			cosThetaOq = cosThetaO;
		}

		cosThetaOq = abs(cosThetaOq);
		pdf += RTXCR_MP(cosThetaI, cosThetaOq, sinThetaI, sinThetaOq, hairMaterialInteraction.v[i]) *
				apPdf[i] *
				RTXCR_NP(dphi, i, hairMaterialInteraction.logisticDistributionScalar, hairMaterialInteraction.gammaI, gammaT);
	}
	pdf += RTXCR_MP(cosThetaI, cosThetaO, sinThetaI, sinThetaO, hairMaterialInteraction.v[RTXCR_Hair_Max_Scattering_Events]) *
			apPdf[RTXCR_Hair_Max_Scattering_Events] *
			RTXCR_ONE_OVER_TWO_PI;

	if (pdf > 1e-3) {
		weight = RTXCR_HairChiangBsdfEval(hairMaterialInteraction, wi, wo);
	} else {
		pdf = 0.0;
		weight = vec3(0.0, 0.0, 0.0);
	}

	return (pdf > 0.0);
}

// VAL: not in RTXCR: the density RTXCR_SampleChiangBsdf draws wi with, for multiple importance sampling against light
// samples (D126 H2). Its sampler's lines, with dphi the azimuth from wo to wi, and its cutoff: a direction whose density
// is at most 1e-3 the sampler rejects, so it has none.
float RTXCR_ChiangBsdfPdf(const RTXCR_HairMaterialInteraction hairMaterialInteraction,
		const vec3 wi,
		const vec3 wo) {
	const float sinThetaO = wo.x;
	const float cosThetaO = RTXCR_Sqrt01(1.0 - sinThetaO * sinThetaO);
	const float phiO = RTXCR_Atan2safe(wo.z, wo.y);

	const float sinThetaI = wi.x;
	const float cosThetaI = RTXCR_Sqrt01(1.0 - sinThetaI * sinThetaI);
	const float phiI = RTXCR_Atan2safe(wi.z, wi.y);

	float apPdf[RTXCR_Hair_Max_Scattering_Events + 1];
	RTXCR_ComputeApPdf(hairMaterialInteraction, cosThetaO, apPdf);

	const float etap = RTXCR_Sqrt0(hairMaterialInteraction.ior * hairMaterialInteraction.ior - sinThetaO * sinThetaO) / cosThetaO;
	const float sinGammaT = hairMaterialInteraction.h / etap;
	const float gammaT = asin(clamp(sinGammaT, -1.0, 1.0));
	const float dphi = phiI - phiO;

	float pdf = 0.0;
	for (int i = 0; i < RTXCR_Hair_Max_Scattering_Events; ++i) {
		float sinThetaOq, cosThetaOq; // VAL: the outgoing angle tilted, as the sampler and the eval tilt it (pbrt).
		if (i == 0) {
			sinThetaOq = sinThetaO * hairMaterialInteraction.cos2kAlpha[1] - cosThetaO * hairMaterialInteraction.sin2kAlpha[1];
			cosThetaOq = cosThetaO * hairMaterialInteraction.cos2kAlpha[1] + sinThetaO * hairMaterialInteraction.sin2kAlpha[1];
		} else if (i == 1) {
			sinThetaOq = sinThetaO * hairMaterialInteraction.cos2kAlpha[0] + cosThetaO * hairMaterialInteraction.sin2kAlpha[0];
			cosThetaOq = cosThetaO * hairMaterialInteraction.cos2kAlpha[0] - sinThetaO * hairMaterialInteraction.sin2kAlpha[0];
		} else if (i == 2) {
			sinThetaOq = sinThetaO * hairMaterialInteraction.cos2kAlpha[2] + cosThetaO * hairMaterialInteraction.sin2kAlpha[2];
			cosThetaOq = cosThetaO * hairMaterialInteraction.cos2kAlpha[2] - sinThetaO * hairMaterialInteraction.sin2kAlpha[2];
		} else {
			sinThetaOq = sinThetaO;
			cosThetaOq = cosThetaO;
		}

		cosThetaOq = abs(cosThetaOq);
		pdf += RTXCR_MP(cosThetaI, cosThetaOq, sinThetaI, sinThetaOq, hairMaterialInteraction.v[i]) *
				apPdf[i] *
				RTXCR_NP(dphi, i, hairMaterialInteraction.logisticDistributionScalar, hairMaterialInteraction.gammaI, gammaT);
	}
	pdf += RTXCR_MP(cosThetaI, cosThetaO, sinThetaI, sinThetaO, hairMaterialInteraction.v[RTXCR_Hair_Max_Scattering_Events]) *
			apPdf[RTXCR_Hair_Max_Scattering_Events] *
			RTXCR_ONE_OVER_TWO_PI;

	return pdf > 1e-3 ? pdf : 0.0;
}

// ----------------------------------------------------------------------------
// HairFarFieldBCSDF.hlsli. wi, wo, the tangent and the normal in any one orthonormal frame (world space here).
// ----------------------------------------------------------------------------

struct RTXCR_FarFieldTrtFit {
	float peak;
	float weight1;
	float weight2;
	float width1;
	float width2;
};

float RTXCR_FarFieldWrapAngle(const float angle) {
	float wrapped = rtxcr_fmod(angle + RTXCR_PI, RTXCR_TWO_PI);
	if (wrapped < 0.0) {
		wrapped += RTXCR_TWO_PI;
	}
	return wrapped - RTXCR_PI;
}

// A normal distribution wrapped onto [-pi, pi).  Spatial aliases converge
// quickly for narrow lobes; the Fourier form is stable for broad lobes.
float RTXCR_FarFieldWrappedGaussian(const float angle, const float stddev) {
	const float sigma = max(stddev, 1e-4);
	const float delta = RTXCR_FarFieldWrapAngle(angle);

	if (sigma < 1.0) {
		return RTXCR_Gaussian1D(delta - RTXCR_TWO_PI, sigma) +
				RTXCR_Gaussian1D(delta, sigma) +
				RTXCR_Gaussian1D(delta + RTXCR_TWO_PI, sigma);
	}

	float result = 1.0;
	for (uint harmonic = 1u; harmonic <= 6u; ++harmonic) {
		const float n = float(harmonic);
		result += 2.0 * exp(-0.5 * sigma * sigma * n * n) * cos(n * delta);
	}
	return max(result * RTXCR_ONE_OVER_TWO_PI, 0.0);
}

void RTXCR_FarFieldComputeTrtFit(const float thetaI,
		const float roughness,
		out RTXCR_FarFieldTrtFit fit) {
	// TRT lobe: custom 3-Gaussian lobe based on fitting to MC simulation
	// The elliptic-hair fit is parameterized by |theta_i|, not theta_d.
	const float ti = abs(thetaI);
	float variance1;
	float variance2;

	if (ti < 0.525) {
		fit.peak = cos(ti) - 0.733;
		fit.weight1 = (-0.000111282) * (-0.103125 + ti) * ti + pow(ti, 15.7265) + 0.00023939;
		fit.weight2 = 0.000322755 * ((ti - pow(1.80972 * ti, 16.7669)) * tan(ti) + 0.991977);
		variance1 = 0.00597578 + 0.000428897 * cos(5.41149 * ti);
		variance2 = 0.0181 * tan(cos(2.121 * ti));
	} else if (ti < 1.1) {
		fit.peak = max(0.0, 0.00493 + 0.579 * ti - 0.775 * ti * ti);
		fit.weight1 = 0.00108 - 0.0014 * ti + 0.0003937 * ti * ti;
		fit.weight2 = -0.00119 + 0.00219 * ti;
		variance1 = 0.0391 - 0.0888 * ti + 0.0581 * ti * ti;
		variance2 = 0.384 - 1.14 * ti + 0.942 * ti * ti;
	} else {
		fit.peak = 0.0;
		fit.weight1 = 0.0;
		fit.weight2 = 0.000239 + 0.00139 * ti * ti * ti - 0.00053124 * ti * ti * ti * ti * ti;
		variance1 = 1.0;
		variance2 = -1.86 + 2.73 * ti - 0.7437 * ti * ti;
	}

	fit.weight1 = max(fit.weight1, 0.0);
	fit.weight2 = max(fit.weight2, 0.0);
	const float widthScale = max(roughness, 1e-4) / 0.06;
	fit.width1 = max(widthScale * sqrt(max(variance1, 0.0)), 1e-4);
	fit.width2 = max(widthScale * sqrt(max(variance2, 0.0)), 1e-4);
}

float RTXCR_FarFieldTrtAzimuth(const float phi,
		const RTXCR_FarFieldTrtFit fit) {
	return fit.weight1 * RTXCR_FarFieldWrappedGaussian(phi - fit.peak, fit.width1) +
			fit.weight1 * RTXCR_FarFieldWrappedGaussian(phi + fit.peak, fit.width1) +
			fit.weight2 * RTXCR_FarFieldWrappedGaussian(phi, fit.width2);
}

float RTXCR_FarFieldVariance(const float beta) {
	// Very large v is already indistinguishable from a uniform elevation
	// distribution, and clamping avoids loss of precision in sinh(1 / v).
	return clamp(beta * beta, 1e-6, 100.0);
}

float RTXCR_FarFieldLongitudinal(const float thetaOp,
		const float sinThetaI,
		const float cosThetaI,
		const float beta) {
	const float sinThetaOp = sin(thetaOp);
	const float cosThetaOp = abs(cos(thetaOp));
	return RTXCR_MP(cosThetaOp, cosThetaI, sinThetaOp, sinThetaI, RTXCR_FarFieldVariance(beta));
}

void RTXCR_FarFieldSampleLongitudinal(const float thetaOp,
		const float beta,
		const vec2 random,
		out float sinThetaI,
		out float cosThetaI) {
	const float v = RTXCR_FarFieldVariance(beta);
	const float u = max(random.x, 1e-5);
	const float cosTheta = clamp(1.0 + v * log(u + (1.0 - u) * exp(-2.0 / v)), -1.0, 1.0);
	const float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
	const float sinThetaOp = sin(thetaOp);
	const float cosThetaOp = abs(cos(thetaOp));

	sinThetaI = clamp(-cosTheta * sinThetaOp + sinTheta * cos(RTXCR_TWO_PI * random.y) * cosThetaOp, -1.0, 1.0);
	cosThetaI = sqrt(max(1.0 - sinThetaI * sinThetaI, 0.0));
}

void RTXCR_FarFieldCoordinateFrame(const RTXCR_HairInteractionSurface hairInteractionSurface,
		const vec3 wo,
		out vec3 tangent,
		out vec3 azimuthOrigin,
		out vec3 azimuthBitangent,
		out float sinThetaO,
		out float cosThetaO) {
	tangent = normalize(hairInteractionSurface.tangent); // tangent of hair
	sinThetaO = clamp(dot(wo, tangent), -1.0, 1.0);
	cosThetaO = sqrt(max(1.0 - sinThetaO * sinThetaO, 0.0));

	vec3 projectedWo = wo - sinThetaO * tangent;
	if (dot(projectedWo, projectedWo) < 1e-10) {
		projectedWo = hairInteractionSurface.shadingNormal -
				dot(hairInteractionSurface.shadingNormal, tangent) * tangent;
		if (dot(projectedWo, projectedWo) < 1e-10) {
			const vec3 fallbackAxis = abs(tangent.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
			projectedWo = fallbackAxis - dot(fallbackAxis, tangent) * tangent;
		}
	}

	azimuthOrigin = normalize(projectedWo);
	azimuthBitangent = normalize(cross(azimuthOrigin, tangent));
}

void RTXCR_FarFieldDirectionCoordinates(const vec3 wi,
		const vec3 tangent,
		const vec3 azimuthOrigin,
		const vec3 azimuthBitangent,
		out float sinThetaI,
		out float cosThetaI,
		out float phi,
		out float cosPhi) {
	sinThetaI = clamp(dot(wi, tangent), -1.0, 1.0);
	cosThetaI = sqrt(max(1.0 - sinThetaI * sinThetaI, 0.0));

	const vec3 projectedWi = wi - sinThetaI * tangent;
	const float invCosThetaI = 1.0 / max(cosThetaI, 1e-7);
	cosPhi = clamp(dot(projectedWi, azimuthOrigin) * invCosThetaI, -1.0, 1.0);
	const float sinPhi = clamp(dot(projectedWi, azimuthBitangent) * invCosThetaI, -1.0, 1.0);
	phi = abs(RTXCR_Atan2safe(sinPhi, cosPhi));
}

float RTXCR_FarFieldTtAzimuthPdf(const float phi,
		const float cosPhi,
		const float etaPrimeInverse,
		out float hTT) {
	const float a = clamp(etaPrimeInverse, 0.0, 1.0);
	const float supportStart = 2.0 * asin(a);
	if (phi < supportStart) {
		hTT = 1.0;
		return 0.0;
	}

	// hTT: root of phi(h) for p = 1
	const float numerator = 0.5 + 0.5 * cosPhi;
	const float denominator = 1.0 + a * a -
			2.0 * a * sqrt(max(0.5 - 0.5 * cosPhi, 0.0));
	hTT = sqrt(clamp(numerator / max(denominator, 1e-7), 0.0, 1.0));

	const float dhR = sqrt(max(1.0 - hTT * hTT, 1e-7));
	const float dhT = sqrt(max(1.0 - a * a * hTT * hTT, 1e-7));
	const float derivative = abs(-2.0 / dhR + 2.0 * a / dhT);
	return derivative > 1e-7 ? 0.5 / derivative : 0.0;
}

vec4 RTXCR_FarFieldLobeProbabilities(const RTXCR_HairMaterialInteractionBcsdf material,
		const float sinThetaO,
		const float cosThetaO) {
	const float diffuseProbability = clamp(material.diffuseReflectionWeight, 0.0, 1.0);
	const float ior = max(material.ior, 1.0001);
	const float f0 = RTXCR_CalculateBaseReflectivity(1.0, ior);
	const float fresnel = RTXCR_EvalFresnelSchlick(f0, cosThetaO);
	const float cosThetaT = sqrt(max(1.0 - sinThetaO * sinThetaO / (ior * ior), 1e-7));
	const vec3 transmittance = exp(-2.0 * max(material.absorptionCoefficient, vec3(0.0)) / cosThetaT);

	const float energyR = fresnel;
	const float energyTT = (1.0 - fresnel) * (1.0 - fresnel) * RTXCR_Luminance(transmittance);

	// A2 at h = 0 is a conservative, inexpensive proxy for the fitted TRT
	// lobe.  Unlike the fit itself, it responds correctly to runtime IOR.
	const float energyTRT = (1.0 - fresnel) * (1.0 - fresnel) * fresnel *
			RTXCR_Luminance(transmittance * transmittance);

	// Keep every non-zero scattering component represented in the proposal.
	vec3 specularEnergy = max(vec3(energyR, energyTT, energyTRT), vec3(1e-4));
	specularEnergy /= specularEnergy.x + specularEnergy.y + specularEnergy.z;
	return vec4(diffuseProbability, (1.0 - diffuseProbability) * specularEnergy);
}

// Custom far-field BCSDF eval() [Eugene d'Eon - 2022]
//  R lobe: [d'Eon et al. 2014 - SIGGRAPH talk]
//  TT lobe: [Marschner et al. 2003]
//  TRT lobe: custom 3-Gaussian lobe based on fitting to MC simulation
// Returns the CRCF (including the hair integration measure) and
// the full solid-angle PDF used by RTXCR_SampleFarFieldBcsdf.
void RTXCR_HairFarFieldBcsdfEval(const RTXCR_HairInteractionSurface hairInteractionSurface,
		const RTXCR_HairMaterialInteractionBcsdf hairMaterialInteractionBcsdf,
		const vec3 wi, // pointing to light
		const vec3 wo, // pointing to camera
		out vec3 bsdf, // Far-field CRCF for hair lobes (R, TT, TRT)
		out vec3 bsdfDiffuse, // [optional] The extension hair diffuse CRCF for artificial hair, set diffuseReflectionWeight to 0 to disable this feature
		out float pdf) { // PDF for the current sample (used for indirect pass)
	vec3 tangent;
	vec3 azimuthOrigin;
	vec3 azimuthBitangent;
	float sinThetaO;
	float cosThetaO;
	RTXCR_FarFieldCoordinateFrame(hairInteractionSurface, wo, tangent,
			azimuthOrigin, azimuthBitangent,
			sinThetaO, cosThetaO);

	// determine cylindrical coordinates (theta/phi) [Marschner et al. 2003]
	float sinThetaI;
	float cosThetaI;
	float phi;
	float cosPhi;
	RTXCR_FarFieldDirectionCoordinates(wi, tangent, azimuthOrigin, azimuthBitangent, sinThetaI, cosThetaI, phi, cosPhi);

	const float thetaI = asin(sinThetaI);
	const float thetaO = asin(sinThetaO);
	const float thetaD = 0.5 * (thetaO - thetaI);
	const float sinThetaD = sin(thetaD);
	const float cosThetaD = cos(thetaD);

	// load fiber properties
	const float roughness = max(hairMaterialInteractionBcsdf.roughness, 1e-4);
	const float alpha = hairMaterialInteractionBcsdf.cuticleAngle;
	const float ior = max(hairMaterialInteractionBcsdf.ior, 1.0001);
	const float iorSqr = ior * ior;
	const float f0 = RTXCR_CalculateBaseReflectivity(1.0, ior);
	const vec3 mua = max(hairMaterialInteractionBcsdf.absorptionCoefficient, vec3(0.0));

	// Compute R lobe - smooth azimuthal N term, normalized longitudinal M term
	// p(phi) = cos(phi / 2) / 4 is normalized on [-pi, pi].
	const float azimuthR = 0.25 * max(cos(0.5 * phi), 0.0);
	// tighten the R lobe (or not) with phi - [d'Eon et al. 2014 SIGGRAPH talk]
	const float betaR = sqrt(2.0) * roughness * max(0.01, cos(0.5 * phi));
	const float longitudinalR = RTXCR_FarFieldLongitudinal(thetaO + 2.0 * alpha, sinThetaI, cosThetaI, betaR);
	// Attenuation is parameterized by the outgoing ray and cylinder offset,
	// as in Ap.  Keeping it independent of the normalized longitudinal term
	// is what makes the lobe's directional integral conservative.
	// [d'Eon et al. 2011 - (12)], using the outgoing-ray approximation.
	const float fresnelR = RTXCR_EvalFresnelSchlick(f0, clamp(cosThetaO * cos(0.5 * phi), 0.0, 1.0));

	// Compute TT lobe - exact azimuthal Jacobian N term, normalized longitudinal M term
	// Sample/evaluate the exact Jacobian of uniform cylinder offset h.
	const float betaTT = 0.5 * roughness * sqrt(max((iorSqr - 1.0) / max(cosThetaO * cosThetaO, 1e-7), 0.0));
	const float longitudinalTT = RTXCR_FarFieldLongitudinal(thetaO - alpha,
			sinThetaI, cosThetaI, betaTT);
	const float etaPrimeInverse = cosThetaD / // 1.0 / eta_prime
			sqrt(max(iorSqr - sinThetaD * sinThetaD, 1e-7));
	float hTT;
	const float azimuthTT = RTXCR_FarFieldTtAzimuthPdf(phi, cosPhi, etaPrimeInverse, hTT);
	const float cosGammaI = sqrt(max(1.0 - hTT * hTT, 0.0));
	// [d'Eon et al. 2011 - (14)], using the outgoing-ray approximation.
	const float fresnelTT = RTXCR_EvalFresnelSchlick(f0, clamp(cosThetaO * cosGammaI, 0.0, 1.0));
	const float cosThetaT = sqrt(max(1.0 - sinThetaO * sinThetaO / iorSqr, 1e-7));
	const float etaPrimeInverseO = cosThetaO / sqrt(max(iorSqr - sinThetaO * sinThetaO, 1e-7));
	const float cosGammaT = sqrt(max(1.0 - hTT * hTT * etaPrimeInverseO * etaPrimeInverseO, 0.0));
	const vec3 attenuationTT = (1.0 - fresnelTT) * (1.0 - fresnelTT) * exp(-2.0 * mua * cosGammaT / cosThetaT); // TODO: absorption with Medulla

	// compute TRT lobe as sum of 3 Gaussians
	// The Gaussians are wrapped periodically and their fit parameters are functions of theta_i.
	const float betaTRT = roughness * (2.0 + pow(abs(thetaO), 1.5));
	const float longitudinalTRT = RTXCR_FarFieldLongitudinal(thetaO - 3.0 * alpha,
			sinThetaI, cosThetaI, betaTRT);
	RTXCR_FarFieldTrtFit trtFit;
	RTXCR_FarFieldComputeTrtFit(thetaI, roughness, trtFit);
	const float trtAzimuthUnnormalized = RTXCR_FarFieldTrtAzimuth(phi, trtFit);
	const float trtAzimuthIntegral = max(2.0 * trtFit.weight1 + trtFit.weight2, 1e-7);
	// assume h = 0 for absorption
	const float fresnelSpec = RTXCR_EvalFresnelSchlick(f0, cosThetaO);
	const vec3 transmittanceSpec = exp(-2.0 * mua / cosThetaT);
	// The fitted weights define the TRT azimuthal shape.  Give that normalized
	// shape the physical A2 energy at h = 0 so IOR and absorption affect TRT,
	// instead of accidentally counting the fit amplitude as extra energy.
	const vec3 attenuationTRT = (1.0 - fresnelSpec) * (1.0 - fresnelSpec) *
			fresnelSpec * transmittanceSpec * transmittanceSpec;

	const float diffuseWeight = clamp(hairMaterialInteractionBcsdf.diffuseReflectionWeight, 0.0, 1.0);
	const float diffuseAzimuth = max((RTXCR_PI - phi) * cosPhi + sin(phi), 0.0);
	const float diffusePdf = cosThetaI * (0.25 / RTXCR_PI) * diffuseAzimuth;

	// eval:
	bsdf = max((1.0 - diffuseWeight) *
					(longitudinalR * azimuthR * fresnelR +
							longitudinalTT * azimuthTT * attenuationTT +
							longitudinalTRT * trtAzimuthUnnormalized / trtAzimuthIntegral * attenuationTRT),
			vec3(0.0));
	bsdfDiffuse = diffuseWeight * diffusePdf * clamp(hairMaterialInteractionBcsdf.diffuseReflectionTint, vec3(0.0), vec3(1.0));

	// PDF is the complete normalized mixture used by the sampling function.
	const vec4 lobeProbability = RTXCR_FarFieldLobeProbabilities(hairMaterialInteractionBcsdf, sinThetaO, cosThetaO);
	const float proposalR = longitudinalR * azimuthR;
	const float proposalTT = longitudinalTT * azimuthTT;
	const float proposalTRT = longitudinalTRT * trtAzimuthUnnormalized / trtAzimuthIntegral;
	pdf = dot(lobeProbability, vec4(diffusePdf, proposalR, proposalTT, proposalTRT));
}

bool RTXCR_SampleFarFieldBcsdf(const RTXCR_HairInteractionSurface hairInteractionSurface,
		const RTXCR_HairMaterialInteractionBcsdf hairMaterialInteractionBcsdf,
		const vec3 wo,
		const float h,
		const vec3 rand3[2], // RTXCR's rand2, renamed: rand2() is a function here.
		out vec3 wi,
		out vec3 bsdf,
		out vec3 bsdfDiffuse,
		out float pdf) {
	vec3 tangent;
	vec3 azimuthOrigin;
	vec3 azimuthBitangent;
	float sinThetaO;
	float cosThetaO;
	RTXCR_FarFieldCoordinateFrame(hairInteractionSurface, wo, tangent,
			azimuthOrigin, azimuthBitangent,
			sinThetaO, cosThetaO);

	const float thetaO = asin(sinThetaO);
	const float roughness = max(hairMaterialInteractionBcsdf.roughness, 1e-4);
	const float alpha = hairMaterialInteractionBcsdf.cuticleAngle;
	const float ior = max(hairMaterialInteractionBcsdf.ior, 1.0001);
	// Select a lobe using outgoing-direction energy proxies at h = 0, then
	// sample its normalized longitudinal and azimuthal proposal.
	const vec4 lobeProbability = RTXCR_FarFieldLobeProbabilities(
			hairMaterialInteractionBcsdf, sinThetaO, cosThetaO);

	float sinThetaI;
	float cosThetaI;

	if (rand3[0].z < lobeProbability.x) {
		// sample diffuse
		// The marginal of cosine hemispheres over uniformly sampled cylinder
		// offsets is exactly the analytic diffuse BCSDF above.
		const float clampedH = clamp(h, -1.0, 1.0);
		const vec3 cylinderNormal = sqrt(max(1.0 - clampedH * clampedH, 0.0)) * azimuthOrigin +
				clampedH * azimuthBitangent;
		const vec3 cylinderBitangent = normalize(cross(cylinderNormal, tangent));
		float unusedPdf;
		const vec3 localWi = RTXCR_SampleHemisphere(rand3[0].xy, unusedPdf);
		wi = localWi.x * tangent + localWi.y * cylinderBitangent +
				localWi.z * cylinderNormal;
	} else if (rand3[0].z < lobeProbability.x + lobeProbability.y) {
		// sample R
		const float phi = RTXCR_PhiR(clamp(h, -1.0, 1.0));
		const float betaR = sqrt(2.0) * roughness * max(0.01, cos(0.5 * phi));
		RTXCR_FarFieldSampleLongitudinal(thetaO + 2.0 * alpha, betaR,
				rand3[0].xy, sinThetaI, cosThetaI);
		wi = cosThetaI * (cos(phi) * azimuthOrigin + sin(phi) * azimuthBitangent) +
				sinThetaI * tangent;
	} else if (rand3[0].z < lobeProbability.x + lobeProbability.y + lobeProbability.z) {
		// sample TT
		const float betaTT = 0.5 * roughness * sqrt(max((ior * ior - 1.0) / max(cosThetaO * cosThetaO, 1e-7), 0.0));
		RTXCR_FarFieldSampleLongitudinal(thetaO - alpha, betaTT, rand3[0].xy, sinThetaI, cosThetaI);
		const float thetaI = asin(sinThetaI);
		const float thetaD = 0.5 * (thetaO - thetaI);
		const float etaPrimeInverse = cos(thetaD) / // 1.0 / eta_prime
				sqrt(max(ior * ior - sin(thetaD) * sin(thetaD), 1e-7));
		const float phi = RTXCR_FarFieldWrapAngle(
				RTXCR_PhiTT(clamp(h, -1.0, 1.0), etaPrimeInverse));
		wi = cosThetaI * (cos(phi) * azimuthOrigin + sin(phi) * azimuthBitangent) +
				sinThetaI * tangent;
	} else {
		// sample TRT
		const float betaTRT = roughness * (2.0 + pow(abs(thetaO), 1.5));
		RTXCR_FarFieldSampleLongitudinal(thetaO - 3.0 * alpha, betaTRT,
				rand3[0].xy, sinThetaI, cosThetaI);

		RTXCR_FarFieldTrtFit trtFit;
		RTXCR_FarFieldComputeTrtFit(asin(sinThetaI), roughness, trtFit);
		const float integral = max(2.0 * trtFit.weight1 + trtFit.weight2, 1e-7);
		const float selector = clamp(0.5 * h + 0.5, 0.0, 1.0) * integral;
		float center;
		float width;
		if (selector < trtFit.weight1) {
			center = trtFit.peak;
			width = trtFit.width1;
		} else if (selector < 2.0 * trtFit.weight1) {
			center = -trtFit.peak;
			width = trtFit.width1;
		} else {
			center = 0.0;
			width = trtFit.width2;
		}

		const float gaussian = RTXCR_RandomGaussian1D(rand3[1].x, min(rand3[1].y, 1.0 - 1e-7));
		const float phi = RTXCR_FarFieldWrapAngle(center + width * gaussian);
		wi = cosThetaI * (cos(phi) * azimuthOrigin + sin(phi) * azimuthBitangent) +
				sinThetaI * tangent;
	}

	RTXCR_HairFarFieldBcsdfEval(hairInteractionSurface,
			hairMaterialInteractionBcsdf,
			wi, wo, bsdf, bsdfDiffuse, pdf);
	return pdf > 0.0;
}

#endif // RTXCR_HAIR_GLSL
