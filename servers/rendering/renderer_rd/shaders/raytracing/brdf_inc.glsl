#ifndef BRDF_HLSLI
#define BRDF_HLSLI

/* Creative Commons CC0 Public Domain. To the extent possible under law,
Jakub Boksansky has waived all copyright and related or neighboring rights
to Crash Course in BRDF Implementation Code Sample.
This work is published from: Germany. */

// This is a code sample accompanying the "Crash Course in BRDF Implementation" article
// v1.1, February 2021

// -------------------------------------------------------------------------
//    C++ / GLSL / HLSL compatibility
// -------------------------------------------------------------------------

#if defined(__cplusplus) && !defined(GLSL)

#pragma once

// Include additional things when compiling HLSL as C++
// Here we use the GLM library to support HLSL types and functions
// Modification: Godot preprocessor doesn't understand #include in #if blocks.. If you need to use C++, make sure to include these yourself.
//#include "glm/glm/glm.hpp"
//#include "glm/glm/gtc/constants.hpp"
//#include "glm/glm/gtx/compatibility.hpp"

#define OUT_PARAMETER(X) X &

using namespace glm;

inline float rsqrt(float x) {
	return inversesqrt(x);
}
inline float saturate(float x) {
	return clamp(x, 0.0f, 1.0f);
}
inline float rcp(float x) {
	return 1.0f / x;
}
template <typename T>
T mad(T a, T b, T c) {
	return a * b + c;
}
template <typename M, typename V>
V mul(const M &m, const V &v) {
	return m * v;
}
// GLM is column-major like GLSL - transpose constructor args
#define MAKE_FLOAT2x2(r0c0, r0c1, r1c0, r1c1) mat2(r0c0, r1c0, r0c1, r1c1)
#define MAKE_FLOAT3x3(r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2) \
	mat3(r0c0, r1c0, r2c0, r0c1, r1c1, r2c1, r0c2, r1c2, r2c2)

#elif defined(GLSL)
// GLSL detected - define HLSL-compatible types and functions

#define float2 vec2
#define float3 vec3
#define float4 vec4
#define float2x2 mat2
#define float3x3 mat3
#define float4x4 mat4

#define lerp mix
#define saturate(x) clamp(x, 0.0, 1.0)
#define rsqrt inversesqrt
#define rcp(x) (1.0 / (x))
#define mad(a, b, c) ((a) * (b) + (c))
#define mul(M, v) ((M) * (v))

// HLSL is row-major, GLSL is column-major - transpose constructor args for GLSL
#define MAKE_FLOAT2x2(r0c0, r0c1, r1c0, r1c1) mat2(r0c0, r1c0, r0c1, r1c1)
#define MAKE_FLOAT3x3(r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2) \
	mat3(r0c0, r1c0, r2c0, r0c1, r1c1, r2c1, r0c2, r1c2, r2c2)

#define OUT_PARAMETER(X) out X

#else
// HLSL
#define OUT_PARAMETER(X) out X
#define MAKE_FLOAT2x2(r0c0, r0c1, r1c0, r1c1) float2x2(r0c0, r0c1, r1c0, r1c1)
#define MAKE_FLOAT3x3(r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2) \
	float3x3(r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2)
#endif

// -------------------------------------------------------------------------
//    Constant Definitions
// -------------------------------------------------------------------------

#define NONE 0

// NDF definitions
#define GGX 1
#define BECKMANN 2

// Specular BRDFs
#define MICROFACET 1
#define PHONG 2

// Diffuse BRDFs
#define LAMBERTIAN 1
#define OREN_NAYAR 2
#define DISNEY 3
#define FROSTBITE 4

// BRDF types
#define DIFFUSE_TYPE 1
#define SPECULAR_TYPE 2

// PIs
#ifndef PI
#define PI 3.141592653589f
#endif

#ifndef TWO_PI
#define TWO_PI (2.0f * PI)
#endif

#ifndef ONE_OVER_PI
#define ONE_OVER_PI (1.0f / PI)
#endif

#ifndef ONE_OVER_TWO_PI
#define ONE_OVER_TWO_PI (1.0f / TWO_PI)
#endif

// -------------------------------------------------------------------------
//    Configuration macros (user editable - set your preferences here)
// -------------------------------------------------------------------------

// Specify what NDF (GGX or BECKMANN you want to use)
#ifndef MICROFACET_DISTRIBUTION
#define MICROFACET_DISTRIBUTION GGX
//#define MICROFACET_DISTRIBUTION BECKMANN
#endif

// Specify default specular and diffuse BRDFs
#ifndef SPECULAR_BRDF
#define SPECULAR_BRDF MICROFACET
//#define SPECULAR_BRDF PHONG
//#define SPECULAR_BRDF NONE
#endif

#ifndef DIFFUSE_BRDF
#define DIFFUSE_BRDF LAMBERTIAN
//#define DIFFUSE_BRDF OREN_NAYAR
//#define DIFFUSE_BRDF DISNEY
//#define DIFFUSE_BRDF FROSTBITE
//#define DIFFUSE_BRDF NONE
#endif

// Specifies minimal reflectance for dielectrics (when metalness is zero)
// Nothing has lower reflectance than 2%, but we use 4% to have consistent results with UE4, Frostbite, et al.
#define MIN_DIELECTRICS_F0 0.04f

// Enable this to weigh diffuse by Fresnel too, otherwise specular and diffuse will be simply added together
// (this is disabled by default for Frostbite diffuse which is normalized to combine well with GGX Specular BRDF)
#if DIFFUSE_BRDF != FROSTBITE
#define COMBINE_BRDFS_WITH_FRESNEL 1
#endif

// Uncomment this to use "general" version of G1 which is not optimized and uses NDF-specific G_Lambda (can be useful for experimenting and debugging)
//#define Smith_G1 Smith_G1_General

// Enable optimized G2 implementation which includes division by specular BRDF denominator (not available for all NDFs, check macro G2_DIVIDED_BY_DENOMINATOR if it was actually used)
#define USE_OPTIMIZED_G2 1

// Enable height correlated version of G2 term. Separable version will be used otherwise
#define USE_HEIGHT_CORRELATED_G2 1

// Master toggle for the shading-normal clamp at grazing angles.
// 0 -> clampShadingNormal becomes a passthrough; only the mirror fallback in
//      shade_and_bounce remains active.
// 1 -> clamp + mirror two-layer fallback (default).
#ifndef RT_SHADING_NORMAL_CLAMP_ENABLED
#define RT_SHADING_NORMAL_CLAMP_ENABLED 1
#endif

// Geometry NdotV below which clampShadingNormal starts pulling the shading
// normal toward the geometry normal. Higher values flatten normal-mapped
// detail at grazing angles but reduce BRDF below-hemisphere rejections.
// Used by both the production indirect-bounce path and the matching debug
// visualization so they remain in sync.
#ifndef RT_SHADING_NORMAL_CLAMP_THRESHOLD
#define RT_SHADING_NORMAL_CLAMP_THRESHOLD 0.4f
#endif

// Enable below-hemisphere sample recovery.
// 0 -> disable recovery and terminate the path on rejection instead.
// 1 -> enable recovery and reuse the original BRDF weight.
#ifndef RT_BELOW_HEMISPHERE_RECOVERY_ENABLED
#define RT_BELOW_HEMISPHERE_RECOVERY_ENABLED 1
#endif

// -------------------------------------------------------------------------
//    Automatically resolved macros based on preferences (don't edit these)
// -------------------------------------------------------------------------

// Select distribution function
#if MICROFACET_DISTRIBUTION == GGX
#define Microfacet_D GGX_D
#elif MICROFACET_DISTRIBUTION == BECKMANN
#define Microfacet_D Beckmann_D
#endif

// Select G functions (masking/shadowing) depending on selected distribution
#if MICROFACET_DISTRIBUTION == GGX
#define Smith_G_Lambda Smith_G_Lambda_GGX
#elif MICROFACET_DISTRIBUTION == BECKMANN
#define Smith_G_Lambda Smith_G_Lambda_Beckmann_Walter
#endif

#ifndef Smith_G1
// Define version of G1 optimized specifically for selected NDF
#if MICROFACET_DISTRIBUTION == GGX
#define Smith_G1 Smith_G1_GGX
#elif MICROFACET_DISTRIBUTION == BECKMANN
#define Smith_G1 Smith_G1_Beckmann_Walter
#endif
#endif

// Select default specular and diffuse BRDF functions
#if SPECULAR_BRDF == MICROFACET
#define evalSpecular evalMicrofacet
#define sampleSpecular sampleSpecularMicrofacet
#if MICROFACET_DISTRIBUTION == GGX
#define sampleSpecularHalfVector sampleGGXVNDF
#else
#define sampleSpecularHalfVector sampleBeckmannWalter
#endif
#elif SPECULAR_BRDF == PHONG
#define evalSpecular evalPhong
#define sampleSpecular sampleSpecularPhong
#define sampleSpecularHalfVector samplePhong
#else
#define evalSpecular evalVoid
#define sampleSpecular sampleSpecularVoid
#define sampleSpecularHalfVector sampleSpecularHalfVectorVoid
#endif

#if MICROFACET_DISTRIBUTION == GGX
#define specularSampleWeight specularSampleWeightGGXVNDF
#define specularPdf sampleGGXVNDFReflectionPdf
#else
#define specularSampleWeight specularSampleWeightBeckmannWalter
#define specularPdf sampleBeckmannWalterReflectionPdf
#endif

#if DIFFUSE_BRDF == LAMBERTIAN
#define evalDiffuse evalLambertian
#define diffuseTerm lambertian
#elif DIFFUSE_BRDF == OREN_NAYAR
#define evalDiffuse evalOrenNayar
#define diffuseTerm orenNayar
#elif DIFFUSE_BRDF == DISNEY
#define evalDiffuse evalDisneyDiffuse
#define diffuseTerm disneyDiffuse
#elif DIFFUSE_BRDF == FROSTBITE
#define evalDiffuse evalFrostbiteDisneyDiffuse
#define diffuseTerm frostbiteDisneyDiffuse
#else
#define evalDiffuse evalVoid
#define evalIndirectDiffuse evalIndirectVoid
#define diffuseTerm none
#endif

// -------------------------------------------------------------------------
//    Structures
// -------------------------------------------------------------------------

struct MaterialProperties {
	float3 baseColor;
	float metalness;

	float3 emissive;
	float roughness;
	float dielectricF0; // Base reflectance for dielectrics (e.g. 0.04 for default 4% reflectance).
	float transmissivness;
	float opacity;
	float3 diffuseTransmission; // Albedo of a thin diffuse-transmission lobe on the other side of N (D103); 0 = none.
	float anisotropy; // VAL: the specular lobe's anisotropy, Godot's ANISOTROPY, at most 1 (D118 MM1); 0 = isotropic.
	float3 anisotropyT; // With anisotropy: the unit tangent its roughness stretches along, orthogonal to the N the BRDF is given.
	float clearcoat; // VAL: the coat's strength, Godot's CLEARCOAT, 0..1 (D118 MM2); 0 = no coat.
	float clearcoatAlpha; // With a coat: its GGX alpha, mix(0.001, 0.1, CLEARCOAT_ROUGHNESS) as the raster takes it.
	float3 clearcoatN; // With a coat: its unit normal, the vertex normal without the normal map.
	float3 sheenColor; // VAL: the sheen layer's reflectivity, from Godot's RIM and RIM_TINT (D118 MM3); 0 = no sheen.
	float sheenRoughness; // With a sheen: Estevez and Kulla's r, the surface's roughness clamped to 0.1..1, where the albedo fit holds.
	float subsurface; // VAL: the diffuse lobe's share that walks under the surface, Godot's SSS_STRENGTH (D120); the direct lobe keeps 1 - it.
	bool hair; // VAL: the hair BCSDF (D126 H2) in place of every other lobe: a hair material on a strand.
	bool hairChiang; // With hair: Chiang's model (rendering/pathtracing/hair_bcsdf), else RTXCR's far-field one.
	float3 hairT; // With hair: the strand's unit tangent, orthogonal to the N the BCSDF is given: Chiang's frame.
	float3 hairAxis; // With hair: the strand's unit axis as the trace gives it, the far field's tangent (its own frame).
	float3 hairAbsorption; // With hair: the fibre's absorption coefficient, RTXCR's (per unit radius).
	float hairIor; // With hair: the fibre's index of refraction.
	float hairTilt; // With hair: the cuticle tilt in radians.
	float hairInset; // With hair: how deep inside the strand its rays start (HitData.strand_inset), 0 for the surface's offset.
};

// A thin surface that passes light to its other side (D103): light below N is evaluated, not rejected.
bool hasDiffuseTransmission(MaterialProperties material) {
	return any(greaterThan(material.diffuseTransmission, float3(0.0f, 0.0f, 0.0f)));
}

// -------------------------------------------------------------------------
//    VAL: hair (D126 H2, docs/r6-strand-hair.md)
// -------------------------------------------------------------------------

// clang-format off
#include "raytracing_hair_inc.glsl"
// clang-format on

bool hasHair(MaterialProperties material) {
	return material.hair;
}

// RTXCR's absorption for a strand: from the melanin with the albedo as a dye tint (Cycles' tint, white for none), or
// from the albedo, the colour mapping at the azimuthal roughness, which is the material's roughness.
float3 hairAbsorptionCoefficient(bool p_color, float3 p_albedo, float p_melanin, float p_redness, float p_roughness) {
	float3 tint = RTXCR_AbsorptionCoefficientFromColor(p_albedo, p_roughness);
	return p_color ? tint : RTXCR_AbsorptionCoefficientFromMelanin(p_melanin, p_redness) + tint;
}

// The strand's colour the denoiser's guide takes for the absorption: the inverse of RTXCR's colour mapping.
float3 hairGuideColor(float3 p_absorption, float p_roughness) {
	const float betaN = p_roughness;
	const float tmp = 5.969f - 0.215f * betaN + 2.532f * betaN * betaN - 10.73f * pow(betaN, 3.0f) + 5.574f * pow(betaN, 4.0f) + 0.245f * pow(betaN, 5.0f);
	return exp(-sqrt(max(p_absorption, float3(0.0f, 0.0f, 0.0f))) * tmp);
}

// The hair's local frame as RTXCR's sample builds it for Chiang's model: x the tangent, y cross(N, T), z the normal.
float3 hairToLocal(float3 v, float3 N, float3 T) {
	return float3(dot(v, T), dot(v, cross(N, T)), dot(v, N));
}

float3 hairFromLocal(float3 v, float3 N, float3 T) {
	return v.x * T + v.y * cross(N, T) + v.z * N;
}

// In Chiang's local frame (p_local), or in world space for the far field, whose frame is the strand's axis as the trace
// gives it: RTXCR's far field builds its own azimuth from the tangent and wo, and reads N only where wo runs along the
// axis, so it takes the axis unchanged (on a tapered segment the normal tilts off the perpendicular to it).
RTXCR_HairInteractionSurface hairSurface(float3 N, float3 V, MaterialProperties material, bool p_local) {
	RTXCR_HairInteractionSurface surface;
	surface.incidentRayDirection = p_local ? hairToLocal(V, N, material.hairT) : V;
	surface.shadingNormal = p_local ? float3(0.0f, 0.0f, 1.0f) : N;
	surface.tangent = p_local ? float3(1.0f, 0.0f, 0.0f) : material.hairAxis;
	return surface;
}

RTXCR_HairMaterialInteraction hairChiangInteraction(float3 N, float3 V, MaterialProperties material) {
	RTXCR_HairMaterialData data;
	data.baseColor = float3(1.0f, 1.0f, 1.0f);
	// One roughness for both (docs/r6-strand-hair.md, H2 design). Both floored as the far field floors its own: a zero
	// roughness gives Chiang's longitudinal variance 0 (its M_p divides by it), an IOR of 1 no refraction to invert.
	const float roughness = max(material.roughness, 1e-4f);
	const float ior = max(material.hairIor, 1.0001f);
	data.longitudinalRoughness = roughness;
	data.azimuthalRoughness = roughness;
	data.ior = ior;
	data.eta = 1.0f / ior;
	data.fresnelApproximation = 0u; // The exact dielectric Fresnel.
	data.absorptionModel = RTXCR_HairAbsorptionModel_Color;
	data.melanin = 0.0f;
	data.melaninRedness = 0.0f;
	data.cuticleAngleInDegrees = degrees(material.hairTilt);
	RTXCR_HairMaterialInteraction interaction = RTXCR_CreateHairMaterialInteraction(data, hairSurface(N, V, material, true));
	interaction.absorptionCoefficient = material.hairAbsorption;
	return interaction;
}

RTXCR_HairMaterialInteractionBcsdf hairFarFieldInteraction(MaterialProperties material) {
	RTXCR_HairMaterialInteractionBcsdf interaction;
	interaction.diffuseReflectionTint = float3(0.0f, 0.0f, 0.0f);
	interaction.diffuseReflectionWeight = 0.0f; // RTXCR's diffuse extension is for artificial hair.
	interaction.roughness = material.roughness;
	interaction.absorptionCoefficient = material.hairAbsorption;
	interaction.ior = material.hairIor;
	interaction.cuticleAngle = material.hairTilt;
	return interaction;
}

// The hair BCSDF for light from L seen from V at a strand of unit normal N: RTXCR's value, which carries the integration
// measure the other lobes bake in as their cosine (so the caller multiplies by nothing), and the density the hair
// sampler below draws L with.
void evalHair(float3 N, float3 L, float3 V, MaterialProperties material, OUT_PARAMETER(float3) r_value, OUT_PARAMETER(float) r_pdf) {
	if (material.hairChiang) {
		const RTXCR_HairMaterialInteraction interaction = hairChiangInteraction(N, V, material);
		const float3 wi = hairToLocal(L, N, material.hairT);
		const float3 wo = hairToLocal(V, N, material.hairT);
		r_value = RTXCR_HairChiangBsdfEval(interaction, wi, wo);
		r_pdf = RTXCR_ChiangBsdfPdf(interaction, wi, wo);
	} else {
		float3 diffuse;
		RTXCR_HairFarFieldBcsdfEval(hairSurface(N, V, material, false), hairFarFieldInteraction(material), L, V, r_value, diffuse, r_pdf);
		r_value = r_pdf > 0.0f ? r_value + diffuse : float3(0.0f, 0.0f, 0.0f); // As RTXCR's sample weighs its light samples.
	}
}

// A direction from the hair BCSDF and its weight, value over density, from six uniform numbers (u.xy, u.z, w.xy, w.z):
// Chiang's lobe and azimuth pair, then its longitudinal pair; the far field's longitudinal pair, its lobe, its azimuth's
// Gaussian pair and the cylinder offset h, uniform over [-1, 1], as RTXCR's sample draws them. False when the sampler
// rejects the direction (a zero density).
bool sampleHair(float3 u, float3 w, float3 N, float3 V, MaterialProperties material, OUT_PARAMETER(float3) r_direction, OUT_PARAMETER(float3) r_weight, OUT_PARAMETER(float) r_pdf) {
	r_direction = N;
	r_weight = float3(0.0f, 0.0f, 0.0f);
	r_pdf = 0.0f;
	if (material.hairChiang) {
		const RTXCR_HairMaterialInteraction interaction = hairChiangInteraction(N, V, material);
		float2 rnd[2];
		rnd[0] = u.xy;
		rnd[1] = w.xy;
		float3 wi;
		float3 value;
		uint lobe;
		if (!RTXCR_SampleChiangBsdf(interaction, hairToLocal(V, N, material.hairT), rnd, wi, r_pdf, value, lobe)) {
			return false;
		}
		r_direction = hairFromLocal(wi, N, material.hairT);
		r_weight = value / r_pdf;
		return true;
	}
	float3 rnd[2];
	rnd[0] = float3(u.xy, u.z);
	rnd[1] = float3(w.xy, 0.0f);
	float3 value;
	float3 diffuse;
	if (!RTXCR_SampleFarFieldBcsdf(hairSurface(N, V, material, false), hairFarFieldInteraction(material), V, 2.0f * w.z - 1.0f, rnd, r_direction, value, diffuse, r_pdf)) {
		return false;
	}
	r_weight = (value + diffuse) / r_pdf;
	return true;
}

// Data needed to evaluate BRDF (surface and material properties at given point + configuration of light and normal vectors)
struct BrdfData {
	// Material properties
	float3 specularF0;
	float3 diffuseReflectance;

	// Roughnesses
	float roughness; //< perceptively linear roughness (artist's input)
	float alpha; //< linear roughness - often 'alpha' in specular BRDF equations
	float alphaSquared; //< alpha squared - pre-calculated value commonly used in BRDF equations
	float2 alpha2D; //< VAL: alpha along T and B (D118 MM1); (alpha, alpha) for an isotropic lobe
	bool anisotropic; //< VAL: the lobe is anisotropic; the *TB terms below are set only then

	// Commonly used terms for BRDF evaluation
	float3 F; //< Fresnel term

	// Vectors
	float3 V; //< Direction to viewer (or opposite direction of incident ray)
	float3 N; //< Shading normal
	float3 H; //< Half vector (microfacet normal)
	float3 L; //< Direction to light (or direction of reflecting ray)

	float NdotL;
	float NdotV;

	float LdotH;
	float NdotH;
	float VdotH;

	// VAL: H, L and V along the anisotropic lobe's tangent and bitangent (D118 MM1)
	float2 HdotTB;
	float2 LdotTB;
	float2 VdotTB;

	// True when V/L is backfacing wrt. shading normal N
	bool Vbackfacing;
	bool Lbackfacing;
};

// -------------------------------------------------------------------------
//    Utilities
// -------------------------------------------------------------------------

// Converts Phong's exponent (shininess) to Beckmann roughness (alpha)
// Source: "Microfacet Models for Refraction through Rough Surfaces" by Walter et al.
float shininessToBeckmannAlpha(float shininess) {
	return sqrt(2.0f / (shininess + 2.0f));
}

// Converts Beckmann roughness (alpha) to Phong's exponent (shininess)
// Source: "Microfacet Models for Refraction through Rough Surfaces" by Walter et al.
float beckmannAlphaToShininess(float alpha) {
	return 2.0f / min(0.9999f, max(0.0002f, (alpha * alpha))) - 2.0f;
}

// Converts Beckmann roughness (alpha) to Oren-Nayar roughness (sigma)
// Source: "Moving Frostbite to Physically Based Rendering" by Lagarde & de Rousiers
float beckmannAlphaToOrenNayarRoughness(float alpha) {
	return 0.7071067f * atan(alpha);
}

float luminance(float3 rgb) {
	return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

float3 baseColorToSpecularF0(float3 baseColor, float metalness, float dielectricF0) {
	return lerp(float3(dielectricF0, dielectricF0, dielectricF0), baseColor, metalness);
}

float3 baseColorToDiffuseReflectance(float3 baseColor, float metalness) {
	return baseColor * (1.0f - metalness);
}

float none(const BrdfData data) {
	return 0.0f;
}

float3 evalVoid(const BrdfData data) {
	return float3(0.0f, 0.0f, 0.0f);
}

void evalIndirectVoid(const BrdfData data, float2 u, OUT_PARAMETER(float3) rayDirection, OUT_PARAMETER(float3) weight) {
	rayDirection = float3(0.0f, 0.0f, 1.0f);
	weight = float3(0.0f, 0.0f, 0.0f);
}

float3 sampleSpecularVoid(float3 Vlocal, float alpha, float alphaSquared, float3 specularF0, float2 u, OUT_PARAMETER(float3) weight) {
	weight = float3(0.0f, 0.0f, 0.0f);
	return float3(0.0f, 0.0f, 0.0f);
}

float3 sampleSpecularHalfVectorVoid(float3 Vlocal, float2 alpha2D, float2 u) {
	return float3(0.0f, 0.0f, 0.0f);
}

// -------------------------------------------------------------------------
//    Quaternion rotations
// -------------------------------------------------------------------------

// Calculates rotation quaternion from input vector to the vector (0, 0, 1)
// Input vector must be normalized!
float4 getRotationToZAxis(float3 v) {
	// Handle special case when v is exact or near opposite of (0, 0, 1)
	if (v.z < -0.99999f) {
		return float4(1.0f, 0.0f, 0.0f, 0.0f);
	}

	return normalize(float4(v.y, -v.x, 0.0f, 1.0f + v.z));
}

// Calculates rotation quaternion from vector (0, 0, 1) to the input vector
// Input vector must be normalized!
float4 getRotationFromZAxis(float3 v) {
	// Handle special case when v is exact or near opposite of (0, 0, 1)
	if (v.z < -0.99999f) {
		return float4(1.0f, 0.0f, 0.0f, 0.0f);
	}

	return normalize(float4(-v.y, v.x, 0.0f, 1.0f + v.z));
}

// Returns the quaternion with inverted rotation
float4 invertRotation(float4 q) {
	return float4(-q.x, -q.y, -q.z, q.w);
}

// Optimized point rotation using quaternion
// Source: https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
float3 rotatePoint(float4 q, float3 v) {
	const float3 qAxis = float3(q.x, q.y, q.z);
	return 2.0f * dot(qAxis, v) * qAxis + (q.w * q.w - dot(qAxis, qAxis)) * v + 2.0f * q.w * cross(qAxis, v);
}

// -------------------------------------------------------------------------
//    Sampling
// -------------------------------------------------------------------------

// Samples a direction within a hemisphere oriented along +Z axis with a cosine-weighted distribution
// Source: "Sampling Transformations Zoo" in Ray Tracing Gems by Shirley et al.
float3 sampleHemisphere(float2 u, OUT_PARAMETER(float) pdf) {
	float a = sqrt(u.x);
	float b = TWO_PI * u.y;

	float3 result = float3(
			a * cos(b),
			a * sin(b),
			sqrt(1.0f - u.x));

	pdf = result.z * ONE_OVER_PI;

	return result;
}

float3 sampleHemisphere(float2 u) {
	float pdf;
	return sampleHemisphere(u, pdf);
}

// For sampling of all our diffuse BRDFs we use cosine-weighted hemisphere sampling, with PDF equal to (NdotL/PI)
float diffusePdf(float NdotL) {
	return NdotL * ONE_OVER_PI;
}

// -------------------------------------------------------------------------
//    Fresnel
// -------------------------------------------------------------------------

// Schlick's approximation to Fresnel term
// f90 should be 1.0, except for the trick used by Schuler (see 'shadowedF90' function)
float3 evalFresnelSchlick(float3 f0, float f90, float NdotS) {
	return f0 + (f90 - f0) * pow(1.0f - NdotS, 5.0f);
}

// Schlick's approximation to Fresnel term calculated using spherical gaussian approximation
// Source: https://seblagarde.wordpress.com/2012/06/03/spherical-gaussien-approximation-for-blinn-phong-phong-and-fresnel/ by Lagarde
float3 evalFresnelSchlickSphericalGaussian(float3 f0, float f90, float NdotV) {
	return f0 + (f90 - f0) * exp2((-5.55473f * NdotV - 6.983146f) * NdotV);
}

// Schlick's approximation to Fresnel term with Hoffman's improvement using the Lazanyi's error term
// Source: "Fresnel Equations Considered Harmful" by Hoffman
// Also see slides http://renderwonk.com/publications/mam2019/naty_mam2019.pdf for examples and explanation of f82 term
float3 evalFresnelHoffman(float3 f0, float f82, float f90, float NdotS) {
	const float alpha = 6.0f; //< Fixed to 6 in order to put peak angle for Lazanyi's error term at 82 degrees (f82)
	float3 a = 17.6513846f * (f0 - f82) + 8.166666f * (float3(1.0f, 1.0f, 1.0f) - f0);
	return saturate(f0 + (f90 - f0) * pow(1.0f - NdotS, 5.0f) - a * NdotS * pow(1.0f - NdotS, alpha));
}

float3 evalFresnel(float3 f0, float f90, float NdotS) {
	// Default is Schlick's approximation
	return evalFresnelSchlick(f0, f90, NdotS);
}

// Attenuates F90 for very low F0 values
// Source: "An efficient and Physically Plausible Real-Time Shading Model" in ShaderX7 by Schuler
// Also see section "Overbright highlights" in Hoffman's 2010 "Crafting Physically Motivated Shading Models for Game Development" for discussion
// IMPORTANT: Note that when F0 is calculated using metalness, it's value is never less than MIN_DIELECTRICS_F0, and therefore,
// this adjustment has no effect. To be effective, F0 must be authored separately, or calculated in different way. See main text for discussion.
float shadowedF90(float3 F0) {
	// This scaler value is somewhat arbitrary, Schuler used 60 in his article. In here, we derive it from MIN_DIELECTRICS_F0 so
	// that it takes effect for any reflectance lower than least reflective dielectrics
	//const float t = 60.0f;
	const float t = (1.0f / MIN_DIELECTRICS_F0);
	return min(1.0f, t * luminance(F0));
}

// -------------------------------------------------------------------------
//    Lambert
// -------------------------------------------------------------------------

float lambertian(const BrdfData data) {
	return 1.0f;
}

float3 evalLambertian(const BrdfData data) {
	return data.diffuseReflectance * (ONE_OVER_PI * data.NdotL);
}

// -------------------------------------------------------------------------
//    Phong
// -------------------------------------------------------------------------

// For derivation see "Phong Normalization Factor derivation" by Giesen
float phongNormalizationTerm(float shininess) {
	return (1.0f + shininess) * ONE_OVER_TWO_PI;
}

float3 evalPhong(const BrdfData data) {
	// First convert roughness to shininess (Phong exponent)
	float shininess = beckmannAlphaToShininess(data.alpha);

	float3 R = reflect(-data.L, data.N);
	return data.specularF0 * (phongNormalizationTerm(shininess) * pow(max(0.0f, dot(R, data.V)), shininess) * data.NdotL);
}

// Samples a Phong distribution lobe oriented along +Z axis
// Source: "Sampling Transformations Zoo" in Ray Tracing Gems by Shirley et al.
float3 samplePhong(float3 Vlocal, float shininess, float2 u, OUT_PARAMETER(float) pdf) {
	float cosTheta = pow(1.0f - u.x, 1.0f / (1.0f + shininess));
	float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

	float phi = TWO_PI * u.y;

	pdf = phongNormalizationTerm(shininess) * pow(cosTheta, shininess);

	return float3(
			cos(phi) * sinTheta,
			sin(phi) * sinTheta,
			cosTheta);
}

float3 samplePhong(float3 Vlocal, float2 alpha2D, float2 u) {
	float shininess = beckmannAlphaToShininess(dot(alpha2D, float2(0.5f, 0.5f)));
	float pdf;
	return samplePhong(Vlocal, shininess, u, pdf);
}

// Sampling the specular BRDF based on Phong, includes normalization term
float3 sampleSpecularPhong(float3 Vlocal, float alpha, float alphaSquared, float3 specularF0, float2 u, OUT_PARAMETER(float3) weight) {
	// First convert roughness to shininess (Phong exponent)
	float shininess = beckmannAlphaToShininess(alpha);

	float pdf;
	float3 LPhong = samplePhong(Vlocal, shininess, u, pdf);

	// Sampled LPhong is in "lobe space" - where Phong lobe is centered around +Z axis
	// We need to rotate it in direction of perfect reflection
	float3 Nlocal = float3(0.0f, 0.0f, 1.0f);
	float3 lobeDirection = reflect(-Vlocal, Nlocal);
	float3 Llocal = rotatePoint(getRotationFromZAxis(lobeDirection), LPhong);

	// Calculate the weight of the sample
	float3 Rlocal = reflect(-Llocal, Nlocal);
	float NdotL = max(0.00001f, dot(Nlocal, Llocal));
	weight = max(float3(0.0f, 0.0f, 0.0f), specularF0 * NdotL);

	// Unoptimized formula was:
	//weight = specularF0 * (phongNormalizationTerm(shininess) * pow(max(0.0f, dot(Rlocal, Vlocal)), shininess) * NdotL) / pdf;

	return Llocal;
}

// -------------------------------------------------------------------------
//    Oren-Nayar
// -------------------------------------------------------------------------

// Based on Oren-Nayar's qualitative model
// Source: "Generalization of Lambert's Reflectance Model" by Oren & Nayar
float orenNayar(BrdfData data) {
	// Oren-Nayar roughness (sigma) is in radians - use conversion from Beckmann roughness here
	float sigma = beckmannAlphaToOrenNayarRoughness(data.alpha);

	float thetaV = acos(data.NdotV);
	float thetaL = acos(data.NdotL);

	float alpha = max(thetaV, thetaL);
	float beta = min(thetaV, thetaL);

	// Calculate cosine of azimuth angles difference - by projecting L and V onto plane defined by N. Assume L, V, N are normalized.
	float3 l = data.L - data.NdotL * data.N;
	float3 v = data.V - data.NdotV * data.N;
	float cosPhiDifference = dot(normalize(v), normalize(l));

	float sigma2 = sigma * sigma;
	float A = 1.0f - 0.5f * (sigma2 / (sigma2 + 0.33f));
	float B = 0.45f * (sigma2 / (sigma2 + 0.09f));

	return (A + B * max(0.0f, cosPhiDifference) * sin(alpha) * tan(beta));
}

float3 evalOrenNayar(const BrdfData data) {
	return data.diffuseReflectance * (orenNayar(data) * ONE_OVER_PI * data.NdotL);
}

// -------------------------------------------------------------------------
//    Disney
// -------------------------------------------------------------------------

// Disney's diffuse term
// Source "Physically-Based Shading at Disney" by Burley
float disneyDiffuse(const BrdfData data) {
	float FD90MinusOne = 2.0f * data.roughness * data.LdotH * data.LdotH - 0.5f;

	float FDL = 1.0f + (FD90MinusOne * pow(1.0f - data.NdotL, 5.0f));
	float FDV = 1.0F + (FD90MinusOne * pow(1.0f - data.NdotV, 5.0f));

	return FDL * FDV;
}

float3 evalDisneyDiffuse(const BrdfData data) {
	return data.diffuseReflectance * (disneyDiffuse(data) * ONE_OVER_PI * data.NdotL);
}

// Frostbite's version of Disney diffuse with energy normalization.
// Source: "Moving Frostbite to Physically Based Rendering" by Lagarde & de Rousiers
float frostbiteDisneyDiffuse(const BrdfData data) {
	float energyBias = 0.5f * data.roughness;
	float energyFactor = lerp(1.0f, 1.0f / 1.51f, data.roughness);

	float FD90MinusOne = energyBias + 2.0 * data.LdotH * data.LdotH * data.roughness - 1.0f;

	float FDL = 1.0f + (FD90MinusOne * pow(1.0f - data.NdotL, 5.0f));
	float FDV = 1.0f + (FD90MinusOne * pow(1.0f - data.NdotV, 5.0f));

	return FDL * FDV * energyFactor;
}

float3 evalFrostbiteDisneyDiffuse(const BrdfData data) {
	return data.diffuseReflectance * (frostbiteDisneyDiffuse(data) * ONE_OVER_PI * data.NdotL);
}

// -------------------------------------------------------------------------
//    Smith G term
// -------------------------------------------------------------------------

// Function to calculate 'a' parameter for lambda functions needed in Smith G term
// This is a version for shape invariant (isotropic) NDFs
// Note: make sure NdotS is not negative
float Smith_G_a(float alpha, float NdotS) {
	return NdotS / (max(0.00001f, alpha) * sqrt(1.0f - min(0.99999f, NdotS * NdotS)));
}

// Lambda function for Smith G term derived for GGX distribution
float Smith_G_Lambda_GGX(float a) {
	return (-1.0f + sqrt(1.0f + (1.0f / (a * a)))) * 0.5f;
}

// Lambda function for Smith G term derived for Beckmann distribution
// This is Walter's rational approximation (avoids evaluating of error function)
// Source: "Real-time Rendering", 4th edition, p.339 by Akenine-Moller et al.
// Note that this formulation is slightly optimized and different from Walter's
float Smith_G_Lambda_Beckmann_Walter(float a) {
	if (a < 1.6f) {
		return (1.0f - (1.259f - 0.396f * a) * a) / ((3.535f + 2.181f * a) * a);
		//return ((1.0f + (2.276f + 2.577f * a) * a) / ((3.535f + 2.181f * a) * a)) - 1.0f; //< Walter's original
	} else {
		return 0.0f;
	}
}

// Smith G1 term (masking function)
// This non-optimized version uses NDF specific lambda function (G_Lambda) resolved bia macro based on selected NDF
float Smith_G1_General(float a) {
	return 1.0f / (1.0f + Smith_G_Lambda(a));
}

// Smith G1 term (masking function) optimized for GGX distribution (by substituting G_Lambda_GGX into G1)
float Smith_G1_GGX(float a) {
	float a2 = a * a;
	return 2.0f / (sqrt((a2 + 1.0f) / a2) + 1.0f);
}

// Smith G1 term (masking function) further optimized for GGX distribution (by substituting G_a into G1_GGX)
float Smith_G1_GGX(float alpha, float NdotS, float alphaSquared, float NdotSSquared) {
	return 2.0f / (sqrt(((alphaSquared * (1.0f - NdotSSquared)) + NdotSSquared) / NdotSSquared) + 1.0f);
}

// Smith G1 term (masking function) optimized for Beckmann distribution (by substituting G_Lambda_Beckmann_Walter into G1)
// Source: "Microfacet Models for Refraction through Rough Surfaces" by Walter et al.
float Smith_G1_Beckmann_Walter(float a) {
	if (a < 1.6f) {
		return ((3.535f + 2.181f * a) * a) / (1.0f + (2.276f + 2.577f * a) * a);
	} else {
		return 1.0f;
	}
}

float Smith_G1_Beckmann_Walter(float alpha, float NdotS, float alphaSquared, float NdotSSquared) {
	return Smith_G1_Beckmann_Walter(Smith_G_a(alpha, NdotS));
}

// Smith G2 term (masking-shadowing function)
// Separable version assuming independent (uncorrelated) masking and shadowing, uses G1 functions for selected NDF
float Smith_G2_Separable(float alpha, float NdotL, float NdotV) {
	float aL = Smith_G_a(alpha, NdotL);
	float aV = Smith_G_a(alpha, NdotV);
	return Smith_G1(aL) * Smith_G1(aV);
}

// Smith G2 term (masking-shadowing function)
// Height correlated version - non-optimized, uses G_Lambda functions for selected NDF
float Smith_G2_Height_Correlated(float alpha, float NdotL, float NdotV) {
	float aL = Smith_G_a(alpha, NdotL);
	float aV = Smith_G_a(alpha, NdotV);
	return 1.0f / (1.0f + Smith_G_Lambda(aL) + Smith_G_Lambda(aV));
}

// Smith G2 term (masking-shadowing function) for GGX distribution
// Separable version assuming independent (uncorrelated) masking and shadowing - optimized by substituting G_Lambda for G_Lambda_GGX and
// dividing by (4 * NdotL * NdotV) to cancel out these terms in specular BRDF denominator
// Source: "Moving Frostbite to Physically Based Rendering" by Lagarde & de Rousiers
// Note that returned value is G2 / (4 * NdotL * NdotV) and therefore includes division by specular BRDF denominator
float Smith_G2_Separable_GGX_Lagarde(float alphaSquared, float NdotL, float NdotV) {
	float a = NdotV + sqrt(alphaSquared + NdotV * (NdotV - alphaSquared * NdotV));
	float b = NdotL + sqrt(alphaSquared + NdotL * (NdotL - alphaSquared * NdotL));
	return 1.0f / (a * b);
}

// Smith G2 term (masking-shadowing function) for GGX distribution
// Height correlated version - optimized by substituting G_Lambda for G_Lambda_GGX and dividing by (4 * NdotL * NdotV) to cancel out
// the terms in specular BRDF denominator
// Source: "Moving Frostbite to Physically Based Rendering" by Lagarde & de Rousiers
// Note that returned value is G2 / (4 * NdotL * NdotV) and therefore includes division by specular BRDF denominator
float Smith_G2_Height_Correlated_GGX_Lagarde(float alphaSquared, float NdotL, float NdotV) {
	float a = NdotV * sqrt(alphaSquared + NdotL * (NdotL - alphaSquared * NdotL));
	float b = NdotL * sqrt(alphaSquared + NdotV * (NdotV - alphaSquared * NdotV));
	return 0.5f / (a + b);
}

// Smith G2 term (masking-shadowing function) for GGX distribution
// Height correlated version - approximation by Hammon
// Source: "PBR Diffuse Lighting for GGX + Smith Microsurfaces", slide 84 by Hammon
// Note that returned value is G2 / (4 * NdotL * NdotV) and therefore includes division by specular BRDF denominator
float Smith_G2_Height_Correlated_GGX_Hammon(float alpha, float NdotL, float NdotV) {
	return 0.5f / (lerp(2.0f * NdotL * NdotV, NdotL + NdotV, alpha));
}

// A fraction G2/G1 where G2 is height correlated can be expressed using only G1 terms
// Source: "Implementing a Simple Anisotropic Rough Diffuse Material with Stochastic Evaluation", Appendix A by Heitz & Dupuy
float Smith_G2_Over_G1_Height_Correlated(float alpha, float alphaSquared, float NdotL, float NdotV) {
	float G1V = Smith_G1(alpha, NdotV, alphaSquared, NdotV * NdotV);
	float G1L = Smith_G1(alpha, NdotL, alphaSquared, NdotL * NdotL);
	return G1L / (G1V + G1L - G1V * G1L);
}

// Evaluates G2 for selected configuration (GGX/Beckmann, optimized/non-optimized, separable/height-correlated)
// Note that some paths aren't optimized too much...
// Also note that when USE_OPTIMIZED_G2 is specified, returned value will be: G2 / (4 * NdotL * NdotV) if GG-X is selected
float Smith_G2(float alpha, float alphaSquared, float NdotL, float NdotV) {
#if USE_OPTIMIZED_G2 && (MICROFACET_DISTRIBUTION == GGX)
#if USE_HEIGHT_CORRELATED_G2
#define G2_DIVIDED_BY_DENOMINATOR 1
	return Smith_G2_Height_Correlated_GGX_Lagarde(alphaSquared, NdotL, NdotV);
#else
#define G2_DIVIDED_BY_DENOMINATOR 1
	return Smith_G2_Separable_GGX_Lagarde(alphaSquared, NdotL, NdotV);
#endif
#else
#if USE_HEIGHT_CORRELATED_G2
	return Smith_G2_Height_Correlated(alpha, NdotL, NdotV);
#else
	return Smith_G2_Separable(alpha, NdotL, NdotV);
#endif
#endif
}

// -------------------------------------------------------------------------
//    Normal distribution functions
// -------------------------------------------------------------------------

float Beckmann_D(float alphaSquared, float NdotH) {
	float cos2Theta = NdotH * NdotH;
	float numerator = exp((cos2Theta - 1.0f) / (alphaSquared * cos2Theta));
	float denominator = PI * alphaSquared * cos2Theta * cos2Theta;
	return numerator / denominator;
}

float GGX_D(float alphaSquared, float NdotH) {
	float b = ((alphaSquared - 1.0f) * NdotH * NdotH + 1.0f);
	return alphaSquared / (PI * b * b);
}

// -------------------------------------------------------------------------
//    Microfacet model
// -------------------------------------------------------------------------

// Samples a microfacet normal for the GGX distribution using VNDF method.
// Source: "Sampling the GGX Distribution of Visible Normals" by Heitz
// See also https://hal.inria.fr/hal-00996995v1/document and http://jcgt.org/published/0007/04/01/
// Random variables 'u' must be in <0;1) interval
// PDF is 'G1(NdotV) * D'
float3 sampleGGXVNDF(float3 Ve, float2 alpha2D, float2 u) {
	// Section 3.2: transforming the view direction to the hemisphere configuration
	float3 Vh = normalize(float3(alpha2D.x * Ve.x, alpha2D.y * Ve.y, Ve.z));

	// Section 4.1: orthonormal basis (with special case if cross product is zero)
	float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	float3 T1 = lensq > 0.0f ? float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
	float3 T2 = cross(Vh, T1);

	// Section 4.2: parameterization of the projected area
	float r = sqrt(u.x);
	float phi = TWO_PI * u.y;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5f * (1.0f + Vh.z);
	t2 = lerp(sqrt(1.0f - t1 * t1), t2, s);

	// Section 4.3: reprojection onto hemisphere
	float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

	// Section 3.4: transforming the normal back to the ellipsoid configuration
	return normalize(float3(alpha2D.x * Nh.x, alpha2D.y * Nh.y, max(0.0f, Nh.z)));
}

// PDF of sampling a reflection vector L using 'sampleGGXVNDF'.
// Note that PDF of sampling given microfacet normal is (G1 * D) when vectors are in local space (in the hemisphere around shading normal).
// Remaining terms (1.0f / (4.0f * NdotV)) are specific for reflection case, and come from multiplying PDF by jacobian of reflection operator
float sampleGGXVNDFReflectionPdf(float alpha, float alphaSquared, float NdotH, float NdotV, float LdotH) {
	NdotH = max(0.00001f, NdotH);
	NdotV = max(0.00001f, NdotV);
	return (GGX_D(max(0.00001f, alphaSquared), NdotH) * Smith_G1_GGX(alpha, NdotV, alphaSquared, NdotV * NdotV)) / (4.0f * NdotV);
}

// "Walter's trick" is an adjustment of alpha value for Walter's sampling to reduce maximal weight of sample to about 4
// Source: "Microfacet Models for Refraction through Rough Surfaces" by Walter et al., page 8
float waltersTrick(float alpha, float NdotV) {
	return (1.2f - 0.2f * sqrt(abs(NdotV))) * alpha;
}

// PDF of sampling a reflection vector L using 'sampleBeckmannWalter'.
// Note that PDF of sampling given microfacet normal is (D * NdotH). Remaining terms (1.0f / (4.0f * LdotH)) are specific for
// reflection case, and come from multiplying PDF by jacobian of reflection operator
float sampleBeckmannWalterReflectionPdf(float alpha, float alphaSquared, float NdotH, float NdotV, float LdotH) {
	NdotH = max(0.00001f, NdotH);
	LdotH = max(0.00001f, LdotH);
	return Beckmann_D(max(0.00001f, alphaSquared), NdotH) * NdotH / (4.0f * LdotH);
}

// Samples a microfacet normal for the Beckmann distribution using walter's method.
// Source: "Microfacet Models for Refraction through Rough Surfaces" by Walter et al.
// PDF is 'D * NdotH'
float3 sampleBeckmannWalter(float3 Vlocal, float2 alpha2D, float2 u) {
	float alpha = dot(alpha2D, float2(0.5f, 0.5f));

	// Equations (28) and (29) from Walter's paper for Beckmann distribution
	float tanThetaSquared = -(alpha * alpha) * log(1.0f - u.x);
	float phi = TWO_PI * u.y;

	// Calculate cosTheta and sinTheta needed for conversion to H vector
	float cosTheta = rsqrt(1.0f + tanThetaSquared);
	float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

	// Convert sampled spherical coordinates to H vector
	return normalize(float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta));
}

// Weight for the reflection ray sampled from GGX distribution using VNDF method
float specularSampleWeightGGXVNDF(float alpha, float alphaSquared, float NdotL, float NdotV, float HdotL, float NdotH) {
#if USE_HEIGHT_CORRELATED_G2
	return Smith_G2_Over_G1_Height_Correlated(alpha, alphaSquared, NdotL, NdotV);
#else
	return Smith_G1_GGX(alpha, NdotL, alphaSquared, NdotL * NdotL);
#endif
}

// Weight for the reflection ray sampled from Beckmann distribution using Walter's method
float specularSampleWeightBeckmannWalter(float alpha, float alphaSquared, float NdotL, float NdotV, float HdotL, float NdotH) {
	return (HdotL * Smith_G2(alpha, alphaSquared, NdotL, NdotV)) / (NdotV * NdotH);
}

// VAL: multiple-scattering compensation from a table (D118), in place of NVIDIA's linear fit of Ems, which put a
// white furnace at F0 = 1 up to 13 % over 1 at normal incidence (roughness 0.5 to 0.85), 16 % over at grazing on
// the roughest and 11 % under at grazing near roughness 0.2.
// E(alpha, mu), the directional albedo of the lobe above at F = 1 (GGX D, height-correlated Smith G2), integrated
// offline over its VNDF by tools/shaders/ggx_albedo_table.py: 32 rows of perceptual roughness sqrt(alpha) = 0 .. 1
// by 32 columns of sqrt(mu) = 0 .. 1. Read bilinearly, it keeps the white furnace within 0.6 % of 1 for mu >= 0.05
// and about 1 % for mu >= 0.01; below that, on near-mirror silhouettes, E dips along mu ~ alpha faster than the grid.
#if !USE_HEIGHT_CORRELATED_G2
#error GGX_ALBEDO integrates the height-correlated G2, regenerate it with tools/shaders/ggx_albedo_table.py
#endif
const float GGX_ALBEDO[1024] = float[](
		1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
		0.9981f, 0.8921f, 0.9646f, 0.9928f, 0.9983f, 0.9996f, 0.9998f, 0.9999f, 0.9999f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
		0.9995f, 0.9526f, 0.8921f, 0.9223f, 0.9646f, 0.9847f, 0.9928f, 0.9962f, 0.9983f, 0.9993f, 0.9996f, 0.9997f, 0.9998f, 0.9999f, 0.9999f, 0.9999f, 0.9999f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
		0.9998f, 0.9775f, 0.9251f, 0.8921f, 0.9060f, 0.9387f, 0.9646f, 0.9799f, 0.9883f, 0.9928f, 0.9955f, 0.9968f, 0.9983f, 0.9992f, 0.9995f, 0.9996f, 0.9997f, 0.9998f, 0.9998f, 0.9999f, 0.9999f, 0.9999f, 0.9999f, 0.9999f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
		0.9999f, 0.9874f, 0.9526f, 0.9128f, 0.8920f, 0.8992f, 0.9222f, 0.9462f, 0.9646f, 0.9769f, 0.9847f, 0.9897f, 0.9928f, 0.9950f, 0.9962f, 0.9972f, 0.9983f, 0.9991f, 0.9994f, 0.9995f, 0.9997f, 0.9997f, 0.9998f, 0.9998f, 0.9999f, 0.9999f, 0.9999f, 0.9999f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
		0.9999f, 0.9919f, 0.9683f, 0.9359f, 0.9064f, 0.8919f, 0.8958f, 0.9121f, 0.9321f, 0.9503f, 0.9645f, 0.9748f, 0.9820f, 0.9869f, 0.9904f, 0.9928f, 0.9947f, 0.9958f, 0.9966f, 0.9974f, 0.9984f, 0.9991f, 0.9994f, 0.9996f, 0.9997f, 0.9997f, 0.9998f, 0.9999f, 0.9999f, 0.9999f, 1.0000f, 1.0000f,
		0.9999f, 0.9944f, 0.9774f, 0.9524f, 0.9248f, 0.9024f, 0.8917f, 0.8938f, 0.9056f, 0.9218f, 0.9383f, 0.9528f, 0.9643f, 0.9732f, 0.9797f, 0.9846f, 0.9882f, 0.9908f, 0.9928f, 0.9945f, 0.9955f, 0.9962f, 0.9969f, 0.9975f, 0.9984f, 0.9992f, 0.9995f, 0.9997f, 0.9998f, 0.9999f, 0.9999f, 1.0000f,
		1.0000f, 0.9958f, 0.9832f, 0.9634f, 0.9399f, 0.9168f, 0.8992f, 0.8907f, 0.8919f, 0.9006f, 0.9138f, 0.9283f, 0.9421f, 0.9540f, 0.9639f, 0.9716f, 0.9777f, 0.9824f, 0.9859f, 0.9888f, 0.9909f, 0.9926f, 0.9941f, 0.9950f, 0.9957f, 0.9963f, 0.9968f, 0.9974f, 0.9984f, 0.9993f, 0.9997f, 0.9999f,
		1.0000f, 0.9967f, 0.9869f, 0.9709f, 0.9509f, 0.9294f, 0.9097f, 0.8951f, 0.8876f, 0.8875f, 0.8940f, 0.9045f, 0.9169f, 0.9295f, 0.9411f, 0.9512f, 0.9597f, 0.9666f, 0.9722f, 0.9766f, 0.9801f, 0.9829f, 0.9852f, 0.9869f, 0.9883f, 0.9896f, 0.9905f, 0.9910f, 0.9914f, 0.9917f, 0.9919f, 0.9921f,
		1.0000f, 0.9973f, 0.9892f, 0.9758f, 0.9586f, 0.9391f, 0.9195f, 0.9033f, 0.8916f, 0.8857f, 0.8858f, 0.8909f, 0.8995f, 0.9102f, 0.9216f, 0.9327f, 0.9427f, 0.9516f, 0.9592f, 0.9655f, 0.9707f, 0.9750f, 0.9785f, 0.9813f, 0.9837f, 0.9857f, 0.9871f, 0.9885f, 0.9899f, 0.9908f, 0.9914f, 0.9918f,
		1.0000f, 0.9977f, 0.9908f, 0.9791f, 0.9639f, 0.9464f, 0.9289f, 0.9123f, 0.8983f, 0.8883f, 0.8831f, 0.8827f, 0.8865f, 0.8936f, 0.9027f, 0.9128f, 0.9231f, 0.9329f, 0.9418f, 0.9497f, 0.9566f, 0.9624f, 0.9674f, 0.9716f, 0.9750f, 0.9779f, 0.9803f, 0.9824f, 0.9841f, 0.9853f, 0.9864f, 0.9882f,
		1.0000f, 0.9979f, 0.9917f, 0.9814f, 0.9673f, 0.9522f, 0.9361f, 0.9197f, 0.9046f, 0.8919f, 0.8830f, 0.8784f, 0.8777f, 0.8807f, 0.8866f, 0.8945f, 0.9036f, 0.9130f, 0.9224f, 0.9312f, 0.9393f, 0.9465f, 0.9528f, 0.9583f, 0.9630f, 0.9670f, 0.9704f, 0.9734f, 0.9759f, 0.9779f, 0.9800f, 0.9816f,
		1.0000f, 0.9981f, 0.9923f, 0.9826f, 0.9698f, 0.9562f, 0.9410f, 0.9250f, 0.9099f, 0.8965f, 0.8854f, 0.8775f, 0.8729f, 0.8718f, 0.8739f, 0.8787f, 0.8855f, 0.8934f, 0.9021f, 0.9108f, 0.9194f, 0.9274f, 0.9348f, 0.9415f, 0.9474f, 0.9525f, 0.9571f, 0.9610f, 0.9644f, 0.9673f, 0.9700f, 0.9721f,
		1.0000f, 0.9981f, 0.9925f, 0.9832f, 0.9716f, 0.9588f, 0.9441f, 0.9289f, 0.9141f, 0.8999f, 0.8874f, 0.8774f, 0.8702f, 0.8659f, 0.8646f, 0.8660f, 0.8699f, 0.8757f, 0.8827f, 0.8905f, 0.8987f, 0.9068f, 0.9147f, 0.9222f, 0.9290f, 0.9353f, 0.9409f, 0.9459f, 0.9503f, 0.9541f, 0.9576f, 0.9606f,
		1.0000f, 0.9981f, 0.9925f, 0.9832f, 0.9726f, 0.9600f, 0.9458f, 0.9313f, 0.9164f, 0.9022f, 0.8889f, 0.8774f, 0.8681f, 0.8613f, 0.8571f, 0.8555f, 0.8565f, 0.8596f, 0.8645f, 0.8706f, 0.8776f, 0.8852f, 0.8929f, 0.9005f, 0.9079f, 0.9148f, 0.9213f, 0.9272f, 0.9326f, 0.9374f, 0.9417f, 0.9457f,
		1.0000f, 0.9981f, 0.9923f, 0.9830f, 0.9728f, 0.9601f, 0.9466f, 0.9321f, 0.9176f, 0.9029f, 0.8893f, 0.8767f, 0.8659f, 0.8571f, 0.8504f, 0.8463f, 0.8445f, 0.8449f, 0.8473f, 0.8512f, 0.8564f, 0.8627f, 0.8696f, 0.8768f, 0.8841f, 0.8913f, 0.8983f, 0.9048f, 0.9110f, 0.9167f, 0.9219f, 0.9266f,
		1.0000f, 0.9980f, 0.9919f, 0.9828f, 0.9724f, 0.9596f, 0.9462f, 0.9318f, 0.9171f, 0.9025f, 0.8883f, 0.8749f, 0.8629f, 0.8525f, 0.8440f, 0.8376f, 0.8332f, 0.8311f, 0.8310f, 0.8326f, 0.8358f, 0.8403f, 0.8457f, 0.8519f, 0.8586f, 0.8656f, 0.8725f, 0.8794f, 0.8860f, 0.8923f, 0.8982f, 0.9036f,
		1.0000f, 0.9979f, 0.9914f, 0.9822f, 0.9715f, 0.9587f, 0.9447f, 0.9303f, 0.9155f, 0.9006f, 0.8859f, 0.8719f, 0.8589f, 0.8472f, 0.8371f, 0.8287f, 0.8223f, 0.8177f, 0.8152f, 0.8145f, 0.8154f, 0.8178f, 0.8215f, 0.8263f, 0.8318f, 0.8379f, 0.8444f, 0.8510f, 0.8577f, 0.8643f, 0.8706f, 0.8767f,
		1.0000f, 0.9977f, 0.9907f, 0.9814f, 0.9701f, 0.9571f, 0.9428f, 0.9279f, 0.9127f, 0.8973f, 0.8822f, 0.8675f, 0.8537f, 0.8409f, 0.8293f, 0.8193f, 0.8110f, 0.8044f, 0.7996f, 0.7965f, 0.7952f, 0.7954f, 0.7970f, 0.7999f, 0.8038f, 0.8086f, 0.8140f, 0.8199f, 0.8262f, 0.8326f, 0.8390f, 0.8454f,
		1.0000f, 0.9975f, 0.9900f, 0.9803f, 0.9683f, 0.9549f, 0.9400f, 0.9247f, 0.9090f, 0.8930f, 0.8772f, 0.8618f, 0.8471f, 0.8332f, 0.8205f, 0.8090f, 0.7990f, 0.7905f, 0.7836f, 0.7784f, 0.7748f, 0.7728f, 0.7723f, 0.7731f, 0.7751f, 0.7781f, 0.7821f, 0.7868f, 0.7921f, 0.7978f, 0.8037f, 0.8099f,
		1.0000f, 0.9973f, 0.9891f, 0.9791f, 0.9664f, 0.9522f, 0.9367f, 0.9207f, 0.9044f, 0.8877f, 0.8711f, 0.8549f, 0.8393f, 0.8244f, 0.8105f, 0.7976f, 0.7861f, 0.7759f, 0.7671f, 0.7599f, 0.7541f, 0.7499f, 0.7472f, 0.7458f, 0.7457f, 0.7468f, 0.7489f, 0.7520f, 0.7558f, 0.7603f, 0.7654f, 0.7708f,
		1.0000f, 0.9970f, 0.9881f, 0.9776f, 0.9643f, 0.9493f, 0.9330f, 0.9160f, 0.8989f, 0.8814f, 0.8639f, 0.8468f, 0.8302f, 0.8143f, 0.7992f, 0.7851f, 0.7720f, 0.7603f, 0.7498f, 0.7406f, 0.7329f, 0.7266f, 0.7216f, 0.7181f, 0.7158f, 0.7147f, 0.7148f, 0.7159f, 0.7180f, 0.7209f, 0.7245f, 0.7286f,
		1.0000f, 0.9967f, 0.9872f, 0.9760f, 0.9618f, 0.9460f, 0.9287f, 0.9107f, 0.8927f, 0.8742f, 0.8558f, 0.8377f, 0.8201f, 0.8030f, 0.7867f, 0.7713f, 0.7569f, 0.7437f, 0.7315f, 0.7206f, 0.7110f, 0.7027f, 0.6957f, 0.6899f, 0.6854f, 0.6822f, 0.6801f, 0.6791f, 0.6791f, 0.6800f, 0.6817f, 0.6843f,
		1.0000f, 0.9964f, 0.9863f, 0.9741f, 0.9591f, 0.9422f, 0.9241f, 0.9049f, 0.8858f, 0.8664f, 0.8468f, 0.8276f, 0.8089f, 0.7907f, 0.7732f, 0.7565f, 0.7407f, 0.7260f, 0.7123f, 0.6998f, 0.6884f, 0.6782f, 0.6692f, 0.6614f, 0.6547f, 0.6493f, 0.6450f, 0.6417f, 0.6395f, 0.6383f, 0.6379f, 0.6385f,
		1.0000f, 0.9961f, 0.9854f, 0.9721f, 0.9561f, 0.9383f, 0.9190f, 0.8988f, 0.8784f, 0.8578f, 0.8370f, 0.8167f, 0.7968f, 0.7773f, 0.7586f, 0.7406f, 0.7235f, 0.7074f, 0.6922f, 0.6781f, 0.6650f, 0.6530f, 0.6422f, 0.6324f, 0.6238f, 0.6162f, 0.6097f, 0.6043f, 0.5999f, 0.5964f, 0.5938f, 0.5921f,
		1.0000f, 0.9958f, 0.9845f, 0.9701f, 0.9531f, 0.9341f, 0.9136f, 0.8922f, 0.8704f, 0.8486f, 0.8267f, 0.8050f, 0.7838f, 0.7631f, 0.7431f, 0.7238f, 0.7054f, 0.6878f, 0.6712f, 0.6556f, 0.6410f, 0.6274f, 0.6148f, 0.6032f, 0.5927f, 0.5831f, 0.5746f, 0.5671f, 0.5605f, 0.5548f, 0.5500f, 0.5460f,
		1.0000f, 0.9954f, 0.9834f, 0.9681f, 0.9499f, 0.9296f, 0.9079f, 0.8852f, 0.8620f, 0.8389f, 0.8157f, 0.7927f, 0.7701f, 0.7482f, 0.7268f, 0.7062f, 0.6865f, 0.6675f, 0.6496f, 0.6325f, 0.6164f, 0.6012f, 0.5870f, 0.5738f, 0.5616f, 0.5502f, 0.5398f, 0.5303f, 0.5217f, 0.5139f, 0.5070f, 0.5009f,
		1.0000f, 0.9950f, 0.9824f, 0.9659f, 0.9466f, 0.9250f, 0.9019f, 0.8779f, 0.8532f, 0.8287f, 0.8042f, 0.7798f, 0.7559f, 0.7326f, 0.7099f, 0.6880f, 0.6669f, 0.6466f, 0.6273f, 0.6089f, 0.5914f, 0.5748f, 0.5591f, 0.5444f, 0.5306f, 0.5176f, 0.5056f, 0.4943f, 0.4839f, 0.4743f, 0.4655f, 0.4574f,
		0.9999f, 0.9946f, 0.9812f, 0.9637f, 0.9430f, 0.9202f, 0.8957f, 0.8702f, 0.8442f, 0.8182f, 0.7922f, 0.7664f, 0.7411f, 0.7164f, 0.6924f, 0.6692f, 0.6468f, 0.6252f, 0.6046f, 0.5849f, 0.5661f, 0.5482f, 0.5312f, 0.5152f, 0.4999f, 0.4856f, 0.4721f, 0.4593f, 0.4474f, 0.4362f, 0.4257f, 0.4159f,
		0.9999f, 0.9942f, 0.9800f, 0.9614f, 0.9394f, 0.9151f, 0.8893f, 0.8623f, 0.8348f, 0.8072f, 0.7798f, 0.7526f, 0.7259f, 0.6999f, 0.6745f, 0.6500f, 0.6263f, 0.6035f, 0.5816f, 0.5607f, 0.5407f, 0.5216f, 0.5035f, 0.4862f, 0.4698f, 0.4542f, 0.4395f, 0.4255f, 0.4123f, 0.3998f, 0.3880f, 0.3769f,
		0.9999f, 0.9938f, 0.9787f, 0.9589f, 0.9357f, 0.9100f, 0.8826f, 0.8542f, 0.8252f, 0.7960f, 0.7672f, 0.7385f, 0.7104f, 0.6830f, 0.6563f, 0.6305f, 0.6056f, 0.5816f, 0.5586f, 0.5365f, 0.5154f, 0.4953f, 0.4761f, 0.4578f, 0.4404f, 0.4238f, 0.4080f, 0.3931f, 0.3789f, 0.3654f, 0.3526f, 0.3405f,
		0.9999f, 0.9933f, 0.9774f, 0.9564f, 0.9318f, 0.9046f, 0.8758f, 0.8458f, 0.8153f, 0.7846f, 0.7542f, 0.7242f, 0.6946f, 0.6659f, 0.6379f, 0.6108f, 0.5847f, 0.5596f, 0.5355f, 0.5124f, 0.4903f, 0.4692f, 0.4491f, 0.4300f, 0.4117f, 0.3944f, 0.3779f, 0.3622f, 0.3473f, 0.3331f, 0.3196f, 0.3069f);

float ggxAlbedo(float alpha, float NdV) {
	float x = sqrt(saturate(alpha)) * 31.0f;
	float y = sqrt(saturate(NdV)) * 31.0f;
	int i = min(int(x), 30);
	int j = min(int(y), 30);
	float fx = x - float(i);
	float fy = y - float(j);
	int k = i * 32 + j;
	float a = lerp(GGX_ALBEDO[k], GGX_ALBEDO[k + 1], fy);
	float b = lerp(GGX_ALBEDO[k + 32], GGX_ALBEDO[k + 33], fy);
	return lerp(a, b, fx);
}

// VAL: Turquin's compensation, 1 + F0 (1 - E) / E, which is 1 / E at F0 = 1 and so exact in the white furnace.
float3 evalMultiScatterApproximation(float alpha, float NdV, float3 F0) {
	float E = ggxAlbedo(alpha, NdV);
	return 1.0f + F0 * ((1.0f - E) / E);
}

// VAL: anisotropic GGX (D118 MM1), the forms of Falcor's AnisotropicGGX.slang and Heitz's "Understanding the
// Masking-Shadowing Function in Microfacet-Based BRDFs". Vectors are in the lobe's (T, B, N) frame, alpha2D its
// roughness along T and B; with alpha2D = (alpha, alpha) each reduces to the isotropic function it stands beside.

// The NDF, each alpha floored as GGX_D floors alpha squared.
float GGX_D_Anisotropic(float2 alpha2D, float3 H) {
	float2 a = max(alpha2D, float2(0.0031623f, 0.0031623f));
	float3 h = float3(H.x / a.x, H.y / a.y, H.z);
	float d = dot(h, h);
	return 1.0f / (PI * a.x * a.y * d * d);
}

// sqrt(ax^2 Sx^2 + ay^2 Sy^2 + Sz^2), the length both Smith terms take: Lambda(S) = (length / Sz - 1) / 2.
float Smith_Length_GGX_Anisotropic(float2 alpha2D, float3 S) {
	return length(float3(alpha2D.x * S.x, alpha2D.y * S.y, S.z));
}

// Smith G1 = 1 / (1 + Lambda(S)), S.z clamped as the isotropic callers clamp NdotS.
float Smith_G1_GGX_Anisotropic(float2 alpha2D, float3 S) {
	float z = max(0.00001f, S.z);
	return 2.0f * z / (z + Smith_Length_GGX_Anisotropic(alpha2D, float3(S.xy, z)));
}

// Height-correlated G2 / (4 * NdotL * NdotV), the anisotropic Smith_G2_Height_Correlated_GGX_Lagarde (Godot's
// V_GGX_anisotropic), with L.z and V.z the clamped NdotL and NdotV.
float Smith_G2_Height_Correlated_GGX_Anisotropic(float2 alpha2D, float3 L, float3 V) {
	float a = V.z * Smith_Length_GGX_Anisotropic(alpha2D, L);
	float b = L.z * Smith_Length_GGX_Anisotropic(alpha2D, V);
	return 0.5f / (a + b);
}

// The density of a reflection sampled by sampleGGXVNDF with alpha2D: D * G1(V) / (4 * NdotV), as sampleGGXVNDFReflectionPdf.
float sampleGGXVNDFReflectionPdfAnisotropic(float2 alpha2D, float3 H, float3 V) {
	float NdotV = max(0.00001f, V.z);
	return GGX_D_Anisotropic(alpha2D, float3(H.xy, max(0.00001f, H.z))) * Smith_G1_GGX_Anisotropic(alpha2D, V) / (4.0f * NdotV);
}

// sampleSpecularMicrofacet for the anisotropic lobe: the half vector from the VNDF with alpha2D and the weight
// F * G2 / G1(V) with the multiple-scattering factor, whose roughness is sqrt(ax * ay) = alpha.
float3 sampleSpecularMicrofacetAnisotropic(float3 Vlocal, float alpha, float2 alpha2D, float3 specularF0, float2 u, OUT_PARAMETER(float3) weight) {
	float3 Hlocal = alpha == 0.0f ? float3(0.0f, 0.0f, 1.0f) : sampleGGXVNDF(Vlocal, alpha2D, u);
	float3 Llocal = reflect(-Vlocal, Hlocal);

	float HdotL = max(0.00001f, min(1.0f, dot(Hlocal, Llocal)));
	float NdotV = max(0.00001f, min(1.0f, Vlocal.z));
	float3 F = evalFresnel(specularF0, shadowedF90(specularF0), HdotL);
	float3 ms = evalMultiScatterApproximation(alpha, NdotV, specularF0);

	float G1V = Smith_G1_GGX_Anisotropic(alpha2D, Vlocal);
	float G1L = Smith_G1_GGX_Anisotropic(alpha2D, Llocal);
	weight = ms * F * (G1L / (G1V + G1L - G1V * G1L));

	return Llocal;
}

// Samples a reflection ray from the rough surface using selected microfacet distribution and sampling method
// Resulting weight includes multiplication by cosine (NdotL) term
float3 sampleSpecularMicrofacet(float3 Vlocal, float alpha, float alphaSquared, float3 specularF0, float2 u, OUT_PARAMETER(float3) weight) {
	// Sample a microfacet normal (H) in local space
	float3 Hlocal;
	if (alpha == 0.0f) {
		// Fast path for zero roughness (perfect reflection), also prevents NaNs appearing due to divisions by zeroes
		Hlocal = float3(0.0f, 0.0f, 1.0f);
	} else {
		// For non-zero roughness, this calls VNDF sampling for GG-X distribution or Walter's sampling for Beckmann distribution
		Hlocal = sampleSpecularHalfVector(Vlocal, float2(alpha, alpha), u);
	}

	// Reflect view direction to obtain light vector
	float3 Llocal = reflect(-Vlocal, Hlocal);

	// Note: HdotL is same as HdotV here
	// Clamp dot products here to small value to prevent numerical instability. Assume that rays incident from below the hemisphere have been filtered
	float HdotL = max(0.00001f, min(1.0f, dot(Hlocal, Llocal)));
	const float3 Nlocal = float3(0.0f, 0.0f, 1.0f);
	float NdotL = max(0.00001f, min(1.0f, dot(Nlocal, Llocal)));
	float NdotV = max(0.00001f, min(1.0f, dot(Nlocal, Vlocal)));
	float NdotH = max(0.00001f, min(1.0f, dot(Nlocal, Hlocal)));
	float3 F = evalFresnel(specularF0, shadowedF90(specularF0), HdotL);

#if MICROFACET_DISTRIBUTION == GGX
	float3 ms = evalMultiScatterApproximation(alpha, NdotV, specularF0);
#else
	float3 ms = vec3(1.0f, 1.0f, 1.0f);
#endif

	// Calculate weight of the sample specific for selected sampling method
	// (this is microfacet BRDF divided by PDF of sampling method - notice how most terms cancel out)
	weight = ms * F * specularSampleWeight(alpha, alphaSquared, NdotL, NdotV, HdotL, NdotH);

	return Llocal;
}

// Evaluates microfacet specular BRDF
float3 evalMicrofacet(const BrdfData data) {
	float D = Microfacet_D(max(0.00001f, data.alphaSquared), data.NdotH);
	float G2 = Smith_G2(data.alpha, data.alphaSquared, data.NdotL, data.NdotV);
	//float3 F = evalFresnel(data.specularF0, shadowedF90(data.specularF0), data.VdotH); //< Unused, F is precomputed already

	// VAL: the same multiple-scattering compensation sampleSpecularMicrofacet's weight carries (D118), so light
	// sampling and the BSDF's own sample estimate one specular lobe under MIS.
#if MICROFACET_DISTRIBUTION == GGX
	float3 ms = evalMultiScatterApproximation(data.alpha, data.NdotV, data.specularF0);
#else
	float3 ms = vec3(1.0f, 1.0f, 1.0f);
#endif

	// VAL: the anisotropic lobe (D118 MM1), its G2 already divided by the denominator.
	if (data.anisotropic) {
		float Da = GGX_D_Anisotropic(data.alpha2D, float3(data.HdotTB, data.NdotH));
		float G2a = Smith_G2_Height_Correlated_GGX_Anisotropic(data.alpha2D, float3(data.LdotTB, data.NdotL), float3(data.VdotTB, data.NdotV));
		return ms * data.F * (G2a * Da * data.NdotL);
	}

#if G2_DIVIDED_BY_DENOMINATOR
	return ms * data.F * (G2 * D * data.NdotL);
#else
	return ms * ((data.F * G2 * D) / (4.0f * data.NdotL * data.NdotV)) * data.NdotL;
#endif
}

// -------------------------------------------------------------------------
//    Shading normal clamping
// -------------------------------------------------------------------------

// Clamps a shading normal toward the geometry normal at grazing view angles.
// This addresses a Monte-Carlo path tracing failure mode: when a strong normal
// map perturbs the shading normal far from the geometry normal and the view
// direction grazes the surface, importance-sampled reflection directions
// frequently land below the geometry hemisphere. Such samples must be
// discarded (they would self-intersect the surface), terminating the path
// before it can collect any environment contribution and producing speckled
// black pixels. The classic example is water with high-frequency wave normals
// viewed at distance.
//
// The clamp is a smooth lerp between the shading and geometry normals based
// on the geometry NdotV. Above 'threshold' the shading normal is unchanged.
// Below it, the lerp factor falls linearly to 0 at NdotV == 0 (full geometry
// normal). 'threshold' should be small (e.g. 0.10..0.20); higher values flatten
// detail at glancing angles unnecessarily, lower values still allow rejection
// black pixels for very aggressive normal maps.
//
// The same clamped normal must be used for *all* BRDF math (sampling, eval,
// direct light NEE, denoiser G-buffers) so the indirect lobe and direct
// lighting agree.
float3 clampShadingNormal(float3 shading_n, float3 geometry_n, float3 V, float threshold) {
#if RT_SHADING_NORMAL_CLAMP_ENABLED
	float NdotV_geo = dot(geometry_n, V);
	if (NdotV_geo >= threshold) {
		return shading_n;
	}
	float t = max(0.0f, NdotV_geo) * rcp(max(0.0001f, threshold));
	return normalize(lerp(geometry_n, shading_n, t));
#else
	return shading_n;
#endif
}

// -------------------------------------------------------------------------
//    Below-hemisphere sample recovery
// -------------------------------------------------------------------------
//
// When BRDF importance sampling against a strongly perturbed shading normal
// produces a direction that lands below the geometry plane, the direction
// would self-intersect the surface and cannot be traced as-is.
//
// Recovery strategy: mirror the rejected direction across the geometry plane
// and reuse the original BRDF weight. Cheap, never produces black pixels,
// but is energy-non-conserving (over-bright at grazing angles) -- a
// deliberate trade-off vs. terminating the path.
//
// Set RT_BELOW_HEMISPHERE_RECOVERY_ENABLED to 0 to disable recovery and
// terminate the path on rejection instead. Useful for A/B comparison or
// when investigating energy conservation.

// Mirrors a direction across the geometry plane. No-op if already above.
float3 mirrorDirectionAcrossPlane(float3 dir, float3 geometry_n) {
	float c = dot(dir, geometry_n);
	if (c >= 0.0f) {
		return dir;
	}
	return normalize(dir - 2.0f * c * geometry_n);
}

// Recover a BRDF sample that landed below the geometry plane.
// Returns true and writes the recovered (above-plane) direction to
// 'recovered_dir' when recovery is enabled. Returns false when recovery is
// disabled at compile time (RT_BELOW_HEMISPHERE_RECOVERY_ENABLED == 0); the
// caller should then terminate the path.
bool recoverBelowHemisphereSample(
		float3 rejected_dir,
		float3 geometry_n,
		OUT_PARAMETER(float3) recovered_dir) {
#if RT_BELOW_HEMISPHERE_RECOVERY_ENABLED
	recovered_dir = mirrorDirectionAcrossPlane(rejected_dir, geometry_n);
	return true;
#else
	recovered_dir = rejected_dir;
	return false;
#endif
}

// -------------------------------------------------------------------------
//    VAL: clearcoat (D118 MM2)
// -------------------------------------------------------------------------
// Godot's coat: a GGX layer of IOR 1.5 (F0 0.04, Schlick's Fresnel) about the vertex normal, its alpha
// mix(0.001, 0.1, CLEARCOAT_ROUGHNESS), its strength CLEARCOAT. The base under it takes the F0 the raster remaps
// from air to the coat, and passes what the coat does not reflect in each direction, (1 - c F_c(n_c.V)) (1 - c
// F_c(n_c.L)): nvpro's layering, reciprocal, and on the light sample the raster's (1 - Fr) taken per direction.
// The coat is picked before the base's lobes; brdf_pdf adds its density with that probability.

// Below this alpha (CLEARCOAT_ROUGHNESS under about 0.055) the coat is a mirror: sampled as one, with no light
// sample term, as a zero-roughness specular lobe is. GGX_D's floor (alpha 0.0032) lies under it.
#define CLEARCOAT_DELTA_ALPHA 0.0064f

bool hasClearcoat(MaterialProperties material) {
	return material.clearcoat > 0.0f;
}

// The base's F0 under the coat: the raster's f0_Clear_Coat_To_Surface (Filament's fit of iorTof0(f0ToIor(f0), 1.5)),
// mixed in by the coat's strength.
float3 clearcoatBaseF0(float3 f0, float clearcoat) {
	float3 under = saturate(f0 * (f0 * (0.941892f - 0.263008f * f0) + 0.346479f) - 0.0285998f);
	return lerp(f0, under, clearcoat);
}

// What the coat reflects of light crossing it along S, c F_c. The cosine is unsigned, so a direction below the
// coat's normal (a transmission, a silhouette) still crosses it.
float clearcoatReflectance(MaterialProperties material, float3 S) {
	float c = min(1.0f, abs(dot(material.clearcoatN, S)));
	return material.clearcoat * (0.04f + 0.96f * pow(1.0f - c, 5.0f));
}

// The share of the base that leaves along S through the coat.
float clearcoatAttenuation(MaterialProperties material, float3 S) {
	return 1.0f - clearcoatReflectance(material, S);
}

// The coat's reflection times its NdotL, for the light sample: c F_c(LdotH) D G2 / (4 NdotL NdotV) NdotL. Zero for
// a mirror coat and for L or V below its normal.
float evalClearcoat(MaterialProperties material, float3 L, float3 V) {
	float NdotL = dot(material.clearcoatN, L);
	float NdotV = dot(material.clearcoatN, V);
	if (material.clearcoatAlpha < CLEARCOAT_DELTA_ALPHA || NdotL <= 0.0f || NdotV <= 0.0f) {
		return 0.0f;
	}
	NdotL = min(NdotL, 1.0f);
	NdotV = min(NdotV, 1.0f);
	float3 H = normalize(L + V);
	float NdotH = saturate(dot(material.clearcoatN, H));
	float LdotH = saturate(dot(L, H));
	float alphaSquared = material.clearcoatAlpha * material.clearcoatAlpha;
	float F = material.clearcoat * (0.04f + 0.96f * pow(1.0f - LdotH, 5.0f));
	return F * GGX_D(alphaSquared, NdotH) * Smith_G2_Height_Correlated_GGX_Lagarde(alphaSquared, NdotL, NdotV) * NdotL;
}

// The density with which sampleClearcoat produces L: the coat's VNDF reflection density; zero for a mirror coat.
float clearcoatPdf(MaterialProperties material, float3 L, float3 V) {
	float NdotL = dot(material.clearcoatN, L);
	float NdotV = dot(material.clearcoatN, V);
	if (material.clearcoatAlpha < CLEARCOAT_DELTA_ALPHA || NdotL <= 0.0f || NdotV <= 0.0f) {
		return 0.0f;
	}
	float3 H = normalize(L + V);
	float alphaSquared = material.clearcoatAlpha * material.clearcoatAlpha;
	return sampleGGXVNDFReflectionPdf(material.clearcoatAlpha, alphaSquared, dot(material.clearcoatN, H), min(NdotV, 1.0f), dot(L, H));
}

float sheenAlbedo(float r, float mu); // The sheen's directional albedo, in the sheen block below (D118 MM3).

// The probability with which the scatter in shade_and_bounce picks the coat before the base's lobes: what the coat
// reflects at V against the albedos under it (the luminances brdfSpecularProbability and the transmission weigh,
// and a sheen's about N, D118 MM3), with specularF0 already remapped. 0 without a coat or with V below its normal
// (no random number drawn), 1 over a black base.
float clearcoatProbability(MaterialProperties material, float3 N, float3 V, float3 specularF0, float3 diffuseReflectance) {
	if (!hasClearcoat(material) || dot(material.clearcoatN, V) <= 0.0f) {
		return 0.0f;
	}
	float coat = clearcoatReflectance(material, V);
	float sheen = luminance(material.sheenColor) > 0.0f ? luminance(material.sheenColor) * sheenAlbedo(material.sheenRoughness, dot(N, V)) : 0.0f;
	float base = (1.0f - coat) * (luminance(specularF0) + luminance(diffuseReflectance) + luminance(material.diffuseTransmission) + sheen);
	if (base < 0.0001f) {
		return 1.0f;
	}
	return clamp(coat / (coat + base), 0.01f, 0.99f);
}

// Whether light arriving from L below the shading normal reaches the BSDF: through a thin surface's
// transmission (D103), or off a coat whose own normal L is above. The light samplers reject the rest.
bool brdfTakesLightBelowN(MaterialProperties material, float3 L) {
	return hasHair(material) || hasDiffuseTransmission(material) || (hasClearcoat(material) && dot(material.clearcoatN, L) > 0.0f);
}

// Samples the coat's reflection: its VNDF about the coat's normal, or the mirror direction below
// CLEARCOAT_DELTA_ALPHA. The weight is the reflection over its density, c F_c(LdotH) G2 / G1(V) (c F_c(NdotV) for
// the mirror). Returns false as evalIndirectCombinedBRDF does: on a zero weight, or a direction below the geometry
// plane, which rayDirection still carries for the caller's recovery.
bool sampleClearcoat(float2 u, float3 geometryNormal, float3 V, MaterialProperties material, OUT_PARAMETER(float3) rayDirection, OUT_PARAMETER(float3) sampleWeight) {
	float4 qRotationToZ = getRotationToZAxis(material.clearcoatN);
	float3 Vlocal = rotatePoint(qRotationToZ, V);
	float alpha = material.clearcoatAlpha;
	bool mirror = alpha < CLEARCOAT_DELTA_ALPHA;
	float3 Hlocal = mirror ? float3(0.0f, 0.0f, 1.0f) : sampleGGXVNDF(Vlocal, float2(alpha, alpha), u);
	float3 Llocal = reflect(-Vlocal, Hlocal);

	float HdotL = max(0.00001f, min(1.0f, dot(Hlocal, Llocal)));
	float NdotL = max(0.00001f, min(1.0f, Llocal.z));
	float NdotV = max(0.00001f, min(1.0f, Vlocal.z));
	float F = material.clearcoat * (0.04f + 0.96f * pow(1.0f - HdotL, 5.0f));
	float G2OverG1 = 1.0f;
	if (!mirror) {
		float alphaSquared = alpha * alpha;
		float G1V = Smith_G1_GGX(alpha, NdotV, alphaSquared, NdotV * NdotV);
		float G1L = Smith_G1_GGX(alpha, NdotL, alphaSquared, NdotL * NdotL);
		G2OverG1 = G1L / (G1V + G1L - G1V * G1L);
	}
	sampleWeight = float3(F * G2OverG1, F * G2OverG1, F * G2OverG1);
	rayDirection = normalize(rotatePoint(invertRotation(qRotationToZ), Llocal));

	if (luminance(sampleWeight) == 0.0f) {
		return false;
	}
	return dot(geometryNormal, rayDirection) > 0.0f;
}

// -------------------------------------------------------------------------
//    VAL: sheen (D118 MM3)
// -------------------------------------------------------------------------
// Godot's rim read as a sheen layer: Estevez and Kulla's "Charlie" sheen ("Production Friendly Microfacet Sheen
// BRDF", 2017) as Falcor's SheenBSDF writes it, about the shading normal, under the coat and over the base. Its
// colour is what the raster's rim term ends up as (RIM times mix(1, albedo, RIM_TINT), then the diffuse light's
// albedo and 1 - metallic); its r is the surface's roughness. The base under it passes what it does not reflect in
// each direction in Kulla and Conty's normalised reciprocal form, (1 - s E(V)) (1 - s E(L)) / (1 - s E_avg), s the
// colour's largest channel, E the lobe's directional albedo from Falcor's fit and E_avg its cosine-weighted mean:
// over a diffuse base the pair then leaves exactly what the sheen does not reflect at V. It is sampled by the
// cosine, as the paper recommends, and picked after the coat.

bool hasSheen(MaterialProperties material) {
	return any(greaterThan(material.sheenColor, float3(0.0f, 0.0f, 0.0f)));
}

// The microfacet distribution, its sin^2 floored as Falcor floors it.
float sheenD(float r, float NdotH) {
	float invR = 1.0f / r;
	float sin2h = max(1.0f - NdotH * NdotH, 0.0078125f);
	return (2.0f + invR) * pow(sin2h, invR * 0.5f) / TWO_PI;
}

// The fit of the shadowing's Lambda, interpolated between the paper's r = 0 and r = 1 fits by (1 - r)^2.
float sheenL(float r, float x) {
	float t = (1.0f - r) * (1.0f - r);
	float a = mix(21.5473f, 25.3245f, t);
	float b = mix(3.82987f, 3.32435f, t);
	float c = mix(0.19823f, 0.16801f, t);
	float d = mix(-1.97760f, -1.27393f, t);
	float e = mix(-4.32054f, -4.85967f, t);
	return a / (1.0f + b * pow(x, c)) + d * x + e;
}

float sheenLambda(float r, float cosTheta) {
	return cosTheta < 0.5f ? exp(sheenL(r, cosTheta)) : exp(2.0f * sheenL(r, 0.5f) - sheenL(r, 1.0f - cosTheta));
}

// The shadowing, height-correlated and reciprocal. Unlike Falcor, without the paper's terminator softening (its
// equation 4): the albedo fit below describes the unsoftened lobe (to 0.025), and the softened one lost up to a
// fifth of its grazing energy that the base's attenuation still charged for.
float sheenG(float r, float NdotL, float NdotV) {
	return 1.0f / (1.0f + sheenLambda(r, NdotL) + sheenLambda(r, NdotV));
}

// The lobe's directional albedo for a white sheen at cosine mu: Falcor's fit, eleven rows at r = 0, 0.1 .. 1 of
// (a, b, c, d, e) in 1 - (e + x^d + a / (1 + b x^c)), x = 20 mu + 1, interpolated linearly in r.
const float SHEEN_ALBEDO_FIT[55] = float[](
		1.01108f, 75.2286f, -9.4671f, 0.000973445f, -1.01325f,
		8.44447f, 0.0135054f, 1.11637f, 0.479139f, -9.30873f,
		5.67124f, 0.0194668f, 1.12489f, 0.453551f, -6.53104f,
		4.50086f, 0.0237758f, 1.09376f, 0.423665f, -5.36521f,
		3.77781f, 0.0273286f, 1.0666f, 0.400267f, -4.65601f,
		3.35338f, 0.0307392f, 0.989189f, 0.371074f, -4.24089f,
		3.39605f, 0.0308828f, 0.845455f, 0.333813f, -4.28662f,
		3.62583f, 0.0303531f, 0.79754f, 0.325588f, -4.51725f,
		4.11524f, 0.0172797f, 0.701166f, 0.275242f, -5.04394f,
		4.4012f, 0.0159469f, 0.685892f, 0.270221f, -5.32997f,
		5.8912f, 0.0171461f, 0.66366f, 0.288735f, -6.78867f);

float sheenAlbedoFitRow(int row, float x) {
	int i = row * 5;
	return SHEEN_ALBEDO_FIT[i + 4] + pow(x, SHEEN_ALBEDO_FIT[i + 3]) + SHEEN_ALBEDO_FIT[i] / (1.0f + SHEEN_ALBEDO_FIT[i + 1] * pow(x, SHEEN_ALBEDO_FIT[i + 2]));
}

float sheenAlbedo(float r, float mu) {
	float x = saturate(mu) * 20.0f + 1.0f;
	int rowB = clamp(int(floor(r * 10.0f)), 0, 10);
	int rowT = min(rowB + 1, 10);
	float wB = float(rowT) - r * 10.0f;
	return 1.0f - saturate(sheenAlbedoFitRow(rowB, x) * wB + sheenAlbedoFitRow(rowT, x) * (1.0f - wB));
}

// The fit's cosine-weighted mean, 2 * integral of E(mu) mu dmu, per row of the fit, interpolated as the rows are.
const float SHEEN_ALBEDO_AVERAGE[11] = float[](
		0.001252f, 0.077709f, 0.123519f, 0.162374f, 0.196127f, 0.225372f, 0.249648f, 0.266983f, 0.283394f, 0.293679f, 0.299293f);

float sheenAlbedoAverage(float r) {
	int rowB = clamp(int(floor(r * 10.0f)), 0, 10);
	int rowT = min(rowB + 1, 10);
	float wB = float(rowT) - r * 10.0f;
	return SHEEN_ALBEDO_AVERAGE[rowB] * wB + SHEEN_ALBEDO_AVERAGE[rowT] * (1.0f - wB);
}

// What the sheen does not reflect along S, 1 - s E(|N.S|). The cosine is unsigned, so a thin surface's
// transmission below N crosses it too. Over a diffuse base this one side is the base's whole share at V, which the
// pick and the RR guides weigh.
float sheenAttenuation(MaterialProperties material, float3 N, float3 S) {
	float s = max(material.sheenColor.r, max(material.sheenColor.g, material.sheenColor.b));
	return 1.0f - s * sheenAlbedo(material.sheenRoughness, abs(dot(N, S)));
}

// The base's share between V and L under the sheen: both sides over 1 - s E_avg, Kulla and Conty's normalisation.
float sheenBaseAttenuation(MaterialProperties material, float3 N, float3 V, float3 L) {
	float s = max(material.sheenColor.r, max(material.sheenColor.g, material.sheenColor.b));
	return sheenAttenuation(material, N, V) * sheenAttenuation(material, N, L) / (1.0f - s * sheenAlbedoAverage(material.sheenRoughness));
}

// The sheen's reflection times its NdotL, for the light sample: colour D G / (4 NdotV). Zero for L or V below N.
float3 evalSheen(MaterialProperties material, float3 N, float3 L, float3 V) {
	float NdotL = dot(N, L);
	float NdotV = dot(N, V);
	if (NdotL <= 0.0f || NdotV <= 0.0f) {
		return float3(0.0f, 0.0f, 0.0f);
	}
	NdotL = min(NdotL, 1.0f);
	NdotV = max(0.00001f, min(NdotV, 1.0f));
	float NdotH = saturate(dot(N, normalize(L + V)));
	float r = material.sheenRoughness;
	return material.sheenColor * (sheenD(r, NdotH) * sheenG(r, NdotL, NdotV) / (4.0f * NdotV));
}

// The density with which sampleSheen produces L: the cosine's; zero for L or V below N.
float sheenPdf(float3 N, float3 L, float3 V) {
	float NdotL = dot(N, L);
	if (NdotL <= 0.0f || dot(N, V) <= 0.0f) {
		return 0.0f;
	}
	return diffusePdf(min(NdotL, 1.0f));
}

// The probability with which the scatter in shade_and_bounce picks the sheen after the coat and before the base's
// lobes: the sheen's albedo at V against the base's albedos under it, as clearcoatProbability weighs the coat's.
// 0 without a sheen or with V below N (no random number drawn), 1 over a black base.
float sheenProbability(MaterialProperties material, float3 N, float3 V, float3 specularF0, float3 diffuseReflectance) {
	float NdotV = dot(N, V);
	if (!hasSheen(material) || NdotV <= 0.0f) {
		return 0.0f;
	}
	float sheen = luminance(material.sheenColor) * sheenAlbedo(material.sheenRoughness, NdotV);
	float base = sheenAttenuation(material, N, V) * (luminance(specularF0) + luminance(diffuseReflectance) + luminance(material.diffuseTransmission));
	if (base < 0.0001f) {
		return 1.0f;
	}
	return clamp(sheen / (sheen + base), 0.01f, 0.99f);
}

// Samples the sheen: the cosine about N, the weight its reflection over that density, colour D G pi / (4 NdotV
// NdotL). Returns false as evalIndirectCombinedBRDF does: on a zero weight, or a direction below the geometry
// plane, which rayDirection still carries for the caller's recovery.
bool sampleSheen(float2 u, float3 shadingNormal, float3 geometryNormal, float3 V, MaterialProperties material, OUT_PARAMETER(float3) rayDirection, OUT_PARAMETER(float3) sampleWeight) {
	float3 Llocal = sampleHemisphere(u);
	rayDirection = normalize(rotatePoint(getRotationFromZAxis(shadingNormal), Llocal));
	float NdotL = max(0.00001f, min(1.0f, Llocal.z));
	float NdotV = max(0.00001f, min(1.0f, dot(shadingNormal, V)));
	float NdotH = saturate(dot(shadingNormal, normalize(rayDirection + V)));
	float r = material.sheenRoughness;
	sampleWeight = material.sheenColor * (sheenD(r, NdotH) * sheenG(r, NdotL, NdotV) * PI / (4.0f * NdotV * NdotL));

	if (luminance(sampleWeight) == 0.0f) {
		return false;
	}
	return dot(geometryNormal, rayDirection) > 0.0f;
}

// -------------------------------------------------------------------------
//    Combined BRDF
// -------------------------------------------------------------------------

// Precalculates commonly used terms in BRDF evaluation
// Clamps around dot products prevent NaNs and ensure numerical stability, but make sure to
// correctly ignore rays outside of the sampling hemisphere, by using 'Vbackfacing' and 'Lbackfacing' flags
BrdfData prepareBRDFData(float3 N, float3 L, float3 V, MaterialProperties material) {
	BrdfData data;

	// Evaluate VNHL vectors
	data.V = V;
	data.N = N;
	data.H = normalize(L + V);
	data.L = L;

	float NdotL = dot(N, L);
	float NdotV = dot(N, V);
	data.Vbackfacing = (NdotV <= 0.0f);
	data.Lbackfacing = (NdotL <= 0.0f);

	// Clamp NdotS to prevent numerical instability. Assume vectors below the hemisphere will be filtered using 'Vbackfacing' and 'Lbackfacing' flags
	data.NdotL = min(max(0.00001f, NdotL), 1.0f);
	data.NdotV = min(max(0.00001f, NdotV), 1.0f);

	data.LdotH = saturate(dot(L, data.H));
	data.NdotH = saturate(dot(N, data.H));
	data.VdotH = saturate(dot(V, data.H));

	// Unpack material properties
	data.specularF0 = baseColorToSpecularF0(material.baseColor, material.metalness, material.dielectricF0);
	data.diffuseReflectance = baseColorToDiffuseReflectance(material.baseColor, material.metalness);
	// VAL: under a coat (D118 MM2) the base's F0 is taken from the coat's medium, as the raster remaps it.
	if (hasClearcoat(material)) {
		data.specularF0 = clearcoatBaseF0(data.specularF0, material.clearcoat);
	}

	// Unpack 'perceptively linear' -> 'linear' -> 'squared' roughness
	data.roughness = material.roughness;
	data.alpha = material.roughness * material.roughness;
	data.alphaSquared = data.alpha * data.alpha;

	// VAL: anisotropy as Godot's raster stretches the lobe (D118 MM1): alpha / aspect along T, alpha * aspect along
	// B, aspect = sqrt(1 - 0.9 * anisotropy). The forms are GGX's.
	data.alpha2D = float2(data.alpha, data.alpha);
	data.anisotropic = (MICROFACET_DISTRIBUTION == GGX) && material.anisotropy != 0.0f;
	data.HdotTB = float2(0.0f, 0.0f);
	data.LdotTB = float2(0.0f, 0.0f);
	data.VdotTB = float2(0.0f, 0.0f);
	if (data.anisotropic) {
		float aspect = sqrt(1.0f - 0.9f * material.anisotropy);
		data.alpha2D = float2(data.alpha / aspect, data.alpha * aspect);
		float3 T = material.anisotropyT;
		float3 B = cross(N, T);
		data.HdotTB = float2(dot(T, data.H), dot(B, data.H));
		data.LdotTB = float2(dot(T, L), dot(B, L));
		data.VdotTB = float2(dot(T, V), dot(B, V));
	}

	// Pre-calculate some more BRDF terms
	data.F = evalFresnel(data.specularF0, shadowedF90(data.specularF0), data.LdotH);

	return data;
}

// This is an entry point for evaluation of all other BRDFs based on selected configuration (for direct light)
float3 evalCombinedBRDF(float3 N, float3 L, float3 V, MaterialProperties material) {
	// Prepare data needed for BRDF evaluation - unpack material properties and evaluate commonly used terms (e.g. Fresnel, NdotL, ...)
	const BrdfData data = prepareBRDFData(N, L, V, material);

	// Ignore V and L rays "below" the hemisphere
	if (data.Vbackfacing || data.Lbackfacing) {
		return float3(0.0f, 0.0f, 0.0f);
	}

	// Eval specular and diffuse BRDFs
	float3 specular = evalSpecular(data);
	float3 diffuse = evalDiffuse(data);

	// Combine specular and diffuse layers
#if COMBINE_BRDFS_WITH_FRESNEL
	// Specular is already multiplied by F, just attenuate diffuse
	return (float3(1.0f, 1.0f, 1.0f) - data.F) * diffuse + specular;
#else
	return diffuse + specular;
#endif
}

// Version that returns diffuse and specular separately.
void evalCombinedBRDFSeparate(float3 N, float3 L, float3 V, MaterialProperties material, OUT_PARAMETER(float3) outDiffuse, OUT_PARAMETER(float3) outSpecular) {
	outDiffuse = float3(0.0f, 0.0f, 0.0f);
	outSpecular = float3(0.0f, 0.0f, 0.0f);
	// VAL: hair (D126 H2) replaces every lobe, from either side of N. It goes in the diffuse output, which a light's
	// specular amount does not scale: its highlight is part of one scattering function, not a lobe an artist dims.
	if (hasHair(material)) {
		float pdf;
		evalHair(N, L, V, material, outDiffuse, pdf);
		return;
	}
	// Prepare data needed for BRDF evaluation (uses shading normal N for all BRDF math).
	const BrdfData data = prepareBRDFData(N, L, V, material);
	// VAL: a coat (D118 MM2) adds its reflection to the specular and passes the base what it does not reflect on
	// either side; its normal is not N, so it reflects light the base faces away from too.
	const bool coated = hasClearcoat(material);
	const float coatSpecular = coated ? evalClearcoat(material, L, V) : 0.0f;
	const float coatAttenuation = coated ? clearcoatAttenuation(material, V) * clearcoatAttenuation(material, L) : 1.0f;
	// VAL: a sheen (D118 MM3), under the coat, adds its reflection to the diffuse, where the raster's rim goes, and
	// passes the base what it does not reflect on either side.
	const bool sheened = hasSheen(material);
	const float sheenBase = sheened ? sheenBaseAttenuation(material, N, V, L) : 1.0f;
	// Ignore L rays "below" the hemisphere, except through a thin surface's transmission lobe
	// (D103): Lambertian about -N, its albedo the transmission's.
	if (data.Lbackfacing) {
		if (hasDiffuseTransmission(material)) {
			outDiffuse = material.diffuseTransmission * (ONE_OVER_PI * max(-dot(N, L), 0.0f)) * sheenBase;
		}
		if (coated) {
			outDiffuse *= coatAttenuation;
			outSpecular = float3(coatSpecular, coatSpecular, coatSpecular);
		}
		return;
	}

	// Eval specular and diffuse BRDFs
	outSpecular = evalSpecular(data);
	outDiffuse = evalDiffuse(data);

#if COMBINE_BRDFS_WITH_FRESNEL
	// Specular is already multiplied by F, just attenuate diffuse
	outDiffuse = (float3(1.0f, 1.0f, 1.0f) - data.F) * outDiffuse;
#endif
	// VAL: the subsurface walk's share of the diffuse lobe (D120) arrives at the walk's exit, not here.
	outDiffuse *= 1.0f - material.subsurface;
	if (sheened) {
		outDiffuse = outDiffuse * sheenBase + evalSheen(material, N, L, V);
		outSpecular *= sheenBase;
	}
	if (coated) {
		outDiffuse *= coatAttenuation;
		outSpecular = outSpecular * coatAttenuation + float3(coatSpecular, coatSpecular, coatSpecular);
	}
}

// The probability with which the scatter in shade_and_bounce picks the specular lobe over the
// diffuse one: the luminance split, 1 when the diffuse lobe is dark and 0 when the specular one is
// (no random number is drawn in either case). One function so the density below matches the choice.
float brdfSpecularProbability(float3 specularF0, float3 diffuseReflectance) {
	float specularLum = luminance(specularF0);
	float diffuseLum = luminance(diffuseReflectance);
	if (diffuseLum < 0.0001f) {
		return 1.0f;
	}
	if (specularLum < 0.0001f) {
		return 0.0f;
	}
	return clamp(specularLum / (specularLum + diffuseLum), 0.01f, 0.99f);
}

// The probability with which the scatter in shade_and_bounce takes a thin surface's diffuse
// transmission (D103) before choosing between the reflection lobes: the transmission's luminance
// against the reflection albedos', 0 without transmission (no random number drawn) and 1 when the
// surface reflects nothing.
float diffuseTransmissionProbability(float3 specularF0, float3 diffuseReflectance, float3 diffuseTransmission) {
	float transmissionLum = luminance(diffuseTransmission);
	if (transmissionLum <= 0.0f) {
		return 0.0f;
	}
	float reflectionLum = luminance(specularF0) + luminance(diffuseReflectance);
	if (reflectionLum < 0.0001f) {
		return 1.0f;
	}
	return clamp(transmissionLum / (transmissionLum + reflectionLum), 0.01f, 0.99f);
}

// The density (per solid angle) with which the scatter in shade_and_bounce produces L from V: the
// lobe probability above times the cosine density of the diffuse lobe and the reflection density
// of the specular sampler (specularPdf). A zero-roughness specular lobe is a delta and adds no
// density here; the scatter reports RT_BSDF_PDF_DELTA for it instead. Emissive light sampling
// weighs its samples against this (D50), so it must describe the sampler exactly.
float brdf_pdf(float3 N, float3 L, float3 V, MaterialProperties material) {
	// VAL: hair (D126 H2): the hair sampler's own density, from either side of N.
	if (hasHair(material)) {
		float3 value;
		float pdf;
		evalHair(N, L, V, material, value, pdf);
		return pdf;
	}
	const BrdfData data = prepareBRDFData(N, L, V, material);
	// VAL: a coat (D118 MM2) is picked first, the base's lobes below take the rest; a mirror coat adds no density.
	float coatProbability = clearcoatProbability(material, N, V, data.specularF0, data.diffuseReflectance);
	float coatPdf = coatProbability > 0.0f ? coatProbability * clearcoatPdf(material, L, V) : 0.0f;
	// VAL: a sheen (D118 MM3) takes its share of the rest next, the base's lobes what remains.
	float sheenP = sheenProbability(material, N, V, data.specularF0, data.diffuseReflectance);
	float baseProbability = (1.0f - coatProbability) * (1.0f - sheenP);
	float sheenDensity = sheenP > 0.0f ? (1.0f - coatProbability) * sheenP * sheenPdf(N, L, V) : 0.0f;
	// A thin surface's transmission (D103) takes its share first: cosine about -N below, the
	// reflection lobes the rest above.
	float transmissionProbability = diffuseTransmissionProbability(data.specularF0, data.diffuseReflectance, material.diffuseTransmission);
	if (!data.Vbackfacing && data.Lbackfacing && transmissionProbability > 0.0f) {
		return coatPdf + baseProbability * transmissionProbability * diffusePdf(max(-dot(N, L), 0.0f));
	}
	if (data.Vbackfacing || data.Lbackfacing) {
		return coatPdf;
	}
	float specularProbability = brdfSpecularProbability(data.specularF0, data.diffuseReflectance);
	float pdf = 0.0f;
	if (specularProbability < 1.0f) {
		// VAL: a diffuse pick walks under the surface with the subsurface share (D120), so the direct lobe's density keeps the rest.
		pdf += (1.0f - specularProbability) * (1.0f - material.subsurface) * diffusePdf(data.NdotL);
	}
	if (specularProbability > 0.0f && data.alpha > 0.0f) {
		if (data.anisotropic) {
			pdf += specularProbability * sampleGGXVNDFReflectionPdfAnisotropic(data.alpha2D, float3(data.HdotTB, data.NdotH), float3(data.VdotTB, data.NdotV));
		} else {
			pdf += specularProbability * specularPdf(data.alpha, data.alphaSquared, data.NdotH, data.NdotV, data.LdotH);
		}
	}
	if (transmissionProbability > 0.0f) {
		pdf *= 1.0f - transmissionProbability;
	}
	return coatPdf + sheenDensity + baseProbability * pdf;
}

// This is an entry point for evaluation of all other BRDFs based on selected configuration (for indirect light)
bool evalIndirectCombinedBRDF(float2 u, float3 shadingNormal, float3 geometryNormal, float3 V, MaterialProperties material, const int brdfType, OUT_PARAMETER(float3) rayDirection, OUT_PARAMETER(float3) sampleWeight, float4 debug) {
	// Ignore incident ray coming from "below" the hemisphere
	//if (dot(shadingNormal, V) <= 0.0f) return false;

	// Transform view direction into local space of our sampling routines
	// (local space is oriented so that its positive Z axis points along the shading normal)
	// VAL: an anisotropic lobe (D118 MM1) samples in its own (T, B, N) frame, where its tangent is the X axis.
	const bool anisotropic = (MICROFACET_DISTRIBUTION == GGX) && material.anisotropy != 0.0f;
	const float3 T = material.anisotropyT;
	const float3 B = cross(shadingNormal, T);
	float4 qRotationToZ = getRotationToZAxis(shadingNormal);
	float3 Vlocal = anisotropic ? float3(dot(V, T), dot(V, B), dot(V, shadingNormal)) : rotatePoint(qRotationToZ, V);
	const float3 Nlocal = float3(0.0f, 0.0f, 1.0f);
	MaterialProperties localMaterial = material;
	localMaterial.anisotropyT = float3(1.0f, 0.0f, 0.0f);

	float3 rayDirectionLocal = float3(0.0f, 0.0f, 0.0f);

	if (brdfType == DIFFUSE_TYPE) {
		// Sample diffuse ray using cosine-weighted hemisphere sampling
		rayDirectionLocal = sampleHemisphere(u);
		const BrdfData data = prepareBRDFData(Nlocal, rayDirectionLocal, Vlocal, localMaterial);

		// Function 'diffuseTerm' is predivided by PDF of sampling the cosine weighted hemisphere
		sampleWeight = data.diffuseReflectance * diffuseTerm(data);

#if COMBINE_BRDFS_WITH_FRESNEL
		// Sample a half-vector of specular BRDF. Note that we're reusing random variable 'u' here, but correctly it should be an new independent random number
		float3 Hspecular = sampleSpecularHalfVector(Vlocal, data.alpha2D, u);

		// Clamp HdotL to small value to prevent numerical instability. Assume that rays incident from below the hemisphere have been filtered
		float VdotH = max(0.00001f, min(1.0f, dot(Vlocal, Hspecular)));
		sampleWeight *= (float3(1.0f, 1.0f, 1.0f) - evalFresnel(data.specularF0, shadowedF90(data.specularF0), VdotH));
#endif

	} else if (brdfType == SPECULAR_TYPE) {
		const BrdfData data = prepareBRDFData(Nlocal, float3(0.0f, 0.0f, 1.0f) /* unused L vector */, Vlocal, localMaterial);
		if (data.anisotropic) {
			rayDirectionLocal = sampleSpecularMicrofacetAnisotropic(Vlocal, data.alpha, data.alpha2D, data.specularF0, u, sampleWeight);
		} else {
			rayDirectionLocal = sampleSpecular(Vlocal, data.alpha, data.alphaSquared, data.specularF0, u, sampleWeight);
		}
	}

	// Transform sampled direction Llocal back to V vector space.
	// Done before the early-out checks so the caller always receives a valid
	// direction in 'rayDirection' (e.g. for below-hemisphere mirroring).
	rayDirection = normalize(anisotropic ? rayDirectionLocal.x * T + rayDirectionLocal.y * B + rayDirectionLocal.z * shadingNormal : rotatePoint(invertRotation(qRotationToZ), rayDirectionLocal));

	// Prevent tracing direction with no contribution
	if (luminance(sampleWeight) == 0.0f) {
		return false;
	}

	// Prevent tracing direction "under" the hemisphere (behind the triangle).
	// The caller is expected to handle this case: either terminate the path,
	// or mirror the direction across the geometry plane to keep it above.
	if (dot(geometryNormal, rayDirection) <= 0.0f) {
		return false;
	}

	return true;
}

// -------------------------------------------------------------------------
//    DLSS-RR G-buffer helpers (Environment BRDF approximation)
// -------------------------------------------------------------------------

// Environment BRDF approximation from https://www.unrealengine.com/fr/blog/physically-based-shading-on-mobile
// Author: Brian Karis, License: 'Epic Games, Inc. gives you permission to use, copy, modify, and distribute the code samples in this article as you see fit with no attribution required.'
// Returns specular albedo directly (F0 * scale + bias).
float3 DLSSRR_envBRDFApprox(float3 specularColor, float roughness, float NdotV) {
	const float4 c0 = { -1, -0.0275, -0.572, 0.022 };
	const float4 c1 = { 1, 0.0425, 1.04, -0.04 };
	float4 r = roughness * c0 + c1;
	float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
	float2 AB = float2(-1.04, 1.04) * a004 + r.zw;
	return (specularColor * AB.x + AB.y) * 2.0f;
}

float3 DLSSRR_computeDiffuseAlbedo(float3 baseColor, float metalness) {
	return baseColor * (1.0f - metalness);
}

float3 DLSSRR_computeSpecularAlbedo(float3 baseColor, float metalness, float dielectricF0, float roughness, float NdotV) {
	float3 F0 = baseColorToSpecularF0(baseColor, metalness, dielectricF0);
	return DLSSRR_envBRDFApprox(F0, roughness, NdotV);
}

// Fudges diffuse albedo so that is	never zero. Needed for stable demodulation.
float3 DLSSRR_encodeDiffuseAlbedo(float3 diffuseAlbedo) {
	return max(diffuseAlbedo, float3(0.04f));
}

#endif // BRDF_HLSLI
