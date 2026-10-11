// 12 material samplers (filter/repeat combinations, matching samplers_inc.glsl).
layout(set = 0, binding = 16) uniform sampler SAMPLER_NEAREST_CLAMP;
layout(set = 0, binding = 17) uniform sampler SAMPLER_LINEAR_CLAMP;
layout(set = 0, binding = 18) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_CLAMP;
layout(set = 0, binding = 19) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP;
layout(set = 0, binding = 20) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_CLAMP;
layout(set = 0, binding = 21) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_CLAMP;
layout(set = 0, binding = 22) uniform sampler SAMPLER_NEAREST_REPEAT;
layout(set = 0, binding = 23) uniform sampler SAMPLER_LINEAR_REPEAT;
layout(set = 0, binding = 24) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT;
layout(set = 0, binding = 25) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT;
layout(set = 0, binding = 26) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_REPEAT;
layout(set = 0, binding = 27) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_REPEAT;

// VAL (GPU audit hot 5): the same twelve samplers as one array, in the order above (clamp then repeat; nearest, linear,
// with mipmaps, with mipmaps and anisotropy), which the standard hit group's fetch indexes from the material's flags.
#define RT_MATERIAL_SAMPLER_NEAREST_REPEAT 6u
#define RT_MATERIAL_SAMPLER_LINEAR_REPEAT 7u
#define RT_MATERIAL_SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT 8u
#define RT_MATERIAL_SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT 9u
#define RT_MATERIAL_SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_REPEAT 11u
layout(set = 0, binding = 56) uniform sampler rt_material_samplers[12];
