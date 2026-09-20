// Emissive mesh lights (D50): surfaces whose material emits a constant radiance are sampled as
// lights with next-event estimation, weighed against the BSDF path with the power heuristic, so
// a lantern, a torch or a lit window lights its surroundings instead of waiting for a bounce to
// land on it, and the bounce that does land on one is not counted twice.
//
// The CPU (RenderRaytracing::gather_emissive) lists the eligible TLAS instances once per frame,
// sorted by instance index, each with its selection probability from a power CDF over the whole
// list (luminance times an area proxy: no position enters it, so a hit can recover the density
// its instance was sampled with). Selection is one binary search of the CDF, a triangle is
// chosen uniformly within the instance and a point uniformly on the triangle; positions are read
// straight from the vertex buffer the BLAS was built over, through its device address, which
// works in every stage, and the instance's current object-to-world rides in the list.
//
// Requires: raytracing_inc.glsl, brdf_inc.glsl (brdf_pdf), raytracing_hit_inc.glsl
// (geometries[], get_triangle_indices_ex), raytracing_lights_inc.glsl (lights_trace_shadow_ray),
// scene_data_block (emissive_exposure_normalization).

#ifndef RT_EMISSIVE_BUFFER_BINDING
#define RT_EMISSIVE_BUFFER_BINDING 42
#endif

#define RT_EMISSIVE_FLAG_TWO_SIDED 1u
// The instance flips the winding (as Godot's front faces, which wind clockwise as seen, need):
// its lit face is the one from which the corners wind clockwise.
#define RT_EMISSIVE_FLAG_FLIP_FACING 2u

// Must match C++ RT_EmissiveInstance (std430, 80 bytes).
struct RTEmissiveInstance {
	float xform[12]; // Object-to-world, trace space (mat3x4, transposed 3x4).
	float radiance_r; // Linear emission before the exposure normalisation.
	float radiance_g;
	float radiance_b;
	float cdf; // Inclusive; the last entry reads 1.0.
	uint instance_index; // gl_InstanceCustomIndexEXT of the instance: its geometries[] entry.
	uint primitive_count;
	uint flags;
	float pdf; // This entry's selection probability.
};

layout(set = 0, binding = RT_EMISSIVE_BUFFER_BINDING, std430) readonly buffer EmissiveBuffer {
	RTEmissiveInstance rt_emissive[];
};

mat4 emissive_get_transform(RTEmissiveInstance e) {
	return transpose(mat4(
			vec4(e.xform[0], e.xform[1], e.xform[2], e.xform[3]),
			vec4(e.xform[4], e.xform[5], e.xform[6], e.xform[7]),
			vec4(e.xform[8], e.xform[9], e.xform[10], e.xform[11]),
			vec4(0.0, 0.0, 0.0, 1.0)));
}

/// One vertex position in object space, as the BLAS reads it: float3, or the unorm16 x 4 of a
/// compressed surface, whose unit cube the instance transform already maps to the surface's box.
vec3 emissive_fetch_position(GeometryData geom, uint index) {
	if ((geom.flags & FLAG_COMPRESSED) != 0u) {
		Uint32Buffer vb = Uint32Buffer(geom.vertex_address);
		uint w0 = vb.v[index * 2u];
		uint w1 = vb.v[index * 2u + 1u];
		return vec3(float(w0 & 0xFFFFu), float(w0 >> 16u), float(w1 & 0xFFFFu)) / 65535.0;
	}
	FloatBuffer vb = FloatBuffer(geom.vertex_address);
	uint stride = geom.position_stride >> 2u;
	return vec3(vb.v[index * stride], vb.v[index * stride + 1u], stride >= 3u ? vb.v[index * stride + 2u] : 0.0);
}

/// The three trace-space corners of a triangle of an emissive instance.
void emissive_fetch_triangle(RTEmissiveInstance e, uint primitive_id, out vec3 p0, out vec3 p1, out vec3 p2) {
	GeometryData geom = geometries[e.instance_index];
	uint i0, i1, i2;
	get_triangle_indices_ex(geom, primitive_id, i0, i1, i2);
	mat4 xform = emissive_get_transform(e);
	p0 = (xform * vec4(emissive_fetch_position(geom, i0), 1.0)).xyz;
	p1 = (xform * vec4(emissive_fetch_position(geom, i1), 1.0)).xyz;
	p2 = (xform * vec4(emissive_fetch_position(geom, i2), 1.0)).xyz;
}

/// The density, per solid angle at the receiver, with which light sampling produces a point at
/// distance dist on a triangle of area, met at cos_l (the cosine at the triangle, against the
/// direction). Zero for a degenerate triangle or a point seen edge-on.
float emissive_solid_angle_pdf(RTEmissiveInstance e, float area, float dist, float cos_l) {
	if (area <= 0.0 || cos_l <= 0.0) {
		return 0.0;
	}
	return e.pdf / float(max(e.primitive_count, 1u)) * (dist * dist) / (area * cos_l);
}

/// The power-heuristic weight of the strategy with density p_this against the other's p_other.
float emissive_mis_weight(float p_this, float p_other) {
	float a = p_this * p_this;
	float b = p_other * p_other;
	return a / max(a + b, 1e-20);
}

/// Next-event estimation toward the emissive list: one instance by the CDF, one triangle, one
/// point, and the power-heuristic weight against the BSDF density of the same direction. Fills
/// the sample the caller traces the shadow ray for and evaluates the BRDF with (the same code the
/// analytic lights share); its weight is the MIS weight over the light's density. Returns false
/// when the sample carries nothing. Three random numbers are drawn whatever the outcome, so the
/// sequence is stable per call.
bool emissive_sample_direct(
		vec3 hit_pos,
		vec3 N,
		vec3 V,
		MaterialProperties material,
		inout uint rng_state,
		float strategy_probability,
		uint count,
		out LightSampleNEE r_sample) {
	r_sample.direction = vec3(0.0, 1.0, 0.0);
	r_sample.shadow_distance = 0.0;
	r_sample.leg_distance = 0.0;
	r_sample.radiance = vec3(0.0);
	r_sample.weight = 0.0;
	r_sample.specular_multiplier = 1.0;

	float u_sel = rand(rng_state);
	vec2 u_tri = rand2(rng_state);

	// Binary search of the inclusive CDF.
	uint lo = 0u;
	uint hi = count - 1u;
	while (lo < hi) {
		uint mid = (lo + hi) >> 1u;
		if (rt_emissive[mid].cdf < u_sel) {
			lo = mid + 1u;
		} else {
			hi = mid;
		}
	}
	RTEmissiveInstance e = rt_emissive[lo];
	if (e.pdf <= 0.0 || e.primitive_count == 0u) {
		return false;
	}

	uint primitive_id = min(uint(u_tri.x * float(e.primitive_count)), e.primitive_count - 1u);
	// The triangle's share of u_tri.x is spent; stretch the remainder back to [0, 1).
	float u_rem = fract(u_tri.x * float(e.primitive_count));
	vec3 p0, p1, p2;
	emissive_fetch_triangle(e, primitive_id, p0, p1, p2);

	// Uniform point on the triangle.
	float su = sqrt(u_rem);
	float b1 = 1.0 - su;
	float b2 = u_tri.y * su;
	vec3 point = p0 * (1.0 - b1 - b2) + p1 * b1 + p2 * b2;

	vec3 n_scaled = cross(p1 - p0, p2 - p0);
	float twice_area = length(n_scaled);
	if (twice_area <= 0.0) {
		return false;
	}
	float area = 0.5 * twice_area;
	vec3 n_tri = n_scaled / twice_area;

	vec3 to_light = point - hit_pos;
	float dist_sq = dot(to_light, to_light);
	if (dist_sq <= 1e-8) {
		return false;
	}
	float dist = sqrt(dist_sq);
	vec3 L = to_light / dist;
	r_sample.direction = L;
	r_sample.shadow_distance = dist;
	r_sample.leg_distance = dist;

	// The cosine at the light on its lit face: n_tri faces the side from which the corners wind
	// counter-clockwise, which is the culled side of a flipped instance.
	float cos_l = dot(n_tri, -L);
	if ((e.flags & RT_EMISSIVE_FLAG_FLIP_FACING) != 0u) {
		cos_l = -cos_l;
	}
	if ((e.flags & RT_EMISSIVE_FLAG_TWO_SIDED) != 0u) {
		cos_l = abs(cos_l);
	}
	if (cos_l <= 0.0 || dot(N, L) <= 0.0) {
		return false;
	}

	float pdf_light = emissive_solid_angle_pdf(e, area, dist, cos_l);
	if (pdf_light <= 0.0) {
		return false;
	}

	float pdf_bsdf = brdf_pdf(N, L, V, material);
	// The density this strategy really has is the list's times the share of samples sent to it.
	float w = emissive_mis_weight(strategy_probability * pdf_light, pdf_bsdf);

	r_sample.radiance = vec3(e.radiance_r, e.radiance_g, e.radiance_b) * scene_data_block.data.emissive_exposure_normalization;
	r_sample.weight = w / pdf_light;
	return true;
}

/// The density with which light sampling from the previous vertex would have produced the hit the
/// path found by its own scatter: the instance's entry (a binary search of the list by instance
/// index; zero when the instance is not listed, so its emission keeps the whole weight), its
/// triangle's area and the cosine at the hit. Zero on the unlit side of a one-sided entry.
float emissive_hit_pdf(uint instance_index, uint primitive_id, vec3 ray_dir, float hit_t, bool is_front_face, uint count) {
	uint lo = 0u;
	uint hi = count;
	while (lo < hi) {
		uint mid = (lo + hi) >> 1u;
		if (rt_emissive[mid].instance_index < instance_index) {
			lo = mid + 1u;
		} else {
			hi = mid;
		}
	}
	if (lo >= count || rt_emissive[lo].instance_index != instance_index) {
		return 0.0;
	}
	RTEmissiveInstance e = rt_emissive[lo];
	if (!is_front_face && (e.flags & RT_EMISSIVE_FLAG_TWO_SIDED) == 0u) {
		return 0.0;
	}
	vec3 p0, p1, p2;
	emissive_fetch_triangle(e, primitive_id, p0, p1, p2);
	vec3 n_scaled = cross(p1 - p0, p2 - p0);
	float twice_area = length(n_scaled);
	if (twice_area <= 0.0) {
		return 0.0;
	}
	// The side was settled above; the cosine is the face's own, as light sampling computes it,
	// not the interpolated normal the hit carries, so the two densities agree exactly.
	float cos_l = abs(dot(n_scaled, ray_dir)) / twice_area;
	return emissive_solid_angle_pdf(e, 0.5 * twice_area, hit_t, cos_l);
}
