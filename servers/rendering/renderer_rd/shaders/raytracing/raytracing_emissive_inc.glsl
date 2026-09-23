// Emissive mesh lights (D50): surfaces whose material emits a constant radiance are sampled as
// lights with next-event estimation, weighed against the BSDF path with the power heuristic, so
// a lantern, a torch or a lit window lights its surroundings instead of waiting for a bounce to
// land on it, and the bounce that does land on one is not counted twice.
//
// The CPU (RenderRaytracing::gather_emissive) lists the eligible TLAS instances once per frame,
// sorted by instance index, each with its selection probability from a power CDF over the whole
// list (luminance times an area proxy: no position enters it, so a hit can recover the density
// its instance was sampled with). Selection resamples the CDF (D51): a few candidates drawn by
// the CDF, one kept in proportion to how much of its power reaches the receiver, judged from the
// instance's bounding sphere, so the density the sample really has is local while the density
// the MIS weight is written in stays the global one, which both strategies can evaluate; the
// weights of the two strategies still sum to one at every point, and each strategy's estimate of
// its own weighted share is unbiased, so the sum is. With one candidate the selection is the CDF
// itself. Within the kept instance a triangle is chosen uniformly and a point uniformly on the
// triangle; positions are read straight from the vertex buffer the BLAS was built over, through
// its device address, which works in every stage, and the instance's current object-to-world
// rides in the list. Under the light tree (D55) the instance is chosen by the tree's resampling
// instead and sampled through emissive_sample_instance with the tree density as the density the
// MIS weight is written in; the chance-hit side recovers it by emissive_hit_pdf_entry.
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

// Must match C++ RT_EmissiveInstance (std430, 96 bytes).
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
	float center_x; // The instance's bounding sphere in trace space (D51).
	float center_y;
	float center_z;
	float radius;
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
float emissive_solid_angle_pdf_p(float p_instance, uint primitive_count, float area, float dist, float cos_l) {
	if (area <= 0.0 || cos_l <= 0.0) {
		return 0.0;
	}
	return p_instance / float(max(primitive_count, 1u)) * (dist * dist) / (area * cos_l);
}

float emissive_solid_angle_pdf(RTEmissiveInstance e, float area, float dist, float cos_l) {
	return emissive_solid_angle_pdf_p(e.pdf, e.primitive_count, area, dist, cos_l);
}

/// The power-heuristic weight of the strategy with density p_this against the other's p_other.
float emissive_mis_weight(float p_this, float p_other) {
	float a = p_this * p_this;
	float b = p_other * p_other;
	return a / max(a + b, 1e-20);
}

/// One entry of the inclusive CDF by binary search: the first whose cdf reaches u.
uint emissive_search_cdf(float u, uint count) {
	uint lo = 0u;
	uint hi = count - 1u;
	while (lo < hi) {
		uint mid = (lo + hi) >> 1u;
		if (rt_emissive[mid].cdf < u) {
			lo = mid + 1u;
		} else {
			hi = mid;
		}
	}
	return lo;
}

/// How much of an entry's power can reach a receiver at hit_pos facing N, judged from the entry's
/// bounding sphere: the cosine at the receiver, raised by the sphere's angular size so no part of
/// an emitter above the horizon reads zero, over the distance squared, both floored by the radius
/// for a receiver inside the sphere. A sphere wholly below the receiver's plane is demoted to a
/// thousandth of its radius rather than excluded: the sphere is the rest pose's box, and a skinned
/// or blend-shaped emitter can pose triangles outside it, so a target of zero there would lose
/// their light. Any positive function of the entry is a valid resampling target; this one is the
/// unshadowed irradiance up to the emitter's own facing.
float emissive_proximity(RTEmissiveInstance e, vec3 hit_pos, vec3 N) {
	vec3 to_c = vec3(e.center_x, e.center_y, e.center_z) - hit_pos;
	float d_sq = dot(to_c, to_c);
	float above = max(dot(N, to_c) + e.radius, 1e-3 * e.radius);
	if (above <= 0.0 || e.pdf <= 0.0) {
		return 0.0;
	}
	float r_sq = e.radius * e.radius;
	float denominator = max(sqrt(d_sq), e.radius) * max(d_sq, r_sq);
	return denominator > 0.0 ? above / denominator : 0.0;
}

bool emissive_sample_instance(RTEmissiveInstance e, float p_instance, float resample_factor, vec3 hit_pos, vec3 N, vec3 V, MaterialProperties material, vec2 u_tri, float strategy_probability, out LightSampleNEE r_sample);

/// Next-event estimation toward the emissive list: candidates by the CDF resampled by their
/// proximity (D51), one triangle, one point, and the power-heuristic weight against the BSDF
/// density of the same direction. Fills the sample the caller traces the shadow ray for and
/// evaluates the BRDF with (the same code the analytic lights share); its weight is the MIS
/// weight over the density the sample was really produced with. Returns false when the sample
/// carries nothing. candidates + 3 random numbers are drawn whatever the outcome, so the
/// sequence is stable per call.
bool emissive_sample_direct(
		vec3 hit_pos,
		vec3 N,
		vec3 V,
		MaterialProperties material,
		inout uint rng_state,
		float strategy_probability,
		uint count,
		uint candidates,
		out LightSampleNEE r_sample) {
	r_sample.direction = vec3(0.0, 1.0, 0.0);
	r_sample.shadow_distance = 0.0;
	r_sample.leg_distance = 0.0;
	r_sample.radiance = vec3(0.0);
	r_sample.weight = 0.0;
	r_sample.specular_multiplier = 1.0;

	candidates = clamp(candidates, 1u, 16u);
	// Weighted reservoir over the candidates on one random number, rescaled after every
	// decision so it stays uniform; the candidates are stratified over the CDF. Each candidate
	// is drawn with probability pdf and targeted at pdf times its proximity, so its resampling
	// weight is the proximity alone.
	// rand() can round to 1.0, which would refuse the first candidate below.
	float u_res = min(rand(rng_state), 0.99999994);
	uint selected = 0u;
	float w_sum = 0.0;
	float w_selected = 0.0;
	for (uint i = 0u; i < candidates; i++) {
		float u_sel = (float(i) + rand(rng_state)) / float(candidates);
		uint idx = emissive_search_cdf(u_sel, count);
		RTEmissiveInstance c = rt_emissive[idx];
		float w = emissive_proximity(c, hit_pos, N);
		if (w <= 0.0) {
			continue;
		}
		w_sum += w;
		float accept = w / w_sum;
		if (u_res < accept) {
			selected = idx;
			w_selected = w;
			u_res = u_res / accept;
		} else {
			u_res = (u_res - accept) / (1.0 - accept);
		}
		// Rounding must not carry the number to 1.0: the first kept candidate accepts at 1.0, and
		// a number at 1.0 would refuse it and divide by zero.
		u_res = min(u_res, 0.99999994);
	}
	vec2 u_tri = rand2(rng_state);
	if (w_selected <= 0.0) {
		return false;
	}
	// The resampling's contribution weight is w_sum / (candidates * pdf * w_selected), so the
	// instance was really selected with candidates * w_selected / w_sum times its pdf; with one
	// candidate that factor is 1 and the selection is the CDF's.
	float resample_factor = float(candidates) * w_selected / w_sum;

	RTEmissiveInstance e = rt_emissive[selected];
	return emissive_sample_instance(e, e.pdf, resample_factor, hit_pos, N, V, material, u_tri, strategy_probability, r_sample);
}

/// One emissive instance's sample from its pair of random numbers: a uniform triangle, a uniform
/// point on it, the power-heuristic weight against the BSDF density written in p_instance (the
/// probability the instance is selected with, as the chance-hit side recovers it: the global CDF's
/// under D51, the light tree's exact one under D55) over the density the sample was really
/// produced with (p_instance times resample_factor). Returns false when the sample carries nothing.
bool emissive_sample_instance(
		RTEmissiveInstance e,
		float p_instance,
		float resample_factor,
		vec3 hit_pos,
		vec3 N,
		vec3 V,
		MaterialProperties material,
		vec2 u_tri,
		float strategy_probability,
		out LightSampleNEE r_sample) {
	r_sample.direction = vec3(0.0, 1.0, 0.0);
	r_sample.shadow_distance = 0.0;
	r_sample.leg_distance = 0.0;
	r_sample.radiance = vec3(0.0);
	r_sample.weight = 0.0;
	r_sample.specular_multiplier = 1.0;
	if (e.primitive_count == 0u) {
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

	// The density the MIS weight is written in: what a chance hit recovers.
	float pdf_light = emissive_solid_angle_pdf_p(p_instance, e.primitive_count, area, dist, cos_l);
	if (pdf_light <= 0.0) {
		return false;
	}
	// The density this sample was really produced with: the resampled instance choice in place
	// of the entry's global probability.
	float pdf_sample = pdf_light * resample_factor;

	float pdf_bsdf = brdf_pdf(N, L, V, material);
	// The weight is a function of the point alone, shared with the chance-hit side, so the two
	// strategies' weights sum to one there whatever density either really sampled with; the
	// share of samples sent to this strategy enters it as it enters the other side.
	float w = emissive_mis_weight(strategy_probability * pdf_light, pdf_bsdf);

	r_sample.radiance = vec3(e.radiance_r, e.radiance_g, e.radiance_b) * scene_data_block.data.emissive_exposure_normalization;
	r_sample.weight = w / pdf_sample;
	return true;
}

/// The density with which light sampling from the previous vertex would have produced the hit the
/// path found by its own scatter: the instance's entry (a binary search of the list by instance
/// index; zero when the instance is not listed, so its emission keeps the whole weight), its
/// triangle's area and the cosine at the hit. Zero on the unlit side of a one-sided entry.
uint emissive_find_entry(uint instance_index, uint count) {
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
		return count;
	}
	return lo;
}

float emissive_hit_pdf_entry(uint entry, float p_instance, uint primitive_id, vec3 ray_dir, float hit_t, bool is_front_face);

float emissive_hit_pdf(uint instance_index, uint primitive_id, vec3 ray_dir, float hit_t, bool is_front_face, uint count) {
	uint entry = emissive_find_entry(instance_index, count);
	if (entry >= count) {
		return 0.0;
	}
	return emissive_hit_pdf_entry(entry, rt_emissive[entry].pdf, primitive_id, ray_dir, hit_t, is_front_face);
}

/// The same density for a listed entry selected with probability p_instance (the light tree's,
/// D55, evaluated at the previous vertex by the caller).
float emissive_hit_pdf_entry(uint entry, float p_instance, uint primitive_id, vec3 ray_dir, float hit_t, bool is_front_face) {
	RTEmissiveInstance e = rt_emissive[entry];
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
	return emissive_solid_angle_pdf_p(p_instance, e.primitive_count, 0.5 * twice_area, hit_t, cos_l);
}
