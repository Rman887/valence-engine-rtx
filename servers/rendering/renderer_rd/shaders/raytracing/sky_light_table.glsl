#[compute]

#version 450

#VERSION_DEFINES

// The sky's sampling table for the path tracer (D54): a luminance distribution over the sky's
// radiance octmap, built once per sky change, from which a hit draws a sky direction in
// proportion to its brightness (raytracing_sky_inc.glsl). The table is a pyramid over the
// octahedral square (D69): its leaves are the octmap's own texels (2^levels a side, at least the
// 256 of the moments' grid and at most 1024), each holding the texel's luminance times its solid
// angle, and every level above holds the sums of four children, the four children of one node
// stored as one vec4 so that a descent reads one 16-byte word per level. A hit draws a leaf by
// descending from the top, choosing a column and a row of each quad in proportion to their sums
// and rescaling its two random numbers inside the chosen part, which keeps a stratified pair
// stratified; a direction's density is its leaf's weight over the sum, per unit of octahedral
// area, and the solid-angle density at any direction follows from the octahedral map's Jacobian,
// which the sampling and the weighing sides evaluate alike, so no per-cell solid angle is ever
// needed for the density. A sun painted in an HDRI (half a degree) is then drawn inside its own
// texel rather than uniformly inside a 0.7-degree cell of the earlier 256 x 256 grid with its
// two inclusive CDFs, which left a sample inside the sun's cell a lottery between the disc and
// the sky beside it. The floor that keeps every leaf reachable (a fraction of the sky's total
// spread evenly over the leaves) is not stored: every reader adds a level's share of it to the
// weights it reads, from the header the build writes, so the pyramid is built in one pass over
// the texels and a reduction. MODE_LEAF writes the leaves, one thread each; MODE_REDUCE sums one
// level from the one below, one thread per node, for the levels too large for one workgroup;
// MODE_HEADER reduces the rest in shared memory, writes the total and the header, and reports how
// much of the sky's light is concentrated (the cells of the 256-grid brighter than
// SKY_CONCENTRATION_RATIO times the sky's mean radiance): a smooth sky is estimated well by the
// BSDF's own directions, and a light sample spent on it is one taken from a light whose sample
// had no variance, so the share the sky gets follows its concentrated light, not its total.
// For the share by variance (D62) MODE_MOMENTS then reports, at each of eight candidate shares, the
// integrals the variance of the sky's irradiance on a Lambertian receiver is built from, when a
// share c of the light samples draws from this table and the receiver's cosine lobe draws the
// bounce, the two combined by the power heuristic as the trace combines them: the second moment of
// the combined estimator, the integral of f^2 (q^3 + p^3) / (q^2 + p^2)^2 with p the lobe's
// density and q = c p_sky, and the two strategies' weighted means, over the 256-grid's cells with
// the pyramid's density at that level. The receiver is not only the one facing the sky's up
// (D65): one workgroup per normal of a grid over the octahedral square in the sky's frame,
// SKY_NORMAL_CELLS cells a side and a normal at every vertex, so the trace reads the four around
// its own normal bilinearly without a wrap (the vertices on the square's border that the
// octahedral fold identifies hold one direction, so the reading is continuous across it). The
// radiance the trace delivers is not the octmap's (D64): sky_radiance mixes the sharpest level L
// toward the fog, whose colour is the Environment's fog light mixed with a blurred level L_f by the
// aerial perspective, so along any direction it is alpha L + beta L_f + gamma, the weights set per
// frame by the fog and the exposure. The moments therefore hold, per normal and share, the
// second-moment integrals of every product of L, L_f and 1 and the two means' integrals of each,
// built only when the sky changes, and MODE_COMBINE folds them with the frame's weights into the
// variance and the sky strategy's mean the trace reads, run when the table or the weights change.
// A smooth sky reads nearly zero variance at c = 0 (the cosine lobe alone estimates it well) and
// rises with c; a sky that carries a small bright source reads high at c = 0 and falls as c grows.
// The Lambertian receiver, whose cosine lobe stands for the material's density, is a proxy for the
// trace's surfaces: it chooses the share, and any share is unbiased.

#ifdef MODE_LEAF
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
#else
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
#endif

#include "../oct_inc.glsl"

#define SKY_CELL_LEVEL 8u // The moments' grid and the concentration's: level 8 of the pyramid, 256 cells a side. Must match RT_SKY_LIGHT_CELL_LEVEL.
#define SKY_CELL_N (1u << SKY_CELL_LEVEL)
#define SKY_CELLS (SKY_CELL_N * SKY_CELL_N)
#define SKY_HEADER_LEVELS 5u // MODE_HEADER reduces this level and the ones above it from the level below; its shared arrays and thread counts are written for 5 and SKY_CELL_LEVEL 8, which render_raytracing.h asserts. Must match RT_SKY_LIGHT_HEADER_LEVELS.
#define SKY_CONCENTRATION_RATIO 4.0
#define SKY_FULL_SPHERE (4.0 * 3.14159265358979)
#define SKY_PI 3.14159265358979
#define SKY_SHARE_COUNT 8u
#define SKY_BASIS 12u // Must match RT_SKY_BASIS in raytracing_sky_inc.glsl and RT_SKY_LIGHT_BASIS.
#define SKY_NORMAL_CELLS 8u // Must match RT_SKY_NORMAL_CELLS in raytracing_sky_inc.glsl and RT_SKY_LIGHT_NORMAL_CELLS.
#define SKY_NORMALS ((SKY_NORMAL_CELLS + 1u) * (SKY_NORMAL_CELLS + 1u))

// The candidate shares, written to the table's sky_share; must match RT_SKY_SHARES, the trace's
// compile-time copy in raytracing_sky_inc.glsl (D67).
const float SKY_SHARES[SKY_SHARE_COUNT] = float[](0.0, 1.0 / 32.0, 1.0 / 16.0, 0.125, 0.25, 0.5, 0.75, 0.9);

#if defined(MODE_LEAF) || defined(MODE_MOMENTS)
// The radiance octmap's sharpest layer as a 2D texture with its mip chain.
layout(set = 0, binding = 0) uniform texture2D radiance_octmap;
layout(set = 0, binding = 1) uniform sampler radiance_sampler;
#endif

#ifdef USE_RADIANCE_OCTMAP_ARRAY
// The octmap's roughness layers, for the fog's level, which is one of them.
layout(set = 0, binding = 3) uniform texture2DArray radiance_layers;
#endif

// Must match SkyLightBuffer in raytracing_sky_inc.glsl and RT_SKY_LIGHT_BUFFER_SIZE.
layout(set = 0, binding = 2, std430) restrict buffer SkyLightBuffer {
	float sky_total; // The sum over the leaves of luminance times solid angle; 0 for a black sky.
	float sky_concentrated; // The part of sky_total in the 256-grid's cells brighter than the ratio times the mean.
	float sky_pdf_scale; // A leaf's floored weight times this is its density per unit octahedral area: the leaf count over the floored total.
	float sky_floor_leaf; // The floor every leaf's weight carries: floor_fraction times the total over the leaf count.
	uint sky_levels; // The pyramid's levels below the top: the leaves are level sky_levels, 2^sky_levels a side.
	uint sky_leaf_n; // 2^sky_levels.
	uint sky_pad[2];
	float sky_share[SKY_SHARE_COUNT]; // The candidate shares, SKY_SHARES (D62).
	// Per receiver normal and share, the sky term's variance and the sky strategy's weighted mean
	// under the frame's weights (D65), what the trace reads.
	vec2 sky_share_moments[SKY_NORMALS * SKY_SHARE_COUNT];
	// Per receiver normal and share, SKY_BASIS integrals before the exposure (D64): the second
	// moment's over the products L L, L_f L_f, 1, L L_f, L, L_f, then the bounce's weighted mean over
	// L, L_f, 1, then the sky strategy's over L, L_f, 1.
	float sky_basis[SKY_NORMALS * SKY_SHARE_COUNT * SKY_BASIS];
	// The pyramid, levels 1 to sky_levels (D69): level l holds 2^l x 2^l node weights as the quads of
	// the level above's nodes, row-major over those, each quad the node's four children in the order
	// (2x, 2y), (2x + 1, 2y), (2x, 2y + 1), (2x + 1, 2y + 1). The floor is not stored.
	vec4 sky_pyramid[];
};

layout(push_constant, std430) uniform Params {
	float border_pad; // The octmap's padding in UV space (radiance_border_size).
	float border_scale; // 1 - 2 * border_pad.
	float lod_cell; // For the moments' cells: the octmap mip whose texel is about half a cell wide (0 when the mips are roughness levels, or the sharpest is already coarser).
	float floor_fraction; // The share of the sky's total spread evenly, so no leaf has zero density.
	// The frame's weights of L, L_f and 1 in the delivered luminance (MODE_COMBINE), as the renderer
	// reads them from the scene data sky_radiance reads.
	float mix_level;
	float mix_fog_level;
	float mix_constant;
	float lod_leaf; // For the leaves: the mip whose texel is half a leaf wide, or the sharpest.
	float tap_offset; // The four taps' offset from a leaf's centre in leaves (0.25), or 0 where a leaf is one texel and one tap at the centre reads it.
	uint leaf_level; // The leaves' level: the octmap's texels, held to [SKY_CELL_LEVEL, RT_SKY_LIGHT_MAX_LEVELS].
	uint reduce_level; // MODE_REDUCE: the level this dispatch writes from the one below.
	uint pad;
}
params;

// The pyramid's layout. Level l's quads start after the (4^(l - 1) - 1) / 3 quads of the levels
// above; node (x, y) of level l is component (y & 1) * 2 + (x & 1) of the quad of its parent
// (x / 2, y / 2), which is row-major over the 2^(l - 1) parents a side. Must match the trace's
// reading in raytracing_sky_inc.glsl.
uint sky_quad_base(uint p_level) {
	return ((1u << (2u * p_level - 2u)) - 1u) / 3u;
}

uint sky_node_quad(uint p_level, uvec2 p_node) {
	return sky_quad_base(p_level) + (p_node.y >> 1u) * (1u << (p_level - 1u)) + (p_node.x >> 1u);
}

uint sky_node_component(uvec2 p_node) {
	return ((p_node.y & 1u) << 1u) | (p_node.x & 1u);
}

// A cell's centre direction on a grid of p_n cells a side and its solid angle from the map's
// Jacobian there: an octahedral area element d(uv) subtends 4 |d|_1^3 d(uv) of solid angle at
// the unit direction d it maps to.
vec3 sky_grid_centre(uvec2 p_cell, uint p_n, out float r_solid_angle) {
	vec2 uv_centre = (vec2(p_cell) + 0.5) / float(p_n);
	vec3 dir_centre = oct_to_vec3(uv_centre * 2.0 - 1.0);
	float l1 = abs(dir_centre.x) + abs(dir_centre.y) + abs(dir_centre.z);
	r_solid_angle = 4.0 * l1 * l1 * l1 / float(p_n * p_n);
	return dir_centre;
}

#if defined(MODE_LEAF) || defined(MODE_MOMENTS)

float sky_luminance(vec3 p_color) {
	return dot(p_color, vec3(0.2126, 0.7152, 0.0722));
}

// A direction inside a cell of a grid of p_n cells a side: tap t of the four the cell is read at,
// at p_offset cells from its centre.
vec3 sky_grid_tap(uvec2 p_cell, uint p_n, uint p_tap, float p_offset) {
	vec2 uv = (vec2(p_cell) + 0.5 + p_offset * vec2(float(p_tap & 1u) * 2.0 - 1.0, float(p_tap >> 1u) * 2.0 - 1.0)) / float(p_n);
	return oct_to_vec3(uv * 2.0 - 1.0);
}

// Four taps inside the cell at the chosen mip: the cell's mean radiance, whatever the distortion
// of the octahedral map inside it. With a zero offset the taps coincide at the centre and one is
// read: the leaf that is one texel or less.
float sky_grid_luminance(uvec2 p_cell, uint p_n, float p_lod, float p_offset) {
	uint taps = (p_offset > 0.0) ? 4u : 1u;
	float lum = 0.0;
	for (uint t = 0u; t < taps; t++) {
		vec2 tex_uv = vec3_to_oct_with_border(sky_grid_tap(p_cell, p_n, t, p_offset), vec2(params.border_pad, params.border_scale));
		lum += sky_luminance(textureLod(sampler2D(radiance_octmap, radiance_sampler), tex_uv, p_lod).rgb);
	}
	return max(lum / float(taps), 0.0);
}

#endif // MODE_LEAF || MODE_MOMENTS

#ifdef MODE_LEAF

// One thread per leaf: its luminance times its solid angle, into its quad of the leaf level.
void main() {
	uvec2 leaf = gl_GlobalInvocationID.xy;
	uint n = 1u << params.leaf_level;
	if (leaf.x >= n || leaf.y >= n) {
		return;
	}
	float solid_angle;
	sky_grid_centre(leaf, n, solid_angle);
	float w = sky_grid_luminance(leaf, n, params.lod_leaf, params.tap_offset) * solid_angle;
	sky_pyramid[sky_node_quad(params.leaf_level, leaf)][sky_node_component(leaf)] = w;
}

#endif // MODE_LEAF

#ifdef MODE_REDUCE

// One thread per node of params.reduce_level: the sum of its quad, which is the level below's
// four children, into its own quad of its level. The four threads of a quad write four components.
void main() {
	uint level = params.reduce_level;
	uint side = 1u << level;
	uint i = gl_GlobalInvocationID.x;
	if (i >= side * side) {
		return;
	}
	uvec2 node = uvec2(i & (side - 1u), i >> level);
	vec4 q = sky_pyramid[sky_quad_base(level + 1u) + node.y * side + node.x];
	sky_pyramid[sky_node_quad(level, node)][sky_node_component(node)] = (q.x + q.y) + (q.z + q.w);
}

#endif // MODE_REDUCE

#ifdef MODE_HEADER

// The levels SKY_HEADER_LEVELS down to 1, reduced in shared memory from the level below the
// first (1024 nodes at level 5, four per thread), and the threads' parts of the concentrated light.
shared float node_5[1024];
shared float node_4[256];
shared float node_3[64];
shared float node_2[16];
shared float node_1[4];
shared float part_concentrated[256];

// The sum of the quad of node (x, y) of a level whose values sit row-major in a shared array of
// p_side nodes a side.
#define SKY_SHARED_QUAD(arr, node, side) ((arr[(node.y * 2u) * side + node.x * 2u] + arr[(node.y * 2u) * side + node.x * 2u + 1u]) + (arr[(node.y * 2u + 1u) * side + node.x * 2u] + arr[(node.y * 2u + 1u) * side + node.x * 2u + 1u]))

// One workgroup. Reads level SKY_HEADER_LEVELS + 1 (written by MODE_REDUCE or MODE_LEAF), writes
// the levels above it and the header.
void main() {
	uint t = gl_LocalInvocationID.x;
	// Level SKY_HEADER_LEVELS (5) from the quads of the level below, four nodes per thread.
	for (uint k = 0u; k < 4u; k++) {
		uint i = t + 256u * k;
		uvec2 node = uvec2(i & 31u, i >> 5u);
		vec4 q = sky_pyramid[sky_quad_base(SKY_HEADER_LEVELS + 1u) + node.y * 32u + node.x];
		float s = (q.x + q.y) + (q.z + q.w);
		node_5[i] = s;
		sky_pyramid[sky_node_quad(SKY_HEADER_LEVELS, node)][sky_node_component(node)] = s;
	}
	barrier();
	{
		uvec2 node = uvec2(t & 15u, t >> 4u);
		float s = SKY_SHARED_QUAD(node_5, node, 32u);
		node_4[t] = s;
		sky_pyramid[sky_node_quad(4u, node)][sky_node_component(node)] = s;
	}
	barrier();
	if (t < 64u) {
		uvec2 node = uvec2(t & 7u, t >> 3u);
		float s = SKY_SHARED_QUAD(node_4, node, 16u);
		node_3[t] = s;
		sky_pyramid[sky_node_quad(3u, node)][sky_node_component(node)] = s;
	}
	barrier();
	if (t < 16u) {
		uvec2 node = uvec2(t & 3u, t >> 2u);
		float s = SKY_SHARED_QUAD(node_3, node, 8u);
		node_2[t] = s;
		sky_pyramid[sky_node_quad(2u, node)][sky_node_component(node)] = s;
	}
	barrier();
	if (t < 4u) {
		uvec2 node = uvec2(t & 1u, t >> 1u);
		float s = SKY_SHARED_QUAD(node_2, node, 4u);
		node_1[t] = s;
		sky_pyramid[sky_node_quad(1u, node)][sky_node_component(node)] = s;
	}
	barrier();
	float total = (node_1[0] + node_1[1]) + (node_1[2] + node_1[3]);
	// The concentrated light: the 256-grid's cells above the ratio times the sphere's mean
	// radiance, a cell's luminance being its weight over its solid angle; 64 quads of that level per
	// thread. The floor is not in the weights, as the ratio compares the sky's own light.
	float threshold = SKY_CONCENTRATION_RATIO * total / SKY_FULL_SPHERE;
	float concentrated = 0.0;
	uint cell_base = sky_quad_base(SKY_CELL_LEVEL);
	for (uint k = 0u; k < 64u; k++) {
		uint q = t * 64u + k;
		uvec2 parent = uvec2(q & (SKY_CELL_N / 2u - 1u), q / (SKY_CELL_N / 2u));
		vec4 w = sky_pyramid[cell_base + q];
		for (uint c = 0u; c < 4u; c++) {
			float solid_angle;
			sky_grid_centre(parent * 2u + uvec2(c & 1u, c >> 1u), SKY_CELL_N, solid_angle);
			if (w[c] > threshold * solid_angle) {
				concentrated += w[c];
			}
		}
	}
	part_concentrated[t] = concentrated;
	barrier();
	if (t == 0u) {
		float concentrated_total = 0.0;
		for (uint i = 0u; i < 256u; i++) {
			concentrated_total += part_concentrated[i];
		}
		uint n = 1u << params.leaf_level;
		float leaves = float(n * n);
		sky_total = total;
		sky_concentrated = concentrated_total;
		// The floor keeps every leaf reachable, so the density is positive wherever the octmap can
		// hold light; a black sky gets a uniform table that is never drawn from (its total is 0).
		sky_floor_leaf = (total > 0.0) ? params.floor_fraction * total / leaves : 1.0;
		sky_pdf_scale = (total > 0.0) ? leaves / (total * (1.0 + params.floor_fraction)) : 1.0;
		sky_levels = params.leaf_level;
		sky_leaf_n = n;
	}
	if (t < SKY_SHARE_COUNT) {
		sky_share[t] = SKY_SHARES[t];
	}
}

#undef SKY_SHARED_QUAD

#endif // MODE_HEADER

#ifdef MODE_MOMENTS

shared float row_basis[SKY_BASIS][SKY_CELL_N + 1u]; // Padded: the twelve summing threads read one row each, a bank apart.

// The luminance of the level the fog's aerial perspective reads: sky_radiance's sample at a
// roughness of 1 / MAX_ROUGHNESS_LOD, which is the second roughness layer at its sharpest in array
// mode and the texture's second mip otherwise.
float sky_fog_luminance(vec2 p_tex_uv) {
#ifdef USE_RADIANCE_OCTMAP_ARRAY
	return sky_luminance(textureLod(sampler2DArray(radiance_layers, radiance_sampler), vec3(p_tex_uv, 1.0), 0.0).rgb);
#else
	return sky_luminance(textureLod(sampler2D(radiance_octmap, radiance_sampler), p_tex_uv, 1.0).rgb);
#endif
}

// The fog's level over the cell, from the same four taps as the cell's own luminance.
float sky_cell_fog_luminance(uvec2 p_cell) {
	float lum = 0.0;
	for (uint t = 0u; t < 4u; t++) {
		lum += sky_fog_luminance(vec3_to_oct_with_border(sky_grid_tap(p_cell, SKY_CELL_N, t, 0.25), vec2(params.border_pad, params.border_scale)));
	}
	return max(lum * 0.25, 0.0);
}

// One workgroup per receiver normal of the grid, one thread per row of the 256-grid, which is the
// pyramid's level SKY_CELL_LEVEL.
void main() {
	uint row = gl_LocalInvocationID.x;
	uint normal_index = gl_WorkGroupID.x;
	uvec2 vertex = uvec2(normal_index % (SKY_NORMAL_CELLS + 1u), normal_index / (SKY_NORMAL_CELLS + 1u));
	vec3 receiver = oct_to_vec3(vec2(vertex) / float(SKY_NORMAL_CELLS) * 2.0 - 1.0);
	// A cell's floored weight is its node's plus the floor of the leaves under it, and its density
	// per unit octahedral area is that over the floored total times the cell count: the leaf's scale
	// over the leaves per cell.
	float leaves_per_cell = float(1u << (2u * (sky_levels - SKY_CELL_LEVEL)));
	float cell_floor = sky_floor_leaf * leaves_per_cell;
	float cell_scale = sky_pdf_scale / leaves_per_cell;
	// Per share, the integrals over this row's cells, in sky_basis's order.
	float basis[SKY_SHARE_COUNT][SKY_BASIS];
	for (uint k = 0u; k < SKY_SHARE_COUNT; k++) {
		for (uint i = 0u; i < SKY_BASIS; i++) {
			basis[k][i] = 0.0;
		}
	}
	for (uint col = 0u; col < SKY_CELL_N; col++) {
		// The cell at its centre as seen from the receiver: the cosine, the cosine lobe's density
		// and the table's, each over solid angle, and the power heuristic's weights.
		uvec2 cell = uvec2(col, row);
		float solid_angle;
		vec3 dir_centre = sky_grid_centre(cell, SKY_CELL_N, solid_angle);
		float cos_theta = dot(receiver, dir_centre);
		// Not merely positive: a receiver in the octahedron's equator at 45 degrees ((+-a, +-a, 0))
		// meets the cells whose centre has y = -x as the rounding residue of a*x - a*x, and at the
		// share 0 the second moment's (cos/pi)^4 then underflows to 0 / 0. Such a cell carries under
		// 1e-6 of a cell's cosine weight; at the floor every term stays a normal float.
		if (!(cos_theta > 1e-6)) {
			continue;
		}
		float lum = sky_grid_luminance(cell, SKY_CELL_N, params.lod_cell, 0.25);
		float lum_fog = sky_cell_fog_luminance(cell);
		float p_bsdf = cos_theta / SKY_PI;
		float p2 = p_bsdf * p_bsdf;
		// The table's density over solid angle: per octahedral area over the Jacobian 4 |d|_1^3, which
		// is the cell's solid angle times the cell count, as sky_pdf reads it at a leaf.
		float w_cell = sky_pyramid[sky_node_quad(SKY_CELL_LEVEL, cell)][sky_node_component(cell)] + cell_floor;
		float p_sky = w_cell * cell_scale / (solid_angle * float(SKY_CELLS));
		for (uint k = 0u; k < SKY_SHARE_COUNT; k++) {
			// With q = c p_sky the weights are q^2 / (q^2 + p^2) and p^2 / (q^2 + p^2), and the two
			// samples' second moments sum to f^2 (q^3 + p^3) / (q^2 + p^2)^2 over solid angle.
			float q = SKY_SHARES[k] * p_sky;
			float q2 = q * q;
			float sum2 = q2 + p2;
			float w_moment = solid_angle * cos_theta * cos_theta * (q2 * q + p2 * p_bsdf) / (sum2 * sum2);
			float w_bsdf = solid_angle * cos_theta * p2 / sum2;
			float w_sky = solid_angle * cos_theta * q2 / sum2;
			basis[k][0] += w_moment * lum * lum;
			basis[k][1] += w_moment * lum_fog * lum_fog;
			basis[k][2] += w_moment;
			basis[k][3] += w_moment * lum * lum_fog;
			basis[k][4] += w_moment * lum;
			basis[k][5] += w_moment * lum_fog;
			basis[k][6] += w_bsdf * lum;
			basis[k][7] += w_bsdf * lum_fog;
			basis[k][8] += w_bsdf;
			basis[k][9] += w_sky * lum;
			basis[k][10] += w_sky * lum_fog;
			basis[k][11] += w_sky;
		}
	}

	// The integrals at each share, the rows reduced through shared memory one share at a time, one
	// thread per integral.
	for (uint k = 0u; k < SKY_SHARE_COUNT; k++) {
		barrier(); // The previous share's reduction has read the rows.
		for (uint i = 0u; i < SKY_BASIS; i++) {
			row_basis[i][row] = basis[k][i];
		}
		barrier();
		if (row < SKY_BASIS) {
			float s = 0.0;
			for (uint r = 0u; r < SKY_CELL_N; r++) {
				s += row_basis[row][r];
			}
			sky_basis[(normal_index * SKY_SHARE_COUNT + k) * SKY_BASIS + row] = s;
		}
	}
}

#endif // MODE_MOMENTS

#ifdef MODE_COMBINE

// One thread per receiver normal and share: the moments' integrals of each product of L, L_f and 1
// combined with the frame's weights into the second moment, and of each level into the two
// strategies' means, then the variance (the two strategies' means both leave the second moment,
// since one sample of each is taken) and the sky strategy's mean the covariance reads.
void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= SKY_NORMALS * SKY_SHARE_COUNT) {
		return;
	}
	uint b = i * SKY_BASIS;
	vec3 m = vec3(params.mix_level, params.mix_fog_level, params.mix_constant);
	float moment = m.x * m.x * sky_basis[b] + m.y * m.y * sky_basis[b + 1u] + m.z * m.z * sky_basis[b + 2u] +
			2.0 * (m.x * m.y * sky_basis[b + 3u] + m.x * m.z * sky_basis[b + 4u] + m.y * m.z * sky_basis[b + 5u]);
	float mean_bsdf = dot(m, vec3(sky_basis[b + 6u], sky_basis[b + 7u], sky_basis[b + 8u]));
	float mean_sky = dot(m, vec3(sky_basis[b + 9u], sky_basis[b + 10u], sky_basis[b + 11u]));
	sky_share_moments[i] = vec2(max(moment - mean_bsdf * mean_bsdf - mean_sky * mean_sky, 0.0), mean_sky);
}

#endif // MODE_COMBINE
