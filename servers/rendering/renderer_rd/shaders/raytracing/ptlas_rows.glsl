#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// The per-instance shader rows of slot-resident instances (D94b): the path tracer's geometry,
// material and motion index buffers are indexed by PTLAS slot and persist across frames, and a
// frame uploads only the rows that changed, as one list of (slot, rows) this pass scatters. A row
// write carries the geometry and material rows when bit 0 of `parts` is set, and the motion index
// always (a slot that moved last frame and not this one gets -1 back). Must match RT_SlotRowWrite
// (render_raytracing.h): 16 bytes of header, the 128-byte geometry row and the 96-byte material row.

struct RowWrite {
	uint slot;
	uint parts;
	int motion_index;
	uint pad;
	uvec4 geometry[8];
	uvec4 material[6];
};

layout(set = 0, binding = 0, std430) restrict readonly buffer Rows {
	RowWrite rows[];
}
row_writes;

layout(set = 0, binding = 1, std430) restrict writeonly buffer Geometries {
	uvec4 geometries[];
}
geometry_rows;

layout(set = 0, binding = 2, std430) restrict writeonly buffer Materials {
	uvec4 materials[];
}
material_rows;

layout(set = 0, binding = 3, std430) restrict writeonly buffer MotionIndices {
	int motion_indices[];
}
motion_rows;

layout(push_constant, std430) uniform Params {
	uint count;
	uint pad0;
	uint pad1;
	uint pad2;
}
params;

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= params.count) {
		return;
	}
	uint slot = row_writes.rows[i].slot;
	if ((row_writes.rows[i].parts & 1u) != 0u) {
		for (uint k = 0u; k < 8u; k++) {
			geometry_rows.geometries[slot * 8u + k] = row_writes.rows[i].geometry[k];
		}
		for (uint k = 0u; k < 6u; k++) {
			material_rows.materials[slot * 6u + k] = row_writes.rows[i].material[k];
		}
	}
	motion_rows.motion_indices[slot] = row_writes.rows[i].motion_index;
}
