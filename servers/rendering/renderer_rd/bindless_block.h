/**************************************************************************/
/*  bindless_block.h                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/rendering_device.h"

class BindlessBlock {
public:
	constexpr static uint32_t MAX_BINDLESS_TEXTURES = 128000;

	uint32_t add_texture(RID p_texture);

	void initialize(RenderingDevice *p_rd);

	void begin_frame();

	void finalize(RID p_shader, uint32_t p_set_index);

	RID get_uniform_set() const { return uniform_set; }
	uint32_t get_texture_count() const { return textures.size(); }
	// VAL: the log2 of each slot's width and height (TexelLog), the term a ray-cone level adds per texture (D116, half
	// their sum) and the per-axis sizes a fetch's gradients need (GPU audit hot 5), so a hit reads two floats instead of
	// querying the texture's size per lookup. Uploaded when a slot changes.
	RID get_texel_log_buffer();
	bool is_initialized() const { return rd != nullptr; }
	bool is_finalized() const { return uniform_set.is_valid(); }

	void clear();
	~BindlessBlock();

private:
	RenderingDevice *rd = nullptr;
	LocalVector<RID> textures;
	// The shaders' vec2 per slot (std430: 8 bytes, no padding): floats, not Vector2, whose real_t is a double in a
	// double build.
	struct TexelLog {
		float width = 0.0f;
		float height = 0.0f;
	};
	LocalVector<TexelLog> texel_log;
	RID texel_log_buffer;
	uint32_t texel_log_capacity = 0;
	bool texel_log_dirty = true;
	HashMap<RID, uint32_t> texture_to_index;
	LocalVector<uint32_t> free_indices;
	uint32_t free_indices_count = 0; // logical top to avoid resize of free_indices array
	RID default_texture;
	RID uniform_set;
	bool needs_refinalize = false;

	TexelLog _texel_log(RID p_texture) const;
};
