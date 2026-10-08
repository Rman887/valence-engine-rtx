/**************************************************************************/
/*  shader_compiler.h                                                     */
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

#include "core/templates/pair.h"
#include "servers/rendering/rendering_server_enums.h"
#include "servers/rendering/shader_language.h"

class ShaderCompiler {
public:
	enum Stage {
		STAGE_VERTEX,
		STAGE_FRAGMENT,
		STAGE_COMPUTE,
		STAGE_RAYGEN,
		STAGE_ANY_HIT,
		STAGE_CLOSEST_HIT,
		STAGE_MISS,
		STAGE_INTERSECTION,
		STAGE_MAX
	};

	struct IdentifierActions {
		HashMap<StringName, Stage> entry_point_stages;

		HashMap<StringName, Pair<int *, int>> render_mode_values;
		HashMap<StringName, bool *> render_mode_flags;
		HashMap<StringName, bool *> usage_flag_pointers;
		HashMap<StringName, bool *> write_flag_pointers;
		HashMap<StringName, Pair<int *, int>> stencil_mode_values;
		int *stencil_reference = nullptr;

		HashMap<StringName, ShaderLanguage::ShaderNode::Uniform> *uniforms = nullptr;
	};

	struct GeneratedCode {
		Vector<String> defines;
		struct Texture {
			StringName name;
			ShaderLanguage::DataType type = ShaderLanguage::DataType::TYPE_VOID;
			ShaderLanguage::ShaderNode::Uniform::Hint hint = ShaderLanguage::ShaderNode::Uniform::Hint::HINT_NONE;
			bool use_color = false;
			ShaderLanguage::TextureFilter filter = ShaderLanguage::TextureFilter::FILTER_DEFAULT;
			ShaderLanguage::TextureRepeat repeat = ShaderLanguage::TextureRepeat::REPEAT_DEFAULT;
			bool global = false;
			int array_size = 0;
		};

		Vector<Texture> texture_uniforms;

		Vector<uint32_t> uniform_offsets;
		uint32_t uniform_total_size = 0;
		String uniforms;
		String stage_globals[STAGE_MAX];

		HashMap<String, String> code;

		bool uses_global_textures = false;
		bool uses_fragment_time = false;
		bool uses_vertex_time = false;
		bool uses_screen_texture_mipmaps = false;
		bool uses_screen_texture = false;
		bool uses_depth_texture = false;
		bool uses_normal_roughness_texture = false;
	};

	struct DefaultIdentifierActions {
		HashMap<StringName, String> renames;
		HashMap<StringName, String> render_mode_defines;
		HashMap<StringName, String> usage_defines;
		HashMap<StringName, String> custom_samplers;
		ShaderLanguage::TextureFilter default_filter = ShaderLanguage::TextureFilter::FILTER_NEAREST;
		ShaderLanguage::TextureRepeat default_repeat = ShaderLanguage::TextureRepeat::REPEAT_DISABLE;
		int base_texture_binding_index = 0;
		int texture_layout_set = 0;
		String base_uniform_string;
		String global_buffer_array_variable;
		String instance_uniform_index_variable;
		uint32_t base_varying_index = 0;
		bool suppress_varying_io = false;
		bool apply_luminance_multiplier = false;
		bool check_multiview_samplers = false;
		// When set, texture(s, uv) on a float 2D or 2D array sampler whose coordinate reads one of
		// texture_lod_coordinates directly becomes textureLod at the level this function returns for
		// textureSize(s, 0), and texture(s, uv, bias) on any sampler but an external one becomes textureLod
		// at the bias (plus that level where it applies). The path tracer's hit groups pick a level of
		// detail there, and their stages accept no bias form (D116).
		String texture_lod_function;
		HashSet<StringName> texture_lod_coordinates;
		// When set, the lookups above through a linear anisotropic sampler, without a bias, whose coordinate reads one
		// of texture_grad_coordinates and none of the other texture_lod_coordinates become
		// (v.on ? textureGrad(s, uv, v.x, v.y) : <the level form>), v being this variable: the hit's UV gradients
		// where it has them (D116). A usage flag pointer under this variable's name is set when a call was rewritten.
		String texture_grad_variable;
		HashSet<StringName> texture_grad_coordinates;
		// The spatial stage function texture_projected(sampler2D, vec2 uv, float uv_per_metre) is texture(s, uv),
		// or when this is set, textureLod at the level this function returns for textureSize(s, 0) and
		// uv_per_metre (the path tracer's hit groups, D116).
		String texture_projected_function;
		// When set, the level, gradient and projected forms above on a float 2D uniform of material data (source_color,
		// hint_normal, hint_roughness_* or hint_anisotropy, through a linear filter) call these
		// in place of textureLod and textureGrad, as f(texture, its sampler, the nearest-with-mipmaps sampler of its
		// repeat mode, uv, then the level and whether the filter has mipmaps, or the two gradients): the path
		// tracer's stochastic texture filtering (D117). A usage flag pointer under the level function's name is set
		// when a call was rewritten.
		String texture_filtering_lod_function;
		String texture_filtering_grad_function;
	};

private:
	ShaderLanguage parser;

	String _get_sampler_name(ShaderLanguage::TextureFilter p_filter, ShaderLanguage::TextureRepeat p_repeat);
	String _get_texture_filtering_arguments(const String &p_texture, bool p_color, ShaderLanguage::ShaderNode::Uniform::Hint p_hint, ShaderLanguage::TextureFilter p_filter, ShaderLanguage::TextureRepeat p_repeat, bool &r_mipmaps);

	void _dump_function_deps(const ShaderLanguage::ShaderNode *p_node, const StringName &p_for_func, const HashMap<StringName, String> &p_func_code, String &r_to_add, HashSet<StringName> &added);
	String _dump_node_code(const ShaderLanguage::Node *p_node, int p_level, GeneratedCode &r_gen_code, IdentifierActions &p_actions, const DefaultIdentifierActions &p_default_actions, bool p_assigning, bool p_scope = true);

	const ShaderLanguage::ShaderNode *shader = nullptr;
	const ShaderLanguage::FunctionNode *function = nullptr;
	StringName current_func_name;
	StringName time_name;
	HashSet<StringName> texture_functions;

	HashSet<StringName> used_name_defines;
	HashSet<StringName> used_flag_pointers;
	HashSet<StringName> used_rmode_defines;
	HashSet<StringName> internal_functions;
	HashSet<StringName> fragment_varyings;

	DefaultIdentifierActions actions;

	static ShaderLanguage::DataType _get_global_shader_uniform_type(const StringName &p_name);

public:
	Error compile(RSE::ShaderMode p_mode, const String &p_code, IdentifierActions *p_actions, const String &p_path, GeneratedCode &r_gen_code);

	void initialize(DefaultIdentifierActions p_actions);
	ShaderCompiler();
};
