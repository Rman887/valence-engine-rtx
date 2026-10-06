/**************************************************************************/
/*  luminance.h                                                           */
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

#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/effects/luminance_histogram.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/luminance_reduce.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/luminance_reduce_raster.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"

#define RB_LUMINANCE_BUFFERS SNAME("luminance_buffers")

namespace RendererRD {

class Luminance {
private:
	bool prefer_raster_effects;

	enum LuminanceReduceMode {
		LUMINANCE_REDUCE_READ,
		LUMINANCE_REDUCE,
		LUMINANCE_REDUCE_WRITE,
		LUMINANCE_REDUCE_MAX
	};

	struct LuminanceReducePushConstant {
		int32_t source_size[2];
		float max_luminance;
		float min_luminance;
		float exposure_adjust;
		float pad[3];
	};

	struct LuminanceReduce {
		LuminanceReduceShaderRD shader;
		RID shader_version;
		RID pipelines[LUMINANCE_REDUCE_MAX];
	} luminance_reduce;

	enum LuminanceReduceRasterMode {
		LUMINANCE_REDUCE_FRAGMENT_FIRST,
		LUMINANCE_REDUCE_FRAGMENT,
		LUMINANCE_REDUCE_FRAGMENT_FINAL,
		LUMINANCE_REDUCE_FRAGMENT_MAX
	};

	struct LuminanceReduceRasterPushConstant {
		int32_t source_size[2];
		int32_t dest_size[2];
		float exposure_adjust;
		float min_luminance;
		float max_luminance;
		uint32_t pad1;
	};

	struct LuminanceReduceFragment {
		LuminanceReduceRasterShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipelines[LUMINANCE_REDUCE_FRAGMENT_MAX];
	} luminance_reduce_raster;

	enum LuminanceHistogramMode {
		LUMINANCE_HISTOGRAM_BLOCK,
		LUMINANCE_HISTOGRAM_BIN,
		LUMINANCE_HISTOGRAM_RESOLVE,
		LUMINANCE_HISTOGRAM_MAX
	};

	struct LuminanceHistogramPushConstant {
		int32_t source_size[2];
		float log_min;
		float log_max;
		float to_scene;
		float clamp_log_min;
		float clamp_log_max;
		float adjust_up;
		float adjust_down;
		float low_fraction;
		float high_fraction;
		float source_scale; // Undoes the frame's pre-exposure on the metered image.
	};

	struct LuminanceHistogram {
		LuminanceHistogramShaderRD shader;
		RID shader_version;
		RID pipelines[LUMINANCE_HISTOGRAM_MAX];
	} luminance_histogram;

public:
	class LuminanceBuffers : public RenderBufferCustomDataRD {
		GDCLASS(LuminanceBuffers, RenderBufferCustomDataRD);

	private:
		bool prefer_raster_effects;

	public:
		Vector<RID> reduce;
		RID current;

		// Histogram metering: block means of the metered image and the bin counts.
		RID blocks;
		Size2i blocks_size;
		RID histogram;

		// False until a frame is metered after (re)configuration, while current holds no value.
		bool metered = false;

		virtual void configure(RenderSceneBuffersRD *p_render_buffers) override;
		virtual void free_data() override;

		void set_prefer_raster_effects(bool p_prefer_raster_effects);
	};

	Ref<LuminanceBuffers> get_luminance_buffers(Ref<RenderSceneBuffersRD> p_render_buffers);
	RID get_current_luminance_buffer(Ref<RenderSceneBuffersRD> p_render_buffers);
	void luminance_reduction(RID p_source_texture, const Size2i p_source_size, Ref<LuminanceBuffers> p_luminance_buffers, float p_min_luminance, float p_max_luminance, float p_adjust, bool p_set = false);

	struct HistogramSettings {
		float min_luminance = 0.0; // The camera's bounds, scene luminance.
		float max_luminance = 0.0;
		float to_scene = 1.0; // Image units to scene luminance.
		float source_scale = 1.0; // The metered image times this is in image units: 1 / the frame's pre-exposure.
		float adjust_up = 1.0;
		float adjust_down = 1.0;
		float low_fraction = 0.5;
		float high_fraction = 0.95;
	};

	bool can_use_histogram() const { return !prefer_raster_effects; }
	bool luminance_histogram_metering(RID p_source_texture, const Size2i p_source_size, Ref<LuminanceBuffers> p_luminance_buffers, const HistogramSettings &p_settings);

	Luminance(bool p_prefer_raster_effects);
	~Luminance();
};

} // namespace RendererRD
