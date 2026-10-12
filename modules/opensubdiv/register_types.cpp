/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "subdivision_builder.h"

#include "core/object/class_db.h"

#include <opensubdiv/far/error.h>

// OpenSubdiv reports topology errors and warnings through process-wide callbacks, printing to
// stdout by default; route them to Godot's log.
static void _opensubdiv_error(OpenSubdiv::Far::ErrorType p_type, const char *p_message) {
	ERR_PRINT(vformat("OpenSubdiv error %d: %s", int(p_type), String::utf8(p_message)));
}

static void _opensubdiv_warning(const char *p_message) {
	WARN_PRINT(vformat("OpenSubdiv: %s", String::utf8(p_message)));
}

void initialize_opensubdiv_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	OpenSubdiv::Far::SetErrorCallback(_opensubdiv_error);
	OpenSubdiv::Far::SetWarningCallback(_opensubdiv_warning);
	GDREGISTER_CLASS(SubdivisionBuilder);
}

void uninitialize_opensubdiv_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	OpenSubdiv::Far::SetErrorCallback(nullptr);
	OpenSubdiv::Far::SetWarningCallback(nullptr);
}
