/**************************************************************************/
/*  test_dependency_tracker.cpp                                           */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_dependency_tracker)

#include "servers/rendering/storage/utilities.h"

namespace TestDependencyTracker {

static int changed_count = 0;

static void count_changed(Dependency::DependencyChangedNotification p_notification, DependencyTracker *p_tracker) {
	changed_count++;
}

TEST_CASE("[DependencyTracker] remove_dependency drops only the dependency it is given") {
	Dependency kept;
	Dependency dropped;
	DependencyTracker tracker;
	tracker.changed_callback = count_changed;

	tracker.update_begin();
	tracker.update_dependency(&kept);
	tracker.update_dependency(&dropped);
	tracker.update_end();

	tracker.remove_dependency(&dropped);
	tracker.remove_dependency(&dropped); // Not tracked any more: nothing changes.

	changed_count = 0;
	dropped.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
	CHECK_MESSAGE(changed_count == 0, "The removed dependency no longer reaches the tracker.");
	kept.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
	CHECK_MESSAGE(changed_count == 1, "The other dependency still reaches the tracker.");

	// A compatible mesh swap's sequence: the old mesh removed, the new one added outside an
	// update_begin/update_end pair.
	Dependency added;
	tracker.update_dependency(&added);
	changed_count = 0;
	added.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
	CHECK_MESSAGE(changed_count == 1, "A dependency added outside an update reaches the tracker.");

	// A later full update that re-registers it keeps it; one that does not drops it.
	tracker.update_begin();
	tracker.update_dependency(&kept);
	tracker.update_dependency(&added);
	tracker.update_end();
	changed_count = 0;
	added.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
	CHECK_MESSAGE(changed_count == 1, "A full update that re-registers it keeps it.");
	tracker.update_begin();
	tracker.update_dependency(&kept);
	tracker.update_end();
	changed_count = 0;
	added.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
	CHECK_MESSAGE(changed_count == 0, "A full update that does not re-register it drops it.");

	// A later full update keeps what it updates and drops the rest, as before.
	tracker.update_begin();
	tracker.update_dependency(&dropped);
	tracker.update_end();
	changed_count = 0;
	kept.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
	dropped.changed_notify(Dependency::DEPENDENCY_CHANGED_MESH);
	CHECK(changed_count == 1);
	tracker.clear();
}

} // namespace TestDependencyTracker
