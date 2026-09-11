# engine/ — Godot fork submodule

This is a git submodule (Rman887/godot-rtx, branch `valence-main`) forked from NVIDIA-RTX/godot `nvidia-pt-dlss`: Godot 4.8-dev plus NVIDIA's path tracer and DLSS integration. The rules for changing it are in `../docs/engine-fork.md`; the map of the rendering code is in `../AGENTS.md`.

Before editing here, check whether the change can be made in `../game/` instead. If not: work on `valence-main`, prefix commits `VAL:`, keep the diff minimal, follow Godot style, do not reformat upstream lines, add a ledger row in `../docs/engine-fork.md`, then bump the submodule pointer from the root repo.

Never hand-edit `thirdparty/`. Build with `../tools/build.ps1`, not ad-hoc `scons` invocations. Runtime DLLs for Streamline are staged into `bin/` by `../tools/stage-deps.ps1`.
