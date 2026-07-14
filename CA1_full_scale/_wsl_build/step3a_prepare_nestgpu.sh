#!/usr/bin/env bash
# Step 3a — restore the NEST-GPU fork (upstream 90f87ab + tracked patch) and
# inspect its CMake options so step 3b configures with the correct flag names.
# No compilation here (fast).
set -euo pipefail

ROOT="$HOME/ca1_full_scale"
NG="$ROOT/nest-gpu"
PATCH="$ROOT/nest-gpu-patches/nest-gpu-local-mods.patch"

echo "== patch present? =="
ls -l "$PATCH"

echo "== clone upstream nest-gpu (full history for checkout) =="
rm -rf "$NG"
git clone --quiet https://github.com/nest/nest-gpu.git "$NG"

echo "== checkout base 90f87ab =="
git -C "$NG" checkout --quiet 90f87ab
git -C "$NG" --no-pager log --oneline -1

echo "== apply tracked fork patch =="
git -C "$NG" apply --stat "$PATCH" | tail -5
git -C "$NG" apply "$PATCH"
echo "patch applied OK"

echo "== confirm user models present (theta stack) =="
ls "$NG"/src/user_m2*.* "$NG"/src/user_m3*.* "$NG"/src/user_m4*.* "$NG"/src/user_m5*.* 2>/dev/null | sed 's#.*/##' | tr '\n' ' '; echo

echo "== CMake options (option()/arch/python/mpi) =="
grep -rniE "option\(|cuda[_-]?arch|compute_|sm_|with[-_]python|with[-_]mpi|Python3?_EXECUTABLE|CMAKE_CUDA_ARCHITECTURES" \
    "$NG/CMakeLists.txt" "$NG"/cmake/*.cmake 2>/dev/null | head -40 || echo "(none matched — inspect manually)"

echo "== STEP3A DONE =="
