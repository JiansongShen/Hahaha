#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./dev/coverage.sh [builddir] [--clean]
#
# Policy:
# - Focus on "core" and exclude display (UI/visualization) code from coverage.
# - Use gcovr merge options to avoid double-counting template instantiations
#   across multiple translation units.
# --clean means clean the builddir before running.


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-builddir}"

cd "${ROOT_DIR}"

if [ -d "${BUILD_DIR}" ]; then
  meson setup --reconfigure "${BUILD_DIR}" -Db_coverage=true -Ddisplay=false
else
  meson setup "${BUILD_DIR}" --buildtype=debug -Db_coverage=true -Ddisplay=false
fi
meson compile -C "${BUILD_DIR}" tests/hahaha_tests
meson test -C "${BUILD_DIR}" -v --no-rebuild

GCOVR_COMMON_ARGS=(
  -r .
  --merge-mode-functions=merge-use-line-min
  --exclude-unreachable-branches
  --exclude-noncode-lines
  --filter 'core/.'
  --exclude 'subprojects/.'
  --exclude 'examples/.'
  --exclude 'core/src/display/.'
  --exclude 'core/include/display/.'
  # --html-details
  # -o cover/coverage.html
)

echo "== line coverage (core, exclude display) =="
gcovr "${GCOVR_COMMON_ARGS[@]}" --txt-metric line --fail-under-line 80

echo
echo "== branch coverage (core, exclude display) =="
gcovr "${GCOVR_COMMON_ARGS[@]}" --txt-metric branch --fail-under-branch 35
