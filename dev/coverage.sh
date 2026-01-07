#!/usr/bin/env bash
###
 # @Author: napbad napbad.sen@gmail.com
 # @Date: 2026-01-07 11:07:08
 # @LastEditors: napbad napbad.sen@gmail.com
 # @LastEditTime: 2026-01-07 11:28:49
 # @FilePath: /Hahaha/dev/coverage.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
###
set -euo pipefail

# Usage:
#   ./dev/coverage.sh [builddir]
#
# Policy:
# - Focus on "core" and exclude display (UI/visualization) code from coverage.
# - Use gcovr merge options to avoid double-counting template instantiations
#   across multiple translation units.

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
  --merge-lines
  --exclude-unreachable-branches
  --filter 'core/.'
  --exclude 'extern/.'
  --exclude 'examples/.'
  --exclude 'core/src/display/.'
  --exclude 'core/include/display/.'
)

echo "== line coverage (core, exclude display) =="
gcovr "${GCOVR_COMMON_ARGS[@]}" --txt-metric line --fail-under-line 85

echo
echo "== branch coverage (core, exclude display) =="
gcovr "${GCOVR_COMMON_ARGS[@]}" --txt-metric branch --fail-under-branch 50
