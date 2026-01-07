
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contributors:
# Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
#

#!/bin/bash
set -euo pipefail


GREEN_BOLD="\033[1;32m"
RESET_STYLE="\033[0m"
TERMINAL_WIDTH=$(tput cols)

echo -e "${GREEN_BOLD}$(printf '=%.0s' $(seq 1 $TERMINAL_WIDTH))${RESET_STYLE}"

SUCCESS_TEXT="Dev Container Build Success"
TEXT_LENGTH=${#SUCCESS_TEXT}
OFFSET=$(( (TERMINAL_WIDTH - TEXT_LENGTH) / 2 ))
echo -e "${GREEN_BOLD}$(printf ' %.0s' $(seq 1 $OFFSET))${SUCCESS_TEXT}${RESET_STYLE}"

echo -e "${GREEN_BOLD}$(printf '=%.0s' $(seq 1 $TERMINAL_WIDTH))${RESET_STYLE}"

# Ensure Meson WrapDB dependencies exist in the workspace.
# The workspace is bind-mounted, so this will populate /workspace/subprojects/.
cd /workspace

if command -v meson >/dev/null 2>&1; then
  # Install googletest + imgui via Meson wraps (idempotent).
  # Note: Meson wrap install does NOT support --force on some versions.
  if [ ! -d "subprojects/googletest-1.17.0" ]; then
    meson wrap install gtest
  fi
  if [ ! -d "subprojects/imgui-1.91.6" ]; then
    meson wrap install imgui
  fi
else
  echo "WARNING: meson not found in PATH; skip Meson wrap install." >&2
fi
