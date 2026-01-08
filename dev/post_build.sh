
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
  # Install/download googletest + imgui via Meson WrapDB (idempotent).
  #
  # IMPORTANT:
  # - `meson wrap install <name>` FAILS (exit=1) when the wrap file already
  #   exists in the repo (e.g. subprojects/gtest.wrap).
  # - `meson subprojects download <name>` is the correct command when wrap
  #   files are already tracked in git.
  mkdir -p subprojects

  # gtest
  if compgen -G "subprojects/googletest-*" > /dev/null; then
    echo "gtest subproject already present."
  else
    if [ -f "subprojects/gtest.wrap" ]; then
      meson subprojects download gtest
    else
      meson wrap install gtest
    fi
  fi

  # imgui
  if compgen -G "subprojects/imgui-*" > /dev/null; then
    echo "imgui subproject already present."
  else
    if [ -f "subprojects/imgui.wrap" ]; then
      meson subprojects download imgui
    else
      meson wrap install imgui
    fi
  fi
else
  echo "WARNING: meson not found in PATH; skip Meson wrap install." >&2
fi
