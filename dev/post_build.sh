
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
# set -euo pipefail


GREEN_BOLD="\033[1;32m"
RESET_STYLE="\033[0m"
TERMINAL_WIDTH=$(tput cols)

echo -e "${GREEN_BOLD}$(printf '=%.0s' $(seq 1 $TERMINAL_WIDTH))${RESET_STYLE}"

SUCCESS_TEXT="Dev Container Build Success"
TEXT_LENGTH=${#SUCCESS_TEXT}
OFFSET=$(( (TERMINAL_WIDTH - TEXT_LENGTH) / 2 ))
echo -e "${GREEN_BOLD}$(printf ' %.0s' $(seq 1 $OFFSET))${SUCCESS_TEXT}${RESET_STYLE}"

echo -e "${GREEN_BOLD}$(printf '=%.0s' $(seq 1 $TERMINAL_WIDTH))${RESET_STYLE}"

# Run the comprehensive setup script
SCRIPT_DIR=$(dirname "$0")
# Since this script and dev_env_setup.py are both in dev/, we can call it directly
SETUP_SCRIPT="$SCRIPT_DIR/dev_env_setup.py"

if [ -f "$SETUP_SCRIPT" ]; then
    python3 "$SETUP_SCRIPT"
fi
