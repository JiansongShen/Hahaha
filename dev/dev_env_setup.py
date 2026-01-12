# Copyright (c) 2026 Contributors of hahaha(https://github.com/Napbad/Hahaha)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contributors:
# Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
# jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
#
import argparse
import logging
import os
import subprocess
from pathlib import Path

VcpkgRepoUrl = "https://github.com/Microsoft/vcpkg"
Dependencies = [
    "gtest",
    "imgui[glfw-binding,opengl3-binding]",
]

VcpkgRootDirName = "vcpkg_root"

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

script_path = os.path.dirname(os.path.abspath(__file__))
default_workdir = os.path.join(script_path, "..")

def _git_clone_repo(
        url: str,
        depth: int = 1,
        dest_dir: Path = Path("."),
) -> bool:
    result = True
    command = [
        "git",
        "clone",
        "--depth",
        str(depth),
        url,
        dest_dir,
    ]
    command_res = subprocess.run(
        command,
        capture_output=True,
        check=False,
    )
    command_stdout = command_res.stdout.decode("utf-8") if command_res.stdout else ""
    command_stderr = command_res.stderr.decode("utf-8") if command_res.stderr else ""
    if command_res.returncode != 0:
        result = False
        logger.error(f"Failed to clone repo: {url},"
                      f"stdout of git clone command: \n\t{command_stdout}, \n"
                      f"stderr of git clone command: \n\t{command_stderr}")
    return result


def _download_vcpkg_root_repo(vcpkg_root_path: Path) -> None:
    dest = vcpkg_root_path / VcpkgRootDirName
    if dest.exists():
        logger.info(f"vcpkg root directory already exists: {dest}")
        return

    _git_clone_repo(
        VcpkgRepoUrl,
        depth=1,
        dest_dir=dest,
    )


def _check_and_bootstrap_vcpkg(vcpkg_root_path: Path) -> bool:
    """
    Check if vcpkg is ready. If local vcpkg repo exists but binary is missing, bootstrap it.
    """
    local_vcpkg_root = vcpkg_root_path / VcpkgRootDirName
    local_vcpkg_exe = local_vcpkg_root / "vcpkg"

    # 1. Check if we have a local vcpkg repo
    if not local_vcpkg_root.exists():
        # If no local repo, we can't bootstrap.
        # But maybe the user relies on system vcpkg?
        # Let's check system vcpkg
        if subprocess.run(["vcpkg", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            return True
        else:
             logger.error(f"vcpkg not found. Local repo not at {local_vcpkg_root}, and 'vcpkg' not in PATH.")
             return False

    # 2. Local repo exists. Check for binary.
    if not local_vcpkg_exe.exists():
        logger.info(f"Local vcpkg repo found at {local_vcpkg_root}, but 'vcpkg' binary is missing. Bootstrapping...")

        bootstrap_script = "./bootstrap-vcpkg.sh"
        if os.name == 'nt':
            bootstrap_script = ".\\bootstrap-vcpkg.bat"

        try:
            subprocess.run(
                bootstrap_script,
                shell=True,
                check=True,
                cwd=str(local_vcpkg_root)
            )
            logger.info("vcpkg bootstrapped successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to bootstrap vcpkg: {e}")
            return False

    return True


def _vcpkg_install_pkg(pkg_name: str, vcpkg_root_path: Path) -> bool:
    # Prefer local vcpkg if available, otherwise system vcpkg
    local_vcpkg_exe = vcpkg_root_path / VcpkgRootDirName / "vcpkg"
    vcpkg_cmd = str(local_vcpkg_exe) if local_vcpkg_exe.exists() else "vcpkg"

    command_str = f"{vcpkg_cmd} install {pkg_name} --recurse"
    command_env = os.environ.copy()
    command_env["VCPKG_ROOT"] = str(vcpkg_root_path / VcpkgRootDirName)

    logger.info(f"running: [{command_str}] (VCPKG_ROOT={command_env['VCPKG_ROOT']})")
    res = subprocess.run(
        command_str,
        shell=True,
        check=False,
        env=command_env,
        capture_output=True,
    )
    if res.returncode == 0:
        logger.info(f"Successfully installed [{pkg_name}]")
        return True

    res_stdout = res.stdout.decode("utf-8") if res.stdout else ""
    res_stderr = res.stderr.decode("utf-8") if res.stderr else ""

    logger.error(
        f"Failed to install [{pkg_name}] \n"
        f"stdout of vcpkg command: \n\t{res_stdout}, \n"
        f"stderr of vcpkg command: \n\t{res_stderr}  \n"
    )

    return False


def _download_dependencies_via_vcpkg(vcpkg_root_path: Path) -> bool:
    if _check_and_bootstrap_vcpkg(vcpkg_root_path):
        for pkg_name in Dependencies:
            if not _vcpkg_install_pkg(pkg_name, vcpkg_root_path):
                return False
        return True
    return False


def main() -> None:

    logger.info("Start to setup develop environments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir",
                        default=default_workdir,
                        help="Working directory, "
                             "if you are using devcontainer, then it is /workspace, "
                             "otherwise, it will be the path that you clone the repo to "
                             "e.g. (/home/username/project/Hahaha)")

    parser.add_argument(
        "--vcpkg-root",
        default=default_workdir + "/vcpkg/",
        help="Path to vcpkg root directory, "
             "if you are using devcontainer, then it is /workspace/vcpkg, "
             "otherwise, it will be the path that you clone the repo to "
             "e.g. (/home/username/project/Hahaha/vcpkg)"
             ", you can set up this by yourself",
    )

    args = parser.parse_args()
    work_dir = Path(args.workdir)
    vcpkg_root_path = Path(args.vcpkg_root)

    logger.info(f"Working directory: {work_dir}")
    logger.info(f"Vcpkg root directory: {vcpkg_root_path}")

    _download_vcpkg_root_repo(vcpkg_root_path)
    _download_dependencies_via_vcpkg(vcpkg_root_path)


if __name__ == "__main__":
    main()
