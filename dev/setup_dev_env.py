#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import platform

# --- Colors for output ---
GREEN_BOLD = "\033[1;32m"
YELLOW_BOLD = "\033[1;33m"
RED_BOLD = "\033[1;31m"
RESET_STYLE = "\033[0m"

REQUIRED_TOOLS = [
    "git", "make", "cmake", "gdb", "python3", "pip3", "ninja",
    "pkg-config", "curl", "wget", "clang", "clangd", "clang-format",
    "meson", "pre-commit", "gcovr", "valgrind"
]

def log_info(msg):
    print(f"{GREEN_BOLD}[INFO] {msg}{RESET_STYLE}")

def log_warn(msg):
    print(f"{YELLOW_BOLD}[WARN] {msg}{RESET_STYLE}")

def log_error(msg):
    print(f"{RED_BOLD}[ERROR] {msg}{RESET_STYLE}")

def check_sudo():
    if os.geteuid() != 0:
        if shutil.which("sudo"):
            return "sudo"
        else:
            log_warn("Running without root privileges. System package installation might fail.")
            return ""
    return ""

def run_cmd(cmd, check=True, check_output=False):
    log_info(f"Running: {cmd}")
    try:
        if check_output:
            return subprocess.check_output(cmd, shell=True, text=True, executable='/bin/bash').strip()
        subprocess.run(cmd, shell=True, check=check, executable='/bin/bash')
        return True
    except subprocess.CalledProcessError as e:
        if check:
            raise e
        return False

def configure_git():
    """Fixes 'detected dubious ownership' errors in containers."""
    log_info("Configuring git safe.directory...")
    run_cmd("git config --global --add safe.directory '*'", check=False)

def install_system_deps(sudo_cmd):
    log_info("Checking and installing system dependencies...")

    packages_apt = [
        "build-essential", "git", "cmake", "gdb", "wget", "curl",
        "libcurl4-openssl-dev", "libc-bin", "vim", "python3", "swig",
        "python3-pip", "python3-venv", "clang", "meson", "fish",
        "clangd", "clang-format", "libglfw3-dev", "libgl1-mesa-dev",
        "libglu1-mesa-dev", "pkg-config", "pipx", "gcovr", "valgrind",
        "ninja-build"
    ]

    packages_pacman = [
        "base-devel", "git", "cmake", "gdb", "wget", "curl",
        "python", "swig", "python-pip", "clang", "meson", "fish",
        "glfw-x11", "mesa", "glu", "python-pipx", "gcovr", "valgrind",
        "ninja"
    ]

    if shutil.which("apt-get"):
        log_info("Detected apt-get.")
        run_cmd(f"{sudo_cmd} apt-get update")
        # Install in batch to save time
        run_cmd(f"{sudo_cmd} apt-get install -y {' '.join(packages_apt)}", check=False)

    elif shutil.which("pacman"):
        log_info("Detected pacman.")
        run_cmd(f"{sudo_cmd} pacman -Syu --noconfirm {' '.join(packages_pacman)}", check=False)
    else:
        log_warn("Unsupported package manager. Skipping system package installation.")

def ensure_local_bin_in_path():
    home_bin = os.path.expanduser("~/.local/bin")
    if home_bin not in os.environ["PATH"]:
        os.environ["PATH"] = f"{home_bin}:{os.environ['PATH']}"
        log_info(f"Temporarily added {home_bin} to PATH")

def install_python_tools(sudo_cmd):
    log_info("Installing Python-based tools (Meson, pre-commit, etc.)...")
    ensure_local_bin_in_path()

    tools = ["meson", "pre-commit", "gcovr"]

    for tool in tools:
        if shutil.which(tool):
            log_info(f"{tool} is already installed.")
            continue

        log_info(f"Installing {tool}...")
        if shutil.which("pipx"):
            # Use --force to avoid 'already installed' errors if the binary is missing but venv exists
            run_cmd(f"pipx install {tool} --force", check=False)
            run_cmd("pipx ensurepath", check=False)
        else:
            cmd = f"pip3 install {tool}"
            if sudo_cmd:
                # Try system install if sudo, else user
                try:
                    run_cmd(f"{sudo_cmd} {cmd} --break-system-packages")
                except:
                    run_cmd(f"{sudo_cmd} {cmd}")
            else:
                run_cmd(f"{cmd} --user")

def setup_pre_commit_hooks():
    log_info("Setting up pre-commit git hooks...")

    if not shutil.which("pre-commit"):
        log_error("pre-commit executable not found. Skipping hook installation.")
        return

    # Check if we are in a git repository
    if not os.path.isdir(".git"):
        log_warn("Current directory is not a git repository (no .git found). Skipping pre-commit install.")
        return

    try:
        run_cmd("pre-commit install")
    except subprocess.CalledProcessError:
        log_warn("Failed to run 'pre-commit install'. Make sure git is configured correctly.")

def setup_subprojects():
    log_info("Setting up Meson subprojects...")
    os.makedirs("subprojects", exist_ok=True)

    if not shutil.which("meson"):
        log_error("Meson not found, skipping subprojects download.")
        return

    subprojects = {
        "gtest": ["googletest-1.17.0", "googletest"],
        "imgui": ["imgui-1.91.6", "imgui"]
    }

    for name, dirs in subprojects.items():
        if any(os.path.isdir(os.path.join("subprojects", d)) for d in dirs):
            log_info(f"Subproject {name} seems populated.")
            continue

        wrap_file = os.path.join("subprojects", f"{name}.wrap")
        if os.path.exists(wrap_file):
            try:
                run_cmd(f"meson subprojects download {name}")
            except subprocess.CalledProcessError:
                log_warn(f"Failed to download {name} subproject")
        else:
            try:
                run_cmd(f"meson wrap install {name}")
            except subprocess.CalledProcessError:
                log_warn(f"Failed to install {name} wrap")

def verify_installation():
    log_info("Verifying installation...")
    all_ok = True
    print(f"{'Tool':<20} | {'Status':<10} | {'Path/Version'}")
    print("-" * 50)

    for tool in REQUIRED_TOOLS:
        path = shutil.which(tool)
        status = f"{GREEN_BOLD}OK{RESET_STYLE}" if path else f"{RED_BOLD}MISSING{RESET_STYLE}"
        if not path:
            all_ok = False
            path_str = "-"
        else:
            # Try to get version for some tools
            path_str = path

        print(f"{tool:<20} | {status:<19} | {path_str}")

    print("-" * 50)
    return all_ok

def main():
    # Ensure we are in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if os.path.isdir(project_root):
        os.chdir(project_root)

    log_info(f"Running setup from {os.getcwd()}")

    # 0. Configure git (critical for devcontainers)
    configure_git()

    sudo_cmd = check_sudo()

    # 1. System Dependencies
    install_system_deps(sudo_cmd)

    # 2. Python Tools (Meson, pre-commit)
    install_python_tools(sudo_cmd)

    # 3. Setup Hooks
    setup_pre_commit_hooks()

    # 4. Subprojects
    setup_subprojects()

    # 5. Verification
    if verify_installation():
        log_info("Development environment setup complete! ✨")
    else:
        log_warn("Some tools are missing. Please check the logs above.")

    log_info("You may need to restart your shell or source your profile to update PATH.")

if __name__ == "__main__":
    main()
