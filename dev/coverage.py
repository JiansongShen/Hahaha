#!/usr/bin/env python3
"""
Coverage script for Hahaha project.

Usage:
    python dev/coverage.py [builddir] [--clean] [--cuda|--no-cuda]

    Use an existing build directory that already passes tests, e.g.:
    python dev/coverage.py cmake-build-vcpkg-windows

Policy:
- Focus on "core" and exclude display (UI/visualization) code from coverage.
- Use gcovr merge options to avoid double-counting template instantiations
  across multiple translation units.
--clean means clean the builddir before running.
--cuda enables CUDA support for coverage testing.
--no-cuda disables CUDA support for coverage testing.
Default behavior: auto-detect CUDA availability and enable if possible.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def get_root_dir():
    """Get the project root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent.resolve()


def detect_cuda():
    """Auto-detect CUDA availability (nvcc + toolkit path for CMake/VS)."""
    # On Windows with Visual Studio, CMake needs the CUDA toolkit directory,
    # not just nvcc in PATH. Check environment variables first, then standard paths.
    if platform.system() == "Windows":
        # Check CUDA_PATH environment variable (most reliable)
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            cuda_dir = Path(cuda_path)
            if cuda_dir.exists():
                nvcc = cuda_dir / "bin" / "nvcc.exe"
                if nvcc.exists():
                    return True
        
        # Check versioned CUDA_PATH_V* environment variables
        for key, value in os.environ.items():
            if key.startswith("CUDA_PATH_") and value:
                cuda_dir = Path(value)
                if cuda_dir.exists():
                    nvcc = cuda_dir / "bin" / "nvcc.exe"
                    if nvcc.exists():
                        return True
        
        # Check standard CUDA installation paths
        cuda_base_paths = [
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
            Path("C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA"),
        ]
        for base in cuda_base_paths:
            if not base.exists():
                continue
            # Toolkit has versioned subdirs like v12.3
            try:
                subdirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("v")]
                if subdirs:
                    # Check nvcc exists in one of them (e.g. bin/nvcc.exe)
                    for ver in subdirs:
                        nvcc = ver / "bin" / "nvcc.exe"
                        if nvcc.exists():
                            return True
            except OSError:
                pass
        
        # Last resort: check if nvcc is in PATH (less reliable for VS)
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                capture_output=True,
            )
            # If nvcc works, check if we can find its installation path
            # by checking common locations relative to where nvcc might be
            return True  # nvcc found, assume it might work
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return False

    # Unix-like: nvcc in PATH and /opt/cuda or similar
    try:
        subprocess.run(
            ["nvcc", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        if Path("/opt/cuda").exists():
            return True
        # Check CUDA_PATH on Unix too
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path and Path(cuda_path).exists():
            return True
        return True  # nvcc found on Unix, assume toolkit is set up
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return False


def get_cmake_generator(build_dir: Path):
    """Get appropriate CMake generator for the platform."""
    # Check if CMakeCache.txt exists and read the generator
    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        try:
            with open(cmake_cache, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("CMAKE_GENERATOR:"):
                        match = re.search(r"CMAKE_GENERATOR:INTERNAL=(.+)", line)
                        if match:
                            generator = match.group(1).strip()
                            print(f"Using existing generator: {generator}")
                            return generator
        except Exception:
            pass  # If we can't read it, fall back to default

    # Choose generator based on platform
    if platform.system() == "Windows":
        # Try to detect Visual Studio version
        # Check for Visual Studio 2025 (VS 18)
        vs2025_paths = [
            Path("C:/Program Files/Microsoft Visual Studio/2025/Community"),
            Path("C:/Program Files/Microsoft Visual Studio/2025/Professional"),
            Path("C:/Program Files/Microsoft Visual Studio/2025/Enterprise"),
            Path("C:/Program Files/Microsoft Visual Studio/2025/Preview"),
        ]
        for vs_path in vs2025_paths:
            if vs_path.exists():
                return "Visual Studio 18 2026"
        
        # Check for Visual Studio 2022 (VS 17)
        vs2022_paths = [
            Path("C:/Program Files/Microsoft Visual Studio/2022/Community"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Professional"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise"),
        ]
        for vs_path in vs2022_paths:
            if vs_path.exists():
                return "Visual Studio 17 2022"
        
        # Check for Visual Studio 2019 (VS 16)
        vs2019_paths = [
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Community"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise"),
        ]
        for vs_path in vs2019_paths:
            if vs_path.exists():
                return "Visual Studio 16 2019"
        
        # Default to Visual Studio 17 2022
        return "Visual Studio 17 2022"
    else:
        # Unix-like systems use Ninja
        return "Ninja"


def clean_build_dir(build_dir: Path):
    """Clean the build directory if it exists."""
    if build_dir.exists():
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)


def configure_cmake(build_dir: Path, enable_cuda: str, capture_output: bool = False):
    """Configure CMake with appropriate settings.
    If capture_output is True, returns (success, stderr_text) for fallback handling.
    """
    generator = get_cmake_generator(build_dir)
    
    cmake_args = [
        "cmake",
        "-S",
        ".",
        "-B",
        str(build_dir),
        "-G",
        generator,
    ]
    
    # Visual Studio generators use --config instead of CMAKE_BUILD_TYPE
    if generator.startswith("Visual Studio"):
        cmake_args.extend([
            "-DHAHAHA_DISPLAY=OFF",
            "-DHAHAHA_BUILD_TESTS=ON",
            "-DHAHAHA_BUILD_EXAMPLES=OFF",
            "-DHAHAHA_ENABLE_COVERAGE=ON",
        ])
    else:
        cmake_args.extend([
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DHAHAHA_DISPLAY=OFF",
            "-DHAHAHA_BUILD_TESTS=ON",
            "-DHAHAHA_BUILD_EXAMPLES=OFF",
            "-DHAHAHA_ENABLE_COVERAGE=ON",
        ])

    if enable_cuda == "on":
        cmake_args.append("-DHAHAHA_USE_CUDA=ON")
        print("Building with CUDA support enabled.")
    else:
        cmake_args.append("-DHAHAHA_USE_CUDA=OFF")
        print("Building without CUDA support.")

    print(f"Running: {' '.join(cmake_args)}")
    if capture_output:
        result = subprocess.run(
            cmake_args,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0, (result.stderr or "") + (result.stdout or "")
    else:
        # Always capture output to show errors even when not using capture_output mode
        result = subprocess.run(
            cmake_args,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmake_args, result.stdout, result.stderr
            )


def build_project(build_dir: Path, parallel: int = 8):
    """Build the project."""
    print(f"Building project with {parallel} parallel jobs...")
    build_args = ["cmake", "--build", str(build_dir), "--parallel", str(parallel)]
    
    # Visual Studio generators need --config Debug for coverage
    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        try:
            with open(cmake_cache, "r", encoding="utf-8") as f:
                content = f.read()
                if "Visual Studio" in content:
                    build_args.extend(["--config", "Debug"])
        except Exception:
            pass
    
    subprocess.run(build_args, check=True)


def find_test_executable(build_dir: Path):
    """Find the test executable in the build directory."""
    # Common locations for test executables (single-config and multi-config)
    test_paths = [
        build_dir / "tests" / "Debug" / "hahaha_tests.exe",  # VS multi-config Debug
        build_dir / "tests" / "Release" / "hahaha_tests.exe",  # VS multi-config Release
        build_dir / "tests" / "hahaha_tests.exe",  # Single-config (Ninja, etc.) or VS run dir
        build_dir / "tests" / "Debug" / "hahaha_tests",  # Unix Debug
        build_dir / "tests" / "Release" / "hahaha_tests",  # Unix Release
        build_dir / "tests" / "hahaha_tests",  # Unix single-config
    ]
    
    for test_path in test_paths:
        if test_path.exists():
            return test_path
    return None


def run_tests(build_dir: Path, enable_cuda: str, continue_on_failure: bool = False):
    """Run tests.
    Returns True if tests passed, False otherwise.
    """
    if enable_cuda == "on":
        print("Running tests with CUDA enabled...")
    else:
        print("Running tests without CUDA...")

    ctest_args = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "--verbose",  # Show detailed test output
    ]
    
    # Visual Studio generators need -C Debug for coverage
    cmake_cache = build_dir / "CMakeCache.txt"
    config = "Debug"
    if cmake_cache.exists():
        try:
            with open(cmake_cache, "r", encoding="utf-8") as f:
                content = f.read()
                if "Visual Studio" in content:
                    ctest_args.extend(["-C", "Debug"])
        except Exception:
            pass
    
    try:
        result = subprocess.run(ctest_args, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        # Print ctest output
        if hasattr(e, 'stdout') and e.stdout:
            print(e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print(e.stderr, file=sys.stderr)
        
        # Try to run the test executable directly to get more detailed output
        test_exe = find_test_executable(build_dir)
        if test_exe:
            print("\n" + "=" * 70, file=sys.stderr)
            print("Tests failed. Running test executable directly for detailed output:", file=sys.stderr)
            print("=" * 70 + "\n", file=sys.stderr)
            try:
                # Run the test executable directly with output capture
                result = subprocess.run(
                    [str(test_exe)],
                    cwd=build_dir / "tests",
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                
                # Check for heap corruption or assertion failures
                output = (result.stdout or "") + (result.stderr or "")
                if "Debug Assertion Failed" in output or "heap" in output.lower() or "__acrt_first_block" in output:
                    print("\n" + "!" * 70, file=sys.stderr)
                    print("HEAP CORRUPTION DETECTED!", file=sys.stderr)
                    print("!" * 70, file=sys.stderr)
                    print(
                        "\nThis indicates a memory management bug (buffer overflow, use-after-free, etc.).\n"
                        "Debugging suggestions:\n"
                        "1. Run tests under a debugger (Visual Studio Debugger or WinDbg)\n"
                        "2. Enable AddressSanitizer (ASan) if available\n"
                        "3. Check for buffer overflows, uninitialized memory, or double-free errors\n"
                        "4. Use --continue-on-test-failure to generate coverage anyway\n",
                        file=sys.stderr,
                    )
            except subprocess.TimeoutExpired:
                print("Test executable timed out after 5 minutes.", file=sys.stderr)
            except Exception as ex:
                print(f"Could not run test executable directly: {ex}", file=sys.stderr)
        
        if continue_on_failure:
            print(
                f"\nWarning: Tests failed (exit code {e.returncode}), but continuing with coverage generation...",
                file=sys.stderr,
            )
            return False
        else:
            print(
                "\nTip: Use --continue-on-test-failure to generate coverage even if tests fail.",
                file=sys.stderr,
            )
            raise


def run_coverage(root_dir: Path, enable_cuda: str):
    """Run coverage analysis using gcovr."""
    gcovr_common_args = [
        "gcovr",
        "-r",
        ".",
        "--merge-mode-functions=merge-use-line-min",
        "--exclude-unreachable-branches",
        "--exclude-noncode-lines",
        "--filter",
        "core/.",
        "--exclude",
        "subprojects/.",
        "--exclude",
        "examples/.",
        "--exclude",
        "core/src/display/.",
        "--exclude",
        "core/include/display/.",
    ]

    print("== line coverage (core, exclude display) ==")
    line_args = gcovr_common_args + [
        "--txt-metric",
        "line",
        "--fail-under-line",
        "80",
    ]
    subprocess.run(line_args, check=True, cwd=root_dir)

    print()
    print("== branch coverage (core, exclude display) ==")
    branch_args = gcovr_common_args + [
        "--txt-metric",
        "branch",
        "--fail-under-branch",
        "35",
    ]
    subprocess.run(branch_args, check=True, cwd=root_dir)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate coverage report for Hahaha project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "builddir",
        nargs="?",
        default="builddir",
        help="Build directory (default: builddir)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean the build directory before running",
    )
    parser.add_argument(
        "--cuda",
        action="store_const",
        const="on",
        dest="enable_cuda",
        help="Enable CUDA support for coverage testing",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_const",
        const="off",
        dest="enable_cuda",
        help="Disable CUDA support for coverage testing",
    )
    parser.add_argument(
        "--continue-on-test-failure",
        action="store_true",
        help="Continue with coverage generation even if tests fail",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests and generate coverage directly (useful when tests have bugs)",
    )

    args = parser.parse_args()

    root_dir = get_root_dir()
    build_dir = root_dir / args.builddir
    enable_cuda = args.enable_cuda if args.enable_cuda else "auto"

    # Change to root directory
    os.chdir(root_dir)

    # Auto-detect CUDA availability if not specified
    if enable_cuda == "auto":
        if detect_cuda():
            enable_cuda = "on"
            cuda_path = os.environ.get("CUDA_PATH", "standard location")
            print(f"CUDA detected (CUDA_PATH={cuda_path}), enabling CUDA support for coverage testing.")
        else:
            enable_cuda = "off"
            print("CUDA not detected, disabling CUDA support for coverage testing.")

    # Clean the build directory if requested
    if args.clean:
        clean_build_dir(build_dir)

    # Create build directory if it doesn't exist
    build_dir.mkdir(parents=True, exist_ok=True)

    # Configure CMake; if CUDA is on and configure fails with CUDA error, retry without CUDA
    if enable_cuda == "on":
        ok, out = configure_cmake(build_dir, enable_cuda, capture_output=True)
        if not ok:
            # Always show the error output
            print("\n" + "=" * 70, file=sys.stderr)
            print("CMake configuration failed with CUDA enabled:", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            print(out, file=sys.stderr)
            
            if "CUDA" in out or "cuda" in out or "No CUDA toolset found" in out:
                print(
                    "\nCMake failed with CUDA enabled (e.g. no CUDA toolset for Visual Studio). "
                    "Retrying with CUDA disabled for coverage.",
                    file=sys.stderr,
                )
                enable_cuda = "off"
                configure_cmake(build_dir, enable_cuda)
            else:
                print(
                    "\nCMake configuration failed. Check the error messages above.",
                    file=sys.stderr,
                )
                raise subprocess.CalledProcessError(1, "cmake", stderr=out)
    else:
        configure_cmake(build_dir, enable_cuda)

    # Build project
    build_project(build_dir)

    # Run tests (unless skipped)
    if args.skip_tests:
        print("Skipping tests as requested (--skip-tests).", file=sys.stderr)
        print("Warning: Coverage will only reflect code executed during build, not test execution.", file=sys.stderr)
    else:
        tests_passed = run_tests(build_dir, enable_cuda, args.continue_on_test_failure)
        
        if not tests_passed and not args.continue_on_test_failure:
            print(
                "\nTests failed. Options:\n"
                "  --continue-on-test-failure : Continue with coverage generation anyway\n"
                "  --skip-tests              : Skip tests entirely (faster, but less accurate coverage)",
                file=sys.stderr,
            )
            sys.exit(8)

    # Generate the coverage report
    run_coverage(root_dir, enable_cuda)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
