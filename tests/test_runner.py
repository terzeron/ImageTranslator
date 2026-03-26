#!/usr/bin/env python

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = (
    SCRIPT_PATH.parent.parent
    if SCRIPT_PATH.parent.name == "tests"
    else SCRIPT_PATH.parent
)
TEST_DIR = PROJECT_ROOT / "tests"
RESOURCES_DIR = TEST_DIR / "resources"

# venv가 있으면 venv의 python을 사용, 없으면 현재 python
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def read_resource(filename: str) -> str:
    resource_path = RESOURCES_DIR / filename
    with open(resource_path, "r", encoding="utf-8") as f:
        return f.read()


def get_last_success_time() -> float:
    last_success_file = PROJECT_ROOT / ".last_test_success"
    if last_success_file.exists():
        try:
            return float(last_success_file.read_text())
        except (ValueError, OSError):
            return 0.0
    return 0.0


def set_last_success_time() -> None:
    last_success_file = PROJECT_ROOT / ".last_test_success"
    try:
        last_success_file.write_text(str(time.time()))
    except (OSError, PermissionError):
        pass


def get_modified_files(since: float) -> list[Path]:
    exclude_paths = {
        ".pytest_cache",
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        ".mypy_cache",
        "tmp",
        "htmlcov",
    }
    modified = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in exclude_paths]
        for f in files:
            if f.endswith(".py"):
                p = Path(root) / f
                try:
                    if p.stat().st_mtime > since and not f.startswith("."):
                        modified.append(p)
                except (OSError, PermissionError):
                    continue
    return modified


def get_test_modules_for_files(files: list[Path]) -> list[Path]:
    test_modules = set()
    for f in files:
        if f.name == "test_runner.py":
            continue
        if f.name.startswith("test_") and f.suffix == ".py":
            test_modules.add(f)
        else:
            test_name = f"test_{f.stem}.py"
            test_path = TEST_DIR / test_name
            if test_path.exists():
                test_modules.add(test_path)
    return list(test_modules)


def run_test_modules(test_targets: list[Path]) -> tuple[bool, int, int, list[Path]]:
    # 이전 커버리지 데이터 삭제
    cov_file = PROJECT_ROOT / ".coverage"
    if cov_file.exists():
        cov_file.unlink()

    passed_count = 0
    failed_count = 0
    failed_files = []

    for idx, t in enumerate(test_targets, 1):
        print(f"--- [{idx}/{len(test_targets)}] Running: {t.name} ---")
        absolute_path = t.resolve()

        start_time = time.time()
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "--tb=short",
                "--disable-warnings",
                "--cov=ocr",
                "--cov=translate",
                "--cov=render",
                "--cov-append",
                "--cov-report=",
                str(absolute_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(PROJECT_ROOT),
        )
        execution_time = time.time() - start_time

        # 필터링된 출력
        for line in result.stdout.splitlines():
            if any(
                skip in line
                for skip in [
                    "test session starts",
                    "platform ",
                    "rootdir",
                    "configfile",
                    "plugins",
                    "cachedir",
                ]
            ):
                continue
            if line.strip():
                print(line)

        if result.returncode != 0:
            print(f"FAILED: {t.name} ({execution_time:.1f}s)")
            if result.stderr:
                for line in result.stderr.splitlines()[-5:]:
                    print(f"  {line}")
            failed_count += 1
            failed_files.append(t)
        else:
            print(f"PASSED: {t.name} ({execution_time:.1f}s)")
            passed_count += 1
        print()

    overall_success = failed_count == 0
    return overall_success, passed_count, failed_count, failed_files


def run_coverage_report() -> None:
    # venv에 coverage가 있는지 확인
    check = subprocess.run(
        [PYTHON, "-c", "import coverage"],
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        print("\ncoverage 패키지가 없어 커버리지 리포트를 건너뜁니다.")
        return

    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)

    result = subprocess.run(
        [
            PYTHON,
            "-m",
            "coverage",
            "report",
            "--show-missing",
            "--include=ocr.py,translate.py,render.py",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    html_result = subprocess.run(
        [PYTHON, "-m", "coverage", "html", "--include=ocr.py,translate.py,render.py"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if html_result.returncode == 0:
        print("Coverage HTML: htmlcov/index.html")


def main() -> bool:
    parser = argparse.ArgumentParser(description="Test runner for image-translate")
    parser.add_argument("-a", "--all", action="store_true", help="Run all tests")
    parser.add_argument("-f", "--file", type=str, help="Run a specific test file")
    args = parser.parse_args()

    # Ensure project root is in sys.path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    if args.file:
        test_path = (
            TEST_DIR / args.file
            if not Path(args.file).is_absolute()
            else Path(args.file)
        )
        if not test_path.exists():
            print(f"Test file not found: {args.file}")
            return False
        test_targets = [test_path]
        print(f"Running: {args.file}")
    elif args.all:
        test_targets = sorted(TEST_DIR.glob("test_*.py"))
        test_targets = [t for t in test_targets if t.name != "test_runner.py"]
        print(f"Running all tests: {len(test_targets)} files")
    else:
        # Default: run changed tests, fallback to all
        last_success = get_last_success_time()
        modified = get_modified_files(last_success)
        test_targets = get_test_modules_for_files(modified)
        if not test_targets:
            test_targets = sorted(TEST_DIR.glob("test_*.py"))
            test_targets = [t for t in test_targets if t.name != "test_runner.py"]
            print(f"No changed tests detected, running all: {len(test_targets)} files")
        else:
            print(f"Running changed tests: {[t.name for t in test_targets]}")

    if not test_targets:
        print("No test files found.")
        return True

    success, passed, failed, failed_files = run_test_modules(test_targets)

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed_files:
        print(f"Failed: {[f.name for f in failed_files]}")
    print("=" * 60)

    if success:
        set_last_success_time()
        run_coverage_report()

    return success


if __name__ == "__main__":
    test_success = main()
    sys.exit(0 if test_success else 1)
