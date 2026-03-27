#!/usr/bin/env python

import sys
import os
import argparse
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from collections import defaultdict
import cProfile
import pstats
import io
import ast


# Always resolve project root (directory containing 'tests' folder)
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = (
    SCRIPT_PATH.parent.parent
    if SCRIPT_PATH.parent.name == "tests"
    else SCRIPT_PATH.parent
)
PYTHON_DIRS = [PROJECT_ROOT, PROJECT_ROOT / "tests"]
TEST_DIR = PROJECT_ROOT / "tests"
RESOURCES_DIR = TEST_DIR / "resources"

# Coverage targets: source modules at project root
COV_TARGETS = ["ocr", "translate", "render"]
COV_ARGS = [f"--cov={t}" for t in COV_TARGETS]
COV_INCLUDE = ",".join(f"{t}.py" for t in COV_TARGETS)


def read_resource(filename: str) -> str:
    """Read a resource file from tests/resources/ directory"""
    resource_path = RESOURCES_DIR / filename
    with open(resource_path, "r", encoding="utf-8") as f:
        return f.read()


def get_last_success_time() -> float:
    LAST_SUCCESS_FILE = PROJECT_ROOT / ".last_test_success"
    if LAST_SUCCESS_FILE.exists():
        return float(LAST_SUCCESS_FILE.read_text())
    return 0.0


def get_last_run_time() -> float:
    """Get the timestamp of the last test runner execution (regardless of success)."""
    LAST_RUN_FILE = PROJECT_ROOT / ".last_test_run"
    if LAST_RUN_FILE.exists():
        try:
            return float(LAST_RUN_FILE.read_text())
        except (ValueError, OSError, PermissionError):
            return 0.0
    return 0.0


def set_last_run_time() -> None:
    """Persist the timestamp of the latest test runner execution."""
    LAST_RUN_FILE = PROJECT_ROOT / ".last_test_run"
    try:
        LAST_RUN_FILE.write_text(str(time.time()))
    except (OSError, PermissionError):
        pass


def get_reference_time_for_changes() -> float:
    """Reference time for change detection.

    Prefer last run time to avoid re-running all tests repeatedly when the last
    successful time is old due to ongoing failures.
    """
    last_run = get_last_run_time()
    if last_run > 0:
        return last_run
    return get_last_success_time()


def _get_file_timestamps() -> dict[str, float]:
    """Get modification timestamps for all Python files"""
    timestamps = {}
    for file_path in collect_python_files():
        try:
            timestamps[str(file_path)] = file_path.stat().st_mtime
        except (OSError, PermissionError):
            pass
    return timestamps


def set_last_success_time() -> None:
    LAST_SUCCESS_FILE = PROJECT_ROOT / ".last_test_success"
    LAST_SUCCESS_FILE.write_text(str(time.time()))


def get_modified_files(
    since: float, exclude_paths: Optional[set[str]] = None
) -> list[Path]:
    """Get modified Python files with better filtering"""
    if exclude_paths is None:
        exclude_paths = {
            ".pytest_cache",
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            ".env",
            ".idea",
            ".vscode",
            ".run",
            ".mypy_cache",
            ".hypothesis",
            ".coverage_out",
            "tmp",
            "logs",
            "htmlcov",
            "fonts",
        }

    modified = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_paths]

        for f in files:
            if f.endswith(".py"):
                p = Path(root) / f
                try:
                    if (
                        p.stat().st_mtime > since
                        and not f.endswith(".bak")
                        and not f.startswith(".")
                        and p.stat().st_size > 0
                    ):
                        modified.append(p)
                except (OSError, PermissionError):
                    continue
    return modified


def get_test_modules_for_files(files: list[Path]) -> list[Path]:
    test_modules = set()
    for f in files:
        # test_runner.py는 테스트 실행기이므로 제외
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


def get_failed_or_skipped_tests() -> list[str]:
    """Get actually failed or skipped tests from pytest cache (robust path resolution)"""
    candidate_cache_dirs = [
        TEST_DIR / ".pytest_cache/v/cache",
        PROJECT_ROOT / ".pytest_cache/v/cache",
    ]

    lastfailed_file = None
    for cand in candidate_cache_dirs:
        lf = cand / "lastfailed"
        if lf.exists():
            lastfailed_file = lf
            break

    if lastfailed_file is None:
        return []

    try:
        with open(lastfailed_file, "r") as f:
            failed_data = json.load(f)

        if not failed_data:
            return []

        normalized: list[str] = []
        for test_key in failed_data.keys():
            if "::" not in test_key:
                continue
            file_part, *rest = test_key.split("::")

            file_path: Path
            if os.path.isabs(file_part):
                file_path = Path(file_part)
            elif file_part.startswith("tests/"):
                file_path = PROJECT_ROOT / file_part
            else:
                file_path = TEST_DIR / file_part

            try:
                rel_to_tests = file_path.resolve().relative_to(TEST_DIR.resolve())
            except ValueError:
                continue

            normalized_key = f"tests/{rel_to_tests.as_posix()}"
            if rest:
                normalized_key += "::" + "::".join(rest)
            normalized.append(normalized_key)

        return normalized

    except (
        json.JSONDecodeError,
        FileNotFoundError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return []


def is_test_actually_failed(test_path: Path) -> bool:
    """Check if a test file actually failed based on pytest cache"""
    failed_tests = get_failed_or_skipped_tests()
    test_file_str = str(test_path.relative_to(PROJECT_ROOT))

    for failed_test in failed_tests:
        if failed_test.startswith(test_file_str):
            return True
    return False


def collect_python_files() -> list[Path]:
    files: list[Path] = []
    for dir_path in PYTHON_DIRS:
        if dir_path.exists():
            if dir_path == PROJECT_ROOT:
                # Project root: only top-level .py files
                files.extend([f.resolve() for f in dir_path.glob("*.py")])
            else:
                files.extend([f.resolve() for f in dir_path.rglob("*.py")])
    return files


def is_test_file(file_path: Path) -> bool:
    """Check if a file is a test file (excluding test_runner.py)"""
    return (
        file_path.parent == TEST_DIR
        and file_path.name.startswith("test_")
        and file_path.name != "test_runner.py"
    )


def get_test_methods(test_file: Path) -> list[str]:
    """Extract all test methods from a test file"""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "--collect-only", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )

    test_methods = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line and "::test_" in line and line.endswith(")") is False:
            test_methods.append(line)

    return test_methods


def _get_coverage_file() -> Path:
    """커버리지 데이터 파일 경로를 반환한다."""
    for candidate in [PROJECT_ROOT / ".coverage", TEST_DIR / ".coverage"]:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / ".coverage"


def _clear_coverage_data() -> None:
    """이전 커버리지 데이터를 초기화한다."""
    for candidate in [PROJECT_ROOT / ".coverage", TEST_DIR / ".coverage"]:
        if candidate.exists():
            candidate.unlink()


def run_test_modules_sequentially(
    test_targets: list[Path],
) -> tuple[bool, int, int, list[Path]]:
    """Run test modules sequentially and return (success, passed_count, failed_count, failed_files)"""
    _clear_coverage_data()
    passed_count = 0
    failed_count = 0
    failed_files = []

    for idx, t in enumerate(test_targets, 1):
        print(f"--- [{idx}/{len(test_targets)}] Running: {t} ---")
        absolute_path = t.resolve()

        start_time = time.time()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--tb=no",
                "--disable-warnings",
                *COV_ARGS,
                "--cov-append",
                "--cov-report=",
                str(absolute_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        end_time = time.time()
        execution_time = end_time - start_time

        # Filter output to remove session start info and show only test results
        filtered_output = []
        in_session_start = False
        for line in result.stdout.splitlines():
            if "test session starts" in line:
                in_session_start = True
                continue
            if in_session_start and (
                "collected" in line
                or "platform" in line
                or "rootdir" in line
                or "configfile" in line
                or "plugins" in line
                or "cachedir" in line
                or "hypothesis profile" in line
            ):
                continue
            if in_session_start and line.strip() == "":
                in_session_start = False
                continue
            if not in_session_start:
                filtered_output.append(line)

        if filtered_output:
            print("\n".join(filtered_output))

        # Update performance cache with actual execution time
        test_file_name = t.name
        update_test_performance_cache(test_file_name, execution_time)

        if result.returncode != 0:
            print(f"❌ {t.name} FAILED.")
            failed_count += 1
            failed_files.append(t)
        else:
            print(f"✅ {t.name} PASSED.")
            passed_count += 1
        print("")

    overall_success = failed_count == 0
    if overall_success:
        print(f"✅ All {len(test_targets)} tests passed.")
    else:
        print(f"❌ {failed_count} out of {len(test_targets)} tests failed.")

    return overall_success, passed_count, failed_count, failed_files


def run_specific_test_file(test_file: str) -> bool:
    """Run a specific test file"""
    print(f"=== Running Specific Test File: {test_file} ===")
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"Error: Test file {test_file} not found")
        return False

    start_time = time.time()
    _clear_coverage_data()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_path),
            "-v",
            *COV_ARGS,
            "--cov-report=",
        ],
        check=False,
    )
    end_time = time.time()
    execution_time = end_time - start_time

    update_test_performance_cache(test_path.name, execution_time)

    return result.returncode == 0


def get_actual_execution_duration() -> float:
    """Get actual execution duration from cache or return 0 if not available"""
    performance_cache_file = PROJECT_ROOT / ".test_performance_cache"

    if performance_cache_file.exists():
        try:
            with open(performance_cache_file, "r") as f:
                cached_data = json.load(f)
                if "actual_total_duration" in cached_data:
                    return cached_data["actual_total_duration"]
        except (json.JSONDecodeError, OSError, PermissionError, KeyError):
            pass

    return 0.0


def update_actual_execution_duration(duration: float) -> None:
    """Update actual execution duration in cache"""
    performance_cache_file = PROJECT_ROOT / ".test_performance_cache"

    cached_data = {}
    if performance_cache_file.exists():
        try:
            with open(performance_cache_file, "r") as f:
                cached_data = json.load(f)
        except (json.JSONDecodeError, OSError, PermissionError):
            cached_data = {}

    cached_data["actual_total_duration"] = duration
    cached_data["last_actual_execution"] = time.time()

    try:
        with open(performance_cache_file, "w") as f:
            json.dump(cached_data, f, indent=2)
    except (OSError, PermissionError, TypeError) as e:
        print(f"⚠️  Failed to save actual execution duration: {e}")


def run_all_tests() -> tuple[bool, list[Path]]:
    """Run all tests in dependency order and return success and failed files"""
    print("=== Test Dependency Analysis and Execution (All Tests) ===")
    test_files = [f for f in TEST_DIR.glob("test_*.py") if f.name != "test_runner.py"]
    ordered_tests = sorted(test_files, key=lambda x: x.name)
    if not ordered_tests:
        print("No tests to run")
        return True, []

    _clear_coverage_data()
    total_start = time.time()
    failed_files = []

    for idx, t in enumerate(ordered_tests, 1):
        print(f"--- [{idx}/{len(ordered_tests)}] Running: {t.name} ---")
        start = time.time()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--tb=no",
                "--disable-warnings",
                *COV_ARGS,
                "--cov-append",
                "--cov-report=",
                str(t),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        end = time.time()

        # Filter output
        filtered_output = []
        in_session_start = False
        for line in result.stdout.splitlines():
            if "test session starts" in line:
                in_session_start = True
                continue
            if in_session_start and (
                "collected" in line
                or "platform" in line
                or "rootdir" in line
                or "configfile" in line
                or "plugins" in line
                or "cachedir" in line
                or "hypothesis profile" in line
            ):
                continue
            if in_session_start and line.strip() == "":
                in_session_start = False
                continue
            if not in_session_start:
                filtered_output.append(line)

        if filtered_output:
            print("\n".join(filtered_output))

        update_test_performance_cache(t.name, end - start)
        if result.returncode != 0:
            failed_files.append(t)

    total_end = time.time()
    update_actual_execution_duration(total_end - total_start)
    print(f"⏱️  실제 총 수행 시간: {total_end - total_start:.1f}초")

    return not failed_files, failed_files


def clear_lastfailed_for_files(test_file_prefixes: list[str]) -> None:
    """Remove stale lastfailed entries for the given test file prefixes."""
    candidate_cache_dirs = [
        TEST_DIR / ".pytest_cache/v/cache",
        PROJECT_ROOT / ".pytest_cache/v/cache",
    ]

    for cache_dir in candidate_cache_dirs:
        lastfailed_file = cache_dir / "lastfailed"
        if not lastfailed_file.exists():
            continue

        try:
            with open(lastfailed_file, "r") as f:
                data = json.load(f)

            if not data:
                continue

            cleaned = {
                key: val
                for key, val in data.items()
                if not any(
                    key.startswith(prefix + "::") for prefix in test_file_prefixes
                )
            }

            if cleaned != data:
                with open(lastfailed_file, "w") as f:
                    json.dump(cleaned, f, indent=2)
        except (json.JSONDecodeError, OSError, PermissionError):
            pass


def run_failed_tests() -> bool:
    """Run only failed tests"""
    failed_tests = get_failed_or_skipped_tests()
    if not failed_tests:
        return True

    test_files: list[Path] = []
    seen: set[Path] = set()
    for t in failed_tests:
        file_path = t.split("::")[0]
        if file_path.startswith("tests/"):
            absolute_path = PROJECT_ROOT / file_path
            if (
                absolute_path.exists()
                and absolute_path.resolve() != Path(__file__).resolve()
                and absolute_path not in seen
            ):
                test_files.append(absolute_path)
                seen.add(absolute_path)

    if not test_files:
        return True

    print("Running only failed test modules:")
    for t in test_files:
        print(f"  - {t}")

    success, _, _, _ = run_test_modules_sequentially(test_files)

    if success:
        file_prefixes = []
        for t in test_files:
            try:
                rel = t.relative_to(PROJECT_ROOT)
                file_prefixes.append(str(rel))
            except ValueError:
                pass
        if file_prefixes:
            clear_lastfailed_for_files(file_prefixes)
        print("All failed tests passed. Now running changed test modules (if any)...")
    else:
        print("Some previously failed tests are still failing.")

    return success


def run_changed_tests() -> bool:
    """Run tests for changed files with dependency consideration"""
    success, _, _, _ = run_changed_tests_with_results()
    return success


def run_changed_tests_with_results() -> tuple[bool, int, int, list[Path]]:
    """Run tests for changed files and return detailed results"""
    last_success = get_reference_time_for_changes()
    if last_success == 0:
        print(
            "⚪ 기준 성공 시각이 없어 변경 테스트를 건너뜁니다. 필요 시 -a 옵션으로 전체 실행하세요."
        )
        return True, 0, 0, []
    modified_files = get_modified_files(last_success)

    modified_files = [f for f in modified_files if f.name != "test_runner.py"]

    if not modified_files:
        print("✅ 변경된 테스트가 없습니다. 아무 테스트도 실행하지 않습니다.")
        return True, 0, 0, []

    test_targets = get_test_targets_with_dependencies(modified_files)

    if not test_targets:
        print("No test targets found for changed files.")
        return True, 0, 0, []

    print("\n🎯 Running tests for changed files (with dependency analysis)...")
    print(
        f"📋 Found {len(test_targets)} test targets for {len(modified_files)} modified files"
    )

    return run_test_modules_sequentially(test_targets)


def get_test_statistics() -> dict[str, Any]:
    """Get comprehensive test performance statistics"""
    test_files = [f for f in TEST_DIR.glob("test_*.py") if f.name != "test_runner.py"]

    last_success = get_reference_time_for_changes()
    last_success_str = (
        "Never"
        if last_success == 0
        else datetime.fromtimestamp(last_success).strftime("%Y-%m-%d %H:%M:%S")
    )

    failed_tests = get_failed_or_skipped_tests()
    performance_data = get_pytest_performance_data()

    return {
        "total_test_files": len(test_files),
        "last_success_time": last_success_str,
        "failed_tests_count": len(failed_tests),
        "failed_tests": failed_tests,
        "performance_data": performance_data,
    }


def get_pytest_performance_data() -> dict[str, Any]:
    """Extract performance data from actual pytest execution"""
    test_files = [f for f in TEST_DIR.glob("test_*.py") if f.name != "test_runner.py"]

    performance_cache_file = PROJECT_ROOT / ".test_performance_cache"

    cached_durations = {}
    if performance_cache_file.exists():
        try:
            with open(performance_cache_file, "r") as f:
                cached_data = json.load(f)
                cached_durations = cached_data.get("file_durations", {})
        except (json.JSONDecodeError, OSError, PermissionError):
            pass

    file_durations = {}
    for test_file in test_files:
        file_name = test_file.name
        if file_name in cached_durations:
            file_durations[file_name] = cached_durations[file_name]
        else:
            file_durations[file_name] = 0.5

    file_test_counts = {}
    file_avg_times = {}

    for test_file in test_files:
        file_name = test_file.name
        try:
            test_methods = get_test_methods(test_file)
            test_count = len(test_methods)
            file_test_counts[file_name] = test_count

            if test_count > 0:
                avg_time = file_durations[file_name] / test_count
                file_avg_times[file_name] = avg_time
            else:
                file_avg_times[file_name] = file_durations[file_name]
        except (subprocess.CalledProcessError, OSError, PermissionError) as e:
            print(f"⚠️  Failed to get test count for {file_name}: {e}")
            file_test_counts[file_name] = 1
            file_avg_times[file_name] = file_durations[file_name]

    total_duration = sum(file_durations.values())

    return {
        "total_duration": total_duration,
        "file_durations": file_durations,
        "file_test_counts": file_test_counts,
        "file_avg_times": file_avg_times,
        "slowest_files": sorted(
            file_durations.items(), key=lambda x: x[1], reverse=True
        )[:10],
        "slowest_avg_times": sorted(
            file_avg_times.items(), key=lambda x: x[1], reverse=True
        )[:10],
    }


def update_test_performance_cache(test_file: str, execution_time: float) -> None:
    """Update performance cache with actual test execution time"""
    performance_cache_file = PROJECT_ROOT / ".test_performance_cache"

    cached_data = {}
    if performance_cache_file.exists():
        try:
            with open(performance_cache_file, "r") as f:
                cached_data = json.load(f)
        except (json.JSONDecodeError, OSError, PermissionError):
            cached_data = {}

    file_durations = cached_data.get("file_durations", {})
    alpha = 0.3

    if test_file in file_durations:
        old_time = file_durations[test_file]
        new_time = alpha * execution_time + (1 - alpha) * old_time
    else:
        new_time = execution_time

    file_durations[test_file] = new_time
    cached_data["file_durations"] = file_durations
    cached_data["last_updated"] = time.time()

    try:
        with open(performance_cache_file, "w") as f:
            json.dump(cached_data, f, indent=2)
        print(f"📊 Updated performance cache: {test_file} = {new_time:.2f}s")
    except (OSError, PermissionError, TypeError) as e:
        print(f"⚠️  Failed to save performance cache: {e}")


def print_test_statistics(stats: dict[str, Any]) -> None:
    """Print comprehensive test performance statistics"""
    print("\n" + "=" * 60)
    print("⚡ TEST PERFORMANCE STATISTICS")
    print("=" * 60)

    perf_data = stats["performance_data"]

    print(f"📊 총 테스트 파일 수: {stats['total_test_files']}개")
    print(f"⏰ 마지막 성공 시간: {stats['last_success_time']}")
    print(f"🕐 예상 총 수행 시간: {perf_data['total_duration']:.1f}초")

    actual_duration = get_actual_execution_duration()
    if actual_duration > 0:
        print(f"⏱️  실제 총 수행 시간: {actual_duration:.1f}초")
        if perf_data["total_duration"] > 0:
            improvement = (
                (perf_data["total_duration"] - actual_duration)
                / perf_data["total_duration"]
            ) * 100
            print(
                f"📈 성능 개선율: {improvement:.1f}% (예상 대비 {actual_duration / perf_data['total_duration']:.1f}배 빠름)"
            )

    print(
        f"📈 평균 파일당 수행 시간: {perf_data['total_duration'] / stats['total_test_files']:.1f}초"
    )

    total_test_count = sum(perf_data["file_test_counts"].values())
    if total_test_count > 0:
        print(f"🧪 총 테스트 개수: {total_test_count}개")
        print(
            f"📊 평균 테스트당 수행 시간: {perf_data['total_duration'] / total_test_count:.2f}초"
        )

    if stats["failed_tests_count"] > 0:
        print(f"❌ 실패한 테스트: {stats['failed_tests_count']}개")
    else:
        print("✅ 실패한 테스트: 없음")

    print("\n🐌 수행 시간이 긴 테스트 파일 TOP 10:")
    for i, (file_name, duration) in enumerate(perf_data["slowest_files"], 1):
        test_count = perf_data["file_test_counts"].get(file_name, 1)
        avg_time = perf_data["file_avg_times"].get(file_name, duration)

        if duration >= 10:
            icon = "🔥"
        elif duration >= 5:
            icon = "⚠️ "
        elif duration >= 1:
            icon = "📝"
        else:
            icon = "⚡"

        print(
            f"   {i:2d}. {icon} {file_name:<35} {duration:>6.1f}초 ({test_count}개 테스트, 평균 {avg_time:.2f}초)"
        )

    print("\n🐌 평균 테스트 수행시간이 긴 파일 TOP 10:")
    for i, (file_name, avg_time) in enumerate(perf_data["slowest_avg_times"], 1):
        test_count = perf_data["file_test_counts"].get(file_name, 1)
        total_time = perf_data["file_durations"].get(file_name, 0)

        if avg_time >= 5:
            icon = "🔥"
        elif avg_time >= 2:
            icon = "⚠️ "
        elif avg_time >= 0.5:
            icon = "📝"
        else:
            icon = "⚡"

        print(
            f"   {i:2d}. {icon} {file_name:<35} 평균 {avg_time:>6.2f}초 ({test_count}개 테스트, 총 {total_time:.1f}초)"
        )

    slow_files = [f for f, d in perf_data["file_durations"].items() if d >= 10]
    slow_avg_files = [f for f, avg in perf_data["file_avg_times"].items() if avg >= 2]

    if slow_files:
        print("\n💡 성능 개선 권장사항:")
        print(f"   • {len(slow_files)}개 파일이 10초 이상 소요됩니다")

    if slow_avg_files:
        print(f"   • {len(slow_avg_files)}개 파일의 평균 테스트 시간이 2초 이상입니다")

    fast_ratio = (
        len([d for d in perf_data["file_durations"].values() if d < 1])
        / len(perf_data["file_durations"])
        * 100
    )
    fast_avg_ratio = (
        len([avg for avg in perf_data["file_avg_times"].values() if avg < 0.5])
        / len(perf_data["file_avg_times"])
        * 100
    )

    print("\n📊 성능 지표:")
    print(f"   • 빠른 테스트 비율 (<1초): {fast_ratio:.1f}%")
    print(f"   • 빠른 평균 테스트 비율 (<0.5초): {fast_avg_ratio:.1f}%")
    print(
        f"   • 병렬 실행 시 예상 시간: {max(perf_data['file_durations'].values()):.1f}초"
    )

    print("=" * 60)


def profile_slow_tests() -> None:
    """Profile the slowest test files to identify bottlenecks"""
    print("\n🔬 STATISTICAL PROFILING MODE")
    print("=" * 60)
    print("소스코드 변경 없이 통계적 샘플링으로 성능 분석을 수행합니다.")

    perf_data = get_pytest_performance_data()
    slow_tests = [
        name for name, duration in perf_data["slowest_files"][:5] if duration >= 5.0
    ]

    print(f"\n📋 분석 대상 테스트 파일 ({len(slow_tests)}개):")
    for test_name in slow_tests:
        print(f"   • {test_name}")

    print("\n🚀 프로파일링 시작...")

    all_analyses = []

    for test_name in slow_tests:
        test_path = TEST_DIR / test_name
        if not test_path.exists():
            print(f"⚠️  파일을 찾을 수 없음: {test_name}")
            continue

        success, analysis = run_test_with_profiling(test_path)
        all_analyses.append(analysis)

        if analysis["total_samples"] > 0:
            print_profiling_analysis(analysis)
        else:
            print(
                f"⚡ {test_name}: 너무 빨라서 샘플링되지 않음 ({analysis['execution_time']:.2f}초)"
            )

    print("\n📈 전체 프로파일링 요약:")
    total_samples = sum(a["total_samples"] for a in all_analyses)
    total_time = sum(a["execution_time"] for a in all_analyses)

    print(f"   • 총 실행 시간: {total_time:.2f}초")
    print(f"   • 총 샘플 수: {total_samples}개")
    if total_time > 0:
        print(f"   • 평균 샘플링 밀도: {total_samples / total_time:.1f} 샘플/초")


def run_test_with_profiling(test_file: Path) -> tuple[bool, dict[str, Any]]:
    """Run a single test file with profiling"""
    print(f"🔬 Profiling: {test_file.name}")

    profiler = cProfile.Profile()

    start_time = time.time()

    try:
        profiler.enable()

        import pytest

        exit_code = pytest.main(
            [
                str(test_file),
                "-v",
                "--tb=short",
            ]
        )

        success = exit_code == 0

    except (ImportError, ModuleNotFoundError, SystemExit) as e:
        print(f"Error running test: {e}")
        success = False
    finally:
        profiler.disable()

    end_time = time.time()
    execution_time = end_time - start_time

    analysis = analyze_cprofile_results(profiler, execution_time, test_file.name)
    analysis["success"] = success

    return success, analysis


def analyze_cprofile_results(
    profiler: cProfile.Profile, execution_time: float, test_file: str
) -> dict[str, Any]:
    """Analyze cProfile results and extract meaningful insights"""

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)

    ps.sort_stats("cumulative")

    total_calls = 0
    stats_dict = getattr(ps, "stats", {})

    for func_data in stats_dict.values():
        total_calls += func_data[0]

    function_stats = []
    for func, (cc, nc, tt, ct, callers) in stats_dict.items():
        filename, line_number, function_name = func

        if any(
            path in filename
            for path in ["/tests/", "ocr.py", "translate.py", "render.py", "main.py"]
        ):
            function_stats.append(
                {
                    "filename": Path(filename).name,
                    "line": line_number,
                    "function": function_name,
                    "calls": nc,
                    "total_calls": cc,
                    "total_time": tt,
                    "cumulative_time": ct,
                    "time_per_call": tt / nc if nc > 0 else 0,
                    "cum_time_per_call": ct / nc if nc > 0 else 0,
                }
            )

    function_stats.sort(key=lambda x: x["cumulative_time"], reverse=True)

    hotspots = function_stats[:15]
    most_called = sorted(function_stats, key=lambda x: x["calls"], reverse=True)[:10]
    slowest_per_call = sorted(
        [f for f in function_stats if f["calls"] > 0],
        key=lambda x: x["time_per_call"],
        reverse=True,
    )[:10]

    file_stats: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"total_time": 0.0, "cumulative_time": 0.0, "calls": 0, "functions": 0}
    )

    for func_stat in function_stats:
        filename = func_stat["filename"]
        file_stats[filename]["total_time"] += func_stat["total_time"]
        file_stats[filename]["cumulative_time"] += func_stat["cumulative_time"]
        file_stats[filename]["calls"] += func_stat["calls"]
        file_stats[filename]["functions"] += 1

    file_stats_list = [
        {"filename": filename, **stats} for filename, stats in file_stats.items()
    ]
    file_stats_list.sort(key=lambda x: x["cumulative_time"], reverse=True)

    return {
        "test_file": test_file,
        "execution_time": execution_time,
        "total_calls": total_calls,
        "total_samples": total_calls,
        "function_count": len(function_stats),
        "hotspots": hotspots,
        "most_called": most_called,
        "slowest_per_call": slowest_per_call,
        "file_stats": file_stats_list[:10],
    }


def print_profiling_analysis(analysis: dict[str, Any]) -> None:
    """Print profiling analysis results"""
    print(f"\n🔬 PROFILING ANALYSIS: {analysis['test_file']}")
    print(f"{'=' * 80}")

    print(f"⏱️  총 실행 시간: {analysis['execution_time']:.3f}초")
    print(f"📞 총 함수 호출: {analysis['total_calls']:,}회")
    print(f"🔧 분석된 함수: {analysis['function_count']}개")

    print("\n🔥 성능 핫스팟 (누적 시간 기준 TOP 15):")
    print(
        f"{'Rank':<4} {'File':<25} {'Function':<25} {'Calls':<8} {'Total(s)':<8} {'Cumul(s)':<8} {'Per Call(ms)':<12}"
    )
    print("-" * 100)
    for i, func in enumerate(analysis["hotspots"], 1):
        print(
            f"{i:<4} {func['filename']:<25} {func['function']:<25} {func['calls']:<8} {func['total_time']:<8.3f} {func['cumulative_time']:<8.3f} {func['time_per_call'] * 1000:<12.2f}"
        )

    print("\n📞 가장 많이 호출된 함수 TOP 10:")
    print(f"{'Rank':<4} {'Calls':<10} {'File':<25} {'Function':<25} {'Total(s)':<8}")
    print("-" * 80)
    for i, func in enumerate(analysis["most_called"], 1):
        print(
            f"{i:<4} {func['calls']:<10} {func['filename']:<25} {func['function']:<25} {func['total_time']:<8.3f}"
        )

    print("\n🐌 호출당 가장 느린 함수 TOP 10:")
    print(
        f"{'Rank':<4} {'Per Call(ms)':<12} {'Calls':<8} {'File':<25} {'Function':<25}"
    )
    print("-" * 80)
    for i, func in enumerate(analysis["slowest_per_call"], 1):
        print(
            f"{i:<4} {func['time_per_call'] * 1000:<12.2f} {func['calls']:<8} {func['filename']:<25} {func['function']:<25}"
        )

    print("\n📁 파일별 성능 요약:")
    print(
        f"{'Rank':<4} {'File':<30} {'Functions':<9} {'Calls':<10} {'Total(s)':<8} {'Cumul(s)':<8}"
    )
    print("-" * 80)
    for i, file_stat in enumerate(analysis["file_stats"], 1):
        print(
            f"{i:<4} {file_stat['filename']:<30} {file_stat['functions']:<9} {file_stat['calls']:<10} {file_stat['total_time']:<8.3f} {file_stat['cumulative_time']:<8.3f}"
        )

    print("\n💡 성능 개선 권장사항:")
    top_hotspot = analysis["hotspots"][0] if analysis["hotspots"] else None
    if top_hotspot:
        print(f"   • 최대 병목점: {top_hotspot['filename']}::{top_hotspot['function']}")
        print(
            f"     - 누적 시간: {top_hotspot['cumulative_time']:.3f}초 ({top_hotspot['cumulative_time'] / analysis['execution_time'] * 100:.1f}%)"
        )

    top_called = analysis["most_called"][0] if analysis["most_called"] else None
    if top_called and top_called["calls"] > 1000:
        print(f"   • 과도한 호출: {top_called['filename']}::{top_called['function']}")
        print(f"     - {top_called['calls']:,}회 호출, 캐싱 또는 최적화 고려")

    slow_per_call = (
        analysis["slowest_per_call"][0] if analysis["slowest_per_call"] else None
    )
    if slow_per_call and slow_per_call["time_per_call"] > 0.001:
        print(
            f"   • 느린 함수: {slow_per_call['filename']}::{slow_per_call['function']}"
        )
        print(
            f"     - 호출당 {slow_per_call['time_per_call'] * 1000:.2f}ms, 내부 로직 최적화 필요"
        )

    print(f"{'=' * 80}")


def get_status_emoji(
    path: Path,
    executed_tests: Optional[set[Path]] = None,
    target_tests: Optional[set[Path]] = None,
    failed_tests: Optional[set[Path]] = None,
    passed_tests: Optional[set[Path]] = None,
    deps: Optional[dict[Path, set[Path]]] = None,
    reverse_deps: Optional[dict[Path, set[Path]]] = None,
    modified_files: Optional[set[Path]] = None,
    affected_files: Optional[set[Path]] = None,
    all_mode: bool = False,
) -> str:
    """Unified function to get status emoji for a file"""
    if executed_tests is None:
        executed_tests = set()
    if target_tests is None:
        target_tests = set()
    if failed_tests is None:
        failed_tests = set()
    if passed_tests is None:
        passed_tests = set()
    if modified_files is None:
        modified_files = set()
    if affected_files is None:
        affected_files = set()

    if is_test_module(path):
        if path in failed_tests:
            return "🔴"
        if path in passed_tests:
            return "🟢"
        if path in modified_files:
            return "🔵"
        if all_mode and path not in modified_files:
            return "🟣"
        if path in target_tests and path not in modified_files:
            return "🟣"
        return "⚪"

    if not is_test_module(path):
        if path in modified_files:
            return "🔵"
        test_modules = set()
        if deps and reverse_deps:
            test_modules = _get_test_modules_for_file(path, deps, reverse_deps)
        if test_modules:
            has_failed = any(test in failed_tests for test in test_modules)
            has_passed = any(test in passed_tests for test in test_modules)
            if has_failed:
                return "🔴"
            elif has_passed:
                return "🟢"
            else:
                if path in affected_files:
                    return "🟡"
                else:
                    return "⚪"
        else:
            if path in affected_files:
                return "🟡"
            else:
                return "⚪"
    return "⚪"


def is_test_module(path: Path) -> bool:
    """테스트 모듈 판별 함수"""
    resolved_path = path.resolve()
    resolved_test_dir = TEST_DIR.resolve()
    return (
        resolved_path.name.startswith("test_")
        and resolved_path.suffix == ".py"
        and resolved_path.parent == resolved_test_dir
        and resolved_path.name != "test_runner.py"
    )


def _get_test_modules_for_file(
    file_path: Path, deps: dict[Path, set[Path]], reverse_deps: dict[Path, set[Path]]
) -> set[Path]:
    """파일을 테스트하는 테스트 모듈들을 찾기"""
    test_modules = set()

    possible_test_names = [f"test_{file_path.stem}.py"]

    for test_name in possible_test_names:
        test_path = TEST_DIR / test_name
        if is_test_module(test_path) and test_path.exists():
            test_modules.add(test_path)

    if file_path in reverse_deps:
        for importing_file in reverse_deps[file_path]:
            if is_test_module(importing_file):
                test_modules.add(importing_file)

    return test_modules


def extract_imports_from_file(file_path: Path) -> set[str]:
    """Extract all import statements from a Python file using AST"""
    imports = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
    except (SyntaxError, OSError, PermissionError) as e:
        print(f"Warning: Could not parse imports from {file_path}: {e}")

    return imports


def build_dependency_graph() -> dict[Path, set[Path]]:
    """Build dependency graph by parsing import statements in all Python files"""
    deps = {}
    all_files = collect_python_files()

    module_to_path = {}
    for file_path in all_files:
        rel_path = file_path.relative_to(PROJECT_ROOT)
        module_name = ".".join(rel_path.with_suffix("").parts)
        module_to_path[module_name] = file_path

    for file_path in all_files:
        deps[file_path] = set()
        imports = extract_imports_from_file(file_path)

        for import_name in imports:
            if not isinstance(import_name, str):
                continue
            if import_name.startswith("."):
                rel_path = file_path.relative_to(PROJECT_ROOT)
                parts = list(rel_path.with_suffix("").parts)
                dots = len(import_name) - len(import_name.lstrip("."))
                if dots <= len(parts):
                    parts = parts[:-dots]
                    import_name = ".".join(parts + [import_name.lstrip(".")])

            if import_name in module_to_path:
                deps[file_path].add(module_to_path[import_name])

    return deps


def analyze_all_dependencies() -> tuple[dict[Path, set[Path]], dict[Path, set[Path]]]:
    """Analyze dependencies between all Python files with performance optimization"""
    if not TEST_DIR.exists():
        print(f"Error: {TEST_DIR} directory not found")
        return {}, {}

    cache_file = PROJECT_ROOT / ".dependency_cache.json"
    cache_data = {}

    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                raw = json.load(f)

            if _is_dependency_cache_valid(raw):
                print("📦 Using cached dependency analysis (performance optimized)")
                deps = {
                    Path(k): {Path(v) for v in vs}
                    for k, vs in raw.get("deps", {}).items()
                }
                reverse_deps = {
                    Path(k): {Path(v) for v in vs}
                    for k, vs in raw.get("reverse_deps", {}).items()
                }
                return deps, reverse_deps

        except (OSError, PermissionError, json.JSONDecodeError):
            pass

    print("🔍 Performing fresh dependency analysis...")
    start_time = time.time()

    deps = build_dependency_graph()
    reverse_deps = get_reverse_dependencies(deps)

    elapsed = time.time() - start_time
    print(f"✅ Dependency analysis completed in {elapsed:.2f}s")

    try:
        serializable = {
            "deps": {str(k): [str(v) for v in vs] for k, vs in deps.items()},
            "reverse_deps": {
                str(k): [str(v) for v in vs] for k, vs in reverse_deps.items()
            },
            "timestamp": time.time(),
            "file_timestamps": _get_file_timestamps(),
        }
        with open(cache_file, "w") as f:
            json.dump(serializable, f, indent=2)
    except (OSError, PermissionError) as e:
        print(f"⚠️  Failed to cache dependencies: {e}")

    return deps, reverse_deps


def get_reverse_dependencies(deps: dict[Path, set[Path]]) -> dict[Path, set[Path]]:
    """Get reverse dependencies (who imports this file)"""
    reverse_deps: dict[Path, set[Path]] = {}

    for src, targets in deps.items():
        for target in targets:
            if target not in reverse_deps:
                reverse_deps[target] = set()
            reverse_deps[target].add(src)

    return reverse_deps


def get_affected_files(
    modified_files: list[Path],
    deps: dict[Path, set[Path]],
    reverse_deps: dict[Path, set[Path]],
    max_depth: int = 1,
) -> set[Path]:
    """Get all files affected by the modified files"""
    affected = set(modified_files)
    to_process = [(f, 0) for f in modified_files]
    processed = set()

    while to_process:
        current_file, depth = to_process.pop(0)
        if current_file in processed or depth >= max_depth:
            continue

        processed.add(current_file)

        if current_file in reverse_deps:
            importers = reverse_deps[current_file]
            for importer in importers:
                if importer not in affected:
                    affected.add(importer)
                    to_process.append((importer, depth + 1))

    return affected


def print_global_dependency_tree(
    deps: dict[Path, set[Path]],
    max_depth: int = 10,
    executed_tests: Optional[set[Path]] = None,
    target_tests: Optional[set[Path]] = None,
    failed_tests: Optional[set[Path]] = None,
    passed_tests: Optional[set[Path]] = None,
    reverse_deps: Optional[dict[Path, set[Path]]] = None,
    modified_files: Optional[set[Path]] = None,
    affected_files: Optional[set[Path]] = None,
    all_mode: bool = False,
):
    """테스트 파일을 루트로 하여 전체 import chain을 트리로 출력"""
    test_modules = [f for f in deps.keys() if is_test_module(f)]
    print("\n=== GLOBAL DEPENDENCY TREE ===")
    for idx, test_file in enumerate(sorted(test_modules, key=str)):
        status = get_status_emoji(
            test_file,
            executed_tests,
            target_tests,
            failed_tests,
            passed_tests,
            deps,
            reverse_deps,
            modified_files,
            affected_files,
            all_mode=all_mode,
        )
        print(f"{status} {test_file.name}")
        _print_global_branch_tree_node(
            test_file,
            deps,
            max_depth,
            1,
            set([test_file]),
            "    ",
            executed_tests,
            target_tests,
            failed_tests,
            passed_tests,
            reverse_deps,
            modified_files,
            affected_files,
            all_mode=all_mode,
        )


def _print_global_branch_tree_node(
    node: Path,
    deps: dict[Path, set[Path]],
    max_depth: int,
    current_depth: int,
    visited: set[Path],
    prefix: str,
    executed_tests: Optional[set[Path]] = None,
    target_tests: Optional[set[Path]] = None,
    failed_tests: Optional[set[Path]] = None,
    passed_tests: Optional[set[Path]] = None,
    reverse_deps: Optional[dict[Path, set[Path]]] = None,
    modified_files: Optional[set[Path]] = None,
    affected_files: Optional[set[Path]] = None,
    all_mode: bool = False,
):
    if current_depth > max_depth:
        return
    for dep in sorted(deps.get(node, set()), key=str):
        if dep in visited:
            continue
        status = get_status_emoji(
            dep,
            executed_tests,
            target_tests,
            failed_tests,
            passed_tests,
            deps,
            reverse_deps,
            modified_files,
            affected_files,
            all_mode=all_mode,
        )
        print(f"{prefix}└─ {status} {dep.name}")
        visited.add(dep)
        _print_global_branch_tree_node(
            dep,
            deps,
            max_depth,
            current_depth + 1,
            visited,
            prefix + "    ",
            executed_tests,
            target_tests,
            failed_tests,
            passed_tests,
            reverse_deps,
            modified_files,
            affected_files,
            all_mode=all_mode,
        )


def get_test_targets_with_dependencies(modified_files: list[Path]) -> list[Path]:
    """Get test targets considering import dependencies"""
    deps, reverse_deps = analyze_all_dependencies()

    modified_files = [f for f in modified_files if f.name != "test_runner.py"]

    if not modified_files:
        print("📋 No relevant files changed (test_runner.py changes are ignored)")
        return []

    affected_files = get_affected_files(modified_files, deps, reverse_deps, max_depth=1)

    failed_tests = get_failed_or_skipped_tests()
    failed_test_paths = set()
    for failed_test in failed_tests:
        if "::" in failed_test:
            test_file = failed_test.split("::")[0]
            test_path = Path(test_file)
            if test_path.exists():
                failed_test_paths.add(test_path)

    test_targets = []
    for file_path in affected_files:
        if is_test_module(file_path):
            test_targets.append(file_path)
        else:
            possible_test_names = [f"test_{file_path.stem}.py"]
            for test_name in possible_test_names:
                test_path = TEST_DIR / test_name
                if is_test_module(test_path) and test_path.exists():
                    test_targets.append(test_path)
                    break

    for failed_test_path in failed_test_paths:
        if is_test_module(failed_test_path):
            test_targets.append(failed_test_path)

    unique_targets = []
    seen = set()
    for target in test_targets:
        if target.name != "test_runner.py" and target not in seen:
            unique_targets.append(target)
            seen.add(target)

    return unique_targets


def run_coverage_report() -> None:
    """전체 테스트를 단일 프로세스로 실행하여 정확한 커버리지 리포트를 생성한다."""
    try:
        import coverage as _cov_mod  # noqa: F401
    except ImportError:
        print("\n⚠️  coverage 패키지가 없어 커버리지 리포트를 건너뜁니다.")
        return

    print("\n📊 전체 커버리지 측정 중...")
    cov_dir = str(PROJECT_ROOT)

    cov_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--tb=no",
            "-q",
            "--disable-warnings",
            *COV_ARGS,
            "--cov-report=",
            str(TEST_DIR),
        ],
        capture_output=True,
        text=True,
        cwd=cov_dir,
    )

    if cov_result.returncode not in (0, 1):
        print("⚠️  커버리지 수집 중 오류가 발생했습니다.")
        if cov_result.stderr:
            for line in cov_result.stderr.splitlines()[-5:]:
                print(f"  {line}")
        return

    print("\n" + "=" * 80)
    print("📊 COVERAGE REPORT")
    print("=" * 80)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "report",
            "--show-missing",
            f"--include={COV_INCLUDE}",
        ],
        capture_output=True,
        text=True,
        cwd=cov_dir,
    )
    if result.stdout:
        print(result.stdout)

    html_result = subprocess.run(
        [sys.executable, "-m", "coverage", "html", f"--include={COV_INCLUDE}"],
        capture_output=True,
        text=True,
        cwd=cov_dir,
    )
    if html_result.returncode == 0:
        print("Coverage HTML written to dir htmlcov")
    else:
        print("⚠️  HTML 리포트 생성 중 오류가 발생했습니다.")
        if html_result.stderr:
            for line in html_result.stderr.splitlines()[-5:]:
                print(f"  {line}")


def _is_dependency_cache_valid(cache_data: dict[str, Any]) -> bool:
    """Check if cached dependency data is still valid"""
    if "timestamp" not in cache_data or "file_timestamps" not in cache_data:
        return False

    if time.time() - cache_data["timestamp"] > 3600:
        return False

    current_timestamps = _get_file_timestamps()
    cached_timestamps = cache_data["file_timestamps"]

    for file_path, current_time in current_timestamps.items():
        if (
            file_path not in cached_timestamps
            or cached_timestamps[file_path] != current_time
        ):
            return False

    return True


def main() -> bool:
    """Main function"""
    os.chdir(TEST_DIR)

    parser = argparse.ArgumentParser(description="Test runner with dependency analysis")
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Run all tests in dependency order (default: run changed modules + previously failed tests)",
    )
    parser.add_argument(
        "-f", "--file", type=str, help="Run a specific test file (e.g., test_ocr.py)"
    )
    parser.add_argument(
        "-p",
        "--profile",
        type=str,
        metavar="TEST_MODULE",
        help="Profile a specific test module using detailed cProfile analysis (e.g., test_ocr.py)",
    )
    args = parser.parse_args()

    print("🔍 Analyzing dependencies...")
    deps, reverse_deps = analyze_all_dependencies()

    executed_tests: set[Path] = set()
    target_tests: set[Path] = set()
    failed_tests: set[Path] = set()
    passed_tests: set[Path] = set()

    actual_passed_count = 0
    actual_failed_count = 0

    last_success = get_last_success_time()
    modified_files = set(get_modified_files(last_success))
    affected_files = (
        get_affected_files(list(modified_files), deps, reverse_deps, max_depth=1)
        if modified_files
        else set()
    )

    if args.file:
        test_path = Path(args.file)
        if not test_path.is_absolute():
            # Try relative to TEST_DIR first, then PROJECT_ROOT
            if (TEST_DIR / test_path).exists():
                test_path = TEST_DIR / test_path
            elif (PROJECT_ROOT / test_path).exists():
                test_path = PROJECT_ROOT / test_path
        if test_path.exists():
            target_tests = {test_path}
            print(f"🎯 Running test: {test_path}")
            print(f"📋 Dependencies: {len(deps.get(test_path, set()))} modules")
        else:
            print(f"❌ Test file not found: {args.file}")
            return False
    elif args.all:
        test_files = [
            f for f in TEST_DIR.glob("test_*.py") if f.name != "test_runner.py"
        ]
        target_tests = set(test_files)
        print(f"🎯 Running all tests: {len(target_tests)} files")
    else:
        if modified_files:
            if last_success == 0:
                print("⚪ 기준 성공 시각이 없어 변경 테스트 목록 계산을 건너뜁니다.")
                target_tests = set()
            else:
                test_targets = get_test_targets_with_dependencies(list(modified_files))
                target_tests = set(test_targets)
                print(f"🎯 Running tests for modified files: {len(target_tests)} files")
                print(f"📋 Modified files: {[f.name for f in modified_files]}")
        else:
            print("📋 No changed files detected")
            target_tests = set()

    print_global_dependency_tree(
        deps,
        max_depth=10,
        executed_tests=executed_tests,
        target_tests=target_tests,
        failed_tests=failed_tests,
        passed_tests=passed_tests,
        reverse_deps=reverse_deps,
        modified_files=modified_files,
        affected_files=affected_files,
        all_mode=args.all,
    )

    success = False

    if args.profile:
        test_module = args.profile
        if not test_module.endswith(".py"):
            test_module += ".py"

        test_path = TEST_DIR / test_module
        if not test_path.exists():
            print(f"❌ 테스트 모듈을 찾을 수 없습니다: {test_module}")
            print(f"   경로: {test_path}")
            return False

        print("\n🔬 DETAILED PROFILING MODE")
        print("=" * 80)
        print(f"대상 모듈: {test_module}")

        success, analysis = run_test_with_profiling(test_path)

        if analysis.get("function_count", 0) > 0:
            print_profiling_analysis(analysis)
        else:
            print("⚠️  프로파일링 데이터를 수집할 수 없었습니다.")

        return success
    elif args.file:
        success = run_specific_test_file(str(test_path))
        executed_tests.add(test_path)
        if success:
            passed_tests.add(test_path)
            actual_passed_count = 1
            actual_failed_count = 0
        else:
            failed_tests.add(test_path)
            actual_passed_count = 0
            actual_failed_count = 1
    elif args.all:
        success, all_failed_files = run_all_tests()
        failed_tests.update(all_failed_files)

        test_files = [
            f for f in TEST_DIR.glob("test_*.py") if f.name != "test_runner.py"
        ]
        executed_tests.update(test_files)
        passed_tests.update(set(test_files) - failed_tests)

        actual_passed_count = len(passed_tests)
        actual_failed_count = len(failed_tests)

    else:
        failed_tests_success = run_failed_tests()
        if not failed_tests_success:
            return False

        success, actual_passed_count, actual_failed_count, changed_failed_files = (
            run_changed_tests_with_results()
        )
        failed_tests.update(changed_failed_files)

        print(
            f"\n📋 Test execution completed: {actual_passed_count} passed, {actual_failed_count} failed"
        )

        last_success = get_last_success_time()
        modified_files = set(get_modified_files(last_success))
        if modified_files:
            test_targets = get_test_targets_with_dependencies(list(modified_files))
            executed_tests.update(test_targets)
            passed_tests.update(set(test_targets) - failed_tests)

        success = not failed_tests

    set_last_run_time()

    if success:
        set_last_success_time()
        print_test_statistics(get_test_statistics())

    if executed_tests:
        print("\n" + "=" * 80)
        print("🎯 TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"✅ Passed: {actual_passed_count} tests")
        print(f"❌ Failed: {actual_failed_count} tests")
        print(f"📋 Total executed: {len(executed_tests)} tests")

        if len(passed_tests) > 0:
            print(
                f"✅ Passed tests: {[t.name for t in list(passed_tests)[:5]]}{'...' if len(passed_tests) > 5 else ''}"
            )
        if len(failed_tests) > 0:
            print(f"❌ Failed tests: {[t.name for t in failed_tests]}")

    if success:
        run_coverage_report()

    return success


if __name__ == "__main__":
    test_success = main()
    sys.exit(0 if test_success else 1)
