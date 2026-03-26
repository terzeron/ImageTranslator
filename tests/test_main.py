import subprocess
import sys
import json

from tests.conftest import RESOURCES_DIR, PROJECT_ROOT


def run_cli(
    *args: str, env_override: dict | None = None, cwd: str | None = None
) -> subprocess.CompletedProcess:
    import os

    env = os.environ.copy()
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    if env_override:
        env.update(env_override)
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd or str(PROJECT_ROOT),
        env=env,
    )


class TestCliNoArgs:
    def test_no_args_shows_help(self):
        result = run_cli()
        assert result.returncode == 1

    def test_help_flag(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "이미지 텍스트 번역 도구" in result.stdout


class TestCliOcrSubcommand:
    def test_ocr_help(self):
        result = run_cli("ocr", "--help")
        assert result.returncode == 0
        assert "image_path" in result.stdout
        assert "--lang" in result.stdout
        assert "--model" in result.stdout

    def test_ocr_nonexistent_file(self):
        result = run_cli("ocr", "/tmp/nonexistent_image.png")
        assert result.returncode != 0

    def test_ocr_with_real_image(self, tmp_path):
        src = RESOURCES_DIR / "synthetic_en.png"
        dest = tmp_path / "cli_test.png"
        dest.write_bytes(src.read_bytes())
        result = run_cli(
            "ocr", "--lang", "en", "--model", "v4", str(dest), cwd=str(tmp_path)
        )
        assert result.returncode == 0, result.stderr
        assert "OCR 결과 저장" in result.stdout

        output_json = tmp_path / "cli_test_ocr.json"
        assert output_json.exists()
        with open(output_json) as f:
            data = json.load(f)
        assert data["lang"] == "en"
        assert data["model"] == "v4"

    def test_ocr_invalid_lang(self):
        result = run_cli("ocr", "--lang", "fr", "dummy.png")
        assert result.returncode != 0

    def test_ocr_invalid_model(self):
        result = run_cli("ocr", "--model", "v99", "dummy.png")
        assert result.returncode != 0
