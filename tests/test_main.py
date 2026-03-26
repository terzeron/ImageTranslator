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


class TestPipelineCache:
    """pipeline 캐시 동작 검증. LLM 호출을 mock하여 테스트."""

    def _make_ocr_cache(self, tmp_path, stem="test"):
        """OCR 캐시 JSON 생성."""
        data = {
            "source": f"{stem}.jpg",
            "lang": "ja",
            "model": "v5",
            "results": [
                {
                    "bbox": [[10, 10], [50, 10], [50, 30], [10, 30]],
                    "text": "テスト",
                    "confidence": 0.95,
                }
            ],
        }
        path = tmp_path / f"{stem}_ocr.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def _make_translated_cache(self, tmp_path, stem="test"):
        """번역 캐시 JSON 생성."""
        data = {
            "source": f"{stem}.jpg",
            "source_lang": "ja",
            "target_lang": "ko",
            "llm": "claude",
            "translations": [
                {
                    "original_texts": ["テスト"],
                    "translated": "테스트",
                    "bboxes": [[[10, 10], [50, 10], [50, 30], [10, 30]]],
                }
            ],
        }
        path = tmp_path / f"{stem}_translated.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def _make_test_image(self, tmp_path, stem="test"):
        from PIL import Image

        img = Image.new("RGB", (100, 100), (255, 255, 255))
        path = tmp_path / f"{stem}.jpg"
        img.save(path)
        return path

    def test_uses_ocr_cache(self, tmp_path):
        """OCR 캐시가 있으면 OCR을 건너뛰고 캐시 사용"""
        img = self._make_test_image(tmp_path)
        ocr_cache = self._make_ocr_cache(tmp_path)
        tr_cache = self._make_translated_cache(tmp_path)
        result = run_cli(
            "pipeline",
            "--lang",
            "ja",
            "--model",
            "v5",
            "--llm",
            "claude",
            str(img),
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert "캐시 사용" in result.stdout

    def test_uses_translated_cache(self, tmp_path):
        """번역 캐시가 있으면 LLM 호출을 건너뛰고 캐시 사용"""
        img = self._make_test_image(tmp_path)
        ocr_cache = self._make_ocr_cache(tmp_path)
        tr_cache = self._make_translated_cache(tmp_path)
        result = run_cli(
            "pipeline",
            "--lang",
            "ja",
            "--model",
            "v5",
            "--llm",
            "claude",
            str(img),
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert result.stdout.count("캐시 사용") == 2  # OCR + 번역 모두 캐시

    def test_reset_ignores_cache(self, tmp_path):
        """--reset 옵션이면 캐시를 무시하고 재생성"""
        img = self._make_test_image(tmp_path)
        self._make_ocr_cache(tmp_path)
        self._make_translated_cache(tmp_path)
        result = run_cli(
            "pipeline",
            "--lang",
            "ja",
            "--model",
            "v5",
            "--llm",
            "claude",
            "--reset",
            str(img),
            cwd=str(tmp_path),
        )
        # --reset이면 OCR을 다시 실행 → "OCR 처리 중" 출력
        assert "OCR 처리 중" in result.stdout
        assert "캐시 사용" not in result.stdout

    def test_no_cache_runs_normally(self, tmp_path):
        """캐시 없으면 정상 실행 (OCR은 실행, 번역은 API 필요하므로 OCR까지만 확인)"""
        img_src = RESOURCES_DIR / "synthetic_en.png"
        img = tmp_path / "nocache.png"
        img.write_bytes(img_src.read_bytes())
        result = run_cli(
            "pipeline",
            "--lang",
            "en",
            "--model",
            "v4",
            "--llm",
            "claude",
            str(img),
            cwd=str(tmp_path),
        )
        # API 한도로 실패할 수 있지만 OCR은 성공해야 함
        assert "OCR 처리 중" in result.stdout
