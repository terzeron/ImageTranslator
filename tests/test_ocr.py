import json
import pytest

from ocr import run_ocr, LANG_MAP, MODEL_MAP, SUPPORTED_EXTENSIONS
from tests.conftest import RESOURCES_DIR


class TestOcrConstants:
    def test_lang_map_has_required_languages(self):
        assert set(LANG_MAP.keys()) == {"ja", "ko", "en", "zh"}

    def test_model_map_has_required_versions(self):
        assert set(MODEL_MAP.keys()) == {"v3", "v4", "v5"}

    def test_supported_extensions_include_common_formats(self):
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            assert ext in SUPPORTED_EXTENSIONS


class TestOcrInputValidation:
    def test_nonexistent_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="파일을 찾을 수 없습니다"):
            run_ocr(str(tmp_path / "nonexistent.png"))

    def test_unsupported_format_raises_value_error(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="지원하지 않는 이미지 포맷"):
            run_ocr(str(txt_file))


class TestOcrOutputFormat:
    """OCR 결과 JSON의 구조와 포맷을 검증한다."""

    @pytest.fixture(scope="class")
    def en_ocr_result(self, tmp_path_factory):
        """영어 합성 이미지 OCR 결과 (클래스 내 공유)"""
        src = RESOURCES_DIR / "synthetic_en.png"
        work = tmp_path_factory.mktemp("ocr")
        dest = work / "synthetic_en.png"
        dest.write_bytes(src.read_bytes())
        output_path = run_ocr(str(dest), lang="en", model="v4")
        with open(output_path) as f:
            return json.load(f)

    def test_output_has_required_keys(self, en_ocr_result):
        assert "source" in en_ocr_result
        assert "lang" in en_ocr_result
        assert "model" in en_ocr_result
        assert "results" in en_ocr_result

    def test_output_source_matches_filename(self, en_ocr_result):
        assert en_ocr_result["source"] == "synthetic_en.png"

    def test_output_lang_matches_input(self, en_ocr_result):
        assert en_ocr_result["lang"] == "en"

    def test_output_model_matches_input(self, en_ocr_result):
        assert en_ocr_result["model"] == "v4"

    def test_results_is_list(self, en_ocr_result):
        assert isinstance(en_ocr_result["results"], list)

    def test_each_result_has_bbox_text_confidence(self, en_ocr_result):
        for item in en_ocr_result["results"]:
            assert "bbox" in item
            assert "text" in item
            assert "confidence" in item

    def test_bbox_is_4_point_polygon(self, en_ocr_result):
        for item in en_ocr_result["results"]:
            assert len(item["bbox"]) == 4
            for point in item["bbox"]:
                assert len(point) == 2
                assert isinstance(point[0], float)
                assert isinstance(point[1], float)

    def test_confidence_is_between_0_and_1(self, en_ocr_result):
        for item in en_ocr_result["results"]:
            assert 0.0 <= item["confidence"] <= 1.0


class TestOcrJsonFile:
    """JSON 파일 저장 동작을 검증한다."""

    def test_output_file_created_in_cwd(self, tmp_path, monkeypatch):
        src = RESOURCES_DIR / "synthetic_en.png"
        dest = tmp_path / "test_img.png"
        dest.write_bytes(src.read_bytes())
        monkeypatch.chdir(tmp_path)
        output_path = run_ocr(str(dest), lang="en", model="v4")
        assert output_path.exists()
        assert output_path.parent == tmp_path
        assert output_path.name == "test_img_ocr.json"

    def test_output_file_is_valid_json(self, tmp_path):
        src = RESOURCES_DIR / "synthetic_en.png"
        dest = tmp_path / "valid.png"
        dest.write_bytes(src.read_bytes())
        output_path = run_ocr(str(dest), lang="en", model="v4")
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_overwrites_existing_ocr_json(self, tmp_path):
        src = RESOURCES_DIR / "synthetic_en.png"
        dest = tmp_path / "overwrite.png"
        dest.write_bytes(src.read_bytes())

        # 첫 번째 실행
        output_path = run_ocr(str(dest), lang="en", model="v4")
        first_mtime = output_path.stat().st_mtime

        # 약간의 시간 차이를 위해
        import time

        time.sleep(0.1)

        # 두 번째 실행 — 덮어쓰기
        output_path2 = run_ocr(str(dest), lang="en", model="v4")
        assert output_path == output_path2
        assert output_path2.stat().st_mtime > first_mtime


class TestOcrEnglishRecognition:
    """영어 합성 이미지의 OCR 인식 품질을 검증한다."""

    @pytest.fixture(scope="class")
    def en_texts(self, tmp_path_factory):
        src = RESOURCES_DIR / "synthetic_en.png"
        work = tmp_path_factory.mktemp("ocr_en")
        dest = work / "synthetic_en.png"
        dest.write_bytes(src.read_bytes())
        output_path = run_ocr(str(dest), lang="en", model="v5")
        with open(output_path) as f:
            data = json.load(f)
        return [r["text"] for r in data["results"]]

    def test_detects_at_least_one_text(self, en_texts):
        assert len(en_texts) >= 1

    def test_recognizes_hello_world(self, en_texts):
        combined = " ".join(en_texts).lower()
        assert "hello" in combined
        assert "world" in combined

    def test_recognizes_paddleocr_test(self, en_texts):
        combined = " ".join(en_texts).lower()
        assert "paddleocr" in combined
        assert "test" in combined


class TestOcrNoText:
    """텍스트 없는 이미지에 대해 빈 결과를 반환하는지 검증한다."""

    def test_no_text_image_returns_empty_results(self, tmp_path):
        src = RESOURCES_DIR / "no_text.png"
        dest = tmp_path / "no_text.png"
        dest.write_bytes(src.read_bytes())
        output_path = run_ocr(str(dest), lang="en", model="v4")
        with open(output_path) as f:
            data = json.load(f)
        assert data["results"] == []


class TestOcrJapaneseRecognition:
    """일본어 만화 이미지의 OCR 인식 품질을 검증한다 (v5 모델)."""

    @pytest.fixture(scope="class")
    def ja_dialog_result(self, tmp_path_factory):
        """대사가 많은 만화 페이지 OCR"""
        src = RESOURCES_DIR / "manga_ja_dialog.jpg"
        work = tmp_path_factory.mktemp("ocr_ja")
        dest = work / "manga_ja_dialog.jpg"
        dest.write_bytes(src.read_bytes())
        output_path = run_ocr(str(dest), lang="ja", model="v5")
        with open(output_path) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def ja_short_result(self, tmp_path_factory):
        """짧은 대사의 만화 페이지 OCR"""
        src = RESOURCES_DIR / "manga_ja_short.jpg"
        work = tmp_path_factory.mktemp("ocr_ja2")
        dest = work / "manga_ja_short.jpg"
        dest.write_bytes(src.read_bytes())
        output_path = run_ocr(str(dest), lang="ja", model="v5")
        with open(output_path) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def ja_sparse_result(self, tmp_path_factory):
        """텍스트가 적은 만화 페이지 OCR"""
        src = RESOURCES_DIR / "manga_ja_sparse.jpg"
        work = tmp_path_factory.mktemp("ocr_ja3")
        dest = work / "manga_ja_sparse.jpg"
        dest.write_bytes(src.read_bytes())
        output_path = run_ocr(str(dest), lang="ja", model="v5")
        with open(output_path) as f:
            return json.load(f)

    # --- 대사가 많은 페이지 ---

    def test_dialog_page_detects_multiple_texts(self, ja_dialog_result):
        assert len(ja_dialog_result["results"]) >= 10

    def test_dialog_page_contains_japanese_characters(self, ja_dialog_result):
        import re

        combined = "".join(r["text"] for r in ja_dialog_result["results"])
        # 히라가나, 카타카나, 한자 중 하나 이상 포함
        assert re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", combined)

    def test_dialog_page_recognizes_key_words(self, ja_dialog_result):
        """이전 OCR 결과에서 높은 confidence로 인식된 핵심 단어 포함 여부"""
        combined = "".join(r["text"] for r in ja_dialog_result["results"])
        # 이 페이지에서 확인된 주요 단어들 중 일부가 포함되어야 함
        keywords = ["日本", "戦場", "通信", "味方", "移動"]
        found = sum(1 for kw in keywords if kw in combined)
        assert found >= 2, f"키워드 매치 {found}/5, 텍스트: {combined[:200]}"

    def test_dialog_page_average_confidence_above_threshold(self, ja_dialog_result):
        scores = [
            r["confidence"] for r in ja_dialog_result["results"] if r["confidence"] > 0
        ]
        if scores:
            avg = sum(scores) / len(scores)
            assert avg >= 0.5, f"평균 confidence {avg:.3f} < 0.5"

    # --- 짧은 대사 페이지 ---

    def test_short_page_detects_some_text(self, ja_short_result):
        assert len(ja_short_result["results"]) >= 3

    def test_short_page_contains_hiragana(self, ja_short_result):
        import re

        combined = "".join(r["text"] for r in ja_short_result["results"])
        assert re.search(r"[\u3040-\u309F]", combined), "히라가나가 검출되지 않음"

    # --- 텍스트 적은 페이지 ---

    def test_sparse_page_handles_few_texts(self, ja_sparse_result):
        """텍스트가 적어도 에러 없이 처리됨"""
        assert isinstance(ja_sparse_result["results"], list)


class TestOcrActionPage:
    """텍스트가 거의 없는 액션 장면 (002-072.jpg)."""

    @pytest.fixture(scope="class")
    def action_result(self, tmp_path_factory):
        src = RESOURCES_DIR / "manga" / "002-072.jpg"
        work = tmp_path_factory.mktemp("ocr_action")
        dest = work / "002-072.jpg"
        dest.write_bytes(src.read_bytes())
        output_path = run_ocr(str(dest), lang="ja", model="v5")
        with open(output_path) as f:
            return json.load(f)

    def test_action_page_returns_valid_structure(self, action_result):
        assert isinstance(action_result["results"], list)

    def test_action_page_has_few_or_no_meaningful_text(self, action_result):
        """액션 장면은 의미 있는 텍스트가 거의 없어야 함"""
        meaningful = [r for r in action_result["results"] if r["confidence"] > 0.5]
        assert len(meaningful) <= 3


class TestOcrWrongLanguage:
    """한국어 이미지를 잘못된 언어(ja)로 인식하면 품질이 낮고,
    올바른 언어(ko)로 인식하면 품질이 높아지는지 검증."""

    @pytest.fixture(scope="class")
    def ko_image_path(self, tmp_path_factory):
        src = RESOURCES_DIR / "manga" / "001-001-006-001-136.jpg"
        work = tmp_path_factory.mktemp("ocr_ko")
        dest = work / "001-001-006-001-136.jpg"
        dest.write_bytes(src.read_bytes())
        return dest

    @pytest.fixture(scope="class")
    def result_with_wrong_lang(self, ko_image_path):
        output_path = run_ocr(str(ko_image_path), lang="ja", model="v5")
        with open(output_path) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def result_with_correct_lang(self, ko_image_path):
        output_path = run_ocr(str(ko_image_path), lang="ko", model="v5")
        with open(output_path) as f:
            return json.load(f)

    def test_wrong_lang_has_lower_confidence(self, result_with_wrong_lang):
        """ja로 한국어 이미지를 인식하면 평균 confidence가 낮음"""
        scores = [
            r["confidence"]
            for r in result_with_wrong_lang["results"]
            if r["confidence"] > 0
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        assert avg < 0.7, f"잘못된 언어인데 confidence가 높음: {avg:.3f}"

    def test_correct_lang_has_higher_confidence(self, result_with_correct_lang):
        """ko로 인식하면 평균 confidence가 높음"""
        scores = [
            r["confidence"]
            for r in result_with_correct_lang["results"]
            if r["confidence"] > 0
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        assert avg >= 0.9, f"올바른 언어인데 confidence가 낮음: {avg:.3f}"

    def test_correct_lang_recognizes_korean(self, result_with_correct_lang):
        """ko로 인식한 결과에 한글이 포함되어야 함"""
        import re

        combined = "".join(r["text"] for r in result_with_correct_lang["results"])
        assert re.search(r"[가-힣]", combined), "한글이 검출되지 않음"

    def test_correct_lang_recognizes_key_words(self, result_with_correct_lang):
        """이미지 속 핵심 한국어 단어가 인식되어야 함"""
        combined = "".join(r["text"] for r in result_with_correct_lang["results"])
        keywords = ["게타", "정부", "자위대", "폭발"]
        found = sum(1 for kw in keywords if kw in combined)
        assert found >= 2, f"키워드 매치 {found}/4, 텍스트: {combined[:200]}"

    def test_correct_lang_detects_more_text(
        self, result_with_wrong_lang, result_with_correct_lang
    ):
        """올바른 언어로 인식하면 더 많은 의미 있는 텍스트를 검출"""
        wrong_meaningful = len(
            [r for r in result_with_wrong_lang["results"] if r["confidence"] > 0.5]
        )
        correct_meaningful = len(
            [r for r in result_with_correct_lang["results"] if r["confidence"] > 0.5]
        )
        assert correct_meaningful > wrong_meaningful
