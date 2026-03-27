import json
import pytest
from unittest.mock import patch

from translate import (
    run_translate,
    _build_prompt,
    _parse_llm_response,
    LLM_CHOICES,
    LANGUAGE_TRAITS,
)


# --- 테스트용 OCR JSON 생성 헬퍼 ---


def _make_ocr_json(tmp_path, texts, lang="ja", filename="test_ocr.json"):
    data = {
        "source": "test.jpg",
        "lang": lang,
        "model": "v5",
        "results": [
            {
                "bbox": [[0, 0], [100, 0], [100, 30], [0, 30]],
                "text": t,
                "confidence": 0.9,
            }
            for t in texts
        ],
    }
    path = tmp_path / filename
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


# --- 상수 테스트 ---


class TestTranslateConstants:
    def test_llm_choices(self):
        assert "claude" in LLM_CHOICES
        assert "gemini" in LLM_CHOICES

    def test_language_traits_has_supported_langs(self):
        for lang in ("ja", "zh", "ko", "en"):
            assert lang in LANGUAGE_TRAITS


# --- 프롬프트 구성 테스트 ---


def _make_ocr_results(texts):
    """텍스트 리스트에서 bbox 포함 OCR 결과 dict 리스트 생성."""
    return [
        {
            "bbox": [[i * 30, 0], [i * 30 + 20, 0], [i * 30 + 20, 20], [i * 30, 20]],
            "text": t,
            "confidence": 0.9,
        }
        for i, t in enumerate(texts)
    ]


class TestBuildPrompt:
    def test_prompt_contains_source_texts(self):
        prompt = _build_prompt(_make_ocr_results(["こんにちは", "世界"]), "ja", "ko")
        assert "こんにちは" in prompt
        assert "世界" in prompt

    def test_prompt_contains_target_language(self):
        prompt = _build_prompt(_make_ocr_results(["hello"]), "en", "ko")
        assert "한국어" in prompt

    def test_prompt_contains_language_traits(self):
        prompt = _build_prompt(_make_ocr_results(["テスト"]), "ja", "ko")
        assert "세로쓰기" in prompt

    def test_prompt_contains_json_format_instruction(self):
        prompt = _build_prompt(_make_ocr_results(["test"]), "en", "ko")
        assert "indices" in prompt
        assert "translated" in prompt

    def test_prompt_for_chinese_mentions_right_to_left(self):
        prompt = _build_prompt(_make_ocr_results(["测试"]), "zh", "ko")
        assert "우→좌" in prompt

    def test_prompt_contains_coordinates(self):
        prompt = _build_prompt(_make_ocr_results(["テスト"]), "ja", "ko")
        assert "x=" in prompt
        assert "y=" in prompt

    def test_prompt_warns_about_distance(self):
        prompt = _build_prompt(_make_ocr_results(["a"]), "ja", "ko")
        assert "멀리 떨어진" in prompt or "위치가 가까운" in prompt


# --- LLM 응답 파싱 테스트 ---


class TestParseLlmResponse:
    def test_parse_plain_json(self):
        response = '[{"indices": [0, 1], "translated": "안녕 세계"}]'
        result = _parse_llm_response(response)
        assert len(result) == 1
        assert result[0]["indices"] == [0, 1]
        assert result[0]["translated"] == "안녕 세계"

    def test_parse_json_with_markdown_code_block(self):
        response = '```json\n[{"indices": [0], "translated": "테스트"}]\n```'
        result = _parse_llm_response(response)
        assert len(result) == 1

    def test_parse_multiple_groups(self):
        response = """[
            {"indices": [0], "translated": "첫번째"},
            {"indices": [1, 2], "translated": "두번째 세번째"}
        ]"""
        result = _parse_llm_response(response)
        assert len(result) == 2

    def test_parse_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_response("이것은 JSON이 아닙니다")


# --- run_translate 입력 검증 ---


class TestRunTranslateValidation:
    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError, match="파일을 찾을 수 없습니다"):
            run_translate("/nonexistent/path.json")

    def test_unsupported_llm_raises(self, tmp_path):
        ocr_json = _make_ocr_json(tmp_path, ["テスト"])
        with pytest.raises(ValueError, match="지원하지 않는 LLM"):
            run_translate(str(ocr_json), llm="unknown")


# --- run_translate 빈 결과 처리 ---


class TestRunTranslateEmptyResults:
    def test_empty_results_returns_empty_translations(self, tmp_path):
        data = {
            "source": "empty.jpg",
            "lang": "ja",
            "model": "v5",
            "results": [],
        }
        path = tmp_path / "empty_ocr.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        output = run_translate(str(path))
        with open(output) as f:
            result = json.load(f)
        assert result["translations"] == []
        assert result["source_lang"] == "ja"
        assert result["target_lang"] == "ko"


# --- run_translate mock 테스트 ---


MOCK_LLM_RESPONSE = json.dumps(
    [
        {"indices": [0, 1], "translated": "안녕 세계"},
        {"indices": [2], "translated": "테스트"},
    ]
)


class TestRunTranslateWithMock:
    @patch("translate._call_claude", return_value=MOCK_LLM_RESPONSE)
    def test_output_file_created(self, mock_claude, tmp_path):
        ocr_json = _make_ocr_json(tmp_path, ["こんにちは", "世界", "テスト"])
        output = run_translate(str(ocr_json), target_lang="ko", llm="claude")
        assert output.exists()
        assert output.name == "test_translated.json"

    @patch("translate._call_claude", return_value=MOCK_LLM_RESPONSE)
    def test_output_has_required_keys(self, mock_claude, tmp_path):
        ocr_json = _make_ocr_json(tmp_path, ["こんにちは", "世界", "テスト"])
        output = run_translate(str(ocr_json))
        with open(output) as f:
            data = json.load(f)
        assert "source" in data
        assert "source_lang" in data
        assert "target_lang" in data
        assert "llm" in data
        assert "translations" in data

    @patch("translate._call_claude", return_value=MOCK_LLM_RESPONSE)
    def test_translations_structure(self, mock_claude, tmp_path):
        ocr_json = _make_ocr_json(tmp_path, ["こんにちは", "世界", "テスト"])
        output = run_translate(str(ocr_json))
        with open(output) as f:
            data = json.load(f)
        translations = data["translations"]
        assert len(translations) == 2
        # 첫 번째 그룹: 합쳐진 텍스트
        assert translations[0]["original_texts"] == ["こんにちは", "世界"]
        assert translations[0]["translated"] == "안녕 세계"
        assert len(translations[0]["bboxes"]) == 2
        # 두 번째 그룹: 단일 텍스트
        assert translations[1]["original_texts"] == ["テスト"]
        assert translations[1]["translated"] == "테스트"

    @patch("translate._call_claude", return_value=MOCK_LLM_RESPONSE)
    def test_claude_called_with_prompt(self, mock_claude, tmp_path):
        ocr_json = _make_ocr_json(tmp_path, ["テスト"])
        run_translate(str(ocr_json), llm="claude")
        mock_claude.assert_called_once()
        prompt = mock_claude.call_args[0][0]
        assert "テスト" in prompt

    @patch("translate._call_gemini", return_value=MOCK_LLM_RESPONSE)
    def test_gemini_called_when_specified(self, mock_gemini, tmp_path):
        ocr_json = _make_ocr_json(tmp_path, ["テスト"])
        run_translate(str(ocr_json), llm="gemini")
        mock_gemini.assert_called_once()

    @patch("translate._call_claude", return_value=MOCK_LLM_RESPONSE)
    def test_target_lang_passed_to_output(self, mock_claude, tmp_path):
        ocr_json = _make_ocr_json(tmp_path, ["hello"])
        output = run_translate(str(ocr_json), target_lang="ja")
        with open(output) as f:
            data = json.load(f)
        assert data["target_lang"] == "ja"

    @patch("translate._call_claude", return_value=MOCK_LLM_RESPONSE)
    def test_source_lang_from_ocr_json(self, mock_claude, tmp_path):
        ocr_json = _make_ocr_json(tmp_path, ["hello"], lang="en")
        output = run_translate(str(ocr_json))
        with open(output) as f:
            data = json.load(f)
        assert data["source_lang"] == "en"

    @patch("translate._call_claude", return_value=MOCK_LLM_RESPONSE)
    def test_llm_field_in_output(self, mock_claude, tmp_path):
        ocr_json = _make_ocr_json(tmp_path, ["テスト"])
        output = run_translate(str(ocr_json), llm="claude")
        with open(output) as f:
            data = json.load(f)
        assert data["llm"] == "claude"

    @patch("translate._call_gemini", return_value=MOCK_LLM_RESPONSE)
    def test_gemini_llm_field_in_output(self, mock_gemini, tmp_path):
        ocr_json = _make_ocr_json(tmp_path, ["テスト"])
        output = run_translate(str(ocr_json), llm="gemini")
        with open(output) as f:
            data = json.load(f)
        assert data["llm"] == "gemini"

    @patch("translate._call_claude", return_value=MOCK_LLM_RESPONSE)
    def test_output_file_in_cwd(self, mock_claude, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ocr_json = _make_ocr_json(tmp_path, ["テスト"])
        output = run_translate(str(ocr_json))
        assert output.parent == tmp_path

    @patch("translate._call_claude")
    def test_out_of_range_index_ignored(self, mock_claude, tmp_path):
        """LLM이 존재하지 않는 인덱스를 반환해도 에러 없이 처리"""
        mock_claude.return_value = json.dumps(
            [
                {"indices": [0, 99], "translated": "테스트"},
            ]
        )
        ocr_json = _make_ocr_json(tmp_path, ["テスト"])
        output = run_translate(str(ocr_json))
        with open(output) as f:
            data = json.load(f)
        # 인덱스 0은 포함, 99는 범위 밖이므로 무시
        assert len(data["translations"]) == 1
        assert data["translations"][0]["original_texts"] == ["テスト"]
        assert len(data["translations"][0]["bboxes"]) == 1


class TestBuildPromptEdgeCases:
    def test_unknown_language_uses_code_fallback(self):
        prompt = _build_prompt(_make_ocr_results(["test"]), "ar", "ko")
        assert "ar" in prompt

    def test_unknown_target_lang_uses_code(self):
        prompt = _build_prompt(_make_ocr_results(["test"]), "en", "fr")
        assert "fr" in prompt

    def test_empty_text_list(self):
        prompt = _build_prompt([], "ja", "ko")
        assert "한국어" in prompt

    def test_text_indices_in_prompt(self):
        prompt = _build_prompt(_make_ocr_results(["aaa", "bbb", "ccc"]), "en", "ko")
        assert "aaa" in prompt
        assert "bbb" in prompt
        assert "ccc" in prompt


class TestParseLlmResponseEdgeCases:
    def test_parse_with_leading_whitespace(self):
        response = '   [{"indices": [0], "translated": "테스트"}]'
        result = _parse_llm_response(response)
        assert len(result) == 1

    def test_parse_code_block_without_language(self):
        response = '```\n[{"indices": [0], "translated": "테스트"}]\n```'
        result = _parse_llm_response(response)
        assert len(result) == 1

    def test_parse_empty_array(self):
        result = _parse_llm_response("[]")
        assert result == []


# --- _call_claude (translate.py:59-71) ---


class TestCallClaude:
    def test_call_claude_returns_text(self):
        from unittest.mock import MagicMock
        import sys

        mock_mod = MagicMock()
        mock_msg = mock_mod.Anthropic.return_value.messages.create.return_value
        mock_msg.stop_reason = "end_turn"
        mock_msg.content = [MagicMock(text="hello")]

        with patch.dict(sys.modules, {"anthropic": mock_mod}):
            from translate import _call_claude

            result = _call_claude("test prompt")
        assert result == "hello"

    def test_call_claude_raises_on_max_tokens(self):
        from unittest.mock import MagicMock
        import sys

        mock_mod = MagicMock()
        mock_msg = mock_mod.Anthropic.return_value.messages.create.return_value
        mock_msg.stop_reason = "max_tokens"

        with patch.dict(sys.modules, {"anthropic": mock_mod}):
            from translate import _call_claude

            with pytest.raises(RuntimeError, match="max_tokens"):
                _call_claude("test prompt")


# --- _call_gemini (translate.py:75-82) ---


class TestCallGemini:
    def test_call_gemini_returns_text(self):
        from unittest.mock import MagicMock
        import sys

        mock_genai = MagicMock()
        mock_response = (
            mock_genai.Client.return_value.models.generate_content.return_value
        )
        mock_response.text = "gemini response"

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict(
            sys.modules, {"google": mock_google, "google.genai": mock_genai}
        ):
            from translate import _call_gemini

            result = _call_gemini("test prompt")
        assert result == "gemini response"


# --- 청크 번역: 200개 초과 OCR 결과 인덱스 변환 (translate.py:149) ---


class TestRunTranslateChunking:
    @patch("translate._call_claude")
    def test_chunked_translation_index_offset(self, mock_claude, tmp_path):
        """201개 텍스트 → 2 청크로 분할, 두 번째 청크 인덱스가 +200 보정됨"""
        # 첫 청크(0~199): 인덱스 [0]
        # 두 번째 청크(200): 인덱스 [0] → 전체 [200]
        mock_claude.side_effect = [
            json.dumps([{"indices": [0], "translated": "첫번째"}]),
            json.dumps([{"indices": [0], "translated": "마지막"}]),
        ]
        texts = [f"text_{i}" for i in range(201)]
        ocr_json = _make_ocr_json(tmp_path, texts)
        output = run_translate(str(ocr_json))
        with open(output) as f:
            data = json.load(f)

        assert len(data["translations"]) == 2
        assert data["translations"][0]["original_texts"] == ["text_0"]
        assert data["translations"][1]["original_texts"] == ["text_200"]
        # _call_claude가 2번 호출됨
        assert mock_claude.call_count == 2

    @patch("translate._call_claude")
    def test_empty_indices_group_skipped(self, mock_claude, tmp_path):
        """indices가 빈 그룹은 무시됨 (line 149 근처)"""
        mock_claude.return_value = json.dumps(
            [
                {"indices": [], "translated": "무시됨"},
                {"indices": [0], "translated": "유효"},
            ]
        )
        ocr_json = _make_ocr_json(tmp_path, ["テスト"])
        output = run_translate(str(ocr_json))
        with open(output) as f:
            data = json.load(f)
        assert len(data["translations"]) == 1
        assert data["translations"][0]["translated"] == "유효"
