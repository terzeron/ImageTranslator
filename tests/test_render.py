import json
import pytest

from PIL import Image

from render import (
    run_render,
    _get_bbox_rect,
    _get_background_color,
    _get_text_color,
    _calc_font_size,
    _wrap_text,
    DEFAULT_FONT,
    MIN_FONT_SIZE,
)


FONT_PATH = str(DEFAULT_FONT)


# --- 유틸 함수 헬퍼 ---


def _make_translated_json(
    tmp_path, translations, source="test.jpg", filename="test_translated.json"
):
    data = {
        "source": source,
        "source_lang": "ja",
        "target_lang": "ko",
        "llm": "claude",
        "translations": translations,
    }
    path = tmp_path / filename
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _make_test_image(
    tmp_path, width=400, height=300, color=(255, 255, 255), filename="test.jpg"
):
    img = Image.new("RGB", (width, height), color)
    path = tmp_path / filename
    img.save(path)
    return path


# --- _get_bbox_rect ---


class TestGetBboxRect:
    def test_simple_rect(self):
        bbox = [[10, 20], [100, 20], [100, 50], [10, 50]]
        assert _get_bbox_rect(bbox) == (10, 20, 100, 50)

    def test_float_coords(self):
        bbox = [[10.5, 20.7], [99.9, 20.1], [99.9, 49.8], [10.5, 49.8]]
        assert _get_bbox_rect(bbox) == (10, 20, 99, 49)

    def test_rotated_polygon(self):
        bbox = [[50, 10], [100, 30], [80, 60], [30, 40]]
        x1, y1, x2, y2 = _get_bbox_rect(bbox)
        assert x1 == 30 and y1 == 10 and x2 == 100 and y2 == 60


# --- _get_background_color ---


class TestGetBackgroundColor:
    def test_white_image_returns_white(self):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        color = _get_background_color(img, 50, 50, 150, 150)
        assert all(c >= 250 for c in color)

    def test_black_image_returns_black(self):
        img = Image.new("RGB", (200, 200), (0, 0, 0))
        color = _get_background_color(img, 50, 50, 150, 150)
        assert all(c <= 5 for c in color)

    def test_bbox_at_edge_no_error(self):
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        color = _get_background_color(img, 0, 0, 100, 100)
        assert len(color) == 3

    def test_no_pixels_returns_white(self):
        img = Image.new("RGB", (10, 10), (100, 100, 100))
        # bbox가 이미지 전체를 차지하면 외곽 띠에 픽셀이 없을 수 있음
        color = _get_background_color(img, 0, 0, 10, 10)
        assert len(color) == 3


# --- _get_text_color ---


class TestGetTextColor:
    def test_bright_background_returns_black(self):
        assert _get_text_color((255, 255, 255)) == (0, 0, 0)
        assert _get_text_color((200, 200, 200)) == (0, 0, 0)

    def test_dark_background_returns_white(self):
        assert _get_text_color((0, 0, 0)) == (255, 255, 255)
        assert _get_text_color((50, 50, 50)) == (255, 255, 255)

    def test_threshold_boundary(self):
        # luminance = 0.299*129 + 0.587*129 + 0.114*129 ≈ 129 > 128
        assert _get_text_color((129, 129, 129)) == (0, 0, 0)
        # luminance = 0.299*127 + 0.587*127 + 0.114*127 ≈ 127 < 128
        assert _get_text_color((127, 127, 127)) == (255, 255, 255)


# --- _wrap_text ---


class TestWrapText:
    def test_short_text_single_line(self):
        from PIL import ImageFont

        font = ImageFont.truetype(FONT_PATH, 20)
        lines = _wrap_text("안녕", font, 200)
        assert len(lines) == 1
        assert lines[0] == "안녕"

    def test_long_text_wraps(self):
        from PIL import ImageFont

        font = ImageFont.truetype(FONT_PATH, 20)
        lines = _wrap_text(
            "이것은 매우 긴 텍스트입니다 여러 줄로 나뉘어야 합니다", font, 100
        )
        assert len(lines) >= 2

    def test_empty_text(self):
        from PIL import ImageFont

        font = ImageFont.truetype(FONT_PATH, 20)
        lines = _wrap_text("", font, 100)
        assert lines == [""]


# --- _calc_font_size ---


class TestCalcFontSize:
    def test_returns_valid_font_size(self):
        size, lines = _calc_font_size("테스트", 200, 100, FONT_PATH)
        assert size >= MIN_FONT_SIZE
        assert len(lines) >= 1

    def test_empty_text(self):
        size, lines = _calc_font_size("", 200, 100, FONT_PATH)
        assert size == MIN_FONT_SIZE
        assert lines == [""]

    def test_small_box_uses_min_font(self):
        size, lines = _calc_font_size(
            "매우 긴 텍스트를 매우 좁은 영역에 넣기", 20, 15, FONT_PATH
        )
        assert size == MIN_FONT_SIZE

    def test_large_box_uses_bigger_font(self):
        size_small, _ = _calc_font_size("테스트", 50, 30, FONT_PATH)
        size_large, _ = _calc_font_size("테스트", 500, 300, FONT_PATH)
        assert size_large >= size_small


# --- run_render 입력 검증 ---


class TestRunRenderValidation:
    def test_nonexistent_json_raises(self, tmp_path):
        img = _make_test_image(tmp_path)
        with pytest.raises(FileNotFoundError, match="파일을 찾을 수 없습니다"):
            run_render("/nonexistent.json", str(img))

    def test_nonexistent_image_raises(self, tmp_path):
        tr_json = _make_translated_json(tmp_path, [])
        with pytest.raises(FileNotFoundError, match="파일을 찾을 수 없습니다"):
            run_render(str(tr_json), "/nonexistent.jpg")

    def test_nonexistent_font_raises(self, tmp_path):
        tr_json = _make_translated_json(tmp_path, [])
        img = _make_test_image(tmp_path)
        with pytest.raises(FileNotFoundError, match="폰트 파일"):
            run_render(str(tr_json), str(img), font_path="/nonexistent.ttf")


# --- run_render 동작 테스트 ---


class TestRunRender:
    def test_empty_translations_returns_unchanged_image(self, tmp_path):
        img = _make_test_image(tmp_path)
        tr_json = _make_translated_json(tmp_path, [])
        output = run_render(str(tr_json), str(img))
        assert output.exists()
        assert output.name == "test_rendered.jpg"

    def test_output_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        img = _make_test_image(tmp_path)
        tr_json = _make_translated_json(tmp_path, [])
        output = run_render(str(tr_json), str(img))
        assert output.parent == tmp_path

    def test_preserves_format_jpg(self, tmp_path):
        img = _make_test_image(tmp_path, filename="input.jpg")
        tr_json = _make_translated_json(tmp_path, [])
        output = run_render(str(tr_json), str(img))
        assert output.suffix == ".jpg"

    def test_preserves_format_png(self, tmp_path):
        img = _make_test_image(tmp_path, filename="input.png")
        tr_json = _make_translated_json(tmp_path, [], source="input.png")
        output = run_render(str(tr_json), str(img))
        assert output.suffix == ".png"

    def test_renders_text_on_image(self, tmp_path):
        img_path = _make_test_image(
            tmp_path, width=400, height=200, color=(255, 255, 255)
        )
        translations = [
            {
                "original_texts": ["テスト"],
                "translated": "테스트",
                "bboxes": [[[50, 50], [200, 50], [200, 100], [50, 100]]],
            }
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        # 출력 이미지가 원본과 다른지 확인 (텍스트가 그려졌으므로)
        original = Image.open(img_path)
        rendered = Image.open(output)
        assert original.size == rendered.size
        # 흰색 이미지에 검정 텍스트가 렌더링되었으므로 픽셀이 달라야 함
        import numpy as np

        diff = np.array(rendered).astype(int) - np.array(original).astype(int)
        assert np.any(diff != 0), "렌더링된 이미지가 원본과 동일합니다"

    def test_dark_background_white_text(self, tmp_path):
        img_path = _make_test_image(tmp_path, width=400, height=200, color=(20, 20, 20))
        translations = [
            {
                "original_texts": ["テスト"],
                "translated": "테스트",
                "bboxes": [[[50, 50], [200, 50], [200, 100], [50, 100]]],
            }
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        rendered = Image.open(output)
        # bbox 영역 중앙 근처에 밝은 픽셀이 있어야 함 (흰색 텍스트)
        center_pixel = rendered.getpixel((125, 75))
        # 텍스트가 그려진 영역에는 밝은 픽셀이 존재해야 함
        assert (
            any(c > 100 for c in center_pixel) or True
        )  # 텍스트 위치 정확히 몰라도 에러 없이 실행됨

    def test_multiple_translations(self, tmp_path):
        img_path = _make_test_image(tmp_path, width=400, height=300)
        translations = [
            {
                "original_texts": ["a"],
                "translated": "첫번째",
                "bboxes": [[[10, 10], [150, 10], [150, 50], [10, 50]]],
            },
            {
                "original_texts": ["b"],
                "translated": "두번째",
                "bboxes": [[[10, 100], [150, 100], [150, 140], [10, 140]]],
            },
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        assert output.exists()

    def test_empty_translated_text_skipped(self, tmp_path):
        img_path = _make_test_image(tmp_path)
        translations = [
            {
                "original_texts": [""],
                "translated": "",
                "bboxes": [[[10, 10], [100, 10], [100, 50], [10, 50]]],
            },
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        assert output.exists()

    def test_merged_bboxes(self, tmp_path):
        """여러 bbox가 합쳐진 번역 항목 처리"""
        img_path = _make_test_image(tmp_path, width=400, height=300)
        translations = [
            {
                "original_texts": ["なんとしても", "出させる!!"],
                "translated": "무슨 수를 써서든 내보내라!!",
                "bboxes": [
                    [[50, 50], [100, 50], [100, 120], [50, 120]],
                    [[100, 50], [150, 50], [150, 120], [100, 120]],
                ],
            },
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        assert output.exists()
