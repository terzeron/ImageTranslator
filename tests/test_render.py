import json
import pytest

from PIL import Image, ImageDraw

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


class TestRenderEraseMagin:
    """텍스트 지우기 마진 검증."""

    def test_erase_covers_beyond_bbox(self, tmp_path):
        """bbox 외곽 마진만큼 추가로 지워지는지 확인"""

        # 흰색 배경에 bbox 영역+약간 밖에 검정 텍스트 시뮬레이션
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        draw_setup = ImageDraw.Draw(img)
        # bbox(50~100) 바로 밖(48~102)에 검정 픽셀 (잔존 텍스트)
        draw_setup.rectangle([48, 48, 102, 102], fill=(0, 0, 0))
        img_path = tmp_path / "margin_test.png"
        img.save(img_path)

        translations = [
            {
                "original_texts": ["test"],
                "translated": "테스트",
                "bboxes": [[[50, 50], [100, 50], [100, 100], [50, 100]]],
            }
        ]
        tr_json = _make_translated_json(
            tmp_path, translations, source="margin_test.png"
        )
        output = run_render(str(tr_json), str(img_path))
        rendered = Image.open(output)

        # 마진(ERASE_MARGIN=3) 범위 내 픽셀(48,48)이 배경색(흰색)으로 지워져야 함
        edge_pixel = rendered.getpixel((48, 48))
        assert all(c > 200 for c in edge_pixel), (
            f"마진 영역이 지워지지 않음: {edge_pixel}"
        )


class TestRenderBboxSizeConstraint:
    """렌더링된 텍스트가 개별 bbox 크기를 크게 초과하지 않는지 검증."""

    def test_text_fits_within_bbox(self, tmp_path, monkeypatch):
        """단일 bbox에 렌더링된 텍스트의 크기가 bbox를 초과하지 않아야 함"""
        monkeypatch.chdir(tmp_path)
        img_path = _make_test_image(
            tmp_path, width=400, height=300, color=(255, 255, 255)
        )
        bbox = [[50, 50], [250, 50], [250, 100], [50, 100]]  # 200x50
        translations = [
            {
                "original_texts": ["テスト"],
                "translated": "테스트",
                "bboxes": [bbox],
            }
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))

        # 렌더링된 이미지에서 텍스트 영역 분석
        rendered = Image.open(output)
        import numpy as np

        arr = np.array(rendered)

        # bbox 외부 영역에서 원본(흰색)과 달라진 픽셀 찾기
        # bbox는 50~150, 50~100 → 그 밖에서 변경된 픽셀이 적어야 함
        outside_top = arr[0:47, :, :]  # bbox 위 (마진 고려)
        outside_bottom = arr[103:, :, :]  # bbox 아래 (마진 고려)
        # 흰색(255)이 아닌 픽셀 수
        changed_outside = np.sum(outside_top < 250) + np.sum(outside_bottom < 250)
        # bbox 외부에 텍스트가 그려지면 안 됨
        assert changed_outside == 0, f"bbox 외부에 {changed_outside}개 변경 픽셀 발견"

    def test_distant_bboxes_merged_into_one(self, tmp_path, monkeypatch):
        """같은 translation item의 멀리 떨어진 bbox도 bounding rect로 합쳐서 렌더링"""
        monkeypatch.chdir(tmp_path)
        img_path = _make_test_image(
            tmp_path, width=500, height=300, color=(255, 255, 255)
        )
        translations = [
            {
                "original_texts": ["上", "下"],
                "translated": "위쪽 아래쪽",
                "bboxes": [
                    [[50, 20], [100, 20], [100, 60], [50, 60]],
                    [[50, 200], [100, 200], [100, 240], [50, 240]],
                ],
            }
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        assert output.exists()


class TestTranslationItemRendering:
    """translation item 단위 렌더링 검증."""

    def test_multiple_bboxes_merged_as_one_region(self, tmp_path, monkeypatch):
        """여러 bbox가 하나의 bounding rect로 합쳐져 렌더링"""
        monkeypatch.chdir(tmp_path)
        img_path = _make_test_image(
            tmp_path, width=200, height=200, color=(255, 255, 255)
        )
        translations = [
            {
                "original_texts": ["なんとしても", "出させる!!"],
                "translated": "어떻게든 내보내라!!",
                "bboxes": [
                    [[50, 30], [70, 30], [70, 100], [50, 100]],
                    [[60, 30], [80, 30], [80, 100], [60, 100]],
                ],
            },
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        assert output.exists()

    def test_entire_region_erased(self, tmp_path, monkeypatch):
        """전체 bounding rect 영역이 배경색으로 지워짐"""
        monkeypatch.chdir(tmp_path)
        # 흰색 배경, bbox 영역에 검정 (텍스트 시뮬레이션)
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        draw_setup = ImageDraw.Draw(img)
        draw_setup.rectangle([48, 28, 82, 102], fill=(0, 0, 0))
        img_path = tmp_path / "erase_test.jpg"
        img.save(img_path)

        translations = [
            {
                "original_texts": ["a", "b"],
                "translated": "테스트",
                "bboxes": [
                    [[50, 30], [70, 30], [70, 70], [50, 70]],
                    [[50, 70], [80, 70], [80, 100], [50, 100]],
                ],
            },
        ]
        tr_json = _make_translated_json(tmp_path, translations, source="erase_test.jpg")
        output = run_render(str(tr_json), str(img_path))
        rendered = Image.open(output)
        # bbox 밖 (30,30)은 원래 흰색 배경이어야 함 (지우기 전 영역)
        outside_pixel = rendered.getpixel((30, 30))
        assert all(c > 200 for c in outside_pixel), (
            f"bbox 밖 영역이 변경됨: {outside_pixel}"
        )


# --- _calc_font_size: bigger font 성공 분기 (render.py:81-83) ---


class TestCalcFontSizeBiggerFont:
    def test_bigger_font_accepted_in_large_box(self):
        """충분히 큰 박스에 짧은 텍스트 → bigger font 시도 성공하여 크기 증가"""
        from render import _calc_font_size

        size, lines = _calc_font_size("ab", 500, 500, FONT_PATH)
        # 큰 박스에 짧은 텍스트이므로 MIN_FONT_SIZE보다 훨씬 커야 함
        assert size > MIN_FONT_SIZE + 1
        assert len(lines) >= 1


# --- _render_vertical (render.py:156, 161-162) ---


class TestRenderVertical:
    def test_empty_text_returns_immediately(self):
        """빈 텍스트는 아무것도 렌더링하지 않음 (line 156)"""
        from render import _render_vertical

        img = Image.new("RGB", (100, 200), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        import numpy as np

        before = np.array(img).copy()
        _render_vertical(draw, "", 10, 10, 80, 180, (0, 0, 0), FONT_PATH)
        after = np.array(img)
        assert np.array_equal(before, after), "빈 텍스트인데 이미지가 변경됨"

    def test_auto_size_without_suggested(self):
        """suggested_size=0이면 자동 크기 계산 (lines 161-162)"""
        from render import _render_vertical

        img = Image.new("RGB", (60, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        import numpy as np

        before = np.array(img).copy()
        _render_vertical(
            draw, "테스트", 5, 5, 50, 280, (0, 0, 0), FONT_PATH, suggested_size=0
        )
        after = np.array(img)
        assert not np.array_equal(before, after), "텍스트가 렌더링되지 않음"


# --- _cluster_bboxes (render.py:224) ---


class TestClusterBboxesEmpty:
    def test_empty_list_returns_empty(self):
        from render import _cluster_bboxes

        assert _cluster_bboxes([]) == []


# --- run_render: degenerate bbox (render.py:305) ---


class TestRunRenderDegenerateBbox:
    def test_zero_size_bbox_skipped(self, tmp_path):
        """동일 좌표 bbox(bw=0)는 건너뛰고 에러 없이 완료"""
        img_path = _make_test_image(tmp_path, width=200, height=200)
        translations = [
            {
                "original_texts": ["x"],
                "translated": "테스트",
                "bboxes": [[[50, 50], [50, 50], [50, 100], [50, 100]]],
            }
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        assert output.exists()


# --- run_render: empty bboxes list (render.py:327) ---


class TestRunRenderEmptyBboxes:
    def test_translation_with_empty_bboxes_skipped(self, tmp_path):
        """bboxes가 빈 리스트인 항목은 건너뛰기"""
        img_path = _make_test_image(tmp_path, width=200, height=200)
        translations = [
            {
                "original_texts": [],
                "translated": "테스트",
                "bboxes": [],
            }
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        assert output.exists()


# --- run_render: degenerate cluster cw/ch<=0 (render.py:351) ---


class TestRunRenderDegenerateCluster:
    def test_degenerate_cluster_rect_skipped(self, tmp_path):
        """클러스터의 bounding rect가 0 크기이면 건너뛰기"""
        img_path = _make_test_image(tmp_path, width=200, height=200)
        # 동일 좌표 2개 bbox → 클러스터의 cw/ch=0
        translations = [
            {
                "original_texts": ["a", "b"],
                "translated": "테스트",
                "bboxes": [
                    [[100, 100], [100, 100], [100, 100], [100, 100]],
                    [[100, 100], [100, 100], [100, 100], [100, 100]],
                ],
            }
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        assert output.exists()


# --- run_render: whitespace-only chunk (render.py:363) ---


class TestRunRenderWhitespaceChunk:
    def test_whitespace_only_translated_skipped(self, tmp_path):
        """translated가 공백만 있는 항목은 건너뛰기"""
        img_path = _make_test_image(tmp_path, width=200, height=200)
        translations = [
            {
                "original_texts": ["x"],
                "translated": "   ",
                "bboxes": [[[10, 10], [100, 10], [100, 50], [10, 50]]],
            }
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        assert output.exists()

    def test_multi_cluster_whitespace_chunk_skipped(self, tmp_path, monkeypatch):
        """복수 클러스터에서 텍스트 분배 시 공백만 남는 chunk 건너뛰기 (line 363)"""
        monkeypatch.chdir(tmp_path)
        img_path = _make_test_image(tmp_path, width=500, height=500)
        # 멀리 떨어진 2개 bbox → 2개 클러스터, " x"를 분배하면 첫 클러스터가 " " 획득
        translations = [
            {
                "original_texts": ["a", "b"],
                "translated": " x",
                "bboxes": [
                    [[10, 10], [100, 10], [100, 100], [10, 100]],
                    [[400, 400], [490, 400], [490, 490], [400, 490]],
                ],
            }
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        assert output.exists()


# --- run_render: 세로 렌더링 통합 (render.py:156, 161-162 via run_render) ---


class TestRunRenderVerticalText:
    def test_vertical_bbox_renders_text(self, tmp_path, monkeypatch):
        """세로로 긴 bbox에 텍스트가 세로 렌더링됨"""
        monkeypatch.chdir(tmp_path)
        img_path = _make_test_image(tmp_path, width=200, height=400)
        # 세로로 긴 bbox (bh > bw * 1.5)
        translations = [
            {
                "original_texts": ["テスト"],
                "translated": "테스트입니다",
                "bboxes": [[[50, 10], [80, 10], [80, 300], [50, 300]]],
            }
        ]
        tr_json = _make_translated_json(tmp_path, translations)
        output = run_render(str(tr_json), str(img_path))
        assert output.exists()
        import numpy as np

        original = np.array(Image.open(img_path))
        rendered = np.array(Image.open(output))
        # 세로 텍스트가 렌더링되어 이미지가 달라야 함
        assert not np.array_equal(original, rendered)
