import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = Path(__file__).resolve().parent / "fonts" / "NanumBarunGothic.ttf"
MIN_FONT_SIZE = 8
LINE_SPACING = 1.2
ERASE_MARGIN = 3  # bbox 지우기 시 추가 마진 (px)


def _get_bbox_rect(bbox: list[list[float]]) -> tuple[int, int, int, int]:
    """4점 polygon에서 (x_min, y_min, x_max, y_max) 추출."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _get_background_color(
    img: Image.Image, x1: int, y1: int, x2: int, y2: int
) -> tuple[int, ...]:
    """bbox 외곽 5px 띠의 중앙값(median)으로 배경색 결정."""
    margin = 5
    w, h = img.size
    pixels = []

    for y in range(max(0, y1 - margin), y1):
        for x in range(max(0, x1 - margin), min(w, x2 + margin)):
            pixels.append(img.getpixel((x, y)))
    for y in range(y2, min(h, y2 + margin)):
        for x in range(max(0, x1 - margin), min(w, x2 + margin)):
            pixels.append(img.getpixel((x, y)))
    for y in range(max(0, y1), min(h, y2)):
        for x in range(max(0, x1 - margin), x1):
            pixels.append(img.getpixel((x, y)))
    for y in range(max(0, y1), min(h, y2)):
        for x in range(x2, min(w, x2 + margin)):
            pixels.append(img.getpixel((x, y)))

    if not pixels:
        return (255, 255, 255)

    arr = np.array(pixels)
    median = np.median(arr, axis=0).astype(int)
    return tuple(median)


def _get_text_color(bg_color: tuple[int, ...]) -> tuple[int, int, int]:
    """배경 밝기에 따라 텍스트 색상 결정 (luminance > 128 → 검정, 아니면 흰색)."""
    luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


def _calc_font_size(
    text: str, box_w: int, box_h: int, font_path: str
) -> tuple[int, list[str]]:
    """bbox 면적을 최대한 채우는 폰트 크기와 줄바꿈된 텍스트 반환."""
    if not text:
        return MIN_FONT_SIZE, [""]

    n = len(text)
    estimated = int(math.sqrt(box_w * box_h / (n * LINE_SPACING)))
    estimated = max(MIN_FONT_SIZE, min(estimated, box_h))

    font_size = estimated
    for _ in range(5):
        font = ImageFont.truetype(font_path, font_size)
        lines = _wrap_text(text, font, box_w)
        total_h = int(len(lines) * font_size * LINE_SPACING)

        if total_h <= box_h:
            bigger = font_size + 1
            font_b = ImageFont.truetype(font_path, bigger)
            lines_b = _wrap_text(text, font_b, box_w)
            total_h_b = int(len(lines_b) * bigger * LINE_SPACING)
            if total_h_b <= box_h:
                font_size = bigger
                lines = lines_b
                continue
            break
        else:
            font_size = max(MIN_FONT_SIZE, font_size - 1)
            if font_size == MIN_FONT_SIZE:
                font = ImageFont.truetype(font_path, font_size)
                lines = _wrap_text(text, font, box_w)
                break

    return font_size, lines


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """텍스트를 max_width에 맞게 줄바꿈."""
    lines = []
    current = ""
    for char in text:
        test = current + char
        bbox = font.getbbox(test)
        w = bbox[2] - bbox[0]
        if w > max_width and current:
            lines.append(current)
            current = char
        else:
            current = test
    if current:
        lines.append(current)
    return lines or [""]


def _erase_bbox(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, ...]:
    """bbox 영역을 마진 포함하여 배경색으로 지우고, 사용한 배경색 반환."""
    w, h = img.size
    bg_color = _get_background_color(img, x1, y1, x2, y2)
    ex1 = max(0, x1 - ERASE_MARGIN)
    ey1 = max(0, y1 - ERASE_MARGIN)
    ex2 = min(w, x2 + ERASE_MARGIN)
    ey2 = min(h, y2 + ERASE_MARGIN)
    draw.rectangle([ex1, ey1, ex2, ey2], fill=bg_color)
    return bg_color


def run_render(
    translated_json_path: str,
    image_path: str,
    font_path: str | None = None,
) -> Path:
    tr_path = Path(translated_json_path)
    img_path = Path(image_path)

    if not tr_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {translated_json_path}")
    if not img_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {image_path}")

    font_file = font_path or str(DEFAULT_FONT)
    if not Path(font_file).exists():
        raise FileNotFoundError(f"폰트 파일을 찾을 수 없습니다: {font_file}")

    with open(tr_path, encoding="utf-8") as f:
        tr_data = json.load(f)

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for item in tr_data["translations"]:
        translated = item["translated"]
        if not translated.strip():
            continue

        bboxes = item["bboxes"]
        if not bboxes:
            continue

        # 1단계: 모든 개별 bbox를 마진 포함하여 배경색으로 지우기
        last_bg = None
        for bbox in bboxes:
            bx1, by1, bx2, by2 = _get_bbox_rect(bbox)
            last_bg = _erase_bbox(draw, img, bx1, by1, bx2, by2)

        # 2단계: 개별 bbox 단위로 텍스트 분배 및 렌더링
        # 번역 텍스트를 각 bbox의 면적 비율로 분배
        bbox_rects = [_get_bbox_rect(b) for b in bboxes]
        bbox_areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in bbox_rects]
        total_area = sum(bbox_areas) or 1

        text_remaining = translated
        for i, (bx1, by1, bx2, by2) in enumerate(bbox_rects):
            box_w = bx2 - bx1
            box_h = by2 - by1
            if box_w <= 0 or box_h <= 0:
                continue

            # 이 bbox에 할당할 글자 수 (면적 비율)
            if i < len(bbox_rects) - 1:
                ratio = bbox_areas[i] / total_area
                char_count = max(1, round(len(translated) * ratio))
                chunk = text_remaining[:char_count]
                text_remaining = text_remaining[char_count:]
            else:
                chunk = text_remaining

            if not chunk.strip():
                continue

            # 배경색/텍스트색 결정
            bg_color = _get_background_color(img, bx1, by1, bx2, by2)
            text_color = _get_text_color(bg_color)

            # 폰트 크기 결정 (개별 bbox 크기 기준)
            font_size, lines = _calc_font_size(chunk, box_w, box_h, font_file)
            font = ImageFont.truetype(font_file, font_size)

            # 텍스트를 bbox 중앙에 렌더링
            total_text_h = int(len(lines) * font_size * LINE_SPACING)
            y_offset = by1 + (box_h - total_text_h) // 2

            for line in lines:
                line_bbox = font.getbbox(line)
                line_w = line_bbox[2] - line_bbox[0]
                x_offset = bx1 + (box_w - line_w) // 2
                draw.text((x_offset, y_offset), line, fill=text_color, font=font)
                y_offset += int(font_size * LINE_SPACING)

    output_path = Path.cwd() / f"{img_path.stem}_rendered{img_path.suffix}"
    img.save(output_path)
    return output_path
