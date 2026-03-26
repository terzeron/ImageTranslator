import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = Path(__file__).resolve().parent / "fonts" / "NanumBarunGothic.ttf"
MIN_FONT_SIZE = 8
LINE_SPACING = 1.2
ERASE_MARGIN = 5


def _get_bbox_rect(bbox: list[list[float]]) -> tuple[int, int, int, int]:
    """4점 polygon에서 (x_min, y_min, x_max, y_max) 추출."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _get_background_color(
    img: Image.Image, x1: int, y1: int, x2: int, y2: int
) -> tuple[int, ...]:
    """bbox 외곽 띠의 중앙값(median)으로 배경색 결정."""
    margin = 7
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


def _render_vertical(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    box_w: int,
    box_h: int,
    text_color: tuple[int, int, int],
    font_path: str,
    suggested_size: int = 0,
) -> None:
    """세로로 긴 영역에 텍스트를 세로쓰기 렌더링 (우→좌, 위→아래)."""
    n = len(text)
    if n == 0:
        return

    if suggested_size > 0:
        font_size = max(MIN_FONT_SIZE, min(suggested_size, box_w))
    else:
        font_size = int(math.sqrt(box_w * box_h / (n * LINE_SPACING * LINE_SPACING)))
        font_size = max(MIN_FONT_SIZE, min(font_size, box_w))

    char_h = int(font_size * LINE_SPACING)
    col_w = int(font_size * LINE_SPACING)
    chars_per_col = max(1, box_h // char_h)
    num_cols = math.ceil(n / chars_per_col)

    # 열이 너무 많으면 폰트 축소
    if num_cols * col_w > box_w:
        font_size = max(MIN_FONT_SIZE, box_w // max(1, num_cols) - 2)
        char_h = int(font_size * LINE_SPACING)
        col_w = int(font_size * LINE_SPACING)
        chars_per_col = max(1, box_h // char_h)

    font = ImageFont.truetype(font_path, font_size)

    # 우→좌 세로쓰기
    total_cols_w = num_cols * col_w
    x_start = x + box_w - col_w - max(0, (box_w - total_cols_w) // 2)

    col = 0
    row = 0
    for char in text:
        cx = x_start - col * col_w
        cy = y + row * char_h
        char_bbox = font.getbbox(char)
        char_w = char_bbox[2] - char_bbox[0]
        cx_centered = cx + (col_w - char_w) // 2
        draw.text((cx_centered, cy), char, fill=text_color, font=font)
        row += 1
        if row >= chars_per_col:
            row = 0
            col += 1


def _erase_rect(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    margin: int = ERASE_MARGIN,
) -> tuple[int, ...]:
    """영역을 마진 포함하여 배경색으로 지우고, 사용한 배경색 반환."""
    w, h = img.size
    bg_color = _get_background_color(img, x1, y1, x2, y2)
    ex1 = max(0, x1 - margin)
    ey1 = max(0, y1 - margin)
    ex2 = min(w, x2 + margin)
    ey2 = min(h, y2 + margin)
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

        # 1단계: 개별 bbox를 rect로 변환 + 배경색으로 지우기
        bbox_rects = []
        for bbox in bboxes:
            bx1, by1, bx2, by2 = _get_bbox_rect(bbox)
            _erase_rect(draw, img, bx1, by1, bx2, by2)
            bbox_rects.append((bx1, by1, bx2, by2))

        # 2단계: 원본 텍스트 + bbox에서 폰트 크기 추정 → 최대값 사용
        original_texts = item.get("original_texts", [])
        estimated_sizes = []
        for idx, (bx1, by1, bx2, by2) in enumerate(bbox_rects):
            bw, bh = bx2 - bx1, by2 - by1
            if bw <= 0 or bh <= 0:
                continue
            n = len(original_texts[idx]) if idx < len(original_texts) else 1
            n = max(1, n)
            if bh > bw * 1.5:
                # 세로쓰기: 높이를 글자 수로 나누고, 폭과 비교하여 작은 값
                estimated_sizes.append(min(bw, int(bh / n)))
            else:
                # 가로쓰기: 폭을 글자 수로 나누고, 높이와 비교하여 작은 값
                estimated_sizes.append(min(bh, int(bw / n)))
        font_size = max(estimated_sizes) if estimated_sizes else MIN_FONT_SIZE
        font_size = max(MIN_FONT_SIZE, font_size)

        # 3단계: 전체 bounding rect 계산 (렌더링 영역)
        all_x = [r[0] for r in bbox_rects] + [r[2] for r in bbox_rects]
        all_y = [r[1] for r in bbox_rects] + [r[3] for r in bbox_rects]
        rx1, ry1 = min(all_x), min(all_y)
        rx2, ry2 = max(all_x), max(all_y)
        rw, rh = rx2 - rx1, ry2 - ry1

        if rw <= 0 or rh <= 0:
            continue

        bg_color = _get_background_color(img, rx1, ry1, rx2, ry2)
        text_color = _get_text_color(bg_color)

        # 4단계: 통일 폰트 크기로 전체 영역에 렌더링
        if rh > rw * 1.5:
            _render_vertical(
                draw,
                translated,
                rx1,
                ry1,
                rw,
                rh,
                text_color,
                font_file,
                suggested_size=font_size,
            )
        else:
            font = ImageFont.truetype(font_file, font_size)
            lines = _wrap_text(translated, font, rw)
            total_text_h = int(len(lines) * font_size * LINE_SPACING)
            y_offset = ry1 + (rh - total_text_h) // 2

            for line in lines:
                line_bbox = font.getbbox(line)
                line_w = line_bbox[2] - line_bbox[0]
                x_offset = rx1 + (rw - line_w) // 2
                draw.text((x_offset, y_offset), line, fill=text_color, font=font)
                y_offset += int(font_size * LINE_SPACING)

    output_path = Path.cwd() / f"{img_path.stem}_rendered{img_path.suffix}"
    img.save(output_path)
    return output_path
