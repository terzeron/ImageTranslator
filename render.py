import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = Path(__file__).resolve().parent / "fonts" / "NanumBarunGothic.ttf"
MIN_FONT_SIZE = 8
LINE_SPACING = 1.2
ERASE_MARGIN = 5
SPACE_RATIO = 0.5  # 공백을 글자 폭의 이 비율로 렌더링


def _get_bbox_rect(bbox: list[list[float]]) -> tuple[int, int, int, int]:
    """4점 polygon에서 (x_min, y_min, x_max, y_max) 추출."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _get_background_color(
    img: Image.Image, x1: int, y1: int, x2: int, y2: int
) -> tuple[int, ...]:
    """bbox 외곽 띠의 중앙값(median)으로 배경색 결정."""
    bw, bh = x2 - x1, y2 - y1
    margin = max(7, int(min(bw, bh) * 0.2))
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


def _measure_text(text: str, font: ImageFont.FreeTypeFont) -> int:
    """공백을 SPACE_RATIO로 계산한 텍스트 폭 반환."""
    w = 0
    for char in text:
        cb = font.getbbox(char)
        cw = cb[2] - cb[0]
        w += int(cw * SPACE_RATIO) if char == " " else cw
    return w


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """텍스트를 max_width에 맞게 줄바꿈 (공백 폭 SPACE_RATIO 적용)."""
    lines = []
    current = ""
    for char in text:
        test = current + char
        w = _measure_text(test, font)
        if w > max_width and current:
            lines.append(current)
            current = char
        else:
            current = test
    if current:
        lines.append(current)
    return lines or [""]


def _draw_text_line(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple,
) -> None:
    """한 줄 텍스트를 공백 폭 SPACE_RATIO로 렌더링."""
    cx = x
    for char in text:
        cb = font.getbbox(char)
        cw = cb[2] - cb[0]
        if char == " ":
            cx += int(cw * SPACE_RATIO)
        else:
            draw.text((cx, y), char, fill=color, font=font)
            cx += cw


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
        font_size = max(MIN_FONT_SIZE, suggested_size)
    else:
        font_size = int(math.sqrt(box_w * box_h / (n * LINE_SPACING * LINE_SPACING)))
        font_size = max(MIN_FONT_SIZE, font_size)

    char_h = int(font_size * LINE_SPACING)
    col_w = int(font_size * LINE_SPACING)
    chars_per_col = max(1, box_h // char_h)
    num_cols = math.ceil(n / chars_per_col)

    font = ImageFont.truetype(font_path, font_size)

    # 우→좌 세로쓰기
    total_cols_w = num_cols * col_w
    x_start = x + box_w - col_w - max(0, (box_w - total_cols_w) // 2)

    col = 0
    cy = y
    for char in text:
        cx = x_start - col * col_w
        if char == " ":
            cy += int(char_h * SPACE_RATIO)
        else:
            char_bbox = font.getbbox(char)
            char_w = char_bbox[2] - char_bbox[0]
            cx_centered = cx + (col_w - char_w) // 2
            draw.text((cx_centered, cy), char, fill=text_color, font=font)
            cy += char_h
        if cy >= y + box_h:
            cy = y
            col += 1


def _should_cluster(
    r1: tuple[int, int, int, int],
    r2: tuple[int, int, int, int],
) -> bool:
    """두 bbox를 같은 클러스터로 묶어야 하면 True.

    겹치거나, 간격이 두 bbox의 평균 짧은변보다 작으면 같은 클러스터.
    """
    x1a, y1a, x1b, y1b = r1
    x2a, y2a, x2b, y2b = r2

    # 겹침 체크
    if not (x1b < x2a or x2b < x1a or y1b < y2a or y2b < y1a):
        return True

    # 두 bbox의 짧은변 평균
    short1 = min(x1b - x1a, y1b - y1a)
    short2 = min(x2b - x2a, y2b - y2a)
    avg_short = (short1 + short2) / 2

    gap_x = max(0, max(x1a, x2a) - min(x1b, x2b))
    gap_y = max(0, max(y1a, y2a) - min(y1b, y2b))

    return gap_x <= avg_short and gap_y <= avg_short


def _cluster_bboxes(
    bbox_rects: list[tuple[int, int, int, int]],
) -> list[list[int]]:
    """겹치거나 인접한 bbox들을 클러스터로 묶어 인덱스 그룹 반환."""
    n = len(bbox_rects)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _should_cluster(bbox_rects[i], bbox_rects[j]):
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    return list(clusters.values())


def _erase_rect(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, ...]:
    """영역을 동적 마진 포함하여 배경색으로 지우고, 사용한 배경색 반환."""
    w, h = img.size
    # 마진: bbox 크기의 15% (최소 ERASE_MARGIN)
    bw, bh = x2 - x1, y2 - y1
    margin = max(ERASE_MARGIN, int(min(bw, bh) * 0.15))
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

    # 1단계: 모든 bbox 지우기 + 전체 폰트 크기 추정
    all_estimated_sizes = []
    for item in tr_data["translations"]:
        original_texts = item.get("original_texts", [])
        for idx, bbox in enumerate(item.get("bboxes", [])):
            bx1, by1, bx2, by2 = _get_bbox_rect(bbox)
            _erase_rect(draw, img, bx1, by1, bx2, by2)
            bw, bh = bx2 - bx1, by2 - by1
            if bw <= 0 or bh <= 0:
                continue
            n = len(original_texts[idx]) if idx < len(original_texts) else 1
            n = max(1, n)
            if bh > bw * 1.5:
                all_estimated_sizes.append(min(bw, int(bh / n)))
            else:
                all_estimated_sizes.append(min(bh, int(bw / n)))

    # 이미지 전체 통일 폰트 크기 (중앙값)
    global_font_size = (
        int(np.median(all_estimated_sizes)) if all_estimated_sizes else MIN_FONT_SIZE
    )
    global_font_size = max(MIN_FONT_SIZE, global_font_size)

    # 2단계: 텍스트 렌더링
    for item in tr_data["translations"]:
        translated = item.get("translated", "")
        if not translated.strip():
            continue

        bboxes = item["bboxes"]
        if not bboxes:
            continue

        bbox_rects = [_get_bbox_rect(b) for b in bboxes]

        # bbox를 클러스터로 분리 (멀리 떨어진 bbox는 별개 클러스터)
        clusters = _cluster_bboxes(bbox_rects)

        # 텍스트를 클러스터 면적 비율로 분배
        cluster_rects = []
        cluster_areas = []
        for indices in clusters:
            cx1 = min(bbox_rects[i][0] for i in indices)
            cy1 = min(bbox_rects[i][1] for i in indices)
            cx2 = max(bbox_rects[i][2] for i in indices)
            cy2 = max(bbox_rects[i][3] for i in indices)
            cluster_rects.append((cx1, cy1, cx2, cy2))
            cluster_areas.append((cx2 - cx1) * (cy2 - cy1))

        total_area = sum(cluster_areas) or 1
        text_remaining = translated

        for ci, (cx1, cy1, cx2, cy2) in enumerate(cluster_rects):
            cw, ch = cx2 - cx1, cy2 - cy1
            if cw <= 0 or ch <= 0:
                continue

            # 텍스트 분배
            if ci < len(cluster_rects) - 1:
                ratio = cluster_areas[ci] / total_area
                char_count = max(1, round(len(translated) * ratio))
                chunk = text_remaining[:char_count]
                text_remaining = text_remaining[char_count:]
            else:
                chunk = text_remaining

            if not chunk.strip():
                continue

            bg_color = _get_background_color(img, cx1, cy1, cx2, cy2)
            text_color = _get_text_color(bg_color)

            # 클러스터 내 bbox 다수결로 방향 결정
            c_bboxes = [bbox_rects[i] for i in clusters[ci]]
            iv = sum(1 for r in c_bboxes if (r[3] - r[1]) > (r[2] - r[0]) * 1.5)
            item_vertical = iv > len(c_bboxes) - iv

            if item_vertical:
                _render_vertical(
                    draw,
                    chunk,
                    cx1,
                    cy1,
                    cw,
                    ch,
                    text_color,
                    font_file,
                    suggested_size=global_font_size,
                )
            else:
                font = ImageFont.truetype(font_file, global_font_size)
                lines = _wrap_text(chunk, font, cw)
                total_text_h = int(len(lines) * global_font_size * LINE_SPACING)
                y_offset = cy1 + (ch - total_text_h) // 2

                for line in lines:
                    line_w = _measure_text(line, font)
                    x_offset = cx1 + (cw - line_w) // 2
                    _draw_text_line(draw, x_offset, y_offset, line, font, text_color)
                    y_offset += int(global_font_size * LINE_SPACING)

    output_path = Path.cwd() / f"{img_path.stem}_rendered{img_path.suffix}"
    img.save(output_path)
    return output_path
