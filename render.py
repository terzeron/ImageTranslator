import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = Path(__file__).resolve().parent / "fonts" / "NanumBarunGothic.ttf"
MIN_FONT_SIZE = 8
LINE_SPACING = 1.2
ERASE_MARGIN = 5  # bbox 지우기 시 추가 마진 (px)


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


def _should_cluster(
    r1: tuple[int, int, int, int],
    r2: tuple[int, int, int, int],
) -> bool:
    """두 rect를 같은 클러스터로 묶어야 하면 True.

    겹치거나, 간격이 두 bbox의 평균 폭/높이보다 작으면 같은 클러스터.
    """
    x1a, y1a, x1b, y1b = r1
    x2a, y2a, x2b, y2b = r2

    # 겹침 체크
    if not (x1b < x2a or x2b < x1a or y1b < y2a or y2b < y1a):
        return True

    # 두 bbox의 평균 크기 기반 동적 gap
    w1, h1 = x1b - x1a, y1b - y1a
    w2, h2 = x2b - x2a, y2b - y2a
    avg_w = (w1 + w2) / 2
    avg_h = (h1 + h2) / 2

    gap_x = max(0, max(x1a, x2a) - min(x1b, x2b))
    gap_y = max(0, max(y1a, y2a) - min(y1b, y2b))

    # x축 간격이 평균 폭 이하이고 y축이 겹치거나 근접
    if gap_x <= avg_w and gap_y <= avg_h:
        return True

    return False


def _cluster_bboxes(
    bbox_rects: list[tuple[int, int, int, int]],
) -> list[list[int]]:
    """겹치거나 인접한 bbox들을 클러스터로 묶어 인덱스 그룹 반환.

    Union-Find로 구현.
    """
    n = len(bbox_rects)
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

        # 1단계: 개별 bbox를 rect로 변환
        bbox_rects = [_get_bbox_rect(b) for b in bboxes]

        # 2단계: 클러스터링
        clusters = _cluster_bboxes(bbox_rects)

        # 3단계: 클러스터별로 처리
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

        # 4단계: 클러스터 전체 영역을 배경색으로 지우기 (개별 bbox 아닌 클러스터 단위)
        for cx1, cy1, cx2, cy2 in cluster_rects:
            _erase_rect(draw, img, cx1, cy1, cx2, cy2)

        # 5단계: 클러스터별 텍스트 렌더링
        text_remaining = translated
        for ci, (cx1, cy1, cx2, cy2) in enumerate(cluster_rects):
            cw = cx2 - cx1
            ch = cy2 - cy1
            if cw <= 0 or ch <= 0:
                continue

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

            font_size, lines = _calc_font_size(chunk, cw, ch, font_file)
            font = ImageFont.truetype(font_file, font_size)

            total_text_h = int(len(lines) * font_size * LINE_SPACING)
            y_offset = cy1 + (ch - total_text_h) // 2

            for line in lines:
                line_bbox = font.getbbox(line)
                line_w = line_bbox[2] - line_bbox[0]
                x_offset = cx1 + (cw - line_w) // 2
                draw.text((x_offset, y_offset), line, fill=text_color, font=font)
                y_offset += int(font_size * LINE_SPACING)

    output_path = Path.cwd() / f"{img_path.stem}_rendered{img_path.suffix}"
    img.save(output_path)
    return output_path
