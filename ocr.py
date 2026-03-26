from pathlib import Path
import json
from paddleocr import PaddleOCR


LANG_MAP = {
    "ja": "japan",
    "ko": "korean",
    "en": "en",
    "zh": "ch",
}

MODEL_MAP = {
    "v3": "PP-OCRv3",
    "v4": "PP-OCRv4",
    "v5": "PP-OCRv5",
}

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
}


def run_ocr(image_path: str, lang: str = "ja", model: str = "v5") -> Path:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {image_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 이미지 포맷입니다: {path.suffix}")

    ocr = PaddleOCR(
        ocr_version=MODEL_MAP[model],
        lang=LANG_MAP[lang],
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    prediction = ocr.predict(str(path))

    results = []
    for result in prediction:
        data = result.json.get("res", result.json)
        if "dt_polys" not in data:
            continue
        for bbox, text, score in zip(
            data["dt_polys"], data["rec_texts"], data["rec_scores"]
        ):
            results.append(
                {
                    "bbox": [[float(x), float(y)] for x, y in bbox],
                    "text": text,
                    "confidence": float(score),
                }
            )

    output = {
        "source": path.name,
        "lang": lang,
        "model": model,
        "results": results,
    }

    output_path = Path.cwd() / f"{path.stem}_ocr.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return output_path
