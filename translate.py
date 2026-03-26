import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LLM_CHOICES = ("claude", "gemini")

LANGUAGE_TRAITS = {
    "ja": "일본어는 세로쓰기(위→아래, 우→좌)와 가로쓰기가 공존합니다. 만화 말풍선은 주로 세로쓰기입니다.",
    "zh": "중국어는 세로쓰기(우→좌)와 가로쓰기가 공존하며, 고전 중국어는 우→좌 가로쓰기도 있습니다.",
    "ko": "한국어는 가로쓰기가 주류이나, 오래된 텍스트에서 세로쓰기가 나타날 수 있습니다.",
    "en": "영어는 좌→우 가로쓰기입니다.",
}


def _build_prompt(ocr_results: list[dict], source_lang: str, target_lang: str) -> str:
    lang_trait = LANGUAGE_TRAITS.get(source_lang, f"원본 언어 코드: {source_lang}")
    target_name = {"ko": "한국어", "ja": "일본어", "en": "영어", "zh": "중국어"}.get(
        target_lang, target_lang
    )

    # 텍스트와 위치/크기 정보를 함께 표시
    text_lines = []
    for i, r in enumerate(ocr_results):
        xs = [p[0] for p in r["bbox"]]
        ys = [p[1] for p in r["bbox"]]
        cx, cy = int((min(xs) + max(xs)) / 2), int((min(ys) + max(ys)) / 2)
        bw, bh = int(max(xs) - min(xs)), int(max(ys) - min(ys))
        text_lines.append(f'{i}: "{r["text"]}" (위치: x={cx}, y={cy}, 크기: {bw}x{bh})')

    return f"""만화 이미지에서 OCR로 인식된 텍스트 조각들입니다. 각 텍스트의 이미지 내 위치(x,y 좌표)와 bbox 크기(w,h)도 함께 제공됩니다.
이것은 만화의 대화 장면이므로, 전체 텍스트의 맥락(대화 흐름)을 이해한 뒤 {target_name}로 번역해주세요.

## 원본 언어 특성
{lang_trait}

## 번역 규칙
1. **전체 맥락 파악**: 모든 텍스트를 먼저 읽고 대화의 흐름을 파악하세요.
2. **위치 기반 그룹핑**: 위치(x,y)가 가까운 텍스트는 같은 말풍선입니다. 합쳐서 하나의 자연스러운 문장으로 번역하세요. 좌표가 멀리 떨어진 텍스트는 별개의 말풍선이므로 합치지 마세요.
3. **모든 텍스트 포함**: 입력의 모든 인덱스가 출력에 포함되어야 합니다. 빠뜨리지 마세요. 효과음, 페이지 번호도 포함하세요.
4. **번역 분량**: 번역된 텍스트의 길이가 원본과 비슷하게 유지되도록 하세요. 원본이 짧은 감탄사면 번역도 짧게, 원본이 긴 문장이면 번역도 적절히 길게.
5. 인식 오류로 보이는 무의미한 문자(깨진 글자)는 무시하되, 출력 JSON에 indices는 포함하고 translated를 빈 문자열로 하세요.

## 입력 텍스트
{chr(10).join(text_lines)}

## 출력 형식
반드시 아래 JSON 배열만 출력하세요. 설명이나 마크다운 없이 순수 JSON만:
[
  {{"indices": [0, 1], "translated": "번역된 문장"}},
  {{"indices": [2], "translated": "번역된 단어"}}
]

indices는 위 입력 텍스트의 인덱스 번호 배열입니다. **모든 인덱스가 정확히 한 번씩 포함**되어야 합니다."""


def _call_claude(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_gemini(prompt: str) -> str:
    from google import genai

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def _parse_llm_response(response_text: str) -> list[dict]:
    text = response_text.strip()
    # 마크다운 코드 블록 제거
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def run_translate(
    ocr_json_path: str,
    target_lang: str = "ko",
    llm: str = "claude",
) -> Path:
    path = Path(ocr_json_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {ocr_json_path}")

    with open(path, encoding="utf-8") as f:
        ocr_data = json.load(f)

    source_lang = ocr_data["lang"]
    results = ocr_data["results"]

    if not results:
        output = {
            "source": ocr_data["source"],
            "source_lang": source_lang,
            "target_lang": target_lang,
            "llm": llm,
            "translations": [],
        }
        output_path = Path.cwd() / path.name.replace("_ocr.json", "_translated.json")
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output_path

    ocr_texts = [r["text"] for r in results]

    prompt = _build_prompt(results, source_lang, target_lang)

    if llm == "claude":
        response_text = _call_claude(prompt)
    elif llm == "gemini":
        response_text = _call_gemini(prompt)
    else:
        raise ValueError(f"지원하지 않는 LLM: {llm}")

    translation_groups = _parse_llm_response(response_text)

    translations = []
    for group in translation_groups:
        indices = group.get("indices", [])
        translated_text = group.get("translated", "")
        if not indices:
            continue
        translations.append(
            {
                "original_texts": [ocr_texts[i] for i in indices if i < len(ocr_texts)],
                "translated": translated_text,
                "bboxes": [results[i]["bbox"] for i in indices if i < len(results)],
            }
        )

    output = {
        "source": ocr_data["source"],
        "source_lang": source_lang,
        "target_lang": target_lang,
        "llm": llm,
        "translations": translations,
    }

    output_path = Path.cwd() / path.name.replace("_ocr.json", "_translated.json")
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return output_path
