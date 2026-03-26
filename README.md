# ImageTranslator

이미지 내 텍스트를 자동으로 인식하고 번역하여 원본 이미지 위에 합성하는 도구.
Google Translate의 이미지 번역 기능과 유사한 파이프라인을 구현합니다.

## Demo

일본어 만화 → 한국어 번역 결과:

| 원본                            | 번역 결과                         |
| ------------------------------- | --------------------------------- |
| ![원본](docs/demo_original.jpg) | ![번역](docs/demo_translated.jpg) |

## Pipeline

```
원본 이미지 → [OCR] → [LLM 번역] → [이미지 합성] → 번역된 이미지
```

1. **OCR** — PaddleOCR로 텍스트 위치(bounding box)와 문자를 인식
2. **번역** — LLM(Claude/Gemini)이 문맥을 고려하여 텍스트를 조합하고 번역
3. **이미지 합성** — 원본 텍스트를 배경색으로 지우고 번역문을 폰트 렌더링

## 설치

```bash
# Python 3.13.7 + uv 필요
uv sync
```

`.env` 파일에 API 키를 설정합니다:

```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

## 사용법

### 한번에 실행

```bash
uv run python main.py pipeline --lang ja --model v5 --llm claude --target-lang ko manga.jpg
```

출력 (중간 산출물이 없을 때):

```
[1/3] OCR 처리 중: manga.jpg
  → manga_ocr.json
[2/3] 번역 중 (claude)
  → manga_translated.json
[3/3] 이미지 합성 중
  → manga_rendered.jpg
완료: manga_rendered.jpg
```

출력 (중간 산출물 캐시가 있을 때):

```
[1/3] OCR 캐시 사용: manga_ocr.json
[2/3] 번역 캐시 사용: manga_translated.json
[3/3] 이미지 합성 중
  → manga_rendered.jpg
완료: manga_rendered.jpg
```

캐시를 무시하고 재생성하려면 `--reset` 옵션:

```bash
uv run python main.py pipeline --reset --lang ja --model v5 --llm claude --target-lang ko manga.jpg
```

### 개별 단계 실행

```bash
# 1. OCR
uv run python main.py ocr --lang ja --model v5 manga.jpg

# 2. 번역
uv run python main.py translate manga_ocr.json --target-lang ko --llm claude

# 3. 이미지 합성
uv run python main.py render manga_translated.json manga.jpg
```

### 옵션

| 옵션            | 값             | 기본값           | 설명                  |
| --------------- | -------------- | ---------------- | --------------------- |
| `--lang`        | ja, ko, en, zh | ja               | OCR 인식 언어         |
| `--model`       | v3, v4, v5     | v5               | PaddleOCR 모델 버전   |
| `--llm`         | claude, gemini | claude           | 번역에 사용할 LLM     |
| `--target-lang` | ko, ja, en, zh | ko               | 번역 대상 언어        |
| `--font`        | 파일 경로      | NanumBarunGothic | 렌더링 폰트           |
| `--reset`       | (플래그)       |                  | 중간 산출물 캐시 무시 |

## 지원 언어

| 언어        | OCR        | 번역 대상 |
| ----------- | ---------- | --------- |
| 일본어 (ja) | v3, v5     | O         |
| 한국어 (ko) | v3, v5     | O         |
| 영어 (en)   | v3, v4, v5 | O         |
| 중국어 (zh) | v3, v4, v5 | O         |

## 테스트

```bash
# 전체 테스트 + 커버리지
tests/test_runner.py -a

# pytest 직접 실행
uv run python -m pytest tests/
```

## 기술 스택

- **OCR**: PaddleOCR (paddleocr >= 3.4.0, paddlepaddle == 3.0.0)
- **LLM**: Anthropic Claude API, Google Gemini API
- **이미지 처리**: Pillow
- **폰트**: 나눔바른고딕 (번들 포함)
