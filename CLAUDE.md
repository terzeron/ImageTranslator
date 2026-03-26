# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

이미지 내 텍스트를 OCR로 인식하고 번역하는 도구. Google Translate의 이미지 번역 기능과 유사한 파이프라인 구현이 목표.

## Pipeline (3단계)

1. **텍스트 검출 + 문자 인식 (OCR)** — PaddleOCR로 이미지 내 텍스트 위치(bounding box)와 문자를 한번에 인식
2. **번역 (LLM)** — LLM AI를 이용하여 인식된 문자들을 문맥에 맞게 합치거나 쪼개서 번역
3. **이미지 합성 (Rendering)** — 원본 텍스트 영역을 바탕색으로 덮어 지운 뒤, 번역된 텍스트를 폰트 렌더링하여 이미지 위에 그려넣기

`pipeline` 서브커맨드로 3단계를 한번에 실행 가능. 모든 중간/최종 산출물은 현재 작업 디렉토리에 생성.

## Development Setup

- Python 3.13.7, uv 패키지 매니저 사용
- 가상환경 생성 및 의존성 설치: `uv sync`
- 실행: `uv run python main.py`

## Usage

```bash
# 파이프라인 한번에 실행 (OCR → 번역 → 이미지 합성)
uv run python main.py pipeline --lang ja --model v5 --llm claude --target-lang ko image.jpg

# 중간 산출물 캐시 무시하고 재생성
uv run python main.py pipeline --reset --lang ja --model v5 --llm claude --target-lang ko image.jpg

# 개별 단계 실행
uv run python main.py ocr --lang ja --model v5 image.jpg
uv run python main.py translate image_ocr.json --target-lang ko --llm claude
uv run python main.py render image_translated.json image.jpg
```

## Architecture

- `main.py` — CLI 진입점 (서브커맨드: `ocr`, `translate`, `render`, `pipeline`)
- `ocr.py` — PaddleOCR 래퍼 (텍스트 검출 + 문자 인식)
- `translate.py` — LLM 번역 모듈 (Claude, Gemini)
- `render.py` — 이미지 합성 (배경색 추출, 텍스트 렌더링)
- `fonts/NanumBarunGothic.ttf` — 번들 폰트
- OCR 엔진: PaddleOCR (paddleocr >= 3.4.0, paddlepaddle == 3.0.0)
- LLM: anthropic (Claude), google-genai (Gemini)
- API 키: `.env` 파일에서 로드 (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`)

## Testing

- `tests/test_runner.py -a` — 전체 테스트 실행 + 커버리지
- `uv run python -m pytest tests/` — pytest 직접 실행
- 테스트 데이터: `tests/resources/` (합성 이미지, 만화 페이지)
