import argparse
import sys

from ocr import run_ocr, LANG_MAP, MODEL_MAP
from translate import run_translate, LLM_CHOICES
from render import run_render

TARGET_LANGS = ("ko", "ja", "en", "zh")


def main():
    parser = argparse.ArgumentParser(description="이미지 텍스트 번역 도구")
    sub = parser.add_subparsers(dest="command")

    ocr_parser = sub.add_parser("ocr", help="이미지에서 텍스트 검출 + 문자 인식")
    ocr_parser.add_argument("image_path", help="이미지 파일 경로")
    ocr_parser.add_argument(
        "--lang", choices=LANG_MAP.keys(), default="ja", help="인식 언어 (기본: ja)"
    )
    ocr_parser.add_argument(
        "--model",
        choices=MODEL_MAP.keys(),
        default="v5",
        help="PaddleOCR 모델 버전 (기본: v5)",
    )

    tr_parser = sub.add_parser("translate", help="OCR 결과를 LLM으로 번역")
    tr_parser.add_argument("ocr_json", help="OCR 결과 JSON 파일 경로")
    tr_parser.add_argument(
        "--target-lang",
        choices=TARGET_LANGS,
        default="ko",
        help="번역 대상 언어 (기본: ko)",
    )
    tr_parser.add_argument(
        "--llm", choices=LLM_CHOICES, default="claude", help="사용할 LLM (기본: claude)"
    )

    rd_parser = sub.add_parser("render", help="번역 결과를 이미지에 합성")
    rd_parser.add_argument("translated_json", help="번역 결과 JSON 파일 경로")
    rd_parser.add_argument("image_path", help="원본 이미지 파일 경로")
    rd_parser.add_argument(
        "--font", default=None, help="폰트 파일 경로 (기본: NanumBarunGothic)"
    )

    pl_parser = sub.add_parser(
        "pipeline", help="OCR → 번역 → 이미지 합성을 한번에 실행"
    )
    pl_parser.add_argument("image_path", help="이미지 파일 경로")
    pl_parser.add_argument(
        "--lang", choices=LANG_MAP.keys(), default="ja", help="인식 언어 (기본: ja)"
    )
    pl_parser.add_argument(
        "--model",
        choices=MODEL_MAP.keys(),
        default="v5",
        help="PaddleOCR 모델 버전 (기본: v5)",
    )
    pl_parser.add_argument(
        "--target-lang",
        choices=TARGET_LANGS,
        default="ko",
        help="번역 대상 언어 (기본: ko)",
    )
    pl_parser.add_argument(
        "--llm", choices=LLM_CHOICES, default="claude", help="사용할 LLM (기본: claude)"
    )
    pl_parser.add_argument(
        "--font", default=None, help="폰트 파일 경로 (기본: NanumBarunGothic)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "ocr":
        output_path = run_ocr(args.image_path, lang=args.lang, model=args.model)
        print(f"OCR 결과 저장: {output_path}")
    elif args.command == "translate":
        output_path = run_translate(
            args.ocr_json, target_lang=args.target_lang, llm=args.llm
        )
        print(f"번역 결과 저장: {output_path}")
    elif args.command == "render":
        output_path = run_render(
            args.translated_json, args.image_path, font_path=args.font
        )
        print(f"렌더링 결과 저장: {output_path}")
    elif args.command == "pipeline":
        print(f"[1/3] OCR 처리 중: {args.image_path}")
        ocr_path = run_ocr(args.image_path, lang=args.lang, model=args.model)
        print(f"  → {ocr_path}")

        print(f"[2/3] 번역 중 ({args.llm})")
        tr_path = run_translate(
            str(ocr_path), target_lang=args.target_lang, llm=args.llm
        )
        print(f"  → {tr_path}")

        print("[3/3] 이미지 합성 중")
        output_path = run_render(str(tr_path), args.image_path, font_path=args.font)
        print(f"  → {output_path}")
        print(f"완료: {output_path}")


if __name__ == "__main__":
    main()
