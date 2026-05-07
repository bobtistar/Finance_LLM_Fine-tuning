"""
EXAONE 금융 분류 모델 smoke test.

이 스크립트의 목적은 전체 평가 전에 아래 4가지만 빠르게 확인하는 것입니다.

1. base model이 로드되는가?
2. LoRA adapter가 붙는가?
3. 문장 1개에 대해 추론이 되는가?
4. 모델 출력이 JSON으로 파싱되는가?

IDE에서 실행하는 방법:
    1. 아래 TEST_TEXTS의 문장을 원하는 문장으로 수정합니다.
    2. 이 파일을 열고 Run Python File로 실행합니다.

터미널에서 문장을 넘기는 방법도 유지했습니다:
    python finance_llm/evaluation/smoke_test_exaone.py --text "12개월 Forward PER 14배를 적용했다."
"""

import argparse
import faulthandler
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.exaone_classifier import (
    FATAL_LOG_FILE,
    LOAD_MODE,
    RESULT_DIR,
    classify_sentence,
    configure_runtime_environment,
    load_tokenizer_and_model,
    print_gpu_info,
    resolve_adapter_dir,
)
from utils.classification_parser import parse_classification_json

configure_runtime_environment()

FORCE_CPU = False
TEST_TEXTS = [
    "HBM 수요 증가로 2025년 메모리 업황 개선이 예상된다.",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default=None,
        help="터미널에서 직접 테스트 문장을 넘기고 싶을 때 사용합니다.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="4-bit 로딩을 끕니다. GTX 1660 Super에서는 보통 사용하지 않는 옵션입니다.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="GPU를 쓰지 않고 CPU로 실행합니다.",
    )
    return parser.parse_args()


def parse_json_output(raw_text):
    return parse_classification_json(raw_text)


def main():
    fatal_log = None

    try:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fatal_log = FATAL_LOG_FILE.open("w", encoding="utf-8")
        faulthandler.enable(file=fatal_log, all_threads=True)

        args = parse_args()
        adapter_dir = resolve_adapter_dir()

        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"LoRA adapter 폴더를 찾을 수 없습니다: {adapter_dir}"
            )

        if FORCE_CPU or args.cpu:
            load_mode = "cpu"
        elif args.no_4bit:
            load_mode = "hybrid"
        else:
            load_mode = LOAD_MODE

        tokenizer, model = load_tokenizer_and_model(load_mode=load_mode)
        print_gpu_info()

        test_texts = [args.text] if args.text else TEST_TEXTS

        for index, text in enumerate(test_texts, start=1):
            print(f"\n========== 테스트 {index} ==========")
            print("[입력 문장]")
            print(text)

            raw_output = classify_sentence(
                tokenizer=tokenizer,
                model=model,
                text=text,
                max_new_tokens=args.max_new_tokens,
            )
            parsed_output = parse_json_output(raw_output)

            print("\n[모델 원본 출력]")
            print(raw_output)

            print("\n[JSON 파싱 결과]")
            if parsed_output is None:
                print("파싱 실패: JSON 형식이 아니거나 카테고리가 잘못되었습니다.")
            else:
                print(json.dumps(parsed_output, ensure_ascii=False))

    except Exception as error:
        print("\n[오류 발생]")
        print(repr(error))
        print(f"자세한 로그 위치: {FATAL_LOG_FILE}")
        raise
    finally:
        if fatal_log is not None:
            fatal_log.flush()
        sys.stderr.flush()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
