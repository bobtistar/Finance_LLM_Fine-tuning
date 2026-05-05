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
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = PROJECT_DIR / ".hf_cache"
ENV_FILE = PROJECT_DIR / ".env"

load_dotenv(ENV_FILE)

# Hugging Face 모델 다운로드 위치를 D드라이브 프로젝트 폴더로 고정합니다.

os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR / "transformers")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
ADAPTER_DIR = PROJECT_DIR / "models" / "exaone-3.5-2.4b-finance-qlora"

# 터미널을 쓰지 않고 테스트하고 싶으면 여기만 수정하면 됩니다.
# 여러 문장을 넣으면 모델은 한 번만 로드하고, 문장을 순서대로 테스트합니다.
TEST_TEXTS = [
    "HBM 수요 증가로 2025년 메모리 업황 개선이 예상된다.",
    # "12개월 Forward PER 14배를 적용해 목표주가를 산정했다.",
    # "미중 반도체 규제 강화 시 수출 차질이 발생할 수 있다.",
]

CATEGORIES = [
    "산업_트렌드",
    "성장_동력",
    "실적_전망",
    "산업_분석",
    "기업_분석",
    "리스크_요인",
    "밸류에이션",
]


# ---------------------------------------------------------------------
# 입력 옵션
# ---------------------------------------------------------------------

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
    return parser.parse_args()


# ---------------------------------------------------------------------
# 프롬프트 생성
# ---------------------------------------------------------------------

def make_prompt(text):
    category_text = ", ".join(CATEGORIES)

    return (
        "당신은 증권사 리서치 리포트 문장을 분류하는 금융 분석 보조 모델입니다.\n"
        "주어진 문장을 primary 1개와 secondary 최대 2개로 분류하세요.\n\n"
        f"카테고리: {category_text}\n\n"
        "반드시 JSON만 출력하세요.\n"
        '출력 형식: {"primary":"카테고리명","secondary":["카테고리명"]}\n\n'
        f"문장: {text}"
    )


# ---------------------------------------------------------------------
# EXAONE + PEFT 호환 패치
# ---------------------------------------------------------------------

def patch_exaone_embedding(model):
    """
    EXAONE 모델은 PEFT가 기대하는 embedding 접근 함수가 바로 동작하지 않을 수 있습니다.
    그래서 실제 embedding layer인 transformer.wte를 직접 연결해줍니다.
    """
    if not hasattr(model, "transformer"):
        return
    if not hasattr(model.transformer, "wte"):
        return

    def get_input_embeddings():
        return model.transformer.wte

    def set_input_embeddings(new_embeddings):
        model.transformer.wte = new_embeddings

    model.get_input_embeddings = get_input_embeddings
    model.set_input_embeddings = set_input_embeddings


# ---------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------

def make_4bit_config():
    """
    GTX 1660 Super는 VRAM이 6GB라서 4-bit 로딩을 사용하는 것이 안전합니다.
    bfloat16 대신 float16을 사용합니다.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_tokenizer_and_model(use_4bit=True):
    print(f"[캐시 위치] {MODEL_CACHE_DIR}")

    print("[1/4] tokenizer 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR,
    )

    print("[2/4] base model 로드 중...")
    quantization_config = make_4bit_config() if use_4bit else None

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )

    print("[3/4] EXAONE embedding patch 적용 중...")
    patch_exaone_embedding(base_model)

    print("[4/4] LoRA adapter 로드 중...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    return tokenizer, model


# ---------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------

def classify_sentence(tokenizer, model, text, max_new_tokens):
    prompt = make_prompt(text)
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    device = model.device
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------
# JSON 파싱
# ---------------------------------------------------------------------

def parse_json_output(raw_text):
    """
    모델이 ```json 코드블록을 붙여도 JSON 부분만 찾아서 파싱합니다.
    파싱 실패 또는 카테고리 오류가 있으면 None을 반환합니다.
    """
    text = raw_text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    primary = data.get("primary")
    secondary = data.get("secondary", [])

    if primary not in CATEGORIES:
        return None
    if not isinstance(secondary, list):
        return None

    secondary = [
        category
        for category in secondary
        if category in CATEGORIES and category != primary
    ][:2]

    return {
        "primary": primary,
        "secondary": secondary,
    }


# ---------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------

def print_gpu_info():
    if not torch.cuda.is_available():
        print("[GPU] CUDA를 사용할 수 없어 CPU로 실행합니다.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(0) / 1024**3

    print(f"[GPU] {gpu_name}")
    print(f"[GPU] allocated={allocated_gb:.2f}GB, reserved={reserved_gb:.2f}GB")


def main():
    args = parse_args()

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"LoRA adapter 폴더를 찾을 수 없습니다: {ADAPTER_DIR}")

    use_4bit = torch.cuda.is_available() and not args.no_4bit

    tokenizer, model = load_tokenizer_and_model(use_4bit=use_4bit)
    print_gpu_info()

    if args.text:
        test_texts = [args.text]
    else:
        test_texts = TEST_TEXTS

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


if __name__ == "__main__":
    main()
