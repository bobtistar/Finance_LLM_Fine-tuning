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
import gc
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config.categories import (  # noqa: E402
    CATEGORIES,
    make_classification_prompt as make_prompt,
    parse_classification_json as parse_json_output,
)

MODEL_CACHE_DIR = PROJECT_DIR / ".hf_cache"
ENV_FILE = PROJECT_DIR / ".env"
RESULT_DIR = PROJECT_DIR / "eval_results"
OFFLOAD_DIR = PROJECT_DIR / ".offload"
FATAL_LOG_FILE = RESULT_DIR / "fatal_error.log"

load_dotenv(ENV_FILE)

# Hugging Face 모델 다운로드 위치를 D드라이브 프로젝트 폴더로 고정합니다.

os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR / "transformers")
# Transformers 5.x crashes on Windows during safetensors mmap materialization.
# HF_DEACTIVATE_ASYNC_LOAD controls HF file I/O but not safetensors' Rust thread pool.
# RAYON_NUM_THREADS=1 forces safetensors to materialize tensors single-threaded,
# which avoids the access violation in torch.storage.__getitem__.
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["RAYON_NUM_THREADS"] = "1"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
fatal_log = FATAL_LOG_FILE.open("w", encoding="utf-8")
faulthandler.enable(file=fatal_log, all_threads=True)

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

# 로딩 방식 선택
#
# "hybrid": bitsandbytes 없이 일부는 GPU, 일부는 CPU/디스크에 올립니다. Windows에서 가장 안전한 기본값입니다.
# "4bit": bitsandbytes 4-bit GPU 로딩입니다. 빠르지만 현재 Windows 환경에서 access violation이 발생했습니다.
# "cpu": GPU를 쓰지 않고 CPU로만 로딩합니다. 가장 느리지만 원인 분리에는 좋습니다.
LOAD_MODE = "4bit"

# CPU 강제 실행용 이전 옵션입니다. True이면 LOAD_MODE보다 우선해서 CPU로 실행합니다.
FORCE_CPU = False

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
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="GPU를 쓰지 않고 CPU로 실행합니다.",
    )
    return parser.parse_args()


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


def load_tokenizer_and_model(load_mode="hybrid"):
    print(f"[캐시 위치] {MODEL_CACHE_DIR}")
    print(f"[오류 로그] {FATAL_LOG_FILE}")
    print(f"[오프로딩 위치] {OFFLOAD_DIR}")

    if load_mode == "cpu":
        print("[로딩 방식] CPU")
    elif load_mode == "4bit":
        print("[로딩 방식] 4-bit GPU")
    else:
        print("[로딩 방식] hybrid GPU/CPU, bitsandbytes 사용 안 함")

    print("[1/4] tokenizer 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR,
    )

    print("[2/4] base model 로드 중...")
    quantization_config = make_4bit_config() if load_mode == "4bit" else None

    if load_mode == "cpu":
        device_map = None
        max_memory = None
        dtype = torch.float32
    elif load_mode == "4bit":
        device_map = "auto"
        max_memory = None
        dtype = torch.float16
    else:
        OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
        device_map = "auto"
        max_memory = {
            0: "3GiB",
            "cpu": "8GiB",
        }
        dtype = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=OFFLOAD_DIR if load_mode == "hybrid" else None,
        offload_state_dict=True if load_mode == "hybrid" else False,
        dtype=dtype,
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
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    del inputs, output_ids, generated_ids
    gc.collect()

    return result


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
    try:
        args = parse_args()

        if not ADAPTER_DIR.exists():
            raise FileNotFoundError(f"LoRA adapter 폴더를 찾을 수 없습니다: {ADAPTER_DIR}")

        if FORCE_CPU or args.cpu:
            load_mode = "cpu"
        elif args.no_4bit:
            load_mode = "hybrid"
        else:
            load_mode = LOAD_MODE

        tokenizer, model = load_tokenizer_and_model(load_mode=load_mode)
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

    except Exception as error:
        print("\n[오류 발생]")
        print(repr(error))
        print(f"자세한 로그 위치: {FATAL_LOG_FILE}")
        raise
    finally:
        fatal_log.flush()
        sys.stderr.flush()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
