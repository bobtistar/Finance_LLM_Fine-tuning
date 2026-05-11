import gc
import importlib.util
import os
import platform
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompts.classifier_prompts import build_exaone_classifier_prompt

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = PROJECT_DIR / ".hf_cache"
ENV_FILE = PROJECT_DIR / ".env"
RESULT_DIR = PROJECT_DIR / "eval_results"
OFFLOAD_DIR = PROJECT_DIR / ".offload"
FATAL_LOG_FILE = RESULT_DIR / "fatal_error.log"

BASE_MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
BASE_MODEL_REVISION = "8e6fc27"
LOAD_MODE = "auto"

ADAPTER_DIR_CANDIDATES = [
    PROJECT_DIR / "outputs" / "exaone-3.5-2.4b-finance-qlora",
    PROJECT_DIR / "models" / "exaone-3.5-2.4b-finance-qlora",
]


def configure_runtime_environment() -> None:
    load_dotenv(ENV_FILE)

    os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR / "transformers")
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_adapter_dir() -> Path:
    for candidate in ADAPTER_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    return ADAPTER_DIR_CANDIDATES[0]


def bitsandbytes_available() -> bool:
    return (
        platform.system() == "Linux"
        and importlib.util.find_spec("bitsandbytes") is not None
    )


def resolve_load_mode(load_mode: str) -> str:
    if load_mode == "cpu":
        return "cpu"

    if load_mode == "hybrid":
        if torch.cuda.is_available():
            return "hybrid"
        print("[로딩 방식] CUDA가 없어 hybrid 대신 CPU로 실행합니다.")
        return "cpu"

    if load_mode not in {"auto", "4bit"}:
        raise ValueError(
            "load_mode는 'auto', '4bit', 'hybrid', 'cpu' 중 하나여야 합니다."
        )

    has_cuda = torch.cuda.is_available()
    has_bitsandbytes = bitsandbytes_available()

    if has_cuda and has_bitsandbytes:
        return "4bit"

    if load_mode == "4bit":
        if not has_cuda:
            print("[로딩 방식] CUDA가 없어 4-bit 대신 CPU로 실행합니다.")
            return "cpu"
        print("[로딩 방식] bitsandbytes를 사용할 수 없어 4-bit 대신 hybrid로 실행합니다.")
        return "hybrid"

    if has_cuda:
        print("[로딩 방식] bitsandbytes를 사용할 수 없어 auto 모드를 hybrid로 실행합니다.")
        return "hybrid"

    print("[로딩 방식] CUDA가 없어 auto 모드를 CPU로 실행합니다.")
    return "cpu"


def patch_exaone_embedding(model) -> None:
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


def make_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_tokenizer_and_model(load_mode: str = LOAD_MODE, adapter_dir: Path | None = None):
    configure_runtime_environment()

    load_mode = resolve_load_mode(load_mode)
    adapter_path = adapter_dir or resolve_adapter_dir()

    print(f"[캐시 위치] {MODEL_CACHE_DIR}")
    print(f"[오류 로그] {FATAL_LOG_FILE}")
    print(f"[오프로딩 위치] {OFFLOAD_DIR}")
    print(f"[어댑터 위치] {adapter_path}")

    if load_mode == "cpu":
        print("[로딩 방식] CPU")
    elif load_mode == "4bit":
        print("[로딩 방식] 4-bit GPU")
    else:
        print("[로딩 방식] hybrid GPU/CPU, bitsandbytes 사용 안 함")

    print("[1/4] tokenizer 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
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
        revision=BASE_MODEL_REVISION,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=OFFLOAD_DIR if load_mode == "hybrid" else None,
        offload_state_dict=True if load_mode == "hybrid" else False,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )

    print("[3/4] EXAONE embedding patch 적용 중...")
    patch_exaone_embedding(base_model)

    print("[4/4] LoRA adapter 로드 중...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return tokenizer, model


def classify_sentence(tokenizer, model, text: str, max_new_tokens: int) -> str:
    prompt = build_exaone_classifier_prompt(text)
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


def print_gpu_info() -> None:
    if not torch.cuda.is_available():
        print("[GPU] CUDA를 사용할 수 없어 CPU로 실행합니다.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(0) / 1024**3

    print(f"[GPU] {gpu_name}")
    print(f"[GPU] allocated={allocated_gb:.2f}GB, reserved={reserved_gb:.2f}GB")
