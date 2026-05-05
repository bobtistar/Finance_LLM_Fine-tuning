# 02. EXAONE QLoRA 학습 과정

## 1. 목적

이번 학습의 목적은 EXAONE이 금융 보고서를 직접 작성하게 하는 것이 아닙니다.

정확한 목적:

```text
증권사 리포트 문장 또는 문단을 보고
7개 카테고리 중 primary category를 선택하고
secondary category를 최대 2개 선택해서
JSON 형태로 출력하게 만들기
```

## 2. 모델

```text
LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
```

## 3. 학습 방식

```text
QLoRA
```

QLoRA는 base model 전체를 업데이트하지 않고, 4-bit로 양자화된 base model 위에 작은 LoRA adapter만 학습합니다.

결과물:

```text
Base model:
LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct

Adapter:
outputs/exaone-3.5-2.4b-finance-qlora/
```

즉, 학습 후 저장되는 것은 전체 모델이 아니라 **LoRA adapter**입니다.

## 4. 주요 하이퍼파라미터

이번 학습은 baseline 설정으로 진행했습니다.

```text
quantization: 4-bit
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: bfloat16

LoRA r: 16
LoRA alpha: 32
LoRA dropout: 0.05

num_train_epochs: 2
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
effective batch size: 16

learning_rate: 2e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05

max_length: 1024
optimizer: paged_adamw_8bit
```

## 5. 데이터 경로

이번 RunPod 환경에서는 데이터가 `/workspace`에 바로 있었습니다.

```python
TRAIN_FILE = "/workspace/train.jsonl"
VALID_FILE = "/workspace/valid.jsonl"
OUTPUT_DIR = "/workspace/outputs/exaone-3.5-2.4b-finance-qlora"
```

## 6. 핵심 학습 코드 구조

학습 스크립트의 주요 흐름:

```text
1. tokenizer 로드
2. 4-bit quantization config 생성
3. EXAONE base model 로드
4. EXAONE embedding patch 적용
5. prepare_model_for_kbit_training 적용
6. train / validation dataset 로드
7. instruction / input / output을 prompt / completion으로 변환
8. LoRA config 설정
9. SFTTrainer로 학습
10. adapter 저장
```

## 7. EXAONE embedding patch

EXAONE custom model은 PEFT가 기대하는 `get_input_embeddings()` 처리가 자동으로 되지 않아 patch가 필요했습니다.

실제로 찾은 embedding layer:

```text
transformer.wte
```

학습 로그:

```text
[INFO] Patched input embedding: transformer.wte
```

이 patch는 학습 스크립트와 추론 스크립트 양쪽에 모두 필요했습니다.

## 8. 학습 실행

```bash
python scripts/train_exaone_qlora.py
```

학습 완료 시 출력:

```text
Saved adapter to: /workspace/outputs/exaone-3.5-2.4b-finance-qlora
```

이 메시지가 나오면 QLoRA 학습과 adapter 저장은 완료된 것입니다.

## 9. 샘플 추론 결과

입력:

```text
HBM 수요 증가로 2025년 메모리 업황 개선이 예상된다.
```

출력:

```json
{"primary":"산업_트렌드","secondary":["실적_전망"]}
```

입력:

```text
12개월 Forward PER 14배를 적용해 목표주가를 산정했다.
```

출력:

```json
{"primary":"밸류에이션","secondary":[]}
```

입력:

```text
미중 반도체 규제 강화 시 수출 차질이 발생할 수 있다.
```

출력:

```json
{"primary":"리스크_요인","secondary":["산업_트렌드"]}
```

## 10. 백업

```bash
cd /workspace
tar -czf exaone-3.5-2.4b-finance-qlora.tar.gz \
  outputs/exaone-3.5-2.4b-finance-qlora
```

압축 확인:

```bash
tar -tzf exaone-3.5-2.4b-finance-qlora.tar.gz | head
```

확인된 핵심 파일:

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
chat_template.jinja
training_args.bin
```
