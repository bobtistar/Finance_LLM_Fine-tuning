# 03. 파인튜닝 중 발생한 문제와 해결

## 문제 1. CUDA: False

### 증상

```text
CUDA: False
```

### 원인 가능성

```text
1. GPU Pod가 아니라 CPU 환경에 접속
2. GPU는 있으나 PyTorch가 CPU 버전으로 설치됨
3. CUDA 버전 PyTorch 설치 문제
```

### 확인

```bash
nvidia-smi
```

### 해결

`nvidia-smi`가 정상인데 PyTorch에서 CUDA를 못 보면 CUDA 버전 PyTorch를 다시 설치합니다.

```bash
pip uninstall -y torch torchvision torchaudio

pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio
```

---

## 문제 2. EXAONE + PEFT embedding 오류

### 증상

```text
NotImplementedError:
get_input_embeddings not auto-handled for ExaoneModel
```

### 원인

PEFT가 LoRA adapter를 주입할 때 `model.get_input_embeddings()`를 호출하는데, EXAONE custom model에서 해당 메서드가 자동 처리되지 않았습니다.

### 해결

EXAONE 내부의 token embedding layer를 직접 찾아서 `get_input_embeddings()`와 `set_input_embeddings()`를 patch했습니다.

실제 확인된 embedding layer:

```text
transformer.wte
```

학습과 추론 모두에서 patch가 필요했습니다.

---

## 문제 3. TRL prompt/completion mismatch warning

### 증상

```text
[RANK 0] Mismatch between tokenized prompt and the start of tokenized prompt+completion.
```

### 원인

`SFTTrainer`가 prompt와 prompt+completion을 따로 토크나이즈할 때, EXAONE tokenizer의 chat template / whitespace / special token 처리로 인해 prefix가 완전히 일치하지 않았습니다.

### 판단

```text
Traceback이 아니고 학습 progress bar가 진행되었으므로 fatal error는 아니었음.
```

### 교훈

처음 baseline에서는 무시하고 진행 가능했습니다. 더 안정적인 2차 실험에서는 prompt/completion 대신 완성된 chat text 하나로 만들어 학습하는 방식을 고려할 수 있습니다.

---

## 문제 4. 추론 시 adapter 로드 오류

### 증상

학습은 완료되었지만 테스트 스크립트에서 다시 같은 오류가 발생했습니다.

```text
NotImplementedError:
get_input_embeddings not auto-handled for ExaoneModel
```

### 원인

`PeftModel.from_pretrained(base_model, ADAPTER_DIR)` 호출 시에도 PEFT가 embedding 관련 검사를 수행합니다.

### 해결

테스트 스크립트에도 학습 스크립트와 동일한 embedding patch를 추가했습니다.

---

## 문제 5. generate 입력 shape 오류

### 증상

```text
KeyError: 'shape'
AttributeError
```

### 원인

`tokenizer.apply_chat_template(..., return_dict=True)`의 반환값은 Tensor가 아니라 dict/BatchEncoding 형태입니다.

그런데 이를 `model.generate(input_ids, ...)`처럼 Tensor로 넘겨서 `shape` 접근 오류가 발생했습니다.

### 해결

입력을 dict로 받고 `generate(**inputs, ...)` 형태로 넘겼습니다.

```python
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

inputs = {k: v.to(device) for k, v in inputs.items()}

output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
)
```

---

## 문제 6. 출력에 json 코드블록이 붙음

### 증상

모델이 다음처럼 출력했습니다.

```text
```json
{"primary":"산업_트렌드","secondary":["실적_전망"]}
```
```

### 문제

사람이 보기에는 괜찮지만 Python에서 바로 `json.loads()`를 하면 실패할 수 있습니다.

### 해결

post-processing parser를 추가합니다.

```python
import json
import re

def parse_model_json(text: str):
    text = text.strip()
    text = re.sub(r"^```json\\s*", "", text)
    text = re.sub(r"^```\\s*", "", text)
    text = re.sub(r"\\s*```$", "", text)

    match = re.search(r"\\{.*\\}", text, flags=re.DOTALL)
    if not match:
        return None

    return json.loads(match.group(0))
```

---

## 문제 7. tar.gz 파일이 열리지 않는 것처럼 보임

### 증상

```text
Error opening archive: Failed to open 'exaone-3.5-2.4b-finance-qlora.tar.gz'
```

### 확인

로컬 터미널에서 확인했습니다.

```bash
tar -tzf exaone-3.5-2.4b-finance-qlora.tar.gz | head
```

정상적으로 파일 목록이 나왔기 때문에 압축파일은 정상으로 판단했습니다.

핵심 파일:

```text
adapter_config.json
adapter_model.safetensors
```
