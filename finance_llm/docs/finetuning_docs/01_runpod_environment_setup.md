# 01. RunPod 환경 세팅

## 1. Deployment type 선택

RunPod에서는 `Serverless`가 아니라 **Pod**를 선택합니다.

```text
Choose your deployment type
→ Pod
```

이유:

```text
파인튜닝은 GPU 서버에 직접 접속해서 학습을 돌려야 하므로,
API 배포용 Serverless가 아니라 GPU Pod가 필요함.
```

## 2. 추천 GPU

EXAONE-3.5-2.4B QLoRA 기준:

```text
GPU: RTX 4090 24GB
```

대안:

```text
RTX A5000
RTX A6000
L40S
```

처음에는 RTX 4090 24GB면 충분합니다. EXAONE-3.5-7.8B로 확장할 때만 A6000 48GB 이상을 고려합니다.

## 3. Storage configuration

추천값:

```text
Container disk: 40GB
Volume disk: 100GB
Network volume: 사용 안 함
```

각 항목 의미:

```text
Container disk
→ 패키지, 임시 파일용
→ Pod stop 시 사라질 수 있음
→ 20GB는 부족할 수 있으므로 40GB 권장

Volume disk
→ /workspace에 마운트되는 저장공간
→ 학습 데이터, checkpoint, adapter 저장
→ 100GB 권장

Network volume
→ 여러 Pod에서 같은 볼륨을 공유할 때 사용
→ 현재 단일 Pod 학습에는 불필요
```

## 4. 체크박스 설정

```text
Encrypt volume          ❌ 체크 X
SSH terminal access     선택
Start Jupyter notebook  ✅ 체크 O
```

### Encrypt volume

체크하지 않았습니다.

이유:

```text
- 성능 저하 가능
- 나중에 resize 불가
- 현재는 민감한 운영 데이터가 아님
```

### SSH terminal access

SSH public key가 없으면 비워둬도 됩니다. Jupyter terminal만으로도 학습 가능했습니다.

### Start Jupyter notebook

체크했습니다. 처음 파인튜닝을 진행하는 경우 Jupyter 파일 업로드와 터미널 사용이 편합니다.

## 5. 실제 파일 위치 확인

처음에는 다음 경로를 예상했습니다.

```text
/workspace/feynman1227/finance_llm
```

하지만 Jupyter로 업로드한 파일들은 실제로 `/workspace`에 바로 위치했습니다.

따라서 학습 전에는 반드시 확인합니다.

```bash
pwd
ls -lh
```

이번 학습에서는 다음 파일들이 `/workspace`에 있었습니다.

```text
train.jsonl
valid.jsonl
all.jsonl
stats.json
```

## 6. CUDA 확인

학습 전 반드시 확인합니다.

```bash
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

기대 출력:

```text
CUDA: True
GPU: NVIDIA ...
```

만약 `CUDA: False`가 나오면:

```bash
nvidia-smi
```

를 먼저 확인합니다.

`nvidia-smi`는 정상인데 PyTorch에서 CUDA를 못 보면 CUDA 버전 PyTorch를 다시 설치합니다.

```bash
pip uninstall -y torch torchvision torchaudio

pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio
```

## 7. RunPod 과금 주의

RunPod는 Pod가 켜져 있는 동안 계속 과금됩니다.

```text
학습 중: Pod Running 유지
학습 완료 + 다운로드 완료: Stop Pod
데이터 삭제 가능성이 있으므로 바로 Terminate하지 않기
```

권장:

```text
Stop      = 우선 선택
Terminate = 로컬 백업과 평가 확인 후 선택
```
