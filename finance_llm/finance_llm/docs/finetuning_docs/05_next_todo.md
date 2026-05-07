# 05. 다음 TODO 리스트

## A. 평가 TODO

```text
[ ] RunPod 또는 로컬 GPU 환경에서 valid.jsonl 전체 평가 실행
[ ] summary.json 확인
[ ] classification_report.txt 확인
[ ] confusion_matrix.csv 확인
[ ] primary_errors.jsonl에서 30개 이상 수동 검토
[ ] invalid_samples.jsonl 확인
```

필수 기록 지표:

```text
JSON valid rate
primary accuracy
primary macro F1
primary weighted F1
secondary micro F1
secondary macro F1
카테고리별 recall
```

## B. 코드 정리 TODO

```text
[ ] EXAONE 추론 코드를 class로 분리
[ ] JSON parser 함수 추가
[ ] 코드블록 제거 post-processing 추가
[ ] category validation 추가
[ ] parsing 실패 시 fallback 로직 추가
```

추천 파일 구조:

```text
src/
  models/
    exaone_classifier.py
  utils/
    json_parser.py
  evaluation/
    eval_exaone_classifier.py
```

## C. LangGraph 통합 TODO

```text
[ ] EXAONE classifier node 작성
[ ] 문장/문단 단위 입력 처리
[ ] primary 기준으로 category bucket 생성
[ ] secondary는 보조 bucket 또는 metadata로 저장
[ ] Claude API section writer node 작성
[ ] Claude API report composer node 작성
[ ] Claude API reviewer node 작성
[ ] end-to-end 샘플 리포트 1개 생성
```

## D. 성능 개선 TODO

### JSON valid rate가 낮을 때

```text
[ ] output parser 강화
[ ] system prompt에 마크다운 금지 추가
[ ] 학습 데이터 output을 순수 JSON으로 재정리
[ ] max_new_tokens 제한
```

### primary accuracy가 낮을 때

```text
[ ] primary_errors.jsonl 분석
[ ] 라벨 기준 애매한 샘플 정리
[ ] 추가 라벨링
[ ] 2차 QLoRA 실험
```

### macro F1이 낮을 때

```text
[ ] 산업_분석 데이터 보강
[ ] 리스크_요인 데이터 보강
[ ] 밸류에이션 데이터 보강
[ ] 소수 클래스 oversampling 고려
```

## F. 2차 학습 실험 후보

현재 baseline:

```text
lr = 2e-4
epoch = 2
r = 16
alpha = 32
dropout = 0.05
max_length = 1024
```

### Experiment v2: 안정성 중심

```text
lr = 1e-4
epoch = 2
r = 16
alpha = 32
dropout = 0.05
```

### Experiment v3: 표현력 증가

```text
lr = 1e-4 또는 2e-4
epoch = 2
r = 32
alpha = 64
dropout = 0.05
```

### Experiment v4: 학습량 증가

```text
lr = 1e-4
epoch = 3
r = 16
alpha = 32
dropout = 0.05
```

주의:

```text
하이퍼파라미터를 감으로 바꾸지 말고,
valid 평가 결과를 보고 한 번에 1~2개만 바꾼다.
```
