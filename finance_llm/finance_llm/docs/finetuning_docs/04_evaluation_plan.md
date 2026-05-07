# 04. 평가 계획

## 1. 평가 목적

이번 평가는 “학습이 끝났는지”를 확인하는 것이 아니라, 이 모델을 LangGraph에 classifier node로 붙여도 되는지를 판단하는 단계입니다.

EXAONE의 역할:

```text
문장/문단 입력
→ primary category 분류
→ secondary category 최대 2개 출력
→ JSON 형태로 반환
```

따라서 평가는 생성 품질이 아니라 **분류 품질과 출력 안정성**을 봅니다.

## 2. 평가 데이터

```text
valid.jsonl
```

현재 validation set:

```text
304 samples
```

## 3. 핵심 평가 지표

```text
1. JSON valid rate
2. Primary accuracy
3. Primary macro F1
4. Primary weighted F1
5. Category별 precision / recall / F1
6. Secondary micro F1
7. Secondary macro F1
8. Confusion matrix
9. Invalid output sample
10. Primary error sample
```

## 4. 판단 기준

### 바로 LangGraph에 붙여도 되는 수준

```text
JSON valid rate >= 0.98
primary accuracy >= 0.85
macro F1 >= 0.75
산업_분석 / 리스크_요인 recall이 너무 낮지 않음
```

### 붙이되 Claude fallback 필요

```text
JSON valid rate >= 0.95
primary accuracy 0.80~0.85
macro F1 0.65~0.75
```

Fallback 대상:

```text
- JSON 파싱 실패
- primary가 7개 카테고리 밖에 있음
- 문장이 너무 길거나 애매함
- 산업_분석 / 리스크_요인처럼 recall이 낮은 클래스 후보
```

### 재학습 권장

```text
JSON valid rate < 0.95
primary accuracy < 0.80
macro F1 < 0.65
```

## 5. 특히 봐야 할 카테고리

주의 클래스:

```text
산업_분석
리스크_요인
밸류에이션
```

자주 헷갈릴 가능성이 높은 쌍:

```text
산업_트렌드 ↔ 산업_분석
성장_동력 ↔ 기업_분석
실적_전망 ↔ 밸류에이션
리스크_요인 ↔ 산업_트렌드
```

## 6. 평가 결과 저장 파일

평가 스크립트는 다음 파일을 생성하도록 구성합니다.

```text
eval_results/
  summary.json
  classification_report.txt
  confusion_matrix.csv
  invalid_samples.jsonl
  primary_errors.jsonl
```

## 7. 결과 해석

### summary.json

먼저 아래 항목을 봅니다.

```text
json_valid_rate
primary_accuracy
primary_macro_f1
primary_weighted_f1
secondary_micro_f1
secondary_macro_f1
invalid_count
primary_error_count
```

해석:

```text
JSON valid rate 높음
→ LangGraph에서 파싱 가능성이 높음

primary accuracy 높음
→ 카테고리 라우터로 사용 가능

macro F1 낮음
→ 일부 소수 클래스 성능이 낮을 가능성

secondary F1 낮음
→ secondary는 참고용으로만 사용
```

### primary_errors.jsonl

에러를 수동으로 30개 정도 확인해서 다음 세 유형으로 나눕니다.

```text
A. 모델이 명백히 틀림
B. gold label이 애매하거나 틀림
C. 둘 다 가능하지만 기준이 불명확함
```

B/C가 많으면 하이퍼파라미터보다 라벨 기준 정리가 먼저입니다.

## 8. 평가 후 결정

```text
1. 기준 통과
   → LangGraph classifier node로 연결

2. primary accuracy는 괜찮지만 macro F1이 낮음
   → 소수 클래스 데이터 보강 또는 oversampling

3. JSON valid rate가 낮음
   → output 포맷 정제, parser 강화, prompt 수정

4. 특정 쌍을 계속 혼동
   → 카테고리 정의 강화, 헷갈리는 케이스 추가 라벨링

5. 전체 성능 낮음
   → 2차 QLoRA 실험
```
