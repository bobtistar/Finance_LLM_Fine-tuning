# 07. Evaluation Result and Prompt Update

## 1. 목적

이 문서는 EXAONE 분류기의 전체 validation 평가 결과와, 오분류 패턴을 바탕으로 수행한 프롬프트 보정 작업을 기록한다.

이번 작업의 목표는 다음 두 가지였다.

```text
1. valid.jsonl 304개 전체 평가로 신뢰 가능한 지표 확보
2. primary_errors.jsonl 분석을 바탕으로 분류 경계가 흔들리는 카테고리 정의 보강
```

## 2. 평가 환경

전체 평가는 클라우드 GPU 환경에서 수행했다.

```text
Base model: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
Adapter: outputs/exaone-3.5-2.4b-finance-qlora/checkpoint-342
Validation set: 304 samples
Inference mode: 4-bit GPU
```

실행 과정에서 다음 이슈를 정리했다.

```text
- EXAONE remote code 최신본과 local transformers 버전 충돌
- adapter config와 peft 버전 충돌
- adapter 경로 불일치
- smoke test / eval prompt가 너무 간단해 경계 규칙이 충분히 명시되지 않음
```

## 3. 전체 평가 결과

### summary.json

```json
{
  "total_count": 304,
  "json_valid_count": 304,
  "invalid_count": 0,
  "json_valid_rate": 1.0,
  "primary_accuracy": 0.7894736842105263,
  "primary_macro_f1": 0.7853915348089203,
  "primary_weighted_f1": 0.7891975620945952,
  "secondary_micro_f1": 0.48148148148148145,
  "secondary_macro_f1": 0.4430848430848431
}
```

### classification_report.txt 요약

```text
산업_트렌드  precision 0.82 / recall 0.80 / f1 0.81
성장_동력    precision 0.77 / recall 0.76 / f1 0.77
실적_전망    precision 0.69 / recall 0.95 / f1 0.80
산업_분석    precision 0.54 / recall 0.58 / f1 0.56
기업_분석    precision 0.87 / recall 0.68 / f1 0.76
리스크_요인  precision 0.85 / recall 0.85 / f1 0.85
밸류에이션   precision 1.00 / recall 0.91 / f1 0.95
```

## 4. 결과 해석

좋았던 점:

```text
- JSON valid rate 1.0: 파싱 실패가 없어 운영 안정성이 높음
- 리스크_요인 / 밸류에이션 성능이 좋음
- primary macro F1 0.785로 클래스 불균형 환경에서도 전체 분류력은 양호
```

아쉬운 점:

```text
- primary accuracy 0.789로 내부 목표 0.80에 근접하지만 약간 부족
- secondary 성능은 보조 신호로는 쓸 수 있으나 강한 판단 근거로 쓰기엔 약함
- 산업_분석, 기업_분석 경계가 특히 흔들림
```

운영 관점 판단:

```text
- primary 단일 분류기로는 사용 가능
- secondary는 참고용 metadata 정도로 사용하는 것이 적절
- 완전 자동 분류기로 보기에는 기업_분석 / 산업_분석 경계 보강이 더 필요
```

## 5. primary_errors.jsonl 분석

오분류 수:

```text
64 errors
```

주요 오분류 쌍:

```text
기업_분석 -> 실적_전망: 10
기업_분석 -> 성장_동력: 7
성장_동력 -> 기업_분석: 6
산업_트렌드 -> 성장_동력: 5
산업_트렌드 -> 실적_전망: 4
성장_동력 -> 산업_트렌드: 4
기업_분석 -> 산업_트렌드: 4
성장_동력 -> 실적_전망: 4
산업_트렌드 -> 산업_분석: 3
```

gold label 기준 오분류가 많이 발생한 카테고리:

```text
기업_분석: 25
성장_동력: 15
산업_트렌드: 12
산업_분석: 5
```

pred label 기준 과대예측 경향:

```text
실적_전망: 23
성장_동력: 14
산업_트렌드: 11
```

### 핵심 문제 1. 기업_분석이 실적_전망으로 자주 이동

대표 예시:

```text
"SDC 1.2조원, MX/NW 3.3조원, VD/가전 0.3조원으로 추정"
gold: 기업_분석
pred: 실적_전망
```

해석:

```text
숫자와 추정 표현이 들어가면 모델이 회사 내부 현황보다 실적 추정 문장으로 읽는 경향이 있음
```

### 핵심 문제 2. 기업_분석과 성장_동력 경계 혼동

대표 예시:

```text
"부산 공장 전장 라인이 3분기부터 본격적으로 가동되면서 전장용 매출 비중이 의미있게 상승하고 있다."
gold: 기업_분석
pred: 성장_동력
```

해석:

```text
생산라인, 투자, 가동, 고객사, 사업부 관련 문장이
현재 현황 설명인지 미래 성장 논리인지 경계가 겹치는 경우가 많음
```

### 핵심 문제 3. 산업_분석과 산업_트렌드 경계 약함

대표 예시:

```text
"IT 범용품의 가격은 하락세가 지속되는 반면, IT 고성능품과 전장/산업용은 여전히 빠듯한 수급 여건을"
gold: 산업_트렌드
pred: 산업_분석
```

해석:

```text
산업 구조/비교/공급망 설명과 업황 방향성 설명이 한 문장 안에 같이 등장하면 경계가 약해짐
```

## 6. 이번 프롬프트 수정 내용

오분류 패턴을 반영해 다음 기준을 프롬프트에 명시했다.

### 보강한 카테고리 정의

```text
- 성장_동력: 기업의 미래 성장 근거, 신사업, 기술 우위, 신규 고객 확보, 증설/투자/신제품이 성장 논리로 제시된 경우
- 실적_전망: 매출/영업이익/EPS/마진 등 수치 기반 실적 추정, 컨센서스, 가이던스, 상향/하향 전망
- 산업_분석: 경쟁사 비교, 점유율 비교, 산업 구조, 공급망, 밸류체인, 업계 내 포지셔닝 비교
- 기업_분석: 기업 내부 현황, 사업부 구조, 생산라인, 제품 믹스, 고객사, 투자 현황, 현재 진행 중인 전략/운영 상태
```

### 추가한 판별 규칙

```text
1. 숫자, 매출, 영업이익, 증가율 표현이 있어도 미래 실적 추정/가이던스/컨센서스가 아니면 실적_전망으로 보내지 않는다.
2. 회사의 현재 사업부 구성, 생산능력, 고객사, 제품군, 투자 집행, 운영 현황 설명은 기업_분석을 우선한다.
3. 증설, 신제품, 기술력, 파트너십, 신규 시장 진입이 미래 성장 근거로 쓰이면 성장_동력을 우선한다.
4. 경쟁사 비교, 점유율 비교, 공급망/밸류체인/산업 구조 설명은 산업_분석을 우선한다.
5. 산업 전체 수요/공급, 업황, 시장 성장 방향은 산업_트렌드를 우선한다.
```

### 혼동 방지 기준

```text
- 기업 내부 현황 + 숫자: 기본은 기업_분석
- 산업 전망 + 숫자: 기본은 산업_트렌드
- 미래 실적 수치 추정/컨센서스/가이던스: 실적_전망
- 기술/증설/파트너십이 성장 논리의 핵심: 성장_동력
- 비교/점유율/공급망/밸류체인: 산업_분석
```

### few-shot 예시 보강

추가한 예시는 특히 다음 오분류 유형을 겨냥한다.

```text
- 숫자가 있지만 기업 내부 현황인 문장
- 생산라인 가동/사업부 설명이지만 성장_동력으로 과대 분류되던 문장
- 점유율/공급 비중이 들어간 산업_분석 문장
```

## 7. 적용 파일

이번 수정은 다음 파일에 반영했다.

```text
finance_llm/evaluation/smoke_test_exaone.py
finance_llm/prepare_qlora_dataset.py
finance_llm/labeler.py
```

적용 내용:

```text
- smoke_test_exaone.py: 실제 추론용 prompt 강화
- prepare_qlora_dataset.py: 학습 데이터 instruction 강화
- labeler.py: Claude 라벨링용 system prompt 강화
```

## 8. 기대 효과

이번 수정으로 특히 다음 개선을 기대한다.

```text
- 기업_분석 -> 실적_전망 오분류 감소
- 기업_분석 <-> 성장_동력 경계 안정화
- 산업_분석과 산업_트렌드 분기 기준 명확화
```

다만 이번 변경은 prompt 보정 중심이므로, 성능 개선 폭에는 한계가 있을 수 있다.

```text
prompt 보정만으로 충분하지 않으면
1. primary_errors 재라벨링
2. 산업_분석 샘플 보강
3. 기업_분석 / 성장_동력 / 실적_전망 경계 샘플 추가
4. 새 instruction으로 dataset 재생성 후 2차 QLoRA
```

## 9. 다음 권장 작업

```text
1. 수정된 prompt 기준으로 eval_exaone_classifier.py 재실행
2. summary.json 전후 비교
3. primary_errors.jsonl 상위 confusion pair 재확인
4. accuracy 0.80 이상 / macro F1 0.79 이상 여부 확인
5. 개선이 제한적이면 데이터 재정비 + 재학습 진행
```
