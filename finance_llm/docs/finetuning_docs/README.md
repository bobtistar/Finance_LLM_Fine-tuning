# Finance LLM Finetuning Notes

이 문서는 EXAONE-3.5 경량 모델을 QLoRA 방식으로 파인튜닝하면서 진행한 환경 세팅, 학습 과정, 오류 해결, 평가 계획, 다음 TODO를 정리한 기록입니다.

## 현재 목표

EXAONE 모델은 최종 금융 보고서를 작성하는 모델이 아니라, LangGraph 파이프라인 안에서 **단순 태스크용 분류기 / 라우터** 역할을 수행합니다.

```text
EXAONE-3.5-2.4B-Instruct
→ 문장/문단 단위 금융 카테고리 분류
→ primary / secondary category 출력

Claude API
→ 추론
→ 섹션별 분석
→ 최종 금융 분석 보고서 작성
```

## 사용 모델

```text
Base model: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
Tuning method: QLoRA
Output: LoRA adapter
```

## 사용 데이터

```text
all.jsonl
train.jsonl
valid.jsonl
stats.json
```

데이터는 이미 train / valid로 분리되어 있었으므로 별도 split을 다시 하지 않았습니다.

```text
Total: 3,036
Train: 2,732
Valid: 304
```

## 카테고리

```text
산업_트렌드
성장_동력
실적_전망
산업_분석
기업_분석
리스크_요인
밸류에이션
```

## 문서 구성

```text
01_runpod_environment_setup.md
02_training_process.md
03_troubleshooting_log.md
04_evaluation_plan.md
05_next_todo.md
06_langgraph_integration_plan.md
```
