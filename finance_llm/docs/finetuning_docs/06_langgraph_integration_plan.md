# 06. LangGraph 통합 계획

## 1. 전체 역할 분담

```text
EXAONE
→ 단순 분류 / 라우팅 / 카테고리 bucket 생성

Claude API
→ 추론 / 금융 해석 / 섹션 작성 / 최종 보고서 작성
```

EXAONE에게 최종 보고서 작성을 맡기지 않습니다.

## 2. 전체 파이프라인

```text
전처리된 PDF JSON
→ 문장 또는 문단 단위 chunking
→ EXAONE classifier
→ category bucket 생성
→ Claude section writer
→ Claude report composer
→ Claude reviewer
→ 최종 금융 분석 보고서
```

## 3. EXAONE classifier 출력

권장 출력:

```json
{
  "primary": "산업_트렌드",
  "secondary": ["실적_전망"]
}
```

출력 후 parser를 반드시 통과시킵니다.

```python
def parse_model_json(text: str) -> dict | None:
    ...
```

## 4. Category bucket 구조

Claude에게 넘길 중간 데이터는 다음 형태가 좋습니다.

```json
{
  "산업_트렌드": [
    "HBM 수요 증가로 2025년 메모리 업황 개선이 예상된다."
  ],
  "성장_동력": [],
  "실적_전망": [],
  "산업_분석": [],
  "기업_분석": [],
  "리스크_요인": [],
  "밸류에이션": []
}
```

secondary category는 초기에는 metadata로만 저장하는 방식을 추천합니다.

```json
{
  "text": "HBM 수요 증가로 2025년 메모리 업황 개선이 예상된다.",
  "primary": "산업_트렌드",
  "secondary": ["실적_전망"]
}
```

secondary bucket에도 복사하면 중복 문장이 많아질 수 있고, Claude가 같은 근거를 반복 사용할 수 있습니다.

## 5. Fallback 전략

EXAONE 결과가 불안정하면 Claude로 fallback합니다.

Fallback 조건:

```text
- JSON parsing 실패
- primary가 7개 카테고리에 없음
- secondary가 list가 아님
- 입력 문장이 너무 길거나 복잡함
- 평가에서 recall이 낮았던 클래스 후보
```

예시:

```python
if parsed is None:
    return classify_with_claude(text)

if parsed["primary"] not in CATEGORIES:
    return classify_with_claude(text)
```

## 6. LangGraph node 설계

### Node 1. preprocess_node

```text
입력: 전처리된 리포트 JSON
출력: 문장/문단 chunks
```

### Node 2. exaone_classify_node

```text
입력: chunks
출력: classified chunks
```

출력 예시:

```json
[
  {
    "text": "...",
    "primary": "산업_트렌드",
    "secondary": ["실적_전망"]
  }
]
```

### Node 3. bucket_node

```text
입력: classified chunks
출력: category_buckets
```

### Node 4. claude_section_writer_node

```text
입력: category_buckets
출력: 섹션별 초안
```

### Node 5. claude_report_composer_node

```text
입력: 섹션별 초안
출력: 최종 보고서 초안
```

### Node 6. claude_reviewer_node

```text
입력: 최종 보고서 초안 + evidence
출력: 검수된 최종 보고서
```

## 7. End-to-End 평가

모델 단독 평가가 끝나면 실제 리포트 하나를 대상으로 전체 파이프라인을 평가합니다.

체크리스트:

```text
[ ] 7개 섹션이 모두 생성되는가
[ ] 각 섹션에 적절한 근거가 들어가는가
[ ] 리스크 요인이 누락되지 않는가
[ ] 밸류에이션 근거가 별도 섹션에 잘 들어가는가
[ ] Claude가 EXAONE의 잘못된 bucket을 어느 정도 보정하는가
[ ] 보고서 문장이 금융 리서치 톤에 맞는가
[ ] 중복 근거가 과도하게 반복되지 않는가
```

## 8. 통합 후 개선 방향

```text
1. EXAONE 분류 오류가 최종 보고서 품질에 얼마나 영향을 주는지 확인
2. 심각한 오류 유형만 골라 데이터 보강
3. Claude fallback 기준 조정
4. category bucket 구성 방식 개선
5. 필요 시 2차 QLoRA 진행
```
