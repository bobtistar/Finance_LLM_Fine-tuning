# TODO

## 진행 중

- [ ] LLM 레이블링 (`labeler.py` 실행 중)

---

## 완료 후 할 것

### 1. 데이터셋 품질 검토

- [ ] 총 레이블링 건수 확인 (목표: 1,000건 이상)
- [ ] 카테고리별 분포 확인
- [ ] 샘플 50개 수동 검토

### 2. RunPod 환경 세팅

- [ ] RunPod 인스턴스 구성
- [ ] 학습 환경 설치 (CUDA, 라이브러리 등)

### 3. Qwen3.5-9B QLoRA Fine-tuning

- [ ] 학습 데이터 포맷 변환
- [ ] QLoRA 학습 실행
- [ ] 모델 평가

### 4. LangGraph Multi-agent 설계

- [ ] 에이전트 역할 분담 설계
- [ ] 추론 에이전트 (Claude API) 연동
- [ ] 단순 태스크 에이전트 (Fine-tuned Qwen3.5-9B) 연동
- [ ] 보고서 생성 파이프라인 통합

---

## 완료

- [x] PDF 수집 (네이버 금융 리서치 60개)
- [x] 전처리 (pdfplumber + Gemini Vision 라우팅)
- [x] 배치 추출 (`output/` 폴더 JSON 저장)
