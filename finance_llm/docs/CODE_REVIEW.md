# 코드 리뷰 및 핵심 로직 맵

## 한 줄 결론

현재 코드는 단계별 스크립트가 잘 분리되어 있어 전체 흐름을 따라가기는 쉽습니다. 다만 실행 경로, 카테고리 정의, API 설정, 에러 처리 방식이 여러 파일에 흩어져 있어 장기 유지보수 전에는 공통 설정과 데이터 계약을 한곳으로 모으는 것이 좋습니다.

## 전체 파이프라인

```text
PDF 파일
  -> extract.py
  -> pdf_processor.py
  -> vision_processor.py
  -> finance_report/output/*.json
  -> labeler.py
  -> finance_report/labeled_dataset.jsonl
  -> review_sampler.py
  -> finance_report/review_samples.csv
  -> prepare_qlora_dataset.py
  -> finance_report/qlora_dataset/*.jsonl
```

## 코드 상태 요약

| 항목 | 평가 | 메모 |
|---|---|---|
| 파일 분리 | 양호 | 추출, Vision, 레이블링, 샘플링, 학습 데이터 변환이 파일별로 나뉘어 있음 |
| 함수 크기 | 대체로 양호 | `extract.py`, `labeler.py`의 `main`은 조금 더 쪼개면 테스트하기 쉬움 |
| 타입 힌트 | 부분 적용 | `prepare_qlora_dataset.py`는 좋고, 나머지는 반환 타입과 자료구조 타입을 보강할 여지가 있음 |
| 설정 관리 | 개선 필요 | 경로, 모델명, 동시성, 카테고리가 여러 파일에 하드코딩되어 있음 |
| 에러 처리 | 개선 필요 | 일부 에러가 문자열로 결과에 섞이거나 조용히 스킵되어 원인 추적이 어려울 수 있음 |
| 테스트 용이성 | 개선 필요 | 외부 API, 파일 입출력, 현재 작업 디렉터리 의존성이 강함 |

## 파일별 핵심 로직

### [extract.py](../extract.py)

역할: PDF 추출 파이프라인의 오케스트레이터입니다.

| 구성 | 설명 |
|---|---|
| `PDF_DIR` | 입력 PDF 폴더. 현재 `./finance_report`에 고정되어 있음 |
| `OUTPUT_DIR` | 추출 JSON 저장 폴더. 현재 `./finance_report/output`에 고정되어 있음 |
| `main()` | PDF 탐색, 차트 페이지 탐지, Vision 배치 처리, 결과 병합, 요약 JSON 저장을 수행 |

주요 흐름:

1. `PDF_DIR`에서 PDF 목록을 정렬해 읽습니다.
2. `process_pdf(..., use_vision=False)`로 먼저 모든 페이지의 라우팅만 판단합니다.
3. 라우팅이 `vision`인 페이지만 모아 `process_vision_pages_batch()`로 배치 처리합니다.
4. Vision 결과를 기존 페이지 결과에 병합합니다.
5. PDF별 JSON과 `summary.json`을 저장합니다.

유지보수 포인트:

- 현재 경로가 실행 위치에 의존합니다. `Path(__file__).resolve().parent` 기준으로 바꾸면 어디서 실행해도 안정적입니다.
- `main()` 내부에 3단계 로직이 모두 들어 있어 테스트 단위가 큽니다. `detect_pages()`, `merge_vision_results()`, `write_outputs()` 정도로 분리하면 좋습니다.
- 결과 스키마를 문서화하거나 타입으로 고정하면 후속 단계가 안전해집니다.

### [pdf_processor.py](../pdf_processor.py)

역할: PDF 페이지에서 텍스트를 추출하고, 페이지별 처리 방식을 결정합니다.

| 구성 | 설명 |
|---|---|
| `NOISE_PATTERNS` | 페이지 번호, 자료 출처, 면책 문구, 이메일, 전화번호 등 제거 대상 정규식 |
| `clean_text(text)` | 줄 단위로 노이즈 패턴을 제거하고 정제 텍스트 반환 |
| `is_chart_page(text)` | 글자 수와 짧은 행 비율로 차트 페이지 여부 판단 |
| `process_pdf(pdf_path, use_vision=True)` | PDF를 페이지별로 순회하며 `pdfplumber` 또는 `vision` 라우팅 결정 |

주요 흐름:

1. `pdfplumber.open()`으로 PDF를 엽니다.
2. 페이지별로 `extract_text()`를 호출합니다.
3. `clean_text()`로 불필요한 줄을 제거합니다.
4. `is_chart_page()`로 Vision 필요 여부를 판단합니다.
5. `use_vision=True`면 차트 페이지를 바로 Vision 처리하고, 아니면 빈 텍스트로 라우팅 정보만 남깁니다.

유지보수 포인트:

- `is_chart_page()` 기준이 간단해서 빠르지만 오탐 가능성이 있습니다. 실제 샘플 검토 후 임계값을 설정값으로 분리하면 좋습니다.
- `except Exception`으로 전체 PDF 오류를 `result["error"]`에 담습니다. 페이지 단위 실패와 파일 단위 실패를 구분하면 재처리가 쉬워집니다.
- `NOISE_PATTERNS`는 금융사 리포트가 늘어날수록 커질 가능성이 있어 별도 설정 파일로 빼도 좋습니다.

### [vision_processor.py](../vision_processor.py)

역할: 차트/표 중심 페이지를 이미지로 변환해 Gemini Vision으로 요약합니다.

| 구성 | 설명 |
|---|---|
| `GEMINI_MODEL` | Gemini 모델명 |
| `RPM_LIMIT` | 분당 요청 제한 기준. 현재 15 |
| `EXTRACTION_PROMPT` | 차트, 표, 텍스트 요약을 JSON으로 받기 위한 프롬프트 |
| `process_vision_page(pdf_path, page_num)` | 단일 페이지 동기 처리 |
| `_process_single_async(model, pdf_path, page_num)` | 단일 페이지 비동기 처리 |
| `_batch_async(pages)` | `RPM_LIMIT` 단위로 병렬 처리 후 60초 윈도우를 맞춤 |
| `process_vision_pages_batch(pages)` | 외부에서 호출하는 배치 처리 진입점 |
| `_parse_and_format(raw_text)` | 모델의 JSON 응답을 사람이 읽기 쉬운 섹션 텍스트로 변환 |

주요 흐름:

1. `pdf2image.convert_from_path()`로 지정 페이지를 이미지로 변환합니다.
2. Gemini에 이미지와 프롬프트를 함께 전달합니다.
3. 응답 JSON의 `charts`, `tables`, `text`를 `[차트]`, `[표]`, `[텍스트]` 섹션으로 변환합니다.
4. JSON 파싱 실패 시 원문 응답을 그대로 반환합니다.

유지보수 포인트:

- `genai.configure(api_key=os.getenv("GEMINI_API_KEY"))`가 import 시점에 실행됩니다. API 키가 없을 때 더 명확한 오류 메시지를 주는 초기화 함수가 있으면 좋습니다.
- `process_vision_pages_batch()`는 내부에서 `asyncio.run()`을 호출합니다. CLI에서는 괜찮지만 Jupyter, FastAPI 등 이미 이벤트 루프가 있는 환경에서는 문제가 될 수 있습니다.
- 오류가 `[Vision 처리 오류: ...]` 문자열로 데이터에 들어갑니다. 후속 학습 데이터에 섞이지 않도록 `status`, `error` 필드를 분리하는 편이 안전합니다.

### [labeler.py](../labeler.py)

역할: 추출된 페이지 텍스트를 문장 단위로 나누고 Claude API로 카테고리를 레이블링합니다.

| 구성 | 설명 |
|---|---|
| `INPUT_DIR` | `extract.py`가 만든 JSON 입력 폴더 |
| `OUTPUT_FILE` | 레이블링 결과 JSONL 경로 |
| `MODEL` | Anthropic 모델명 |
| `CONCURRENCY` | 동시 API 호출 수 |
| `CATEGORIES` | 허용 카테고리 집합 |
| `SYSTEM_PROMPT` | 분류 기준과 few-shot 예시 |
| `split_sentences(text)` | 문장 경계 기준으로 텍스트 분할 |
| `filter_sentences(sentences)` | 너무 짧은 문장 제거 |
| `classify_sentence(...)` | 단일 문장을 API로 분류하고 결과 검증 |
| `process_file(...)` | 파일 하나의 모든 문장을 분류해 JSONL 라인 생성 |
| `main()` | 입력 파일 순회, 전체 레이블링 실행, 출력 파일 저장 |

주요 흐름:

1. `INPUT_DIR`에서 `summary.json`을 제외한 JSON 파일을 찾습니다.
2. 페이지 텍스트를 문장으로 분리하고 20자 이상만 남깁니다.
3. 문장별 `classify_sentence()` 작업을 만들고 `asyncio.gather()`로 실행합니다.
4. API 응답이 JSON이고 카테고리가 유효한 경우에만 저장합니다.
5. `source`, `page`, `text`, `primary`, `secondary` 필드로 JSONL을 생성합니다.

유지보수 포인트:

- `CATEGORIES`가 다른 파일에도 중복됩니다. 공통 모듈 또는 설정 파일로 분리해야 카테고리 변경 시 누락이 줄어듭니다.
- 레이트 리밋 발생 시 10초 대기 후 해당 문장을 스킵합니다. 재시도 횟수와 백오프를 두면 데이터 손실을 줄일 수 있습니다.
- 파일 하나의 문장 작업을 한 번에 모두 생성합니다. 큰 파일에서는 배치 단위 처리로 메모리와 API 실패 범위를 줄이는 편이 좋습니다.
- `ANTHROPIC_API_KEY`가 없을 때 명시적으로 실패시키면 원인 파악이 쉽습니다.

### [review_sampler.py](../review_sampler.py)

역할: 레이블링 결과에서 카테고리별 수동 검토 샘플 CSV를 생성합니다.

| 구성 | 설명 |
|---|---|
| `JSONL_PATH` | 레이블링 결과 입력 |
| `OUTPUT_PATH` | 수동 검토 CSV 출력 |
| `CATEGORIES` | 카테고리 출력 순서 |
| `SAMPLES_PER_CAT` | 카테고리별 추출 샘플 수 |
| `SEED` | 재현 가능한 샘플링을 위한 랜덤 시드 |
| `load_by_category(path)` | JSONL을 카테고리별로 그룹화 |
| `sample(by_cat)` | 카테고리별 지정 수만큼 무작위 추출 |
| `write_csv(rows, output)` | Excel 호환 UTF-8 BOM CSV 저장 |
| `print_summary(rows)` | 샘플 분포와 검토 가이드 출력 |

주요 흐름:

1. `labeled_dataset.jsonl`을 한 줄씩 읽습니다.
2. `primary` 기준으로 데이터를 그룹화합니다.
3. `SAMPLES_PER_CAT` 기준으로 카테고리별 샘플을 뽑습니다.
4. 검토자가 `O`, `X`, `?`를 입력할 수 있는 CSV를 만듭니다.

유지보수 포인트:

- 현재 입력 레코드에 `_idx`를 직접 추가합니다. 작은 스크립트에서는 괜찮지만, 원본 데이터 보존을 위해 별도 래핑 구조를 쓰면 더 깔끔합니다.
- `JSONL_PATH`, `OUTPUT_PATH`, 샘플 수를 CLI 인자로 받을 수 있으면 반복 검토가 쉬워집니다.
- 카테고리 목록 역시 공통화 대상입니다.

### [prepare_qlora_dataset.py](../prepare_qlora_dataset.py)

역할: 레이블링 JSONL을 QLoRA 학습용 instruction 데이터셋으로 변환합니다.

| 구성 | 설명 |
|---|---|
| `DEFAULT_INPUT` | 레이블링 JSONL 기본 경로 |
| `DEFAULT_OUTPUT_DIR` | 학습 데이터 출력 폴더 |
| `CATEGORIES`, `CATEGORY_SET` | 허용 카테고리와 검증용 집합 |
| `INSTRUCTION` | 학습 데이터에 들어갈 고정 지시문 |
| `parse_args()` | 입력, 출력, 검증 비율, 시드, 중복 제거 옵션 파싱 |
| `normalize_record(item)` | 원본 레코드 검증 및 정규화 |
| `load_records(path, dedupe_text)` | JSONL 전체 로드, 오류/중복 통계 수집 |
| `stratified_split(records, valid_ratio, seed)` | 카테고리별 분포를 유지하며 train/valid 분리 |
| `to_training_row(record)` | instruction/input/output 포맷으로 변환 |
| `write_jsonl(path, records)` | 변환 결과 저장 |
| `distribution(records)` | 카테고리별 분포 계산 |
| `main()` | 전체 변환 실행, `all/train/valid/stats` 저장 |

주요 흐름:

1. CLI 인자를 읽습니다.
2. JSONL을 로드하며 유효하지 않은 레코드를 스킵합니다.
3. 필요하면 같은 `text`를 가진 중복 레코드를 제거합니다.
4. 카테고리별 층화 방식으로 train/valid를 나눕니다.
5. 학습용 JSONL 3종과 통계 JSON을 저장합니다.

유지보수 포인트:

- 이 파일은 현재 코드 중 가장 관리하기 좋은 형태입니다. CLI 인자, 검증, 통계 저장이 잘 들어가 있습니다.
- `CATEGORIES`와 `INSTRUCTION`은 `labeler.py`의 기준과 반드시 동기화되어야 합니다. 프롬프트/카테고리 공통화가 필요합니다.
- 데이터가 적은 카테고리는 `valid_count`가 최소 1개로 잡히므로 작은 데이터셋에서는 검증셋 분포를 확인해야 합니다.

## 데이터 계약

### 추출 결과 JSON

생성 위치: `finance_report/output/{pdf_stem}.json`

```json
{
  "filename": "report.pdf",
  "pages": [
    {
      "page": 1,
      "char_count": 1234,
      "is_chart": false,
      "routing": "pdfplumber",
      "text": "..."
    }
  ],
  "error": null
}
```

주의:

- `error`는 오류가 있을 때만 들어갈 수 있습니다.
- `routing`은 현재 `"pdfplumber"` 또는 `"vision"`입니다.
- Vision 오류 문자열이 `text`에 들어갈 수 있으므로 학습 전 필터링 기준이 필요합니다.

### 레이블링 결과 JSONL

생성 위치: `finance_report/labeled_dataset.jsonl`

```json
{"source":"report.pdf","page":1,"text":"문장","primary":"산업_트렌드","secondary":["실적_전망"]}
```

주의:

- `primary`는 7개 카테고리 중 하나여야 합니다.
- `secondary`는 최대 2개이며, `primary`와 중복되지 않는 것이 좋습니다.

### QLoRA 학습 JSONL

생성 위치: `finance_report/qlora_dataset/{all,train,valid}.jsonl`

```json
{
  "instruction": "분류 지시문",
  "input": "문장: ...",
  "output": "{\"primary\":\"산업_트렌드\",\"secondary\":[]}"
}
```

주의:

- `output`은 문자열화된 JSON입니다. 학습/평가 코드에서 다시 파싱해야 합니다.

## 개선 우선순위

### 1순위: 실행 안정성

- 모든 경로를 `Path(__file__).resolve().parent` 기준으로 정리합니다.
- API 키 누락 시 명확한 예외를 발생시킵니다.
- `requirements.txt` 또는 `pyproject.toml`로 의존성을 고정합니다.

### 2순위: 공통 설정 분리

- 7개 카테고리를 `categories.py` 또는 `config/categories.json`으로 이동합니다.
- Gemini/Claude 모델명, 동시성, RPM 제한, 입력/출력 경로를 설정으로 분리합니다.
- `labeler.py`와 `prepare_qlora_dataset.py`의 프롬프트 기준이 어긋나지 않도록 한곳에서 관리합니다.

### 3순위: 실패 추적 강화

- Vision/API 실패를 일반 텍스트와 분리해 `status`, `error` 필드로 저장합니다.
- 레이트 리밋과 일시적 API 오류에는 재시도 정책을 둡니다.
- `print()` 중심 로그를 `logging`으로 바꾸고 로그 레벨을 구분합니다.

### 4순위: 테스트 추가

- `clean_text()`, `is_chart_page()`, `split_sentences()`, `normalize_record()`, `stratified_split()`부터 단위 테스트를 추가합니다.
- 외부 API 호출 함수는 mock으로 테스트합니다.
- 샘플 JSONL을 두고 변환 결과 스키마를 검증합니다.

## 리팩터링 후보 체크리스트

- [ ] `BASE_DIR`, `DATA_DIR`, `OUTPUT_DIR` 경로 공통화
- [ ] `CATEGORIES` 단일 소스화
- [ ] 프롬프트 파일 또는 프롬프트 모듈 분리
- [ ] API 키 검증 함수 추가
- [ ] Vision 실패 레코드 필터링 기준 추가
- [ ] 레이블링 API 재시도 정책 추가
- [ ] `review_sampler.py` CLI 인자 추가
- [ ] 최소 단위 테스트 추가

## 현재 기준으로 좋은 점

- 파이프라인이 파일 단위로 직관적으로 나뉘어 있어 처음 보는 사람이 흐름을 잡기 쉽습니다.
- 외부 API를 쓰는 부분과 데이터셋 변환 부분이 분리되어 있어 나중에 모델이나 API를 바꾸기 쉽습니다.
- `prepare_qlora_dataset.py`는 인자 파싱, 검증, 통계 저장이 포함되어 있어 유지보수성이 가장 좋습니다.
- 추출 결과, 레이블링 결과, 학습 데이터가 각각 JSON/JSONL/CSV로 남아 중간 산출물 점검이 가능합니다.
