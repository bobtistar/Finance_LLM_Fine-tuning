import os
import json
import re
import asyncio
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

INPUT_DIR = "./finance_report/output"
OUTPUT_FILE = "./finance_report/labeled_dataset.jsonl"

MODEL = "claude-haiku-4-5"
CONCURRENCY = 10  # 동시 API 호출 수

CATEGORIES = {
    "산업_트렌드",
    "성장_동력",
    "실적_전망",
    "산업_분석",
    "기업_분석",
    "리스크_요인",
    "밸류에이션",
}

SYSTEM_PROMPT = """당신은 증권사 리서치 리포트 문장을 분류하는 전문가입니다.
주어진 문장을 아래 7개 카테고리 중 하나로 분류하세요.

## 카테고리 정의
- 산업_트렌드: 해당 산업 전반의 방향성, 외부 수요/공급 변화
- 성장_동력: 기업 내부 역량, 신사업, 경쟁우위
- 실적_전망: 매출/영업이익/EPS 등 수치 기반 미래 예측
- 산업_분석: 경쟁사 비교, 산업 구조, 밸류체인
- 기업_분석: 기업 내부 현황, 사업부 구조, 전략
- 리스크_요인: 투자 thesis를 훼손할 하방 요인
- 밸류에이션: 목표주가 산정 근거, PER/PBR 등 배수 기반 평가

## 출력 규칙
- JSON만 출력 (앞뒤 설명 없이)
- secondary는 최대 2개 (없으면 빈 배열)
- 분류 불가 문장은 {"primary": null, "secondary": []} 반환

## 출력 형식
{"primary": "카테고리명", "secondary": ["카테고리명"]}

## Few-shot 예시

입력: "글로벌 HBM 수요가 2025년 40% 증가할 전망"
출력: {"primary": "산업_트렌드", "secondary": []}

입력: "HBM 수요 증가로 영업이익 반등 예상"
출력: {"primary": "산업_트렌드", "secondary": ["실적_전망"]}

입력: "12개월 Forward PER 14배 적용, 목표주가 9만원"
출력: {"primary": "밸류에이션", "secondary": []}

입력: "미중 반도체 규제 강화시 수출 차질 우려"
출력: {"primary": "리스크_요인", "secondary": ["산업_트렌드"]}

입력: "삼성전자는 HBM3E 양산을 통해 고객사 점유율 회복을 추진 중이다"
출력: {"primary": "성장_동력", "secondary": ["기업_분석"]}

입력: "2024년 매출액 32조원, 영업이익 4.2조원으로 컨센서스를 상회할 전망"
출력: {"primary": "실적_전망", "secondary": []}

입력: "SK하이닉스 대비 원가 경쟁력이 낮아 수익성 개선에 한계가 있다"
출력: {"primary": "산업_분석", "secondary": ["리스크_요인"]}

입력: "메모리 반도체 업황은 공급 과잉 해소와 AI 서버 수요 확대로 회복 국면에 진입했다"
출력: {"primary": "산업_트렌드", "secondary": []}"""


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[。！？])\s*|[\n]+', text)
    return [s.strip() for s in sentences if s.strip()]


def filter_sentences(sentences: list[str]) -> list[str]:
    return [s for s in sentences if len(s) >= 20]


async def classify_sentence(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    sentence: str,
) -> dict | None:
    async with semaphore:
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=128,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": sentence}],
            )

            raw = response.content[0].text.strip()
            cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', raw).strip()
            data = json.loads(cleaned)

            if data.get("primary") is None:
                return None
            if data["primary"] not in CATEGORIES:
                return None

            secondary = [c for c in data.get("secondary", []) if c in CATEGORIES][:2]
            return {"primary": data["primary"], "secondary": secondary}

        except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
            return None
        except anthropic.RateLimitError:
            print("  → 레이트 리밋 도달, 10초 대기 중...")
            await asyncio.sleep(10)
            return None
        except anthropic.APIError as e:
            print(f"  → API 오류 (스킵): {e}")
            return None


async def process_file(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    json_path: Path,
    out_f,
    write_lock: asyncio.Lock,
) -> tuple[int, int]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  → 파일 읽기 오류 (스킵): {e}")
        return 0, 0

    filename = data.get("filename", json_path.name)
    pages = data.get("pages", [])

    # 모든 (page_num, sentence) 쌍 수집
    tasks = []
    meta = []
    for page_data in pages:
        page_num = page_data.get("page", 0)
        text = page_data.get("text", "")
        if not text:
            continue
        for sentence in filter_sentences(split_sentences(text)):
            tasks.append(classify_sentence(client, semaphore, sentence))
            meta.append((page_num, sentence))

    results = await asyncio.gather(*tasks)

    labeled = 0
    skipped = 0
    lines = []
    for (page_num, sentence), result in zip(meta, results):
        if result is None:
            skipped += 1
            continue
        record = {
            "source": filename,
            "page": page_num,
            "text": sentence,
            "primary": result["primary"],
            "secondary": result["secondary"],
        }
        lines.append(json.dumps(record, ensure_ascii=False))
        labeled += 1

    if lines:
        async with write_lock:
            out_f.write("\n".join(lines) + "\n")

    return labeled, skipped


async def main():
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    semaphore = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()

    input_files = sorted(
        f for f in Path(INPUT_DIR).glob("*.json")
        if f.name != "summary.json"
    )
    total_files = len(input_files)
    print(f"총 {total_files}개 JSON 파일 발견 (동시 처리: {CONCURRENCY})\n")

    total_labeled = 0
    total_skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for file_idx, json_path in enumerate(input_files, 1):
            print(f"[{file_idx:02d}/{total_files}] 처리 중: {json_path.name}")
            labeled, skipped = await process_file(
                client, semaphore, json_path, out_f, write_lock
            )
            total_labeled += labeled
            total_skipped += skipped
            print(f"  → 레이블링 완료: {labeled}건 (스킵: {skipped}건)")

    print(f"\n{'=' * 50}")
    print(f"레이블링 완료")
    print(f"총 레이블링 건수: {total_labeled}")
    print(f"스킵된 문장 수: {total_skipped}")
    print(f"저장 위치: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
