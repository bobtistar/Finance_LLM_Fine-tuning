import os
import json
import re
import asyncio
from pathlib import Path

import anthropic
from dotenv import load_dotenv

try:
    from config.categories import LABELING_SYSTEM_PROMPT, parse_classification_json
except ModuleNotFoundError:
    from finance_llm.config.categories import (
        LABELING_SYSTEM_PROMPT,
        parse_classification_json,
    )

load_dotenv()

INPUT_DIR = "./finance_report/output"
OUTPUT_FILE = "./finance_report/labeled_dataset.jsonl"

MODEL = "claude-haiku-4-5"
CONCURRENCY = 10  # 동시 API 호출 수


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
                        "text": LABELING_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": sentence}],
            )

            raw = response.content[0].text.strip()
            return parse_classification_json(raw, allow_null_primary=True)

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
