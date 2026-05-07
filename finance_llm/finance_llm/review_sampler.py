"""
카테고리별 층화 샘플링으로 수동 검토용 CSV 생성

출력: finance_report/review_samples.csv
- 카테고리당 7-8개씩 총 50개 (소수 카테고리 우선 확보)
- review 열 비워둠 → 검토 후 O/X/? 입력
"""

import json
import csv
import random
from collections import defaultdict
from pathlib import Path

JSONL_PATH = "finance_report/labeled_dataset.jsonl"
OUTPUT_PATH = "finance_report/review_samples.csv"

CATEGORIES = [
    "산업_트렌드",
    "성장_동력",
    "실적_전망",
    "산업_분석",
    "기업_분석",
    "리스크_요인",
    "밸류에이션",
]

# 카테고리당 샘플 수 (소수 카테고리에 가중치)
SAMPLES_PER_CAT = {
    "산업_분석":   10,  # 120건으로 가장 적음
    "리스크_요인": 10,  # 134건으로 두 번째로 적음
    "밸류에이션":   7,
    "실적_전망":    6,
    "산업_트렌드":  6,
    "성장_동력":    6,
    "기업_분석":    5,
}  # 합계: 50개

SEED = 42


def load_by_category(path: str) -> dict:
    by_cat = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            item["_idx"] = i + 1  # 원본 행 번호
            by_cat[item["primary"]].append(item)
    return by_cat


def sample(by_cat: dict) -> list:
    random.seed(SEED)
    rows = []
    for cat in CATEGORIES:
        pool = by_cat.get(cat, [])
        n = SAMPLES_PER_CAT.get(cat, 7)
        picked = random.sample(pool, min(n, len(pool)))
        rows.extend(picked)
    # 카테고리 순서대로 정렬
    rows.sort(key=lambda x: (CATEGORIES.index(x["primary"]), x["_idx"]))
    return rows


def write_csv(rows: list, output: str):
    with open(output, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel 호환
        writer = csv.writer(f)
        writer.writerow(["#", "주카테고리", "보조카테고리", "텍스트", "출처", "페이지", "review"])
        for i, row in enumerate(rows, 1):
            secondary = ", ".join(row.get("secondary", []))
            writer.writerow([
                i,
                row["primary"],
                secondary,
                row["text"],
                row["source"],
                row["page"],
                "",  # 검토자 입력란: O(정확) / X(오분류) / ?(애매)
            ])


def print_summary(rows: list):
    from collections import Counter
    dist = Counter(r["primary"] for r in rows)
    print(f"\n샘플 추출 완료: 총 {len(rows)}개")
    print("\n카테고리별 샘플 수:")
    for cat in CATEGORIES:
        print(f"  {cat}: {dist.get(cat, 0)}개")
    print(f"\n저장 위치: {OUTPUT_PATH}")
    print("\n[review 열 입력 가이드]")
    print("  O  → 카테고리 정확")
    print("  X  → 카테고리 오분류 (올바른 카테고리 옆에 메모)")
    print("  ?  → 애매함")


if __name__ == "__main__":
    by_cat = load_by_category(JSONL_PATH)

    print("카테고리별 전체 건수:")
    for cat in CATEGORIES:
        print(f"  {cat}: {len(by_cat.get(cat, []))}건")

    rows = sample(by_cat)
    write_csv(rows, OUTPUT_PATH)
    print_summary(rows)
