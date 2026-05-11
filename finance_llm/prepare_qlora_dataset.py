import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "finance_report" / "labeled_dataset.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "finance_report" / "qlora_dataset"

CATEGORIES = [
    "산업_트렌드",
    "성장_동력",
    "실적_전망",
    "산업_분석",
    "기업_분석",
    "리스크_요인",
    "밸류에이션",
]
CATEGORY_SET = set(CATEGORIES)

INSTRUCTION = """당신은 증권사 리서치 리포트 문장을 분류하는 금융 분석 보조 모델입니다.
주어진 문장을 아래 7개 카테고리 중 하나의 주카테고리(primary)와 최대 2개의 보조카테고리(secondary)로 분류하세요.

카테고리 정의:
- 산업_트렌드: 해당 산업 전반의 방향성, 외부 수요/공급 변화
- 성장_동력: 기업 내부 역량, 신사업, 경쟁우위
- 실적_전망: 매출/영업이익/EPS 등 수치 기반 미래 예측
- 산업_분석: 경쟁사 비교, 산업 구조, 밸류체인
- 기업_분석: 기업 내부 현황, 사업부 구조, 전략
- 리스크_요인: 투자 thesis를 훼손할 하방 요인
- 밸류에이션: 목표주가 산정 근거, PER/PBR 등 배수 기반 평가

출력은 반드시 JSON만 반환하세요.
출력 형식: {"primary": "카테고리명", "secondary": ["카테고리명"]}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert labeled finance-report JSONL into QLoRA instruction-input-output files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dedupe-text",
        action="store_true",
        help="Drop exact duplicate text rows after validation.",
    )
    return parser.parse_args()


def normalize_record(item: dict[str, Any]) -> dict[str, Any] | None:
    text = str(item.get("text", "")).strip()
    primary = item.get("primary")
    secondary = item.get("secondary", [])

    if not text or primary not in CATEGORY_SET or not isinstance(secondary, list):
        return None

    clean_secondary = []
    for category in secondary:
        if category in CATEGORY_SET and category != primary and category not in clean_secondary:
            clean_secondary.append(category)

    return {
        "text": text,
        "primary": primary,
        "secondary": clean_secondary[:2],
    }


def load_records(path: Path, dedupe_text: bool) -> tuple[list[dict[str, Any]], Counter]:
    records = []
    skipped = Counter()
    seen_text = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                skipped["empty_line"] += 1
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped["json_error"] += 1
                continue

            record = normalize_record(item)
            if record is None:
                skipped["invalid_record"] += 1
                continue

            if dedupe_text and record["text"] in seen_text:
                skipped["duplicate_text"] += 1
                continue

            seen_text.add(record["text"])
            records.append(record)

    return records, skipped


def stratified_split(
    records: list[dict[str, Any]],
    valid_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= valid_ratio < 1:
        raise ValueError("--valid-ratio must be in the range [0, 1).")

    rng = random.Random(seed)
    by_category = defaultdict(list)
    for record in records:
        by_category[record["primary"]].append(record)

    train = []
    valid = []
    for category in CATEGORIES:
        bucket = by_category.get(category, [])
        rng.shuffle(bucket)

        if valid_ratio == 0 or len(bucket) <= 1:
            valid_count = 0
        else:
            valid_count = max(1, round(len(bucket) * valid_ratio))
            valid_count = min(valid_count, len(bucket) - 1)

        valid.extend(bucket[:valid_count])
        train.extend(bucket[valid_count:])

    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def to_training_row(record: dict[str, Any]) -> dict[str, str]:
    output = {
        "primary": record["primary"],
        "secondary": record["secondary"],
    }
    return {
        "instruction": INSTRUCTION,
        "input": f"문장: {record['text']}",
        "output": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_training_row(record), ensure_ascii=False) + "\n")


def distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["primary"] for record in records)
    return {category: counts.get(category, 0) for category in CATEGORIES}


def main() -> None:
    args = parse_args()
    records, skipped = load_records(args.input, args.dedupe_text)
    train, valid = stratified_split(records, args.valid_ratio, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "all.jsonl", records)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)

    stats = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "dedupe_text": args.dedupe_text,
        "total_records": len(records),
        "train_records": len(train),
        "valid_records": len(valid),
        "skipped": dict(skipped),
        "distribution": {
            "all": distribution(records),
            "train": distribution(train),
            "valid": distribution(valid),
        },
    }

    with (args.output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"변환 완료: {len(records)}건")
    print(f"  train: {len(train)}건 -> {args.output_dir / 'train.jsonl'}")
    print(f"  valid: {len(valid)}건 -> {args.output_dir / 'valid.jsonl'}")
    print(f"  all:   {len(records)}건 -> {args.output_dir / 'all.jsonl'}")
    print(f"  stats: {args.output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
