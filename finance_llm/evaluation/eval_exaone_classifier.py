"""
EXAONE 금융 분류 모델 간단 평가 스크립트.

처음에는 전체 304개를 바로 돌리지 말고 EVAL_LIMIT = 10으로 테스트합니다.
문제가 없으면 30, None 순서로 늘리면 됩니다.

IDE에서 실행하는 방법:
    1. 아래 EVAL_LIMIT 값을 원하는 개수로 수정합니다.
    2. 이 파일을 Run Python File로 실행합니다.

결과 저장 위치:
    finance_llm/eval_results/
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from smoke_test_exaone import (
    CATEGORIES,
    PROJECT_DIR,
    classify_sentence,
    load_tokenizer_and_model,
    parse_json_output,
    print_gpu_info,
)


# ---------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------

VALID_FILE = PROJECT_DIR / "finance_report" / "qlora_dataset" / "valid.jsonl"
RESULT_DIR = PROJECT_DIR / "eval_results"

# 처음에는 10개만 테스트하세요.
# 10개가 성공하면 30으로 바꾸고, 마지막 전체 평가는 None으로 바꾸면 됩니다.
EVAL_LIMIT = 10


# ---------------------------------------------------------------------
# 입력 옵션
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=EVAL_LIMIT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="4-bit 로딩을 끕니다. GTX 1660 Super에서는 보통 사용하지 않는 옵션입니다.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------

def load_valid_rows(limit):
    rows = []

    with VALID_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            text = item["input"].replace("문장:", "", 1).strip()
            gold = json.loads(item["output"])

            rows.append({
                "text": text,
                "gold_primary": gold["primary"],
                "gold_secondary": gold.get("secondary", []),
            })

            if limit is not None and len(rows) >= limit:
                break

    return rows


# ---------------------------------------------------------------------
# 평가 계산
# ---------------------------------------------------------------------

def calculate_secondary_f1(gold_list, pred_list):
    labeler = MultiLabelBinarizer(classes=CATEGORIES)
    gold_binary = labeler.fit_transform(gold_list)
    pred_binary = labeler.transform(pred_list)

    micro_f1 = f1_score(gold_binary, pred_binary, average="micro", zero_division=0)
    macro_f1 = f1_score(gold_binary, pred_binary, average="macro", zero_division=0)

    return micro_f1, macro_f1


def make_summary(rows, predictions, invalid_count):
    if not rows:
        return {
            "total_count": 0,
            "json_valid_count": 0,
            "invalid_count": invalid_count,
            "json_valid_rate": 0,
            "primary_accuracy": 0,
            "primary_macro_f1": 0,
            "primary_weighted_f1": 0,
            "secondary_micro_f1": 0,
            "secondary_macro_f1": 0,
        }

    gold_primary = [row["gold_primary"] for row in rows]
    pred_primary = [pred["primary"] for pred in predictions]

    gold_secondary = [row["gold_secondary"] for row in rows]
    pred_secondary = [pred["secondary"] for pred in predictions]
    secondary_micro_f1, secondary_macro_f1 = calculate_secondary_f1(
        gold_secondary,
        pred_secondary,
    )

    return {
        "total_count": len(rows),
        "json_valid_count": len(predictions),
        "invalid_count": invalid_count,
        "json_valid_rate": len(predictions) / len(rows) if rows else 0,
        "primary_accuracy": accuracy_score(gold_primary, pred_primary),
        "primary_macro_f1": f1_score(
            gold_primary,
            pred_primary,
            labels=CATEGORIES,
            average="macro",
            zero_division=0,
        ),
        "primary_weighted_f1": f1_score(
            gold_primary,
            pred_primary,
            labels=CATEGORIES,
            average="weighted",
            zero_division=0,
        ),
        "secondary_micro_f1": secondary_micro_f1,
        "secondary_macro_f1": secondary_macro_f1,
    }


# ---------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------

def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_confusion_matrix(path, gold_primary, pred_primary):
    matrix = confusion_matrix(gold_primary, pred_primary, labels=CATEGORIES)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + CATEGORIES)

        for category, values in zip(CATEGORIES, matrix):
            writer.writerow([category] + list(values))


def save_results(rows, predictions, invalid_samples, primary_errors, summary):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    gold_primary = [row["gold_primary"] for row in rows]
    pred_primary = [pred["primary"] for pred in predictions]

    with (RESULT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if rows:
        report = classification_report(
            gold_primary,
            pred_primary,
            labels=CATEGORIES,
            zero_division=0,
        )
    else:
        report = "No valid predictions. Check invalid_samples.jsonl first.\n"

    with (RESULT_DIR / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    save_confusion_matrix(
        RESULT_DIR / "confusion_matrix.csv",
        gold_primary,
        pred_primary,
    )
    save_jsonl(RESULT_DIR / "invalid_samples.jsonl", invalid_samples)
    save_jsonl(RESULT_DIR / "primary_errors.jsonl", primary_errors)


# ---------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    if not VALID_FILE.exists():
        raise FileNotFoundError(f"valid.jsonl을 찾을 수 없습니다: {VALID_FILE}")

    use_4bit = torch.cuda.is_available() and not args.no_4bit

    print("[준비] 평가 데이터 로드")
    rows = load_valid_rows(args.limit)
    print(f"[준비] 평가 샘플 수: {len(rows)}")

    print("[준비] 모델 로드")
    tokenizer, model = load_tokenizer_and_model(use_4bit=use_4bit)
    print_gpu_info()

    predictions = []
    invalid_samples = []
    primary_errors = []

    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] 추론 중...")

        raw_output = classify_sentence(
            tokenizer=tokenizer,
            model=model,
            text=row["text"],
            max_new_tokens=args.max_new_tokens,
        )
        parsed = parse_json_output(raw_output)

        if parsed is None:
            invalid_samples.append({
                "index": index,
                "text": row["text"],
                "gold_primary": row["gold_primary"],
                "gold_secondary": row["gold_secondary"],
                "raw_output": raw_output,
            })
            continue

        predictions.append(parsed)

        if parsed["primary"] != row["gold_primary"]:
            primary_errors.append({
                "index": index,
                "text": row["text"],
                "gold_primary": row["gold_primary"],
                "pred_primary": parsed["primary"],
                "gold_secondary": row["gold_secondary"],
                "pred_secondary": parsed["secondary"],
                "raw_output": raw_output,
            })

    # invalid 샘플을 제외하고 primary/secondary 점수를 계산합니다.
    # JSON valid rate는 summary에 따로 기록됩니다.
    valid_rows = []
    valid_predictions = []
    invalid_indexes = {sample["index"] for sample in invalid_samples}

    for index, row in enumerate(rows, start=1):
        if index in invalid_indexes:
            continue
        valid_rows.append(row)
        valid_predictions.append(predictions[len(valid_predictions)])

    summary = make_summary(
        rows=valid_rows,
        predictions=valid_predictions,
        invalid_count=len(invalid_samples),
    )
    summary["total_count"] = len(rows)
    summary["json_valid_count"] = len(valid_predictions)
    summary["json_valid_rate"] = len(valid_predictions) / len(rows) if rows else 0

    save_results(
        rows=valid_rows,
        predictions=valid_predictions,
        invalid_samples=invalid_samples,
        primary_errors=primary_errors,
        summary=summary,
    )

    print("\n[완료] 평가 결과 저장")
    print(f"저장 위치: {RESULT_DIR}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
