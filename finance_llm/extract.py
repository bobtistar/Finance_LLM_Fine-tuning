import os
import json
from pathlib import Path

from pdf_processor import process_pdf
from vision_processor import process_vision_pages_batch

PDF_DIR = "./finance_report"
OUTPUT_DIR = "./finance_report/output"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))
    total = len(pdf_files)
    print(f"총 {total}개 PDF 발견\n")

    # --- 1단계: 전체 PDF 구조 파악 (Vision 호출 없이) ---
    print("=== 1단계: 차트 페이지 탐지 ===")
    all_results = {}
    vision_pages = []  # (pdf_path, page_num) 전체 목록

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{idx:02d}/{total}] 탐지 중: {pdf_path.name}")
        result = process_pdf(str(pdf_path), use_vision=False)
        all_results[str(pdf_path)] = result

        for page in result["pages"]:
            if page["routing"] == "vision":
                vision_pages.append((str(pdf_path), page["page"]))

    print(f"\n총 Vision 페이지: {len(vision_pages)}개\n")

    # --- 2단계: Vision 페이지 async 배치 처리 ---
    print("=== 2단계: Vision 배치 처리 (15개 동시) ===")
    vision_results = {}
    if vision_pages:
        vision_results = process_vision_pages_batch(vision_pages)

    # --- 3단계: 결과 병합 후 JSON 저장 ---
    print("\n=== 3단계: 결과 저장 ===")
    summary = []

    for pdf_path in pdf_files:
        result = all_results[str(pdf_path)]

        # vision 결과 채우기
        for page in result["pages"]:
            if page["routing"] == "vision":
                key = (str(pdf_path), page["page"])
                page["text"] = vision_results.get(key, "")

        pdfplumber_count = sum(1 for p in result["pages"] if p["routing"] == "pdfplumber")
        vision_count = sum(1 for p in result["pages"] if p["routing"] == "vision")

        print(f"  저장: {pdf_path.name}  (pdfplumber: {pdfplumber_count}, vision: {vision_count})")

        out_path = Path(OUTPUT_DIR) / f"{pdf_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        summary.append({
            "filename": pdf_path.name,
            "total_pages": len(result["pages"]),
            "pdfplumber_pages": pdfplumber_count,
            "vision_pages": vision_count,
            "error": result.get("error", None)
        })

    summary_path = Path(OUTPUT_DIR) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    total_pdfplumber = sum(s["pdfplumber_pages"] for s in summary)
    total_vision = sum(s["vision_pages"] for s in summary)
    errors = [s for s in summary if s["error"]]

    print(f"\n{'=' * 50}")
    print(f"배치 완료")
    print(f"총 pdfplumber 페이지: {total_pdfplumber}")
    print(f"총 Vision 페이지: {total_vision}")
    if errors:
        print(f"에러 발생 파일 ({len(errors)}개):")
        for e in errors:
            print(f"  - {e['filename']}: {e['error']}")
    else:
        print("에러 없음")
    print(f"결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
