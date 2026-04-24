import os
import json
import time
from pathlib import Path

from pdf_processor import process_pdf

# 입력/출력 경로 설정
PDF_DIR = "./finance_report"
OUTPUT_DIR = "./finance_report/output"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 처리할 PDF 파일 목록 수집
    pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))
    total = len(pdf_files)
    print(f"총 {total}개 PDF 발견\n")

    summary = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{idx:02d}/{total}] 처리 중: {pdf_path.name}")

        try:
            result = process_pdf(str(pdf_path), use_vision=True)

            pdfplumber_pages = [p for p in result["pages"] if p["routing"] == "pdfplumber"]
            vision_pages = [p for p in result["pages"] if p["routing"] == "vision"]

            print(f"  → pdfplumber 페이지 수: {len(pdfplumber_pages)}")
            print(f"  → vision 페이지 수: {len(vision_pages)}")

            # 파일별 JSON 결과 저장
            out_path = Path(OUTPUT_DIR) / f"{pdf_path.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            summary.append({
                "filename": pdf_path.name,
                "total_pages": len(result["pages"]),
                "pdfplumber_pages": len(pdfplumber_pages),
                "vision_pages": len(vision_pages),
                "error": result.get("error", None)
            })

        except Exception as e:
            print(f"  → 오류 발생: {e}")
            summary.append({
                "filename": pdf_path.name,
                "total_pages": 0,
                "pdfplumber_pages": 0,
                "vision_pages": 0,
                "error": str(e)
            })

        # API 레이트 리밋 방지
        time.sleep(0.5)

    # 전체 summary.json 저장
    summary_path = Path(OUTPUT_DIR) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 완료 통계 출력
    total_pdfplumber = sum(s["pdfplumber_pages"] for s in summary)
    total_vision = sum(s["vision_pages"] for s in summary)
    errors = [s for s in summary if s["error"]]

    print(f"\n{'=' * 50}")
    print(f"배치 완료")
    print(f"총 pdfplumber 페이지 수: {total_pdfplumber}")
    print(f"총 Vision 페이지 수: {total_vision}")
    if errors:
        print(f"에러 발생 파일 ({len(errors)}개):")
        for e in errors:
            print(f"  - {e['filename']}: {e['error']}")
    else:
        print("에러 없음")
    print(f"결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
