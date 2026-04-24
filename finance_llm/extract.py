import pdfplumber
import re
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ================================
# 노이즈 패턴
# ================================
NOISE_PATTERNS = [
    r'^\d+$',                        # 페이지 번호 단독
    r'^자료\s*:\s*.+',               # "자료 : 카카오, SK증권"
    r'작성자.*신의성실',
    r'본 보고서에 언급된',
    r'본 보고서는 기관',
    r'당사는 자료공표일',
    r'종목별 투자의견',
    r'투자판단 3단계',
    r'유니버스 투자등급',
    r'Compliance Notice',
    r'^\s*$',
    r'@\w+\.\w+',                    # 이메일 주소
    r'^\d{3,4}-\d{4}$',             # 전화번호
]

def clean_text(text):
    lines = text.split('\n')
    cleaned = [l for l in lines 
               if not any(re.search(p, l.strip()) for p in NOISE_PATTERNS)]
    return '\n'.join(cleaned).strip()

def is_chart_page(text):
    """짧은 행 비율 50% 이상이면 차트 페이지"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return True
    short_lines = [l for l in lines if len(l) <= 5]
    return (len(short_lines) / len(lines)) > 0.5

# ================================
# 단일 PDF 처리
# ================================
def process_pdf(pdf_path: str) -> dict:
    result = {
        "filename": Path(pdf_path).name,
        "pages": []
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                raw = page.extract_text() or ""
                cleaned = clean_text(raw)
                char_count = len(cleaned)
                is_chart = is_chart_page(cleaned) or char_count < 200
                
                result["pages"].append({
                    "page": i,
                    "char_count": char_count,
                    "is_chart": is_chart,
                    "routing": "vision" if is_chart else "pdfplumber",
                    "text": cleaned if not is_chart else ""
                })
    except Exception as e:
        result["error"] = str(e)
    
    return result

# ================================
# 테스트 실행
# ================================
# ================================
# 배치 실행
# ================================
if __name__ == "__main__":
    import time

    PDF_DIR = "./finance_report"
    OUTPUT_DIR = "./finance_report/output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    print(f"총 {len(pdf_files)}개 PDF 발견\n")

    summary = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{idx:02d}/{len(pdf_files)}] 처리 중: {pdf_path.name}")
        
        result = process_pdf(str(pdf_path))
        
        # 페이지별 라우팅 요약
        pdfplumber_pages = [p["page"] for p in result["pages"] if p["routing"] == "pdfplumber"]
        vision_pages = [p["page"] for p in result["pages"] if p["routing"] == "vision"]
        
        print(f"  → pdfplumber: {pdfplumber_pages}")
        print(f"  → vision 필요: {vision_pages}")
        
        # 개별 결과 저장
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
        
        time.sleep(0.1)  # 파일 I/O 안정성

    # 전체 요약 저장
    summary_path = Path(OUTPUT_DIR) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 통계 출력
    total_vision = sum(s["vision_pages"] for s in summary)
    total_pdfplumber = sum(s["pdfplumber_pages"] for s in summary)
    errors = [s for s in summary if s["error"]]

    print(f"\n{'='*50}")
    print(f"배치 완료")
    print(f"총 pdfplumber 페이지: {total_pdfplumber}")
    print(f"총 Vision 필요 페이지: {total_vision}")
    print(f"에러 발생 파일: {len(errors)}개")
    if errors:
        for e in errors:
            print(f"  - {e['filename']}: {e['error']}")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
        