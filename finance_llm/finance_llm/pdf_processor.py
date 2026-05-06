import pdfplumber
import re
from pathlib import Path

# 제거할 노이즈 패턴 목록
NOISE_PATTERNS = [
    r'^\d+$',                                               # 페이지 번호 단독 행
    r'^자료\s*:\s*.+',                                      # "자료 : 카카오, SK증권" 형태
    r'작성자',                                               # 면책조항: 작성자
    r'본 보고서',                                            # 면책조항: 본 보고서
    r'당사는',                                               # 면책조항: 당사는
    r'종목별',                                               # 면책조항: 종목별
    r'투자판단',                                              # 면책조항: 투자판단
    r'유니버스',                                              # 면책조항: 유니버스
    r'Compliance\s*Notice',                                 # Compliance Notice
    r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', # 이메일 주소
    r'^\d{2,4}[\-\s]\d{3,4}[\-\s]\d{4}',                  # 전화번호
    r'^\s*$',                                               # 빈 줄
]


def clean_text(text: str) -> str:
    """노이즈 패턴에 해당하는 행을 제거하고 정제된 텍스트 반환"""
    lines = text.split('\n')
    cleaned = [
        line for line in lines
        if not any(re.search(pattern, line.strip()) for pattern in NOISE_PATTERNS)
    ]
    return '\n'.join(cleaned).strip()


def is_chart_page(text: str) -> bool:
    """차트 페이지 여부 판단

    - 전체 글자 수 200자 미만이면 차트 페이지
    - 5자 이하 짧은 행 비율이 50% 이상이면 차트 페이지
    """
    if len(text) < 200:
        return True

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return True

    short_lines = [line for line in lines if len(line) <= 5]
    return (len(short_lines) / len(lines)) > 0.5


def process_pdf(pdf_path: str, use_vision: bool = True) -> dict:
    """PDF 파일을 페이지별로 처리하여 텍스트 추출 및 라우팅 결정

    Args:
        pdf_path: PDF 파일 경로
        use_vision: 차트 페이지에 Vision 처리 적용 여부

    Returns:
        {
            "filename": str,
            "pages": [
                {
                    "page": int,
                    "char_count": int,
                    "is_chart": bool,
                    "routing": str,  # "pdfplumber" | "vision"
                    "text": str
                }
            ]
        }
    """
    # vision import는 use_vision=True일 때만 실행 (순환 의존 방지)
    if use_vision:
        from vision_processor import process_vision_page

    result = {
        "filename": Path(pdf_path).name,
        "pages": []
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                raw_text = page.extract_text() or ""
                cleaned = clean_text(raw_text)
                char_count = len(cleaned)
                chart_page = is_chart_page(cleaned)

                if chart_page:
                    routing = "vision"
                    # use_vision=True일 때만 Gemini API 호출
                    if use_vision:
                        text = process_vision_page(pdf_path, i)
                    else:
                        text = ""
                else:
                    routing = "pdfplumber"
                    text = cleaned

                result["pages"].append({
                    "page": i,
                    "char_count": char_count,
                    "is_chart": chart_page,
                    "routing": routing,
                    "text": text
                })

    except Exception as e:
        result["error"] = str(e)

    return result
