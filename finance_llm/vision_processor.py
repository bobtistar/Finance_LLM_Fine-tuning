import os
import json
import re
import time
import google.generativeai as genai
from pdf2image import convert_from_path
from dotenv import load_dotenv

# .env에서 API 키 로드
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL = "gemini-1.5-flash"

# 차트/표/텍스트 추출 프롬프트
EXTRACTION_PROMPT = """이 이미지는 증권사 리서치 리포트의 한 페이지입니다.
페이지에 있는 차트, 표, 텍스트를 분석하여 반드시 아래 JSON 형식으로만 반환하세요.

{
  "charts": ["차트 설명"],
  "tables": ["표 핵심 수치"],
  "text": "텍스트 요약"
}

JSON 외 다른 텍스트는 절대 포함하지 마세요."""


def process_vision_page(pdf_path: str, page_num: int) -> str:
    """특정 PDF 페이지를 이미지로 변환 후 Gemini로 내용 추출

    Args:
        pdf_path: PDF 파일 경로
        page_num: 처리할 페이지 번호 (1-based)

    Returns:
        추출된 내용의 한국어 텍스트 (JSON 파싱 실패 시 raw 텍스트)
    """
    try:
        # 해당 페이지만 이미지로 변환 (dpi=150)
        images = convert_from_path(
            pdf_path,
            dpi=150,
            first_page=page_num,
            last_page=page_num
        )

        if not images:
            return ""

        image = images[0]

        # Gemini API 호출
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content([EXTRACTION_PROMPT, image])
        raw_text = response.text.strip()

        result = _parse_and_format(raw_text)

        # 무료 티어 RPM 한도(15회/분) 준수
        time.sleep(4)
        return result

    except Exception as e:
        return f"[Vision 처리 오류: {str(e)}]"


def _parse_and_format(raw_text: str) -> str:
    """Gemini 응답 JSON을 파싱하여 한국어 텍스트로 변환

    JSON 파싱 실패 시 raw 텍스트 그대로 반환
    """
    try:
        # 마크다운 코드블록(```json ... ```) 제거
        cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', raw_text).strip()
        data = json.loads(cleaned)

        parts = []

        charts = data.get("charts", [])
        if charts:
            parts.append("[차트]")
            for chart in charts:
                parts.append(f"- {chart}")

        tables = data.get("tables", [])
        if tables:
            parts.append("[표]")
            for table in tables:
                parts.append(f"- {table}")

        text = data.get("text", "")
        if text:
            parts.append("[텍스트]")
            parts.append(text)

        return '\n'.join(parts)

    except (json.JSONDecodeError, KeyError, AttributeError):
        # JSON 파싱 실패 시 raw 텍스트 반환
        return raw_text
