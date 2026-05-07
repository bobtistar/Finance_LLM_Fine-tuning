import os
import json
import re
import time
import asyncio
import google.generativeai as genai
from pdf2image import convert_from_path
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL = "gemini-1.5-flash"
RPM_LIMIT = 15

EXTRACTION_PROMPT = """이 이미지는 증권사 리서치 리포트의 한 페이지입니다.
페이지에 있는 차트, 표, 텍스트를 분석하여 반드시 아래 JSON 형식으로만 반환하세요.

{
  "charts": ["차트 설명"],
  "tables": ["표 핵심 수치"],
  "text": "텍스트 요약"
}

JSON 외 다른 텍스트는 절대 포함하지 마세요."""


def process_vision_page(pdf_path: str, page_num: int) -> str:
    """단일 페이지 동기 처리 (하위 호환용)"""
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=page_num, last_page=page_num)
        if not images:
            return ""
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content([EXTRACTION_PROMPT, images[0]])
        time.sleep(4)  # 무료 15 RPM 준수
        return _parse_and_format(response.text.strip())
    except Exception as e:
        return f"[Vision 처리 오류: {str(e)}]"


async def _process_single_async(model, pdf_path: str, page_num: int) -> str:
    loop = asyncio.get_event_loop()
    try:
        images = await loop.run_in_executor(
            None,
            lambda: convert_from_path(pdf_path, dpi=150, first_page=page_num, last_page=page_num)
        )
        if not images:
            return ""
        response = await model.generate_content_async([EXTRACTION_PROMPT, images[0]])
        return _parse_and_format(response.text.strip())
    except Exception as e:
        return f"[Vision 처리 오류: {str(e)}]"


async def _batch_async(pages: list) -> dict:
    """15개씩 동시 처리, 60초 윈도우 단위 배치"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    results = {}

    for i in range(0, len(pages), RPM_LIMIT):
        batch = pages[i:i + RPM_LIMIT]
        batch_start = time.time()

        print(f"  [Vision 배치 {i // RPM_LIMIT + 1}] {i + 1}-{min(i + RPM_LIMIT, len(pages))} / {len(pages)} 동시 처리 중...")

        tasks = [_process_single_async(model, pdf_path, page_num) for pdf_path, page_num in batch]
        texts = await asyncio.gather(*tasks, return_exceptions=True)

        for (pdf_path, page_num), text in zip(batch, texts):
            if isinstance(text, Exception):
                results[(pdf_path, page_num)] = f"[Vision 처리 오류: {str(text)}]"
            else:
                results[(pdf_path, page_num)] = text

        if i + RPM_LIMIT < len(pages):
            elapsed = time.time() - batch_start
            wait = max(0, 60 - elapsed)
            if wait > 0:
                print(f"  → 다음 배치까지 {wait:.1f}초 대기 중...")
                await asyncio.sleep(wait)

    return results


def process_vision_pages_batch(pages: list) -> dict:
    """여러 페이지를 async 배치로 처리 (15 RPM 무료 한도 내 최대 속도)

    Args:
        pages: (pdf_path, page_num) 튜플 리스트

    Returns:
        {(pdf_path, page_num): text} 딕셔너리
    """
    return asyncio.run(_batch_async(pages))


def _parse_and_format(raw_text: str) -> str:
    try:
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
        return raw_text
