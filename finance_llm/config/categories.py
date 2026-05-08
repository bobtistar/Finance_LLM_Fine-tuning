"""Shared category definitions, prompts, and JSON parsing rules."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any


CATEGORY_DEFINITIONS = OrderedDict(
    [
        (
            "산업_트렌드",
            "산업 전반의 방향성, 외부 수요/공급 변화, 시장 성장률, 업황 변화",
        ),
        (
            "성장_동력",
            "기업의 미래 성장 근거, 신사업, 기술 우위, 신규 고객 확보, 증설/투자/신제품이 성장 논리로 제시된 경우",
        ),
        (
            "실적_전망",
            "매출/영업이익/EPS/마진 등 수치 기반 실적 추정, 컨센서스, 가이던스, 상향/하향 전망",
        ),
        (
            "산업_분석",
            "경쟁사 비교, 점유율 비교, 산업 구조, 공급망, 밸류체인, 업계 내 포지셔닝 비교",
        ),
        (
            "기업_분석",
            "기업 내부 현황, 사업부 구조, 생산라인, 제품 믹스, 고객사, 투자 현황, 현재 진행 중인 전략/운영 상태",
        ),
        (
            "리스크_요인",
            "투자 thesis를 훼손할 하방 요인, 규제, 수요 둔화, 비용 부담, 경쟁 심화",
        ),
        (
            "밸류에이션",
            "목표주가 산정 근거, PER/PBR/EV/EBITDA 등 밸류에이션 배수와 평가 논리",
        ),
    ]
)

CATEGORIES = list(CATEGORY_DEFINITIONS.keys())
CATEGORY_SET = set(CATEGORIES)
MAX_SECONDARY_CATEGORIES = 2


def format_category_names() -> str:
    return ", ".join(CATEGORIES)


def format_category_definitions() -> str:
    return "\n".join(
        f"- {category}: {description}"
        for category, description in CATEGORY_DEFINITIONS.items()
    )


CLASSIFICATION_RULES = """중요한 판별 규칙:
1. 숫자, 매출, 영업이익, 증가율 표현이 있어도 미래 실적 추정/가이던스/컨센서스가 아니면 실적_전망으로 분류하지 않습니다.
2. 회사의 현재 사업부 구성, 생산능력, 고객사, 제품군, 투자 집행, 운영 현황 설명은 기업_분석을 우선합니다.
3. 증설, 신제품, 기술력, 파트너십, 신규 시장 진입이 미래 성장 근거로 쓰이면 성장_동력을 우선합니다.
4. 경쟁사 비교, 점유율 비교, 공급망/밸류체인/산업 구조 설명은 산업_분석을 우선합니다.
5. 산업 전체 수요/공급, 업황, 시장 성장 방향은 산업_트렌드를 우선합니다.
6. primary는 문장의 중심 논지를 가장 잘 설명하는 하나만 고르고, secondary는 보조 맥락일 때만 넣습니다.

혼동 방지 기준:
- 기업 내부 현황 + 숫자: 기본은 기업_분석
- 산업 전망 + 숫자: 기본은 산업_트렌드
- 미래 실적 수치 추정/컨센서스/가이던스: 실적_전망
- 기술/증설/파트너십이 성장 논리의 핵심: 성장_동력
- 비교/점유율/공급망/밸류체인: 산업_분석"""

OUTPUT_RULES = """출력은 반드시 JSON만 반환하세요.
출력 형식: {"primary": "카테고리명", "secondary": ["카테고리명"]}"""

NULLABLE_OUTPUT_RULES = """출력 규칙:
- JSON만 출력
- secondary는 최대 2개
- 분류 불가 문장은 {"primary": null, "secondary": []} 반환

출력 형식:
{"primary": "카테고리명", "secondary": ["카테고리명"]}"""

FEW_SHOT_EXAMPLES = """Few-shot 예시:

입력: "글로벌 HBM 수요가 2025년 40% 증가할 전망"
출력: {"primary": "산업_트렌드", "secondary": []}

입력: "2024년 매출액 32조원, 영업이익 4.2조원으로 컨센서스를 상회할 전망"
출력: {"primary": "실적_전망", "secondary": []}

입력: "MX/NW 3.3조원, VD/가전 0.3조원으로 추정"
출력: {"primary": "기업_분석", "secondary": []}

입력: "부산 공장 전장 라인이 3분기부터 본격 가동되며 전장용 매출 비중이 상승하고 있다."
출력: {"primary": "기업_분석", "secondary": ["실적_전망"]}

입력: "삼성전자는 HBM3E 양산을 통해 고객사 점유율 회복을 추진 중이다"
출력: {"primary": "성장_동력", "secondary": ["기업_분석"]}

입력: "아이폰13용 OLED는 삼성 73%, LG 27%로 공급된다."
출력: {"primary": "산업_분석", "secondary": ["기업_분석"]}

입력: "12개월 Forward PER 14배 적용, 목표주가 9만원"
출력: {"primary": "밸류에이션", "secondary": []}

입력: "미중 반도체 규제 강화시 수출 차질 우려"
출력: {"primary": "리스크_요인", "secondary": ["산업_트렌드"]}"""


def build_classification_instruction(include_nullable_rule: bool = False) -> str:
    output_rules = NULLABLE_OUTPUT_RULES if include_nullable_rule else OUTPUT_RULES
    return (
        "당신은 증권사 리서치 리포트 문장을 분류하는 금융 분석 보조 모델입니다.\n"
        "주어진 문장을 아래 7개 카테고리 중 하나의 주카테고리(primary)와 "
        "최대 2개의 보조카테고리(secondary)로 분류하세요.\n\n"
        "카테고리 정의:\n"
        f"{format_category_definitions()}\n\n"
        f"{CLASSIFICATION_RULES}\n\n"
        f"{output_rules}"
    )


TRAINING_INSTRUCTION = build_classification_instruction()
LABELING_SYSTEM_PROMPT = (
    build_classification_instruction(include_nullable_rule=True)
    + "\n\n"
    + FEW_SHOT_EXAMPLES
)


def make_classification_prompt(text: str) -> str:
    return (
        "당신은 증권사 리서치 리포트 문장을 분류하는 금융 분석 보조 모델입니다.\n"
        "주어진 문장을 아래 7개 카테고리 중 primary 1개와 secondary 최대 2개로 분류하세요.\n\n"
        f"카테고리: {format_category_names()}\n\n"
        "카테고리 정의:\n"
        f"{format_category_definitions()}\n\n"
        f"{CLASSIFICATION_RULES}\n\n"
        "반드시 JSON만 출력하세요.\n"
        '출력 형식: {"primary":"카테고리명","secondary":["카테고리명"]}\n\n'
        f"문장: {text}"
    )


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return data if isinstance(data, dict) else None


def normalize_classification_result(
    data: dict[str, Any],
    allow_null_primary: bool = False,
) -> dict[str, list[str] | str] | None:
    primary = data.get("primary")
    secondary = data.get("secondary", [])

    if primary is None and allow_null_primary:
        return None
    if primary not in CATEGORY_SET:
        return None
    if not isinstance(secondary, list):
        return None

    clean_secondary = []
    for category in secondary:
        if (
            category in CATEGORY_SET
            and category != primary
            and category not in clean_secondary
        ):
            clean_secondary.append(category)

    return {
        "primary": primary,
        "secondary": clean_secondary[:MAX_SECONDARY_CATEGORIES],
    }


def parse_classification_json(
    raw_text: str,
    allow_null_primary: bool = False,
) -> dict[str, list[str] | str] | None:
    data = _extract_json_object(raw_text)
    if data is None:
        return None
    return normalize_classification_result(
        data,
        allow_null_primary=allow_null_primary,
    )
