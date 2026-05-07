import json

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
CATEGORIES_SET = CATEGORY_SET

CATEGORY_DEFINITIONS = {
    "산업_트렌드": "산업 전반의 방향성, 외부 수요/공급 변화, 시장 성장률, 업황 변화",
    "성장_동력": "기업의 미래 성장 근거, 신사업, 기술 우위, 신규 고객 확보, 증설/투자/신제품이 성장 논리로 제시된 경우",
    "실적_전망": "매출/영업이익/EPS/마진 등 수치 기반 실적 추정, 컨센서스, 가이던스, 상향/하향 전망",
    "산업_분석": "경쟁사 비교, 점유율 비교, 산업 구조, 공급망, 밸류체인, 업계 내 포지셔닝 비교",
    "기업_분석": "기업 내부 현황, 사업부 구조, 생산라인, 제품 믹스, 고객사, 투자 현황, 현재 진행 중인 전략/운영 상태",
    "리스크_요인": "투자 thesis를 훼손할 하방 요인, 규제, 수요 둔화, 비용 부담, 경쟁 심화",
    "밸류에이션": "목표주가 산정 근거, PER/PBR/EV/EBITDA 등 밸류에이션 배수와 평가 논리",
}

CLASSIFICATION_RULES = [
    "숫자, 매출, 영업이익, 증가율이 있어도 미래 실적 추정/가이던스/컨센서스가 아니면 실적_전망으로 보내지 마세요.",
    "회사의 현재 사업부 구성, 생산능력, 고객사, 제품군, 투자 집행, 운영 현황 설명은 기업_분석을 우선하세요.",
    "증설, 신제품, 기술력, 파트너십, 신규 시장 진입이 미래 성장 근거로 쓰이면 성장_동력을 우선하세요.",
    "경쟁사 비교, 점유율 비교, 공급망/밸류체인/산업 구조 설명은 산업_분석을 우선하세요.",
    "산업 전체 수요/공급, 업황, 시장 성장 방향은 산업_트렌드를 우선하세요.",
    "primary는 문장의 중심 논지를 가장 잘 설명하는 하나만 고르고, secondary는 보조 맥락일 때만 넣으세요.",
]

DISAMBIGUATION_RULES = [
    "기업 내부 현황 + 숫자: 기본은 기업_분석",
    "산업 전망 + 숫자: 기본은 산업_트렌드",
    "미래 실적 수치 추정/컨센서스/가이던스: 실적_전망",
    "기술/증설/파트너십이 성장 논리의 핵심: 성장_동력",
    "비교/점유율/공급망/밸류체인: 산업_분석",
]

FEW_SHOT_EXAMPLES = [
    (
        "글로벌 HBM 수요가 2025년 40% 증가할 전망",
        {"primary": "산업_트렌드", "secondary": []},
    ),
    (
        "2024년 매출액 32조원, 영업이익 4.2조원으로 컨센서스를 상회할 전망",
        {"primary": "실적_전망", "secondary": []},
    ),
    (
        "MX/NW 3.3조원, VD/가전 0.3조원으로 추정",
        {"primary": "기업_분석", "secondary": []},
    ),
    (
        "부산 공장 전장 라인이 3분기부터 본격 가동되며 전장용 매출 비중이 상승하고 있다.",
        {"primary": "기업_분석", "secondary": ["실적_전망"]},
    ),
    (
        "삼성전자는 HBM3E 양산을 통해 고객사 점유율 회복을 추진 중이다",
        {"primary": "성장_동력", "secondary": ["기업_분석"]},
    ),
    (
        "아이폰13용 OLED는 삼성 73%, LG 27%로 공급된다.",
        {"primary": "산업_분석", "secondary": ["기업_분석"]},
    ),
    (
        "12개월 Forward PER 14배 적용, 목표주가 9만원",
        {"primary": "밸류에이션", "secondary": []},
    ),
    (
        "미중 반도체 규제 강화시 수출 차질 우려",
        {"primary": "리스크_요인", "secondary": ["산업_트렌드"]},
    ),
    (
        "HBM 수요 증가로 ASP 상승이 이어지며 메모리 업체들의 내년 영업이익 추정치도 상향되고 있다.",
        {"primary": "산업_트렌드", "secondary": ["실적_전망"]},
    ),
    (
        "신규 AI 가속기 출시와 북미 고객사 확보가 본격화되며 중장기 매출 성장 기반이 강화되고 있다.",
        {"primary": "성장_동력", "secondary": ["기업_분석"]},
    ),
    (
        "서버 증설이 4분기부터 반영되면서 생산능력 확대가 내년 매출 증가로 이어질 전망이다.",
        {"primary": "실적_전망", "secondary": ["성장_동력"]},
    ),
    (
        "주요 고객사향 출하 비중이 확대되고 있지만 원재료 가격 상승 부담으로 수익성 개선 폭은 제한적일 수 있다.",
        {"primary": "리스크_요인", "secondary": ["기업_분석"]},
    ),
    (
        "경쟁사 대비 HBM 수율은 우위에 있으나 신규 증설 속도에서는 뒤처져 점유율 방어 부담이 존재한다.",
        {"primary": "산업_분석", "secondary": ["리스크_요인"]},
    ),
    (
        "스마트폰 사업부 재고 정상화와 플래그십 판매 회복으로 하반기 실적 반등 가능성이 높아지고 있다.",
        {"primary": "실적_전망", "secondary": ["기업_분석"]},
    ),
    (
        "북미 빅테크와의 공동 개발 경험은 향후 맞춤형 AI 서버 수주 확대의 핵심 근거가 될 것이다.",
        {"primary": "성장_동력", "secondary": ["산업_분석"]},
    ),
    (
        "12개월 Forward PER 18배를 적용하되, 내년 EPS 상향 가능성을 반영해 목표주가를 11만원으로 높였다.",
        {"primary": "밸류에이션", "secondary": ["실적_전망"]},
    ),
]


def format_category_list() -> str:
    return ", ".join(CATEGORIES)


def format_category_definitions() -> str:
    return "\n".join(
        f"- {category}: {CATEGORY_DEFINITIONS[category]}"
        for category in CATEGORIES
    )


def format_classification_rules() -> str:
    return "\n".join(
        f"{index}. {rule}"
        for index, rule in enumerate(CLASSIFICATION_RULES, start=1)
    )


def format_disambiguation_rules() -> str:
    return "\n".join(f"- {rule}" for rule in DISAMBIGUATION_RULES)


def format_few_shot_examples() -> str:
    blocks = []
    for input_text, output_json in FEW_SHOT_EXAMPLES:
        blocks.append(
            f'입력: "{input_text}"\n출력: {json.dumps(output_json, ensure_ascii=False)}'
        )
    return "\n\n".join(blocks)
