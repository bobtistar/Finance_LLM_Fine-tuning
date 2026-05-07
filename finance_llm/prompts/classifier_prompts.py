import json

from common.category import (
    FEW_SHOT_EXAMPLES,
    format_category_definitions,
    format_category_list,
    format_classification_rules,
    format_disambiguation_rules,
    format_few_shot_examples,
)


def build_exaone_classifier_prompt(text: str) -> str:
    return (
        "당신은 증권사 리서치 리포트 문장을 분류하는 금융 분석 보조 모델입니다.\n"
        "주어진 문장을 아래 7개 카테고리 중 primary 1개와 secondary 최대 2개로 분류하세요.\n\n"
        f"카테고리: {format_category_list()}\n\n"
        "카테고리 정의:\n"
        f"{format_category_definitions()}\n\n"
        "중요한 판별 규칙:\n"
        f"{format_classification_rules()}\n\n"
        "혼동 방지 기준:\n"
        f"{format_disambiguation_rules()}\n\n"
        "예시:\n"
        f"{format_few_shot_examples()}\n\n"
        "반드시 JSON만 출력하세요.\n"
        '출력 형식: {"primary":"카테고리명","secondary":["카테고리명"]}\n\n'
        f"문장: {text}"
    )


def build_claude_classifier_system_prompt() -> str:
    few_shot_lines = []
    for input_text, output_json in FEW_SHOT_EXAMPLES:
        few_shot_lines.append(f'입력: "{input_text}"')
        few_shot_lines.append(
            f"출력: {json.dumps(output_json, ensure_ascii=False)}"
        )

    few_shot_text = "\n\n".join(few_shot_lines)

    return (
        "당신은 증권사 리서치 리포트 문장을 분류하는 전문가입니다.\n"
        "주어진 문장을 아래 7개 카테고리 중 하나의 primary와 최대 2개의 secondary로 분류하세요.\n\n"
        "## 카테고리 정의\n"
        f"{format_category_definitions()}\n\n"
        "## 중요 규칙\n"
        + "\n".join(
            f"- {line.split('. ', 1)[1]}"
            for line in format_classification_rules().splitlines()
        )
        + "\n\n## 혼동 방지 기준\n"
        f"{format_disambiguation_rules()}\n\n"
        "## 출력 규칙\n"
        "- JSON만 출력\n"
        "- secondary는 최대 2개\n"
        '- 분류 불가 문장은 {"primary": null, "secondary": []} 반환\n\n'
        "## 출력 형식\n"
        '{"primary": "카테고리명", "secondary": ["카테고리명"]}\n\n'
        "## Few-shot 예시\n\n"
        f"{few_shot_text}"
    )
