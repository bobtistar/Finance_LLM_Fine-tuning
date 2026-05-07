import json
import re

from common.category import CATEGORY_SET


def parse_classification_json(
    raw_text: str,
    *,
    allow_null_primary: bool = False,
) -> dict | None:
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

    primary = data.get("primary")
    secondary = data.get("secondary", [])

    if not isinstance(secondary, list):
        return None

    if primary is None:
        if not allow_null_primary:
            return None
        return {"primary": None, "secondary": []}

    if primary not in CATEGORY_SET:
        return None

    normalized_secondary = [
        category
        for category in secondary
        if category in CATEGORY_SET and category != primary
    ][:2]

    return {
        "primary": primary,
        "secondary": normalized_secondary,
    }
