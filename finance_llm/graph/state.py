"""Shared LangGraph state definitions for report generation."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict, cast

try:
    from finance_llm.config.categories import CATEGORIES
except ModuleNotFoundError:
    from config.categories import CATEGORIES


CategoryName = Literal[
    "산업_트렌드",
    "성장_동력",
    "실적_전망",
    "산업_분석",
    "기업_분석",
    "리스크_요인",
    "밸류에이션",
]

RawReportInput = str | dict[str, Any] | list[dict[str, Any]]


class SentenceItem(TypedDict):
    """Text chunk produced by the preprocess node."""

    text: str
    source: NotRequired[str]
    page: NotRequired[int]
    index: NotRequired[int]
    metadata: NotRequired[dict[str, Any]]


class ClassifiedItem(TypedDict):
    """Classifier output for one sentence or paragraph chunk."""

    text: str
    primary: CategoryName
    secondary: list[CategoryName]
    source: NotRequired[str]
    page: NotRequired[int]
    index: NotRequired[int]
    confidence: NotRequired[float]
    raw_output: NotRequired[str]
    classifier: NotRequired[Literal["exaone", "claude_fallback", "manual"]]
    metadata: NotRequired[dict[str, Any]]


class BucketItem(TypedDict):
    """Evidence item grouped under the primary category bucket."""

    text: str
    primary: CategoryName
    secondary: list[CategoryName]
    source: NotRequired[str]
    page: NotRequired[int]
    index: NotRequired[int]
    confidence: NotRequired[float]
    metadata: NotRequired[dict[str, Any]]


CategoryBuckets = dict[CategoryName, list[BucketItem]]
DraftSections = dict[CategoryName, str]


class ReviewNote(TypedDict):
    """Reviewer feedback for the composed report."""

    section: NotRequired[CategoryName | Literal["overall"]]
    severity: NotRequired[Literal["info", "warning", "error"]]
    note: str
    suggested_fix: NotRequired[str]


class FinanceReportState(TypedDict, total=False):
    """State shared across all LangGraph nodes.

    The state is `total=False` because LangGraph nodes usually return partial
    updates. Use `make_initial_state` when starting a graph run if every key
    should exist from the beginning.
    """

    raw_input: RawReportInput
    sentences: list[SentenceItem]
    classified_items: list[ClassifiedItem]
    buckets: CategoryBuckets
    draft_sections: DraftSections
    final_report: str
    review_notes: list[ReviewNote]


def empty_buckets() -> CategoryBuckets:
    return cast(CategoryBuckets, {category: [] for category in CATEGORIES})


def empty_draft_sections() -> DraftSections:
    return cast(DraftSections, {category: "" for category in CATEGORIES})


def make_initial_state(raw_input: RawReportInput) -> FinanceReportState:
    """Create a fully initialized graph state for a report run."""

    return {
        "raw_input": raw_input,
        "sentences": [],
        "classified_items": [],
        "buckets": empty_buckets(),
        "draft_sections": empty_draft_sections(),
        "final_report": "",
        "review_notes": [],
    }
