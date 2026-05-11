"""
LangGraph multi-agent 리포트 생성 파이프라인의 공유 state 정의.

파이프라인 흐름:
  raw_input
    → (splitter) sentences
    → (classifier) classified_items / failed_sentences
    → (bucketer) buckets
    → (drafter) draft_sections
    → (composer) final_report
    → (reviewer) review_notes / review_passed

설계 원칙:
- 모든 필드는 plain Python 타입만 허용 (LangGraph 체크포인팅 호환)
- 모델·토크나이저 객체는 state에 두지 않음 (직렬화 불가)
- sentences와 classified_items는 별도 필드로 유지 (원본 소실 방지)
- errors만 operator.add 리듀서 적용 (병렬 노드에서 덮어쓰지 않고 누적)
- total=False로 선언해 중간 단계 state에서 KeyError 방지
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


# ---------------------------------------------------------------------------
# 데이터 단위 타입
# ---------------------------------------------------------------------------

class ClassifiedSentence(TypedDict):
    sentence: str
    source_page: int | None   # PDF 페이지 번호. 텍스트 직접 입력이면 None
    primary: str
    secondary: list[str]      # 최대 2개, primary와 중복 없음


class SectionDraft(TypedDict):
    category: str
    sentences: list[str]      # 이 섹션에 사용된 원문 문장 (추적용)
    draft: str                # LLM이 생성한 단락 초안


class ReviewNote(TypedDict):
    section: str              # 검토 대상 섹션 카테고리
    issue: str                # 발견된 문제
    suggestion: str           # 수정 제안


class GraphError(TypedDict):
    node: str                 # 오류가 발생한 노드 이름
    message: str              # 오류 내용


# ---------------------------------------------------------------------------
# 그래프 전체 공유 state
# ---------------------------------------------------------------------------
#
# 필드 생애주기:
#   [진입]      raw_input, source_name
#   [splitter]  sentences
#   [classifier] classified_items, failed_sentences
#   [bucketer]  buckets
#   [drafter]   draft_sections
#   [composer]  final_report
#   [reviewer]  review_notes, review_passed
#   [모든 노드] errors (누적)

class ReportState(TypedDict, total=False):
    # ── 입력 ───────────────────────────────────────────────────────────────
    raw_input: str       # PDF에서 추출하거나 직접 전달된 원본 텍스트
    source_name: str     # 파일명 또는 리포트 식별자

    # ── 전처리 ─────────────────────────────────────────────────────────────
    sentences: list[str]  # 문장 분리 결과. 이후 노드에서 수정하지 않음

    # ── 분류 ───────────────────────────────────────────────────────────────
    classified_items: list[ClassifiedSentence]
    failed_sentences: list[str]  # 파싱 실패 문장. 재시도 or 스킵 판단용

    # ── 버킷팅 ─────────────────────────────────────────────────────────────
    buckets: dict[str, list[str]]  # category → [sentence, ...]

    # ── 초안 생성 ───────────────────────────────────────────────────────────
    draft_sections: list[SectionDraft]

    # ── 최종 리포트 ─────────────────────────────────────────────────────────
    final_report: str

    # ── 검토 ───────────────────────────────────────────────────────────────
    review_notes: list[ReviewNote]
    review_passed: bool   # 조건부 엣지(conditional edge) 분기 기준

    # ── 오류 누적 ───────────────────────────────────────────────────────────
    # operator.add: 노드가 새 오류 목록을 반환하면 기존 목록에 append (덮어쓰지 않음)
    # 병렬 노드가 동시에 오류를 기록해도 소실되지 않음
    errors: Annotated[list[GraphError], operator.add]
