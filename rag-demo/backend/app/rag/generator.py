from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .index import tokenize
from .models import AnswerPoint, ContextBlock, RetrievalHit


AnswerIntent = Literal["list", "process", "fact", "general", "fallback"]

PROCESS_QUESTION_MARKERS = ("流程", "步骤", "怎么加入", "如何加入", "怎样加入", "怎么申请", "如何申请")
LIST_QUESTION_MARKERS = ("哪些", "哪几", "有哪", "列出", "种类", "分类", "类型")
FACT_QUESTION_MARKERS = ("多少", "什么时候", "几点", "多久", "是否", "能否", "时间", "几本")
PROCESS_SENTENCE_PATTERN = re.compile(r"加入方式|报名流程|申请流程|具体步骤|第一步|首先|然后|最后|→|->")
LIST_SENTENCE_PATTERN = re.compile(r"分为|包括|包含|分别是|主要有|共有")
LIST_VALUE_PATTERN = re.compile(r"(?:分为|包括|包含|分别是|主要有|共有)([^，,。；;]+)")
GENERIC_QUERY_TERMS = {
    "学社", "创新学社", "需要", "经过", "流程", "步骤", "时候", "多少", "相关", "情况", "介绍", "告诉",
}


@dataclass(frozen=True)
class SentenceCandidate:
    score: float
    sentence: str
    chunk_id: str
    matched_terms: set[str]
    matched_topic_terms: set[str]
    intent_signal: str | None


@dataclass
class GenerationResult:
    answer: str
    citation_chunk_ids: list[str]
    intent: AnswerIntent
    points: list[AnswerPoint] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=lambda: {
        "requested_method": "extractive",
        "effective_method": "extractive",
        "provider": "local",
    })
    warning: str | None = None


def _matched_query_terms(query_terms: set[str], sentence: str) -> set[str]:
    sentence_terms = set(tokenize(sentence))
    return {
        term
        for term in query_terms
        if term in sentence_terms or term in sentence
    }


def detect_intent(question: str) -> AnswerIntent:
    if any(marker in question for marker in PROCESS_QUESTION_MARKERS):
        return "process"
    if any(marker in question for marker in LIST_QUESTION_MARKERS):
        return "list"
    if any(marker in question for marker in FACT_QUESTION_MARKERS):
        return "fact"
    return "general"


def _intent_signal(intent: AnswerIntent, sentence: str) -> str | None:
    if intent == "list" and LIST_SENTENCE_PATTERN.search(sentence):
        return "检测到枚举结构"
    if intent == "process" and PROCESS_SENTENCE_PATTERN.search(sentence):
        return "检测到流程结构"
    return None


def _parse_list_items(sentence: str) -> list[str]:
    match = LIST_VALUE_PATTERN.search(sentence)
    if not match:
        return []
    value = match.group(1).strip(" ：:")
    parts = [item.strip(" ，,。；;：:") for item in re.split(r"、|以及|和|及", value)]
    items = [item for item in parts if 1 < len(item) <= 30]
    return items if len(items) >= 2 else []


def _parse_process_steps(sentence: str) -> list[str]:
    if "→" not in sentence and "->" not in sentence:
        return []
    value = sentence.split("：", 1)[1] if "：" in sentence else sentence.split(":", 1)[-1]
    parts = [item.strip(" ，,。；;：:") for item in re.split(r"\s*(?:→|->)\s*", value)]
    return [item for item in parts if item]


class ExtractiveTeachingGenerator:
    """Intent-aware deterministic generation that keeps every answer point traceable."""

    id = "extractive"

    def build_context(self, hits: list[RetrievalHit]) -> list[ContextBlock]:
        return [
            ContextBlock(
                chunk_id=hit.chunk.id,
                citation=f"[{index}]",
                section=hit.chunk.section,
                text=hit.chunk.text,
            )
            for index, hit in enumerate(hits, start=1)
        ]

    def generate(self, question: str, hits: list[RetrievalHit]) -> GenerationResult:
        intent = detect_intent(question)
        query_terms = set(tokenize(question))
        topic_terms = query_terms - GENERIC_QUERY_TERMS
        if not topic_terms:
            topic_terms = query_terms
        citation_by_chunk = {hit.chunk.id: f"[{index}]" for index, hit in enumerate(hits, start=1)}
        candidates = self._build_candidates(intent, query_terms, topic_terms, hits)
        matched_topic_terms = set().union(*(candidate.matched_topic_terms for candidate in candidates)) if candidates else set()
        topic_coverage = len(topic_terms.intersection(matched_topic_terms)) / max(len(topic_terms), 1)
        base_evidence_sufficient = bool(hits and hits[0].score >= 0.07 and topic_coverage >= 0.34)

        structured_result = self._structured_answer(intent, candidates, citation_by_chunk) if base_evidence_sufficient else None
        if structured_result:
            structured_result.detail.update(self._generation_detail(intent, query_terms, topic_terms, candidates, structured_result.points))
            return structured_result

        selected = self._select_candidates(candidates, topic_terms)
        points = [
            AnswerPoint(
                text=candidate.sentence,
                citation=citation_by_chunk[candidate.chunk_id],
                chunk_id=candidate.chunk_id,
                selection_reason=candidate.intent_signal or "覆盖新的问题关键词",
            )
            for candidate in selected
        ]

        evidence_sufficient = bool(points and base_evidence_sufficient)
        if not evidence_sufficient:
            return GenerationResult(
                answer="当前知识库中没有检索到足够相关的内容，建议换一种更具体的问法。",
                citation_chunk_ids=[],
                intent="fallback",
                detail=self._generation_detail(intent, query_terms, topic_terms, candidates, []),
            )

        citation_chunk_ids = list(dict.fromkeys(point.chunk_id for point in points))
        answer = self._format_answer(intent, points)
        return GenerationResult(
            answer=answer,
            citation_chunk_ids=citation_chunk_ids,
            intent=intent,
            points=points,
            detail=self._generation_detail(intent, query_terms, topic_terms, candidates, points),
        )

    def _build_candidates(
        self,
        intent: AnswerIntent,
        query_terms: set[str],
        topic_terms: set[str],
        hits: list[RetrievalHit],
    ) -> list[SentenceCandidate]:
        candidates: list[SentenceCandidate] = []
        max_hit_score = max((hit.score for hit in hits), default=1.0)
        for hit in hits:
            normalized_hit_score = hit.score / max_hit_score if max_hit_score > 0 else 0
            sentences = [sentence.strip() for sentence in re.split(r"(?<=[。！？；])", hit.chunk.text) if sentence.strip()]
            for sentence in sentences:
                matched_terms = _matched_query_terms(query_terms, sentence)
                matched_topic_terms = _matched_query_terms(topic_terms, sentence)
                signal = _intent_signal(intent, sentence)
                if not matched_topic_terms:
                    continue
                topic_coverage = len(matched_topic_terms) / max(len(topic_terms), 1)
                query_coverage = len(matched_terms) / max(len(query_terms), 1)
                score = topic_coverage * 0.5 + query_coverage * 0.15 + normalized_hit_score * 0.2
                if signal:
                    score += 0.45
                candidates.append(SentenceCandidate(
                    score=score,
                    sentence=sentence,
                    chunk_id=hit.chunk.id,
                    matched_terms=matched_terms,
                    matched_topic_terms=matched_topic_terms,
                    intent_signal=signal,
                ))
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def _structured_answer(
        self,
        intent: AnswerIntent,
        candidates: list[SentenceCandidate],
        citation_by_chunk: dict[str, str],
    ) -> GenerationResult | None:
        for candidate in candidates:
            if not candidate.intent_signal:
                continue
            values = _parse_list_items(candidate.sentence) if intent == "list" else _parse_process_steps(candidate.sentence)
            if not values:
                continue
            points = [
                AnswerPoint(
                    text=value,
                    citation=citation_by_chunk[candidate.chunk_id],
                    chunk_id=candidate.chunk_id,
                    selection_reason=candidate.intent_signal,
                )
                for value in values
            ]
            return GenerationResult(
                answer=self._format_answer(intent, points),
                citation_chunk_ids=[candidate.chunk_id],
                intent=intent,
                points=points,
            )
        return None

    @staticmethod
    def _select_candidates(candidates: list[SentenceCandidate], topic_terms: set[str]) -> list[SentenceCandidate]:
        selected: list[SentenceCandidate] = []
        selected_texts: list[str] = []
        covered_topic_terms: set[str] = set()
        best_score = candidates[0].score if candidates else 0
        for candidate in candidates:
            normalized = candidate.sentence.strip(" ，。；：")
            duplicate = any(normalized in text or text in normalized for text in selected_texts)
            adds_new_terms = bool(candidate.matched_topic_terms - covered_topic_terms)
            if duplicate:
                continue
            if selected and not adds_new_terms and candidate.score <= best_score * 0.78:
                continue
            selected.append(candidate)
            selected_texts.append(normalized)
            covered_topic_terms.update(candidate.matched_topic_terms)
            if len(selected) == min(4, max(2, len(topic_terms))):
                break
        return selected

    @staticmethod
    def _format_answer(intent: AnswerIntent, points: list[AnswerPoint]) -> str:
        if intent == "process":
            heading = "根据当前知识库，流程如下："
        elif intent == "list":
            heading = f"根据当前知识库，共检索到 {len(points)} 项："
        else:
            heading = "根据当前知识库："
        lines = [f"{index}. {point.text} {point.citation}" for index, point in enumerate(points, start=1)]
        return "\n".join([heading, *lines])

    @staticmethod
    def _generation_detail(
        intent: AnswerIntent,
        query_terms: set[str],
        topic_terms: set[str],
        candidates: list[SentenceCandidate],
        points: list[AnswerPoint],
    ) -> dict[str, Any]:
        return {
            "detected_intent": intent,
            "query_terms": sorted(query_terms),
            "topic_terms": sorted(topic_terms),
            "candidate_sentence_count": len(candidates),
            "selected_points": [
                {
                    "text": point.text,
                    "chunk_id": point.chunk_id,
                    "reason": point.selection_reason,
                }
                for point in points
            ],
        }

    def prompt_preview(self, question: str, context: list[ContextBlock]) -> str:
        context_text = "\n\n".join(
            f"{block.citation} 章节：{block.section}\n{block.text}" for block in context
        )
        return (
            "你是知识库问答助手。只根据给定资料回答；资料不足时明确说明；"
            "回答中的事实必须标注引用编号。先判断问题是枚举、流程还是事实问题，"
            "再用列表或步骤组织答案。\n\n"
            f"资料：\n{context_text}\n\n问题：{question}\n回答："
        )
