from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .generator import GenerationResult, detect_intent
from .models import AnswerPoint, ContextBlock


SYSTEM_PROMPT = (
    "你是一个严格依据知识库回答问题的助手。只能使用用户提供的资料，不得补充资料外的事实。"
    "每条事实后必须标注对应的引用编号，例如 [1]。资料不足时直接回答“当前资料不足，无法确定”。"
    "枚举问题使用项目列表，流程问题使用有序步骤，其他问题分点回答。不要输出思考过程。"
)
CITATION_PATTERN = re.compile(r"\[(\d+)]")
POINT_PREFIX_PATTERN = re.compile(r"^(?:[-*•]|\d+[.、)])\s*")


class LongCatConfigurationError(RuntimeError):
    pass


class LongCatRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class LongCatSettings:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 60.0
    max_tokens: int = 900

    @classmethod
    def from_env_file(cls, env_path: Path) -> "LongCatSettings":
        values = dotenv_values(env_path)
        api_key = (values.get("LONGCAT_API_KEY") or "").strip()
        base_url = (values.get("LONGCAT_BASE_URL") or "https://api.longcat.chat/openai/v1").strip().rstrip("/")
        model = (values.get("LONGCAT_MODEL") or "LongCat-2.0").strip()
        if not api_key:
            raise LongCatConfigurationError("未配置 LONGCAT_API_KEY，无法调用 LongCat")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        try:
            timeout_seconds = float(values.get("LONGCAT_TIMEOUT_SECONDS") or 60)
            max_tokens = int(values.get("LONGCAT_MAX_TOKENS") or 900)
        except ValueError as exc:
            raise LongCatConfigurationError("LongCat 超时或 max_tokens 配置格式不正确") from exc
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )


def longcat_configuration_summary(env_path: Path) -> tuple[bool, str | None]:
    values = dotenv_values(env_path)
    configured = bool((values.get("LONGCAT_API_KEY") or "").strip())
    model = (values.get("LONGCAT_MODEL") or "LongCat-2.0").strip() if configured else None
    return configured, model


class LongCatGenerator:
    id = "longcat"

    def __init__(self, settings: LongCatSettings, client: Any | None = None):
        self.settings = settings
        self.client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=1,
        )

    @classmethod
    def from_env_file(cls, env_path: Path) -> "LongCatGenerator":
        return cls(LongCatSettings.from_env_file(env_path))

    def generate(self, question: str, context: list[ContextBlock]) -> GenerationResult:
        messages = self._messages(question, context)
        try:
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                temperature=0.2,
                max_tokens=self.settings.max_tokens,
            )
        except APITimeoutError as exc:
            raise LongCatRequestError("LongCat 请求超时") from exc
        except APIConnectionError as exc:
            raise LongCatRequestError("无法连接 LongCat 服务") from exc
        except APIStatusError as exc:
            raise LongCatRequestError(f"LongCat 服务返回 HTTP {exc.status_code}") from exc
        except Exception as exc:
            raise LongCatRequestError(f"LongCat 调用失败：{type(exc).__name__}") from exc

        choice = response.choices[0] if response.choices else None
        content = (choice.message.content if choice and choice.message else None) or ""
        content = content.strip()
        if not content:
            raise LongCatRequestError("LongCat 返回了空回答")

        intent = detect_intent(question)
        points, citation_chunk_ids = self._parse_points(content, context)
        if not citation_chunk_ids and self._is_insufficient(content):
            intent = "fallback"
        usage = getattr(response, "usage", None)
        finish_reason = getattr(choice, "finish_reason", None) if choice else None
        metadata = {
            "requested_method": "longcat",
            "effective_method": "longcat",
            "provider": "LongCat",
            "model": self.settings.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "finish_reason": finish_reason,
            "fallback_used": False,
        }
        warning = None if citation_chunk_ids else "模型回答没有包含可识别的有效引用编号"
        return GenerationResult(
            answer=content,
            citation_chunk_ids=citation_chunk_ids,
            intent=intent,
            points=points,
            detail={
                "detected_intent": detect_intent(question),
                "selected_points": [
                    {"text": point.text, "chunk_id": point.chunk_id, "reason": point.selection_reason}
                    for point in points
                ],
            },
            metadata=metadata,
            warning=warning,
        )

    def prompt_preview(self, question: str, context: list[ContextBlock]) -> str:
        messages = self._messages(question, context)
        return (
            f"[system]\n{messages[0]['content']}\n\n"
            f"[user]\n{messages[1]['content']}"
        )

    @staticmethod
    def _messages(question: str, context: list[ContextBlock]) -> list[dict[str, str]]:
        context_text = "\n\n".join(
            f"{block.citation} 章节：{block.section}\n{block.text}" for block in context
        )
        user_prompt = f"资料：\n{context_text}\n\n问题：{question}\n\n请直接给出带引用的答案。"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _parse_points(content: str, context: list[ContextBlock]) -> tuple[list[AnswerPoint], list[str]]:
        context_by_number = {str(index): block for index, block in enumerate(context, start=1)}
        citation_numbers = [number for number in CITATION_PATTERN.findall(content) if number in context_by_number]
        unique_numbers = list(dict.fromkeys(citation_numbers))
        points: list[AnswerPoint] = []
        for line in (item.strip() for item in content.splitlines()):
            if not line or line.startswith("#"):
                continue
            line_citations = [number for number in CITATION_PATTERN.findall(line) if number in context_by_number]
            if not line_citations:
                continue
            cleaned = POINT_PREFIX_PATTERN.sub("", line).strip()
            cleaned = CITATION_PATTERN.sub("", cleaned).strip()
            block = context_by_number[line_citations[0]]
            points.append(AnswerPoint(
                text=cleaned,
                citation=f"[{line_citations[0]}]",
                chunk_id=block.chunk_id,
                selection_reason="LongCat 基于检索上下文生成",
            ))
        if not points and unique_numbers:
            block = context_by_number[unique_numbers[0]]
            points.append(AnswerPoint(
                text=content,
                citation=f"[{unique_numbers[0]}]",
                chunk_id=block.chunk_id,
                selection_reason="LongCat 基于检索上下文生成",
            ))
        citation_chunk_ids = [context_by_number[number].chunk_id for number in unique_numbers]
        return points, citation_chunk_ids

    @staticmethod
    def _is_insufficient(content: str) -> bool:
        return any(marker in content for marker in ("资料不足", "无法确定", "无法回答", "没有足够"))
