from __future__ import annotations

import re
from pathlib import Path

from .models import SourceDocument


TITLE_PATTERN = re.compile(r"^([^：:\n]{2,24})[：:]")


def split_paragraphs(content: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", content) if paragraph.strip()]


def infer_section(paragraph: str, fallback: str = "学社概览") -> str:
    match = TITLE_PATTERN.match(paragraph)
    return match.group(1).strip() if match else fallback


class MarkdownLoader:
    def __init__(self, source_path: Path):
        self.source_path = source_path

    def load(self) -> SourceDocument:
        content = self.source_path.read_text(encoding="utf-8").strip()
        paragraphs = split_paragraphs(content)
        sections: list[str] = []
        for paragraph in paragraphs:
            section = infer_section(paragraph)
            if section not in sections:
                sections.append(section)
        return SourceDocument(
            name=self.source_path.name,
            path=str(self.source_path),
            content=content,
            character_count=len(content),
            paragraph_count=len(paragraphs),
            section_count=len(sections),
            sections=sections,
        )
