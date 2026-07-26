from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .index import tokenize
from .loader import infer_section, split_paragraphs
from .models import Chunk, SourceDocument


class Chunker(ABC):
    id: str

    @abstractmethod
    def split(self, document: SourceDocument) -> list[Chunk]:
        raise NotImplementedError


class StructureChunker(Chunker):
    """Use paragraph structure and retain inferred section metadata."""

    id = "structure"

    def __init__(self, max_chars: int = 620):
        self.max_chars = max_chars

    def split(self, document: SourceDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        cursor = 0
        current_section = "学社概览"

        for paragraph in split_paragraphs(document.content):
            paragraph_start = document.content.find(paragraph, cursor)
            if paragraph_start < 0:
                paragraph_start = cursor
            current_section = infer_section(paragraph, current_section)

            for part, local_start in self._split_long_paragraph(paragraph):
                start = paragraph_start + local_start
                chunks.append(
                    Chunk(
                        id=f"chunk-{len(chunks) + 1:03d}",
                        order=len(chunks) + 1,
                        section=current_section,
                        text=part,
                        character_count=len(part),
                        start_char=start,
                        end_char=start + len(part),
                    )
                )
            cursor = paragraph_start + len(paragraph)
        return chunks

    def _split_long_paragraph(self, paragraph: str) -> list[tuple[str, int]]:
        if len(paragraph) <= self.max_chars:
            return [(paragraph, 0)]

        sentences = [item for item in re.split(r"(?<=[。！？；])", paragraph) if item]
        parts: list[tuple[str, int]] = []
        buffer = ""
        buffer_start = 0
        consumed = 0
        for sentence in sentences:
            if buffer and len(buffer) + len(sentence) > self.max_chars:
                parts.append((buffer.strip(), buffer_start))
                buffer = ""
                buffer_start = consumed
            if len(sentence) > self.max_chars:
                for index in range(0, len(sentence), self.max_chars):
                    piece = sentence[index : index + self.max_chars].strip()
                    if piece:
                        parts.append((piece, consumed + index))
                consumed += len(sentence)
                continue
            if not buffer:
                buffer_start = consumed
            buffer += sentence
            consumed += len(sentence)
        if buffer.strip():
            parts.append((buffer.strip(), buffer_start))
        return parts


class FixedLengthChunker(Chunker):
    """Split the original character stream with a fixed sliding window."""

    id = "fixed_length"

    def __init__(self, chunk_size: int = 260, overlap: int = 40):
        if overlap >= chunk_size:
            raise ValueError("固定长度切分要求 overlap 小于 chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: SourceDocument) -> list[Chunk]:
        content = document.content
        section_positions = self._section_positions(content)
        chunks: list[Chunk] = []
        raw_start = 0
        previous_end = 0

        while raw_start < len(content):
            raw_end = min(raw_start + self.chunk_size, len(content))
            raw_text = content[raw_start:raw_end]
            left_trim = len(raw_text) - len(raw_text.lstrip())
            right_trimmed = raw_text.rstrip()
            start = raw_start + left_trim
            end = raw_start + len(right_trimmed)
            text = content[start:end]

            if text:
                chunks.append(
                    Chunk(
                        id=f"chunk-{len(chunks) + 1:03d}",
                        order=len(chunks) + 1,
                        section=self._sections_in_range(start, end, section_positions),
                        text=text,
                        character_count=len(text),
                        start_char=start,
                        end_char=end,
                        overlap_chars=max(0, previous_end - start),
                    )
                )
                previous_end = end

            if raw_end == len(content):
                break
            raw_start += self.chunk_size - self.overlap

        return chunks

    @staticmethod
    def _section_positions(content: str) -> list[tuple[int, str]]:
        positions: list[tuple[int, str]] = [(0, "学社概览")]
        cursor = 0
        current_section = "学社概览"
        for paragraph in split_paragraphs(content):
            start = content.find(paragraph, cursor)
            if start < 0:
                start = cursor
            section = infer_section(paragraph, current_section)
            if section != current_section:
                positions.append((start, section))
                current_section = section
            cursor = start + len(paragraph)
        return positions

    @staticmethod
    def _section_at(position: int, positions: list[tuple[int, str]]) -> str:
        section = positions[0][1]
        for start, candidate in positions:
            if start > position:
                break
            section = candidate
        return section

    @classmethod
    def _sections_in_range(cls, start: int, end: int, positions: list[tuple[int, str]]) -> str:
        sections = [cls._section_at(start, positions)]
        for position, section in positions:
            if start < position < end and section not in sections:
                sections.append(section)
        return " / ".join(sections)


@dataclass(frozen=True)
class SemanticUnit:
    text: str
    start: int
    end: int
    section: str


class SemanticChunker(Chunker):
    """Group adjacent paragraphs while their local TF-IDF semantics remain cohesive."""

    id = "semantic"

    def __init__(self, similarity_threshold: float = 0.05, max_chars: int = 620):
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("语义相似度阈值必须在 0 到 1 之间")
        if max_chars < 160:
            raise ValueError("语义切分最大长度不能小于 160")
        self.similarity_threshold = similarity_threshold
        self.max_chars = max_chars

    def split(self, document: SourceDocument) -> list[Chunk]:
        units = self._build_units(document)
        if not units:
            return []

        similarities = self._adjacent_similarities(units)
        groups: list[tuple[list[SemanticUnit], str, float | None]] = []
        current = [units[0]]
        start_reason = "document_start"
        boundary_similarity: float | None = None

        for index, unit in enumerate(units[1:], start=1):
            similarity = similarities[index - 1]
            projected_chars = unit.end - current[0].start
            if projected_chars > self.max_chars or similarity < self.similarity_threshold:
                groups.append((current, start_reason, boundary_similarity))
                current = [unit]
                start_reason = "max_chars" if projected_chars > self.max_chars else "semantic_drop"
                boundary_similarity = similarity
            else:
                current.append(unit)
        groups.append((current, start_reason, boundary_similarity))

        chunks: list[Chunk] = []
        for group, split_reason, similarity in groups:
            start = group[0].start
            end = group[-1].end
            sections = list(dict.fromkeys(unit.section for unit in group))
            text = document.content[start:end].strip()
            chunks.append(Chunk(
                id=f"chunk-{len(chunks) + 1:03d}",
                order=len(chunks) + 1,
                section=" / ".join(sections),
                text=text,
                character_count=len(text),
                start_char=start,
                end_char=end,
                boundary_similarity=round(similarity, 4) if similarity is not None else None,
                split_reason=split_reason,
                semantic_unit_count=len(group),
            ))
        return chunks

    def _build_units(self, document: SourceDocument) -> list[SemanticUnit]:
        units: list[SemanticUnit] = []
        cursor = 0
        current_section = "学社概览"
        long_paragraph_splitter = StructureChunker(max_chars=self.max_chars)

        for paragraph in split_paragraphs(document.content):
            paragraph_start = document.content.find(paragraph, cursor)
            if paragraph_start < 0:
                paragraph_start = cursor
            current_section = infer_section(paragraph, current_section)
            for part, local_start in long_paragraph_splitter._split_long_paragraph(paragraph):
                start = paragraph_start + local_start
                units.append(SemanticUnit(
                    text=part,
                    start=start,
                    end=start + len(part),
                    section=current_section,
                ))
            cursor = paragraph_start + len(paragraph)
        return units

    @staticmethod
    def _adjacent_similarities(units: list[SemanticUnit]) -> list[float]:
        if len(units) < 2:
            return []
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        try:
            matrix = vectorizer.fit_transform([unit.text for unit in units])
        except ValueError:
            return [0.0] * (len(units) - 1)
        return [
            float(linear_kernel(matrix[index], matrix[index + 1])[0, 0])
            for index in range(len(units) - 1)
        ]


def get_chunker(
    method: str,
    chunk_size: int = 260,
    overlap: int = 40,
    semantic_threshold: float = 0.05,
    semantic_max_chars: int = 620,
) -> Chunker:
    if method == "structure":
        return StructureChunker()
    if method == "fixed_length":
        if overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return FixedLengthChunker(chunk_size=chunk_size, overlap=overlap)
    if method == "semantic":
        return SemanticChunker(
            similarity_threshold=semantic_threshold,
            max_chars=semantic_max_chars,
        )
    raise ValueError(f"切分方法 {method!r} 尚未实现")
