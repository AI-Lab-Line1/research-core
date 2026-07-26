from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

from app.rag.llm import LongCatGenerator, LongCatRequestError, LongCatSettings
from app.rag.models import ContextBlock, QueryRequest
from app.rag.pipeline import RagPipeline


SOURCE_PATH = Path(__file__).resolve().parents[2] / "data" / "source" / "知识库.md"


class FakeCompletions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="1. 特别勋章 [2]\n2. 技术勋章 [2]"),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=26, total_tokens=146),
        )


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


class FailingLongCatGenerator:
    settings = SimpleNamespace(model="LongCat-2.0")

    @staticmethod
    def prompt_preview(question, context):
        return f"LongCat prompt: {question} / {len(context)} blocks"

    @staticmethod
    def generate(question, context):
        raise LongCatRequestError("LongCat 请求超时")


class LongCatGeneratorTest(TestCase):
    def test_openai_compatible_response_is_parsed_with_usage_and_citations(self) -> None:
        client = FakeClient()
        generator = LongCatGenerator(
            LongCatSettings(
                api_key="test-only-key",
                base_url="https://example.invalid/v1",
                model="LongCat-2.0",
            ),
            client=client,
        )
        context = [
            ContextBlock(chunk_id="chunk-001", citation="[1]", section="概览", text="概览资料"),
            ContextBlock(chunk_id="chunk-014", citation="[2]", section="勋章机制", text="勋章资料"),
        ]

        result = generator.generate("创新学社有哪些勋章？", context)

        self.assertEqual([point.text for point in result.points], ["特别勋章", "技术勋章"])
        self.assertEqual(result.citation_chunk_ids, ["chunk-014"])
        self.assertEqual(result.metadata["provider"], "LongCat")
        self.assertEqual(result.metadata["total_tokens"], 146)
        self.assertEqual(result.metadata["finish_reason"], "stop")
        self.assertEqual(client.chat.completions.request["model"], "LongCat-2.0")
        self.assertEqual(client.chat.completions.request["temperature"], 0.2)
        self.assertNotIn("test-only-key", generator.prompt_preview("问题", context))

    def test_pipeline_falls_back_to_extractive_when_longcat_fails(self) -> None:
        pipeline = RagPipeline(SOURCE_PATH)
        pipeline.longcat_generator = FailingLongCatGenerator()

        result = pipeline.query(QueryRequest(
            question="创新学社有哪些勋章？",
            generation_method="longcat",
            top_k=4,
        ))

        self.assertTrue(result.generation_metadata.fallback_used)
        self.assertEqual(result.generation_metadata.requested_method, "longcat")
        self.assertEqual(result.generation_metadata.effective_method, "extractive")
        self.assertIn("请求超时", result.generation_warning)
        self.assertEqual(len(result.answer_points), 5)
        self.assertFalse(result.trace[-1].detail["llm_called"])
        self.assertTrue(result.trace[-1].detail["generation_metadata"]["fallback_used"])
