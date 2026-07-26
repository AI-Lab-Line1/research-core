from pathlib import Path
from unittest import TestCase

from app.rag.models import QueryRequest
from app.rag.pipeline import RagPipeline


SOURCE_PATH = Path(__file__).resolve().parents[2] / "data" / "source" / "知识库.md"


class RagPipelineTest(TestCase):
    def setUp(self) -> None:
        self.pipeline = RagPipeline(SOURCE_PATH)

    def test_build_index_exposes_chunks_and_vector_metadata(self) -> None:
        result = self.pipeline.build_index()
        self.assertTrue(result.status.ready)
        self.assertGreater(result.status.chunk_count, 20)
        self.assertGreater(result.status.vector_dimension, 0)
        self.assertEqual([step.id for step in result.trace], ["load", "chunk", "index"])
        self.assertIsNotNone(result.chunks[0].vector)

    def test_query_returns_evidence_and_complete_trace(self) -> None:
        result = self.pipeline.query(QueryRequest(question="创新学社有哪些勋章？", top_k=4))
        self.assertIn("勋章", result.answer)
        self.assertEqual(result.answer_intent, "list")
        self.assertEqual(
            [point.text for point in result.answer_points],
            ["特别勋章", "技术勋章", "宣传勋章", "组织勋章", "学社勋章"],
        )
        self.assertTrue(all(point.selection_reason == "检测到枚举结构" for point in result.answer_points))
        self.assertEqual(len(result.retrieval_hits), 4)
        self.assertTrue(all(hit.score > 0 for hit in result.retrieval_hits))
        self.assertTrue(result.context)
        self.assertIn("资料：", result.prompt_preview)
        self.assertEqual(
            [step.id for step in result.trace],
            ["load", "chunk", "index", "retrieve", "rerank", "context", "generate"],
        )
        self.assertEqual(result.trace[4].status, "skipped")
        self.assertEqual(result.trace[-1].detail["detected_intent"], "list")
        self.assertEqual(len(result.trace[-1].detail["selected_points"]), 5)

    def test_process_question_is_split_into_ordered_steps(self) -> None:
        result = self.pipeline.query(QueryRequest(question="加入学社需要经过哪些流程？", top_k=4))

        self.assertEqual(result.answer_intent, "process")
        self.assertEqual(
            [point.text for point in result.answer_points],
            ["扫码加入面试群", "填写报名表", "完成一个月考核期", "成为正式成员，开启“变强之旅”"],
        )
        self.assertNotIn("勋章分为", result.answer)
        self.assertEqual(len({point.chunk_id for point in result.answer_points}), 1)

    def test_unrelated_question_uses_no_answer_fallback(self) -> None:
        result = self.pipeline.query(QueryRequest(question="火星基地的空气循环系统是什么？", top_k=3))
        self.assertIn("没有检索到", result.answer)
        self.assertEqual(result.answer_intent, "fallback")
        self.assertEqual(result.answer_points, [])
        self.assertEqual(result.citations, [])

    def test_zero_score_chunks_are_not_added_to_context(self) -> None:
        result = self.pipeline.query(QueryRequest(question="图书馆什么时候开放，可以借多少本书？", top_k=4))

        self.assertEqual(len(result.retrieval_hits), 1)
        self.assertEqual(len(result.context), 1)
        self.assertEqual(result.context[0].section, "图书馆与资源利用")
        self.assertEqual(result.answer_intent, "fact")
        self.assertEqual(len(result.answer_points), 2)
        self.assertIn("20本书", result.answer)
        self.assertNotIn("电子资源", result.answer)

    def test_fixed_length_chunking_records_overlap(self) -> None:
        result = self.pipeline.build_index(
            chunking_method="fixed_length",
            retrieval_method="tfidf",
            chunk_size=180,
            chunk_overlap=30,
        )

        self.assertTrue(result.status.ready)
        self.assertEqual(result.status.chunking_method, "fixed_length")
        self.assertTrue(any(chunk.overlap_chars > 0 for chunk in result.chunks[1:]))
        self.assertTrue(all(chunk.character_count <= 180 for chunk in result.chunks))
        self.assertTrue(any(" / " in chunk.section for chunk in result.chunks))
        self.assertEqual(result.trace[1].detail["chunk_size"], 180)

        answer = self.pipeline.query(QueryRequest(
            question="图书馆什么时候开放，可以借多少本书？",
            chunking_method="fixed_length",
            retrieval_method="hybrid",
            reranking_method="term_coverage",
            chunk_size=180,
            chunk_overlap=30,
        ))
        self.assertIn("20本书", answer.answer)
        self.assertNotIn("电子资源", answer.answer)

    def test_semantic_chunking_exposes_boundary_scores_and_responds_to_threshold(self) -> None:
        coarse = self.pipeline.build_index(
            chunking_method="semantic",
            retrieval_method="tfidf",
            semantic_threshold=0.0,
            semantic_max_chars=1000,
        )
        fine = self.pipeline.build_index(
            chunking_method="semantic",
            retrieval_method="tfidf",
            semantic_threshold=0.05,
            semantic_max_chars=620,
        )

        self.assertTrue(fine.status.ready)
        self.assertEqual(fine.status.chunking_method, "semantic")
        self.assertLess(len(coarse.chunks), len(fine.chunks))
        self.assertTrue(any(chunk.semantic_unit_count > 1 for chunk in fine.chunks))
        self.assertTrue(any(chunk.split_reason == "semantic_drop" for chunk in fine.chunks))
        self.assertTrue(any(chunk.split_reason == "max_chars" for chunk in fine.chunks))
        self.assertTrue(all(chunk.character_count <= 620 for chunk in fine.chunks))
        self.assertTrue(all(
            chunk.boundary_similarity is not None
            for chunk in fine.chunks
            if chunk.split_reason != "document_start"
        ))
        self.assertEqual(fine.trace[1].detail["semantic_threshold"], 0.05)
        self.assertGreater(fine.trace[1].detail["semantic_boundaries"], 0)

        answer = self.pipeline.query(QueryRequest(
            question="创新学社有哪些勋章？",
            chunking_method="semantic",
            retrieval_method="hybrid",
            semantic_threshold=0.05,
            semantic_max_chars=620,
        ))
        self.assertEqual(answer.answer_intent, "list")
        self.assertIn("特别勋章", answer.answer)

    def test_all_local_retrieval_methods_return_explainable_scores(self) -> None:
        for method in ("tfidf", "bm25", "hybrid"):
            with self.subTest(method=method):
                pipeline = RagPipeline(SOURCE_PATH)
                result = pipeline.query(QueryRequest(
                    question="创新学社有哪些勋章？",
                    retrieval_method=method,
                    top_k=3,
                ))
                self.assertTrue(result.retrieval_hits)
                self.assertIn("勋章", result.answer)
                self.assertEqual(len(result.answer_points), 5)
                self.assertTrue(result.retrieval_hits[0].score_components)
                self.assertEqual(result.index_status.retrieval_method, method)

    def test_rule_reranker_exposes_before_and_after_ranking(self) -> None:
        result = self.pipeline.query(QueryRequest(
            question="图书馆开放时间和借书规则",
            retrieval_method="hybrid",
            reranking_method="term_coverage",
            top_k=2,
        ))

        rerank_step = next(step for step in result.trace if step.id == "rerank")
        self.assertEqual(rerank_step.status, "completed")
        self.assertGreaterEqual(len(rerank_step.detail["before"]), len(rerank_step.detail["after"]))
        self.assertTrue(all(hit.rerank_score is not None for hit in result.retrieval_hits))
        self.assertTrue(all(hit.selected_for_context for hit in result.retrieval_hits))

    def test_invalid_fixed_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.pipeline.build_index(
                chunking_method="fixed_length",
                retrieval_method="tfidf",
                chunk_size=100,
                chunk_overlap=100,
            )
