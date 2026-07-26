from unittest import TestCase

from app.main import compare_methods, get_methods, get_runtime_status
from app.rag.models import ComparisonConfig, ComparisonRequest


class RagApiTest(TestCase):
    def test_runtime_status_is_safe_and_exposes_capabilities(self) -> None:
        status = get_runtime_status()

        self.assertEqual(status.api_version, "0.3.0")
        self.assertTrue(status.source_ready)
        self.assertIn("longcat", status.available_methods)
        self.assertFalse(hasattr(status, "api_key"))

    def test_methods_expose_local_comparison_options(self) -> None:
        available = {item.id for item in get_methods() if item.status == "available"}
        self.assertTrue({"structure", "fixed_length", "semantic", "tfidf", "bm25", "hybrid", "term_coverage", "longcat"} <= available)

    def test_compare_runs_two_isolated_pipelines(self) -> None:
        response = compare_methods(ComparisonRequest(
            question="图书馆什么时候开放，可以借多少本书？",
            top_k=3,
            configs=[
                ComparisonConfig(label="TF-IDF", retrieval_method="tfidf"),
                ComparisonConfig(
                    label="混合检索",
                    chunking_method="fixed_length",
                    retrieval_method="hybrid",
                    reranking_method="term_coverage",
                    chunk_size=180,
                    chunk_overlap=30,
                ),
            ],
        ))

        self.assertEqual(len(response.runs), 2)
        self.assertTrue(all(run.result.retrieval_hits for run in response.runs))
        self.assertTrue(all("20本书" in run.result.answer for run in response.runs))
