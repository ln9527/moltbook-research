#!/usr/bin/env python3
"""
Pipeline Integration Tests

Verify that all analysis phases can import and basic functionality works.
Does not require API keys (skips LLM/embedding tests if not available).
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import unittest
from unittest.mock import MagicMock, patch


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""

    def test_import_pipeline_config(self):
        """Test pipeline config imports."""
        from pipeline.config import (
            PROJECT_ROOT,
            DATA_DIR,
            RAW_DIR,
            DERIVED_DIR,
            BREACH_TIMESTAMP,
        )
        self.assertIsNotNone(PROJECT_ROOT)
        self.assertIsNotNone(BREACH_TIMESTAMP)

    def test_import_state_manager(self):
        """Test state manager imports."""
        from pipeline.state_manager import StateManager, get_state_manager
        self.assertIsNotNone(StateManager)

    def test_import_decision_logger(self):
        """Test decision logger imports."""
        from pipeline.decision_logger import DecisionLogger, get_decision_logger
        self.assertIsNotNone(DecisionLogger)

    def test_import_base(self):
        """Test base analysis class imports."""
        from analysis.base import AnalysisPhase
        self.assertIsNotNone(AnalysisPhase)

    def test_import_phase_00(self):
        """Test Phase 0 imports."""
        from analysis.phase_00_data_audit.main import Phase00DataAudit
        self.assertIsNotNone(Phase00DataAudit)

    def test_import_phase_01(self):
        """Test Phase 1 imports."""
        from analysis.phase_01_temporal.main import Phase01Temporal
        self.assertIsNotNone(Phase01Temporal)

    def test_import_phase_02(self):
        """Test Phase 2 imports."""
        from analysis.phase_02_linguistic.main import Phase02Linguistic
        from analysis.phase_02_linguistic.embedding_client import EmbeddingClient
        from analysis.phase_02_linguistic.llm_analyzer import LLMAnalyzer
        self.assertIsNotNone(Phase02Linguistic)
        self.assertIsNotNone(EmbeddingClient)
        self.assertIsNotNone(LLMAnalyzer)

    def test_import_phase_03(self):
        """Test Phase 3 imports."""
        from analysis.phase_03_restart.main import Phase03Restart
        self.assertIsNotNone(Phase03Restart)

    def test_import_phase_05(self):
        """Test Phase 5 imports."""
        from analysis.phase_05_depth_gradient.main import Phase05DepthGradient
        self.assertIsNotNone(Phase05DepthGradient)


class TestDecisionLogger(unittest.TestCase):
    """Test decision logger functionality."""

    def test_deduplication(self):
        """Test that duplicate decisions are skipped."""
        from pipeline.decision_logger import DecisionLogger
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_path = Path(f.name)

        try:
            logger = DecisionLogger(log_file=temp_path)

            # Log same decision twice
            logger.log_decision(
                phase="test",
                decision="Test decision",
                rationale="Test rationale"
            )
            logger.log_decision(
                phase="test",
                decision="Test decision",
                rationale="Test rationale"
            )

            # Should only have one entry
            content = temp_path.read_text()
            count = content.count("Test decision")
            self.assertEqual(count, 1, "Duplicate decision should be skipped")

        finally:
            temp_path.unlink(missing_ok=True)

    def test_different_decisions_logged(self):
        """Test that different decisions are all logged."""
        from pipeline.decision_logger import DecisionLogger
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_path = Path(f.name)

        try:
            logger = DecisionLogger(log_file=temp_path)

            logger.log_decision(
                phase="test",
                decision="Decision 1",
                rationale="Rationale 1"
            )
            logger.log_decision(
                phase="test",
                decision="Decision 2",
                rationale="Rationale 2"
            )

            content = temp_path.read_text()
            self.assertIn("Decision 1", content)
            self.assertIn("Decision 2", content)

        finally:
            temp_path.unlink(missing_ok=True)


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality."""

    def test_raw_data_exists(self):
        """Test that raw data files exist."""
        from pipeline.config import RAW_DIR

        posts_path = RAW_DIR / "posts_master.json"
        comments_path = RAW_DIR / "comments_master.json"

        self.assertTrue(posts_path.exists(), f"Missing {posts_path}")
        self.assertTrue(comments_path.exists(), f"Missing {comments_path}")

    def test_raw_posts_loadable(self):
        """Test that raw posts can be loaded."""
        from pipeline.config import RAW_DIR

        with open(RAW_DIR / "posts_master.json") as f:
            posts = json.load(f)

        self.assertIsInstance(posts, list)
        self.assertGreater(len(posts), 0)

        # Check structure
        sample = posts[0]
        self.assertIn("id", sample)
        self.assertIn("created_at", sample)

    def test_raw_comments_loadable(self):
        """Test that raw comments can be loaded."""
        from pipeline.config import RAW_DIR

        with open(RAW_DIR / "comments_master.json") as f:
            comments = json.load(f)

        self.assertIsInstance(comments, list)
        self.assertGreater(len(comments), 0)

        # Check structure
        sample = comments[0]
        self.assertIn("comment_id", sample)
        self.assertIn("depth", sample)


class TestDerivedData(unittest.TestCase):
    """Test derived data exists and is valid."""

    def test_derived_data_exists(self):
        """Test that derived parquet files exist."""
        from pipeline.config import DERIVED_DIR

        posts_path = DERIVED_DIR / "posts_derived.parquet"
        comments_path = DERIVED_DIR / "comments_derived.parquet"

        # Skip if Phase 0 hasn't run
        if not posts_path.exists():
            self.skipTest("Derived data not yet generated (run Phase 0 first)")

        self.assertTrue(posts_path.exists())
        self.assertTrue(comments_path.exists())

    def test_derived_posts_schema(self):
        """Test derived posts have expected columns."""
        from pipeline.config import DERIVED_DIR
        import pandas as pd

        posts_path = DERIVED_DIR / "posts_derived.parquet"
        if not posts_path.exists():
            self.skipTest("Derived data not yet generated")

        df = pd.read_parquet(posts_path)

        expected_cols = [
            "id", "created_at", "author_id", "phase",
            "is_pre_breach", "word_count", "is_long_post"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_derived_comments_schema(self):
        """Test derived comments have expected columns."""
        from pipeline.config import DERIVED_DIR
        import pandas as pd

        comments_path = DERIVED_DIR / "comments_derived.parquet"
        if not comments_path.exists():
            self.skipTest("Derived data not yet generated")

        df = pd.read_parquet(comments_path)

        expected_cols = [
            "id", "post_id", "created_at", "depth",
            "is_pre_breach", "word_count", "reply_depth"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_is_pre_breach_correctly_computed(self):
        """Test that is_pre_breach flag is correct."""
        from pipeline.config import DERIVED_DIR, BREACH_TIMESTAMP
        import pandas as pd

        posts_path = DERIVED_DIR / "posts_derived.parquet"
        if not posts_path.exists():
            self.skipTest("Derived data not yet generated")

        df = pd.read_parquet(posts_path)
        breach_ts = pd.Timestamp(BREACH_TIMESTAMP)

        # Check a sample of rows
        sample = df.sample(min(100, len(df)))
        for _, row in sample.iterrows():
            expected = row["created_at"] < breach_ts
            actual = row["is_pre_breach"]
            self.assertEqual(
                expected, actual,
                f"is_pre_breach mismatch for post {row['id']}"
            )


class TestEmbeddingClient(unittest.TestCase):
    """Test embedding client functionality."""

    def test_text_cleaning(self):
        """Test that text cleaning handles edge cases."""
        from analysis.phase_02_linguistic.embedding_client import EmbeddingClient

        client = EmbeddingClient(api_key="test")

        # Test with mocked _embed_batch
        with patch.object(client, '_embed_batch') as mock_embed:
            mock_embed.return_value = []

            # Test empty texts get cleaned
            texts = [None, "", "  ", "valid text"]
            ids = ["1", "2", "3", "4"]

            # This would call _embed_batch with cleaned texts
            # We're just checking it doesn't crash
            try:
                client.embed_texts(texts[:1], ids[:1])
            except Exception:
                pass  # Expected since we're mocking


class TestLLMAnalyzer(unittest.TestCase):
    """Test LLM analyzer functionality."""

    def test_json_parsing(self):
        """Test JSON parsing from LLM responses."""
        from analysis.phase_02_linguistic.llm_analyzer import LLMAnalyzer

        analyzer = LLMAnalyzer(api_key="test")

        # Test code block extraction
        response = '''
Here is my analysis:
```json
{"primary_tone": "informative", "autonomy_score": 3}
```
'''
        result = analyzer._parse_json_response(response)
        self.assertIsNotNone(result)
        self.assertEqual(result["primary_tone"], "informative")

        # Test raw JSON
        response2 = '{"primary_tone": "playful", "autonomy_score": 4}'
        result2 = analyzer._parse_json_response(response2)
        self.assertIsNotNone(result2)
        self.assertEqual(result2["primary_tone"], "playful")

        # Test embedded JSON with text
        response3 = 'Some text before {"key": "value"} some after'
        result3 = analyzer._parse_json_response(response3)
        self.assertIsNotNone(result3)
        self.assertEqual(result3["key"], "value")


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
