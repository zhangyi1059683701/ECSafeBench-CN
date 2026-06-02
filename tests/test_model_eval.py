from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from energy_chem_benchmark.model_eval import (
    ApiKeyPool,
    DEFAULT_MODEL_NAMES,
    _chat_completion_payload,
    _dedupe_records_by_job,
    _is_permanent_api_error,
    _job_id,
    select_models,
    run_multi_model_eval,
)


class ModelEvalTest(unittest.TestCase):
    def test_balanced_preset_selects_expected_model_mix(self) -> None:
        models = select_models("preset:balanced")
        names = {item.model for item in models}
        self.assertTrue(set(DEFAULT_MODEL_NAMES) <= names)
        self.assertIn("closed", {item.openness for item in models})
        self.assertIn("open", {item.openness for item in models})
        self.assertIn("thinking", {item.reasoning_style for item in models})
        self.assertIn("instruct", {item.reasoning_style for item in models})

    def test_api_key_pool_round_robins(self) -> None:
        pool = ApiKeyPool(["k1", "k2", "k3", "k4"])
        self.assertEqual([pool.next_key() for _ in range(6)], ["k1", "k2", "k3", "k4", "k1", "k2"])

    def test_qwen_small_non_streaming_payload_disables_thinking(self) -> None:
        payload = _chat_completion_payload("qwen3-8b", "hello")
        self.assertIs(payload["enable_thinking"], False)
        thinking_payload = _chat_completion_payload("qwen3-235b-a22b-thinking-2507", "hello")
        self.assertNotIn("enable_thinking", thinking_payload)

    def test_dedupe_records_prefers_later_successful_retry(self) -> None:
        records = [
            {"job_id": "model::s1", "error": "RemoteDisconnected", "judge": {"correct": False}},
            {"job_id": "model::s1", "judge": {"correct": True}},
            {"job_id": "model::s2", "judge": {"correct": False}},
        ]
        deduped = _dedupe_records_by_job(records)
        self.assertEqual(len(deduped), 2)
        by_job = {item["job_id"]: item for item in deduped}
        self.assertNotIn("error", by_job["model::s1"])
        self.assertTrue(by_job["model::s1"]["judge"]["correct"])

    def test_job_id_is_stable_for_resume(self) -> None:
        self.assertEqual(_job_id("model-a", "sample-1"), "model-a::sample-1")

    def test_sensitive_word_error_is_permanent_api_error(self) -> None:
        self.assertTrue(_is_permanent_api_error("HTTP Error 500: sensitive words detected"))
        self.assertFalse(_is_permanent_api_error("RemoteDisconnected: Remote end closed connection without response"))

    def test_mock_multi_model_eval_writes_analysis_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_multi_model_eval(
                models="Qwen3-8B,Qwen3-8B-thinking",
                workers=2,
                limit=4,
                run_name="unit_mock_eval",
                mock=True,
            )
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "scores.jsonl").exists())
            self.assertTrue((run_dir / "summary_by_model.csv").exists())
            self.assertTrue((run_dir / "thinking_vs_instruct.csv").exists())
            self.assertTrue((run_dir / "report.md").exists())
            self.assertGreater(result["records"], 0)


if __name__ == "__main__":
    unittest.main()
