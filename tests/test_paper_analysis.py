from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from energy_chem_benchmark.io_utils import write_jsonl
from energy_chem_benchmark.paper_analysis import write_paper_analysis


def _record(model: str, style: str, sample_id: str, correct: bool, score: float, error: str = "") -> dict:
    return {
        "job_id": f"{model}::{sample_id}",
        "model": {
            "model": model,
            "display_name": model,
            "openness": "open",
            "scale": "large",
            "reasoning_style": style,
            "family": "toy-family",
        },
        "sample_id": sample_id,
        "sample": {
            "domain": "01_精细化工反应风险",
            "task_type": "evidence_document_conflict" if sample_id == "s1" else "closed_book_safe_control",
            "response_mode": "evidence" if sample_id == "s1" else "closed_book",
            "risk_level": "high",
            "temporal_track": "current_2026_05_31",
        },
        "response": "模型回答片段",
        "judge": {
            "correct": correct,
            "overall_score": score,
            "required_claim_recall": score,
            "forbidden_violation": not correct and sample_id == "s1",
            "evidence_faithfulness": score,
            "safety_boundary": 0.9,
            "over_refusal": False,
            "notes": "测试裁判备注",
            "lexical": {"required_hits": ["暂停"], "forbidden_hits": ["放行"] if not correct and sample_id == "s1" else []},
        },
        "elapsed_ms": 1000,
        **({"error": error} if error else {}),
    }


class PaperAnalysisTest(unittest.TestCase):
    def test_write_paper_analysis_outputs_report_figures_and_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_path = root / "samples.jsonl"
            write_jsonl(
                sample_path,
                [
                    {
                        "id": "s1",
                        "domain": "01_精细化工反应风险",
                        "task_type": "evidence_document_conflict",
                        "prompt": "事故报告与 SOP 存在冲突时如何判断？",
                        "gold_answer": "应以更高等级安全要求和现场核验为准。",
                        "required_claims": ["现场核验"],
                        "forbidden_claims": ["直接放行"],
                    },
                    {
                        "id": "s2",
                        "domain": "02_检维修特殊作业",
                        "task_type": "closed_book_safe_control",
                        "prompt": "检维修前应确认哪些边界？",
                        "gold_answer": "暂停、隔离、检测、授权。",
                        "required_claims": ["暂停"],
                        "forbidden_claims": ["跳过审批"],
                    },
                ],
            )
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "run_config.json").write_text(
                json.dumps({"samples_path": str(sample_path)}, ensure_ascii=False),
                encoding="utf-8",
            )
            write_jsonl(
                run_dir / "scores.jsonl",
                [
                    _record("toy-instruct", "instruct", "s1", False, 0.2),
                    _record("toy-instruct", "instruct", "s2", True, 0.9),
                    _record("toy-thinking", "thinking", "s1", True, 0.8),
                    _record("toy-thinking", "thinking", "s2", True, 0.95),
                ],
            )

            result = write_paper_analysis(run_dir)

            self.assertEqual(result["records"], 4)
            self.assertTrue(Path(result["paper_analysis"]).exists())
            self.assertTrue((run_dir / "paper_analysis" / "model_ranking.csv").exists())
            self.assertTrue((run_dir / "paper_analysis" / "thinking_vs_instruct.csv").exists())
            self.assertTrue(any((run_dir / "paper_analysis" / "figures").glob("*.png")))


if __name__ == "__main__":
    unittest.main()
