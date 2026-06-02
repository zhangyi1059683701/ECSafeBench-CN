from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark"
CORPUS_ROOT = PROJECT_ROOT / "corpus"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DEFAULT_BENCHMARK_PATH = (
    BENCHMARK_ROOT
    / "releases"
    / "benchmark_v2_internal_ai_reviewed_current961"
    / "benchmark_v2_internal_ai_reviewed_current961.jsonl"
)
EVIDENCE_PATH = CORPUS_ROOT / "curated_evidence_units.jsonl"

CURRENT_TRACK = "current_2026_05_31"
READINESS_TRACK = "readiness_2026_09_30"

EVIDENCE_TASKS = {
    "evidence_fact_verification",
    "evidence_standard_temporal",
    "evidence_incident_reasoning",
    "evidence_sds_verification",
    "evidence_document_conflict",
}

