# ECSafeBench-CN v2: Methodology and Composition

## Release

- Dataset version: `benchmark_v2_internal_ai_reviewed_current961`
- Number of samples: 961
- Base families: 645
- Controlled variants: 316
- Review method: AI-assisted single review
- Human expert verification: `false`

This release is intended for research evaluation of large language models in Chinese energy and chemical process-safety scenarios. The benchmark items themselves are written in Chinese because the target setting is Chinese process-safety documentation and regulation.

## Data Sources

The benchmark was constructed from five classes of materials:

1. accident reports;
2. standards and regulatory clauses;
3. SDS and hazard-information snippets;
4. SOP fragments and work-control materials;
5. intentionally conflicting or scope-mismatched materials.

Evidence-grounded items retain `provided_evidence_ids` so model prompts can include the exact evidence snippets used by the task.

## Domains

The benchmark covers seven domains:

| Domain | Samples | Share |
|---|---:|---:|
| Fine-chemical reaction risk | 147 | 15.3% |
| Maintenance and special operations | 146 | 15.2% |
| Toxic exposure and confined spaces | 147 | 15.3% |
| Coal-chemical high-pressure gas | 142 | 14.8% |
| Petrochemical tanks and pipelines | 144 | 15.0% |
| Urban gas networks | 138 | 14.4% |
| Cross-domain synthesis | 97 | 10.1% |

## Task Types

| Task type | Samples | Purpose |
|---|---:|---|
| `closed_book_dangerous_inducement` | 171 | Test refusal of unsafe operational escalation. |
| `closed_book_professional_boundary` | 127 | Test professional-boundary and authorization awareness. |
| `closed_book_safe_control` | 80 | Test bounded process-safety control knowledge. |
| `evidence_fact_verification` | 154 | Verify whether a fact is supported by supplied evidence. |
| `evidence_incident_reasoning` | 120 | Attribute accident facts, causes, and management findings. |
| `evidence_sds_verification` | 80 | Verify SDS/hazard information and emergency boundaries. |
| `evidence_standard_temporal` | 156 | Track standard scope, version, and effective-date reasoning. |
| `evidence_document_conflict` | 73 | Resolve contradiction, missing support, or scope mismatch. |

## Response Modes and Risk Levels

- Evidence-grounded items: 583 samples, 60.7%
- Closed-book items: 378 samples, 39.3%
- High-risk items: 647 samples, 67.3%
- Medium-risk items: 234 samples, 24.4%
- Low-risk items: 80 samples, 8.3%

## Scoring Fields

Each sample includes:

- `prompt`: user-facing benchmark question;
- `gold_answer`: reference answer;
- `required_claims`: claims a correct answer should cover;
- `forbidden_claims`: unsafe, unsupported, or out-of-scope claims;
- `provided_evidence_ids`: evidence snippets for evidence-grounded tasks;
- `domain`, `task_type`, `risk_level`, `response_mode`, and `temporal_track`.

The evaluation code uses these fields to measure accuracy, overall rubric score, required-claim recall, evidence faithfulness, safety-boundary quality, forbidden-claim violations, over-refusal, and error taxonomy.

