# Data Construction and Accuracy-Evaluation Protocol

## 1. Construction Objective

ECSafeBench-CN evaluates whether a model can reason about Chinese energy and chemical process-safety questions while remaining faithful to evidence and preserving professional safety boundaries. The benchmark is not designed as an operational procedure manual. It is a controlled evaluation suite.

## 2. Item Generation

Generation used bounded source packets rather than free-form model invention. A source packet contained one or more accident-report snippets, standard clauses, SDS/hazard snippets, SOP fragments, or conflict materials. The generation prompt required the model to produce:

- a user-facing prompt;
- a reference answer;
- required claims;
- forbidden claims;
- evidence identifiers for evidence-grounded items;
- domain, task type, response mode, risk level, and temporal-track labels.

The generator was instructed not to add unsupported accident facts, not to introduce operational bypass steps, and not to convert local SOP fragments into universal regulations.

## 3. AI Review and Repair

An independent AI-review pass checked whether each generated item was answerable, evidence-grounded where required, single-target, safe, and scoreable. The review decision was one of:

- `accept`;
- `repair`;
- `delete`.

The first 200 generated items produced 171 accepted items. Of the 29 flagged items, 4 were repaired and retained, while 25 were deleted. The 175 retained seed items were then expanded toward the final release. After additional AI review, filtering, duplicate removal, and malformed-item removal, the final release contains 961 items.

## 4. Model Answering Prompt

All evaluated models receive the same safety-audit style instruction. The model is told to answer the benchmark question, rely strictly on supplied evidence for evidence-grounded tasks, distinguish supported content from unsupported content, and refuse unsafe procedural escalation. The model does not see the reference answer, required claims, forbidden claims, or scoring rubric.

## 5. Judge-Based Scoring

Open-ended process-safety answers cannot be evaluated by exact matching alone. The evaluation therefore uses a structured rubric judge. The judge receives the item metadata, reference answer, required claims, forbidden claims, evidence IDs, and candidate response. It outputs a JSON object with:

- `correct`;
- `overall_score`;
- `required_claim_recall`;
- `forbidden_violation`;
- `evidence_faithfulness`;
- `safety_boundary`;
- `over_refusal`;
- `notes`.

The judge is used as a comparator against the hidden rubric, not as a second answer generator.

## 6. Accuracy and Error Taxonomy

Accuracy is the mean of the binary `correct` field. The mean rubric score is reported separately because partially correct but unsafe or evidence-unfaithful answers should not be treated as fully correct.

Primary error types include:

- forbidden violation;
- incomplete or uncertain reasoning;
- over-refusal;
- low evidence faithfulness;
- low required-claim recall;
- weak safety boundary;
- API or runtime error.

This design separates wrong answers, unsafe answers, over-conservative refusals, and evidence-mismatch failures.

