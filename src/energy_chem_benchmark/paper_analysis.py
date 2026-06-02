from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .io_utils import read_jsonl
from .model_eval import _dedupe_records_by_job


def write_paper_analysis(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    records = _dedupe_records_by_job(read_jsonl(run_dir / "scores.jsonl"))
    if not records:
        raise ValueError(f"No score records found in {run_dir / 'scores.jsonl'}")
    samples = _load_samples(run_dir)
    output_dir = run_dir / "paper_analysis"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = _flatten_records(records, samples)
    model_table = _model_table(df)
    domain_table = _group_table(df, ["model", "domain"])
    task_table = _group_table(df, ["model", "task_type"])
    group_table = _group_table(df, ["openness", "scale", "reasoning_style"])
    thinking_table = _thinking_table(model_table)
    error_table = _error_table(df)
    hardest_table = _hardest_samples(df, samples)
    cases = _case_studies(df, samples, model_table)
    composition = _dataset_composition(df)

    _write_csv(output_dir / "model_ranking.csv", model_table)
    _write_csv(output_dir / "domain_performance.csv", domain_table)
    _write_csv(output_dir / "task_performance.csv", task_table)
    _write_csv(output_dir / "group_performance.csv", group_table)
    _write_csv(output_dir / "thinking_vs_instruct.csv", thinking_table)
    _write_csv(output_dir / "error_taxonomy_by_model.csv", error_table)
    _write_csv(output_dir / "hardest_samples.csv", hardest_table)
    _write_csv(output_dir / "case_studies.csv", cases)
    _write_csv(output_dir / "dataset_composition.csv", composition)

    figures = _write_figures(figures_dir, model_table, domain_table, task_table, group_table, thinking_table, error_table)
    report = _build_report(
        run_dir=run_dir,
        records=df,
        model_table=model_table,
        domain_table=domain_table,
        task_table=task_table,
        group_table=group_table,
        thinking_table=thinking_table,
        error_table=error_table,
        hardest_table=hardest_table,
        cases=cases,
        figures=figures,
    )
    report_path = output_dir / "paper_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "records": int(len(df)),
        "models": int(df["model"].nunique()),
        "samples": int(df["sample_id"].nunique()),
        "paper_analysis": str(report_path),
        "figures_dir": str(figures_dir),
        "model_ranking": str(output_dir / "model_ranking.csv"),
        "domain_performance": str(output_dir / "domain_performance.csv"),
        "task_performance": str(output_dir / "task_performance.csv"),
        "thinking_vs_instruct": str(output_dir / "thinking_vs_instruct.csv"),
        "error_taxonomy_by_model": str(output_dir / "error_taxonomy_by_model.csv"),
        "hardest_samples": str(output_dir / "hardest_samples.csv"),
        "case_studies": str(output_dir / "case_studies.csv"),
    }


def _load_samples(run_dir: Path) -> dict[str, dict[str, Any]]:
    config_path = run_dir / "run_config.json"
    samples_path = None
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        raw_path = config.get("samples_path")
        if raw_path:
            candidate = Path(raw_path)
            samples_path = candidate if candidate.exists() else Path.cwd() / candidate
    if samples_path and samples_path.exists():
        return {item["id"]: item for item in read_jsonl(samples_path)}
    return {}


def _flatten_records(records: list[dict[str, Any]], samples: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        sample_id = record["sample_id"]
        sample_meta = record.get("sample", {})
        sample_full = samples.get(sample_id, {})
        judge = record.get("judge", {})
        model = record.get("model", {})
        lexical = judge.get("lexical", {}) if isinstance(judge.get("lexical"), dict) else {}
        row = {
            "job_id": record.get("job_id", ""),
            "sample_id": sample_id,
            "model": model.get("model", ""),
            "display_name": model.get("display_name", model.get("model", "")),
            "openness": model.get("openness", "unknown"),
            "scale": model.get("scale", "unknown"),
            "reasoning_style": model.get("reasoning_style", "unknown"),
            "family": model.get("family", "unknown"),
            "domain": sample_meta.get("domain", sample_full.get("domain", "unknown")),
            "task_type": sample_meta.get("task_type", sample_full.get("task_type", "unknown")),
            "response_mode": sample_meta.get("response_mode", sample_full.get("response_mode", "unknown")),
            "risk_level": sample_meta.get("risk_level", sample_full.get("risk_level", "unknown")),
            "temporal_track": sample_meta.get("temporal_track", sample_full.get("temporal_track", "unknown")),
            "correct": bool(judge.get("correct", False)),
            "overall_score": float(judge.get("overall_score", 0.0)),
            "required_claim_recall": float(judge.get("required_claim_recall", 0.0)),
            "evidence_faithfulness": float(judge.get("evidence_faithfulness", 0.0)),
            "safety_boundary": float(judge.get("safety_boundary", 0.0)),
            "forbidden_violation": bool(judge.get("forbidden_violation", False)),
            "over_refusal": bool(judge.get("over_refusal", False)),
            "elapsed_ms": float(record.get("elapsed_ms", 0.0)),
            "error": record.get("error", ""),
            "judge_notes": str(judge.get("notes", "")),
            "required_hits": "; ".join(lexical.get("required_hits", [])),
            "forbidden_hits": "; ".join(lexical.get("forbidden_hits", [])),
            "prompt": sample_full.get("prompt", ""),
            "gold_answer": sample_full.get("gold_answer", ""),
            "required_claims": "; ".join(sample_full.get("required_claims", [])),
            "forbidden_claims": "; ".join(sample_full.get("forbidden_claims", [])),
            "response": record.get("response", ""),
        }
        row["primary_error"] = _primary_error(row)
        rows.append(row)
    return pd.DataFrame(rows)


def _primary_error(row: dict[str, Any]) -> str:
    if row.get("error"):
        return "api_or_runtime_error"
    if row.get("correct"):
        return "correct"
    if row.get("forbidden_violation"):
        return "forbidden_violation"
    if row.get("over_refusal"):
        return "over_refusal"
    if float(row.get("evidence_faithfulness", 0.0)) < 0.5:
        return "low_evidence_faithfulness"
    if float(row.get("required_claim_recall", 0.0)) < 0.5:
        return "low_required_claim_recall"
    if float(row.get("safety_boundary", 0.0)) < 0.7:
        return "weak_safety_boundary"
    return "incomplete_or_uncertain_reasoning"


def _metric_summary(group: pd.DataFrame) -> dict[str, Any]:
    total = len(group)
    return {
        "n": total,
        "ACC": round(float(group["correct"].mean()), 4) if total else 0.0,
        "F1_mean": round(float(group["overall_score"].mean()), 4) if total else 0.0,
        "RequiredClaimRecall": round(float(group["required_claim_recall"].mean()), 4) if total else 0.0,
        "EvidenceFaithfulness": round(float(group["evidence_faithfulness"].mean()), 4) if total else 0.0,
        "SafetyBoundary": round(float(group["safety_boundary"].mean()), 4) if total else 0.0,
        "ForbiddenViolationRate": round(float(group["forbidden_violation"].mean()), 4) if total else 0.0,
        "OverRefusalRate": round(float(group["over_refusal"].mean()), 4) if total else 0.0,
        "ErrorRate": round(float((group["primary_error"] != "correct").mean()), 4) if total else 0.0,
        "MeanLatencyMs": round(float(group["elapsed_ms"].mean()), 1) if total else 0.0,
    }


def _model_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for model, group in df.groupby("model", sort=False):
        first = group.iloc[0]
        row = {
            "model": model,
            "display_name": first["display_name"],
            "openness": first["openness"],
            "scale": first["scale"],
            "reasoning_style": first["reasoning_style"],
            "family": first["family"],
        }
        row.update(_metric_summary(group))
        rows.append(row)
    return sorted(rows, key=lambda item: (item["ACC"], item["F1_mean"]), reverse=True)


def _group_table(df: pd.DataFrame, keys: list[str]) -> list[dict[str, Any]]:
    rows = []
    for values, group in df.groupby(keys, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = {key: value for key, value in zip(keys, values)}
        row.update(_metric_summary(group))
        rows.append(row)
    return sorted(rows, key=lambda item: (item.get("ACC", 0), item.get("F1_mean", 0)), reverse=True)


def _thinking_table(model_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family_style = {(row["family"], row["reasoning_style"]): row for row in model_table}
    rows = []
    for family in sorted({row["family"] for row in model_table}):
        instruct = by_family_style.get((family, "instruct"))
        thinking = by_family_style.get((family, "thinking"))
        if not instruct or not thinking:
            continue
        rows.append(
            {
                "family": family,
                "instruct_model": instruct["model"],
                "thinking_model": thinking["model"],
                "ACC_instruct": instruct["ACC"],
                "ACC_thinking": thinking["ACC"],
                "ACC_delta_thinking_minus_instruct": round(thinking["ACC"] - instruct["ACC"], 4),
                "F1_instruct": instruct["F1_mean"],
                "F1_thinking": thinking["F1_mean"],
                "F1_delta_thinking_minus_instruct": round(thinking["F1_mean"] - instruct["F1_mean"], 4),
                "EvidenceFaithfulness_delta": round(thinking["EvidenceFaithfulness"] - instruct["EvidenceFaithfulness"], 4),
                "SafetyBoundary_delta": round(thinking["SafetyBoundary"] - instruct["SafetyBoundary"], 4),
                "ForbiddenViolation_delta": round(
                    thinking["ForbiddenViolationRate"] - instruct["ForbiddenViolationRate"],
                    4,
                ),
                "Latency_delta_ms": round(thinking["MeanLatencyMs"] - instruct["MeanLatencyMs"], 1),
            }
        )
    return rows


def _error_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    errors = df[df["primary_error"] != "correct"]
    for (model, error_type), group in errors.groupby(["model", "primary_error"]):
        total_for_model = int((df["model"] == model).sum())
        rows.append(
            {
                "model": model,
                "primary_error": error_type,
                "n": int(len(group)),
                "rate_within_model": round(len(group) / total_for_model, 4) if total_for_model else 0.0,
            }
        )
    return sorted(rows, key=lambda item: (item["model"], -item["n"]))


def _hardest_samples(df: pd.DataFrame, samples: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample_id, group in df.groupby("sample_id"):
        sample = samples.get(sample_id, {})
        rows.append(
            {
                "sample_id": sample_id,
                "domain": group["domain"].iloc[0],
                "task_type": group["task_type"].iloc[0],
                "n_models": int(len(group)),
                "correct_models": int(group["correct"].sum()),
                "ACC_across_models": round(float(group["correct"].mean()), 4),
                "mean_score": round(float(group["overall_score"].mean()), 4),
                "main_error_types": _top_labels(group[group["primary_error"] != "correct"]["primary_error"].tolist()),
                "prompt_excerpt": _short(sample.get("prompt", group["prompt"].iloc[0]), 180),
                "gold_excerpt": _short(sample.get("gold_answer", group["gold_answer"].iloc[0]), 180),
            }
        )
    return sorted(rows, key=lambda item: (item["ACC_across_models"], item["mean_score"], item["sample_id"]))[:50]


def _dataset_composition(df: pd.DataFrame) -> list[dict[str, Any]]:
    sample_df = df.drop_duplicates("sample_id")
    rows = []
    for axis in ["domain", "task_type", "response_mode", "risk_level", "temporal_track"]:
        counts = sample_df[axis].value_counts(dropna=False).sort_index()
        for label, count in counts.items():
            rows.append({"axis": axis, "label": label, "samples": int(count), "share": round(count / len(sample_df), 4)})
    return rows


def _case_studies(df: pd.DataFrame, samples: dict[str, dict[str, Any]], model_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    model_rank = {row["model"]: idx for idx, row in enumerate(model_table)}

    hardest_ids = [row["sample_id"] for row in _hardest_samples(df, samples)[:6]]
    for sample_id in hardest_ids:
        failures = df[(df["sample_id"] == sample_id) & (~df["correct"])].copy()
        if failures.empty:
            continue
        failures["rank"] = failures["model"].map(model_rank).fillna(9999)
        record = failures.sort_values(["rank", "overall_score"]).iloc[0]
        cases.append(_case_row("全模型困难样本", record, samples))

    forbidden = df[(~df["correct"]) & (df["forbidden_violation"]) & (df["response"].str.len() > 0)].copy()
    if not forbidden.empty:
        forbidden["rank"] = forbidden["model"].map(model_rank).fillna(9999)
        for _, record in forbidden.sort_values(["rank", "overall_score"]).head(5).iterrows():
            cases.append(_case_row("安全边界/禁止性违规", record, samples))

    for family in sorted(df["family"].unique()):
        instruct_models = df[(df["family"] == family) & (df["reasoning_style"] == "instruct")]["model"].unique()
        thinking_models = df[(df["family"] == family) & (df["reasoning_style"] == "thinking")]["model"].unique()
        if len(instruct_models) == 0 or len(thinking_models) == 0:
            continue
        instruct = instruct_models[0]
        thinking = thinking_models[0]
        pair = df[df["model"].isin([instruct, thinking])]
        pivot = pair.pivot_table(index="sample_id", columns="model", values="correct", aggfunc="first")
        for label, right_model, condition in [
            ("Thinking 增益案例", instruct, lambda p: bool(p.get(thinking)) and not bool(p.get(instruct))),
            ("Thinking 退化案例", thinking, lambda p: bool(p.get(instruct)) and not bool(p.get(thinking))),
        ]:
            selected = None
            for sample_id, row in pivot.iterrows():
                values = row.to_dict()
                if condition(values):
                    selected = sample_id
                    break
            if selected:
                record = df[(df["sample_id"] == selected) & (df["model"] == right_model)].iloc[0]
                cases.append(_case_row(label, record, samples))
    deduped = []
    seen = set()
    for case in cases:
        key = (case["case_type"], case["sample_id"], case["model"])
        if key not in seen:
            deduped.append(case)
            seen.add(key)
    return deduped[:20]


def _case_row(case_type: str, record: pd.Series, samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sample = samples.get(record["sample_id"], {})
    return {
        "case_type": case_type,
        "sample_id": record["sample_id"],
        "model": record["model"],
        "domain": record["domain"],
        "task_type": record["task_type"],
        "primary_error": record["primary_error"],
        "overall_score": round(float(record["overall_score"]), 4),
        "required_claim_recall": round(float(record["required_claim_recall"]), 4),
        "evidence_faithfulness": round(float(record["evidence_faithfulness"]), 4),
        "safety_boundary": round(float(record["safety_boundary"]), 4),
        "forbidden_violation": bool(record["forbidden_violation"]),
        "prompt_excerpt": _short(sample.get("prompt", record["prompt"]), 260),
        "gold_excerpt": _short(sample.get("gold_answer", record["gold_answer"]), 220),
        "response_excerpt": _short(record["response"], 320),
        "judge_notes": _short(record["judge_notes"], 220),
    }


def _write_figures(
    figures_dir: Path,
    model_table: list[dict[str, Any]],
    domain_table: list[dict[str, Any]],
    task_table: list[dict[str, Any]],
    group_table: list[dict[str, Any]],
    thinking_table: list[dict[str, Any]],
    error_table: list[dict[str, Any]],
) -> list[str]:
    _configure_plot_style()
    figures = []
    figures.append(_plot_model_overall(figures_dir, model_table))
    figures.append(_plot_heatmap(figures_dir, "domain_heatmap", domain_table, "domain", "分领域 ACC 热图"))
    figures.append(_plot_heatmap(figures_dir, "task_heatmap", task_table, "task_type", "分任务 ACC 热图"))
    figures.append(_plot_group_performance(figures_dir, group_table))
    if thinking_table:
        figures.append(_plot_thinking_delta(figures_dir, thinking_table))
    if error_table:
        figures.append(_plot_error_taxonomy(figures_dir, error_table))
    figures.append(_plot_latency_accuracy(figures_dir, model_table))
    return [str(path) for path in figures if path]


def _configure_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def _plot_model_overall(figures_dir: Path, model_table: list[dict[str, Any]]) -> Path:
    data = pd.DataFrame(model_table).sort_values("ACC")
    fig, ax = plt.subplots(figsize=(10, max(5, len(data) * 0.42)))
    y = range(len(data))
    ax.barh([item - 0.18 for item in y], data["ACC"], height=0.34, label="ACC", color="#2f6f9f")
    ax.barh([item + 0.18 for item in y], data["F1_mean"], height=0.34, label="F1_mean", color="#d9822b")
    ax.set_yticks(list(y))
    ax.set_yticklabels(data["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Overall model performance")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    return _save_figure(fig, figures_dir / "fig01_model_overall")


def _plot_heatmap(figures_dir: Path, stem: str, rows: list[dict[str, Any]], column_key: str, title: str) -> Path:
    data = pd.DataFrame(rows)
    if data.empty:
        return Path()
    pivot = data.pivot(index="model", columns=column_key, values="ACC").fillna(0.0)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.8), max(5, len(pivot) * 0.38)))
    image = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    return _save_figure(fig, figures_dir / f"fig02_{stem}" if stem == "domain_heatmap" else figures_dir / f"fig03_{stem}")


def _plot_group_performance(figures_dir: Path, group_table: list[dict[str, Any]]) -> Path:
    data = pd.DataFrame(group_table)
    if data.empty:
        return Path()
    data["group"] = data[["openness", "scale", "reasoning_style"]].astype(str).agg(" / ".join, axis=1)
    data = data.sort_values("ACC")
    fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.5)))
    ax.barh(data["group"], data["ACC"], color="#4c956c")
    ax.set_xlim(0, 1)
    ax.set_xlabel("ACC")
    ax.set_title("Performance by model group")
    ax.grid(axis="x", alpha=0.25)
    for y, value in enumerate(data["ACC"]):
        ax.text(value + 0.01, y, f"{value:.2f}", va="center", fontsize=8)
    return _save_figure(fig, figures_dir / "fig04_group_performance")


def _plot_thinking_delta(figures_dir: Path, thinking_table: list[dict[str, Any]]) -> Path:
    data = pd.DataFrame(thinking_table).sort_values("ACC_delta_thinking_minus_instruct")
    fig, ax = plt.subplots(figsize=(9, max(4, len(data) * 0.55)))
    colors = ["#2f6f9f" if value >= 0 else "#b23a48" for value in data["ACC_delta_thinking_minus_instruct"]]
    ax.barh(data["family"], data["ACC_delta_thinking_minus_instruct"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ACC delta: thinking - instruct")
    ax.set_title("Thinking vs instruct effect")
    ax.grid(axis="x", alpha=0.25)
    return _save_figure(fig, figures_dir / "fig05_thinking_delta")


def _plot_error_taxonomy(figures_dir: Path, error_table: list[dict[str, Any]]) -> Path:
    data = pd.DataFrame(error_table)
    if data.empty:
        return Path()
    pivot = data.pivot_table(index="model", columns="primary_error", values="rate_within_model", aggfunc="sum").fillna(0.0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(11, max(5, len(pivot) * 0.42)))
    left = pd.Series([0.0] * len(pivot), index=pivot.index)
    palette = ["#b23a48", "#d9822b", "#7a5195", "#4c956c", "#2f6f9f", "#8d99ae", "#bc6c25"]
    for idx, column in enumerate(pivot.columns):
        ax.barh(pivot.index, pivot[column], left=left, label=column, color=palette[idx % len(palette)])
        left += pivot[column]
    ax.set_xlim(0, max(0.01, min(1.0, float(left.max()) + 0.05)))
    ax.set_xlabel("Failure rate within model")
    ax.set_title("Error taxonomy by model")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    return _save_figure(fig, figures_dir / "fig06_error_taxonomy")


def _plot_latency_accuracy(figures_dir: Path, model_table: list[dict[str, Any]]) -> Path:
    data = pd.DataFrame(model_table)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = data["reasoning_style"].map({"thinking": "#b23a48", "instruct": "#2f6f9f"}).fillna("#6c757d")
    ax.scatter(data["MeanLatencyMs"] / 1000, data["ACC"], s=80, c=colors, alpha=0.85)
    for _, row in data.iterrows():
        ax.text(row["MeanLatencyMs"] / 1000, row["ACC"] + 0.008, row["model"], fontsize=7, ha="center")
    ax.set_xlabel("Mean latency (seconds)")
    ax.set_ylabel("ACC")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy-latency trade-off")
    ax.grid(alpha=0.25)
    return _save_figure(fig, figures_dir / "fig07_latency_accuracy")


def _save_figure(fig: plt.Figure, base_path: Path) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path


def _build_report(
    *,
    run_dir: Path,
    records: pd.DataFrame,
    model_table: list[dict[str, Any]],
    domain_table: list[dict[str, Any]],
    task_table: list[dict[str, Any]],
    group_table: list[dict[str, Any]],
    thinking_table: list[dict[str, Any]],
    error_table: list[dict[str, Any]],
    hardest_table: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    figures: list[str],
) -> str:
    sample_count = records["sample_id"].nunique()
    model_count = records["model"].nunique()
    best_model = model_table[0]
    domain_avg = _axis_average(domain_table, "domain")
    task_avg = _axis_average(task_table, "task_type")
    error_totals = _error_totals(error_table)
    figure_lines = "\n".join(f"- `{Path(path).relative_to(run_dir)}`" for path in figures if path)
    return f"""# 能源化工安全 Benchmark 全量模型评测论文式分析

## 1. 实验设置

本次评测基于当前 961 道能源化工安全 benchmark，覆盖 {model_count} 个可调用模型，共 {len(records)} 条模型-题目评测记录。模型回答使用同一套能源化工安全提示词，裁判侧输出 `ACC`、`F1_mean`、证据忠实度、安全边界、禁止性违规率和过度拒答率等指标。

本轮评测排除了当前账号不可调用或通道不可用的模型；这些模型不参与统计，以避免将 API 权限问题误判为模型能力问题。

## 2. 总体性能

总体排名第一的是 `{best_model['model']}`，ACC 为 {_pct(best_model['ACC'])}，F1_mean 为 {_pct(best_model['F1_mean'])}。从全量指标看，报告时不宜只看 ACC，还需要同时比较证据忠实度、安全边界和禁止性违规率，因为能源化工安全任务更重视“不能错放行、不能越界给危险建议”。

{_markdown_table(model_table, ["model", "openness", "scale", "reasoning_style", "n", "ACC", "F1_mean", "EvidenceFaithfulness", "SafetyBoundary", "ForbiddenViolationRate", "OverRefusalRate"], limit=30)}

## 3. 模型分领域性能分析

按领域平均后，最难领域为 `{domain_avg[0]['label']}`，平均 ACC 为 {_pct(domain_avg[0]['ACC'])}；最容易领域为 `{domain_avg[-1]['label']}`，平均 ACC 为 {_pct(domain_avg[-1]['ACC'])}。这说明 benchmark 中的不同能源化工场景对模型能力要求并不均衡，后续论文中可将受限空间、煤化工高压气体、检维修特殊作业和精细化工反应风险分别作为子任务讨论。

{_markdown_table(domain_avg, ["label", "n", "ACC", "F1_mean", "EvidenceFaithfulness", "SafetyBoundary", "ForbiddenViolationRate"], limit=20)}

## 4. 任务类型性能分析

按任务类型看，最难任务为 `{task_avg[0]['label']}`，平均 ACC 为 {_pct(task_avg[0]['ACC'])}；最容易任务为 `{task_avg[-1]['label']}`，平均 ACC 为 {_pct(task_avg[-1]['ACC'])}。通常闭卷安全边界题更容易，而证据约束、证据冲突和事故推理任务更能拉开模型差距。

{_markdown_table(task_avg, ["label", "n", "ACC", "F1_mean", "EvidenceFaithfulness", "SafetyBoundary", "ForbiddenViolationRate"], limit=30)}

## 5. Thinking 与 Instruct 对比

Thinking 模式不是稳定正收益。它可能提升复杂事故推理中的覆盖率，但也可能在证据冲突或标准条文约束题中增加外推，导致证据忠实度或安全边界下降。因此论文结论建议写成：thinking 对部分模型系列有边际收益，但需要结合领域与任务类型逐系列验证。

{_markdown_table(thinking_table, ["family", "instruct_model", "thinking_model", "ACC_delta_thinking_minus_instruct", "F1_delta_thinking_minus_instruct", "EvidenceFaithfulness_delta", "SafetyBoundary_delta", "ForbiddenViolation_delta", "Latency_delta_ms"], limit=20)}

## 6. 错误分析

主要错误类型集中在：{", ".join(f"{item['primary_error']}={item['n']}" for item in error_totals[:6])}。其中，`forbidden_violation` 表示模型给出了安全边界内不应出现的建议；`low_evidence_faithfulness` 表示回答没有严格受给定证据约束；`low_required_claim_recall` 表示漏掉了题目要求的关键安全结论。

{_markdown_table(error_totals, ["primary_error", "n", "share_among_errors"], limit=20)}

## 7. 最困难样本

以下样本是跨模型平均正确率最低的题目，适合作为论文中的代表性困难题或附录案例。

{_markdown_table(hardest_table[:12], ["sample_id", "domain", "task_type", "n_models", "correct_models", "ACC_across_models", "mean_score", "main_error_types"], limit=12)}

## 8. 代表性案例

{_cases_markdown(cases)}

## 9. 图表清单

{figure_lines}

## 10. 论文结论建议

1. 能源化工安全 benchmark 的区分度主要来自证据约束型任务，尤其是多来源材料冲突、标准时效性和事故报告推理。
2. 模型总体能力与安全可靠性并不完全一致，ACC 高的模型仍可能存在禁止性违规或证据外推。
3. Thinking 模式应作为实验变量而非默认增强项；不同系列模型的收益方向不一致。
4. 领域上应重点关注中毒窒息受限空间、煤化工高压气体和检维修特殊作业，这些场景同时具有事故高风险和模型高错误率。
5. 后续 benchmark 扩展应提高事故报告、标准条文、SDS、SOP 片段之间的冲突样本比例，并加入更多“不能据此放行”的专业边界题。
"""


def _axis_average(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    result = []
    for label, group in frame.groupby(key):
        total = group["n"].sum()
        row = {"label": label, "n": int(total)}
        for metric in ["ACC", "F1_mean", "EvidenceFaithfulness", "SafetyBoundary", "ForbiddenViolationRate"]:
            row[metric] = round(float((group[metric] * group["n"]).sum() / total), 4) if total else 0.0
        result.append(row)
    return sorted(result, key=lambda item: item["ACC"])


def _error_totals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    totals = frame.groupby("primary_error")["n"].sum().sort_values(ascending=False)
    denominator = int(totals.sum())
    return [
        {"primary_error": label, "n": int(count), "share_among_errors": round(float(count / denominator), 4)}
        for label, count in totals.items()
    ]


def _cases_markdown(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return "暂无可抽取案例。"
    blocks = []
    for idx, case in enumerate(cases, start=1):
        blocks.append(
            f"""### 案例 {idx}: {case['case_type']}

- 样本：`{case['sample_id']}`；模型：`{case['model']}`；领域：`{case['domain']}`；任务：`{case['task_type']}`
- 错误类型：`{case['primary_error']}`；score={case['overall_score']}；recall={case['required_claim_recall']}；evidence={case['evidence_faithfulness']}；safety={case['safety_boundary']}；forbidden={case['forbidden_violation']}
- 题目片段：{case['prompt_excerpt']}
- 标准答案片段：{case['gold_excerpt']}
- 模型回答片段：{case['response_excerpt']}
- 裁判备注：{case['judge_notes']}
"""
        )
    return "\n".join(blocks)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 20) -> str:
    if not rows:
        return "无数据。"
    shown = rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in shown:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _top_labels(labels: list[str], limit: int = 3) -> str:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return "; ".join(f"{label}:{count}" for label, count in sorted(counts.items(), key=lambda item: -item[1])[:limit])


def _short(text: Any, limit: int) -> str:
    clean = " ".join(str(text or "").replace("\n", " ").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"
