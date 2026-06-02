from __future__ import annotations

import shutil
import textwrap
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "multi_model_eval" / "full_usable_models_961_20260601"
ANALYSIS_DIR = RUN_DIR / "paper_analysis"
PAPER_DIR = ROOT / "paper_neurips"
FIG_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"


DOMAIN_LABELS_BY_PREFIX = {
    "01_": "Fine-chemical\nreaction risk",
    "02_": "Maintenance and\nspecial operations",
    "03_": "Toxic exposure and\nconfined spaces",
    "04_": "Coal-chemical\nhigh-pressure gas",
    "05_": "Petrochemical tanks\nand pipelines",
    "06_": "Urban gas\nnetworks",
    "07_": "Cross-domain\nsynthesis",
}

TASK_LABELS = {
    "closed_book_dangerous_inducement": "Dangerous\ninducement",
    "closed_book_professional_boundary": "Professional\nboundary",
    "closed_book_safe_control": "Safe-control\nknowledge",
    "evidence_document_conflict": "Document\nconflict",
    "evidence_fact_verification": "Evidence fact\nverification",
    "evidence_incident_reasoning": "Incident\nreasoning",
    "evidence_sds_verification": "SDS / hazard\nverification",
    "evidence_standard_temporal": "Temporal standard\nreasoning",
}

MODEL_SHORT = {
    "claude-opus-4-5-20251101-ssvip": "Claude Opus 4.5",
    "DeepSeek-V3.2": "DeepSeek-V3.2",
    "DeepSeek-V3.2-Thinking": "DeepSeek-V3.2-Thinking",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "glm-4.6": "GLM-4.6",
    "glm-4.6-thinking": "GLM-4.6-Thinking",
    "gpt-5.2": "GPT-5.2",
    "gpt-oss-20b": "GPT-OSS-20B",
    "kimi-k2-0905-preview": "Kimi-K2",
    "qwen3-14b": "Qwen3-14B",
    "qwen3-235b-a22b-instruct-2507": "Qwen3-235B-Instruct",
    "qwen3-235b-a22b-thinking-2507": "Qwen3-235B-Thinking",
    "qwen3-30b-a3b-thinking-2507": "Qwen3-30B-A3B-Thinking",
    "qwen3-8b": "Qwen3-8B",
    "qwen3-max-2025-09-23": "Qwen3-Max",
}

ERROR_LABELS = {
    "forbidden_violation": "unsafe / forbidden\nclaim",
    "incomplete_or_uncertain_reasoning": "incomplete or\nuncertain reasoning",
    "over_refusal": "over-refusal",
    "low_evidence_faithfulness": "low evidence\nfaithfulness",
    "low_required_claim_recall": "low required-\nclaim recall",
    "weak_safety_boundary": "weak safety\nboundary",
    "api_or_runtime_error": "provider\nfilter/error",
}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    _configure()

    model = pd.read_csv(ANALYSIS_DIR / "model_ranking.csv")
    domain = pd.read_csv(ANALYSIS_DIR / "domain_performance.csv")
    task = pd.read_csv(ANALYSIS_DIR / "task_performance.csv")
    group = pd.read_csv(ANALYSIS_DIR / "group_performance.csv")
    thinking = pd.read_csv(ANALYSIS_DIR / "thinking_vs_instruct.csv")
    errors = pd.read_csv(ANALYSIS_DIR / "error_taxonomy_by_model.csv")
    hardest = pd.read_csv(ANALYSIS_DIR / "hardest_samples.csv")
    cases = pd.read_csv(ANALYSIS_DIR / "case_studies.csv")

    for frame in (model, domain, task, errors):
        if "model" in frame:
            frame["model_short"] = frame["model"].map(MODEL_SHORT).fillna(frame["model"])

    _plot_overall(model)
    _plot_domain_heatmap(domain)
    _plot_task_heatmap(task)
    _plot_thinking(thinking)
    _plot_error_taxonomy(errors)
    _plot_group(group)
    _plot_latency(model)
    _plot_combined_figures()
    _write_tables(model, domain, task, thinking, errors, hardest, cases)
    _write_readme()


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def _domain_label(raw: object) -> str:
    value = str(raw)
    for prefix, label in DOMAIN_LABELS_BY_PREFIX.items():
        if value.startswith(prefix):
            return label
    return value


def _plain_label(label: str) -> str:
    return label.replace("\n", " ")


def _plot_overall(model: pd.DataFrame) -> None:
    data = model.sort_values("ACC")
    fig, ax = plt.subplots(figsize=(8.3, 6.5))
    y = list(range(len(data)))
    ax.barh([i - 0.18 for i in y], data["ACC"], height=0.34, color="#2C7FB8", label="Accuracy")
    ax.barh([i + 0.18 for i in y], data["F1_mean"], height=0.34, color="#F28E2B", label="Mean judge score")
    ax.set_yticks(y)
    ax.set_yticklabels(data["model_short"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Overall performance across 961 benchmark items")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "fig1_overall_performance")


def _plot_domain_heatmap(domain: pd.DataFrame) -> None:
    data = domain.copy()
    data["domain_en"] = data["domain"].map(_domain_label)
    pivot = data.pivot(index="model_short", columns="domain_en", values="ACC").fillna(0.0)
    order = data.groupby("model_short")["ACC"].mean().sort_values().index
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(9.4, 6.7))
    image = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=7.3)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7.7)
    ax.set_title("Accuracy by process-safety domain")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=6.1)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    _save(fig, "fig2_domain_heatmap")


def _plot_task_heatmap(task: pd.DataFrame) -> None:
    data = task.copy()
    data["task_en"] = data["task_type"].map(TASK_LABELS).fillna(data["task_type"])
    pivot = data.pivot(index="model_short", columns="task_en", values="ACC").fillna(0.0)
    order = data.groupby("model_short")["ACC"].mean().sort_values().index
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(10.7, 6.8))
    image = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=7.1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7.7)
    ax.set_title("Accuracy by task type")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=5.8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    _save(fig, "fig3_task_heatmap")


def _plot_thinking(thinking: pd.DataFrame) -> None:
    data = thinking.copy()
    data["family"] = data["family"].replace(
        {"deepseek-v3.2": "DeepSeek-V3.2", "glm": "GLM", "qwen3-235b": "Qwen3-235B"}
    )
    data = data.sort_values("ACC_delta_thinking_minus_instruct")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    colors = ["#D1495B" if x < 0 else "#2E8B57" for x in data["ACC_delta_thinking_minus_instruct"]]
    ax.barh(data["family"], data["ACC_delta_thinking_minus_instruct"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy delta (thinking - instruct)")
    ax.set_title("Thinking mode is not uniformly beneficial")
    _save(fig, "fig4_thinking_delta")


def _plot_error_taxonomy(errors: pd.DataFrame) -> None:
    data = errors.copy()
    data["error_label"] = data["primary_error"].map(ERROR_LABELS).fillna(data["primary_error"])
    pivot = data.pivot_table(
        index="model_short", columns="error_label", values="rate_within_model", aggfunc="sum"
    ).fillna(0.0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9.4, 6.5))
    left = pd.Series([0.0] * len(pivot), index=pivot.index)
    palette = ["#D1495B", "#F28E2B", "#8E6C8A", "#4E79A7", "#76B7B2", "#59A14F", "#BAB0AC"]
    for idx, column in enumerate(pivot.columns):
        ax.barh(pivot.index, pivot[column], left=left, label=column, color=palette[idx % len(palette)])
        left += pivot[column]
    ax.set_xlabel("Failure rate within model")
    ax.set_title("Error taxonomy")
    ax.tick_params(axis="y", labelsize=7.4)
    ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="lower right")
    _save(fig, "fig5_error_taxonomy")


def _plot_group(group: pd.DataFrame) -> None:
    data = group.copy()
    data["group"] = data[["openness", "scale", "reasoning_style"]].agg(" / ".join, axis=1)
    data = data.sort_values("ACC")
    fig, ax = plt.subplots(figsize=(7.3, 3.9))
    ax.barh(data["group"], data["ACC"], color="#4E79A7")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Performance by model group")
    _save(fig, "fig6_group_performance")


def _plot_latency(model: pd.DataFrame) -> None:
    data = model.copy()
    colors = data["reasoning_style"].map({"thinking": "#D1495B", "instruct": "#2C7FB8"}).fillna("#777777")
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    ax.scatter(data["MeanLatencyMs"] / 1000, data["ACC"], s=58, c=colors, alpha=0.86)
    for _, row in data.iterrows():
        ax.text(row["MeanLatencyMs"] / 1000, row["ACC"] + 0.0065, row["model_short"], fontsize=6.0, ha="center")
    ax.set_xlabel("Mean latency (s)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.35, 0.78)
    ax.set_title("Accuracy--latency trade-off")
    _save(fig, "fig7_latency_accuracy")


def _plot_combined_figures() -> None:
    _combine_pngs(
        "fig_domain_task_heatmaps",
        [
            ("(a) Domain-wise accuracy", FIG_DIR / "fig2_domain_heatmap.png"),
            ("(b) Task-wise accuracy", FIG_DIR / "fig3_task_heatmap.png"),
        ],
        figsize=(13.0, 6.0),
    )
    _combine_pngs(
        "fig_group_latency",
        [
            ("(a) Group-level accuracy", FIG_DIR / "fig6_group_performance.png"),
            ("(b) Accuracy--latency trade-off", FIG_DIR / "fig7_latency_accuracy.png"),
        ],
        figsize=(12.0, 5.2),
    )


def _combine_pngs(name: str, panels: list[tuple[str, Path]], figsize: tuple[float, float]) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, path) in zip(axes, panels, strict=True):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=6)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def _write_tables(
    model: pd.DataFrame,
    domain: pd.DataFrame,
    task: pd.DataFrame,
    thinking: pd.DataFrame,
    errors: pd.DataFrame,
    hardest: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    top = model.head(8).copy()
    top["Model"] = top["model_short"]
    top["ACC"] = top["ACC"].map(lambda x: f"{100*x:.1f}")
    top["Score"] = top["F1_mean"].map(lambda x: f"{100*x:.1f}")
    top["Faith."] = top["EvidenceFaithfulness"].map(lambda x: f"{100*x:.1f}")
    top["Safety"] = top["SafetyBoundary"].map(lambda x: f"{100*x:.1f}")
    _latex_table(top[["Model", "ACC", "Score", "Faith.", "Safety"]], TABLE_DIR / "top_models.tex")

    dom = _weighted_axis(domain, "domain", _domain_label).sort_values("ACC")
    dom["ACC"] = dom["ACC"].map(lambda x: f"{100*x:.1f}")
    dom["Domain"] = dom["label"]
    _latex_table(dom[["Domain", "n", "ACC"]], TABLE_DIR / "domain_summary.tex")

    tasks = _weighted_axis(task, "task_type", lambda value: TASK_LABELS.get(str(value), str(value))).sort_values("ACC")
    tasks["ACC"] = tasks["ACC"].map(lambda x: f"{100*x:.1f}")
    tasks["Task"] = tasks["label"]
    _latex_table(tasks[["Task", "n", "ACC"]], TABLE_DIR / "task_summary.tex")

    th = thinking.copy()
    th["Family"] = th["family"].replace({"deepseek-v3.2": "DeepSeek", "glm": "GLM", "qwen3-235b": "Qwen3-235B"})
    th["Delta ACC"] = th["ACC_delta_thinking_minus_instruct"].map(lambda x: f"{100*x:+.1f}")
    th["Delta F1"] = th["F1_delta_thinking_minus_instruct"].map(lambda x: f"{100*x:+.1f}")
    th["Delta faith."] = th["EvidenceFaithfulness_delta"].map(lambda x: f"{100*x:+.1f}")
    th["Unsafe delta"] = th["ForbiddenViolation_delta"].map(lambda x: f"{100*x:+.1f}")
    _latex_table(th[["Family", "Delta ACC", "Delta F1", "Delta faith.", "Unsafe delta"]], TABLE_DIR / "thinking_summary.tex")

    total_errors = errors.groupby("primary_error")["n"].sum().sort_values(ascending=False).reset_index()
    total_errors["Error type"] = total_errors["primary_error"].str.replace("_", " ")
    _latex_table(total_errors[["Error type", "n"]].head(7), TABLE_DIR / "error_summary.tex")

    hardest.head(12).to_csv(TABLE_DIR / "hardest_samples_for_appendix.csv", index=False)
    cases.to_csv(TABLE_DIR / "case_studies_for_appendix.csv", index=False)


def _weighted_axis(frame: pd.DataFrame, axis: str, labeler: Callable[[object], str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value, group in frame.groupby(axis):
        n = group["n"].sum()
        rows.append(
            {
                "label": _plain_label(labeler(value)),
                "n": int(n),
                "ACC": float((group["ACC"] * group["n"]).sum() / n),
            }
        )
    return pd.DataFrame(rows)


def _latex_table(frame: pd.DataFrame, path: Path) -> None:
    cols = list(frame.columns)
    lines = ["\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", "\\toprule"]
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")
    for _, row in frame.iterrows():
        values = [_escape_latex(str(row[col])) for col in cols]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _escape_latex(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _write_readme() -> None:
    shutil.copyfile(ANALYSIS_DIR / "model_ranking.csv", TABLE_DIR / "model_ranking.csv")
    shutil.copyfile(ANALYSIS_DIR / "dataset_composition.csv", TABLE_DIR / "dataset_composition.csv")
    readme = textwrap.dedent(
        f"""\
        # NeurIPS Paper Assets

        Generated from `{RUN_DIR}`.

        - Figures are in `figures/` as both PDF and PNG.
        - Compact LaTeX tables are in `tables/`.
        - The paper source is `main.tex`.
        """
    )
    (PAPER_DIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
