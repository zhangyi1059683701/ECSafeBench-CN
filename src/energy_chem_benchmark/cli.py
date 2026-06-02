from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_BENCHMARK_PATH
from .model_eval import model_catalog_rows, run_multi_model_eval, write_multi_model_analysis
from .paper_analysis import write_paper_analysis


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_model_catalog(_: argparse.Namespace) -> int:
    _print(model_catalog_rows())
    return 0


def command_multi_model_eval(args: argparse.Namespace) -> int:
    _print(
        run_multi_model_eval(
            samples_path=Path(args.samples),
            models=args.models,
            workers=args.workers,
            limit=args.limit,
            stratified=not args.no_stratified,
            base_url=args.base_url,
            auth_style=args.auth_style,
            timeout=args.timeout,
            retries=args.retries,
            judge_model=args.judge_model,
            run_name=args.run_name,
            mock=args.mock,
        )
    )
    return 0


def command_analyze_model_eval(args: argparse.Namespace) -> int:
    _print(write_multi_model_analysis(Path(args.run_dir)))
    return 0


def command_paper_analysis(args: argparse.Namespace) -> int:
    _print(write_paper_analysis(Path(args.run_dir)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ECSafeBench-CN evaluation utilities for energy and chemical process-safety LLM benchmarking."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser("model-catalog", help="List model groups used by the benchmark evaluation.")
    catalog.set_defaults(func=command_model_catalog)

    multi_eval = commands.add_parser(
        "multi-model-evaluate",
        help="Evaluate multiple models on ECSafeBench-CN through an OpenAI-compatible chat-completions API.",
    )
    multi_eval.add_argument("--samples", default=str(DEFAULT_BENCHMARK_PATH))
    multi_eval.add_argument("--models", default="preset:balanced", help="preset:balanced, preset:all, or comma-separated model names")
    multi_eval.add_argument("--workers", type=int, default=16)
    multi_eval.add_argument("--limit", type=int, default=0, help="Optional sample limit; stratified by domain/task by default")
    multi_eval.add_argument("--no-stratified", action="store_true", help="Use the first N samples instead of stratified sampling")
    multi_eval.add_argument("--base-url", default="https://www.dmxapi.cn/v1")
    multi_eval.add_argument("--auth-style", default="raw", choices=["raw", "bearer"])
    multi_eval.add_argument("--timeout", type=int, default=90)
    multi_eval.add_argument("--retries", type=int, default=3)
    multi_eval.add_argument("--judge-model", default="gpt-5.4-mini")
    multi_eval.add_argument("--run-name", default="")
    multi_eval.add_argument("--mock", action="store_true")
    multi_eval.set_defaults(func=command_multi_model_eval)

    analyze_eval = commands.add_parser("analyze-model-eval", help="Regenerate aggregate CSVs and report for a multi-model run.")
    analyze_eval.add_argument("--run-dir", required=True)
    analyze_eval.set_defaults(func=command_analyze_model_eval)

    paper_analysis = commands.add_parser("paper-analysis", help="Generate paper-style tables, figures, and error cases for a model run.")
    paper_analysis.add_argument("--run-dir", required=True)
    paper_analysis.set_defaults(func=command_paper_analysis)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

