"""CompRAG CLI entry point."""

import argparse
import sys


def cmd_retrieve(args: argparse.Namespace) -> None:
    """Run retrieval step."""
    print(f"retrieve: Not implemented yet (dataset={args.dataset}, subset={args.subset})")


def cmd_generate(args: argparse.Namespace) -> None:
    """Run generation step."""
    print(
        f"generate: Not implemented yet (model={args.model}, quant={args.quant}, "
        f"dataset={args.dataset}, pass={getattr(args, 'pass')}, "
        f"frontier={args.frontier}, provider={args.provider}, seed={args.seed})"
    )


def cmd_score(args: argparse.Namespace) -> None:
    """Run scoring step."""
    print(f"score: Not implemented yet (input={args.input})")


def cmd_aggregate(args: argparse.Namespace) -> None:
    """Run aggregation step."""
    print(f"aggregate: Not implemented yet (input_dir={args.input_dir}, output_dir={args.output_dir})")


def cmd_visualize(args: argparse.Namespace) -> None:
    """Run visualization step."""
    print(f"visualize: Not implemented yet (input_dir={args.input_dir}, output_dir={args.output_dir})")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(prog="comprag", description="CompRAG benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)

    # retrieve
    p_ret = sub.add_parser("retrieve", help="Run retrieval on a dataset")
    p_ret.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. rgb, nq, halueval)")
    p_ret.add_argument("--subset", type=str, default=None, help="Dataset subset (e.g. counterfactual)")
    p_ret.set_defaults(func=cmd_retrieve)

    # generate
    p_gen = sub.add_parser("generate", help="Run generation on a dataset")
    p_gen.add_argument("--model", type=str, required=True, help="Model name")
    p_gen.add_argument("--quant", type=str, default=None, help="Quantization level (e.g. Q4_K_M)")
    p_gen.add_argument("--dataset", type=str, required=True, help="Dataset name")
    p_gen.add_argument("--pass", type=str, required=True,
                       choices=["pass1_baseline", "pass2_loose", "pass3_strict"],
                       help="Evaluation pass")
    p_gen.add_argument("--frontier", action="store_true", help="Use frontier API model")
    p_gen.add_argument("--provider", type=str, default=None,
                       choices=["openai", "anthropic", "google", "deepseek", "zhipu"],
                       help="API provider (required with --frontier)")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.set_defaults(func=cmd_generate)

    # score
    p_score = sub.add_parser("score", help="Score a results JSONL file")
    p_score.add_argument("--input", type=str, required=True, help="Path to raw results JSONL")
    p_score.set_defaults(func=cmd_score)

    # aggregate
    p_agg = sub.add_parser("aggregate", help="Aggregate scored results")
    p_agg.add_argument("--input-dir", type=str, required=True, help="Directory of scored JSONL files")
    p_agg.add_argument("--output-dir", type=str, required=True, help="Output directory for aggregated results")
    p_agg.set_defaults(func=cmd_aggregate)

    # visualize
    p_viz = sub.add_parser("visualize", help="Generate figures from aggregated results")
    p_viz.add_argument("--input-dir", type=str, required=True, help="Directory of aggregated results")
    p_viz.add_argument("--output-dir", type=str, required=True, help="Output directory for figures")
    p_viz.set_defaults(func=cmd_visualize)

    return parser


def main() -> None:
    """Parse arguments and dispatch to subcommand handler."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
