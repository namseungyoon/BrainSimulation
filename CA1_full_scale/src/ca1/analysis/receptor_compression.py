from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ca1.analysis.receptor_compression_rank import (
    rank_receptor_compression_strategies,
    write_report,
)


class _CliNamespace(argparse.Namespace):
    variant: int = 120
    n_budget: int = 20
    json_path: Path = Path("docs/generated/receptor_compression_ranking.json")
    csv_path: Path = Path("docs/generated/receptor_compression_ranking.csv")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    report = rank_receptor_compression_strategies(
        variant=args.variant,
        n_budget=args.n_budget,
    )
    write_report(report, args.json_path, args.csv_path)
    for score in report.scores:
        line = (
            f"{score.strategy}: objective={score.rank_objective:.6f} "
            f"utility_loss={score.utility_loss:.6f} ports={score.n_ports} "
            f"eff_ports={score.effective_ports_per_item:.3f}"
        )
        print(line)
    return 0


def _parse_args(argv: list[str]) -> _CliNamespace:
    namespace = _CliNamespace()
    _ = _parser().parse_args(argv, namespace=namespace)
    return namespace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ca1.analysis.receptor_compression",
        description="Rank 39-to-20 synaptic receptor-port compression candidates.",
    )
    _ = parser.add_argument(
        "--variant",
        type=int,
        choices=(120, 137),
        default=120,
    )
    _ = parser.add_argument("--n-budget", type=int, choices=(20,), default=20)
    _ = parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("docs/generated/receptor_compression_ranking.json"),
    )
    _ = parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("docs/generated/receptor_compression_ranking.csv"),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
