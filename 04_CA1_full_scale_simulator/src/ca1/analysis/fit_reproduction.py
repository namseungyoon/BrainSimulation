from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ca1.analysis.fit_reproduction_data import load_reproduction_dataset
from ca1.analysis.fit_reproduction_plots import aglif_rates_available, save_reproduction_figures
from ca1.analysis.fit_reproduction_response_plots import save_response_outputs
from ca1.analysis.fit_reproduction_replay import (
    build_aglif_replay_report,
    write_aglif_replay_report,
)
from ca1.analysis.fit_reproduction_traces import (
    build_response_trace_report,
    load_response_traces,
    write_response_trace_report,
)

_PARAM_DIR = Path(__file__).resolve().parents[1] / "params"
_DEFAULT_GT = _PARAM_DIR / "ground_truth.json"
_DEFAULT_AEIF = _PARAM_DIR / "neuron_parameters_fitted.json"
_DEFAULT_AGLIF = _PARAM_DIR / "aglif_parameters_fitted.json"
_DEFAULT_OUT_DIR = Path("results") / "fit_reproduction"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--gt", type=Path, default=_DEFAULT_GT)
    _ = parser.add_argument("--aeif", type=Path, default=_DEFAULT_AEIF)
    _ = parser.add_argument("--aglif", type=Path, default=_DEFAULT_AGLIF)
    _ = parser.add_argument("--aglif-report", type=Path, default=None)
    _ = parser.add_argument("--trace-report", type=Path, default=None)
    _ = parser.add_argument("--refresh-aglif", action="store_true")
    _ = parser.add_argument("--refresh-traces", action="store_true")
    _ = parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    _ = parser.add_argument("--prefix", default="single_cell_fit_reproduction")
    _ = parser.add_argument("--formats", default="png,pdf")
    _ = parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    gt_path = _path_arg(args, "gt")
    aeif_path = _path_arg(args, "aeif")
    aglif_path = _path_arg(args, "aglif")
    out_dir = _path_arg(args, "out_dir")
    formats = _parse_formats(_str_arg(args, "formats"))
    if _bool_arg(args, "refresh_aglif") and _bool_arg(args, "refresh_traces"):
        raise ValueError("Run --refresh-aglif and --refresh-traces in separate commands.")
    aglif_report = _resolve_aglif_report(
        gt_path=gt_path,
        aglif_path=aglif_path,
        requested_report=_optional_path_arg(args, "aglif_report"),
        out_dir=out_dir,
        refresh=_bool_arg(args, "refresh_aglif"),
    )
    trace_report = _resolve_trace_report(
        gt_path=gt_path,
        aeif_path=aeif_path,
        aglif_path=aglif_path,
        requested_report=_optional_path_arg(args, "trace_report"),
        out_dir=out_dir,
        refresh=_bool_arg(args, "refresh_traces"),
    )
    dataset = load_reproduction_dataset(
        gt_path=gt_path,
        aeif_path=aeif_path,
        aglif_path=aglif_path,
        aglif_report_path=aglif_report,
    )
    saved = save_reproduction_figures(
        dataset,
        out_dir,
        _str_arg(args, "prefix"),
        formats,
        _int_arg(args, "dpi"),
    )
    traces = None if trace_report is None else load_response_traces(trace_report)
    saved = (
        *saved,
        *save_response_outputs(
            traces,
            dataset,
            out_dir,
            _str_arg(args, "prefix"),
            formats,
            _int_arg(args, "dpi"),
        ),
    )
    for path in saved:
        print(path)
    if not aglif_rates_available(dataset):
        print("A-GLIF f-I curves omitted; rerun with --refresh-aglif to replay NEST-GPU rates.")


def _parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(item.strip().lstrip(".") for item in value.split(",") if item.strip())
    if not formats:
        raise ValueError("--formats must contain at least one extension")
    return formats


def _path_arg(args: argparse.Namespace, name: str) -> Path:
    value = getattr(args, name)
    if isinstance(value, Path):
        return value
    raise TypeError(f"{name} must be a Path")


def _optional_path_arg(args: argparse.Namespace, name: str) -> Path | None:
    value = getattr(args, name)
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    raise TypeError(f"{name} must be a Path or None")


def _str_arg(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name)
    if isinstance(value, str):
        return value
    raise TypeError(f"{name} must be a str")


def _int_arg(args: argparse.Namespace, name: str) -> int:
    value = getattr(args, name)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be an int")


def _bool_arg(args: argparse.Namespace, name: str) -> bool:
    value = getattr(args, name)
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a bool")


def _resolve_aglif_report(
    *,
    gt_path: Path,
    aglif_path: Path,
    requested_report: Path | None,
    out_dir: Path,
    refresh: bool,
) -> Path | None:
    if refresh:
        report_path = requested_report or out_dir / "aglif_replay_report.json"
        report = build_aglif_replay_report(gt_path, aglif_path)
        write_aglif_replay_report(report, report_path)
        return report_path
    if requested_report is not None:
        return requested_report
    candidate = out_dir / "aglif_replay_report.json"
    if candidate.exists():
        return candidate
    return None


def _resolve_trace_report(
    *,
    gt_path: Path,
    aeif_path: Path,
    aglif_path: Path,
    requested_report: Path | None,
    out_dir: Path,
    refresh: bool,
) -> Path | None:
    if refresh:
        report_path = requested_report or out_dir / "stimulus_response_traces.json"
        report = build_response_trace_report(gt_path, aeif_path, aglif_path)
        write_response_trace_report(report, report_path)
        return report_path
    if requested_report is not None:
        return requested_report
    candidate = out_dir / "stimulus_response_traces.json"
    if candidate.exists():
        return candidate
    return None


if __name__ == "__main__":
    main()
