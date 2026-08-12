"""Stage 2.5 item 1 (GPU path): reproduce the A-GLIF (user_m1) single-cell f-I
replay via NEST-GPU and z-score vs the NEURON ground truth. This calls the exact
function (build_aglif_replay_report -> BatchAGLIFFI) that produced the stored
'validation' block in aglif_parameters_fitted.json, so it reproduces the deployed
GPU model AND lets us check our numbers match Dr. Kim's stored ones.
Output JSON -> _study/single_cell_gpu_aglif_repro.json (Windows side via /mnt/d)."""
import json
from pathlib import Path

_OUT = Path(
    "/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/"
    "02_CA1_full_scale_Simulation/_study/single_cell_gpu_aglif_repro.json"
)


def main() -> None:
    import ca1.params as P
    from ca1.analysis.fit_reproduction_replay import build_aglif_replay_report

    pdir = Path(P.__file__).resolve().parent
    report = build_aglif_replay_report(
        pdir / "ground_truth.json", pdir / "aglif_parameters_fitted.json"
    )
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'cell':16s} {'pass':6s} {'med_z':>6s} {'max_z':>6s}")
    for name, r in report.items():
        print(f"{name:16s} {str(r['passed']):6s} {r['median_z']:6.2f} {r['max_z']:6.2f}")
    print("saved:", _OUT)


if __name__ == "__main__":
    main()
