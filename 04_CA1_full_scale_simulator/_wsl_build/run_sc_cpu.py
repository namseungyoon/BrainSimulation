"""Stage 2.5 item 1 (CPU path): re-run the reduced AEIF point neuron in CPU NEST
under current clamp and z-score its single-cell electrophysiology vs the NEURON
multi-compartment ground truth. This is OUR reproduction (a live NEST run), not a
re-read of stored numbers. Validates neuron_parameters_fitted.json (AEIF fit;
Pyramidal/PV are analytic fallbacks -- expected to fail, which is honest).
Output JSON -> _study/single_cell_cpu_aeif_repro.json (Windows side via /mnt/d)."""
import json
from pathlib import Path

_OUT = Path(
    "/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/"
    "04_CA1_full_scale_simulator/_study/single_cell_cpu_aeif_repro.json"
)


def main() -> None:
    import ca1.params as P
    from ca1.validation.single_cell import validate_fits

    pdir = Path(P.__file__).resolve().parent
    gt = json.loads((pdir / "ground_truth.json").read_text(encoding="utf-8"))
    fitted = json.loads((pdir / "neuron_parameters_fitted.json").read_text(encoding="utf-8"))
    # Driver-side input cleanup (no model/validation logic change): drop the
    # free-text 'note' metadata key that the analytic-fallback records carry;
    # single_cell.py only strips I_e/loss/fit_provenance/validation, so 'note'
    # would otherwise leak into NEST SetStatus and raise DictError.
    for _cell in fitted.values():
        if isinstance(_cell, dict):
            _cell.pop("note", None)
    res = validate_fits(fitted, gt, nproc=1)  # serial: 9 cells, fast, avoids spawn
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"{'cell':16s} {'pass':6s} {'med_z':>6s} {'max_z':>6s}  hard_fails")
    for name in fitted:
        r = res[name]
        print(f"{name:16s} {str(r['passed']):6s} {r['median_z']:6.2f} {r['max_z']:6.2f}  {r['hard_fails']}")
    print("saved:", _OUT)


if __name__ == "__main__":
    main()
