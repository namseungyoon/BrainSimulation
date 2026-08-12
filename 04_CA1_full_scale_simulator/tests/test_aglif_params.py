from __future__ import annotations

import json
from pathlib import Path

import pytest

import ca1.params.aglif as aglif_mod
from ca1.params.aglif import load_aglif_params
from ca1.params.groundtruth import CELL_TEMPLATES


def _aglif_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "V_th": -52.0,
        "E_L": -64.0,
        "C_m": 100.0,
        "tau_m": 8.0,
        "k_adap": 0.02,
        "k1": 0.01,
        "k2": 0.03,
        "A1": 7.0,
        "A2": 11.0,
        "I_e": 0.0,
        "V_peak": -47.0,
        "V_reset": -66.0,
        "t_ref": 2.0,
        "fit_provenance": "nestgpu-fi-fit",
    }
    record.update(overrides)
    return record


def _complete_aglif_fit(**overrides_by_cell: dict[str, object]) -> dict[str, object]:
    return {
        name: _aglif_record(**overrides_by_cell.get(name, {}))
        for name in CELL_TEMPLATES
    }


def test_aglif_params_default_loader_uses_complete_fitted_file() -> None:
    fit_path = Path(__file__).parents[1] / "src/ca1/params/aglif_parameters_fitted.json"
    raw = json.loads(fit_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    pyramidal = raw["Pyramidal"]
    assert isinstance(pyramidal, dict)
    cm = pyramidal["C_m"]
    assert isinstance(cm, int | float)

    params = load_aglif_params()

    assert params["Pyramidal"].C_m == pytest.approx(cm)
    assert "current_gain" not in params["Pyramidal"].as_nest()


def test_aglif_params_expected_cells_do_not_depend_on_adex_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called() -> object:
        raise AssertionError("A-GLIF loader must not inspect AdEx fitted fallback")

    monkeypatch.setattr(
        aglif_mod,
        "load_neuron_params",
        fail_if_called,
        raising=False,
    )
    fit_path = tmp_path / "aglif_parameters_fitted.json"
    _ = fit_path.write_text(json.dumps(_complete_aglif_fit()), encoding="utf-8")

    params = load_aglif_params(fit_path)

    assert sorted(params) == sorted(CELL_TEMPLATES)


def test_aglif_params_missing_fit_file_raises_not_fallback(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="missing A-GLIF fitted params"):
        _ = load_aglif_params(tmp_path / "missing.json")


def test_aglif_params_prefer_validated_fitted_file(tmp_path: Path) -> None:
    fit_path = tmp_path / "aglif_parameters_fitted.json"
    _ = fit_path.write_text(
        json.dumps(
            _complete_aglif_fit(Pyramidal={"C_m": 123.0})
        ),
        encoding="utf-8",
    )

    params = load_aglif_params(fit_path)
    assert params["Pyramidal"].C_m == pytest.approx(123.0)
    assert params["Pyramidal"].A1 == pytest.approx(7.0)


def test_aglif_params_reject_nonstandard_nan_validation_payload(
    tmp_path: Path,
) -> None:
    fit_path = tmp_path / "aglif_parameters_fitted.json"
    payload = _complete_aglif_fit()
    pyramidal = payload["Pyramidal"]
    assert isinstance(pyramidal, dict)
    pyramidal["validation"] = {
        "passed": True,
        "passive": {"sag": float("nan")},
    }
    _ = fit_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non-standard JSON constant 'NaN'"):
        _ = load_aglif_params(fit_path)


def test_aglif_params_failed_fit_raises_not_fallback(tmp_path: Path) -> None:
    fit_path = tmp_path / "aglif_parameters_fitted.json"
    _ = fit_path.write_text(
        json.dumps(
            {
                "PV_Basket": {
                    "V_th": -40.0,
                    "E_L": -65.0,
                    "C_m": 1.0,
                    "tau_m": 1.0,
                    "k_adap": 1.0,
                    "k1": 1.0,
                    "k2": 1.0,
                    "A1": 1.0,
                    "A2": 1.0,
                    "I_e": 0.0,
                    "V_peak": -35.0,
                    "V_reset": -60.0,
                    "t_ref": 1.0,
                    "fit_provenance": "FAILED",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PV_Basket.*marked FAILED"):
        _ = load_aglif_params(fit_path)


def test_aglif_params_missing_fit_provenance_raises_not_default(
    tmp_path: Path,
) -> None:
    fit_path = tmp_path / "aglif_parameters_fitted.json"
    payload = _complete_aglif_fit(Pyramidal={"fit_provenance": None})
    pyramidal = payload["Pyramidal"]
    assert isinstance(pyramidal, dict)
    del pyramidal["fit_provenance"]
    _ = fit_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fit_provenance.*required"):
        _ = load_aglif_params(fit_path)


def test_aglif_params_missing_fit_raises_not_fallback(tmp_path: Path) -> None:
    fit_path = tmp_path / "aglif_parameters_fitted.json"
    _ = fit_path.write_text(
        json.dumps({"Pyramidal": _aglif_record()}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing A-GLIF params"):
        _ = load_aglif_params(fit_path)


def test_aglif_params_unknown_cell_type_raises_not_fallback(
    tmp_path: Path,
) -> None:
    fit_path = tmp_path / "aglif_parameters_fitted.json"
    _ = fit_path.write_text(
        json.dumps(
            {
                "PVBasket": {
                    "V_th": -40.0,
                    "E_L": -65.0,
                    "C_m": 1.0,
                    "tau_m": 1.0,
                    "k_adap": 1.0,
                    "k1": 1.0,
                    "k2": 1.0,
                    "A1": 1.0,
                    "A2": 1.0,
                    "I_e": 0.0,
                    "V_peak": -35.0,
                    "V_reset": -60.0,
                    "t_ref": 1.0,
                    "fit_provenance": "nestgpu-fi-fit",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown cell type 'PVBasket'"):
        _ = load_aglif_params(fit_path)
