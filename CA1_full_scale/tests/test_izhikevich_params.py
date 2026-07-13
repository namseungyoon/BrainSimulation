from __future__ import annotations

import json
from pathlib import Path

import pytest

from ca1.params.izhikevich import (
    IzhikevichParams,
    izhikevich_params_for_cell_type,
    load_izhikevich_params,
)

_CELL_TYPES = {
    "Pyramidal",
    "PV_Basket",
    "CCK_Basket",
    "Axo",
    "Bistratified",
    "Ivy",
    "O_LM",
    "SCA",
    "Neurogliaform",
}


def _izh_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "V_m": -70.0,
        "u": -14.0,
        "V_th": 30.0,
        "a": 0.02,
        "b": 0.2,
        "c": -65.0,
        "d": 8.0,
        "I_bias": 0.0,
        "I_gain": 1.0,
        "fit_provenance": "nestgpu-fi-fit",
    }
    record.update(overrides)
    return record


def _complete_izh_fit(**overrides_by_cell: dict[str, object]) -> dict[str, object]:
    return {
        name: _izh_record(**overrides_by_cell.get(name, {}))
        for name in _CELL_TYPES
    }


class TestIzhikevichParams:
    def test_as_nest_transports_cond_beta_status_keys(self) -> None:
        params = IzhikevichParams(V_m=-70.0, u=-14.0, V_th=30.0, a=0.1, b=0.2, c=-65.0, d=2.0)

        assert params.as_nest() == {
            "V_m": -70.0,
            "u": -14.0,
            "V_th": 30.0,
            "a": 0.1,
            "b": 0.2,
            "c": -65.0,
            "d": 2.0,
            "I_e": 0.0,
            "t_ref": 0.0,
            "h_min_rel": 0.1,
            "h0_rel": 0.1,
        }

    def test_missing_fit_file_raises_not_preset_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(
            FileNotFoundError,
            match="missing Izhikevich fitted params",
        ):
            _ = load_izhikevich_params(tmp_path / "missing.json")

    def test_fast_spiking_interneurons_use_fitted_params_by_default(self) -> None:
        params = izhikevich_params_for_cell_type("PV_Basket")

        assert params.a == pytest.approx(0.083696002323701)
        assert params.b == pytest.approx(0.2802478181536967)
        assert params.c == pytest.approx(-72.84304949826722)
        assert params.d == pytest.approx(4.239480143813836)

    def test_pyramidal_cells_use_fitted_params_by_default(self) -> None:
        params = izhikevich_params_for_cell_type("Pyramidal")

        assert params.a == pytest.approx(0.028423827184016797)
        assert params.b == pytest.approx(0.10162730473703946)
        assert params.c == pytest.approx(-45.020362237801706)
        assert params.d == pytest.approx(14.483177863235783)

    def test_fitted_seed_json_sets_bias_and_current_gain(self, tmp_path: Path) -> None:
        seed_path = tmp_path / "izh_seed.json"
        seed_path.write_text(
            json.dumps(
                _complete_izh_fit(
                    Pyramidal={
                        "V_m": -70.0,
                        "u": -6.5,
                        "V_th": 26.0,
                        "a": 0.03,
                        "b": 0.09,
                        "c": -45.0,
                        "d": 14.0,
                        "I_bias": -1.25,
                        "I_gain": 0.068,
                    }
                )
            ),
            encoding="utf-8",
        )

        params = load_izhikevich_params(seed_path)

        assert params["Pyramidal"].I_e == pytest.approx(-1.25)
        assert params["Pyramidal"].current_gain == pytest.approx(0.068)

    def test_malformed_validation_passed_raises_not_preset_default(
        self,
        tmp_path: Path,
    ) -> None:
        seed_path = tmp_path / "izh_malformed_validation.json"
        seed_path.write_text(
            json.dumps(
                _complete_izh_fit(
                    Pyramidal={"validation": {"passed": "true"}}
                )
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="validation.passed must be true"):
            _ = load_izhikevich_params(seed_path)

    def test_partial_fitted_seed_json_raises_not_preset_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        seed_path = tmp_path / "izh_partial.json"
        seed_path.write_text(
            json.dumps({"Pyramidal": _izh_record()}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing Izhikevich params"):
            _ = load_izhikevich_params(seed_path)

    def test_unknown_fitted_seed_cell_type_raises_not_ignored(
        self,
        tmp_path: Path,
    ) -> None:
        seed_path = tmp_path / "izh_unknown.json"
        seed_path.write_text(
            json.dumps({"Basket": _izh_record()}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="unknown cell type 'Basket'"):
            _ = load_izhikevich_params(seed_path)

    def test_failed_fitted_seed_raises_not_preset_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        seed_path = tmp_path / "izh_failed.json"
        seed_path.write_text(
            json.dumps(
                _complete_izh_fit(PV_Basket={"fit_provenance": "FAILED"})
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="PV_Basket.*marked FAILED"):
            _ = load_izhikevich_params(seed_path)

    def test_missing_fit_provenance_raises_not_preset_default(
        self,
        tmp_path: Path,
    ) -> None:
        seed_path = tmp_path / "izh_missing_provenance.json"
        payload = _complete_izh_fit()
        record = _izh_record()
        record.pop("fit_provenance")
        payload["Pyramidal"] = record
        seed_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="fit_provenance.*required"):
            _ = load_izhikevich_params(seed_path)
