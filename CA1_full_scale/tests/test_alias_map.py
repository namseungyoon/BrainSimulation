"""Tests that verify the alias-map / name-mapping correctness.

These tests do NOT import nest or nestgpu -- they inspect source text and
attempt pure-Python imports only, skipping gracefully when impossible.

Bug guarded against
-------------------
The old run_scaled_bezaire.py:55 used:
    pop.replace('cell', '').capitalize()
to map population names to neuron-parameter keys. This silently mismapped
interneurons to pyramidal-cell parameters, producing silent/wrong networks.

Requirements verified here
--------------------------
1. ca1/config.py contains an explicit alias dict mapping lowercase BSB-style
   cell type names (e.g. 'pyramidalcell') to canonical param keys (e.g.
   'Pyramidal').
2. The NestBackend source does NOT contain the .replace('cell','').capitalize()
   heuristic.
3. load_neuron_params returns all 9 cell types with Title-case keys.
4. O-LM g_L is correct (~3.735 nS, not ~0.56 nS).
5. build_network_spec raises when a neuron-parameter type is missing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Source file paths
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent / "src" / "ca1"
_BACKEND_SRC = _SRC / "sim" / "nest_backend.py"
_CONFIG_SRC = _SRC / "config.py"


def _backend_source() -> str:
    if not _BACKEND_SRC.exists():
        pytest.skip("ca1/sim/nest_backend.py not yet written by the backend slice")
    return _BACKEND_SRC.read_text()


def _config_source() -> str:
    if not _CONFIG_SRC.exists():
        pytest.skip("ca1/config.py not yet written")
    return _CONFIG_SRC.read_text()


# ---------------------------------------------------------------------------
# 1. config.py must have an explicit alias dict (not a string heuristic)
# ---------------------------------------------------------------------------

class TestConfigAliasMap:
    def test_config_has_explicit_alias_dict(self) -> None:
        """config.py must map BSB-style names to canonical param keys via a dict."""
        src = _config_source()
        # The config slice uses a dict literal mapping e.g. "pyramidalcell" -> "Pyramidal"
        has_alias = (
            re.search(r'"pyramidalcell"\s*:', src) is not None
            or re.search(r"'pyramidalcell'\s*:", src) is not None
        )
        assert has_alias, (
            "config.py must contain an explicit dict mapping 'pyramidalcell' -> 'Pyramidal' "
            "(or similar). A string heuristic like .replace('cell','').capitalize() is forbidden."
        )

    def test_config_no_replace_heuristic(self) -> None:
        """config.py must not use .replace('cell','') for name mapping."""
        src = _config_source()
        pattern = re.compile(r"\.replace\s*\(\s*['\"]cell['\"]")
        matches = pattern.findall(src)
        assert not matches, (
            "Found forbidden .replace('cell',...) heuristic in config.py. "
            "Use an explicit alias dict instead."
        )


# ---------------------------------------------------------------------------
# 2. NestBackend source must not use the heuristic
# ---------------------------------------------------------------------------

class TestNoReplaceHeuristic:
    def test_replace_cell_heuristic_absent(self) -> None:
        """nest_backend.py must not use .replace('cell','').capitalize()."""
        src = _backend_source()
        pattern = re.compile(r"\.replace\s*\(\s*['\"]cell['\"]")
        matches = pattern.findall(src)
        assert not matches, (
            "Found forbidden .replace('cell',...) heuristic in nest_backend.py. "
            "Use an explicit alias dict instead."
        )

    def test_no_bare_capitalize_after_replace(self) -> None:
        """No pop.replace(...).capitalize() chain."""
        src = _backend_source()
        combined = re.compile(
            r"\.replace\s*\(\s*['\"]cell['\"].*?\.capitalize\(\)",
            re.DOTALL,
        )
        assert not combined.search(src), (
            "Found .replace('cell',...).capitalize() chained call in nest_backend.py"
        )


# ---------------------------------------------------------------------------
# 3. load_neuron_params: all 9 types, Title-case keys
# ---------------------------------------------------------------------------

class TestNeuronParamsAllNineTypes:
    def test_load_neuron_params_has_all_nine_types(self) -> None:
        """All 9 Bezaire cell types must be present (including Neurogliaform)."""
        neurons_mod = pytest.importorskip("ca1.params.neurons")

        params = neurons_mod.load_neuron_params()
        # Canonical Title-case keys used by config.py and the rest of the package
        expected_types = {
            "Pyramidal", "PV_Basket", "CCK_Basket", "Axo",
            "Bistratified", "Ivy", "O_LM", "SCA", "Neurogliaform",
        }
        missing = expected_types - set(params.keys())
        assert not missing, f"Missing cell types in neuron params: {missing}"

    def test_neurogliaform_present(self) -> None:
        """Neurogliaform was the silently-dropped 9th type in the original bug."""
        neurons_mod = pytest.importorskip("ca1.params.neurons")
        params = neurons_mod.load_neuron_params()
        assert "Neurogliaform" in params, (
            "Neurogliaform missing from neuron params (was silently dropped in original bug)"
        )

    def test_load_neuron_params_raises_for_unknown_key(self) -> None:
        """Accessing an unknown type via [] must raise, not return None."""
        neurons_mod = pytest.importorskip("ca1.params.neurons")
        params = neurons_mod.load_neuron_params()
        with pytest.raises((KeyError, ValueError)):
            _ = params["DEFINITELY_NOT_A_CELL_TYPE_XYZZY_12345"]

    def test_default_fitted_file_has_no_failed_records(self) -> None:
        neurons_mod = pytest.importorskip("ca1.params.neurons")
        fitted_path = Path(neurons_mod.__file__).parent / "neuron_parameters_fitted.json"
        analytic_path = Path(neurons_mod.__file__).parent / "neuron_parameters.json"
        if not fitted_path.exists():
            pytest.skip("no fitted neuron-parameter file present")

        raw = json.loads(fitted_path.read_text(encoding="utf-8"))
        expected = set(neurons_mod.load_neuron_params(analytic_path))
        assert set(raw) == expected, (
            "Default fitted params must cover every cell type; missing records "
            f"would silently fall back to analytic values: {sorted(expected - set(raw))}"
        )
        failed = [
            name
            for name, record in raw.items()
            if isinstance(record, dict)
            and (
                record.get("fit_provenance") == "FAILED"
                or (
                    isinstance(record.get("validation"), dict)
                    and record["validation"].get("passed") is False
                )
            )
        ]
        assert not failed, (
            "Default fitted params must contain only validated records; failed "
            f"records hide invalid fits behind fallback: {failed}"
        )

    def test_failed_fitted_entries_raise_not_fallback(self, tmp_path: Path) -> None:
        neurons_mod = pytest.importorskip("ca1.params.neurons")
        analytic_path = Path(neurons_mod.__file__).parent / "neuron_parameters.json"
        fallback = neurons_mod.load_neuron_params(analytic_path)
        fitted_path = tmp_path / "neuron_parameters_fitted.json"
        _ = fitted_path.write_text(
            json.dumps(
                {
                    "Pyramidal": {
                        "C_m": 1.0,
                        "g_L": 1.0,
                        "E_L": -65.0,
                        "V_th": -50.0,
                        "V_reset": -65.0,
                        "Delta_T": 1.0,
                        "a": 0.0,
                        "b": 0.0,
                        "tau_w": 100.0,
                        "t_ref": 2.0,
                        "fit_provenance": "FAILED",
                    }
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Pyramidal.*marked FAILED"):
            _ = neurons_mod._load_fitted(fitted_path, fallback)

    def test_missing_fit_provenance_raises_not_default(
        self,
        tmp_path: Path,
    ) -> None:
        neurons_mod = pytest.importorskip("ca1.params.neurons")
        analytic_path = Path(neurons_mod.__file__).parent / "neuron_parameters.json"
        fallback = neurons_mod.load_neuron_params(analytic_path)
        axo = fallback["Axo"]
        fitted_path = tmp_path / "neuron_parameters_fitted.json"
        _ = fitted_path.write_text(
            json.dumps(
                {
                    name: {
                        "C_m": params.C_m,
                        "g_L": params.g_L,
                        "E_L": params.E_L,
                        "V_th": params.V_th,
                        "V_reset": params.V_reset,
                        "Delta_T": params.Delta_T,
                        "a": params.a,
                        "b": params.b,
                        "tau_w": params.tau_w,
                        "t_ref": params.t_ref,
                        "V_peak": params.V_peak,
                        "fit_provenance": "nest-validated",
                    }
                    for name, params in fallback.items()
                }
            ),
            encoding="utf-8",
        )
        raw = json.loads(fitted_path.read_text(encoding="utf-8"))
        raw["Axo"] = {
            "C_m": axo.C_m,
            "g_L": axo.g_L,
            "E_L": axo.E_L,
            "V_th": axo.V_th,
            "V_reset": axo.V_reset,
            "Delta_T": axo.Delta_T,
            "a": axo.a,
            "b": axo.b,
            "tau_w": axo.tau_w,
            "t_ref": axo.t_ref,
            "V_peak": axo.V_peak,
        }
        _ = fitted_path.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ValueError, match="fit_provenance.*required"):
            _ = neurons_mod._load_fitted(fitted_path, fallback)

    def test_missing_fitted_entries_raise_not_fallback(self, tmp_path: Path) -> None:
        neurons_mod = pytest.importorskip("ca1.params.neurons")
        analytic_path = Path(neurons_mod.__file__).parent / "neuron_parameters.json"
        fallback = neurons_mod.load_neuron_params(analytic_path)
        axo = fallback["Axo"]
        fitted_path = tmp_path / "neuron_parameters_fitted.json"
        _ = fitted_path.write_text(
            json.dumps(
                {
                    "Axo": {
                        "C_m": axo.C_m,
                        "g_L": axo.g_L,
                        "E_L": axo.E_L,
                        "V_th": axo.V_th,
                        "V_reset": axo.V_reset,
                        "Delta_T": axo.Delta_T,
                        "a": axo.a,
                        "b": axo.b,
                        "tau_w": axo.tau_w,
                        "t_ref": axo.t_ref,
                        "V_peak": axo.V_peak,
                        "fit_provenance": "nest-validated",
                    }
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing fitted neuron params"):
            _ = neurons_mod._load_fitted(fitted_path, fallback)

    def test_missing_fitted_v_peak_raises_not_default(
        self,
        tmp_path: Path,
    ) -> None:
        neurons_mod = pytest.importorskip("ca1.params.neurons")
        analytic_path = Path(neurons_mod.__file__).parent / "neuron_parameters.json"
        fallback = neurons_mod.load_neuron_params(analytic_path)
        fitted_path = tmp_path / "neuron_parameters_fitted.json"
        records = {
            name: {
                "C_m": params.C_m,
                "g_L": params.g_L,
                "E_L": params.E_L,
                "V_th": params.V_th,
                "V_reset": params.V_reset,
                "Delta_T": params.Delta_T,
                "a": params.a,
                "b": params.b,
                "tau_w": params.tau_w,
                "t_ref": params.t_ref,
                "V_peak": params.V_peak,
                "fit_provenance": "nest-validated",
            }
            for name, params in fallback.items()
        }
        del records["Axo"]["V_peak"]
        _ = fitted_path.write_text(json.dumps(records), encoding="utf-8")

        with pytest.raises(KeyError, match="V_peak"):
            _ = neurons_mod._load_fitted(fitted_path, fallback)

    def test_validated_fitted_entries_remain_preferred(self) -> None:
        neurons_mod = pytest.importorskip("ca1.params.neurons")
        params = neurons_mod.load_neuron_params()
        assert params["O_LM"].fit_provenance == "nest-validated"


# ---------------------------------------------------------------------------
# 4. O-LM g_L correctness
# ---------------------------------------------------------------------------

class TestOlmGl:
    def _analytic_path(self):
        from pathlib import Path
        import ca1.params.neurons as n
        return Path(n.__file__).parent / "neuron_parameters.json"

    def test_olm_g_l_correct(self) -> None:
        """The ANALYTIC O-LM g_L must be ~3.735 nS (bug 7: old code used ~0.56 nS).

        The default loader now returns the CMA-ES-fitted params (where g_L is a
        free, fit value), so this checks the analytic source file explicitly.
        """
        neurons_mod = pytest.importorskip("ca1.params.neurons")
        params = neurons_mod.load_neuron_params(self._analytic_path())
        olm = params.get("O_LM")
        assert olm is not None, "O_LM key not found in neuron params"
        assert abs(olm.g_L - 3.735) < 0.5, (
            f"analytic O-LM g_L = {olm.g_L} nS; expected ~3.735 nS "
            "(old bug was ~0.56 nS)."
        )

    def test_olm_g_l_not_near_old_buggy_value(self) -> None:
        """Confirm analytic g_L is not near the buggy 0.56 nS value."""
        neurons_mod = pytest.importorskip("ca1.params.neurons")
        olm = neurons_mod.load_neuron_params(self._analytic_path())["O_LM"]
        assert olm.g_L > 1.0, (
            f"O-LM g_L = {olm.g_L} nS is suspiciously low (old bug was ~0.56 nS)"
        )


# ---------------------------------------------------------------------------
# 5. build_network_spec always builds all 9 types
# ---------------------------------------------------------------------------

class TestBuildNetworkSpecCompleteness:
    def test_build_network_spec_includes_all_nine_types(self) -> None:
        """build_network_spec must return a NetworkSpec with all 9 cell types."""
        config_mod = pytest.importorskip("ca1.config")

        # Minimal valid config (no cell_types override -- uses canonical counts)
        spec = config_mod.build_network_spec({"name": "test_complete"})
        assert len(spec.cell_types) == 9, (
            f"Expected 9 cell types, got {len(spec.cell_types)}: "
            f"{list(spec.cell_types.keys())}"
        )

    def test_build_network_spec_includes_neurogliaform(self) -> None:
        """Neurogliaform must never be silently dropped."""
        config_mod = pytest.importorskip("ca1.config")
        spec = config_mod.build_network_spec({"name": "test_ngf"})
        assert "Neurogliaform" in spec.cell_types, (
            "Neurogliaform missing from built NetworkSpec (was the silently-dropped 9th type)"
        )

    def test_build_network_spec_raises_on_missing_neuron_type(self) -> None:
        """build_network_spec must raise if a neuron-parameter type is absent.

        config.py imports load_neuron_params with:
            from ca1.params.neurons import load_neuron_params
        so we must patch the reference in ca1.config, not in ca1.params.neurons.
        """
        pytest.importorskip("ca1.config")
        pytest.importorskip("ca1.params.neurons")

        import ca1.params.neurons as neurons_module
        import ca1.config as config_module

        original = neurons_module.load_neuron_params
        original_in_config = config_module.load_neuron_params  # type: ignore[attr-defined]

        def _incomplete_params(path=None):
            full = original(path)
            # Remove Neurogliaform to simulate the original build bug
            return {k: v for k, v in full.items() if k != "Neurogliaform"}

        # Patch both the source module and the reference bound into config's namespace
        neurons_module.load_neuron_params = _incomplete_params
        config_module.load_neuron_params = _incomplete_params  # type: ignore[attr-defined]
        try:
            with pytest.raises((ValueError, KeyError)):
                config_module.build_network_spec({"name": "test_missing"})
        finally:
            neurons_module.load_neuron_params = original
            config_module.load_neuron_params = original_in_config  # type: ignore[attr-defined]
