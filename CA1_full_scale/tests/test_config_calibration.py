from __future__ import annotations

import pytest


def _receptor_prefix(receptor: str) -> str:
    return receptor.split("__", maxsplit=1)[0]


def test_build_network_spec_applies_global_calibration_by_default() -> None:
    config_mod = pytest.importorskip("ca1.config")
    provenance_mod = pytest.importorskip("ca1.params.provenance")

    base = config_mod.build_network_spec({"name": "base"})
    calibrated = config_mod.build_network_spec({
        "name": "calibrated",
        "calibration": {
            "recurrent_weight_scale": 0.1,
            "recurrent_receptor_weight_scales": {"AMPA_fast": 0.2},
            "afferent_weight_scale": 0.5,
            "afferent_source_weight_scales": {"ECIII": 0.2},
        },
    })

    base_inh = next(
        p for p in base.projections if _receptor_prefix(p.receptor) == "GABA_A_fast"
    )
    calibrated_inh = next(
        p for p in calibrated.projections
        if p.pre == base_inh.pre and p.post == base_inh.post and p.receptor == base_inh.receptor
    )
    base_exc = next(p for p in base.projections if p.pre == "Pyramidal" and p.post == "SCA")
    calibrated_exc = next(
        p for p in calibrated.projections if p.pre == "Pyramidal" and p.post == "SCA"
    )
    base_eciii = next(a for a in base.afferents if a.name == "ECIII_to_Pyramidal")
    calibrated_eciii = next(a for a in calibrated.afferents if a.name == base_eciii.name)
    base_ca3 = next(a for a in base.afferents if a.name == "CA3_to_Pyramidal")
    calibrated_ca3 = next(a for a in calibrated.afferents if a.name == base_ca3.name)

    assert calibrated_inh.weight_nS == pytest.approx(base_inh.weight_nS * 0.1)
    assert calibrated_exc.weight_nS == pytest.approx(base_exc.weight_nS * 0.02)
    assert calibrated_eciii.weight_nS == pytest.approx(base_eciii.weight_nS * 0.1)
    assert calibrated_ca3.weight_nS == pytest.approx(base_ca3.weight_nS * 0.5)
    provenance = provenance_mod.parameter_provenance_for_spec(calibrated)
    assert provenance["calibration.mode"] == "paper_reduction"
    assert provenance["calibration.recurrent_weight_scale"] == "0.1"
    assert (
        provenance["calibration.recurrent_receptor_weight_scales"]
        == '{"AMPA_fast": 0.2}'
    )
    assert provenance["calibration.afferent_weight_scale"] == "0.5"
    assert (
        provenance["calibration.afferent_source_weight_scales"]
        == '{"ECIII": 0.2}'
    )


def test_build_network_spec_scales_dendritic_ampa_calibration_by_default() -> None:
    config_mod = pytest.importorskip("ca1.config")

    # Given: a compartment-aware conndata/syndata fixture with dendritic AMPA ports.
    base_config = {
        "name": "base",
        "conndata_index": 192,
        "syndata_variant": 137,
        "compartment_aware_synapses": True,
    }
    dendritic_scale = 2.5

    # When: the paper-reduction-safe dendritic AMPA scale is applied globally.
    base = config_mod.build_network_spec(base_config)
    calibrated = config_mod.build_network_spec({
        **base_config,
        "name": "calibrated",
        "calibration": {"dendritic_ampa_weight_scale": dendritic_scale},
    })

    # Then: recurrent and afferent dendritic AMPA weights are scaled, while GABA is not.
    base_recurrent = next(
        p
        for p in base.projections
        if p.pre == "Pyramidal" and p.post == "Bistratified"
    )
    calibrated_recurrent = next(
        p
        for p in calibrated.projections
        if p.pre == base_recurrent.pre
        and p.post == base_recurrent.post
        and p.receptor == base_recurrent.receptor
    )
    base_afferent = next(a for a in base.afferents if a.name == "CA3_to_Ivy")
    calibrated_afferent = next(
        a for a in calibrated.afferents if a.name == base_afferent.name
    )
    base_gaba = next(
        p for p in base.projections if _receptor_prefix(p.receptor).startswith("GABA")
    )
    calibrated_gaba = next(
        p
        for p in calibrated.projections
        if p.pre == base_gaba.pre
        and p.post == base_gaba.post
        and p.receptor == base_gaba.receptor
    )

    assert base_recurrent.receptor.endswith("__dend")
    assert _receptor_prefix(base_recurrent.receptor).startswith("AMPA")
    assert base_afferent.receptor.endswith("__dend")
    assert _receptor_prefix(base_afferent.receptor).startswith("AMPA")
    assert calibrated_recurrent.weight_nS == pytest.approx(
        base_recurrent.weight_nS * dendritic_scale
    )
    assert calibrated_afferent.weight_nS == pytest.approx(
        base_afferent.weight_nS * dendritic_scale
    )
    assert calibrated_gaba.weight_nS == pytest.approx(base_gaba.weight_nS)


def test_full_scale_source_transfer_allows_post_transfer_paper_calibration() -> None:
    config_mod = pytest.importorskip("ca1.config")
    provenance_mod = pytest.importorskip("ca1.params.provenance")

    base_config = config_mod.load_config("configs/full_scale.yaml")
    base = config_mod.build_network_spec(base_config)
    calibrated_config = {
        **base_config,
        "calibration": {
            "mode": "paper_reduction",
            "recurrent_weight_scale": 1.5,
            "recurrent_receptor_weight_scales": {"GABA_A_slow": 0.5},
            "afferent_weight_scale": 0.75,
        },
    }
    calibrated = config_mod.build_network_spec(calibrated_config)

    base_fast = next(
        p
        for p in base.projections
        if p.pre == "Axo"
        and p.post == "Pyramidal"
        and _receptor_prefix(p.receptor) == "GABA_A_fast"
    )
    calibrated_fast = next(
        p
        for p in calibrated.projections
        if p.pre == base_fast.pre
        and p.post == base_fast.post
        and p.receptor == base_fast.receptor
    )
    base_slow = next(
        p
        for p in base.projections
        if _receptor_prefix(p.receptor) == "GABA_A_slow"
    )
    calibrated_slow = next(
        p
        for p in calibrated.projections
        if p.pre == base_slow.pre
        and p.post == base_slow.post
        and p.receptor == base_slow.receptor
    )
    base_afferent = next(a for a in base.afferents if a.name == "CA3_to_Pyramidal")
    calibrated_afferent = next(
        a for a in calibrated.afferents if a.name == base_afferent.name
    )

    assert calibrated_fast.weight_nS == pytest.approx(base_fast.weight_nS * 1.5)
    assert calibrated_slow.weight_nS == pytest.approx(base_slow.weight_nS * 0.75)
    assert calibrated_afferent.weight_nS == pytest.approx(base_afferent.weight_nS * 0.75)
    provenance = provenance_mod.parameter_provenance_for_spec(calibrated)
    assert provenance["calibration.mode"] == "paper_reduction"
    assert provenance["calibration.recurrent_weight_scale"] == "1.5"
    assert (
        provenance["calibration.recurrent_receptor_weight_scales"]
        == '{"GABA_A_slow": 0.5}'
    )
    assert provenance["calibration.afferent_weight_scale"] == "0.75"
    assert provenance["source_location_transfer.table"].startswith(
        "source-location-transfer-m2-row-validation-passed"
    )


def test_build_network_spec_accepts_syndata_variant_config_key() -> None:
    config_mod = pytest.importorskip("ca1.config")

    with pytest.raises(ValueError, match="variant must be 120 or 137"):
        config_mod.build_network_spec({
            "name": "bad_syndata_alias",
            "syndata_variant": 999,
        })


def test_build_network_spec_preserves_gpu_neuron_model() -> None:
    config_mod = pytest.importorskip("ca1.config")

    spec = config_mod.build_network_spec({
        "name": "izh_candidate",
        "neuron_model": "izhikevich_cond_beta",
    })

    assert spec.neuron_model == "izhikevich_cond_beta"


def test_build_network_spec_preserves_aglif_neuron_model() -> None:
    config_mod = pytest.importorskip("ca1.config")

    spec = config_mod.build_network_spec({
        "name": "aglif_candidate",
        "neuron_model": "aglif_cond_beta",
    })

    assert spec.neuron_model == "aglif_cond_beta"


def test_build_network_spec_rejects_targeted_projection_calibration_by_default() -> None:
    config_mod = pytest.importorskip("ca1.config")

    with pytest.raises(ValueError, match="diagnostic"):
        config_mod.build_network_spec({
            "name": "overfit_projection",
            "calibration": {
                "projection_weight_scales": {"Pyramidal->SCA": 0.0},
            },
        })


def test_build_network_spec_rejects_targeted_afferent_calibration_by_default() -> None:
    config_mod = pytest.importorskip("ca1.config")

    with pytest.raises(ValueError, match="diagnostic"):
        config_mod.build_network_spec({
            "name": "overfit_afferent",
            "calibration": {
                "afferent_post_weight_scales": {"SCA": 0.3},
            },
        })


def test_build_network_spec_allows_targeted_calibration_in_diagnostic_mode() -> None:
    config_mod = pytest.importorskip("ca1.config")

    base = config_mod.build_network_spec({"name": "base"})
    calibrated = config_mod.build_network_spec({
        "name": "diagnostic",
        "calibration": {
            "mode": "diagnostic",
            "recurrent_weight_scale": 0.1,
            "projection_weight_scales": {"Pyramidal->SCA": 0.01},
            "afferent_weight_scale": 0.5,
            "afferent_post_weight_scales": {"SCA": 0.3},
        },
    })

    base_pair = next(p for p in base.projections if p.pre == "Pyramidal" and p.post == "SCA")
    calibrated_pair = next(
        p for p in calibrated.projections if p.pre == "Pyramidal" and p.post == "SCA"
    )
    base_sca = next(a for a in base.afferents if a.post == "SCA")
    calibrated_sca = next(a for a in calibrated.afferents if a.name == base_sca.name)

    assert calibrated_pair.weight_nS == pytest.approx(base_pair.weight_nS * 0.001)
    assert calibrated_sca.weight_nS == pytest.approx(base_sca.weight_nS * 0.15)
