"""Assembly point: load a YAML/dict config and build a canonical NetworkSpec.

``connectivity.json`` is the single authoritative source for:
  - population counts (populations_used)
  - recurrent projections (excitatory_connections + inhibitory_connections)
  - afferent synapse budgets (afferents section)

``neuron_parameters.json`` is authoritative for per-type AdEx parameters.
``syndata_{variant}.json`` is authoritative for receptor kinetics.

Usage
-----
    from ca1.config import load_config, build_network_spec

    # From a YAML file:
    spec = build_network_spec("path/to/ca1.yaml")

    # From a plain dict (e.g. in tests):
    spec = build_network_spec({}, scale=0.01)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import TypeAlias, assert_never, cast

from .aglif_overrides import (
    parse_aglif_dend_overrides,
    parse_aglif_gc_scale_overrides,
    parse_aglif_receive_domain_overrides,
)
from .analysis.location_transfer import apply_location_transfer, parse_transfer_mode
from .calibration import (
    CalibrationConfig,
    calibration_provenance,
    calibrated_afferent,
    calibrated_projection,
    validate_calibration,
    validate_calibration_targets,
)
from .params.neurons  import load_neuron_params
from .params.receptor_ports import parse_port_strategy, receptor_port_provenance
from .params.synapses import load_afferents, load_projections, load_receptor_config
from .types import (
    AfferentTopology,
    Afferent,
    CellType,
    ConndataCountMode,
    NetworkSpec,
    NeuronModel,
    Projection,
    ReceptorConfig,
    ReceptorTableScope,
    RecurrentTopology,
    WorkingPointMode,
    parse_neuron_model,
    parse_receptor_table_scope,
    parse_recurrent_topology,
    parse_working_point_mode,
)
from .validation.targets import MODEL_RATES_HZ

# --------------------------------------------------------------------------- #
# Explicit population counts (Bezaire 2016, Table 1 / ModelDB cellnumbers)   #
# --------------------------------------------------------------------------- #
# Keys are the CANONICAL names used throughout the package.
# These duplicate populations_used in connectivity.json for offline access;
# connectivity.json is authoritative and overrides these if present.
_FULL_SCALE_COUNTS: dict[str, int] = {
    "Pyramidal":     311500,
    "PV_Basket":       5530,
    "CCK_Basket":      3600,
    "Axo":             1470,
    "Bistratified":    2210,
    "Ivy":             8810,
    "O_LM":            1640,
    "SCA":              400,
    "Neurogliaform":   3580,
}

# JSON key -> canonical name (mirrors neurons.py alias map exactly)
_JSON_POP_TO_NAME: dict[str, str] = {
    "pyramidalcell":   "Pyramidal",
    "pvbasketcell":    "PV_Basket",
    "cckcell":         "CCK_Basket",
    "axoaxoniccell":   "Axo",
    "bistratifiedcell":"Bistratified",
    "ivycell":         "Ivy",
    "olmcell":         "O_LM",
    "scacell":         "SCA",
    "ngfcell":         "Neurogliaform",
}

# CA1 strata each population principally occupies (for CellType metadata)
_LAYERS: dict[str, tuple[str, ...]] = {
    "Pyramidal":     ("SP", "SR", "SLM"),
    "PV_Basket":     ("SP",),
    "CCK_Basket":    ("SP",),
    "Axo":           ("SP",),
    "Bistratified":  ("SP", "SR", "SO"),
    "Ivy":           ("SP", "SR", "SO"),
    "O_LM":          ("SO",),
    "SCA":           ("SR",),
    "Neurogliaform": ("SLM",),
}

_PARAMS_DIR = Path(__file__).parent / "params"
ConfigDict: TypeAlias = dict[str, object]


def _config_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"config field {field!r} must be integral, got {value!r}")
    if isinstance(value, str | int | float):
        return int(value)
    raise TypeError(f"config field {field!r} must be integral, got {value!r}")


def _config_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"config field {field!r} must be numeric, got {value!r}")
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"config field {field!r} must be numeric, got {value!r}")


def _config_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"config field {field!r} must be boolean, got {value!r}")


def _config_str(value: object, field: str) -> str:
    if isinstance(value, str):
        return value
    raise TypeError(f"config field {field!r} must be a string, got {value!r}")


def _config_path(value: object, field: str, base_dir: Path) -> Path:
    path = Path(_config_str(value, field))
    if path.is_absolute():
        return path
    candidates = (base_dir / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"config field {field!r} does not point to an existing file: {path!s}"
    )


def _config_calibration(value: object) -> CalibrationConfig:
    if isinstance(value, dict):
        return cast(CalibrationConfig, value)
    raise TypeError(f"config field 'calibration' must be a mapping, got {value!r}")


def _config_afferent_topology(value: object) -> AfferentTopology:
    raw = _config_str(value, "afferent_topology")
    if raw in (
        "compound",
        "source_pool",
        "literal_source_graph",
        "literal_source_graph_binned",
    ):
        return raw
    message = " ".join(
        (
            "config field 'afferent_topology' must be 'compound',",
            "'source_pool', 'literal_source_graph', or",
            "'literal_source_graph_binned'",
        )
    )
    raise ValueError(message)


def _config_recurrent_topology(value: object) -> RecurrentTopology:
    return parse_recurrent_topology(
        _config_str(value, "recurrent_topology")
    )


def _config_receptor_table_scope(value: object) -> ReceptorTableScope:
    return parse_receptor_table_scope(
        _config_str(value, "receptor_table_scope")
    )


def _config_working_point_mode(value: object) -> WorkingPointMode:
    return parse_working_point_mode(
        _config_str(value, "working_point_mode")
    )


def _config_working_point_clamp_rates_hz(
    value: object | None,
    *,
    mode: WorkingPointMode,
    cell_types: Mapping[str, CellType],
) -> Mapping[str, float]:
    if value is None:
        if mode == "off":
            return {}
        value = "table5"
    if value == "table5":
        return {
            cell_type: float(rate_hz)
            for cell_type, rate_hz in MODEL_RATES_HZ.items()
            if cell_type != "Pyramidal"
        }
    if not isinstance(value, Mapping):
        raise TypeError(
            "config field 'working_point_clamp_rates_hz' must be a mapping "
            "or 'table5'"
        )
    if any(not isinstance(cell_type, str) for cell_type in value):
        raise TypeError(
            "config field 'working_point_clamp_rates_hz' keys must be strings"
        )

    unknown = sorted(set(value) - set(cell_types))
    if unknown:
        raise ValueError(
            "config field 'working_point_clamp_rates_hz' contains unknown "
            f"cell types: {unknown}"
        )
    rates: dict[str, float] = {}
    for cell_type, raw_rate in value.items():
        assert isinstance(cell_type, str)
        rate_hz = _config_float(
            raw_rate,
            f"working_point_clamp_rates_hz.{cell_type}",
        )
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError(
                "config field 'working_point_clamp_rates_hz' rates must be "
                f"finite and positive, got {cell_type}={raw_rate!r}"
            )
        rates[cell_type] = rate_hz
    return rates


def _config_conndata_count_mode(value: object) -> ConndataCountMode:
    raw = _config_str(value, "conndata_count_mode")
    if raw in ("network_total", "per_cell"):
        return raw
    raise ValueError(
        "config field 'conndata_count_mode' must be 'network_total' or 'per_cell'"
    )


def _config_positive_int(value: object, field: str) -> int:
    parsed = _config_int(value, field)
    if parsed < 1:
        raise ValueError(f"config field {field!r} must be positive, got {value!r}")
    return parsed


def _config_nonnegative_float(value: object, field: str) -> float:
    parsed = _config_float(value, field)
    if parsed < 0.0:
        raise ValueError(f"config field {field!r} must be nonnegative, got {value!r}")
    return parsed

# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def load_config(path: str | Path | None = None) -> ConfigDict:
    """Load a YAML configuration file and return it as a plain dict.

    Falls back gracefully if PyYAML is not installed -- in that case the
    caller may pass a dict path directly to :func:`build_network_spec`.

    Parameters
    ----------
    path:
        Path to a ``.yaml`` / ``.yml`` file.  Pass ``None`` to get an empty
        dict suitable for default-only operation.

    Returns
    -------
    dict
        Parsed YAML content.  Empty dict if ``path`` is ``None``.

    Raises
    ------
    FileNotFoundError
        If a path is given but the file does not exist.
    ImportError
        If PyYAML is unavailable and a non-None path is given.
    """
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML config files. Install it with: pip install pyyaml"
        ) from exc

    with open(path, "r", encoding="utf-8") as fh:
        loaded_raw = cast(object, yaml.safe_load(fh))
    if loaded_raw is None:
        return {}
    if not isinstance(loaded_raw, dict):
        raise TypeError(f"Config file must contain a mapping: {path}")
    loaded_mapping = cast(dict[object, object], loaded_raw)
    return {str(key): value for key, value in loaded_mapping.items()}


def _resolve_counts(config: ConfigDict) -> dict[str, int]:
    """Extract per-type full-scale counts from config or connectivity.json.

    Config ``cell_types`` block (if present) takes priority over the JSON
    defaults so that users can override individual counts in YAML.
    """
    counts: dict[str, int] = dict(_FULL_SCALE_COUNTS)

    conndata_index, cellnumbers_index = _connectivity_indices(config)
    if conndata_index is not None or "cellnumbers_index" in config:
        from .extract.modeldb_tables import extract_cellnumbers

        for canonical, count in extract_cellnumbers(index=cellnumbers_index).items():
            if canonical in _FULL_SCALE_COUNTS:
                counts[canonical] = count

    # Override from connectivity.json populations_used
    conn_path = _PARAMS_DIR / "connectivity.json"
    if conndata_index is None and conn_path.exists():
        with open(conn_path, "r", encoding="utf-8") as fh:
            conn = cast(ConfigDict, json.load(fh))
        pops = cast(ConfigDict, conn.get("populations_used", {}))
        for json_key, canonical in _JSON_POP_TO_NAME.items():
            if json_key in pops:
                pop_value: object = pops[json_key]
                counts[canonical] = _config_int(
                    pop_value, f"populations_used.{json_key}"
                )

    # Finally, apply any YAML-level overrides
    cell_types_config = cast(ConfigDict, config.get("cell_types", {}))
    for name, block in cell_types_config.items():
        if isinstance(block, dict) and "count" in block:
            block_mapping = cast(dict[object, object], block)
            count_value = block_mapping["count"]
            counts[str(name)] = _config_int(
                count_value,
                f"cell_types.{name}.count",
            )

    return counts


def _connectivity_indices(config: ConfigDict) -> tuple[int | None, int]:
    conndata_raw = config.get("conndata_index", config.get("connectivity_index"))
    conndata_index = (
        None if conndata_raw is None else _config_int(conndata_raw, "conndata_index")
    )
    cellnumbers_index = _config_int(config.get("cellnumbers_index", 101), "cellnumbers_index")
    return conndata_index, cellnumbers_index


def _validate_conndata_count_mode(
    conndata_index: int | None,
    conndata_count_mode: ConndataCountMode,
) -> None:
    if conndata_index == 430 and conndata_count_mode != "per_cell":
        raise ValueError(
            f"conndata_430 is the paper Table 1 per-cell convergence table; set conndata_count_mode='per_cell' instead of silently treating it as {conndata_count_mode!r}"
        )


def _load_spec_neuron_params(neuron_model: NeuronModel):
    if neuron_model == "aeif_cond_beta_multisynapse":
        return load_neuron_params()

    analytic = load_neuron_params(_PARAMS_DIR / "neuron_parameters.json")
    marker = f"unused-by-{neuron_model}-runtime"
    return {
        name: replace(params, fit_provenance=marker)
        for name, params in analytic.items()
    }


def _used_receptors_by_post(
    cell_types: Mapping[str, CellType],
    projections: list[Projection],
    afferents: list[Afferent],
) -> dict[str, set[str]]:
    used = {name: set() for name in cell_types}
    for proj in projections:
        used.setdefault(proj.post, set()).add(proj.receptor)
    for aff in afferents:
        used.setdefault(aff.post, set()).add(aff.receptor)
    return used


def _target_receptor_tables(
    receptors: ReceptorConfig,
    used_by_post: Mapping[str, set[str]],
) -> Mapping[str, ReceptorConfig]:
    source_names = set(receptors.names)
    tables: dict[str, ReceptorConfig] = {}
    for post, used_names in used_by_post.items():
        missing = sorted(used_names - source_names)
        if missing:
            raise KeyError(
                f"receptor names for post target {post!r} are absent from "
                f"the global receptor table: {missing}"
            )
        names = tuple(name for name in receptors.names if name in used_names)
        indices = tuple(receptors.port_index(name) for name in names)
        tables[post] = ReceptorConfig(
            names=names,
            E_rev=tuple(receptors.E_rev[index] for index in indices),
            tau_rise=tuple(receptors.tau_rise[index] for index in indices),
            tau_decay=tuple(receptors.tau_decay[index] for index in indices),
        )
    return tables


def build_network_spec(
    config: ConfigDict | str | Path,
    scale: float = 1.0,
    seed: int = 12345,
) -> NetworkSpec:
    """Assemble the canonical ``NetworkSpec`` from config + authoritative data.

    ``connectivity.json`` is the single authoritative source for population
    counts, recurrent projections, and afferent synapse budgets.
    ``neuron_parameters.json`` is authoritative for intrinsic parameters.
    ``syndata_120.json`` is the default receptor-kinetics variant.

    All 9 CA1 types (including Neurogliaform) are always built.  A
    ``ValueError`` is raised if any type is missing from the parameter files.

    Parameters
    ----------
    config:
        Either a pre-loaded dict (e.g. from :func:`load_config`) or a path to
        a YAML file (loaded automatically).
    scale:
        Network scale factor applied to population counts.
        1.0 = full Bezaire (311 500 + interneurons).
        <1.0 = reduced for debugging; see ``ca1.build.downscale``.
    seed:
        Global RNG seed propagated into the NetworkSpec.

    Returns
    -------
    NetworkSpec
        Fully assembled spec; feed directly into any SimulatorBackend.

    Raises
    ------
    ValueError
        If any of the 9 expected cell types is missing from parameter files,
        or if a receptor variant is invalid.
    """
    # Normalise config to dict
    config_dir = Path.cwd()
    if isinstance(config, (str, Path)):
        config_dir = Path(config).parent
        config = load_config(config)

    # --- Receptor kinetics -------------------------------------------------- #
    syn_variant = _config_int(
        config.get("synapse_variant", config.get("syndata_variant", 120)),
        "synapse_variant",
    )
    compartment_aware = _config_bool(
        config.get("compartment_aware_synapses", False),
        "compartment_aware_synapses",
    )
    receptor_port_strategy = parse_port_strategy(
        _config_str(
            config.get("receptor_port_strategy", "budget_weighted"),
            "receptor_port_strategy",
        )
    )
    receptor_table_scope = _config_receptor_table_scope(
        config.get("receptor_table_scope", "global")
    )
    match receptor_table_scope:
        case "global":
            runtime_port_strategy = receptor_port_strategy
        case "per_target":
            runtime_port_strategy = "exact"
        case unreachable:
            assert_never(unreachable)
    afferent_topology = _config_afferent_topology(
        config.get("afferent_topology", "compound")
    )
    recurrent_topology = _config_recurrent_topology(
        config.get("recurrent_topology", "fixed_indegree")
    )
    afferent_source_pool_size = _config_positive_int(
        config.get("afferent_source_pool_size", 4096),
        "afferent_source_pool_size",
    )
    afferent_source_pool_indegree = _config_positive_int(
        config.get("afferent_source_pool_indegree", 64),
        "afferent_source_pool_indegree",
    )
    afferent_source_rate_cv = _config_nonnegative_float(
        config.get("afferent_source_rate_cv", 0.0),
        "afferent_source_rate_cv",
    )
    receptors = load_receptor_config(
        variant=syn_variant,
        compartment_aware=compartment_aware,
        port_strategy=runtime_port_strategy,
    )
    neuron_model = parse_neuron_model(
        _config_str(
            config.get("neuron_model", "aeif_cond_beta_multisynapse"),
            "neuron_model",
        )
    )
    working_point_mode = _config_working_point_mode(
        config.get("working_point_mode", "off")
    )

    # --- Neuron parameters -------------------------------------------------- #
    neuron_params = _load_spec_neuron_params(neuron_model)

    # --- Population counts -------------------------------------------------- #
    conndata_index, cellnumbers_index = _connectivity_indices(config)
    conndata_count_mode = _config_conndata_count_mode(
        config.get("conndata_count_mode", "network_total")
    )
    _validate_conndata_count_mode(conndata_index, conndata_count_mode)
    counts = _resolve_counts(config)

    # --- CellType objects (all 9 types, no exceptions swallowed) ------------ #
    cell_types: dict[str, CellType] = {}
    for name in _FULL_SCALE_COUNTS:  # iterate in canonical order
        if name not in neuron_params:
            raise ValueError(
                f"build_network_spec: neuron params missing for type '{name}'. Cannot build NetworkSpec without all 9 types."
            )
        if name not in counts:
            raise ValueError(
                f"build_network_spec: population count missing for type '{name}'."
            )
        cell_types[name] = CellType(
            name   = name,
            count  = counts[name],
            layers = _LAYERS.get(name, ()),
            params = neuron_params[name],
        )

    working_point_clamp_rates_hz = _config_working_point_clamp_rates_hz(
        config.get("working_point_clamp_rates_hz"),
        mode=working_point_mode,
        cell_types=cell_types,
    )

    aglif_receive_domain_overrides = parse_aglif_receive_domain_overrides(
        config.get("aglif_receive_domain_overrides"),
        cell_types,
    )
    aglif_gc_scale_overrides = parse_aglif_gc_scale_overrides(
        config.get("aglif_gc_scale_overrides"),
        cell_types,
    )
    aglif_dend_overrides = parse_aglif_dend_overrides(
        config.get("aglif_dend_overrides"),
        cell_types,
    )

    calibration = _config_calibration(config.get("calibration", {}))
    validate_calibration(calibration)

    # --- Recurrent projections ---------------------------------------------- #
    raw_projections = load_projections(
        conndata_index=conndata_index,
        cellnumbers_index=cellnumbers_index,
        conndata_count_mode=conndata_count_mode,
        synapse_variant=syn_variant,
        compartment_aware=compartment_aware,
        port_strategy=runtime_port_strategy,
    )

    # --- Afferent drives ---------------------------------------------------- #
    afferent_rate = _config_float(config.get("afferent_rate_hz", 0.65), "afferent_rate_hz")
    raw_afferents = load_afferents(
        rate_hz=afferent_rate,
        conndata_index=conndata_index,
        cellnumbers_index=cellnumbers_index,
        conndata_count_mode=conndata_count_mode,
        synapse_variant=syn_variant,
        compartment_aware=compartment_aware,
        port_strategy=runtime_port_strategy,
    )
    validate_calibration_targets(calibration, raw_projections, raw_afferents)
    match receptor_table_scope:
        case "global":
            target_receptors: Mapping[str, ReceptorConfig] = {}
        case "per_target":
            target_receptors = _target_receptor_tables(
                receptors,
                _used_receptors_by_post(cell_types, raw_projections, raw_afferents),
            )
        case unreachable:
            assert_never(unreachable)
    # --- Assemble ----------------------------------------------------------- #
    name = _config_str(config.get("name", "CA1_Bezaire2016"), "name")

    spec = NetworkSpec(
        name              = name,
        cell_types        = cell_types,
        projections       = raw_projections,
        afferents         = raw_afferents,
        receptors         = receptors,
        target_receptors  = target_receptors,
        receptor_table_scope = receptor_table_scope,
        neuron_model      = neuron_model,
        scale             = scale,
        seed              = seed,
        aglif_receive_domain_overrides = aglif_receive_domain_overrides,
        aglif_gc_scale_overrides = aglif_gc_scale_overrides,
        aglif_dend_overrides = aglif_dend_overrides,
        calibration_provenance = calibration_provenance(calibration),
        receptor_provenance = receptor_port_provenance(
            syn_variant,
            compartment_aware,
            runtime_port_strategy, receptors,
        ),
        afferent_topology = afferent_topology,
        recurrent_topology = recurrent_topology,
        afferent_source_pool_size = afferent_source_pool_size,
        afferent_source_pool_indegree = afferent_source_pool_indegree,
        afferent_source_rate_cv = afferent_source_rate_cv,
        conndata_index = conndata_index,
        cellnumbers_index = cellnumbers_index,
        conndata_count_mode = conndata_count_mode,
        weight_compensation = 1.0,  # full scale; downscale module adjusts this
        working_point_mode = working_point_mode,
        working_point_clamp_rates_hz = working_point_clamp_rates_hz,
    )
    transferred = _apply_source_location_transfer(
        spec,
        config=config,
        config_dir=config_dir,
    )
    calibrated = _apply_calibration(transferred, calibration)
    from .source_grounded_stack import apply_source_grounded_stack

    return apply_source_grounded_stack(
        calibrated,
        config.get("source_grounded_stack"),
        config_dir=config_dir,
    )


def _apply_calibration(
    spec: NetworkSpec,
    calibration: CalibrationConfig,
) -> NetworkSpec:
    return replace(
        spec,
        projections=[
            calibrated_projection(proj, calibration)
            for proj in spec.projections
        ],
        afferents=[
            calibrated_afferent(aff, calibration)
            for aff in spec.afferents
        ],
    )


def _apply_source_location_transfer(
    spec: NetworkSpec,
    *,
    config: ConfigDict,
    config_dir: Path,
) -> NetworkSpec:
    raw_mode = config.get("source_location_transfer_mode", "none")
    mode = parse_transfer_mode(
        _config_str(raw_mode, "source_location_transfer_mode")
    )
    if mode == "none":
        return spec

    if "source_location_transfer_table" not in config:
        raise ValueError(
            "source_location_transfer_table is required when source_location_transfer_mode is not 'none'"
        )
    table = _config_path(
        config["source_location_transfer_table"],
        "source_location_transfer_table",
        config_dir,
    )
    updated, _, missing = apply_location_transfer(spec, mode, table)
    if missing:
        raise ValueError(
            f"source-location transfer has missing rows after strict apply: {missing}"
        )
    return updated
