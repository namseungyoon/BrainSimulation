"""Shared HOC-name-to-canonical-type mapping for CA1 ModelDB extraction.

The ModelDB conndata and cellnumbers files use HOC-script identifiers
(e.g. ``pyramidalcell``, ``pvbasketcell``).  Every parser in this package
normalises them through the single alias table defined here so there is
no duplicated mapping logic that could drift.

Usage::

    from ca1.extract.connectivity import canonical_name, AFFERENT_TYPES

    canonical = canonical_name("pvbasketcell")   # -> "PV_Basket"
    canonical_name("unknown")                    # raises KeyError with message

Note on receptor assignment
---------------------------
* Pyramidal -> * : AMPA_fast
* CA3 -> *       : AMPA_fast  (Schaffer collaterals)
* ECIII -> *     : AMPA_slow  (perforant path; syndata entries use slower
                               MyExp2Sid kinetics, typically tau_rise 2.0 ms /
                               tau_decay 6.3 ms with E_rev 0 mV.)
* Interneurons -> * : determined by syndata kinetics; here we return "GABA_A_fast"
                      as the structural default; syndata.py carries the per-pair
                      ground truth.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# HOC identifier -> canonical population name
# ---------------------------------------------------------------------------

_HOC_TO_CANONICAL: dict[str, str] = {
    "pyramidalcell":   "Pyramidal",
    "pvbasketcell":    "PV_Basket",
    "cckcell":         "CCK_Basket",
    "axoaxoniccell":   "Axo",
    "bistratifiedcell": "Bistratified",
    "ivycell":         "Ivy",
    "olmcell":         "O_LM",
    "ngfcell":         "Neurogliaform",
    "scacell":         "SCA",
    # External afferent sources – kept in the mapping so the same helper
    # covers both CA1-internal populations and afferents.
    "ca3cell":         "CA3",
    "eccell":          "ECIII",
    # Alternative HOC identifiers found in some SimTracker variants
    "poolosyncell":    "Pyramidal",   # alternative pyramidal label
    "ppspont":         "Poisson",     # generic spontaneous Poisson source
}

# Population names that represent afferent (external) drive
AFFERENT_TYPES: frozenset[str] = frozenset({"CA3", "ECIII"})

# Population names that represent internal CA1 cell types (9 types)
CA1_INTERNAL_TYPES: frozenset[str] = frozenset({
    "Pyramidal", "PV_Basket", "CCK_Basket", "Axo",
    "Bistratified", "Ivy", "O_LM", "Neurogliaform", "SCA",
})


def canonical_name(hoc_name: str) -> str:
    """Return the canonical population name for a HOC identifier.

    Parameters
    ----------
    hoc_name:
        Lower-case HOC cell-type string as it appears in conndata / cellnumbers.

    Returns
    -------
    str
        The canonical name used throughout the ``ca1`` package.

    Raises
    ------
    KeyError
        If ``hoc_name`` is not present in the alias table.  This is intentional
        (see RECOVERY_PLAN bug #2): a silent fall-through would silently drop or
        mislabel populations as in the old ``pop.replace('cell','').capitalize()``
        pattern.
    """
    try:
        return _HOC_TO_CANONICAL[hoc_name.lower()]
    except KeyError:
        known = ", ".join(sorted(_HOC_TO_CANONICAL))
        raise KeyError(
            f"HOC name {hoc_name!r} not found in alias table. "
            f"Known identifiers: {known}"
        ) from None


def is_afferent(canonical: str) -> bool:
    """Return True if *canonical* is an external afferent source."""
    return canonical in AFFERENT_TYPES


def is_ca1_internal(canonical: str) -> bool:
    """Return True if *canonical* is one of the 9 CA1 internal types."""
    return canonical in CA1_INTERNAL_TYPES
