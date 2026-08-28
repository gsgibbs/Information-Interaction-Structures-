"""
src/pipeline.py
===============
Top-level IIS pipeline orchestration.

This module composes the entropy estimators (src/estimators.py) and
Möbius decomposition (src/mobius.py) into the full Information
Interaction Structure (IIS) pipeline:

    1. Build entropy table over all variable subsets
    2. Möbius inversion → interaction coefficients M(S)
    3. Aggregate by order → raw architecture mass A₂–A₅
    4. Null calibration → corrected mass A₂*–A₅*
    5. Normalize → IIS fingerprint p₂, p₃, p₄, p₅
    6. Compute architecture entropy H_IIS

Public API
----------
    run_iis(df, is_discrete, null_mass, ...)  →  full pipeline
    iis_entropy(iis)                          →  H_IIS scalar
"""

import numpy as np
from itertools import combinations

from .estimators import copula_transform, entropy
from .mobius import mobius_inversion, architecture_mass


# ─────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────

def run_iis(df, is_discrete, null_mass, orders=(2, 3, 4, 5), k=5, verbose=False):
    """
    Run the full IIS pipeline on a single dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input data, shape (n_samples, n_variables).
    is_discrete : bool
        True for integer/binary data (uses plugin estimator);
        False for continuous data (uses k-NN estimator after
        copula transform).
    null_mass : dict or None
        Architecture mass from the type-matched null dataset —
        output of ``architecture_mass()`` on the null.
        Pass None to skip null calibration (raw mass is returned
        as corrected mass).
    orders : tuple of int, optional
        Interaction orders to include (default: 2, 3, 4, 5).
        Must not include 1 (singleton; carries no interaction).
    k : int, optional
        k-NN neighbours for continuous entropy estimation (default 5).
    verbose : bool, optional
        Print subset-size progress (default False).

    Returns
    -------
    iis : dict
        Normalized IIS fingerprint, e.g.
        {'p2': 0.67, 'p3': 0.22, 'p4': 0.08, 'p5': 0.03}.
        Values sum to 1 (or are all 0 if no signal survives calibration).
    raw_mass : dict
        Raw architecture mass {'A2': ..., 'A3': ..., ...}.
    corrected_mass : dict
        Null-corrected mass (max(0, raw − null)).
    entropy_table : dict
        Maps tuple-of-column-names → entropy (bits) for every
        non-empty subset up to max(orders).
    mobius : dict
        Maps tuple-of-column-names → Möbius coefficient M(S).

    Notes
    -----
    Null calibration subtracts the expected architecture mass under
    independence (estimated from a type-matched null dataset) from
    the observed mass, then floors at zero.  This removes baseline
    mass that arises purely from finite-sample bias and the estimator's
    noise floor, leaving only mass attributable to genuine dependence
    structure.

    The IIS fingerprint (p₂, …, p₅) describes the *shape* of
    dependence — at what interaction order it concentrates — conditional
    on dependence existing at all.  Interpret the fingerprint together
    with the total corrected mass (sum of corrected_mass values), which
    measures the *magnitude* of dependence (architecture strength).
    """
    cols    = list(df.columns)
    max_ord = max(orders)
    data    = (df.values.astype(int) if is_discrete
               else copula_transform(df).values)

    # ── Step 1: entropy table ──────────────────────────────────────
    entropy_table = {}
    for size in range(1, max_ord + 1):
        subsets = list(combinations(range(len(cols)), size))
        for subset in subsets:
            col_names = tuple(cols[i] for i in subset)
            entropy_table[col_names] = entropy(data, subset, is_discrete, k=k)
        if verbose:
            print(f"  |S|={size}: {len(subsets)} subsets computed")

    # ── Step 2: Möbius inversion ───────────────────────────────────
    mob = mobius_inversion(entropy_table)

    # ── Step 3: raw architecture mass ─────────────────────────────
    raw_mass = architecture_mass(mob, orders=orders)

    # ── Step 4: null calibration ───────────────────────────────────
    if null_mass is not None:
        corrected_mass = {
            f"A{o}": max(0.0, raw_mass[f"A{o}"] - null_mass.get(f"A{o}", 0.0))
            for o in orders
        }
    else:
        corrected_mass = raw_mass.copy()

    # ── Step 5: normalize → IIS fingerprint ───────────────────────
    total = sum(corrected_mass.values())
    if total == 0:
        iis = {f"p{o}": 0.0 for o in orders}
    else:
        iis = {f"p{o}": corrected_mass[f"A{o}"] / total for o in orders}

    return iis, raw_mass, corrected_mass, entropy_table, mob


# ─────────────────────────────────────────────────────────────────
# Architecture entropy
# ─────────────────────────────────────────────────────────────────

def iis_entropy(iis):
    """
    Compute architecture entropy H_IIS.

        H_IIS = −Σ_k p_k · log₂(p_k)

    H_IIS = 0 means all dependence mass concentrates at a single
    interaction order (maximally structured).
    H_IIS = log₂(K) means mass is spread uniformly across all K
    orders (maximally mixed architecture).

    Parameters
    ----------
    iis : dict
        Output of ``run_iis`` — the normalized fingerprint.

    Returns
    -------
    float
        Architecture entropy in bits.
    """
    vals = np.array(list(iis.values()))
    vals = vals[vals > 0]
    return float(-np.sum(vals * np.log2(vals))) if len(vals) > 0 else 0.0
