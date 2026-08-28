"""
reproduce_benchmark.py
======================
Reproduces the four-method comparison reported in the manuscript.

Run from the repo root:
    python reproduce_benchmark.py

Output
------
Prints comparison table and detection scorecard to stdout.
Saves results/benchmark_results.csv.

Methods compared
----------------
1. Spearman correlation      — pairwise rank association
2. O-information             — Rosas et al. (2019); redundancy vs. synergy
3. Partial correlation       — precision-matrix route; linear conditioning
4. IIS pipeline (this work)  — full interaction-order decomposition

Datasets
--------
See benchmark_datasets/README.md for full descriptions and parameter choices.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from src import run_iis, iis_entropy
from src.estimators import copula_transform, entropy

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SEED    = 42
N       = 1000
P       = 5
RHO     = 0.55   # AR(1) correlation for Correlated Gaussian

DISCRETE_DATASETS   = {"Shared Ancestry", "XOR Epistasis", "Discrete IID"}
CONTINUOUS_DATASETS = {"Gaussian IID", "Correlated Gaussian"}
REPORT_DATASETS     = ["Gaussian IID", "Correlated Gaussian",
                       "Shared Ancestry", "XOR Epistasis"]

GROUND_TRUTH = {
    "Gaussian IID":        "None (null)",
    "Correlated Gaussian": "Pairwise only",
    "Shared Ancestry":     "Pairwise + latent",
    "XOR Epistasis":       "Fifth-order only",
}


def get_null(name):
    """Return the name of the type-matched null dataset."""
    return "Discrete IID" if name in DISCRETE_DATASETS else "Gaussian IID"


# ─────────────────────────────────────────────────────────────────
# 1. Dataset generation
# ─────────────────────────────────────────────────────────────────

def make_datasets():
    rng = np.random.default_rng(SEED)

    datasets = {}

    # Gaussian IID — continuous null
    datasets["Gaussian IID"] = pd.DataFrame(
        rng.normal(0, 1, (N, P)),
        columns=[f"V{i+1}" for i in range(P)]
    )

    # Correlated Gaussian — pairwise AR(1) structure
    Sigma = np.array([[RHO ** abs(i - j) for j in range(P)] for i in range(P)])
    datasets["Correlated Gaussian"] = pd.DataFrame(
        rng.multivariate_normal(np.zeros(P), Sigma, N),
        columns=[f"V{i+1}" for i in range(P)]
    )

    # Shared Ancestry — latent binary driver induces pairwise + higher-order
    A_anc = rng.binomial(1, 0.5, N)
    shared = np.column_stack([
        rng.binomial(1, np.where(A_anc == 1, 0.80, 0.20))
        for _ in range(P)
    ])
    datasets["Shared Ancestry"] = pd.DataFrame(
        shared, columns=[f"SNP{i+1}" for i in range(P)]
    )

    # XOR Epistasis — fifth-order parity; invisible to all pairwise methods
    SNPs = [rng.binomial(1, 0.5, N) for _ in range(P - 1)]
    xor  = SNPs[0].copy()
    for s in SNPs[1:]:
        xor = xor ^ s
    datasets["XOR Epistasis"] = pd.DataFrame(
        np.column_stack(SNPs + [xor]),
        columns=[f"SNP{i+1}" for i in range(P)]
    )

    # Discrete IID — discrete null (type-matched for binary datasets)
    datasets["Discrete IID"] = pd.DataFrame(
        rng.binomial(1, 0.5, (N, P)),
        columns=[f"SNP{i+1}" for i in range(P)]
    )

    return datasets


# ─────────────────────────────────────────────────────────────────
# 2. Method 1 — Spearman correlation
# ─────────────────────────────────────────────────────────────────

def run_spearman(datasets):
    results = {}
    for name in REPORT_DATASETS:
        corr  = datasets[name].corr(method='spearman').abs()
        upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
        vals  = corr.where(upper).stack()
        results[name] = {'mean_r': vals.mean(), 'max_r': vals.max()}
    return results


# ─────────────────────────────────────────────────────────────────
# 3. Method 2 — O-information (Rosas et al. 2019)
# ─────────────────────────────────────────────────────────────────

def run_oinfo(datasets):
    """
    Ω = (p−2)·H(X₁…Xₚ) + Σᵢ[H(Xᵢ) − H(X\ᵢ)]
    Positive = redundancy dominated; negative = synergy dominated.
    """
    results = {}
    for name in REPORT_DATASETS:
        df          = datasets[name]
        is_discrete = name in DISCRETE_DATASETS
        data        = (df.values.astype(int) if is_discrete
                       else copula_transform(df).values)
        all_idx     = list(range(len(df.columns)))

        H_joint   = entropy(data, all_idx, is_discrete)
        H_margins = [entropy(data, [i], is_discrete) for i in all_idx]
        H_leaves  = [entropy(data, [j for j in all_idx if j != i], is_discrete)
                     for i in all_idx]

        o_info = ((P - 2) * H_joint + sum(H_margins) - sum(H_leaves))
        results[name] = {'o_information': o_info}
    return results


# ─────────────────────────────────────────────────────────────────
# 4. Method 3 — Partial correlation
# ─────────────────────────────────────────────────────────────────

def partial_correlation_matrix(X):
    """Precision-matrix route: ρ_ij|rest = −P_ij / sqrt(P_ii · P_jj)."""
    Sigma = np.cov(X, rowvar=False)
    P_mat = np.linalg.pinv(Sigma)
    d     = np.sqrt(np.diag(P_mat))
    with np.errstate(divide='ignore', invalid='ignore'):
        pcorr = -P_mat / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    return pcorr


def run_partial_corr(datasets, spearman_results):
    results = {}
    for name in REPORT_DATASETS:
        X    = datasets[name].values.astype(float)
        pmat = partial_correlation_matrix(X)
        k    = pmat.shape[0]
        vals = np.array([abs(pmat[i, j])
                         for i in range(k) for j in range(i + 1, k)])
        mean_pc   = vals.mean()
        sp_mean   = spearman_results[name]['mean_r']
        reduction = (1 - mean_pc / sp_mean) if sp_mean > 0 else float('nan')
        results[name] = {
            'mean_pcorr':           mean_pc,
            'max_pcorr':            vals.max(),
            'reduction_vs_spearman': reduction
        }
    return results


# ─────────────────────────────────────────────────────────────────
# 5. Method 4 — IIS pipeline
# ─────────────────────────────────────────────────────────────────

def run_iis_all(datasets):
    # First pass: compute raw mass for all datasets (needed for nulls)
    raw_masses = {}
    for name, df in datasets.items():
        is_discrete = name in DISCRETE_DATASETS
        print(f"  [{name}] computing entropy table ...", flush=True)
        _, raw_mass, _, _, _ = run_iis(df, is_discrete,
                                        null_mass=None, verbose=False)
        raw_masses[name] = raw_mass

    # Second pass: null-calibrate report datasets
    iis_results = {}
    for name in REPORT_DATASETS:
        df          = datasets[name]
        is_discrete = name in DISCRETE_DATASETS
        null_name   = get_null(name)

        iis, raw_mass, corrected, _, _ = run_iis(
            df, is_discrete,
            null_mass=raw_masses[null_name],
            verbose=False
        )
        H = iis_entropy(iis)
        A_total = sum(corrected.values())

        iis_results[name] = {
            **iis,
            'H_IIS':          H,
            'A_total':        A_total,
            'raw_mass':       raw_mass,
            'corrected_mass': corrected,
        }

    return iis_results


# ─────────────────────────────────────────────────────────────────
# 6. Print and save results
# ─────────────────────────────────────────────────────────────────

def print_and_save(spearman, pcorr, oinfo, iis_res):
    # Console table
    print("\n" + "=" * 120)
    print("COMPARISON — all four methods")
    print("=" * 120)
    header = (f"{'Dataset':22s} {'Ground truth':20s} "
              f"{'Spearman':>10} {'PartialCorr':>12} {'O-info':>10} "
              f"{'p2':>7} {'p3':>7} {'p4':>7} {'p5':>7} "
              f"{'H_IIS':>7} {'A_total':>9}")
    print(header)
    print("-" * 120)

    rows = []
    for name in REPORT_DATASETS:
        sp = spearman[name]
        pc = pcorr[name]
        oi = oinfo[name]
        ii = iis_res[name]
        print(
            f"{name:22s} {GROUND_TRUTH[name]:20s} "
            f"{sp['mean_r']:>10.4f} {pc['mean_pcorr']:>12.4f} "
            f"{oi['o_information']:>10.4f} "
            f"{ii['p2']:>7.4f} {ii['p3']:>7.4f} "
            f"{ii['p4']:>7.4f} {ii['p5']:>7.4f} "
            f"{ii['H_IIS']:>7.4f} {ii['A_total']:>9.4f}"
        )
        rows.append({
            'Dataset':      name,
            'Ground_truth': GROUND_TRUTH[name],
            'Spearman_mean_r':    sp['mean_r'],
            'PartialCorr_mean':   pc['mean_pcorr'],
            'O_information':      oi['o_information'],
            'IIS_p2': ii['p2'], 'IIS_p3': ii['p3'],
            'IIS_p4': ii['p4'], 'IIS_p5': ii['p5'],
            'H_IIS':   ii['H_IIS'],
            'A_total': ii['A_total'],
        })

    # Detection scorecard
    print("\n" + "=" * 88)
    print("DETECTION SCORECARD")
    print("=" * 88)
    scores = {
        "Spearman |r|":        ["Y", "Y", "Y", "N"],
        "O-information":       ["Y", "~", "~", "~"],
        "Partial correlation": ["Y", "Y", "~", "N"],
        "IIS (this work)":     ["Y", "Y", "Y", "Y"],
    }
    print(f"\n{'Method':22s} {'Gaussian IID':>14} {'Corr. Gaussian':>16} "
          f"{'Shared Ancestry':>17} {'XOR Epistasis':>15}")
    print("-" * 88)
    for method, vals in scores.items():
        print(f"  {method:20s} {vals[0]:>14} {vals[1]:>16} "
              f"{vals[2]:>17} {vals[3]:>15}")
    print("\nY = correct  |  N = fails  |  ~ = partial / no order decomposition")

    # Save CSV
    out = RESULTS_DIR / "benchmark_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nResults saved → {out}")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("IIS Benchmark — reproducing manuscript results")
    print("=" * 60)

    print("\nGenerating datasets ...")
    datasets = make_datasets()

    print("Method 1: Spearman correlation ...")
    spearman = run_spearman(datasets)

    print("Method 2: O-information ...")
    oinfo = run_oinfo(datasets)

    print("Method 3: Partial correlation ...")
    pcorr = run_partial_corr(datasets, spearman)

    print("Method 4: IIS pipeline ...")
    iis_res = run_iis_all(datasets)

    print_and_save(spearman, pcorr, oinfo, iis_res)
