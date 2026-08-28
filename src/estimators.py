"""
src/estimators.py
=================
Entropy estimators for the IIS pipeline.

Two estimators are provided so the pipeline works on both continuous
and discrete data without modification:

    - knn_entropy   : Kozachenko–Leonenko k-NN estimator (continuous)
    - discrete_entropy : Plugin maximum-likelihood estimator (discrete)

A copula transform is also provided to map continuous margins to a
common [0,1] scale before k-NN estimation.

To substitute a different estimator, replace or extend either function
here — nothing in mobius.py or pipeline.py needs to change.
"""

import math
import numpy as np
from scipy.spatial import KDTree
from scipy.special import digamma


# ─────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────

def copula_transform(df):
    """
    Rank-based copula transform.

    Maps each column to (0, 1) by replacing values with their
    fractional rank: rank / (n + 1).  This removes marginal
    distributional differences so that k-NN entropy estimation
    is not confounded by scale or shape differences across variables.

    Parameters
    ----------
    df : pd.DataFrame
        Continuous input data (samples × variables).

    Returns
    -------
    pd.DataFrame
        Same shape as input, with each column mapped to (0, 1).
    """
    return df.rank(method='average').apply(lambda col: col / (len(col) + 1))


# ─────────────────────────────────────────────────────────────────
# Continuous estimator
# ─────────────────────────────────────────────────────────────────

def knn_entropy(X, k=5):
    """
    Kozachenko–Leonenko k-nearest-neighbour entropy estimator.

    Estimates differential entropy of a continuous multivariate
    distribution from samples, without assuming any parametric form.
    Recommended pre-processing: apply copula_transform before passing
    continuous data here.

    Parameters
    ----------
    X : ndarray, shape (n_samples, d)
        Continuous data matrix.
    k : int, optional (default 5)
        Number of nearest neighbours.  Larger k reduces variance
        but increases bias; k=5 is a standard choice.

    Returns
    -------
    float
        Estimated entropy in bits.

    References
    ----------
    Kozachenko, L. F., & Leonenko, N. N. (1987). Sample estimate of
    the entropy of a random vector. Problems of Information Transmission,
    23(2), 95–101.
    """
    n, d = X.shape
    tree = KDTree(X)
    dist, _ = tree.query(X, k=k + 1)          # k+1 because point 0 is self
    rho = np.maximum(dist[:, k], 1e-10)        # distance to k-th neighbour

    # Volume of unit ball in R^d (Chebyshev / max norm via KDTree default)
    if d % 2 == 0:
        cd = (np.pi ** (d / 2) /
              (2 ** d * (d / 2 + 1) * math.factorial(d // 2)))
    else:
        cd = (np.pi ** ((d - 1) / 2) *
              2 ** ((d + 1) / 2) / math.factorial(d))

    H_nats = (d * np.mean(np.log(2 * rho)) +
              np.log(cd) + digamma(n) - digamma(k))
    return H_nats / np.log(2)                  # convert nats → bits


# ─────────────────────────────────────────────────────────────────
# Discrete estimator
# ─────────────────────────────────────────────────────────────────

def discrete_entropy(X):
    """
    Plugin (maximum-likelihood) entropy estimator for discrete data.

    Counts the empirical frequency of each unique row pattern and
    computes Shannon entropy from those frequencies.  No continuity
    correction or shrinkage is applied; adequate for n >> 2^d.

    Parameters
    ----------
    X : ndarray, shape (n_samples, d)
        Integer-coded discrete data (e.g. SNP alleles coded 0/1).

    Returns
    -------
    float
        Estimated entropy in bits.
    """
    rows = [tuple(row) for row in X]
    counts = {}
    for r in rows:
        counts[r] = counts.get(r, 0) + 1
    n = len(rows)
    probs = np.array(list(counts.values())) / n
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


# ─────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────

def entropy(data, indices, is_discrete, k=5):
    """
    Dispatch entropy estimation to the correct estimator.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_variables)
        Full data matrix (copula-transformed if continuous).
    indices : sequence of int
        Column indices of the subset to compute entropy over.
    is_discrete : bool
        True → discrete_entropy; False → knn_entropy.
    k : int
        k-NN neighbours (ignored for discrete data).

    Returns
    -------
    float
        Entropy in bits.
    """
    X = data[:, list(indices)]
    return discrete_entropy(X) if is_discrete else knn_entropy(X, k=k)

