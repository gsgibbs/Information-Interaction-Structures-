# Benchmark Datasets

All datasets are **synthetically generated** at runtime by `reproduce_benchmark.py`
using `numpy.random.default_rng(seed=42)`.  No external data files are required.

---

## Dataset descriptions

### Gaussian IID  *(continuous null)*
| Parameter | Value |
|-----------|-------|
| n         | 1000  |
| p         | 5     |
| Distribution | N(0, 1), independent across all variables |

**Purpose:** Establishes the continuous null baseline.  A correctly
calibrated pipeline should detect no dependence architecture here.
Used as the null reference for all continuous datasets.

---

### Correlated Gaussian  *(pairwise dependence only)*
| Parameter | Value |
|-----------|-------|
| n         | 1000  |
| p         | 5     |
| Covariance | AR(1): Σᵢⱼ = ρ^|i−j|, ρ = 0.55 |

**Purpose:** Tests detection of classical linear pairwise dependence.
All dependence is genuinely pairwise; the IIS fingerprint should
concentrate at p₂ with minimal higher-order mass.

---

### Shared Ancestry  *(pairwise + latent driver)*
| Parameter | Value |
|-----------|-------|
| n         | 1000  |
| p         | 5     |
| Generative model | Latent binary A ~ Bernoulli(0.5); each SNP drawn from Bernoulli(0.8) if A=1, Bernoulli(0.2) if A=0 |

**Purpose:** Simulates a common-driver genetic architecture (e.g. population
stratification or shared ancestry block).  Pairwise methods detect
correlation, but partial correlation partially explains it away.
The IIS fingerprint should show elevated p₂ from the shared driver.

---

### XOR Epistasis  *(pure fifth-order interaction)*
| Parameter | Value |
|-----------|-------|
| n         | 1000  |
| p         | 5     |
| Generative model | SNP1–SNP4 ~ Bernoulli(0.5) iid; SNP5 = SNP1 ⊕ SNP2 ⊕ SNP3 ⊕ SNP4 |

**Purpose:** The critical stress test.  XOR parity creates a pure
fifth-order interaction that is **invisible to all pairwise methods**
(Spearman |r| ≈ 0, partial correlation ≈ 0) but produces a large A₅
in the IIS pipeline.  This dataset demonstrates the unique capability
of the framework.

---

### Discrete IID  *(discrete null)*
| Parameter | Value |
|-----------|-------|
| n         | 1000  |
| p         | 5     |
| Distribution | Bernoulli(0.5), independent across all variables |

**Purpose:** Type-matched null for binary/discrete datasets (Shared Ancestry,
XOR Epistasis).  Using a Gaussian null for discrete data would
conflate estimator bias with genuine dependence; Discrete IID provides
a calibration baseline with the same marginal distribution and estimator.

---

## Parameter choices

**n = 1000, p = 5:**  The number of subsets grows as 2^p − 1 = 31.
At p = 5 this is tractable for k-NN estimation without approximation.
For larger p, subset sampling or approximate methods are needed
(see manuscript §3.4).

**k = 5 (k-NN):**  Standard choice balancing bias and variance for
n = 1000.  Sensitivity analysis across k ∈ {3, 5, 10} showed stable
architecture mass estimates for the datasets above.

**ρ = 0.55 (AR(1)):**  Moderate correlation — strong enough that pairwise
methods detect it clearly, but not so strong that higher-order terms
dominate through nonlinear amplification.

**Noise:**  No observation noise is added to any dataset.  The
finite-sample variability of the entropy estimators plays the role of
noise in the calibration step.
