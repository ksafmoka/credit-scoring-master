"""PSI helper tests."""

import numpy as np
import pandas as pd

from src.monitoring.data_drift import compute_psi


def test_psi_identical_distribution_near_zero():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 1, 2000))
    psi = compute_psi(x, x)
    assert psi < 0.05


def test_psi_shifted_distribution_positive():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 2000))
    cur = pd.Series(rng.normal(1.5, 1, 2000))
    psi = compute_psi(ref, cur)
    assert psi > 0.1
