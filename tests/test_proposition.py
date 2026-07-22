"""Guard 3 (protocol section 1): the Proposition's observable consequence.
Under positive dependence (beta>0), naive KM survival >= net truth (optimism);
under random censoring (beta=0), they coincide. Regression test for H1's direction."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.dgp import calibrate_c0, sample_synthetic, NET
from src.estimand import closed_form_net_survival
from src.arms.naive_survival import NaiveSurvivalArm


def test_km_dominates_net_for_positive_dependence():
    grid = np.linspace(0.05, 2.0, 60)
    S_net = closed_form_net_survival(grid, NET)
    rng = np.random.default_rng(1)
    for beta, expect_optimism in [(0.0, False), (1.0, True)]:
        c0 = calibrate_c0(beta, 0.4, rng)
        ds, _, _ = sample_synthetic(50000, beta=beta, sigma=0.5, c0=c0, rng=rng)
        S_km = NaiveSurvivalArm().fit(ds).survival(grid)
        gap = float(np.mean(S_km - S_net))
        if expect_optimism:
            assert gap > 0.004, f"beta={beta}: expected KM optimism, gap={gap:.4f}"
        else:
            assert abs(gap) < 0.004, f"beta={beta}: expected equality, gap={gap:.4f}"


if __name__ == "__main__":
    test_km_dominates_net_for_positive_dependence()
    print("PASS test_proposition")
