"""Tests de non-régression : lancer `pytest` à la racine du dépôt."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sofr_cap_pricing import (NOTIONAL, SIGMA, STRIKE, SWAP_RATES, SWAP_TENORS,
                              bootstrap_swap_curve, cap_floor_parity, cap_price,
                              caplet_table, swap_par_rate)


def _curve():
    curve, info = bootstrap_swap_curve(SWAP_TENORS, SWAP_RATES)
    return curve, info


def test_swaps_repriced_exactly():
    curve, info = _curve()
    for T, s in zip(info["T"], info["s"]):
        assert abs(swap_par_rate(curve, int(T)) - s) < 1e-12


def test_cap_price_reference_value():
    curve, _ = _curve()
    assert abs(cap_price(NOTIONAL, STRIKE, curve, SIGMA) - 223_799) < 1.0


def test_cap_floor_parity():
    curve, _ = _curve()
    diff, swap = cap_floor_parity(curve, caplet_table(curve))
    assert abs(diff - swap) < 1e-6


def test_cap_increases_with_vol():
    curve, _ = _curve()
    prices = [cap_price(NOTIONAL, STRIKE, curve, s) for s in (0.1, 0.2, 0.3)]
    assert prices[0] < prices[1] < prices[2]


def test_flat_curve():
    curve, info = bootstrap_swap_curve(SWAP_TENORS, [0.03, 0.03, 0.03])
    assert np.ptp(info["zeros"]) < 1e-5
    F = caplet_table(curve)["F"]
    assert np.ptp(F) < 1e-5
