"""
Shared constants for the approximate PELT simulation.

PELT mathematical model (from Linux kernel/sched/pelt.c):
    L(t) = sum_i r_i * y^((t - t_i) / T_p)

where:
    y  = 2^(-1/32) ≈ 0.97857  (decay base, chosen so y^32 = 0.5)
    alpha = ln(2)/32           (|ln y|, used in Taylor approximation)
    T_p = 1 ms                 (scheduling period)

The kernel implements this in Q32 fixed-point:
    L_{t+1} = L_t * d_n + r_t * SCHED_CAPACITY_SCALE
where d_n = runnable_avg_yN_inv[n] is a precomputed 32-bit table entry.
"""

import math

# ---------------------------------------------------------------------------
# Core decay constants
# ---------------------------------------------------------------------------

Y = 2 ** (-1.0 / 32)
"""Decay base: y = 2^(-1/32) ≈ 0.97857.
   Chosen so that y^32 = 0.5 exactly (a task's contribution
   from 32 ms ago counts for half its current-period contribution)."""

ALPHA = math.log(2) / 32
"""Alpha = |ln y| = ln(2)/32 ≈ 0.021660.
   Used in Taylor series expansion: y = e^(-alpha)."""

SCHED_CAPACITY_SCALE = 1024
"""Linux kernel SCHED_CAPACITY_SCALE constant.
   Load accumulator is scaled by this value per period."""

MAX_PERIODS = 32
"""Maximum number of periods in the lookup table (n in [0, 32])."""

# ---------------------------------------------------------------------------
# Kernel imbalance tolerance (used in Error-to-Fairness Theorem)
# ---------------------------------------------------------------------------

IMBALANCE_THRESHOLD = 25
"""Load imbalance migration threshold (25/1024 = 2.44%).
   The kernel migrates tasks when load imbalance exceeds this fraction
   of SCHED_CAPACITY_SCALE. Our approximate error must stay below this."""

FAIRNESS_THRESHOLD_PCT = IMBALANCE_THRESHOLD / SCHED_CAPACITY_SCALE * 100
"""Fairness safety threshold in percent: 25/1024 ≈ 2.44%.
   All approximate techniques must have delta < this value."""

# ---------------------------------------------------------------------------
# Exact Q32 lookup table  (mirrors runnable_avg_yN_inv in the Linux kernel)
# ---------------------------------------------------------------------------

EXACT_TABLE_Q32 = [Y ** n for n in range(MAX_PERIODS + 1)]
"""Exact decay table: EXACT_TABLE_Q32[n] = y^n for n in [0, 32].
   This corresponds to the kernel's runnable_avg_yN_inv[] array stored
   in Q32 fixed-point format (32 fractional bits)."""

# ---------------------------------------------------------------------------
# Quick sanity checks on the constants
# ---------------------------------------------------------------------------

assert abs(Y ** 32 - 0.5) < 1e-10, "y^32 must equal 0.5 by construction"
assert abs(math.exp(-ALPHA) - Y) < 1e-15, "alpha = |ln y| must satisfy e^(-alpha) = y"
assert len(EXACT_TABLE_Q32) == MAX_PERIODS + 1, "Table must have 33 entries (n=0..32)"
