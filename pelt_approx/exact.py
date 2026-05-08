"""
Exact PELT (Per-Entity Load Tracking) accumulator.

This module faithfully replicates the exact update logic from
Linux kernel/sched/pelt.c: ___update_load_avg().

The update equation is:
    L_{t+1} = L_t * d_n + r_t * SCHED_CAPACITY_SCALE

where d_n = y^n is looked up from a precomputed Q32 table.

This serves as the golden reference against which all approximate
techniques are measured for error.
"""

from .constants import EXACT_TABLE_Q32, SCHED_CAPACITY_SCALE, MAX_PERIODS


class ExactPELT:
    """
    Exact PELT accumulator (Q32 fixed-point equivalent).

    Mirrors ___update_load_avg() in kernel/sched/pelt.c.
    Uses a 33-entry precomputed table (y^0 through y^32).
    """

    def __init__(self):
        self.accumulator: float = 0.0
        self._table = EXACT_TABLE_Q32

    def update(self, r: float, n: int = 1) -> float:
        """
        Perform one PELT accumulator update.

        Parameters
        ----------
        r : float
            CPU utilization fraction for this period, in [0.0, 1.0].
            The kernel uses runnable time / period length.
        n : int
            Number of full scheduling periods elapsed since last update.
            Clamped to [0, MAX_PERIODS].

        Returns
        -------
        float
            Updated accumulator value.
        """
        n = min(n, MAX_PERIODS)
        d = self._table[n]
        self.accumulator = self.accumulator * d + r * SCHED_CAPACITY_SCALE
        return self.accumulator

    def reset(self) -> None:
        """Reset accumulator to zero (e.g., for a newly created task)."""
        self.accumulator = 0.0

    @property
    def value(self) -> float:
        return self.accumulator

    def __repr__(self) -> str:
        return f"ExactPELT(accumulator={self.accumulator:.6f})"


# ---------------------------------------------------------------------------
# Standalone functional interface (mirrors kernel function signature)
# ---------------------------------------------------------------------------

def exact_update(accumulator: float, r: float, n: int = 1) -> float:
    """
    Functional version of the exact PELT update.

    Equivalent to ___update_load_avg() in kernel/sched/pelt.c.

    Parameters
    ----------
    accumulator : float  Current load average value.
    r           : float  CPU utilization fraction in [0, 1].
    n           : int    Number of elapsed scheduling periods (default 1).

    Returns
    -------
    float  Updated load average.
    """
    n = min(n, MAX_PERIODS)
    d = EXACT_TABLE_Q32[n]
    return accumulator * d + r * SCHED_CAPACITY_SCALE


# ---------------------------------------------------------------------------
# Steady-state analysis
# ---------------------------------------------------------------------------

def steady_state_value(r_constant: float = 1.0) -> float:
    """
    Compute the exact steady-state accumulator value for a constant utilization r.

    At steady state:  L* = L* * y + r * 1024
                      L* (1 - y) = r * 1024
                      L* = r * 1024 / (1 - y)

    Parameters
    ----------
    r_constant : float  Constant CPU utilization fraction (default 1.0 = 100%).

    Returns
    -------
    float  Steady-state load average.
    """
    from .constants import Y
    return r_constant * SCHED_CAPACITY_SCALE / (1.0 - Y)
