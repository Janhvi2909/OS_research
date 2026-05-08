"""
Algorithm I: Reduced-Precision Fixed-Point (RPF) Approximation.

Phase 2 Report — Section IV.B

The current Linux kernel stores the decay table in Q32 format (32 fractional bits),
requiring 64-bit multiply-shift operations. RPF replaces this with a Q15 table
(16-bit precision), halving memory consumption and enabling 16-bit multiplications.

Precompute:  rpf_table[n] = round(y^n * 2^15) / 2^15   for n in [0, 32]
Runtime:     L_{t+1} = L_t * rpf_table[n] + r * 1024

Theorem 1 (RPF Error Bound):
    delta_RPF <= 1 / (2^(k-1) * (1 - y)) = 1 / (2^(k-1) * alpha_eff)

    For k=16 bits:  delta_RPF <= 1 / (32768 * 0.02143) ≈ 0.1430%

Proof sketch:
    Quantization error per entry: eps_q = 2^(-(k-1))
    Steady-state error: |L*_approx - L*_exact| / L*_exact = eps_q / (1-y) <= eps_q / alpha_eff
"""

import math
from .constants import Y, SCHED_CAPACITY_SCALE, MAX_PERIODS


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def build_rpf_table(bits: int = 16) -> list[float]:
    """
    Build a Reduced-Precision Fixed-Point decay table.

    Parameters
    ----------
    bits : int
        Total bit width. Uses Q(bits-1) format (1 integer bit, bits-1 fractional).
        Default is 16 bits (Q15), matching a 16-bit unsigned integer.

    Returns
    -------
    list[float]
        33-entry list of rounded decay values y^n for n in [0, 32].
    """
    scale = 2 ** (bits - 1)
    table = []
    for n in range(MAX_PERIODS + 1):
        exact_val = Y ** n
        quantized = round(exact_val * scale) / scale
        table.append(quantized)
    return table


def rpf_error_bound(bits: int = 16) -> float:
    """
    Compute the theoretical worst-case relative error bound for RPF.

    From Theorem 1:  delta_RPF <= 1 / (2^(k-1) * (1-y))

    Parameters
    ----------
    bits : int  Bit width of the Q-format table (default 16 for Q15).

    Returns
    -------
    float  Worst-case relative error bound as a percentage.
    """
    eps_q = 2 ** (-(bits - 1))
    alpha_eff = 1.0 - Y
    return (eps_q / alpha_eff) * 100.0


# ---------------------------------------------------------------------------
# Pre-built default Q15 table
# ---------------------------------------------------------------------------

RPF_TABLE_Q15 = build_rpf_table(bits=16)
"""Default 16-bit Q15 RPF lookup table (33 entries, n=0..32)."""

RPF_ERROR_BOUND_PCT = rpf_error_bound(bits=16)
"""Theoretical worst-case error bound for Q15 RPF: ≈ 0.1430%"""


# ---------------------------------------------------------------------------
# RPF accumulator class
# ---------------------------------------------------------------------------

class RPFApprox:
    """
    Reduced-Precision Fixed-Point PELT approximation (Algorithm I).

    Uses a 16-bit Q15 lookup table instead of the kernel's 32-bit Q32 table.
    Memory footprint: 64 bytes (vs 128 bytes for Q32).
    Cycle reduction: ~50% (16-bit multiply vs 64-bit multiply-shift).

    Worst-case error bound: 0.1430% (Theorem 1, Phase 2 Report).
    """

    def __init__(self, bits: int = 16):
        self.bits = bits
        self.accumulator: float = 0.0
        self._table = build_rpf_table(bits)
        self.error_bound_pct = rpf_error_bound(bits)

    def update(self, r: float, n: int = 1) -> float:
        """
        Perform one RPF approximate PELT update.

        Parameters
        ----------
        r : float  CPU utilization fraction in [0, 1].
        n : int    Number of elapsed scheduling periods (default 1).

        Returns
        -------
        float  Updated approximate load average.
        """
        n = min(n, MAX_PERIODS)
        d = self._table[n]
        self.accumulator = self.accumulator * d + r * SCHED_CAPACITY_SCALE
        return self.accumulator

    def reset(self) -> None:
        self.accumulator = 0.0

    @property
    def value(self) -> float:
        return self.accumulator

    def table_memory_bytes(self) -> int:
        """Memory used by the RPF table in bytes."""
        bytes_per_entry = (self.bits + 7) // 8
        return (MAX_PERIODS + 1) * bytes_per_entry

    def __repr__(self) -> str:
        return (f"RPFApprox(bits={self.bits}, accumulator={self.accumulator:.6f}, "
                f"error_bound={self.error_bound_pct:.4f}%)")


# ---------------------------------------------------------------------------
# Standalone functional interface
# ---------------------------------------------------------------------------

def rpf_update(accumulator: float, r: float, n: int = 1,
               table: list[float] = None) -> float:
    """
    Functional RPF update using the default Q15 table.

    Parameters
    ----------
    accumulator : float  Current approximate load average.
    r           : float  CPU utilization fraction in [0, 1].
    n           : int    Elapsed scheduling periods (default 1).
    table       : list   Optional custom RPF table (defaults to RPF_TABLE_Q15).

    Returns
    -------
    float  Updated approximate load average.
    """
    if table is None:
        table = RPF_TABLE_Q15
    n = min(n, MAX_PERIODS)
    d = table[n]
    return accumulator * d + r * SCHED_CAPACITY_SCALE


# ---------------------------------------------------------------------------
# Per-entry quantization error analysis
# ---------------------------------------------------------------------------

def rpf_table_quantization_errors(bits: int = 16) -> list[float]:
    """
    Compute the absolute quantization error at each table entry.

    Returns a list of |rpf_table[n] - y^n| for n in [0, 32].
    """
    from .constants import EXACT_TABLE_Q32
    table = build_rpf_table(bits)
    return [abs(table[n] - EXACT_TABLE_Q32[n]) for n in range(MAX_PERIODS + 1)]
