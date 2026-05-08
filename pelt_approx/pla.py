"""
Algorithm III: Piecewise Linear Approximation (PLA).

Phase 2 Report — Section IV.D

PLA approximates the 32-entry decay lookup table y^n for n in [0, 32] by
storing k segment descriptors. Each segment covers a uniform sub-interval
of width h = 32/k and uses linear interpolation within that interval.

Algorithm:
    Precompute: S[j] = (n_j, y^n_j, slope_j)  for j = 0, ..., k-1
                where n_j = j * h  and  slope_j = (y^n_{j+1} - y^n_j) / h
    Query(n):   j = floor(n / h)
                return S[j].y0 + (n - S[j].n0) * S[j].slope

Theorem 3 (PLA Error Bound):
    |eps_PLA,k| <= (alpha^2 / 8) * h^2 = (alpha^2 * 1024) / (8 * k^2)

    k=4:  |eps| <= 3.75e-3   (0.375%)
    k=8:  |eps| <= 9.38e-4   (0.094%)
    k=16: |eps| <= 2.34e-4   (0.024%)

Proof sketch:
    Linear interpolation error on a C^2 function f over [a, a+h] is
    bounded by ||f''||_inf * h^2 / 8.
    For f(n) = y^n = e^(-alpha*n): f''(n) = alpha^2 * e^(-alpha*n),
    maximum at n=0:  ||f''||_inf = alpha^2.
    Therefore: |eps| <= alpha^2 * h^2 / 8.

IMPORTANT LIMITATION (Phase 2 Finding):
    PLA is suitable for approximating full-range table lookups where n
    varies over [0, 32]. It is NOT suitable for per-period updates where
    n is always 1 — this causes systematic bias because n=1 always falls
    in the first segment, whose slope averages the decay rate over h ms
    instead of computing the instantaneous 1-ms decay.
"""

import math
from dataclasses import dataclass
from typing import List

from .constants import Y, ALPHA, SCHED_CAPACITY_SCALE, MAX_PERIODS


# ---------------------------------------------------------------------------
# Segment descriptor
# ---------------------------------------------------------------------------

@dataclass
class PLASegment:
    """One segment in the PLA lookup table."""
    n0:    float  # left endpoint of segment (period index)
    y0:    float  # exact y^n0 at left endpoint
    slope: float  # linear slope = (y^n1 - y^n0) / h


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def build_pla_table(k: int) -> tuple[List[PLASegment], float]:
    """
    Build a PLA segment table with k uniform segments over [0, 32].

    Parameters
    ----------
    k : int  Number of segments. Must evenly divide 32.

    Returns
    -------
    (segments, h)
        segments : list of PLASegment  — k segment descriptors
        h        : float               — segment width (32/k)
    """
    if 32 % k != 0:
        raise ValueError(f"k={k} must evenly divide 32 (MAX_PERIODS). "
                         f"Valid choices: 1, 2, 4, 8, 16, 32.")
    h = MAX_PERIODS / k
    segments = []
    for j in range(k):
        n0 = j * h
        n1 = (j + 1) * h
        y0 = Y ** n0
        y1 = Y ** n1
        slope = (y1 - y0) / h
        segments.append(PLASegment(n0=n0, y0=y0, slope=slope))
    return segments, h


def pla_query(n: float, segments: List[PLASegment], h: float) -> float:
    """
    Query the PLA table for a given period count n.

    Parameters
    ----------
    n        : float           Period count to approximate y^n for.
    segments : list[PLASegment] PLA segment table.
    h        : float           Segment width.

    Returns
    -------
    float  Approximate value of y^n via linear interpolation.
    """
    j = min(int(n / h), len(segments) - 1)
    seg = segments[j]
    return seg.y0 + (n - seg.n0) * seg.slope


def pla_error_bound(k: int) -> float:
    """
    Theoretical maximum interpolation error for PLA with k segments.

    From Theorem 3:  |eps| <= alpha^2 * h^2 / 8  where h = 32/k.

    Parameters
    ----------
    k : int  Number of segments.

    Returns
    -------
    float  Absolute error bound.
    """
    h = MAX_PERIODS / k
    return (ALPHA ** 2) * (h ** 2) / 8.0


def pla_error_bound_pct(k: int) -> float:
    """Theoretical relative error bound as a percentage."""
    return (pla_error_bound(k) / Y) * 100.0


# ---------------------------------------------------------------------------
# Pre-built default tables (k = 4, 8, 16)
# ---------------------------------------------------------------------------

PLA_TABLE_K4,  PLA_H_K4  = build_pla_table(4)
PLA_TABLE_K8,  PLA_H_K8  = build_pla_table(8)
PLA_TABLE_K16, PLA_H_K16 = build_pla_table(16)


# ---------------------------------------------------------------------------
# PLA accumulator class
# ---------------------------------------------------------------------------

class PLAApprox:
    """
    Piecewise Linear Approximation PELT (Algorithm III).

    Uses k linear segments to approximate the 32-entry decay curve.
    Memory footprint: k * 3 * 8 bytes (three floats per segment).
    For k=8: 192 bytes — slightly larger than the original 128 bytes.

    NOTE: Designed for full-range table lookups (n varies in [0, 32]).
    NOT recommended for per-period updates where n is always 1.
    """

    def __init__(self, k: int = 8):
        self.k = k
        self.accumulator: float = 0.0
        self._segments, self._h = build_pla_table(k)
        self.error_bound_abs = pla_error_bound(k)
        self.error_bound_pct = pla_error_bound_pct(k)

    def update(self, r: float, n: int = 1) -> float:
        """
        Perform one PLA approximate PELT update.

        Parameters
        ----------
        r : float  CPU utilization fraction in [0, 1].
        n : int    Number of elapsed scheduling periods.

        Returns
        -------
        float  Updated approximate load average.
        """
        n = min(n, MAX_PERIODS)
        d = pla_query(float(n), self._segments, self._h)
        self.accumulator = self.accumulator * d + r * SCHED_CAPACITY_SCALE
        return self.accumulator

    def reset(self) -> None:
        self.accumulator = 0.0

    @property
    def value(self) -> float:
        return self.accumulator

    def table_memory_bytes(self) -> int:
        """Bytes used by the PLA segment table (3 floats per segment at 8 bytes each)."""
        return self.k * 3 * 8

    def __repr__(self) -> str:
        return (f"PLAApprox(k={self.k}, h={self._h:.2f}, "
                f"error_bound={self.error_bound_pct:.4f}%)")


# ---------------------------------------------------------------------------
# Standalone functional interfaces
# ---------------------------------------------------------------------------

def pla_update(accumulator: float, r: float, n: int,
               segments: List[PLASegment], h: float) -> float:
    """
    Functional PLA update.

    Parameters
    ----------
    accumulator : float             Current approximate load average.
    r           : float             CPU utilization fraction in [0, 1].
    n           : int               Elapsed scheduling periods.
    segments    : list[PLASegment]  PLA segment table.
    h           : float             Segment width.

    Returns
    -------
    float  Updated approximate load average.
    """
    n = min(n, MAX_PERIODS)
    d = pla_query(float(n), segments, h)
    return accumulator * d + r * SCHED_CAPACITY_SCALE


# ---------------------------------------------------------------------------
# Analysis: per-n interpolation error
# ---------------------------------------------------------------------------

def pla_pointwise_errors(k: int) -> dict:
    """
    Compute the interpolation error |PLA(n) - y^n| for every integer n in [0, 32].

    This is the per-entry accuracy check for the approximation table.
    The maximum across all n should be <= pla_error_bound(k).

    Returns
    -------
    dict with keys:
        'n_values'    : list of n values [0..32]
        'exact'       : list of exact y^n values
        'approx'      : list of PLA-approximated values
        'abs_errors'  : list of absolute errors
        'max_error'   : float, maximum absolute error
        'theory_bound': float, theoretical bound
    """
    segments, h = build_pla_table(k)
    n_values = list(range(MAX_PERIODS + 1))
    exact  = [Y ** n for n in n_values]
    approx = [pla_query(float(n), segments, h) for n in n_values]
    errors = [abs(approx[i] - exact[i]) for i in range(len(n_values))]
    return {
        "n_values":     n_values,
        "exact":        exact,
        "approx":       approx,
        "abs_errors":   errors,
        "max_error":    max(errors),
        "theory_bound": pla_error_bound(k),
    }
