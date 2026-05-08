"""
Algorithm II: Taylor Series Approximation (TSA).

Phase 2 Report — Section IV.C

Since y = e^(-alpha) where alpha = ln(2)/32, the per-period decay factor
can be approximated by truncating the Taylor series of e^(-alpha):

    d̃_k = sum_{j=0}^{k} (-alpha)^j / j!

Degree-2:  d ≈ 1 - alpha + alpha^2/2
Degree-3:  d ≈ 1 - alpha + alpha^2/2 - alpha^3/6

Unlike RPF (which requires a memory lookup), TSA computes d entirely from
compile-time constants using only multiply-add instructions.
Memory footprint: 12 bytes (degree-2) or 16 bytes (degree-3) for coefficients.

Theorem 2 (TSA Error Bound — Lagrange remainder):
    |eps_TSA,k| <= alpha^(k+1) / (k+1)!

    Degree-2: |eps| <= alpha^3 / 6  ≈ 1.69e-6  (relative: 0.000169%)
    Degree-3: |eps| <= alpha^4 / 24 ≈ 9.17e-9  (relative: 9.17e-7%)

Proof sketch:
    By Taylor's theorem, e^(-alpha) = sum_{j=0}^{k} (-alpha)^j/j!
                                     + e^(-xi) * (-alpha)^(k+1) / (k+1)!
    for some xi in (0, alpha). Since e^(-xi) <= 1, the absolute remainder
    is bounded by alpha^(k+1) / (k+1)!.
"""

import math
from .constants import Y, ALPHA, SCHED_CAPACITY_SCALE, MAX_PERIODS


# ---------------------------------------------------------------------------
# Precomputed TSA decay factors (compile-time constants)
# ---------------------------------------------------------------------------

_TSA_D2 = 1.0 - ALPHA + (ALPHA ** 2) / 2.0
"""Degree-2 Taylor approximation: d ≈ 1 - alpha + alpha^2/2."""

_TSA_D3 = 1.0 - ALPHA + (ALPHA ** 2) / 2.0 - (ALPHA ** 3) / 6.0
"""Degree-3 Taylor approximation: d ≈ 1 - alpha + alpha^2/2 - alpha^3/6."""

_EXACT_D1 = Y
"""Exact per-period decay factor for comparison."""


def tsa_decay_factor(degree: int) -> float:
    """
    Compute the Taylor series approximation of y = e^(-alpha) at given degree.

    Parameters
    ----------
    degree : int  Number of terms to include beyond the constant 1 (degree >= 1).

    Returns
    -------
    float  Approximate decay factor d̃.
    """
    d = 0.0
    for j in range(degree + 1):
        d += ((-ALPHA) ** j) / math.factorial(j)
    return d


def tsa_error_bound(degree: int) -> float:
    """
    Compute the theoretical Lagrange remainder error bound for TSA.

    From Theorem 2:  |eps| <= alpha^(k+1) / (k+1)!

    Parameters
    ----------
    degree : int  Taylor polynomial degree.

    Returns
    -------
    float  Absolute error bound per update step.
    """
    return (ALPHA ** (degree + 1)) / math.factorial(degree + 1)


def tsa_relative_error_bound_pct(degree: int) -> float:
    """
    Theoretical relative error bound as a percentage of the exact decay factor.

    Returns
    -------
    float  Relative error bound in percent.
    """
    abs_bound = tsa_error_bound(degree)
    return (abs_bound / Y) * 100.0


# ---------------------------------------------------------------------------
# TSA accumulator class
# ---------------------------------------------------------------------------

class TSAApprox:
    """
    Taylor Series Approximation PELT (Algorithm II).

    Replaces the 32-entry lookup table with a degree-k polynomial
    evaluated from compile-time constants. No memory access at runtime.

    Supported degrees: 2 (default), 3.
    """

    def __init__(self, degree: int = 2):
        if degree not in (2, 3):
            raise ValueError(f"Only degrees 2 and 3 are supported; got degree={degree}")
        self.degree = degree
        self.accumulator: float = 0.0

        # Precompute the single decay constant (same every period when n=1)
        self._d = tsa_decay_factor(degree)
        self.error_bound_abs = tsa_error_bound(degree)
        self.error_bound_pct = tsa_relative_error_bound_pct(degree)

    def update(self, r: float, n: int = 1) -> float:
        """
        Perform one TSA approximate PELT update.

        Note: TSA computes d̃ = e^(-n*alpha) for each n. For the standard
        per-period update (n=1), this is the precomputed constant.
        For multi-period gaps, we compute y^n ≈ d̃^n.

        Parameters
        ----------
        r : float  CPU utilization fraction in [0, 1].
        n : int    Number of elapsed scheduling periods (default 1).

        Returns
        -------
        float  Updated approximate load average.
        """
        n = min(n, MAX_PERIODS)
        if n == 1:
            d = self._d
        else:
            # For n>1: approximate y^n using the Taylor-approximated single step
            # d^n  (cheaper than re-expanding the full series for n*alpha)
            d = self._d ** n
        self.accumulator = self.accumulator * d + r * SCHED_CAPACITY_SCALE
        return self.accumulator

    def reset(self) -> None:
        self.accumulator = 0.0

    @property
    def value(self) -> float:
        return self.accumulator

    @property
    def coefficient_memory_bytes(self) -> int:
        """Bytes used to store TSA coefficients (degree+1 floats at 4 bytes each)."""
        return (self.degree + 1) * 4

    def __repr__(self) -> str:
        return (f"TSAApprox(degree={self.degree}, d={self._d:.10f}, "
                f"error_bound={self.error_bound_pct:.2e}%)")


# ---------------------------------------------------------------------------
# Standalone functional interface
# ---------------------------------------------------------------------------

def tsa_update_degree2(accumulator: float, r: float, n: int = 1) -> float:
    """
    Functional TSA degree-2 update.

    d = 1 - alpha + alpha^2/2

    Parameters
    ----------
    accumulator : float  Current approximate load average.
    r           : float  CPU utilization fraction in [0, 1].
    n           : int    Elapsed periods (default 1).

    Returns
    -------
    float  Updated approximate load average.
    """
    n = min(n, MAX_PERIODS)
    d = _TSA_D2 if n == 1 else _TSA_D2 ** n
    return accumulator * d + r * SCHED_CAPACITY_SCALE


def tsa_update_degree3(accumulator: float, r: float, n: int = 1) -> float:
    """
    Functional TSA degree-3 update.

    d = 1 - alpha + alpha^2/2 - alpha^3/6

    Parameters
    ----------
    accumulator : float  Current approximate load average.
    r           : float  CPU utilization fraction in [0, 1].
    n           : int    Elapsed periods (default 1).

    Returns
    -------
    float  Updated approximate load average.
    """
    n = min(n, MAX_PERIODS)
    d = _TSA_D3 if n == 1 else _TSA_D3 ** n
    return accumulator * d + r * SCHED_CAPACITY_SCALE


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def compare_tsa_to_exact() -> dict:
    """
    Compare TSA approximation values against the exact decay factor.

    Returns a dict with exact value, TSA-2, TSA-3, and their errors.
    """
    exact = Y
    tsa2  = tsa_decay_factor(2)
    tsa3  = tsa_decay_factor(3)
    return {
        "exact":            exact,
        "tsa_degree2":      tsa2,
        "tsa_degree3":      tsa3,
        "abs_error_tsa2":   abs(tsa2 - exact),
        "abs_error_tsa3":   abs(tsa3 - exact),
        "rel_error_tsa2_%": abs(tsa2 - exact) / exact * 100,
        "rel_error_tsa3_%": abs(tsa3 - exact) / exact * 100,
        "bound_tsa2_%":     tsa_relative_error_bound_pct(2),
        "bound_tsa3_%":     tsa_relative_error_bound_pct(3),
    }
