"""
pelt_approx — Approximate PELT Simulation Package

Approximate Computing in OS Process Scheduling:
Error-Bounded Priority Decay Functions for CPU Fair-Share Scheduling

Phase 2 implementation — three approximate algorithms for the Linux
PELT (Per-Entity Load Tracking) decay computation:

    Algorithm I  : RPF — Reduced-Precision Fixed-Point
    Algorithm II : TSA — Taylor Series Approximation (degrees 2 and 3)
    Algorithm III: PLA — Piecewise Linear Approximation

Plus:
    - ExactPELT   : reference exact Q32 implementation
    - ErrorMonitor: 1%-sampling self-healing safety component

Quick start
-----------
    from pelt_approx import ExactPELT, RPFApprox, TSAApprox, PLAApprox

    exact = ExactPELT()
    rpf   = RPFApprox(bits=16)
    tsa2  = TSAApprox(degree=2)
    tsa3  = TSAApprox(degree=3)
    pla8  = PLAApprox(k=8)

    for r in workload:
        exact.update(r)
        rpf.update(r)
        tsa2.update(r)
        # ...
"""

from .constants     import Y, ALPHA, SCHED_CAPACITY_SCALE, FAIRNESS_THRESHOLD_PCT
from .exact         import ExactPELT, exact_update, steady_state_value
from .rpf           import RPFApprox, rpf_update, RPF_TABLE_Q15, RPF_ERROR_BOUND_PCT
from .tsa           import TSAApprox, tsa_update_degree2, tsa_update_degree3
from .pla           import PLAApprox, pla_update, build_pla_table
from .error_monitor import ErrorMonitor

__all__ = [
    # Constants
    "Y", "ALPHA", "SCHED_CAPACITY_SCALE", "FAIRNESS_THRESHOLD_PCT",
    # Exact baseline
    "ExactPELT", "exact_update", "steady_state_value",
    # Algorithm I
    "RPFApprox", "rpf_update", "RPF_TABLE_Q15", "RPF_ERROR_BOUND_PCT",
    # Algorithm II
    "TSAApprox", "tsa_update_degree2", "tsa_update_degree3",
    # Algorithm III
    "PLAApprox", "pla_update", "build_pla_table",
    # Error monitor
    "ErrorMonitor",
]

__version__ = "1.0.0"
__author__  = "Ronit Kumar"
__paper__   = "Approximate Computing in OS Process Scheduling (Phase 2)"
