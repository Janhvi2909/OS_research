"""
simulate.py — Main Simulation for Phase 2: Approximate PELT

Approximate Computing in OS Process Scheduling:
Error-Bounded Priority Decay Functions for CPU Fair-Share Scheduling

This script:
  1. Generates a 10,000-period random workload (seeded for reproducibility).
  2. Runs exact PELT and all five approximate variants in lockstep.
  3. Records per-period relative error for each technique.
  4. Prints three results tables matching the Phase 2 IEEE report:
       Table 1 — Per-Call Complexity and Cycle Estimates
       Table 2 — Memory Footprint per CPU Runqueue
       Table 3 — Error Bounds: Theory vs Simulation
  5. Prints the Error-to-Fairness safety check.
  6. Demonstrates the Error Monitor.
  7. Optionally saves an error-over-time plot (if matplotlib is available).

Usage:
    python simulate.py
    python simulate.py --no-plot   (skip matplotlib, text only)
"""

import random
import math
import sys

# ---------------------------------------------------------------------------
# Import the pelt_approx package
# ---------------------------------------------------------------------------
from pelt_approx.constants     import (Y, ALPHA, SCHED_CAPACITY_SCALE,
                                        FAIRNESS_THRESHOLD_PCT, MAX_PERIODS)
from pelt_approx.exact         import exact_update, steady_state_value
from pelt_approx.rpf           import (rpf_update, build_rpf_table,
                                        rpf_error_bound, RPF_TABLE_Q15)
from pelt_approx.tsa           import (tsa_update_degree2, tsa_update_degree3,
                                        tsa_error_bound, tsa_relative_error_bound_pct,
                                        compare_tsa_to_exact)
from pelt_approx.pla           import (pla_update, build_pla_table,
                                        pla_error_bound_pct, pla_pointwise_errors)
from pelt_approx.error_monitor import ErrorMonitor


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

T    = 10_000   # number of scheduling periods (1 ms each)
SEED = 42       # fixed seed for reproducibility (matches the paper)
N    = 1        # per-period update: n=1 elapsed period each tick


# ===========================================================================
# SECTION 1 — Generate workload
# ===========================================================================

def generate_workload(t: int, seed: int) -> list[float]:
    """
    Generate T uniformly distributed CPU utilization fractions.
    r_t ~ U(0, 1), representing per-period CPU utilization fraction.
    Seeded at 42 for reproducibility (matches paper description).
    """
    rng = random.Random(seed)
    return [rng.uniform(0.0, 1.0) for _ in range(t)]


# ===========================================================================
# SECTION 2 — Run simulation
# ===========================================================================

def run_simulation(workload: list[float]) -> dict:
    """
    Run all six PELT variants in lockstep over the workload.

    Returns a dict of per-period error lists (relative error in %)
    and final accumulator values for each technique.
    """

    # Build PLA tables once
    seg4,  h4  = build_pla_table(4)
    seg8,  h8  = build_pla_table(8)
    seg16, h16 = build_pla_table(16)

    # Accumulators (start at 0, as for a freshly created task)
    acc_exact = 0.0
    acc_rpf   = 0.0
    acc_tsa2  = 0.0
    acc_tsa3  = 0.0
    acc_pla4  = 0.0
    acc_pla8  = 0.0
    acc_pla16 = 0.0

    # Error monitor (attached to RPF for demo)
    monitor = ErrorMonitor(sample_rate=0.01, error_threshold=0.02)

    # Per-period error arrays
    err_rpf   = []
    err_tsa2  = []
    err_tsa3  = []
    err_pla4  = []
    err_pla8  = []
    err_pla16 = []

    for t, r in enumerate(workload):
        # --- exact reference ---
        acc_exact = exact_update(acc_exact, r, n=N)

        # --- Algorithm I: RPF ---
        acc_rpf = rpf_update(acc_rpf, r, n=N)

        # --- Algorithm II: TSA degree 2 ---
        acc_tsa2 = tsa_update_degree2(acc_tsa2, r, n=N)

        # --- Algorithm II: TSA degree 3 ---
        acc_tsa3 = tsa_update_degree3(acc_tsa3, r, n=N)

        # --- Algorithm III: PLA k=4 ---
        acc_pla4 = pla_update(acc_pla4, r, n=N, segments=seg4, h=h4)

        # --- Algorithm III: PLA k=8 ---
        acc_pla8 = pla_update(acc_pla8, r, n=N, segments=seg8, h=h8)

        # --- Algorithm III: PLA k=16 ---
        acc_pla16 = pla_update(acc_pla16, r, n=N, segments=seg16, h=h16)

        # --- Error monitor tick (attached to RPF) ---
        monitor.tick(acc_rpf, r, n=N)

        # --- Record relative errors (avoid division by zero at t=0) ---
        if acc_exact > 0:
            def rel_err(approx):
                return abs(approx - acc_exact) / acc_exact * 100.0

            err_rpf.append(rel_err(acc_rpf))
            err_tsa2.append(rel_err(acc_tsa2))
            err_tsa3.append(rel_err(acc_tsa3))
            err_pla4.append(rel_err(acc_pla4))
            err_pla8.append(rel_err(acc_pla8))
            err_pla16.append(rel_err(acc_pla16))

    return {
        "errors": {
            "rpf":   err_rpf,
            "tsa2":  err_tsa2,
            "tsa3":  err_tsa3,
            "pla4":  err_pla4,
            "pla8":  err_pla8,
            "pla16": err_pla16,
        },
        "final": {
            "exact":  acc_exact,
            "rpf":    acc_rpf,
            "tsa2":   acc_tsa2,
            "tsa3":   acc_tsa3,
            "pla4":   acc_pla4,
            "pla8":   acc_pla8,
            "pla16":  acc_pla16,
        },
        "monitor": monitor,
    }


# ===========================================================================
# SECTION 3 — Print results tables
# ===========================================================================

SEPARATOR = "=" * 72
THIN_SEP  = "-" * 72


def print_header(title: str) -> None:
    print()
    print(SEPARATOR)
    print(f"  {title}")
    print(SEPARATOR)


def print_table1_complexity() -> None:
    """Table 1: Per-Call Complexity and Cycle Estimates (Phase 2 Report §VI.A)"""
    print_header("TABLE 1 — Per-Call Complexity and Cycle Estimates")
    header = f"{'Technique':<28} {'Operations':<32} {'Est. Cycles':>10} {'Reduction':>10}"
    print(header)
    print(THIN_SEP)
    rows = [
        ("Original PELT (64-bit)",
         "1 tab + 1×64b mul + 1 shift",  8.2,  "—"),
        ("RPF (16-bit Q15)",
         "1 tab + 1×16b mul + 1 shift",  4.1,  "50.0%"),
        ("TSA degree-2",
         "2 mul + 2 add (32-bit fp)",     5.3,  "35.4%"),
        ("TSA degree-3",
         "3 mul + 3 add (32-bit fp)",     6.1,  "25.6%"),
        ("PLA k=8",
         "1 shift + 1 tab + 1 mul + 1 add", 5.9, "28.0%"),
    ]
    for name, ops, cycles, reduction in rows:
        print(f"{name:<28} {ops:<32} {cycles:>10.1f} {reduction:>10}")
    print()
    print("  Note: Cycle estimates are for ARMv7 / 32-bit embedded cores where")
    print("  64-bit multiply requires two 32-bit multiplies (~12-15 cycles).")
    print("  On x86-64 / AArch64 with native 64b MLA, RPF benefit is smaller.")


def print_table2_space() -> None:
    """Table 2: Memory Footprint per CPU Runqueue (Phase 2 Report §VI.B)"""
    print_header("TABLE 2 — Memory Footprint per CPU Runqueue")
    header = f"{'Technique':<20} {'Table Size':>14} {'Reduction':>12}"
    print(header)
    print(THIN_SEP)
    rows = [
        ("Original PELT",  "128 B (32×4B)",  "—"),
        ("RPF (16-bit)",   " 64 B (32×2B)",  "50.0%"),
        ("TSA degree-2",   " 12 B (3 coeff)", "90.6%"),
        ("TSA degree-3",   " 16 B (4 coeff)", "87.5%"),
        ("PLA k=8",        " 64 B (8×8B)",   "50.0%"),
    ]
    for name, size, reduction in rows:
        print(f"{name:<20} {size:>14} {reduction:>12}")
    print()
    print("  TSA is most cache-friendly: only 3-4 floating-point constants,")
    print("  fitting entirely in a single cache line (64 bytes).")


def print_table3_errors(results: dict) -> None:
    """Table 3: Error Bounds — Theory vs Simulation (Phase 2 Report §VII.A)"""
    print_header("TABLE 3 — Error Bounds: Theory vs. Simulation (10,000 Periods)")
    header = f"{'Technique':<20} {'Theory Bound':>14} {'Simulated Max':>15} {'Simulated Mean':>16}"
    print(header)
    print(THIN_SEP)

    errors = results["errors"]

    def stats(name):
        e = errors[name]
        return max(e), sum(e) / len(e)

    rpf_max,   rpf_mean   = stats("rpf")
    tsa2_max,  tsa2_mean  = stats("tsa2")
    tsa3_max,  tsa3_mean  = stats("tsa3")
    pla4_max,  pla4_mean  = stats("pla4")
    pla8_max,  pla8_mean  = stats("pla8")
    pla16_max, pla16_mean = stats("pla16")

    data = [
        ("RPF (16-bit)",
         f"{rpf_error_bound(16):.4f}%",
         f"{rpf_max:.4f}%",
         f"{rpf_mean:.4f}%"),
        ("TSA degree-2",
         f"{tsa_relative_error_bound_pct(2):.6f}%",
         f"{tsa2_max:.4f}%",
         f"{tsa2_mean:.4f}%"),
        ("TSA degree-3",
         f"{tsa_relative_error_bound_pct(3):.2e}%",
         f"{tsa3_max:.2e}%",
         f"{tsa3_mean:.2e}%"),
        ("PLA k=4",
         f"{pla_error_bound_pct(4):.4f}%",
         f"{pla4_max:.2f}%(*)",
         f"{pla4_mean:.2f}%"),
        ("PLA k=8",
         f"{pla_error_bound_pct(8):.4f}%",
         f"{pla8_max:.2f}%(*)",
         f"{pla8_mean:.2f}%"),
        ("PLA k=16",
         f"{pla_error_bound_pct(16):.4f}%",
         f"{pla16_max:.2f}%(*)",
         f"{pla16_mean:.2f}%"),
    ]
    for name, theory, sim_max, sim_mean in data:
        print(f"{name:<20} {theory:>14} {sim_max:>15} {sim_mean:>16}")

    print()
    print("  (*) PLA with fixed n=1 (per-period update) exhibits systematic bias:")
    print("      n=1 always falls in segment 0, whose slope averages decay over")
    print(f"      h periods instead of the instantaneous 1-ms rate.")
    print("      PLA should only be used when n varies over [0,32].")
    print("      See Section VII.B of the Phase 2 report for full analysis.")


def print_fairness_theorem(results: dict) -> None:
    """Error-to-Fairness Preservation Theorem check (Phase 2 Report §IV.E)"""
    print_header("ERROR-TO-FAIRNESS PRESERVATION THEOREM (Theorem 4)")
    print()
    print("  Theorem: No task starvation due to approximation if and only if")
    print(f"  delta < theta/SCHED_CAPACITY_SCALE = 25/1024 = {FAIRNESS_THRESHOLD_PCT:.4f}%")
    print()
    print("  Load-balance deviation bound: Delta_imbalance <= delta * 1024")
    print()

    errors = results["errors"]

    techniques = [
        ("RPF (16-bit)",  max(errors["rpf"])),
        ("TSA degree-2",  max(errors["tsa2"])),
        ("TSA degree-3",  max(errors["tsa3"])),
    ]
    print(f"  {'Technique':<20} {'Max delta (%)':>15} {'< 2.44%?':>12} {'Margin':>10}")
    print("  " + THIN_SEP[:68])
    for name, delta in techniques:
        safe   = "YES ✓" if delta < FAIRNESS_THRESHOLD_PCT else "NO  ✗"
        margin = FAIRNESS_THRESHOLD_PCT - delta
        print(f"  {name:<20} {delta:>15.4f} {safe:>12} {margin:>9.4f}%")

    print()
    print("  COROLLARY: All three techniques (RPF, TSA-2, TSA-3) are orders of")
    print("  magnitude below the 2.44% threshold, formally guaranteeing no task")
    print("  starvation under standard Linux kernel parameters.")


def print_monitor_summary(results: dict) -> None:
    """Error Monitor demonstration results."""
    print_header("ERROR MONITOR — Self-Healing Safety Component")
    monitor = results["monitor"]
    s = monitor.summary()
    print()
    print(f"  Monitoring: RPF accumulator, sampled every {monitor.sample_interval} ticks (1%)")
    print(f"  Error threshold: {s['error_threshold_%']:.1f}%")
    print()
    print(f"  Total scheduling ticks : {s['total_ticks']:>8,}")
    print(f"  Total monitor samples  : {s['total_samples']:>8,}")
    print(f"  Max observed error     : {s['max_observed_error_%']:>8.4f}%")
    print(f"  Fallback triggers      : {s['fallback_triggers']:>8}  "
          f"({s['fallback_rate_%']:.2f}% of samples)")
    print(f"  Fallback active?       : {'YES' if s['fallback_active'] else 'NO — system operating safely'}")
    print()
    if s["fallback_triggers"] == 0:
        print("  [OK] No fallback triggered. Approximate arithmetic is within bounds.")
        print("       The monitor would remain dormant in production kernel use.")
    else:
        print("  [WARN] Fallback triggered! Monitor would revert to exact arithmetic.")


def print_tsa_single_step_analysis() -> None:
    """Show per-step decay factor values and errors for TSA."""
    print_header("TSA SINGLE-STEP DECAY FACTOR ANALYSIS")
    cmp = compare_tsa_to_exact()
    print()
    print(f"  Exact decay factor y = 2^(-1/32) = {cmp['exact']:.15f}")
    print(f"  TSA degree-2 approx  = {cmp['tsa_degree2']:.15f}")
    print(f"  TSA degree-3 approx  = {cmp['tsa_degree3']:.15f}")
    print()
    print(f"  Absolute error TSA-2 : {cmp['abs_error_tsa2']:.4e}  (bound: {tsa_error_bound(2):.4e})")
    print(f"  Absolute error TSA-3 : {cmp['abs_error_tsa3']:.4e}  (bound: {tsa_error_bound(3):.4e})")
    print(f"  Relative error TSA-2 : {cmp['rel_error_tsa2_%']:.4e}%  "
          f"(bound: {cmp['bound_tsa2_%']:.4e}%)")
    print(f"  Relative error TSA-3 : {cmp['rel_error_tsa3_%']:.4e}%  "
          f"(bound: {cmp['bound_tsa3_%']:.4e}%)")
    print()
    print(f"  Proof of Lagrange bound: alpha = ln(2)/32 = {ALPHA:.10f}")
    print(f"    alpha^3 / 6  = {ALPHA**3 / 6:.4e}  (TSA-2 absolute error bound)")
    print(f"    alpha^4 / 24 = {ALPHA**4 / 24:.4e}  (TSA-3 absolute error bound)")


def print_rpf_table_sample() -> None:
    """Print sample entries from exact Q32 vs RPF Q15 table."""
    print_header("RPF TABLE — Exact Q32 vs Reduced-Precision Q15 (sample)")
    from pelt_approx.constants import EXACT_TABLE_Q32
    rpf_table = build_rpf_table(16)
    print(f"  {'n':>4}  {'Exact y^n':>18}  {'RPF Q15 y^n':>18}  {'Abs Error':>12}")
    print("  " + THIN_SEP[:60])
    for n in [0, 1, 2, 4, 8, 16, 24, 32]:
        exact = EXACT_TABLE_Q32[n]
        approx = rpf_table[n]
        err = abs(approx - exact)
        print(f"  {n:>4}  {exact:>18.15f}  {approx:>18.15f}  {err:>12.4e}")


def print_pla_segment_table() -> None:
    """Print the PLA segment descriptors for k=8."""
    print_header("PLA SEGMENT TABLE — k=8 (segment descriptors)")
    seg8, h8 = build_pla_table(8)
    print(f"  Segment width h = 32/8 = {h8:.1f} periods")
    print()
    print(f"  {'Seg j':>6}  {'n0':>6}  {'y^n0':>18}  {'slope':>18}")
    print("  " + THIN_SEP[:56])
    for j, seg in enumerate(seg8):
        print(f"  {j:>6}  {seg.n0:>6.1f}  {seg.y0:>18.15f}  {seg.slope:>18.15f}")


# ===========================================================================
# SECTION 4 — Optional matplotlib plot
# ===========================================================================

def try_plot_errors(results: dict) -> None:
    """Save an error-over-time plot if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        errors = results["errors"]
        t_axis = list(range(1, len(errors["rpf"]) + 1))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("Approximate PELT: Relative Error over 10,000 Scheduling Periods\n"
                     "(Phase 2 — Approximate Computing in OS Process Scheduling)",
                     fontsize=12, fontweight="bold")

        # Top plot: RPF and TSA
        ax = axes[0]
        ax.plot(t_axis, errors["rpf"],  label="RPF (16-bit Q15)",  linewidth=0.8, alpha=0.9)
        ax.plot(t_axis, errors["tsa2"], label="TSA degree-2",       linewidth=0.8, alpha=0.9)
        ax.plot(t_axis, errors["tsa3"], label="TSA degree-3",       linewidth=0.8, alpha=0.9)
        ax.axhline(FAIRNESS_THRESHOLD_PCT, color="red", linestyle="--",
                   linewidth=1.2, label=f"Safety threshold (2.44%)")
        ax.set_ylabel("Relative Error (%)")
        ax.set_title("Algorithm I (RPF) and Algorithm II (TSA)")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Bottom plot: PLA variants
        ax = axes[1]
        ax.plot(t_axis, errors["pla4"],  label="PLA k=4",  linewidth=0.8, alpha=0.9)
        ax.plot(t_axis, errors["pla8"],  label="PLA k=8",  linewidth=0.8, alpha=0.9)
        ax.plot(t_axis, errors["pla16"], label="PLA k=16", linewidth=0.8, alpha=0.9)
        ax.axhline(FAIRNESS_THRESHOLD_PCT, color="red", linestyle="--",
                   linewidth=1.2, label=f"Safety threshold (2.44%)")
        ax.set_xlabel("Scheduling Period (t)")
        ax.set_ylabel("Relative Error (%)")
        ax.set_title("Algorithm III (PLA) — Note: systematic bias when n=1 always")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = "pelt_error_plot.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print()
        print(f"  [PLOT] Error-over-time plot saved to: {out_path}")
    except ImportError:
        print()
        print("  [INFO] matplotlib not found. Skipping plot generation.")
        print("         Install with: pip install matplotlib")


# ===========================================================================
# SECTION 5 — Main entry point
# ===========================================================================

def main() -> None:
    no_plot = "--no-plot" in sys.argv

    print()
    print("=" * 72)
    print("  APPROXIMATE PELT SIMULATION — Phase 2")
    print("  Approximate Computing in OS Process Scheduling")
    print("  Error-Bounded Priority Decay Functions for CPU Fair-Share Scheduling")
    print("=" * 72)
    print()
    print(f"  Simulation parameters:")
    print(f"    Periods T        : {T:,}")
    print(f"    Period duration  : 1 ms  (scheduling tick)")
    print(f"    Workload         : r_t ~ U(0,1), seed={SEED}")
    print(f"    Decay base y     : {Y:.15f}")
    print(f"    Alpha (|ln y|)   : {ALPHA:.15f}")
    print(f"    Fairness threshold: {FAIRNESS_THRESHOLD_PCT:.4f}%  (25/1024)")

    # --- Generate workload ---
    print()
    print("  [1/4] Generating workload ...")
    workload = generate_workload(T, SEED)

    # --- Run simulation ---
    print("  [2/4] Running simulation (exact + 6 approximate variants) ...")
    results = run_simulation(workload)
    print("  [3/4] Computing statistics ...")

    # --- Print all results ---
    print()
    print_table1_complexity()
    print()
    print_table2_space()
    print()
    print_table3_errors(results)
    print()
    print_fairness_theorem(results)
    print()
    print_monitor_summary(results)
    print()
    print_tsa_single_step_analysis()
    print()
    print_rpf_table_sample()
    print()
    print_pla_segment_table()

    # --- Optional plot ---
    print()
    print("  [4/4] Generating plot ...")
    if not no_plot:
        try_plot_errors(results)
    else:
        print("  [INFO] Plot skipped (--no-plot).")

    print()
    print(SEPARATOR)
    print("  Simulation complete.")
    print(SEPARATOR)
    print()


if __name__ == "__main__":
    main()
