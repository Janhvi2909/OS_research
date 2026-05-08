"""
Error Monitor — Self-Healing Safety Component.

Phase 2 Report — Section V.C

The Error Monitor runs at a 1% sampling rate: on every 100th scheduling tick
it computes the exact PELT value and compares it against the approximate
accumulator. If the relative error exceeds the configured threshold (default 2%),
it triggers automatic fallback to exact arithmetic and logs a kernel warning.

This provides a safety net even if the static error analysis misses an edge case
(e.g., highly bursty workloads, SMP migration interference).

In the Linux kernel this would map to:
    - A per-CPU flag: sched_approx_fallback_active
    - A kernel warning via pr_warn_ratelimited()
    - A tracepoint: trace_sched_approx_fallback()
"""

from .constants import SCHED_CAPACITY_SCALE
from .exact import exact_update


class ErrorMonitor:
    """
    Sampling-based error monitor for the approximate PELT module.

    Samples 1% of scheduling ticks (every 100th by default) and computes
    the relative deviation between the approximate and exact accumulator.
    Automatically reverts to exact arithmetic if error exceeds threshold.

    Parameters
    ----------
    sample_rate     : float  Fraction of ticks to monitor (default 0.01 = 1%).
    error_threshold : float  Relative error threshold in [0, 1] (default 0.02 = 2%).
    """

    def __init__(self, sample_rate: float = 0.01, error_threshold: float = 0.02):
        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if not 0 < error_threshold < 1:
            raise ValueError(f"error_threshold must be in (0, 1), got {error_threshold}")

        self.sample_rate = sample_rate
        self.error_threshold = error_threshold

        # Shadow exact accumulator (used only during sampling ticks)
        self._exact_accumulator: float = 0.0

        # State
        self.fallback_active: bool = False
        self.tick_count: int = 0
        self.sample_interval: int = max(1, round(1.0 / sample_rate))

        # Statistics
        self.total_samples: int = 0
        self.fallback_triggers: int = 0
        self.max_observed_error: float = 0.0
        self._last_error: float = 0.0

        # Logged events
        self.events: list[dict] = []

    # -----------------------------------------------------------------------
    # Core interface
    # -----------------------------------------------------------------------

    def tick(self, approx_accumulator: float, r: float, n: int = 1) -> bool:
        """
        Process one scheduling tick through the error monitor.

        Updates the shadow exact accumulator on every tick. On sampling ticks,
        compares the approximate and exact accumulators. Triggers fallback if
        the error exceeds the threshold.

        Parameters
        ----------
        approx_accumulator : float  Current value of the approximate accumulator.
        r                  : float  CPU utilization fraction (same as used by approx).
        n                  : int    Elapsed periods (same as used by approx).

        Returns
        -------
        bool  True if fallback was triggered this tick, False otherwise.
        """
        self.tick_count += 1

        # Always maintain the exact shadow accumulator
        self._exact_accumulator = exact_update(self._exact_accumulator, r, n)

        # Only run the comparison on sampling ticks
        if self.tick_count % self.sample_interval != 0:
            return False

        return self._run_check(approx_accumulator)

    def _run_check(self, approx_value: float) -> bool:
        """Run the error check and return True if fallback was triggered."""
        self.total_samples += 1
        exact_value = self._exact_accumulator

        if exact_value == 0.0:
            return False

        relative_error = abs(approx_value - exact_value) / exact_value
        self._last_error = relative_error

        if relative_error > self.max_observed_error:
            self.max_observed_error = relative_error

        if relative_error > self.error_threshold:
            self.fallback_active = True
            self.fallback_triggers += 1
            self.events.append({
                "tick":            self.tick_count,
                "error":           relative_error,
                "threshold":       self.error_threshold,
                "approx_value":    approx_value,
                "exact_value":     exact_value,
            })
            return True
        return False

    def reset(self) -> None:
        """Reset all state (e.g., for a new task after migration)."""
        self._exact_accumulator = 0.0
        self.fallback_active = False
        self.tick_count = 0
        self.total_samples = 0
        self.fallback_triggers = 0
        self.max_observed_error = 0.0
        self._last_error = 0.0
        self.events.clear()

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def last_error(self) -> float:
        """Most recently measured relative error (on last sampling tick)."""
        return self._last_error

    @property
    def fallback_rate(self) -> float:
        """Fraction of sampling checks that triggered fallback."""
        if self.total_samples == 0:
            return 0.0
        return self.fallback_triggers / self.total_samples

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary dict of monitor statistics."""
        return {
            "total_ticks":          self.tick_count,
            "total_samples":        self.total_samples,
            "fallback_triggers":    self.fallback_triggers,
            "fallback_rate_%":      self.fallback_rate * 100,
            "max_observed_error_%": self.max_observed_error * 100,
            "fallback_active":      self.fallback_active,
            "sample_rate":          self.sample_rate,
            "error_threshold_%":    self.error_threshold * 100,
        }

    def __repr__(self) -> str:
        return (f"ErrorMonitor(sample_rate={self.sample_rate}, "
                f"threshold={self.error_threshold*100:.1f}%, "
                f"fallback={'ACTIVE' if self.fallback_active else 'off'}, "
                f"triggers={self.fallback_triggers}/{self.total_samples})")
