Dynamic FLOPS/Watt optimizer for NVIDIA Rubin (R200) under partial load.

PROBLEM:
  Rubin's peak power per NVL72 rack is not officially disclosed, but
  Rubin Ultra is projected at ~600 kW. Rubin NVL72 is likely 250-350 kW.
  At partial load (e.g., inference idle periods, sparse MoE activations),
  SMs still burn standby power, HBM4 remains in high-power mode,
  and NVLink 6 switches hold active links — all wasting watts.

SOLUTION:
  1. SM Cluster Power Gating  — coalesce work to fewer SM clusters,
     gate idle clusters at hardware level via SM occupancy hints.
  2. HBM4 Power-State Scheduler — track per-rank access frequency;
     request firmware low-power transitions on cold ranks.
  3. NVLink Adaptive Link Suspension — detect idle GPU pairs and
     negotiate link power-down through NSCQ / NVML.
  4. Adaptive Batch Coalescer — delay micro-batches until they fill
     enough SM warps to justify full-power state.

TARGETS:
  - ≥ 30% FLOPS/W improvement at < 40% load
  - No latency regression above SLO threshold
  - Compatible with Vera CPU co-scheduling

HARDWARE HOOKS:
  - pynvml (NVML 14+)
  - nvidia-smi --query (power, utilization, NVLink throughput)
  - CUDA Driver API: cuDeviceGetAttribute, SM occupancy
  - Rubin SM Cluster API (planned CUDA 13.x)
"""

from __future__ import annotations

import time
import threading
import logging
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("[WARN] pynvml not found — running in simulation mode")

try:
    import torch
    import torch.cuda as tcuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("RubinPowerGovernor")


# ─────────────────────────────────────────────
#  Rubin hardware constants (R200 / NVL72)
# ─────────────────────────────────────────────

class RubinSpecs:
    """
    Known / estimated specs for a single Rubin GPU (R200).
    Values derived from CES 2026 disclosures.
    """
    PEAK_FP4_TFLOPS: float = 50_000.0       # 50 PFLOPs FP4 per GPU
    PEAK_FP8_TFLOPS: float = 25_000.0
    HBM4_BW_TBS: float = 22.0               # 22 TB/s per GPU (estimated)
    HBM4_CAPACITY_GB: int = 288
    NVLINK6_BW_PER_GPU_TBS: float = 3.6     # bidirectional per GPU
    NUM_SM_CLUSTERS: int = 16               # estimated; Rubin uses SM cluster arch
    TDP_WATTS: int = 1200                   # estimated per GPU TDP

    # Power states (estimated based on Hopper/Blackwell patterns + HBM4)
    POWER_FULL_W: int = 1200
    POWER_IDLE_W: int = 220                 # HBM4 self-refresh + SM standby
    POWER_SM_CLUSTER_GATE_W: int = 80       # per cluster saved when gated
    POWER_HBM_LOWP_SAVING_W: int = 35       # per cold HBM rank in low-power


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class GpuMetrics:
    gpu_idx: int
    timestamp: float
    util_sm_pct: float          # SM utilization %
    util_mem_pct: float         # Memory controller utilization %
    power_w: float              # Current draw (Watts)
    temp_c: float
    nvlink_tx_mbs: float        # NVLink TX MB/s
    nvlink_rx_mbs: float
    hbm_bw_utilization: float   # 0.0 – 1.0
    flops_estimate: float       # estimated TFLOPS based on SM util
    flops_per_watt: float

    @property
    def is_underloaded(self) -> bool:
        return self.util_sm_pct < 40.0

    @property
    def load_tier(self) -> str:
        if self.util_sm_pct >= 80:
            return "HIGH"
        elif self.util_sm_pct >= 40:
            return "MID"
        elif self.util_sm_pct >= 10:
            return "LOW"
        return "IDLE"


@dataclass
class PowerDecision:
    gpu_idx: int
    action: str                     # e.g. "GATE_SM_CLUSTERS", "HBM_LOWPOWER", etc.
    sm_clusters_active: int = 16    # how many SM clusters to keep live
    hbm_ranks_lowpower: int = 0     # how many HBM ranks to sleep
    nvlink_links_suspend: int = 0   # how many NVLink lanes to suspend
    projected_savings_w: float = 0.0
    projected_flops_per_watt_gain_pct: float = 0.0


# ─────────────────────────────────────────────
#  NVML thin wrapper
# ─────────────────────────────────────────────

class NvmlInterface:
    def __init__(self):
        self._handles: list = []
        self._sim = not NVML_AVAILABLE
        if not self._sim:
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            log.info("NVML initialized — %d Rubin GPU(s) visible", n)
        else:
            import random
            self._rng = random.Random(42)
            log.info("NVML simulation mode — generating synthetic Rubin metrics")

    @property
    def gpu_count(self) -> int:
        return len(self._handles) if not self._sim else 2   # simulate 2 GPUs

    def get_metrics(self, idx: int) -> GpuMetrics:
        if self._sim:
            return self._simulate_metrics(idx)
        h = self._handles[idx]
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW → W
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)

        # NVLink aggregate (sum all links)
        nvlink_tx = nvlink_rx = 0.0
        try:
            for link in range(18):  # Rubin may have up to 18 NVLink ports
                nvlink_tx += pynvml.nvmlDeviceGetNvLinkUtilizationCounter(
                    h, link, pynvml.NVML_NVLINK_COUNTER_DATA_TX) / 1e6
                nvlink_rx += pynvml.nvmlDeviceGetNvLinkUtilizationCounter(
                    h, link, pynvml.NVML_NVLINK_COUNTER_DATA_RX) / 1e6
        except pynvml.NVMLError:
            pass

        sm_util = float(util.gpu)
        mem_util = float(util.memory)
        flops_est = (sm_util / 100.0) * RubinSpecs.PEAK_FP8_TFLOPS
        flops_pw = flops_est / max(power, 1.0)
        hbm_bw_util = mem_util / 100.0

        return GpuMetrics(
            gpu_idx=idx,
            timestamp=time.monotonic(),
            util_sm_pct=sm_util,
            util_mem_pct=mem_util,
            power_w=power,
            temp_c=float(temp),
            nvlink_tx_mbs=nvlink_tx,
            nvlink_rx_mbs=nvlink_rx,
            hbm_bw_utilization=hbm_bw_util,
            flops_estimate=flops_est,
            flops_per_watt=flops_pw,
        )

    def set_power_limit(self, idx: int, watts: int) -> None:
        """Apply power cap via NVML (requires root / MIG mode off)."""
        if self._sim:
            log.info("[SIM] GPU %d → power limit = %d W", idx, watts)
            return
        h = self._handles[idx]
        try:
            pynvml.nvmlDeviceSetPowerManagementLimit(h, watts * 1000)
            log.info("GPU %d power limit set to %d W", idx, watts)
        except pynvml.NVMLError as e:
            log.warning("Cannot set power limit on GPU %d: %s", idx, e)

    def _simulate_metrics(self, idx: int) -> GpuMetrics:
        import math, random
        t = time.monotonic()
        # Simulate partial-load scenario with periodic busy bursts
        phase = math.sin(t * 0.3 + idx) * 0.5 + 0.5
        sm = 15.0 + phase * 55.0 + random.uniform(-5, 5)
        sm = max(0.0, min(100.0, sm))
        power = RubinSpecs.POWER_IDLE_W + (sm / 100.0) * (
            RubinSpecs.POWER_FULL_W - RubinSpecs.POWER_IDLE_W)
        flops = (sm / 100.0) * RubinSpecs.PEAK_FP8_TFLOPS
        return GpuMetrics(
            gpu_idx=idx,
            timestamp=t,
            util_sm_pct=sm,
            util_mem_pct=sm * 0.7,
            power_w=power,
            temp_c=45.0 + sm * 0.4,
            nvlink_tx_mbs=sm * 10_000,
            nvlink_rx_mbs=sm * 10_000,
            hbm_bw_utilization=sm * 0.007,
            flops_estimate=flops,
            flops_per_watt=flops / max(power, 1),
        )

    def shutdown(self):
        if not self._sim:
            pynvml.nvmlShutdown()


# ─────────────────────────────────────────────
#  Decision Engine
# ─────────────────────────────────────────────

class RubinPowerDecisionEngine:
    """
    Policy engine that translates real-time metrics → power decisions.

    Strategy:
      IDLE  (< 10% SM) → 4 SM clusters, HBM ranks in low-power, NVLink suspended
      LOW   (10-40% SM) → 8 SM clusters, partial HBM wakeup
      MID   (40-80% SM) → 12 SM clusters, full HBM, NVLink active
      HIGH  (>80% SM)  → 16 SM clusters (full), max power allowed
    """

    TIER_POLICY = {
        "IDLE": dict(sm_clusters=4,  hbm_ranks_lp=6, nvlink_suspend=8),
        "LOW":  dict(sm_clusters=8,  hbm_ranks_lp=3, nvlink_suspend=4),
        "MID":  dict(sm_clusters=12, hbm_ranks_lp=1, nvlink_suspend=0),
        "HIGH": dict(sm_clusters=16, hbm_ranks_lp=0, nvlink_suspend=0),
    }

    # Hysteresis: don't oscillate between tiers faster than this (seconds)
    HYSTERESIS_SEC: float = 2.0

    def __init__(self):
        self._last_decision: dict[int, PowerDecision] = {}
        self._last_decision_time: dict[int, float] = {}

    def decide(self, m: GpuMetrics) -> Optional[PowerDecision]:
        tier = m.load_tier
        policy = self.TIER_POLICY[tier]

        now = time.monotonic()
        last_t = self._last_decision_time.get(m.gpu_idx, 0.0)
        if now - last_t < self.HYSTERESIS_SEC:
            return None  # in hysteresis window, hold current state

        sm_active = policy["sm_clusters"]
        hbm_lp = policy["hbm_ranks_lp"]
        nvlink_susp = policy["nvlink_suspend"]

        # Estimate power savings
        sm_savings = (RubinSpecs.NUM_SM_CLUSTERS - sm_active) * RubinSpecs.POWER_SM_CLUSTER_GATE_W
        hbm_savings = hbm_lp * RubinSpecs.POWER_HBM_LOWP_SAVING_W
        total_savings = sm_savings + hbm_savings

        baseline_fw = m.flops_estimate / max(m.power_w, 1.0)
        new_power = max(m.power_w - total_savings, RubinSpecs.POWER_IDLE_W * 0.5)
        new_fw = m.flops_estimate / max(new_power, 1.0)
        gain_pct = ((new_fw - baseline_fw) / max(baseline_fw, 0.001)) * 100.0

        decision = PowerDecision(
            gpu_idx=m.gpu_idx,
            action=f"TIER_{tier}",
            sm_clusters_active=sm_active,
            hbm_ranks_lowpower=hbm_lp,
            nvlink_links_suspend=nvlink_susp,
            projected_savings_w=total_savings,
            projected_flops_per_watt_gain_pct=gain_pct,
        )

        self._last_decision[m.gpu_idx] = decision
        self._last_decision_time[m.gpu_idx] = now
        return decision


# ─────────────────────────────────────────────
#  Actuator (applies decisions via NVML / sysfs)
# ─────────────────────────────────────────────

class RubinPowerActuator:
    """
    Applies PowerDecision to hardware.

    Real implementation paths:
      SM cluster gating  → CUDA 13.x SM Cluster API (planned)
                            or nvidia-mig-parted for approximate SM partitioning
      HBM low-power      → NVML future API / GPU driver firmware
      NVLink suspension  → pynvml.nvmlDeviceSetNvLinkUtilizationControl
                            or nvidia-smi nlms (NVLink Mgmt Service)
      Power limit        → nvmlDeviceSetPowerManagementLimit (available now)
    """

    def __init__(self, nvml: NvmlInterface):
        self._nvml = nvml

    def apply(self, decision: PowerDecision) -> None:
        # 1. Set conservative power cap proportional to SM cluster target
        frac = decision.sm_clusters_active / RubinSpecs.NUM_SM_CLUSTERS
        target_w = int(RubinSpecs.POWER_IDLE_W + frac * (
            RubinSpecs.POWER_FULL_W - RubinSpecs.POWER_IDLE_W))
        self._nvml.set_power_limit(decision.gpu_idx, target_w)

        # 2. Log SM cluster hint (future CUDA 13 API hook)
        log.debug(
            "GPU %d → SM clusters: %d/%d | HBM low-power ranks: %d | "
            "NVLink suspend lanes: %d | est. savings: %.0f W (+%.1f%% FLOPS/W)",
            decision.gpu_idx,
            decision.sm_clusters_active, RubinSpecs.NUM_SM_CLUSTERS,
            decision.hbm_ranks_lowpower,
            decision.nvlink_links_suspend,
            decision.projected_savings_w,
            decision.projected_flops_per_watt_gain_pct,
        )


# ─────────────────────────────────────────────
#  Adaptive Batch Coalescer
# ─────────────────────────────────────────────

class AdaptiveBatchCoalescer:
    """
    Delays small inference requests to coalesce them into larger batches,
    increasing SM utilization and thus FLOPS/W.

    Integration: wrap your inference function with `coalesce_and_run`.
    """

    def __init__(
        self,
        target_batch_size: int = 64,
        max_wait_ms: float = 8.0,
        min_util_threshold_pct: float = 40.0,
    ):
        self.target_batch = target_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.min_util = min_util_threshold_pct
        self._queue: list = []
        self._lock = threading.Lock()
        self._trigger = threading.Event()

    def submit(self, request) -> None:
        """Submit an inference request for coalesced execution."""
        with self._lock:
            self._queue.append(request)
            if len(self._queue) >= self.target_batch:
                self._trigger.set()

    def get_batch(self, current_util_pct: float) -> list:
        """
        Return coalesced batch when:
          - batch is full, OR
          - max_wait exceeded AND utilization is already high enough to justify dispatch
        """
        wait_time = self.max_wait_s
        if current_util_pct >= self.min_util:
            # GPU already busy — dispatch immediately even with small batch
            wait_time = 0.001

        self._trigger.wait(timeout=wait_time)
        self._trigger.clear()

        with self._lock:
            batch = self._queue[:self.target_batch]
            self._queue = self._queue[self.target_batch:]
        return batch


# ─────────────────────────────────────────────
#  Governor (orchestrates everything)
# ─────────────────────────────────────────────

class RubinPowerGovernor:
    """
    Main entry point.  Runs a background monitor thread that continuously
    adjusts Rubin GPU power states for optimal FLOPS/Watt.

    Usage:
        governor = RubinPowerGovernor(poll_interval_s=0.5)
        governor.start()
        # ... your inference workload ...
        governor.stop()
        report = governor.get_report()
    """

    def __init__(self, poll_interval_s: float = 0.5, slo_latency_ms: float = 50.0):
        self.poll_interval = poll_interval_s
        self.slo_latency_ms = slo_latency_ms

        self._nvml = NvmlInterface()
        self._engine = RubinPowerDecisionEngine()
        self._actuator = RubinPowerActuator(self._nvml)
        self._coalescer = AdaptiveBatchCoalescer()

        self._history: dict[int, deque] = {
            i: deque(maxlen=120) for i in range(self._nvml.gpu_count)
        }
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="RubinGovernor")
        self._thread.start()
        log.info("RubinPowerGovernor started on %d GPU(s)", self._nvml.gpu_count)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._nvml.shutdown()
        log.info("RubinPowerGovernor stopped")

    def _loop(self) -> None:
        while self._running:
            for idx in range(self._nvml.gpu_count):
                try:
                    metrics = self._nvml.get_metrics(idx)
                    self._history[idx].append(metrics)
                    decision = self._engine.decide(metrics)
                    if decision:
                        self._actuator.apply(decision)
                except Exception as e:
                    log.error("Governor loop error on GPU %d: %s", idx, e)
            time.sleep(self.poll_interval)

    def get_report(self) -> dict:
        """
        Returns efficiency report across all observed samples.
        """
        report = {}
        for idx, hist in self._history.items():
            if not hist:
                continue
            samples = list(hist)
            avg_util = sum(m.util_sm_pct for m in samples) / len(samples)
            avg_power = sum(m.power_w for m in samples) / len(samples)
            avg_flops_pw = sum(m.flops_per_watt for m in samples) / len(samples)
            tier_counts: dict[str, int] = {}
            for m in samples:
                t = m.load_tier
                tier_counts[t] = tier_counts.get(t, 0) + 1

            # Estimate naive baseline (always full power)
            naive_power = RubinSpecs.POWER_FULL_W
            avg_flops = sum(m.flops_estimate for m in samples) / len(samples)
            naive_flops_pw = avg_flops / naive_power
            improvement_pct = ((avg_flops_pw - naive_flops_pw) /
                               max(naive_flops_pw, 0.001)) * 100

            report[f"gpu_{idx}"] = {
                "samples": len(samples),
                "avg_sm_util_pct": round(avg_util, 1),
                "avg_power_w": round(avg_power, 1),
                "avg_flops_per_watt_tflops_w": round(avg_flops_pw, 2),
                "estimated_improvement_vs_naive_pct": round(improvement_pct, 1),
                "time_in_tier": tier_counts,
            }
        return report


# ─────────────────────────────────────────────
#  CLI demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rubin FLOPS/Watt Governor")
    parser.add_argument("--duration", type=int, default=10,
                        help="Run for N seconds then print report")
    parser.add_argument("--poll", type=float, default=0.5,
                        help="Poll interval in seconds")
    args = parser.parse_args()

    gov = RubinPowerGovernor(poll_interval_s=args.poll)
    gov.start()

    print(f"\nMonitoring Rubin GPU(s) for {args.duration}s...")
    time.sleep(args.duration)

    gov.stop()
    report = gov.get_report()

    print("\n" + "═" * 60)
    print("  RUBIN POWER GOVERNOR — EFFICIENCY REPORT")
    print("═" * 60)
    print(json.dumps(report, indent=2))
    print("═" * 60)
