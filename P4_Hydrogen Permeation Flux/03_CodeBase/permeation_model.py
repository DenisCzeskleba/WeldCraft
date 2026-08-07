"""Numerical 1D hydrogen-permeation models used by P4.

The production path in this module is deliberately numerical and explicit.  A
mobile lattice concentration is advanced with Fick's second law.  Optional
trap occupancy is advanced at the same time, and the mobile concentration is
reduced/increased by exactly the corresponding trapped amount.

Trapping terminology matters:

* McNabb and Foster introduced kinetic capture/detrapping equations in
  "A new analysis of the diffusion of hydrogen in iron and ferritic steels",
  Transactions of the Metallurgical Society of AIME 227 (1963), 618-627.
* Oriani's local-equilibrium treatment is a related fast-exchange
  approximation, not another name for the kinetic model implemented here:
  R. A. Oriani, Acta Metallurgica 18 (1970), 147-157,
  DOI: 10.1016/0001-6160(70)90078-7.

P4 uses the McNabb-Foster kinetic picture.  The energy-well drawings produced
by the renderer are qualitative symbols for retention and capacity; they are
not literal spatial geometries or an Oriani-equilibrium calculation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
from numba import njit


class SimulationError(RuntimeError):
    """Raised when a requested explicit calculation is invalid or unstable."""


class SimulationCancelled(SimulationError):
    """Raised when a cooperative GUI cancellation request is observed."""


@dataclass(frozen=True)
class SurfaceHistory:
    """Effective entry-side concentration history in units of C_ref."""

    base_concentration: float = 1.0
    delta_concentration: float = 0.0
    onset_time_ref: float = 0.5
    time_constant_ref: float = 0.3
    transition_mode: str = "exponential"

    def value(self, time_ref: float) -> float:
        if time_ref < self.onset_time_ref or self.delta_concentration == 0.0:
            return self.base_concentration
        if self.transition_mode == "step":
            return self.final_concentration
        elapsed = time_ref - self.onset_time_ref
        return self.base_concentration + self.delta_concentration * (
            1.0 - math.exp(-elapsed / self.time_constant_ref)
        )

    @property
    def final_concentration(self) -> float:
        return self.base_concentration + self.delta_concentration


@dataclass(frozen=True)
class TrapConfig:
    """Dimensionless McNabb-Foster kinetic parameters.

    ``capacity_ratio`` is the maximum trapped concentration divided by C_ref.
    ``capture_rate_ref`` has units 1/tau_ref at C=C_ref.  Release is expressed
    as an intuitive half-time and converted internally to k_det=ln(2)/t_half.
    """

    enabled: bool = False
    capacity_ratio: float = 0.0
    capture_rate_ref: float = 20.0
    release_half_time_ref: float = 1.0
    initial_occupancy: float = 0.0

    @property
    def release_rate_ref(self) -> float:
        return math.log(2.0) / self.release_half_time_ref


@dataclass(frozen=True)
class PrefillConfig:
    """Prepare an aged, symmetric concentration bulge before permeation."""

    enabled: bool = False
    initial_fraction: float = 0.20
    target_center_fraction: float = 0.10
    maximum_age_time_ref: float = 20.0


@dataclass(frozen=True)
class SimulationConfig:
    """Complete configuration for one normalized 1D permeation case."""

    label: str = "Reference"
    length_ratio: float = 1.0
    diffusivity_ratio: float = 1.0
    solubility_ratio: float = 1.0
    downstream_concentration: float = 0.0
    end_time_ref: float = 4.0
    n_nodes: int = 201
    n_output: int = 401
    diffusion_safety: float = 0.45
    reaction_safety: float = 0.08
    max_internal_steps: int = 50_000_000
    reference_length_mm: float = 1.0
    reference_diffusivity_mm2_s: float = 1.0
    reference_concentration_mol_mm3: Optional[float] = None
    surface: SurfaceHistory = field(default_factory=SurfaceHistory)
    traps: TrapConfig = field(default_factory=TrapConfig)
    prefill: PrefillConfig = field(default_factory=PrefillConfig)

    @property
    def tau_ref_seconds(self) -> float:
        return self.reference_length_mm**2 / self.reference_diffusivity_mm2_s

    @property
    def physical_reference_flux(self) -> Optional[float]:
        if self.reference_concentration_mol_mm3 is None:
            return None
        return (
            self.reference_diffusivity_mm2_s
            * self.reference_concentration_mol_mm3
            / self.reference_length_mm
        )

    @property
    def steady_flux_common_reference(self) -> float:
        inlet = self.solubility_ratio * self.surface.final_concentration
        return self.diffusivity_ratio * (
            inlet - self.downstream_concentration
        ) / self.length_ratio

    def with_changes(self, **changes: Any) -> "SimulationConfig":
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SimulationConfig":
        data = dict(values)
        data["surface"] = SurfaceHistory(**dict(data.get("surface", {})))
        data["traps"] = TrapConfig(**dict(data.get("traps", {})))
        data["prefill"] = PrefillConfig(**dict(data.get("prefill", {})))
        return cls(**data)


@dataclass
class SimulationResult:
    """Time-resolved fields and downstream response from one simulation."""

    config: SimulationConfig
    x_ref: np.ndarray
    time_ref: np.ndarray
    mobile_concentration: np.ndarray
    trap_occupancy: np.ndarray
    outlet_flux_common: np.ndarray
    inlet_concentration: np.ndarray
    total_hydrogen: np.ndarray
    initial_profile: np.ndarray
    prefill_age_time_ref: float = 0.0
    internal_steps: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def fourier_number(self) -> np.ndarray:
        return (
            self.config.diffusivity_ratio
            * self.time_ref
            / self.config.length_ratio**2
        )

    @property
    def time_seconds(self) -> np.ndarray:
        return self.time_ref * self.config.tau_ref_seconds

    @property
    def time_minutes(self) -> np.ndarray:
        return self.time_seconds / 60.0

    def flux(self, mode: str = "common_reference") -> np.ndarray:
        if mode == "common_reference":
            return self.outlet_flux_common.copy()
        if mode == "per_curve":
            steady = self.config.steady_flux_common_reference
            if steady == 0.0:
                raise SimulationError("Per-curve normalization requires non-zero steady flux.")
            return self.outlet_flux_common / steady
        if mode == "physical":
            reference_flux = self.config.physical_reference_flux
            if reference_flux is None:
                raise SimulationError(
                    "Physical flux requires reference_concentration_mol_mm3."
                )
            return self.outlet_flux_common * reference_flux
        raise ValueError(f"Unknown flux mode: {mode}")


def mixed_output_times(end_time_ref: float, count: int) -> np.ndarray:
    """Return exactly ``count`` times with extra resolution near breakthrough."""

    if end_time_ref <= 0.0:
        raise ValueError("end_time_ref must be positive.")
    if count < 3:
        raise ValueError("At least three output samples are required.")
    remaining = count - 1
    early_count = max(0, min(remaining - 2, int(round(remaining * 0.35))))
    later_count = remaining - early_count
    transition = 0.15 * end_time_ref
    early = (
        np.geomspace(
            max(end_time_ref * 1.0e-6, 1.0e-12),
            transition,
            early_count,
            endpoint=False,
        )
        if early_count
        else np.empty(0, dtype=np.float64)
    )
    later = np.linspace(transition, end_time_ref, later_count)
    return np.concatenate(([0.0], early, later)).astype(np.float64)


def validate_config(config: SimulationConfig) -> None:
    if config.length_ratio <= 0.0 or config.diffusivity_ratio <= 0.0:
        raise ValueError("Length and diffusivity ratios must be positive.")
    if config.solubility_ratio <= 0.0:
        raise ValueError("Solubility ratio must be positive.")
    if config.n_nodes < 11 or config.n_nodes % 2 == 0:
        raise ValueError("n_nodes must be an odd integer of at least 11.")
    if config.n_output < 3:
        raise ValueError("n_output must be at least three.")
    if not (0.0 < config.diffusion_safety < 0.5):
        raise ValueError("diffusion_safety must lie between zero and 0.5.")
    if not (0.0 < config.reaction_safety < 0.5):
        raise ValueError("reaction_safety must lie between zero and 0.5.")
    if config.surface.time_constant_ref <= 0.0:
        raise ValueError("Surface time constant must be positive.")
    if config.surface.transition_mode not in {"exponential", "step"}:
        raise ValueError("Surface transition_mode must be 'exponential' or 'step'.")
    if config.surface.base_concentration < 0.0 or config.surface.final_concentration < 0.0:
        raise ValueError("Surface concentrations cannot be negative.")
    trap = config.traps
    if trap.capacity_ratio < 0.0 or trap.capture_rate_ref < 0.0:
        raise ValueError("Trap capacity and capture rate cannot be negative.")
    if trap.release_half_time_ref <= 0.0:
        raise ValueError("Trap release half-time must be positive.")
    if not (0.0 <= trap.initial_occupancy <= 1.0):
        raise ValueError("Initial trap occupancy must lie between zero and one.")
    prefill = config.prefill
    if prefill.enabled:
        if not (0.0 < prefill.target_center_fraction < prefill.initial_fraction):
            raise ValueError("Prefill target must lie between zero and the initial fraction.")
        if prefill.maximum_age_time_ref <= 0.0:
            raise ValueError("Maximum prefill age time must be positive.")


@njit(cache=True)
def reaction_exchange_step(
    mobile: float,
    occupancy: float,
    capacity: float,
    capture_rate: float,
    release_rate: float,
    dt: float,
) -> tuple[float, float]:
    """Advance one local trap exchange while conserving C_mobile+N_t*theta."""

    exchange_rate = capture_rate * mobile * (1.0 - occupancy) - release_rate * occupancy
    new_occupancy = occupancy + dt * exchange_rate
    new_mobile = mobile - capacity * dt * exchange_rate
    return new_mobile, new_occupancy


@njit(cache=True)
def _surface_value(
    time_ref: float,
    base: float,
    delta: float,
    onset: float,
    tau: float,
    step_mode: bool,
) -> float:
    if time_ref < onset or delta == 0.0:
        return base
    if step_mode:
        return base + delta
    return base + delta * (1.0 - math.exp(-(time_ref - onset) / tau))


@njit(cache=True)
def _outlet_flux(concentration: np.ndarray, diffusivity: float, dx: float) -> float:
    derivative = (
        3.0 * concentration[-1]
        - 4.0 * concentration[-2]
        + concentration[-3]
    ) / (2.0 * dx)
    return -diffusivity * derivative


@njit(cache=True)
def _total_hydrogen(
    concentration: np.ndarray,
    occupancy: np.ndarray,
    capacity: float,
    dx: float,
) -> float:
    total = 0.0
    last = concentration.size - 1
    for index in range(concentration.size):
        weight = 0.5 if index == 0 or index == last else 1.0
        total += weight * (concentration[index] + capacity * occupancy[index])
    return total * dx


@njit(cache=True, nogil=True)
def _age_uniform_prefill(
    n_nodes: int,
    length: float,
    diffusivity: float,
    initial_fraction: float,
    target_fraction: float,
    diffusion_safety: float,
    maximum_time: float,
    max_steps: int,
    cancel_flag: np.ndarray,
) -> tuple[np.ndarray, float, int, int]:
    dx = length / (n_nodes - 1)
    dt_limit = diffusion_safety * dx * dx / diffusivity
    concentration = np.full(n_nodes, initial_fraction, dtype=np.float64)
    concentration[0] = 0.0
    concentration[-1] = 0.0
    updated = concentration.copy()
    previous = concentration.copy()
    age_time = 0.0
    steps = 0
    center = n_nodes // 2

    while concentration[center] > target_fraction:
        if steps % 4096 == 0 and cancel_flag[0] != 0:
            return concentration, age_time, steps, 2
        if steps >= max_steps or age_time >= maximum_time:
            return concentration, age_time, steps, 1
        dt = min(dt_limit, maximum_time - age_time)
        previous[:] = concentration
        previous_center = previous[center]
        updated[0] = 0.0
        updated[-1] = 0.0
        for index in range(1, n_nodes - 1):
            laplacian = (
                concentration[index + 1]
                - 2.0 * concentration[index]
                + concentration[index - 1]
            ) / (dx * dx)
            updated[index] = concentration[index] + dt * diffusivity * laplacian
        concentration, updated = updated, concentration
        steps += 1
        if concentration[center] <= target_fraction:
            denominator = previous_center - concentration[center]
            fraction = 1.0 if denominator <= 0.0 else (previous_center - target_fraction) / denominator
            for index in range(n_nodes):
                concentration[index] = previous[index] + fraction * (
                    concentration[index] - previous[index]
                )
            age_time += fraction * dt
            concentration[0] = 0.0
            concentration[-1] = 0.0
            return concentration, age_time, steps, 0
        age_time += dt
    return concentration, age_time, steps, 0


@njit(cache=True, nogil=True)
def _run_explicit(
    initial_concentration: np.ndarray,
    initial_occupancy: float,
    output_times: np.ndarray,
    length: float,
    diffusivity: float,
    solubility: float,
    downstream: float,
    surface_base: float,
    surface_delta: float,
    surface_onset: float,
    surface_tau: float,
    surface_step_mode: bool,
    trap_capacity: float,
    capture_rate: float,
    release_rate: float,
    diffusion_safety: float,
    reaction_safety: float,
    max_steps: int,
    cancel_flag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    n_output = output_times.size
    n_nodes = initial_concentration.size
    dx = length / (n_nodes - 1)
    diffusion_dt = diffusion_safety * dx * dx / diffusivity
    concentration = initial_concentration.copy()
    occupancy = np.full(n_nodes, initial_occupancy, dtype=np.float64)
    occupancy[0] = 0.0
    occupancy[-1] = 0.0
    next_concentration = concentration.copy()
    next_occupancy = occupancy.copy()

    mobile_history = np.empty((n_output, n_nodes), dtype=np.float64)
    trap_history = np.empty((n_output, n_nodes), dtype=np.float64)
    flux_history = np.empty(n_output, dtype=np.float64)
    inlet_history = np.empty(n_output, dtype=np.float64)
    total_history = np.empty(n_output, dtype=np.float64)

    time_now = 0.0
    inlet = solubility * _surface_value(
        time_now,
        surface_base,
        surface_delta,
        surface_onset,
        surface_tau,
        surface_step_mode,
    )
    concentration[0] = inlet
    concentration[-1] = downstream
    mobile_history[0, :] = concentration
    trap_history[0, :] = occupancy
    flux_history[0] = _outlet_flux(concentration, diffusivity, dx)
    inlet_history[0] = inlet
    total_history[0] = _total_hydrogen(concentration, occupancy, trap_capacity, dx)
    steps = 0
    tolerance = 1.0e-10

    for output_index in range(1, n_output):
        if cancel_flag[0] != 0:
            return (
                mobile_history,
                trap_history,
                flux_history,
                inlet_history,
                total_history,
                steps,
                3,
            )
        target_time = output_times[output_index]
        while time_now < target_time - 1.0e-15:
            if steps % 4096 == 0 and cancel_flag[0] != 0:
                return (
                    mobile_history,
                    trap_history,
                    flux_history,
                    inlet_history,
                    total_history,
                    steps,
                    3,
                )
            if steps >= max_steps:
                return (
                    mobile_history,
                    trap_history,
                    flux_history,
                    inlet_history,
                    total_history,
                    steps,
                    1,
                )

            dt = min(diffusion_dt, target_time - time_now)
            if trap_capacity > 0.0 and capture_rate + release_rate > 0.0:
                maximum_mobile = 0.0
                for index in range(n_nodes):
                    maximum_mobile = max(maximum_mobile, concentration[index])
                reaction_scale = capture_rate * (maximum_mobile + trap_capacity) + release_rate
                if reaction_scale > 0.0:
                    dt = min(dt, reaction_safety / reaction_scale)

            new_time = time_now + dt
            new_inlet = solubility * _surface_value(
                new_time,
                surface_base,
                surface_delta,
                surface_onset,
                surface_tau,
                surface_step_mode,
            )
            next_concentration[0] = new_inlet
            next_concentration[-1] = downstream
            next_occupancy[0] = 0.0
            next_occupancy[-1] = 0.0

            for index in range(1, n_nodes - 1):
                laplacian = (
                    concentration[index + 1]
                    - 2.0 * concentration[index]
                    + concentration[index - 1]
                ) / (dx * dx)
                exchange_rate = (
                    capture_rate * concentration[index] * (1.0 - occupancy[index])
                    - release_rate * occupancy[index]
                )
                next_concentration[index] = concentration[index] + dt * (
                    diffusivity * laplacian - trap_capacity * exchange_rate
                )
                next_occupancy[index] = occupancy[index] + dt * exchange_rate

                if (
                    next_concentration[index] < -tolerance
                    or next_occupancy[index] < -tolerance
                    or next_occupancy[index] > 1.0 + tolerance
                ):
                    return (
                        mobile_history,
                        trap_history,
                        flux_history,
                        inlet_history,
                        total_history,
                        steps,
                        2,
                    )

            concentration, next_concentration = next_concentration, concentration
            occupancy, next_occupancy = next_occupancy, occupancy
            time_now = new_time
            steps += 1

        mobile_history[output_index, :] = concentration
        trap_history[output_index, :] = occupancy
        flux_history[output_index] = _outlet_flux(concentration, diffusivity, dx)
        inlet_history[output_index] = concentration[0]
        total_history[output_index] = _total_hydrogen(
            concentration, occupancy, trap_capacity, dx
        )

    return (
        mobile_history,
        trap_history,
        flux_history,
        inlet_history,
        total_history,
        steps,
        0,
    )


def calculate_metrics(result: SimulationResult) -> Dict[str, float]:
    time = result.time_ref
    flux = result.outlet_flux_common
    target = result.config.steady_flux_common_reference
    metrics: Dict[str, float] = {}

    for percentage in (10, 50, 90):
        threshold = target * percentage / 100.0
        indices = np.flatnonzero(flux >= threshold)
        key = f"t{percentage}"
        if indices.size == 0:
            metrics[key] = float("nan")
            continue
        index = int(indices[0])
        if index == 0:
            metrics[key] = float(time[0])
            continue
        f0, f1 = flux[index - 1], flux[index]
        fraction = 0.0 if f1 == f0 else (threshold - f0) / (f1 - f0)
        metrics[key] = float(time[index - 1] + fraction * (time[index] - time[index - 1]))

    cumulative = np.zeros_like(flux)
    cumulative[1:] = np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(time))
    late_start = min(time.size - 2, max(0, int(0.75 * time.size)))
    slope, intercept = np.polyfit(time[late_start:], cumulative[late_start:], 1)
    metrics["time_lag"] = float(-intercept / slope) if slope > 0.0 else float("nan")
    metrics["peak_flux"] = float(np.max(flux))
    tail_start = max(0, int(0.95 * flux.size))
    metrics["final_flux"] = float(np.mean(flux[tail_start:]))
    metrics["overshoot"] = max(0.0, metrics["peak_flux"] - metrics["final_flux"])
    return metrics


def simulate_case(
    config: SimulationConfig,
    output_times: Optional[Iterable[float]] = None,
    cancel_flag: Optional[np.ndarray] = None,
) -> SimulationResult:
    """Run one explicit numerical case and return time-resolved flux/fields."""

    validate_config(config)
    if cancel_flag is None:
        cancel_flag = np.zeros(1, dtype=np.int8)
    if cancel_flag[0] != 0:
        raise SimulationCancelled("Simulation cancelled.")
    if output_times is None:
        times = mixed_output_times(config.end_time_ref, config.n_output)
    else:
        times = np.asarray(list(output_times), dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or times[0] != 0.0:
            raise ValueError("Output times must be a one-dimensional sequence starting at zero.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("Output times must be strictly increasing.")

    length = config.length_ratio
    diffusivity = config.diffusivity_ratio
    initial_profile = np.zeros(config.n_nodes, dtype=np.float64)
    prefill_age = 0.0
    prefill_steps = 0
    if config.prefill.enabled:
        initial_profile, prefill_age, prefill_steps, age_status = _age_uniform_prefill(
            config.n_nodes,
            length,
            diffusivity,
            config.prefill.initial_fraction,
            config.prefill.target_center_fraction,
            config.diffusion_safety,
            config.prefill.maximum_age_time_ref,
            config.max_internal_steps,
            cancel_flag,
        )
        if age_status == 2:
            raise SimulationCancelled("Simulation cancelled during prefill ageing.")
        if age_status != 0:
            raise SimulationError(
                "Prefill aging did not reach the target centre concentration within the configured limit."
            )

    trap = config.traps
    trap_active = trap.enabled and trap.capacity_ratio > 0.0
    trap_capacity = trap.capacity_ratio if trap_active else 0.0
    capture_rate = trap.capture_rate_ref if trap_active else 0.0
    release_rate = trap.release_rate_ref if trap_active else 0.0
    initial_occupancy = trap.initial_occupancy if trap_active else 0.0

    (
        mobile,
        trapped,
        flux,
        inlet,
        total,
        steps,
        status,
    ) = _run_explicit(
        initial_profile,
        initial_occupancy,
        times,
        length,
        diffusivity,
        config.solubility_ratio,
        config.downstream_concentration,
        config.surface.base_concentration,
        config.surface.delta_concentration,
        config.surface.onset_time_ref,
        config.surface.time_constant_ref,
        config.surface.transition_mode == "step",
        trap_capacity,
        capture_rate,
        release_rate,
        config.diffusion_safety,
        config.reaction_safety,
        config.max_internal_steps,
        cancel_flag,
    )
    if status == 3:
        raise SimulationCancelled("Simulation cancelled.")
    if status == 1:
        raise SimulationError("Explicit solver exceeded max_internal_steps.")
    if status == 2:
        raise SimulationError(
            "Explicit trap update produced a negative concentration or invalid occupancy; "
            "reduce reaction_safety or use less extreme kinetics."
        )

    x_ref = np.linspace(0.0, length, config.n_nodes)
    result = SimulationResult(
        config=config,
        x_ref=x_ref,
        time_ref=times,
        mobile_concentration=mobile,
        trap_occupancy=trapped,
        outlet_flux_common=flux,
        inlet_concentration=inlet,
        total_hydrogen=total,
        initial_profile=initial_profile,
        prefill_age_time_ref=prefill_age,
        internal_steps=steps + prefill_steps,
    )
    result.metrics = calculate_metrics(result)
    return result
