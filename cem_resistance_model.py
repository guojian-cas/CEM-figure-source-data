"""Reference implementation of the Conversion Energy Model (CEM).

This module contains only the local CEM resistance closure, state update, and
a small explanatory plot. It does not solve the depth-integrated mass or
momentum equations, route flow over a DEM, or read case files.

The implementation follows the manuscript equations:

    C_eff = rho h e_c
    D_s = eta_s tau_b |u|
    D_i = eta_i rho h |u|**3 K_plus
    chi = 1 - exp(-Omega)
    c(chi) = c0 (1 - chi)**p
    mu(chi) = mu_f + (mu_s - mu_f) (1 - chi)**q
    1 / xi_eff(chi) = chi**r / xi

Arrays and scalars are both accepted through NumPy broadcasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
# Edit this block first, then run:
#
#     python cem_resistance_model.py
#
# The script will write OUTPUT_PNG in the same folder by default.
USER_PARAMETERS = {
    # Local flow state supplied by an external model, experiment, or scenario.
    "flow_depth_m": 3.0,
    "speed_m_s": 5.0,
    "slope_deg": 12.0,
    "k_plus_1_m": 0.02,
    "initial_chi": 0.0,
    "duration_s": 40.0,
    "dt_s": 0.25,
    # Material and resistance parameters.
    "rho_kg_m3": 2000.0,
    "cohesion_pa": 15000.0,
    "mu_s": 0.55,
    "mu_f": 0.08,
    "xi_m_s2": 2500.0,
    "ec_j_kg": 120.0,
    "eta_s": 1.0,
    "eta_i": 1.0,
    "p": 1.0,
    "q": 1.0,
    "r": 1.0,
    "lambda_w": 0.0,
    # Output.
    "output_png": "cem_model_demo.png",
}


@dataclass(frozen=True)
class CEMParameters:
    """Material and resistance parameters for the CEM closure."""

    rho: float = 2000.0
    g: float = 9.81
    cohesion_pa: float = 15000.0
    mu_s: float = 0.55
    mu_f: float = 0.08
    xi: float = 2500.0
    ec_j_kg: float = 120.0
    eta_s: float = 1.0
    eta_i: float = 1.0
    p: float = 1.0
    q: float = 1.0
    r: float = 1.0
    lambda_w: float = 0.0
    alpha_w: float = 0.0
    min_capacity_j_m2: float = 1.0e-12
    eps_speed: float = 1.0e-12


@dataclass(frozen=True)
class ResistanceTerms:
    """Basal resistance components in stress units (Pa)."""

    chi: np.ndarray
    cohesion_pa: np.ndarray
    friction_pa: np.ndarray
    velocity_pa: np.ndarray
    total_pa: np.ndarray
    mu: np.ndarray
    activation: np.ndarray
    xi_eff: np.ndarray


@dataclass(frozen=True)
class ConversionPower:
    """Conversion-power terms per unit basal area."""

    shear_w_m2: np.ndarray
    impact_w_m2: np.ndarray
    total_w_m2: np.ndarray
    capacity_j_m2: np.ndarray
    domega_dt: np.ndarray


@dataclass(frozen=True)
class CEMStep:
    """Result of one explicit local CEM update."""

    chi: np.ndarray
    omega: np.ndarray
    resistance: ResistanceTerms
    conversion_power: ConversionPower


def _clip01(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def state_from_omega(omega):
    """Return chi = 1 - exp(-Omega), clipped to [0, 1]."""

    omega_arr = np.maximum(np.asarray(omega, dtype=float), 0.0)
    return _clip01(1.0 - np.exp(-np.minimum(omega_arr, 745.0)))


def omega_from_state(chi):
    """Return Omega implied by chi = 1 - exp(-Omega)."""

    chi_arr = np.clip(np.asarray(chi, dtype=float), 0.0, 1.0 - 1.0e-15)
    return -np.log1p(-chi_arr)


def effective_capacity(
    h,
    params: CEMParameters = CEMParameters(),
    *,
    water_index=0.0,
    material_factor=1.0,
):
    """Return effective conversion capacity C_eff in J m^-2.

    water_index is a normalized hydrologic or pore-pressure weakening index.
    material_factor can be used for lithology, cementation, or grain-size
    modifiers. The default values reduce the expression to C_eff = rho h e_c.
    """

    h_arr = np.maximum(np.asarray(h, dtype=float), 0.0)
    water_factor = np.exp(-float(params.alpha_w) * np.asarray(water_index, dtype=float))
    ec_eff = float(params.ec_j_kg) * water_factor * np.asarray(material_factor, dtype=float)
    capacity = float(params.rho) * h_arr * np.maximum(ec_eff, 0.0)
    return np.maximum(capacity, float(params.min_capacity_j_m2))


def resistance_terms(
    h,
    speed,
    chi,
    params: CEMParameters = CEMParameters(),
    *,
    cos_slope=1.0,
    effective_normal_stress_pa: Optional[float] = None,
) -> ResistanceTerms:
    """Compute the CEM basal resistance components.

    Parameters
    ----------
    h:
        Flow depth in m.
    speed:
        Depth-averaged speed |u| in m s^-1.
    chi:
        Conversion state, where 0 is unconverted and 1 is fully converted.
    cos_slope:
        cos(theta), used only when effective_normal_stress_pa is not supplied.
    effective_normal_stress_pa:
        Optional effective normal stress. If omitted, the function uses
        rho g h cos(theta) and multiplies the friction term by (1 - lambda_w).
    """

    h_arr = np.maximum(np.asarray(h, dtype=float), 0.0)
    speed_arr = np.maximum(np.asarray(speed, dtype=float), 0.0)
    chi_arr = _clip01(chi)
    one_minus_chi = 1.0 - chi_arr

    cohesion = float(params.cohesion_pa) * np.power(one_minus_chi, float(params.p))
    mu = float(params.mu_f) + (float(params.mu_s) - float(params.mu_f)) * np.power(
        one_minus_chi, float(params.q)
    )
    activation = np.power(chi_arr, float(params.r))

    if effective_normal_stress_pa is None:
        normal = float(params.rho) * float(params.g) * h_arr * np.asarray(cos_slope, dtype=float)
        normal_eff = normal * (1.0 - np.clip(float(params.lambda_w), 0.0, 0.99))
    else:
        normal_eff = np.maximum(np.asarray(effective_normal_stress_pa, dtype=float), 0.0)

    friction = mu * normal_eff
    velocity = float(params.rho) * float(params.g) * speed_arr * speed_arr * activation / max(
        float(params.xi), 1.0e-12
    )
    total = cohesion + friction + velocity

    xi_eff = np.divide(
        float(params.xi),
        activation,
        out=np.full(np.broadcast_shapes(np.shape(activation), np.shape(speed_arr)), np.inf),
        where=activation > 0.0,
    )

    return ResistanceTerms(
        chi=chi_arr,
        cohesion_pa=cohesion,
        friction_pa=friction,
        velocity_pa=velocity,
        total_pa=total,
        mu=mu,
        activation=activation,
        xi_eff=xi_eff,
    )


def conversion_power(
    h,
    speed,
    chi,
    k_plus=0.0,
    params: CEMParameters = CEMParameters(),
    *,
    tau_b_pa=None,
    cos_slope=1.0,
    effective_normal_stress_pa: Optional[float] = None,
    water_index=0.0,
    material_factor=1.0,
) -> ConversionPower:
    """Compute CEM conversion power and dOmega/dt.

    k_plus is the positive topographic compression index K_plus. If tau_b_pa is
    omitted, the trial basal resistance is computed from the current chi.
    """

    h_arr = np.maximum(np.asarray(h, dtype=float), 0.0)
    speed_arr = np.maximum(np.asarray(speed, dtype=float), 0.0)
    if tau_b_pa is None:
        tau_b_pa = resistance_terms(
            h_arr,
            speed_arr,
            chi,
            params,
            cos_slope=cos_slope,
            effective_normal_stress_pa=effective_normal_stress_pa,
        ).total_pa
    else:
        tau_b_pa = np.maximum(np.asarray(tau_b_pa, dtype=float), 0.0)

    shear = float(params.eta_s) * tau_b_pa * speed_arr
    impact = (
        float(params.eta_i)
        * float(params.rho)
        * h_arr
        * np.power(speed_arr, 3.0)
        * np.maximum(np.asarray(k_plus, dtype=float), 0.0)
    )
    total = shear + impact
    capacity = effective_capacity(
        h_arr,
        params,
        water_index=water_index,
        material_factor=material_factor,
    )
    domega_dt = np.divide(total, capacity, out=np.zeros_like(total, dtype=float), where=capacity > 0.0)

    return ConversionPower(
        shear_w_m2=shear,
        impact_w_m2=impact,
        total_w_m2=total,
        capacity_j_m2=capacity,
        domega_dt=domega_dt,
    )


def update_state(
    chi,
    dt,
    h,
    speed,
    k_plus=0.0,
    params: CEMParameters = CEMParameters(),
    *,
    cos_slope=1.0,
    effective_normal_stress_pa: Optional[float] = None,
    water_index=0.0,
    material_factor=1.0,
) -> CEMStep:
    """Advance the local CEM state by one explicit time step.

    The update uses the current resistance as a trial resistance for conversion
    power and then recomputes the resistance with the updated chi.
    """

    chi_old = _clip01(chi)
    power = conversion_power(
        h,
        speed,
        chi_old,
        k_plus,
        params,
        cos_slope=cos_slope,
        effective_normal_stress_pa=effective_normal_stress_pa,
        water_index=water_index,
        material_factor=material_factor,
    )
    domega = np.maximum(np.asarray(dt, dtype=float), 0.0) * power.domega_dt
    chi_new = 1.0 - (1.0 - chi_old) * np.exp(-domega)
    chi_new = _clip01(chi_new)
    omega_new = omega_from_state(chi_new)
    resistance = resistance_terms(
        h,
        speed,
        chi_new,
        params,
        cos_slope=cos_slope,
        effective_normal_stress_pa=effective_normal_stress_pa,
    )
    return CEMStep(chi=chi_new, omega=omega_new, resistance=resistance, conversion_power=power)


def basal_traction_vector(u, v, tau_b_pa, eps=1.0e-12):
    """Return the basal traction vector opposite to the horizontal velocity."""

    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    tau_arr = np.asarray(tau_b_pa, dtype=float)
    speed = np.sqrt(u_arr * u_arr + v_arr * v_arr)
    denom = np.maximum(speed, float(eps))
    return -tau_arr * u_arr / denom, -tau_arr * v_arr / denom


def parameters_from_user_inputs(config=USER_PARAMETERS) -> CEMParameters:
    """Build CEMParameters from the editable USER_PARAMETERS block."""

    return CEMParameters(
        rho=float(config["rho_kg_m3"]),
        cohesion_pa=float(config["cohesion_pa"]),
        mu_s=float(config["mu_s"]),
        mu_f=float(config["mu_f"]),
        xi=float(config["xi_m_s2"]),
        ec_j_kg=float(config["ec_j_kg"]),
        eta_s=float(config["eta_s"]),
        eta_i=float(config["eta_i"]),
        p=float(config["p"]),
        q=float(config["q"]),
        r=float(config["r"]),
        lambda_w=float(config["lambda_w"]),
    )


def run_local_cem_demo(config=USER_PARAMETERS):
    """Run a single-cell explanatory CEM calculation.

    The prescribed h, speed, slope, and K_plus are held fixed. This is not a
    runout simulation; it only shows how conversion work drives chi and how chi
    changes the local basal resistance law.
    """

    params = parameters_from_user_inputs(config)
    h = float(config["flow_depth_m"])
    speed = float(config["speed_m_s"])
    k_plus = float(config["k_plus_1_m"])
    cos_slope = np.cos(np.deg2rad(float(config["slope_deg"])))
    dt = float(config["dt_s"])
    duration = float(config["duration_s"])
    n_steps = max(1, int(np.ceil(duration / dt)))

    time = np.linspace(0.0, n_steps * dt, n_steps + 1)
    chi = np.zeros(n_steps + 1)
    omega = np.zeros(n_steps + 1)
    ds = np.zeros(n_steps)
    di = np.zeros(n_steps)
    capacity = np.zeros(n_steps)
    tau_total = np.zeros(n_steps + 1)
    tau_cohesion = np.zeros(n_steps + 1)
    tau_friction = np.zeros(n_steps + 1)
    tau_velocity = np.zeros(n_steps + 1)

    chi[0] = float(config["initial_chi"])
    first_resistance = resistance_terms(h, speed, chi[0], params, cos_slope=cos_slope)
    tau_total[0] = first_resistance.total_pa
    tau_cohesion[0] = first_resistance.cohesion_pa
    tau_friction[0] = first_resistance.friction_pa
    tau_velocity[0] = first_resistance.velocity_pa

    for i in range(n_steps):
        step = update_state(
            chi=chi[i],
            dt=dt,
            h=h,
            speed=speed,
            k_plus=k_plus,
            params=params,
            cos_slope=cos_slope,
        )
        chi[i + 1] = step.chi
        omega[i + 1] = step.omega
        ds[i] = step.conversion_power.shear_w_m2
        di[i] = step.conversion_power.impact_w_m2
        capacity[i] = step.conversion_power.capacity_j_m2
        tau_total[i + 1] = step.resistance.total_pa
        tau_cohesion[i + 1] = step.resistance.cohesion_pa
        tau_friction[i + 1] = step.resistance.friction_pa
        tau_velocity[i + 1] = step.resistance.velocity_pa

    return {
        "params": params,
        "time_s": time,
        "power_time_s": time[:-1],
        "chi": chi,
        "omega": omega,
        "ds_w_m2": ds,
        "di_w_m2": di,
        "capacity_j_m2": capacity,
        "tau_total_pa": tau_total,
        "tau_cohesion_pa": tau_cohesion,
        "tau_friction_pa": tau_friction,
        "tau_velocity_pa": tau_velocity,
        "h_m": h,
        "speed_m_s": speed,
        "k_plus_1_m": k_plus,
        "cos_slope": cos_slope,
    }


def make_explanatory_figure(result, output_png):
    """Create a compact figure explaining work, state, and resistance."""

    import matplotlib.pyplot as plt

    out = Path(output_png)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out

    time = result["time_s"]
    power_time = result["power_time_s"]
    params = result["params"]
    h = result["h_m"]
    speed = result["speed_m_s"]
    cos_slope = result["cos_slope"]

    chi_curve = np.linspace(0.0, 1.0, 301)
    resistance_curve = resistance_terms(h, speed, chi_curve, params, cos_slope=cos_slope)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), dpi=180)
    fig.suptitle("Conversion Energy Model (CEM): local resistance closure", fontsize=13)

    ax = axes[0, 0]
    ax.plot(power_time, result["ds_w_m2"], color="#d55e00", lw=1.8, label="shear input Ds")
    ax.plot(power_time, result["di_w_m2"], color="#009e73", lw=1.8, label="impact input Di")
    ax.set_title("1. Motion supplies conversion work")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("power (W m$^{-2}$)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    line_omega = ax.plot(time, result["omega"], color="#777777", lw=1.8, label="Omega")[0]
    ax.set_title("2. Work capacity becomes state")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Omega")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    line_chi = ax2.plot(time, result["chi"], color="#cc79a7", lw=2.0, label="chi")[0]
    ax2.set_ylabel("chi")
    ax2.set_ylim(0.0, 1.02)
    ax.legend([line_omega, line_chi], ["Omega", "chi"], frameon=False, loc="center right")

    ax = axes[1, 0]
    ax.plot(time, result["tau_cohesion_pa"] / 1000.0, color="#0072b2", lw=1.8, label="cohesion")
    ax.plot(time, result["tau_friction_pa"] / 1000.0, color="#e69f00", lw=1.8, label="friction")
    ax.plot(time, result["tau_velocity_pa"] / 1000.0, color="#009e73", lw=1.8, label="velocity-squared")
    ax.plot(time, result["tau_total_pa"] / 1000.0, color="#000000", lw=2.0, label="total")
    ax.set_title("3. State changes resistance components")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("basal resistance (kPa)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(chi_curve, resistance_curve.cohesion_pa / 1000.0, color="#0072b2", lw=1.7, label="c(chi)")
    ax.plot(chi_curve, resistance_curve.friction_pa / 1000.0, color="#e69f00", lw=1.7, label="mu(chi)")
    ax.plot(chi_curve, resistance_curve.velocity_pa / 1000.0, color="#009e73", lw=1.7, label="Voellmy term")
    ax.plot(chi_curve, resistance_curve.total_pa / 1000.0, color="#000000", lw=2.0, label="total")
    ax.annotate("MC-like\nchi=0", xy=(0.0, resistance_curve.total_pa[0] / 1000.0), xytext=(0.06, 0.84),
                textcoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=8)
    ax.annotate("Voellmy-like\nchi=1", xy=(1.0, resistance_curve.total_pa[-1] / 1000.0), xytext=(0.63, 0.18),
                textcoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=8)
    ax.set_title("4. CEM connects the two end-member states")
    ax.set_xlabel("conversion state chi")
    ax.set_ylabel("basal resistance (kPa)")
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    fig.text(
        0.02,
        0.01,
        "This is a local closure demonstration only: h, speed, slope, and K_plus are prescribed inputs.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.95))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    result = run_local_cem_demo(USER_PARAMETERS)
    output = make_explanatory_figure(result, USER_PARAMETERS["output_png"])
    print("CEM local resistance demo")
    print(f"depth h = {result['h_m']:.3g} m")
    print(f"speed |u| = {result['speed_m_s']:.3g} m/s")
    print(f"K_plus = {result['k_plus_1_m']:.3g} 1/m")
    print(f"final chi = {result['chi'][-1]:.4f}")
    print(f"final Omega = {result['omega'][-1]:.4f}")
    print(f"final basal resistance = {result['tau_total_pa'][-1] / 1000.0:.2f} kPa")
    print(f"figure saved to: {output}")


if __name__ == "__main__":
    main()
