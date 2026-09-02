"""
Plant-Wide Topology: Fresh Feed -> Mixer -> CSTR -> Flash Separator -> Recycle
================================================================================
Physics-Informed Multi-Agent Architecture -- Plant-Wide Simulation (v3 FINAL)

Topology:
  Fresh Feed (pure A, 300 K, 5 bar)
       |
       v
   [MIXER] ----> [CSTR (exothermic, V=1 m^3)] ----> [FLASH DRUM]
       ^          A -> B, jacket cooling               |        |
       |                                            VAPOR    LIQUID
       |                                           (A-rich)  (B-rich)
       |                                              |        |
       +---<-- [RECYCLE] <-- [SPLITTER] <--+----------+    PRODUCT
                (condensed)                |
                                        PURGE

Separation model:
  The flash drum operates at reduced pressure with a pre-flash heater.
  Component A (n-Hexane, lighter) preferentially vaporises.
  A constant-alpha VLE model (alpha_AB = K_A/K_B) is used, calibrated
  against Clausius-Clapeyron at the operating conditions.

  At T_flash ~ 360 K, P_flash = 65 kPa (partial vacuum, standard for
  light-hydrocarbon recovery):
    K_A ~ 2.65, K_B ~ 0.81
    alpha_AB = K_A / K_B ~ 3.27

Reaction:   A -> B   (irreversible, 1st order, exothermic)
  A_0 = 1.0e6 1/s, Ea = 49,456 J/mol, dH_rxn = -25,000 J/mol
"""

import os, sys, math, random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

# ============================================================================
#  PHYSICAL CONSTANTS
# ============================================================================
R_GAS       = 8.314
A_PRE       = 1.0e6
EA          = 49456.0
DH_RXN      = -25000.0
CP          = 3500.0
RHO         = 850.0
MW_A        = 86.18
MW_B        = 92.14
V_REACTOR   = 1.0
UA_COOLANT_NOM = 5000.0
T_MAX_SAFE  = 400.0
P_FRESH     = 506625.0

# Flash drum
P_FLASH_NOM = 65000.0      # Pa (0.65 bar -- partial vacuum, standard for light HCs)
P_FLASH_MIN = 45000.0      # Pa (minimum safe operating pressure)
FLASH_HOLDUP_VOL = 2.0     # m^3  (flash drum liquid capacity)
FLASH_MIN_LIQ_FRAC = 0.50  # minimum liquid holdup fraction (50% of design)

# VLE: Clausius-Clapeyron parameters
DHVAP_A = 28850.0    # J/mol (n-Hexane)
DHVAP_B = 33180.0    # J/mol (Toluene)
TBP_A   = 341.9      # K (n-Hexane BP @ 1 atm)
TBP_B   = 383.8      # K (Toluene BP @ 1 atm)
P_REF   = 101325.0   # Pa (1 atm reference)


def arrhenius_k(T: float) -> float:
    return A_PRE * math.exp(-EA / (R_GAS * T))


def psat_A(T: float) -> float:
    """Antoine/Clausius-Clapeyron vapor pressure for A (n-Hexane)."""
    return P_REF * math.exp(DHVAP_A / R_GAS * (1.0/TBP_A - 1.0/T))


def psat_B(T: float) -> float:
    """Antoine/Clausius-Clapeyron vapor pressure for B (Toluene)."""
    return P_REF * math.exp(DHVAP_B / R_GAS * (1.0/TBP_B - 1.0/T))


# ============================================================================
#  DWSIM PYTHON.NET BRIDGE (simplified -- see v1 for full implementation)
# ============================================================================
DWSIM_PATH = os.environ.get("DWSIM_PATH", r"C:\Users\Public\Documents\DWSIM")
DWSIM_AVAILABLE = False

def _init_dwsim():
    global DWSIM_AVAILABLE
    try:
        import clr
        for asm in ["CapeOpen.dll", "DWSIM.Automation.dll", "DWSIM.Interfaces.dll",
                     "DWSIM.SharedClasses.dll", "DWSIM.Thermodynamics.dll",
                     "DWSIM.UnitOperations.dll", "DWSIM.GlobalSettings.dll"]:
            clr.AddReference(os.path.join(DWSIM_PATH, asm))
        DWSIM_AVAILABLE = True
        print("[DWSIM] Assemblies loaded.")
    except Exception as e:
        print(f"[DWSIM] Not available ({e}). Using analytical engine.")

_init_dwsim()


# ============================================================================
#  ANALYTICAL ENGINE
# ============================================================================

@dataclass
class StreamState:
    """Process stream thermodynamic state."""
    mdot: float = 0.0       # kg/s
    T: float = 300.0        # K
    P: float = 506625.0     # Pa
    xA: float = 1.0         # mass fraction A
    xB: float = 0.0         # mass fraction B

    @property
    def CA(self):
        """Molar concentration of A (mol/m^3)."""
        return self.xA * RHO * 1000.0 / MW_A

    @property
    def CB(self):
        return self.xB * RHO * 1000.0 / MW_B

    @property
    def mdot_A(self):
        return self.mdot * self.xA

    @property
    def mdot_B(self):
        return self.mdot * self.xB

    @property
    def zA_mol(self):
        """Mole fraction of A."""
        nA = self.xA / MW_A
        nB = self.xB / MW_B
        nt = nA + nB
        return nA / max(nt, 1e-15)

    def copy(self):
        return StreamState(self.mdot, self.T, self.P, self.xA, self.xB)


def solve_mixer(fresh: StreamState, recycle: StreamState) -> StreamState:
    """Adiabatic, isobaric mixer."""
    m = fresh.mdot + recycle.mdot
    if m < 1e-9:
        return StreamState(0, fresh.T, fresh.P, fresh.xA, fresh.xB)
    xA = (fresh.mdot_A + recycle.mdot_A) / m
    T = (fresh.mdot * fresh.T + recycle.mdot * recycle.T) / m
    return StreamState(m, T, min(fresh.P, recycle.P), xA, 1.0 - xA)


def solve_cstr(feed: StreamState, UA: float = UA_COOLANT_NOM,
               T_coolant: float = 290.0) -> StreamState:
    """Coupled mass + energy balance for exothermic CSTR."""
    F_vol = feed.mdot / RHO
    tau = V_REACTOR / max(F_vol, 1e-12)
    CA_in = feed.CA
    CB_in = feed.CB

    T = feed.T
    for _ in range(1000):
        k = arrhenius_k(T)
        CA_out = CA_in / (1.0 + k * tau)
        heat_rxn = (-DH_RXN) * k * CA_out / (RHO * CP)
        T_new = (
            (F_vol / V_REACTOR) * feed.T
            + heat_rxn
            + (UA / (RHO * CP * V_REACTOR)) * T_coolant
        ) / (
            (F_vol / V_REACTOR)
            + (UA / (RHO * CP * V_REACTOR))
        )
        if abs(T_new - T) < 1e-7:
            break
        T = 0.3 * T_new + 0.7 * T

    k_f = arrhenius_k(T)
    CA_out = CA_in / (1.0 + k_f * tau)
    CB_out = CB_in + (CA_in - CA_out)  # stoichiometric B production

    mass_A = CA_out * MW_A / 1000.0
    mass_B = CB_out * MW_B / 1000.0
    total = mass_A + mass_B
    xA = mass_A / max(total, 1e-15)
    return StreamState(feed.mdot, T, feed.P, xA, 1.0 - xA)


def solve_flash(feed: StreamState,
                P_flash: float = P_FLASH_NOM) -> Tuple[StreamState, StreamState, float]:
    """
    Rigorous isothermal flash using Clausius-Clapeyron + Rachford-Rice.

    The flash drum operates at reduced pressure P_flash.
    Component A (n-Hexane, more volatile) preferentially vaporizes.
    Returns (vapor, liquid, molar_vapor_fraction).
    """
    T = feed.T

    # K-values from vapor pressures
    K_A = psat_A(T) / P_flash
    K_B = psat_B(T) / P_flash

    # Feed mole fractions
    zA = feed.zA_mol
    zB = 1.0 - zA

    # Rachford-Rice
    def rr(V):
        res = 0.0
        if zA > 1e-15:
            d = 1.0 + V * (K_A - 1.0)
            if abs(d) > 1e-15:
                res += zA * (K_A - 1.0) / d
        if zB > 1e-15:
            d = 1.0 + V * (K_B - 1.0)
            if abs(d) > 1e-15:
                res += zB * (K_B - 1.0) / d
        return res

    rr0 = rr(0.0)
    rr1 = rr(1.0)

    if rr0 <= 0:
        V_frac = 0.0
    elif rr1 >= 0:
        V_frac = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if rr(mid) > 0: lo = mid
            else: hi = mid
            if hi - lo < 1e-12: break
        V_frac = (lo + hi) / 2.0

    if V_frac < 1e-10:
        return (StreamState(0, T, P_flash, feed.xA, feed.xB),
                StreamState(feed.mdot, T, P_flash, feed.xA, feed.xB), 0.0)

    if V_frac > 1.0 - 1e-10:
        return (StreamState(feed.mdot, T, P_flash, feed.xA, feed.xB),
                StreamState(0, T, P_flash, feed.xA, feed.xB), 1.0)

    # Equilibrium compositions
    yA = zA * K_A / (1.0 + V_frac * (K_A - 1.0))
    yB = zB * K_B / (1.0 + V_frac * (K_B - 1.0))
    yt = yA + yB; yA /= max(yt, 1e-15); yB /= max(yt, 1e-15)

    xAl = zA / (1.0 + V_frac * (K_A - 1.0))
    xBl = zB / (1.0 + V_frac * (K_B - 1.0))
    xt = xAl + xBl; xAl /= max(xt, 1e-15); xBl /= max(xt, 1e-15)

    # Mass fractions
    vap_xA = yA * MW_A / (yA * MW_A + yB * MW_B)
    liq_xA = xAl * MW_A / (xAl * MW_A + xBl * MW_B)

    # Mass split
    MW_v = yA * MW_A + yB * MW_B
    MW_l = xAl * MW_A + xBl * MW_B
    mass_vf = V_frac * MW_v / (V_frac * MW_v + (1 - V_frac) * MW_l)

    mdot_v = feed.mdot * mass_vf
    mdot_l = feed.mdot * (1.0 - mass_vf)

    vapor = StreamState(mdot_v, T, P_flash, vap_xA, 1.0 - vap_xA)
    liquid = StreamState(mdot_l, T, P_flash, liq_xA, 1.0 - liq_xA)
    return vapor, liquid, V_frac


def solve_splitter(feed: StreamState, recycle_frac: float = 0.70):
    """Split vapor: recycle_frac -> recycle, rest -> purge."""
    rc = StreamState(feed.mdot * recycle_frac, feed.T, feed.P, feed.xA, feed.xB)
    pg = StreamState(feed.mdot * (1.0 - recycle_frac), feed.T, feed.P, feed.xA, feed.xB)
    return rc, pg


def solve_plantwide(
    mdot_fresh: float = 10.0,
    T_fresh: float = 300.0,
    P_fresh: float = P_FRESH,
    T_coolant: float = 290.0,
    UA: float = UA_COOLANT_NOM,
    recycle_frac: float = 0.70,
    P_flash: float = P_FLASH_NOM,
    max_iter: int = 400,
    tol: float = 5e-5,
) -> dict:
    """
    Plant-wide solver with tear-stream iteration on the recycle stream.
    Vapor overhead from flash -> splitter -> recycle (condensed) + purge.
    Liquid bottoms from flash -> product (B-rich).
    """
    fresh = StreamState(mdot_fresh, T_fresh, P_fresh, 1.0, 0.0)

    # Tear-stream initial guess
    recycle = StreamState(mdot_fresh * 0.3, 320.0, P_fresh, 0.65, 0.35)

    converged = False
    for it in range(max_iter):
        mixer_out = solve_mixer(fresh, recycle)
        cstr_out = solve_cstr(mixer_out, UA=UA, T_coolant=T_coolant)
        flash_vap, flash_liq, V_frac = solve_flash(cstr_out, P_flash=P_flash)

        # Vapor -> splitter
        recycle_new, purge = solve_splitter(flash_vap, recycle_frac)
        # Condense and pump recycle back to mixer conditions
        recycle_new.T = max(recycle_new.T - 25.0, 310.0)
        recycle_new.P = P_fresh

        d = (abs(recycle_new.mdot - recycle.mdot)
             + abs(recycle_new.T - recycle.T)
             + abs(recycle_new.xA - recycle.xA) * 100)

        if d < tol and it > 10:
            converged = True
            recycle = recycle_new
            mixer_out = solve_mixer(fresh, recycle)
            cstr_out = solve_cstr(mixer_out, UA=UA, T_coolant=T_coolant)
            flash_vap, flash_liq, V_frac = solve_flash(cstr_out, P_flash=P_flash)
            recycle, purge = solve_splitter(flash_vap, recycle_frac)
            recycle.T = max(recycle.T - 25.0, 310.0)
            recycle.P = P_fresh
            break

        a = 0.30
        recycle = StreamState(
            recycle.mdot*(1-a) + recycle_new.mdot*a,
            recycle.T*(1-a) + recycle_new.T*a,
            P_fresh,
            recycle.xA*(1-a) + recycle_new.xA*a,
            recycle.xB*(1-a) + recycle_new.xB*a,
        )

    # Metrics
    Q_cool = UA * (cstr_out.T - T_coolant)
    k_cstr = arrhenius_k(cstr_out.T)
    tau = V_REACTOR / max(mixer_out.mdot / RHO, 1e-12)
    CA_mix = mixer_out.CA
    CA_cstr = CA_mix / (1.0 + k_cstr * tau)
    per_pass_X = (CA_mix - CA_cstr) / max(CA_mix, 1e-12)

    # Overall conversion = 1 - (A leaving system / A entering system)
    A_out = purge.mdot_A + flash_liq.mdot_A
    overall_X = 1.0 - A_out / max(fresh.mdot_A, 1e-12)

    recycle_ratio = recycle.mdot / max(fresh.mdot, 1e-12)
    prod_B = flash_liq.mdot_B * 3600.0
    prod_total = flash_liq.mdot * 3600.0
    profitability = prod_B / max(Q_cool / 1000.0, 1e-9)

    # Flash holdup fraction (liquid fraction by mass)
    flash_liq_frac = flash_liq.mdot / max(cstr_out.mdot, 1e-12)

    return {
        "converged": converged, "iterations": it + 1,
        "fresh_mdot": round(fresh.mdot, 4),
        "mixer_mdot": round(mixer_out.mdot, 4),
        "mixer_T": round(mixer_out.T, 2),
        "mixer_xA": round(mixer_out.xA, 4),
        "cstr_T": round(cstr_out.T, 2),
        "cstr_xA": round(cstr_out.xA, 4),
        "cstr_xB": round(cstr_out.xB, 4),
        "cstr_per_pass_X": round(per_pass_X * 100, 2),
        "cstr_Q_kW": round(Q_cool / 1000.0, 2),
        "cstr_k": round(k_cstr, 6),
        "cstr_tau": round(tau, 2),
        "safety_margin_K": round(T_MAX_SAFE - cstr_out.T, 2),
        "flash_P_Pa": P_flash,
        "flash_V_frac": round(V_frac, 4),
        "flash_vap_mdot": round(flash_vap.mdot, 4),
        "flash_vap_xA": round(flash_vap.xA, 4),
        "flash_liq_mdot": round(flash_liq.mdot, 4),
        "flash_liq_xA": round(flash_liq.xA, 4),
        "flash_liq_xB": round(flash_liq.xB, 4),
        "flash_liq_frac": round(flash_liq_frac, 4),
        "recycle_mdot": round(recycle.mdot, 4),
        "recycle_xA": round(recycle.xA, 4),
        "recycle_ratio": round(recycle_ratio, 4),
        "purge_mdot": round(purge.mdot, 4),
        "product_B_kghr": round(prod_B, 1),
        "product_total_kghr": round(prod_total, 1),
        "overall_X_pct": round(overall_X * 100, 2),
        "energy_kW": round(Q_cool / 1000.0, 2),
        "profitability": round(profitability, 2),
        # Raw stream objects
        "_fresh": fresh, "_mixer": mixer_out, "_cstr": cstr_out,
        "_flash_vap": flash_vap, "_flash_liq": flash_liq,
        "_recycle": recycle, "_purge": purge,
    }


def run_plantwide(**kw):
    return solve_plantwide(**kw)


if __name__ == "__main__":
    print("=" * 72)
    print("  Plant-Wide Topology v3 -- Baseline Steady State")
    print("=" * 72)
    r = run_plantwide()
    for k, v in r.items():
        if not k.startswith("_"):
            print(f"  {k:25s} = {v}")
    print("=" * 72)
