"""
Plant-Wide Topology: Fresh Feed -> Mixer -> CSTR -> Flash Separator -> Recycle
================================================================================
Multi-Objective Architecture (3E: Economics, Energy, Environment)

Topology:
  Fresh Feed (pure A, 300 K, 5 bar)
       |
       v
   [MIXER] ------> [CSTR (exothermic)] ------> [FLASH DRUM (0.65 bar)]
       ^            A -> B, V = 1 m^3              |           |
       |            Cooling jacket UA            VAPOR       LIQUID
       |                                       (A-rich)     (B-rich)
       |                                          |            |
       +---<-- [RECYCLE] <-- [SPLITTER] <---------+        PRODUCT
              (condensed)          |
                                PURGE --> recovery / flare

Reaction:   A -> B   (irreversible, 1st order, exothermic)
  A_0 = 1.0e6 1/s,  Ea = 49,456 J/mol,  dH_rxn = -25,000 J/mol

Surrogate compounds:
  A = n-Hexane (MW=86.18, BP=341.9 K @ 1 atm)
  B = Toluene  (MW=92.14, BP=383.8 K @ 1 atm)

DWSIM Python.NET bridge included; analytical engine used as fallback.
"""

import os, sys, math
from dataclasses import dataclass
from typing import Tuple, Dict

# ============================================================================
#  PHYSICAL & THERMODYNAMIC CONSTANTS
# ============================================================================
R_GAS       = 8.314        # J/(mol*K)
A_PRE       = 1.0e6        # 1/s        pre-exponential
EA          = 49456.0       # J/mol      activation energy
DH_RXN      = -25000.0     # J/mol      exothermic
CP          = 3500.0        # J/(kg*K)   liquid heat capacity
RHO         = 850.0         # kg/m^3     liquid density
MW_A        = 86.18         # g/mol      n-Hexane
MW_B        = 92.14         # g/mol      Toluene
V_REACTOR   = 1.0           # m^3
UA_NOM      = 5000.0        # W/K        baseline coolant UA
T_MAX_SAFE  = 400.0         # K          runaway threshold
P_FRESH     = 506625.0      # Pa         5.0 bar

# Flash drum
P_FLASH_NOM = 65000.0       # Pa  (0.65 bar vacuum flash)
P_FLASH_MIN = 45000.0       # Pa  (minimum safe flash pressure)
FLASH_MIN_LIQ_FRAC = 0.50   # minimum liquid holdup fraction

# Clausius-Clapeyron VLE parameters
DHVAP_A = 28850.0            # J/mol  (n-Hexane heat of vaporization)
DHVAP_B = 33180.0            # J/mol  (Toluene heat of vaporization)
TBP_A   = 341.9              # K      (n-Hexane boiling point @ 1 atm)
TBP_B   = 383.8              # K      (Toluene boiling point @ 1 atm)
P_REF   = 101325.0           # Pa     (1 atm reference)

# ============================================================================
#  ECONOMIC PARAMETERS (Multi-Objective 3E Framework)
# ============================================================================
PRICE_B       = 2.50         # $/kg   product B selling price
COST_A        = 0.80         # $/kg   feed A purchase cost
COST_COOLING  = 0.08         # $/kWh  cooling utility cost
COST_PURGE    = 0.50         # $/kg   routine purge treatment cost
FLARE_PENALTY = 10.00        # $/kg   emergency flare penalty


def arrhenius_k(T: float) -> float:
    """First-order rate constant k(T) via Arrhenius equation."""
    return A_PRE * math.exp(-EA / (R_GAS * T))


def psat_A(T: float) -> float:
    """Vapor pressure of A (n-Hexane) via Clausius-Clapeyron."""
    return P_REF * math.exp(DHVAP_A / R_GAS * (1.0/TBP_A - 1.0/T))


def psat_B(T: float) -> float:
    """Vapor pressure of B (Toluene) via Clausius-Clapeyron."""
    return P_REF * math.exp(DHVAP_B / R_GAS * (1.0/TBP_B - 1.0/T))


# ============================================================================
#  DWSIM PYTHON.NET BRIDGE
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


def build_dwsim_plantwide(**kwargs):
    """
    Build the full plant-wide DWSIM flowsheet:
      Fresh Feed -> Mixer -> CSTR (kinetic A->B) -> Flash Drum -> Splitter
    Using Peng-Robinson EOS with N-Hexane/Toluene.
    Recycle tear stream resolved by DWSIM's built-in convergence block.
    """
    if not DWSIM_AVAILABLE:
        raise RuntimeError("DWSIM not loaded.")

    from DWSIM.Automation import Automation3
    from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType
    from DWSIM.GlobalSettings import Settings
    from System import Array, Double

    Settings.AutomationMode = True
    mgr = Automation3()
    sim = mgr.CreateFlowsheet()

    sim.CreateAndAddPropertyPackage("Peng-Robinson (PR)")
    sim.AddCompound("N-Hexane")
    sim.AddCompound("Toluene")

    mf = kwargs.get("mdot_fresh", 10.0)
    # -- streams --
    ff = sim.AddObject(ObjectType.MaterialStream, 50, 100, "FreshFeed")
    ff.GetAsObject().SetTemperature(300.0)
    ff.GetAsObject().SetPressure(P_FRESH)
    ff.GetAsObject().SetMassFlow(mf)
    ff.GetAsObject().SetOverallComposition(Array[Double]([1.0, 0.0]))
    ff.GetAsObject().SetFlashSpec("PT")

    mx_o = sim.AddObject(ObjectType.MaterialStream, 200, 100, "MixerOut")
    cr_o = sim.AddObject(ObjectType.MaterialStream, 350, 100, "CSTROut")
    fv   = sim.AddObject(ObjectType.MaterialStream, 500,  50, "FlashVap")
    fl   = sim.AddObject(ObjectType.MaterialStream, 500, 150, "FlashLiq")
    rc   = sim.AddObject(ObjectType.MaterialStream, 200, 200, "Recycle")
    pg   = sim.AddObject(ObjectType.MaterialStream, 600, 200, "Purge")
    qc   = sim.AddObject(ObjectType.EnergyStream,   350, 200, "QCool")

    mixer = sim.AddObject(ObjectType.NodeIn, 150, 100, "MIX-01")
    cstr  = sim.AddObject(ObjectType.Reactor_CSTR, 300, 100, "CSTR-01")
    flash = sim.AddObject(ObjectType.FlashDrum, 450, 100, "FLASH-01")
    split = sim.AddObject(ObjectType.NodeOut, 550, 50, "SPLIT-01")

    cstr.GetAsObject().Volume = V_REACTOR

    sim.ConnectObjects(ff.GraphicObject, mixer.GraphicObject, 0, 0)
    sim.ConnectObjects(rc.GraphicObject, mixer.GraphicObject, 0, 1)
    sim.ConnectObjects(mixer.GraphicObject, mx_o.GraphicObject, 0, 0)
    sim.ConnectObjects(mx_o.GraphicObject, cstr.GraphicObject, 0, 0)
    sim.ConnectObjects(cstr.GraphicObject, cr_o.GraphicObject, 0, 0)
    sim.ConnectObjects(qc.GraphicObject, cstr.GraphicObject, 0, 0)
    sim.ConnectObjects(cr_o.GraphicObject, flash.GraphicObject, 0, 0)
    sim.ConnectObjects(flash.GraphicObject, fv.GraphicObject, 0, 0)
    sim.ConnectObjects(flash.GraphicObject, fl.GraphicObject, 1, 0)
    sim.ConnectObjects(fv.GraphicObject, split.GraphicObject, 0, 0)
    sim.ConnectObjects(split.GraphicObject, rc.GraphicObject, 0, 0)
    sim.ConnectObjects(split.GraphicObject, pg.GraphicObject, 1, 0)

    mgr.CalculateFlowsheet2(sim)
    return {"dwsim_solved": True}


# ============================================================================
#  ANALYTICAL ENGINE -- STREAM STATE
# ============================================================================

@dataclass
class Stream:
    """Thermodynamic state of a process stream."""
    mdot: float = 0.0     # kg/s
    T: float    = 300.0    # K
    P: float    = 506625.0 # Pa
    xA: float   = 1.0      # mass fraction A
    xB: float   = 0.0      # mass fraction B

    @property
    def CA(self):
        """Molar concentration of A (mol/m^3 of liquid)."""
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
        return Stream(self.mdot, self.T, self.P, self.xA, self.xB)


# ============================================================================
#  UNIT OPERATION SOLVERS
# ============================================================================

def solve_mixer(fresh: Stream, recycle: Stream) -> Stream:
    """Adiabatic, mass-weighted mixer. T_out = mass-average temperature."""
    m = fresh.mdot + recycle.mdot
    if m < 1e-9:
        return Stream(0, fresh.T, fresh.P, fresh.xA, fresh.xB)
    xA = (fresh.mdot_A + recycle.mdot_A) / m
    T  = (fresh.mdot * fresh.T + recycle.mdot * recycle.T) / m
    return Stream(m, T, min(fresh.P, recycle.P), xA, 1.0 - xA)


def solve_cstr(feed: Stream, UA: float = UA_NOM,
               T_coolant: float = 290.0) -> Stream:
    """
    Steady-state CSTR: coupled mass + energy balance.

    Mass (species A):  0 = (F/V)(CA_in - CA) - k(T)*CA
    Energy:  0 = (F/V)(T_in - T) + (-dH)*k*CA/(rho*Cp) - UA/(rho*Cp*V)*(T-Tc)

    Solved by damped fixed-point iteration.
    """
    F_vol = feed.mdot / RHO
    tau = V_REACTOR / max(F_vol, 1e-12)
    CA_in = feed.CA
    CB_in = feed.CB

    T = feed.T
    for _ in range(1000):
        k = arrhenius_k(T)
        CA_out = CA_in / (1.0 + k * tau)
        heat = (-DH_RXN) * k * CA_out / (RHO * CP)
        T_new = (
            (F_vol / V_REACTOR) * feed.T
            + heat
            + (UA / (RHO * CP * V_REACTOR)) * T_coolant
        ) / (
            (F_vol / V_REACTOR)
            + (UA / (RHO * CP * V_REACTOR))
        )
        if abs(T_new - T) < 1e-7:
            break
        T = 0.3 * T_new + 0.7 * T   # damping factor

    k_f = arrhenius_k(T)
    CA_out = CA_in / (1.0 + k_f * tau)
    CB_out = CB_in + (CA_in - CA_out)    # stoichiometric 1:1 molar

    mass_A = CA_out * MW_A / 1000.0      # kg per m^3 liquid
    mass_B = CB_out * MW_B / 1000.0
    total = mass_A + mass_B
    xA = mass_A / max(total, 1e-15)
    return Stream(feed.mdot, T, feed.P, xA, 1.0 - xA)


def solve_flash(feed: Stream,
                P_flash: float = P_FLASH_NOM) -> Tuple[Stream, Stream, float]:
    """
    Isothermal flash using Clausius-Clapeyron + Rachford-Rice.

    Component A (n-Hexane, more volatile) preferentially vaporizes.
    Returns (vapor_stream, liquid_stream, molar_vapor_fraction).
    """
    T = feed.T
    KA = psat_A(T) / P_flash
    KB = psat_B(T) / P_flash

    zA = feed.zA_mol
    zB = 1.0 - zA

    def rr(V):
        """Rachford-Rice objective function."""
        res = 0.0
        if zA > 1e-15:
            d = 1.0 + V * (KA - 1.0)
            if abs(d) > 1e-15: res += zA * (KA - 1.0) / d
        if zB > 1e-15:
            d = 1.0 + V * (KB - 1.0)
            if abs(d) > 1e-15: res += zB * (KB - 1.0) / d
        return res

    rr0, rr1 = rr(0.0), rr(1.0)

    if rr0 <= 0:
        V_frac = 0.0   # subcooled liquid
    elif rr1 >= 0:
        V_frac = 1.0   # superheated vapor
    else:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if rr(mid) > 0: lo = mid
            else: hi = mid
            if hi - lo < 1e-12: break
        V_frac = (lo + hi) / 2.0

    # Edge cases
    if V_frac < 1e-10:
        return (Stream(0, T, P_flash, feed.xA, feed.xB),
                Stream(feed.mdot, T, P_flash, feed.xA, feed.xB), 0.0)
    if V_frac > 1.0 - 1e-10:
        return (Stream(feed.mdot, T, P_flash, feed.xA, feed.xB),
                Stream(0, T, P_flash, feed.xA, feed.xB), 1.0)

    # Equilibrium compositions (mole fractions)
    yA = zA * KA / (1.0 + V_frac * (KA - 1.0))
    yB = zB * KB / (1.0 + V_frac * (KB - 1.0))
    yt = yA + yB; yA /= max(yt, 1e-15); yB /= max(yt, 1e-15)

    xAl = zA / (1.0 + V_frac * (KA - 1.0))
    xBl = zB / (1.0 + V_frac * (KB - 1.0))
    xt = xAl + xBl; xAl /= max(xt, 1e-15); xBl /= max(xt, 1e-15)

    # Mole fracs -> mass fracs
    vap_xA = yA * MW_A / (yA * MW_A + yB * MW_B)
    liq_xA = xAl * MW_A / (xAl * MW_A + xBl * MW_B)

    # Mass vapor fraction
    MW_v = yA * MW_A + yB * MW_B
    MW_l = xAl * MW_A + xBl * MW_B
    mass_vf = V_frac * MW_v / (V_frac * MW_v + (1 - V_frac) * MW_l)

    mdot_v = feed.mdot * mass_vf
    mdot_l = feed.mdot * (1.0 - mass_vf)

    vap = Stream(mdot_v, T, P_flash, vap_xA, 1.0 - vap_xA)
    liq = Stream(mdot_l, T, P_flash, liq_xA, 1.0 - liq_xA)
    return vap, liq, V_frac


def solve_splitter(feed: Stream,
                   recycle_frac: float = 0.70) -> Tuple[Stream, Stream]:
    """Split vapor overhead: recycle_frac -> recycle, rest -> purge."""
    rc = Stream(feed.mdot * recycle_frac, feed.T, feed.P, feed.xA, feed.xB)
    pg = Stream(feed.mdot * (1.0 - recycle_frac), feed.T, feed.P, feed.xA, feed.xB)
    return rc, pg


# ============================================================================
#  PLANT-WIDE SOLVER (Tear-Stream Iteration)
# ============================================================================

def solve_plantwide(
    mdot_fresh: float  = 10.0,
    T_fresh: float     = 300.0,
    P_fresh: float     = P_FRESH,
    T_coolant: float   = 290.0,
    UA: float          = UA_NOM,
    recycle_frac: float = 0.70,
    P_flash: float     = P_FLASH_NOM,
    max_iter: int      = 400,
    tol: float         = 5e-5,
    flare_extra: float = 0.0,      # kg/s of extra emergency flare
) -> dict:
    """
    Solve the plant-wide topology with tear-stream convergence.

    Tear stream = recycle (damped direct substitution, alpha=0.30).
    Vapor overhead -> splitter -> recycle (condensed) + purge.
    Liquid bottoms -> product (B-rich).
    """
    fresh = Stream(mdot_fresh, T_fresh, P_fresh, 1.0, 0.0)

    # Initial tear-stream guess
    recycle = Stream(mdot_fresh * 0.3, 320.0, P_fresh, 0.65, 0.35)

    converged = False
    for it in range(max_iter):
        mixer_out = solve_mixer(fresh, recycle)
        cstr_out  = solve_cstr(mixer_out, UA=UA, T_coolant=T_coolant)
        flash_vap, flash_liq, V_frac = solve_flash(cstr_out, P_flash=P_flash)

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
            # Final consistent pass
            mixer_out = solve_mixer(fresh, recycle)
            cstr_out  = solve_cstr(mixer_out, UA=UA, T_coolant=T_coolant)
            flash_vap, flash_liq, V_frac = solve_flash(cstr_out, P_flash=P_flash)
            recycle, purge = solve_splitter(flash_vap, recycle_frac)
            recycle.T = max(recycle.T - 25.0, 310.0)
            recycle.P = P_fresh
            break

        a = 0.30
        recycle = Stream(
            recycle.mdot*(1-a) + recycle_new.mdot*a,
            recycle.T*(1-a)    + recycle_new.T*a,
            P_fresh,
            recycle.xA*(1-a)   + recycle_new.xA*a,
            recycle.xB*(1-a)   + recycle_new.xB*a,
        )

    # ---- Derived metrics ----
    Q_cool = UA * (cstr_out.T - T_coolant)
    k_cstr = arrhenius_k(cstr_out.T)
    tau = V_REACTOR / max(mixer_out.mdot / RHO, 1e-12)
    CA_mix = mixer_out.CA
    CA_cstr = CA_mix / (1.0 + k_cstr * tau)
    per_pass_X = (CA_mix - CA_cstr) / max(CA_mix, 1e-12)
    A_out = purge.mdot_A + flash_liq.mdot_A
    overall_X = 1.0 - A_out / max(fresh.mdot_A, 1e-12)
    recycle_ratio = recycle.mdot / max(fresh.mdot, 1e-12)
    flash_liq_frac = flash_liq.mdot / max(cstr_out.mdot, 1e-12)

    # Product & purge rates in kg/hr
    prod_B_kghr = flash_liq.mdot_B * 3600.0
    prod_total_kghr = flash_liq.mdot * 3600.0
    purge_kghr = purge.mdot * 3600.0

    # ---- 3E METRICS ----
    revenue = prod_B_kghr * PRICE_B                        # $/hr
    feed_cost = fresh.mdot * 3600.0 * COST_A               # $/hr
    cool_cost = (Q_cool / 1000.0) * COST_COOLING            # $/hr  (kW -> $/hr)
    purge_cost = purge_kghr * COST_PURGE                    # $/hr
    flare_kghr = flare_extra * 3600.0                       # $/hr
    flare_cost = flare_kghr * FLARE_PENALTY                 # $/hr
    profit = revenue - feed_cost - cool_cost - purge_cost - flare_cost

    sec = (Q_cool / 1000.0) / max(prod_total_kghr, 1e-12)  # kW / (kg/hr) = kWh/kg
    sec_kw_per_kgs = Q_cool / max(flash_liq.mdot, 1e-12)   # kW / (kg/s)

    return {
        "converged": converged, "iterations": it + 1,
        # Process
        "fresh_mdot": round(fresh.mdot, 4),
        "mixer_mdot": round(mixer_out.mdot, 4),
        "mixer_T":    round(mixer_out.T, 2),
        "mixer_xA":   round(mixer_out.xA, 4),
        "cstr_T":     round(cstr_out.T, 2),
        "cstr_xA":    round(cstr_out.xA, 4),
        "cstr_xB":    round(cstr_out.xB, 4),
        "per_pass_X": round(per_pass_X * 100, 2),
        "Q_cool_kW":  round(Q_cool / 1000.0, 2),
        "k_rate":     round(k_cstr, 6),
        "tau_s":      round(tau, 2),
        "safety_margin_K": round(T_MAX_SAFE - cstr_out.T, 2),
        "flash_P":    P_flash,
        "flash_V_frac": round(V_frac, 4),
        "flash_vap_mdot": round(flash_vap.mdot, 4),
        "flash_vap_xA":   round(flash_vap.xA, 4),
        "flash_liq_mdot": round(flash_liq.mdot, 4),
        "flash_liq_xA":   round(flash_liq.xA, 4),
        "flash_liq_xB":   round(flash_liq.xB, 4),
        "flash_liq_frac": round(flash_liq_frac, 4),
        "recycle_mdot":   round(recycle.mdot, 4),
        "recycle_xA":     round(recycle.xA, 4),
        "recycle_ratio":  round(recycle_ratio, 4),
        "purge_mdot":     round(purge.mdot, 4),
        "overall_X":      round(overall_X * 100, 2),
        # Product
        "product_B_kghr":   round(prod_B_kghr, 1),
        "product_tot_kghr": round(prod_total_kghr, 1),
        "purge_kghr":       round(purge_kghr, 1),
        # 3E Economics
        "revenue_hr":   round(revenue, 2),
        "feed_cost_hr": round(feed_cost, 2),
        "cool_cost_hr": round(cool_cost, 2),
        "purge_cost_hr": round(purge_cost, 2),
        "flare_kghr":   round(flare_kghr, 1),
        "flare_cost_hr": round(flare_cost, 2),
        "profit_hr":    round(profit, 2),
        # 3E Energy
        "SEC_kWh_per_kg": round(sec, 6),
        "SEC_kW_per_kgs": round(sec_kw_per_kgs, 2),
        # 3E Environment
        "waste_kghr":   round(purge_kghr + flare_kghr, 1),
        "flare_only_kghr": round(flare_kghr, 1),
        # Raw streams
        "_fresh": fresh, "_mixer": mixer_out, "_cstr": cstr_out,
        "_vap": flash_vap, "_liq": flash_liq,
        "_recycle": recycle, "_purge": purge,
    }


def run_plantwide(**kw):
    """Plant-wide solver: DWSIM if available, else analytical."""
    if DWSIM_AVAILABLE:
        try:
            return build_dwsim_plantwide(**kw)
        except Exception as e:
            print(f"[DWSIM] Failed ({e}); using analytical engine.")
    return solve_plantwide(**kw)


# ============================================================================
#  SELF-TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("  Plant-Wide Topology -- Multi-Objective Baseline")
    print("=" * 72)
    r = run_plantwide()
    for k, v in r.items():
        if not k.startswith("_"):
            print(f"  {k:25s} = {v}")
    print("=" * 72)
