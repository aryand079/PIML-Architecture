"""
DWSIM Plant Model: Exothermic CSTR (A  B)
=============================================
Physics-Informed Multi-Agent Architecture -- Plant Simulation Bridge

Reaction:  A -> B   (irreversible, first-order, exothermic)
  DeltaH_rxn = -25,000  J/mol
  A (pre-exponential) = 1.0 x 10^6  1/s
  Ea (activation energy) = 49,456  J/mol

Operating envelope:
  Feed temperature:   300 K
  Feed pressure:      5.0 bar  (506,625 Pa)
  Feed mass flow:     10.0 kg/s
  Reactor volume:     1.0 m3
  Coolant UA:         5,000  W/K
  Max safe temp:      400 K  (runaway threshold)
  Target conversion:  >= 85 %

This script can be run against a live DWSIM installation.
When DWSIM is unavailable, the companion `piml_multiagent_sim.py`
uses the same thermodynamic model as an analytical fallback.
"""

import os
import sys
import math

# ---------------------------------------------------------------------------
# 1.  DWSIM Python.NET Automation Interface
# ---------------------------------------------------------------------------
# The block below requires:
#    DWSIM >= 8.0 installed at DWSIM_PATH
#    pythonnet  (pip install pythonnet)
# If DWSIM is not present the script falls through to the analytical engine.

DWSIM_PATH = os.environ.get(
    "DWSIM_PATH",
    r"C:\Users\Public\Documents\DWSIM"
)

DWSIM_AVAILABLE = False

def _init_dwsim():
    """Attempt to load DWSIM assemblies via pythonnet CLR."""
    global DWSIM_AVAILABLE
    try:
        import clr  # pythonnet

        assemblies = [
            "CapeOpen.dll",
            "DWSIM.Automation.dll",
            "DWSIM.Interfaces.dll",
            "DWSIM.SharedClasses.dll",
            "DWSIM.Thermodynamics.dll",
            "DWSIM.UnitOperations.dll",
            "DWSIM.GlobalSettings.dll",
        ]
        for asm in assemblies:
            clr.AddReference(os.path.join(DWSIM_PATH, asm))

        DWSIM_AVAILABLE = True
        print("[DWSIM] Assemblies loaded successfully.")
    except Exception as e:
        print(f"[DWSIM] Not available ({e}). Using analytical engine.")
        DWSIM_AVAILABLE = False

_init_dwsim()


# ---------------------------------------------------------------------------
# 2.  DWSIM Flowsheet Builder
# ---------------------------------------------------------------------------
def build_dwsim_flowsheet(
    T_feed: float = 300.0,      # K
    P_feed: float = 506625.0,   # Pa  (5.0 bar)
    mdot_feed: float = 10.0,    # kg/s
    T_coolant: float = 290.0,   # K
):
    """
    Construct and solve a DWSIM CSTR flowsheet.

    Returns a dict of key simulation results:
      T_out, P_out, mdot_out, conversion, Q_cooling
    """
    if not DWSIM_AVAILABLE:
        raise RuntimeError("DWSIM assemblies are not loaded.")

    from DWSIM.Automation import Automation3 as Automation
    from DWSIM.Interfaces.Enums import ObjectType
    from System import Array, Double  # .NET interop types

    manager = Automation()
    flowsheet = manager.CreateFlowsheet()

    # --- Thermodynamic property package (Peng-Robinson) ---
    pp = flowsheet.CreateAndAddPropertyPackage("Peng-Robinson (PR)")

    # --- Compounds: pseudo-components A and B (modelled as n-Hexane / Toluene) ---
    flowsheet.AddCompound("N-Hexane")   # surrogate for reactant A
    flowsheet.AddCompound("Toluene")    # surrogate for product B

    # --- Feed material stream ---
    feed = flowsheet.AddObject(ObjectType.MaterialStream, 50, 100, "Feed")
    feed_obj = feed.GetAsObject()
    feed_obj.SetTemperature(T_feed)
    feed_obj.SetPressure(P_feed)
    feed_obj.SetMassFlow(mdot_feed)
    feed_obj.SetOverallComposition(Array[Double]([1.0, 0.0]))  # pure A
    feed_obj.SetFlashSpec("Temperature and Pressure")

    # --- Product material stream ---
    product = flowsheet.AddObject(ObjectType.MaterialStream, 250, 100, "Product")

    # --- Cooling energy stream ---
    q_cool = flowsheet.AddObject(ObjectType.EnergyStream, 150, 200, "Q_Cooling")

    # --- CSTR unit operation ---
    cstr = flowsheet.AddObject(ObjectType.RCT_CSTR, 150, 100, "CSTR-01")
    cstr_obj = cstr.GetAsObject()

    # Reactor volume = 1.0 m3
    cstr_obj.Volume = 1.0

    # --- Kinetic reaction set  (A  B, first order in A) ---
    rxn_set = flowsheet.CreateAndAddReactionSet("CSTR_RxnSet")
    rxn = flowsheet.CreateAndAddReaction(
        "A_to_B", "N-Hexane", "Toluene",
        reactionType=0,              # Kinetic
        reactionBasis=0,             # Molar concentration
        reactionPhase=0,             # Liquid
        forwardPreExpFactor=1.0e6,   # 1/s
        forwardActivationEnergy=49456.0,  # J/mol
        forwardReactionOrder=1.0,
        enthalpyOfReaction=-25000.0,  # J/mol (exothermic, moderate)
    )
    rxn_set.AddReaction(rxn)
    cstr_obj.ReactionSetID = rxn_set.ID

    # --- Connect topology ---
    flowsheet.ConnectObjects(feed, cstr, -1, -1)
    flowsheet.ConnectObjects(cstr, product, -1, -1)
    flowsheet.ConnectObjects(q_cool, cstr, -1, -1)

    # --- Solve ---
    flowsheet.RequestCalculation()

    # --- Extract results ---
    prod_obj = flowsheet.GetFlowsheetSimulationObject("Product")
    results = {
        "T_out_K": prod_obj.GetTemperature(),
        "P_out_Pa": prod_obj.GetPressure(),
        "mdot_out_kgs": prod_obj.GetMassFlow(),
        "conversion_pct": (
            1.0 - prod_obj.GetOverallComposition()[0]
        ) * 100.0,
        "Q_cooling_W": abs(
            flowsheet.GetFlowsheetSimulationObject("Q_Cooling")
                     .GetEnergyFlow()
        ),
    }
    return results


# ---------------------------------------------------------------------------
# 3.  Analytical CSTR Engine  (standalone, mirrors DWSIM thermodynamics)
# ---------------------------------------------------------------------------
# Physical constants & reaction parameters
R_GAS       = 8.314          # J/(molK)
A_PRE       = 1.0e6          # 1/s          (pre-exponential factor)
EA          = 49456.0        # J/mol        (activation energy)
DH_RXN      = -25000.0       # J/mol        (exothermic, moderate)
CP          = 3500.0         # J/(kgK)     (liquid heat capacity)
RHO         = 850.0          # kg/m3        (liquid density)
MW_A        = 86.18          # g/mol        (n-Hexane molecular weight)
V_REACTOR   = 1.0            # m3
UA_COOLANT  = 5000.0         # W/K          (coolant heat-transfer coeff x area)
T_MAX_SAFE  = 400.0          # K            (runaway threshold)


def arrhenius_k(T: float) -> float:
    """First-order rate constant k(T) via Arrhenius equation."""
    return A_PRE * math.exp(-EA / (R_GAS * T))


def cstr_steady_state(
    T_feed: float   = 300.0,
    P_feed: float   = 506625.0,
    mdot: float     = 10.0,
    CA_in: float    = None,
    T_coolant: float = 290.0,
):
    """
    Solve the coupled mass + energy balance for a first-order
    exothermic CSTR at steady state.

    Mass balance (molar):
        0 = F/V  (CA_in  CA)  k(T)CA

    Energy balance:
        0 = F/V  (T_feed  T)
            + (DeltaH_rxn)k(T)CA / (Cp)
             UA/(CpV)  (T  T_coolant)

    Solved iteratively (simple successive substitution).

    Returns dict with T_out, P_out, CA_out, conversion, Q_cooling, k_rate.
    """
    # Volumetric flow
    F_vol = mdot / RHO                          # m3/s
    tau   = V_REACTOR / F_vol                    # residence time (s)

    # Inlet concentration (mol/m3)
    if CA_in is None:
        CA_in = RHO * 1000.0 / MW_A             # ~= 9,864 mol/m3

    # Iterative solution (damped fixed-point)
    T = T_feed  # initial guess
    for _ in range(500):
        k = arrhenius_k(T)
        CA = CA_in / (1.0 + k * tau)             # mass balance
        # energy balance rearranged for T
        heat_rxn = (-DH_RXN) * k * CA / (RHO * CP)
        T_new = (
            (F_vol / V_REACTOR) * T_feed
            + heat_rxn
            + (UA_COOLANT / (RHO * CP * V_REACTOR)) * T_coolant
        ) / (
            (F_vol / V_REACTOR)
            + (UA_COOLANT / (RHO * CP * V_REACTOR))
        )
        if abs(T_new - T) < 1e-6:
            break
        T = 0.3 * T_new + 0.7 * T  # damping for stability

    conversion = (CA_in - CA) / CA_in
    Q_cool = UA_COOLANT * (T - T_coolant)

    return {
        "T_out_K":        round(T, 2),
        "P_out_Pa":       P_feed,             # negligible drop in CSTR
        "mdot_out_kgs":   mdot,
        "CA_out_molm3":   round(CA, 2),
        "conversion_pct": round(conversion * 100, 2),
        "Q_cooling_W":    round(Q_cool, 2),
        "k_rate_1s":      round(arrhenius_k(T), 6),
        "tau_s":          round(tau, 4),
    }


# ---------------------------------------------------------------------------
# 4.  Convenience wrapper -- prefer DWSIM, fallback to analytical
# ---------------------------------------------------------------------------
def run_plant(T_feed=300.0, P_feed=506625.0, mdot=10.0, T_coolant=290.0):
    """Run the CSTR simulation through DWSIM or the analytical fallback."""
    if DWSIM_AVAILABLE:
        try:
            return build_dwsim_flowsheet(T_feed, P_feed, mdot, T_coolant)
        except Exception as e:
            print(f"[DWSIM] Solver failed ({e}); falling back to analytical.")

    return cstr_steady_state(T_feed, P_feed, mdot, T_coolant=T_coolant)


# ---------------------------------------------------------------------------
# 5.  Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("  DWSIM / Analytical CSTR Plant Model  --  Baseline Run")
    print("=" * 65)
    baseline = run_plant()
    for k, v in baseline.items():
        print(f"  {k:20s} = {v}")
    print("=" * 65)
