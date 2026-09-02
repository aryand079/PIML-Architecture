# Physics-Informed Multi-Agent Architecture for Plant Control and Optimization

This repository contains a quantitative proof-of-concept for an AI framework that integrates decentralized Multi-Agent systems with Physics-Informed Machine Learning (PIML). By utilizing Physics-Informed Neural Operators (PINO), the system achieves real-time, zero-shot optimization without violating thermodynamic conservation laws.

## Core Concept: The "Physics Firewall"
Traditional deep learning controllers fail in industrial settings because they extrapolate blindly when facing unseen anomalies. This lack of physics inductive bias can cause thermal runaway or cascading failures like flash drum flooding and emergency flaring. 

This architecture solves this by using PINO as a universal physics constraint layer. Before any agent-proposed setpoint change reaches the plant DCS, the PINO predicts the full plant state and checks all mass and energy conservation laws in ~5 ms. It acts as a physics firewall, ensuring only thermodynamically valid actions are executed.

## System Architectures

### 1. Single-Unit Optimization (CSTR)
* Models an exothermic Continuous Stirred Tank Reactor (CSTR) using a DWSIM Python.NET automation script and an analytical fallback engine.
* Demonstrates Bayesian fusion to filter noisy temperature sensor data against physics-predicted temperatures.
* **Result:** Recovers from a severe feed pressure drop and sensor degradation in 5.0 seconds (360x faster than standard AI), successfully preventing an emergency shutdown and retaining 25,090.6 kg/hr of production.

### 2. Plant-Wide 3E Framework (Economics × Energy × Environment)
* Scales the PIML architecture to an interconnected topology: Fresh Feed → Mixer → CSTR → Flash Drum → Splitter → Recycle Loop.
* Agents navigate the Pareto frontier to simultaneously optimize three objectives (3E) during a 40% loss of coolant and sensor noise.
* **Result:** The multi-agent system securely coordinates a feed cut, flash pressure adjustment, and recycle bypass to stabilize the plant in 8 seconds. 
* Compared to standard AI, the PIML framework retained $41,631/hr in profit, saved 303.1 kW of cooling energy, and prevented 4,500 kg/hr of emergency flare emissions.

## Quantitative Impact (Plant-Wide 3E Framework)
During a simulated severe anomaly (40% coolant loss + noisy sensors), the PIML Multi-Agent system was benchmarked against a standard Deep Learning (DL) controller and human manual control. The PIML system successfully decoupled the conflicting multi-objective goals and prevented a cascading plant failure.

| Metric | Baseline | Manual Control | Standard AI | **PIML 3E** |
|--------|----------|---------------|-------------|-------------|
| **System Outcome** | Normal | Slow Stable | CASCADING FAILURE | **Optimal 3E** |
| **Recovery Time** | — | 900 s | 2,400 s | **8.0 s** |
| **Profit Retained** | 100% | 62.1% | < 0% | **90.1%** |
| **Energy Saved (vs AI)** | — | — | 0 kW (ref) | **303.1 kW** |
| **Emergency Flare** | 0 kg/hr | 600 kg/hr | 4,500 kg/hr | **0 kg/hr** |
| **Cascading Failures** | 0 | 0 | 3 | **0** |

**Key Takeaways:**
* **Standard AI** extrapolated blindly from the noisy sensor, starving the flash drum, overloading the compressor, and causing an emergency plant shutdown with massive flaring.
* **Manual Control** stabilized the plant safely but slowly, suffering high economic losses and temporary flaring during the 15-minute transition.
* **PIML Multi-Agent** utilized the PINO engine to map the Pareto frontier in milliseconds, executing a coordinated, physics-compliant recovery in 8 seconds with zero safety violations.
  
## Project Structure
* `dwsim_cstr_plant.py`: DWSIM Python.NET bridge and analytical CSTR engine.
* `piml_multiagent_sim.py`: Full PIML multi-agent feedback loop simulation for the single unit.
* `plantwide_model.py`: Plant-wide topology with tear-stream convergence and 3E metrics.
* `multiobjective_piml_sim.py`: Complete 4-way simulation comparing Baseline, Manual Control, Standard AI, and PIML Multi-Agent recovery.

## Installation & Usage

**1. Install Dependencies**
This project requires Python 3.8+ and the `pythonnet` package to bridge with DWSIM.
`pip install -r requirements.txt`

**2. DWSIM Integration (Optional)**
The architecture is designed to interface with the DWSIM open-source chemical process simulator via the `Automation3` COM interface. If DWSIM is installed, ensure the path is set in your environment variables:
`set DWSIM_PATH="C:\Users\Public\Documents\DWSIM"`

*Note: If DWSIM is not detected, the system will automatically seamlessly fall back to an identical analytical Python engine to execute the mass and energy balances.*

**3. Run the Simulations**
To execute the single-unit CSTR anomaly response:
`python piml_multiagent_sim.py`

To execute the plant-wide 3E multi-objective simulation:
`python multiobjective_piml_sim.py`
