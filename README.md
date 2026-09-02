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

## Project Structure
* `dwsim_cstr_plant.py`: DWSIM Python.NET bridge and analytical CSTR engine.
* `piml_multiagent_sim.py`: Full PIML multi-agent feedback loop simulation for the single unit.
* `plantwide_model.py`: Plant-wide topology with tear-stream convergence and 3E metrics.
* `multiobjective_piml_sim.py`: Complete 4-way simulation comparing Baseline, Manual Control, Standard AI, and PIML Multi-Agent recovery.
