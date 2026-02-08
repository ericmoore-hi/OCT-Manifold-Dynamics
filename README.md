# Deterministic Solution of NP-complete Problems via Inertial Manifold Dynamics within the BSS Model Framework

**Author:** Eric Moore  
**Contact:** eric@oct-compute.com

---

## Overview

This repository contains the reference implementation, simulation scripts, and validation data for the **Ontological Computing Theory (OCT)** applied to NP-complete problems (specifically 3-SAT). 

This project demonstrates a novel **physics-based computational paradigm** operating within the theoretical framework of the **Blum-Shub-Smale (BSS)** real computation model. The code simulates continuous inertial dynamics to overcome energy barriers in non-convex landscapes, providing numerical evidence for polynomial-time convergence in continuous-time analog systems.

### ⚠️ Important Note on Performance and Simulation Methodology

This codebase is a **numerical simulation** of an analog continuous-time system running on a discrete digital computer. It involves heavy numerical integration (solving ODEs via Runge-Kutta/Euler methods), which is inherently computationally expensive on digital hardware. 

* **Purpose:** The code serves to validate the **scaling laws** (complexity class) and dynamical behavior of the proposed analog architecture.
* **Comparison:** It is **NOT** intended to compete with highly optimized digital SAT solvers (like CDCL) in terms of "wall-clock time" on current CPU/GPU hardware. The theoretical speed advantage of OCT is realized only in physical analog implementation (e.g., photonics or continuous-time CMOS circuits), not in digital emulation.

---

## Repository Structure & File Descriptions

The repository includes 15 Python scripts used to reproduce the figures, tables, and analytical results presented in the manuscript.

### 1. Core Solvers & Dynamics Engine
* **`oct_core_solver.py`** (Generates Figures 5 & 6)  
    The fundamental continuous dynamics engine implementing the core OCT equations via explicit Euler integration. It parses standard DIMACS `.cnf` files and demonstrates the system's ability to converge to a global minimum for SAT instances (Figure 5) or exhibit characteristic limit cycles for UNSAT instances (Figure 6).
* **`main_solver_N1000.py`** (Generates Figure 7)  
    A large-scale simulation script designed for high-dimensional instances (e.g., $N=1000$ variables at the critical phase transition ratio $\alpha \approx 4.25$). It utilizes an adaptive Runge-Kutta (RK45) integrator to simulate the Inertial Manifold Dynamics, demonstrating the system's capability to navigate complex energy landscapes and find global minima in polynomial physical time.
* **`phase_dynamics_solver.py`** (Generates Figure 3)  
    Performs numerical integration of representative dynamical equations to visualize the fundamental spectral distinction between SAT regimes (converging to a fixed point/DC) and frustrated UNSAT regimes (exhibiting persistent limit cycles/AC) in the presence of noise.

### 2. Scaling & Complexity Analysis
* **`inertial_sat_scaling.py`** (Generates Figure 9)  
    Performs a compact empirical benchmark on random 3-SAT instances in the range $N \in [10, 20]$. It conducts multiple trials per data point to determine the median convergence time and fits the initial polynomial scaling trend ($T \propto N^k$), providing experimental validation of the efficient scaling dynamics at small scales.
* **`analytical_complexity_model.py`** (Generates Figure 10)  
    Visualizes the theoretical performance envelope of the OCT solver, plotting the median ($O(N^{1.6})$) and worst-case ($O(N^{2.1})$) polynomial scaling bounds. It highlights the divergence from the classical exponential complexity barrier ($O(2^N)$).
* **`asymptotic_scaling_model.py`** (Generates Figure 8)  
    Performs a theoretical extrapolation of the empirically observed polynomial complexity law ($T \propto N^{1.6}$) to extreme scales ($N=100,000$). It utilizes the $N=1000$ benchmark result as a calibration anchor to demonstrate the divergence between the proposed OCT architecture and the classical exponential barrier.
* **`integration_timescale_scaling.py`** (Generates Figure 14)  
    Evaluates the scaling behavior of the critical integration time-step ($dt$) required for numerical stability. It compares the observed polynomial scaling law ($dt \propto N^{-0.5}$) against a hypothetical exponential decay barrier, demonstrating that the system avoids the "Precision Wall" in numerical integration.

### 3. Physics & Noise Analysis
* **`Analytical_Frequency_Model.py`** (Generates Figure 4 & Table 1)  
    Computes the theoretical bandwidth scaling of the physical architecture based on coupled oscillator dynamics ($\omega \propto N^{-1/2}$). Includes a Monte Carlo analysis to verify signal stability and distinguishability under simulated thermal noise conditions.
* **`oct_noise_stability_analysis.py`** (Generates Figure 11)  
    Models the theoretical stability threshold of OCT topological invariants against thermal noise power as a function of system size ($N$). It visualizes the phase boundary between the deterministic phase-locked region and the noise-induced chaotic region.
* **`oscillator_stochastic_dynamics.py`** (Generates Figure 12)  
    Simulates the circuit-level stochastic dynamics of the OCT oscillator core under thermal noise (Johnson-Nyquist noise) using Euler-Maruyama integration. It demonstrates the robustness of collective phase-locking against significant thermal fluctuations.
* **`analytical_phase_diagram.py`** (Generates Figure 13)  
    Visualizes the analytical comparison between the classical sharp phase transition ("freezing") and the proposed OCT tunneling-based dynamics near the critical constraint density $\alpha_c \approx 4.26$.

### 4. Mechanism Validation (Proof-of-Concept)
* **`inertial_dynamics_simulation.py`** (Generates Figure 15)  
    Simulates the trajectory of a dynamical system within an adversarial potential landscape. It visually demonstrates the **kinetic barrier crossing** mechanism, showing how sufficient inertial mass ($M$) allows the system to escape local traps via momentum (ballistic smoothing).
* **`pigeonhole_stiffness_test.py`** (Generates Figure 16)  
    Simulates the dynamics of a "Hard-UNSAT" benchmark (Pigeonhole Principle). It demonstrates the "digital stiffness" and dynamic divergence phenomena characteristic of unsatisfiable instances, verifying the system's Positive Lyapunov Exponent behavior when no global solution exists.
* **`stochastic_resonance_tunneling.py`** (Generates Figure 17)  
    Demonstrates the phenomenon of Stochastic Resonance Pumping. The simulation proves that thermal noise ($k_B T$) facilitates the escape from local minima via barrier tunneling, accelerating convergence to the global solution.
* **`adaptive_damping_proof.py`** (Generates Figure 18)  
    Demonstrates the resolution of the "Inertial Flattening Paradox" via the **Adaptive Damping** mechanism. Compares constant friction regimes against state-dependent dissipation ($\gamma(V)$), proving that the system can ballistically bypass shallow local minima while triggering "smart braking" to capture the deep global solution.

---

## Usage

All scripts are written in Python 3. Dependencies include `numpy`, `scipy`, `matplotlib`, `pandas`, and standard libraries.

To run a simulation (e.g., the core N=1000 test), simply execute the corresponding script:

```bash
python main_solver_N1000.py

```
## License and Usage Policy

**Academic Use Only**

The code, algorithms, and methodologies presented in this repository are the intellectual property of the author and are provided as a **Proof-of-Concept** for the Ontological Computing Theory (OCT).

* **Academic & Research Use:** Researchers are free to view, download, modify, and use this code for non-commercial academic research and validation purposes. Proper citation of the associated paper is required.
* **Commercial Use:** Any commercial application, reproduction, or implementation (including digital software, FPGA, or analog hardware realizations) is **strictly prohibited** without prior written permission from the author (Eric Moore).

For commercial licensing inquiries, please contact: eric@oct-compute.com
