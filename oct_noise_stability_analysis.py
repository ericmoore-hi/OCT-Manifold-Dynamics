"""
Script: Noise Stability and Phase Boundary Analysis
Description: Models the theoretical stability threshold of OCT topological
invariants against thermal noise power as a function of system size (N).
Visualizes the deterministic phase-locked region vs. noise-induced chaos.

Copyright (c) 2025 Eric Moore.
This code is part of the research manuscript:
"Deterministic Solution of NP-complete Problems via Inertial Manifold Dynamics within the BSS Model Framework"
"""

import matplotlib.pyplot as plt
import numpy as np


def generate_noise_phase_diagram():
    # Variable range (N) for scaling analysis
    N = np.linspace(10, 2000, 200)

    # Stability Threshold: Logarithmic scaling based on topological protection theory
    # Represents the critical noise density before phase-locking is lost
    stability_threshold = 0.5 * np.log10(N) + 1

    plt.figure(figsize=(10, 6))

    # Deterministic SAT Zone: Region where OCT dynamics remain stable
    plt.fill_between(N, 0, stability_threshold, color='#2ecc71', alpha=0.2,
                     label='Deterministic SAT Zone (Phase-Locked State)')

    # Unstable Zone: Region where thermal noise dominates global dynamics
    plt.fill_between(N, stability_threshold, 5, color='#e74c3c', alpha=0.1,
                     label='Thermal Instability Zone (Noise-Induced Chaos)')

    # Plot the phase boundary line
    plt.plot(N, stability_threshold, color='#27ae60', linestyle='-', linewidth=2.5)

    # Labeling and Physical Units
    plt.xlabel('System Size (Variable Count $N$)', fontsize=12, fontweight='bold')
    plt.ylabel('Thermal Noise Power Density (dB/Hz equivalent)', fontsize=12, fontweight='bold')
    plt.title('OCT Stability Phase Diagram: Resilience Scaling', fontsize=14, fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', frameon=True, fontsize=10)

    # Save output for publication
    output_filename = 'OCT_Noise_Stability_Phase_Diagram.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[INFO] Phase diagram generated: {output_filename}")
    plt.show()


if __name__ == "__main__":
    generate_noise_phase_diagram()