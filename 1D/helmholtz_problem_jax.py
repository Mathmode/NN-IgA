from __future__ import annotations

"""Problem data for the 1D Helmholtz interface experiment (torch-free).

This is a lightweight replacement for the original PyTorch-based container so
Experiment 2 can run fully in JAX/JIT environments where importing PyTorch is
undesirable or unsupported.
"""

import numpy as np


class OneInterfaceHelmholtzPaper:
    """Two-zone Helmholtz interface problem from the paper (manufactured exact solution)."""

    def __init__(self, xI: float = 0.5):
        self.xI = float(xI)

        self.sigma_L = 1.0
        self.sigma_R = 4.0

        self.k_L = 31.0 * np.pi / 2.0
        self.k_R = 5.0 * np.pi

        self.alpha_L = -(self.k_L**2)
        self.alpha_R = -(10.0 * np.pi) ** 2  # -100*pi^2

        self.gN = 10.0 * np.pi

        # Manufactured constants
        self.A_L = 1.0 / np.sqrt(2.0)
        self.A_R = -0.5
        self.B_R = -31.0 / 80.0

    # -------------------------
    # Exact solution (NumPy)
    # -------------------------
    def u_exact(self, x):
        x = np.asarray(x, dtype=float)
        uL = self.A_L * np.sin(self.k_L * x)
        uR = self.A_R * np.sin(self.k_R * x) + self.B_R * np.cos(self.k_R * x)
        return np.where(x <= self.xI, uL, uR)

    def du_exact(self, x):
        x = np.asarray(x, dtype=float)
        duL = self.A_L * self.k_L * np.cos(self.k_L * x)
        duR = self.A_R * self.k_R * np.cos(self.k_R * x) - self.B_R * self.k_R * np.sin(self.k_R * x)
        return np.where(x <= self.xI, duL, duR)

    def d2u_exact(self, x):
        x = np.asarray(x, dtype=float)
        d2uL = -(self.k_L**2) * self.A_L * np.sin(self.k_L * x)
        d2uR = -(self.k_R**2) * (self.A_R * np.sin(self.k_R * x) + self.B_R * np.cos(self.k_R * x))
        return np.where(x <= self.xI, d2uL, d2uR)

    # -------------------------
    # Boundary data
    # -------------------------
    def u0(self) -> float:
        return 0.0

    def u1(self) -> float:
        return float(self.u_exact(np.array([1.0], dtype=float))[0])

    def gN_right(self) -> float:
        return float(self.gN)


__all__ = ["OneInterfaceHelmholtzPaper"]

