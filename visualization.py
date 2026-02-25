from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from solver import FluidSolver2D


@dataclass
class VisualizationConfig:
    interval_ms: int = 30
    stride: int = 4
    cmap_density: str = "viridis"
    cmap_vorticity: str = "RdBu_r"
    show_vorticity: bool = False


class FluidVisualizer:
    def __init__(
        self,
        solver: FluidSolver2D,
        cfg: VisualizationConfig,
        forcing_callback: Optional[Callable[[int, FluidSolver2D], tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    ):
        self.solver = solver
        self.cfg = cfg
        self.forcing_callback = forcing_callback

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_title("2D Incompressible Viscous Flow")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        field0 = self.solver.vorticity() if self.cfg.show_vorticity else self.solver.density
        self.img = self.ax.imshow(
            field0.T,
            origin="lower",
            cmap=self.cfg.cmap_vorticity if self.cfg.show_vorticity else self.cfg.cmap_density,
            extent=[0, solver.nx * solver.dx, 0, solver.ny * solver.dy],
            aspect="auto",
        )
        self.cbar = self.fig.colorbar(self.img, ax=self.ax)
        self.cbar.set_label("vorticity" if self.cfg.show_vorticity else "density")

        self.quiver = None

    def _update_quiver(self):
        uc, vc = self.solver.velocity_at_cell_centers()
        stride = self.cfg.stride

        x = (np.arange(self.solver.nx) + 0.5) * self.solver.dx
        y = (np.arange(self.solver.ny) + 0.5) * self.solver.dy
        xx, yy = np.meshgrid(x[::stride], y[::stride], indexing="ij")

        uq = uc[::stride, ::stride]
        vq = vc[::stride, ::stride]

        if self.quiver is None:
            self.quiver = self.ax.quiver(xx, yy, uq, vq, color="white", alpha=0.7, scale=20)
        else:
            self.quiver.set_UVC(uq, vq)

    def _frame(self, frame_id: int):
        fx = fy = src = None
        if self.forcing_callback is not None:
            fx, fy, src = self.forcing_callback(frame_id, self.solver)

        metrics = self.solver.step(fx=fx, fy=fy, density_source=src)

        field = self.solver.vorticity() if self.cfg.show_vorticity else self.solver.density
        self.img.set_data(field.T)
        self.img.set_clim(vmin=float(np.min(field)), vmax=float(np.max(field) + 1e-12))

        self._update_quiver()
        self.ax.set_title(
            f"step={frame_id} | max|u|={metrics['max_speed']:.3e} | "
            f"E={metrics['mean_kinetic_energy']:.3e} | max|div|={metrics['max_abs_divergence']:.3e}"
        )

        artists = [self.img]
        if self.quiver is not None:
            artists.append(self.quiver)
        return artists

    def animate(self, frames: int = 500) -> FuncAnimation:
        anim = FuncAnimation(
            self.fig,
            self._frame,
            frames=frames,
            interval=self.cfg.interval_ms,
            blit=False,
        )
        return anim

    def show(self, frames: int = 500):
        self.animate(frames=frames)
        plt.tight_layout()
        plt.show()
