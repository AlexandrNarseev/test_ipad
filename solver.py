import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class SolverConfig:
    """Конфигурация численной схемы и физических параметров."""

    nx: int = 128
    ny: int = 96
    width: float = 1.0
    height: float = 0.75
    dt: float = 0.01
    viscosity: float = 1e-3
    density_diffusion: float = 0.0
    fluid_density: float = 1.0
    pressure_iterations: int = 80
    diffusion_iterations: int = 30


class FluidSolver2D:
    """
    2D-солвер несжимаемой вязкой жидкости на MAC-сетке (staggered grid).

    Почему MAC-сетка:
    - Скорости хранятся на гранях ячеек, давление — в центрах.
    - Такая расстановка снижает checkerboard-артефакты давления и повышает
      устойчивость проекционной схемы для несжимаемых течений.
    """

    def __init__(self, config: SolverConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.dx = config.width / config.nx
        self.dy = config.height / config.ny
        self.dt = config.dt
        self.nu = config.viscosity
        self.kappa = config.density_diffusion
        self.rho = config.fluid_density

        # MAC-сетка:
        # u(i+1/2, j) -> (nx+1, ny)
        # v(i, j+1/2) -> (nx, ny+1)
        # p(i, j), density(i, j) -> (nx, ny)
        self.u = np.zeros((self.nx + 1, self.ny), dtype=np.float64)
        self.v = np.zeros((self.nx, self.ny + 1), dtype=np.float64)
        self.p = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.density = np.zeros((self.nx, self.ny), dtype=np.float64)

        self._u_prev = np.zeros_like(self.u)
        self._v_prev = np.zeros_like(self.v)
        self._density_prev = np.zeros_like(self.density)

    @staticmethod
    def from_json(path: str | Path) -> SolverConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SolverConfig(**data)

    def apply_no_slip_boundaries(self) -> None:
        """No-slip: нулевая скорость на твердых стенках."""
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0

        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0

    def add_forces(
        self,
        fx: Optional[np.ndarray] = None,
        fy: Optional[np.ndarray] = None,
        density_source: Optional[np.ndarray] = None,
    ) -> None:
        """
        Добавление внешних сил и источников плотности.

        fx, fy задаются в центрах ячеек (nx, ny) и интерполируются
        к соответствующим компонентам скорости на MAC-сетке.
        """
        if fx is not None:
            assert fx.shape == (self.nx, self.ny)
            # Интерполяция fx в u-узлы (грани по x)
            self.u[1:-1, :] += self.dt * 0.5 * (fx[:-1, :] + fx[1:, :]) / self.rho
        if fy is not None:
            assert fy.shape == (self.nx, self.ny)
            # Интерполяция fy в v-узлы (грани по y)
            self.v[:, 1:-1] += self.dt * 0.5 * (fy[:, :-1] + fy[:, 1:]) / self.rho
        if density_source is not None:
            assert density_source.shape == (self.nx, self.ny)
            self.density += self.dt * density_source

        self.apply_no_slip_boundaries()

    def velocity_at_cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Скорость в центрах ячеек для анализа/визуализации."""
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        return uc, vc

    def _clamp_position(self, x: float, y: float) -> tuple[float, float]:
        x = float(np.clip(x, 0.0, self.nx * self.dx))
        y = float(np.clip(y, 0.0, self.ny * self.dy))
        return x, y

    def _sample_center_field(self, field: np.ndarray, x: float, y: float) -> float:
        """Билинейная интерполяция поля, заданного в центрах ячеек."""
        x, y = self._clamp_position(x, y)
        gx = x / self.dx - 0.5
        gy = y / self.dy - 0.5

        i0 = int(np.floor(gx))
        j0 = int(np.floor(gy))
        tx = gx - i0
        ty = gy - j0

        i0 = np.clip(i0, 0, self.nx - 1)
        j0 = np.clip(j0, 0, self.ny - 1)
        i1 = np.clip(i0 + 1, 0, self.nx - 1)
        j1 = np.clip(j0 + 1, 0, self.ny - 1)

        f00 = field[i0, j0]
        f10 = field[i1, j0]
        f01 = field[i0, j1]
        f11 = field[i1, j1]

        return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11

    def _sample_u(self, field: np.ndarray, x: float, y: float) -> float:
        """Билинейная интерполяция u(i+1/2, j)."""
        x, y = self._clamp_position(x, y)
        gx = x / self.dx
        gy = y / self.dy - 0.5

        i0 = int(np.floor(gx))
        j0 = int(np.floor(gy))
        tx = gx - i0
        ty = gy - j0

        i0 = np.clip(i0, 0, self.nx)
        j0 = np.clip(j0, 0, self.ny - 1)
        i1 = np.clip(i0 + 1, 0, self.nx)
        j1 = np.clip(j0 + 1, 0, self.ny - 1)

        f00 = field[i0, j0]
        f10 = field[i1, j0]
        f01 = field[i0, j1]
        f11 = field[i1, j1]

        return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11

    def _sample_v(self, field: np.ndarray, x: float, y: float) -> float:
        """Билинейная интерполяция v(i, j+1/2)."""
        x, y = self._clamp_position(x, y)
        gx = x / self.dx - 0.5
        gy = y / self.dy

        i0 = int(np.floor(gx))
        j0 = int(np.floor(gy))
        tx = gx - i0
        ty = gy - j0

        i0 = np.clip(i0, 0, self.nx - 1)
        j0 = np.clip(j0, 0, self.ny)
        i1 = np.clip(i0 + 1, 0, self.nx - 1)
        j1 = np.clip(j0 + 1, 0, self.ny)

        f00 = field[i0, j0]
        f10 = field[i1, j0]
        f01 = field[i0, j1]
        f11 = field[i1, j1]

        return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11

    def _velocity_at(self, x: float, y: float) -> tuple[float, float]:
        return self._sample_u(self.u, x, y), self._sample_v(self.v, x, y)

    def advect(self) -> None:
        """
        Semi-Lagrangian advection.

        Для каждой степени свободы трассируем назад характеристику:
        x_back = x - dt * u(x), y_back = y - dt * v(y).
        Это более устойчиво (безусловно по CFL для линейной части),
        но добавляет численную диффузию.
        """
        self._u_prev[:] = self.u
        self._v_prev[:] = self.v
        self._density_prev[:] = self.density

        # Адвекция u на его сетке
        for i in range(1, self.nx):
            x = i * self.dx
            for j in range(0, self.ny):
                y = (j + 0.5) * self.dy
                ux, vy = self._velocity_at(x, y)
                xb, yb = self._clamp_position(x - self.dt * ux, y - self.dt * vy)
                self.u[i, j] = self._sample_u(self._u_prev, xb, yb)

        # Адвекция v на его сетке
        for i in range(0, self.nx):
            x = (i + 0.5) * self.dx
            for j in range(1, self.ny):
                y = j * self.dy
                ux, vy = self._velocity_at(x, y)
                xb, yb = self._clamp_position(x - self.dt * ux, y - self.dt * vy)
                self.v[i, j] = self._sample_v(self._v_prev, xb, yb)

        # Адвекция плотности в центрах
        for i in range(self.nx):
            x = (i + 0.5) * self.dx
            for j in range(self.ny):
                y = (j + 0.5) * self.dy
                ux, vy = self._velocity_at(x, y)
                xb, yb = self._clamp_position(x - self.dt * ux, y - self.dt * vy)
                self.density[i, j] = self._sample_center_field(self._density_prev, xb, yb)

        self.apply_no_slip_boundaries()

    def _diffuse_scalar(self, field: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
        """Неявная диффузия (Gauss-Seidel) для скалярного поля в центрах ячеек."""
        out = field.copy()
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy

        for _ in range(iterations):
            out[1:-1, 1:-1] = (
                field[1:-1, 1:-1]
                + alpha
                * (
                    (out[2:, 1:-1] + out[:-2, 1:-1]) / dx2
                    + (out[1:-1, 2:] + out[1:-1, :-2]) / dy2
                )
            ) / (1.0 + 2.0 * alpha * (1.0 / dx2 + 1.0 / dy2))

            out[0, :] = out[1, :]
            out[-1, :] = out[-2, :]
            out[:, 0] = out[:, 1]
            out[:, -1] = out[:, -2]

        return out

    def diffuse(self) -> None:
        """
        Неявный шаг вязкости/диффузии:
        (I - dt*nu*L) u^{n+1} = u^*
        (I - dt*nu*L) v^{n+1} = v^*

        Схема устойчива для больших dt относительно чисто явной диффузии.
        """
        alpha_u = self.dt * self.nu
        alpha_d = self.dt * self.kappa

        self._u_prev[:] = self.u
        self._v_prev[:] = self.v

        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        denom = 1.0 + 2.0 * alpha_u * (1.0 / dx2 + 1.0 / dy2)

        for _ in range(self.cfg.diffusion_iterations):
            self.u[1:-1, 1:-1] = (
                self._u_prev[1:-1, 1:-1]
                + alpha_u
                * (
                    (self.u[2:, 1:-1] + self.u[:-2, 1:-1]) / dx2
                    + (self.u[1:-1, 2:] + self.u[1:-1, :-2]) / dy2
                )
            ) / denom

            self.v[1:-1, 1:-1] = (
                self._v_prev[1:-1, 1:-1]
                + alpha_u
                * (
                    (self.v[2:, 1:-1] + self.v[:-2, 1:-1]) / dx2
                    + (self.v[1:-1, 2:] + self.v[1:-1, :-2]) / dy2
                )
            ) / denom
            self.apply_no_slip_boundaries()

        if alpha_d > 0.0:
            self.density = self._diffuse_scalar(self.density, alpha_d, self.cfg.diffusion_iterations)

    def divergence(self) -> np.ndarray:
        """Дивергенция скорости в центрах ячеек (контроль несжимаемости)."""
        return (
            (self.u[1:, :] - self.u[:-1, :]) / self.dx
            + (self.v[:, 1:] - self.v[:, :-1]) / self.dy
        )

    def project(self) -> None:
        """
        Проекционный шаг:
        1) решаем Пуассон: ∇²p = (ρ/dt) div(u*)
        2) корректируем скорость: u = u* - dt/ρ ∇p

        После шага div(u) -> 0 (в пределах точности итерационного решателя).
        """
        div = self.divergence()
        rhs = (self.rho / self.dt) * div

        self.p.fill(0.0)
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        denom = 2.0 * (1.0 / dx2 + 1.0 / dy2)

        for _ in range(self.cfg.pressure_iterations):
            self.p[1:-1, 1:-1] = (
                (self.p[2:, 1:-1] + self.p[:-2, 1:-1]) / dx2
                + (self.p[1:-1, 2:] + self.p[1:-1, :-2]) / dy2
                - rhs[1:-1, 1:-1]
            ) / denom

            # Neumann для давления (dp/dn=0) у твердых стен
            self.p[0, :] = self.p[1, :]
            self.p[-1, :] = self.p[-2, :]
            self.p[:, 0] = self.p[:, 1]
            self.p[:, -1] = self.p[:, -2]

        self.u[1:-1, :] -= (self.dt / self.rho) * (self.p[1:, :] - self.p[:-1, :]) / self.dx
        self.v[:, 1:-1] -= (self.dt / self.rho) * (self.p[:, 1:] - self.p[:, :-1]) / self.dy
        self.apply_no_slip_boundaries()

    def vorticity(self) -> np.ndarray:
        """Вихрь ω = dv/dx - du/dy в центрах ячеек."""
        uc, vc = self.velocity_at_cell_centers()

        du_dy = np.zeros_like(uc)
        dv_dx = np.zeros_like(vc)

        du_dy[:, 1:-1] = (uc[:, 2:] - uc[:, :-2]) / (2.0 * self.dy)
        du_dy[:, 0] = (uc[:, 1] - uc[:, 0]) / self.dy
        du_dy[:, -1] = (uc[:, -1] - uc[:, -2]) / self.dy

        dv_dx[1:-1, :] = (vc[2:, :] - vc[:-2, :]) / (2.0 * self.dx)
        dv_dx[0, :] = (vc[1, :] - vc[0, :]) / self.dx
        dv_dx[-1, :] = (vc[-1, :] - vc[-2, :]) / self.dx

        return dv_dx - du_dy

    def metrics(self) -> Dict[str, float]:
        uc, vc = self.velocity_at_cell_centers()
        speed = np.sqrt(uc * uc + vc * vc)
        kinetic_energy = 0.5 * self.rho * (uc * uc + vc * vc)
        div = self.divergence()

        return {
            "max_speed": float(np.max(speed)),
            "mean_kinetic_energy": float(np.mean(kinetic_energy)),
            "max_abs_divergence": float(np.max(np.abs(div))),
            "l2_divergence": float(np.sqrt(np.mean(div * div))),
        }

    def step(
        self,
        fx: Optional[np.ndarray] = None,
        fy: Optional[np.ndarray] = None,
        density_source: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        self.add_forces(fx=fx, fy=fy, density_source=density_source)
        self.advect()
        self.diffuse()
        self.project()
        return self.metrics()

    def save_state(self, path: str | Path, extra: Optional[Dict[str, float]] = None) -> None:
        payload = {
            "u": self.u,
            "v": self.v,
            "p": self.p,
            "density": self.density,
            "config_json": json.dumps(asdict(self.cfg)),
        }
        if extra:
            for k, v in extra.items():
                payload[f"meta_{k}"] = np.array(v)

        np.savez_compressed(path, **payload)
