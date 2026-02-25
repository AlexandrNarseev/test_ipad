import argparse
import logging
from pathlib import Path

import numpy as np

from solver import FluidSolver2D, SolverConfig
from visualization import FluidVisualizer, VisualizationConfig


def build_default_forcing(amplitude: float = 30.0, source_strength: float = 3.0):
    """Создает внешний форсинг: инжекция плотности и локальный импульс."""

    def forcing(step: int, solver: FluidSolver2D):
        fx = np.zeros((solver.nx, solver.ny), dtype=np.float64)
        fy = np.zeros((solver.nx, solver.ny), dtype=np.float64)
        src = np.zeros((solver.nx, solver.ny), dtype=np.float64)

        cx = solver.nx // 2
        cy = solver.ny // 5
        rad = max(2, min(solver.nx, solver.ny) // 15)

        x = np.arange(solver.nx)[:, None]
        y = np.arange(solver.ny)[None, :]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= rad**2

        # Подъемная сила формирует струю, полезно для тестирования устойчивости.
        fy[mask] = amplitude
        src[mask] = source_strength

        # Пульсирующий боковой сдвиг для генерации вихревых структур.
        fx[mask] = 0.25 * amplitude * np.sin(0.03 * step)
        return fx, fy, src

    return forcing


def parse_args():
    parser = argparse.ArgumentParser(description="2D incompressible viscous fluid simulation")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config for SolverConfig")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--show-vorticity", action="store_true", help="Visualize vorticity instead of density")
    parser.add_argument("--save-every", type=int, default=0, help="Save state every N steps (0 disables)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for state files")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = SolverConfig()
    if args.config:
        cfg = FluidSolver2D.from_json(args.config)

    solver = FluidSolver2D(cfg)
    forcing = build_default_forcing()

    # Ограничение по dt для точности (не строгая граница устойчивости из-за semi-Lagrangian):
    # CFL ~ max(|u|) * dt / min(dx,dy) <= O(1) желательно для меньшей численной диффузии.
    logging.info(
        "Grid=(%d,%d), dt=%.4g, nu=%.4g, dx=%.4g, dy=%.4g",
        solver.nx,
        solver.ny,
        solver.dt,
        solver.nu,
        solver.dx,
        solver.dy,
    )

    out_dir = Path(args.output_dir)
    if args.save_every > 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.headless:
        for step in range(args.steps):
            fx, fy, src = forcing(step, solver)
            metrics = solver.step(fx=fx, fy=fy, density_source=src)
            if step % 25 == 0 or step == args.steps - 1:
                logging.info(
                    "step=%d max|u|=%.3e E=%.3e max|div|=%.3e l2(div)=%.3e",
                    step,
                    metrics["max_speed"],
                    metrics["mean_kinetic_energy"],
                    metrics["max_abs_divergence"],
                    metrics["l2_divergence"],
                )
            if args.save_every > 0 and step % args.save_every == 0:
                solver.save_state(out_dir / f"state_{step:06d}.npz", extra={"step": step})
        return

    vis_cfg = VisualizationConfig(show_vorticity=args.show_vorticity)
    visualizer = FluidVisualizer(solver=solver, cfg=vis_cfg, forcing_callback=forcing)
    anim = visualizer.animate(frames=args.steps)

    # Удерживаем ссылку на анимацию, чтобы Matplotlib не удалил объект GC.
    _ = anim

    if args.save_every > 0:
        # При интерактивном режиме сохраняем начальное состояние.
        solver.save_state(out_dir / "state_initial.npz", extra={"step": 0})

    visualizer.show(frames=args.steps)


if __name__ == "__main__":
    main()
