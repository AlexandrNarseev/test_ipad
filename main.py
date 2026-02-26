import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.ndimage import label


@dataclass
class ModelParams:
    """Параметры 2D фазово-диффузионной модели (калибруются под эксперимент)."""

    # Геометрия и сетка
    nx: int = 200
    ny: int = 200
    lx: float = 2e-3
    ly: float = 2e-3

    # Температура/кристаллизация
    t_init: float = 980.0
    t_liq: float = 933.0
    t_sol: float = 893.0
    cooling_rate: float = 5.0

    # Диффузия водорода
    d_l: float = 1.0e-8
    d_s: float = 1.0e-11

    # Поверхностные/термодинамические параметры
    gamma: float = 0.9
    v_m_h2: float = 2.24e-2  # м^3/моль
    r_gas: float = 8.314

    # TiH2
    t_decomp: float = 700.0
    k0: float = 1e5
    q_act: float = 140e3
    n_particles: int = 40
    ti_radius_min: int = 2
    ti_radius_max: int = 3
    source_scale: float = 5e3  # моль/(м^3*c)

    # Фазовое поле пор
    m_phi: float = 1e-10
    beta: float = 2.0e2
    supersat: float = 0.2
    nuc_rate: float = 0.08
    eps_cells: float = 2.0

    # Усадка
    k_shrink: float = 0.02

    # Численная схема
    dt_safety: float = 0.35
    sink_coeff: float = 0.4

    # Перевод растворимости (мл H2 / 100 г Al -> моль/м^3)
    rho_al: float = 2700.0


class PoreFormationModel:
    def __init__(self, p: ModelParams, seed: int = 42):
        self.p = p
        self.rng = np.random.default_rng(seed)

        self.dx = p.lx / p.nx
        self.dy = p.ly / p.ny
        self.dt = p.dt_safety * min(self.dx, self.dy) ** 2 / (4.0 * p.d_l)

        self.phi = np.zeros((p.nx, p.ny), dtype=np.float64)
        self.c = np.zeros((p.nx, p.ny), dtype=np.float64)
        self.t_field = np.full((p.nx, p.ny), p.t_init, dtype=np.float64)
        self.f_s = self.compute_solid_fraction(self.t_field)
        self.f_s_prev = self.f_s.copy()

        self.f_tih2 = np.zeros_like(self.c)
        self.init_tih2_particles()

        c_eq0 = self.c_eq_mol_m3(self.t_field)
        self.c[:] = 0.85 * c_eq0

        self.time = 0.0

    def init_tih2_particles(self):
        x = np.arange(self.p.nx)[:, None]
        y = np.arange(self.p.ny)[None, :]
        for _ in range(self.p.n_particles):
            cx = self.rng.integers(0, self.p.nx)
            cy = self.rng.integers(0, self.p.ny)
            r = self.rng.integers(self.p.ti_radius_min, self.p.ti_radius_max + 1)
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
            self.f_tih2[mask] = 1.0

    def c_eq_ml_per_100g(self, temp):
        return 0.69 * np.exp(-3200.0 / np.maximum(temp, 1.0))

    def c_eq_mol_m3(self, temp):
        # 1 мл H2 = 1e-6 м^3 при Н.У.; n = V/Vm; затем делим на объем 100 г Al.
        # coeff = rho_al * 1e-5 / Vm (для исходных единиц мл/100г).
        coeff = self.p.rho_al * 1e-5 / self.p.v_m_h2
        return coeff * self.c_eq_ml_per_100g(temp)

    def compute_solid_fraction(self, temp):
        fs = (self.p.t_liq - temp) / (self.p.t_liq - self.p.t_sol)
        return np.clip(fs, 0.0, 1.0)

    @staticmethod
    def laplacian(field):
        return (
            np.roll(field, 1, 0)
            + np.roll(field, -1, 0)
            + np.roll(field, 1, 1)
            + np.roll(field, -1, 1)
            - 4.0 * field
        )

    def div_d_grad(self, field, d_eff):
        d_xp = 0.5 * (d_eff + np.roll(d_eff, -1, 0))
        d_xm = 0.5 * (d_eff + np.roll(d_eff, 1, 0))
        d_yp = 0.5 * (d_eff + np.roll(d_eff, -1, 1))
        d_ym = 0.5 * (d_eff + np.roll(d_eff, 1, 1))

        flux_xp = d_xp * (np.roll(field, -1, 0) - field) / self.dx
        flux_xm = d_xm * (field - np.roll(field, 1, 0)) / self.dx
        flux_yp = d_yp * (np.roll(field, -1, 1) - field) / self.dy
        flux_ym = d_ym * (field - np.roll(field, 1, 1)) / self.dy

        return (flux_xp - flux_xm) / self.dx + (flux_yp - flux_ym) / self.dy

    def pore_radius_from_phi(self):
        dphix = (np.roll(self.phi, -1, 0) - np.roll(self.phi, 1, 0)) / (2 * self.dx)
        dphiy = (np.roll(self.phi, -1, 1) - np.roll(self.phi, 1, 1)) / (2 * self.dy)
        norm = np.sqrt(dphix**2 + dphiy**2 + 1e-20)
        nx = dphix / norm
        ny = dphiy / norm
        curvature = (
            (np.roll(nx, -1, 0) - np.roll(nx, 1, 0)) / (2 * self.dx)
            + (np.roll(ny, -1, 1) - np.roll(ny, 1, 1)) / (2 * self.dy)
        )
        radius = 1.0 / (np.abs(curvature) + 5.0)
        return np.clip(radius, self.dx, 5e-4)

    def update(self):
        p = self.p

        # Охлаждение и кристаллизация
        self.time += self.dt
        new_t = p.t_init - p.cooling_rate * self.time
        self.t_field.fill(new_t)
        self.f_s_prev[:] = self.f_s
        self.f_s[:] = self.compute_solid_fraction(self.t_field)
        df_s_dt = np.maximum((self.f_s - self.f_s_prev) / self.dt, 0.0)

        # Растворимость с поправкой Лапласа
        c_eq = self.c_eq_mol_m3(self.t_field)
        r_pore = self.pore_radius_from_phi()
        laplace_arg = (2.0 * p.gamma * p.v_m_h2) / (p.r_gas * self.t_field * r_pore)
        c_eq_eff = c_eq * np.exp(np.clip(laplace_arg, 0.0, 8.0))

        # Разложение TiH2
        arrh = p.k0 * np.exp(-p.q_act / (p.r_gas * np.maximum(self.t_field, 1.0)))
        active = (self.t_field >= p.t_decomp).astype(np.float64)
        s_decomp = p.source_scale * arrh * self.f_tih2 * active
        self.f_tih2 = np.clip(self.f_tih2 - arrh * self.f_tih2 * self.dt * active, 0.0, 1.0)

        # Диффузия + источник + сток в поры
        d_eff = p.d_l * (1.0 - self.f_s) + p.d_s * self.f_s
        sink_por = p.sink_coeff * np.maximum(self.c - c_eq_eff, 0.0) * self.phi
        self.c += self.dt * (self.div_d_grad(self.c, d_eff) + s_decomp - sink_por)
        self.c = np.clip(self.c, 0.0, None)

        # Зарождение пор
        nucleation_zone = (self.c > c_eq * (1.0 + p.supersat)) & (self.f_s < 0.8) & (self.phi < 0.2)
        random_gate = self.rng.random(self.phi.shape) < p.nuc_rate
        self.phi[nucleation_zone & random_gate] += 0.25

        # Рост пор (фазовое поле)
        eps = p.eps_cells * self.dx
        double_well = self.phi * (1.0 - self.phi) * (1.0 - 2.0 * self.phi) / (eps**2)
        driving = p.beta * (self.c - c_eq_eff)
        lap_phi = self.laplacian(self.phi) / (self.dx * self.dy)
        s_shrink = p.k_shrink * df_s_dt

        dphi_dt = p.m_phi * (lap_phi - double_well + driving)
        dphi_dt += s_shrink * (self.f_s > 0.7)
        dphi_dt[self.f_s > 0.95] = np.maximum(dphi_dt[self.f_s > 0.95], 0.0) * 0.0

        self.phi += self.dt * dphi_dt
        self.phi = np.clip(self.phi, 0.0, 1.0)

        return self.metrics(df_s_dt)

    def metrics(self, df_s_dt):
        pore_mask = self.phi > 0.5
        structure = np.ones((3, 3), dtype=np.int8)
        labels, n_pores = label(pore_mask, structure=structure)

        area_cell = self.dx * self.dy
        radii = []
        for idx in range(1, n_pores + 1):
            area = np.sum(labels == idx) * area_cell
            if area > 0:
                radii.append(np.sqrt(area / np.pi))

        avg_radius = float(np.mean(radii)) if radii else 0.0
        total_porosity = float(np.mean(self.phi))
        shrink_porosity = float(np.mean(self.phi[self.f_s > 0.9])) if np.any(self.f_s > 0.9) else 0.0

        return {
            "time": self.time,
            "porosity": total_porosity,
            "avg_radius": avg_radius,
            "n_pores": int(n_pores),
            "shrink_porosity": shrink_porosity,
            "temp": float(self.t_field[0, 0]),
            "mean_fs": float(np.mean(self.f_s)),
            "max_df_s_dt": float(np.max(df_s_dt)),
        }


def make_animation(model: PoreFormationModel, n_steps: int, every: int = 20):
    fig, axs = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    ax_phi, ax_c, ax_fs, ax_t = axs.ravel()

    im_phi = ax_phi.imshow(model.phi.T, origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax_phi.set_title("Поры (phi)")
    fig.colorbar(im_phi, ax=ax_phi, fraction=0.046)

    im_c = ax_c.imshow(model.c.T, origin="lower", cmap="viridis")
    ax_c.set_title("Концентрация H2 (моль/м^3)")
    fig.colorbar(im_c, ax=ax_c, fraction=0.046)

    im_fs = ax_fs.imshow(model.f_s.T, origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax_fs.set_title("Доля твердой фазы f_s")
    fig.colorbar(im_fs, ax=ax_fs, fraction=0.046)

    im_t = ax_t.imshow(model.t_field.T, origin="lower", cmap="coolwarm", vmin=model.p.t_sol - 50, vmax=model.p.t_init)
    ax_t.set_title("Температура T (K)")
    fig.colorbar(im_t, ax=ax_t, fraction=0.046)

    text = fig.text(0.01, 0.01, "", fontsize=10)

    def update(frame):
        metrics = None
        for _ in range(every):
            metrics = model.update()

        im_phi.set_data(model.phi.T)
        im_c.set_data(model.c.T)
        im_c.set_clim(float(np.min(model.c)), float(np.max(model.c) + 1e-12))
        im_fs.set_data(model.f_s.T)
        im_t.set_data(model.t_field.T)

        text.set_text(
            f"t={metrics['time']:.3f} c | T={metrics['temp']:.1f}K | porosity={metrics['porosity']:.4f} | "
            f"Ravg={metrics['avg_radius']*1e6:.1f} um | pores={metrics['n_pores']} | "
            f"shrink={metrics['shrink_porosity']:.4f}"
        )

        if frame % 5 == 0:
            print(
                f"step={frame*every:05d} t={metrics['time']:.3f}s por={metrics['porosity']:.5f} "
                f"Ravg={metrics['avg_radius']*1e6:.2f}um N={metrics['n_pores']} "
                f"shrink={metrics['shrink_porosity']:.5f} fs={metrics['mean_fs']:.3f}"
            )

        return [im_phi, im_c, im_fs, im_t, text]

    anim = FuncAnimation(fig, update, frames=n_steps // every, interval=50, blit=False)
    plt.show()
    return anim


def run_headless(model: PoreFormationModel, n_steps: int, print_every: int = 50):
    for i in range(n_steps):
        metrics = model.update()
        if i % print_every == 0 or i == n_steps - 1:
            print(
                f"step={i:05d} t={metrics['time']:.3f}s T={metrics['temp']:.1f}K "
                f"porosity={metrics['porosity']:.5f} Ravg={metrics['avg_radius']*1e6:.2f}um "
                f"pores={metrics['n_pores']} shrink={metrics['shrink_porosity']:.5f}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="2D phase-diffusion model of pore formation in Al melt")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    params = ModelParams()
    model = PoreFormationModel(params, seed=args.seed)

    print(f"Auto dt={model.dt:.3e} s (stability limit {model.dx**2/(4*params.d_l):.3e} s)")

    if args.headless:
        run_headless(model, n_steps=args.steps)
    else:
        _ = make_animation(model, n_steps=args.steps)


if __name__ == "__main__":
    main()
