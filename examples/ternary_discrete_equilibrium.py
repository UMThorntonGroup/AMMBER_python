"""
Example: ternary isothermal free energies, lower convex hull, and equilibrium.

Builds a synthetic A-B-C system (independent components B, C; solvent A) with
three solution phases, then plots:
  1) composition-plane scatter of each phase colored by G, hull facets, and
     equilibrium at a chosen overall composition
  2) 3D composition-energy view of the same data

Run from the repository root:
    python examples/ternary_discrete_equilibrium.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Allow running from a source checkout without an editable install
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ammber.Systems import IsothermalDiscreteSystem, get_lower_convex_hull_facets

# Overall composition (mole fractions of B and C) at which to evaluate equilibrium
QUERY_COMPOSITION = np.array([0.28, 0.22])
TEMPERATURE = 1000.0
OUTPUT_PATH = Path(__file__).resolve().parent / "ternary_discrete_equilibrium.png"


def sample_simplex(center, scale, npts, rng, max_sum=0.95):
    """Sample composition points near ``center`` that remain inside the A-B-C simplex."""
    pts = []
    while len(pts) < npts:
        cand = center + scale * rng.normal(size=(npts, 2))
        cand = cand[(cand >= 0).all(axis=1) & (cand.sum(axis=1) <= max_sum)]
        pts.extend(cand.tolist())
    return np.asarray(pts[:npts])


def parabolic_energy(x, cmin, fmin, kwell):
    dx = x - cmin
    return fmin + 0.5 * kwell * np.sum(dx * dx, axis=1)


def build_synthetic_ternary(rng=None):
    """Three overlapping wells chosen so QUERY_COMPOSITION sits in a multiphase region."""
    if rng is None:
        rng = np.random.default_rng(7)

    phase_specs = {
        "ALPHA": dict(center=np.array([0.12, 0.10]), scale=0.08, npts=120,
                      cmin=np.array([0.12, 0.10]), fmin=-12000.0, kwell=8.0e4),
        "BETA": dict(center=np.array([0.55, 0.12]), scale=0.09, npts=120,
                     cmin=np.array([0.55, 0.12]), fmin=-11000.0, kwell=7.0e4),
        "GAMMA": dict(center=np.array([0.20, 0.50]), scale=0.09, npts=120,
                      cmin=np.array([0.20, 0.50]), fmin=-11500.0, kwell=7.5e4),
    }

    system = IsothermalDiscreteSystem(components=["B", "C"], solution_component="A")
    for name, spec in phase_specs.items():
        xdata = sample_simplex(spec["center"], spec["scale"], spec["npts"], rng)
        Gdata = parabolic_energy(xdata, spec["cmin"], spec["fmin"], spec["kwell"])
        system.add_phase(name, xdata, Gdata, T=TEMPERATURE)
    return system


def plot_example(system, query, output_path=OUTPUT_PATH):
    hull = system.get_lc_hull()
    facets = get_lower_convex_hull_facets(system._all_points())
    equilibrium = system.get_equilibrium(query)

    # Energy of the query from the containing lower-hull facet (for 3D marker)
    query_G = None
    containing_facet = None
    for facet in facets:
        comps = facet[:, :-1]
        # barycentric test reused via get_equilibrium already; reconstruct G on facet
        A = (comps[1:] - comps[0]).T
        try:
            rest = np.linalg.solve(A, query - comps[0])
        except np.linalg.LinAlgError:
            continue
        lambdas = np.concatenate(([1.0 - rest.sum()], rest))
        if np.all(lambdas >= -1e-10):
            query_G = float(lambdas @ facet[:, -1])
            containing_facet = facet
            break

    phase_colors = {"ALPHA": "C0", "BETA": "C1", "GAMMA": "C2"}
    all_G = np.concatenate([ph.Gdata for ph in system.phases.values()])
    norm = Normalize(vmin=all_G.min(), vmax=np.percentile(all_G, 90))

    fig = plt.figure(figsize=(12.5, 5.5))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    # --- 2D composition plane ---
    for name, phase in system.phases.items():
        sc = ax2d.scatter(
            phase.xdata[:, 0],
            phase.xdata[:, 1],
            c=phase.Gdata,
            cmap="viridis",
            norm=norm,
            s=12,
            alpha=0.75,
            label=f"{name} points",
            edgecolors="none",
        )
    fig.colorbar(sc, ax=ax2d, shrink=0.85, label="G (arb.)")

    for facet in facets:
        tri = np.vstack([facet[:, :2], facet[0, :2]])
        ax2d.plot(tri[:, 0], tri[:, 1], color="0.35", lw=0.8, alpha=0.7)
    ax2d.scatter(
        hull[:, 0],
        hull[:, 1],
        s=18,
        facecolors="none",
        edgecolors="k",
        linewidths=0.8,
        label="hull vertices",
        zorder=3,
    )

    ax2d.scatter(
        query[0],
        query[1],
        marker="*",
        s=220,
        c="crimson",
        edgecolors="k",
        linewidths=0.6,
        label=f"query x={query.tolist()}",
        zorder=5,
    )
    if containing_facet is not None:
        tri = np.vstack([containing_facet[:, :2], containing_facet[0, :2]])
        ax2d.plot(tri[:, 0], tri[:, 1], color="crimson", lw=2.0, zorder=4)

    for phase_name, comp in equilibrium.items():
        comp = np.atleast_1d(comp).astype(float)
        if comp.size == 1:
            continue
        if comp.ndim > 1:
            comp = comp.ravel()[:2]
        color = phase_colors.get(phase_name.split("_")[0], "crimson")
        ax2d.scatter(
            comp[0],
            comp[1],
            s=90,
            c=color,
            edgecolors="k",
            linewidths=0.8,
            marker="D",
            label=f"eq: {phase_name}",
            zorder=6,
        )
        ax2d.plot(
            [query[0], comp[0]],
            [query[1], comp[1]],
            color=color,
            ls="--",
            lw=1.2,
            zorder=4,
        )

    # simplex bounds
    ax2d.plot([0, 1, 0, 0], [0, 0, 1, 0], color="0.5", lw=1.0)
    ax2d.set_xlabel(r"$x_\mathrm{B}$")
    ax2d.set_ylabel(r"$x_\mathrm{C}$")
    ax2d.set_xlim(-0.02, 1.02)
    ax2d.set_ylim(-0.02, 1.02)
    ax2d.set_aspect("equal")
    ax2d.set_title("Composition plane: energies, hull, equilibrium")
    ax2d.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # --- 3D energy surface ---
    for name, phase in system.phases.items():
        ax3d.scatter(
            phase.xdata[:, 0],
            phase.xdata[:, 1],
            phase.Gdata,
            c=phase_colors.get(name, "gray"),
            s=8,
            alpha=0.45,
            label=name,
        )

    hull_polys = []
    for facet in facets:
        hull_polys.append(facet)
    if hull_polys:
        coll = Poly3DCollection(
            hull_polys,
            alpha=0.25,
            facecolor="silver",
            edgecolor="0.2",
            linewidths=0.6,
        )
        ax3d.add_collection3d(coll)

    if query_G is not None:
        ax3d.scatter(
            [query[0]],
            [query[1]],
            [query_G],
            marker="*",
            s=180,
            c="crimson",
            edgecolors="k",
            depthshade=False,
            label="query on hull",
        )
    for phase_name, comp in equilibrium.items():
        comp = np.asarray(comp, dtype=float).ravel()
        if comp.size < 2:
            continue
        # match energy of nearest hull vertex for that phase composition
        d = np.linalg.norm(hull[:, :2] - comp[:2], axis=1)
        G_eq = hull[np.argmin(d), -1]
        ax3d.scatter(
            [comp[0]],
            [comp[1]],
            [G_eq],
            s=70,
            c=phase_colors.get(phase_name.split("_")[0], "crimson"),
            edgecolors="k",
            marker="D",
            depthshade=False,
            label=f"eq: {phase_name}",
        )

    ax3d.set_xlabel(r"$x_\mathrm{B}$")
    ax3d.set_ylabel(r"$x_\mathrm{C}$")
    ax3d.set_zlabel("G")
    ax3d.set_title("3D free energies and lower convex hull")
    ax3d.view_init(elev=22, azim=-55)
    ax3d.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Synthetic ternary A-B-C at T={TEMPERATURE:.0f}  |  "
        f"equilibrium phases: {', '.join(equilibrium.keys()) or '(none)'}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Saved figure to {output_path}")
    print(f"Query composition (x_B, x_C) = {query.tolist()}")
    print("Equilibrium:")
    for name, comp in equilibrium.items():
        print(f"  {name}: {np.asarray(comp).tolist()}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def main():
    system = build_synthetic_ternary()
    plot_example(system, QUERY_COMPOSITION)


if __name__ == "__main__":
    main()
