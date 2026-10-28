"""
Tests for multicomponent isothermal discrete classes.
"""

import numpy as np

from ammber.Systems import (
    IsothermalDiscretePhase,
    IsothermalDiscreteSystem,
    get_lower_convex_hull,
)


def test_isothermal_discrete_phase_potentials_and_free_energy():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.05, 0.4, size=(80, 2))
    x = x[x.sum(axis=1) < 0.9]
    G = 10 * ((x[:, 0] - 0.2) ** 2 + (x[:, 1] - 0.15) ** 2) - 1000

    phase = IsothermalDiscretePhase("FCC", x, G, T=800.0)
    assert phase.n_indep == 2
    assert phase.xdata.shape[0] == len(phase.Gdata)

    G_before = phase.Gdata.copy()
    phase.remove_potential([1.0, -2.0], xref=[0.0, 0.0], Gref=0)
    phase.add_potential([1.0, -2.0], xref=[0.0, 0.0], Gref=0)
    np.testing.assert_allclose(phase.Gdata, G_before)

    fe = phase.free_energy([0.2, 0.15])
    assert fe is not None
    assert abs(fe - (-1000.0)) < 1.0


def test_isothermal_discrete_system_hull_and_equilibrium():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.05, 0.4, size=(80, 2))
    x = x[x.sum(axis=1) < 0.9]
    G = 10 * ((x[:, 0] - 0.2) ** 2 + (x[:, 1] - 0.15) ** 2) - 1000

    x2 = rng.uniform(0.3, 0.7, size=(60, 2))
    x2 = x2[x2.sum(axis=1) < 0.95]
    G2 = 12 * ((x2[:, 0] - 0.5) ** 2 + (x2[:, 1] - 0.2) ** 2) - 900

    sys = IsothermalDiscreteSystem(components=["B", "C"], solution_component="A")
    sys.add_phase("FCC", x, G, T=800.0)
    sys.add_phase("BCC", x2, G2, T=800.0)

    hull = sys.get_lc_hull()
    assert hull.ndim == 2
    assert hull.shape[1] == 3
    assert hull.shape[0] > 0

    eq_sys = sys.get_equilibrium_system(miscibility_gap_threshold=0.15)
    assert len(eq_sys.phases) >= 1
    assert any(name.startswith("FCC") or name.startswith("BCC") for name in eq_sys.phases)

    eq = sys.get_equilibrium([0.25, 0.15])
    assert "FCC" in eq
    comp, G = eq["FCC"]
    np.testing.assert_allclose(comp, [0.25, 0.15])
    assert np.isscalar(G) or np.size(G) == 1
    assert np.isfinite(G)


def test_get_lower_convex_hull_1d_parabola():
    xb = np.linspace(0.01, 0.99, 50)
    Gb = (xb - 0.5) ** 2
    pts = np.column_stack([xb, Gb])
    lh = get_lower_convex_hull(pts)

    assert lh.shape[1] == 2
    assert lh.shape[0] > 0
    assert lh[:, -1].min() <= Gb.min() + 1e-12
