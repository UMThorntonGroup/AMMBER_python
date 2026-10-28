"""
Tests for multicomponent isothermal discrete classes.
"""

import numpy as np

from ammber.Systems import (
    IsothermalDiscretePhase,
    IsothermalDiscreteSystem,
    _simplex_distance,
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


def test_simplex_distance_is_invariant_to_solvent_label():
    full_compositions = np.array(
        [
            [0.20, 0.30, 0.50],
            [0.22, 0.31, 0.47],
            [0.30, 0.30, 0.40],
            [0.18, 0.28, 0.54],
        ]
    )
    reference = full_compositions[0]

    distances_with_c_solvent = _simplex_distance(
        full_compositions[1:, :2], reference[:2]
    )
    distances_with_a_solvent = _simplex_distance(
        full_compositions[1:, 1:], reference[1:]
    )

    np.testing.assert_allclose(distances_with_c_solvent, distances_with_a_solvent)
    np.testing.assert_array_equal(
        distances_with_c_solvent <= 0.05, distances_with_a_solvent <= 0.05
    )


def test_resample_near_xpoint_and_equilibrium_use_simplex_ball():
    xdata = np.array([[0.2, 0.2], [0.28, 0.2], [0.2, 0.28]])
    system = IsothermalDiscreteSystem(components=["B", "C"], solution_component="A")
    system.add_phase("FCC", xdata, [0.0, 0.0, 0.0])
    system.add_phase("LIQUID", xdata, [10.0, 10.0, 10.0])

    xeq = np.array([0.22, 0.22])
    xdist = 0.04
    phase = system.phases["FCC"].resample_near_xpoint(
        xeq.reshape(1, -1), xdist=xdist, npts=21
    )
    assert phase is not system.phases["FCC"]
    assert np.all(phase.xdata >= 0)
    assert np.all(phase.xdata.sum(axis=1) <= 1)
    assert np.all(_simplex_distance(phase.xdata, xeq) <= xdist)

    near_equilibrium = system.resample_near_equilibrium(
        xeq, xdist=xdist, npts=21
    )
    assert set(near_equilibrium.phases) == {"FCC"}
    resampled = near_equilibrium.phases["FCC"]
    assert resampled.xdata.shape[0] > 0
    assert np.all(_simplex_distance(resampled.xdata, xeq) <= xdist)
    assert np.all(np.isfinite(resampled.Gdata))


def test_get_lower_convex_hull_1d_parabola():
    xb = np.linspace(0.01, 0.99, 50)
    Gb = (xb - 0.5) ** 2
    pts = np.column_stack([xb, Gb])
    lh = get_lower_convex_hull(pts)

    assert lh.shape[1] == 2
    assert lh.shape[0] > 0
    assert lh[:, -1].min() <= Gb.min() + 1e-12
