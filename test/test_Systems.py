"""
Tests for multicomponent isothermal discrete classes.
"""

import numpy as np

from ammber.Systems import (
    IsothermalDiscretePhase,
    IsothermalDiscreteSystem,
    IsothermalQuadraticPhase,
    IsothermalQuadraticSystem,
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


def test_isothermal_quadratic_phase_energy_and_potentials():
    phase = IsothermalQuadraticPhase(
        "FCC",
        A=[[2.0, 0.5], [0.5, 3.0]],
        B=[1.0, -2.0],
        C=4.0,
        c_ref=[0.2, 0.3],
        T=800.0,
    )
    x = np.array([0.4, 0.1])
    delta = x - np.array([0.2, 0.3])
    expected = 0.5 * delta @ np.array([[2.0, 0.5], [0.5, 3.0]]) @ delta
    expected += np.array([1.0, -2.0]) @ delta + 4.0
    assert phase.free_energy(x) == expected

    sample_points = np.array([[0.4, 0.1], [0.2, 0.3]])
    np.testing.assert_allclose(
        phase.free_energy(sample_points), [expected, 4.0]
    )

    before = phase.free_energy(x)
    phase.add_potential([1.5, -0.5], xref=[0.1, 0.2], Gref=3.0)
    expected_added = before + (x - [0.1, 0.2]) @ [1.5, -0.5] + 3.0
    np.testing.assert_allclose(phase.free_energy(x), expected_added)
    phase.remove_potential([1.5, -0.5], xref=[0.1, 0.2], Gref=3.0)
    np.testing.assert_allclose(phase.free_energy(x), before)


def test_isothermal_quadratic_phase_requires_symmetric_matrix():
    import pytest

    with pytest.raises(ValueError, match="symmetric"):
        IsothermalQuadraticPhase(
            "FCC", A=[[1.0, 2.0], [0.0, 1.0]], B=[0.0, 0.0], C=0, c_ref=[0, 0]
        )


def test_isothermal_quadratic_phase_single_component_inputs():
    phase = IsothermalQuadraticPhase(
        "BCC", A=[[2.0]], B=[-1.0], C=3.0, c_ref=[0.25]
    )
    assert phase.free_energy(0.25) == 3.0
    np.testing.assert_allclose(phase.free_energy([0.25, 0.5]), [3.0, 2.8125])
    discrete = phase.discretize(0.25)
    np.testing.assert_array_equal(discrete.xdata, [[0.25]])
    np.testing.assert_array_equal(discrete.Gdata, [3.0])


def test_isothermal_quadratic_system_discretization_and_potential():
    system = IsothermalQuadraticSystem(
        components=["B", "C"], solution_component="A", T=900.0
    )
    system.add_phase(
        "FCC",
        A=[[2.0, 0.0], [0.0, 4.0]],
        B=[1.0, -1.0],
        C=-10.0,
        c_ref=[0.1, 0.2],
    )
    xdata = np.array([[0.1, 0.2], [0.3, 0.1], [0.2, 0.2]])
    system.add_potential([0.5, 1.0], xref=[0.0, 0.0], Gref=2.0)

    discrete = system.to_discrete(xdata=xdata)
    phase = discrete.phases["FCC"]
    assert phase.T == 900.0
    np.testing.assert_array_equal(phase.xdata, xdata)
    np.testing.assert_allclose(
        phase.Gdata, system.phases["FCC"].free_energy(xdata)
    )

    gridded = system.to_discrete(npts=3).phases["FCC"]
    assert np.all(gridded.xdata.sum(axis=1) <= 1.0)
    assert gridded.xdata.shape[1] == 2


def test_isothermal_quadratic_fit_without_equilibrium():
    xdata = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.1],
            [0.3, 0.1],
            [0.1, 0.2],
            [0.2, 0.2],
            [0.3, 0.2],
            [0.1, 0.3],
            [0.2, 0.3],
            [0.3, 0.3],
        ]
    )
    source = IsothermalQuadraticPhase(
        "FCC",
        A=[[4.0, 1.0], [1.0, 6.0]],
        B=[2.0, -3.0],
        C=-10.0,
        c_ref=[0.2, 0.2],
    ).discretize(xdata)
    fitted = IsothermalQuadraticPhase(
        "placeholder", A=np.zeros((2, 2)), B=np.zeros(2), C=0, c_ref=[0, 0]
    )

    fitted.fit_phase(source)

    np.testing.assert_allclose(fitted.free_energy(xdata), source.Gdata)


def test_isothermal_quadratic_fit_warns_when_underdetermined():
    import pytest

    source = IsothermalDiscretePhase("BCC", [0.2, 0.8], [1.0, 2.0], T=500.0)
    fitted = IsothermalQuadraticPhase(
        "placeholder", A=[[0.0]], B=[0.0], C=0, c_ref=[0.0]
    )

    with pytest.warns(UserWarning, match="underdetermined"):
        fitted.fit_phase(source)


def test_isothermal_quadratic_fit_with_equilibrium_and_system():
    potential = np.array([5.0, -3.0])
    intercept = 10.0
    xdata = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.2, 0.3], [0.8, 0.4], [0.5, 0.8]]
    )
    Gdata = xdata @ potential + intercept + 2.0 * xdata[:, 1] ** 2
    discrete_system = IsothermalDiscreteSystem(
        components=["B", "C"], solution_component="A"
    )
    discrete_system.add_phase("FCC", xdata, Gdata, T=800.0)
    discrete_system.add_phase(
        "BCC", [[0.5, 1.0]], [potential @ [0.5, 1.0] + intercept], T=800.0
    )
    equilibrium = discrete_system.get_equilibrium([0.5, 0.3])

    fitted_potential, fitted_intercept = equilibrium.get_tangent_plane()
    np.testing.assert_allclose(fitted_potential, potential)
    np.testing.assert_allclose(fitted_intercept, intercept)
    assert len(equilibrium.points) == 3

    fitted_phase = IsothermalQuadraticPhase(
        "placeholder", A=np.zeros((2, 2)), B=np.zeros(2), C=0, c_ref=[0, 0]
    )
    fitted_phase.fit_phase(discrete_system.phases["FCC"], equilibrium)
    np.testing.assert_allclose(
        fitted_phase.A, [[0.0, 0.0], [0.0, 4.0]], atol=1e-12
    )
    np.testing.assert_allclose(fitted_phase.B, potential)
    np.testing.assert_allclose(
        fitted_phase.free_energy(xdata), discrete_system.phases["FCC"].Gdata
    )
    np.testing.assert_array_equal(
        discrete_system.phases["FCC"].Gdata, Gdata
    )

    liquid_x = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.1],
            [0.3, 0.1],
            [0.1, 0.2],
            [0.2, 0.2],
            [0.3, 0.2],
            [0.1, 0.3],
            [0.2, 0.3],
            [0.3, 0.3],
        ]
    )
    liquid_ref = np.array([0.2, 0.2])
    liquid_A = np.array([[2.0, 0.2], [0.2, 4.0]])
    liquid_delta = liquid_x - liquid_ref
    liquid_residual = (
        3.0
        + 0.5
        * np.einsum("ni,ij,nj->n", liquid_delta, liquid_A, liquid_delta)
    )
    discrete_system.add_phase(
        "LIQUID",
        liquid_x,
        liquid_x @ potential + intercept + liquid_residual,
        T=800.0,
    )

    fitted_system = IsothermalQuadraticSystem().from_discrete(
        discrete_system, equilibrium=equilibrium
    )
    assert set(fitted_system.phases) == {"FCC", "BCC", "LIQUID"}
    np.testing.assert_allclose(fitted_system.phases["FCC"].B, potential)
    np.testing.assert_allclose(fitted_system.phases["BCC"].B, potential)
    np.testing.assert_allclose(
        fitted_system.phases["BCC"].A,
        5e8 * np.array([[2.0, 1.0], [1.0, 2.0]]),
    )
    np.testing.assert_allclose(fitted_system.phases["LIQUID"].A, liquid_A)
    np.testing.assert_allclose(fitted_system.phases["LIQUID"].B, potential)
    np.testing.assert_allclose(fitted_system.phases["LIQUID"].C, 13.4)
    np.testing.assert_allclose(fitted_system.phases["LIQUID"].c_ref, liquid_ref)
    np.testing.assert_allclose(
        fitted_system.phases["LIQUID"].free_energy(liquid_x),
        discrete_system.phases["LIQUID"].Gdata,
    )


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
    assert any(name == "FCC" for name, _, _ in eq)
    assert all(len(comp) == 2 and np.isfinite(G) for _, comp, G in eq)


def test_get_equilibrium_keeps_repeated_phase_occurrences():
    system = IsothermalDiscreteSystem(components=["B", "C"], solution_component="A")
    system.add_phase("FCC", [[0.0, 0.0], [1.0, 0.0]], [0.0, 0.0])
    system.add_phase("BCC", [[0.0, 1.0]], [0.0])
    system._is_equilibrium_system = True

    equilibrium = system.get_equilibrium([0.3, 0.3])

    assert [name for name, _, _ in equilibrium] == ["FCC", "FCC", "BCC"]
    np.testing.assert_allclose(equilibrium[0][1], [0.0, 0.0])
    np.testing.assert_allclose(equilibrium[1][1], [1.0, 0.0])
    np.testing.assert_allclose(equilibrium[2][1], [0.0, 1.0])
    np.testing.assert_allclose([energy for _, _, energy in equilibrium], 0.0)

    near_equilibrium = system.resample_near_equilibrium(
        [0.3, 0.3], xdist=0.1, npts=5
    )
    assert set(near_equilibrium.phases) == {"FCC", "BCC"}
    np.testing.assert_array_equal(
        near_equilibrium.phases["FCC"].xdata, [[0.0, 0.0], [1.0, 0.0]]
    )


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
