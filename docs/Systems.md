# Multicomponent isothermal systems

`ammber.Systems` provides discrete and quadratic representations of phases
whose Gibbs free energy depends on one or more independent composition
coordinates at a fixed temperature. The dependent (solvent) fraction is not
stored; for a ternary A-B-C system with independent components B and C,
`x = [x_B, x_C]` and `x_A = 1 - x_B - x_C`.

The primary classes are:

- `IsothermalDiscretePhase` and `IsothermalDiscreteSystem` for tabulated
  composition-energy data and lower-convex-hull equilibrium calculations.
- `IsothermalQuadraticPhase` and `IsothermalQuadraticSystem` for fitting and
  evaluating quadratic free-energy models.
- `IsothermalEquilibrium` for the vertices of a facet containing a queried
  overall composition, and for calculating that facet's affine tangent plane.

## Discrete phases and systems

Create a system with the independent component names and the dependent solvent
name, then add a phase with one energy for each composition:

```python
import numpy as np

from ammber.Systems import IsothermalDiscreteSystem

system = IsothermalDiscreteSystem(
    components=["B", "C"],
    solution_component="A",
)

xdata = np.array([
    [0.10, 0.10],
    [0.20, 0.10],
    [0.10, 0.20],
    [0.20, 0.20],
])
Gdata = np.array([-100.0, -98.0, -97.0, -94.0])
system.add_phase("FCC", xdata, Gdata, T=1000.0)
```

`xdata` has shape `(n_points, n_independent_components)` and `Gdata` has shape
`(n_points,)`. The constructor accepts a one-dimensional composition array as
multiple points for a one-component system, or as one multicomponent point
when there is only one energy value. Duplicate compositions are removed,
keeping the first corresponding energy.

Alternatively, `IsothermalDiscreteSystem.fromTDB(db, components,
solution_component, temperature, phase_list=None)` samples phases from a
pycalphad database. If `phase_list` is omitted, the system determines which
phases are present at the requested temperature.

### Lower hull and equilibrium

The system computes its lower convex hull lazily:

```python
hull_points = system.get_lc_hull()
equilibrium = system.get_equilibrium([0.15, 0.15])
```

`get_lc_hull()` returns an array whose final column is Gibbs free energy.
`get_equilibrium(x)` returns an `IsothermalEquilibrium`. Its `points` are
always the labeled vertices of the containing facet, each represented as
`(phase_name, composition, G)`. A phase name can occur more than once if that
phase supplies multiple facet vertices. If `x` is outside the lower-hull
composition domain, `points` is empty. The query composition is available as
`equilibrium.query`.

`IsothermalEquilibrium` is iterable and indexable:

```python
for phase_name, composition, energy in equilibrium:
    print(phase_name, composition, energy)
```

When the facet vertices span a unique affine plane,
`get_tangent_plane()` returns `(potential, intercept)` such that
`G = composition @ potential + intercept` on that facet. It raises
`ValueError` if the points do not uniquely define a plane (including an empty
result). In a two-independent-component system, a full facet has three
non-collinear vertices.

`get_equilibrium_system(miscibility_gap_threshold=0.1)` returns the phases
that contribute vertices to the lower hull, splitting disconnected
composition branches when they are farther apart than the threshold.
`resample_near_equilibrium(x, xdist=0.001, npts=21)` samples source phases in a
local neighborhood of equilibrium compositions. The distance is measured in
full composition space, including the dependent solvent fraction.

### Adding or removing a linear potential

Both discrete phases and systems can add or remove a linear term:

```python
system.add_potential(potential=[2.0, -1.0], xref=[0.0, 0.0], Gref=5.0)
system.remove_potential(potential=[2.0, -1.0], xref=[0.0, 0.0], Gref=5.0)
```

The term added or removed is
`(x - xref) @ potential + Gref`. A scalar potential or reference composition
is broadcast to all independent components.

## Quadratic phases

`IsothermalQuadraticPhase` uses the centered multicomponent form

```text
G(x) = 0.5 * (x - c_ref).T @ A @ (x - c_ref)
       + B.T @ (x - c_ref) + C
```

`A` must be a symmetric `n x n` matrix, `B` and `c_ref` must each have `n`
entries, and `C` is the energy at `c_ref`. Positive definiteness is not
required.

```python
from ammber.Systems import IsothermalQuadraticPhase

phase = IsothermalQuadraticPhase(
    "FCC",
    A=[[8.0, 1.0], [1.0, 6.0]],
    B=[2.0, -3.0],
    C=-100.0,
    c_ref=[0.20, 0.15],
    T=1000.0,
)

energy_at_x = phase.free_energy([0.25, 0.20])
energies = phase.free_energy([[0.20, 0.15], [0.25, 0.20]])
```

For one independent component, a one-dimensional input to `free_energy` is
interpreted as a vector of compositions; for multiple independent components,
a one-dimensional input is one composition. To apply a linear potential to a
quadratic phase, use its `add_potential` or `remove_potential` method.

## Fitting quadratic phases

`IsothermalQuadraticPhase.fit_phase(discrete_phase, equilibrium=None)` fits a
discrete phase in place and returns the quadratic phase object.

- A **single-point phase** is treated as a point compound. Its reference and
  energy are the point's composition and energy. Its curvature is fixed to
  `5e8 * (I + ones)`, giving diagonal entries `1e9` and off-diagonal entries
  `5e8`. If an equilibrium is provided, its tangent potential is used as `B`.
- With **no equilibrium**, the fit uses `c_ref = mean(xdata)` and least squares
  to determine all symmetric quadratic coefficients, the linear vector `B`,
  and `C`.
- With an **equilibrium**, the facet's tangent plane is removed before fitting
  the quadratic curvature. If the discrete phase contains one or more facet
  vertices, `c_ref` is their mean and the residual at that reference is fixed
  to zero. Otherwise, `c_ref` is the composition with the minimum residual
  energy and the residual energy there is fixed to the minimum. The tangent
  plane is then restored.

Underdetermined fits emit `UserWarning`. A warning is also emitted when there
are enough samples in count but their composition geometry does not identify
all fitted coefficients. A fit can still be returned in those cases, but
coefficients may not be uniquely determined; use adequate composition
coverage for the model dimension.

The equilibrium object should come from the discrete system and query of
interest. The vertices provide both the tangent plane and the phase points
used to select the corresponding fit branch:

```python
from ammber.Systems import IsothermalQuadraticSystem

discrete_system = system
query = [0.15, 0.15]
equilibrium = discrete_system.get_equilibrium(query)

quadratic_system = IsothermalQuadraticSystem(
    components=discrete_system.components,
    solution_component=discrete_system.solution_component,
)
quadratic_system.from_discrete(discrete_system, equilibrium=equilibrium)
```

`IsothermalQuadraticSystem.from_discrete(discrete_system,
equilibrium=None)` fits every phase and replaces the system's phase mapping.
It copies the component metadata and temperature from the discrete phases.
Pass `None` to fit each solution phase without an equilibrium constraint, or
pass one `IsothermalEquilibrium` result to use its tangent plane for each
phase. This is most appropriate when all phases are being fitted relative to
the same equilibrium facet.

`IsothermalQuadraticSystem.add_phase(name, A, B, C, c_ref)` adds a manually
specified phase. `to_discrete(xdata=None, npts=21)` samples all quadratic
phases into an `IsothermalDiscreteSystem`. Provide `xdata` to choose exact
sample compositions, or omit it to generate a regular grid over the independent
composition simplex with `npts` points along each axis.

## Ternary example

The repository includes a complete discrete ternary equilibrium and
near-equilibrium plotting example:

```bash
python examples/ternary_discrete_equilibrium.py
```

See [`examples/ternary_discrete_equilibrium.py`](../examples/ternary_discrete_equilibrium.py).
