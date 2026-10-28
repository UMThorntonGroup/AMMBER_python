from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.interpolate import LinearNDInterpolator
import numpy as np
from pycalphad import calculate


def _as_composition_array(xdata, Gdata):
    """
    Coerce composition and energy data to shapes (npts, n_indep) and (npts,).

    A 1D xdata is treated as (npts, 1) when it matches Gdata length, otherwise
    as a single composition point (1, n_indep) when Gdata is scalar/length-1.
    """
    xdata = np.asarray(xdata, dtype=float)
    Gdata = np.asarray(Gdata, dtype=float).ravel()
    if xdata.ndim == 1:
        if Gdata.size == 1 and xdata.size != 1:
            xdata = xdata.reshape(1, -1)
        else:
            xdata = xdata.reshape(-1, 1)
    elif xdata.ndim != 2:
        raise ValueError("xdata must be 1D or 2D (npts, n_indep)")
    if xdata.shape[0] != Gdata.shape[0]:
        raise ValueError(
            f"xdata and Gdata length mismatch: {xdata.shape[0]} vs {Gdata.shape[0]}"
        )
    return xdata, Gdata


def _rows_in(points, reference, tol=1e-8):
    """Boolean mask of rows in points that match any row in reference within tol."""
    if points.size == 0 or reference.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    diffs = np.abs(points[:, None, :] - reference[None, :, :])
    return np.any(np.all(diffs < tol, axis=2), axis=1)


def _point_in_simplex(x, simplex_vertices, tol=1e-10):
    """
    Test whether composition x lies in the simplex spanned by simplex_vertices.

    Parameters
    ----------
    x : array (n_indep,)
    simplex_vertices : array (n_indep + 1, n_indep)

    Returns
    -------
    inside : bool
    lambdas : array or None
        barycentric coordinates if inside
    """
    x = np.asarray(x, dtype=float).ravel()
    verts = np.asarray(simplex_vertices, dtype=float)
    n = x.size
    if verts.shape != (n + 1, n):
        return False, None
    A = (verts[1:] - verts[0]).T
    try:
        lambdas_rest = np.linalg.solve(A, x - verts[0])
    except np.linalg.LinAlgError:
        return False, None
    lambdas = np.concatenate(([1.0 - lambdas_rest.sum()], lambdas_rest))
    if np.all(lambdas >= -tol):
        return True, lambdas
    return False, None


def _simplex_distance(x, xref):
    """Return Euclidean distance including the dependent solvent fraction."""
    delta = np.asarray(x, dtype=float) - np.asarray(xref, dtype=float)
    return np.sqrt(np.sum(delta**2, axis=-1) + np.sum(delta, axis=-1) ** 2)


def get_lower_convex_hull(inputpoints):
    """
    Returns the set of points that lie on the lower convex hull of inputpoints
    with respect to the last coordinate (energy).

    Parameters
    ----------
    inputpoints : numpy array (N, n_indep + 1)
        points in composition-energy space; last column is energy

    Returns
    -------
    numpy array
        unique vertices on the lower convex hull
    """
    inputpoints = np.asarray(inputpoints, dtype=float)
    if inputpoints.ndim != 2:
        raise ValueError("inputpoints must be 2D")
    npts, ndim = inputpoints.shape
    if npts == 0:
        return inputpoints
    if npts <= ndim:
        return np.unique(inputpoints, axis=0)

    hull = ConvexHull(inputpoints)
    # Outward normals: equations[:, :-1]; offset in equations[:, -1].
    # Lower envelope in energy has outward normals pointing toward -G.
    lower_mask = hull.equations[:, -2] < 0
    if not np.any(lower_mask):
        # Degenerate / flipped orientation fallback: keep all hull vertices
        return np.unique(inputpoints[hull.vertices], axis=0)

    lower_vertex_inds = np.unique(hull.simplices[lower_mask].ravel())
    lower_points = inputpoints[lower_vertex_inds]
    # Stable ordering: lexicographic on composition then energy
    order = np.lexsort(lower_points.T[::-1])
    return lower_points[order]


def get_lower_convex_hull_facets(inputpoints):
    """
    Returns lower-hull facet vertex coordinates.

    Parameters
    ----------
    inputpoints : numpy array (N, n_indep + 1)

    Returns
    -------
    list of numpy arrays
        each array has shape (n_indep + 1, n_indep + 1) for one lower facet
    """
    inputpoints = np.asarray(inputpoints, dtype=float)
    npts, ndim = inputpoints.shape
    if npts <= ndim:
        return [inputpoints] if npts else []

    hull = ConvexHull(inputpoints)
    lower_mask = hull.equations[:, -2] < 0
    if not np.any(lower_mask):
        return [inputpoints[simplex] for simplex in hull.simplices]
    return [inputpoints[simplex] for simplex in hull.simplices[lower_mask]]


class IsothermalDiscretePhase:
    """
    A single phase at one temperature described by points in
    multicomponent composition-Gibbs free energy space.
    """

    def __init__(self, name, xdata, Gdata, T):
        """
        Constructor. Initializes the phase object from points in
        composition-Gibbs free energy space.

        Parameters
        ----------
        name : string
            name of phase to be added
        xdata : list, array
            compositions, shape (npts, n_indep) or 1D for a single independent
            component / single point
        Gdata : list, array
            Gibbs free energies corresponding to compositions
        T : float
            temperature of this isotherm
        """
        xdata, Gdata = _as_composition_array(xdata, Gdata)
        _, ordering = np.unique(xdata, axis=0, return_index=True)
        ordering = np.sort(ordering)
        self.name = name
        self.xdata = xdata[ordering]
        self.Gdata = Gdata[ordering]
        self.T = T

    @property
    def n_indep(self):
        return self.xdata.shape[1]

    def remove_potential(self, potential, xref=0, Gref=0):
        """
        Removes a potential from the free energy of this phase.
        G = G - (x - xref) · potential - Gref

        Parameters
        ----------
        potential : float or array
            chemical potential(s) for the independent components
        xref : float or array
            composition corresponding to Gref
        Gref : float
            an energy that will be mapped to zero at composition xref,
            after removing the potential
        """
        potential = np.asarray(potential, dtype=float).ravel()
        xref = np.asarray(xref, dtype=float).ravel()
        if xref.size == 1:
            xref = np.full(self.n_indep, float(xref))
        if potential.size == 1:
            potential = np.full(self.n_indep, float(potential))
        self.Gdata = self.Gdata - (self.xdata - xref) @ potential - Gref

    def add_potential(self, potential, xref=0, Gref=0):
        """
        Adds a potential to the free energy of this phase.
        G = G + (x - xref) · potential + Gref

        Parameters
        ----------
        potential : float or array
            chemical potential(s) for the independent components
        xref : float or array
            composition corresponding to Gref
        Gref : float
            an energy that will be mapped to zero at composition xref,
            after adding the potential
        """
        potential = np.asarray(potential, dtype=float).ravel()
        xref = np.asarray(xref, dtype=float).ravel()
        if xref.size == 1:
            xref = np.full(self.n_indep, float(xref))
        if potential.size == 1:
            potential = np.full(self.n_indep, float(potential))
        self.Gdata = self.Gdata + (self.xdata - xref) @ potential + Gref

    def resample(self, xpoints):
        """
        Returns a new IsothermalDiscretePhase by interpolating onto xpoints
        (LinearNDInterpolator).

        Parameters
        ----------
        xpoints : list, array
            compositions to be resampled over, shape (npts, n_indep)

        Returns
        -------
        IsothermalDiscretePhase
        """
        xpoints = np.asarray(xpoints, dtype=float)
        if xpoints.ndim == 1:
            if self.n_indep == 1:
                xpoints = xpoints.reshape(-1, 1)
            else:
                xpoints = xpoints.reshape(1, -1)
        mins = self.xdata.min(axis=0)
        maxs = self.xdata.max(axis=0)
        mask = np.all((xpoints >= mins) & (xpoints <= maxs), axis=1)
        newx = xpoints[mask]
        min_pts = self.n_indep + 1
        if newx.shape[0] >= min_pts and self.xdata.shape[0] >= min_pts:
            interp = LinearNDInterpolator(self.xdata, self.Gdata)
            newG = interp(newx)
            valid = np.isfinite(newG)
            if np.count_nonzero(valid) >= min_pts:
                return IsothermalDiscretePhase(self.name, newx[valid], newG[valid], self.T)
        return self

    def resample_near_xpoint(self, x, xdist=0.04, npts=21):
        """
        Resample this phase within a solvent-invariant distance of x.

        Parameters
        ----------
        x : list, array
            Composition of the independent components, shape (n_indep,) or
            (1, n_indep).
        xdist : float
            Radius in full mole-fraction space, including the solvent.
        npts : positive integer
            Number of grid points along each independent composition axis.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n_indep:
            raise ValueError(f"x must contain {self.n_indep} independent components")
        if xdist < 0:
            raise ValueError("xdist must be non-negative")

        extent = xdist / np.sqrt(2)
        axes = [np.linspace(value - extent, value + extent, npts) for value in x]
        grid = np.meshgrid(*axes, indexing="ij")
        xpoints = np.stack([axis.ravel() for axis in grid], axis=-1)
        in_simplex = np.all(xpoints >= 0, axis=1) & (xpoints.sum(axis=1) <= 1)
        in_ball = _simplex_distance(xpoints, x) <= xdist
        return self.resample(xpoints[in_simplex & in_ball])

    def free_energy(self, x):
        """
        Returns the estimated free energy by interpolating at composition x
        (LinearNDInterpolator).

        Parameters
        ----------
        x : float, list, or array
            composition to be sampled; shape (n_indep,) or (npts, n_indep)

        Returns
        -------
        float or array
            free energy at composition x
        """
        x = np.asarray(x, dtype=float)
        single = x.ndim == 0 or (x.ndim == 1 and self.n_indep > 1) or (
            x.ndim == 1 and self.n_indep == 1 and x.size == 1
        )
        if self.n_indep == 1:
            x_eval = np.atleast_1d(x).reshape(-1, 1)
            single = np.size(x) == 1
        elif x.ndim == 1:
            x_eval = x.reshape(1, -1)
            single = True
        else:
            x_eval = x
            single = False

        if self.xdata.shape[0] == 1:
            if np.allclose(x_eval, self.xdata[0]):
                vals = np.full(x_eval.shape[0], self.Gdata[0])
                return float(vals[0]) if single else vals
            print(
                "Attempted to evaluate free energy of a compound outside its range. "
                "This will cause errors."
            )
            return None

        if self.xdata.shape[0] < self.n_indep + 1:
            print(
                "Attempted to evaluate free energy with too few points for interpolation. "
                "This will cause errors."
            )
            return None

        interp = LinearNDInterpolator(self.xdata, self.Gdata)
        vals = interp(x_eval)
        if single:
            val = vals.ravel()[0]
            if not np.isfinite(val):
                print(
                    "Attempted to evaluate free energy of a compound outside its range. "
                    "This will cause errors."
                )
                return None
            return float(val)
        return vals


class IsothermalDiscreteSystem:
    """
    A set of phases at one temperature described by points in
    multicomponent composition-Gibbs free energy space.
    """

    def __init__(self, components=None, solution_component=""):
        """
        Constructor. Initializes with an empty set of phases.

        Parameters
        ----------
        components : list (string)
            names of the independent composition components
        solution_component : string
            name of the solvent / dependent component
        """
        self.components = list(components) if components is not None else []
        self.solution_component = solution_component
        self.phases = {}
        self.lc_hull = None
        self._lower_facets = None
        self._is_equilibrium_system = False

    def fromTDB(self, db, components, solution_component, temperature, phase_list=None):
        """
        Constructs discrete phases from a pycalphad database.

        Parameters
        ----------
        db : pycalphad database
        components : list (string)
            Element abbreviations for independent components (all-caps)
        solution_component : string
            Element abbreviation for the solvent / dependent component
        temperature : float
            isotherm temperature
        phase_list : list (string)
            If specified, only listed phases will be constructed; otherwise,
            all available phases will be constructed.
        """
        components = list(components)
        elements = components + [solution_component, "VA"]
        self.components = components
        self.solution_component = solution_component
        if phase_list is None:
            phase_list = list(db.phases.keys())
            cal = calculate(db, elements, phase_list, P=101325, T=temperature, output="GM", pdens=1001)
            phase_list = set(cal.Phase.data[0][0][0])
        for phase_name in phase_list:
            result = calculate(db, elements, phase_name, P=101325, T=temperature, output="GM", pdens=1001)
            x_cols = [result.X.sel(component=c).data[0][0][0] for c in components]
            xdata = np.stack(x_cols, axis=-1)
            self.phases[phase_name] = IsothermalDiscretePhase(
                phase_name, xdata, result.GM.data[0][0][0], T=temperature
            )

    def add_phase(self, name, xdata, Gdata, T=0.0):
        """
        Constructs a new IsothermalDiscretePhase from points in
        composition-Gibbs free energy space.

        Parameters
        ----------
        name : string
            name of phase to be added
        xdata : list, array
            compositions, shape (npts, n_indep)
        Gdata : list, array
            Gibbs free energies corresponding to compositions
        T : float
            temperature of this isotherm
        """
        self.phases[name] = IsothermalDiscretePhase(name, xdata, Gdata, T)

    def _all_points(self):
        xdata_list = []
        Gdata_list = []
        for phase in self.phases:
            xdata_list.append(self.phases[phase].xdata)
            Gdata_list.append(self.phases[phase].Gdata)
        x_all = np.vstack(tuple(xdata_list))
        G_all = np.concatenate(tuple(Gdata_list))
        return np.hstack((x_all, G_all[:, None]))

    def get_lc_hull(self, recalculate=False):
        """
        Returns the set of composition-G points on the lower convex hull of
        all points in all phases.

        Parameters
        ----------
        recalculate : bool
            if true, forces recalculation

        Returns
        -------
        numpy array
            points on the lower convex hull of all phases
        """
        if self.lc_hull is None or recalculate:
            allpoints = self._all_points()
            self.lc_hull = get_lower_convex_hull(allpoints)
            self._lower_facets = get_lower_convex_hull_facets(allpoints)
        return self.lc_hull

    def get_equilibrium_system(self, miscibility_gap_threshold=0.1):
        """
        Returns the phases that may be present at any composition at equilibrium.

        Parameters
        ----------
        miscibility_gap_threshold : float
            (optional) Composition-space distance threshold used to separate
            miscibility-gap branches of the same phase

        Returns
        -------
        IsothermalDiscreteSystem
        """
        lc_hull = self.get_lc_hull()
        equilibrium_system = IsothermalDiscreteSystem(
            components=self.components, solution_component=self.solution_component
        )
        equilibrium_system._is_equilibrium_system = True

        for phase in self.phases:
            phase_pts = np.hstack(
                (self.phases[phase].xdata, self.phases[phase].Gdata[:, None])
            )
            mask = _rows_in(phase_pts, lc_hull)
            if np.count_nonzero(mask) > 0:
                equilibrium_system.phases[phase] = IsothermalDiscretePhase(
                    phase,
                    self.phases[phase].xdata[mask],
                    self.phases[phase].Gdata[mask],
                    self.phases[phase].T,
                )

        # Split miscibility-gap branches by connected components in composition space
        phases_to_split = list(equilibrium_system.phases.keys())
        for phase in phases_to_split:
            phase_obj = equilibrium_system.phases[phase]
            xdata = phase_obj.xdata
            Gdata = phase_obj.Gdata
            n = xdata.shape[0]
            if n <= 1:
                continue
            dist = cdist(xdata, xdata)
            n_comp, labels = connected_components(dist <= miscibility_gap_threshold)
            if n_comp <= 1:
                continue
            equilibrium_system.phases.pop(phase)
            for i in range(n_comp):
                mask = labels == i
                n_pts = np.count_nonzero(mask)
                new_name = phase + "_" + str(i)
                if n_pts >= 1:
                    if n_pts == 1:
                        print(
                            "phase "
                            + new_name
                            + " is represented by a single point, a finer "
                            "discretization may be needed"
                        )
                    equilibrium_system.phases[new_name] = IsothermalDiscretePhase(
                        new_name, xdata[mask], Gdata[mask], phase_obj.T
                    )
        return equilibrium_system

    def get_equilibrium(self, x):
        """
        Returns the composition(s) and free energy of the phase(s) present at
        equilibrium at composition x.

        Parameters
        ----------
        x : list, array
            composition of the independent components, shape (n_indep,)

        Returns
        -------
        dict {string phase_name : (array compositions, float G)}
            composition and free energy of each phase present at equilibrium at x.
            For a single-phase region, G is the hull energy at x.
        """
        x = np.asarray(x, dtype=float).ravel()
        self.get_lc_hull()
        if self._is_equilibrium_system:
            equilibrium_system = self
        else:
            equilibrium_system = self.get_equilibrium_system()

        facets = self._lower_facets
        if facets is None:
            facets = get_lower_convex_hull_facets(self._all_points())

        containing = None
        lambdas = None
        for facet in facets:
            comps = facet[:, :-1]
            inside, lam = _point_in_simplex(x, comps)
            if inside:
                containing = facet
                lambdas = lam
                break
        if containing is None:
            return {}

        # Map facet vertices to phases that own them
        vertex_phase_comp = []
        for vertex in containing:
            for phase_name, phase_obj in equilibrium_system.phases.items():
                phase_pts = np.hstack((phase_obj.xdata, phase_obj.Gdata[:, None]))
                if _rows_in(vertex.reshape(1, -1), phase_pts)[0]:
                    vertex_phase_comp.append(
                        (phase_name, vertex[:-1].copy(), float(vertex[-1]))
                    )
                    break

        if not vertex_phase_comp:
            return {}

        phase_names = [p for p, _, _ in vertex_phase_comp]
        if len(set(phase_names)) == 1:
            G_x = float(lambdas @ containing[:, -1])
            return {phase_names[0]: (x.copy(), G_x)}

        equilibrium = {}
        for phase_name, comp, G in vertex_phase_comp:
            if phase_name not in equilibrium:
                equilibrium[phase_name] = (comp, G)
        return equilibrium

    def resample_near_equilibrium(self, x, xdist=0.001, npts=21):
        """
        Resample phases near their equilibrium compositions at x.

        Parameters
        ----------
        x : list, array
            Composition of the independent components.
        xdist : float
            Radius in full mole-fraction space around each equilibrium
            composition.
        npts : positive integer
            Number of grid points along each independent composition axis.
        """
        equilibrium = self.get_equilibrium(x)
        new_system = IsothermalDiscreteSystem(
            components=self.components,
            solution_component=self.solution_component,
        )
        for phase, (xeq, _) in equilibrium.items():
            if phase in self.phases:
                source_phase = phase
            elif phase[:-2] in self.phases:
                source_phase = phase[:-2]
            else:
                continue
            new_system.phases[phase] = self.phases[source_phase].resample_near_xpoint(
                xeq, xdist=xdist, npts=npts
            )
        return new_system
