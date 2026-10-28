"""Real, unweighted degree-two persistent Hodge response.

The increasing atom index sets a storage orientation only. Source and probe
cochains both alternate, so the scalar readout is independent of this choice.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import null_space, pinvh, solve


def distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Coordinates must be a finite n by 3 array")
    return np.linalg.norm(x[:, None] - x[None, :], axis=2)


def rips(x: np.ndarray, radius: float) -> dict[int, list[tuple[int, ...]]]:
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("Radius must be finite and nonnegative")
    d = distances(x)
    out = {0: [(i,) for i in range(len(x))]}
    for q in (1, 2, 3):
        out[q] = [s for s in combinations(range(len(x)), q + 1)
                  if all(d[i, j] <= radius for i, j in combinations(s, 2))]
    return out


def boundary(lower: list[tuple], upper: list[tuple]) -> np.ndarray:
    index = {s: k for k, s in enumerate(lower)}
    b = np.zeros((len(lower), len(upper)), dtype=float)
    for j, simplex in enumerate(upper):
        for k in range(len(simplex)):
            b[index[simplex[:k] + simplex[k + 1:]], j] = (-1) ** k
    return b


def signed_transport(old: list[tuple], new: list[tuple], new_to_old: np.ndarray) -> np.ndarray:
    """Map old oriented chains to new coordinates after atom relabelling."""
    index = {s: i for i, s in enumerate(old)}
    t = np.zeros((len(new), len(old)))
    for row, simplex in enumerate(new):
        mapped = tuple(int(new_to_old[i]) for i in simplex)
        inversions = sum(mapped[i] > mapped[j] for i in range(len(mapped))
                         for j in range(i + 1, len(mapped)))
        t[row, index[tuple(sorted(mapped))]] = (-1) ** inversions
    return t


@dataclass
class PersistentOperator:
    matrix: np.ndarray
    small: dict[int, list[tuple]]
    large: dict[int, list[tuple]]
    b2: np.ndarray
    b3: np.ndarray
    inside: np.ndarray
    outside: np.ndarray
    null_basis: np.ndarray
    r: float
    s: float


def persistent_operator(x: np.ndarray, r: float, s: float, rcond: float = 1e-10) -> PersistentOperator:
    """B2^T B2 + A P_ker(D) A^T, using an orthonormal kernel basis."""
    if s < r:
        raise ValueError("Persistent scales require r <= s")
    small, large = rips(x, r), rips(x, s)
    return persistent_from_complexes(small, large, rcond=rcond, r=r, s=s)


def persistent_from_complexes(small: dict, large: dict, rcond: float = 1e-10,
                             r: float = 0, s: float = 1) -> PersistentOperator:
    """Construct the same operator for an explicitly supplied simplicial pair."""
    for q in range(4):
        if not set(small[q]).issubset(large[q]):
            raise ValueError("Small complex must be a subcomplex of large")
        for complex_ in (small, large):
            if any(tuple(sorted(simplex)) != simplex for simplex in complex_[q]):
                raise ValueError("Input simplices must use increasing orientation")
            if q and any(face not in complex_[q-1] for simplex in complex_[q]
                         for face in combinations(simplex,q)):
                raise ValueError("Input must be closed under taking faces")
    b2 = boundary(small[1], small[2])
    b3 = boundary(large[2], large[3])
    row_lookup = {t: i for i, t in enumerate(large[2])}
    inside_rows = [row_lookup[t] for t in small[2]]
    outside_rows = [i for i, t in enumerate(large[2]) if t not in set(small[2])]
    a = b3[inside_rows, :]
    d = b3[outside_rows, :]
    if d.shape[1] == 0:
        n = np.empty((0, 0))
    elif d.shape[0] == 0:
        n = np.eye(d.shape[1])
    else:
        n = null_space(d, rcond=rcond)
    c = a @ n
    lap = b2.T @ b2 + c @ c.T
    lap = (lap + lap.T) / 2
    return PersistentOperator(lap, small, large, b2, b3, a, d, n, r, s)


def schur_operator(op: PersistentOperator, rcond: float = 1e-10) -> np.ndarray:
    a, d = op.inside, op.outside
    result = op.b2.T @ op.b2 + a @ a.T
    if d.shape[0] and d.shape[1]:
        adt = a @ d.T
        result -= adt @ pinvh(d @ d.T, rtol=rcond) @ adt.T
    return (result + result.T) / 2


def cochains(x: np.ndarray, triangles: list[tuple], atom_features: np.ndarray,
             bandwidth: float = 2.4, length_unit: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Three odd sources and six even alternating probes.

    Columns of atom_features: atomic number, heavy-atom degree, total H count.
    No CIP labels, optical signs, or atom ranks enter the features.
    The quadruple kernel uses all six squared interatomic distances.
    """
    if bandwidth <= 0 or length_unit <= 0:
        raise ValueError("Bandwidth and length unit must be positive")
    d = distances(x)
    af = np.asarray(atom_features, dtype=float)
    if af.shape != (len(x), 3) or not np.isfinite(af).all():
        raise ValueError("atom_features must have shape n by 3")
    z = af[:, 0] / 8.0
    degree, hydrogens = af[:, 1] / 4.0, af[:, 2] / 4.0
    radial = np.exp(-d*d / (2 * bandwidth**2)) @ z / max(len(x), 1)
    fields = np.column_stack((z, degree, hydrogens, radial))
    weights = np.column_stack((np.ones(len(x)), z, 1 + degree))
    sources = np.zeros((len(triangles), 3))
    probes = np.zeros((len(triangles), 6))
    field_pairs = list(combinations(range(4), 2))
    for row, (i, j, k) in enumerate(triangles):
        tri_sq = d[i,j]**2 + d[i,k]**2 + d[j,k]**2
        for ell in range(len(x)):
            if ell in (i, j, k):
                continue
            sq = tri_sq + d[i,ell]**2 + d[j,ell]**2 + d[k,ell]**2
            kernel = np.exp(-sq / (2 * bandwidth**2))
            volume = np.linalg.det(np.column_stack((x[j]-x[i], x[k]-x[i], x[ell]-x[i])))
            sources[row] += weights[ell] * kernel * volume / length_unit**3
        tri_kernel = np.exp(-tri_sq / (2 * bandwidth**2))
        for col, (a, b) in enumerate(field_pairs):
            f, g = fields[:, a], fields[:, b]
            probes[row,col] = tri_kernel * ((f[j]-f[i])*(g[k]-g[i])
                                           - (f[k]-f[i])*(g[j]-g[i]))
    return sources, probes


def response(op: PersistentOperator, chi: np.ndarray, probes: np.ndarray,
             tau: float = 0.3) -> dict[str, np.ndarray | float]:
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    m = op.matrix.shape[0]
    if chi.shape[0] != m or probes.shape[0] != m:
        raise ValueError("Source and probe dimensions must match chain space")
    if m == 0:
        return {"u": chi.copy(), "signed": np.zeros((probes.shape[1], chi.shape[1])),
                "energy": np.zeros(chi.shape[1]), "residual": 0.0,
                "residual_bound": 0.0, "eigenvalues": np.empty(0)}
    matrix = op.matrix + tau * np.eye(m)
    u = solve(matrix, chi, assume_a="pos")
    residual = float(np.linalg.norm(matrix @ u - chi))
    return {"u": u, "signed": probes.T @ u, "energy": np.sum(chi*u, axis=0),
            "residual": residual, "residual_bound": residual / tau,
            "eigenvalues": np.linalg.eigvalsh(op.matrix)}


def atom_energy(u: np.ndarray, triangles: list[tuple], n: int) -> np.ndarray:
    """Even, orientation-free atom map; entries sum to ||u||_F^2."""
    scores = np.zeros(n)
    for tri, value in zip(triangles, np.sum(u*u, axis=1)):
        scores[list(tri)] += value / 3
    return scores


def local_signed_map(u: np.ndarray, probes: np.ndarray, triangles: list[tuple], n: int) -> np.ndarray:
    """Odd atom map using first source and first probe; sums to that readout."""
    scores = np.zeros(n)
    for tri, value in zip(triangles, u[:,0] * probes[:,0]):
        scores[list(tri)] += value / 3
    return scores
