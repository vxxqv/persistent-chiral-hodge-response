"""Certified reduced solves and exact integration over filtration events."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve

from .core import cochains_jvp, distances, persistent_operator


def galerkin(laplacian, source, basis, tau=.3):
    """Galerkin solution with a posteriori Euclidean and energy error bounds.

    The columns of basis must be orthonormal; the empty subspace is allowed.
    This is a general symmetric positive semidefinite matrix calculation.
    """
    l, c, q = map(lambda a: np.asarray(a, dtype=float), (laplacian, source, basis))
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if l.ndim != 2 or l.shape[0] != l.shape[1] or c.shape != (len(l),):
        raise ValueError("incompatible matrix and source")
    if q.ndim != 2 or q.shape[0] != len(l):
        raise ValueError("incompatible basis")
    if not all(np.isfinite(a).all() for a in (l,c,q)):
        raise ValueError("inputs must be finite")
    if not np.allclose(l,l.T,atol=1e-12,rtol=1e-12):
        raise ValueError("Laplacian must be symmetric")
    if len(l) and np.linalg.eigvalsh(l)[0] < -1e-10:
        raise ValueError("Laplacian must be positive semidefinite")
    if not np.allclose(q.T@q,np.eye(q.shape[1]),atol=1e-12,rtol=1e-12):
        raise ValueError("basis columns must be orthonormal")
    a=l+tau*np.eye(len(l))
    u=q@solve(q.T@a@q,q.T@c,assume_a='pos') if q.shape[1] else np.zeros_like(c)
    residual=c-a@u
    norm=float(np.linalg.norm(residual))
    return dict(u=u,energy=float(c@u),residual=norm,
                solution_error_bound=norm/tau,energy_error_bound=norm**2/tau)


@dataclass
class ScaleAverage:
    signed: np.ndarray
    derivative: np.ndarray | None
    interior_derivative: np.ndarray | None
    boundary_derivative: np.ndarray | None
    events: np.ndarray
    widths: np.ndarray
    pieces: np.ndarray


def scale_average(x, atom_features, lower=2.2, upper=3.4, gap=.6,
                  tau=.3, bandwidth=2.4, direction=None):
    """Uniform average of b^T (L_{r,r+gap}+tau I)^-1 chi over r.

    Integration is exact in r up to floating arithmetic: the integrand changes
    only at pair distances and pair distances minus gap. The derivative also
    includes motion of these event boundaries. With direction supplied, events
    must be distinct, positive-distance, and strictly away from the endpoints.
    Ties are supported for values, but no derivative is returned through a tie.
    """
    if not np.isfinite([lower,upper,gap,tau,bandwidth]).all():
        raise ValueError("parameters must be finite")
    if lower < 0 or upper <= lower or gap < 0 or tau <= 0 or bandwidth <= 0:
        raise ValueError("invalid scales or regularization")
    x=np.asarray(x,dtype=float)
    d=distances(x)
    v=np.zeros_like(x) if direction is None else np.asarray(direction,dtype=float)
    if v.shape != x.shape or not np.isfinite(v).all():
        raise ValueError("direction must be finite and match coordinates")
    ii,jj=np.triu_indices(len(x),1)
    ds=d[ii,jj]
    if direction is not None and np.any(ds==0):
        raise ValueError("event derivative requires distinct points")
    speeds=np.zeros_like(ds) if direction is None else np.einsum(
        'ij,ij->i',x[ii]-x[jj],v[ii]-v[jj])/ds
    candidates=np.concatenate([ds,ds-gap])
    speeds=np.tile(speeds,2)
    if direction is not None and np.any(np.minimum(abs(candidates-lower),abs(candidates-upper))<1e-9):
        raise ValueError("event at an integration endpoint")
    inside=(candidates>lower)&(candidates<upper)
    order=np.argsort(candidates[inside],kind='stable')
    events=candidates[inside][order]
    speeds=speeds[inside][order]
    if direction is not None and np.any(np.diff(events)<1e-9):
        raise ValueError("event derivative requires distinct events")
    if direction is None:
        events=np.unique(events)
    cuts=np.concatenate([[lower],events,[upper]])
    widths=np.diff(cuts)
    values=[]; derivatives=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        r=(a+b)/2
        op=persistent_operator(x,r,r+gap)
        chi,probe,dchi,dprobe=cochains_jvp(x,op.small[2],atom_features,v,bandwidth)
        matrix=op.matrix+tau*np.eye(len(op.matrix))
        if len(matrix):
            u=solve(matrix,chi,assume_a='pos')
            du=solve(matrix,dchi,assume_a='pos')
        else:
            u,du=chi,dchi
        values.append(probe.T@u)
        derivatives.append(dprobe.T@u+probe.T@du)
    values=np.asarray(values)
    signed=np.einsum('i,ijk->jk',widths,values)/(upper-lower)
    if direction is None:
        return ScaleAverage(signed,None,None,None,events,widths,values)
    interior=np.einsum('i,ijk->jk',widths,np.asarray(derivatives))/(upper-lower)
    # Moving an event right expands its left interval and shrinks its right.
    boundary=-np.einsum('i,ijk->jk',speeds,np.diff(values,axis=0))/(upper-lower)
    return ScaleAverage(signed,interior+boundary,interior,boundary,events,widths,values)
