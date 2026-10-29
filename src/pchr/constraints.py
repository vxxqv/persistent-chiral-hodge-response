"""Distance constraints and necessary second-order compatibility.

The kernel of the rigidity matrix is an infinitesimal space. At a singular
configuration it need not equal velocities of actual constraint curves.
"""
from __future__ import annotations
import numpy as np


def distance_pairs(n, bonds):
    adjacency=[set() for _ in range(n)]
    for i,j,*_ in bonds:
        adjacency[i].add(j);adjacency[j].add(i)
    pairs=set()
    for i in range(n):
        for j in adjacency[i] | set().union(*(adjacency[k] for k in adjacency[i])):
            if i!=j:pairs.add(tuple(sorted((i,j))))
    return sorted(pairs)


def rigidity_matrix(x,pairs):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():
        raise ValueError('finite n by 3 coordinates required')
    result=np.zeros((len(pairs),x.size))
    for row,(i,j) in enumerate(pairs):
        if not (0<=i<len(x) and 0<=j<len(x)) or i==j:
            raise ValueError('distinct in-range pair required')
        result[row,3*i:3*i+3]=x[i]-x[j]
        result[row,3*j:3*j+3]=x[j]-x[i]
    return result


def normal_acceleration(x,pairs,velocity,masses,rcond=1e-12):
    """Return mass-minimal candidate acceleration and its compatibility residual.

    On a constant-rank constraint manifold with tangent velocity, a vanishing
    compatibility residual gives the normal acceleration of an induced-metric
    geodesic. At singular points this is only a necessary second-order test.
    """
    x=np.asarray(x,float);v=np.asarray(velocity,float);m=np.asarray(masses,float)
    if v.shape!=x.shape or m.shape!=(len(x),) or np.any(m<=0) or not np.isfinite(m).all() or not np.isfinite(v).all():
        raise ValueError('matching finite velocity and positive masses required')
    r=rigidity_matrix(x,pairs);inv=1/np.repeat(m,3)
    b=np.array([np.dot(v[i]-v[j],v[i]-v[j]) for i,j in pairs])
    w=(r*inv)@r.T
    a=-inv*(r.T@np.linalg.pinv(w,rcond=rcond,hermitian=True)@b)
    return {'acceleration':a.reshape(x.shape),'first_residual':r@v.ravel(),
            'second_residual':r@a+b,'quadratic_term':b}
