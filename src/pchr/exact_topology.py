"""Exact rational topology oracle; no production numerical functions imported."""
from __future__ import annotations

from itertools import combinations
import sympy as sp


def differential(lower, upper):
    result=sp.zeros(len(lower),len(upper))
    lookup={t:i for i,t in enumerate(lower)}
    for j,simplex in enumerate(upper):
        for k in range(len(simplex)):
            result[lookup[simplex[:k]+simplex[k+1:]],j]=(-1)**k
    return result


def exact_persistent(small, large):
    b2=differential(small[1],small[2])
    b3=differential(large[2],large[3])
    lookup={t:i for i,t in enumerate(large[2])}
    inside=[lookup[t] for t in small[2]]
    outside=[i for i,t in enumerate(large[2]) if t not in small[2]]
    a=b3[inside,:]; d=b3[outside,:]
    ns=d.nullspace()
    z=sp.Matrix.hstack(*ns) if ns else sp.zeros(len(large[3]),0)
    gram=z.T*z
    up=a*z*gram.inv()*z.T*a.T if z.cols else sp.zeros(len(small[2]))
    lap=b2.T*b2+up
    beta=len(small[2])-b2.rank()-b3.rank()+d.rank()
    assert lap.rank()==len(small[2])-beta
    assert b2*differential(small[2],small[3])==sp.zeros(len(small[1]),len(small[3]))
    return lap, beta, z, a, d


def closure(maximal, n):
    result={q:set() for q in range(4)}
    result[0]={(i,) for i in range(n)}
    for simplex in maximal:
        for q in range(min(3,len(simplex)-1)+1):
            result[q].update(combinations(simplex,q+1))
    return {q:sorted(value) for q,value in result.items()}


def octahedral_filling():
    triangles=[tuple(sorted((i,j,k))) for i in (0,1) for j in (2,3) for k in (4,5)]
    small=closure(triangles,7)
    large=closure([tuple(sorted(t+(6,))) for t in triangles],7)
    return small,large


def tetrahedron_subcomplexes():
    optional=[t for q in (1,2,3) for t in combinations(range(4),q+1)]
    for mask in range(1<<len(optional)):
        selected=[s for i,s in enumerate(optional) if mask & (1<<i)]
        k={0:[(i,) for i in range(4)],1:[],2:[],3:[]}
        for s in selected: k[len(s)-1].append(s)
        if all(all(face in k[len(s)-2] for face in combinations(s,len(s)-1))
               for s in selected):
            yield k
