from itertools import combinations

import numpy as np
import pytest
import sympy as sp

from pchr.core import (atom_energy, boundary, cochains, local_signed_map,
                       persistent_operator, response, rips, schur_operator,
                       signed_transport)


@pytest.fixture
def cloud():
    rng = np.random.default_rng(3821)
    return rng.normal(size=(8,3)), np.column_stack(([6,7,8,9,6,16,8,6], [3,2,1,1,4,2,1,2], [1,1,0,0,0,0,1,2]))


def test_boundary_squared(cloud):
    x, _ = cloud
    k = rips(x, 3)
    for q in (2,3):
        assert np.array_equal(boundary(k[q-2],k[q-1]) @ boundary(k[q-1],k[q]),
                              np.zeros((len(k[q-2]),len(k[q]))))


def test_full_simplex_collapse(cloud):
    x, f = cloud
    op = persistent_operator(x, 100, 100)
    np.testing.assert_allclose(op.matrix, len(x)*np.eye(len(op.small[2])), atol=1e-12)
    chi,b = cochains(x,op.small[2],f)
    out = response(op,chi,b,.3)
    np.testing.assert_allclose(out['u'],chi/(len(x)+.3),atol=1e-13)


def test_persistent_filling_tetrahedron():
    x = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    # Boundary-only K and filled L constructed directly to isolate persistence.
    triangles = list(combinations(range(4),3))
    edges = list(combinations(range(4),2))
    b2 = boundary(edges,triangles)
    b3 = boundary(triangles,[(0,1,2,3)])
    assert sp.Matrix(b2).rank() == 3
    assert sp.Matrix(b2.T@b2).nullspace()
    np.testing.assert_allclose(b2.T@b2+b3@b3.T,4*np.eye(4))


@pytest.mark.parametrize('seed',range(12))
def test_exact_persistent_betti(seed):
    rng=np.random.default_rng(seed)
    x=rng.integers(-5,6,size=(7,3)).astype(float)
    op=persistent_operator(x,6.3,8.2)
    # Exact rational ranks are independent of floating SVD null-space ranks.
    beta=len(op.small[2])-sp.Matrix(op.b2).rank()-sp.Matrix(op.b3).rank()+sp.Matrix(op.outside).rank()
    numerical=int(np.sum(np.linalg.eigvalsh(op.matrix)<1e-8))
    assert numerical == beta
    np.testing.assert_allclose(op.matrix,schur_operator(op),atol=2e-10)


def test_symmetry_laws_and_orientation(cloud):
    x,f=cloud
    op=persistent_operator(x,2.1,2.7)
    chi,b=cochains(x,op.small[2],f)
    out=response(op,chi,b)
    rng=np.random.default_rng(481)
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(q)<0: q[:,0]*=-1
    for matrix in (q,q@np.diag([-1,1,1])):
        sign=round(np.linalg.det(matrix))
        y=x@matrix.T + np.array([10,-4,8])
        op2=persistent_operator(y,2.1,2.7)
        c2,b2=cochains(y,op2.small[2],f)
        out2=response(op2,c2,b2)
        np.testing.assert_allclose(c2,sign*chi,atol=2e-13)
        np.testing.assert_allclose(b2,b,atol=2e-13)
        np.testing.assert_allclose(out2['signed'],sign*out['signed'],atol=2e-12)
        np.testing.assert_allclose(out2['energy'],out['energy'],atol=2e-12)
    for _ in range(10):
        p=rng.permutation(len(x))
        op2=persistent_operator(x[p],2.1,2.7)
        c2,b2=cochains(x[p],op2.small[2],f[p])
        t=signed_transport(op.small[2],op2.small[2],p)
        np.testing.assert_allclose(c2,t@chi,atol=1e-13)
        np.testing.assert_allclose(b2,t@b,atol=1e-13)
        np.testing.assert_allclose(op2.matrix,t@op.matrix@t.T,atol=2e-12)
        out2=response(op2,c2,b2)
        np.testing.assert_allclose(out2['signed'],out['signed'],atol=2e-12)


def test_orientation_free_atom_maps(cloud):
    x,f=cloud; op=persistent_operator(x,2.1,2.7)
    chi,b=cochains(x,op.small[2],f); out=response(op,chi,b)
    energy=atom_energy(out['u'],op.small[2],len(x))
    signed=local_signed_map(out['u'],b,op.small[2],len(x))
    np.testing.assert_allclose(energy.sum(),np.sum(out['u']**2))
    np.testing.assert_allclose(signed.sum(),out['signed'][0,0])


def test_empty_space_and_invalid_parameters():
    x=np.zeros((1,3)); f=np.array([[6,0,4]])
    op=persistent_operator(x,0,1)
    chi,b=cochains(x,op.small[2],f)
    assert response(op,chi,b)['signed'].shape==(6,3)
    with pytest.raises(ValueError): persistent_operator(x,2,1)
    with pytest.raises(ValueError): response(op,chi,b,0)
    with pytest.raises(ValueError): rips(x,-1)


def test_resolvent_and_residual_bounds():
    rng=np.random.default_rng(432)
    for _ in range(20):
        a=rng.normal(size=(7,7)); c=rng.normal(size=(7,7))
        l=a.T@a; lp=c.T@c; chi=rng.normal(size=7); chip=rng.normal(size=7); tau=.4
        u=np.linalg.solve(l+tau*np.eye(7),chi); up=np.linalg.solve(lp+tau*np.eye(7),chip)
        bound=np.linalg.norm(chi-chip)/tau+np.linalg.norm(l-lp,2)*np.linalg.norm(chip)/tau**2
        assert np.linalg.norm(u-up)<=bound+1e-12
        approx=u+rng.normal(size=7)*1e-6
        residual=(l+tau*np.eye(7))@approx-chi
        assert np.linalg.norm(u-approx)<=np.linalg.norm(residual)/tau+1e-12
