"""Independent numerical checks for the analytical extensions."""
from itertools import combinations

import numpy as np
import pytest

from pchr.analysis import galerkin,scale_average
from pchr.core import cochains,cochains_jvp,distances,rips


def example():
    rng=np.random.default_rng(61723)
    x=rng.normal(size=(7,3))
    f=np.array([[6,3,1],[7,2,1],[8,1,0],[6,2,2],[9,1,0],[16,2,0],[6,4,0]])
    return x,f,rng.normal(size=x.shape)


def literal_cochains(x,triangles,af,h=2.4):
    # Literal determinants and scalar loops are independent of the vectorized
    # cross-product/JVP implementation. Kept only as a scientific oracle.
    d=distances(x); z=af[:,0]/8; deg=af[:,1]/4; hyd=af[:,2]/4
    radial=np.exp(-d*d/(2*h*h))@z/len(x)
    fields=np.column_stack([z,deg,hyd,radial]); w=np.column_stack([np.ones(len(x)),z,1+deg])
    c=np.zeros((len(triangles),3)); b=np.zeros((len(triangles),6))
    for row,t in enumerate(triangles):
        i,j,k=t
        for ell in set(range(len(x)))-set(t):
            s=sum(d[a,bb]**2 for a,bb in combinations((*t,ell),2))
            c[row]+=w[ell]*np.exp(-s/(2*h*h))*np.linalg.det(np.stack([x[j]-x[i],x[k]-x[i],x[ell]-x[i]],axis=1))
        weight=np.exp(-sum(d[a,bb]**2 for a,bb in combinations(t,2))/(2*h*h))
        for col,(a,bb) in enumerate(combinations(range(4),2)):
            b[row,col]=weight*np.linalg.det(np.array([[fields[j,a]-fields[i,a],fields[k,a]-fields[i,a]],
                                                    [fields[j,bb]-fields[i,bb],fields[k,bb]-fields[i,bb]]]))
    return c,b


def test_clique_enumeration_against_all_subsets():
    rng=np.random.default_rng(321)
    for n in (0,1,4,8,13):
        x=rng.normal(size=(n,3)); d=distances(x)
        for r in (0,1.3,3,100):
            actual=rips(x,r)
            for q in (1,2,3):
                expected=[t for t in combinations(range(n),q+1) if all(d[i,j]<=r for i,j in combinations(t,2))]
                assert actual[q]==expected


def test_cochains_literal_reference_and_coordinate_derivative():
    x,f,v=example(); triangles=rips(x,2.7)[2]
    c,b,dc,db=cochains_jvp(x,triangles,f,v)
    cr,br=literal_cochains(x,triangles,f)
    np.testing.assert_allclose(c,cr,atol=3e-15)
    np.testing.assert_allclose(b,br,atol=3e-16)
    errors=[]
    for h in (1e-2,1e-3,1e-4):
        cp,bp=cochains(x+h*v,triangles,f); cm,bm=cochains(x-h*v,triangles,f)
        errors.append(np.linalg.norm((cp-cm)/(2*h)-dc)+np.linalg.norm((bp-bm)/(2*h)-db))
    assert errors[1]<errors[0]/80 and errors[2]<errors[1]/80
    translation=np.tile([1.,-2.,3.],(len(x),1))
    omega=np.array([.3,.4,-.7]); rotation=np.cross(omega,x)
    for tangent in (translation,rotation):
        _,_,cc,bb=cochains_jvp(x,triangles,f,tangent)
        assert np.linalg.norm(cc)+np.linalg.norm(bb)<1e-13


def test_scale_average_boundary_term_and_finite_difference():
    x,f,v=example()
    result=scale_average(x,f,1.3,2.9,.53,direction=v)
    assert np.linalg.norm(result.boundary_derivative)>1e-6
    errors=[]
    for h in (1e-4,1e-5,1e-6):
        p=scale_average(x+h*v,f,1.3,2.9,.53).signed
        m=scale_average(x-h*v,f,1.3,2.9,.53).signed
        errors.append(np.linalg.norm((p-m)/(2*h)-result.derivative))
    assert errors[1]<errors[0]/50
    assert errors[2]<1e-8
    np.testing.assert_allclose(result.signed,
        np.einsum('i,ijk->jk',result.widths,result.pieces)/1.6,atol=1e-14)
    reflected=x*np.array([-1,1,1])
    np.testing.assert_allclose(scale_average(reflected,f,1.3,2.9,.53).signed,-result.signed,atol=1e-13)


def test_galerkin_certificates_and_nested_energy():
    rng=np.random.default_rng(4412); z=rng.normal(size=(12,12)); l=z.T@z
    c=rng.normal(size=12); tau=.17; a=l+tau*np.eye(12); exact=np.linalg.solve(a,c)
    q,_=np.linalg.qr(rng.normal(size=(12,12))); energies=[]
    for k in range(13):
        out=galerkin(l,c,q[:,:k],tau)
        error=exact-out['u']; energies.append(out['energy'])
        assert np.linalg.norm(error)<=out['solution_error_bound']+1e-12
        energy_error=c@exact-out['energy']
        np.testing.assert_allclose(energy_error,error@a@error,atol=2e-13)
        assert energy_error<=out['energy_error_bound']+2e-13
    assert np.min(np.diff(energies))>=-1e-12
    np.testing.assert_allclose(energies[-1],c@exact,atol=2e-13)
    with pytest.raises(ValueError): galerkin(l,c,q*2,tau)
