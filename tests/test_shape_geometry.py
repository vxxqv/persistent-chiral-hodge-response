import numpy as np
import pytest

from pchr.core import cochains_jvp,rips
from pchr.shape_geometry import bond_rotation,cochains_jet2,horizontal_projection,metric


def test_projection_is_orthogonal_idempotent_and_equivariant():
    rng=np.random.default_rng(7811);x=rng.normal(size=(9,3));v=rng.normal(size=x.shape);m=rng.uniform(1,20,len(x))
    p=horizontal_projection(x,v,m);h=p['horizontal'];y=p['centered_coordinates']
    np.testing.assert_allclose(np.sum(m[:,None]*h,axis=0),0,atol=1e-12)
    np.testing.assert_allclose(np.sum(m[:,None]*np.cross(y,h),axis=0),0,atol=1e-12)
    np.testing.assert_allclose(horizontal_projection(x,h,m)['horizontal'],h,atol=1e-13)
    np.testing.assert_allclose(metric(v,v,m),metric(h,h,m)+metric(p['rotation'],p['rotation'],m)+m.sum()*np.sum(p['translation']**2),atol=2e-12)
    for sign in (1,-1):
        q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,0]*=sign*np.linalg.det(q)
        pp=horizontal_projection(x@q.T+[2,3,-4],v@q.T,m)
        np.testing.assert_allclose(pp['horizontal'],h@q.T,atol=1e-12)
        np.testing.assert_allclose(pp['angular_velocity'],np.linalg.det(q)*q@p['angular_velocity'],atol=1e-12)
    rigid=np.tile([2,3,4],(len(x),1))+np.cross([.2,.7,-.1],y)
    np.testing.assert_allclose(horizontal_projection(x,rigid,m)['horizontal'],0,atol=1e-12)
    with pytest.raises(ValueError):horizontal_projection(np.array([[0.,0,0],[1.,0,0],[2.,0,0]]),np.ones((3,3)),np.ones(3))


def test_analytic_second_derivatives():
    rng=np.random.default_rng(5812);x=rng.normal(size=(7,3));v=rng.normal(size=x.shape)
    f=np.array([[6,3,1],[7,2,1],[8,1,0],[6,2,2],[9,1,0],[16,2,0],[6,4,0]])
    tri=rips(x,2.7)[2];c,b,dc,db,d2c,d2b=cochains_jet2(x,tri,f,v)
    errors=[]
    for step in (1e-2,1e-3,1e-4):
        _,_,cp,bp=cochains_jvp(x+step*v,tri,f,v);_,_,cm,bm=cochains_jvp(x-step*v,tri,f,v)
        errors.append(np.linalg.norm((cp-cm)/(2*step)-d2c)+np.linalg.norm((bp-bm)/(2*step)-d2b))
    assert errors[1]<errors[0]/80 and errors[2]<errors[1]/80
    for actual,expected in zip(cochains_jet2(x,tri,f,np.zeros_like(v))[4:],(np.zeros_like(c),np.zeros_like(b))):
        np.testing.assert_allclose(actual,expected,atol=0)


def test_torsion_preserves_bond_lengths_and_derivatives():
    x=np.array([[0.,0,0],[1.,.2,0],[2.,0,0],[3.,.4,1],[4.,1.,.7]])
    bonds=[[0,1,1],[1,2,1],[2,3,1],[3,4,1]]
    y,v,a,indices=bond_rotation(x,bonds,(1,2),0,.37)
    for i,j,*_ in bonds:np.testing.assert_allclose(np.linalg.norm(y[i]-y[j]),np.linalg.norm(x[i]-x[j]),atol=1e-14)
    h=1e-5;yp=bond_rotation(x,bonds,(1,2),0,.37+h)[0];ym=bond_rotation(x,bonds,(1,2),0,.37-h)[0]
    np.testing.assert_allclose((yp-ym)/(2*h),v,atol=1e-10)
    np.testing.assert_allclose((yp-2*y+ym)/h**2,a,atol=1e-5)
    with pytest.raises(ValueError):bond_rotation(x,bonds+[[0,4,1]],(1,2),0)
