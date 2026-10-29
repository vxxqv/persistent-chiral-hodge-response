import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.integrate import quad
from pchr.quantum_rotor import galerkin,coefficient_matrix,modes
from pchr.rotational_kinematics import mechanical_connection,square_holonomy
from pchr.torsion_geometry import bridge_axes,chart


def test_fourier_modes_cannot_be_silently_rounded():
    values=np.ones((5,5))
    for invalid in [[[.5,0]],[[np.nan,0]],[[np.inf,0]]]:
        with pytest.raises(ValueError):coefficient_matrix(values,invalid)


def test_exact_bilinear_fourier_integrals():
    n=7;values=np.cos(2*np.pi*np.arange(n)/n)+2;grid=np.arange(n+1)*2*np.pi/n;data=np.r_[values,values[0]]
    tensor=np.repeat(values[:,None],5,axis=1);p=np.array([[0,0],[1,0],[6,0],[8,0]])
    actual=coefficient_matrix(tensor,p)
    for i in range(len(p)):
        for j in range(len(p)):
            k=p[i,0]-p[j,0]
            re=quad(lambda q:np.interp(q,grid,data)*np.cos(k*q),0,2*np.pi,points=grid,epsabs=1e-11)[0]/(2*np.pi)
            im=quad(lambda q:-np.interp(q,grid,data)*np.sin(k*q),0,2*np.pi,points=grid,epsabs=1e-11)[0]/(2*np.pi)
            np.testing.assert_allclose(actual[i,j],re+1j*im,atol=3e-15)


def test_free_rotor_spectrum_and_variational_order():
    n=11;e=np.full((n,n),2.3);rho=np.ones_like(e);a=np.broadcast_to(np.eye(2),(n,n,2,2)).copy()
    h,b,p=galerkin(e,rho,a,3,kappa=.7)
    np.testing.assert_allclose(b,np.eye(len(p)),atol=2e-15)
    np.testing.assert_allclose(h,np.diag(2.3+.7*np.sum(p*p,axis=1)),atol=3e-15)
    values=eigh(h,b,eigvals_only=True);np.testing.assert_allclose(values,np.sort(2.3+.7*np.sum(p*p,axis=1)),atol=1e-13)
    q=2*np.pi*np.arange(n)/n;e=3+np.cos(q[:,None])+np.cos(q[None,:]);v=[]
    for k in (1,2,3):
        h,b,p=galerkin(e,rho,a,k);v.append(eigh(h,b,eigvals_only=True)[:6])
    assert np.all(np.diff(v,axis=0)<1e-10)


def test_connection_derivative_and_holonomy_sign():
    x=np.array([[0.,0,0],[1.,.1,0],[1.8,1.,.2],[2.7,.7,1.],[3.4,1.6,.3],[4.5,1.,1.2]])
    bonds=[[i,i+1,1] for i in range(5)];axes=bridge_axes(6,bonds)[:2];m=np.arange(1.,7)
    y,v,h=chart(x,axes,[0.,0.]);a,inertia,ell,da=mechanical_connection(y,v,m,h)
    curvature=da[0,1]-da[1,0]-np.cross(a[0],a[1]);errors=[]
    for epsilon in (.08,.04,.02):
        loop=square_holonomy(x,axes,m,epsilon);errors.append(np.linalg.norm(np.array(loop['rotation_log'])/epsilon**2+curvature))
        assert loop['orthogonality_error']<1e-10 and loop['max_rhs_angular_momentum_residual']<1e-12
    assert errors[1]<.6*errors[0] and errors[2]<.6*errors[1]
