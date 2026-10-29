import itertools
import numpy as np
import sympy as sp
import gudhi
from pchr.constraints import rigidity_matrix,normal_acceleration,distance_pairs
from pchr.torsion_geometry import bridge_axes,chart,quotient_metric,christoffel,response_jets


def fixture():
    x=np.array([[0.,0,0],[1.1,.1,0],[1.8,1.1,.2],[2.9,.8,.7],[3.5,1.6,-.2],[4.3,1.1,.7],[5.,2.,1.2]])
    bonds=[[i,i+1,1] for i in range(6)];axes=bridge_axes(len(x),bonds)
    return x,bonds,axes,np.arange(1,8.)


def test_exact_singular_k4_not_a_realizable_flex():
    x=sp.Matrix([[0,0,0],[1,0,0],[0,1,0],[1,1,0]]);pairs=list(itertools.combinations(range(4),2))
    r=sp.zeros(6,12)
    for row,(i,j) in enumerate(pairs):
        for d in range(3):r[row,3*i+d]=x[i,d]-x[j,d];r[row,3*j+d]=x[j,d]-x[i,d]
    u=sp.Matrix([1,-1,-1,1]);stress=sp.Matrix([-u[i]*u[j] for i,j in pairs]);v=sp.zeros(12,1);v[11]=1
    b=sp.Matrix([sum((v[3*i+d]-v[3*j+d])**2 for d in range(3)) for i,j in pairs])
    assert r.rank()==5 and r*v==sp.zeros(6,1) and r.T*stress==sp.zeros(12,1) and (stress.T*b)[0]==1
    check=normal_acceleration(np.array(x,float),pairs,np.array(v,float).reshape(4,3),np.ones(4))
    assert np.linalg.norm(check['second_residual'])>.1


def test_integrable_chart_mixed_jets_metric_and_reflection():
    x,bonds,axes,m=fixture();q=np.array([.23,-.37,.51,.18]);y,v,h=chart(x,axes,q)
    pairs=distance_pairs(len(x),bonds)
    for i,j in pairs:np.testing.assert_allclose(np.linalg.norm(y[i]-y[j]),np.linalg.norm(x[i]-x[j]),atol=2e-14)
    np.testing.assert_allclose(rigidity_matrix(y,pairs)@v.reshape(len(v),-1).T,0,atol=2e-14)
    g,vh,dg=quotient_metric(y,v,m,h);gamma=christoffel(g,dg)
    np.testing.assert_allclose(gamma,gamma.transpose(0,2,1),atol=1e-13)
    step=2e-5
    for a in range(len(q)):
        d=np.eye(len(q))[a]*step;yp,vp=chart(x,axes,q+d,False);ym,vm=chart(x,axes,q-d,False)
        np.testing.assert_allclose((yp-ym)/(2*step),v[a],atol=5e-10)
        np.testing.assert_allclose((vp-vm)/(2*step),h[a],atol=5e-10)
        gp=quotient_metric(yp,vp,m)[0];gm=quotient_metric(ym,vm,m)[0]
        np.testing.assert_allclose((gp-gm)/(2*step),dg[a],atol=2e-8)
    refl=np.diag([-1.,1.,1.]);yr,vr,hr=chart(x@refl,axes,-q)
    np.testing.assert_allclose(yr,y@refl,atol=5e-15)
    np.testing.assert_allclose(vr,-v@refl,atol=5e-15)
    np.testing.assert_allclose(quotient_metric(yr,vr,m,hr)[0],g,atol=5e-14)
    check=normal_acceleration(y,pairs,v[0],m)
    np.testing.assert_allclose(check['first_residual'],0,atol=1e-13)
    np.testing.assert_allclose(check['second_residual'],0,atol=2e-12)


def test_response_hessian_chart_acceleration_and_covariance():
    x,bonds,axes,m=fixture();axes=axes[:2];q=np.array([.21,-.31]);y,v,h=chart(x,axes,q)
    f=np.array([[6,1,3],[6,2,2],[7,2,1],[8,2,0],[6,2,2],[16,2,0],[9,1,0]])
    jets=response_jets(y,f,v,h,r=2.7,s=3.3)
    step=2e-5
    for i in range(2):
        d=np.eye(2)[i]*step;yp,vp=chart(x,axes,q+d,False);ym,vm=chart(x,axes,q-d,False)
        plus=response_jets(yp,f,vp,r=2.7,s=3.3);minus=response_jets(ym,f,vm,r=2.7,s=3.3)
        for name in jets:
            np.testing.assert_allclose((plus[name]['value']-minus[name]['value'])/(2*step),jets[name]['jacobian'][:,i],atol=2e-10)
            np.testing.assert_allclose((plus[name]['jacobian']-minus[name]['jacobian'])/(2*step),jets[name]['hessian'][:,:,i],atol=3e-9)
    refl=np.diag([-1.,1.,1.]);yr,vr,hr=chart(x@refl,axes,-q);mirror=response_jets(yr,f,vr,hr,r=2.7,s=3.3)
    for name in jets:
        np.testing.assert_allclose(mirror[name]['value'],-jets[name]['value'],atol=1e-15)
        np.testing.assert_allclose(mirror[name]['jacobian'],jets[name]['jacobian'],atol=1e-15)
        np.testing.assert_allclose(mirror[name]['hessian'],-jets[name]['hessian'],atol=1e-15)


def test_periodic_vertex_filtration_includes_top_class():
    for n in (3,5):
        complex=gudhi.PeriodicCubicalComplex(vertices=np.zeros((n,n)),periodic_dimensions=[True,True])
        complex.persistence(homology_coeff_field=2,min_persistence=0)
        assert complex.betti_numbers()==[1,2,1]
        assert [len(complex.persistence_intervals_in_dimension(d)) for d in range(3)]==[1,2,1]
