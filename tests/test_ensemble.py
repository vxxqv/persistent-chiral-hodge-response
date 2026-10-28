import numpy as np
import pytest
from scipy.linalg import solve
from sklearn.metrics import roc_auc_score,silhouette_score
from pchr.ensemble import (gaussian,odd_kernel,mmd_squared,odd_statistics,odd_witness,
                          odd_norm_jets,empirical_w1,separation_metrics,parity_mmd_decomposition,rms_normalized_jets,
                          odd_pullback_metric,odd_derivative_gram)


def test_odd_projection_and_mean_cancellation():
    x=np.array([[1.],[2.],[-3.]])
    assert x.mean()==0
    g=odd_kernel(x)
    full=gaussian(np.r_[x,-x]);projection=np.c_[np.eye(3),-np.eye(3)]/2
    np.testing.assert_allclose(g,projection@full@projection.T,atol=1e-15)
    assert np.linalg.eigvalsh(g).min()>-1e-14
    stats=odd_statistics(x)
    assert stats['odd_norm_squared']>0
    np.testing.assert_allclose(stats['mirror_mmd_squared'],mmd_squared(x,-x),atol=1e-14)
    np.testing.assert_allclose(odd_witness(-x,x),-odd_witness(x,x),atol=1e-15)
    assert odd_statistics(np.r_[x,-x])['odd_norm_squared']<1e-15


def test_centered_resolvent_independent_explicit_feature_space():
    rng=np.random.default_rng(81);x=rng.normal(size=(7,3));g=odd_kernel(x)
    values,vectors=np.linalg.eigh(g);psi=vectors@np.diag(np.sqrt(np.maximum(values,0)))
    mean=psi.mean(axis=0);center=psi-mean;cov=center.T@center/len(x);lam=.05
    w=solve(cov+lam*np.eye(len(g)),mean,assume_a='pos')
    stats=odd_statistics(x,regularization=lam)
    np.testing.assert_allclose(stats['centered_fisher_score'],mean@w,rtol=1e-12)
    np.testing.assert_allclose(g@stats['witness_coefficients'],psi@w,rtol=1e-12,atol=1e-13)
    assert stats['empirical_nonpositive_fraction']<=stats['cantelli_upper_bound']+1e-14
    sequence=[odd_statistics(x,regularization=v)['centered_fisher_score'] for v in [.01,.05,.2]]
    assert sequence[0]>=sequence[1]>=sequence[2]


def test_path_acceleration_and_second_derivative():
    rng=np.random.default_rng(29);x=rng.normal(size=(5,4));v=rng.normal(size=x.shape);a=rng.normal(size=x.shape)
    value,first,second=odd_norm_jets(x,v,a)
    h=1e-5
    at=lambda t:x+t*v+.5*t*t*a
    np.testing.assert_allclose(value,odd_statistics(x)['odd_norm_squared'],atol=1e-14)
    np.testing.assert_allclose(first,(odd_norm_jets(at(h),v+h*a,a)[0]-odd_norm_jets(at(-h),v-h*a,a)[0])/(2*h),atol=2e-9)
    np.testing.assert_allclose(second,(odd_norm_jets(at(h),v+h*a,a)[1]-odd_norm_jets(at(-h),v-h*a,a)[1])/(2*h),atol=2e-8)
    reflected=odd_norm_jets(-x,-v,-a)
    np.testing.assert_allclose(reflected,[value,first,second],atol=1e-14)


def test_transport_bound_and_known_one_dimensional_solution():
    x=np.array([[-3.],[1.],[2.]]);y=np.array([[-2.],[2.],[2.5]])
    w=empirical_w1(x,y)
    np.testing.assert_allclose(w,np.mean(np.abs(np.sort(x[:,0])-np.sort(y[:,0]))),atol=1e-12)
    cx=np.sqrt(odd_statistics(x)['odd_norm_squared']);cy=np.sqrt(odd_statistics(y)['odd_norm_squared'])
    assert abs(cx-cy)<=w+1e-14
    assert np.sqrt(mmd_squared(x,y))<=w+1e-14


def test_metrics_against_independent_definitions_and_ties():
    x=np.array([[0.,0.],[0.,0.],[2.,0.],[2.1,.1]])
    y=np.array([0,0,1,1]);result=separation_metrics(x,y)
    d=np.linalg.norm(x[:,None]-x[None,:],axis=2);i,j=np.triu_indices(4,1)
    np.testing.assert_allclose(result['es_auc'],roc_auc_score(y[i]!=y[j],d[i,j]))
    np.testing.assert_allclose(result['silhouette'],silhouette_score(d,y,metric='precomputed'))
    assert result['nn_tie_aware']==1
    tied=separation_metrics(np.zeros((4,2)),[0,0,1,1])
    assert tied['es_auc']==.5
    np.testing.assert_allclose(tied['nn_tie_aware'],1/3)


def test_invalid_kernel_inputs():
    for v in [0,-1,np.nan]:
        with pytest.raises(ValueError):gaussian([[1]],sigma=v)
    with pytest.raises(ValueError):odd_statistics([[1]],regularization=0)


def test_parity_balancing_is_an_orthogonal_projection():
    rng=np.random.default_rng(13);x=rng.normal(size=(4,5));y=rng.normal(size=(7,5))
    result=parity_mmd_decomposition(x,y)
    np.testing.assert_allclose(result['total_squared'],result['odd_squared']+result['even_squared'],atol=1e-14)
    np.testing.assert_allclose(result['odd_squared'],result['balanced_squared'],atol=1e-14)
    identical=mmd_squared(np.r_[x,y],np.r_[y,x])
    assert identical<1e-14


def test_rms_normalization_second_jet_and_finite_support_divergence():
    rng=np.random.default_rng(47);x=rng.normal(size=(4,3));v=rng.normal(size=x.shape);a=rng.normal(size=x.shape)
    z,dz,ddz=rms_normalized_jets(x,v,a);h=1e-5
    at=lambda t:rms_normalized_jets(x+t*v+.5*t*t*a,v+t*a,a)
    np.testing.assert_allclose((at(h)[0]-at(-h)[0])/(2*h),dz,atol=1e-9)
    np.testing.assert_allclose((at(h)[1]-at(-h)[1])/(2*h),ddz,atol=1e-8)
    g=odd_kernel(x);coefficient=1/(np.ones(4)@np.linalg.solve(g,np.ones(4)))
    lam=1e-7;observed=lam*odd_statistics(x,regularization=lam)['centered_fisher_score']
    np.testing.assert_allclose(observed,coefficient,rtol=2e-5)


def test_pullback_metric_spectrum_and_derivative_definition():
    rng=np.random.default_rng(72);c0=(1-2*np.exp(-1.5))/2
    for norm in [0,.2,np.sqrt(.75),1.,3.]:
        z=np.zeros(4);z[0]=norm;g=odd_pullback_metric(z)
        e=np.linalg.eigvalsh(g)
        assert e.min()>=c0-1e-14 and e.max()<=1+1e-14
        np.testing.assert_allclose(odd_derivative_gram(z[None]),g,atol=1e-15)
        v=rng.normal(size=4);h=2e-4
        # Mixed finite difference of the kernel is independent of the closed
        # metric formula; quadratic cancellation controls this looser tolerance.
        numeric=(odd_kernel([z+h*v],[z+h*v])[0,0]-odd_kernel([z+h*v],[z-h*v])[0,0]
                 -odd_kernel([z-h*v],[z+h*v])[0,0]+odd_kernel([z-h*v],[z-h*v])[0,0])/(4*h*h)
        np.testing.assert_allclose(numeric,v@g@v,rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(odd_pullback_metric([np.sqrt(.75),0])[0,0],c0,atol=1e-15)


def test_odd_law_immersion_generic_and_antipodal_failure():
    x=np.array([[.7,.1],[1.2,-.4],[-.1,1.1]])
    assert np.linalg.eigvalsh(odd_derivative_gram(x)).min()>1e-4
    antipodal=np.array([[.7,.1],[-.7,-.1]]);v=np.array([1.,2.,-1.,-2.])
    np.testing.assert_allclose(v@odd_derivative_gram(antipodal)@v,0,atol=1e-14)
