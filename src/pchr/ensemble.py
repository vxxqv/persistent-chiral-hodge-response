"""Reflection-odd RKHS geometry of empirical molecular-response measures.

The Gaussian kernel is characteristic on descriptor space, not necessarily on
molecular configuration space. Conformer support points are not assumed to be
independent thermodynamic samples by these descriptive computations.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


def points(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('finite nonempty point matrix required')
    return x


def gaussian(x,y=None,sigma=1.):
    x=points(x);y=x if y is None else points(y)
    if x.shape[1]!=y.shape[1] or not np.isfinite(sigma) or sigma<=0:
        raise ValueError('matching feature dimensions and positive sigma required')
    return np.exp(-cdist(x,y,'sqeuclidean')/(2*sigma*sigma))


def probability_weights(n,weights=None):
    w=np.full(n,1/n) if weights is None else np.asarray(weights,float)
    if w.shape!=(n,) or np.any(w<0) or not np.isfinite(w).all() or not np.isclose(w.sum(),1):
        raise ValueError('probability weights must match the support')
    return w


def odd_kernel(x,y=None,sigma=1.):
    x=points(x);y=x if y is None else points(y)
    return .5*(gaussian(x,y,sigma)-gaussian(x,-y,sigma))


def parity_mmd_decomposition(x,y,sigma=1.):
    """For odd input coordinates, split MMD into even/odd RKHS energies."""
    x,y=points(x),points(y)
    signed=float(odd_kernel(x,sigma=sigma).mean()+odd_kernel(y,sigma=sigma).mean()-2*odd_kernel(x,y,sigma).mean())
    total=mmd_squared(x,y,sigma)
    even=total-signed
    if min(even,signed)<-1e-10:raise ArithmeticError('negative parity energy')
    balanced_x=np.r_[x,-y];balanced_y=np.r_[y,-x]
    wx=np.r_[np.full(len(x),.5/len(x)),np.full(len(y),.5/len(y))]
    wy=np.r_[np.full(len(y),.5/len(y)),np.full(len(x),.5/len(x))]
    balanced=mmd_squared(balanced_x,balanced_y,sigma,wx,wy)
    return {'total_squared':total,'odd_squared':max(signed,0.),'even_squared':max(even,0.),
            'balanced_squared':balanced,'projection_identity_error':abs(balanced-signed)}


def rms_normalized_jets(x,velocity,acceleration):
    """Differentiate the common RMS normalization; no translation centering."""
    x=points(x);v=np.asarray(velocity,float);a=np.asarray(acceleration,float)
    if x.shape!=v.shape or x.shape!=a.shape or not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError('finite jets with matching dimensions required')
    s=np.sqrt(np.mean(np.sum(x*x,axis=1)))
    if s<=1e-12:raise ValueError('RMS normalization is at its numerical floor')
    first=np.mean(np.sum(x*v,axis=1))/s
    second=(np.mean(np.sum(v*v+x*a,axis=1))-first*first)/s
    return (x/s,v/s-x*first/s**2,
            a/s-2*v*first/s**2-x*second/s**2+2*x*first**2/s**3)


def odd_pullback_metric(z,sigma=1.):
    z=np.asarray(z,float)
    if z.ndim!=1 or not np.isfinite(z).all() or not np.isfinite(sigma) or sigma<=0:
        raise ValueError('finite feature vector and positive sigma required')
    q=np.exp(-2*np.dot(z,z)/sigma**2)
    return (1+q)*np.eye(len(z))/(2*sigma**2)-2*q*np.outer(z,z)/sigma**4


def odd_derivative_gram(x,sigma=1.):
    """Block matrix of inner products of feature-map partial derivatives."""
    x=points(x);n,d=x.shape
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('positive sigma required')
    result=np.empty((n*d,n*d));eye=np.eye(d)/sigma**2
    for i in range(n):
        for j in range(n):
            minus=x[i]-x[j];plus=x[i]+x[j]
            block=.5*(np.exp(-minus@minus/(2*sigma**2))*(eye-np.outer(minus,minus)/sigma**4)
                       +np.exp(-plus@plus/(2*sigma**2))*(eye-np.outer(plus,plus)/sigma**4))
            result[i*d:(i+1)*d,j*d:(j+1)*d]=block
    return result


def mmd_squared(x,y,sigma=1.,weights_x=None,weights_y=None):
    x,y=points(x),points(y)
    wx,wy=probability_weights(len(x),weights_x),probability_weights(len(y),weights_y)
    value=float(wx@gaussian(x,sigma=sigma)@wx+wy@gaussian(y,sigma=sigma)@wy-2*wx@gaussian(x,y,sigma)@wy)
    if value < -1e-10:raise ArithmeticError('negative squared RKHS distance')
    return max(value,0.)


def odd_statistics(x,sigma=1.,regularization=.05):
    x=points(x);n=len(x)
    if not np.isfinite(regularization) or regularization<=0:raise ValueError('positive regularization required')
    g=odd_kernel(x,sigma=sigma);g=(g+g.T)/2
    u=np.full(n,1/n)
    coeff0=solve(g/n+regularization*np.eye(n),u,assume_a='pos')
    t=float(u@g@coeff0)
    if not -1e-10<=t<1:raise ArithmeticError('invalid centered covariance resolvent')
    t=max(t,0.);coeff=coeff0/(1-t)
    squared=float(u@g@u)
    if squared < -1e-10:raise ArithmeticError('invalid odd mean norm')
    scores=g@coeff
    empirical_error=float(np.mean(scores<=0))
    fisher=t/(1-t)
    unbiased=(float((g.sum()-np.trace(g))/(n*(n-1))) if n>1 else None)
    return {'odd_norm_squared':max(squared,0.),'mirror_mmd_squared':4*max(squared,0.),
            'centered_fisher_score':fisher,'uncentered_t':t,'cantelli_upper_bound':1/(1+fisher),
            'empirical_nonpositive_fraction':empirical_error,'witness_coefficients':coeff,
            'odd_u_statistic':unbiased,'diagonal_mean':float(np.trace(g)/n)}


def odd_witness(query,support,sigma=1.,coefficients=None):
    support=points(support)
    weights=(np.full(len(support),1/len(support)) if coefficients is None else np.asarray(coefficients,float))
    if weights.shape!=(len(support),):raise ValueError('coefficient dimensions do not match support')
    return odd_kernel(query,support,sigma)@weights


def odd_norm_jets(x,velocity,acceleration=None,sigma=1.,weights=None):
    """Exact first and second derivatives along a twice differentiable path.

    Acceleration is the second coordinate derivative, not a Hessian surrogate.
    Weights and bandwidth are fixed along the differentiated path.
    """
    x=points(x);v=np.asarray(velocity,float)
    a=np.zeros_like(x) if acceleration is None else np.asarray(acceleration,float)
    if v.shape!=x.shape or a.shape!=x.shape or not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError('finite path jets must match point shape')
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('positive sigma required')
    w=np.full(len(x),1/len(x)) if weights is None else np.asarray(weights,float)
    if w.shape!=(len(x),) or np.any(w<0) or not np.isfinite(w).all() or not np.isclose(w.sum(),1):
        raise ValueError('probability weights required')
    product=w[:,None]*w[None,:];output=np.zeros(3)
    for sign,kernel_sign in [(-1,1),(1,-1)]:
        q=x[:,None]+sign*x[None,:];dq=v[:,None]+sign*v[None,:];ddq=a[:,None]+sign*a[None,:]
        value=np.exp(-np.sum(q*q,axis=2)/(2*sigma*sigma))
        linear=np.sum(q*dq,axis=2)/(sigma*sigma)
        curvature=(np.sum(dq*dq,axis=2)+np.sum(q*ddq,axis=2))/(sigma*sigma)
        output+=.5*kernel_sign*np.array([np.sum(product*value),
             np.sum(product*(-linear*value)),np.sum(product*value*(linear*linear-curvature))])
    return output


def empirical_w1(x,y):
    """Uniform empirical W1 by finite transport linear programming."""
    x,y=points(x),points(y);n,m=len(x),len(y);cost=cdist(x,y)
    rows=np.zeros((n+m,n*m))
    for i in range(n):rows[i,i*m:(i+1)*m]=1
    for j in range(m):rows[n+j,j::m]=1
    result=linprog(cost.ravel(),A_eq=rows,b_eq=np.r_[np.full(n,1/n),np.full(m,1/m)],bounds=(0,None),method='highs')
    if not result.success:raise RuntimeError(result.message)
    return float(result.fun)


def separation_metrics(features,labels):
    x=points(features);y=np.asarray(labels)
    if y.shape!=(len(x),) or len(np.unique(y))<2:raise ValueError('at least two labelled classes required')
    d=cdist(x,x);i,j=np.triu_indices(len(x),1);positive=y[i]!=y[j]
    npair=int(positive.sum());nnegative=len(positive)-npair
    if not npair or not nnegative:auc=None
    else:
        ranks=rankdata(d[i,j],method='average')
        auc=float((ranks[positive].sum()-npair*(npair+1)/2)/(npair*nnegative))
    nearest=[];silhouette=[]
    for k in range(len(x)):
        row=d[k].copy();row[k]=np.inf;mask=row==row.min()
        same=(y==y[k]);same[k]=False
        if same.any():
            nearest.append(float(np.mean(y[mask]==y[k])))
            aa=float(d[k,same].mean());bb=min(float(d[k,y==c].mean()) for c in np.unique(y) if c!=y[k])
            silhouette.append((bb-aa)/max(aa,bb) if max(aa,bb)>0 else 0.)
        else:silhouette.append(0.)
    return {'es_auc':auc,'nn_tie_aware':float(np.mean(nearest)) if nearest else None,
            'silhouette':float(np.mean(silhouette)),'same_pairs':nnegative,'different_pairs':npair}
