"""Integrable rooted bridge-torsion families and their mass quotient metric.

Fixed labelled atoms and fixed internal geometry are used. Single bridges can
have large chemical barriers: eligibility is kinematic, not an energy claim.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve
from .shape_geometry import horizontal_projection,cochains_jet2
from .core import cochains,cochains_jvp,persistent_operator


def bridge_axes(n,bonds,root=0):
    adjacency=[set() for _ in range(n)]
    for i,j,*_ in bonds:adjacency[i].add(j);adjacency[j].add(i)
    depth={root:0};pending=[root]
    for i in pending:
        for j in sorted(adjacency[i]):
            if j not in depth:depth[j]=depth[i]+1;pending.append(j)
    if len(depth)!=n:raise ValueError('connected molecule required')
    axes=[]
    for a,b,order in bonds:
        if order!=1 or min(len(adjacency[a]),len(adjacency[b]))<2:continue
        component={root};pending=[root]
        for i in pending:
            for j in adjacency[i]:
                if {i,j}=={a,b}:continue
                if j not in component:component.add(j);pending.append(j)
        if (a in component)==(b in component):continue
        parent,child=(a,b) if a in component else (b,a)
        axes.append({'parent':parent,'child':child,'moving':sorted(set(range(n))-component),'depth':depth[parent]})
    return sorted(axes,key=lambda a:(a['depth'],a['parent'],a['child']))


def chart(x,axes,angles,second=True):
    """Exact chart value, Jacobian and symmetric coordinate second derivatives."""
    y=np.asarray(x,float).copy();q=np.asarray(angles,float);k=len(axes)
    if y.ndim!=2 or y.shape[1]!=3 or q.shape!=(k,) or not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError('finite coordinates and one angle per ordered axis required')
    for ax,theta in zip(axes,q):
        a,b=ax['parent'],ax['child'];unit=y[b]-y[a];length=np.linalg.norm(unit)
        if length<1e-12:raise ValueError('degenerate bond axis')
        unit/=length;idx=ax['moving'];r=y[idx]-y[a]
        y[idx]=y[a]+r*np.cos(theta)+np.cross(unit,r)*np.sin(theta)+(r@unit)[:,None]*unit*(1-np.cos(theta))
    v=np.zeros((k,len(y),3));units=[]
    for i,ax in enumerate(axes):
        a,b=ax['parent'],ax['child'];u=y[b]-y[a];u/=np.linalg.norm(u);units.append(u)
        v[i,ax['moving']]=np.cross(u,y[ax['moving']]-y[a])
    if not second:return y,v
    h=np.zeros((k,k,len(y),3))
    for i in range(k):
        h[i,i]=np.cross(units[i],v[i])
        for j in range(i+1,k):
            # Root ordering makes intersecting bridge components nested.
            if set(axes[j]['moving']).issubset(axes[i]['moving']):
                h[i,j]=h[j,i]=np.cross(units[i],v[j])
    return y,v,h


def quotient_metric(x,v,masses,h=None):
    """g_ij = <V_i,V_j>_M - ell_i^T I^-1 ell_j after translation removal.

    If h is provided, dg[a,i,j] is the analytic coordinate derivative of g.
    """
    x=np.asarray(x,float);v=np.asarray(v,float);m=np.asarray(masses,float)
    if v.ndim!=3 or v.shape[1:]!=x.shape:raise ValueError('one molecular velocity per coordinate required')
    zero=horizontal_projection(x,np.zeros_like(x),m);y=zero['centered_coordinates'];inertia=zero['inertia']
    vc=v-np.einsum('n,inj->ij',m,v)[:,None,:]/m.sum()
    ell=np.einsum('n,inj->ij',m,np.cross(y,vc));omega=np.linalg.solve(inertia,ell.T).T
    vh=vc-np.cross(omega[:,None,:],y)
    g=np.einsum('n,inj,knj->ik',m,vh,vh)
    if h is None:return g,vh
    h=np.asarray(h,float)
    if h.shape!=(len(v),len(v),len(x),3):raise ValueError('matching symmetric chart Hessian required')
    hc=h-np.einsum('n,ainj->aij',m,h)[:,:,None,:]/m.sum()
    dg=np.empty((len(v),len(v),len(v)))
    for a in range(len(v)):
        di=2*np.einsum('n,nj,nj->',m,y,vc[a])*np.eye(3)-np.einsum('n,nj,nk->jk',m,vc[a],y)-np.einsum('n,nj,nk->jk',m,y,vc[a])
        de=np.einsum('n,inj->ij',m,np.cross(vc[a],vc)+np.cross(y,hc[a]))
        dg[a]=(np.einsum('n,inj,knj->ik',m,hc[a],vc)+np.einsum('n,inj,knj->ik',m,vc,hc[a])
               -de@omega.T-omega@de.T+omega@di@omega.T)
    return g,vh,dg


def christoffel(g,dg):
    g=np.asarray(g,float);dg=np.asarray(dg,float)
    if g.shape[0]!=g.shape[1] or dg.shape!=(len(g),)*3:raise ValueError('matching metric and derivative required')
    if len(g) and np.linalg.eigvalsh(g)[0]<=1e-12*np.linalg.eigvalsh(g)[-1]:raise ValueError('singular torsion metric')
    out=np.empty((len(g),)*3)
    for i in range(len(g)):
        for j in range(len(g)):out[:,i,j]=np.linalg.solve(g,.5*(dg[i,:,j]+dg[j,:,i]-dg[:,i,j]))
    return out


def response_jets(x,features,v,h=None,r=2.6,s=3.2,tau=.3):
    """Source/same/persistent jets on the fixed filtration chamber of x.

    The chart-acceleration term is included in each mixed second derivative.
    No derivative through discrete filtration events is asserted.
    """
    x=np.asarray(x,float);v=np.asarray(v,float);op=persistent_operator(x,r,s)
    tri=op.small[2];c,b=cochains(x,tri,features);den=max(len(tri),1);size=len(tri)
    same=op.b2.T@op.b2
    if op.small[3]:
        from .core import boundary
        d=boundary(tri,op.small[3]);same+=d@d.T
    matrices={'source':None,'same_scale':same+tau*np.eye(size),'persistent':op.matrix+tau*np.eye(size)}
    first=[cochains_jvp(x,tri,features,w)[2:] for w in v]
    second={}
    if h is not None:
        diagonal=[cochains_jet2(x,tri,features,w)[4:] for w in v]
        for i in range(len(v)):
            for j in range(i,len(v)):
                if i==j:cc,bb=diagonal[i]
                else:
                    cc0,bb0=cochains_jet2(x,tri,features,v[i]+v[j])[4:]
                    cc=(cc0-diagonal[i][0]-diagonal[j][0])/2;bb=(bb0-diagonal[i][1]-diagonal[j][1])/2
                ac,ab=cochains_jvp(x,tri,features,h[i,j])[2:]
                second[i,j]=(cc+ac,bb+ab)
    result={}
    for name,matrix in matrices.items():
        def inverse(a):return a if matrix is None or size==0 else solve(matrix,a,assume_a='pos')
        u=inverse(c);du=[inverse(dc) for dc,db in first];p=(b.T@u/den).ravel()
        jac=np.array([((db.T@u+b.T@d)/den).ravel() for (dc,db),d in zip(first,du)]).reshape(len(v),18).T
        payload={'value':p,'jacobian':jac}
        if h is not None:
            hh=np.empty((18,len(v),len(v)))
            for (i,j),(cc,bb) in second.items():
                mixed=(bb.T@u+first[i][1].T@du[j]+first[j][1].T@du[i]+b.T@inverse(cc))/den
                hh[:,i,j]=hh[:,j,i]=mixed.ravel()
            payload['hessian']=hh
        result[name]=payload
    return result
