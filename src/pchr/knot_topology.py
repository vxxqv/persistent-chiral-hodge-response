"""Exact two-strand Jones fixtures and polygon Fox determinants.

The polygon determinant is abs(Delta(-1)), not a complete knot classifier.
For open protein chains every value is conditional on the stated closure.
"""
import numpy as np
import sympy as sp


def two_strand_jones(exponent):
    """Normalized bracket of closure sigma1**m, as a Laurent polynomial in A.

    TL2: e^2=delta e; normalized closure tr(I)=delta, tr(e)=1.
    Positive generator A I+A^-1 e; writhe correction (-A^3)^-m.
    """
    if not isinstance(exponent,int):raise ValueError('integer braid exponent required')
    a=sp.Symbol('A');delta=-a*a-a**-2;one=sp.Integer(1);coefficient=sp.Integer(0)
    b,c=(a,a**-1) if exponent>=0 else (a**-1,a)
    for _ in range(abs(exponent)):
        one,coefficient=sp.expand(one*b),sp.expand(one*c+coefficient*b+coefficient*c*delta)
    return sp.expand((-a**3)**(-exponent)*(one*delta+coefficient))


def crossing_records(polygon,projection):
    """Transverse crossings of a closed polygon in a proper orthogonal frame."""
    x=np.asarray(polygon,float);q=np.asarray(projection,float)
    if x.ndim!=2 or x.shape[1]!=3 or len(x)<3 or not np.isfinite(x).all():raise ValueError('finite polygon vertices required')
    if q.shape!=(3,3) or not np.isfinite(q).all() or np.linalg.norm(q.T@q-np.eye(3))>1e-10 or np.linalg.det(q)<0:raise ValueError('finite proper orthogonal projection frame required')
    y=x@q;d=np.roll(y,-1,axis=0)-y;n=len(y);ii,jj=np.triu_indices(n,2);keep=~((ii==0)&(jj==n-1));ii,jj=ii[keep],jj[keep]
    di,dj=d[ii,:2],d[jj,:2];gap=y[jj,:2]-y[ii,:2]
    cross=lambda a,b:a[:,0]*b[:,1]-a[:,1]*b[:,0]
    denominator=cross(di,dj);possible=np.abs(denominator)>1e-12
    i,j=ii[possible],jj[possible];gap=gap[possible];di,dj=di[possible],dj[possible];denominator=denominator[possible]
    s=cross(gap,dj)/denominator;t=cross(gap,di)/denominator;take=(s>0)&(s<1)&(t>0)&(t<1)
    records=[]
    for a,b,ss,tt,dd in zip(i[take],j[take],s[take],t[take],denominator[take]):
        if min(ss,tt,1-ss,1-tt)<1e-8:raise ValueError('projection crossing too close to a vertex')
        zi=y[a,2]+ss*d[a,2];zj=y[b,2]+tt*d[b,2]
        if abs(zi-zj)<1e-8:raise ValueError('spatial intersection or numerically nongeneric projection')
        over,under=(a+ss,b+tt) if zi>zj else (b+tt,a+ss)
        records.append({'over':float(over),'under':float(under),'sign':int(np.sign(dd)*(1 if zi>zj else -1))})
    return sorted(records,key=lambda c:c['under'])


def fox_determinant(polygon,projection):
    records=crossing_records(polygon,projection);n=len(records)
    if not n:return {'determinant':1,'crossings':0,'writhe':0}
    under=np.array([c['under'] for c in records]);matrix=sp.zeros(n,n)
    for i,c in enumerate(records):
        # Arc i runs from undercrossing i to undercrossing i+1 along the polygon.
        overarc=(np.searchsorted(under,c['over'],side='right')-1)%n
        matrix[i,overarc]+=2;matrix[i,(i-1)%n]-=1;matrix[i,i]-=1
    minor=matrix[:-1,:-1];det=abs(int(minor.det(method='domain-ge'))) if n>1 else 1
    if det<1 or det%2==0:raise AssertionError('invalid knot determinant')
    return {'determinant':det,'crossings':n,'writhe':sum(c['sign'] for c in records)}


def trefoil_points(n=100):
    q=2*np.pi*np.arange(n)/n
    return np.column_stack([(2+np.cos(3*q))*np.cos(2*q),(2+np.cos(3*q))*np.sin(2*q),np.sin(3*q)])


def _pairing(pairs):return tuple(sorted(tuple(sorted(p)) for p in pairs))


def tl_identity(n):return _pairing((i,n+i) for i in range(n))


def tl_generator(n,i):
    if not 0<=i<n-1:raise ValueError('adjacent TL generator required')
    return _pairing([(i,i+1),(n+i,n+i+1)]+[(j,n+j) for j in range(n) if j not in (i,i+1)])


def tl_product(a,b,n):
    """Stack two noncrossing pairing diagrams; count removed internal circles."""
    parent=list(range(3*n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(i)]=find(j)
    for i,j in a:union(i,j)
    for i,j in b:union(n+i,n+j)
    components={}
    for i in range(3*n):components.setdefault(find(i),[]).append(i)
    pairs=[];loops=0
    for component in components.values():
        external=[i if i<n else i-n for i in component if i<n or i>=2*n]
        if not external:loops+=1
        elif len(external)==2:pairs.append(external)
        else:raise AssertionError('invalid TL pairing composition')
    return _pairing(pairs),loops


def braid_jones_A(n,word):
    """Exact normalized TL state sum for a braid word, retaining formal A.

    Letters +/-i are generators sigma_i with one-based braid indexing.
    This classical state sum is not a quantum-computing speedup.
    """
    a=sp.Symbol('A');delta=-a*a-a**-2;state={tl_identity(n):sp.Integer(1)};writhe=0
    for letter in word:
        if not isinstance(letter,int) or not 1<=abs(letter)<n:raise ValueError('braid generator out of range')
        writhe+=1 if letter>0 else -1;power=1 if letter>0 else -1;new={}
        for diagram,c in state.items():
            new[diagram]=new.get(diagram,0)+c*a**power
            composed,loops=tl_product(diagram,tl_generator(n,abs(letter)-1),n)
            new[composed]=new.get(composed,0)+c*a**(-power)*delta**loops
        state={d:sp.expand(c) for d,c in new.items() if c!=0}
    bracket=0
    for diagram,c in state.items():
        parent=list(range(2*n))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i,j in [*diagram,*[(i,n+i) for i in range(n)]]:parent[find(i)]=find(j)
        loops=len({find(i) for i in range(2*n)});bracket+=c*delta**(loops-1)
    return sp.expand((-a**3)**(-writhe)*bracket)


def unitary_braid_matrices(root_order=5):
    """Two-dimensional TL3 representation at t=exp(2pi i/r), r>=3.

    Its ordinary matrix trace is not the Jones Markov trace on the full TL
    algebra. The construction verifies a unitary braid representation only.
    """
    if not isinstance(root_order,int) or root_order<3:raise ValueError('integer root order at least three required')
    delta=2*np.cos(np.pi/root_order);a=np.exp(1j*(np.pi/2-np.pi/(2*root_order)))
    e1=np.array([[delta,0.],[0.,0.]]);off=np.sqrt(max(1-delta**-2,0.));e2=np.array([[1/delta,off],[off,delta-1/delta]])
    return a*np.eye(2)+e1/a,a*np.eye(2)+e2/a,e1,e2
