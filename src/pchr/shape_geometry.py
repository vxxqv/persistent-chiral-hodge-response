"""Rigid-motion quotient geometry for labelled heavy-atom configurations."""
from __future__ import annotations

import numpy as np

from .core import cochains_jvp, distances


def horizontal_projection(x, velocity, masses):
    """Mass-orthogonal removal of translation and infinitesimal rotation.

    Collinear configurations are rejected because their inertia tensor is
    singular. No scale normalization or atom-permutation quotient is applied.
    """
    x=np.asarray(x,dtype=float);v=np.asarray(velocity,dtype=float);m=np.asarray(masses,dtype=float)
    if x.ndim!=2 or x.shape[1]!=3 or v.shape!=x.shape or m.shape!=(len(x),):
        raise ValueError('incompatible geometry, velocity or masses')
    if not all(np.isfinite(a).all() for a in (x,v,m)) or np.any(m<=0):
        raise ValueError('finite coordinates and positive masses required')
    if not len(x):raise ValueError('nonempty configuration required')
    y=x-np.average(x,axis=0,weights=m);translation=np.average(v,axis=0,weights=m)
    centered=v-translation
    inertia=np.einsum('i,ij,ij->',m,y,y)*np.eye(3)-np.einsum('i,ij,ik->jk',m,y,y)
    ev=np.linalg.eigvalsh(inertia)
    if ev[-1]<=0 or ev[0]<=1e-12*ev[-1]:raise ValueError('collinear or numerically degenerate configuration')
    torque=np.sum(m[:,None]*np.cross(y,centered),axis=0)
    angular=np.linalg.solve(inertia,torque)
    rotation=np.cross(angular,y)
    horizontal=centered-rotation
    return dict(horizontal=horizontal,translation=translation,rotation=rotation,
                angular_velocity=angular,inertia=inertia,centered_coordinates=y)


def metric(u,v,masses):
    return float(np.sum(np.asarray(masses)[:,None]*np.asarray(u)*np.asarray(v)))


def cochains_jet2(x,triangles,atom_features,direction,bandwidth=2.4,length_unit=1.):
    """Analytic values, first and second derivatives along x+t*direction."""
    c,b,dc,db=cochains_jvp(x,triangles,atom_features,direction,bandwidth,length_unit)
    x=np.asarray(x,dtype=float);v=np.asarray(direction,dtype=float);f=np.asarray(atom_features,dtype=float)
    delta=x[:,None]-x[None,:];dv=v[:,None]-v[None,:]
    s=np.einsum('ijk,ijk->ij',delta,delta)
    ds=2*np.einsum('ijk,ijk->ij',delta,dv);d2s=2*np.einsum('ijk,ijk->ij',dv,dv)
    den=2*bandwidth**2;kernel=np.exp(-s/den)
    dk=-kernel*ds/den;d2k=kernel*((ds/den)**2-d2s/den)
    z=f[:,0]/8;degree=f[:,1]/4;hyd=f[:,2]/4;n=max(len(x),1)
    fields=np.column_stack([z,degree,hyd,kernel@z/n])
    df=np.column_stack([np.zeros((len(x),3)),dk@z/n])
    d2f=np.column_stack([np.zeros((len(x),3)),d2k@z/n])
    weights=np.column_stack([np.ones(len(x)),z,1+degree])
    d2c=np.zeros_like(c);d2b=np.zeros_like(b)
    pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for row,(i,j,k) in enumerate(triangles):
        ts=s[i,j]+s[i,k]+s[j,k];dts=ds[i,j]+ds[i,k]+ds[j,k];d2ts=d2s[i,j]+d2s[i,k]+d2s[j,k]
        total=ts+s[i]+s[j]+s[k];dtotal=dts+ds[i]+ds[j]+ds[k];d2total=d2ts+d2s[i]+d2s[j]+d2s[k]
        kk=np.exp(-total/den);dkk=-kk*dtotal/den;d2kk=kk*((dtotal/den)**2-d2total/den)
        a=x[j]-x[i];aa=v[j]-v[i];bb=x[k]-x[i];dbb=v[k]-v[i];cc=x-x[i];dcc=v-v[i]
        vol=cc@np.cross(a,bb)
        dvol=cc@(np.cross(aa,bb)+np.cross(a,dbb))+dcc@np.cross(a,bb)
        d2vol=2*(cc@np.cross(aa,dbb)+dcc@(np.cross(aa,bb)+np.cross(a,dbb)))
        for value in (vol,dvol,d2vol):value[[i,j,k]]=0
        d2c[row]=(d2kk*vol+2*dkk*dvol+kk*d2vol)@weights/length_unit**3
        kt=np.exp(-ts/den);dkt=-kt*dts/den;d2kt=kt*((dts/den)**2-d2ts/den)
        for col,(a,bb) in enumerate(pairs):
            fj=fields[j,a]-fields[i,a];fk=fields[k,a]-fields[i,a]
            gj=fields[j,bb]-fields[i,bb];gk=fields[k,bb]-fields[i,bb]
            dfj=df[j,a]-df[i,a];dfk=df[k,a]-df[i,a];dgj=df[j,bb]-df[i,bb];dgk=df[k,bb]-df[i,bb]
            ffj=d2f[j,a]-d2f[i,a];ffk=d2f[k,a]-d2f[i,a];ggj=d2f[j,bb]-d2f[i,bb];ggk=d2f[k,bb]-d2f[i,bb]
            minor=fj*gk-fk*gj;dminor=dfj*gk+fj*dgk-dfk*gj-fk*dgj
            d2minor=ffj*gk+2*dfj*dgk+fj*ggk-ffk*gj-2*dfk*dgj-fk*ggj
            d2b[row,col]=d2kt*minor+2*dkt*dminor+kt*d2minor
    return c,b,dc,db,d2c,d2b


def bond_rotation(x,bonds,axis_edge,moving_vertex,angle=0.):
    """Rotate one component around a bridge bond, preserving bond geometry.

    Returns positions and their first two derivatives with respect to angle.
    Rings are rejected. This is a geometrical torsion, not an energy model.
    """
    x=np.asarray(x,dtype=float);distances(x)
    a,b=axis_edge;n=len(x)
    if not (0<=a<n and 0<=b<n and 0<=moving_vertex<n) or a==b:
        raise ValueError('invalid axis or moving vertex')
    adjacency=[set() for _ in range(n)];present=False
    for i,j,*_ in bonds:
        if {i,j}=={a,b}:present=True;continue
        adjacency[i].add(j);adjacency[j].add(i)
    if not present:raise ValueError('axis must be a bond')
    component={moving_vertex};pending=[moving_vertex]
    while pending:
        i=pending.pop()
        for j in adjacency[i]-component:component.add(j);pending.append(j)
    if (a in component)==(b in component):raise ValueError('axis must separate the graph into components')
    axis=x[b]-x[a];norm=np.linalg.norm(axis)
    if norm==0 or not np.isfinite(angle):raise ValueError('invalid axis geometry or angle')
    axis/=norm;indices=sorted(component);r=x[indices]-x[a]
    moved=(r*np.cos(angle)+np.cross(axis,r)*np.sin(angle)
           +(r@axis)[:,None]*axis*(1-np.cos(angle)))
    y=x.copy();y[indices]=x[a]+moved
    velocity=np.zeros_like(x);acceleration=np.zeros_like(x)
    velocity[indices]=np.cross(axis,moved);acceleration[indices]=np.cross(axis,velocity[indices])
    return y,velocity,acceleration,indices
