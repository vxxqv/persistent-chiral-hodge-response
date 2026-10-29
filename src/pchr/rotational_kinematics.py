"""Mechanical connection and zero-angular-momentum reconstruction.

Body convention: lab positions are R y(q), Rdot=R hat(Omega),
Omega=-A_i qdot_i. Therefore F_12=d1 A2-d2 A1-A1 cross A2.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from .torsion_geometry import chart
from .shape_geometry import horizontal_projection


def hat(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def mechanical_connection(x,v,masses,h=None):
    m=np.asarray(masses,float);x=np.asarray(x,float);v=np.asarray(v,float)
    projected=horizontal_projection(x,np.zeros_like(x),m);y=projected['centered_coordinates'];inertia=projected['inertia']
    vc=v-np.einsum('n,inj->ij',m,v)[:,None,:]/m.sum()
    ell=np.einsum('n,inj->ij',m,np.cross(y,vc));connection=np.linalg.solve(inertia,ell.T).T
    if h is None:return connection,inertia,ell
    h=np.asarray(h,float);hc=h-np.einsum('n,ainj->aij',m,h)[:,:,None,:]/m.sum();da=np.empty((len(v),len(v),3))
    for a in range(len(v)):
        di=2*np.einsum('n,nj,nj->',m,y,vc[a])*np.eye(3)-np.einsum('n,nj,nk->jk',m,vc[a],y)-np.einsum('n,nj,nk->jk',m,y,vc[a])
        de=np.einsum('n,inj->ij',m,np.cross(vc[a],vc)+np.cross(y,hc[a]))
        da[a]=np.linalg.solve(inertia,(de-connection@di.T).T).T
    return connection,inertia,ell,da


def square_holonomy(x,axes,masses,epsilon):
    if len(axes)!=2 or not np.isfinite(epsilon) or epsilon<=0:raise ValueError('two axes and a positive loop size required')
    corners=epsilon*np.array([[0.,0.],[1,0],[1,1],[0,1],[0,0]])
    orientation=np.eye(3);maxmoment=0.;maxorth=0.;steps=0;path=[]
    for start,end in zip(corners[:-1],corners[1:]):
        velocity=end-start
        def rhs(t,flat):
            nonlocal maxmoment,maxorth
            y,v=chart(x,axes,start+t*velocity,False);a,inertia,ell=mechanical_connection(y,v,masses)
            omega=-velocity@a;maxmoment=max(maxmoment,float(np.linalg.norm(inertia@omega+velocity@ell)))
            r=flat.reshape(3,3);maxorth=max(maxorth,float(np.linalg.norm(r.T@r-np.eye(3))))
            return (r@hat(omega)).ravel()
        solution=solve_ivp(rhs,(0,1),orientation.ravel(),method='DOP853',rtol=1e-11,atol=1e-13,dense_output=True)
        if not solution.success:raise RuntimeError(solution.message)
        for t in np.linspace(0,1,17):path.append({'q':(start+t*velocity).tolist(),'rotation':solution.sol(t).reshape(3,3).tolist()})
        orientation=solution.y[:,-1].reshape(3,3);steps+=solution.nfev
    # Rotation.from_matrix projects tiny integration roundoff onto SO(3);
    # the unprojected matrix and its residual remain in the record.
    logarithm=Rotation.from_matrix(orientation).as_rotvec()
    return {'epsilon':float(epsilon),'rotation':orientation.tolist(),'rotation_log':logarithm.tolist(),'angle_radians':float(np.linalg.norm(logarithm)),
        'orthogonality_error':float(np.linalg.norm(orientation.T@orientation-np.eye(3))),'determinant':float(np.linalg.det(orientation)),
        'max_rhs_angular_momentum_residual':maxmoment,'max_internal_solver_stage_orthogonality_error':maxorth,'rhs_evaluations':steps,'path':path}
