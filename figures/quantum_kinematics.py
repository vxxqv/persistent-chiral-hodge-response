"""Five figures connecting molecular fields, quantum analysis and knot sectors."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.ticker import NullLocator
from scipy.spatial.transform import Rotation
import pyvista as pv
import rebuild as v
from pchr.knot_topology import trefoil_points,unitary_braid_matrices

ROOT=v.ROOT;Q=ROOT/'results/quantum_kinematics';F=ROOT/'results/molecular_fields';P=ROOT/'results/protein_knots'


def load(p):return json.loads(Path(p).read_text())


def field_parallel_scale(x,velocity,scale=1.2):
    eye=np.array([.15,-.65,1.5]);direction=-eye/np.linalg.norm(eye)
    right=np.cross(direction,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,direction)
    allpoints=np.r_[x,x+velocity*scale];aspect=1900/1300
    return max((np.max(abs(allpoints@up))+.4)*1.15,(np.max(abs(allpoints@right))+.4)/aspect*1.15)


def field_render(key,x,velocity,bonds,elements,scale=1.2,parallel_scale=None):
    path=v.CACHE/(key+'.png');meta=v.CACHE/(key+'.json')
    if path.exists() and meta.exists():
        metadata=load(meta);v.MANIFEST['renders'][key]=metadata;return plt.imread(path),metadata
    size=(1900,1300);p=pv.Plotter(off_screen=True,window_size=size,lighting='three lights');p.set_background('white');p.enable_anti_aliasing('ssaa')
    for i,j,order in bonds:p.add_mesh(pv.Line(x[i],x[j]).tube(radius=.055,n_sides=16),color='#929B9C',smooth_shading=True)
    for i,pos in enumerate(x):p.add_mesh(pv.Sphere(radius=.18,center=pos,theta_resolution=20,phi_resolution=16),color=v.ELEMENT.get(elements[i],v.GREY),smooth_shading=True)
    take=np.linalg.norm(velocity,axis=1)>1e-8
    p.add_arrows(x[take],velocity[take]*scale,mag=1,color=v.TEAL)
    bound=np.max(np.linalg.norm(x,axis=1));eye=np.array([.15,-.65,1.5])*bound;p.camera_position=[eye,(0,0,0),(0,1,0)];p.enable_parallel_projection()
    aspect=size[0]/size[1];parallel=field_parallel_scale(x,velocity,scale) if parallel_scale is None else float(parallel_scale);p.camera.parallel_scale=parallel
    metadata={'parallel_scale':float(parallel),'window_size':size,'one_angstrom_width_fraction':1/(2*parallel*aspect),'arrow_factor_radians':scale,'camera_position':[eye.tolist(),[0,0,0],[0,1,0]]}
    p.show(screenshot=str(path),auto_close=True);v.dump(meta,metadata);v.MANIFEST['renders'][key]=metadata;return plt.imread(path),metadata


def fig20():
    z=np.load(F/'vector_fields_a.npz');case=load(ROOT/'results/constrained/case_a_n65.json');nh=len(np.load(ROOT/'results/constrained/case_a_n65.npz')['heavy_coordinates']);x=z['baseline_coordinates'][:nh];coords,rot,center=v.canonical(x);vel=z['baseline_horizontal_velocity_fields'][:,:nh]@rot
    bonds=[b for b in case['bonds'] if max(b[:2])<nh];elements=case['elements'][:nh]
    fig=plt.figure(figsize=(178/25.4,151/25.4));gs=fig.add_gridspec(2,2,left=.13,right=.95,bottom=.15,top=.92,hspace=.62,wspace=.50)
    common_scale=max(field_parallel_scale(coords,velocity) for velocity in vel)
    for i in range(2):
        ax=fig.add_subplot(gs[0,i]);im,meta=field_render('field20_common_'+str(i),coords,vel[i],bonds,elements,parallel_scale=common_scale);v.image_panel(ax,im,meta);v.panel(ax,'AB'[i],f'Horizontal velocity field {i+1}')
    ax=fig.add_subplot(gs[1,0]);q=(np.arange(25)-12)*360/25;field=np.fft.fftshift(np.linalg.norm(z['curvature_field'],axis=2));im=ax.pcolormesh(q,q,field.T,cmap='cividis',shading='nearest',rasterized=True)
    ax.set_xticks([-180,0,180]);ax.set_yticks([-180,0,180]);ax.set_xlabel(r'$q_1$ (degrees)');ax.set_ylabel(r'$q_2$ (degrees)');v.panel(ax,'C',r'Connection curvature norm $\|F_{12}\|$')
    cb=fig.colorbar(im,ax=ax,pad=.03,fraction=.045);cb.ax.tick_params(labelsize=8)
    ax=fig.add_subplot(gs[1,1]);kin=load(Q/'kinematics.json')
    for side,color in [('a',v.BLUE),('b',v.ORANGE)]:
        eps=np.array([r['epsilon'] for r in kin[side]['loops']]);err=np.array([r['curvature_vector_error'] for r in kin[side]['loops']]);ax.loglog(eps,err,'o-',color=color,label='Isomer '+side.upper())
    ax.set_xlabel(r'Loop side $\epsilon$ (rad)');ax.set_ylabel('Scaled curvature-vector error');ax.set_xticks([.025,.05,.1,.2],['0.025','0.05','0.10','0.20']);ax.xaxis.set_minor_locator(NullLocator());v.panel(ax,'D','Commutator predicts the phase');ax.legend(loc='lower right',fontsize=8)
    caption=('Figure 20. Vector fields and rotational curvature of the constrained molecular family. A,B, The two mass-horizontal velocity fields at isomer A’s base conformation, displayed on its experimental-identity-linked generated heavy-atom geometry. Arrows show0.35rad times the instantaneous all-atom horizontal velocity, restricted to the displayed heavy atoms; they are tangent vectors, not finite displacements. The mass projection includes explicit hydrogens. Atom positions and the camera are fixed between panels. C, Norm of the body-section connection curvature on a25-by25 periodic grid. Its angular-coordinate convention is fixed; individual body components are archived. D, The square-loop reconstruction error compared with the predicted negative curvature vector, decreasing with loop size for both independently generated case slices. The loop starts along positive q1 then positive q2. The fields remove overall angular momentum, and their nonzero vertical Lie bracket explains rotational holonomy. No receptor mechanism or experimental trajectory is inferred.')
    caption=caption.replace('show0.35rad','show 1.2 rad').replace('a25-by25','a 25-by25')
    caption=caption.replace('The loop starts along positive q1', 'The scaled error is the norm of log(R)/epsilon squared plus F12, with the rotation log identified with its axial vector. The loop starts along positive q1')
    v.save(fig,'Figure_20_Molecular_vector_fields',caption,{'case':case,'coordinates':x,'display_rotation':rot,'horizontal_velocity_fields':z['baseline_horizontal_velocity_fields'],'curvature_grid':z['curvature_field'],'kinematics':kin})


def fig21():
    fig=plt.figure(figsize=(178/25.4,147/25.4));gs=fig.add_gridspec(2,2,left=.13,right=.97,bottom=.15,top=.92,hspace=.65,wspace=.50);source={}
    ax=fig.add_subplot(gs[0,0])
    for side,color in [('a',v.BLUE),('b',v.ORANGE)]:
        z=np.load(F/f'vector_fields_{side}.npz');source[side]={k:z[k] for k in ['physical_time_fs','state','total_energy','kinetic_energy']};t=z['physical_time_fs']/1000
        ax.plot(t,np.degrees(z['state'][:,0]),color=color,label='Isomer '+side.upper());ax.plot(t,np.degrees(z['state'][:,1]),color=color,ls='--')
    v.clean(ax,'Reduced-model time (ps)','Angle displacement (degrees)');v.panel(ax,'A','Coupled torsional motion')
    ax=fig.add_subplot(gs[0,1]);z=np.load(F/'vector_fields_a.npz');t=z['physical_time_fs']/1000
    ax.plot(t,z['kinetic_energy'],color=v.TEAL,label='Kinetic');ax.plot(t,z['total_energy']-z['kinetic_energy'],color=v.GREY,ls='--',label='Potential');ax.axhline(.5,color=v.RULE,lw=1);v.clean(ax,'Reduced-model time (ps)','Energy (kcal/mol)');v.panel(ax,'B','One conserved model Hamiltonian')
    ax=fig.add_subplot(gs[1,0])
    for side,color in [('a',v.BLUE),('b',v.ORANGE)]:
        z=np.load(F/f'vector_fields_{side}.npz');angle=Rotation.from_matrix(z['state'][:,4:].reshape(-1,3,3)).magnitude();ax.plot(z['physical_time_fs']/1000,np.degrees(angle),color=color)
    v.clean(ax,'Reduced-model time (ps)','Laboratory rotation angle (degrees)');v.panel(ax,'C','Rotation compensates internal motion')
    ax=fig.add_subplot(gs[1,1]);checks=load(F/'forced_geodesic_validation.json')
    for side,color in [('a',v.BLUE),('b',v.ORANGE)]:
        rows=checks['cases'][side];dt=np.array([r['step'] for r in rows]);error=np.array([r['endpoint_canonical_error'] for r in rows]);ax.loglog(dt,error,'o-',color=color)
    dt=np.array([.05,.2]);ax.loglog(dt,1e-8*(dt/.05)**4,color=v.GREY,ls=':',label='Fourth-order slope');ax.set_xlabel('Independent RK4 step');ax.set_ylabel('Canonical endpoint difference');ax.set_xticks([.05,.1,.2],['0.05','0.10','0.20']);ax.xaxis.set_minor_locator(NullLocator());v.panel(ax,'D','Forced-geodesic cross-check');ax.legend(loc='upper left',fontsize=8)
    handles=[Line2D([],[],color=v.BLUE,label='Isomer A'),Line2D([],[],color=v.ORANGE,label='Isomer B'),Line2D([],[],color=v.INK,label=r'$q_1$'),Line2D([],[],color=v.INK,ls='--',label=r'$q_2$'),Line2D([],[],color=v.TEAL,label='Kinetic'),Line2D([],[],color=v.GREY,ls='--',label='Potential')]
    caption=('Figure 21. Reduced Hamiltonian motion on the same fixed molecular slices. A, The two angle coordinates from zero initial angles and a positive q1 velocity normalized to0.5kcal/mol kinetic energy. B, Kinetic and restricted MMFF94s potential energy for isomer A; their sum remains0.5kcal/mol to the recorded numerical precision. C, Proper-rotation angle reconstructed at zero angular momentum. Time conversion uses48.8882fs per declared natural unit. D, An additional validation after the primary trajectories: fixed-step RK4 solves the forced geodesic equation in angles and velocities, independently organized from the canonical DOP853 momentum integration. Endpoint errors decrease at approximately fourth order under step halving; the dotted line indicates slope only. No physical parameter or endpoint was retuned. These are deterministic constrained-model paths without solvent, thermostat or a receptor, and are not full molecular dynamics or measured motions.')
    for a,b in [('to0.5kcal','to 0.5 kcal'),('remains0.5kcal','remains 0.5 kcal'),('uses48.8882fs','uses 48.8882 fs')]:caption=caption.replace(a,b)
    v.save(fig,'Figure_21_Reduced_Hamiltonian_motion',caption,{'trajectories':source,'validation':checks},handles)


def fig22():
    fig=plt.figure(figsize=(178/25.4,151/25.4));gs=fig.add_gridspec(2,2,left=.13,right=.83,bottom=.15,top=.92,hspace=.62,wspace=.55);source={};ax=fig.add_subplot(gs[0,0])
    for side,color in [('a',v.BLUE),('b',v.ORANGE)]:
        for n,ls in [(33,':'),(65,'--'),(129,'-')]:
            data=load(Q/f'rotor_{side}_n{n}.json');source[f'{side}_{n}']=data;k=np.array([int(vv) for vv in data['spectra']]);energy=[data['spectra'][str(j)]['low_eigenvalues'][0] for j in k];ax.plot(k,energy,color=color,ls=ls,marker='o' if n==129 else None,ms=3)
    v.clean(ax,'Fourier cutoff K','Ground Ritz energy (kcal/mol)');ax.set_xticks([3,9,15]);v.panel(ax,'A','Ground-state basis sensitivity')
    ax=fig.add_subplot(gs[0,1]);k=np.array([3,5,7,9,12,15])
    for side,color in [('a',v.BLUE),('b',v.ORANGE)]:
        values=[]
        with np.load(Q/f'rotor_{side}_n129.npz') as data:
            for cutoff in k:
                e=data[f'eigenvalues_K{cutoff}'];values.append(np.log(np.exp(-(e-e[0])/(.00198720425864083*298.15)).sum())-e[0]/(.00198720425864083*298.15))
        ax.plot(k,values,'o-',color=color,ms=3)
    v.clean(ax,'Fourier cutoff K',r'Finite-model $\log Z$');ax.set_xticks([3,9,15]);v.panel(ax,'B','Partition sums increase')
    maxabs=0.;diffs=[]
    for side in ('a','b'):
        z=np.load(Q/f'rotor_{side}_n129.npz');diff=(z['quantum_probability']-z['classical_probability'])*129**2;diffs.append(diff);maxabs=max(maxabs,float(abs(diff).max()));source['density_'+side]={'difference_from_classical':diff,'quantum':z['quantum_probability'],'classical':z['classical_probability']}
    for i,side in enumerate(('a','b')):
        ax=fig.add_subplot(gs[1,i]);q=np.arange(129)*360/129;im=ax.pcolormesh(q,q,diffs[i].T,cmap=v.DIVERGE,vmin=-maxabs,vmax=maxabs,shading='nearest',rasterized=True);ax.set_xticks([0,180,360]);ax.set_yticks([0,180,360]);ax.set_xlabel(r'$q_1$ (degrees)');ax.set_ylabel(r'$q_2$ (degrees)');v.panel(ax,'CD'[i],'Isomer '+side.upper()+': finite-model density')
    cax=fig.add_axes([.875,.17,.015,.29]);bar=fig.colorbar(im,cax=cax);bar.set_label('Density difference');bar.ax.tick_params(labelsize=8)
    handles=[Line2D([],[],color=v.BLUE,label='Isomer A'),Line2D([],[],color=v.ORANGE,label='Isomer B'),Line2D([],[],color=v.GREY,ls=':',label='N = 33'),Line2D([],[],color=v.GREY,ls='--',label='N = 65'),Line2D([],[],color=v.GREY,label='N = 129')]
    caption=('Figure 22. Effective quantum torsional model with approximation limits retained. A, Ground-state Ritz values for both cases at three coefficient-grid sizes and six nested Fourier cutoffs. Each fixed-grid sequence decreases variationally; differences at the largest cutoff remain material. B, Complete finite-basis partition sums at298.15K and N129 increase with nested basis size, as the fixed-operator convergence theorem requires. C,D, N129,K15 quantum minus classical position-density quadratures, multiplied by129 squared so the plotted density is relative to uniform angle measure. Blue is negative and orange positive on one symmetric scale. The underlying coefficient model uses exact Fourier integrals of periodic bilinear rho, rho g-inverse and rho E interpolants. These densities are not claimed converged quantum corrections or ab initio populations. An exact reflected model is isospectral; the two independently generated case slices need not be. Matrix residuals and spectral-weight truncation checks concern the finite model and do not certify excluded modes or physical accuracy.')
    for a,b in [('at298.15K','at 298.15 K'),('N129','N = 129'),('K15','K = 15'),('by129','by 129')]:caption=caption.replace(a,b)
    v.save(fig,'Figure_22_Quantum_rotor_analysis',caption,source,handles)


def protein_render(pdb,data):
    path=v.CACHE/('protein_'+pdb+'.png');meta=v.CACHE/('protein_'+pdb+'.json')
    if path.exists() and meta.exists():
        metadata=load(meta);v.MANIFEST['renders']['protein_'+pdb]=metadata;return plt.imread(path),metadata
    x=np.array(data['coordinates']);coords,rot,center=v.canonical(x);p=pv.Plotter(off_screen=True,window_size=(1700,1500),lighting='three lights');p.set_background('white');p.enable_anti_aliasing('ssaa')
    points=pv.PolyData(coords);points.lines=np.r_[len(coords),np.arange(len(coords))];points['position']=np.linspace(0,1,len(coords));tube=points.tube(radius=.48,n_sides=18);p.add_mesh(tube,scalars='position',cmap='viridis',show_scalar_bar=False,smooth_shading=True)
    for index,color in [(0,v.BLUE),(-1,v.ORANGE)]:p.add_mesh(pv.Sphere(radius=1.0,center=coords[index]),color=color,smooth_shading=True)
    eye=np.array([1.4,-1.5,1.2])*np.linalg.norm(coords,axis=1).max();p.camera_position=[eye,(0,0,0),(0,1,0)];p.enable_parallel_projection();p.reset_camera();scale=p.camera.parallel_scale
    metadata={'display_rotation':rot.tolist(),'center':center.tolist(),'parallel_scale':float(scale),'one_angstrom_width_fraction':1/(2*scale*(1700/1500)),'window_size':[1700,1500],'coordinates':'experimental CA trace; tube is a rendering, not a molecular surface'}
    p.show(screenshot=str(path),auto_close=True);v.dump(meta,metadata);v.MANIFEST['renders']['protein_'+pdb]=metadata;return plt.imread(path),metadata


def fig23():
    fig=plt.figure(figsize=(178/25.4,154/25.4));gs=fig.add_gridspec(2,3,left=.08,right=.98,bottom=.17,top=.91,hspace=.50,wspace=.32,height_ratios=[1.3,1]);source={}
    for i,pdb in enumerate(['1J85','2EFV','1UBQ']):
        data=load(P/(pdb+'.json'));source[pdb]=data;ax=fig.add_subplot(gs[0,i]);im,meta=protein_render(pdb,data);v.image_panel(ax,im,meta,bar=False);v.panel(ax,'ABC'[i],pdb+' · '+str(data['n_residues'])+' residues')
        # A physical scale bar uses exactly the stored orthographic camera.
        length=10*meta['one_angstrom_width_fraction'];ax.plot([.06,.06+length],[.05,.05],transform=ax.transAxes,color=v.INK,lw=1.2);ax.text(.06+length/2,.075,'10 Å',transform=ax.transAxes,ha='center',fontsize=8)
    ax=fig.add_subplot(gs[1,:]);ax.set_position([.23,.17,.74,.30]);determinants=['1','3','5','9'];colors=[v.LIGHT,v.TEAL,v.BLUE,v.ORANGE];bottom=np.zeros(3)
    for det,color in zip(determinants,colors):
        values=np.array([int(source[pdb]['determinant_counts'].get(det,0)) for pdb in ['1J85','2EFV','1UBQ']]);ax.barh(np.arange(3),values,left=bottom,color=color,height=.55,label='Determinant '+det)
        for i,(left,value) in enumerate(zip(bottom,values)):
            if value>=4:ax.text(left+value/2,i,str(value),ha='center',va='center',color='white' if det!='1' else v.INK,fontsize=8)
        bottom+=values
    ax.set_yticks(range(3),['YibK · 1J85','MJ0366 · 2EFV','Ubiquitin · 1UBQ']);ax.invert_yaxis();ax.set_xlim(0,64);ax.set_xticks([0,16,32,48,64]);ax.set_xlabel('Number of the 64 deterministic closure directions');v.panel(ax,'D','Open-chain closure sensitivity remains visible')
    caption=('Figure 23. Experimental protein backbones and closure-dependent knot evidence. A-C, First-model chain-A C-alpha traces of1J85,YibK;2EFV,MJ0366; and1UBQ,ubiquitin. Tube colour progresses from N to C along each independent chain; endpoint beads are blue and orange. The physical10Å scale bars use the stored orthographic cameras; the proteins are not shown at a common magnification. D, Exact Fox cofactor determinants after64 far-point closures in fixed Fibonacci-sphere directions. Counts are57/4/2/1 for determinants3/5/1/9 in1J85,50/14 for3/1 in2EFV, and63/1 for1/3 in1UBQ. Every closure is retained. Determinant1 does not prove an unknot, determinant3 does not uniquely prove a trefoil, and the determinant cannot distinguish mirror handedness. Three independent new projections of each closed polygon give576 additional matching determinant checks. These frequencies describe the closure protocol and are not random-sampling probabilities or biological-function estimates.')
    for a,b in [('of1J85,YibK;2EFV,MJ0366; and1UBQ,ubiquitin','of 1J85 (YibK), 2EFV (MJ0366), and 1UBQ (ubiquitin)'),('physical10Å','physical 10 Å'),('after64','after 64'),('are57','are 57'),('determinants3','determinants 3'),('in1J85,50','in 1J85, 50'),('for3/1 in2EFV','for 3/1 in 2EFV'),('and63','and 63'),('for1/3 in1UBQ','for 1/3 in 1UBQ'),('Determinant1','Determinant 1'),('determinant3','determinant 3'),('give576','give 576')]:caption=caption.replace(a,b)
    handles=[Line2D([],[],color=color,lw=5,label='Determinant '+det) for det,color in zip(determinants,colors)]
    v.save(fig,'Figure_23_Protein_closure_topology',caption,source,handles)


def fig24():
    x=trefoil_points(240);fig=plt.figure(figsize=(178/25.4,142/25.4));gs=fig.add_gridspec(2,2,left=.12,right=.97,bottom=.15,top=.92,hspace=.57,wspace=.48)
    for i,y in enumerate([x,x*np.array([-1,1,1])]):
        ax=fig.add_subplot(gs[0,i],projection='3d');ax.plot(*np.vstack([y,y[0]]).T,color=v.BLUE if i==0 else v.ORANGE,lw=3);ax.set_box_aspect((1,1,.4));ax.view_init(35,25);ax.set_axis_off();v.panel(ax,'AB'[i],['Closed trefoil fixture','Exact geometric mirror'][i])
    ax=fig.add_subplot(gs[1,0]);exponents=np.arange(-4,5);positive=np.array([int(k in [1,3])-int(k==4) for k in exponents]);mirror=positive[::-1];ax.bar(exponents-.15,positive,width=.3,color=v.BLUE);ax.bar(exponents+.15,mirror,width=.3,color=v.ORANGE);ax.axhline(0,color=v.RULE,lw=.7);v.clean(ax,'Laurent exponent in t','Jones coefficient');ax.set_xticks([-4,-2,0,2,4]);ax.set_yticks([-1,0,1]);v.panel(ax,'C','Knot invariant resolves the mirror')
    ax=fig.add_subplot(gs[1,1]);b1,b2,e1,e2=unitary_braid_matrices(5);checks=load(P/'quantum_braid_checks.json');values=[checks['root_order5_unitarity_errors'][0],checks['root_order5_braid_relation_error'],checks['root_order5_noncommutativity']]
    ax.scatter(values,range(3),c=[v.GREY,v.GREY,v.TEAL],s=28,marker='D');ax.set_xscale('log');ax.set_xlim(1e-17,10);ax.set_yticks(range(3),['Unitarity','Braid relation','Noncommutativity']);ax.invert_yaxis();ax.set_xlabel('Frobenius norm (log scale)');v.panel(ax,'D','Non-Abelian unitary braid action')
    handles=[Line2D([],[],color=v.BLUE,lw=3,label='Chosen trefoil convention'),Line2D([],[],color=v.ORANGE,lw=3,label='Mirror')]
    caption=('Figure 24. Genuine quantum-topological algebra, distinguished from the quantum rotor. A,B, A mathematical closed trefoil polygon and its exact mirror, with the same camera. This is not a protein coordinate model. C, The positive two-strand braid convention gives V(t)=t+t^3-t^4; reflection inverts t, giving different Laurent coefficients. These exact polynomials obstruct knot achirality but are not complete invariants. D, A two-dimensional Temperley-Lieb representation at t=exp(2pi i/5) obeys unitarity and the braid relation to floating precision while the two generators do not commute. The exact diagram state sum separately checks braid relations and both Markov stabilization signs. The ordinary trace of these2-by2 matrices is not the Jones closure trace. No quantum algorithm or molecular braid operation was executed. Reflection-even Hamiltonian spectra can remain identical between mirror sectors even when the Jones coefficients differ.')
    caption=caption.replace('these2-by2','these 2-by2')
    caption=caption.replace('C, The positive', 'C, The algebraic crossing convention is fixed independently of a visual handedness label. The positive')
    v.save(fig,'Figure_24_Quantum_braid_invariants',caption,{'trefoil_points':x,'exponents':exponents,'jones_coefficients':positive,'mirror_coefficients':mirror,'matrix_checks':checks},handles)


def main(out,cache):
    v.OUT=out;v.CACHE=cache;out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    for function in (fig20,fig21,fig22,fig23,fig24):function()
    v.dump(out/'quantum_kinematics_figure_manifest.json',v.MANIFEST)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--cache',type=Path,required=True);args=parser.parse_args();main(args.out,args.cache)
