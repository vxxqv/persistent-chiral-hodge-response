"""Rebuild the scientific figures from frozen coordinates and numerical results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.linalg import eigh, solve

from pchr.core import (atom_energy, cochains, distances, local_signed_map,
                       persistent_operator, persistent_from_complexes, response, rips)
from pchr.exact_topology import octahedral_filling
from pchr.analysis import scale_average
from pchr.shape_geometry import bond_rotation

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'data/conformers.json').read_text())['records']
RECORDS={r['name']:r for r in DATA}
INK='#171717'; GREY='#777E7B'; LIGHT='#B9BEBA'; TEAL='#1F6F5F'; RULE='#D8DCD9'
BLUE='#356D9A'; ORANGE='#AE6732'
DIVERGE=LinearSegmentedColormap.from_list('signed_response',[BLUE,'#FAFAF7',ORANGE])
GREEN=LinearSegmentedColormap.from_list('response_energy',['#ECEFEA','#A2C1B5',TEAL,'#123C32'])
ELEMENT={'C':'#515B60','N':'#4F75A3','O':'#BB5A51','S':'#C69C37','H':'#E2E4E0'}
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.titlesize':9.5,
 'axes.titleweight':'bold','axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':8,'legend.frameon':False,'axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.65,
 'ytick.major.width':.65,'lines.linewidth':1.25,'figure.facecolor':'white',
 'axes.facecolor':'white','text.color':INK,'axes.labelcolor':INK,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','savefig.facecolor':'white'})

OUT:Path; CACHE:Path
MANIFEST={'input_sha256':{},'figures':{},'renders':{}}
CAPTIONS={}


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(x):
    x=np.asarray(x,dtype=float); y=x-x.mean(axis=0)
    _,_,vh=np.linalg.svd(y,full_matrices=False)
    rot=vh.T
    if np.linalg.det(rot)<0:rot[:,-1]*=-1
    # Choose a reproducible camera side, using atom order only for the display.
    for k in (0,1):
        if y[0]@rot[:,k]<0:rot[:,k]*=-1;rot[:,2]*=-1
    assert np.linalg.det(rot)>.999999
    return y@rot,rot,x.mean(axis=0)


def aligned(x,ref):
    y=x-x.mean(axis=0); t=ref-ref.mean(axis=0)
    u,_,vh=np.linalg.svd(y.T@t)
    d=np.eye(3);d[2,2]=np.linalg.det(u@vh)
    return y@u@d@vh


def case(name,conf=0,r=2.6,s=3.2,tau=.3):
    record=RECORDS[name]
    x=np.asarray(record['conformers'][conf]['coordinates']);f=np.asarray(record['atom_features'])
    op=persistent_operator(x,r,s);c,b=cochains(x,op.small[2],f);out=response(op,c,b,tau)
    return record,x,f,op,c,b,out


def viewport(key,x,bonds=None,elements=None,triangles=None,values=None,norm=None,
             atom_values=None,atom_norm=None,edges=None,ensemble=None,highlight=None,
             size=(1800,1180),extent=None,face_opacity=.75,camera='molecule'):
    """Orthographic physical geometry; all labels are added later as vectors."""
    cache=CACHE/(key+'.png');meta=CACHE/(key+'.json')
    if cache.exists() and meta.exists():
        metadata=json.loads(meta.read_text());MANIFEST['renders'][key]=metadata
        return plt.imread(cache),metadata
    x=np.asarray(x,dtype=float)
    p=pv.Plotter(off_screen=True,window_size=size,lighting='three lights')
    p.set_background('white');p.enable_anti_aliasing('ssaa')
    p.enable_depth_peeling(number_of_peels=12,occlusion_ratio=0)
    if triangles is not None and len(triangles):
        faces=np.column_stack([np.full(len(triangles),3),np.asarray(triangles)]).ravel()
        mesh=pv.PolyData(x,faces)
        if values is None:
            p.add_mesh(mesh,color='#A5C5B8',opacity=face_opacity,smooth_shading=False,
                       show_edges=True,edge_color='#668B7D',line_width=.7)
        else:
            colors=np.uint8(DIVERGE(norm(np.asarray(values)))*255)
            mesh.cell_data['color']=colors
            p.add_mesh(mesh,scalars='color',rgb=True,opacity=face_opacity,
                       show_scalar_bar=False,show_edges=True,edge_color='#8C9894',line_width=.5)
    if edges is not None:
        for i,j in edges:
            p.add_mesh(pv.Line(x[i],x[j]).tube(radius=.018,n_sides=10),color='#A6B7AE',
                       smooth_shading=True,opacity=.7)
    if ensemble is not None:
        for conf in ensemble:
            for i,j,*_ in bonds:
                p.add_mesh(pv.Line(conf[i],conf[j]).tube(radius=.024,n_sides=12),
                           color='#9AA9A0',opacity=.35,smooth_shading=True)
    if bonds:
        for i,j,order in bonds:
            p.add_mesh(pv.Line(x[i],x[j]).tube(radius=.062,n_sides=20),color='#929B9C',smooth_shading=True)
    for i,pos in enumerate(x):
        radius=.20 if elements is None else {'C':.235,'O':.26,'N':.25,'S':.30}.get(elements[i],.22)
        color=ELEMENT.get(elements[i],GREY) if elements is not None else TEAL
        if atom_values is not None:color=GREEN(atom_norm(atom_values[i]))[:3]
        p.add_mesh(pv.Sphere(radius=radius,center=pos,theta_resolution=32,phi_resolution=24),
                   color=color,smooth_shading=True,specular=.18,specular_power=18,
                   ambient=.35,diffuse=.65)
    if highlight is not None:
        for i in highlight:
            p.add_mesh(pv.Sphere(radius=.32,center=x[i],theta_resolution=20,phi_resolution=14),
                       style='wireframe',color=INK,line_width=.9,opacity=.5)
    allpoints=np.vstack([x,*ensemble]) if ensemble is not None else x
    bound=float(np.max(np.linalg.norm(allpoints,axis=1)))
    aspect=size[0]/size[1]
    # x is the horizontal viewing direction; the common camera gives honest
    # mirror comparisons. Parallel projection avoids depth-dependent scaling.
    eye=np.array([1.5,-1.8,1.4]) if camera=='oblique' else np.array([0,-.65,1.5])
    eye=eye*max(bound,1)
    p.camera_position=[eye,(0,0,0),(0,1,0)]
    p.enable_parallel_projection()
    view=-eye/np.linalg.norm(eye);right=np.cross(view,[0,1,0]);right/=np.linalg.norm(right)
    up=np.cross(right,view);vertical=allpoints@up;horizontal=allpoints@right
    scale=max(.95,(np.max(abs(vertical))+.36)*1.15,
              (np.max(abs(horizontal))+.36)/aspect*1.15,extent or 0)
    p.camera.parallel_scale=scale
    camera={'position':list(p.camera.position),'focal_point':list(p.camera.focal_point),
            'up':list(p.camera.up),'parallel_scale':scale,'window_size':size,
            'projection':'orthographic','one_angstrom_width_fraction':1/(2*scale*aspect)}
    p.show(screenshot=str(cache),auto_close=True)
    dump(meta,camera);MANIFEST['renders'][key]=camera
    return plt.imread(cache),camera


def image_panel(ax,im,meta=None,bar=True):
    ax.imshow(im);ax.set_axis_off()
    if meta and bar:
        length=meta['one_angstrom_width_fraction']
        ax.plot([.065,.065+length],[.065,.065],transform=ax.transAxes,color=INK,lw=1.25,clip_on=False)
        ax.text(.065+length/2,.09,'1 Å',ha='center',va='bottom',transform=ax.transAxes,fontsize=8)


def panel(ax,letter,title):
    ax.set_title(title,loc='left',pad=10)
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-17,10),textcoords='offset points',
                fontweight='bold',fontsize=10.5,ha='left',va='baseline',annotation_clip=False)


def clean(ax,xlabel=None,ylabel=None):
    if xlabel:ax.set_xlabel(xlabel)
    if ylabel:ax.set_ylabel(ylabel)
    ax.tick_params(length=3,pad=3)
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))


def save(fig,key,caption,source,legend=None):
    OUT.mkdir(parents=True,exist_ok=True)
    if legend:
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.005),ncol=min(4,len(legend)),
                   handlelength=1.7,columnspacing=1.3)
    fig.canvas.draw()
    bounds=[]
    renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b=text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            bounds.append({'text':text.get_text(),'box':list(b.bounds),'font_pt':text.get_fontsize()})
    for ext in ('png','pdf','svg'):
        options={'dpi':600}
        if ext=='pdf':options['metadata']={'Title':key,'Author':'Vivaan Patni','Creator':'','Producer':''}
        if ext=='svg':options['metadata']={'Creator':'','Date':None}
        fig.savefig(OUT/f'{key}.{ext}',**options)
    (OUT/f'{key}.txt').write_text(caption,encoding='utf-8')
    dump(OUT/'source_data'/f'{key}.json',source)
    MANIFEST['figures'][key]={'size_inches':fig.get_size_inches().tolist(),
                            'caption':caption,'text_bounds':bounds,
                            'source_sha256':sha(OUT/'source_data'/f'{key}.json')}
    CAPTIONS[key]=caption
    plt.close(fig)
    print(key,flush=True)


def fig01():
    record,x,f,op,c,b,out=case('(S)-naproxen')
    y,rot,center=canonical(x);ym=y*np.array([-1,1,1]);extent=3.5
    mirror=persistent_operator(ym,2.6,3.2);cm,bm=cochains(ym,mirror.small[2],f);om=response(mirror,cm,bm)
    np.testing.assert_allclose(om['signed'],-out['signed'],atol=1e-13)
    fig=plt.figure(figsize=(178/25.4,132/25.4));gs=fig.add_gridspec(2,2,left=.095,right=.97,
                 bottom=.14,top=.91,hspace=.55,wspace=.42,height_ratios=[1.1,1])
    for col,coords in enumerate((y,ym)):
        ax=fig.add_subplot(gs[0,col]);im,meta=viewport(f'01_{col}',coords,record['bonds'],record['elements'],extent=extent)
        image_panel(ax,im,meta);panel(ax,'AB'[col],['(S)-naproxen','Exact geometric mirror'][col])
    ax=fig.add_subplot(gs[1,0]);eig=out['eigenvalues'];idx=np.arange(1,len(eig)+1)
    ax.plot(idx,eig,color=TEAL,lw=1.4,marker='o',ms=3.5)
    ax.plot(idx,om['eigenvalues'],color=INK,lw=0,marker='x',ms=4,mew=.7)
    clean(ax,'Eigenvalue index',r'$\lambda(L_2^{2.6,3.2})$');panel(ax,'C','Persistent spectrum')
    ax=fig.add_subplot(gs[1,1]);yy=np.arange(6)
    labels=[r'$Z,d$',r'$Z,H$',r'$Z,\rho$',r'$d,H$',r'$d,\rho$',r'$H,\rho$']
    for i,(a,bb) in enumerate(zip(out['signed'][:,0],om['signed'][:,0])):
        ax.plot([a,bb],[i,i],color=RULE,lw=1)
    ax.scatter(out['signed'][:,0],yy,color=TEAL,s=22,zorder=3)
    ax.scatter(om['signed'][:,0],yy,color=INK,marker='x',s=22,lw=.8,zorder=3)
    ax.axvline(0,color=RULE,lw=.7);ax.set_yticks(yy,labels);ax.invert_yaxis()
    ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2),useMathText=True)
    ax.set_xlabel(r'Signed readout $b^\top u$');panel(ax,'D','Odd readouts')
    handles=[Line2D([],[],color=TEAL,marker='o',ls='none',label='Original'),
             Line2D([],[],color=INK,marker='x',ls='none',label='Mirror')]
    caption=('Figure 1. Exact reflection of one neutral (S)-naproxen conformer (PubChem CID 156391). '
      'A,B, Heavy-atom coordinates generated by ETKDGv3/MMFF94s; bonds are the molecular graph. '
      'The views use the same orthographic camera and physical scale. C, All degree-two persistent '
      'Laplacian eigenvalues at r=2.6 Å, s=3.2 Å. D, The six even probes paired with the first odd source, '
      'with bandwidth 2.4 Å and regularization 0.3. Z, d, H and ρ are scaled atomic number, heavy-atom '
      'degree, total hydrogen count and radial field. Reflection preserves distances and spectrum and '
      'reverses the signed readouts. This is a generated geometry control, not an assay result. '
      'Atoms: carbon graphite, nitrogen blue, oxygen muted red, sulfur ochre; hydrogen atoms are omitted.')
    save(fig,'Figure_01_Molecular_reflection',caption,dict(name=record['name'],coordinates=x,
        display_rotation=rot,center=center,bonds=record['bonds'],eigenvalues=eig,mirror_eigenvalues=om['eigenvalues'],
        signed=out['signed'],mirror_signed=om['signed'],distance_error=np.max(abs(distances(y)-distances(ym)))),handles)


def fig02():
    record,x,f,_,_,_,_=case('L-alanine');y,rot,center=canonical(x)
    radii=[2.0,2.4,2.8,3.2]
    fig=plt.figure(figsize=(178/25.4,140/25.4));gs=fig.add_gridspec(3,2,left=.105,right=.97,
        bottom=.12,top=.92,hspace=.67,wspace=.43,height_ratios=[1,1,.95])
    for k,r in enumerate(radii):
        ax=fig.add_subplot(gs[k//2,k%2]);simp=rips(x,r)
        im,meta=viewport(f'02_{k}',y,record['bonds'],record['elements'],triangles=simp[2],edges=simp[1],
                         extent=2.0,face_opacity=.28)
        image_panel(ax,im,meta);panel(ax,'ABCD'[k],f'r = {r:.1f} Å')
    rs=np.linspace(.8,4.,130);counts=[];spectra=[]
    for r in rs:
        op=persistent_operator(x,r,r+.6);counts.append([len(op.small[q]) for q in (1,2,3)])
        ev=np.linalg.eigvalsh(op.matrix);spectra.extend((r,float(v)) for v in ev)
    counts=np.asarray(counts)
    ax=fig.add_subplot(gs[2,0])
    for k,(color,ls) in enumerate([(GREY,'--'),(TEAL,'-'),(INK,':')]):
        ax.step(rs,counts[:,k],where='post',color=color,ls=ls,lw=1.3)
    clean(ax,'Earlier scale r (Å)','Simplex count');panel(ax,'E','Chain-space growth')
    ax=fig.add_subplot(gs[2,1]);sp=np.asarray(spectra)
    ax.scatter(sp[:,0],sp[:,1],s=2.2,color=TEAL,alpha=.5,rasterized=False)
    clean(ax,'Earlier scale r (Å)',r'$\lambda(L_2^{r,r+0.6})$');panel(ax,'F','Spectrum across scale')
    handles=[Line2D([],[],color=GREY,ls='--',label='Edges'),Line2D([],[],color=TEAL,label='Triangles'),
             Line2D([],[],color=INK,ls=':',label='Tetrahedra')]
    caption=('Figure 2. Filtration of a fixed neutral L-alanine conformer (CID 5950). A-D, The same '
      'coordinates and camera at four Vietoris-Rips distance thresholds. Solid molecular bonds come '
      'from the chemical graph; thin proximity edges and translucent triangular faces come from the '
      'distance filtration and are not additional chemical bonds. E, Counts in the earlier complex. '
      'F, Every degree-two persistent eigenvalue at 130 displayed scales with s=r+0.6 Å; eigenvalues '
      'are unordered between scales because the chain space changes. No molecular two-cycle is '
      'inferred merely from a visible ring or overlapping triangular faces.')
    save(fig,'Figure_02_Molecular_filtration',caption,dict(coordinates=x,rotation=rot,center=center,
         radii=radii,scan_r=rs,simplex_counts=counts,spectrum=sp),handles)


def fig03():
    small,large=octahedral_filling();op=persistent_from_complexes(small,large)
    x=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[0,0,0]],float)*1.5
    down=op.b2.T@op.b2;z=op.null_basis[:,0]
    if z[0]<0:z=-z
    assert np.linalg.norm(op.outside@z)<1e-12
    fig=plt.figure(figsize=(178/25.4,135/25.4));gs=fig.add_gridspec(2,2,left=.105,right=.96,
        bottom=.14,top=.91,hspace=.60,wspace=.48,height_ratios=[1,1.2])
    ax=fig.add_subplot(gs[0,0]);im,meta=viewport('03_sphere_oblique',x,triangles=small[2],extent=1.7,face_opacity=.45,camera='oblique')
    image_panel(ax,im,bar=False);panel(ax,'A','Octahedral boundary K')
    ax=fig.add_subplot(gs[0,1]);im,meta=viewport('03_cone_oblique',x,triangles=large[2],extent=1.7,face_opacity=.2,
                         edges=[(i,6) for i in range(6)],camera='oblique')
    image_panel(ax,im,bar=False);panel(ax,'B','Cone filling L')
    ax=fig.add_subplot(gs[1,0]);im=ax.imshow(op.outside,cmap=DIVERGE,vmin=-1,vmax=1,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(8),range(1,9));ax.set_yticks([0,3,6,9,11],[1,4,7,10,12])
    ax.set_xlabel('Tetrahedron index');ax.set_ylabel('Outside-face index');panel(ax,'C','Outside boundary block D')
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.045,ticks=[-1,0,1]);cb.set_label('Boundary coefficient')
    ax=fig.add_subplot(gs[1,1]);e0=np.linalg.eigvalsh(down);e1=np.linalg.eigvalsh(op.matrix);yy=np.arange(8)
    for i in range(8):ax.plot([e0[i],e1[i]],[i,i],color=RULE,lw=.8)
    ax.scatter(e0,yy,s=36,marker='o',facecolors='white',edgecolors=GREY,linewidths=1)
    ax.scatter(e1,yy,s=19,color=TEAL,marker='D')
    ax.set_yticks([0,2,4,6],[1,3,5,7]);ax.set_xlabel('Eigenvalue');ax.set_ylabel('Ordered mode')
    ax.set_xlim(-.25,6.5);panel(ax,'D','Filling lifts the zero mode')
    handles=[Line2D([],[],marker='o',color=GREY,mfc='white',ls='none',label='Down Laplacian'),
             Line2D([],[],marker='D',color=TEAL,ls='none',label='Persistent Laplacian')]
    caption=('Figure 3. Exact persistent filling fixture. K is the eight-face boundary of an octahedron '
      'with an isolated central vertex; L adds the cone of each face to that vertex. This is an abstract '
      'simplicial pair, not a Vietoris-Rips molecular example. A,B, The outer faces and internal cone '
      'faces. C, The 12-by-8 outside-face boundary matrix D has rank 7. Its one-dimensional kernel '
      'allows internal faces to cancel in a linear combination of all eight tetrahedra, although no '
      'single tetrahedron has boundary entirely in K. D, The persistent operator lifts the down '
      'Laplacian zero eigenvalue to 1. The up term uses an orthonormal kernel basis; its trace is 1. '
      'Deleting all tetrahedra with outside faces would incorrectly retain the unfilled two-cycle.')
    save(fig,'Figure_03_Persistent_filling',caption,dict(coordinates=x,small=small,large=large,
         outside=op.outside,null_vector=z,down_eigenvalues=e0,persistent_eigenvalues=e1),handles)


def fig04():
    record,x,f,op,c,b,out=case('(S)-naproxen');y,rot,center=canonical(x)
    raw=b[:,0]*c[:,0];filtered=b[:,0]*out['u'][:,0]
    bound=max(abs(raw).max(),abs(filtered).max());norm=TwoSlopeNorm(0,-bound,bound)
    fig=plt.figure(figsize=(178/25.4,138/25.4));gs=fig.add_gridspec(2,2,left=.105,right=.94,
        bottom=.15,top=.91,hspace=.63,wspace=.50,height_ratios=[1.1,1])
    for col,(vals,title) in enumerate([(raw,'Probe-weighted source'),(filtered,'Probe-weighted response')]):
        ax=fig.add_subplot(gs[0,col]);im,meta=viewport(f'04_{col}',y,record['bonds'],record['elements'],
            triangles=op.small[2],values=vals,norm=norm,extent=3.5,face_opacity=.88)
        image_panel(ax,im,meta);panel(ax,'AB'[col],title)
    cax=fig.add_axes([.35,.565,.35,.012]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=DIVERGE),cax=cax,orientation='horizontal')
    cb.set_label('Signed face contribution',fontsize=8);cb.formatter.set_powerlimits((-2,2));cb.update_ticks()
    ev,q=eigh(op.matrix);a=q.T@c[:,0];u=q.T@out['u'][:,0]
    groups=[]
    for i,value in enumerate(ev):
        if not groups or not np.isclose(value,ev[groups[-1][0]],rtol=1e-8,atol=1e-8):groups.append([i])
        else:groups[-1].append(i)
    unique=np.array([ev[g].mean() for g in groups])
    aw=np.array([np.sum(a[g]**2) for g in groups])/np.sum(a*a)
    uw=np.array([np.sum(u[g]**2) for g in groups])/np.sum(u*u)
    ax=fig.add_subplot(gs[1,0]);ax.scatter(unique,aw,s=22,color=GREY,marker='o',facecolors='none')
    ax.scatter(unique,uw,s=20,color=TEAL,marker='D')
    clean(ax,'Eigenvalue','Fraction of squared norm');panel(ax,'C','Spectral redistribution')
    taus=np.geomspace(.03,30,150);energy=np.array([np.sum(a*a/(ev+t)) for t in taus])
    ax=fig.add_subplot(gs[1,1]);ax.plot(taus,energy,color=TEAL,lw=1.6)
    ax.set_xscale('log');clean(ax,'Regularization τ (log scale)',r'$\chi^\top (L+\tau I)^{-1}\chi$')
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10,numticks=4));panel(ax,'D','Source energy across τ')
    handles=[Line2D([],[],color=GREY,marker='o',mfc='none',ls='none',label='Source'),
             Line2D([],[],color=TEAL,marker='D',ls='none',label='Response')]
    caption=('Figure 4. Propagation on the naproxen complex from Figure 1. A,B, Each triangular face '
      'is colored by bσ χσ or bσ uσ for the first probe and source, using one symmetric numerical '
      'color scale. These products are invariant under a change of storage orientation; the bare '
      'cochain coefficients are not being presented as invariant face colors. All faces are retained, '
      'although depth can occlude them in a static view. C, Eigenspace fractions of the squared '
      'norm of the first source and its resolved response, at τ=0.3. Numerically equal eigenvalues '
      'are grouped within relative/absolute tolerance 1e-8, giving basis-independent total weights. D, The corresponding positive source '
      'energy decreases with regularization. Filtering suppresses high eigenvalues; this standard '
      'resolvent effect is not evidence of improved biological prediction.')
    save(fig,'Figure_04_Source_propagation',caption,dict(coordinates=x,triangles=op.small[2],
        raw_face=raw,response_face=filtered,color_limits=[-bound,bound],eigenvalues=ev,
        grouped_eigenvalues=unique,source_spectral_weight=aw,response_spectral_weight=uw,tau=taus,energy=energy),handles)


def fig05():
    names=['L-alanine','L-cysteine','L-isoleucine','(S)-naproxen'];cases=[case(n) for n in names]
    fractions=[atom_energy(a[-1]['u'],a[3].small[2],len(a[1])) for a in cases]
    fractions=[v/v.sum() for v in fractions];vmax=max(v.max() for v in fractions);norm=Normalize(0,vmax)
    fig=plt.figure(figsize=(178/25.4,150/25.4));gs=fig.add_gridspec(3,2,left=.105,right=.94,bottom=.15,top=.92,
       hspace=.65,wspace=.47,height_ratios=[1,1,1.05])
    sources=[]
    for k,(case_,v) in enumerate(zip(cases,fractions)):
        record,x,f,op,c,b,out=case_;y,rot,center=canonical(x)
        ax=fig.add_subplot(gs[k//2,k%2]);im,meta=viewport(f'05_{k}',y,record['bonds'],record['elements'],
          atom_values=v,atom_norm=norm,highlight=[i for i,_ in record['stereocenters']],extent=3.4 if k==3 else 2.)
        image_panel(ax,im,meta);panel(ax,'ABCD'[k],names[k])
        sources.append(dict(name=record['name'],atom_fraction=v,stereocenters=record['stereocenters']))
    cax=fig.add_axes([.955,.59,.012,.28]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=GREEN),cax=cax)
    cb.set_label('Fraction of response norm²',fontsize=8)
    cb.ax.yaxis.set_ticks_position('left');cb.ax.yaxis.set_label_position('left')
    ax=fig.add_subplot(gs[2,0]);record,x,f,op,c,b,out=cases[1];v=fractions[1];raw=atom_energy(c,op.small[2],len(x));raw/=raw.sum()
    cysteine_source=raw.copy();cysteine_response=v.copy()
    yy=np.arange(len(x));labels=[f'{record["elements"][i]}{i+1}'+(' *' if i in [j for j,_ in record['stereocenters']] else '') for i in yy]
    for i in yy:ax.plot([raw[i],v[i]],[i,i],color=RULE,lw=1)
    ax.scatter(raw,yy,s=19,facecolors='white',edgecolors=GREY,linewidths=.8)
    ax.scatter(v,yy,s=19,color=TEAL,marker='D');ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlim(-.025,.43)
    ax.set_xlabel('Fraction of squared norm');panel(ax,'E','Cysteine atom fractions')
    ax=fig.add_subplot(gs[2,1]);raws=[];fils=[]
    for a,v in zip(cases,fractions):
        record,x,f,op,c,b,out=a;idx=[i for i,_ in record['stereocenters']]
        raw=atom_energy(c,op.small[2],len(x));raw/=raw.sum()
        raws.append(raw[idx].sum()/(len(idx)/len(x)));fils.append(v[idx].sum()/(len(idx)/len(x)))
    yy=np.arange(4)
    for i in yy:ax.plot([raws[i],fils[i]],[i,i],color=RULE,lw=1)
    ax.scatter(raws,yy,s=20,facecolors='white',edgecolors=GREY,linewidths=.8)
    ax.scatter(fils,yy,s=20,color=TEAL,marker='D');ax.set_yticks(yy,['Ala','Cys','Ile','Naproxen']);ax.invert_yaxis()
    ax.axvline(1,color=RULE,ls=':',lw=.8);ax.set_xlabel('Stereocentre enrichment');panel(ax,'F','Annotated-centre enrichment')
    handles=[Line2D([],[],marker='o',color=GREY,mfc='white',ls='none',label='Source'),
             Line2D([],[],marker='D',color=TEAL,ls='none',label='Response')]
    caption=('Figure 5. Orientation-independent atom localization at r=2.6 Å, s=3.2 Å and τ=0.3. '
      'A-D, Each atom receives one third of the sum of squared response coefficients on incident '
      'triangles, summed over the three sources, then divided by the total squared response norm. '
      'All four views use the same color normalization; thin wire shells mark annotated tetrahedral '
      'stereocentres. E, Source and response fractions for every cysteine heavy atom; * marks its '
      'stereocentre. F, Total annotated-stereocentre fraction divided by the uniform atom fraction '
      'for each displayed first conformer. Enrichment is descriptive, not attribution to a molecular '
      'mechanism or proof that the descriptor detects every stereocentre. L-cysteine has R absolute '
      'configuration at its alpha carbon; L does not universally mean S. Isoleucine has two annotated centres.')
    save(fig,'Figure_05_Atom_localization',caption,dict(atoms=sources,cysteine_source_fraction=cysteine_source,
        cysteine_response_fraction=cysteine_response,
        exemplar_source_enrichment=raws,exemplar_response_enrichment=fils,atom_color_limits=[0,vmax]),handles)


def fig06():
    record=RECORDS['L-methionine'];xs=[np.asarray(c['coordinates']) for c in record['conformers']]
    y,rot,center=canonical(xs[0]);ensemble=[aligned(x,xs[0])@rot for x in xs]
    table=pd.read_csv(ROOT/'results/main/molecule_summary.csv')
    fig=plt.figure(figsize=(178/25.4,180/25.4));gs=fig.add_gridspec(2,2,left=.24,right=.97,bottom=.13,top=.92,
        hspace=.38,wspace=.45,height_ratios=[1,2.6])
    ax=fig.add_subplot(gs[0,0]);im,meta=viewport('06_ensemble',y,record['bonds'],record['elements'],ensemble=ensemble[1:],extent=3.1)
    image_panel(ax,im,meta);panel(ax,'A','Methionine conformers')
    ax=fig.add_subplot(gs[0,1]);p=table.raw_conformer_dispersion;f=table.pchr_conformer_dispersion
    lim=max(p.max(),f.max())*1.07
    ax.plot([0,lim],[0,lim],color=LIGHT,lw=.9,ls=':');ax.scatter(p,f,s=19,color=TEAL)
    ax.set_xlim(0,lim);ax.set_ylim(0,lim);clean(ax,'Source dispersion','Response dispersion');panel(ax,'B','Conformer dispersion')
    ax=fig.add_subplot(gs[1,:]);yy=np.arange(len(table))
    for i,row in table.iterrows():ax.plot([row.raw_conformer_dispersion,row.pchr_conformer_dispersion],[i,i],color=RULE,lw=1)
    ax.scatter(table.raw_conformer_dispersion,yy,s=18,facecolors='white',edgecolors=GREY,linewidths=.85,zorder=3)
    ax.scatter(table.same_conformer_dispersion,yy,s=19,color=INK,marker='x',linewidths=.8,zorder=3)
    ax.scatter(table.pchr_conformer_dispersion,yy,s=18,color=TEAL,marker='D',zorder=4)
    ax.set_yticks(yy,table.name);ax.invert_yaxis();ax.set_xlabel('Relative conformer dispersion')
    ax.tick_params(axis='y',length=0,pad=8);panel(ax,'C','All 24 molecules')
    handles=[Line2D([],[],color=GREY,mfc='white',marker='o',ls='none',label='Source'),
             Line2D([],[],color=INK,marker='x',ls='none',label='Same-scale response'),
             Line2D([],[],color=TEAL,marker='D',ls='none',label='Persistent response')]
    caption=('Figure 6. Conformer variation in the frozen 24-molecule pilot. A, Eight independently '
      'generated methionine conformers aligned on the first using translations and proper Kabsch '
      'rotations; pale bonds show the other seven conformers. This is not a dynamical trajectory. '
      'B,C, Molecule-level relative dispersion of the 18 pooled features across eight conformers, '
      'at r=2.6 Å, persistent s=3.2 Å, τ=0.3; the same-scale response uses s=r. Every feature '
      'matrix is divided by the triangle count. Dispersion is median distance from the '
      'coordinatewise median, divided by median vector norm plus 1e-12. All 24 molecules '
      'are displayed in the source-selected order. Lower values mean smaller conformer variation, '
      'not necessarily greater chemical accuracy. The persistent response improves this metric '
      'for 8/24 molecules. Its paired median difference from the source is +0.0592, with a molecule '
      'bootstrap 95% interval [-0.000003, 0.2184] from the frozen 5,000-resample analysis. The '
      'pilot therefore does not establish a general stability advantage.')
    save(fig,'Figure_06_Conformer_comparison',caption,dict(molecules=table.to_dict('records'),
        ensemble=ensemble,alignment='proper Kabsch rotations, first conformer reference'),handles)


def fig07():
    record,x,f,_,_,_,_=case('L-tryptophan')
    # On every filtration interval the matrix and cochain list are constant;
    # the tau sweep is evaluated from one eigendecomposition per scale.
    rs=np.linspace(2.0,3.5,61);taus=np.geomspace(.03,10,60);surface=np.zeros((len(rs),len(taus)))
    counts=[]
    for i,r in enumerate(rs):
        op=persistent_operator(x,r,r+.6);c,b=cochains(x,op.small[2],f);ev,q=eigh(op.matrix)
        a=q.T@c;bb=q.T@b
        for j,t in enumerate(taus):surface[i,j]=np.linalg.norm(bb.T@(a/(ev[:,None]+t)))
        counts.append(len(op.small[2]))
    fig=plt.figure(figsize=(178/25.4,135/25.4))
    ax=fig.add_axes([.07,.50,.52,.42],projection='3d');xx,yy=np.meshgrid(rs,np.log10(taus),indexing='ij')
    ax.plot_surface(xx,yy,surface,cmap=GREEN,linewidth=0,antialiased=True,alpha=1)
    ax.view_init(24,-54);ax.set_xlabel('r (Å)',labelpad=5);ax.set_ylabel(r'$\log_{10}\tau$',labelpad=9)
    ax.set_zlabel(r'$\|B^\top U\|_F$',labelpad=3);ax.tick_params(labelsize=8,pad=0)
    ax.set_yticks([-1,0,1]);ax.set_zticks([0,.01,.02])
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.pane.fill=False;axis._axinfo['grid']['color']=(.88,.89,.88,.5)
    fig.text(.075,.93,'A',fontweight='bold',fontsize=10.5)
    fig.text(.115,.93,'Response landscape',fontweight='bold',fontsize=9.5)
    ax=fig.add_axes([.67,.57,.27,.31]);im=ax.pcolormesh(rs,taus,surface.T,cmap=GREEN,shading='nearest',rasterized=True)
    ax.set_yscale('log');ax.set_xlabel('r (Å)');ax.set_ylabel('τ (log scale)');panel(ax,'B','Same values in 2D')
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.07,pad=.30);cb.set_label(r'$\|B^\top U\|_F$')
    ax=fig.add_axes([.13,.16,.81,.23]);j=int(np.argmin(abs(taus-.3)))
    ax.step(rs,surface[:,j],where='mid',color=TEAL,lw=1.5)
    ax.set_xlabel('Earlier scale r (Å)');ax.set_ylabel(r'$\|B^\top U\|_F$');panel(ax,'C',f'Fixed τ = {taus[j]:.3f}')
    caption=('Figure 7. Parameter sensitivity of the first generated tryptophan conformer. A,B, '
      'Frobenius norm of the 6-by-3 signed-readout matrix over 61 earlier scales and 60 positive '
      'regularization values, with later scale s=r+0.6 Å and source bandwidth 2.4 Å. The 3D surface '
      'connects evaluated grid values only to display the landscape; the true hard-filtration '
      'descriptor can jump at distance events. B displays identical values without perspective. '
      'C, The grid section nearest τ=0.3; its actual regularization is printed in the panel title. '
      'The norm is reflection-even and loses sign, although its underlying pooled components '
      'are odd. No robustness or biological performance optimum is inferred from this surface.')
    save(fig,'Figure_07_Parameter_landscape',caption,dict(name=record['name'],r=rs,tau=taus,
        signed_matrix_norm=surface,triangle_counts=counts,section_tau=taus[j]))


def fig08():
    record,x,f,_,_,_,_=case('L-alanine')
    # A deterministic coordinate direction, not an energy minimization or MD.
    v=np.zeros_like(x);v[0]=[.65,-.2,.25]
    tt=np.linspace(-.22,.22,65);hard=[];avg=[];inside=[];boundary=[]
    for t in tt:
        coords=x+t*v;op=persistent_operator(coords,2.6,3.2);c,b=cochains(coords,op.small[2],f)
        hard.append(response(op,c,b)['signed'][0,0])
        a=scale_average(coords,f,2.35,2.85,.6,direction=v)
        avg.append(a.signed[0,0]);inside.append(a.interior_derivative[0,0]);boundary.append(a.boundary_derivative[0,0])
    fig=plt.figure(figsize=(178/25.4,135/25.4));gs=fig.add_gridspec(2,2,left=.105,right=.97,
        bottom=.15,top=.91,hspace=.65,wspace=.45)
    y,rot,center=canonical(x)
    ax=fig.add_subplot(gs[0,0]);im,meta=viewport('08_displacement_marked',y,record['bonds'],record['elements'],
         ensemble=[(x+.22*v-center)@rot,(x-.22*v-center)@rot],highlight=[0],extent=2.)
    image_panel(ax,im,meta);panel(ax,'A','Single-atom displacement')
    ax=fig.add_subplot(gs[0,1]);ax.plot(tt,hard,color=GREY,lw=1.25,ls='--');ax.plot(tt,avg,color=TEAL,lw=1.5)
    clean(ax,'Displacement parameter t',r'$b^\top u$');panel(ax,'B','Hard and averaged response')
    ax=fig.add_subplot(gs[1,:]);total=np.array(inside)+np.array(boundary)
    ax.plot(tt,total,color=TEAL,lw=1.6);ax.plot(tt,inside,color=GREY,ls='--',lw=1.2)
    ax.plot(tt,boundary,color=INK,ls=':',lw=1.3)
    clean(ax,'Displacement parameter t',r'$d\bar p/dt$');panel(ax,'C','Moving filtration boundaries')
    handles=[Line2D([],[],color=GREY,ls='--',label='Hard / interior'),
             Line2D([],[],color=TEAL,label='Average / total'),
             Line2D([],[],color=INK,ls=':',label='Boundary motion')]
    caption=('Figure 8. Filtration-event sensitivity on a generated alanine geometry. A, The first '
      'heavy atom, marked with a wire shell, is displaced along the fixed vector (0.65, -0.20, 0.25) Å multiplied by t; pale '
      'bonds show the endpoints. This is a coordinate sensitivity experiment, not molecular '
      'dynamics. B, First probe/source readout at r=2.6 Å and its uniform average over '
      'r in [2.35,2.85] Å, with s=r+0.6 Å and τ=0.3. The average is integrated exactly between '
      'distance and shifted-distance events, up to floating arithmetic. C, The analytic derivative '
      'of the average, decomposed into within-complex and moving-event-boundary terms. Ignoring '
      'the latter gives an incomplete derivative. Values at tied or endpoint events are not '
      'assigned a derivative by this routine.')
    save(fig,'Figure_08_Filtration_calculus',caption,dict(t=tt,direction=v,hard=hard,averaged=avg,
        interior_derivative=inside,boundary_derivative=boundary,total_derivative=total),handles)


def supp01():
    maps=json.loads((ROOT/'results/main/atom_maps.json').read_text());first={r['name']:r for r in maps if r['conformer_id']==0}
    vals=[np.asarray(first[r['name']]['atom_energy']) for r in DATA];vals=[v/v.sum() for v in vals]
    norm=Normalize(0,max(v.max() for v in vals))
    fig,axes=plt.subplots(6,4,figsize=(178/25.4,225/25.4));fig.subplots_adjust(left=.025,right=.975,bottom=.105,top=.97,hspace=.30,wspace=.02)
    sources=[]
    for k,(record,v,ax) in enumerate(zip(DATA,vals,axes.ravel())):
        x=np.asarray(record['conformers'][0]['coordinates']);y,rot,center=canonical(x)
        im,meta=viewport(f'S01_{k}',y,record['bonds'],record['elements'],atom_values=v,atom_norm=norm,size=(840,580))
        image_panel(ax,im,meta,bar=False);ax.set_title(record['name'],fontsize=8,pad=2,loc='center')
        sources.append(dict(name=record['name'],cid=record['pubchem_cid'],coordinates=x,
                            elements=record['elements'],bonds=record['bonds'],atom_fractions=v))
    cax=fig.add_axes([.34,.055,.32,.010]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=GREEN),cax=cax,orientation='horizontal')
    cb.set_label('Fraction of response norm²',fontsize=8);cb.ax.tick_params(labelsize=8)
    caption=('Supplementary Figure S1. Complete 24-molecule first-conformer atlas in the frozen '
      'source-selected order. Atom color shows the fraction of the total squared persistent-response '
      'norm assigned to that atom at r=2.6 Å, s=3.2 Å, τ=0.3. One shared color scale is used across '
      'all molecules. Molecular views are individually fitted for legibility, so their apparent '
      'sizes are not a physical cross-molecule size comparison. The exact coordinates, bonds, '
      'chemical identities and per-atom fractions are provided in source data. These are '
      'ETKDGv3/MMFF94s generated neutral conformers, not deposited experimental structures or '
      'physiological protonation states. No cases are omitted according to response quality.')
    save(fig,'Figure_S01_Full_molecular_atlas',caption,dict(molecules=sources,color_limits=[0,norm.vmax]))


def fig09():
    geometry=json.loads((ROOT/'results/geometry/intrinsic_sensitivity.json').read_text())
    records=geometry['records'];item=next(r for r in records if r['name']=='L-methionine');torsion=item['torsion']
    record,x,f,op,c,b,out=case('L-methionine');y,rot,center=canonical(x)
    ensemble=[(bond_rotation(x,record['bonds'],(2,3),0,angle)[0]-center)@rot for angle in (-.4,.4)]
    limit=torsion['minimum_distance_event_margin']/(12*np.linalg.norm(np.asarray(torsion['velocity']),axis=1).max())
    tt=np.linspace(-limit,limit,61);observed=[]
    for t in tt:
        xx=bond_rotation(x,record['bonds'],(2,3),0,float(t))[0]
        oo=persistent_operator(xx,2.6,3.2)
        assert oo.small==op.small and oo.large==op.large
        cc,bb=cochains(xx,oo.small[2],f)
        pp=response(oo,cc,bb)['signed'][0,0]
        observed.append(pp-item['readout']-t*torsion['derivative'])
    full=.5*tt**2*torsion['total_second_derivative'];hessian=.5*tt**2*torsion['coordinate_hessian_term']
    fig=plt.figure(figsize=(178/25.4,140/25.4));gs=fig.add_gridspec(2,2,left=.18,right=.96,bottom=.15,top=.91,
      hspace=.65,wspace=.92,height_ratios=[1.08,1])
    ax=fig.add_subplot(gs[0,0]);im,meta=viewport('09_torsion',y,record['bonds'],record['elements'],ensemble=ensemble,
                       highlight=[2,3],extent=3.2)
    image_panel(ax,im,meta);panel(ax,'A','Methionine bond rotation')
    ax=fig.add_subplot(gs[0,1]);yy=np.arange(4);values=[r['intrinsic_gradient_norm'] for r in records]
    ax.scatter(values,yy,s=23,color=TEAL);ax.set_yticks(yy,['Alanine','Cysteine','Methionine','Naproxen']);ax.invert_yaxis()
    ax.set_xlim(0,.003);ax.set_xticks([0,.001,.002,.003]);ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2),useMathText=True)
    ax.set_xlabel(r'$\|\mathrm{grad}_{\mathcal{S}}\,p\|$');panel(ax,'B','Intrinsic sensitivity')
    ax=fig.add_subplot(gs[1,0]);values=[torsion['coordinate_hessian_term'],torsion['acceleration_term'],torsion['total_second_derivative']]
    yy=np.arange(3);ax.axvline(0,color=RULE,lw=.8)
    ax.scatter(values[:2],yy[:2],s=24,color=GREY,marker='o');ax.scatter(values[2:],yy[2:],s=24,color=TEAL,marker='D')
    ax.set_yticks(yy,['Hessian term','Acceleration term','Full curvature']);ax.invert_yaxis()
    ax.set_xlim(-.00037,.00029);ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2),useMathText=True)
    ax.set_xlabel(r'$d^2p/d\theta^2$ (rad$^{-2}$)');panel(ax,'C','Torsional curvature')
    ax=fig.add_subplot(gs[1,1]);ax.plot(tt*1000,np.asarray(observed)/1e-10,color=TEAL,lw=1.5)
    ax.plot(tt*1000,hessian/1e-10,color=GREY,ls='--',lw=1.2)
    ax.plot(tt*1000,full/1e-10,color=INK,ls=':',lw=1.15)
    ax.set_xlabel('Bond rotation θ (mrad)');ax.set_ylabel(r'Nonlinear readout ($10^{-10}$)')
    panel(ax,'D','Nonlinear response')
    handles=[Line2D([],[],color=TEAL,label='Computed torsion response'),
             Line2D([],[],color=GREY,ls='--',label='Hessian term alone'),
             Line2D([],[],color=INK,ls=':',label='Full quadratic expansion')]
    caption=('Figure 9. Differential geometry of the molecular response, using the first source '
      'and probe at r=2.6 Å, s=3.2 Å, τ=0.3. A, Methionine with the terminal methyl-sulfur component '
      'rotated about the bridge bond joining heavy-atom indices 2 and 3 (zero based); wire shells '
      'mark the axis atoms and pale bonds show rotations of ±0.4 radians for illustration. All '
      'graph bond lengths are preserved. B, Norm of the mass-metric shape gradient after removing '
      'translation and rotation, for the four prespecified first conformers; weights are standard '
      'heavy-atom atomic weights and the unit is readout per Å per square-root atomic mass unit. '
      'C, Methionine curvature decomposes into a coordinate Hessian term and a path-acceleration '
      'term. Omitting the latter reverses its sign. D, The nonlinear part after removing the '
      'constant and linear response, within a much smaller angle window chosen to remain in '
      'one filtration chamber; the computed curve agrees with the full local quadratic expansion. '
      'A geometric torsion is not a molecular-dynamics trajectory or a claim about occupied rotamers.')
    save(fig,'Figure_09_Intrinsic_geometry',caption,dict(geometry=geometry,theta=tt,
      nonlinear_readout=observed,full_quadratic=full,hessian_only_quadratic=hessian,
      maximum_angle=limit,display_angles=[-.4,.4],bonds=record['bonds']),handles)


def supp02():
    table=pd.read_csv(ROOT/'results/main/molecule_summary.csv')
    allrows=pd.read_csv(ROOT/'results/main/all_results.csv')
    fig=plt.figure(figsize=(178/25.4,160/25.4));gs=fig.add_gridspec(1,2,left=.26,right=.97,bottom=.16,top=.91,wspace=.35,width_ratios=[1.2,1])
    ax=fig.add_subplot(gs[0,0]);yy=np.arange(24)
    for i,row in table.iterrows():ax.plot([row.median_source_enrichment,row.median_stereocenter_enrichment],[i,i],color=RULE,lw=.9)
    ax.scatter(table.median_source_enrichment,yy,s=20,facecolors='white',edgecolors=GREY,lw=.8)
    ax.scatter(table.median_stereocenter_enrichment,yy,s=20,color=TEAL,marker='D')
    ax.axvline(1,color=RULE,ls=':',lw=.8);ax.set_yticks(yy,table.name);ax.invert_yaxis();ax.tick_params(axis='y',length=0,pad=7)
    ax.set_xlabel('Median stereocentre enrichment');panel(ax,'A','All molecules')
    ax=fig.add_subplot(gs[0,1]);col='response_source_cosine'
    if col not in allrows.columns:col=next(c for c in allrows.columns if 'cosine' in c)
    values=allrows[col].to_numpy();values=np.sort(values[np.isfinite(values)])
    ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=TEAL,lw=1.5)
    ax.set_xlabel('Response-source cosine');ax.set_ylabel('Cumulative fraction of configurations')
    ax.set_ylim(0,1);ax.set_xlim(.75,1.005);panel(ax,'B','Source-response alignment')
    handles=[Line2D([],[],color=GREY,marker='o',mfc='white',ls='none',label='Source'),
             Line2D([],[],color=TEAL,marker='D',ls='none',label='Persistent response')]
    caption=('Supplementary Figure S2. Localization and source alignment in the pilot. A, Median '
      'stereocentre enrichment across eight conformers per molecule, using all three source channels '
      'at r=2.6 Å, s=3.2 Å, τ=0.3. All 24 molecules are shown; the vertical reference is uniform atom '
      'allocation. The paired median response-minus-source difference is 0.0052 with a molecule '
      'bootstrap 95% interval [-0.1094,0.0484]; an enrichment advantage is not established. '
      'B, Response-source Frobenius cosine for configurations with a nonzero source. Empty '
      'triangle spaces have no defined cosine and are excluded from this ECDF; the exact '
      'denominator is recorded in source data. Strong alignment means the filter often preserves '
      'much of the source direction and does not itself establish added biological information.')
    values=values[np.isfinite(values)]
    save(fig,'Figure_S02_Localization_and_alignment',caption,dict(molecules=table.to_dict('records'),
        response_source_cosine=values,n_finite=len(values),n_total=len(allrows)),handles)


def main():
    global OUT,CACHE
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--only',nargs='*');args=ap.parse_args();OUT=args.output;CACHE=args.cache;CACHE.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():MANIFEST.update(json.loads((OUT/'manifest.json').read_text()))
    for p in [ROOT/'data/conformers.json',ROOT/'results/main/molecule_summary.csv',ROOT/'results/main/summary.json',Path(__file__)]:
        MANIFEST['input_sha256'][str(p.relative_to(ROOT))]=sha(p)
    MANIFEST['software']={'matplotlib':matplotlib.__version__,'pyvista':pv.__version__,'numpy':np.__version__}
    functions={'01':fig01,'02':fig02,'03':fig03,'04':fig04,'05':fig05,'06':fig06,'07':fig07,'08':fig08,'09':fig09,'S01':supp01,'S02':supp02}
    for name,fn in functions.items():
        if not args.only or name in args.only:
            fn();dump(OUT/'manifest.json',MANIFEST)


if __name__=='__main__':main()
