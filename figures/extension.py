"""Six source-backed figures for the single ensemble-stereochemistry extension."""
from pathlib import Path
import argparse,json,hashlib
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from scipy.linalg import eigh
from rdkit import Chem
import rebuild as v
from pchr.ensemble import odd_kernel,odd_witness,odd_pullback_metric
from pchr.ensemble_diagnostics import mirror_smiles
from pchr.shape_geometry import bond_rotation

ROOT=v.ROOT;EXT=ROOT/'results/external_full';BIO=ROOT/'results/biology_cliffs'
v.ELEMENT.update({'F':'#6B9B58','Cl':'#5A9A54','Br':'#95634D','I':'#7C69A4','P':'#BD8E3D'})
LABELS={'source':'Odd source','persistent':'Persistent response','same_scale':'Same-scale response',
 'distance_histogram':'Distance histogram','gemnet':'GemNet','mace':'MACE','fmg':'FMG'}


def load(path):return json.loads(path.read_text())


def interval(values,seed=28102026):
    values=np.asarray(values,float);rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(values,size=(5000,len(values)),replace=True),axis=1)
    return float(np.mean(values)),np.quantile(boot,[.025,.975])


def native_case():
    selected=load(ROOT/'data/extension_selection.json')['selected']
    for p in selected:
        parent=load(EXT/'parents'/(p['mol_id']+'.json'));records=parent.get('records',[])
        labels=sorted({r['en_id'] for r in records})
        for a,b in combinations(labels,2):
            ra=[r for r in records if r['en_id']==a];rb=[r for r in records if r['en_id']==b]
            if mirror_smiles(ra[0]['stereo_smiles'])==rb[0]['stereo_smiles']:
                return parent,ra,rb
    raise ValueError('no enantiomer case')


def case_projection(a,b):
    x=np.array([r['r2.6']['persistent'] for r in a+b]);scale=max(np.sqrt(np.mean(np.sum(x*x,axis=1))),1e-12)
    x/=scale;_,_,vh=np.linalg.svd(x,full_matrices=False);basis=vh[:2].T
    return x[:len(a)],x[len(a):],basis,x@basis


def fig10():
    parent,a,b=native_case();fig=plt.figure(figsize=(178/25.4,156/25.4))
    gs=fig.add_gridspec(2,2,left=.145,right=.97,bottom=.13,top=.92,hspace=.60,wspace=.48,height_ratios=[1,1.10])
    ref=np.array(a[0]['coordinates']);coords,rot,center=v.canonical(ref)
    ensemble=[v.aligned(np.array(r['coordinates']),ref)@rot for r in a[1:]]
    ax=fig.add_subplot(gs[0,0]);im,meta=v.viewport('ext10_native',coords,a[0]['bonds'],a[0]['elements'],ensemble=ensemble,
        highlight=[i for i,_ in a[0]['stereocentres']],size=(1900,1300))
    v.image_panel(ax,im,meta);v.panel(ax,'A','External conformer ensemble')
    xa,xb,basis,proj=case_projection(a,b)
    ax=fig.add_subplot(gs[0,1]);ax.axhline(0,color=v.RULE,lw=.6);ax.axvline(0,color=v.RULE,lw=.6)
    ax.scatter(proj[:len(a),0],proj[:len(a),1],s=30,color=v.BLUE,marker='o')
    ax.scatter(proj[len(a):,0],proj[len(a):,1],s=34,color=v.ORANGE,marker='^')
    v.clean(ax,'Projection 1','Projection 2');v.panel(ax,'B','Native enantiomer supports')
    table=pd.read_csv(EXT/'separation.csv');primary=table[table.radius=='r2.6'];order=['distance_histogram','source','fmg','same_scale','persistent','gemnet','mace']
    ax=fig.add_subplot(gs[1,0]);data=[]
    for i,name in enumerate(order):
        vals=primary[primary.representation==name].es_auc.to_numpy();mean,ci=interval(vals)
        color=v.TEAL if name=='persistent' else v.GREY
        ax.plot(ci,[i,i],color=color,lw=1.3);ax.scatter([mean],[i],s=23,color=color,marker='D' if name=='persistent' else 'o')
        data.append({'representation':name,'mean':mean,'interval95':ci.tolist()})
    ax.axvline(.5,color=v.RULE,ls=':',lw=1);ax.set_xlim(.42,.96);ax.set_yticks(range(len(order)),['Distance only','Odd source','FMG','Same-scale','Persistent','GemNet','MACE']);ax.invert_yaxis();ax.tick_params(axis='y',length=0)
    ax.set_xlabel('Mean parent ES-AUC');v.panel(ax,'C','Same 512 molecular parents')
    ax=fig.add_subplot(gs[1,1]);wide=primary.pivot(index='mol_id',columns='representation',values='es_auc')
    ax.scatter(wide.source,wide.persistent,s=10,alpha=.40,color=v.TEAL,linewidths=0)
    ax.plot([0,1],[0,1],color=v.LIGHT,lw=.8);ax.set_xlim(-.03,1.03);ax.set_ylim(-.03,1.03)
    v.clean(ax,'Odd-source ES-AUC','Persistent-response ES-AUC');v.panel(ax,'D','Paired propagation comparison')
    legend=[Line2D([],[],ls='none',marker='o',color=v.BLUE,label='Native isomer A'),Line2D([],[],ls='none',marker='^',color=v.ORANGE,label='Native isomer B')]
    caption=('Figure 10. External stereoisomer discrimination. A, The first eligible enantiomer example in the frozen parent-hash order ('+parent['mol_id']+'), showing its first conformer and the other conformers of that isomer after proper Kabsch alignment; wire shells mark assigned tetrahedral centres. B, The two native supports in an uncentered two-axis SVD projection of their 18-component persistent readouts; this visualization retains an origin and does not enter any score. C, Equal-parent mean ES-AUC and 5,000-resample parent bootstrap 95% intervals on 512 parents and 5,514 conformers, at r=2.6 Å, s=3.2 Å, tau=0.3. Published GemNet, MACE and FMG embeddings use the identical source offset indices. D, Every parent in the source versus response comparison; overlapping points are retained. Persistent-minus-source mean ES-AUC is -0.0425, interval [-0.0523,-0.0326]. Native conformer differences can make an even distance representation discriminate stereoisomer labels; this does not establish reflection sensitivity. The cohort includes diastereomers and generated conformer sets, not binding measurements.')
    v.save(fig,'Figure_10_External_ensembles',caption,{'case_parent':parent['mol_id'],'case_a':a,'case_b':b,'projection':proj,'basis':basis,'mean_intervals':data,'paired_scores':wide.reset_index().to_dict('records')},legend)


def fig11():
    parent,a,b=native_case();xa,xb,basis,projection=case_projection(a,b)
    bound=max(np.abs(projection).max()*1.35,.8);axis=np.linspace(-bound,bound,75);xx,yy=np.meshgrid(axis,axis)
    query=np.c_[xx.ravel(),yy.ravel()]@basis.T
    odd=(odd_kernel(query,xa).mean(axis=1)-odd_kernel(query,xb).mean(axis=1)).reshape(xx.shape)
    fig=plt.figure(figsize=(178/25.4,160/25.4));gs=fig.add_gridspec(2,2,left=.12,right=.96,bottom=.14,top=.91,hspace=.65,wspace=.48)
    ax=fig.add_subplot(gs[0,0],projection='3d');ax.plot_surface(xx,yy,odd,cmap=v.DIVERGE,linewidth=0,rasterized=True,antialiased=True,alpha=.95)
    ax.set_xlabel('Projection 1',labelpad=6);ax.set_ylabel('Projection 2',labelpad=14);ax.set_zlabel('Odd witness',labelpad=7)
    ax.set_xticks([-bound,0,bound],['-b','0','b']);ax.set_yticks([-bound,0,bound],['-b','0','b']);ax.locator_params(axis='z',nbins=3);ax.view_init(26,-55);ax.set_box_aspect((1.1,1,.65));v.panel(ax,'A','Odd witness section')
    ax=fig.add_subplot(gs[0,1]);maximum=max(abs(odd.min()),abs(odd.max()));im=ax.pcolormesh(xx,yy,odd,cmap=v.DIVERGE,vmin=-maximum,vmax=maximum,shading='nearest',rasterized=True)
    ax.contour(xx,yy,odd,levels=[0],colors=[v.GREY],linewidths=.7)
    ax.scatter(projection[:len(a),0],projection[:len(a),1],s=27,facecolor='white',edgecolor=v.BLUE,marker='o',lw=1)
    ax.scatter(projection[len(a):,0],projection[len(a):,1],s=30,facecolor='white',edgecolor=v.ORANGE,marker='^',lw=1)
    ax.set_xlabel('Projection 1');ax.set_ylabel('Projection 2');ax.set_aspect('equal');v.panel(ax,'B','Same values without perspective')
    cax=fig.add_axes([.65,.545,.26,.012]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_label('Witness value',labelpad=2);cb.ax.tick_params(labelsize=8)
    frame=pd.read_csv(EXT/'parity_decomposition.csv');frame=frame[frame.radius=='r2.6'];order=['source','persistent','distance_histogram']
    means=frame.groupby(['mol_id','representation'])[['odd_squared','even_squared']].mean().groupby('representation').mean()
    ax=fig.add_subplot(gs[1,0]);pos=np.arange(len(order));ax.bar(pos,[means.loc[n,'even_squared'] for n in order],color=v.LIGHT,width=.56)
    ax.bar(pos,[means.loc[n,'odd_squared'] for n in order],bottom=[means.loc[n,'even_squared'] for n in order],color=v.TEAL,width=.56)
    ax.set_xticks(pos,['Source','Persistent','Distance']);ax.set_ylabel('Mean squared MMD');v.panel(ax,'C','Even and odd ensemble disparity')
    ax=fig.add_subplot(gs[1,1]);fractions=[]
    for name,style,color in [('source','--',v.GREY),('persistent','-',v.TEAL)]:
        sub=frame[frame.representation==name].copy();sub['fraction']=sub.odd_squared/sub.total_squared.replace(0,np.nan)
        values=np.sort(sub.groupby('mol_id').fraction.mean().dropna().to_numpy());ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',ls=style,color=color,label=LABELS[name]);fractions.append({'representation':name,'parent_fractions':values})
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Odd fraction of squared MMD');ax.set_ylabel('Cumulative parent fraction');v.panel(ax,'D','Native enantiomer pairs')
    handles=[Line2D([],[],color=v.LIGHT,lw=6,label='Mirror-even disparity'),Line2D([],[],color=v.TEAL,lw=6,label='Mirror-odd disparity'),Line2D([],[],color=v.GREY,ls='--',label='Odd source')]
    caption=('Figure 11. Post-primary parity decomposition of native enantiomer ensembles. A,B, The odd Gaussian-kernel witness contrast between the same two supports as Figure 10, evaluated on the displayed two-dimensional slice through the full 18-dimensional normalized descriptor space (sigma=1). Markers in B are orthogonal projections; the full vectors define the kernel. b is the symmetric plot bound recorded in source data. C, Mean squared Gaussian MMD split into orthogonal mirror-even and mirror-odd RKHS contributions, using 465 verified tetrahedral enantiomer pairs from 337 parents; pair values are averaged within each parent before the displayed parent means. D, Parent ECDF of the odd fraction. Balancing each pair as (mu+mirror(nu))/2 and (nu+mirror(mu))/2 preserves exactly the odd contribution; even distance-histogram laws become identical under that operation. Unequal conformer counts retain one-half mixture weights. The projection identity holds to 6.7e-16. This diagnostic was added after the primary test and is not presented as a confirmatory endpoint or an equilibrium-distribution estimate.')
    v.save(fig,'Figure_11_Parity_geometry',caption,{'parent':parent['mol_id'],'projection_basis':basis,'projection':projection,'plot_bound':bound,'grid':axis,'witness':odd,'parent_mean_energies':means.reset_index().to_dict('records'),'parent_odd_fractions':fractions},handles)


def fig12():
    fixture=load(EXT/'ensemble_geometry.json')['fixture'];points=np.array(fixture['points']);g=np.array(fixture['gram']);lam=np.array(fixture['lambda']);score=np.array(fixture['scores'])
    evals,vecs=np.linalg.eigh(g);phi=vecs@np.diag(np.sqrt(np.maximum(evals,0)));mean=phi.mean(axis=0);center=phi-mean
    cov=center.T@center/3;rho,axes=np.linalg.eigh(cov);weights=(axes.T@mean)**2
    fig=plt.figure(figsize=(178/25.4,146/25.4));gs=fig.add_gridspec(2,2,left=.13,right=.96,bottom=.14,top=.91,hspace=.62,wspace=.46)
    ax=fig.add_subplot(gs[0,0]);im=ax.imshow(g,cmap=v.DIVERGE,vmin=-.5,vmax=.5)
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{g[i,j]:.3f}',ha='center',va='center',fontsize=8,color=v.INK)
    ax.set_xticks(range(3),['-3','1','2']);ax.set_yticks(range(3),['-3','1','2']);ax.set_xlabel('Support point');ax.set_ylabel('Support point');v.panel(ax,'A','Odd Gram matrix')
    ax=fig.add_subplot(gs[0,1]);ax.semilogx(lam,lam*score,color=v.TEAL)
    ax.axhline(fixture['divergence_coefficient'],color=v.ORANGE,ls='--',lw=1)
    ax.set_xlabel(r'Ensemble regularization $\lambda$');ax.set_ylabel(r'$\lambda S_\lambda$');v.panel(ax,'B','Sharp small-regularization limit')
    ax=fig.add_subplot(gs[1,0]);parts=[]
    for i,(value,weight) in enumerate(zip(rho,weights)):
        component=weight/(max(value,0)+lam);parts.append(component)
        ax.loglog(lam,component,color=v.ORANGE if i==0 else v.GREY,ls='-' if i==0 else ['--',':'][i-1],label='Null covariance mode' if i==0 else f'Positive mode {i}')
    ax.set_xlabel(r'Ensemble regularization $\lambda$');ax.set_ylabel('Spectral contribution');v.panel(ax,'C','Where the divergence comes from')
    ax=fig.add_subplot(gs[1,1]);z=np.linspace(-4,4,301);witness=odd_witness(z[:,None],points)
    ax.plot(z,witness,color=v.TEAL);ax.axhline(0,color=v.RULE,lw=.7);ax.axvline(0,color=v.LIGHT,ls=':',lw=.8)
    ax.scatter(points[:,0],odd_witness(points,points),color=v.INK,s=22,zorder=3)
    ax.set_xlabel('One-dimensional feature coordinate');ax.set_ylabel('Odd witness');v.panel(ax,'D','Zero mean, nonzero odd law')
    handles=[Line2D([],[],color=v.TEAL,label=r'$\lambda S_\lambda$ / odd witness'),Line2D([],[],color=v.ORANGE,ls='--',label='Exact limiting coefficient'),Line2D([],[],color=v.ORANGE,label='Null covariance mode')]
    caption=('Figure 12. A finite-support warning derived from an exact abstract descriptor law. The three equally weighted points are -3, 1 and 2, with Euclidean mean zero. A, Its positive-definite odd Gaussian Gram matrix at sigma=1. B, The scaled centered-covariance resolvent score tends to the explicit coefficient 1/(1^T G^-1 1). C, The covariance-null component creates a 1/lambda divergence; the positive covariance modes remain bounded. D, The odd kernel witness detects reflection asymmetry despite cancellation of the Euclidean mean. This is an abstract example, not an asserted molecular realization. The same finite-support mechanism can inflate an empirical conformer score as regularization decreases, so that score alone is not evidence of physical rigidity, statistical generalization or biological selectivity.')
    v.save(fig,'Figure_12_Finite_support_limits',caption,{'fixture':fixture,'covariance_eigenvalues':rho,'mean_spectral_weights':weights,'spectral_components':parts,'witness_grid':z,'witness':witness},handles)


def fig13():
    data=load(EXT/'ensemble_geometry.json');x=np.array(data['coordinates']);y,rot,center=v.canonical(x)
    ghosts=[(bond_rotation(x,data['bonds'],data['axis_edge'],data['moving_vertex'],a)[0]-center)@rot for a in [-.3,.3]]
    fig=plt.figure(figsize=(178/25.4,154/25.4));gs=fig.add_gridspec(2,2,left=.14,right=.96,bottom=.15,top=.91,hspace=.65,wspace=.47)
    ax=fig.add_subplot(gs[0,0]);im,meta=v.viewport('ext13_torsion_oblique',y,data['bonds'],data['elements'],ensemble=ghosts,highlight=data['axis_edge'],size=(1900,1250),camera='oblique')
    v.image_panel(ax,im,meta);v.panel(ax,'A','One conformer in the ensemble')
    ax=fig.add_subplot(gs[0,1],projection='3d');tau=np.array(data['tau_grid']);sigma=np.array(data['sigma_grid']);tt,ss=np.meshgrid(np.log10(tau),sigma)
    values=np.array(data['curvature_surface']);ax.plot_surface(tt,ss,values,cmap=v.GREEN,linewidth=0,rasterized=True,antialiased=True)
    ax.set_xlabel(r'$\log_{10}\tau$',labelpad=6);ax.set_ylabel(r'$\sigma$',labelpad=6);ax.set_zlabel('Kernel-law curvature',labelpad=8)
    ax.set_xticks([-1.5,-.5,.5]);ax.set_yticks([.5,1.25,2]);ax.locator_params(axis='z',nbins=3);ax.view_init(26,-55);ax.set_box_aspect((1.2,1,.7));v.panel(ax,'B','Two regularization scales')
    ax=fig.add_subplot(gs[1,0]);checks=data['finite_difference_checks'];step=np.array([c['step'] for c in checks]);error=np.array([c['absolute_error'] for c in checks])
    ax.semilogy(step*1e6,error,'o-',color=v.TEAL,ms=4);ax.invert_xaxis();ax.set_xticks([.55,.2,.06]);ax.set_xlabel('Difference step (µrad)');ax.set_ylabel('Absolute curvature difference');v.panel(ax,'C','Separate derivative check')
    ax=fig.add_subplot(gs[1,1]);theta=np.array(data['theta']);jets=np.array(data['curve_jets']);base=np.array(data['kernel_jets']);wrong=np.array(data['omitted_path_acceleration_jets'])
    ax.plot(theta*1e6,(jets[:,1]-base[1])*1e8,color=v.TEAL,lw=1.4)
    ax.plot(theta*1e6,base[2]*theta*1e8,color=v.INK,ls=':',lw=1.2)
    ax.plot(theta*1e6,wrong[2]*theta*1e8,color=v.GREY,ls='--',lw=1.1)
    ax.set_xlabel('Bond rotation (µrad)');ax.set_ylabel(r'Derivative change ($10^{-8}$)');v.panel(ax,'D','Acceleration changes the prediction')
    handles=[Line2D([],[],color=v.TEAL,label='Computed first derivative'),Line2D([],[],color=v.INK,ls=':',label='Full curvature prediction'),Line2D([],[],color=v.GREY,ls='--',label='Path acceleration omitted')]
    caption=('Figure 13. Analytic calculus of a conformer probability measure. The deterministic first eligible bridge-bond example is '+data['mol_id']+', with '+str(data['n_parent_conformers'])+' parent conformers and '+str(data['n_class_conformers'])+' support points in the differentiated isomer law. A, Only the first conformer rotates; axis atoms have wire shells. Pale rotations of +/-0.3 rad illustrate the moving component and are not the tiny window used for derivatives. B, Exact second derivative of squared odd-mean norm over source-response regularization tau and Gaussian bandwidth sigma, with r=2.6 Å and s=3.2 Å. C, Centered differences of the analytic first derivative check the second derivative; the smallest step shows floating cancellation, not monotonic convergence. D, The actual local derivative change and its first-order curvature prediction. The maximum angle is '+f"{data['theta_window_radians']:.3g}"+' rad, chosen from the 2.94e-5 Å filtration margin; every evaluated simplex list is checked unchanged. RMS normalization is recomputed and differentiated analytically. Full curvature is -0.0655425 rad^-2; omitting molecular path acceleration gives -0.141804 rad^-2. These are descriptor-law sensitivities, not forces or occupied-rotamer probabilities.')
    v.save(fig,'Figure_13_Ensemble_calculus',caption,data,handles)


def fig14():
    frame=pd.read_csv(BIO/'descriptor_activity_pairs.csv');summary=load(BIO/'association_summary.json');limits=load(BIO/'target_conditioning_limits.json')
    limits.sort(key=lambda r:(-r['n_targets'],r['parent_key']));case=limits[0]
    # The case is selected by coverage, not by the magnitude or sign of affinity.
    rows=frame[frame.parent_key==case['parent_key']];a=load(BIO/'ensembles'/(hashlib.sha256(case['stereo_a'].encode()).hexdigest()[:20]+'.json'))
    b=load(BIO/'ensembles'/(hashlib.sha256(case['stereo_b'].encode()).hexdigest()[:20]+'.json'))
    fig=plt.figure(figsize=(178/25.4,202/25.4));gs=fig.add_gridspec(3,2,left=.18,right=.96,bottom=.12,top=.93,hspace=.66,wspace=.54,height_ratios=[.9,1.1,.85])
    ref=np.array(a['conformers'][0]['coordinates']);coords,rot,center=v.canonical(ref)
    for col,record in enumerate([a,b]):
        xx=np.array(record['conformers'][0]['coordinates']);display=coords if col==0 else v.aligned(xx,ref)@rot
        ghosts=[v.aligned(np.array(c['coordinates']),ref)@rot for c in record['conformers'][1:3]]
        ax=fig.add_subplot(gs[0,col]);im,meta=v.viewport('ext14_three_isomer_'+str(col),display,record['conformers'][0]['bonds'],record['conformers'][0]['elements'],ensemble=ghosts,
            highlight=[i for i,_ in record['conformers'][0]['stereocentres']],size=(1900,1250),camera='oblique')
        v.image_panel(ax,im,meta);v.panel(ax,'AB'[col],['Assayed isomer A','Assayed isomer B'][col])
    ax=fig.add_subplot(gs[1,0]);target=rows.groupby(['target_chembl_id','target_name']).delta_pki.median().reset_index().sort_values('target_chembl_id')
    names=target.target_name.str.replace('5-hydroxytryptamine receptor','5-HT',regex=False).str.replace('Sodium-dependent serotonin transporter','Serotonin transporter',regex=False)
    ax.axvline(0,color=v.RULE,lw=.8);ax.scatter(target.delta_pki,range(len(target)),s=28,color=v.TEAL,marker='D')
    ax.set_yticks(range(len(target)),names);ax.invert_yaxis();ax.tick_params(axis='y',length=0);ax.set_xlabel(r'Observed $pK_i(A)-pK_i(B)$');v.panel(ax,'C','One pair, different protein targets')
    ax=fig.add_subplot(gs[1,1]);parents=pd.read_csv(BIO/'parent_level_activity.csv');ax.scatter(parents['persistent_r2.6_mmd'],parents.absolute_delta_pki,s=16,color=v.TEAL,alpha=.7,linewidth=0)
    ax.set_xlabel('Persistent ensemble MMD');ax.set_ylabel(r'Mean same-assay $|\Delta pK_i|$');v.panel(ax,'D','All 103 molecular parents')
    ax=fig.add_subplot(gs[2,:]);names=['source_r2.6_mmd','same_scale_r2.6_mmd','persistent_r2.6_mmd','distance_histogram_r2.6_mmd','ecfp4_chiral']
    labels=['Odd source','Same-scale','Persistent','Distance histogram','Chiral ECFP4'];associations=[]
    for i,name in enumerate(names):
        row=next(r for r in summary['activity_associations'] if r['representation']==name);ci=row['bootstrap95'];color=v.TEAL if name.startswith('persistent') else v.GREY
        ax.plot(ci,[i,i],color=color,lw=1.3);ax.scatter([row['spearman']],[i],s=24,color=color,marker='D' if name.startswith('persistent') else 'o');associations.append(row)
    ax.axvline(0,color=v.RULE,lw=.8);ax.set_yticks(range(len(names)),labels);ax.invert_yaxis();ax.tick_params(axis='y',length=0);ax.set_xlim(-.32,.32)
    ax.set_xlabel(r'Parent-level Spearman $\rho$ and bootstrap 95% interval');v.panel(ax,'E','Association with measured stereoselectivity')
    caption=('Figure 14. Same-assay biological validation. A,B, ETKDGv3/MMFF94s geometries for measured stereoisomers CHEMBL2159489 and CHEMBL2159492, properly aligned; the first is colored and the next two have pale bonds for legibility. All eight conformers per isomer enter the analysis. These are generated geometries, not receptor-bound structures. The example is chosen by the greatest number of retained protein targets, breaking ties by parent hash, without selecting on affinity. C, Differences in ChEMBL pKi within the same assay, displayed separately for four human protein targets; positive values favour A. Target medians summarize multiple retained assays only after each within-assay difference has been formed. D, All 103 constitutional parents from 143 comparisons against 65 human single-protein targets. Pair/target values receive equal weight within a parent. E, Gaussian ensemble MMD versus absolute affinity difference, with 5,000 parent bootstrap replicates. The primary persistent-response association is rho=0.0499, interval [-0.1537,0.2463], and does not establish predictive value. ECFP4 uses radius two, 2,048 bits and chiral tags; the achiral fingerprint is identical within these graphs. Records require exact Ki, nM units, human single-protein targets, assay confidence nine and no retained validity/duplicate flag. Assay and activity identifiers remain in source data. The observed target mix is fixed; the interval is not evidence of generalization to unseen targets.')
    v.save(fig,'Figure_14_Biological_recognition',caption,{'case_parent':case,'case_a':a,'case_b':b,'case_target_differences':target.to_dict('records'),'parent_data':parents.to_dict('records'),'correlations':associations,'summary':summary})


def fig15():
    data=load(EXT/'kernel_shape_metric.json');x=np.array(data['coordinates']);coords,rot,center=v.canonical(x)
    r=np.linspace(0,3,250);q=np.exp(-2*r*r);radial=(1+q*(1-4*r*r))/2;tangent=(1+q)/2;c0=(1-2*np.exp(-1.5))/2
    fig=plt.figure(figsize=(178/25.4,198/25.4));gs=fig.add_gridspec(3,2,left=.13,right=.96,bottom=.13,top=.93,hspace=.68,wspace=.48,height_ratios=[1,1.05,.95])
    ax=fig.add_subplot(gs[0,0]);ax.plot(r,radial,color=v.TEAL,label='Radial');ax.plot(r,tangent,color=v.GREY,ls='--',label='Tangential');ax.axhline(c0,color=v.LIGHT,ls=':',lw=.8)
    ax.set_xlabel(r'$\|z\|/\sigma$');ax.set_ylabel(r'$\sigma^2$ × metric eigenvalue');ax.set_ylim(.2,1.05);v.panel(ax,'A','Explicit odd-feature metric')
    ax=fig.add_subplot(gs[0,1],projection='3d');uu,vv=np.meshgrid(np.linspace(0,2*np.pi,45),np.linspace(0,np.pi,23));rv=np.sqrt(.75);gt=(1+np.exp(-2*rv*rv))/2
    xx=np.cos(uu)*np.sin(vv)/np.sqrt(c0);yy=np.sin(uu)*np.sin(vv)/np.sqrt(gt);zz=np.cos(vv)/np.sqrt(gt)
    ax.plot_surface(xx,yy,zz,color='#91B4A5',alpha=.75,linewidth=.2,edgecolor='#567E6D',rasterized=True)
    ax.set_xlabel('Radial',labelpad=8);ax.set_ylabel('Tangent 1',labelpad=13);ax.set_zlabel('Tangent 2',labelpad=8);ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_zticks([-1,0,1]);ax.set_xlim(-2.05,2.05);ax.set_ylim(-2.05,2.05);ax.set_zlim(-2.05,2.05);ax.view_init(24,-55);ax.set_box_aspect((1,1,1));v.panel(ax,'B','Metric unit-ball section')
    fractions=[]
    for key in ['maximum_velocity','null_velocity']:
        velocity=np.array(data[key]);mass=np.array([Chem.GetPeriodicTable().GetAtomicWeight(Chem.GetPeriodicTable().GetAtomicNumber(e)) for e in data['elements']]);part=mass*np.sum(velocity*velocity,axis=1);fractions.append(part/part.sum())
    norm=Normalize(0,max(np.max(a) for a in fractions))
    for col,part in enumerate(fractions):
        ax=fig.add_subplot(gs[1,col]);im,meta=v.viewport('ext15_mode_oblique_'+str(col),coords,data['bonds'],data['elements'],atom_values=part,atom_norm=norm,size=(1900,1250),camera='oblique')
        v.image_panel(ax,im,meta);v.panel(ax,'CD'[col],['Most sensitive shape direction','An invisible first-order direction'][col])
    cax=fig.add_axes([.32,.405,.42,.011]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=v.GREEN),cax=cax,orientation='horizontal');cb.set_label('Fraction of mass-metric tangent norm',labelpad=1);cb.ax.tick_params(labelsize=8)
    ax=fig.add_subplot(gs[2,0]);singular=np.array(data['singular_values']);ax.semilogy(np.arange(1,len(singular)+1),singular,'o-',color=v.TEAL,ms=3)
    ax.axhline(singular[0]*1e-8,color=v.LIGHT,ls=':',lw=.8);ax.set_xticks([1,6,12,18]);ax.set_xlabel('Singular-value index');ax.set_ylabel('Response shape Jacobian');v.panel(ax,'E','Eighteen visible dimensions')
    ax=fig.add_subplot(gs[2,1]);ax.barh([0],[data['numerical_rank']],color=v.TEAL,height=.38);ax.barh([0],[data['numerical_nullity']],left=[data['numerical_rank']],color=v.LIGHT,height=.38)
    ax.text(data['numerical_rank']/2,0,str(data['numerical_rank']),ha='center',va='center',fontsize=9,color='white');ax.text(data['numerical_rank']+data['numerical_nullity']/2,0,str(data['numerical_nullity']),ha='center',va='center',fontsize=9)
    ax.set_xlim(0,data['shape_dimension']);ax.set_ylim(-.6,.6);ax.set_yticks([]);ax.set_xlabel('Mass-orthonormal shape dimensions');v.panel(ax,'F','A kernel cannot repair a bottleneck')
    handles=[Line2D([],[],color=v.TEAL,label='Radial / visible response'),Line2D([],[],color=v.GREY,ls='--',label='Tangential'),Line2D([],[],color=v.LIGHT,lw=5,label='Invisible shape directions')]
    caption=('Figure 15. Differential geometry and the rank bottleneck of the odd kernel. A, Exact radial and tangential eigenvalues of the pullback metric of the odd Gaussian feature map. The sharp uniform lower coefficient is c0=(1-2 exp(-3/2))/2. B, A three-dimensional section of its tangent unit ball at ||z||/sigma=sqrt(3/4), sigma=1; this is feature-space geometry, not a molecular surface. C,D, The same 31-atom geometry '+data['mol_id']+' with atom participation in a maximum-sensitivity direction and a computed null direction of its response Jacobian. Colors share one normalization and sum to one within each mass-normalized tangent. E, The 18 singular values in mass-orthonormal proper-rigid quotient coordinates, at r=2.6 Å, s=3.2 Å and tau=0.3, using a fixed baseline response scale for this local experiment. F, The unconstrained labelled shape space has dimension 87; numerical rank is 18 and 69 first-order directions are invisible. This saturates the dimension lower bound. The local kernel metric has the same null space, with the proved spectral sandwich. A null first derivative is not a claim of global invariance, unchanged chemistry, or zero physical force; bond-constrained conformer manifolds have a different dimension. Central differences check both displayed directions to vector errors below 6.1e-11.')
    v.save(fig,'Figure_15_Kernel_shape_metric',caption,{'metric':data,'radius_grid':r,'radial_eigenvalues':radial,'tangential_eigenvalues':tangent,'sharp_lower_coefficient':c0,'atom_fractions':fractions},handles)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--only',nargs='*');a=p.parse_args()
    v.OUT=a.output;v.CACHE=a.cache;v.CACHE.mkdir(parents=True,exist_ok=True)
    if (v.OUT/'manifest.json').exists():v.MANIFEST.update(load(v.OUT/'manifest.json'))
    for path in [Path(__file__),EXT/'summary.json',EXT/'parity_diagnostic_summary.json',EXT/'ensemble_geometry.json',EXT/'kernel_shape_metric.json',BIO/'association_summary.json']:
        v.MANIFEST['input_sha256'][str(path.relative_to(ROOT))]=v.sha(path)
    for key,fn in [('10',fig10),('11',fig11),('12',fig12),('13',fig13),('14',fig14),('15',fig15)]:
        if not a.only or key in a.only:fn();v.dump(v.OUT/'manifest.json',v.MANIFEST)


if __name__=='__main__':main()
