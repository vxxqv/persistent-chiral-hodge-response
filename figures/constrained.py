"""Four figures from the frozen realizable-conformer analyses."""
from pathlib import Path
import argparse,json,itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import rebuild as v
from pchr.torsion_geometry import chart
from pchr.constraints import rigidity_matrix,normal_acceleration

ROOT=v.ROOT;RESULT=ROOT/'results/constrained'


def load(path):return json.loads(Path(path).read_text())


def fig16():
    x=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[1.,1.,0]])
    pairs=list(itertools.combinations(range(4),2));u=np.array([1.,-1,-1,1]);stress=np.array([-u[i]*u[j] for i,j in pairs])
    velocity=np.zeros_like(x);velocity[3,2]=1;r=rigidity_matrix(x,pairs);check=normal_acceleration(x,pairs,velocity,np.ones(4))
    fig=plt.figure(figsize=(178/25.4,135/25.4));gs=fig.add_gridspec(2,2,left=.12,right=.97,bottom=.14,top=.92,hspace=.69,wspace=.48)
    ax=fig.add_subplot(gs[0,0],projection='3d')
    for (i,j),w in zip(pairs,stress):ax.plot(*x[[i,j]].T,color=v.TEAL if w>0 else v.ORANGE,lw=1.6)
    ax.scatter(*x.T,c=v.INK,s=30,depthshade=False)
    for i,point in enumerate(x):ax.text(*(point+[-.04,-.04,-.16]),str(i+1),fontsize=8)
    ax.quiver(*x[3],0,0,.7,color=v.BLUE,arrow_length_ratio=.2,lw=1.8);ax.text(1.03,1.03,.65,r'$v_4$',fontsize=8)
    ax.set_xlim(-.2,1.2);ax.set_ylim(-.2,1.2);ax.set_zlim(-.2,.9);ax.view_init(23,-52);ax.set_axis_off();v.panel(ax,'A','Planar complete-distance fixture')
    ax=fig.add_subplot(gs[0,1]);matrix=np.outer(u,u);im=ax.imshow(matrix,cmap=v.DIVERGE,vmin=-1,vmax=1)
    for i in range(4):
        for j in range(4):ax.text(j,i,str(int(matrix[i,j])),ha='center',va='center',fontsize=8,color='white')
    ax.set_xticks(range(4),range(1,5));ax.set_yticks(range(4),range(1,5));ax.set_xlabel('Vertex');ax.set_ylabel('Vertex');v.panel(ax,'B',r'Equilibrium stress $u u^\top$')
    ax=fig.add_subplot(gs[1,0]);t=np.linspace(-.6,.6,201);changes=[]
    for i in (0,1,2):
        length=np.linalg.norm(x[3]-x[i]);change=np.sqrt(length*length+t*t)-length;changes.append(change)
        ax.plot(t,change,color=v.BLUE if i==0 else v.GREY,ls='-' if i!=2 else ':',label='Diagonal edge' if i==0 else ('Side edges' if i==1 else None))
    ax.axhline(0,color=v.RULE,lw=.8);v.clean(ax,r'Candidate displacement $t$','Distance change');v.panel(ax,'C','First derivative misses the obstruction');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.00),fontsize=8)
    ax=fig.add_subplot(gs[1,1]);labels=[r'$\|Rv\|$',r'$\|R^\top\omega\|$',r'$\omega^\top b(v)$'];values=[np.linalg.norm(r@velocity.ravel()),np.linalg.norm(r.T@stress),stress@check['quadratic_term']]
    ax.barh(range(3),values,color=[v.GREY,v.GREY,v.ORANGE],height=.45);ax.scatter(values[:2],[0,1],color=v.INK,s=25,marker='|');ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlim(-.08,1.2);ax.set_xlabel('Exact fixture value');v.panel(ax,'D','No compatible acceleration exists')
    caption=('Figure 16. An infinitesimal flex need not be a realizable constrained motion. A, A planar labelled K4 framework with all six pair distances fixed. Blue arrow moves vertex 4 out of plane. Teal/orange edges carry positive/negative stress coefficients; this is a mathematical fixture, not a chemical molecule. B, The rank-one stress matrix with u=(1,-1,-1,1); off-diagonal entries are minus the edge stresses. C, The proposed straight displacement changes the three incident distances quadratically, although every first derivative is zero. D, Symbolic rational arithmetic gives rank R=5, Rv=0, R-transpose omega=0 and omega-transpose b(v)=1. Thus Ra+b(v)=0 has no solution. The actual full-distance shape quotient is a singleton. A first-order kernel therefore overstates realizable flexibility at this singular point.')
    v.save(fig,'Figure_16_Second_order_obstruction',caption,{'coordinates':x,'pairs':pairs,'stress':stress,'stress_matrix':matrix,'velocity':velocity,'rank':int(np.linalg.matrix_rank(r)),'obstruction':float(stress@check['quadratic_term']),'candidate_t':t,'distance_changes':changes})


def fig17():
    records=[load(p) for p in sorted((RESULT/'cohort').glob('*.json'))];valid=[r for r in records if r['status']=='completed']
    case=load(RESULT/'case_a_n65.json');z=np.load(RESULT/'case_a_n65.npz');x=z['heavy_coordinates'];coords,rotation,center=v.canonical(x)
    bonds=[b for b in case['bonds'] if max(b[:2])<len(x)];elements=case['elements'][:len(x)]
    fig=plt.figure(figsize=(178/25.4,152/25.4));gs=fig.add_gridspec(2,2,left=.12,right=.97,bottom=.14,top=.92,hspace=.66,wspace=.48)
    ax=fig.add_subplot(gs[0,0]);im,meta=v.viewport('constrained_17_case',coords,bonds,elements,highlight=sorted({a[k] for a in case['axes'] for k in ['parent','child']}),size=(1900,1250));v.image_panel(ax,im,meta)
    direction=np.array(meta['focal_point'])-np.array(meta['position']);direction/=np.linalg.norm(direction)
    right=np.cross(direction,meta['up']);right/=np.linalg.norm(right);up=np.cross(right,direction);height,width=im.shape[:2];scale=meta['parallel_scale'];aspect=width/height
    projected=np.column_stack([(.5+coords@right/(2*scale*aspect))*width,(.5-coords@up/(2*scale))*height])
    for i,axis in enumerate(case['axes']):
        pp=projected[[axis['parent'],axis['child']]];color=[v.BLUE,v.ORANGE][i]
        ax.plot(pp[:,0],pp[:,1],color=color,lw=2.1)
        ax.annotate(str(i+1),xy=pp.mean(axis=0),xytext=(-12 if i==0 else 12,13 if i==0 else -14),textcoords='offset points',ha='center',fontsize=8,color=color,arrowprops={'arrowstyle':'-','color':color,'lw':.7})
    v.panel(ax,'A','Biological case: two bridge axes')
    ax.text(.5,-.07,'S–N and S–aryl rotations; fixed internal geometry',ha='center',va='top',transform=ax.transAxes,fontsize=8)
    ax=fig.add_subplot(gs[0,1]);counts={}
    for r in valid:
        pair=(r['torsion_dimension'],r['representations']['persistent']['ranks']['1e-08']);counts[pair]=counts.get(pair,0)+1
    for (k,rank),count in counts.items():ax.scatter(k,rank,s=20+count*3,color=v.TEAL,edgecolors='white',lw=.5)
    ax.plot([0,20],[0,20],color=v.RULE,lw=1);ax.axhline(18,color=v.GREY,ls=':',lw=1);ax.set_xlim(0,21);ax.set_ylim(0,21);ax.set_xticks([0,5,10,15,20]);ax.set_yticks([0,5,10,15,20]);v.clean(ax,'Bridge coordinates k','Numerical response rank');v.panel(ax,'B','204 of 206 families have full rank')
    ax.text(.03,.95,'Marker area increases with count',transform=ax.transAxes,va='top',fontsize=8)
    ax=fig.add_subplot(gs[1,0]);colours=[v.GREY,v.BLUE,v.TEAL]
    for name,color in zip(['source','same_scale','persistent'],colours):
        vals=[]
        for r in valid:
            rep=r['representations'][name];s=np.array(rep['singular_values']);vals.append(s[r['torsion_dimension']-1]/s[0] if len(s)>=r['torsion_dimension'] else 0)
        positive=np.sort(np.array(vals)[np.array(vals)>0]);ax.plot(positive,np.arange(1,len(positive)+1)/len(valid),color=color,label=name.replace('_',' ').capitalize())
    ax.set_xscale('log');ax.axvline(1e-8,color=v.RULE,ls=':',lw=1);ax.set_xlim(1e-8,1);ax.set_ylim(0,1.02);v.clean(ax,r'$s_k/s_1$ (zero values omitted)','Fraction of all 206 isomers');ax.set_xticks([1e-8,1e-4,1.]);ax.set_xticklabels([r'$10^{-8}$',r'$10^{-4}$',r'$10^0$']);v.panel(ax,'C','Conditioning of the realizable response');ax.legend(loc='lower right',fontsize=8)
    ax=fig.add_subplot(gs[1,1]);local=load(RESULT/'case_local_checks.json');vals=[]
    for side in ('a','b'):
        c=local[side];g=np.array(c['metric']);ev,u=np.linalg.eigh(g);lift=u@np.diag(1/np.sqrt(ev))@u.T
        vals.append([np.linalg.eigvalsh(lift@np.array(c[key])[0]@lift) for key in ['coordinate_hessian','intrinsic_hessian']])
    for j,(label,color) in enumerate(zip(['Coordinate second derivative','Intrinsic Hessian'],[v.GREY,v.TEAL])):
        for i in range(2):
            ax.scatter(np.array(vals)[i,j]*1e4,np.full(2,i)+(-.12 if j==0 else .12),color=color,s=27,marker='o' if j==0 else 'D')
    ax.axvline(0,color=v.RULE,lw=.8);ax.set_yticks([0,1],['Isomer A','Isomer B']);ax.invert_yaxis();ax.set_ylim(1.5,-.5);ax.set_xlabel(r'Hessian eigenvalue ($\times 10^{-4}$, mass basis)');v.panel(ax,'D','Curvature includes the connection')
    legend=[Line2D([],[],ls='none',marker='o',color=v.GREY,label='Coordinate second derivative'),Line2D([],[],ls='none',marker='D',color=v.TEAL,label='Intrinsic Hessian')]
    caption=('Figure 17. Realizable torsional sensitivity on the biological cohort. A, The first archived conformer of isomer A in the previously selected 81dbd5f36b7abada case; coloured segments1 and2 mark the first two eligible root-oriented bridge bonds (S–N and S–aryl), with wire shells at their endpoints. These are kinematic rotations and may have large energetic barriers. B, Persistent response rank on all206 first conformers from103 parents, using a heavy-atom mass-orthonormal torsion basis, fixed baseline response normalization, relative tolerance10^-8 and absolute floor10^-12. Multiplicities share a marker. Two isomers have k=20 and ranks16 and17; the other204 have rank k. C, Cumulative smallest-to-largest singular-value ratios; two k>18 cases have s_k=0 by dimension and are omitted from the log axis, so curves terminate at204/206. All three representations retain204 full-rank cases at relative tolerances10^-6,10^-8,10^-10. D, Eigenvalues of the first fixed normalized response coordinate: ordinary chart second derivative versus the Levi-Civita Hessian, expressed in the same all-atom mass-orthonormal basis of the two-coordinate case family. These local derivatives hold inside the archived filtration chamber; they do not predict activity.')
    for a,b in [('segments1','segments 1'),('and2','and 2'),('all206','all 206'),('from103','from 103'),('tolerance10','tolerance 10'),('floor10','floor 10'),('ranks16','ranks 16'),('and17','and 17'),('other204','other 204'),('at204','at 204'),('retain204','retain 204'),('tolerances10','tolerances 10')]:caption=caption.replace(a,b)
    v.save(fig,'Figure_17_Realizable_sensitivity',caption,{'records':valid,'case':case,'case_curvature_eigenvalues':vals,'local_checks':local},legend)


def fig18():
    fig=plt.figure(figsize=(178/25.4,156/25.4));gs=fig.add_gridspec(2,2,left=.11,right=.82,bottom=.13,top=.92,hspace=.58,wspace=.53,height_ratios=[1,1.1])
    norm=LogNorm(1,2e5);cmap=plt.get_cmap('cividis');source={}
    for i,side in enumerate(('a','b')):
        z=np.load(RESULT/f'case_{side}_n65.npz');q=z['angles'];e=z['energy'];source[side]={'angles':q,'energy':e};closed=np.pad(e,((0,1),(0,1)),mode='wrap');qq=np.r_[q,2*np.pi];a,b=np.meshgrid(qq,qq,indexing='ij')
        ax=fig.add_subplot(gs[0,i],projection='3d');xx=(2+.7*np.cos(b))*np.cos(a);yy=(2+.7*np.cos(b))*np.sin(a);zz=.7*np.sin(b)
        ax.plot_surface(xx,yy,zz,facecolors=cmap(norm(1+np.maximum(closed,0))),rstride=1,cstride=1,shade=False,linewidth=0,antialiased=True)
        ax.set_box_aspect((1,1,.37));ax.view_init(40,-47);ax.set_axis_off();v.panel(ax,'AB'[i],'Isomer '+side.upper()+': angle torus')
        ax=fig.add_subplot(gs[1,i]);im=ax.pcolormesh(np.degrees(qq),np.degrees(qq),(1+np.maximum(closed,0)).T,cmap=cmap,norm=norm,shading='nearest',rasterized=True)
        ax.contour(np.degrees(q),np.degrees(q),e.T,levels=[1,5,20],colors=['white']*3,linewidths=[.6,.8,1.]);ax.scatter([0],[0],marker='o',color=v.ORANGE,s=25,edgecolor='white',lw=.6)
        ax.set_xlim(0,360);ax.set_ylim(0,360);v.clean(ax,r'S–N angle $q_1$ (degrees)',r'S–aryl angle $q_2$ (degrees)');ax.set_xticks([0,180,360]);ax.set_yticks([0,180,360]);v.panel(ax,'CD'[i],'Restricted energy')
    cax=fig.add_axes([.865,.19,.018,.61]);bar=fig.colorbar(im,cax=cax);bar.set_label(r'$1+\Delta E/(\mathrm{kcal\,mol}^{-1})$');bar.ax.tick_params(labelsize=8)
    caption=('Figure 18. One two-coordinate molecular case, with its periodic parameter topology made explicit. A,B, The same square grids wrapped onto an illustrative torus; the torus is the angle domain and is not a molecular surface. C,D, Unwrapped65-by65 energy samples for the first archived conformer of each of the two stereoisomers. Opposite edges identify. The energy is evaluated with MMFF94s after hydrogen-only optimization at the base, followed by exact bridge rotations without further minimization. Each isomer uses its own E(0,0) baseline shared with its33-by33 grid. Colour displays1+DeltaE on a common logarithmic scale; white contours are DeltaE=1,5,20 kcal/mol and orange marks the base. Small negative roundoff at the baseline is clipped only for logarithmic display. All raw energies, including severe high-energy clashes, are archived. The different independently generated base conformers are not an exact mirror control, and differences between these restricted surfaces are not an enantiomer free-energy difference.')
    caption=caption.replace('Colour displays1+DeltaE on','Colour displays 1+DeltaE/(1 kcal/mol) on')
    for a,b in [('Unwrapped65-by65','Unwrapped 65-by65'),('its33-by33','its 33-by33'),('displays1+DeltaE','displays 1+DeltaE'),('DeltaE=1,5,20','DeltaE=1, 5, 20')]:caption=caption.replace(a,b)
    v.save(fig,'Figure_18_Periodic_energy_domains',caption,source)


def fig19():
    fig=plt.figure(figsize=(178/25.4,160/25.4));gs=fig.add_gridspec(2,2,left=.13,right=.97,bottom=.15,top=.92,hspace=.70,wspace=.50)
    source={};colors={33:v.GREY,65:v.TEAL}
    for i,side in enumerate(('a','b')):
        ax=fig.add_subplot(gs[0,i]);rows=[]
        for n in (33,65):
            data=load(RESULT/f'case_{side}_n{n}.json');source[f'{side}_{n}']=data
            ints=data['persistence']['intervals']['1'];finite=sorted([[b,e] for b,e in ints if e is not None],key=lambda x:x[0]);essential=sorted([b for b,e in ints if e is None])
            start=len(rows)
            for b,e in finite:
                y=len(rows);ax.plot([1+b,1+e],[y,y],color=colors[n],lw=2);rows.append((n,False,b,e))
            for b in essential:
                y=len(rows);ax.annotate('',xy=(3e5,y),xytext=(1+b,y),arrowprops={'arrowstyle':'->','color':colors[n],'lw':1.4});rows.append((n,True,b,None))
            ax.text(.02,(start+len(rows)-1)/2,str(n)+'×'+str(n),transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8,color=colors[n])
        ax.set_xscale('log');ax.set_xlim(1,3e5);ax.set_ylim(len(rows),-1);ax.set_yticks([]);ax.set_xlabel(r'$1+\Delta E/(\mathrm{kcal\,mol}^{-1})$');v.panel(ax,'AB'[i],'Isomer '+side.upper()+r': all $H_1$ intervals')
    measures=load(RESULT/'measure_summary.json');comparison=load(RESULT/'grid_comparison.json');ax=fig.add_subplot(gs[1,0]);labels=['Uniform angle','Boltzmann angle','Boltzmann + volume'];x=np.arange(3)
    for side,color,offset in [('a',v.BLUE,-.07),('b',v.ORANGE,.07)]:
        for n,marker in [(33,'o'),(65,'D')]:ax.plot(x+offset,measures['grids'][str(n)][side]['odd_norm_squared'],ls='-' if n==65 else ':',marker=marker,color=color,ms=4,lw=1)
    ax.set_xticks(x,['Uniform','Energy','Energy + volume'],rotation=20,ha='right');ax.set_ylim(0,.55);ax.set_ylabel(r'Odd squared norm $\|a_\mu\|^2$');v.panel(ax,'C','Weight choice changes the represented law')
    ax=fig.add_subplot(gs[1,1]);dims=np.arange(3);width=.32
    for side,color,offset in [('a',v.BLUE,-width/2),('b',v.ORANGE,width/2)]:
        vals=[comparison['bottleneck_distances_kcal_mol'][side][str(d)] for d in range(3)];ax.bar(dims+offset,vals,width,color=color)
    ax.set_yscale('log');ax.set_xticks(dims,[r'$H_0$',r'$H_1$',r'$H_2$']);ax.set_ylabel('33-to-65 bottleneck (kcal/mol)');v.panel(ax,'D','High-energy topology is not converged')
    handles=[Line2D([],[],ls='-',color=v.BLUE,label='Isomer A'),Line2D([],[],ls='-',color=v.ORANGE,label='Isomer B'),Line2D([],[],ls='none',marker='o',color=v.GREY,label='33×33'),Line2D([],[],ls='none',marker='D',color=v.GREY,label='65×65')]
    caption=('Figure 19. Periodic sublevel persistence and measure sensitivity on the same frozen case. A,B, Every H1 interval in the periodic vertex cubical energy filtration over F2, shown separately at33 and65 samples per angle. Arrowed bars are essential, with births near5 and16 kcal/mol; horizontal coordinates are1+DeltaE on a log axis. Both terminal complexes have Betti numbers(1,2,1), forced by the toroidal parameter domain. C, Odd-kernel mean squared norms at sigma1 under uniform angle, Boltzmann angle, and Boltzmann-times-metric-volume weights at298.15K. One shared fine-grid uniform RMS normalizes both isomers for all weight rules. The metric-volume rule belongs to an explicitly defined reduced quotient kinetic model, not the full molecular canonical distribution. Coarse/fine markers nearly coincide; this is finite quadrature sensitivity, not a sampling confidence interval. D, Full untruncated diagram bottleneck distances expose severe resolution sensitivity of the high-energy landscape, particularly H1. These large differences prevent a continuum persistence claim even though the low-energy weighted summaries change little. No diagram represents molecular spatial cavities or receptor binding.')
    caption=caption.replace('coordinates are1+DeltaE on','coordinates are 1+DeltaE/(1 kcal/mol) on')
    for a,b in [('at33 and65','at 33 and 65'),('near5 and16','near 5 and 16'),('numbers(1,2,1)','numbers (1,2,1)'),('at sigma1','at sigma=1'),('at298.15K','at 298.15 K')]:caption=caption.replace(a,b)
    v.save(fig,'Figure_19_Persistence_and_measure',caption,{'case_diagrams':source,'measures':measures,'grid_comparison':comparison},handles)


def main(out,cache):
    v.OUT=out;v.CACHE=cache;out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    for function in (fig16,fig17,fig18,fig19):function()
    v.dump(out/'constrained_figure_manifest.json',v.MANIFEST)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--cache',type=Path,required=True);args=parser.parse_args();main(args.out,args.cache)
