import numpy as np
import sympy as sp
import gudhi

from pchr.core import persistent_from_complexes, persistent_operator, rips, schur_operator
from pchr.exact_topology import (closure, exact_persistent, octahedral_filling,
                                 tetrahedron_subcomplexes)


def test_cancelling_tetrahedra_fill_octahedral_sphere():
    k,l=octahedral_filling()
    exact,beta,z,a,d=exact_persistent(k,l)
    assert beta==0 and z.cols==1 and d.rank()==7
    assert d.cols==8 and all(d[:,j]!=sp.zeros(d.rows,1) for j in range(d.cols))
    assert (a*z*(z.T*z).inv()*z.T*a.T).trace()==1
    op=persistent_from_complexes(k,l)
    np.testing.assert_allclose(op.matrix,np.array(exact,dtype=float),atol=2e-12)
    np.testing.assert_allclose(op.matrix,schur_operator(op),atol=2e-12)
    # Discarding every tetrahedron with an outside face misses the filling.
    naive=op.b2.T@op.b2
    assert np.sum(np.linalg.eigvalsh(naive)<1e-8)==1
    assert np.sum(np.linalg.eigvalsh(op.matrix)<1e-8)==0
    assert abs(np.linalg.eigvalsh(op.matrix)[0]-1)<1e-10


def test_nonorthogonal_basis_requires_gram_inverse():
    k,l=octahedral_filling(); _,_,z,a,d=exact_persistent(k,l)
    correct=a*z*(z.T*z).inv()*z.T*a.T
    zz=3*z
    np.testing.assert_array_equal(np.array(correct,dtype=float),
                                 np.array(a*zz*(zz.T*zz).inv()*zz.T*a.T,dtype=float))
    assert a*zz*zz.T*a.T != correct


def test_all_tetrahedral_subcomplexes():
    complexes=list(tetrahedron_subcomplexes())
    assert len(complexes)==114
    count=0
    for k in complexes:
        for l in complexes:
            if all(set(k[q]).issubset(l[q]) for q in range(4)):
                exact,beta,_,_,_=exact_persistent(k,l)
                op=persistent_from_complexes(k,l)
                np.testing.assert_allclose(op.matrix,np.array(exact,dtype=float),atol=2e-12)
                assert len(op.small[2])-np.linalg.matrix_rank(op.matrix,tol=1e-8)==beta
                count+=1
    # Independent count: choose K's edges inside L; each supported triangle
    # of L is independently optional, with one extra filled-tetrahedron case.
    independent=0
    from itertools import combinations
    for l in complexes:
        for mask in range(1<<len(l[1])):
            edges={e for i,e in enumerate(l[1]) if mask & (1<<i)}
            supported=sum(set(combinations(t,2)).issubset(edges) for t in l[2])
            independent+=2**supported
        if l[3]: independent+=1
    assert count==independent==3046


def test_gudhi_persistent_betti_independent_library():
    for seed in range(10):
        rng=np.random.default_rng(520+seed)
        x=rng.normal(size=(8,3)); r,s=1.6,2.2
        op=persistent_operator(x,r,s)
        st=gudhi.RipsComplex(points=x,max_edge_length=s).create_simplex_tree(max_dimension=3)
        st.compute_persistence(homology_coeff_field=11,persistence_dim_max=True)
        # Integer boundary fixtures use real homology; these small geometric
        # examples are also checked over F_11 by an unrelated implementation.
        intervals=st.persistence_intervals_in_dimension(2)
        beta=sum(birth<=r and death>s for birth,death in intervals)
        assert beta==int(np.sum(np.linalg.eigvalsh(op.matrix)<1e-8))
