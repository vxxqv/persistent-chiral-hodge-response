import numpy as np
import sympy as sp
import pytest
from scipy.spatial.transform import Rotation
from pchr.knot_topology import two_strand_jones,fox_determinant,trefoil_points,braid_jones_A,tl_product,tl_identity,tl_generator,unitary_braid_matrices


def test_invalid_projection_cannot_report_an_unknot_determinant():
    knot=trefoil_points(40)
    for invalid in [np.full((3,3),np.nan),np.diag([1.,1.,-1.]),np.ones((3,3))]:
        with pytest.raises(ValueError):fox_determinant(knot,invalid)


def test_exact_jones_normalization_and_mirror():
    a=sp.Symbol('A');assert two_strand_jones(1)==1 and two_strand_jones(-1)==1
    assert sp.expand(two_strand_jones(3)-(a**-4+a**-12-a**-16))==0
    for m in range(1,10,2):assert sp.expand(two_strand_jones(-m)-two_strand_jones(m).subs(a,1/a))==0
    # Multiplying a positive and negative TL2 generator gives I exactly.
    delta=-a*a-a**-2;assert sp.expand(a*a**-1)==1 and sp.expand(a*a+a**-1*a**-1+delta)==0


def test_fox_determinant_projection_mirror_and_subdivision():
    knot=trefoil_points(80);circle=np.column_stack([np.cos(np.arange(80)*2*np.pi/80),np.sin(np.arange(80)*2*np.pi/80),np.zeros(80)])
    rng=np.random.default_rng(7311)
    for _ in range(5):
        q=Rotation.random(random_state=rng).as_matrix();original=fox_determinant(knot,q)
        assert original['determinant']==3 and fox_determinant(circle,q)['determinant']==1
        mirror=fox_determinant(knot*np.array([-1,1,1]),np.diag([-1,1,1])@q@np.diag([1,1,-1]))
        assert mirror['determinant']==3 and mirror['writhe']==-original['writhe']
        subdivided=np.empty((160,3));subdivided[::2]=knot;subdivided[1::2]=(knot+np.roll(knot,-1,axis=0))/2
        assert fox_determinant(subdivided,q)['determinant']==3


def test_tl3_relations_and_markov_braid_checks():
    identity=tl_identity(3);e1=tl_generator(3,0);e2=tl_generator(3,1)
    assert tl_product(e1,e1,3)==(e1,1)
    product,_=tl_product(e1,e2,3);assert tl_product(product,e1,3)==(e1,0)
    for word in ([1,2,1],[1,1,1,2],[1,2,-1,2]):
        assert sp.expand(braid_jones_A(3,[1,-1,*word])-braid_jones_A(3,word))==0
        assert sp.expand(braid_jones_A(3,[1,2,1,*word])-braid_jones_A(3,[2,1,2,*word]))==0
    assert sp.expand(braid_jones_A(3,[1,1,1,2])-two_strand_jones(3))==0
    assert sp.expand(braid_jones_A(3,[1,1,1,-2])-two_strand_jones(3))==0
    assert sp.expand(braid_jones_A(3,[1,2,-1,-2])-braid_jones_A(3,[2,-1,-2,1]))==0


def test_unitary_quantum_braid_representation():
    for r in (4,5,7):
        b1,b2,e1,e2=unitary_braid_matrices(r);delta=2*np.cos(np.pi/r)
        np.testing.assert_allclose(e1@e1,delta*e1,atol=1e-15)
        np.testing.assert_allclose(e1@e2@e1,e1,atol=1e-15)
        np.testing.assert_allclose(b1.conj().T@b1,np.eye(2),atol=1e-15)
        np.testing.assert_allclose(b2.conj().T@b2,np.eye(2),atol=1e-15)
        np.testing.assert_allclose(b1@b2@b1,b2@b1@b2,atol=2e-15)
        assert np.linalg.norm(b1@b2-b2@b1)>.5
