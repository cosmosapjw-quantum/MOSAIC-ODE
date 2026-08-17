import numpy as np
from weaveode.vector_pipeline import VectorCandidatePipeline
def _pipeline(matrix,h=.1):
 def rhs(_t,y):return matrix@y
 def jac(_t,_y):return matrix
 return VectorCandidatePipeline(rhs,jacobian=jac,step_size=h,candidate_radius=.08,candidate_count=9,eess_points=8,component_radius=.12,atol=1e-10,rtol=1e-8,certification_tol=1e-6),rhs
def test_vector_candidate_pipeline_selects_certified_linear_bdf1_root():
 m=np.diag([-2.,-.5]);p,_=_pipeline(m);y0=np.array([1.,2.]);r=p.step(0.,y0);assert r.accepted;np.testing.assert_allclose(r.y,np.linalg.solve(np.eye(2)-.1*m,y0),atol=1e-11);assert r.work.native_candidate_evaluations>0
def test_vector_candidate_pipeline_transports_eess_between_steps():
 m=np.diag([-1.,-.25]);p,_=_pipeline(m,.05);first=p.step(0.,np.array([1.,2.]));assert first.accepted;second=p.step(.05,first.y);assert second.accepted;assert p.eess.generation==1
def test_vector_candidate_pipeline_rolls_back_eess_after_rejected_step():
 def rhs(_t,y):return y*y
 def jac(_t,y):return np.diag(2*y)
 p=VectorCandidatePipeline(rhs,jacobian=jac,step_size=1.,candidate_radius=.05,candidate_count=7,eess_points=6,component_radius=.2,atol=1e-10,rtol=1e-8,certification_tol=1e-6);r=p.step(0.,np.array([1.]));assert not r.accepted;assert p.eess.points.size==0
