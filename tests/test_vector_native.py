import numpy as np
def test_vector_bdf1_scores_from_rhs_matches_python_reference():
 from weaveode import _native
 c=np.array([[1.,2.],[.5,-1.],[2.5,.25]]);rv=np.array([[.1,-.5],[.2,1.],[-.25,.75]]);p=np.array([.75,1.5]);a=np.array([1e-8,2e-8]);rtol=1e-6;h=.1;got=_native.vector_bdf1_scores_from_rhs(c,rv,p,h,a,rtol);res=c-p[None,:]-h*rv;scale=a[None,:]+rtol*np.maximum(np.abs(p)[None,:],np.abs(c));np.testing.assert_allclose(got,np.sqrt(np.mean((res/scale)**2,axis=1)),atol=2e-12)
