import importlib
import numpy as np
def _native():return importlib.import_module('weaveode._native')
def test_wrms_scores_matches_numpy():
 native=_native();r=np.array([[1.,-2.],[.25,.5]]);s=np.array([2.,4.]);np.testing.assert_allclose(native.wrms_scores(r,s),np.sqrt(np.mean((r/s)**2,axis=1)),atol=1e-15)
def test_poly_bdf1_scores_uses_original_residual_scaling():
 native=_native();c=np.array([-1.,0.,1.,2.]);coeff=np.array([0.,0.,1.]);got=native.poly_bdf1_scores(c,.25,.1,coeff,1e-8,1e-6);res=c-.25-.1*c**2;scale=1e-8+1e-6*np.maximum(.25,np.abs(c));np.testing.assert_allclose(got,np.abs(res)/scale,atol=1e-12)
def test_radius_components_labels_connected_components():assert _native().radius_components(np.array([[0.,0.],[.1,0.],[3.,3.],[3.1,3.],[8.,0.]]),.2).tolist()==[0,0,1,1,2]
