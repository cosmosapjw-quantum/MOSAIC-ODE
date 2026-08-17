import math
import numpy as np
from weaveode.eess import PersistentEESSState
from weaveode.topology import merge_profile,vietoris_rips_persistence
def test_merge_profile_tracks_two_modes_until_bridge_radius():assert merge_profile(np.array([[-1.],[-.9],[1.],[1.1]]),[.11,.5,2.1])==[(.11,2),(.5,2),(2.1,1)]
def test_vr_persistence_detects_circle_loop():
 angles=np.linspace(0,2*math.pi,20,endpoint=False);points=np.column_stack([np.cos(angles),np.sin(angles)]);p=vietoris_rips_persistence(points,max_dimension=1);finite=[d-b for b,d in p[1] if np.isfinite(d)];assert finite;assert max(finite)>.8
def test_topology_preserving_selection_keeps_modes():
 state=PersistentEESSState(4,.25);state.update(np.array([[-1.],[-.95],[.95],[1.]]),np.array([.2,.1,.15,.05]));state.update(np.array([[-1.04],[-1.02],[-.98],[-.96]]),np.array([.01,.02,.03,.04]));assert np.any(state.points[:,0]>.8);assert len(np.unique(state.component_labels))==2
def test_eess_transport_moves_persistent_population():
 state=PersistentEESSState(3,.3);state.update(np.array([[-1.],[0.],[1.]]),np.array([.2,.1,.3]));state.transport(np.array([.25]));np.testing.assert_allclose(np.sort(state.points[:,0]),[-.75,.25,1.25]);assert state.generation==1
