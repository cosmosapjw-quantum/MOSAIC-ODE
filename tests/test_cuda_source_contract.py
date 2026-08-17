from pathlib import Path
def test_cuda_preproduct_sources_cover_device_resident_architecture():
 expected={'cuda/candidate_bundle.cuh':['CandidateBundleDevice','active_mask'],'cuda/vector_candidate_score.cu':['vector_bdf1','cudaStream_t'],'cuda/topology_h0.cu':['radius','atomic'],'cuda/online_low_rank.cu':['low_rank','tanh'],'cuda/runtime_rhs_nvrtc.cpp':['nvrtc','PTX']}
 for name,markers in expected.items():
  text=Path(name).read_text()
  for marker in markers:assert marker in text
