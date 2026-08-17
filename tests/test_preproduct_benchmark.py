from benchmarks.run_preproduct_benchmark import run_preproduct_benchmark
def test_preproduct_benchmark_smoke():
 r=run_preproduct_benchmark();assert r['schema_version']==1;assert r['project']=='MOSAIC-ODE';assert r['linear_vector']['success'];assert r['linear_vector']['max_discrete_error']<1e-10;assert r['robertson']['accepted'];assert r['robertson']['minimum_state']>=-1e-12;assert r['robertson']['mass_error']<1e-10;assert r['cuda']['executed'] is False
