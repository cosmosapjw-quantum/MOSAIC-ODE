from benchmarks.run_integrated_benchmark import deterministic_projection,run_benchmark
def test_integrated_benchmark_schema_and_reproducibility():
 first=run_benchmark(seed=20260817,steps=4);second=run_benchmark(seed=20260817,steps=4);assert first['schema_version']==1
 for result in first['variants'].values():assert result['success'];assert result['false_acceptances']==0;assert result['max_principal_branch_error']<1e-8
 assert deterministic_projection(first)==deterministic_projection(second)
