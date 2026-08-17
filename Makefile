.PHONY: build-native build-c test benchmark preproduct-benchmark verify package clean

build-native:
	python3 setup.py build_ext --inplace

build-c:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel
	ctest --test-dir build --output-on-failure

test: build-native build-c
	PYTHONPATH=src:. pytest -q

benchmark: build-native
	PYTHONPATH=src:. python3 benchmarks/run_integrated_benchmark.py \
		--output artifacts/generated/integrated_benchmark.json

preproduct-benchmark: build-native
	PYTHONPATH=src:. python3 benchmarks/run_preproduct_benchmark.py

verify:
	./scripts/verify_preproduct.sh

package:
	./scripts/package_bootstrap.sh

clean:
	rm -rf build build-* .pytest_cache htmlcov .coverage dist *.egg-info
	rm -f src/weaveode/_native*.so
