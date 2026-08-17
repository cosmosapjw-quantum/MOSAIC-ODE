# FAILURE_LOG.md

## 2026-08-17 — Performance hypothesis not supported on scalar benchmark

The fully integrated `B+E+T+H+M` path solved the multibranch BDF1 benchmark with zero false acceptance and exact principal-branch tracking, but consumed 1882 abstract work units versus 84 for the direct BDF1 backbone. This is retained as a negative result. The integrated slice is promoted only as a correctness prototype; performance and GPU claims remain on hold.

## 2026-08-17 — CUDA execution unavailable

The validation environment exposes PyTorch CPU only and no CUDA compiler/device. Optional CUDA C++ source was written and isolated behind a disabled-by-default CMake gate, but it was not compiled or executed.
