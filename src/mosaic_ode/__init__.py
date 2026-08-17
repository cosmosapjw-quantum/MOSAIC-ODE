"""Canonical Python facade for the MOSAIC-ODE integrated pre-product."""
from __future__ import annotations
from weaveode import OnlineLowRankAdapter, SolveTrace, WeaveBDF1Solver
from weaveode.vector_pipeline import VectorCandidatePipeline, VectorPipelineResult, VectorPipelineTrace
__project_name__ = "MOSAIC-ODE"
__version__ = "0.2.0a0"
__all__ = ["OnlineLowRankAdapter", "SolveTrace", "WeaveBDF1Solver", "VectorCandidatePipeline", "VectorPipelineResult", "VectorPipelineTrace"]
