"""Vector integrated candidate pipeline for the MOSAIC-ODE pre-product.

This is intentionally a small vertical slice. It connects a classical
predictor, native vector residual scoring, persistent EESS/topology state,
mode-specific homotopy paths, current-IVP online adaptation, and original BDF1
residual validation for vector ODEs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import _native
from .contracts import FloatArray, Jacobian, Rhs, as_state, positive_step, tolerance_scale
from .eess import PersistentEESSState
from .homotopy import continuation_newton
from .integrators import bdf1_residual
from .online import OnlineLowRankAdapter
from .telemetry import WorkCounter


@dataclass(slots=True)
class VectorPipelineTrace:
    candidate_count: int
    component_count: int
    homotopy_paths: int
    selected_component: int
    selected_source: str
    online_update_accepted: bool


@dataclass(slots=True)
class VectorPipelineResult:
    y: FloatArray
    accepted: bool
    scaled_residual: float
    message: str
    trace: VectorPipelineTrace
    work: WorkCounter


class VectorCandidatePipeline:
    """One-step vector BDF1 co-solver with topology/EESS/homotopy participation."""

    def __init__(self, rhs: Rhs, *, step_size: float, jacobian: Jacobian | None = None,
                 atol: float | npt.ArrayLike = 1e-10, rtol: float = 1e-8,
                 certification_tol: float = 1e-6, candidate_radius: float = 0.1,
                 candidate_count: int = 17, eess_points: int = 12,
                 component_radius: float = 0.25, homotopy_nodes: int = 7,
                 online_rank: int = 2, seed: int = 0) -> None:
        self.rhs = rhs
        self.jacobian = jacobian
        self.step_size = positive_step(step_size)
        self.atol = np.asarray(atol, dtype=np.float64)
        if np.any(~np.isfinite(self.atol)) or np.any(self.atol < 0.0): raise ValueError("atol must be finite and nonnegative")
        if not np.isfinite(rtol) or rtol < 0.0: raise ValueError("rtol must be finite and nonnegative")
        if self.atol.ndim > 1: raise ValueError("atol must be scalar or one-dimensional")
        if candidate_radius < 0.0 or not np.isfinite(candidate_radius): raise ValueError("candidate_radius must be finite and nonnegative")
        if candidate_count < 3: raise ValueError("candidate_count must be at least 3")
        if homotopy_nodes < 2: raise ValueError("homotopy_nodes must be at least 2")
        if certification_tol <= 0.0 or not np.isfinite(certification_tol): raise ValueError("certification_tol must be positive and finite")
        self.rtol = float(rtol); self.certification_tol = float(certification_tol); self.candidate_radius=float(candidate_radius); self.candidate_count=int(candidate_count); self.homotopy_nodes=int(homotopy_nodes); self.seed=int(seed); self._rng=np.random.default_rng(seed); self.eess=PersistentEESSState(eess_points,component_radius); self.online_rank=int(online_rank); self.online:OnlineLowRankAdapter|None=None; self._last_correction:FloatArray|None=None; self._last_predictor:FloatArray|None=None

    def _atol_for(self, dimension: int) -> FloatArray:
        if self.atol.ndim == 0: return np.full(dimension, float(self.atol), dtype=np.float64)
        if self.atol.shape != (dimension,): raise ValueError("atol has the wrong state dimension")
        return np.ascontiguousarray(self.atol)

    def _ensure_online(self, dimension: int) -> OnlineLowRankAdapter:
        if self.online is None:
            self.online = OnlineLowRankAdapter(state_dimension=dimension, feature_dimension=3*dimension, rank=min(self.online_rank,dimension), learning_rate=0.02, max_relative_correction=0.20, seed=self.seed)
        elif self.online.state_dimension != dimension: raise ValueError("state dimension changed after online adapter initialization")
        return self.online

    def _features(self, previous: FloatArray, predictor: FloatArray, f_previous: FloatArray) -> FloatArray:
        return np.ascontiguousarray(np.concatenate([previous, predictor-previous, f_previous]))

    def _candidate_cloud(self, predictor: FloatArray, online_correction: FloatArray) -> FloatArray:
        dimension=predictor.size; scale=np.maximum(1.0,np.abs(predictor)); candidates=[predictor.copy(),predictor+online_correction]
        for axis in range(dimension):
            if len(candidates)>=self.candidate_count: break
            direction=np.zeros(dimension,dtype=np.float64); direction[axis]=self.candidate_radius*scale[axis]; candidates.append(predictor+direction)
            if len(candidates)<self.candidate_count: candidates.append(predictor-direction)
        while len(candidates)<self.candidate_count:
            direction=self._rng.normal(size=dimension); norm=float(np.linalg.norm(direction))
            if norm==0.0: continue
            direction/=norm; candidates.append(predictor+self.candidate_radius*scale*direction)
        return np.ascontiguousarray(np.asarray(candidates[:self.candidate_count],dtype=np.float64))

    def _native_scores(self,t:float,previous:FloatArray,h:float,candidates:FloatArray,work:WorkCounter)->FloatArray:
        rhs_values=np.empty_like(candidates)
        for index,candidate in enumerate(candidates):
            value=np.asarray(self.rhs(float(t)+h,candidate.copy()),dtype=np.float64); work.rhs_evaluations+=1
            if value.shape!=previous.shape or not np.all(np.isfinite(value)): raise ValueError("RHS returned an invalid vector candidate value")
            rhs_values[index]=value
        scores=np.asarray(_native.vector_bdf1_scores_from_rhs(candidates,rhs_values,previous,h,self._atol_for(previous.size),self.rtol),dtype=np.float64); work.native_candidate_evaluations+=candidates.shape[0]; return np.ascontiguousarray(scores)

    def _homotopy_root(self,t:float,previous:FloatArray,h:float,seed:FloatArray,work:WorkCounter)->tuple[FloatArray,bool]:
        parameters=np.linspace(0.0,1.0,self.homotopy_nodes)
        def residual(candidate:FloatArray,lam:float)->FloatArray:
            work.rhs_evaluations+=1; endpoint=bdf1_residual(self.rhs,t,previous,h,candidate); return (1.0-lam)*(candidate-seed)+lam*endpoint
        jacobian=None
        if self.jacobian is not None:
            def jacobian(candidate:FloatArray,lam:float)->FloatArray:
                jf=np.asarray(self.jacobian(float(t)+h,candidate.copy()),dtype=np.float64); work.jacobian_evaluations+=1
                if jf.shape!=(candidate.size,candidate.size) or not np.all(np.isfinite(jf)): raise ValueError("Jacobian returned an invalid matrix")
                endpoint_j=np.eye(candidate.size,dtype=np.float64)-h*jf; return (1.0-lam)*np.eye(candidate.size,dtype=np.float64)+lam*endpoint_j
        trace=continuation_newton(residual,seed,parameters,jacobian=jacobian,tol=1e-12,max_iterations=30); work+=trace.work; return trace.states[-1].copy(),bool(trace.converged)

    def step(self,t:float,y_previous:npt.ArrayLike,*,h:float|None=None)->VectorPipelineResult:
        snapshot=(self.eess.points.copy(),self.eess.scores.copy(),self.eess.component_labels.copy(),self.eess.generation); last_predictor=None if self._last_predictor is None else self._last_predictor.copy()
        try: result=self._step_impl(t,y_previous,h=h)
        except Exception:
            self._restore_eess_snapshot(snapshot); self._last_predictor=last_predictor; raise
        if not result.accepted:
            self._restore_eess_snapshot(snapshot); self._last_predictor=last_predictor
        return result

    def _restore_eess_snapshot(self,snapshot:tuple[FloatArray,FloatArray,npt.NDArray[np.int64],int])->None:
        points,scores,labels,generation=snapshot; self.eess.points=np.ascontiguousarray(points); self.eess.scores=np.ascontiguousarray(scores); self.eess.component_labels=np.ascontiguousarray(labels); self.eess.generation=int(generation)

    def _step_impl(self,t:float,y_previous:npt.ArrayLike,*,h:float|None=None)->VectorPipelineResult:
        previous=as_state(y_previous,name="y_previous"); step=self.step_size if h is None else positive_step(h); work=WorkCounter(); f_previous=np.asarray(self.rhs(float(t),previous.copy()),dtype=np.float64); work.rhs_evaluations+=1
        if f_previous.shape!=previous.shape or not np.all(np.isfinite(f_previous)): raise ValueError("RHS returned an invalid predictor value")
        predictor=previous+step*f_previous; scale=tolerance_scale(previous,predictor,self._atol_for(previous.size),self.rtol); online=self._ensure_online(previous.size); features=self._features(previous,predictor,f_previous); online_correction=online.propose(features,scale)
        if self._last_predictor is not None and self.eess.points.size: self.eess.transport(predictor-self._last_predictor)
        cloud=self._candidate_cloud(predictor,online_correction); scores=self._native_scores(float(t),previous,step,cloud,work); self.eess.update(cloud,scores); work.eess_candidate_evaluations+=cloud.shape[0]; work.topology_evaluations+=1; representatives=self.eess.best_per_component()
        if representatives.size==0: return VectorPipelineResult(predictor,False,float("inf"),"EESS produced no representatives",VectorPipelineTrace(cloud.shape[0],0,0,-1,"none",False),work)
        roots=[]; root_sources=[]
        for index,representative in enumerate(representatives):
            root,converged=self._homotopy_root(float(t),previous,step,representative,work)
            if converged and np.all(np.isfinite(root)): roots.append(root); root_sources.append(f"homotopy:{index}")
        from .trajroot import solve_bdf1_step
        direct=solve_bdf1_step(self.rhs,float(t),previous,step,predictor=predictor,jacobian=self.jacobian,atol=self._atol_for(previous.size),rtol=self.rtol,certification_tol=self.certification_tol); work+=direct.work
        if direct.accepted: roots.append(direct.y.copy()); root_sources.append("direct")
        if not roots: return VectorPipelineResult(predictor,False,float("inf"),"no converged root candidate",VectorPipelineTrace(cloud.shape[0],int(np.unique(self.eess.component_labels).size),representatives.shape[0],-1,"none",False),work)
        root_matrix=np.ascontiguousarray(np.vstack(roots)); root_scores=self._native_scores(float(t),previous,step,root_matrix,work); certified=np.flatnonzero(root_scores<=self.certification_tol)
        if certified.size==0: return VectorPipelineResult(predictor,False,float(np.min(root_scores)),"no root passed original-residual validation",VectorPipelineTrace(cloud.shape[0],int(np.unique(self.eess.component_labels).size),representatives.shape[0],-1,"none",False),work)
        selected=int(min(certified,key=lambda idx:(float(np.linalg.norm(root_matrix[idx]-predictor)),float(root_scores[idx]),root_sources[idx]))); y=root_matrix[selected].copy(); selected_score=float(root_scores[selected]); self.eess.update(y[None,:],np.array([selected_score],dtype=np.float64)); work.eess_candidate_evaluations+=1; update=online.update(features,y-predictor,scale); work.online_updates+=int(update.accepted); self._last_correction=y-predictor; self._last_predictor=predictor.copy(); labels=self.eess.component_labels; nearest=int(np.argmin(np.linalg.norm(self.eess.points-y[None,:],axis=1))); selected_component=int(labels[nearest]); trace=VectorPipelineTrace(cloud.shape[0],int(np.unique(labels).size),representatives.shape[0],selected_component,root_sources[selected],update.accepted); return VectorPipelineResult(y,True,selected_score,"accepted",trace,work)
