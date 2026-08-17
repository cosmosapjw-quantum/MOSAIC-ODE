"""Integrated BDF1 vertical slice connecting all core WeaveODE ideas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from .contracts import FloatArray, Jacobian, Rhs, as_state, positive_step, tolerance_scale, wrms_norm
from .eess import PersistentEESSState
from .homotopy import continuation_newton
from .integrators import bdf1_residual
from .online import OnlineLowRankAdapter
from .telemetry import WorkCounter
from .trajroot import solve_bdf1_step

CandidateBuilder = Callable[[float, float, float], npt.NDArray[np.float64]]


@dataclass(slots=True)
class IntegratedStepTrace:
    candidate_count: int
    component_count: int
    used_topology: bool
    used_homotopy: bool
    online_update_attempted: bool
    selected_source: str
    predictor: float
    online_correction: float
    sources_considered: tuple[str, ...]


@dataclass(slots=True)
class IntegratedStepResult:
    y: FloatArray
    accepted: bool
    scaled_residual: float
    message: str
    trace: IntegratedStepTrace
    work: WorkCounter


@dataclass(slots=True)
class SolveTrace:
    t: FloatArray
    y: FloatArray
    success: bool
    message: str
    steps: list[IntegratedStepTrace]
    work: WorkCounter


def _accumulate(target: WorkCounter, source: WorkCounter) -> None:
    target += source


class WeaveBDF1Solver:
    """A scalar integrated co-solver with BDF1 as acceptance authority."""

    def __init__(self, rhs: Rhs, *, step_size: float, polynomial_coefficients: npt.ArrayLike | None = None,
                 jacobian: Jacobian | None = None, atol: float = 1e-10, rtol: float = 1e-8,
                 certification_tol: float = 1e-6, search_radius: float = 1.0,
                 grid_candidates: int = 65, eess_points: int = 16, component_radius: float = 0.25,
                 homotopy_nodes: int = 9, seed: int = 0,
                 online_adapter: OnlineLowRankAdapter | None = None,
                 candidate_builder: CandidateBuilder | None = None) -> None:
        self.rhs = rhs
        self.jacobian = jacobian
        self.step_size = positive_step(step_size)
        if atol < 0.0 or rtol < 0.0 or (atol == 0.0 and rtol == 0.0): raise ValueError("atol/rtol define an invalid scale")
        if certification_tol <= 0.0: raise ValueError("certification_tol must be positive")
        if search_radius < 0.0 or not np.isfinite(search_radius): raise ValueError("search_radius must be finite and nonnegative")
        if grid_candidates < 1: raise ValueError("grid_candidates must be positive")
        if eess_points < 2: raise ValueError("eess_points must be at least 2")
        if homotopy_nodes < 2: raise ValueError("homotopy_nodes must be at least 2")
        self.atol=float(atol); self.rtol=float(rtol); self.certification_tol=float(certification_tol); self.search_radius=float(search_radius); self.grid_candidates=int(grid_candidates); self.homotopy_nodes=int(homotopy_nodes)
        self.coefficients=None
        if polynomial_coefficients is not None:
            coefficients=np.asarray(polynomial_coefficients,dtype=np.float64)
            if coefficients.ndim!=1 or coefficients.size==0 or not np.all(np.isfinite(coefficients)): raise ValueError("polynomial_coefficients must be a finite vector")
            self.coefficients=np.ascontiguousarray(coefficients)
        self.eess=PersistentEESSState(eess_points,component_radius)
        self.online=online_adapter or OnlineLowRankAdapter(state_dimension=1,feature_dimension=6,rank=3,learning_rate=0.04,max_relative_correction=0.25,seed=seed)
        self.candidate_builder=candidate_builder or self._default_candidate_builder
        self._last_predictor:float|None=None
        self._last_selected_delta=0.0

    def _default_candidate_builder(self,predictor:float,_previous:float,_h:float)->FloatArray:
        if self.grid_candidates==1: return np.array([[predictor]],dtype=np.float64)
        return np.linspace(predictor-self.search_radius,predictor+self.search_radius,self.grid_candidates,dtype=np.float64)[:,None]

    def _features(self,previous:float,h:float,f_previous:float,predictor:float)->FloatArray:
        component_count=int(np.unique(self.eess.component_labels).size) if self.eess.component_labels.size else 1
        return np.array([previous,h,f_previous,predictor,self._last_selected_delta,float(component_count)],dtype=np.float64)

    def _python_scores(self,candidates:FloatArray,t:float,previous:float,h:float)->FloatArray:
        scores=np.empty(candidates.shape[0],dtype=np.float64); previous_state=np.array([previous],dtype=np.float64)
        for index,value in enumerate(candidates[:,0]):
            candidate=np.array([value],dtype=np.float64); residual=bdf1_residual(self.rhs,t,previous_state,h,candidate); scale=tolerance_scale(previous_state,candidate,self.atol,self.rtol); scores[index]=wrms_norm(residual,scale)
        return scores

    def _score_candidates(self,candidates:FloatArray,t:float,previous:float,h:float,work:WorkCounter)->FloatArray:
        if candidates.shape[0]==0: return np.empty(0,dtype=np.float64)
        if self.coefficients is not None:
            try:
                from . import _native
                scores=np.asarray(_native.poly_bdf1_scores(np.ascontiguousarray(candidates[:,0]),previous,h,self.coefficients,self.atol,self.rtol),dtype=np.float64); work.native_candidate_evaluations+=candidates.shape[0]; return scores
            except (ImportError,AttributeError): pass
        scores=self._python_scores(candidates,t,previous,h); work.residual_evaluations+=candidates.shape[0]; work.rhs_evaluations+=candidates.shape[0]; return scores

    @staticmethod
    def _preselect(candidates:FloatArray,scores:FloatArray,capacity:int)->tuple[FloatArray,FloatArray]:
        if candidates.shape[0]<=capacity: return candidates,scores
        indices=np.asarray(np.argsort(scores,kind="stable")[:capacity],dtype=np.int64); return np.ascontiguousarray(candidates[indices]),np.ascontiguousarray(scores[indices])

    def _natural_homotopy(self,t:float,previous:float,h:float,work:WorkCounter)->tuple[float|None,bool]:
        parameters=np.linspace(0.0,1.0,self.homotopy_nodes)
        def residual(x:FloatArray,lam:float)->FloatArray:
            value=np.asarray(self.rhs(t+h,x.copy()),dtype=np.float64); return x-previous-lam*h*value
        local_jacobian=None
        if self.jacobian is not None:
            def local_jacobian(x:FloatArray,lam:float)->FloatArray:
                return np.eye(1,dtype=np.float64)-lam*h*np.asarray(self.jacobian(t+h,x.copy()),dtype=np.float64)
        trace=continuation_newton(residual,np.array([previous],dtype=np.float64),parameters,jacobian=local_jacobian,tol=1e-12); _accumulate(work,trace.work); work.rhs_evaluations+=trace.work.residual_evaluations
        if not trace.converged: return None,True
        return float(trace.states[-1,0]),True

    def _classical_fallback(self,t:float,previous_state:FloatArray,h:float,work:WorkCounter,*,predictor:float,features:FloatArray,proposal_scale:FloatArray,online_correction:float)->IntegratedStepResult:
        fallback=solve_bdf1_step(self.rhs,t,previous_state,h,jacobian=self.jacobian,atol=self.atol,rtol=self.rtol,certification_tol=self.certification_tol); _accumulate(work,fallback.work); work.fallbacks+=1; update_attempted=bool(fallback.accepted)
        if fallback.accepted:
            update=self.online.update(features,np.array([float(fallback.y[0])-predictor]),proposal_scale)
            if update.accepted: work.online_updates+=1
            self._last_predictor=predictor; self._last_selected_delta=float(fallback.y[0])-predictor
        trace=IntegratedStepTrace(0,0,False,False,update_attempted,"classical_fallback",predictor,online_correction,("classical_fallback",))
        return IntegratedStepResult(fallback.y,fallback.accepted,fallback.scaled_residual,fallback.message,trace,work)

    def step(self,t:float,y_previous:npt.ArrayLike,*,h:float|None=None)->IntegratedStepResult:
        previous_state=as_state(y_previous,name="y_previous")
        if previous_state.shape!=(1,): raise ValueError("the integrated V0 solver currently supports scalar ODEs")
        step=positive_step(self.step_size if h is None else h); work=WorkCounter(); f_previous=float(self.rhs(float(t),previous_state.copy())[0]); work.rhs_evaluations+=1; previous=float(previous_state[0]); predictor=previous+step*f_previous
        if self.eess.points.size and self._last_predictor is not None: self.eess.transport(np.array([predictor-self._last_predictor],dtype=np.float64))
        features=self._features(previous,step,f_previous,predictor); proposal_scale=np.array([max(1.0,abs(previous),abs(predictor))],dtype=np.float64); online_correction=float(self.online.propose(features,proposal_scale)[0])
        base=np.asarray(self.candidate_builder(predictor,previous,step),dtype=np.float64)
        if base.size==0: return self._classical_fallback(float(t),previous_state,step,work,predictor=predictor,features=features,proposal_scale=proposal_scale,online_correction=online_correction)
        if base.ndim!=2 or base.shape[1]!=1 or not np.all(np.isfinite(base)): raise ValueError("candidate_builder must return a finite array with shape (n, 1)")
        online_point=np.array([[predictor+online_correction]],dtype=np.float64); combined=np.vstack([base,np.array([[predictor]],dtype=np.float64),online_point]);
        if self.eess.points.size: combined=np.vstack([combined,self.eess.points])
        scores=self._score_candidates(combined,float(t),previous,step,work); pre_points,pre_scores=self._preselect(combined,scores,max(2*self.eess.max_points,self.eess.max_points+4)); self.eess.update(pre_points,pre_scores); work.eess_candidate_evaluations+=pre_points.shape[0]; work.topology_evaluations+=1; representatives=self.eess.best_per_component(); component_count=int(np.unique(self.eess.component_labels).size)
        root_candidates=[("predictor",predictor),("online",predictor+online_correction)]; homotopy_candidate,used_homotopy=self._natural_homotopy(float(t),previous,step,work)
        if homotopy_candidate is not None: root_candidates.append(("natural_homotopy",homotopy_candidate))
        for component,representative in enumerate(representatives):
            refined=solve_bdf1_step(self.rhs,float(t),previous_state,step,predictor=representative,jacobian=self.jacobian,atol=self.atol,rtol=self.rtol,certification_tol=self.certification_tol); _accumulate(work,refined.work)
            if refined.nonlinear.converged: root_candidates.append((f"eess_component_{component}",float(refined.y[0])))
        unique_sources=[]; unique_values=[]
        for source,value in root_candidates:
            if not np.isfinite(value): continue
            if any(abs(value-existing)<=1e-10*max(1.0,abs(value),abs(existing)) for existing in unique_values): continue
            unique_sources.append(source); unique_values.append(value)
        root_array=np.asarray(unique_values,dtype=np.float64)[:,None] if unique_values else np.empty((0,1)); root_scores=self._score_candidates(root_array,float(t),previous,step,work); certified=[index for index,score in enumerate(root_scores) if np.isfinite(score) and score<=self.certification_tol]
        if not certified: return self._classical_fallback(float(t),previous_state,step,work,predictor=predictor,features=features,proposal_scale=proposal_scale,online_correction=online_correction)
        selected_index=min(certified,key=lambda index:(abs(unique_values[index]-predictor),root_scores[index],unique_sources[index])); selected=float(unique_values[selected_index]); selected_score=float(root_scores[selected_index]); selected_source=unique_sources[selected_index]; update=self.online.update(features,np.array([selected-predictor]),proposal_scale)
        if update.accepted: work.online_updates+=1
        self.eess.update(np.array([[selected]],dtype=np.float64),np.array([selected_score])); work.eess_candidate_evaluations+=1; self._last_predictor=predictor; self._last_selected_delta=selected-predictor
        trace=IntegratedStepTrace(combined.shape[0],component_count,True,used_homotopy,True,selected_source,predictor,online_correction,tuple(unique_sources)); return IntegratedStepResult(np.array([selected],dtype=np.float64),True,selected_score,"accepted",trace,work)

    def solve(self,t_span:tuple[float,float],y0:npt.ArrayLike)->SolveTrace:
        t0,t_end=map(float,t_span)
        if not np.isfinite(t0) or not np.isfinite(t_end) or t_end<=t0: raise ValueError("t_span must be finite and increasing")
        state=as_state(y0,name="y0")
        if state.shape!=(1,): raise ValueError("the integrated V0 solver currently supports scalar ODEs")
        times=[t0]; states=[state.copy()]; step_traces=[]; work=WorkCounter(); t=t0; tolerance=16.0*np.finfo(np.float64).eps*max(1.0,abs(t_end))
        while t<t_end-tolerance:
            h=min(self.step_size,t_end-t); result=self.step(t,state,h=h); _accumulate(work,result.work); step_traces.append(result.trace)
            if not result.accepted: return SolveTrace(np.asarray(times),np.vstack(states),False,result.message,step_traces,work)
            t=min(t_end,t+h); state=result.y.copy(); times.append(t); states.append(state)
        return SolveTrace(np.asarray(times),np.vstack(states),True,"success",step_traces,work)
