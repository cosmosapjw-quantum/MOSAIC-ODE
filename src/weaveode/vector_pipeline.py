"""Vector integrated candidate pipeline for the MOSAIC-ODE pre-product."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from . import _native
from .contracts import FloatArray,Jacobian,Rhs,as_state,positive_step,tolerance_scale
from .eess import PersistentEESSState
from .homotopy import continuation_newton
from .integrators import bdf1_residual
from .online import OnlineLowRankAdapter
from .telemetry import WorkCounter
@dataclass(slots=True)
class VectorPipelineTrace:
    candidate_count:int; component_count:int; homotopy_paths:int; selected_component:int; selected_source:str; online_update_accepted:bool
@dataclass(slots=True)
class VectorPipelineResult:
    y:FloatArray; accepted:bool; scaled_residual:float; message:str; trace:VectorPipelineTrace; work:WorkCounter
class VectorCandidatePipeline:
    def __init__(self,rhs:Rhs,*,step_size:float,jacobian:Jacobian|None=None,atol:float|npt.ArrayLike=1e-10,rtol:float=1e-8,certification_tol:float=1e-6,candidate_radius:float=.1,candidate_count:int=17,eess_points:int=12,component_radius:float=.25,homotopy_nodes:int=7,online_rank:int=2,seed:int=0):
        self.rhs=rhs; self.jacobian=jacobian; self.step_size=positive_step(step_size); self.atol=np.asarray(atol,dtype=np.float64); self.rtol=float(rtol); self.certification_tol=float(certification_tol); self.candidate_radius=float(candidate_radius); self.candidate_count=int(candidate_count); self.homotopy_nodes=int(homotopy_nodes); self.seed=int(seed); self._rng=np.random.default_rng(seed); self.eess=PersistentEESSState(eess_points,component_radius); self.online_rank=int(online_rank); self.online=None; self._last_correction=None; self._last_predictor=None
    def _atol_for(self,d):
        if self.atol.ndim==0:return np.full(d,float(self.atol),dtype=np.float64)
        if self.atol.shape!=(d,):raise ValueError("atol has the wrong state dimension")
        return np.ascontiguousarray(self.atol)
    def _ensure_online(self,d):
        if self.online is None:self.online=OnlineLowRankAdapter(state_dimension=d,feature_dimension=3*d,rank=min(self.online_rank,d),learning_rate=.02,max_relative_correction=.20,seed=self.seed)
        return self.online
    def _features(self,previous,predictor,f_previous): return np.ascontiguousarray(np.concatenate([previous,predictor-previous,f_previous]))
    def _candidate_cloud(self,predictor,online_correction):
        d=predictor.size; scale=np.maximum(1.0,np.abs(predictor)); c=[predictor.copy(),predictor+online_correction]
        for axis in range(d):
            if len(c)>=self.candidate_count: break
            direction=np.zeros(d); direction[axis]=self.candidate_radius*scale[axis]; c.append(predictor+direction)
            if len(c)<self.candidate_count:c.append(predictor-direction)
        while len(c)<self.candidate_count:
            direction=self._rng.normal(size=d); norm=float(np.linalg.norm(direction))
            if norm: c.append(predictor+self.candidate_radius*scale*direction/norm)
        return np.ascontiguousarray(np.asarray(c[:self.candidate_count],dtype=np.float64))
    def _native_scores(self,t,previous,h,candidates,work):
        rhs_values=np.empty_like(candidates)
        for i,candidate in enumerate(candidates): rhs_values[i]=np.asarray(self.rhs(float(t)+h,candidate.copy()),dtype=np.float64); work.rhs_evaluations+=1
        scores=np.asarray(_native.vector_bdf1_scores_from_rhs(candidates,rhs_values,previous,h,self._atol_for(previous.size),self.rtol),dtype=np.float64); work.native_candidate_evaluations+=candidates.shape[0]; return np.ascontiguousarray(scores)
    def _homotopy_root(self,t,previous,h,seed,work):
        parameters=np.linspace(0,1,self.homotopy_nodes)
        def residual(candidate,lam): work.rhs_evaluations+=1; return (1-lam)*(candidate-seed)+lam*bdf1_residual(self.rhs,t,previous,h,candidate)
        jacobian=None
        if self.jacobian is not None:
            def jacobian(candidate,lam): work.jacobian_evaluations+=1; jf=np.asarray(self.jacobian(float(t)+h,candidate.copy()),dtype=np.float64); return (1-lam)*np.eye(candidate.size)+lam*(np.eye(candidate.size)-h*jf)
        trace=continuation_newton(residual,seed,parameters,jacobian=jacobian,tol=1e-12,max_iterations=30); work+=trace.work; return trace.states[-1].copy(),bool(trace.converged)
    def _restore(self,snapshot):
        p,s,l,g=snapshot; self.eess.points=np.ascontiguousarray(p); self.eess.scores=np.ascontiguousarray(s); self.eess.component_labels=np.ascontiguousarray(l); self.eess.generation=int(g)
    def step(self,t,y_previous,*,h=None):
        snapshot=(self.eess.points.copy(),self.eess.scores.copy(),self.eess.component_labels.copy(),self.eess.generation); last=None if self._last_predictor is None else self._last_predictor.copy()
        try: result=self._step_impl(t,y_previous,h=h)
        except Exception: self._restore(snapshot); self._last_predictor=last; raise
        if not result.accepted:self._restore(snapshot); self._last_predictor=last
        return result
    def _step_impl(self,t,y_previous,*,h=None):
        previous=as_state(y_previous,name="y_previous"); step=self.step_size if h is None else positive_step(h); work=WorkCounter(); f_previous=np.asarray(self.rhs(float(t),previous.copy()),dtype=np.float64); work.rhs_evaluations+=1; predictor=previous+step*f_previous; scale=tolerance_scale(previous,predictor,self._atol_for(previous.size),self.rtol); online=self._ensure_online(previous.size); features=self._features(previous,predictor,f_previous); correction=online.propose(features,scale)
        if self._last_predictor is not None and self.eess.points.size:self.eess.transport(predictor-self._last_predictor)
        cloud=self._candidate_cloud(predictor,correction); scores=self._native_scores(float(t),previous,step,cloud,work); self.eess.update(cloud,scores); work.eess_candidate_evaluations+=cloud.shape[0]; work.topology_evaluations+=1; representatives=self.eess.best_per_component(); roots=[]; sources=[]
        for i,rep in enumerate(representatives):
            root,ok=self._homotopy_root(float(t),previous,step,rep,work)
            if ok: roots.append(root); sources.append(f"homotopy:{i}")
        from .trajroot import solve_bdf1_step
        direct=solve_bdf1_step(self.rhs,float(t),previous,step,predictor=predictor,jacobian=self.jacobian,atol=self._atol_for(previous.size),rtol=self.rtol,certification_tol=self.certification_tol); work+=direct.work
        if direct.accepted:roots.append(direct.y.copy());sources.append("direct")
        if not roots:return VectorPipelineResult(predictor,False,float("inf"),"no converged root candidate",VectorPipelineTrace(cloud.shape[0],0,representatives.shape[0],-1,"none",False),work)
        rm=np.ascontiguousarray(np.vstack(roots)); rs=self._native_scores(float(t),previous,step,rm,work); certified=np.flatnonzero(rs<=self.certification_tol)
        if certified.size==0:return VectorPipelineResult(predictor,False,float(np.min(rs)),"no root passed original-residual validation",VectorPipelineTrace(cloud.shape[0],0,representatives.shape[0],-1,"none",False),work)
        selected=int(min(certified,key=lambda idx:(float(np.linalg.norm(rm[idx]-predictor)),float(rs[idx]),sources[idx]))); y=rm[selected].copy(); score=float(rs[selected]); self.eess.update(y[None,:],np.array([score])); work.eess_candidate_evaluations+=1; update=online.update(features,y-predictor,scale); work.online_updates+=int(update.accepted); self._last_correction=y-predictor; self._last_predictor=predictor.copy(); labels=self.eess.component_labels; nearest=int(np.argmin(np.linalg.norm(self.eess.points-y[None,:],axis=1))); component=int(labels[nearest]); trace=VectorPipelineTrace(cloud.shape[0],int(np.unique(labels).size),representatives.shape[0],component,sources[selected],update.accepted); return VectorPipelineResult(y,True,score,"accepted",trace,work)
