"""Integrated scalar BDF1 vertical slice."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from .contracts import FloatArray,Jacobian,Rhs,as_state,positive_step,tolerance_scale,wrms_norm
from .eess import PersistentEESSState
from .homotopy import continuation_newton
from .integrators import bdf1_residual
from .online import OnlineLowRankAdapter
from .telemetry import WorkCounter
from .trajroot import solve_bdf1_step
@dataclass(slots=True)
class IntegratedStepTrace:
    candidate_count:int; component_count:int; used_topology:bool; used_homotopy:bool; online_update_attempted:bool; selected_source:str; predictor:float; online_correction:float; sources_considered:tuple[str,...]
@dataclass(slots=True)
class IntegratedStepResult:
    y:FloatArray; accepted:bool; scaled_residual:float; message:str; trace:IntegratedStepTrace; work:WorkCounter
@dataclass(slots=True)
class SolveTrace:
    t:FloatArray; y:FloatArray; success:bool; message:str; steps:list[IntegratedStepTrace]; work:WorkCounter
class WeaveBDF1Solver:
    def __init__(self,rhs:Rhs,*,step_size:float,polynomial_coefficients=None,jacobian:Jacobian|None=None,atol:float=1e-10,rtol:float=1e-8,certification_tol:float=1e-6,search_radius:float=1,grid_candidates:int=65,eess_points:int=16,component_radius:float=.25,homotopy_nodes:int=9,seed:int=0,online_adapter=None,candidate_builder=None):
        self.rhs=rhs;self.jacobian=jacobian;self.step_size=positive_step(step_size);self.atol=float(atol);self.rtol=float(rtol);self.certification_tol=float(certification_tol);self.search_radius=float(search_radius);self.grid_candidates=int(grid_candidates);self.homotopy_nodes=int(homotopy_nodes);self.coefficients=None if polynomial_coefficients is None else np.ascontiguousarray(np.asarray(polynomial_coefficients,dtype=np.float64));self.eess=PersistentEESSState(eess_points,component_radius);self.online=online_adapter or OnlineLowRankAdapter(state_dimension=1,feature_dimension=6,rank=3,learning_rate=.04,max_relative_correction=.25,seed=seed);self.candidate_builder=candidate_builder or self._default_candidate_builder;self._last_predictor=None;self._last_selected_delta=0.0
    def _default_candidate_builder(self,predictor,_previous,_h): return np.linspace(predictor-self.search_radius,predictor+self.search_radius,self.grid_candidates)[:,None]
    def _features(self,previous,h,f_previous,predictor): return np.array([previous,h,f_previous,predictor,self._last_selected_delta,float(np.unique(self.eess.component_labels).size if self.eess.component_labels.size else 1)])
    def _score(self,candidates,t,previous,h,work):
        if self.coefficients is not None:
            try:
                from . import _native
                scores=np.asarray(_native.poly_bdf1_scores(np.ascontiguousarray(candidates[:,0]),previous,h,self.coefficients,self.atol,self.rtol));work.native_candidate_evaluations+=len(candidates);return scores
            except (ImportError,AttributeError):pass
        out=[];prev=np.array([previous])
        for v in candidates[:,0]:
            cand=np.array([v]);out.append(wrms_norm(bdf1_residual(self.rhs,t,prev,h,cand),tolerance_scale(prev,cand,self.atol,self.rtol)));work.rhs_evaluations+=1;work.residual_evaluations+=1
        return np.asarray(out)
    def _natural(self,t,previous,h,work):
        params=np.linspace(0,1,self.homotopy_nodes)
        def r(x,lam):return x-previous-lam*h*np.asarray(self.rhs(t+h,x.copy()))
        j=None
        if self.jacobian is not None:
            def j(x,lam):return np.eye(1)-lam*h*np.asarray(self.jacobian(t+h,x.copy()))
        tr=continuation_newton(r,np.array([previous]),params,jacobian=j,tol=1e-12);work+=tr.work;work.rhs_evaluations+=tr.work.residual_evaluations;return (float(tr.states[-1,0]) if tr.converged else None)
    def step(self,t,y_previous,*,h=None):
        previous_state=as_state(y_previous);step=positive_step(self.step_size if h is None else h);work=WorkCounter();fprev=float(self.rhs(float(t),previous_state.copy())[0]);work.rhs_evaluations+=1;previous=float(previous_state[0]);predictor=previous+step*fprev
        if self.eess.points.size and self._last_predictor is not None:self.eess.transport(np.array([predictor-self._last_predictor]))
        features=self._features(previous,step,fprev,predictor);scale=np.array([max(1,abs(previous),abs(predictor))]);cor=float(self.online.propose(features,scale)[0]);base=np.asarray(self.candidate_builder(predictor,previous,step),dtype=np.float64)
        combined=np.vstack([base,[[predictor]],[[predictor+cor]],self.eess.points]) if self.eess.points.size else np.vstack([base,[[predictor]],[[predictor+cor]]]);scores=self._score(combined,float(t),previous,step,work);idx=np.argsort(scores)[:max(2*self.eess.max_points,self.eess.max_points+4)];self.eess.update(combined[idx],scores[idx]);work.eess_candidate_evaluations+=len(idx);work.topology_evaluations+=1;reps=self.eess.best_per_component();roots=[("predictor",predictor),("online",predictor+cor)];hc=self._natural(float(t),previous,step,work)
        if hc is not None:roots.append(("natural_homotopy",hc))
        for i,rep in enumerate(reps):
            rr=solve_bdf1_step(self.rhs,float(t),previous_state,step,predictor=rep,jacobian=self.jacobian,atol=self.atol,rtol=self.rtol,certification_tol=self.certification_tol);work+=rr.work
            if rr.nonlinear.converged:roots.append((f"eess_component_{i}",float(rr.y[0])))
        unique=[]
        for source,value in roots:
            if not any(abs(value-v)<=1e-10*max(1,abs(value),abs(v)) for _,v in unique):unique.append((source,value))
        arr=np.array([v for _,v in unique])[:,None];rs=self._score(arr,float(t),previous,step,work);cert=[i for i,s in enumerate(rs) if s<=self.certification_tol]
        if not cert:
            fallback=solve_bdf1_step(self.rhs,float(t),previous_state,step,jacobian=self.jacobian,atol=self.atol,rtol=self.rtol,certification_tol=self.certification_tol);work+=fallback.work;work.fallbacks+=1;trace=IntegratedStepTrace(len(combined),int(np.unique(self.eess.component_labels).size),True,True,False,"classical_fallback",predictor,cor,tuple(s for s,_ in unique));return IntegratedStepResult(fallback.y,fallback.accepted,fallback.scaled_residual,fallback.message,trace,work)
        si=min(cert,key=lambda i:(abs(unique[i][1]-predictor),rs[i],unique[i][0]));source,selected=unique[si];update=self.online.update(features,np.array([selected-predictor]),scale);work.online_updates+=int(update.accepted);self.eess.update(np.array([[selected]]),np.array([rs[si]]));self._last_predictor=predictor;self._last_selected_delta=selected-predictor;trace=IntegratedStepTrace(len(combined),int(np.unique(self.eess.component_labels).size),True,True,True,source,predictor,cor,tuple(s for s,_ in unique));return IntegratedStepResult(np.array([selected]),True,float(rs[si]),"accepted",trace,work)
    def solve(self,t_span,y0):
        t0,t_end=map(float,t_span);state=as_state(y0);times=[t0];states=[state.copy()];traces=[];work=WorkCounter();t=t0
        while t<t_end-1e-15:
            h=min(self.step_size,t_end-t);result=self.step(t,state,h=h);work+=result.work;traces.append(result.trace)
            if not result.accepted:return SolveTrace(np.asarray(times),np.vstack(states),False,result.message,traces,work)
            t+=h;state=result.y.copy();times.append(t);states.append(state)
        return SolveTrace(np.asarray(times),np.vstack(states),True,"success",traces,work)
