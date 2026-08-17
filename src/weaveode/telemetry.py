"""Work accounting shared by the research integrators and co-solver."""
from __future__ import annotations
from dataclasses import asdict,dataclass
@dataclass(slots=True)
class WorkCounter:
    residual_evaluations:int=0; rhs_evaluations:int=0; jacobian_evaluations:int=0; jvp_evaluations:int=0; accepted_steps:int=0; rejected_steps:int=0; newton_iterations:int=0; krylov_iterations:int=0; native_candidate_evaluations:int=0; topology_evaluations:int=0; eess_candidate_evaluations:int=0; online_updates:int=0; fallbacks:int=0
    def to_dict(self): return asdict(self)
    def __iadd__(self,other):
        for name in self.__dataclass_fields__: setattr(self,name,getattr(self,name)+getattr(other,name))
        return self
