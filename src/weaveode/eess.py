"""Persistent, topology-aware finite expected-extended solution state."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from .topology import radius_components
FloatArray=npt.NDArray[np.float64]
IntArray=npt.NDArray[np.int64]
@dataclass(slots=True)
class PersistentEESSState:
    max_points:int
    component_radius:float
    points:FloatArray=field(default_factory=lambda:np.empty((0,0),dtype=np.float64))
    scores:FloatArray=field(default_factory=lambda:np.empty(0,dtype=np.float64))
    component_labels:IntArray=field(default_factory=lambda:np.empty(0,dtype=np.int64))
    generation:int=0
    candidate_evaluations:int=0
    def __post_init__(self)->None:
        if self.max_points<=0: raise ValueError("max_points must be positive")
        if not np.isfinite(self.component_radius) or self.component_radius<0.0: raise ValueError("component_radius must be finite and nonnegative")
    def update(self,points:npt.ArrayLike,scores:npt.ArrayLike)->None:
        incoming=np.asarray(points,dtype=np.float64); values=np.asarray(scores,dtype=np.float64)
        if incoming.ndim!=2 or incoming.shape[0]==0 or incoming.shape[1]==0: raise ValueError("points must be a non-empty two-dimensional array")
        if values.shape!=(incoming.shape[0],): raise ValueError("scores must contain one value per point")
        if not np.all(np.isfinite(incoming)) or not np.all(np.isfinite(values)): raise ValueError("points and scores must be finite")
        self.candidate_evaluations+=incoming.shape[0]
        if self.points.size==0: combined_points=np.ascontiguousarray(incoming); combined_scores=np.ascontiguousarray(values)
        else:
            if incoming.shape[1]!=self.points.shape[1]: raise ValueError("incoming point dimension changed")
            combined_points=np.vstack([self.points,incoming]); combined_scores=np.concatenate([self.scores,values])
        labels=radius_components(combined_points,self.component_radius); unique_labels=np.unique(labels)
        if unique_labels.size>self.max_points: raise RuntimeError(f"{unique_labels.size} topology components exceed EESS capacity {self.max_points}")
        selected=[]
        for label in unique_labels:
            members=np.flatnonzero(labels==label); selected.append(int(members[np.argmin(combined_scores[members])]))
        selected_set=set(selected)
        for index in np.argsort(combined_scores,kind="stable"):
            integer_index=int(index)
            if integer_index not in selected_set: selected.append(integer_index); selected_set.add(integer_index)
            if len(selected)>=self.max_points: break
        selected_array=np.asarray(selected,dtype=np.int64)
        self.points=np.ascontiguousarray(combined_points[selected_array]); self.scores=np.ascontiguousarray(combined_scores[selected_array]); self.component_labels=radius_components(self.points,self.component_radius)
    def transport(self,offset:npt.ArrayLike)->None:
        if self.points.size==0: raise RuntimeError("cannot transport an empty EESS population")
        shift=np.asarray(offset,dtype=np.float64)
        if shift.shape!=(self.points.shape[1],) or not np.all(np.isfinite(shift)): raise ValueError("offset must be a finite vector matching point dimension")
        self.points=np.ascontiguousarray(self.points+shift); self.component_labels=radius_components(self.points,self.component_radius); self.generation+=1
    def best_per_component(self)->FloatArray:
        if self.points.size==0: return np.empty((0,0),dtype=np.float64)
        representatives=[]
        for label in np.unique(self.component_labels):
            members=np.flatnonzero(self.component_labels==label); best=int(members[np.argmin(self.scores[members])]); representatives.append(self.points[best])
        return np.vstack(representatives)
