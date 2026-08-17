"""Small-cloud topology primitives used to allocate EESS and homotopy work."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from math import inf
from typing import Iterable
import numpy as np
import numpy.typing as npt
FloatArray=npt.NDArray[np.float64]; IntArray=npt.NDArray[np.int64]
def _as_points(points):
    a=np.asarray(points,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]==0 or a.shape[1]==0 or not np.all(np.isfinite(a)): raise ValueError("points must be a non-empty finite two-dimensional array")
    return np.ascontiguousarray(a)
def radius_components(points,radius):
    a=_as_points(points); radius=float(radius)
    try:
        from . import _native
        return np.asarray(_native.radius_components(a,radius),dtype=np.int64)
    except (ImportError,AttributeError): pass
    parent=np.arange(a.shape[0],dtype=np.int64)
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=int(parent[x])
        return x
    def union(i,j):
        i=find(i); j=find(j)
        if i!=j: parent[j]=i
    r2=radius*radius
    for i in range(a.shape[0]):
        d=a[i+1:]-a[i]; ds=np.einsum("ij,ij->i",d,d)
        for rel in np.flatnonzero(ds<=r2): union(i,i+1+int(rel))
    mapping={}; labels=np.empty(a.shape[0],dtype=np.int64)
    for i in range(a.shape[0]): labels[i]=mapping.setdefault(find(i),len(mapping))
    return labels
def merge_profile(points,radii:Iterable[float]):
    a=_as_points(points); return [(float(r),int(np.unique(radius_components(a,float(r))).size)) for r in radii]
@dataclass(frozen=True,slots=True)
class _Simplex:
    vertices:tuple[int,...]; dimension:int; filtration:float
def vietoris_rips_persistence(points,*,max_dimension=1,max_radius=None,point_limit=64):
    a=_as_points(points); n=a.shape[0]
    if n>point_limit: raise ValueError(f"point cloud exceeds point_limit={point_limit}")
    d=a[:,None,:]-a[None,:,:]; dist=np.sqrt(np.einsum("ijk,ijk->ij",d,d)); cutoff=float(np.max(dist)) if max_radius is None else float(max_radius)
    simplices=[_Simplex((i,),0,0.0) for i in range(n)]; ef={}
    for i,j in combinations(range(n),2):
        v=float(dist[i,j])
        if v<=cutoff: ef[(i,j)]=v; simplices.append(_Simplex((i,j),1,v))
    if max_dimension>=1:
        for i,j,k in combinations(range(n),3):
            edges=((i,j),(i,k),(j,k))
            if all(e in ef for e in edges): simplices.append(_Simplex((i,j,k),2,max(ef[e] for e in edges)))
    simplices.sort(key=lambda s:(s.filtration,s.dimension,s.vertices)); si={s.vertices:i for i,s in enumerate(simplices)}; reduced=[]; piv={}; positive=[]; death={}
    for ci,s in enumerate(simplices):
        col=set() if s.dimension==0 else ({si[(s.vertices[0],)],si[(s.vertices[1],)]} if s.dimension==1 else {si[(s.vertices[0],s.vertices[1])],si[(s.vertices[0],s.vertices[2])],si[(s.vertices[1],s.vertices[2])]})
        while col:
            p=max(col); prev=piv.get(p)
            if prev is None: break
            col.symmetric_difference_update(reduced[prev])
        reduced.append(col)
        if not col: positive.append(ci)
        else: p=max(col); piv[p]=ci; death[p]=s.filtration
    out={dim:[] for dim in range(max_dimension+1)}
    for bi in positive:
        s=simplices[bi]
        if s.dimension<=max_dimension: out[s.dimension].append((s.filtration,death.get(bi,inf)))
    return out
