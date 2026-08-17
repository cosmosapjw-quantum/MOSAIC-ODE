"""Small analytic problems used by the integrated coding research loop."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
FloatArray=npt.NDArray[np.float64]
@dataclass(frozen=True,slots=True)
class PolynomialScalarProblem:
    coefficients:FloatArray
    def __post_init__(self):
        c=np.asarray(self.coefficients,dtype=np.float64)
        if c.ndim!=1 or c.size==0 or not np.all(np.isfinite(c)): raise ValueError("coefficients must be a non-empty finite vector")
        object.__setattr__(self,"coefficients",np.ascontiguousarray(c))
    def rhs(self,_t,y):
        state=np.asarray(y,dtype=np.float64); value=0.0
        for coefficient in self.coefficients[::-1]: value=value*state[0]+float(coefficient)
        return np.array([value],dtype=np.float64)
    def jacobian(self,_t,y):
        state=np.asarray(y,dtype=np.float64); derivative=sum(power*self.coefficients[power]*state[0]**(power-1) for power in range(1,self.coefficients.size)); return np.array([[derivative]],dtype=np.float64)
