"""Cold-start online adaptation for the current IVP only."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
FloatArray=npt.NDArray[np.float64]
@dataclass(slots=True)
class OnlineUpdateResult:
    accepted:bool; loss_before:float; loss_after:float; message:str
class OnlineLowRankAdapter(nn.Module):
    def __init__(self,*,state_dimension:int,feature_dimension:int,rank:int=4,learning_rate:float=.03,max_relative_correction:float=.25,seed:int=0,device:str|torch.device="cpu"):
        super().__init__(); torch.manual_seed(seed); self.state_dimension=state_dimension; self.feature_dimension=feature_dimension; self.rank=rank; self.learning_rate=float(learning_rate); self.max_relative_correction=float(max_relative_correction); self.device=torch.device(device); self.encoder=nn.Linear(feature_dimension,rank,bias=False,dtype=torch.float64); self.decoder=nn.Linear(rank,state_dimension,bias=False,dtype=torch.float64); nn.init.normal_(self.encoder.weight,mean=0,std=1/max(1,feature_dimension)**.5); nn.init.zeros_(self.decoder.weight); self.to(self.device); self._initial_encoder=self.encoder.weight.detach().clone(); self._optimizer=torch.optim.SGD(self.parameters(),lr=self.learning_rate)
    def _tensor_inputs(self,features,scale):
        f=np.asarray(features,dtype=np.float64); s=np.asarray(scale,dtype=np.float64)
        if f.shape!=(self.feature_dimension,) or s.shape!=(self.state_dimension,) or np.any(s<=0): raise ValueError("invalid online inputs")
        return torch.as_tensor(f,dtype=torch.float64,device=self.device),torch.as_tensor(s,dtype=torch.float64,device=self.device)
    def forward(self,features,scale): return self.max_relative_correction*scale*torch.tanh(self.decoder(torch.tanh(self.encoder(features))))
    def propose(self,features,scale):
        ft,st=self._tensor_inputs(features,scale)
        with torch.no_grad(): p=self(ft,st)
        return p.detach().cpu().numpy().astype(np.float64,copy=True)
    def update(self,features,target_correction,scale):
        ft,st=self._tensor_inputs(features,scale); target=np.asarray(target_correction,dtype=np.float64)
        if target.shape!=(self.state_dimension,) or not np.all(np.isfinite(target)): raise ValueError("invalid target")
        tt=torch.as_tensor(target,dtype=torch.float64,device=self.device); snap={n:p.detach().clone() for n,p in self.named_parameters()}; self._optimizer.zero_grad(set_to_none=True); before=self(ft,st); loss=torch.mean(((before-tt)/st)**2); lb=float(loss.detach().cpu()); loss.backward(); torch.nn.utils.clip_grad_norm_(self.parameters(),10.0); self._optimizer.step()
        with torch.no_grad(): la=float(torch.mean(((self(ft,st)-tt)/st)**2).cpu()); finite=all(torch.all(torch.isfinite(p)) for p in self.parameters())
        if not np.isfinite(la) or not finite or la>lb*(1+1e-10):
            with torch.no_grad():
                for n,p in self.named_parameters(): p.copy_(snap[n])
            return OnlineUpdateResult(False,lb,la,"update rolled back")
        return OnlineUpdateResult(True,lb,la,"accepted")
    def reset(self):
        with torch.no_grad(): self.encoder.weight.copy_(self._initial_encoder); self.decoder.weight.zero_()
        self._optimizer=torch.optim.SGD(self.parameters(),lr=self.learning_rate)
