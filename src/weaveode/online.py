"""Cold-start online adaptation for the current IVP only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True)
class OnlineUpdateResult:
    accepted: bool
    loss_before: float
    loss_after: float
    message: str


class OnlineLowRankAdapter(nn.Module):
    """A tiny bounded proposal adapter trained only on the active IVP."""

    def __init__(
        self,
        *,
        state_dimension: int,
        feature_dimension: int,
        rank: int = 4,
        learning_rate: float = 0.03,
        max_relative_correction: float = 0.25,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if state_dimension <= 0 or feature_dimension <= 0 or rank <= 0:
            raise ValueError("dimensions and rank must be positive")
        if learning_rate <= 0.0 or max_relative_correction <= 0.0:
            raise ValueError("learning_rate and correction bound must be positive")
        torch.manual_seed(seed)
        self.state_dimension = state_dimension
        self.feature_dimension = feature_dimension
        self.rank = rank
        self.learning_rate = float(learning_rate)
        self.max_relative_correction = float(max_relative_correction)
        self.device = torch.device(device)
        self.encoder = nn.Linear(feature_dimension, rank, bias=False, dtype=torch.float64)
        self.decoder = nn.Linear(rank, state_dimension, bias=False, dtype=torch.float64)
        nn.init.normal_(self.encoder.weight, mean=0.0, std=1.0 / max(1, feature_dimension) ** 0.5)
        nn.init.zeros_(self.decoder.weight)
        self.to(self.device)
        self._initial_encoder = self.encoder.weight.detach().clone()
        self._optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def _tensor_inputs(self, features: npt.ArrayLike, scale: npt.ArrayLike) -> tuple[torch.Tensor, torch.Tensor]:
        feature_array = np.asarray(features, dtype=np.float64)
        scale_array = np.asarray(scale, dtype=np.float64)
        if feature_array.shape != (self.feature_dimension,):
            raise ValueError("features have the wrong shape")
        if scale_array.shape != (self.state_dimension,):
            raise ValueError("scale has the wrong shape")
        if not np.all(np.isfinite(feature_array)) or not np.all(np.isfinite(scale_array)):
            raise ValueError("features and scale must be finite")
        if np.any(scale_array <= 0.0):
            raise ValueError("scale must be positive")
        return (
            torch.as_tensor(feature_array, dtype=torch.float64, device=self.device),
            torch.as_tensor(scale_array, dtype=torch.float64, device=self.device),
        )

    def forward(self, features: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        raw = self.decoder(torch.tanh(self.encoder(features)))
        return self.max_relative_correction * scale * torch.tanh(raw)

    def propose(self, features: npt.ArrayLike, scale: npt.ArrayLike) -> FloatArray:
        feature_tensor, scale_tensor = self._tensor_inputs(features, scale)
        with torch.no_grad():
            proposal = self(feature_tensor, scale_tensor)
        return proposal.detach().cpu().numpy().astype(np.float64, copy=True)

    def update(self, features: npt.ArrayLike, target_correction: npt.ArrayLike, scale: npt.ArrayLike) -> OnlineUpdateResult:
        feature_tensor, scale_tensor = self._tensor_inputs(features, scale)
        target = np.asarray(target_correction, dtype=np.float64)
        if target.shape != (self.state_dimension,) or not np.all(np.isfinite(target)):
            raise ValueError("target_correction has the wrong shape or contains NaN/Inf")
        target_tensor = torch.as_tensor(target, dtype=torch.float64, device=self.device)
        snapshot = {name: parameter.detach().clone() for name, parameter in self.named_parameters()}
        self._optimizer.zero_grad(set_to_none=True)
        before_prediction = self(feature_tensor, scale_tensor)
        before_loss_tensor = torch.mean(((before_prediction - target_tensor) / scale_tensor) ** 2)
        before_loss = float(before_loss_tensor.detach().cpu())
        if not np.isfinite(before_loss):
            return OnlineUpdateResult(False, before_loss, before_loss, "non-finite initial loss")
        before_loss_tensor.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self._optimizer.step()
        with torch.no_grad():
            after_prediction = self(feature_tensor, scale_tensor)
            after_loss_tensor = torch.mean(((after_prediction - target_tensor) / scale_tensor) ** 2)
            after_loss = float(after_loss_tensor.detach().cpu())
            finite_parameters = all(torch.all(torch.isfinite(parameter)) for parameter in self.parameters())
        if not np.isfinite(after_loss) or not finite_parameters or after_loss > before_loss * (1.0 + 1e-10):
            with torch.no_grad():
                for name, parameter in self.named_parameters():
                    parameter.copy_(snapshot[name])
            return OnlineUpdateResult(False, before_loss, after_loss, "update rolled back")
        return OnlineUpdateResult(True, before_loss, after_loss, "accepted")

    def reset(self) -> None:
        with torch.no_grad():
            self.encoder.weight.copy_(self._initial_encoder)
            self.decoder.weight.zero_()
        self._optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
