import torch
import torch.nn as nn

class NeuralNDCGLoss(nn.Module):
    def __init__(
        self,
        device: torch.device,
        padded_value_indicator: int,
        temperature: float = 1.0,
        powered_relevancies: bool = True,
        k: int = -1,
        stochastic: bool = False,
        n_samples: int = 32,
        beta: float = 0.1,
        log_scores: bool = True,
        max_iter: int = 50,
        tol: float = 1e-6,
        transposed: bool = False,
    ) -> None: ...
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor: ...
    def compute_P_hat(
        self, y_pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor: ...
    def compute_discounted_gains_not_transposed(
        self, P_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, k: int
    ) -> torch.Tensor: ...
    def compute_discounted_gains_transposed(
        self, P_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, k: int
    ) -> torch.Tensor: ...
    def compute_idcg(self, y_true: torch.Tensor, k: int) -> torch.Tensor: ...
    def compute_ndcg(
        self, discounted_gains: torch.Tensor, idcg: torch.Tensor, k: int
    ) -> torch.Tensor: ...
