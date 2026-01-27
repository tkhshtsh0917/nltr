import torch
import torch.nn as nn

from .utils import (
    DEFAULT_EPS,
    PADDED_Y_VALUE,
    dcg,
    deterministic_neural_sort,
    sinkhorn_scaling,
    stochastic_neural_sort,
)


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class NeuralNDCGLoss(nn.Module):
    def __init__(
        self,
        device: torch.device = get_torch_device(),
        padded_value_indicator: int = PADDED_Y_VALUE,
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
    ) -> None:
        super(NeuralNDCGLoss, self).__init__()

        self.device = device
        self.padded_value_indicator = padded_value_indicator
        self.temperature = temperature
        self.powered_relevancies = powered_relevancies
        self.k = k
        self.stochastic = stochastic
        self.n_samples = n_samples
        self.beta = beta
        self.log_scores = log_scores
        self.max_iter = max_iter
        self.tol = tol
        self.transposed = transposed

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        k: int = self.k if self.k != -1 else y_true.shape[1]
        mask: torch.Tensor = y_true == self.padded_value_indicator

        P_hat: torch.Tensor = self.compute_P_hat(y_pred, mask)
        if not self.transposed:
            discounted_gains: torch.Tensor = (
                self.compute_discounted_gains_not_transposed(P_hat, y_true, mask, k)
            )
        else:
            discounted_gains = self.compute_discounted_gains_transposed(
                P_hat, y_true, mask, k
            )

        idcg: torch.Tensor = self.compute_idcg(y_true, k)
        ndcg: torch.Tensor = self.compute_ndcg(discounted_gains, idcg, k)
        assert (ndcg < 0.0).sum() == 0, "every ndcg should be non-negative"

        if torch.all(idcg == 0.0):
            return torch.tensor(0.0)

        mean_ndcg = ndcg.sum() / ((~(idcg == 0.0)).sum() * ndcg.shape[0])  # type: ignore
        return -1.0 * mean_ndcg  # -1 cause we want to maximize NDCG

    def compute_P_hat(self, y_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.stochastic:
            return stochastic_neural_sort(
                device=self.device,
                s=y_pred.unsqueeze(-1),
                n_samples=self.n_samples,
                tau=self.temperature,
                mask=mask,
                beta=self.beta,
                log_scores=self.log_scores,
            )
        else:
            return deterministic_neural_sort(
                device=self.device,
                s=y_pred.unsqueeze(-1),
                tau=self.temperature,
                mask=mask,
            ).unsqueeze(0)

    def compute_discounted_gains_not_transposed(
        self, P_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, k: int
    ) -> torch.Tensor:
        P_hat = sinkhorn_scaling(
            P_hat.view(-1, P_hat.shape[2], P_hat.shape[3]),
            mask.repeat_interleave(P_hat.shape[0], dim=0),
            tol=self.tol,
            max_iter=self.max_iter,
        ).view(-1, y_true.shape[0], P_hat.shape[2], P_hat.shape[3])
        P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.0)

        y_true_masked: torch.Tensor = (
            y_true.masked_fill(mask, 0.0).unsqueeze(-1).unsqueeze(0)
        )
        if self.powered_relevancies:
            y_true_masked = torch.pow(2.0, y_true_masked) - 1.0

        ground_truth: torch.Tensor = torch.matmul(P_hat, y_true_masked).squeeze(-1)
        discounts: torch.Tensor = 1.0 / torch.log2(
            torch.arange(y_true.shape[-1], dtype=torch.float, device=self.device) + 2.0
        )

        return ground_truth * discounts

    def compute_discounted_gains_transposed(
        self, P_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, k: int
    ) -> torch.Tensor:
        P_hat_masked: torch.Tensor = sinkhorn_scaling(
            P_hat.view(-1, y_true.shape[1], y_true.shape[1]),
            mask.repeat_interleave(P_hat.shape[0], dim=0),
            tol=self.tol,
            max_iter=self.max_iter,
        ).view(P_hat.shape[0], y_true.shape[0], y_true.shape[1], y_true.shape[1])

        discounts: torch.Tensor = 1.0 / torch.log2(
            torch.arange(y_true.shape[-1], dtype=torch.float, device=self.device) + 2.0
        )
        discounts[k:] = 0.0
        discounts = discounts[None, None, :, None]
        discounts = torch.matmul(P_hat_masked.permute(0, 1, 3, 2), discounts).squeeze(
            -1
        )

        if self.powered_relevancies:
            gains: torch.Tensor = torch.pow(2.0, y_true) - 1
        else:
            gains = y_true

        return gains.unsqueeze(0) * discounts

    def compute_idcg(self, y_true: torch.Tensor, k: int) -> torch.Tensor:
        dcg_func: torch.Tensor = dcg(y_true, y_true, ats=[k])
        if not self.powered_relevancies:
            dcg_func = dcg(y_true, y_true, ats=[k], gain_fn=lambda x: x)

        return dcg_func.permute(1, 0) if not self.transposed else dcg_func.squeeze()

    def compute_ndcg(
        self, discounted_gains: torch.Tensor, idcg: torch.Tensor, k: int
    ) -> torch.Tensor:
        if not self.transposed:
            discounted_gains = discounted_gains[:, :, :k]

        ndcg: torch.Tensor = discounted_gains.sum(
            dim=-1 if not self.transposed else 2
        ) / (idcg + DEFAULT_EPS)
        idcg_mask: torch.Tensor = idcg == 0.0

        return ndcg.masked_fill(
            idcg_mask.repeat(ndcg.shape[0], 1) if not self.transposed else idcg_mask,
            0.0,
        )
