from typing import Callable

import numpy as np
import torch

DEFAULT_EPS: float = 1e-10
PADDED_Y_VALUE: int = -1


def sinkhorn_scaling(
    mat: torch.Tensor,
    mask: torch.Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> torch.Tensor:
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)

    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=DEFAULT_EPS)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=DEFAULT_EPS)

        if (
            torch.max(torch.abs(mat.sum(dim=2) - 1.0)) < tol
            and torch.max(torch.abs(mat.sum(dim=1) - 1.0)) < tol
        ):
            break

    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)

    return mat


def deterministic_neural_sort(
    device: torch.device, s: torch.Tensor, tau: float, mask: torch.Tensor
) -> torch.Tensor:
    n: int = s.size(1)
    one: torch.Tensor = torch.ones((n, 1), dtype=torch.float32, device=device)

    s = s.masked_fill(mask[:, :, None], -1e8)
    a_s: torch.Tensor = torch.abs(s - s.permute(0, 2, 1)).masked_fill(
        mask[:, :, None] | mask[:, None, :], 0.0
    )
    B: torch.Tensor = torch.matmul(a_s, torch.matmul(one, one.T))

    temp: list[torch.Tensor] = [
        n - m + 1 - 2 * (torch.arange(n - int(m), device=device) + 1)
        for m in mask.squeeze(-1).sum(dim=1)
    ]
    scaling: torch.Tensor = torch.stack(
        [
            torch.cat((t.type(torch.float32), torch.zeros(n - len(t), device=device)))
            for t in temp
        ]
    )

    C: torch.Tensor = torch.matmul(
        s.masked_fill(mask[:, :, None], 0.0), scaling.unsqueeze(-2)
    )
    p_max: torch.Tensor = (
        (C - B)
        .permute(0, 2, 1)
        .masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    )

    return torch.nn.functional.softmax(p_max / tau, dim=-1)


def stochastic_neural_sort(
    device: torch.device,
    s: torch.Tensor,
    n_samples: int,
    tau: float,
    mask: torch.Tensor,
    beta: float = 1.0,
    log_scores: bool = True,
    eps: float = 1e-10,
) -> torch.Tensor:
    s_positive = s + torch.abs(s.min())
    samples = beta * __sample_gumbel(
        [n_samples, s.size(0), s.size(1), 1], device=device
    )

    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * s.size(0), s.size(1), 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)
    p_hat = deterministic_neural_sort(device, s_perturb, tau, mask_repeated)

    return p_hat.view(n_samples, s.size(0), s.size(1), s.size(1))


def __sample_gumbel(
    samples_shape: list[int], device: torch.device, eps: float = 1e-10
) -> torch.Tensor:
    U: torch.Tensor = torch.rand(samples_shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def dcg(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    ats: list[int] | None = None,
    gain_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.pow(2, x) - 1,
    padding_indicator: int = PADDED_Y_VALUE,
) -> torch.Tensor:
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length: int = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds: torch.Tensor = __apply_mask_and_get_true_sorted_by_preds(
        y_pred, y_true, padding_indicator
    )

    discounts: torch.Tensor = (
        torch.tensor(1)
        / torch.log2(
            torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0
        )
    ).to(device=true_sorted_by_preds.device)

    gains: torch.Tensor = gain_fn(true_sorted_by_preds)

    discounted_gains: torch.Tensor = (gains * discounts)[:, : np.max(ats)]

    cum_dcg: torch.Tensor = torch.cumsum(discounted_gains, dim=1)

    ats_tensor: torch.Tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    return cum_dcg[:, ats_tensor]


def __apply_mask_and_get_true_sorted_by_preds(
    y_pred: torch.Tensor, y_true: torch.Tensor, padding_indicator: int = PADDED_Y_VALUE
) -> torch.Tensor:
    mask: torch.Tensor = y_true == padding_indicator

    y_pred[mask] = float("-inf")
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)
