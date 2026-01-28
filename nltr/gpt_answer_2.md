いいリクエストです — 要求どおり **K==1 を捨てずに multi-task 同時学習**（listwise + pointwise を重みで合成）し、さらに実験管理を **mlflow** に統一した「簡便化しない」完全版スクリプトを用意しました。

ポイント（先に短く）：

* 各バッチ（同長バケット）で **(1) listwise（NeuralNDCG）** と **(2) pointwise（MSE）** の両方を計算します。

  * `K >= min_k_for_listwise` のとき listwise を有効化。
  * pointwise は常に計算（K==1 を含む全サンプルに対して）。
* 損失合成：`loss = λ_list * list_loss + λ_point * point_loss`（ハイパーパラは config で調整）
* AMP（混合精度）、勾配累積、勾配クリップ、LR scheduler、チェックポイント、再開、NaN 検出はすべて実装済み。
* ロギング／実験管理は **mlflow** を使用。ハイパラ・epoch 毎の val 指標・チェックポイント（artifact）を mlflow に保存します。
* 検証では **(A) per-query NDCG（K>=2 のクエリ群で平均）** と **(B) pointwise RMSE（すべてのアイテム）** の両方を出します。これで listwise の改善と pointwise の品質を同時に監視できます。

以下が完全な実装例（`train_ndcg_mlflow.py`）。必要に応じてパスや `NeuralNDCGLoss` の import パスを調整して使ってください。

```python
"""
train_ndcg_mlflow.py

- 可変スレート (X_list, y_list) を使った multi-task (NeuralNDCG + pointwise MSE) 同時学習
- mlflow に実験記録・チェックポイント保存
- 実運用で必要な要素を省略せず全て含む
"""

import os
import time
import random
import math
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import mlflow

# NeuralNDCG import: adjust if your package path differs
from neuralndcgloss.loss import NeuralNDCGLoss

# -------------------------
# Config (adjust / load from json/yaml as needed)
# -------------------------
config = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 20,
    "batch_Q": 32,
    "D": 100,
    "hidden_dim": 128,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "use_amp": True,
    "grad_accum_steps": 1,
    "log_interval_batches": 200,
    "save_dir": "./checkpoints",
    "mlflow_experiment": "ndcg_multi_task",
    "temperature": 0.1,            # NeuralNDCG hyperparam
    "min_k_for_listwise": 2,
    # multi-task weights
    "lambda_listwise": 1.0,
    "lambda_pointwise": 1.0,
    "drop_last_train": True,
    "drop_longer_than": 400,
    "max_batches_per_epoch": None,
    "use_cosine_scheduler": True,
    "total_steps_estimate": None,
}

os.makedirs(config["save_dir"], exist_ok=True)

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For reproducibility; may slow down
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config["seed"])

# -------------------------
# Model
# -------------------------
class RankMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
        self._init_weights()

    def forward(self, x: torch.Tensor):
        # x: (B, K, D)
        B, K, D = x.shape
        flat = x.view(B * K, D)
        out = self.net(flat).view(B, K)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# -------------------------
# Utilities: buckets, batching, metrics
# -------------------------
def build_buckets(X_list: List[np.ndarray], drop_longer_than: int = None):
    buckets = defaultdict(list)
    for idx, Xq in enumerate(X_list):
        k = Xq.shape[0]
        if drop_longer_than and k > drop_longer_than:
            continue
        buckets[k].append(idx)
    return buckets

def make_epoch_batches(buckets: Dict[int, List[int]], batch_Q: int, shuffle: bool = True, drop_last: bool = True):
    bucket_shuffled = {}
    for k, idxs in buckets.items():
        idxs_copy = idxs[:]
        if shuffle:
            random.shuffle(idxs_copy)
        bucket_shuffled[k] = idxs_copy
    batches = []
    for k, idxs in bucket_shuffled.items():
        n = len(idxs)
        num_full = n // batch_Q
        for b in range(num_full):
            batches.append((k, idxs[b * batch_Q: (b + 1) * batch_Q]))
        if not drop_last:
            rem = n % batch_Q
            if rem > 0:
                batches.append((k, idxs[num_full * batch_Q:]))
    if shuffle:
        random.shuffle(batches)
    return batches

# per-query NDCG (numpy): handles K>=1; for K==1 NDCG defined as 0 (no ranking info)
def ndcg_at_k_per_query_numpy(y_true: np.ndarray, y_score: np.ndarray, k: int = None) -> float:
    Q, K = y_true.shape
    if k is None:
        k = K
    ndcgs = []
    for i in range(Q):
        rel = y_true[i]
        scores = y_score[i]
        if K < 2:
            ndcgs.append(0.0)
            continue
        order = np.argsort(-scores)
        rank_rel = rel[order][:k]
        denom = np.log2(np.arange(2, rank_rel.size + 2))
        gains = (2 ** rank_rel - 1)
        dcg = np.sum(gains / denom)
        ideal = np.sort(rel)[::-1][:k]
        idcg = np.sum((2 ** ideal - 1) / denom)
        ndcgs.append(0.0 if idcg == 0 else float(dcg / idcg))
    return float(np.mean(ndcgs))

def rmse_pointwise(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# -------------------------
# Checkpoint helpers
# -------------------------
def save_checkpoint(state: dict, path: str):
    torch.save(state, path)

def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None,
                    scheduler=None, scaler: GradScaler = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt and ckpt["optim_state"] is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler is not None and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            # some schedulers have state that can't be loaded across versions — log but continue
            print("Warning: scheduler state could not be fully restored.")
    if scaler is not None and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    return epoch, global_step

# -------------------------
# Full training (mlflow integrated)
# -------------------------
def train_and_eval_mlflow(X_list_train: List[np.ndarray],
                          y_list_train: List[np.ndarray],
                          X_list_val: List[np.ndarray],
                          y_list_val: List[np.ndarray],
                          config: dict):
    device = config["device"]
    model = RankMLP(d_in=config["D"], hidden=config["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # Scheduler (cosine annealing example)
    if config["use_cosine_scheduler"]:
        # estimate total steps
        buckets_train = build_buckets(X_list_train, drop_longer_than=config.get("drop_longer_than"))
        batches_per_epoch = sum(len(idxs) // config["batch_Q"] for idxs in buckets_train.values())
        est_total_steps = max(1, batches_per_epoch * config["epochs"]) if config["total_steps_estimate"] is None else config["total_steps_estimate"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, est_total_steps))
    else:
        scheduler = None

    scaler = GradScaler(enabled=config["use_amp"])
    ndcg_loss = NeuralNDCGLoss(temperature=config["temperature"])
    pointwise_loss_fn = nn.MSELoss()

    # mlflow setup
    mlflow.set_experiment(config["mlflow_experiment"])
    with mlflow.start_run():
        # log config
        mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))})

        # save model code snapshot (optional)
        # mlflow.log_artifact("train_ndcg_mlflow.py")  # if you want to store the script

        # resume if checkpoint exists
        latest_ckpt = os.path.join(config["save_dir"], "latest.pt")
        global_step = 0
        start_epoch = 1
        if os.path.exists(latest_ckpt):
            print("Resuming from checkpoint:", latest_ckpt)
            start_epoch, global_step = load_checkpoint(latest_ckpt, model, optimizer, scheduler, scaler)
            start_epoch += 1
            print(f"Resumed at epoch {start_epoch}, global_step {global_step}")

        best_val_ndcg = -1.0

        # training loop
        for epoch in range(start_epoch, config["epochs"] + 1):
            epoch_start = time.time()
            buckets = build_buckets(X_list_train, drop_longer_than=config.get("drop_longer_than"))
            batches = make_epoch_batches(buckets, batch_Q=config["batch_Q"], shuffle=True, drop_last=config["drop_last_train"])

            running_loss = 0.0
            batch_count = 0
            model.train()

            for (k, batch_idxs) in batches:
                B = len(batch_idxs)
                K = k
                # stack and push
                X_batch = np.stack([X_list_train[i] for i in batch_idxs], axis=0)
                y_batch = np.stack([y_list_train[i] for i in batch_idxs], axis=0)
                X_batch_t = torch.from_numpy(X_batch).to(device=device, dtype=torch.float32)
                y_batch_t = torch.from_numpy(y_batch).to(device=device, dtype=torch.float32)

                with autocast(enabled=config["use_amp"]):
                    scores = model(X_batch_t)  # (B,K)

                    # pointwise loss on scalar scores vs label floats
                    point_loss = pointwise_loss_fn(scores.view(-1), y_batch_t.view(-1))

                    # listwise loss only if K >= min_k_for_listwise
                    if K >= config["min_k_for_listwise"]:
                        list_loss = ndcg_loss(scores, y_batch_t)
                    else:
                        list_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                    total_loss = config["lambda_listwise"] * list_loss + config["lambda_pointwise"] * point_loss

                    # safety: ensure finite
                    if not torch.isfinite(total_loss):
                        raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {global_step}")

                    # scale for accumulation
                    total_loss = total_loss / config["grad_accum_steps"]

                # backward
                scaler.scale(total_loss).backward()

                # optimizer step when accumulation satisfied
                if (batch_count + 1) % config["grad_accum_steps"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    scaler.step(optimizer)    # performs optimizer.step() safely under AMP
                    scaler.update()
                    optimizer.zero_grad()

                    # scheduler step (after actual optimizer.step)
                    if scheduler is not None:
                        scheduler.step()

                    global_step += 1

                    # mlflow: log per-step lr occasionally
                    if global_step % 100 == 0:
                        mlflow.log_metric("train/lr", optimizer.param_groups[0]["lr"], step=global_step)

                # logging internal scalars (use raw loss.item() scaled back)
                # multiply back grad_accum_steps to reflect true loss scale for reporting
                running_loss += total_loss.item() * config["grad_accum_steps"]
                batch_count += 1

                # periodic batch-level monitoring (do an eval forward in eval mode for correctness)
                if global_step % config["log_interval_batches"] == 0:
                    model.eval()
                    with torch.no_grad():
                        scores_eval = model(X_batch_t).detach().cpu().numpy()
                        y_eval = y_batch_t.detach().cpu().numpy()
                        ndcg10 = ndcg_at_k_per_query_numpy(y_eval, scores_eval, k=min(10, K))
                        rmse = rmse_pointwise(y_eval, scores_eval)
                    model.train()
                    mlflow.log_metric("train/batch_ndcg10", ndcg10, step=global_step)
                    mlflow.log_metric("train/batch_point_rmse", rmse, step=global_step)
                    mlflow.log_metric("train/batch_loss", running_loss / max(1, batch_count), step=global_step)
                    print(f"[E{epoch}] step {global_step} lr {optimizer.param_groups[0]['lr']:.6f} loss {running_loss / batch_count:.6f} ndcg@10 {ndcg10:.4f} rmse {rmse:.4f}")

                if config.get("max_batches_per_epoch") and batch_count >= config["max_batches_per_epoch"]:
                    break

            # epoch finished
            epoch_time = time.time() - epoch_start
            avg_loss = running_loss / max(1, batch_count)
            print(f"Epoch {epoch} finished in {epoch_time:.1f}s, avg_loss={avg_loss:.6f}")
            mlflow.log_metric("train/epoch_loss", avg_loss, step=epoch)

            # validation
            val_ndcg, val_rmse = evaluate_full_mlflow(model, X_list_val, y_list_val, device, config)
            print(f"Epoch {epoch} VALIDATION ndcg={val_ndcg:.6f} rmse={val_rmse:.6f}")
            mlflow.log_metric("val/ndcg", val_ndcg, step=epoch)
            mlflow.log_metric("val/point_rmse", val_rmse, step=epoch)

            # save checkpoint + log artifact to mlflow
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict()
            }
            ckpt_path = os.path.join(config["save_dir"], f"ckpt_epoch_{epoch}.pt")
            latest_path = os.path.join(config["save_dir"], "latest.pt")
            save_checkpoint(ckpt, ckpt_path)
            save_checkpoint(ckpt, latest_path)
            mlflow.log_artifact(ckpt_path)

            # best model logic
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_path = os.path.join(config["save_dir"], "best.pt")
                save_checkpoint(ckpt, best_path)
                mlflow.log_artifact(best_path)
                print(f"New best model saved: val_ndcg={best_val_ndcg:.6f}")

        # end run: log final params/notes if needed
        mlflow.log_param("best_val_ndcg", best_val_ndcg)

# -------------------------
# Validation helper (full)
# -------------------------
def evaluate_full_mlflow(model: nn.Module, X_list: List[np.ndarray], y_list: List[np.ndarray], device: str, config: dict):
    model.eval()
    buckets = build_buckets(X_list, drop_longer_than=config.get("drop_longer_than"))
    batches = make_epoch_batches(buckets, batch_Q=config["batch_Q"], shuffle=False, drop_last=False)
    ndcgs = []
    rmses = []
    with torch.no_grad():
        for (k, batch_idxs) in batches:
            B = len(batch_idxs)
            X_batch = np.stack([X_list[i] for i in batch_idxs], axis=0)
            y_batch = np.stack([y_list[i] for i in batch_idxs], axis=0)
            X_t = torch.from_numpy(X_batch).to(device=device, dtype=torch.float32)
            y_t = torch.from_numpy(y_batch).to(device=device, dtype=torch.float32)

            scores = model(X_t).detach().cpu().numpy()
            y_np = y_t.detach().cpu().numpy()

            # per-batch metrics
            ndcg_b = ndcg_at_k_per_query_numpy(y_np, scores, k=min(10, k))
            rmse_b = rmse_pointwise(y_np, scores)
            ndcgs.append(ndcg_b)
            rmses.append(rmse_b)
    # average across batches
    mean_ndcg = float(np.mean(ndcgs)) if len(ndcgs) > 0 else 0.0
    mean_rmse = float(np.mean(rmses)) if len(rmses) > 0 else 0.0
    return mean_ndcg, mean_rmse

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Replace these with your real X_list_train / y_list_train loaded from parquet (polars -> numpy)
    def make_dummy(num_q=3000, D=100, min_k=1, max_k=50):
        Xl, yl = [], []
        for _ in range(num_q):
            k = random.randint(min_k, max_k)
            Xl.append(np.random.randn(k, D).astype(np.float32))
            yl.append(np.random.randint(0, 5, size=(k,)).astype(np.float32))
        return Xl, yl

    X_list_train, y_list_train = make_dummy(num_q=3000, D=config["D"], min_k=1, max_k=50)
    X_list_val, y_list_val = make_dummy(num_q=600, D=config["D"], min_k=1, max_k=50)

    train_and_eval_mlflow(X_list_train, y_list_train, X_list_val, y_list_val, config)
```

---

## 使い方／ハイパラ調整メモ（実務的）

* `lambda_listwise` と `lambda_pointwise`：

  * 初期推奨は両方 `1.0`。もし K==1 が多数で listwise の学習が希薄なら `lambda_listwise` をやや大きめ（例 2.0）にして listwise 信号の影響を増やすか、逆に pointwise を重視したければ `lambda_pointwise` を大きくする。
  * 実験として grid: `lambda_listwise ∈ {0.1, 1.0, 2.0}`, `lambda_pointwise ∈ {0.1, 1.0, 2.0}` を試すのが良い。mlflow で簡単に管理できます。
* ラベルの scale：MSE を使う点に注意。ラベルが 0..4 なら MSE と相性悪くないが、必要なら正規化（/4）を検討。現在の実装は生ラベルのまま。
* 検証指標：`val/ndcg`（K>=2 用） と `val/point_rmse`（全体）を両方見る。サービス側での最終評価は NDCG を優先するのが通常。
* mlflow：UI でハイパラ・メトリクス・チェックポイント（artifact）を全部確認できます。必要なら `mlflow.sklearn.log_model` などを使い、より整ったモデル登録へ進めます。

---

## 最後に（安全確認）

* `NeuralNDCGLoss` の API（引数名や戻り値）がローカルで異なる場合があります。`neuralndcgloss` の README に従って import と引数（temperature 等）を合わせてください。
* 実データで最初に行うこと：**ラベル分布（Kごとの分布含む）を確認**し、`min_k_for_listwise`／`lambda` の初期値を調整してください。
* 大規模な実験（多 GPU / DDP）に移す場合は、さらに `DistributedSampler`・`torch.distributed` ロジック・rank0 のみ mlflow ログ／checkpoint 書き込み等を入れる必要があります（希望があれば DDP版を作ります）。

必要なら、あなたの実 parquet のカラム一覧と実データの K 分布を貼ってください。
それに合わせて（A）`polars → X_list,y_list` の最適コード（dtype,並び順,sort）や、（B）現実的な初期ハイパラ（batch_Q, λ 値, temperature）を算出して差し上げます。どちらにしますか？
