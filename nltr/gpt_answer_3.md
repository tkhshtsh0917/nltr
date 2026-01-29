良い指摘 — その3点（**勾配累積の反映 / `autocast`・`GradScaler` の `enabled` フラグ / `if __name__=="__main__":` の追加**）をすべて取り込んだ「簡便化しない」完全版スクリプトを作りました。前回提示の設計方針・mlflow・RAdamScheduleFree・multi-task（listwise + pointwise）の方針はそのまま維持します。

下のスクリプトは実行可能な形で、かつ運用的に必要な要素（再現性設定、チェックポイント、resume、NaN 検知、grad accumulation、安全な AMP の使い方、mlflow ログ、`optimizer.train()/eval()` の呼び出しなど）をすべて含めています。`load_dataset()` 部分だけご自分の実データ読み込み (polars → `X_list`, `y_list`) に差し替えてください（サンプルとして簡易 loader を入れています）。

重要：

* `GradScaler(enabled=...)` と `autocast(enabled=...)` を **明示的に config["use_amp"]** と連動させています。
* `grad_accum_steps` を適用し、`scaler.step(optimizer)`（AMP 用）を **実際に optimizer.step の代わりに使う**よう統一しました。
* `if __name__ == "__main__":` 部で、mlflow run を開始し、データの読み込み → train 関数の呼び出しを行います。

以下が完全版スクリプトです。必要ならあなたの parquet 読み込み関数 (`load_parquet_to_torch` 等) を挿入してすぐ動くようにします（要 polars）。

```python
# train_with_RAdamScheduleFree_mlflow.py
"""
完全版トレーニングスクリプト（簡便化なし）
- Optimizer: RAdamScheduleFree (schedulefree パッケージ)
- Multi-task: pointwise (MSE) + listwise (NeuralNDCG)
- grad_accum_steps 対応
- AMP: autocast(enabled=config["use_amp"]) + GradScaler(enabled=config["use_amp"])
- mlflow により実験管理・チェックポイント保存
- model weight init via self.apply(self._init_weights)
"""

import os
import time
import random
import math
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import mlflow

# User must have installed schedulefree and JohnYKiyo NeuralNDCG implementation
# pip install schedulefree
# pip install git+https://github.com/JohnYKiyo/ListwiseRankingLoss-NeuralNDCG.git  # or local path
from schedulefree import RAdamScheduleFree
from neuralndcgloss.loss import NeuralNDCGLoss

# ----------------------------
# Config
# ----------------------------
config = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 20,
    "batch_Q": 32,
    "D": 100,
    "hidden_dim": 128,
    "lambda_listwise": 1.0,
    "lambda_pointwise": 1.0,
    "grad_accum_steps": 4,
    "grad_clip": 1.0,
    "mlflow_experiment": "ndcg_radam_schedulefree",
    "temperature": 0.1,
    "min_k_for_listwise": 2,
    "drop_last_train": True,
    "drop_longer_than": 400,
    "max_batches_per_epoch": None,
    "checkpoint_dir": "./checkpoints",
    # AMP switch:
    "use_amp": True,
    # optimizer lr for RAdamScheduleFree:
    "optimizer_lr": 1e-4,
}

os.makedirs(config["checkpoint_dir"], exist_ok=True)

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic may slow down; set as desired
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config["seed"])

# ----------------------------
# Model
# ----------------------------
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
        # use apply() style as requested
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # x: (B, K, D)
        B, K, D = x.shape
        flat = x.view(B * K, D)
        out = self.net(flat).view(B, K)
        return out

# ----------------------------
# Utilities: buckets, batching, metrics
# ----------------------------
def build_buckets(X_list: List[np.ndarray], drop_longer_than: int = None):
    buckets = defaultdict(list)
    for i, x in enumerate(X_list):
        k = x.shape[0]
        if drop_longer_than and k > drop_longer_than:
            continue
        buckets[k].append(i)
    return buckets

def make_epoch_batches(buckets, batch_Q, shuffle=True, drop_last=True):
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
            batches.append((k, idxs[b * batch_Q:(b + 1) * batch_Q]))
        if not drop_last:
            rem = n % batch_Q
            if rem > 0:
                batches.append((k, idxs[num_full * batch_Q:]))
    if shuffle:
        random.shuffle(batches)
    return batches

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

# ----------------------------
# Checkpoint helpers
# ----------------------------
def save_checkpoint(state: dict, path: str):
    torch.save(state, path)

def load_checkpoint(path: str, model: nn.Module, optimizer=None, scaler: GradScaler = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt and ckpt["optim_state"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optim_state"])
        except Exception:
            print("Warning: optimizer state could not be fully restored (version mismatch?).")
    if scaler is not None and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except Exception:
            print("Warning: scaler state could not be fully restored.")
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    return epoch, global_step

# ----------------------------
# Training + mlflow
# ----------------------------
def train_and_eval_mlflow(X_list_train, y_list_train, X_list_val, y_list_val, config):
    device = config["device"]
    model = RankMLP(d_in=config["D"], hidden=config["hidden_dim"]).to(device)

    # Optimizer: RAdamScheduleFree
    optimizer = RAdamScheduleFree(model.parameters(), lr=config["optimizer_lr"])

    # AMP scaler with explicit enabled flag
    scaler = GradScaler(enabled=config["use_amp"])
    ndcg_loss = NeuralNDCGLoss(temperature=config["temperature"])
    pointwise_loss_fn = nn.MSELoss()

    # mlflow setup
    mlflow.set_experiment(config["mlflow_experiment"])
    with mlflow.start_run():
        # log scalar + config params
        mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))})

        # resume if checkpoint exists
        latest_ckpt = os.path.join(config["checkpoint_dir"], "latest.pt")
        global_step = 0
        start_epoch = 1
        if os.path.exists(latest_ckpt):
            start_epoch, global_step = load_checkpoint(latest_ckpt, model, optimizer, scaler)
            start_epoch += 1
            print(f"Resumed from checkpoint at epoch {start_epoch-1} global_step {global_step}")

        best_val_ndcg = -1.0

        # training loop
        for epoch in range(start_epoch, config["epochs"] + 1):
            t0 = time.time()
            buckets = build_buckets(X_list_train, drop_longer_than=config.get("drop_longer_than"))
            batches = make_epoch_batches(buckets, batch_Q=config["batch_Q"], shuffle=True, drop_last=config["drop_last_train"])
            model.train()
            # if optimizer has train/eval toggles (RAdamScheduleFree article mentions optimizer.train())
            if hasattr(optimizer, "train"):
                try:
                    optimizer.train()
                except Exception:
                    pass

            running_loss = 0.0
            batch_count = 0
            # iterate batches
            for batch_idx, (k, batch_idxs) in enumerate(batches):
                B = len(batch_idxs)
                K = k

                # build batch tensors
                X_batch = np.stack([X_list_train[i] for i in batch_idxs], axis=0)  # (B, K, D)
                y_batch = np.stack([y_list_train[i] for i in batch_idxs], axis=0)  # (B, K)
                X_t = torch.from_numpy(X_batch).to(device=device, dtype=torch.float32)
                y_t = torch.from_numpy(y_batch).to(device=device, dtype=torch.float32)

                # forward + loss under autocast(enabled=...)
                with autocast(enabled=config["use_amp"]):
                    scores = model(X_t)  # (B, K)
                    p_loss = pointwise_loss_fn(scores.view(-1), y_t.view(-1))
                    if K >= config["min_k_for_listwise"]:
                        l_loss = ndcg_loss(scores, y_t)
                    else:
                        l_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                    loss = config["lambda_pointwise"] * p_loss + config["lambda_listwise"] * l_loss

                # scale down for accumulation
                loss = loss / float(config["grad_accum_steps"])

                # backward with scaler
                scaler.scale(loss).backward()

                # step when accumulation satisfied
                if (batch_count + 1) % config["grad_accum_steps"] == 0:
                    # unscale, clip, step, update scaler
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

                    # scaler.step will call optimizer.step() internally
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # optional optimizer.eval()/train switching is handled outside eval
                    global_step += 1

                running_loss += loss.item() * float(config["grad_accum_steps"])  # scale back
                batch_count += 1

                # safety: detect NaN/Inf in loss
                if not math.isfinite(running_loss):
                    raise RuntimeError(f"Non-finite running_loss at epoch {epoch} batch {batch_idx}")

                # logging (periodic)
                if global_step % 100 == 0 and global_step > 0:
                    # compute batch-level metrics in eval mode (accurate BN/Dropout behavior)
                    model.eval()
                    with torch.no_grad():
                        scores_eval = model(X_t).detach().cpu().numpy()
                        y_eval = y_t.detach().cpu().numpy()
                        ndcg10 = ndcg_at_k_per_query_numpy(y_eval, scores_eval, k=min(10, K))
                        rmse = rmse_pointwise(y_eval, scores_eval)
                    model.train()
                    mlflow.log_metric("train/batch_ndcg10", ndcg10, step=global_step)
                    mlflow.log_metric("train/batch_point_rmse", rmse, step=global_step)
                    mlflow.log_metric("train/batch_loss", running_loss / max(1, batch_count), step=global_step)
                    print(f"[E{epoch}] step {global_step} lr=N/A loss={running_loss / batch_count:.6f} ndcg@10={ndcg10:.4f} rmse={rmse:.4f}")

                # optional debug cap
                if config.get("max_batches_per_epoch") and batch_count >= config["max_batches_per_epoch"]:
                    break

            # epoch end: validation over full val set
            epoch_time = time.time() - t0
            avg_loss = running_loss / max(1, batch_count)
            mlflow.log_metric("train/epoch_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch} finished in {epoch_time:.1f}s avg_loss={avg_loss:.6f}")

            # ensure optimizer in eval mode if it exposes such API
            if hasattr(optimizer, "eval"):
                try:
                    optimizer.eval()
                except Exception:
                    pass

            val_ndcg, val_rmse = evaluate_full_mlflow(model, X_list_val, y_list_val, device, config)
            mlflow.log_metric("val/ndcg", val_ndcg, step=epoch)
            mlflow.log_metric("val/point_rmse", val_rmse, step=epoch)
            print(f"Validation: ndcg={val_ndcg:.6f} rmse={val_rmse:.6f}")

            # checkpointing: save model, optimizer and scaler state
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
                "scaler_state": scaler.state_dict()
            }
            ckpt_path = os.path.join(config["checkpoint_dir"], f"ckpt_epoch_{epoch}.pt")
            latest_path = os.path.join(config["checkpoint_dir"], "latest.pt")
            save_checkpoint(ckpt, ckpt_path)
            save_checkpoint(ckpt, latest_path)
            mlflow.log_artifact(ckpt_path)

            # best model save
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_path = os.path.join(config["checkpoint_dir"], "best.pt")
                save_checkpoint(ckpt, best_path)
                mlflow.log_artifact(best_path)
                print(f"New best model saved: val_ndcg={best_val_ndcg:.6f}")

    # end mlflow run

# ----------------------------
# Validation helper
# ----------------------------
def evaluate_full_mlflow(model: nn.Module, X_list: List[np.ndarray], y_list: List[np.ndarray], device: str, config: dict):
    model.eval()
    ndcgs = []; rmses = []
    buckets = build_buckets(X_list, drop_longer_than=config.get("drop_longer_than"))
    batches = make_epoch_batches(buckets, batch_Q=config["batch_Q"], shuffle=False, drop_last=False)
    with torch.no_grad():
        for (k, batch_idxs) in batches:
            X_batch = np.stack([X_list[i] for i in batch_idxs], axis=0)
            y_batch = np.stack([y_list[i] for i in batch_idxs], axis=0)
            X_t = torch.from_numpy(X_batch).to(device=device, dtype=torch.float32)
            y_t = torch.from_numpy(y_batch).to(device=device, dtype=torch.float32)
            scores = model(X_t).detach().cpu().numpy()
            y_np = y_t.detach().cpu().numpy()
            ndcgs.append(ndcg_at_k_per_query_numpy(y_np, scores, k=min(10, k)))
            rmses.append(rmse_pointwise(y_np, scores))
    mean_ndcg = float(np.mean(ndcgs)) if len(ndcgs) > 0 else 0.0
    mean_rmse = float(np.mean(rmses)) if len(rmses) > 0 else 0.0
    return mean_ndcg, mean_rmse

# ----------------------------
# Minimal dataset loader (replace with polars->X_list,y_list in your env)
# ----------------------------
def load_dummy_data(num_q=3000, D=100, min_k=1, max_k=50):
    X_list, y_list = [], []
    for _ in range(num_q):
        k = random.randint(min_k, max_k)
        X_list.append(np.random.randn(k, D).astype(np.float32))
        y_list.append(np.random.randint(0, 5, size=(k,)).astype(np.float32))
    return X_list, y_list

# ----------------------------
# __main__ driver
# ----------------------------
if __name__ == "__main__":
    # Example: replace this with your own parquet -> X_list,y_list loader (polars)
    # e.g. data = load_parquet_to_torch(...); X_list = [x.numpy() for x in data["X_list"]]; etc.
    print("Loading data...")
    X_list_train, y_list_train = load_dummy_data(num_q=3000, D=config["D"], min_k=1, max_k=50)
    X_list_val, y_list_val = load_dummy_data(num_q=600, D=config["D"], min_k=1, max_k=50)
    print("Starting training (mlflow experiment: {})".format(config["mlflow_experiment"]))
    train_and_eval_mlflow(X_list_train, y_list_train, X_list_val, y_list_val, config)
```

---

### 補足（実務注意）

* `grad_accum_steps`：バッチ内で `loss /= grad_accum_steps` → `scaler.scale(loss).backward()` → 実際の `scaler.step(optimizer)` は accumulation 毎に行う、という正しい流れを採用しました。
* `autocast(enabled=...)` と `GradScaler(enabled=...)`：AMP を `config["use_amp"]` で完全制御できます。CPU 実行やデバッグ時は `use_amp=False` にしてください。
* `optimizer_lr`：RAdamScheduleFree はスケジューラ不要だが学習率は依然チューニング対象です（config の値を実データに合わせて調整してください）。
* `optimizer.train()/eval()` と `optimizer.state_dict()` の互換：RAdamScheduleFree の実装によるため `train()/eval()` が存在すれば呼び出します。`state_dict()` が無ければ checkpoint の保存に `None` が入ります（上のコードは安全なtry/except を入れていませんが load 時に警告処理をしています）。
* 実データに差し替える際、`load_parquet_to_torch`（polars）を作って `X_list = [arr.numpy() if torch else arr for ...]` のように渡してください。`X_list` 要素は `np.ndarray` (K_i, D) である必要があります。

---

必要なら次を出します（今すぐ作成できます）：

1. あなたの実 parquet のカラム名を使って `polars -> X_list,y_list` に変換する関数（そのままこのスクリプトに貼れる形）。
2. `grad_accum_steps` を検証するための小型ベンチマーク（ステップ時間とメモリ比較）スクリプト。
3. RAdamScheduleFree の挙動（optimizer.train()/eval() の期待動作）確認用の簡単テストケース。

どれを作りましょうか？
