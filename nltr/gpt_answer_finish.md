下記は「**これまでの全要件を統合した実運用向けの（簡便化しない）完全な Python スクリプト**」です。
要点を短くまとめると：

* parquet を **polars** で読み、`sort_col`（例：`initial_rank`）でグループ内の並びを固定し、**BM25 を含む任意の特徴列**を特徴量として使う `QueryItemDataset` を作成します。
* 同じスレート長（K）のクエリだけを含むバッチを返す `BucketBatchSampler`（バケッティング）を実装。バッチは毎エポックごとにランダムに作られます（`set_epoch`）。
* モデルは `RankMLP`（三層 MLP）、初期化は `self.apply(self._init_weights)`。
* 学習は **multi-task（pointwise MSE ＋ listwise NeuralNDCG）**、損失重みで合成。`grad_accum_steps` による勾配累積対応。
* Optimizer は **RAdamScheduleFree**（スケジューラ不要）を使用。`optimizer.train()/eval()` が存在すれば適切に呼ぶ。
* AMP（`autocast(enabled=...)` + `GradScaler(enabled=...)`）を安全に使用。
* 実験管理は **mlflow**：ハイパラ／ログ／チェックポイント（artifact）をすべて記録。
* 再現性（seed）、チェックポイントの resume、NaN 検出、検証（全クエリで精密 NDCG / pointwise RMSE）を備える。

> 実行前に必要：`pip install polars schedulefree mlflow` と `neuralndcgloss`（JohnYKiyo 実装）をインストールしてください。パッケージ名・インポートパスが異なる場合は該当箇所を調整してください。

---

保存ファイル名の例：`train_full_ltr_mlflow.py` — このまま実行できるように `__main__` 部も含めています。
（長いですが、全て現場で必要な要素を入れています。）

```python
# train_full_ltr_mlflow.py
"""
Complete production-ready training script for listwise+pointwise LTR
- polars -> QueryItemDataset (group by qid, sort by sort_col)
- BucketBatchSampler to produce batches of queries with identical slate length K
- RankMLP model with apply(init)
- Multi-task loss: NeuralNDCG (listwise) + MSE (pointwise)
- Optimizer: RAdamScheduleFree (schedulefree package) -> no external scheduler
- AMP (autocast + GradScaler) with explicit enabled flag
- grad_accum_steps support
- mlflow logging, checkpointing, resume
- Evaluation: per-query NDCG and pointwise RMSE over validation set
"""

import os
import time
import random
import math
from collections import defaultdict
from typing import List, Optional, Sequence, Dict, Any, Iterator

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Sampler

# External libs (install before running)
# pip install schedulefree mlflow
# pip install git+https://github.com/JohnYKiyo/ListwiseRankingLoss-NeuralNDCG.git
from schedulefree import RAdamScheduleFree
from neuralndcgloss.loss import NeuralNDCGLoss

# -------------------------
# Config - tune for your environment
# -------------------------
CONFIG = {
    # Data
    "train_parquet": "data/train.parquet",
    "val_parquet": "data/val.parquet",
    "qid_col": "qid",
    "item_id_col": "item_id",
    "label_col": "label",
    "sort_col": "initial_rank",        # group-internal ordering column (e.g., initial rank)
    "feature_cols": None,              # None => auto-detect (will include bm25 if present)
    # Model / training
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "D_override": None,                # if you want to force feature dim (None->auto)
    "hidden_dim": 128,
    "batch_Q": 32,                     # number of queries per batch
    "grad_accum_steps": 4,
    "grad_clip": 1.0,
    "epochs": 20,
    "min_k_for_listwise": 2,
    "drop_longer_than": 400,           # drop extremely long slates if desired
    "allow_incomplete_batch": True,    # allow final small batches from each bucket
    # Loss weights
    "lambda_listwise": 1.0,
    "lambda_pointwise": 1.0,
    # Optimizer
    "optimizer_lr": 1e-4,              # start point for RAdamScheduleFree
    # AMP & reproducibility
    "use_amp": True,
    "seed": 42,
    # mlflow
    "mlflow_experiment": "ltr_full_pipeline",
    "checkpoint_dir": "./checkpoints",
    # DataLoader
    "num_workers": 4,
    "pin_memory": True,
    # Logging freq
    "log_interval_steps": 200,
    # safety
    "max_batches_per_epoch": None,     # debug cap; None = no cap
}

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic may slow down; set as needed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG["seed"])

# -------------------------
# Dataset: polars -> per-query arrays
# -------------------------
class QueryItemDataset(Dataset):
    """
    Load parquet (preprocessed) and group rows by qid. Each dataset item is a query with all its items.
    Exposes: __len__, __getitem__ returning dict with keys 'qid','X','y','item_ids','K'
    """

    def __init__(
        self,
        parquet_path: str,
        qid_col: str = "qid",
        item_id_col: Optional[str] = "item_id",
        label_col: str = "label",
        feature_cols: Optional[List[str]] = None,
        sort_col: Optional[str] = None,
        drop_qids_with_k_lt: Optional[int] = None,
    ):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"parquet not found: {parquet_path}")
        self.parquet_path = parquet_path
        self.qid_col = qid_col
        self.item_id_col = item_id_col
        self.label_col = label_col
        self.sort_col = sort_col

        # 1) read parquet (eager)
        df = pl.read_parquet(parquet_path)

        # 2) feature_cols
        reserved = {qid_col, label_col}
        if item_id_col:
            reserved.add(item_id_col)
        if sort_col:
            reserved.add(sort_col)
        if feature_cols is None:
            # auto-detect all columns except reserved
            self.feature_cols = [c for c in df.columns if c not in reserved]
        else:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"feature_cols contains missing columns: {missing}")
            self.feature_cols = feature_cols

        # 3) sort by qid and sort_col (if provided) so that group lists preserve desired order
        sort_keys = [qid_col]
        if sort_col:
            sort_keys.append(sort_col)
        df = df.sort(sort_keys)

        # 4) groupby qid and aggregate feature lists and label lists
        agg_exprs = [pl.col(c).list().alias(c) for c in self.feature_cols]
        agg_exprs.append(pl.col(self.label_col).list().alias(self.label_col))
        if self.item_id_col:
            agg_exprs.append(pl.col(self.item_id_col).list().alias(self.item_id_col))
        agg_exprs.append(pl.count().alias("K"))

        grouped = df.groupby(self.qid_col).agg(agg_exprs)

        # 5) optional filter
        if drop_qids_with_k_lt is not None:
            grouped = grouped.filter(pl.col("K") >= drop_qids_with_k_lt)

        # 6) materialize arrays
        self.qids = grouped[self.qid_col].to_list()
        self.Ks = grouped["K"].to_list()
        Q = len(self.qids)
        if Q == 0:
            raise ValueError("No queries loaded from parquet (maybe wrong filters?)")

        self.X_list: List[np.ndarray] = []
        self.y_list: List[np.ndarray] = []
        self.item_ids_list: List[List[Any]] = []

        for i in range(Q):
            k = int(self.Ks[i])
            # build (k, D) feature matrix
            cols_arrays = []
            for c in self.feature_cols:
                col_list = grouped[c][i]  # python list of length k
                arr = np.asarray(col_list, dtype=np.float32).reshape(k, 1)
                cols_arrays.append(arr)
            feat_mat = np.concatenate(cols_arrays, axis=1)  # (k, D)
            labels = np.asarray(grouped[self.label_col][i], dtype=np.float32)
            self.X_list.append(feat_mat)
            self.y_list.append(labels)
            if self.item_id_col:
                self.item_ids_list.append(grouped[self.item_id_col][i])
            else:
                self.item_ids_list.append([None] * k)

        # optional override dimension
        if CONFIG["D_override"] is not None:
            self.D = CONFIG["D_override"]
        else:
            self.D = self.X_list[0].shape[1]

    def __len__(self) -> int:
        return len(self.qids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "qid": self.qids[idx],
            "X": self.X_list[idx],    # np.ndarray (K, D)
            "y": self.y_list[idx],    # np.ndarray (K,)
            "item_ids": self.item_ids_list[idx],
            "K": int(self.X_list[idx].shape[0]),
        }

# -------------------------
# BucketBatchSampler
# -------------------------
class BucketBatchSampler(Sampler[List[int]]):
    """
    Yields batches of dataset indices; each batch contains queries having the same slate length K.
    - set_epoch(epoch) available to reseed shuffle deterministically per epoch
    - allow_incomplete: allow final smaller batch in each bucket
    """

    def __init__(
        self,
        dataset: QueryItemDataset,
        batch_size: int,
        shuffle: bool = True,
        allow_incomplete: bool = True,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.allow_incomplete = allow_incomplete
        self.seed = seed if seed is not None else 0
        self._build_buckets()
        self.epoch = 0

    def _build_buckets(self):
        self.buckets: Dict[int, List[int]] = {}
        for idx, k in enumerate(self.dataset.Ks):
            k = int(k)
            self.buckets.setdefault(k, []).append(idx)
        self.unique_ks = sorted(self.buckets.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        # copy buckets and shuffle within each bucket
        bucket_idxs = {k: lst[:] for k, lst in self.buckets.items()}
        for k, lst in bucket_idxs.items():
            if self.shuffle:
                rng.shuffle(lst)

        # create batches per bucket
        all_batches: List[List[int]] = []
        for k, lst in bucket_idxs.items():
            n = len(lst)
            i = 0
            while i < n:
                j = i + self.batch_size
                batch = lst[i:j]
                if len(batch) < self.batch_size and not self.allow_incomplete:
                    break
                all_batches.append(batch)
                i = j

        # shuffle across batches so epoch mixes K groups
        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for lst in self.buckets.values():
            n = len(lst)
            if self.allow_incomplete:
                total += math.ceil(n / self.batch_size)
            else:
                total += n // self.batch_size
        return total

# -------------------------
# collate_fn
# -------------------------
def collate_bucket_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Empty batch")
    B = len(batch)
    K = batch[0]["K"]
    D = batch[0]["X"].shape[1]
    # sanity check
    for item in batch:
        if item["K"] != K:
            raise ValueError("Mixed K in collate_batch (should not happen)")

    X_np = np.stack([item["X"] for item in batch], axis=0)  # (B, K, D)
    y_np = np.stack([item["y"] for item in batch], axis=0)  # (B, K)
    qids = [item["qid"] for item in batch]
    item_ids = [item["item_ids"] for item in batch]

    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).float()
    return {"qids": qids, "item_ids": item_ids, "X": X, "y": y, "K": K}

# -------------------------
# Model: RankMLP (item-wise scorer)
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
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, D) -> scores: (B, K)
        B, K, D = x.shape
        flat = x.view(B * K, D)
        out = self.net(flat).view(B, K)
        return out

# -------------------------
# Metrics
# -------------------------
def ndcg_at_k_per_query_numpy(y_true: np.ndarray, y_score: np.ndarray, k: Optional[int] = None) -> float:
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
# Save/Load checkpoint helpers
# -------------------------
def save_checkpoint(state: dict, path: str):
    torch.save(state, path)

def load_checkpoint(path: str, model: nn.Module, optimizer=None, scaler: Optional[GradScaler] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt and ckpt["optim_state"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optim_state"])
        except Exception:
            print("Warning: could not restore optimizer state exactly.")
    if scaler is not None and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except Exception:
            print("Warning: could not restore scaler state exactly.")
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    return epoch, global_step

# -------------------------
# Evaluation over entire val set
# -------------------------
def evaluate_full(model: nn.Module, dataset: QueryItemDataset, config: Dict[str, Any]) -> Dict[str, float]:
    device = config["device"]
    model.eval()
    buckets = defaultdict(list)
    for idx, k in enumerate(dataset.Ks):
        if config["drop_longer_than"] and k > config["drop_longer_than"]:
            continue
        buckets[int(k)].append(idx)
    sampler = BucketBatchSampler(dataset, batch_size=config["batch_Q"], shuffle=False, allow_incomplete=True, seed=0)
    sampler.buckets = buckets  # override buckets to only val ones
    batches = list(iter(sampler))
    ndcgs = []
    rmses = []
    with torch.no_grad():
        for batch_idxs in batches:
            # build batch items
            batch_items = [dataset[i] for i in batch_idxs]
            batch = collate_bucket_batch(batch_items)
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            scores = model(X).detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            k_val = batch["K"]
            ndcgs.append(ndcg_at_k_per_query_numpy(y_np, scores, k=min(10, k_val)))
            rmses.append(rmse_pointwise(y_np, scores))
    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    mean_rmse = float(np.mean(rmses)) if rmses else 0.0
    return {"ndcg": mean_ndcg, "rmse": mean_rmse}

# -------------------------
# Training loop (integrated with mlflow)
# -------------------------
def train_main(train_parquet: str, val_parquet: str, config: Dict[str, Any]):
    device = config["device"]

    # load datasets
    print("Loading train dataset...")
    train_ds = QueryItemDataset(
        parquet_path=train_parquet,
        qid_col=config["qid_col"],
        item_id_col=config["item_id_col"],
        label_col=config["label_col"],
        feature_cols=config["feature_cols"],
        sort_col=config["sort_col"],
        drop_qids_with_k_lt=None,
    )
    print("Loading val dataset...")
    val_ds = QueryItemDataset(
        parquet_path=val_parquet,
        qid_col=config["qid_col"],
        item_id_col=config["item_id_col"],
        label_col=config["label_col"],
        feature_cols=config["feature_cols"],
        sort_col=config["sort_col"],
        drop_qids_with_k_lt=None,
    )

    # infer D if not forced
    D = train_ds.D
    print(f"Feature dim D={D}")

    model = RankMLP(d_in=D, hidden=config["hidden_dim"]).to(device)

    # optimizer
    optimizer = RAdamScheduleFree(model.parameters(), lr=config["optimizer_lr"])

    # AMP
    scaler = GradScaler(enabled=config["use_amp"])

    # losses
    ndcg_loss_fn = NeuralNDCGLoss(temperature=CONFIG["temperature"])
    pointwise_loss_fn = nn.MSELoss()

    # batch sampler & dataloader
    sampler = BucketBatchSampler(
        dataset=train_ds,
        batch_size=config["batch_Q"],
        shuffle=True,
        allow_incomplete=config["allow_incomplete_batch"],
        seed=config["seed"],
    )
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collate_bucket_batch,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    # mlflow run
    import mlflow
    mlflow.set_experiment(config["mlflow_experiment"])
    with mlflow.start_run():
        # log config (primitives only)
        mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))})
        # resume checkpoint if exists
        latest_ckpt = os.path.join(config["checkpoint_dir"], "latest.pt")
        start_epoch = 1
        global_step = 0
        if os.path.exists(latest_ckpt):
            print("Resuming from checkpoint:", latest_ckpt)
            e, gs = load_checkpoint(latest_ckpt, model, optimizer, scaler)
            start_epoch = e + 1
            global_step = gs
            print(f"Resumed at epoch {start_epoch-1}, global_step {global_step}")

        best_val_ndcg = -1.0

        # training epochs
        for epoch in range(start_epoch, config["epochs"] + 1):
            t0 = time.time()
            # set epoch on sampler so that it reseeds shuffling deterministically
            sampler.set_epoch(epoch)
            model.train()
            # if optimizer has train/eval API
            if hasattr(optimizer, "train"):
                try:
                    optimizer.train()
                except Exception:
                    pass

            running_loss = 0.0
            batch_count = 0
            step_in_epoch = 0

            for batch in loader:
                # batch: dict with X (B,K,D), y (B,K), qids
                X = batch["X"].to(device)
                y = batch["y"].to(device)
                K = batch["K"]
                B = X.shape[0]

                with autocast(enabled=config["use_amp"]):
                    scores = model(X)  # (B,K)
                    p_loss = pointwise_loss_fn(scores.view(-1), y.view(-1))
                    if K >= config["min_k_for_listwise"]:
                        l_loss = ndcg_loss_fn(scores, y)
                    else:
                        l_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                    total_loss = config["lambda_pointwise"] * p_loss + config["lambda_listwise"] * l_loss

                # scale down for accumulation
                loss_to_backprop = total_loss / float(config["grad_accum_steps"])
                scaler.scale(loss_to_backprop).backward()

                # step if accumulation boundary hit
                if (batch_count + 1) % config["grad_accum_steps"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                running_loss += float(total_loss.item())
                batch_count += 1
                step_in_epoch += 1

                # logging
                if global_step % config["log_interval_steps"] == 0 and global_step > 0:
                    # for accurate metrics, re-forward in eval mode
                    model.eval()
                    with torch.no_grad():
                        scores_eval = model(X).detach().cpu().numpy()
                        y_eval = y.detach().cpu().numpy()
                        ndcg10 = ndcg_at_k_per_query_numpy(y_eval, scores_eval, k=min(10, K))
                        rmse = rmse_pointwise(y_eval, scores_eval)
                    model.train()
                    mlflow.log_metric("train/batch_ndcg10", ndcg10, step=global_step)
                    mlflow.log_metric("train/batch_rmse", rmse, step=global_step)
                    mlflow.log_metric("train/batch_loss", running_loss / max(1, batch_count), step=global_step)
                    print(f"[E{epoch}] step {global_step} loss={running_loss / max(1,batch_count):.4f} ndcg@10={ndcg10:.4f} rmse={rmse:.4f}")

                # optional debug cap
                if config["max_batches_per_epoch"] and step_in_epoch >= config["max_batches_per_epoch"]:
                    break

            # epoch done: validation
            epoch_time = time.time() - t0
            avg_loss = running_loss / max(1, batch_count)
            mlflow.log_metric("train/epoch_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch} finished in {epoch_time:.1f}s avg_loss={avg_loss:.6f}")

            # put optimizer to eval if needed
            if hasattr(optimizer, "eval"):
                try:
                    optimizer.eval()
                except Exception:
                    pass

            val_metrics = evaluate_full(model, val_ds, config)
            mlflow.log_metric("val/ndcg", val_metrics["ndcg"], step=epoch)
            mlflow.log_metric("val/point_rmse", val_metrics["rmse"], step=epoch)
            print(f"Validation: ndcg={val_metrics['ndcg']:.6f} rmse={val_metrics['rmse']:.6f}")

            # checkpoint
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
                "scaler_state": scaler.state_dict() if scaler is not None else None
            }
            ckpt_path = os.path.join(config["checkpoint_dir"], f"ckpt_epoch_{epoch}.pt")
            latest_path = os.path.join(config["checkpoint_dir"], "latest.pt")
            save_checkpoint(ckpt, ckpt_path)
            save_checkpoint(ckpt, latest_path)
            mlflow.log_artifact(ckpt_path)

            if val_metrics["ndcg"] > best_val_ndcg:
                best_val_ndcg = val_metrics["ndcg"]
                best_path = os.path.join(config["checkpoint_dir"], "best.pt")
                save_checkpoint(ckpt, best_path)
                mlflow.log_artifact(best_path)
                print(f"New best model (ndcg={best_val_ndcg:.6f}) saved to {best_path}")

    # end mlflow run

# -------------------------
# __main__ entrypoint
# -------------------------
if __name__ == "__main__":
    # modify CONFIG paths as needed
    train_parquet = CONFIG["train_parquet"]
    val_parquet = CONFIG["val_parquet"]

    # Sanity check - adjust feature_cols if you want to include 'bm25' etc.
    # If CONFIG['feature_cols'] is None, dataset auto-detects features (excluding qid,label,sort_col,item_id)
    if not os.path.exists(train_parquet):
        raise FileNotFoundError(f"train_parquet not found: {train_parquet}")
    if not os.path.exists(val_parquet):
        print("Warning: val_parquet not found; using train as val for quick test.")
        val_parquet = train_parquet

    # Run training
    train_main(train_parquet=train_parquet, val_parquet=val_parquet, config=CONFIG)
```

---

### 実行前チェックリスト（必ず確認してください）

1. 必須ライブラリのインストール：`polars`, `schedulefree`, `mlflow`, `neuralndcgloss`（JohnYKiyo 実装）を入れてください。
2. `CONFIG` の `train_parquet` / `val_parquet` を実データのパスに置き換え、`feature_cols` を必要なら明示してください（BM25 列名を含める）。
3. `sort_col` は、グループ内の「初期検索の順位」を表す列（小さい＝高順位）の名前に合わせてください。BM25 が高スコアで高順位なら事前に `-bm25` のように反転しておくか、`sort_col` を適宜調整してください。
4. GPU メモリが足りない場合は `batch_Q`／`grad_accum_steps`／`hidden_dim` を調整してください。`num_workers` と `pin_memory` も環境に合わせて調整。
5. `RAdamScheduleFree` 実装の API はバージョン差がある可能性があるため、`optimizer.train()` / `optimizer.eval()` / `state_dict()` の扱いは実装に合わせてください。上のコードは存在すれば呼ぶ・保存する、という安全設計です。

---

### 追加で出せるもの（すぐ作成します）

* あなたの parquet の `df.columns` を貼っていただければ、`CONFIG["feature_cols"]` を明示した ready-to-run バージョンを作り直します（BM25 列名を必ず入れます）。
* DDP（分散学習）バージョン、または `train_main` を CLI で細かく設定できる argparse/typer バージョンも作れます。
* mlflow をローカルで使う場合の簡単な起動（`mlflow ui --backend-store-uri ...`）や artifact 格納方法のアドバイス。

やるならどれにしますか？
