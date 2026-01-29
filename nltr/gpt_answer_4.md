いい指定ありがとうございます — バケッティング（同一スレート長ごとにグループ化）を**PyTorch の Dataset / BatchSampler / DataLoader**できちんと実装した完全版を出します。

ポイントまとめ（この実装で満たす内容）

* parquet を **polars** で読み込み、行（query×item）を **qid ごとに集約**して内部で `X_list: List[np.ndarray]` / `y_list: List[np.ndarray]` として保持する `QueryItemDataset` を実装。
* 同一スレート長（K）のクエリ群からランダムに `batch_size` 個のクエリを選ぶ **BucketBatchSampler** を実装。

  * 各バッチ内は同一 K であることを厳密に保証。
  * エポックごとにバケッティング内の順序がランダムに変わる（`set_epoch(epoch)` を呼ぶことで再現可能）。
  * バッチサイズに満たない余り（残り）からのバッチ作成を**許可**する設定（`allow_incomplete=True`）。
* `collate_fn` はバッチ（同一 K）を `(X: torch.FloatTensor (B,K,D), y: torch.FloatTensor (B,K), qids: List)` に組み上げる。
* サンプルの使い方（`DataLoader` の生成とエポックループ）を示す。

以下がそのコードです。必要なら `load_parquet_to_torch` 部分をあなたのカラムに合わせて微調整可能です（デフォルトは `qid_col="qid"`, `label_col="label"`, `sort_col=None`）。

```python
"""
bucketed_dataloader.py

- QueryItemDataset: polars -> per-query groups (X_list, y_list, qids, item_ids)
- BucketBatchSampler: yields batches of query-indices where each batch contains queries with the same slate size K
- collate_batch: stacks a batch into (B, K, D) tensors
- Usage example at the bottom
"""

from typing import List, Sequence, Optional, Tuple, Iterator, Dict, Any
import os
import random
import math

import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# ------------------------------
# Dataset: group rows by qid -> per-query arrays
# ------------------------------
class QueryItemDataset(Dataset):
    """
    Load parquet and group by qid. Each item is a single query (all its items).
    __getitem__(i) -> dict with keys:
      - 'qid': original qid
      - 'X': np.ndarray shape (K_i, D)  (float32)
      - 'y': np.ndarray shape (K_i,)    (float32)
      - 'item_ids': list of item ids (optional)
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
        """
        Args:
            parquet_path: path to parquet file
            qid_col, item_id_col, label_col: column names
            feature_cols: if None, all columns except qid_col,label_col,sort_col,item_id_col are used
            sort_col: if provided, group rows will be sorted by this column to preserve initial rank
            drop_qids_with_k_lt: if provided, remove queries with K < this value
        """
        self.parquet_path = parquet_path
        self.qid_col = qid_col
        self.item_id_col = item_id_col
        self.label_col = label_col
        self.sort_col = sort_col

        # read parquet
        df = pl.read_parquet(parquet_path)

        # determine feature columns
        reserved = {qid_col, label_col}
        if sort_col:
            reserved.add(sort_col)
        if item_id_col:
            reserved.add(item_id_col)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in reserved]
        else:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"feature_cols contains missing columns: {missing}")
        self.feature_cols = feature_cols

        # optional sort
        sort_keys = [qid_col]
        if sort_col:
            sort_keys.append(sort_col)
        df = df.sort(sort_keys)

        # group and aggregate (polars list aggregation)
        agg_cols = [pl.col(c).list().alias(c) for c in feature_cols]
        agg_cols.append(pl.col(label_col).list().alias(label_col))
        if item_id_col:
            agg_cols.append(pl.col(item_id_col).list().alias(item_id_col))
        agg_cols.append(pl.count().alias("K"))
        grouped = df.groupby(qid_col).agg(agg_cols)

        # optional filtering by min K
        if drop_qids_with_k_lt is not None:
            grouped = grouped.filter(pl.col("K") >= drop_qids_with_k_lt)

        # finalize lists
        self.qids = grouped[qid_col].to_list()
        self.Ks = grouped["K"].to_list()
        Q = len(self.qids)
        self.X_list: List[np.ndarray] = []
        self.y_list: List[np.ndarray] = []
        self.item_ids_list: List[Optional[List[Any]]] = []

        for i in range(Q):
            k = int(self.Ks[i])
            # features: each feature column is a list of length k
            cols_arrays = []
            for c in feature_cols:
                col_list = grouped[c][i]  # python list
                arr = np.asarray(col_list, dtype=np.float32).reshape(k, 1)
                cols_arrays.append(arr)
            feat_mat = np.concatenate(cols_arrays, axis=1)  # (k, D)
            labels = np.asarray(grouped[self.label_col][i], dtype=np.float32)
            self.X_list.append(feat_mat)
            self.y_list.append(labels)
            if item_id_col:
                self.item_ids_list.append(grouped[item_id_col][i])
            else:
                self.item_ids_list.append([None] * k)

    def __len__(self) -> int:
        return len(self.qids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "qid": self.qids[idx],
            "X": self.X_list[idx],       # np.ndarray (K,D) float32
            "y": self.y_list[idx],       # np.ndarray (K,) float32
            "item_ids": self.item_ids_list[idx],  # list
            "K": self.X_list[idx].shape[0],
        }

# ------------------------------
# Sampler: bucketed batch sampler
# ------------------------------
class BucketBatchSampler(Sampler[List[int]]):
    """
    Yields lists of dataset indices. Each yielded batch contains queries
    that have identical slate size K.

    Args:
        dataset: QueryItemDataset
        batch_size: number of queries per batch
        shuffle: shuffle within each bucket each epoch
        allow_incomplete: if True, allow final leftover batch smaller than batch_size
        seed: base seed for reproducibility
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
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.allow_incomplete = allow_incomplete
        self.seed = seed if seed is not None else 0
        # build bucket mapping: K -> list of indices
        self._build_buckets()

        # internal epoch (for set_epoch)
        self.epoch = 0

    def _build_buckets(self):
        self.buckets = {}
        for idx, k in enumerate(self.dataset.Ks):
            k = int(k)
            self.buckets.setdefault(k, []).append(idx)
        # keep list(sorted bucket sizes) for stable order if needed
        self.unique_ks = sorted(self.buckets.keys())

    def set_epoch(self, epoch: int):
        """Set epoch number to deterministically reseed shuffling"""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        # For each bucket, create shuffled order of indices
        bucket_idxs_copy = {k: idxs[:] for k, idxs in self.buckets.items()}
        for k, idxs in bucket_idxs_copy.items():
            if self.shuffle:
                rng.shuffle(idxs)

        # produce batches: iterate all buckets and produce their batches,
        # but to make cross-bucket mixing we can collect all batches and shuffle them at the end
        all_batches: List[List[int]] = []
        for k, idxs in bucket_idxs_copy.items():
            n = len(idxs)
            i = 0
            while i < n:
                j = i + self.batch_size
                batch = idxs[i:j]
                if len(batch) < self.batch_size and not self.allow_incomplete:
                    break
                all_batches.append(batch)
                i = j
        # shuffle across batches to mix Ks each epoch but keep each batch internal K-homogenous
        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        # number of batches produced per epoch
        total = 0
        for idxs in self.buckets.values():
            n = len(idxs)
            if self.allow_incomplete:
                total += math.ceil(n / self.batch_size)
            else:
                total += n // self.batch_size
        return total

# ------------------------------
# collate_fn
# ------------------------------
def collate_bucket_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    batch: list of dataset.__getitem__ outputs; all elements must have same K
    returns dict with:
      - 'qids': list
      - 'X': tensor float32 (B, K, D)
      - 'y': tensor float32 (B, K)
      - 'item_ids': list of lists
      - 'K': K
    """
    B = len(batch)
    if B == 0:
        raise ValueError("Empty batch passed to collate_bucket_batch")
    K = batch[0]["K"]
    D = batch[0]["X"].shape[1]
    # sanity: check same K
    for item in batch:
        if item["K"] != K:
            raise ValueError("collate_bucket_batch received mixed K in batch (must be equal)")

    X_np = np.stack([item["X"] for item in batch], axis=0)  # (B,K,D)
    y_np = np.stack([item["y"] for item in batch], axis=0)  # (B,K)
    qids = [item["qid"] for item in batch]
    item_ids = [item["item_ids"] for item in batch]

    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).float()
    return {"qids": qids, "item_ids": item_ids, "X": X, "y": y, "K": K}

# ------------------------------
# Example usage
# ------------------------------
def example_usage(parquet_path: str):
    # create dataset
    ds = QueryItemDataset(
        parquet_path=parquet_path,
        qid_col="qid",
        item_id_col="item_id",
        label_col="label",
        feature_cols=None,   # auto-detect
        sort_col=None,       # if you have initial_rank column use that
        drop_qids_with_k_lt=None
    )

    batch_Q = 16  # number of queries per batch
    sampler = BucketBatchSampler(ds, batch_size=batch_Q, shuffle=True, allow_incomplete=True, seed=123)

    # DataLoader: when using batch_sampler, set batch_size=None
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=collate_bucket_batch,
        num_workers=4,   # tune for your env
        pin_memory=True
    )

    # epoch loop: set epoch on sampler for reproducibility
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for batch in loader:
            # batch is dict with tensors (B,K,D) and (B,K)
            X = batch["X"]       # (B,K,D)
            y = batch["y"]       # (B,K)
            qids = batch["qids"]
            K = batch["K"]
            # send to device, feed into model, etc.
            # e.g., X = X.to(device); y = y.to(device)
            print(f"epoch {epoch} batch K={K} X.shape={X.shape} y.shape={y.shape} q0={qids[0]}")
        print("epoch done")

# ------------------------------
# If run as script, run example (requires a parquet file)
# ------------------------------
if __name__ == "__main__":
    # replace with your parquet path; we provide a check to avoid accidental run
    PARQUET_EXAMPLE = "data/example.parquet"
    if os.path.exists(PARQUET_EXAMPLE):
        example_usage(PARQUET_EXAMPLE)
    else:
        print(f"No parquet at {PARQUET_EXAMPLE}. Replace with your path or import this module in training code.")
```

---

## 実装上の補足・注意点（短め）

* polars の `groupby().agg(list)` は高速ですが、メモリ使用は `Q × K × D` に比例します。大規模データはチャンク処理や on-disk indexing（parquet scan）を検討してください。
* `BatchSampler.__len__()` はエポックあたりの**概算バッチ数**を返します（`allow_incomplete=True` の場合、切り上げで計算）。DataLoader によっては `len(loader)` が利用できます。
* `num_workers` を >0 にすると DataLoader は worker プロセスで `Dataset.__getitem__` を呼びますが、今回は Dataset の内部データは numpy arrays in memory なので worker 側でのメモリ複製が起きます。必要なら `pin_memory` と `persistent_workers` を調整してください。
* `collate_bucket_batch` はバッチ内 K が同じ前提です。もし `allow_incomplete=False` で drop_last の設定が有ればこの前提は常に満たされます。今回実装では `BucketBatchSampler` が同一 K を保証します。
* エポックごとの「同一 K 群からのランダム選択」は `BucketBatchSampler` の `set_epoch` & internal shuffle により実現しています（seed+epoch により再現可能）。
* 「バッチサイズに満たない余りのデータからもバッチを作成してよい」という要件は `allow_incomplete=True` でかなえています。もし代わりに `drop_last=True` を望む場合は sampler を変更してください。

---

これで「Dataset/DataLoader/BatchSampler を駆使して、同一スレート長のみを含むランダムバッチを毎エポック作る」要件は満たされます。
次にやると良いこと（提案）：

1. polars の `sort_col`（例えば `initial_rank`）でグループ内ソートする — これを有効にすればモデルは初期順位を尊重した学習が可能。
2. 大規模データなら `X_list` を memory-mapped npy に吐く・または lazy ロード式の Dataset にする（現在は全ロード）。
3. `BucketBatchSampler` を少し拡張して「バケットの重みづけサンプリング（各エポックで K の頻度を調整）」も可能です。

必要なら、あなたの parquet のカラム一覧をいただければ `feature_cols` を自動検出せず、コードに直接固定して動くバージョンを作ります。どうしますか？
