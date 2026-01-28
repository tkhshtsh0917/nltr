# variable_slate_training.py
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from neuralndcgloss.loss import NeuralNDCGLoss  # adjust import if needed

# --------------------
# Hyperparams
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_Q = 32               # number of queries per batch (within a bucket)
D = 100                    # feature dim per item
hidden_dim = 128
lr = 1e-3
epochs = 10
use_mixed_precision = True
gradient_accumulation_steps = 1
clip_norm = 1.0

# --------------------
# Example: prepare X_list, y_list
# --------------------
# You should prepare these from parquet grouping: 
# X_list[i]: numpy array shape (K_i, D), y_list[i]: numpy array shape (K_i,)
# For demonstration, create a toy variable-length dataset:
def make_toy_data(num_queries=3000, D=100, min_k=1, max_k=178):
    X_list = []
    y_list = []
    for _ in range(num_queries):
        k = random.randint(min_k, min(20, max_k))  # use smaller Ks for the toy
        X_list.append(np.random.randn(k, D).astype(np.float32))
        y_list.append(np.random.randint(0, 5, size=(k,)).astype(np.float32))
    return X_list, y_list

# Replace with your actual loaded data instead of toy:
# X_list, y_list = load_grouped_parquet(...)
X_list, y_list = make_toy_data(num_queries=3000, D=D, min_k=1, max_k=178)

# --------------------
# Build buckets by slate length
# --------------------
from collections import defaultdict
buckets = defaultdict(list)
for i, X_q in enumerate(X_list):
    k = X_q.shape[0]
    buckets[k].append(i)

# Optionally filter out extremely long slates or subsample them
MAX_K_ALLOWED = 200  # if you want to cap
for k in list(buckets.keys()):
    if k > MAX_K_ALLOWED:
        # simple strategy: sample down to MAX_K_ALLOWED or drop those queries
        indices = buckets[k]
        # Here we drop them for simplicity
        print(f"Dropping {len(indices)} queries with K={k} > {MAX_K_ALLOWED}")
        del buckets[k]

# --------------------
# Model: item-wise MLP
# --------------------
class RankMLP(nn.Module):
    def __init__(self, d_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        # x: (B, K, D)
        B, K, D = x.shape
        x = x.view(B * K, D)
        out = self.net(x).view(B, K)
        return out

model = RankMLP(d_in=D, hidden=hidden_dim).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = GradScaler(enabled=use_mixed_precision)

# Losses
ndcg_loss = NeuralNDCGLoss(temperature=0.1)  # tune temperature
pointwise_loss_fn = nn.MSELoss()  # used for K==1 batches

# --------------------
# Helper: make batches for an epoch (bucketed batching)
# --------------------
def make_epoch_batches(buckets, batch_Q, shuffle=True, drop_last=True):
    """
    buckets: dict length -> list of indices
    yields: list of indices (length=batch_Q) where each index has same slate length
    """
    # Create per-bucket shuffled lists
    bucket_shuffled = {}
    for k, idxs in buckets.items():
        idxs_copy = idxs[:]  # shallow copy
        if shuffle:
            random.shuffle(idxs_copy)
        bucket_shuffled[k] = idxs_copy

    # For each bucket produce batches
    all_batches = []
    for k, idxs in bucket_shuffled.items():
        n = len(idxs)
        num_full = n // batch_Q
        for b in range(num_full):
            batch_idxs = idxs[b * batch_Q : (b + 1) * batch_Q]
            all_batches.append((k, batch_idxs))
        if not drop_last:
            rem = n % batch_Q
            if rem > 0:
                batch_idxs = idxs[num_full * batch_Q :]
                all_batches.append((k, batch_idxs))  # smaller batch
    # Shuffle across buckets to mix lengths during epoch
    if shuffle:
        random.shuffle(all_batches)
    return all_batches

# --------------------
# Training loop (bucketed batches)
# --------------------
def train_one_epoch(model, optimizer, scaler, X_list, y_list, buckets):
    model.train()
    batches = make_epoch_batches(buckets, batch_Q=batch_Q, shuffle=True, drop_last=True)
    total_loss = 0.0
    seen_batches = 0
    for (k, batch_idxs) in batches:
        # build batch tensors: all queries in this batch have same k
        B = len(batch_idxs)
        K = k
        # stack arrays
        X_batch = np.stack([X_list[i] for i in batch_idxs], axis=0)  # (B, K, D)
        y_batch = np.stack([y_list[i] for i in batch_idxs], axis=0)  # (B, K)
        X_batch = torch.from_numpy(X_batch).to(DEVICE)
        y_batch = torch.from_numpy(y_batch).to(DEVICE)

        with autocast(enabled=use_mixed_precision):
            scores = model(X_batch)  # (B, K)
            if K == 1:
                # pointwise fallback
                # map label (0..4) to float target (optionally normalize)
                loss = pointwise_loss_fn(scores.view(-1), y_batch.view(-1))
            else:
                # listwise
                loss = ndcg_loss(scores, y_batch)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (seen_batches + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        seen_batches += 1

        # optional logging
        if seen_batches % 100 == 0:
            # quick batch-level ndcg estimate (cpu)
            with torch.no_grad():
                s = scores.detach().cpu().numpy()
                yv = y_batch.detach().cpu().numpy()
                # compute simple ndcg@k for this batch (vectorized small utility)
                def dcg_batch(rel, scores, k=None):
                    # rel shape (B,K), scores shape (B,K)
                    order = np.argsort(-scores, axis=1)
                    if k is None: k = rel.shape[1]
                    top_rel = np.take_along_axis(rel, order[:, :k], axis=1)
                    denom = np.log2(np.arange(2, k + 2))
                    gains = (2 ** top_rel - 1)
                    return np.sum(gains / denom, axis=1)
                k_eval = min(10, K)
                dcg_v = dcg_batch(yv, s, k=k_eval)
                ideal_v = dcg_batch(yv, yv, k=k_eval)  # sorting by true rel
                ndcg_batch = np.mean(np.where(ideal_v == 0, 0.0, dcg_v / ideal_v))
            print(f"[batch {seen_batches}] loss={total_loss/seen_batches:.4f} ndcg@{k_eval}={ndcg_batch:.4f}")

    return total_loss / max(1, seen_batches)

# --------------------
# Run training
# --------------------
for epoch in range(1, epochs + 1):
    avg_loss = train_one_epoch(model, optimizer, scaler, X_list, y_list, buckets)
    print(f"Epoch {epoch} avg_loss={avg_loss:.6f}")
    # (Optional) validation pass, checkpointing, etc.
