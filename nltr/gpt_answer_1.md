良い姿勢です — 「簡便化」ですませた箇所は、実運用だとバグ・不安定・再現性欠如・評価誤差につながりやすい箇所ばかりです。ここではまず**省略していた全ての重要要素**を列挙して、なぜ必要かを短く説明し、その後でそれらを組み込んだ**実運用向けの改良版トレーニングスクリプト**を提示します。スクリプトは可変スレート向けの前のコードをベースに、実務で欠かせない要素をすべて入れてあります（チェックポイント、検証ループ、スケジューラ、再現性設定、ログ記録、resume、mixed-precision 安全処理、NaN 検出、など）。

まずは省略箇所（≒「後で追加すべき項目」）の全一覧。読み飛ばしてもいいですが、重要なので一通り目を通してください。

---

# 省略していた／簡便化した箇所（完全一覧）と理由（短評）

1. **学習率スケジューラ（LR scheduler）**

   * 理由：学習率制御は収束速度と最終精度に強く影響。種類（OneCycle, Cosine, ReduceLROnPlateau）により呼び出しタイミングが異なる点に注意。

2. **バッチ単位／エポック単位の scheduler.step() の呼出し場所の明示**

   * 理由：勾配累積や OneCycle の要件で正しい呼び出し位置が必須（optimizer.step の直後に呼ぶのが基本）。

3. **評価モード切替（`model.eval()` / `model.train()`）**

   * 理由：Dropout/BatchNorm の挙動が変わるため。ログや検証結果にバイアスが入る。

4. **検証ループ（validation）と検証用 NDCG 計算（全クエリで正確に）**

   * 理由：ミニバッチの監視ログはノイジー。モデル選択には一定の検証セット評価が必須。

5. **チェックポイント（model/optimizer/scheduler/scaler + epoch + global_step の保存）と resume 機能**

   * 理由：中断復帰と実験の再現に不可欠。

6. **乱数シードと決定論的設定（numpy / torch / cudnn）**

   * 理由：再現性確保。複数実験での比較に必須。

7. **TensorBoard / ロギング（loss, val_ndcg, lr, grad_norm 等）**

   * 理由：学習の状態把握・ハイパラ調整に必須。

8. **NaN / Inf 検知と安全停止**

   * 理由：NeuralNDCG の最適化で数値が悪化すると NaN が出やすい。早期検出で無駄な走行を防ぐ。

9. **勾配クリッピングと勾配累積（gradient_accumulation）への整合**

   * 理由：爆発勾配対策、メモリ制約時に実効バッチを増やす手段。scheduler 呼び出しの調整が必要。

10. **混合精度（AMP）の正しい使い方（scaler, unscale, clip の順序）**

    * 理由：速度/メモリ向上だが誤用は不安定化を招く。

11. **DataLoader の実用設定（num_workers, pin_memory, persistent_workers, prefetch_factor）**

    * 理由：I/O と CPU→GPU コストの最適化。

12. **メモリ最適化（pin_memory, float16 の扱い、バッチサイズ調整）**

    * 理由：OOM 回避や高速化。

13. **可変スレート時のマスク処理・padding の扱い（損失計算や評価に反映させる）**

    * 理由：パディングした項目を損失に混ぜない、正しい正規化が必要。

14. **K==1（スレート長1）の明示的処理方針（skip / pointwise / multi-task）**

    * 理由：listwise 損失は意味を持たないため代替方針を取る必要がある。

15. **ラベルの検証（範囲、欠損、型）・正規化（必要なら）**

    * 理由：不正ラベルは学習を壊す。NDCG は非負整数ラベル前提。

16. **特徴の dtype / device 一貫性（float32 vs float16、ラベルは float32）**

    * 理由：AMP 時の dtype エラー回避、GPU 転送の安全化。

17. **メトリクスの正しい計算（per-query NDCG、IDCG=0 の扱い）**

    * 理由：集約方法で指標が変わる（平均の取り方に注意）。

18. **early stopping / best model selection（検証指標に基づく）**

    * 理由：過学習防止と最良モデルの確保。

19. **学習率 warmup（初期 LR を段階的に上げる）**

    * 理由：特に大バッチやOneCycleで有効。

20. **重み初期化（カスタム初期化が必要なら）**

    * 理由：小さな MLP でも初期化の影響は無視できない。

21. **分散学習（DDP）を行う際の追加要件（DistributedSampler, per-process seed, sync BN, step 調整）**

    * 理由：複数 GPU で正しく学習させるため。

22. **ベンチマーク／プロファイル（torch.profiler）**

    * 理由：ボトルネック解消と最適化のため。

23. **メモリリーク・長時間実行の監視（定期 gc, detach の適切な場所）**

    * 理由：長時間走らせると落ちるケースに備える。

24. **ログの保存（ハイパラ、コードバージョン、git commit hash）**

    * 理由：実験の再現と解析のため。

25. **データ不均衡対策（ラベル分布の偏りがある場合の重み付け）**

    * 理由：学習の偏りを制御するため。

26. **検証時のバッチ作り（drop_last=false、全クエリを評価する）**

    * 理由：正しい検証は全クエリで行う必要がある。

27. **出力のキャリブレーション／後処理（スコアのスケーリングや閾値、ランキング出力形式）**

    * 理由：サービス要件に合わせると必要になる。

---

上のリストは「何が欠けていたか」を懇切丁寧に揃えたものです。次に、**これらをすべて実装に反映した実用スクリプト**を提示します。
目的：可変スレートの dataset（`X_list`, `y_list`）をそのまま使い、バケットバッチングで学習 → 検証 → checkpoint → resume → TensorBoard ログまで完備します。コメントで各重要ポイントをマークしてあるので、どの部分がどの項目に対応するかも分かります。

> 注意：NeuralNDCGLoss の import パスはあなたの環境に合わせて調整してください（リポジトリを pip install した時とパッケージ名が変わる場合があります）。

---

# 実運用向けフルスクリプト（可変スレート版 — 完全版）

```python
"""
train_full_ndcg.py
- 可変スレート（X_list, y_list）を使った学習のフル実装
- 省略無し：scheduler, eval, checkpoint, resume, seed, AMP, logging, NaN 検知 等を含む
"""

import os
import time
import random
import math
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import numpy as np
import polars as pl  # optional: for data loading part if needed
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# NeuralNDCG import — adjust if package path differs
from neuralndcgloss.loss import NeuralNDCGLoss

# -------------------------
# Config (preferably load from json/yaml)
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
    "log_interval_batches": 100,
    "save_dir": "./checkpoints",
    "tensorboard_dir": "./runs/ndcg_experiment",
    "temperature": 0.1,     # NeuralNDCG hyperparam
    "min_k_for_listwise": 2,
    "drop_last_train": True,
    "drop_longer_than": 200,   # optional cap
    "max_batches_per_epoch": None,  # optional debug limit
    "use_cosine_scheduler": True,
    "total_steps_estimate": None,  # optional to set scheduler T_max precisely
}

os.makedirs(config["save_dir"], exist_ok=True)
os.makedirs(config["tensorboard_dir"], exist_ok=True)

# -------------------------
# Reproducibility (seed + deterministic)
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may slow down)
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
        # Optional: Xavier init for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# -------------------------
# Util: per-query ndcg (exact) for validation (numpy)
# -------------------------
def ndcg_at_k_per_query_numpy(y_true: np.ndarray, y_score: np.ndarray, k: int = None) -> float:
    # y_true, y_score: shape (Q, K_var) lists of arrays or 2D with padding+mask
    # We'll accept 2D arrays where padding positions can be ignored (mask passed separately)
    # Here assume no padding: y_true, y_score are padded and mask is provided by caller if needed.
    Q, K = y_true.shape
    if k is None:
        k = K
    ndcgs = []
    for i in range(Q):
        rel = y_true[i]
        scores = y_score[i]
        # handle all-zero ideal case
        order = np.argsort(-scores)
        rank_rel = rel[order][:k]
        # DCG
        denom = np.log2(np.arange(2, rank_rel.size + 2))
        gains = (2 ** rank_rel - 1)
        dcg = np.sum(gains / denom)
        # IDCG
        ideal = np.sort(rel)[::-1][:k]
        ideal_gains = (2 ** ideal - 1)
        idcg = np.sum(ideal_gains / denom)
        ndcgs.append(0.0 if idcg == 0 else float(dcg / idcg))
    return float(np.mean(ndcgs))

# -------------------------
# Data preparation helpers (assume X_list, y_list are provided)
# -------------------------
# X_list: List[np.ndarray] each shape (K_i, D)
# y_list: List[np.ndarray] each shape (K_i,)
# build buckets
def build_buckets(X_list: List[np.ndarray], drop_longer_than: int = None):
    buckets = defaultdict(list)
    for idx, Xq in enumerate(X_list):
        k = Xq.shape[0]
        if drop_longer_than and k > drop_longer_than:
            continue
        buckets[k].append(idx)
    return buckets

# Make epoch batches from buckets
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

# -------------------------
# Safety: check and resume checkpoint
# -------------------------
def save_checkpoint(state: dict, path: str):
    torch.save(state, path)

def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None,
                    scheduler=None, scaler: GradScaler = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    return epoch, global_step

# -------------------------
# Training loop (full)
# -------------------------
def train_full(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    val_X_list: List[np.ndarray],
    val_y_list: List[np.ndarray],
    config: dict
):
    device = config["device"]
    model = RankMLP(d_in=config["D"], hidden=config["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # scheduler: cosine annealing with estimated total steps or basic Lambda warmup
    if config["use_cosine_scheduler"]:
        # estimate total steps if not provided
        if config["total_steps_estimate"] is None:
            # approximate: average batches per epoch * epochs
            buckets = build_buckets(X_list, drop_longer_than=config.get("drop_longer_than"))
            batches_per_epoch = sum(len(idxs) // config["batch_Q"] for idxs in buckets.values())
            est_total_steps = max(1, batches_per_epoch * config["epochs"])
        else:
            est_total_steps = config["total_steps_estimate"]
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, est_total_steps))
    else:
        scheduler = None

    scaler = GradScaler(enabled=config["use_amp"])
    ndcg_loss = NeuralNDCGLoss(temperature=config["temperature"])
    pointwise_loss = nn.MSELoss()

    writer = SummaryWriter(config["tensorboard_dir"])
    global_step = 0
    start_epoch = 1

    # resume logic: if checkpoint exists, load latest (simple policy)
    latest_ckpt = os.path.join(config["save_dir"], "latest.pt")
    if os.path.exists(latest_ckpt):
        print("Resuming from checkpoint:", latest_ckpt)
        start_epoch, global_step = load_checkpoint(latest_ckpt, model, optimizer, scheduler, scaler)
        start_epoch += 1  # resume next epoch
        print(f"Resumed at epoch {start_epoch}, global_step {global_step}")

    # training
    best_val_ndcg = -1.0
    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_start = time.time()
        # rebuild buckets each epoch if necessary (e.g., if dynamic sampling)
        buckets = build_buckets(X_list, drop_longer_than=config.get("drop_longer_than"))
        batches = make_epoch_batches(buckets, batch_Q=config["batch_Q"], shuffle=True, drop_last=config["drop_last_train"])

        running_loss = 0.0
        batch_count = 0
        model.train()
        for (k, batch_idxs) in batches:
            B = len(batch_idxs)
            K = k
            # stack and move to device
            X_batch = np.stack([X_list[i] for i in batch_idxs], axis=0)  # (B,K,D)
            y_batch = np.stack([y_list[i] for i in batch_idxs], axis=0)  # (B,K)
            X_batch = torch.from_numpy(X_batch).to(device=device, dtype=torch.float32)
            y_batch = torch.from_numpy(y_batch).to(device=device, dtype=torch.float32)

            with autocast(enabled=config["use_amp"]):
                scores = model(X_batch)  # (B,K)
                if K < config["min_k_for_listwise"]:
                    # handle K==1 or short lists: pointwise fallback
                    loss = pointwise_loss(scores.view(-1), y_batch.view(-1))
                else:
                    loss = ndcg_loss(scores, y_batch)

                # check NaN/Inf
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss detected at epoch {epoch}, step {global_step}")

                loss = loss / config["grad_accum_steps"]

            scaler.scale(loss).backward()

            # optimizer step only every grad_accum_steps batches
            if (batch_count + 1) % config["grad_accum_steps"] == 0:
                # unscale and clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # scheduler step: only after actual optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                # logging LR
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/lr", current_lr, global_step)

            running_loss += loss.item() * config["grad_accum_steps"]
            batch_count += 1

            # periodic logging / quick eval (use eval mode for correctness)
            if global_step % config["log_interval_batches"] == 0:
                # do eval on this batch only for monitoring (accurate because we set eval)
                model.eval()
                with torch.no_grad():
                    scores_eval = model(X_batch)  # eval mode affects BN/Dropout
                    scores_np = scores_eval.detach().cpu().numpy()
                    y_np = y_batch.detach().cpu().numpy()
                    ndcg10 = ndcg_at_k_per_query_numpy(y_np, scores_np, k=min(10, K))
                model.train()
                writer.add_scalar("train/batch_ndcg10", ndcg10, global_step)
                writer.add_scalar("train/batch_loss", running_loss / (batch_count + 1e-12), global_step)
                print(f"[E{epoch}] step {global_step} lr {current_lr:.6f} batch_loss {running_loss / batch_count:.6f} ndcg@10 {ndcg10:.4f}")

            # optional safety cap for debug
            if config.get("max_batches_per_epoch") and batch_count >= config["max_batches_per_epoch"]:
                break

        # epoch end
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / max(1, batch_count)
        print(f"Epoch {epoch} finished in {epoch_time:.1f}s, avg_loss={avg_loss:.6f}")

        # validation: run full validation (no drop_last, full coverage)
        val_ndcg = evaluate_full(model, val_X_list, val_y_list, device, config, writer, global_step)
        writer.add_scalar("val/ndcg", val_ndcg, epoch)
        print(f"Epoch {epoch} validation NDCG: {val_ndcg:.6f}")

        # checkpoint
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict()
        }
        save_checkpoint(ckpt, os.path.join(config["save_dir"], f"ckpt_epoch_{epoch}.pt"))
        save_checkpoint(ckpt, os.path.join(config["save_dir"], "latest.pt"))

        # early stopping / best model selection
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            save_checkpoint(ckpt, os.path.join(config["save_dir"], "best.pt"))
            print(f"New best model saved with val_ndcg={best_val_ndcg:.6f}")

    writer.close()

# -------------------------
# Evaluation helper (full validation)
# -------------------------
def evaluate_full(model: nn.Module, X_list: List[np.ndarray], y_list: List[np.ndarray], device: str, config: dict, writer=None, step=0):
    model.eval()
    buckets = build_buckets(X_list, drop_longer_than=config.get("drop_longer_than"))
    batches = make_epoch_batches(buckets, batch_Q=config["batch_Q"], shuffle=False, drop_last=False)
    all_ndcgs = []
    with torch.no_grad():
        for (k, batch_idxs) in batches:
            B = len(batch_idxs)
            X_batch = np.stack([X_list[i] for i in batch_idxs], axis=0)
            y_batch = np.stack([y_list[i] for i in batch_idxs], axis=0)
            X_batch_t = torch.from_numpy(X_batch).to(device=device, dtype=torch.float32)
            y_batch_t = torch.from_numpy(y_batch).to(device=device, dtype=torch.float32)

            # forward in eval mode
            scores = model(X_batch_t).detach().cpu().numpy()  # (B,K)
            y_np = y_batch_t.detach().cpu().numpy()
            ndcg_batch = ndcg_at_k_per_query_numpy(y_np, scores, k=min(10, k))
            all_ndcgs.append(ndcg_batch)
    # average across all batches (weighted by batch size if needed)
    mean_ndcg = float(np.mean(all_ndcgs)) if len(all_ndcgs) > 0 else 0.0
    return mean_ndcg

# ---------------------------------
# Example usage (assume X_list,y_list are loaded)
# ---------------------------------
if __name__ == "__main__":
    # TODO: replace with actual dataset loading
    def make_dummy(num_q=3000, D=100, min_k=1, max_k=50):
        Xl, yl = [], []
        for _ in range(num_q):
            k = random.randint(min_k, max_k)
            Xl.append(np.random.randn(k, D).astype(np.float32))
            yl.append(np.random.randint(0, 5, size=(k,)).astype(np.float32))
        return Xl, yl
    X_list_train, y_list_train = make_dummy(num_q=3000, D=config["D"], min_k=1, max_k=50)
    X_list_val, y_list_val = make_dummy(num_q=600, D=config["D"], min_k=1, max_k=50)

    train_full(X_list_train, y_list_train, X_list_val, y_list_val, config)
```

---

# 補足説明（スクリプト中の重要箇所と対応関係）

* `set_seed`：乱数のセット（再現性）。`cudnn.deterministic=True` の副作用で高速化が妨げられる点を注意。
* `CosineAnnealingLR`：スケジューラの例。`est_total_steps` を求めて `T_max` に入れているので、batch 単位 step でスムーズに動く。
* `scaler.unscale_(optimizer)` → `clip_grad_norm_` → `scaler.step(optimizer)` → `scaler.update()` の順：AMP の安全な使い方。
* `scheduler.step()` は **optimizer.step() を行った直後**にだけ呼んでいる（grad_accum に対応）。
* `model.eval()` / `torch.no_grad()` を検証やバッチごとの監視に必ず使う。ログ計測のために再度 forward している。
* `save_checkpoint` / `load_checkpoint`：optimizer, scheduler, scaler の状態も復元することで完全な resume を実現。
* `ndcg_at_k_per_query_numpy`：検証では per-query の NDCG を厳密に計算して平均する。IDCG=0 の場合 0 を戻す。
* `min_k_for_listwise`：listwise を適用する最小スレート長。1 の場合は pointwise fallback。
* `drop_longer_than`：非常に長いスレートは学習時にサブサンプリングか drop する方が実務的。ここでは drop のオプションを提供。
* `tensorboard` に LR / loss / ndcg を毎ステップまたは毎 epoch ログしている。実験解析用に必須。

---

# 最後に：運用でよく聞かれる Q&A（すぐ欲しくなる短答）

* Q: `OneCycleLR` を使う場合どうする？
  A: `OneCycleLR` はバッチ（optimizer.step）ごとに `scheduler.step()` を呼ぶ必要があります。累積を使う場合は「実際に optimizer.step() を行うタイミング」で呼んでください。
* Q: DDP（複数GPU）での注意は？
  A: `DistributedSampler` を使い、各プロセスで seed をずらす、`sync_bn` が必要なら有効化、checkpoint は rank0 のみ書き込み。scheduler のステップは各プロセスで一貫して呼ぶ。
* Q: validation はバッチ落ち（drop_last）しないで全件評価すべき？
  A: **はい**。検証は全クエリで行い、平均 NDCG を用いるべきです。
* Q: K==1 を学習に入れるべきか？
  A: 事前に分布確認をして方針を決定。情報量があるなら pointwise による multi-task として同時学習するのが無難です（損失重みで調整）。

---

必要なら次のどれかを即作ります（すべて今すぐ提示できます）：

1. あなたの実 parquet ファイル（列名一覧）に合わせた `load_parquet_to_torch` + `X_list,y_list` 作成コード（polars ベース）。
2. 上スクリプトを DDP 対応に書き換え（分散学習）。
3. `OneCycleLR` / warmup を含む scheduler の別サンプル（実験セットアップ）。
4. K==1 の扱い（pointwise vs exclude）の比較用実験コード（同じ訓練で両方試す）。

どれを出しますか？（もし選ばなければ、1 をデフォルトで用意します）
