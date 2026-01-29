良い問い — 実運用での **観測バイアス（特にポジションバイアス）** に対する対策は重要かつ奥が深いです。あなたが *IPW（逆率重み付け）を直接使う手法を避けたい* と言っている点を踏まえ、ここでは **（A）モデル設計／損失設計による「デバイアス（逆因果的介入を使わない）」アプローチ** と **（B）関連する別アプローチの短い整理** を示し、最後に **実装コード例（Gradient Reversal と Observation Dropout を Two-Tower に組み込む）** を提示します。重要な論拠は KDD’23 論文 “Towards Disentangling Relevance and Bias in Unbiased Learning to Rank” をベースにしています。([arXiv][1])

# 要旨（結論ファースト）

* IPW を避けたいなら、**二塔（two-tower）構造における「観測（bias）塔が関連性情報を奪わないようにする」手法**が有力です。KDD’23 は **（1）Gradient Reversal（敵対的に観測塔から関連性情報を“取り除く”）と（2）Observation Dropout（観測塔の出力を確率的にゼロ化してショートカット学習を防ぐ）** を提案し効果を示しています。([arXiv][1])
* 同系統の別アプローチ（ベクトル化 EH / LBD 等）も有望で、観測×関連性の相互作用をより表現力豊かにモデリングすることで誤差を減らします。([arXiv][2])
* 現実的には **これらを既存の listwise+pointwise 学習パイプラインに組み込み、ハイパラ（勾配反転係数、ドロップアウト率、敵対損失重み）をチューニング** するのが実用的です。([arXiv][1])

# 理由（なぜ IPW を避けてこうするのか）

* 実ログの `position`（あるいは initial_rank）は **過去のランカー（logging policy）に依存**しており、その policy 自体が関連性情報を持つため、単純に位置特徴を観測塔にいれてしまうと「観測塔が関連性を表現してしまう（共変／交絡）」ことがあります。すると relevance tower が学習されず性能が落ちる。これが KDD’23 の指摘点です。([arXiv][1])
* IPW 系は理論的に正しいが、propensity の推定が難しい／高分散／ログにpropensityが無い等の実務課題があります。したがって「観測塔の学習能力を制御して共役を解く」やり方は実運用で有用です。([research.google.com][3])

# 改良案（設計ベクトル）

1. **Two-Tower ベースライン → 拡張**

   * Relevance tower (f_\theta(x_R))：あなたが最初に使っていた RankMLP をここで使う（listwiseスコアはこの出力を使う）。
   * Observation tower (g_\phi(x_O))：`position`, `platform`, `bm25`（ただし bm25 は relevance 兼 observation になり得るので注意）などを入力。観測確率（observation）や bias スコアを出す。
   * Click モデル：`click_prob = sigmoid(f + g)`（加法）など。学習は click の cross-entropy。([arXiv][1])
2. **Gradient Reversal（敵対）**

   * Observation tower の表現から **「関連性を予測しないように」** 強制するために、observation 出力の上流に小さな head を付け、そこに対して「relevance label（または relevance tower の予測）」を教師として学習するが、バックプロパゲーション時にその勾配を *反転*（負にスケーリング）して観測塔へ流す。これにより観測塔は関連性情報を保持しにくくなる。([arXiv][1])
3. **Observation Dropout（出力ドロップ）**

   * Observation tower の最終出力（bias スコア）に対して確率的にゼロ化（rate α）を行う。これで観測塔が関連性のショートカットを学ぶのを抑制する。Paper は固定 rate を使うが、学習可能な gating にしても良い（ただし安定性に注意）。([arXiv][1])
4. **損失合成**（既存の multi-task 設計との統合）

   * relevance tower は listwise（NeuralNDCG）＋ pointwise（MSE）で学習（あなたの既存設定）。
   * click prediction loss（CE）で全体（f + g）を訓練。
   * adversarial loss（relevance を obs から予測する head の MSE/CE）を *negative* にスケールして観測塔へ逆伝播（梯子のために GradientReversal を使う）。もしくは「grad reversal を入れた上で head に正の損失を課す」。
   * 合成例： `total = λ_rel*(listwise + pointwise) + λ_click*CE_click + λ_adv*L_adv`（ただし gradient reversal によって obs tower は L_adv を minimize しない方向に更新される）. ([arXiv][1])
5. **実務トリック**

   * adversarial ラベルは「click」「relevance_tower_pred（detach）」などが利用可能で、KDD’23 はどれも実用上うまく働くと報告しています（クリックでも可）。([arXiv][1])
   * GradRev の強さ（スカラー η）や dropout rate α、各損失の λ はデータ次第で敏感なので mlflow でハイパラ探索を強く推奨。([arXiv][1])

# 実装コード例（最小限かつ実装可能な形）

以下は **(A)** TwoTower モデル（Relevance tower + Observation tower）に **GradientReversal** と **ObservationDropout** を組み込み、**あなたの既存 training loop の total_loss** に合流させるための補助実装です。

* 既存の `RankMLP`（relevance tower）と NeuralNDCG ベースの listwise 損失はそのまま使います（ここでは `relevance_model` を `RankMLP` と仮定）。
* 具体的には `TwoTowerDebiasModel` を作り、training では `scores_rel = model.relevance(X)` を listwise に使い、click_pred = sigmoid(scores_rel + g_obs) を click loss に使います。
* `GradientReversal` は PyTorch の custom autograd を使った典型的な実装です。

```python
# debias_two_tower.py  — 部分抜粋（組み込み例）
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Gradient Reversal Layer ---
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # reverse and scale gradient
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradientReversalFn.apply(x, lambd)

# --- Observation Dropout layer (simple) ---
class ObservationDropout(nn.Module):
    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        if not self.training or self.drop_rate <= 0.0:
            return x
        # apply same dropout value across batch (or elementwise if desired)
        mask = (torch.rand_like(x) > self.drop_rate).float()
        # scale to keep expectation stable
        mask = mask / (1.0 - self.drop_rate)
        return x * mask

# --- Two-Tower Debias model ---
class TwoTowerDebiasModel(nn.Module):
    def __init__(self, d_in: int, hidden: int, obs_input_dim: int, obs_hidden: int,
                 gradrev_lambda=0.6, obs_dropout_rate=0.3):
        """
        - d_in: feature dim for relevance tower (query-doc features)
        - obs_input_dim: feature dim for observation tower (e.g., position embedding + maybe bm25)
        """
        super().__init__()
        # relevance tower: can reuse your RankMLP (item-wise)
        self.relevance = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # observation tower: small MLP that consumes observation features (position, platform, maybe bm25)
        self.observation = nn.Sequential(
            nn.Linear(obs_input_dim, obs_hidden),
            nn.ReLU(),
            nn.Linear(obs_hidden, 1)  # output single bias score per item
        )

        # adversarial head: from observation representation to relevance proxy
        # we will use a small head that tries to predict relevance labels from obs tower,
        # but its gradients are reversed before flowing back into observation tower.
        self.adv_head = nn.Sequential(
            nn.Linear(1, obs_hidden),  # note: taking observation tower's scalar output as input; you can instead use deeper representation
            nn.ReLU(),
            nn.Linear(obs_hidden, 1)
        )

        self.gradrev_lambda = gradrev_lambda
        self.obs_dropout = ObservationDropout(obs_dropout_rate)

        # init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_relevance: torch.Tensor, x_obs: torch.Tensor, return_components=False):
        """
        x_relevance: (B, K, D_relevance)   -> processed item-wise via relevance tower
        x_obs:       (B, K, D_obs)         -> processed item-wise via observation tower
        Returns:
          scores_rel: (B, K)   relevance tower raw scores (not sigmoid)
          obs_score:  (B, K)   observation tower raw scores (not sigmoid)
          click_prob: (B, K)   sigmoid(scores_rel + obs_score)
        """
        B, K, _ = x_relevance.shape
        # flatten item-wise
        rel_flat = x_relevance.view(B * K, -1)
        obs_flat = x_obs.view(B * K, -1)

        scores_rel = self.relevance(rel_flat).view(B, K)         # (B,K)
        obs_raw = self.observation(obs_flat).view(B, K)         # (B,K)

        # observation dropout (elementwise)
        obs_raw_drop = self.obs_dropout(obs_raw)

        click_logits = scores_rel + obs_raw_drop                  # additive interaction
        click_prob = torch.sigmoid(click_logits)

        if return_components:
            return scores_rel, obs_raw, obs_raw_drop, click_prob
        return click_prob

    def adversarial_loss(self, obs_raw: torch.Tensor, adv_label: torch.Tensor, adv_criterion):
        """
        obs_raw: (B, K) (before dropout)
        adv_label: (B, K) e.g. relevance proxy (detached) or clicks
        adv_criterion: loss function (e.g., MSELoss or BCE)
        We apply grad reversal to obs_raw before passing to adv_head.
        """
        B, K = obs_raw.shape
        inp = obs_raw.view(B * K, 1)            # shape (B*K, 1)
        # reversed representation
        rev = grad_reverse(inp, self.gradrev_lambda)
        pred_rev = self.adv_head(rev).view(B, K)  # predicted relevance proxy
        # adv_criterion compares pred_rev to adv_label
        loss_adv = adv_criterion(pred_rev, adv_label)
        return loss_adv
```

## トレーニングでの損失合成（擬似コード）

（あなたの既存スクリプトに挿入する形で）

```python
# inside training loop, for a batch:
# X_rel: (B,K,D_rel) -> features for relevance tower (the item features)
# X_obs: (B,K,D_obs) -> features for observation tower (e.g. one-hot pos, platform; optionally include bm25 but be careful)
# y_click: binary clicks (B,K)  -- if not available, could use click proxies
# y_rel_labels: graded 0..4 (B,K)  -- for listwise and pointwise

scores_rel, obs_raw, obs_raw_drop, click_prob = model.forward(X_rel, X_obs, return_components=True)

# 1) listwise and pointwise losses operate on scores_rel (relevance tower)
list_loss = ndcg_loss_fn(scores_rel, y_rel_labels)   # NeuralNDCG
point_loss = F.mse_loss(scores_rel.view(-1), y_rel_labels.view(-1))

# 2) click prediction loss on combined logit (sigmoid used inside)
click_logits = scores_rel + obs_raw_drop
click_loss = F.binary_cross_entropy_with_logits(click_logits, y_click)

# 3) adversarial loss: try to predict relevance proxy from obs_raw, but gradient reversed inside
# adv_label choice: could be y_click (click), or scores_rel.detach() (relevance tower prediction), or a pretrain label
adv_label = scores_rel.detach()   # option: use relevance tower's current preds as proxy
adv_criterion = nn.MSELoss()
adv_loss = model.adversarial_loss(obs_raw, adv_label, adv_criterion)

# total
total_loss = (lambda_rel_list * list_loss) + (lambda_rel_point * point_loss) \
             + (lambda_click * click_loss) + (lambda_adv * adv_loss)

# backward & optimizer step as usual (with AMP/GradScaler etc.)
```

### ハイパラ（実務的初期値）

* `gradrev_lambda`（η）: 0.5–0.8（KDD’23 は 0.6–0.8 が有効領域であると報告）。([arXiv][1])
* `obs_dropout_rate`（α）: 0.2–0.5（感度あり）。([arXiv][1])
* `lambda_adv`（敵対損失重み）: 0.1〜1.0（探索が必要）
* `lambda_click`（クリック損失重み）: 環境により 0.1–1.0

# 追加の（補助的）工夫と別手法のまとめ

* **LBD / Vectorization**：ユーザ／アイテムレベルで relev/obs をよりリッチにデカップリングする LBD（NeurIPS’22）や Vectorization（KDD’22）は、EH（検査仮説＝scalar）を緩める方向で、複雑な相互作用がある場面で有利。([NeurIPS Proceedings][4])
* **Doubly-Robust / SNIPS / DR estimators**：IPW の高分散問題を軽減するための補完的手法。ただしこれらは根本的に propensity を使う系です（あなたは避けたいとのこと）。([arXiv][5])
* **Interleaving / online randomization**：実際にランダム化データを小規模に収集して推定／検証に使う。ユーザ体験とのトレードオフあり。([arXiv][1])
* **Representation-level deconfounding（因果表現学習）**：観測変数と関連性変数を分ける因果表現学習を導入する研究も進んでいます。
* **Propensity as feature (learned)**：propensity を明示的にモデルし、その予測値を利用して安定化する手法（属性ベース推定など）。([research.google.com][3])

# 実運用での注意（リスクと確認事項）

* GradRev は **収束がやや遅く不安定**になりやすい（KDD’23 も注意）。学習曲線・validation をきめ細かく監視し、`lambda_adv` と `gradrev_lambda` を小さめから探索してください。([arXiv][1])
* ObservationDropout は **適切なドロップ率探索が重要**（高すぎると観測情報を失い click 予測が落ちる、低すぎると共変が残る）。([arXiv][1])
* adversarial ラベルの選択（クリック／relevance_pred）で結果変わるが、KDD’23 は大きな差はなかったと報告（実環境で試す価値あり）。([arXiv][1])

# 最後に：あなたのパイプラインへの統合提案（実務順）

1. まずは **Two-Tower + ObservationDropout** を既存スクリプトに軽く追加して小規模検証。
2. 次に **GradRev** を導入（より強力だが扱いは慎重に）→ mlflow で η・α・λ を grid/beam 探索。
3. 成果が出れば **LBD/Vectorization** 等の次世代手法で改善。
4. 必要に応じて限定的な A/B（ランダム化）で因果的裏付けを取得。([arXiv][1])

---

もしよければ、次のどれを作りますか？（すぐ作ります）

1. 上の TwoTower を **あなたの既存フル training スクリプト（mlflow + grad_accum + RAdamScheduleFree）に統合した完全版**（つまり今までの `train_full_ltr_mlflow.py` にこの debias モジュールを組み込む）。
2. GradRev / Dropout の **ハイパラ探索用 mlflow スクリプト（ベイジアン／grid 用）テンプレ**。
3. LBD や Vectorization の **簡易実装（参考実装）** を示す（やや手間）。

どれを最優先で出しましょうか？

[1]: https://arxiv.org/pdf/2212.13937 "Towards Disentangling Relevance and Bias in Unbiased Learning to Rank"
[2]: https://arxiv.org/abs/2206.01702?utm_source=chatgpt.com "Scalar is Not Enough: Vectorization-based Unbiased Learning to Rank"
[3]: https://research.google.com/pubs/archive/46485.pdf?utm_source=chatgpt.com "Position Bias Estimation for Unbiased Learning to Rank in ..."
[4]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/d81cb1f4dc6e13aeb45553f80b3d6837-Abstract-Conference.html?utm_source=chatgpt.com "LBD: Decouple Relevance and Observation for Individual- ..."
[5]: https://arxiv.org/pdf/1804.05938?utm_source=chatgpt.com "Unbiased Learning to Rank with Unbiased Propensity ..."

---

いい選択です — 要求どおり、**Two-Tower（relevance + observation）にDebias手法（Gradient Reversal + Observation Dropout）を組み込み**、これまでの要件（polars→parquet、BucketBatchSampler、grad_accum、AMP、RAdamScheduleFree、mlflow、NeuralNDCG を用いた listwise、pointwise、click loss の multi-task 学習、チェックポイント・resume 等）すべてを統合した**実運用向けの完全スクリプト**を用意しました。

* 重要実装点

  * `QueryItemDataset` は `feature_cols`（relevance 用）と `obs_feature_cols`（observation 用）を受け取り、グループ内の並びを `sort_col` で固定。`pos` を `obs_feature_cols` に入れれば位置情報が観測特徴として使われます。
  * `BucketBatchSampler` により「同一スレート長（K）だけを含むバッチ」を毎エポックランダムに作ります（`set_epoch` で再現可能）。
  * `TwoTowerDebiasModel`：relevance tower（item-wise MLP）、observation tower（item-wise MLP）、`ObservationDropout`、`adv_head`（敵対ヘッド）、`GradientReversal` を実装。`forward` は `(scores_rel, obs_raw, obs_raw_drop, click_prob)` を返します。
  * 学習損失は合成：`λ_rel*(listwise + pointwise) + λ_click*click_loss + λ_adv*adv_loss`。`adv_loss` は `adv_head` の予測と `adv_label`（設定で `scores` または `click`）で計算、`grad_reverse` により observation tower の重みは「関連性推定をしない方向」に更新されます。
  * `click_col` が parquet に無ければ、`click` を `label > 0` のバイナリ proxy として使います（実データに click があるならその列を使うことを推奨）。
  * 完全な運用機能：AMP（`autocast(enabled=...)` + `GradScaler(enabled=...)`）、`grad_accum_steps`、`RAdamScheduleFree`（scheduler不要）、mlflow ロギング（ハイパラ、ステップ/epoch メトリクス、チェックポイント保存）、検証（NDCG @ k と pointwise RMSE）、resume を含みます。

以下がその**完全スクリプト**（長いですがそのまま実運用可能な形）。
実行前に必要なライブラリをインストールしてください（例）:

```bash
pip install polars schedulefree mlflow
pip install git+https://github.com/JohnYKiyo/ListwiseRankingLoss-NeuralNDCG.git
```

---

```python
# train_two_tower_debias_mlflow.py
"""
Two-Tower Debias LTR training (production-ready)
- polars -> QueryItemDataset (q-group, sort_col ordering)
- BucketBatchSampler ensures same-slate batches
- TwoTowerDebiasModel: relevance tower + observation tower + grad-reversal + obs-dropout + adv_head
- Multi-task loss: listwise(NeuralNDCG) + pointwise(MSE) + click BCE + adversarial MSE
- Optimizer: RAdamScheduleFree (no external scheduler)
- AMP (autocast + GradScaler), grad_accum_steps, checkpointing, mlflow logging
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
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Sampler

# external libs - ensure installed
from schedulefree import RAdamScheduleFree
from neuralndcgloss.loss import NeuralNDCGLoss

# ----------------------------
# Config - tune to your env
# ----------------------------
CONFIG = {
    # Data paths
    "train_parquet": "data/train.parquet",
    "val_parquet": "data/val.parquet",
    "qid_col": "qid",
    "item_id_col": "item_id",
    "label_col": "label",       # graded relevance 0..4
    "click_col": "click",       # optional; if missing, proxy used (label>0)
    "sort_col": "initial_rank", # group-internal ordering (small=top)
    # features: two groups (relevance input, observation input)
    # If None, auto-detect relevance features (excluding reserved); obs_feature_cols defaults to ['pos','bm25'] if bm25 present
    "feature_cols": None,
    "obs_feature_cols": None,
    # Model / training
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_dim": 128,
    "batch_Q": 32,
    "grad_accum_steps": 4,
    "grad_clip": 1.0,
    "epochs": 20,
    "min_k_for_listwise": 2,
    "drop_longer_than": 400,
    "allow_incomplete_batch": True,
    "max_batches_per_epoch": None,
    # loss weights
    "lambda_listwise": 1.0,
    "lambda_pointwise": 1.0,
    "lambda_click": 1.0,
    "lambda_adv": 0.5,
    # adversarial settings
    "gradrev_lambda": 0.6,
    "obs_dropout_rate": 0.3,
    "adv_label_source": "scores",  # "scores" or "click"
    # optimizer & amp
    "optimizer_lr": 1e-4,
    "use_amp": True,
    "seed": 42,
    # mlflow
    "mlflow_experiment": "ltr_two_tower_debias",
    "checkpoint_dir": "./checkpoints",
    # dataloader
    "num_workers": 4,
    "pin_memory": True,
    # logging
    "log_interval_steps": 200,
}

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

# ----------------------------
# Utility: reproducibility
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG["seed"])

# ----------------------------
# Dataset: polars -> per-query arrays with obs features support
# ----------------------------
class QueryItemDataset(Dataset):
    """
    Groups parquet rows by qid. Returns per-query:
      - 'qid', 'X' (relevance features) shape (K,D_rel),
      - 'X_obs' (obs features) shape (K,D_obs),
      - 'y' labels shape (K,),
      - 'click' shape (K,) (binary)
      - 'item_ids', 'K'
    """
    def __init__(
        self,
        parquet_path: str,
        qid_col: str,
        item_id_col: Optional[str],
        label_col: str,
        click_col: Optional[str],
        feature_cols: Optional[List[str]],
        obs_feature_cols: Optional[List[str]],
        sort_col: Optional[str],
        drop_qids_with_k_lt: Optional[int] = None,
    ):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(parquet_path)
        self.parquet_path = parquet_path
        self.qid_col = qid_col
        self.item_id_col = item_id_col
        self.label_col = label_col
        self.click_col = click_col
        self.sort_col = sort_col

        df = pl.read_parquet(parquet_path)

        # reserved columns
        reserved = {qid_col, label_col}
        if item_id_col:
            reserved.add(item_id_col)
        if sort_col:
            reserved.add(sort_col)
        if click_col:
            reserved.add(click_col)

        # if feature_cols not provided -> auto-detect (exclude reserved)
        if feature_cols is None:
            self.feature_cols = [c for c in df.columns if c not in reserved]
        else:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"feature_cols missing: {missing}")
            self.feature_cols = feature_cols

        # obs_feature_cols default: include 'pos' and 'bm25' if exists
        if obs_feature_cols is None:
            self.obs_feature_cols = []
            if "bm25" in df.columns:
                self.obs_feature_cols.append("bm25")
            # include 'pos' to represent position within group
            self.obs_feature_cols.append("pos")
        else:
            self.obs_feature_cols = obs_feature_cols

        # global sort by qid and sort_col (if provided)
        sort_keys = [qid_col]
        if sort_col:
            sort_keys.append(sort_col)
        df = df.sort(sort_keys)

        # group and aggregate lists for all columns we need
        # prepare aggregation expressions for feature_cols and obs_feature_cols and label/click/item_id
        agg_cols = [pl.col(c).list().alias(c) for c in set(self.feature_cols + self.obs_feature_cols)]
        agg_cols.append(pl.col(label_col).list().alias(label_col))
        if click_col:
            agg_cols.append(pl.col(click_col).list().alias(click_col))
        if item_id_col:
            agg_cols.append(pl.col(item_id_col).list().alias(item_id_col))
        agg_cols.append(pl.count().alias("K"))

        grouped = df.groupby(qid_col).agg(agg_cols)

        # optional filter by min K
        if drop_qids_with_k_lt is not None:
            grouped = grouped.filter(pl.col("K") >= drop_qids_with_k_lt)

        self.qids = grouped[qid_col].to_list()
        self.Ks = grouped["K"].to_list()
        Q = len(self.qids)
        if Q == 0:
            raise ValueError("No queries after grouping/filters")

        self.X_list = []
        self.X_obs_list = []
        self.y_list = []
        self.click_list = []
        self.item_ids_list = []

        # materialize per-query arrays
        for i in range(Q):
            k = int(self.Ks[i])
            # relevance features: build matrix from self.feature_cols
            rel_cols_arrays = []
            for c in self.feature_cols:
                arr = np.asarray(grouped[c][i], dtype=np.float32).reshape(k, 1)
                rel_cols_arrays.append(arr)
            X_rel = np.concatenate(rel_cols_arrays, axis=1) if rel_cols_arrays else np.zeros((k, 0), dtype=np.float32)

            # observation features: for 'pos' generate range; for others fetch lists
            obs_cols_arrays = []
            for c in self.obs_feature_cols:
                if c == "pos":
                    arr = np.arange(k, dtype=np.float32).reshape(k, 1)  # 0-indexed position
                else:
                    arr = np.asarray(grouped[c][i], dtype=np.float32).reshape(k, 1)
                obs_cols_arrays.append(arr)
            X_obs = np.concatenate(obs_cols_arrays, axis=1) if obs_cols_arrays else np.zeros((k, 0), dtype=np.float32)

            labels = np.asarray(grouped[self.label_col][i], dtype=np.float32)
            # click handling: if click_col present use it; else proxy as label>0
            if self.click_col and self.click_col in grouped.columns:
                clicks = np.asarray(grouped[self.click_col][i], dtype=np.float32)
            else:
                clicks = (labels > 0).astype(np.float32)

            self.X_list.append(X_rel)
            self.X_obs_list.append(X_obs)
            self.y_list.append(labels)
            self.click_list.append(clicks)
            if self.item_id_col:
                self.item_ids_list.append(grouped[self.item_id_col][i])
            else:
                self.item_ids_list.append([None] * k)

        # feature dims
        self.D_rel = self.X_list[0].shape[1]
        self.D_obs = self.X_obs_list[0].shape[1]

    def __len__(self) -> int:
        return len(self.qids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "qid": self.qids[idx],
            "X": self.X_list[idx],          # (K, D_rel)
            "X_obs": self.X_obs_list[idx],  # (K, D_obs)
            "y": self.y_list[idx],          # (K,)
            "click": self.click_list[idx],  # (K,)
            "item_ids": self.item_ids_list[idx],
            "K": int(self.X_list[idx].shape[0]),
        }

# ----------------------------
# BucketBatchSampler (same-slate batches)
# ----------------------------
class BucketBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: QueryItemDataset, batch_size: int,
                 shuffle: bool = True, allow_incomplete: bool = True, seed: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.allow_incomplete = allow_incomplete
        self.seed = seed if seed is not None else 0
        self._build_buckets()
        self.epoch = 0

    def _build_buckets(self):
        self.buckets = {}
        for idx, k in enumerate(self.dataset.Ks):
            k = int(k)
            self.buckets.setdefault(k, []).append(idx)
        self.unique_ks = sorted(self.buckets.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        bucket_idxs = {k: lst[:] for k, lst in self.buckets.items()}
        for k, lst in bucket_idxs.items():
            if self.shuffle:
                rng.shuffle(lst)

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

        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for lst in self.buckets.values():
            n = len(lst)
            total += math.ceil(n / self.batch_size) if self.allow_incomplete else (n // self.batch_size)
        return total

# ----------------------------
# collate_fn - returns relevance X and obs X_obs stacked
# ----------------------------
def collate_bucket_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    if B == 0:
        raise ValueError("Empty batch")
    K = batch[0]["K"]
    # sanity check
    for item in batch:
        if item["K"] != K:
            raise ValueError("collate received mixed K")

    X_np = np.stack([item["X"] for item in batch], axis=0)      # (B,K,D_rel)
    Xobs_np = np.stack([item["X_obs"] for item in batch], axis=0)  # (B,K,D_obs)
    y_np = np.stack([item["y"] for item in batch], axis=0)      # (B,K)
    click_np = np.stack([item["click"] for item in batch], axis=0)
    qids = [item["qid"] for item in batch]
    item_ids = [item["item_ids"] for item in batch]

    return {
        "qids": qids,
        "item_ids": item_ids,
        "X": torch.from_numpy(X_np).float(),
        "X_obs": torch.from_numpy(Xobs_np).float(),
        "y": torch.from_numpy(y_np).float(),
        "click": torch.from_numpy(click_np).float(),
        "K": K,
    }

# ----------------------------
# Gradient Reversal + Observation Dropout + TwoTower Model
# ----------------------------
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradientReversalFn.apply(x, lambd)

class ObservationDropout(nn.Module):
    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        if not self.training or self.drop_rate <= 0.0:
            return x
        # elementwise dropout, scale to keep expectation
        mask = (torch.rand_like(x) > self.drop_rate).float()
        mask = mask / (1.0 - self.drop_rate)
        return x * mask

class ItemMLP(nn.Module):
    """Item-wise scorer / encoder: flatten (B*K, D) -> out (B*K, 1)"""
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

    def forward(self, x):  # x: (B*K, D)
        return self.net(x)  # (B*K,1)

class TwoTowerDebiasModel(nn.Module):
    def __init__(self, d_rel: int, d_obs: int, hidden: int,
                 gradrev_lambda: float = 0.6, obs_dropout_rate: float = 0.3):
        super().__init__()
        self.relevance_item = ItemMLP(d_rel, hidden)
        self.obs_item = ItemMLP(d_obs, max(32, hidden//2))  # smaller obs MLP
        # adv head takes observation scalar (B*K,1) -> predicts scalar relevance proxy
        self.adv_head = nn.Sequential(
            nn.Linear(1, max(32, hidden//2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(32, hidden//2), 1)
        )
        self.obs_dropout = ObservationDropout(obs_dropout_rate)
        self.gradrev_lambda = gradrev_lambda

    def forward(self, X_rel: torch.Tensor, X_obs: torch.Tensor, return_all: bool = False):
        """
        X_rel: (B,K,D_rel), X_obs: (B,K,D_obs)
        returns:
          scores_rel: (B,K)    -- relevance raw scores (no sigmoid)
          obs_raw: (B,K)      -- observation raw scores
          obs_raw_drop: (B,K) -- after dropout
          click_prob: (B,K)   -- sigmoid(scores_rel + obs_raw_drop)
        """
        B, K, D_rel = X_rel.shape
        _, K2, D_obs = X_obs.shape
        assert K == K2

        # flatten
        rel_flat = X_rel.view(B * K, D_rel)
        obs_flat = X_obs.view(B * K, D_obs)

        scores_rel_flat = self.relevance_item(rel_flat)  # (B*K,1)
        obs_raw_flat = self.obs_item(obs_flat)           # (B*K,1)

        scores_rel = scores_rel_flat.view(B, K)
        obs_raw = obs_raw_flat.view(B, K)

        # dropout
        obs_raw_drop = self.obs_dropout(obs_raw)

        click_logits = scores_rel + obs_raw_drop
        click_prob = torch.sigmoid(click_logits)

        if return_all:
            return scores_rel, obs_raw, obs_raw_drop, click_prob
        return click_prob

    def adversarial_loss(self, obs_raw: torch.Tensor, adv_label: torch.Tensor, adv_criterion):
        """
        obs_raw: (B,K)
        adv_label: (B,K) - target (e.g., detached scores_rel or click)
        adv_criterion: loss function (e.g., MSE or BCE)
        returns scalar loss (torch.Tensor)
        """
        B, K = obs_raw.shape
        x = obs_raw.view(B * K, 1)
        # apply gradient reversal
        x_rev = grad_reverse(x, self.gradrev_lambda)
        pred = self.adv_head(x_rev).view(B, K)
        loss = adv_criterion(pred, adv_label)
        return loss

# ----------------------------
# Evaluation util
# ----------------------------
def ndcg_at_k_per_query_numpy(y_true: np.ndarray, y_score: np.ndarray, k: Optional[int] = None) -> float:
    Q, K = y_true.shape
    if k is None:
        k = K
    ndcgs = []
    for i in range(Q):
        rel = y_true[i]
        scores = y_score[i]
        if K < 2:
            ndcgs.append(0.0); continue
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

def load_checkpoint(path: str, model: nn.Module, optimizer=None, scaler: Optional[GradScaler] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt and ckpt["optim_state"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optim_state"])
        except Exception:
            print("Warning: optimizer state restore failed (version mismatch?)")
    if scaler is not None and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except Exception:
            print("Warning: scaler state restore failed.")
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    return epoch, global_step

# ----------------------------
# Full training loop (mlflow integrated)
# ----------------------------
def train_two_tower_debias(train_parquet: str, val_parquet: str, config: Dict[str, Any]):
    device = config["device"]

    print("Loading train dataset...")
    train_ds = QueryItemDataset(
        parquet_path=train_parquet,
        qid_col=config["qid_col"],
        item_id_col=config["item_id_col"],
        label_col=config["label_col"],
        click_col=config.get("click_col"),
        feature_cols=config.get("feature_cols"),
        obs_feature_cols=config.get("obs_feature_cols"),
        sort_col=config.get("sort_col"),
        drop_qids_with_k_lt=None,
    )
    print("Loading val dataset...")
    val_ds = QueryItemDataset(
        parquet_path=val_parquet,
        qid_col=config["qid_col"],
        item_id_col=config["item_id_col"],
        label_col=config["label_col"],
        click_col=config.get("click_col"),
        feature_cols=config.get("feature_cols"),
        obs_feature_cols=config.get("obs_feature_cols"),
        sort_col=config.get("sort_col"),
        drop_qids_with_k_lt=None,
    )

    D_rel = train_ds.D_rel
    D_obs = train_ds.D_obs
    print(f"D_rel={D_rel} D_obs={D_obs}")

    model = TwoTowerDebiasModel(
        d_rel=D_rel,
        d_obs=D_obs,
        hidden=config["hidden_dim"],
        gradrev_lambda=config["gradrev_lambda"],
        obs_dropout_rate=config["obs_dropout_rate"],
    ).to(device)

    optimizer = RAdamScheduleFree(model.parameters(), lr=config["optimizer_lr"])
    scaler = GradScaler(enabled=config["use_amp"])

    ndcg_loss_fn = NeuralNDCGLoss(temperature=0.1)
    pointwise_loss_fn = nn.MSELoss()
    adv_loss_fn = nn.MSELoss()
    click_loss_fn = nn.BCEWithLogitsLoss()

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

    # mlflow
    import mlflow
    mlflow.set_experiment(config["mlflow_experiment"])
    with mlflow.start_run():
        # log config (primitives)
        mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))})

        # resume
        latest_ckpt = os.path.join(config["checkpoint_dir"], "latest.pt")
        start_epoch = 1
        global_step = 0
        if os.path.exists(latest_ckpt):
            print("Resuming from", latest_ckpt)
            e, gs = load_checkpoint(latest_ckpt, model, optimizer, scaler)
            start_epoch = e + 1
            global_step = gs
            print(f"Resumed at epoch {start_epoch-1}, global_step {global_step}")

        best_val_ndcg = -1.0

        for epoch in range(start_epoch, config["epochs"] + 1):
            t0 = time.time()
            sampler.set_epoch(epoch)
            model.train()
            if hasattr(optimizer, "train"):
                try:
                    optimizer.train()
                except Exception:
                    pass

            running_loss = 0.0
            batch_count = 0

            for batch_idx, batch in enumerate(loader):
                X = batch["X"].to(device)         # (B,K,D_rel)
                X_obs = batch["X_obs"].to(device) # (B,K,D_obs)
                y = batch["y"].to(device)         # (B,K)
                click = batch["click"].to(device) # (B,K)
                K = batch["K"]
                B = X.shape[0]

                with autocast(enabled=config["use_amp"]):
                    scores_rel, obs_raw, obs_raw_drop, click_prob = model.forward(X, X_obs, return_all=True)
                    # listwise on raw scores
                    if K >= config["min_k_for_listwise"]:
                        list_loss = ndcg_loss_fn(scores_rel, y)
                    else:
                        list_loss = torch.tensor(0.0, device=device)

                    point_loss = pointwise_loss_fn(scores_rel.view(-1), y.view(-1))

                    click_logits = scores_rel + obs_raw_drop  # raw logits for click
                    click_loss = click_loss_fn(click_logits.view(-1), click.view(-1))

                    # adversarial label choice
                    if config["adv_label_source"] == "scores":
                        adv_label = scores_rel.detach()
                    else:
                        adv_label = click.detach()

                    adv_loss = model.adversarial_loss(obs_raw, adv_label, adv_loss_fn)

                    total_loss = (config["lambda_listwise"] * list_loss +
                                  config["lambda_pointwise"] * point_loss +
                                  config["lambda_click"] * click_loss +
                                  config["lambda_adv"] * adv_loss)

                # grad accumulation handling
                loss_to_backprop = total_loss / float(config["grad_accum_steps"])
                scaler.scale(loss_to_backprop).backward()

                if (batch_count + 1) % config["grad_accum_steps"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                running_loss += float(total_loss.item())
                batch_count += 1

                # logging (periodic)
                if global_step % config["log_interval_steps"] == 0 and global_step > 0:
                    model.eval()
                    with torch.no_grad():
                        s_eval = model.forward(X, X_obs, return_all=True)[0].detach().cpu().numpy()
                        y_eval = y.detach().cpu().numpy()
                        ndcg10 = ndcg_at_k_per_query_numpy(y_eval, s_eval, k=min(10, K))
                        rmse = rmse_pointwise(y_eval, s_eval)
                        # click metrics quick: click logit preds
                        click_logits_eval = (s_eval + model.obs_dropout(model.obs_item(X_obs.view(B*K, -1)).view(B,K))).astype(float)
                    model.train()
                    mlflow.log_metric("train/batch_ndcg10", ndcg10, step=global_step)
                    mlflow.log_metric("train/batch_rmse", rmse, step=global_step)
                    mlflow.log_metric("train/batch_loss", running_loss / max(1, batch_count), step=global_step)
                    print(f"[E{epoch}] step {global_step} loss={running_loss / max(1,batch_count):.4f} ndcg@10={ndcg10:.4f} rmse={rmse:.4f}")

                if config.get("max_batches_per_epoch") and batch_count >= config["max_batches_per_epoch"]:
                    break

            # epoch done
            epoch_time = time.time() - t0
            avg_loss = running_loss / max(1, batch_count)
            mlflow.log_metric("train/epoch_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch} finished in {epoch_time:.1f}s avg_loss={avg_loss:.6f}")

            # optimizer.eval if exists
            if hasattr(optimizer, "eval"):
                try:
                    optimizer.eval()
                except Exception:
                    pass

            # validation
            val_res = evaluate_two_tower(model, val_ds, config)
            mlflow.log_metric("val/ndcg", val_res["ndcg"], step=epoch)
            mlflow.log_metric("val/rmse", val_res["rmse"], step=epoch)
            print(f"Validation: ndcg={val_res['ndcg']:.6f}, rmse={val_res['rmse']:.6f}")

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

            if val_res["ndcg"] > best_val_ndcg:
                best_val_ndcg = val_res["ndcg"]
                best_path = os.path.join(config["checkpoint_dir"], "best.pt")
                save_checkpoint(ckpt, best_path)
                mlflow.log_artifact(best_path)
                print(f"New best model saved (ndcg={best_val_ndcg:.6f})")

    # end mlflow run

# ----------------------------
# validation helper
# ----------------------------
def evaluate_two_tower(model: TwoTowerDebiasModel, dataset: QueryItemDataset, config: Dict[str, Any]) -> Dict[str, float]:
    device = config["device"]
    model.eval()
    buckets = defaultdict(list)
    for idx, k in enumerate(dataset.Ks):
        if config["drop_longer_than"] and k > config["drop_longer_than"]:
            continue
        buckets[int(k)].append(idx)
    sampler = BucketBatchSampler(dataset, batch_size=config["batch_Q"], shuffle=False, allow_incomplete=True, seed=0)
    sampler.buckets = buckets
    batches = list(iter(sampler))
    ndcgs = []
    rmses = []
    with torch.no_grad():
        for batch_idxs in batches:
            items = [dataset[i] for i in batch_idxs]
            batch = collate_bucket_batch(items)
            X = batch["X"].to(device); X_obs = batch["X_obs"].to(device)
            y = batch["y"].to(device)
            B = X.shape[0]; K = batch["K"]
            scores_rel, obs_raw, obs_raw_drop, click_prob = model.forward(X, X_obs, return_all=True)
            scores_np = scores_rel.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            ndcgs.append(ndcg_at_k_per_query_numpy(y_np, scores_np, k=min(10, K)))
            rmses.append(rmse_pointwise(y_np, scores_np))
    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    mean_rmse = float(np.mean(rmses)) if rmses else 0.0
    return {"ndcg": mean_ndcg, "rmse": mean_rmse}

# ----------------------------
# __main__ runner
# ----------------------------
if __name__ == "__main__":
    # Sanity checks and defaults
    train_parquet = CONFIG["train_parquet"]
    val_parquet = CONFIG["val_parquet"]
    if not os.path.exists(train_parquet):
        raise FileNotFoundError(f"train_parquet not found: {train_parquet}")
    if not os.path.exists(val_parquet):
        print("val_parquet missing, using train_parquet as val (for quick test).")
        val_parquet = train_parquet

    # If user did not specify obs_feature_cols, we already created defaults in dataset (bm25,pos)
    train_two_tower_debias(train_parquet, val_parquet, CONFIG)
```

---

## 実務上の注意（短く）

1. `click_col` が存在すればそれを使い、なければ `label>0` を proxy としますが、**可能なら実クリックデータを用意してください**。proxy は性能差が生じます。
2. `adv_label_source` を `"scores"` にすると adversarial は relevance prediction（detached scores）を抑制する方向に働きます。別の選択（`"click"`) も試してください。
3. ハイパラ感度が高い要素：`gradrev_lambda`, `obs_dropout_rate`, `lambda_adv`, `lambda_click`。mlflow のパラメータ管理で grid/検索を推奨します。
4. `RAdamScheduleFree` の API/挙動はパッケージ版によって若干差がある可能性があるので、`optimizer.train()` / `optimizer.eval()` 等がなければ safe to ignore（スクリプトは存在すれば呼びます）。
5. プロダクションでの評価はオンライン A/B（差分因果推定）や持続的ログ収集（ランダム化された小規模テスト）で検証するのが最終的には必要です。

---

もし良ければ、次にやること（選んでください）：

1. あなたの実 parquet の `df.columns` を貼ってもらえれば、`CONFIG["feature_cols"]` と `CONFIG["obs_feature_cols"]` を固定してすぐ動くスクリプトに直します。
2. mlflow を使ったハイパラ探索（簡易 grid runner）テンプレを作成します（`gradrev_lambda` / `obs_dropout_rate` / `lambda_adv` 等）。
3. DDP（複数 GPU）対応版（`DistributedSampler`、rank0 のみ mlflow／checkpoint）を出します。

どれにしますか？
