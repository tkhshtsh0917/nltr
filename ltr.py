import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from nltr.losses import NeuralNDCGLoss
    return NeuralNDCGLoss, nn, optim, torch


@app.cell
def _(nn, torch):
    class RankNet(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
            super(RankNet, self).__init__()

            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
            return out
    return (RankNet,)


@app.cell
def _():
    # モデル、データ、ロス、オプティマイザの準備
    input_size = 10  # 入力特徴量の数
    hidden_size = 50  # 隠れ層のユニット数
    output_size = 5  # 出力スコアの数（ランキングのスレートサイズ）
    return hidden_size, input_size, output_size


@app.cell
def _(NeuralNDCGLoss, RankNet, hidden_size, input_size, optim, output_size):
    model = RankNet(input_size, hidden_size, output_size).to('mps')
    criterion = NeuralNDCGLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return criterion, model, optimizer


@app.cell
def _(input_size, output_size, torch):
    # ダミーデータの作成
    batch_size = 16
    X = torch.randn(batch_size, input_size)  # ランダムな入力データ
    y_true = torch.randint(0, 5, (batch_size, output_size)).float()  # ランダムなターゲットランキング
    return X, y_true


@app.cell
def _(X, model):
    y_init = model(X.to('mps'))
    y_init
    return


@app.cell
def _(X, criterion, model, optimizer, y_true):
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()

        # 順伝播
        optimizer.zero_grad()
        y_pred = model(X.to('mps'))

        # Neural NDCG Lossの計算
        loss = criterion(y_pred.to('mps'), y_true.to('mps'))

        # 逆伝播と最適化
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    return


@app.cell
def _(X, model):
    y_test = model(X.to('mps'))
    y_test
    return


@app.cell
def _(y_true):
    y_true
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
