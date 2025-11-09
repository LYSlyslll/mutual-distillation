"""
mutual_distillation_model.py

可执行示例：实现双层 GRU + 四层 CNN 互蒸馏模块 + 全连接预测头
输入: (batch, t, n_features)
输出: (batch, 1) 价格变化率回归预测

使用方法:
    python mutual_distillation_model.py
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# --------------------
# 配置 / 超参数（可按需调整）
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 超参数示例
CONFIG = {
    "t": 20,                 # 窗口长度（时间步数）
    "n_features": 16,        # 每步特征维度（n）
    "gru_hidden": 64,        # GRU 隐藏单元
    "gru_layers": 2,         # GRU 层数
    "gru_bidirectional": False,
    "cnn_channels": [32, 64, 64, 128],  # 四层 CNN 输出通道数
    "cnn_kernel_sizes": [5, 5, 3, 3],   # 四层卷积核大小
    "pred_hidden": [128, 64],  # 预测机全连接层结构
    "dropout": 0.2,
    "lr": 1e-3,
    "batch_size": 256,
    "num_epochs": 5,
    "lambda_distill": 0.5,    # 互蒸馏权重 lambda
    "alternating": False,     # 是否使用交替训练（True/False）
    "save_dir": "./checkpoints",
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# --------------------
# 简单 Dataset（示例用伪数据）
# --------------------
class DummyTimeSeriesDataset(Dataset):
    def __init__(self, n_samples, t, n_features):
        self.n_samples = n_samples
        self.t = t
        self.n_features = n_features
        # 造一些随机数据：特征和目标（回归）
        self.X = np.random.normal(size=(n_samples, t, n_features)).astype(np.float32)
        # 目标：用一些可学的函数生成（例如每样本的最后时间步特征加噪）
        self.y = (self.X.mean(axis=(1,2)) + 0.1 * np.random.randn(n_samples)).astype(np.float32)
        self.y = self.y.reshape(-1,1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------
# 模型组件
# --------------------
class GRULearner(nn.Module):
    """双层 GRU 学习器，输出一个因子向量"""
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, bidirectional=False, dropout=0.0, pool="last"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=n_layers,
                          batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if n_layers>1 else 0.0)
        self.pool = pool
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        # x: (batch, t, input_dim)
        out, h_n = self.gru(x)  # out: (batch, t, hidden*direc)
        if self.pool == "last":
            # use last time step
            # out[:, -1, :] is fine when not packed
            feat = out[:, -1, :]
        elif self.pool == "mean":
            feat = out.mean(dim=1)
        elif self.pool == "max":
            feat, _ = out.max(dim=1)
        else:
            feat = out[:, -1, :]
        return feat  # (batch, out_dim)


class CNNCorr(nn.Module):
    """四层 1D CNN 校正器，输出一个因子向量"""
    def __init__(self, input_dim, channels=[32,64,64,128], kernel_sizes=[5,5,3,3], dropout=0.0, pool="avg"):
        super().__init__()
        layers = []
        in_ch = input_dim  # we will treat features as channels by transposing
        for i, (out_ch, k) in enumerate(zip(channels, kernel_sizes)):
            conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, padding=k//2)
            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_ch))
            # optionally downsample a bit (这里不强制池化以保留时序)
            if dropout and dropout>0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.pool = pool
        self.out_dim = channels[-1]

    def forward(self, x):
        # x: (batch, t, n_features) -> 转为 (batch, n_features, t)
        x = x.permute(0, 2, 1)
        out = self.encoder(x)  # (batch, out_ch, t)
        if self.pool == "avg":
            feat = out.mean(dim=2)
        elif self.pool == "max":
            feat, _ = out.max(dim=2)
        else:
            feat = out.mean(dim=2)
        return feat  # (batch, out_dim)


class Predictor(nn.Module):
    """简单 MLP 预测机（回归）"""
    def __init__(self, input_dim, hidden_dims=[128,64], dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # 回归输出
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)  # (batch,1)


class MutualDistillationModel(nn.Module):
    """
    包含 GRU 学习器、CNN 校正器、各自的预测头，并在特征级进行互蒸馏
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gru = GRULearner(input_dim=cfg["n_features"],
                              hidden_dim=cfg["gru_hidden"],
                              n_layers=cfg["gru_layers"],
                              bidirectional=cfg["gru_bidirectional"],
                              dropout=cfg["dropout"],
                              pool="last")
        self.cnn = CNNCorr(input_dim=cfg["n_features"],
                           channels=cfg["cnn_channels"],
                           kernel_sizes=cfg["cnn_kernel_sizes"],
                           dropout=cfg["dropout"],
                           pool="avg")
        # 如果两者特征维不同，先投影到相同维度便于计算蒸馏损失和融合
        feat_dim = max(self.gru.out_dim, self.cnn.out_dim)
        self.project_gru = nn.Linear(self.gru.out_dim, feat_dim) if self.gru.out_dim != feat_dim else nn.Identity()
        self.project_cnn = nn.Linear(self.cnn.out_dim, feat_dim) if self.cnn.out_dim != feat_dim else nn.Identity()

        # 各自的预测头（也可以共享同一个预测头，但此处分别建立）
        self.pred_gru = Predictor(input_dim=feat_dim, hidden_dims=cfg["pred_hidden"], dropout=cfg["dropout"])
        self.pred_cnn = Predictor(input_dim=feat_dim, hidden_dims=cfg["pred_hidden"], dropout=cfg["dropout"])

        # 融合后的预测头（可选）：把两个因子连接起来用于最终预测
        self.pred_fuse = Predictor(input_dim=feat_dim*2, hidden_dims=cfg["pred_hidden"], dropout=cfg["dropout"])

    def forward(self, x):
        """
        返回字典:
            feat_gru_proj: (batch, feat_dim)
            feat_cnn_proj: (batch, feat_dim)
            y_gru: (batch,1)
            y_cnn: (batch,1)
            y_fuse: (batch,1)
        """
        feat_gru = self.gru(x)
        feat_cnn = self.cnn(x)

        feat_gru_p = self.project_gru(feat_gru)
        feat_cnn_p = self.project_cnn(feat_cnn)

        y_gru = self.pred_gru(feat_gru_p)
        y_cnn = self.pred_cnn(feat_cnn_p)

        fused = torch.cat([feat_gru_p, feat_cnn_p], dim=1)
        y_fuse = self.pred_fuse(fused)

        return {
            "feat_gru": feat_gru_p,
            "feat_cnn": feat_cnn_p,
            "y_gru": y_gru,
            "y_cnn": y_cnn,
            "y_fuse": y_fuse
        }


# --------------------
# 损失与训练逻辑
# --------------------
def mse_loss(a, b):
    return torch.mean((a - b) ** 2)


def train_one_epoch(model, dataloader, optimizer_gru, optimizer_cnn, cfg, device, epoch, alternating=False):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    lambda_d = cfg["lambda_distill"]
    for X_batch, y_batch in pbar:
        X = X_batch.to(device)  # (batch, t, n)
        y = y_batch.to(device)  # (batch,1)

        out = model(X)
        feat_g = out["feat_gru"]
        feat_c = out["feat_cnn"]
        y_g = out["y_gru"]
        y_c = out["y_cnn"]
        y_f = out["y_fuse"]

        # 预测损失（各自与融合一起都可计算；这里以fusion为主评估，但也保留个体损失）
        loss_pred_gru = mse_loss(y_g, y)
        loss_pred_cnn = mse_loss(y_c, y)
        loss_pred_fuse = mse_loss(y_f, y)

        # 互蒸馏（特征级别 MSE）
        dist_loss = mse_loss(feat_g, feat_c)

        # 总损失示例（joint update）
        loss_gru = loss_pred_gru + lambda_d * dist_loss
        loss_cnn = loss_pred_cnn + lambda_d * dist_loss
        # optionally also include fusion loss
        loss_fuse = loss_pred_fuse

        if not alternating:
            # 联合更新：把所有参数一起优化（通过单个 optimizer 来管理）
            # 我们假设 optimizer_gru == optimizer_cnn == a joint optimizer
            assert optimizer_gru is optimizer_cnn, "For joint update pass same optimizer twice"
            optimizer = optimizer_gru
            optimizer.zero_grad()
            # 我们可以合并损失（也可按论文交替更新）
            loss = loss_gru + loss_cnn + loss_fuse
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (len(dataloader)* (1))})
        else:
            # 交替更新：先更新 GRU（固定 CNN），再更新 CNN（固定 GRU）
            # 更新 GRU 分支
            optimizer_gru.zero_grad()
            loss_gru.backward(retain_graph=True)
            optimizer_gru.step()

            # 更新 CNN 分支
            optimizer_cnn.zero_grad()
            loss_cnn.backward()
            optimizer_cnn.step()

            # 可以单独优化 fusion head 用于快速收敛（可选）
            # 此处不单独优化 fusion head

            total_loss += (loss_gru.item() + loss_cnn.item() + loss_fuse.item())
            pbar.set_postfix({"loss": total_loss / (len(dataloader)*2)})
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_mse = 0.0
    n = 0
    for X_batch, y_batch in dataloader:
        X = X_batch.to(device)
        y = y_batch.to(device)
        out = model(X)
        # 以融合预测为最终输出
        y_pred = out["y_fuse"]
        total_mse += ((y_pred - y)**2).sum().item()
        n += y.shape[0]
    rmse = math.sqrt(total_mse / n)
    return rmse


# --------------------
# 主执行：构建模型、数据、训练并演示推理
# --------------------
def main():
    cfg = CONFIG
    device = DEVICE
    print("Device:", device)

    # 数据集（示例）
    train_ds = DummyTimeSeriesDataset(n_samples=5000, t=cfg["t"], n_features=cfg["n_features"])
    val_ds = DummyTimeSeriesDataset(n_samples=1000, t=cfg["t"], n_features=cfg["n_features"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = MutualDistillationModel(cfg).to(device)

    # 优化器配置：如果不交替更新，可以用单个 optimizer
    if not cfg["alternating"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        optimizer_gru = optimizer_cnn = optimizer
    else:
        # 交替更新：分别为 GRU/CNN & 各自头创建优化器（这里只做演示，可按需细分参数）
        params_gru = list(model.gru.parameters()) + list(model.project_gru.parameters()) + list(model.pred_gru.parameters())
        params_cnn = list(model.cnn.parameters()) + list(model.project_cnn.parameters()) + list(model.pred_cnn.parameters())
        # fusion head 单独优化器
        params_fuse = list(model.pred_fuse.parameters())

        optimizer_gru = torch.optim.Adam(params_gru, lr=cfg["lr"])
        optimizer_cnn = torch.optim.Adam(params_cnn, lr=cfg["lr"])
        optimizer_fuse = torch.optim.Adam(params_fuse, lr=cfg["lr"])
        # we won't use optimizer_fuse in the simple alternation loop, but could

    best_rmse = float("inf")
    for epoch in range(1, cfg["num_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer_gru, optimizer_cnn, cfg, device, epoch, alternating=cfg["alternating"])
        val_rmse = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | Train Loss(avg per batch): {train_loss:.6f} | Val RMSE: {val_rmse:.6f}")

        # 保存模型
        ckpt_path = os.path.join(cfg["save_dir"], f"md_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "cfg": cfg
        }, ckpt_path)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "best_model.pt"))
            print("Saved best model.")

    # 演示推理
    model.eval()
    sample = torch.from_numpy(np.random.normal(size=(4, cfg["t"], cfg["n_features"])).astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(sample)
    print("Demo outputs (fusion predictions):", out["y_fuse"].cpu().numpy().flatten())

if __name__ == "__main__":
    main()
