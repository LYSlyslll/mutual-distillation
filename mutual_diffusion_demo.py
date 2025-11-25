"""
mutual_distillation_model.py

可执行示例：实现双层 GRU + 四层 CNN 互蒸馏模块 + 全连接预测头
输入: (batch, t, n_features)
输出: (batch, 1) 价格变化率回归预测

使用方法:
    python mutual_distillation_model.py
"""

import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import baostock as bs

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
    "n_features": 10,        # 每步特征维度（n），会在数据集加载后进行对齐
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
    "hs300_max_symbols": 30,
    "hs300_start_date": None,
    "hs300_end_date": None,
    "hs300_cache_dir": "./data",
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# --------------------
# 沪深300成分股数据集
# --------------------
def _format_stock_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    if symbol.startswith("6"):
        return f"sh{symbol}"
    if symbol.startswith(("0", "3")):
        return f"sz{symbol}"
    return symbol


def _default_date_range(years: int = 3):
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    # Baostock 必须使用 YYYY-MM-DD 格式
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

# --- 新增：Baostock 专用格式化函数 ---
def _format_baostock_symbol(symbol: str) -> str:
    """
    将股票代码强制转换为 Baostock 需要的 'sh.600000' 格式
    """
    # 1. 清理空格并转小写
    symbol = symbol.strip().lower()
    
    # 2. 如果已经是 sh. 或 sz. 开头且长度为9，直接返回
    if (symbol.startswith("sh.") or symbol.startswith("sz.")) and len(symbol) == 9:
        return symbol
        
    # 3. 如果是 akshare 格式 (sh600000)，去掉前缀重新加
    if symbol.startswith("sh") and "." not in symbol:
        return "sh." + symbol[2:]
    if symbol.startswith("sz") and "." not in symbol:
        return "sz." + symbol[2:]
        
    # 4. 如果是纯数字 (600000)
    if symbol.isdigit():
        if symbol.startswith("6"):
            return f"sh.{symbol}"
        elif symbol.startswith(("0", "3")):
            return f"sz.{symbol}"
            
    # 5. 其他情况尝试直接返回，或根据需求抛错
    return symbol

class HS300TimeSeriesDataset(Dataset):
    """
    从开源金融数据接口（Baostock）抓取沪深300指数成分股日线数据，
    构造时间序列监督学习数据集。
    """

    FEATURE_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "return",
        "volume_change",
        "volatility",
        "high_low_range",
    ]

    def __init__(
        self,
        window_size: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_symbols: int = 20,
        cache_dir: str | os.PathLike[str] = "./data",
    ):
        if start_date is None or end_date is None:
            default_start, default_end = _default_date_range(years=3)
            start_date = start_date or default_start
            end_date = end_date or default_end

        self.window_size = window_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples = []
        self.targets = []

        data_frames = self._load_data(start_date, end_date, max_symbols)
        self._build_samples(data_frames)

        if not self.samples:
            raise RuntimeError("HS300 数据集中没有生成任何样本，请检查时间范围或网络连接。")

        self.n_features = len(self.FEATURE_COLUMNS)

    
    def _load_data(self, start_date: str, end_date: str, max_symbols: int):
        # --- 修复日期格式 (兼容 YYYYMMDD 和 YYYY-MM-DD) ---
        def fix_date(d):
            d = d.replace("-", "").replace("/", "") # 先去掉可能存在的符号
            if len(d) == 8:
                return f"{d[:4]}-{d[4:6]}-{d[6:]}" # 强制转为 YYYY-MM-DD
            return d # 如果格式很怪就不动了，让报错暴露问题

        start_date = fix_date(start_date)
        end_date = fix_date(end_date)
        # ----------------------------------------------------

        lg = bs.login()
        if lg.error_code != '0':
            raise RuntimeError(f"Baostock login failed: {lg.error_msg}")

        print(f"Baostock login success! Date range: {start_date} to {end_date}")
        
        cache_file = self.cache_dir / f"hs300_baostock_{start_date}_{end_date}_{max_symbols}.pkl"
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            bs.logout()
            return pd.read_pickle(cache_file)

        rs = bs.query_hs300_stocks()
        if rs.error_code != '0':
            raise RuntimeError("获取沪深300成分股失败")

        hs300_codes = []
        while rs.next():
            row = rs.get_row_data()
            hs300_codes.append(row[1])

        print(f"获取到 {len(hs300_codes)} 只成分股，准备下载前 {max_symbols} 只...")
        
        records = []
        for raw_code in hs300_codes[:max_symbols]:
            code = _format_baostock_symbol(raw_code)
            
            rs_daily = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,amount,pctChg,turn",
                start_date=start_date, 
                end_date=end_date,
                frequency="d", 
                adjustflag="2"
            )

            # --- 修复 NoneType 报错 ---
            if rs_daily is None or rs_daily.error_code != '0':
                # 如果 rs_daily 是 None，说明参数严重错误（如日期格式）
                msg = rs_daily.error_msg if rs_daily is not None else "Baostock returned None (Check date format)"
                print(f"获取 {code} 失败: {msg}")
                continue
            # ------------------------

            data_list = []
            while rs_daily.next():
                data_list.append(rs_daily.get_row_data())

            if not data_list:
                continue

            df = pd.DataFrame(data_list, columns=rs_daily.fields)
            
            # 类型转换
            numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pctChg", "turn"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            rename_map = {
                "amount": "turnover",
                "pctChg": "pct_change",
                "turn": "turnover_rate"
            }
            df = df.rename(columns=rename_map)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            
            df["return"] = df["close"].pct_change().fillna(0.0)
            df["volume_change"] = df["volume"].pct_change().fillna(0.0)
            df["volatility"] = df["return"].rolling(window=5, min_periods=1).std().fillna(0.0)
            df["high_low_range"] = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).fillna(0.0)
            df["target"] = df["close"].pct_change().shift(-1)
            df["symbol"] = code

            feature_df = df[self.FEATURE_COLUMNS + ["target", "symbol"]].dropna(subset=["target"])
            
            if not feature_df.empty:
                records.append(feature_df)

        bs.logout()

        if not records:
            raise RuntimeError("未能下载任何数据，请检查日期范围是否为非交易日，或网络问题。")

        combined = pd.concat(records, axis=0)
        combined.to_pickle(cache_file)
        return combined

    def _build_samples(self, combined_df: pd.DataFrame):
        grouped = combined_df.groupby("symbol")
        for _, df in grouped:
            values = df[self.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            targets = df["target"].to_numpy(dtype=np.float32)
            if len(values) <= self.window_size:
                continue
            for idx in range(self.window_size, len(values)):
                window = values[idx - self.window_size : idx]
                target = targets[idx]
                if np.isnan(window).any() or np.isnan(target):
                    continue
                self.samples.append(window.copy())
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = np.array([self.targets[idx]], dtype=np.float32)
        return x.astype(np.float32), y


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

    dataset = HS300TimeSeriesDataset(
        window_size=cfg["t"],
        start_date=cfg["hs300_start_date"],
        end_date=cfg["hs300_end_date"],
        max_symbols=cfg["hs300_max_symbols"],
        cache_dir=cfg["hs300_cache_dir"],
    )

    # 根据真实数据的特征维度更新配置
    cfg["n_features"] = dataset.n_features

    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    if n_train == 0 or n_val == 0:
        raise RuntimeError("数据量不足以划分训练/验证集，请扩大时间范围或放宽筛选条件。")

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

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

    # 演示推理：取验证集中一小批样本
    model.eval()
    try:
        sample_batch, _ = next(iter(val_loader))
    except StopIteration:
        sample_batch = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    if sample_batch.shape[0] > 4:
        sample_batch = sample_batch[:4]
    with torch.no_grad():
        out = model(sample_batch)
    print("Demo outputs (fusion predictions):", out["y_fuse"].cpu().numpy().flatten())

if __name__ == "__main__":
    main()

