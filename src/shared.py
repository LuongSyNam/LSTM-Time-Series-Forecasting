import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

AQ_COLS = [
    "pm25", "pm10", "no2", "o3", "so2", "co",
    "aod", "dust", "uv_index", "co2",
    "aqi",
]
TIME_COLS = ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]

def get_device():
    """Lấy device ưu tiên: GPU (CUDA) > MPS (Apple Silicon) > CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"      🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        if torch.cuda.get_device_properties(0).total_memory / 1e9 < 5:
            print(f"      GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"      🚀 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print(f"      💻 Using CPU")
    return device

def build_advanced_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm time features"""
    df = df.copy()
    ts = pd.to_datetime(df["Time"], utc=True)
    
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24).astype("float32")
    df["day_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7).astype("float32")
    df["day_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12).astype("float32")
    
    df["hour_norm"] = ts.dt.hour.astype("float32") / 24.0
    df["day_norm"] = ts.dt.dayofweek.astype("float32") / 7.0
    df["month_norm"] = ts.dt.month.astype("float32") / 12.0
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype("float32")
    df["is_business_hour"] = ((ts.dt.hour >= 8) & (ts.dt.hour < 18)).astype("float32")
    
    return df

def add_lag_features(df: pd.DataFrame, target_col: str, lags: list = None, dropna: bool = True):
    """Thêm lag features cho target"""
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 72]
    
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    
    for window in [6, 12, 24]:
        df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(window).mean()
        df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(window).std()
    
    if dropna:
        df = df.dropna().reset_index(drop=True)
    return df

class ImprovedTemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 200):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5
        self.dropout = nn.Dropout(0.1)
        
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, lstm_out: torch.Tensor):
        lstm_out = lstm_out + self.pe[:, :lstm_out.size(1), :]
        
        q = self.query(lstm_out[:, -1:, :])
        k = self.key(lstm_out)
        v = self.value(lstm_out)
        
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        return torch.bmm(weights, v).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + self.dropout(x)


class ImprovedLSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 24,
        num_locations: int = 0,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.horizon = horizon
        self.use_embedding = num_locations > 0

        lstm_input = input_size + embed_dim if self.use_embedding else input_size
        
        self.input_proj = nn.Sequential(
            nn.Linear(lstm_input, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        if self.use_embedding:
            self.loc_embedding = nn.Embedding(num_locations, embed_dim)
        
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.attention = ImprovedTemporalAttention(hidden_size)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_size, dropout),
            ResidualBlock(hidden_size, dropout)
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )
        
    def forward(self, x: torch.Tensor, loc_ids: torch.Tensor | None = None):
        if self.use_embedding and loc_ids is not None:
            emb = self.loc_embedding(loc_ids)
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, emb], dim=-1)
        
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        context = self.res_blocks(context)
        out = self.fc(self.dropout(context))
        
        return out


class OverfittingDetector:
    def __init__(self, patience=10, min_delta=0.001, overfit_threshold=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.overfit_threshold = overfit_threshold
        self.best_val_loss = float('inf')
        self.counter = 0
        self.overfit_warning_count = 0
        self.history = []
        
    def check(self, train_loss, val_loss, epoch):
        should_stop = False
        is_overfitting = False
        message = ""
        
        self.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'gap': val_loss - train_loss
        })
        
        gap = val_loss - train_loss
        
        if gap > self.overfit_threshold:
            is_overfitting = True
            self.overfit_warning_count += 1
            message = f"⚠️ Overfitting detected! (gap={gap:.4f} > {self.overfit_threshold})"
            
            if self.overfit_warning_count >= 3:
                message += f" | Stopping after {self.overfit_warning_count} warnings"
                should_stop = True
        else:
            if gap < self.overfit_threshold * 0.5:
                self.overfit_warning_count = max(0, self.overfit_warning_count - 1)
        
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                message += f" | Early stopping (no improvement for {self.patience} epochs)"
                should_stop = True
        
        return should_stop, is_overfitting, message


def make_windows(X: np.ndarray, target_idx: int, lookback: int, horizon: int):
    windows, targets = [], []
    n_samples = len(X) - lookback - horizon + 1
    
    for i in range(n_samples):
        windows.append(X[i : i + lookback])
        targets.append(X[i + lookback : i + lookback + horizon, target_idx])
    
    return np.array(windows, dtype="float32"), np.array(targets, dtype="float32")


def build_last_window(df: pd.DataFrame, scaler: StandardScaler, lookback: int, feat_cols: list, target_idx: int) -> np.ndarray:
    """Chuẩn bị dữ liệu đầu vào cho mô hình"""
    if len(df) < lookback:
        raise ValueError(f"Dữ liệu không đủ lookback={lookback} (chỉ có {len(df)})")
    
    last_chunk = df[feat_cols].values[-lookback:].copy()
    scaled_chunk = scaler.transform(last_chunk.astype("float32"))
    
    return np.expand_dims(scaled_chunk, axis=0)


def get_feature_columns(df: pd.DataFrame, target_col: str):
    """Lấy danh sách feature columns"""
    base_features = [c for c in AQ_COLS + TIME_COLS if c in df.columns]
    lag_features = [c for c in df.columns if 'lag_' in c or 'rolling_' in c]
    feat_cols = base_features + lag_features
    target_idx = feat_cols.index(target_col) if target_col in feat_cols else -1
    return feat_cols, target_idx