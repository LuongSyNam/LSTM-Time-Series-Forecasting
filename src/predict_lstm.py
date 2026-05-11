import argparse, os, json, pickle
import pandas as pd, numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

AQ_COLS = [
    "pm25", "pm10", "no2", "o3", "so2", "co",
    "aod", "dust", "uv_index", "co2",
    "aqi",
]
TIME_COLS = ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]
EXTRA_TIME_COLS = ["hour_norm", "day_norm", "month_norm", "is_weekend", "is_business_hour"]

def get_device():
    """Lấy device ưu tiên: GPU (CUDA) > MPS (Apple Silicon) > CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"      🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"      🚀 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print(f"      💻 Using CPU")
    return device

def add_lag_features_for_inference(df: pd.DataFrame, target_col: str, lags: list = None):
    """Thêm lag features cho inference"""
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 72]
    
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    
    for window in [6, 12, 24]:
        df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(window).mean()
        df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(window).std()
    
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

def build_last_window(
    df: pd.DataFrame,
    scaler: StandardScaler,
    lookback: int,
    feat_cols: list,
    target_idx: int
) -> np.ndarray:
    """Chuẩn bị dữ liệu đầu vào cho mô hình"""
    if len(df) < lookback:
        raise ValueError(f"Dữ liệu không đủ lookback={lookback} (chỉ có {len(df)})")
    
    last_chunk = df[feat_cols].values[-lookback:].copy()
    scaled_chunk = scaler.transform(last_chunk.astype("float32"))
    
    return np.expand_dims(scaled_chunk, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_lstm.pt", help="Path to checkpoint (relative to outdir)")
    ap.add_argument("--input", default="data/air_quality.csv")
    ap.add_argument("--location", default="khanhhoa_nhatrang")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--value_column", type=str, default="aqi")
    
    args = ap.parse_args()

    device = get_device()

    # ── [1/4] Loading checkpoint ──────────────────────────────────
    # SỬA: Tìm checkpoint trong outdir
    ckpt_path = os.path.join(args.outdir, args.ckpt)
    if not os.path.exists(ckpt_path):
        # Thử tìm best_lstm.pt nếu không thấy
        alt_path = os.path.join(args.outdir, "best_lstm.pt")
        if os.path.exists(alt_path):
            ckpt_path = alt_path
        else:
            print(f"[ERROR] Checkpoint không tồn tại: {ckpt_path}")
            return
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    lookback = ckpt["lookback"]
    horizon = ckpt["horizon"]
    feat_cols = ckpt["feat_cols"]
    target_idx = ckpt["target_idx"]
    num_locations = ckpt["num_locations"]
    hidden_size = ckpt["hidden_size"]
    num_layers = ckpt["num_layers"]
    embed_dim = ckpt.get("embed_dim", 16)

    print(f"[1/4] Loading checkpoint: {ckpt_path}")
    print(f"      lookback={lookback}  horizon={horizon}")
    print(f"      features={len(feat_cols)}  target_idx={target_idx}")

    # Khởi tạo model
    model = ImprovedLSTMForecaster(
        input_size=len(feat_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        horizon=horizon,
        num_locations=num_locations,
        embed_dim=embed_dim,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    
    print(f"      Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"      Device: {device}")

    # ── [2/4] Loading data & Feature Engineering ───────────────────
    print(f"[2/4] Loading data: {args.input}")
    df = pd.read_csv(args.input, parse_dates=["Time"])
    df = build_advanced_time_features(df)
    
    print(f"      Adding lag features for target '{args.value_column}'...")
    df = add_lag_features_for_inference(df, args.value_column)
    
    if args.location:
        all_locs = [args.location]
    else:
        all_locs = sorted(df["location_key"].unique())
    
    print(f"      Locations: {all_locs}")

    # ── [3/4] Loading scaler(s) ───────────────────────────────────
    print("[3/4] Loading scaler(s)...")
    single_location = num_locations == 0
    
    if single_location:
        scaler_path = os.path.join(args.outdir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            print(f"[ERROR] Không tìm thấy scaler: {scaler_path}")
            return
        
        with open(scaler_path, "rb") as f:
            bundle = pickle.load(f)
        
        scaler = bundle["scaler"]
        scalers_map = {all_locs[0]: scaler}
        print(f"      Loaded scaler for single location")
        
    else:
        scaler_path = os.path.join(args.outdir, "scalers.pkl")
        if not os.path.exists(scaler_path):
            print(f"[ERROR] Không tìm thấy scalers: {scaler_path}")
            return
        
        with open(scaler_path, "rb") as f:
            bundle = pickle.load(f)
        
        scalers_map = bundle["scalers"]
        loc2idx = bundle["loc2idx"]
        print(f"      Loaded scalers for {len(scalers_map)} locations")

    # ── [4/4] Direct forecast ─────────────────────────────────────
    print(f"[4/4] Direct forecast ({horizon} bước)...")
    all_preds = []

    for loc in all_locs:
        sub = df[df["location_key"] == loc].copy()
        
        missing_feats = [f for f in feat_cols if f not in sub.columns]
        if missing_feats:
            print(f"  [Skip] {loc}: thiếu {len(missing_feats)} features: {missing_feats[:5]}")
            continue
        
        if len(sub) < lookback:
            print(f"  [Skip] {loc}: chỉ có {len(sub)} rows, cần {lookback}")
            continue
        
        sub = sub.sort_values("Time").reset_index(drop=True)
        
        sc = scalers_map.get(loc)
        if sc is None:
            print(f"  [Skip] {loc}: không tìm thấy scaler")
            continue
        
        try:
            window = build_last_window(sub, sc, lookback, feat_cols, target_idx)
            X_t = torch.tensor(window, dtype=torch.float32).to(device)
            
            l_t = None
            if not single_location:
                if loc not in loc2idx:
                    print(f"  [Skip] {loc}: không có trong loc2idx")
                    continue
                l_t = torch.tensor([loc2idx[loc]]).to(device)
            
            with torch.no_grad():
                pred_scaled = model(X_t, l_t).cpu().numpy()[0]
            
            n_features = len(feat_cols)
            dummy = np.zeros((horizon, n_features), dtype="float32")
            dummy[:, target_idx] = pred_scaled
            
            pred_final = sc.inverse_transform(dummy)[:, target_idx]
            
            last_ts = sub["Time"].max()
            for i in range(horizon):
                future_ts = last_ts + pd.Timedelta(hours=i+1)
                all_preds.append({
                    "location": loc,
                    "forecast_date": future_ts,
                    "hour_ahead": i + 1,
                    "predicted_value": float(pred_final[i])
                })
            
            print(f"  [OK] {loc}: predicted {horizon} steps")
            
        except Exception as e:
            print(f"  [Error] {loc}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if all_preds:
        out_path = os.path.join(args.outdir, args.out)
        out_df = pd.DataFrame(all_preds)
        out_df.to_csv(out_path, index=False)
        print(f"\n[OK] Saved {len(out_df)} predictions to {out_path}")
        print("\n📊 24-hour forecast:")
        for i, pred in enumerate(all_preds[:24]):
            print(f"  Hour +{pred['hour_ahead']:2d}: {pred['predicted_value']:.2f}")
    else:
        print("[ERROR] No predictions generated!")

if __name__ == "__main__":
    main()