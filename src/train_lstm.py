import argparse, os, json, pickle, time
import pandas as pd, numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import rmse, mae, mse, mape, R2

AQ_COLS = [
    "pm25", "pm10", "no2", "o3", "so2", "co",
    "aod", "dust", "uv_index", "co2",
    "aqi",
]
TIME_COLS = ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]
DEFAULT_HORIZON = 24

# ==================== CLASS PHÁT HIỆN OVERFITTING ====================
class OverfittingDetector:
    """Phát hiện và xử lý overfitting tự động"""
    def __init__(self, patience=10, min_delta=0.001, overfit_threshold=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.overfit_threshold = overfit_threshold
        self.best_val_loss = float('inf')
        self.counter = 0
        self.overfit_warning_count = 0
        self.history = []
        
    def check(self, train_loss, val_loss, epoch):
        """
        Kiểm tra overfitting
        Returns: (should_stop, is_overfitting, message)
        """
        should_stop = False
        is_overfitting = False
        message = ""
        
        # Lưu lịch sử
        self.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'gap': val_loss - train_loss
        })
        
        gap = val_loss - train_loss
        
        # 1. Phát hiện overfitting (val loss cao hơn train loss quá nhiều)
        if gap > self.overfit_threshold:
            is_overfitting = True
            self.overfit_warning_count += 1
            message = f"⚠️ Overfitting detected! (gap={gap:.4f} > {self.overfit_threshold})"
            
            # Nếu overfitting kéo dài quá 3 epochs thì dừng
            if self.overfit_warning_count >= 3:
                message += f" | Stopping after {self.overfit_warning_count} warnings"
                should_stop = True
        else:
            # Reset warning count nếu hết overfitting
            if gap < self.overfit_threshold * 0.5:
                self.overfit_warning_count = max(0, self.overfit_warning_count - 1)
        
        # 2. Early stopping dựa trên validation loss
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                message += f" | Early stopping (no improvement for {self.patience} epochs)"
                should_stop = True
        
        return should_stop, is_overfitting, message

# ==================== HÀM KIỂM TRA GPU ====================
def get_device():
    """Lấy device ưu tiên: GPU (CUDA) > MPS (Apple Silicon) > CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"      🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"      GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"      🚀 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print(f"      💻 Using CPU")
    return device

def build_advanced_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm nhiều time features hơn"""
    df = df.copy()
    ts = pd.to_datetime(df["Time"], utc=True)
    
    # Basic cyclic features
    df["hour_sin"]  = np.sin(2 * np.pi * ts.dt.hour / 24).astype("float32")
    df["hour_cos"]  = np.cos(2 * np.pi * ts.dt.hour / 24).astype("float32")
    df["day_sin"]   = np.sin(2 * np.pi * ts.dt.dayofweek / 7).astype("float32")
    df["day_cos"]   = np.cos(2 * np.pi * ts.dt.dayofweek / 7).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12).astype("float32")
    
    # ADD: Additional time features
    df["hour_norm"] = ts.dt.hour.astype("float32") / 24.0
    df["day_norm"] = ts.dt.dayofweek.astype("float32") / 7.0
    df["month_norm"] = ts.dt.month.astype("float32") / 12.0
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype("float32")
    df["is_business_hour"] = ((ts.dt.hour >= 8) & (ts.dt.hour < 18)).astype("float32")
    
    return df

def add_lag_features(df: pd.DataFrame, target_col: str, lags: list = None):
    """Thêm lag features cho target"""
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 72]
    
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [6, 12, 24]:
        df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(window).mean()
        df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(window).std()
    
    return df

class ImprovedTemporalAttention(nn.Module):
    """Improved attention (CPU-optimized)"""
    def __init__(self, hidden_size: int, max_len: int = 200):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5
        self.dropout = nn.Dropout(0.1)
        
        # Simplified positional encoding
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
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = DEFAULT_HORIZON,
        num_locations: int = 0,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.horizon = horizon
        self.use_embedding = num_locations > 0

        lstm_input = input_size + embed_dim if self.use_embedding else input_size
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(lstm_input, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        if self.use_embedding:
            self.loc_embedding = nn.Embedding(num_locations, embed_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.attention = ImprovedTemporalAttention(hidden_size)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_size, dropout),
            ResidualBlock(hidden_size, dropout)        
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
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

def make_windows(
    X: np.ndarray,
    target_idx: int,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows, targets = [], []
    n_samples = len(X) - lookback - horizon + 1
    
    for i in range(n_samples):
        windows.append(X[i : i + lookback])
        targets.append(X[i + lookback : i + lookback + horizon, target_idx])
    
    return np.array(windows, dtype="float32"), np.array(targets, dtype="float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/air_quality.csv")
    ap.add_argument("--lookback", type=int, default=72)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--value_column", type=str, default="aqi")
    ap.add_argument("--location", type=str, default=None)
    ap.add_argument("--locations", type=str, nargs="+", default=None)
    ap.add_argument("--embed_dim", type=int, default=16)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--loss", type=str, default="huber")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    # ==================== THÊM THAM SỐ CHO OVERFITTING ====================
    ap.add_argument("--early_stop_patience", type=int, default=10,
                    help="Số epoch chờ nếu val_loss không cải thiện")
    ap.add_argument("--overfit_threshold", type=float, default=0.05,
                    help="Ngưỡng gap (val_loss - train_loss) để phát hiện overfitting")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ==================== KIỂM TRA DEVICE ====================
    device = get_device()
    
    # Load & preprocess
    print("[1/5] Loading data...")
    df = pd.read_csv(args.input, parse_dates=["Time"])
    df = build_advanced_time_features(df)
    df = add_lag_features(df, args.value_column)
    
    # Update feature columns
    base_features = [c for c in AQ_COLS + TIME_COLS if c in df.columns]
    lag_features = [c for c in df.columns if 'lag_' in c or 'rolling_' in c]
    feat_cols = base_features + lag_features
    
    n_features = len(feat_cols)
    print(f"      Features: {n_features} total")
    print(f"      - Base: {len(base_features)}")
    print(f"      - Lag/Rolling: {len(lag_features)}")
    print(f"      Target: '{args.value_column}'")
    print(f"      Horizon: {args.horizon} | Lookback: {args.lookback}")

    # Data processing
    if args.location is not None:
        sub = df[df["location_key"] == args.location].copy()
        if sub.empty:
            print(f"[ERROR] Location not found: {args.location}")
            return
        
        sub = sub.sort_values("Time").reset_index(drop=True)
        sub = sub.dropna()
        
        # Scale ALL features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sub[feat_cols].values.astype("float32"))
        
        target_idx = feat_cols.index(args.value_column)
        
        with open(os.path.join(args.outdir, "scaler.pkl"), "wb") as f:
            pickle.dump({"scaler": scaler, "feat_cols": feat_cols, "target_idx": target_idx}, f)
        
        X_win, y_win = make_windows(X_scaled, target_idx, args.lookback, args.horizon)
        loc_ids_arr = None
        num_locations = 0
        
        print(f"      Location: {args.location} | Rows: {len(sub)}")
        
    elif args.locations is not None:
        # Multi-location
        df = df[df["location_key"].isin(args.locations)].copy()
        all_locs = sorted(args.locations)
        loc2idx = {loc: i for i, loc in enumerate(all_locs)}
        num_locations = len(all_locs)
        
        X_parts, y_parts, lid_parts = [], [], []
        scalers = {}
        
        for loc in all_locs:
            grp = df[df["location_key"] == loc].sort_values("Time").dropna()
            if len(grp) < args.lookback + args.horizon:
                print(f"  Warning: {loc} has insufficient data, skipping")
                continue
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(grp[feat_cols].values.astype("float32"))
            scalers[loc] = scaler
            
            X_w, y_w = make_windows(X_scaled, target_idx, args.lookback, args.horizon)
            if len(X_w) > 0:
                X_parts.append(X_w)
                y_parts.append(y_w)
                lid_parts.append(np.full(len(X_w), loc2idx[loc], dtype=np.int64))
        
        if not X_parts:
            print("[ERROR] No valid locations with sufficient data")
            return
        
        X_win = np.concatenate(X_parts)
        y_win = np.concatenate(y_parts)
        loc_ids_arr = np.concatenate(lid_parts) if num_locations > 1 else None
        
        with open(os.path.join(args.outdir, "scalers.pkl"), "wb") as f:
            pickle.dump({"scalers": scalers, "loc2idx": loc2idx, "feat_cols": feat_cols, "target_idx": target_idx}, f)
        
        print(f"      Locations: {len(X_parts)}/{len(all_locs)} | Total windows: {len(X_win):,}")
        
    else:
        print("[ERROR] Must specify --location or --locations")
        return

    print(f"[2/5] Windows: {len(X_win):,} | X shape: {X_win.shape} | y shape: {y_win.shape}")

    # Split
    idx = np.arange(len(X_win))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, random_state=args.seed, shuffle=True)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.667, random_state=args.seed, shuffle=True)
    print(f"      Split: Train={len(idx_tr):,} | Val={len(idx_val):,} | Test={len(idx_te):,}")

    def make_ds(idx_set):
        Xt = torch.tensor(X_win[idx_set])
        yt = torch.tensor(y_win[idx_set])
        if loc_ids_arr is not None:
            return TensorDataset(Xt, yt, torch.tensor(loc_ids_arr[idx_set]))
        return TensorDataset(Xt, yt)

    pin_memory = device.type == "cuda" or device.type == "mps"
    tr_dl = DataLoader(
        make_ds(idx_tr), 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )
    va_dl = DataLoader(
        make_ds(idx_val), 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    # Model
    model = ImprovedLSTMForecaster(
        input_size=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        horizon=args.horizon,
        num_locations=num_locations,
        embed_dim=args.embed_dim,
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[3/5] Model: {total_params:,} params")
    print(f"      Hidden: {args.hidden_size} | Layers: {args.num_layers} | Dropout: 0.2")
    
    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    
    # Loss function
    def combined_loss(pred, target):
        return nn.MSELoss()(pred, target) + 0.5 * nn.L1Loss()(pred, target)
    
    crit = combined_loss if args.loss == "combined" else nn.HuberLoss(delta=1.0)
    print(f"      Loss: {'Combined (MSE+MAE)' if args.loss == 'combined' else 'Huber'}")
    print(f"      LR: {args.lr} | Batch: {args.batch_size}")
    print(f"      Device: {device}")
    print(f"      Num workers: {args.num_workers}")
    
    # ==================== KHỞI TẠO OVERFITTING DETECTOR ====================
    overfit_detector = OverfittingDetector(
        patience=args.early_stop_patience,
        overfit_threshold=args.overfit_threshold
    )
    print(f"      Early stop patience: {args.early_stop_patience}")
    print(f"      Overfit threshold: {args.overfit_threshold}")
    
    # Training
    best_val = float("inf")
    best_path = os.path.join(args.outdir, "best_lstm.pt")
    history = {"train_loss": [], "val_loss": [], "gap": [], "is_overfitting": []}
    
    print(f"\n[4/5] Training ({args.epochs} epochs)...")
    start = time.time()
    
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(tr_dl, desc=f"Epoch {ep:02d}/train", leave=False):
            xb, yb = batch[0], batch[1]
            lb = batch[2] if len(batch) == 3 else None
            
            xb = xb.to(device)
            yb = yb.to(device)
            if lb is not None:
                lb = lb.to(device)
            
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb, lb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)
        
        train_loss /= len(tr_dl.dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(va_dl, desc=f"Epoch {ep:02d}/val", leave=False):
                xb, yb = batch[0], batch[1]
                lb = batch[2] if len(batch) == 3 else None
                
                xb = xb.to(device)
                yb = yb.to(device)
                if lb is not None:
                    lb = lb.to(device)
                
                loss = crit(model(xb, lb), yb)
                val_loss += loss.item() * xb.size(0)
        
        val_loss /= len(va_dl.dataset)
        gap = val_loss - train_loss
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["gap"].append(gap)
        
        scheduler.step()
        current_lr = opt.param_groups[0]['lr']
        
        # ==================== KIỂM TRA OVERFITTING ====================
        should_stop, is_overfitting, overfit_msg = overfit_detector.check(train_loss, val_loss, ep)
        history["is_overfitting"].append(is_overfitting)
        
        # Hiển thị thông tin GPU
        gpu_info = ""
        if device.type == "cuda":
            mem_alloc = torch.cuda.memory_allocated(device) / 1e9
            mem_reserved = torch.cuda.memory_reserved(device) / 1e9
            gpu_info = f" | GPU: {mem_alloc:.1f}/{mem_reserved:.1f}GB"
        
        # Hiển thị warning nếu overfitting
        overfit_flag = " 🔴 OVERFITTING" if is_overfitting else ""
        print(f"  Epoch {ep:02d}/{args.epochs} | train={train_loss:.4f} | val={val_loss:.4f} | gap={gap:.4f}{overfit_flag} | lr={current_lr:.2e} | {time.time()-t0:.1f}s{gpu_info}")
        
        if overfit_msg and is_overfitting:
            print(f"      {overfit_msg}")
        
        # Lưu model tốt nhất
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.cpu().state_dict(),
                "horizon": args.horizon,
                "lookback": args.lookback,
                "feat_cols": feat_cols,
                "target_idx": target_idx,
                "num_locations": num_locations,
                "embed_dim": args.embed_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
            }, best_path)
            model.to(device)
            print(f"            ✓ Best saved (val={val_loss:.4f})")
        
        # ==================== DỪNG NẾU OVERFITTING KÉO DÀI ====================
        if should_stop:
            print(f"\n  🛑 Training stopped early at epoch {ep} due to overfitting/no improvement")
            break
    
    elapsed = time.time() - start
    print(f"\n  Training completed in {int(elapsed//60)}m {int(elapsed%60)}s")
    
    # ==================== PHÂN TÍCH OVERFITTING CUỐI CÙNG ====================
    print("\n[5/5] Evaluation & Overfitting Analysis...")
    
    # Tính toán thống kê overfitting
    final_gap = history["gap"][-1] if history["gap"] else 0
    max_gap = max(history["gap"]) if history["gap"] else 0
    overfitting_epochs = sum(history["is_overfitting"])
    
    print(f"\n  📊 Overfitting Analysis:")
    print(f"      Final gap (val - train): {final_gap:.4f}")
    print(f"      Max gap: {max_gap:.4f}")
    print(f"      Overfitting epochs: {overfitting_epochs}/{len(history['is_overfitting'])}")
    
    if final_gap > args.overfit_threshold:
        print(f"\n  ⚠️  WARNING: Model shows signs of overfitting!")
        print(f"     Suggestions to reduce overfitting:")
        print(f"     1. Increase dropout rate (--dropout 0.3 or 0.4)")
        print(f"     2. Increase weight decay (--weight_decay 1e-4)")
        print(f"     3. Reduce model complexity (--hidden_size 96 --num_layers 1)")
        print(f"     4. Increase training data or use data augmentation")
        print(f"     5. Use --loss combined for better generalization")
    else:
        print(f"\n  ✅ Model shows good generalization!")
    
    # Evaluation
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    def eval_split(idx_set, split_name):
        Xt = torch.tensor(X_win[idx_set]).to(device)
        lb = torch.tensor(loc_ids_arr[idx_set]).to(device) if loc_ids_arr is not None else None
        
        with torch.no_grad():
            pred_s = model(Xt, lb).cpu().numpy()
        
        yt_true = y_win[idx_set]
        pf, tf = pred_s.flatten(), yt_true.flatten()
        
        results = {
            "norm_mae": mae(tf, pf),
            "norm_mse": mse(tf, pf),
            "norm_rmse": rmse(tf, pf),
            "norm_mape": mape(tf, pf),
            "norm_r2": R2(tf, pf)
        }
        
        print(f"  [{split_name:5s}] R²={results['norm_r2']:.4f} | MAE={results['norm_mae']:.4f} | RMSE={results['norm_rmse']:.4f}")
        return results
    
    val_m = eval_split(idx_val, "Val")
    test_m = eval_split(idx_te, "Test")
    
    # Save metrics
    metrics = {f"val_{k}": v for k, v in val_m.items()}
    metrics.update({f"test_{k}": v for k, v in test_m.items()})
    metrics["best_epoch"] = len(history["val_loss"])
    metrics["device"] = str(device)
    metrics["final_gap"] = final_gap
    metrics["max_gap"] = max_gap
    metrics["overfitting_epochs"] = overfitting_epochs
    
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save history
    pd.DataFrame(history).to_csv(os.path.join(args.outdir, "training_history.csv"), index=False)
    
    print(f"\n✅ Complete! Results saved to {args.outdir}/")
    print(f"   Best R² Score: {test_m['norm_r2']:.4f}")

if __name__ == "__main__":
    main()