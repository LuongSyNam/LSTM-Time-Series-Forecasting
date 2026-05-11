import argparse, os, json, pickle, time
import pandas as pd, numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import rmse, mae, mse, mape, R2

# Tất cả các feature khí / AQI có trong CSV
AQ_COLS = [
    "pm25", "pm10", "no2", "o3", "so2", "co",
    "aod", "dust", "uv_index", "co2",
    "aqi", "aqi_pm25", "aqi_pm10", "aqi_no2", "aqi_o3", "aqi_so2", "aqi_co",
]

# Cyclic time encodings — tạo tự động từ ts_utc
TIME_COLS = ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm time encoding cyclic (sin/cos) vào df."""
    df = df.copy()
    ts = pd.to_datetime(df["ts_utc"], utc=True)
    df["hour_sin"]  = np.sin(2 * np.pi * ts.dt.hour      / 24).astype("float32")
    df["hour_cos"]  = np.cos(2 * np.pi * ts.dt.hour      / 24).astype("float32")
    df["day_sin"]   = np.sin(2 * np.pi * ts.dt.dayofweek / 7 ).astype("float32")
    df["day_cos"]   = np.cos(2 * np.pi * ts.dt.dayofweek / 7 ).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * ts.dt.month     / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * ts.dt.month     / 12).astype("float32")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Trả về AQ_COLS + TIME_COLS có mặt trong df (theo đúng thứ tự)."""
    return [c for c in AQ_COLS + TIME_COLS if c in df.columns]

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size ** 0.5

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        lstm_out : (B, T, H)
        returns  : (B, H)  — context vector
        """
        q = self.query(lstm_out[:, -1:, :])   # (B, 1, H) — query từ bước cuối
        k = self.key(lstm_out)                 # (B, T, H)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, 1, T)
        weights = torch.softmax(scores, dim=-1)                 # (B, 1, T)
        context = torch.bmm(weights, lstm_out).squeeze(1)       # (B, H)
        return context


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size:    int,
        hidden_size:   int   = 128,
        num_layers:    int   = 2,
        dropout:       float = 0.2,
        horizon:       int   = 24,
        num_locations: int   = 0, 
        embed_dim:     int   = 16,
    ):
        super().__init__()
        self.use_embedding = num_locations > 0
        if self.use_embedding:
            self.loc_embedding = nn.Embedding(num_locations, embed_dim)
            lstm_input = input_size + embed_dim
        else:
            lstm_input = input_size

        self.lstm = nn.LSTM(
            lstm_input, hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size)
        self.norm      = nn.LayerNorm(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor, loc_ids: torch.Tensor | None = None):
        """
        x       : (B, T, n_features)
        loc_ids : (B,)   — bắt buộc khi use_embedding=True
        """
        if self.use_embedding and loc_ids is not None:
            emb = self.loc_embedding(loc_ids)                  # (B, embed_dim)
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, embed_dim)
            x = torch.cat([x, emb], dim=-1)                    # (B, T, F+embed_dim)

        lstm_out, _ = self.lstm(x)                             # (B, T, H)
        context     = self.attention(lstm_out)                 # (B, H)
        # Residual: context + h_last → ổn định hơn, giữ thông tin cuối
        h_last  = lstm_out[:, -1, :]                           # (B, H)
        out     = self.norm(context + h_last)                  # (B, H)
        out     = self.dropout(out)
        return self.fc(out)                                    # (B, horizon)

def make_windows(
    X: np.ndarray,          # (T, n_features) — đã scale
    target_idx: int,        # index của cột target trong feat_cols
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        windows : (N, lookback, n_features)
        targets : (N, horizon)
    """
    T = len(X)
    windows, targets = [], []
    for i in range(T - lookback - horizon + 1):
        windows.append(X[i : i + lookback])
        targets.append(X[i + lookback : i + lookback + horizon, target_idx])
    return (
        np.array(windows, dtype="float32"),
        np.array(targets,  dtype="float32"),
    )

def inverse_target(
    values: np.ndarray,     # (N,) hoặc (N, horizon) — đã flat hoặc chưa
    scaler: StandardScaler,
    target_idx: int,
    n_features: int,
) -> np.ndarray:
    """Inverse-scale chỉ cột target, bỏ qua các cột còn lại."""
    flat = values.flatten()
    dummy = np.zeros((len(flat), n_features), dtype="float32")
    dummy[:, target_idx] = flat
    return scaler.inverse_transform(dummy)[:, target_idx]

FORECAST_STEPS = 24   # Số giờ dự báo (cố định, phải bằng --horizon khi train)


def direct_forecast(
    model:        nn.Module,
    last_window:  np.ndarray,      # (lookback, n_features) — đã scale
    scaler:       StandardScaler,
    target_idx:   int,
    n_features:   int,
    steps:        int = FORECAST_STEPS,
    loc_id:       int | None = None,
) -> np.ndarray:
    """
    Direct multi-step forecast: 1 lần forward pass → ra `horizon` bước cùng lúc.
    Không có error propagation như auto-regressive.

    Yêu cầu: model phải được train với horizon == steps (fc output = steps).

    Input : last_window (lookback, n_features) — đã scale
    Output: preds (steps,) — original scale
    """
    model.eval()
    lb_t = torch.tensor([loc_id]) if loc_id is not None else None

    with torch.no_grad():
        x            = torch.tensor(last_window).unsqueeze(0)  # (1, lookback, F)
        preds_scaled = model(x, lb_t).numpy().flatten()        # (horizon,)

    # Chỉ lấy `steps` bước đầu (phòng khi horizon > steps)
    preds_scaled = preds_scaled[:steps].astype("float32")

    return inverse_target(preds_scaled, scaler, target_idx, n_features)


def plot_curves(history: dict, outpath: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, vals in history.items():
        ax.plot(vals, label=key)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training / Validation / Test Loss")
    ax.legend(); fig.tight_layout()
    fig.savefig(outpath, dpi=160); plt.close(fig)


def plot_forecast(dates, values, pred_start_idx, preds, outpath):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, values, label="actual")
    fut = dates[pred_start_idx : pred_start_idx + len(preds)]
    ax.plot(fut, preds, label="forecast", linestyle="--")
    ax.set_title("Forecast vs Actual")
    ax.set_xlabel("Date"); ax.set_ylabel("Value")
    ax.legend(); fig.tight_layout()
    fig.savefig(outpath, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",        default="data/2025.csv")
    ap.add_argument("--lookback",     type=int,   default=24)
    ap.add_argument("--horizon",      type=int,   default=1)
    ap.add_argument("--value_column", type=str,   default="aqi",
                    help="Tên cột target cần dự đoán (phải nằm trong AQ_COLS)")
    ap.add_argument("--location",     type=str,   default=None,
                    help="1 tỉnh duy nhất — multivariate, không embedding")
    ap.add_argument("--locations",    type=str,   nargs="+", default=None,
                    help="Một số tỉnh chọn lọc — dùng embedding. VD: --locations a b c")
    ap.add_argument("--embed_dim",    type=int,   default=16,
                    help="Kích thước embedding (chỉ dùng khi multi-location)")
    ap.add_argument("--hidden_size",  type=int,   default=128)
    ap.add_argument("--num_layers",   type=int,   default=2)
    ap.add_argument("--epochs",       type=int,   default=5)
    ap.add_argument("--batch-size",   type=int,   default=512)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--outdir",       type=str,   default="outputs")
    ap.add_argument("--loss",         type=str,   default="huber",
                    help="Loss function: huber (default) hoac mse")
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ── Load & preprocess ──────────────────────────────────────────
    print("[1/5] Loading data...")
    df = pd.read_csv(args.input, parse_dates=["ts_utc"])
    df = build_time_features(df)
    feat_cols  = get_feature_cols(df)
    target_idx = feat_cols.index(args.value_column)
    n_features = len(feat_cols)

    print(f"      Features ({n_features}): {feat_cols}")
    print(f"      Target  : '{args.value_column}'  (idx {target_idx})")

    # Xác định mode
    # • single_location : --location X        → 1 tỉnh, không embedding
    # • multi_location  : --locations X Y Z   → tỉnh chọn lọc, có embedding
    #                   : (không arg nào)      → toàn bộ tỉnh, có embedding
    single_location = args.location is not None
    if args.locations is not None and args.location is not None:
        print("[ERROR] Chỉ dùng một trong --location hoặc --locations, không dùng cả hai.")
        return
    selected_locations = args.locations  # None = toàn bộ tỉnh

    
    if single_location:
        sub = df[df["location_key"] == args.location].copy()
        if sub.empty:
            print(f"[ERROR] Không tìm thấy dữ liệu cho location: {args.location}")
            return
        sub = sub.sort_values("ts_utc").reset_index(drop=True)
        print(f"\n[Mode] SINGLE-LOCATION — '{args.location}'")
        print(f"       {len(sub)} rows  |  {n_features} features  |  NO embedding\n")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sub[feat_cols].values.astype("float32"))
        with open(os.path.join(args.outdir, "scaler.pkl"), "wb") as f:
            pickle.dump({"scaler": scaler, "feat_cols": feat_cols,
                         "target_idx": target_idx}, f)

        X_win, y_win = make_windows(X_scaled, target_idx, args.lookback, args.horizon)
        loc_ids_arr  = None   # không cần
        num_locations = 0

        # giữ lại để plot forecast
        plot_dates  = sub["ts_utc"].values
        plot_values = sub[args.value_column].values
        forecast_scaler = scaler
        forecast_loc_id = None

    else:
        all_locs = sorted(df["location_key"].unique())

        # Nếu --locations được chỉ định, kiểm tra và lọc
        if selected_locations is not None:
            not_found = [l for l in selected_locations if l not in all_locs]
            if not_found:
                print(f"[ERROR] Không tìm thấy location: {not_found}")
                print(f"        Các location hợp lệ: {all_locs[:5]}...")
                return
            all_locs = sorted(selected_locations)
            df = df[df["location_key"].isin(all_locs)].copy()

        loc2idx   = {loc: i for i, loc in enumerate(all_locs)}
        num_locations = len(all_locs)
        # Nếu chỉ 1 tỉnh trong --locations → không cần embedding
        use_embedding_multi = num_locations > 1
        mode_label = f"{num_locations} tỉnh chọn lọc" if selected_locations else f"toàn bộ {num_locations} tỉnh"
        embed_info = f"embed_dim={args.embed_dim}" if use_embedding_multi else "NO embedding (chỉ 1 tỉnh)"
        print(f"\n[Mode] MULTI-LOCATION — {mode_label}")
        print(f"       {n_features} features  |  {embed_info}\n")

        scalers: dict[str, StandardScaler] = {}
        X_parts, y_parts, lid_parts = [], [], []

        for loc, grp in df.groupby("location_key", sort=True):
            grp = grp.sort_values("ts_utc")
            sc  = StandardScaler()
            vals = sc.fit_transform(grp[feat_cols].values.astype("float32"))
            scalers[loc] = sc

            X_w, y_w = make_windows(vals, target_idx, args.lookback, args.horizon)
            if len(X_w) == 0:
                continue
            X_parts.append(X_w)
            y_parts.append(y_w)
            lid_parts.append(np.full(len(X_w), loc2idx[loc], dtype=np.int64))

        with open(os.path.join(args.outdir, "scalers.pkl"), "wb") as f:
            pickle.dump({"scalers": scalers, "loc2idx": loc2idx,
                         "feat_cols": feat_cols, "target_idx": target_idx}, f)

        X_win = np.concatenate(X_parts)
        y_win = np.concatenate(y_parts)
        loc_ids_arr = np.concatenate(lid_parts) if use_embedding_multi else None
        # Nếu 1 tỉnh trong multi-mode: tắt embedding giống single-location
        if not use_embedding_multi:
            num_locations = 0

        # dùng tỉnh cuối cùng để vẽ forecast
        last_loc = all_locs[-1]
        last_sub = df[df["location_key"] == last_loc].sort_values("ts_utc")
        plot_dates  = last_sub["ts_utc"].values
        plot_values = last_sub[args.value_column].values
        forecast_scaler = scalers[last_loc]
        forecast_loc_id = loc2idx[last_loc] if use_embedding_multi else None

        last_lid = loc2idx[last_loc]
        if use_embedding_multi:
            last_loc_mask = np.where(loc_ids_arr == last_lid)[0]
        else:
            last_loc_mask = np.arange(len(X_win))   # chỉ 1 tỉnh → dùng tất cả
        forecast_last_win_idx = last_loc_mask[-1]

    print(f"[2/5] Windows: {len(X_win):,}  shape={X_win.shape}")

    # ── Train/Val/Test split (giữ thứ tự thời gian) ───────────────
    idx = np.arange(len(X_win))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, shuffle=False)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.667, shuffle=False)
    print(f"       Split → Train={len(idx_tr):,}  Val={len(idx_val):,}  Test={len(idx_te):,}")

    def make_ds(idx_set):
        Xt = torch.tensor(X_win[idx_set])
        yt = torch.tensor(y_win[idx_set])
        if loc_ids_arr is not None:
            lt = torch.tensor(loc_ids_arr[idx_set])
            return TensorDataset(Xt, yt, lt)
        return TensorDataset(Xt, yt)

    tr_dl = DataLoader(make_ds(idx_tr),  batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(make_ds(idx_val), batch_size=args.batch_size, shuffle=False)
    te_dl = DataLoader(make_ds(idx_te),  batch_size=args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────
    model = LSTMForecaster(
        input_size    = n_features,
        hidden_size   = args.hidden_size,
        num_layers    = args.num_layers,
        horizon       = args.horizon,
        num_locations = num_locations,
        embed_dim     = args.embed_dim,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[3/5] Model  — {total_params:,} parameters")
    if model.use_embedding:
        print(f"       Embedding: {num_locations} locations × {args.embed_dim} dims")

    opt  = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Tự giảm LR khi val loss không cải thiện sau 2 epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2
    )
    _loss_map = {"huber": nn.HuberLoss(delta=1.0), "mse": nn.MSELoss()}
    crit = _loss_map.get(args.loss.lower(), nn.HuberLoss(delta=1.0))
    print(f"       Loss    : {args.loss.upper()}")
    history   = {"train_loss": [], "val_loss": [], "test_loss": []}
    best_val  = float("inf"); stale = 0
    best_path = os.path.join(args.outdir, "best_lstm.pt")

    def run_epoch(dl, train=False):
        total, n = 0.0, 0
        model.train() if train else model.eval()
        ctx = torch.enable_grad() if train else torch.no_grad()
        desc = "train" if train else "eval"
        with ctx:
            for batch in tqdm(dl, desc=f"  {desc}", leave=False):
                if loc_ids_arr is not None:
                    xb, yb, lb = batch
                else:
                    xb, yb = batch; lb = None
                if train: opt.zero_grad()
                preds = model(xb, lb)
                loss  = crit(preds, yb)
                if train: loss.backward(); opt.step()
                total += loss.item() * xb.size(0)
                n     += xb.size(0)
        return total / n

    # ── Training loop ─────────────────────────────────────────────
    print(f"\n[4/5] Training ({args.epochs} epochs)...")
    start = time.time()
    for ep in range(1, args.epochs + 1):
        ep_start = time.time()
        tl   = run_epoch(tr_dl, train=True)
        vl   = run_epoch(va_dl, train=False)
        tel  = run_epoch(te_dl, train=False)
        ep_sec = time.time() - ep_start
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["test_loss"].append(tel)
        scheduler.step(vl)
        cur_lr = opt.param_groups[0]["lr"]
        print(f"  [Epoch {ep:02d}/{args.epochs}]  "
              f"train={tl:.4f}  val={vl:.4f}  test={tel:.4f}  "
              f"lr={cur_lr:.2e}  ({ep_sec:.1f}s)")
        if vl < best_val:
            best_val = vl; stale = 0
            torch.save({
                "model_state":   model.state_dict(),
                "horizon":       args.horizon,
                "lookback":      args.lookback,
                "feat_cols":     feat_cols,
                "target_idx":    target_idx,
                "num_locations": num_locations,
                "embed_dim":     args.embed_dim,
                "hidden_size":   args.hidden_size,
                "num_layers":    args.num_layers,
            }, best_path)
            print(f"            ✓ Best model saved (val={vl:.4f})")
        else:
            stale += 1
            if stale >= 5:
                print("  Early stopping."); break

    elapsed = time.time() - start
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"\n  Training time: {h}h {m}m {s}s\n")
    plot_curves(history, os.path.join(args.outdir, "training_curves.png"))

    # ── Evaluation ────────────────────────────────────────────────
    print("[5/5] Evaluation...")
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"]); model.eval()

    def eval_split(idx_set, split_name):
        Xt = torch.tensor(X_win[idx_set])
        lb = torch.tensor(loc_ids_arr[idx_set]) if loc_ids_arr is not None else None
        with torch.no_grad():
            pred_s = model(Xt, lb).numpy()   # (N, horizon) — normalized scale
        yt_true = y_win[idx_set]             # (N, horizon) — normalized scale

        # ── [A] Normalized scale — để so sánh với bài báo ──────────
        pred_flat = pred_s.flatten()
        true_flat = yt_true.flatten()
        norm_mae  = mae(true_flat,  pred_flat)
        norm_mse  = mse(true_flat,  pred_flat)
        norm_rmse = rmse(true_flat, pred_flat)
        norm_mape = mape(true_flat, pred_flat)   # 0-1
        norm_r2   = R2(true_flat, pred_flat)

        # ── [B] Original scale — ý nghĩa thực tế (AQI units) ───────
        if single_location:
            pred_orig = inverse_target(pred_s,   forecast_scaler, target_idx, n_features)
            true_orig = inverse_target(yt_true,  forecast_scaler, target_idx, n_features)
        else:
            with open(os.path.join(args.outdir, "scalers.pkl"), "rb") as f:
                bundle = pickle.load(f)
            sc_map  = bundle["scalers"]
            idx2loc = {v: k for k, v in bundle["loc2idx"].items()}
            lids    = loc_ids_arr[idx_set]
            pred_orig_list, true_orig_list = [], []
            for i, lid in enumerate(lids):
                sc_i = sc_map[idx2loc[lid]]
                pred_orig_list.extend(inverse_target(pred_s[i],  sc_i, target_idx, n_features))
                true_orig_list.extend(inverse_target(yt_true[i], sc_i, target_idx, n_features))
            pred_orig = np.array(pred_orig_list)
            true_orig = np.array(true_orig_list)

        orig_mae  = mae(true_orig,  pred_orig)
        orig_rmse = rmse(true_orig, pred_orig)
        orig_mape = mape(true_orig, pred_orig)   # 0-1
        orig_r2   = R2(true_orig, pred_orig)

        print(f"  [{split_name:5s}]  MAE={norm_mae:.4f}  MSE={norm_mse:.4f}  "
              f"RMSE={norm_rmse:.4f}  MAPE={norm_mape:.4f}  R²={norm_r2:.4f}")

        return {
            "norm_mae":  norm_mae,
            "norm_mse":  norm_mse,
            "norm_rmse": norm_rmse,
            "norm_mape": norm_mape,
            "norm_r2":   norm_r2,
            "orig_mae":  orig_mae,
            "orig_rmse": orig_rmse,
            "orig_mape": orig_mape,
            "orig_r2":   orig_r2,
        }

    val_m  = eval_split(idx_val, "Val")
    test_m = eval_split(idx_te,  "Test")

    metrics = {f"val_{k}": v for k, v in val_m.items()}
    metrics.update({f"test_{k}": v for k, v in test_m.items()})
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Future forecast (tất cả tỉnh đã train) ───────────────────
    all_pred_rows = []

    if single_location or (not single_location and num_locations == 0):
        # ── 1 tỉnh — direct multi-step 24h ──────────────────────
        pred = direct_forecast(
            model       = model,
            last_window = X_win[-1],          # (lookback, n_features) scaled
            scaler      = forecast_scaler,
            target_idx  = target_idx,
            n_features  = n_features,
            steps       = FORECAST_STEPS,
            loc_id      = None,
        )

        last_date      = pd.to_datetime(plot_dates[-1], utc=True)
        forecast_dates = [last_date + pd.Timedelta(hours=i+1) for i in range(FORECAST_STEPS)]

        loc_label = args.location if single_location else all_locs[0]
        for i, (fd, pv) in enumerate(zip(forecast_dates, pred)):
            all_pred_rows.append({"location": loc_label, "forecast_date": fd,
                                  "hour_ahead": i + 1, "predicted_value": pv})

        pred_start = max(0, len(plot_values) - FORECAST_STEPS)
        plot_forecast(plot_dates, plot_values, pred_start, pred,
                      os.path.join(args.outdir, "forecast_plot.png"))
        print(f"  Forecast → {loc_label}  (direct {FORECAST_STEPS}h)")

    else:
        # ── Nhiều tỉnh: lặp qua từng tỉnh ──────────────────────
        with open(os.path.join(args.outdir, "scalers.pkl"), "rb") as f:
            bundle = pickle.load(f)
        sc_map = bundle["scalers"]

        fig, axes = plt.subplots(len(all_locs), 1,
                                 figsize=(10, 4 * len(all_locs)), sharex=False)
        if len(all_locs) == 1:
            axes = [axes]

        for ax, loc in zip(axes, all_locs):
            lid    = loc2idx[loc]
            sc_loc = sc_map[loc]

            # Window cuối cùng của tỉnh này
            mask     = np.where(loc_ids_arr == lid)[0]
            last_win = X_win[mask[-1]]          # (lookback, n_features) scaled

            # Direct multi-step 24h
            pred = direct_forecast(
                model       = model,
                last_window = last_win,
                scaler      = sc_loc,
                target_idx  = target_idx,
                n_features  = n_features,
                steps       = FORECAST_STEPS,
                loc_id      = lid,
            )

            # Forecast dates tính từ thời điểm cuối của tỉnh này
            loc_sub    = df[df["location_key"] == loc].sort_values("ts_utc")
            loc_dates  = loc_sub["ts_utc"].values
            loc_values = loc_sub[args.value_column].values
            last_date  = pd.to_datetime(loc_dates[-1], utc=True)
            forecast_dates = [last_date + pd.Timedelta(hours=i+1) for i in range(FORECAST_STEPS)]

            for i, (fd, pv) in enumerate(zip(forecast_dates, pred)):
                all_pred_rows.append({"location": loc, "forecast_date": fd,
                                      "hour_ahead": i + 1, "predicted_value": pv})

            # Subplot cho tỉnh này
            pred_start = max(0, len(loc_values) - FORECAST_STEPS)
            fut = loc_dates[pred_start : pred_start + len(pred)]
            ax.plot(loc_dates, loc_values, label="actual", linewidth=0.8)
            ax.plot(fut, pred, label="forecast", linestyle="--")
            ax.set_title(loc); ax.set_ylabel(args.value_column)
            ax.legend(fontsize=8)
            print(f"  Forecast → {loc}  (direct {FORECAST_STEPS}h)")

        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "forecast_plot.png"), dpi=160)
        plt.close(fig)

    # Lưu predictions của tất cả tỉnh
    pred_df = pd.DataFrame(all_pred_rows)
    pred_df.to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)
    print(f"  → predictions.csv: {len(pred_df)} rows ({len(pred_df) // FORECAST_STEPS} tỉnh × {FORECAST_STEPS}h direct)")

    print(f"\n[OK] Hoàn thành! Output → {args.outdir}/")
    print(f"     best_lstm.pt  |  metrics.json  |  predictions.csv")
    print(f"     training_curves.png  |  forecast_plot.png")


if __name__ == "__main__":
    main()