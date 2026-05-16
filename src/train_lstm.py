import argparse, os, json, pickle, time
import pandas as pd, numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import rmse, mae, mse, mape, R2
torch.cuda.empty_cache()  # Giải phóng bộ nhớ cache
from src.model import (
    get_device, build_advanced_time_features, add_lag_features,
    ImprovedLSTMForecaster, make_windows, OverfittingDetector,
    get_feature_columns, AQ_COLS, TIME_COLS
)

DEFAULT_HORIZON = 24

def main(args=None):
    # Nếu args được truyền từ run.py thì dùng, không thì parse từ command line
    if args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--input", default="data/air_quality.csv")
        ap.add_argument("--lookback", type=int, default=96)
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
        ap.add_argument("--early_stop_patience", type=int, default=10)
        ap.add_argument("--overfit_threshold", type=float, default=0.05)
        args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = get_device()
    
    print("[1/5] Loading data...")
    df = pd.read_csv(args.input, parse_dates=["Time"])
    df = build_advanced_time_features(df)
    df = add_lag_features(df, args.value_column, dropna=False)
    
    feat_cols, target_idx = get_feature_columns(df, args.value_column)
    n_features = len(feat_cols)
    
    print(f"      Features: {n_features} total")
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
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sub[feat_cols].values.astype("float32"))
        
        with open(os.path.join(args.outdir, "scaler.pkl"), "wb") as f:
            pickle.dump({"scaler": scaler, "feat_cols": feat_cols, "target_idx": target_idx}, f)
        
        X_win, y_win = make_windows(X_scaled, target_idx, args.lookback, args.horizon)
        loc_ids_arr = None
        num_locations = 0
        
        print(f"      Location: {args.location} | Rows: {len(sub)}")
        
    elif args.locations is not None:
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

    print(f"[2/5] Windows: {len(X_win):,} | X shape: {X_win.shape}")

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
    print(f"      Hidden: {args.hidden_size} | Layers: {args.num_layers}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    
    def combined_loss(pred, target):
        return nn.MSELoss()(pred, target) + 0.5 * nn.L1Loss()(pred, target)
    
    crit = combined_loss if args.loss == "combined" else nn.HuberLoss(delta=1.0)
    print(f"      Loss: {'Combined' if args.loss == 'combined' else 'Huber'}")
    print(f"      LR: {args.lr} | Batch: {args.batch_size}")
    print(f"      Device: {device}")
    print(f"      Num workers: {args.num_workers}")
    
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
        
        model.train()
        train_loss = 0
        for batch in tqdm(tr_dl, desc=f"Epoch {ep:02d}/train", leave=False):
            xb, yb = batch[0], batch[1]
            lb = batch[2] if len(batch) == 3 else None
            
            xb, yb = xb.to(device), yb.to(device)
            if lb is not None:
                lb = lb.to(device)
            
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb, lb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)
        
        train_loss /= len(tr_dl.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(va_dl, desc=f"Epoch {ep:02d}/val", leave=False):
                xb, yb = batch[0], batch[1]
                lb = batch[2] if len(batch) == 3 else None
                
                xb, yb = xb.to(device), yb.to(device)
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
        
        should_stop, is_overfitting, overfit_msg = overfit_detector.check(train_loss, val_loss, ep)
        history["is_overfitting"].append(is_overfitting)
        
        overfit_flag = " 🔴 OVERFITTING" if is_overfitting else ""
        print(f"  Epoch {ep:02d}/{args.epochs} | train={train_loss:.4f} | val={val_loss:.4f} | gap={gap:.4f}{overfit_flag} | lr={current_lr:.2e} | {time.time()-t0:.1f}s")
        
        if overfit_msg and is_overfitting:
            print(f"      {overfit_msg}")
        
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
        
        if should_stop:
            print(f"\n  🛑 Training stopped early at epoch {ep}")
            break
    
    elapsed = time.time() - start
    print(f"\n  Training completed in {int(elapsed//60)}m {int(elapsed%60)}s")
    
    # Evaluation
    print("\n[5/5] Evaluation...")
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
    
    final_gap = history["gap"][-1] if history["gap"] else 0
    metrics = {f"val_{k}": v for k, v in val_m.items()}
    metrics.update({f"test_{k}": v for k, v in test_m.items()})
    metrics["best_epoch"] = len(history["val_loss"])
    metrics["final_gap"] = final_gap
    
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    pd.DataFrame(history).to_csv(os.path.join(args.outdir, "training_history.csv"), index=False)
    
    print(f"\n✅ Complete! Results saved to {args.outdir}/")
    print(f"   Best R² Score: {test_m['norm_r2']:.4f}")

if __name__ == "__main__":
    main()