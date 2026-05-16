import argparse, os, pickle
import pandas as pd, numpy as np
import torch
from src.model import (
    get_device, build_advanced_time_features, add_lag_features,
    ImprovedLSTMForecaster, build_last_window, get_feature_columns
)

def main(args=None):
    # Nếu args được truyền từ run.py thì dùng, không thì parse từ command line
    if args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--ckpt", default="best_lstm.pt")
        ap.add_argument("--input", default="data/air_quality.csv")
        ap.add_argument("--location", default="khanhhoa_nhatrang")
        ap.add_argument("--out", default="predictions.csv")
        ap.add_argument("--outdir", type=str, default="outputs")
        ap.add_argument("--value_column", type=str, default="aqi")
        args = ap.parse_args()

    device = get_device()

    # Load checkpoint
    ckpt_path = os.path.join(args.outdir, args.ckpt)
    if not os.path.exists(ckpt_path):
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

    # Model
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

    # Load data
    print(f"[2/4] Loading data: {args.input}")
    df = pd.read_csv(args.input, parse_dates=["Time"])
    df = build_advanced_time_features(df)
    df = add_lag_features(df, args.value_column, dropna=True)
    
    if args.location:
        all_locs = [args.location]
    else:
        all_locs = sorted(df["location_key"].unique())
    print(f"      Locations: {all_locs}")

    # Load scaler
    print("[3/4] Loading scaler...")
    scaler_path = os.path.join(args.outdir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print(f"[ERROR] Không tìm thấy scaler: {scaler_path}")
        return
    
    with open(scaler_path, "rb") as f:
        bundle = pickle.load(f)
    scaler = bundle["scaler"]

    # Predict
    print(f"[4/4] Direct forecast ({horizon} bước)...")
    all_preds = []

    for loc in all_locs:
        sub = df[df["location_key"] == loc].copy()
        
        if len(sub) < lookback:
            print(f"  [Skip] {loc}: chỉ có {len(sub)} rows, cần {lookback}")
            continue
        
        sub = sub.sort_values("Time").reset_index(drop=True)
        
        try:
            window = build_last_window(sub, scaler, lookback, feat_cols, target_idx)
            X_t = torch.tensor(window, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pred_scaled = model(X_t, None).cpu().numpy()[0]
            
            n_features = len(feat_cols)
            dummy = np.zeros((horizon, n_features), dtype="float32")
            dummy[:, target_idx] = pred_scaled
            pred_final = scaler.inverse_transform(dummy)[:, target_idx]
            
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

    if all_preds:
        out_path = os.path.join(args.outdir, args.out)
        pd.DataFrame(all_preds).to_csv(out_path, index=False)
        print(f"\n[OK] Saved {len(all_preds)} predictions to {out_path}")
        print("\n📊 24-hour forecast:")
        for i, pred in enumerate(all_preds[:24]):
            print(f"  Hour +{pred['hour_ahead']:2d}: {pred['predicted_value']:.2f}")
    else:
        print("[ERROR] No predictions generated!")

if __name__ == "__main__":
    main()