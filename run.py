import argparse, sys, os
from datetime import datetime
from src.train_lstm   import main as run_train
from src.predict_lstm import main as run_predict


def parse_args():
    ap = argparse.ArgumentParser(
        description="Pipeline LSTM: train + predict (direct multi-step, horizon=24)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Shared ────────────────────────────────────────────────────
    ap.add_argument("--input",        required=True,  help="CSV dữ liệu đầu vào")
    ap.add_argument("--outdir",       default="outputs", help="Thư mục gốc chứa kết quả")
    ap.add_argument("--value_column", default="aqi",  help="Tên cột target")
    ap.add_argument("--location",     default=None,   help="1 tỉnh (single-location)")
    ap.add_argument("--locations",    nargs="+", default=None,
                    help="Một số tỉnh chọn lọc (multi-location)")
    ap.add_argument("--run_name",     default=None,   help="Tên riêng cho lần chạy (nếu không sẽ tự động tạo timestamp)")

    # ── Train ─────────────────────────────────────────────────────
    ap.add_argument("--lookback",    type=int,   default=72,
                    help="Số giờ nhìn lại (nên >= 2×horizon)")
    ap.add_argument("--horizon",     type=int,   default=24,
                    help="Số giờ dự báo trực tiếp")
    ap.add_argument("--hidden_size", type=int,   default=128)
    ap.add_argument("--num_layers",  type=int,   default=2)
    ap.add_argument("--embed_dim",   type=int,   default=16)
    ap.add_argument("--epochs",      type=int,   default=20)
    ap.add_argument("--batch-size",  type=int,   default=128)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--loss",        default="huber", help="huber | mse | combined")
    ap.add_argument("--seed",        type=int,   default=42)

    # ── Predict ───────────────────────────────────────────────────
    ap.add_argument("--output_csv",  default=None,
                    help="Đường dẫn CSV kết quả (mặc định: <run_dir>/predictions.csv)")

    return ap.parse_args()


def create_run_directory(base_outdir: str, run_name: str = None) -> str:
    """Tạo thư mục riêng cho lần chạy với timestamp"""
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    run_dir = os.path.join(base_outdir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"📁 Creating run directory: {run_dir}")
    return run_dir


def build_train_argv(args, run_dir: str) -> list[str]:
    argv = [
        "--input",        args.input,
        "--outdir",       run_dir,
        "--value_column", args.value_column,
        "--lookback",     str(args.lookback),
        "--horizon",      str(args.horizon),
        "--hidden_size",  str(args.hidden_size),
        "--num_layers",   str(args.num_layers),
        "--embed_dim",    str(args.embed_dim),
        "--epochs",       str(args.epochs),
        "--batch-size",   str(args.batch_size),
        "--lr",           str(args.lr),
        "--loss",         args.loss,
        "--seed",         str(args.seed),
    ]
    if args.location:
        argv += ["--location", args.location]
    elif args.locations:
        argv += ["--locations"] + args.locations
    return argv


def build_predict_argv(args, run_dir: str) -> list[str]:
    output_csv = args.output_csv if args.output_csv else "predictions.csv"
    
    argv = [
        "--input",        args.input,
        "--outdir",       run_dir,
        "--value_column", args.value_column,
        "--out",          output_csv,
    ]
    if args.location:
        argv += ["--location", args.location]
    return argv


def main():
    args = parse_args()
    
    # Tạo thư mục riêng cho lần chạy này
    run_dir = create_run_directory(args.outdir, args.run_name)
    run_name = os.path.basename(run_dir)

    print("=" * 60)
    print(f"  RUN: {run_name}")
    print(f"  OUTPUT DIR: {run_dir}")
    print("=" * 60)
    
    # ========== STEP 1: TRAINING ==========
    print("\n" + "=" * 60)
    print("  STEP 1 / 2 — TRAINING")
    print(f"  horizon={args.horizon}  lookback={args.lookback}  epochs={args.epochs}")
    print("=" * 60)
    sys.argv = ["train_lstm.py"] + build_train_argv(args, run_dir)
    run_train()

    # ========== STEP 2: PREDICTION ==========
    print("\n" + "=" * 60)
    print("  STEP 2 / 2 — PREDICTION (direct multi-step)")
    print("=" * 60)
    sys.argv = ["predict_lstm.py"] + build_predict_argv(args, run_dir)
    run_predict()

    print("\n" + "=" * 60)
    print("  PIPELINE HOÀN THÀNH")
    print(f"  Run name: {run_name}")
    print(f"  Output → {run_dir}/")
    print(f"  best_lstm.pt | metrics.json | predictions.csv | training_history.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()