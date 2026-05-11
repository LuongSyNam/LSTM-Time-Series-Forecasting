"""
run.py — Pipeline đầy đủ: train → predict (direct multi-step).

Ví dụ:
  python run.py --input data/2025.csv --location angiang_longxuyen
  python run.py --input data/2025.csv --locations angiang_longxuyen hanoi --epochs 30
  python run.py --input data/2025.csv --epochs 20 --lookback 48

Lưu ý: --input được dùng cho cả train lẫn predict.
Nếu muốn predict trên CSV khác (không bao gồm ngày cần dự báo),
hãy chạy predict_lstm.py trực tiếp với --input riêng.
"""

import argparse, sys
from src.train_lstm   import main as run_train
from src.predict_lstm import main as run_predict


def parse_args():
    ap = argparse.ArgumentParser(
        description="Pipeline LSTM: train + predict (direct multi-step, horizon=24)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Shared ────────────────────────────────────────────────────
    ap.add_argument("--input",        required=True,  help="CSV dữ liệu đầu vào")
    ap.add_argument("--outdir",       default="outputs")
    ap.add_argument("--value_column", default="aqi",  help="Tên cột target")
    ap.add_argument("--location",     default=None,   help="1 tỉnh (single-location)")
    ap.add_argument("--locations",    nargs="+", default=None,
                    help="Một số tỉnh chọn lọc (multi-location)")

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
    ap.add_argument("--loss",        default="huber", help="huber | mse")
    ap.add_argument("--seed",        type=int,   default=42)

    # ── Predict ───────────────────────────────────────────────────
    ap.add_argument("--output_csv",  default=None,
                    help="Đường dẫn CSV kết quả (mặc định: <outdir>/predictions.csv)")

    return ap.parse_args()


def build_train_argv(args) -> list[str]:
    argv = [
        "--input",        args.input,
        "--outdir",       args.outdir,
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


def build_predict_argv(args) -> list[str]:
    argv = [
        "--input",        args.input,
        "--outdir",       args.outdir,
        "--value_column", args.value_column,
    ]
    if args.location:
        argv += ["--location", args.location]
    if args.output_csv:
        argv += ["--output_csv", args.output_csv]
    return argv


def main():
    args = parse_args()

    print("=" * 60)
    print("  STEP 1 / 2 — TRAINING")
    print(f"  horizon={args.horizon}  lookback={args.lookback}  epochs={args.epochs}")
    print("=" * 60)
    sys.argv = ["train_lstm.py"] + build_train_argv(args)
    run_train()

    print()
    print("=" * 60)
    print("  STEP 2 / 2 — PREDICTION (direct multi-step)")
    print("=" * 60)
    sys.argv = ["predict_lstm.py"] + build_predict_argv(args)
    run_predict()

    print()
    print("=" * 60)
    print("  PIPELINE HOÀN THÀNH")
    print(f"  Output → {args.outdir}/")
    print(f"  best_lstm.pt | metrics.json | predictions.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()