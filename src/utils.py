import numpy as np, torch
from sklearn.preprocessing import StandardScaler
import joblib, os

def make_windows(series: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)

def scale_series(arr: np.ndarray, out_path: str):
    scaler = StandardScaler()
    arr2d = arr.reshape(-1, 1)
    scaled = scaler.fit_transform(arr2d).flatten()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(scaler, out_path)
    return scaled, scaler

def scale_with(arr: np.ndarray, scaler):
    return scaler.transform(arr.reshape(-1, 1)).flatten()

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))

def mspe(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.square((y_true - y_pred) / (np.abs(y_true) + eps))))

def rse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.sqrt(np.sum((y_true - y_pred) ** 2))
        / np.sqrt(np.sum((y_true - y_true.mean()) ** 2))
    )

def R2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae":  mae(y_true, y_pred),
        "mse":  mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mspe": mspe(y_true, y_pred),
        "rse":  rse(y_true, y_pred),
        "R2":   R2(y_true, y_pred),
    }