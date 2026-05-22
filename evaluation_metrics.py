import numpy as np
from typing import List

def calculate_smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) を計算します。
    """
    actual = np.nan_to_num(actual)
    forecast = np.nan_to_num(forecast)
    denominator = (np.abs(actual) + np.abs(forecast)) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs(actual - forecast) / denominator * 100.0
        ape[denominator == 0] = 0.0
        
    return float(np.mean(ape))

def calculate_mase_denominator(train_series: np.ndarray, sp: int = 1) -> float:
    """
    MASEの分母となる、学習データ期間におけるナイーブ予測の平均絶対誤差 (MAE) を計算します。
    """
    clean_series = train_series[~np.isnan(train_series)]
    if len(clean_series) <= sp:
        return 1.0
        
    diff = np.abs(clean_series[sp:] - clean_series[:-sp])
    mean_diff = float(np.mean(diff))
    return mean_diff if mean_diff > 1e-8 else 1.0

def calculate_mase(actual: np.ndarray, forecast: np.ndarray, mase_denominator: float) -> float:
    """
    Mean Absolute Scaled Error (MASE) を計算します。
    """
    actual = np.nan_to_num(actual)
    forecast = np.nan_to_num(forecast)
    mae = np.mean(np.abs(actual - forecast))
    return float(mae / mase_denominator)