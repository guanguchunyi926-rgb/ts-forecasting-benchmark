import logging
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

def calculate_smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) を計算します。
    """
    logger.debug("calculate_smape called: actual_shape=%s forecast_shape=%s", actual.shape, forecast.shape)
    actual = np.nan_to_num(actual)
    forecast = np.nan_to_num(forecast)
    denominator = (np.abs(actual) + np.abs(forecast)) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs(actual - forecast) / denominator * 100.0
        ape[denominator == 0] = 0.0
        
    result = float(np.mean(ape))
    logger.debug("calculate_smape result=%s", result)
    return result

def calculate_mase_denominator(train_series: np.ndarray, sp: int = 1) -> float:
    """
    MASEの分母となる、学習データ期間におけるナイーブ予測の平均絶対誤差 (MAE) を計算します。
    """
    clean_series = train_series[~np.isnan(train_series)]
    logger.debug("calculate_mase_denominator called: train_len=%s sp=%s", len(clean_series), sp)
    if len(clean_series) <= sp:
        logger.debug("calculate_mase_denominator: insufficient data for sp=%s, using fallback denominator=1.0", sp)
        return 1.0
    
    diff = np.abs(clean_series[sp:] - clean_series[:-sp])
    mean_diff = float(np.mean(diff))
    result = mean_diff if mean_diff > 1e-8 else 1.0
    logger.debug("calculate_mase_denominator result=%s", result)
    return result

def calculate_mase(actual: np.ndarray, forecast: np.ndarray, mase_denominator: float) -> float:
    """
    Mean Absolute Scaled Error (MASE) を計算します。
    """
    logger.debug("calculate_mase called: actual_shape=%s forecast_shape=%s denominator=%s", actual.shape, forecast.shape, mase_denominator)
    actual = np.nan_to_num(actual)
    forecast = np.nan_to_num(forecast)
    mae = np.mean(np.abs(actual - forecast))
    result = float(mae / mase_denominator)
    logger.debug("calculate_mase result=%s", result)
    return result