from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def impute_missing_values(series: List[Optional[float]], strategy: str = "ffill_bfill") -> List[float]:
    """
    時系列データの欠損値（None や NaN）を補完します。
    """
    logger.debug("preprocessing.impute_missing_values called: length=%s strategy=%s", len(series), strategy)
    arr = np.array([x if x is not None else np.nan for x in series], dtype=float)
    if np.isnan(arr).all():
        logger.warning("preprocessing.impute_missing_values: all values are NaN/None, using fallback 100.0")
        return [100.0] * len(series)

    if strategy == "ffill_bfill":
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr = arr[idx]

        if np.isnan(arr).any():
            mask = np.isnan(arr)
            rev_arr = arr[::-1]
            rev_mask = np.isnan(rev_arr)
            rev_idx = np.where(~rev_mask, np.arange(len(rev_arr)), 0)
            np.maximum.accumulate(rev_idx, out=rev_idx)
            rev_arr = rev_arr[rev_idx]
            arr = rev_arr[::-1]

    elif strategy == "mean":
        mean_val = np.nanmean(arr)
        arr[np.isnan(arr)] = mean_val if not np.isnan(mean_val) else 100.0

    logger.debug("preprocessing.impute_missing_values finished: nan_count=%s", int(np.isnan(arr).sum()))
    return arr.tolist()


class StandardScaler1D:
    """
    1次元時系列データのための標準化スケーラー（Z-Score Normalization）。
    """
    def __init__(self):
        self.mean_: float = 0.0
        self.scale_: float = 1.0
        self.is_fitted: bool = False

    def fit(self, series: List[float]) -> "StandardScaler1D":
        arr = np.array(series, dtype=float)
        self.mean_ = float(np.mean(arr))
        std = float(np.std(arr))
        self.scale_ = std if std > 1e-8 else 1.0
        self.is_fitted = True
        logger.debug("StandardScaler1D.fit: mean=%s scale=%s", self.mean_, self.scale_)
        return self

    def transform(self, series: List[float]) -> List[float]:
        if not self.is_fitted:
            raise ValueError("スケーラーが fit されていません。")
        arr = np.array(series, dtype=float)
        scaled = (arr - self.mean_) / self.scale_
        logger.debug("StandardScaler1D.transform: input_len=%s", len(arr))
        return scaled.tolist()

    def fit_transform(self, series: List[float]) -> List[float]:
        logger.debug("StandardScaler1D.fit_transform called: length=%s", len(series))
        return self.fit(series).transform(series)

    def inverse_transform(self, series: List[float]) -> List[float]:
        if not self.is_fitted:
            raise ValueError("スケーラーが fit されていません。")
        arr = np.array(series, dtype=float)
        original = (arr * self.scale_) + self.mean_
        logger.debug("StandardScaler1D.inverse_transform: input_len=%s", len(arr))
        return original.tolist()