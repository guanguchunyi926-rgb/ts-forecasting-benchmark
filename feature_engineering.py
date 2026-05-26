from typing import List, Tuple
import numpy as np

def select_lag_width(series_length: int, sp: int = 1, min_window: int = 3, max_window: int = 24) -> int:
    """
    時系列データの長さと周期(sp)に基づいて、最適なラグ幅(ラグ特徴量の次元数)を返します。
    - sp が 1 の場合は、系列長の 1/4 程度を目安にします。
    - sp > 1 の場合は、周期を考慮して、少なくとも一周期分の特徴量を確保します。
    """
    if series_length <= min_window:
        return max(1, series_length - 1)

    default_window = min(max_window, max(min_window, series_length // 4))
    if sp > 1:
        seasonal_window = min(max_window, sp, series_length - 1)
        if series_length > sp + 1:
            return max(default_window, seasonal_window)
    return default_window


def create_lag_features(series: List[float], window_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    1次元の時系列データから、過去のラグ(タイムウインドウ)を特徴量とした
    教師あり学習用のデータセット X とターゲット y を作成します。
    """
    arr = np.array(series, dtype=float)
    n = len(arr)
    
    if n <= window_length:
        raise ValueError(f"データ件数 ({n}) は、ラグ幅 window_length ({window_length}) より大きい必要があります。")
        
    X_list = []
    y_list = []
    
    for i in range(n - window_length):
        X_list.append(arr[i : i + window_length])
        y_list.append(arr[i + window_length])
        
    return np.array(X_list, dtype=float), np.array(y_list, dtype=float)