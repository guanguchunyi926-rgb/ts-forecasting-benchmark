from typing import List, Tuple
import numpy as np

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