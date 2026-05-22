from typing import List, Dict, Any
import numpy as np
from evaluation_metrics import calculate_smape, calculate_mase

def rank_models_by_owa(
    all_forecasts: Dict[str, List[List[float]]],
    actual_series_list: List[List[float]],
    mase_denominators: List[float],
    baseline_id: str = "naive2"
) -> List[Dict[str, Any]]:
    """
    全モデルの平均SMAPE、平均MASE、およびベースラインに対する OWA を集計してランキングを生成します。
    """
    num_series = len(actual_series_list)
    if num_series == 0:
        return []
        
    model_ids = list(all_forecasts.keys())
    metrics_pool = {m_id: {"smapes": [], "mases": []} for m_id in model_ids}
    
    for s in range(num_series):
        actual = np.array(actual_series_list[s], dtype=float)
        denom = mase_denominators[s]
        
        for m_id in model_ids:
            pred = np.array(all_forecasts[m_id][s], dtype=float)
            valid_mask = ~np.isnan(actual) & ~np.isnan(pred)
            if not np.any(valid_mask):
                smape, mase = 0.0, 1.0
            else:
                smape = calculate_smape(actual[valid_mask], pred[valid_mask])
                mase = calculate_mase(actual[valid_mask], pred[valid_mask], denom)
                
            metrics_pool[m_id]["smapes"].append(smape)
            metrics_pool[m_id]["mases"].append(mase)
            
    avg_metrics = {}
    for m_id in model_ids:
        avg_metrics[m_id] = {
            "avg_smape": float(np.mean(metrics_pool[m_id]["smapes"])),
            "avg_mase": float(np.mean(metrics_pool[m_id]["mases"]))
        }
        
    if baseline_id in avg_metrics:
        baseline_smape = avg_metrics[baseline_id]["avg_smape"]
        baseline_mase = avg_metrics[baseline_id]["avg_mase"]
    else:
        baseline_smape = 1.0
        baseline_mase = 1.0
        
    ranking_results = []
    for m_id in model_ids:
        avg_smape = avg_metrics[m_id]["avg_smape"]
        avg_mase = avg_metrics[m_id]["avg_mase"]
        
        rel_smape = avg_smape / baseline_smape if baseline_smape > 0 else 1.0
        rel_mase = avg_mase / baseline_mase if baseline_mase > 0 else 1.0
        owa = 0.5 * (rel_smape + rel_mase)
        
        ranking_results.append({
            "id": m_id,
            "avg_smape": avg_smape,
            "avg_mase": avg_mase,
            "owa": owa
        })
        
    ranking_results.sort(key=lambda x: x["owa"])
    return ranking_results