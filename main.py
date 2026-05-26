from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import json
import os
import uvicorn
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 各各自作モジュールのインポート
from preprocessing import impute_missing_values
from evaluation_metrics import calculate_mase_denominator
from ranking_system import rank_models_by_owa
from statistical_models import predict_statistical_model
from machine_learning_models import predict_ml_model

# CORSの設定
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="複数時系列モデル ベンチマーク評価 API",
    description="外部構成ファイル models_config.json と manual.json の定義を自動解釈して動的予測・OWAランキング・マニュアル配信を行うAPI",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 構成設定ファイルとユーザーマニュアル (JSON) のロード
# ==========================================
CONFIG_FILE = "models_config.json"
MANUAL_FILE = "manual.json"

try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        models_config = json.load(f)
except Exception:
    models_config = {"categories": {}, "models": [], "ensemble_rules": {}}

try:
    with open(MANUAL_FILE, "r", encoding="utf-8") as f:
        manual_config = json.load(f)
except Exception:
    manual_config = {"title": "ユーザーマニュアル", "sections": []}

# ==========================================
# 期待リクエスト / レスポンスのスキーマ
# ==========================================
class EvaluationRequest(BaseModel):
    data_series: List[List[Optional[float]]]
    horizon: int
    selected_ids: List[str]
    existing_rule_series: Optional[List[List[Optional[float]]]] = None
    settings: Optional[Dict[str, Any]] = None

class ModelResult(BaseModel):
    id: str
    forecasts: List[List[Optional[float]]]
    mape: float  # SMAPE
    mase: float
    owa: float

class EvaluationResponse(BaseModel):
    status: str
    train_data: List[List[Optional[float]]]
    actual_val_data: List[List[Optional[float]]]
    results: List[ModelResult]

# ==========================================
# 動的なアンサンブル予測値ブレンド器
# ==========================================
def calculate_dynamic_ensemble(
    rule_def: Dict[str, Any],
    all_forecasts: Dict[str, List[List[float]]],
    sorted_individual_ids: List[str],
    subcat_groups: Dict[str, List[str]],
    temp_ranked_list: List[Dict[str, Any]],
    horizon: int,
    series_idx: int,
    sp: int
) -> List[float]:
    rule_type = rule_def.get("type")
    
    def get_f_val(m_id: str, idx: int) -> float:
        return all_forecasts[m_id][series_idx][idx]

    try:
        if rule_type == "fixed":
            members = rule_def["members"]
            return [float(np.mean([get_f_val(m_id, i) for m_id in members])) for i in range(horizon)]
            
        elif rule_type == "top_n":
            source = rule_def["source"]
            n = rule_def["n"]
            
            if source == "statistical":
                filtered_ids = [m_id for m_id in sorted_individual_ids if m_id in subcat_groups.get("naive", []) + subcat_groups.get("smoothing", []) + subcat_groups.get("arima_grp", []) + subcat_groups.get("state_space", [])]
            elif source == "ml":
                filtered_ids = [m_id for m_id in sorted_individual_ids if m_id in subcat_groups.get("linear_reg", []) + subcat_groups.get("tree", []) + subcat_groups.get("svm_grp", []) + subcat_groups.get("nn", [])]
            else:
                filtered_ids = sorted_individual_ids
                
            top_ids = filtered_ids[:n]
            return [float(np.mean([get_f_val(m_id, i) for m_id in top_ids])) for i in range(horizon)]
            
        elif rule_type == "top_subcats":
            n_subcats = rule_def["n_subcats"]
            subcat_bests = {}
            for subcat_id, members in subcat_groups.items():
                best_item = next((item for item in temp_ranked_list if item["id"] in members), None)
                if best_item:
                    subcat_bests[subcat_id] = best_item
                    
            sorted_subcat_bests = sorted(subcat_bests.items(), key=lambda x: x[1]["owa"])
            best_subcat_ids = [item[1]["id"] for item in sorted_subcat_bests[:n_subcats]]
            
            while len(best_subcat_ids) < n_subcats and len(sorted_individual_ids) > len(best_subcat_ids):
                for m_id in sorted_individual_ids:
                    if m_id not in best_subcat_ids:
                        best_subcat_ids.append(m_id)
                        break
                        
            return [float(np.mean([get_f_val(m_id, i) for m_id in best_subcat_ids])) for i in range(horizon)]
            
        elif rule_type == "top_n_multi":
            sources_rules = rule_def["sources"]
            selected_ids = []
            
            stat_pool = subcat_groups.get("naive", []) + subcat_groups.get("smoothing", []) + subcat_groups.get("arima_grp", []) + subcat_groups.get("state_space", [])
            ml_pool = subcat_groups.get("linear_reg", []) + subcat_groups.get("tree", []) + subcat_groups.get("svm_grp", []) + subcat_groups.get("nn", [])
            
            for s_rule in sources_rules:
                src_name = s_rule["source"]
                n_count = s_rule["n"]
                if src_name == "statistical":
                    top_stats = [m_id for m_id in sorted_individual_ids if m_id in stat_pool][:n_count]
                    selected_ids.extend(top_stats)
                elif src_name == "ml":
                    top_mls = [m_id for m_id in sorted_individual_ids if m_id in ml_pool][:n_count]
                    selected_ids.extend(top_mls)
                    
            return [float(np.mean([get_f_val(m_id, i) for m_id in selected_ids])) for i in range(horizon)]
            
        else:
            return [all_forecasts["naive1"][series_idx][i] for i in range(horizon)]
            
    except Exception:
        return [all_forecasts["naive1"][series_idx][i] for i in range(horizon)]

# ==========================================
# エンドポイント
# ==========================================
@app.get("/api/config")
async def get_config():
    """フロントエンドに予測手法マスタ・カテゴリ設定を配信"""
    return models_config

@app.get("/api/manual")
async def get_manual():
    """フロントエンドにユーザーマニュアル設定を配信"""
    return manual_config

@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_models(request: EvaluationRequest):
    try:
        horizon = request.horizon
        data_series = request.data_series
        selected_ids = request.selected_ids
        existing_rule = request.existing_rule_series
        has_existing = existing_rule is not None

        logger.info("API /api/evaluate called")
        logger.info("Request: horizon=%s selected_ids=%s has_existing=%s series_count=%s",
                    horizon, selected_ids, has_existing, len(data_series))

        freq = "M"
        if request.settings and "freq" in request.settings:
            freq = request.settings["freq"]

        freq_alias_map = {"M": "ME", "Q": "QE", "Y": "YE"}
        freq = freq_alias_map.get(freq, freq)
        sp_map = {"ME": 12, "W": 52, "D": 7, "QE": 4, "YE": 1}

        sp = sp_map.get(freq, 1)
        logger.info("Step 1/7: settings parsed (freq=%s, sp=%s)", freq, sp)

        # 1. 前処理
        train_data_series = []
        actual_val_data_series = []
        mase_denominators = []

        for idx, data in enumerate(data_series):
            imputed_data = impute_missing_values(data, strategy="ffill_bfill")
            train_part = imputed_data[:-horizon]
            val_part = imputed_data[-horizon:]

            train_data_series.append(train_part)
            actual_val_data_series.append(val_part)

            denom = calculate_mase_denominator(np.array(train_part), sp)
            mase_denominators.append(denom)
            logger.debug("Preprocessed series %s: train_len=%s val_len=%s denom=%s",
                         idx, len(train_part), len(val_part), denom)

        logger.info("Step 2/7: preprocessing complete for %s series", len(data_series))

        # 2. 設定JSONマスタに基づく全個別基礎キーの動的分類
        models_meta = models_config.get("models", [])
        subcat_groups = models_config.get("subcat_groups", {})
        ensemble_rules = models_config.get("ensemble_rules", {})

        statistical_models_keys = [m["id"] for m in models_meta if m["category"] == "statistical"]
        ml_models_keys = [m["id"] for m in models_meta if m["category"] == "ml"]
        ensemble_models_keys = [m["id"] for m in models_meta if m["category"] == "ensemble"]

        individual_keys = list(set(statistical_models_keys + ml_models_keys))
        all_core_models = individual_keys + ensemble_models_keys
        
        extended_selected_ids = list(selected_ids)
        for model_id in all_core_models:
            if model_id not in extended_selected_ids:
                extended_selected_ids.append(model_id)

        logger.info("Step 3/7: model keys loaded (statistical=%s ml=%s ensemble=%s)",
                    len(statistical_models_keys), len(ml_models_keys), len(ensemble_models_keys))
        logger.debug("Selected models extended: %s", extended_selected_ids)

        all_forecasts = {m_id: [] for m_id in extended_selected_ids}
        if has_existing:
            all_forecasts['existing'] = []

        # 3. 各時系列系列における予測値の並列計算
        for s in range(len(data_series)):
            train_data = train_data_series[s]
            last_val = train_data[-1] if train_data else 100.0

            n1_forecast = predict_statistical_model('naive1', train_data, horizon, sp, last_val)
            all_forecasts['naive1'].append(n1_forecast)
            logger.debug("Series %s naive1 forecast generated", s)

            for m_id in statistical_models_keys:
                if m_id == 'naive1':
                    continue
                forecast = predict_statistical_model(m_id, train_data, horizon, sp, last_val, naive1_forecast=n1_forecast)
                all_forecasts[m_id].append(forecast)
                logger.debug("Series %s statistical model %s forecast generated", s, m_id)

            for m_id in ml_models_keys:
                forecast = predict_ml_model(m_id, train_data, horizon, last_val, naive1_forecast=n1_forecast)
                all_forecasts[m_id].append(forecast)
                logger.debug("Series %s ml model %s forecast generated", s, m_id)

            if has_existing:
                existing_slice = impute_missing_values(existing_rule[s], strategy="ffill_bfill")[-horizon:]
                all_forecasts['existing'].append(existing_slice)
                logger.debug("Series %s existing rule series added", s)

        logger.info("Step 4/7: forecasts generated for %s series", len(data_series))

        # 4. 個別モデルの仮ランキングと順位の決定
        individual_forecasts_only = {k: all_forecasts[k] for k in individual_keys}
        temp_ranked_list = rank_models_by_owa(
            all_forecasts=individual_forecasts_only,
            actual_series_list=actual_val_data_series,
            mase_denominators=mase_denominators,
            baseline_id="naive2"
        )

        sorted_individual_ids = [item["id"] for item in temp_ranked_list]
        logger.info("Step 5/7: temporary ranking complete (%s individual models)", len(sorted_individual_ids))
        logger.debug("Temporary ranking order: %s", sorted_individual_ids)

        # 5. アンサンブル Comb 1 〜 6 の動的ルールベース適用
        for s in range(len(data_series)):
            for comb_id, rule_def in ensemble_rules.items():
                blend_forecast = calculate_dynamic_ensemble(
                    rule_def=rule_def,
                    all_forecasts=all_forecasts,
                    sorted_individual_ids=sorted_individual_ids,
                    subcat_groups=subcat_groups,
                    temp_ranked_list=temp_ranked_list,
                    horizon=horizon,
                    series_idx=s,
                    sp=sp
                )
                all_forecasts[comb_id].append(blend_forecast)
                logger.debug("Series %s ensemble model %s generated", s, comb_id)

        logger.info("Step 6/7: ensemble forecasts generated for %s series", len(data_series))

        # 6. アンサンブルを含んだ最終総合ランキング（OWA）を算出
        final_ranked_list = rank_models_by_owa(
            all_forecasts=all_forecasts,
            actual_series_list=actual_val_data_series,
            mase_denominators=mase_denominators,
            baseline_id="naive2"
        )
        logger.info("Step 7/7: final ranking complete (%s models evaluated)", len(final_ranked_list))

        # 7. レスポンスの構築
        final_results = []
        for rank_item in final_ranked_list:
            m_id = rank_item["id"]
            if m_id not in selected_ids and m_id != 'existing':
                continue

            clean_forecasts = [[None if np.isnan(x) else float(x) for x in f] for f in all_forecasts[m_id]]
            final_results.append(ModelResult(
                id=m_id,
                forecasts=clean_forecasts,
                mape=rank_item["avg_smape"],
                mase=rank_item["avg_mase"],
                owa=rank_item["owa"]
            ))

        logger.info("Response prepared: %s result models", len(final_results))
        return EvaluationResponse(
            status="success",
            train_data=[[float(x) for x in series] for series in train_data_series],
            actual_val_data=[[float(x) for x in series] for series in actual_val_data_series],
            results=final_results
        )

    except Exception as e:
        logger.exception("Error in evaluate_models")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)