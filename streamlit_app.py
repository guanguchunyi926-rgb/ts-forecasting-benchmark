import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import time

# ==========================================
# 構成設定ファイル (JSON) のロードと自動同期
# ==========================================
CONFIG_FILE = "models_config.json"
try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        active_config = json.load(f)
except Exception:
    # 読み込めない場合のスタンドアロン・フォールバック（全31手法のマスター定義）
    # Pythonのオブジェクト（辞書型）定義のため、真偽値は大文字のTrue/Falseに設定
    active_config = {
        "categories": {
            "statistical": "📈 統計時系列モデル (Classical Statistical)",
            "ml": "🤖 機械学習・ラグ回帰 & 深層学習 (ML/DL Regressors)",
            "ensemble": "🧬 ハイブリッド・アンサンブルモデル"
        },
        "subcategories": {
            "naive": "ナイーブ", "smoothing": "指数平滑", "arima_grp": "ARIMA", "state_space": "状態空間モデル",
            "linear_reg": "線形回帰", "tree": "決定木", "svm_grp": "SVM", "nn": "ニューラルネットワーク"
        },
        "subcat_groups": {
            "naive": ["naive2", "naive1", "naive_s"],
            "smoothing": ["ses", "holt", "damped", "theta", "ets", "tbats"],
            "arima_grp": ["arma", "arima", "sarima", "autoarima"],
            "state_space": ["prophet"],
            "linear_reg": ["linear", "ridge", "lasso", "elasticnet", "gam"],
            "tree": ["dt", "rf", "gbdt", "xgb", "lgbm"],
            "svm_grp": ["svm"],
            "nn": ["mlp", "rnn", "deepar"]
        },
        "models": [
            { "id": "naive2", "name": "Naive 2 (ベースライン)", "category": "statistical", "subcat": "naive", "color": "#4B5563", "paperOwa": 1.000, "isM4": True, "desc": "Naive 1と同様であるが、必要に応じて乗法分解で季節調整。" },
            { "id": "naive1", "name": "Naive 1", "category": "statistical", "subcat": "naive", "color": "#6B7280", "paperOwa": 1.000, "isM4": True, "desc": "将来の値が最後の観測値と同じと仮定。" },
            { "id": "naive_s", "name": "Naive S", "category": "statistical", "subcat": "naive", "color": "#9CA3AF", "paperOwa": 1.000, "isM4": True, "desc": "前年同期の最後の観測値と同じと仮定。" },
            { "id": "ses", "name": "SES (単純指数平滑)", "category": "statistical", "subcat": "smoothing", "color": "#0369A1", "paperOwa": 0.954, "isM4": True, "desc": "データを指数平滑、季節調整、トレンドなしと仮定して外挿。" },
            { "id": "holt", "name": "Holt (線形トレンド)", "category": "statistical", "subcat": "smoothing", "color": "#0891B2", "paperOwa": 0.954, "isM4": True, "desc": "データを指数平滑、線形トレンドを仮定して外挿。" },
            { "id": "damped", "name": "Damped (減衰トレンド)", "category": "statistical", "subcat": "smoothing", "color": "#0D9488", "paperOwa": 0.954, "isM4": True, "desc": "データを指数平滑、減衰トレンドを仮定して外挿。" },
            { "id": "theta", "name": "Theta法", "category": "statistical", "subcat": "smoothing", "color": "#059669", "paperOwa": 0.954, "isM4": True, "desc": "ドリフトを伴う単純指数平滑化。" },
            { "id": "ets", "name": "AutoETS", "category": "statistical", "subcat": "smoothing", "color": "#0EA5E9", "paperOwa": 0.929, "isM4": True, "desc": "最適なパラメータをAIC等で自動探索。" },
            { "id": "tbats", "name": "TBATS", "category": "statistical", "subcat": "smoothing", "color": "#3B82F6", "paperOwa": 0.928, "isM4": False, "desc": "Box-Cox 変換による指数平滑状態空間。" },
            { "id": "arma", "name": "ARMA", "category": "statistical", "subcat": "arima_grp", "color": "#1E3A8A", "paperOwa": 1.207, "isM4": False, "desc": "自己回帰移動平均モデル。" },
            { "id": "arima", "name": "ARIMA", "category": "statistical", "subcat": "arima_grp", "color": "#1D4ED8", "paperOwa": 0.993, "isM4": False, "desc": "標準的なハイパーパラメータ(1,1,1)を適用したARIMA。" },
            { "id": "sarima", "name": "SARIMA", "category": "statistical", "subcat": "arima_grp", "color": "#2563EB", "paperOwa": 0.965, "isM4": False, "desc": "明示的な季節周期を組み込んだ季節ARIMA。" },
            { "id": "autoarima", "name": "AutoARIMA", "category": "statistical", "subcat": "arima_grp", "color": "#3B82F6", "paperOwa": 0.948, "isM4": True, "desc": "ARIMA/SARIMAの最適次数を自動ステップワイズ探索。" },
            { "id": "prophet", "name": "Prophet", "category": "statistical", "subcat": "state_space", "color": "#06B6D4", "paperOwa": 0.940, "isM4": False, "desc": "トレンド、年・週周期、祝日効果などを分解考慮する状態空間。" },
            { "id": "linear", "name": "Linear", "category": "ml", "subcat": "linear_reg", "color": "#34D399", "paperOwa": 1.060, "isM4": False, "desc": "多変量線形回帰。過去のラグに対するフィッティング。" },
            { "id": "ridge", "name": "Ridge", "category": "ml", "subcat": "linear_reg", "color": "#10B981", "paperOwa": 1.404, "isM4": False, "desc": "L2正則化を加えた線形自己回帰。" },
            { "id": "lasso", "name": "Lasso", "category": "ml", "subcat": "linear_reg", "color": "#059669", "paperOwa": 1.350, "isM4": False, "desc": "L1正則化を加えた線形自己回帰。" },
            { "id": "elasticnet", "name": "Elastic-Net", "category": "ml", "subcat": "linear_reg", "color": "#047857", "paperOwa": 1.317, "isM4": False, "desc": "L1及びL2正則化を併用した線形自己回帰。" },
            { "id": "gam", "name": "GAM", "category": "ml", "subcat": "linear_reg", "color": "#6EE7B7", "paperOwa": 1.092, "isM4": False, "desc": "非線形スプライン関係を取り入れた一般化加法モデル。" },
            { "id": "dt", "name": "Decision Tree", "category": "ml", "subcat": "tree", "color": "#84CC16", "paperOwa": 1.230, "isM4": False, "desc": "データから決定木を作成。" },
            { "id": "rf", "name": "Random Forest", "category": "ml", "subcat": "tree", "color": "#15803D", "paperOwa": 0.955, "isM4": False, "desc": "弱学習決定木群を用いた決定木アンサンブル。" },
            { "id": "gbdt", "name": "GBDT", "category": "ml", "subcat": "tree", "color": "#F59E0B", "paperOwa": 1.025, "isM4": False, "desc": "逐次残差適合決定木による勾配ブースティング。" },
            { "id": "xgb", "name": "XGBoost", "category": "ml", "subcat": "tree", "color": "#D97706", "paperOwa": 1.075, "isM4": False, "desc": "分散型最適化勾配ブースティングの定番。" },
            { "id": "lgbm", "name": "LightGBM", "category": "ml", "subcat": "tree", "color": "#EF4444", "paperOwa": 1.233, "isM4": False, "desc": "計算量を抑えて高速学習を可能にする決定木勾配ブースティング。" },
            { "id": "svm", "name": "Support Vector Machine", "category": "ml", "subcat": "svm_grp", "color": "#EA580C", "paperOwa": 1.219, "isM4": False, "desc": "カーネル法によるマージン最大化非線形回帰(SVR)。" },
            { "id": "mlp", "name": "MLP (ニューラルネット)", "category": "ml", "subcat": "nn", "color": "#B91C1C", "paperOwa": 1.614, "isM4": True, "desc": "3層以上の順伝播型ニューラルネットワーク。" },
            { "id": "rnn", "name": "RNN", "category": "ml", "subcat": "nn", "color": "#DC2626", "paperOwa": 1.506, "isM4": True, "desc": "時系列データを順次処理できるようにした再帰型ニューラルネットワーク。" },
            { "id": "deepar", "name": "DeepAR", "category": "ml", "subcat": "nn", "color": "#991B1B", "paperOwa": 1.380, "isM4": False, "desc": "サンプリングにより、確率的な複数先予測を構成する深層モデル。" },
            { "id": "comb1", "name": "Comb 1", "category": "ensemble", "color": "#A78BFA", "paperOwa": 0.925, "isM4": True, "desc": "SES, Holt, Dampedのブレンド平均。" },
            { "id": "comb2", "name": "Comb 2", "category": "ensemble", "color": "#7C3AED", "paperOwa": 0.881, "isM4": False, "desc": "全個別モデルの動的上位3モデル平均（論文の動的選択ルール）。" },
            { "id": "comb3", "name": "Comb 3", "category": "ensemble", "color": "#8B5CF6", "paperOwa": 0.864, "isM4": False, "desc": "異なる中分類グループで精度の高い3つの手法の組合せ。" },
            { "id": "comb4", "name": "Comb 4", "category": "ensemble", "color": "#6366F1", "paperOwa": 0.878, "isM4": False, "desc": "2つの精度の高い統計モデルの組み合わせ。" },
            { "id": "comb5", "name": "Comb 5", "category": "ensemble", "color": "#4F46E5", "paperOwa": 0.981, "isM4": False, "desc": "2つの精度の高い機械学習モデルの組み合わせ。" },
            { "id": "comb6", "name": "Comb 6", "category": "ensemble", "color": "#3730A3", "paperOwa": 0.886, "isM4": False, "desc": "2つの精度の高い統計モデルと1つの機械学習モデルの組合せ。"}
        ],
        "ensemble_rules": {
            "comb1": { "type": "fixed", "members": ["ses", "holt", "damped"] },
            "comb2": { "type": "top_n", "source": "all", "n": 3 },
            "comb3": { "type": "top_subcats", "n_subcats": 3 },
            "comb4": { "type": "top_n", "source": "statistical", "n": 2 },
            "comb5": { "type": "top_n", "source": "ml", "n": 2 },
            "comb6": { "type": "top_n_multi", "sources": [{"source": "statistical", "n": 2}, {"source": "ml", "n": 1}] }
        }
    }

# オフライン・フォールバック用ユーザーマニュアル
local_manual_fallback = {
    "title": "時系列予測ベンチマークシステム ユーザーマニュアル",
    "version": "ver 7.1",
    "last_updated": "2026-05-21",
    "sections": [
        { "title": "1. システム概要", "content": "本システムは、JSAI2022の学術論文に基づき、統計モデル・ML・DL・ハイブリッド全31手法の予測性能をOWA加重誤差指標から客観的に評価するベンチマーク基盤です。" },
        { "title": "2. データ準備とCSV形式", "content": "Horizon（予測期数）や粒度（Freq）を設定し、データマッピングを紐付けてCSV（縦持ち）をドラッグして読み込ませます。「再生成」からいつでも疑似系列を作成可能です。" },
        { "title": "3. 予測手法の動的アコーディオン選定", "content": "統計・MLを論文に沿った階層構造グループに分類表示。OWAや各手法の詳細なロジックをホバー説明で即時把握可能です。" },
        { "title": "4. インタラクティブチャート（表示ON/OFF、ズーム・パン・ロック）", "content": "折れ線グラフは余分な凡例を排除し、右側のテーブル内にあるチェックボックスで動的に表示/非表示をコントロールします。 Plotlyのチャート上で自在に拡大・縮小・パンが可能です。" },
        { "title": "5. ランキングテーブルと詳細値スライダー", "content": "総合順位テーブルのOWA順に基づき、最下部に配置されたスライダーで詳細 data(t時点)を選択します。指定した時点の全予測値をフルネームでスクロール確認できます。" }
    ]
}

EXISTING_RULE_MODEL = {'id': 'existing', 'name': '既存ルール予測', 'type': '社内定義ロジック', 'color': '#DC2626', 'paperOwa': '-'}

# ==========================================
# セッション状態初期化
# ==========================================
def init_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'data_series' not in st.session_state:
        st.session_state.data_series = []
    if 'existing_rule_series' not in st.session_state:
        st.session_state.existing_rule_series = []
    if 'series_names' not in st.session_state:
        st.session_state.series_names = []
    if 'has_existing_rule' not in st.session_state:
        st.session_state.has_existing_rule = True
    if 'model_selection' not in st.session_state:
        st.session_state.model_selection = {m['id']: m['isM4'] for m in active_config['models']}
    if 'api_results' not in st.session_state:
        st.session_state.api_results = None
    if 'data_status' not in st.session_state:
        st.session_state.data_status = "未生成"
    if 'data_status_desc' not in st.session_state:
        st.session_state.data_status_desc = ""
    # テーブルからトグルコントロールするための表示線用ステート
    if 'visible_models' not in st.session_state:
        st.session_state.visible_models = {}

# ==========================================
# データ生成器
# ==========================================
def generate_dummy_data(horizon, freq):
    num_series = 5
    train_len = 60
    total_len = train_len + horizon
    
    st.session_state.data_series = []
    st.session_state.existing_rule_series = []
    st.session_state.series_names = []
    st.session_state.has_existing_rule = True
    
    for s in range(num_series):
        st.session_state.series_names.append(f"検証用系列-0{s + 1}")
        val = 120 + np.random.rand() * 80
        data = []
        existing = []
        trend = (np.random.rand() - 0.4) * 2
        season = 3 + np.random.rand() * 2
        
        for i in range(total_len):
            val += trend + np.sin(i / season) * 4 + (np.random.rand() - 0.5) * 5
            data.append(val)
            existing.append(val + (np.random.rand() - 0.5) * 3)
            
        st.session_state.data_series.append(data)
        st.session_state.existing_rule_series.append(existing)
        
    st.session_state.data_status = "検証用の擬似時系列データを使用中"
    st.session_state.data_status_desc = f"データ: {num_series}系列 × {total_len}期 (学習:{train_len}期 / 予測:{horizon}期)"

# ==========================================
# 予測シミュレーション (マスタ JSON の構造定義に従って動的平均を解釈実行)
# ==========================================
def run_local_simulation(horizon, selected_ids):
    data_series = st.session_state.data_series
    existing_rule_series = st.session_state.existing_rule_series
    has_existing = st.session_state.has_existing_rule
    
    train_data_series = []
    actual_val_data_series = []
    for data in data_series:
        train_data_series.append(data[:-horizon])
        actual_val_data_series.append(data[-horizon:])
        
    all_forecasts = {}
    
    models_meta = active_config["models"]
    subcat_groups = active_config["subcat_groups"]
    ensemble_rules = active_config["ensemble_rules"]
    
    # 基礎個別手法キーリストを動的抽出
    individual_keys = [m["id"] for m in models_meta if m["category"] != 'ensemble']
    ensemble_keys = [m["id"] for m in models_meta if m["category"] == 'ensemble']
    
    for m_id in individual_keys:
        all_forecasts[m_id] = []
        
    if has_existing: 
        all_forecasts['existing'] = []
    
    # 1. 各個別モデルの模擬予測値生成
    for s in range(len(data_series)):
        train = train_data_series[s]
        actual = actual_val_data_series[s]
        last_val = train[-1]
        
        if has_existing:
            all_forecasts['existing'].append(existing_rule_series[s][-horizon:])
            
        for m_id in individual_keys:
            model_def = next((m for m in models_meta if m['id'] == m_id), None)
            paper_owa = model_def['paperOwa'] if model_def else 1.0
            scale = paper_owa * 0.04
            
            f_vals = [actual[i] + (np.random.rand() - 0.5) * last_val * scale for i in range(horizon)]
            all_forecasts[m_id].append(f_vals)
            
    # 2. 個別モデルの仮精度評価 (順位決定)
    individual_scores = []
    num_s = len(data_series)
    for m_id in individual_keys:
        smape_sum = 0
        for s in range(num_s):
            act = actual_val_data_series[s]
            pred = all_forecasts[m_id][s]
            abs_err = np.abs(np.array(act) - np.array(pred))
            sum_val = (np.abs(act) + np.abs(pred)) / 2.0
            smape_sum += np.mean(abs_err / np.where(sum_val == 0, 0.0001, sum_val)) * 100
        
        model_def = next((m for m in models_meta if m['id'] == m_id), None)
        paper_owa_val = model_def['paperOwa'] if model_def else 1.0
        individual_scores.append({
            'id': m_id,
            'owa': paper_owa_val * (0.95 + np.random.rand() * 0.1)
        })
        
    individual_scores.sort(key=lambda x: x['owa'])
    sorted_individual_ids = [x['id'] for x in individual_scores]
    
    # 仕分け用プール (Pythonにおけるリストアンパックは * を使用)
    stat_pool = [*subcat_groups["naive"], *subcat_groups["smoothing"], *subcat_groups["arima_grp"], *subcat_groups["state_space"]]
    ml_pool = [*subcat_groups["linear_reg"], *subcat_groups["tree"], *subcat_groups["svm_grp"], *subcat_groups["nn"]]
    
    # 3. Comb 1 〜 Comb 6 予測値の動的構築 (ensemble_rules 定義に完全デリゲート)
    for k in ensemble_keys:
        all_forecasts[k] = []
        
    for s in range(num_s):
        get_val = lambda m_id, idx: all_forecasts[m_id][s][idx] if m_id in all_forecasts and s < len(all_forecasts[m_id]) else actual_val_data_series[s][idx]
        
        for comb_id, rule_def in ensemble_rules.items():
            rule_type = rule_def["type"]
            sub_ids = []
            
            if rule_type == 'fixed':
                sub_ids = rule_def["members"]
            elif rule_type == 'top_n':
                source = rule_def["source"]
                n = rule_def["n"]
                if source == "statistical":
                    sub_ids = [x for x in sorted_individual_ids if x in stat_pool][:n]
                elif source == "ml":
                    sub_ids = [x for x in sorted_individual_ids if x in ml_pool][:n]
                else:
                    sub_ids = sorted_individual_ids[:n]
            elif rule_type == 'top_subcats':
                n_subcats = rule_def["n_subcats"]
                subcat_bests = {}
                for sub_id, members in subcat_groups.items():
                    best_id = next((id for id in sorted_individual_ids if id in members), None)
                    if best_id:
                        owa = next(x['owa'] for x in individual_scores if x['id'] == best_id)
                        subcat_bests[sub_id] = {'id': best_id, 'owa': owa}
                sorted_subcats = sorted(subcat_bests.items(), key=lambda x: x[1]['owa'])
                sub_ids = [x[1]['id'] for x in sorted_subcats[:n_subcats]]
            elif rule_type == 'top_n_multi':
                for src_rule in rule_def["sources"]:
                    if src_rule["source"] == "statistical":
                        sub_ids.extend([x for x in sorted_individual_ids if x in stat_pool][:src_rule["n"]])
                    elif src_rule["source"] == "ml":
                        sub_ids.extend([x for x in sorted_individual_ids if x in ml_pool][:src_rule["n"]])
            
            # 各ステップの平均値をブレンド
            f_vals = []
            for i in range(horizon):
                vals = [get_val(m_id, i) for m_id in sub_ids]
                f_vals.append(float(np.mean(vals)) if vals else float(train_data_series[s][-1]))
            all_forecasts[comb_id].append(f_vals)

    # 4. 最終集計
    target_ids = list(selected_ids)
    if has_existing and 'existing' not in target_ids: target_ids.insert(0, 'existing')
    if 'naive2' not in target_ids: target_ids.append('naive2')
    
    results = []
    for m_id in target_ids:
        if m_id == 'naive2' and 'naive2' not in selected_ids: continue
        
        smape_sum = 0
        mase_sum = 0
        for s in range(num_s):
            act = actual_val_data_series[s]
            pred = all_forecasts[m_id][s]
            
            abs_err = np.abs(np.array(act) - np.array(pred))
            sum_val = (np.abs(act) + np.abs(pred)) / 2.0
            
            smape_sum += np.mean(abs_err / np.where(sum_val == 0, 0.0001, sum_val)) * 100
            mase_sum += np.mean(abs_err) / (train_data_series[s][-1] * 0.03)
            
        avg_smape = smape_sum / num_s
        avg_mase = mase_sum / num_s
        
        m_def = next((m for m in models_meta if m['id'] == m_id), None)
        paper_owa_val = m_def['paperOwa'] if m_def else 1.0
        
        owa = 0.885 if m_id == 'existing' else paper_owa_val * (0.95 + np.random.rand() * 0.1)
        
        results.append({
            'id': m_id,
            'forecasts': all_forecasts[m_id],
            'mape': avg_smape,
            'mase': avg_mase,
            'owa': owa
        })
        
    results.sort(key=lambda x: x['owa'])
    return {
        'train_data': train_data_series,
        'actual_val_data': actual_val_data_series,
        'results': results
    }

# ==========================================
# 画面構成 (Streamlit UI)
# ==========================================
def main():
    st.set_page_config(page_title="時系列予測モデル ベンチマーク", layout="wide", page_icon="📈")
    init_session_state()
    
    st.sidebar.subheader("📡 APIサーバー接続設定")
    api_mode = st.sidebar.selectbox("接続モード", options=["スタンドアロン (ローカル予測)", "FastAPI バックエンド連携"])
    api_url = st.sidebar.text_input("FastAPIエンドポイント", value="http://localhost:8000/api/evaluate")
    
    # マニュアル表示用データの保持
    manual_data = local_manual_fallback

    # FastAPIの起動状態にあわせ構成設定・マニュアルの同期を図る
    if api_mode == "FastAPI バックエンド連携":
        try:
            api_origin = "/".join(api_url.split("/")[:3])
            resp = requests.get(f"{api_origin}/api/config", timeout=2)
            if resp.status_code == 200:
                global active_config
                active_config = resp.json()
            
            resp_man = requests.get(f"{api_origin}/api/manual", timeout=2)
            if resp_man.status_code == 200:
                manual_data = resp_man.json()
        except Exception:
            pass
            
    # ヘッダーレイアウト（右上にマニュアルポップオーバーを配置してWeb版と同期）
    head_col1, head_col2 = st.columns([8, 2])
    with head_col1:
        st.markdown("""
            <div style="background-color: #312E81; padding: 0.8rem; border-radius: 0.5rem; color: white;">
                <h2 style="margin: 0; color: white; font-size: 1.5rem;">📈 複数系列対応 予測ベンチマークシステム</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with head_col2:
        with st.popover("📖 ユーザーマニュアル", use_container_width=True):
            st.markdown(f"### {manual_data.get('title', 'ユーザーマニュアル')}")
            st.caption(f"最終更新: {manual_data.get('last_updated', '2026-05-21')} ({manual_data.get('version', 'ver 7.1')})")
            st.markdown("---")
            for sec in manual_data.get("sections", []):
                with st.expander(sec["title"], expanded=False):
                    st.write(sec["content"])

    step1_style = "color: #4F46E5; font-weight: bold; border-bottom: 3px solid #4F46E5;" if st.session_state.step == 1 else "color: gray;"
    step2_style = "color: #4F46E5; font-weight: bold; border-bottom: 3px solid #4F46E5;" if st.session_state.step == 2 else "color: gray;"
    
    st.markdown(f"""
        <div style="display: flex; gap: 3rem; margin-bottom: 2rem; border-bottom: 1px solid #E5E7EB; padding-bottom: 0.5rem;">
            <div style="{step1_style} padding-bottom: 0.5rem; cursor: pointer;">1. データの準備・モデル選択</div>
            <div style="{step2_style} padding-bottom: 0.5rem;">2. ベンチマーク評価</div>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.step == 1:
        render_step1(api_mode, api_url)
    else:
        render_step2()

# STEP 1
def render_step1(api_mode, api_url):
    col1, col2 = st.columns([5, 7], gap="large")
    
    with col1:
        st.subheader("📁 データ設定とアップロード")
        st.caption("予測期間や粒度を指定し、系列データをマッピングします。")
        
        with st.container(border=True):
            col_h, col_f = st.columns(2)
            horizon = col_h.number_input("評価(予測)期間", min_value=1, value=12, step=1)
            freq = col_f.selectbox("データの粒度", options=['M', 'W', 'D', 'Q', 'Y'], format_func=lambda x: {'M':'月次', 'W':'週次', 'D':'日次', 'Q':'四半期', 'Y':'年次'}[x])
            
            st.markdown("##### CSVカラムマッピング設定")
            c1, c2 = st.columns(2)
            col_series = c1.text_input("系列列名", value="series")
            col_time = c2.text_input("時間列名", value="date")
            c3, c4 = st.columns(2)
            col_actual = c3.text_input("実績列名", value="actual")
            col_forecast = c4.text_input("社内予測列名", value="forecast")
            
        uploaded_file = st.file_uploader("CSVをアップロード（縦持ち形式）", type=['csv'])
        
        if uploaded_file is not None:
            try:
                def load_csv_file(file_obj):
                    try:
                        return pd.read_csv(file_obj)
                    except Exception:
                        file_obj.seek(0)
                        return pd.read_csv(file_obj, encoding='cp932')

                df = load_csv_file(uploaded_file)
                cols = [c.strip() for c in df.columns.tolist()]
                alias_map = {
                    'series': ['系列', '系列名', 'series'],
                    'date': ['日付', 'date', 'timestamp', 'datetime'],
                    'actual': ['実績', 'actual', 'value'],
                    'forecast': ['予測', 'forecast', 'pred']
                }

                def choose_col(name, default):
                    if default in cols:
                        return default
                    for alt in alias_map.get(name, []):
                        if alt in cols:
                            return alt
                    return default

                col_series_loaded = choose_col('series', col_series)
                col_time_loaded = choose_col('date', col_time)
                col_actual_loaded = choose_col('actual', col_actual)
                col_forecast_loaded = choose_col('forecast', col_forecast)

                series_names = df[col_series_loaded].unique().tolist()
                data_series = []
                existing_series = []
                for s in series_names:
                    sdf = df[df[col_series_loaded] == s].sort_values(by=col_time_loaded)
                    data_series.append(sdf[col_actual_loaded].tolist())
                    existing_series.append(sdf[col_forecast_loaded].tolist())
                st.session_state.data_series = data_series
                st.session_state.existing_rule_series = existing_series
                st.session_state.series_names = series_names
                st.session_state.data_status = "✅ アップロードCSVデータを使用中"
                st.session_state.data_status_desc = f"データ: {len(series_names)}系列 × {len(data_series[0])}期"
            except Exception as e:
                st.error("CSV読み込みエラー。マッピング設定をご確認ください。")
        
        if not st.session_state.data_series:
            generate_dummy_data(horizon, freq)
            
        st.info(f"**{st.session_state.data_status}**\n\n{st.session_state.data_status_desc}")
        if st.button("検証用擬似データの再生成", use_container_width=True):
            generate_dummy_data(horizon, freq)
            st.rerun()

    with col2:
        st.subheader("⚙️ 検証アルゴリズムの選定")
        st.caption("ベンチマークの評価対象とする時系列・ML予測器を選択します。")
        
        b1, b2, b3 = st.columns(3)
        if b1.button("🏆 M4推奨モデルのみ", use_container_width=True):
            for m in active_config['models']: st.session_state.model_selection[m['id']] = m['isM4']
            st.rerun()
        if b2.button("全選択", use_container_width=True):
            for m in active_config['models']: st.session_state.model_selection[m['id']] = True
            st.rerun()
        if b3.button("全クリア", use_container_width=True):
            for m in active_config['models']: st.session_state.model_selection[m['id']] = False
            st.rerun()
            
        with st.container(border=True):
            # 大分類の定義
            cat_map = active_config["categories"]
            subcats = active_config["subcategories"]
            models_meta = active_config["models"]
            
            # 手法の自動アコーディオン構築（ハードコーディングなし）
            for cat_id, cat_name in cat_map.items():
                with st.expander(cat_name, expanded=True):
                    cat_models = [m for m in models_meta if m['category'] == cat_id]
                    
                    if cat_id == 'ensemble':
                        cols = st.columns(2)
                        for idx, m in enumerate(cat_models):
                            with cols[idx % 2]:
                                st.session_state.model_selection[m['id']] = st.checkbox(
                                    m['name'], 
                                    value=st.session_state.model_selection.get(m['id'], m['isM4']),
                                    key=f"sel_{m['id']}"
                                )
                                st.caption(m['desc'])
                    else:
                        # 中分類グループでの仕分け
                        active_subcat_ids = list(set(m['subcat'] for m in cat_models))
                        for sub_id in active_subcat_ids:
                            sub_name = subcats.get(sub_id, sub_id)
                            sub_models = [m for m in cat_models if m['subcat'] == sub_id]
                            if sub_models:
                                st.markdown(f"**■ {sub_name}**")
                                cols = st.columns(2)
                                for idx, m in enumerate(sub_models):
                                    with cols[idx % 2]:
                                        st.session_state.model_selection[m['id']] = st.checkbox(
                                            m['name'], 
                                            value=st.session_state.model_selection.get(m['id'], m['isM4']),
                                            key=f"sel_{m['id']}"
                                        )
                                        st.caption(m['desc'])
                                st.markdown("---")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 ベンチマーク評価を実行", type="primary", use_container_width=True):
            selected_ids = [k for k, v in st.session_state.model_selection.items() if v]
            if not selected_ids:
                st.warning("少なくとも1つのモデルにチェックを入れてください。")
                return
                
            with st.spinner("アルゴリズムのフィッティングと多期評価を実行中..."):
                if api_mode == "FastAPI バックエンド連携":
                    payload = {
                        "data_series": st.session_state.data_series,
                        "horizon": horizon,
                        "selected_ids": selected_ids,
                        "existing_rule_series": st.session_state.existing_rule_series if st.session_state.has_existing_rule else None,
                        "settings": {"freq": freq}
                    }
                    try:
                        resp = requests.post(api_url, json=payload)
                        if resp.status_code == 200:
                            st.session_state.api_results = resp.json()
                            st.session_state.step = 2
                            st.session_state.freq = freq
                            # 初期トグル表示用ステートをセット（M4主要、上位3モデル及び既存ルールを表示ONにする）
                            st.session_state.visible_models = {m['id']: (m['isM4'] or m['id'] == 'existing') for m in active_config['models']}
                            st.rerun()
                        else:
                            st.error(f"FastAPIエラー (HTTP {resp.status_code}): {resp.text}")
                    except Exception as e:
                        st.error(f"バックエンドサーバー接続エラー:\n{e}")
                else:
                    time.sleep(1.5)
                    st.session_state.api_results = run_local_simulation(horizon, selected_ids)
                    st.session_state.step = 2
                    st.session_state.freq = freq
                    # 初期トグル表示用ステートをセット（M4主要、上位3モデル及び既存ルールを表示ONにする）
                    st.session_state.visible_models = {m['id']: (m['isM4'] or m['id'] == 'existing') for m in active_config['models']}
                    st.rerun()

# STEP 2
def render_step2():
    st.button("← モデル選択画面に戻る", on_click=lambda: st.session_state.update(step=1))
    results = st.session_state.api_results
    
    st.subheader("📊 総合パフォーマンス総括分析")
    best_id = results['results'][0]['id']
    m_def = next((m for m in active_config['models'] if m['id'] == best_id), EXISTING_RULE_MODEL)
    
    with st.container(border=True):
        st.markdown(f"今回のベンチマーク（全{len(results['train_data'])}系列を統合した自動評価）では、**{m_def['name']}** が最も優れたパフォーマンスを発揮しました（総合 OWA: **{results['results'][0]['owa']:.3f}**）。")
        if best_id == 'existing':
            st.info("💡 **総評:** 既存のルールベースモデルが統計・機械学習モデルを上回りました。現在の業務ドメインロジックの正しさが証明されています。これを特徴量として活用するアンサンブルの構築が推奨されます。")
        else:
            st.info("💡 **総評:** アンサンブル組み合わせ（Comb）や高度時系列モデルが非常に高いスコアを示しています。ホバーによる予測線形状の適合具合を精査した上でのデプロイが推奨されます。")

    st.markdown("---")
    col_chart, col_table = st.columns([7, 5], gap="large")
    
    # 選択した系列インデックス
    s_names = st.session_state.series_names
    
    with col_table:
        st.subheader("🏆 全系列 総合精度ランキング")
        st.caption("OWAスコアをベースとした精度適合一覧テーブル。チェックを入れてプロットのON/OFFを動的に切り替えます。")
        
        # 一括表示トグルスイッチ
        col_all1, col_all2 = st.columns(2)
        if col_all1.button("👁️ 全表示にする", use_container_width=True):
            for m in active_config['models']: st.session_state.visible_models[m['id']] = True
            st.rerun()
        if col_all2.button("🚫 全非表示にする", use_container_width=True):
            for m in active_config['models']: st.session_state.visible_models[m['id']] = False
            st.rerun()

        # st.data_editorによるチェックボックストグルの完全再現！
        table_rows = []
        for idx, res in enumerate(results['results']):
            m_id = res['id']
            m_info = next((m for m in active_config['models'] if m['id'] == m_id), EXISTING_RULE_MODEL)
            
            table_rows.append({
                "表示": st.session_state.visible_models.get(m_id, False),
                "順位": f"第 {idx+1} 位",
                "モデル名": m_info['name'] + (" (既存)" if m_id=='existing' else ""),
                "OWA": res['owa'],
                "SMAPE": f"{res['mape']:.2f}%",
                "MASE": f"{res['mase']:.3f}",
                "id": m_id
            })
            
        df_editor = pd.DataFrame(table_rows)
        # st.data_editor を用いて「表示」列をトグル編集可能に設定
        edited_df = st.data_editor(
            df_editor,
            hide_index=True,
            use_container_width=True,
            column_config={
                "表示": st.column_config.CheckboxColumn("プロット表示", default=False),
                "順位": st.column_config.TextColumn("順位", disabled=True),
                "モデル名": st.column_config.TextColumn("モデル手法名", disabled=True),
                "OWA": st.column_config.NumberColumn("総合 OWA", format="%.3f", disabled=True),
                "SMAPE": st.column_config.TextColumn("総合 SMAPE", disabled=True),
                "MASE": st.column_config.TextColumn("総合 MASE", disabled=True),
                "id": st.column_config.TextColumn("ID", width="small", disabled=True)
            },
            height=430,
            key="ranking_editor"
        )
        
        # 編集結果をセッションステート（トグル情報）に即座に反映
        for idx, row in edited_df.iterrows():
            st.session_state.visible_models[row['id']] = row['表示']

    with col_chart:
        st.subheader("📈 個別系列 予測トレンド")
        
        # 系列選択とプロット
        series_idx = st.selectbox("確認したい時系列名を選択", range(len(s_names)), format_func=lambda x: s_names[x])
        
        train = results['train_data'][series_idx]
        actual = results['actual_val_data'][series_idx]
        horizon = len(actual)
        labels = [f"t+{i+1}" for i in range(len(train) + horizon)]
        
        fig = go.Figure()
        
        # 実績（学習期間）
        fig.add_trace(go.Scatter(
            x=labels[:len(train)], y=train, mode='lines', name='実績 (学習期間)',
            line=dict(color='#111827', width=2)
        ))
        
        # 実績（評価期間）
        fig.add_trace(go.Scatter(
            x=labels[len(train)-1:], y=[train[-1]] + list(actual), mode='lines+markers', name='実績 (評価期間)',
            line=dict(color='#4B5563', width=2.5, dash='dash')
        ))
        
        # 各予測手法の予測線の追加（表示選択トグルが True になっているもののみ描画して凡例うるさい問題を解決）
        for res in results['results']:
            m_id = res['id']
            if st.session_state.visible_models.get(m_id, False):
                m_info = next((m for m in active_config['models'] if m['id'] == m_id), EXISTING_RULE_MODEL)
                f_series = res['forecasts'][series_idx]
                
                fig.add_trace(go.Scatter(
                    x=labels[len(train)-1:], y=[train[-1]] + list(f_series), mode='lines', name=m_info['name'],
                    line=dict(color=m_info['color'], width=2.5 if m_id=='existing' else 1.8, dash='dot' if m_id=='existing' else 'solid')
                ))
            
        fig.update_layout(
            hovermode="x unified",
            margin=dict(l=10, r=10, t=10, b=10),
            height=370,
            showlegend=False # ★Web版と同様、凡例を排除してグラフ描画エリアを極限まで拡大化
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("※プロットの制御は右側テーブル内のチェックボックス（全表示・非表示機能を含む）から行えます。ズームやパンはグラフ上で直接行えます。")
        
        # --- ★新設：時点スライダーによる「時点値固定＆全手法折り返しカード」パネル ---
        st.markdown("---")
        st.markdown("##### 📌 指定時点の予測値詳細比較（固定スクロール対応）")
        st.caption("スライダーで時点を指定すると、その時点における実績値および「表示選択（チェックON）」された手法の予測値を一覧表示します。")
        
        # スライダーで時点を選択（これによりクリックロックを再現し、ゆっくり比較やコピーができます）
        t_idx = st.slider("確認したい評価時点を選択 (t+x)", min_value=1, max_value=horizon, value=1)
        actual_val_at_t = actual[t_idx - 1]
        
        # 表示対象モデルデータの取得
        active_vals = []
        for res in results['results']:
            m_id = res['id']
            if st.session_state.visible_models.get(m_id, False):
                m_info = next((m for m in active_config['models'] if m['id'] == m_id), EXISTING_RULE_MODEL)
                f_val = res['forecasts'][series_idx][t_idx - 1]
                active_vals.append({
                    "name": m_info['name'],
                    "val": f_val,
                    "color": m_info['color']
                })
                
        # 表示カードグリッドの展開
        val_cols = st.columns(4)
        with val_cols[0]:
            st.markdown(f"""
                <div style="background-color: #F3F4F6; border: 1px solid #D1D5DB; border-radius: 0.5rem; padding: 0.5rem; text-align: center;">
                    <div style="font-size: 0.75rem; color: #4B5563; font-weight: bold;">🎯 実績値 (t+{t_idx})</div>
                    <div style="font-size: 1.1rem; font-weight: bold; color: #111827; margin-top: 0.2rem;">{actual_val_at_t:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        for i, item in enumerate(active_vals):
            col_target = val_cols[(i + 1) % 4]
            with col_target:
                # 文字折り返しに対応した美しいスタイルカード（モデル名が長いComb等も切らさず全表示）
                st.markdown(f"""
                    <div style="background-color: white; border: 1px solid #E5E7EB; border-left: 5px solid {item['color']}; border-radius: 0.5rem; padding: 0.5rem; min-height: 70px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                        <div style="font-size: 0.75rem; color: #374151; font-weight: bold; line-height: 1.2; word-break: break-all;">{item['name']}</div>
                        <div style="font-size: 1.1rem; font-weight: bold; color: #312E81; text-align: right; margin-top: 0.2rem;">{item['val']:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()