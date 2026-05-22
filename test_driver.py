import json
import os
import csv
import sys
import requests
from typing import Dict, Any, List

def load_json_config(filepath: str) -> Dict[str, Any]:
    """設定ファイルをロードします。"""
    if not os.path.exists(filepath):
        print(f"[-] エラー: 設定ファイル '{filepath}' が見つかりません。")
        sys.exit(1)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[-] エラー: 設定ファイルの解析に失敗しました: {e}")
        sys.exit(1)

def run_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """FastAPIバックエンドへ予測・評価のリクエストを送信します。"""
    api_url = config.get("api_url", "http://localhost:8000/api/evaluate")
    
    # リクエストペイロードの成形 (FastAPIのスキーマに準拠)
    payload = {
        "data_series": config.get("data_series", []),
        "horizon": config.get("horizon", 12),
        "selected_ids": config.get("selected_ids", []),
        "existing_rule_series": config.get("existing_rule_series"),
        "settings": config.get("settings", {"freq": "M"})
    }
    
    print(f"[*] APIサーバーへ接続中... URL: {api_url}")
    try:
        response = requests.post(api_url, json=payload, timeout=60)
        if response.status_code != 200:
            print(f"[-] エラー: APIサーバーが異常応答を返しました (HTTP {response.status_code})")
            print(f"[-] 詳細: {response.text}")
            sys.exit(1)
        return response.json()
    except requests.exceptions.ConnectionError:
        print("[-] エラー: APIサーバーに接続できません。FastAPI(main.py)が起動しているか確認してください。")
        sys.exit(1)
    except Exception as e:
        print(f"[-] 予測実行中に例外が発生しました: {e}")
        sys.exit(1)

def export_results_to_csv(api_response: Dict[str, Any], output_score_path: str, output_forecast_path: str):
    """
    評価結果のOWA精度ランキングと、各系列ごとの予測値をマージしたCSVを出力します。
    """
    results_list = api_response.get("results", [])
    train_data = api_response.get("train_data", [])
    actual_val_data = api_response.get("actual_val_data", [])
    
    if not results_list:
        print("[-] 評価結果にデータが存在しません。")
        return

    # 1. 総合評価スコア一覧（OWA順）をCSVへ書き出し
    print(f"[*] 総合OWAランキングをエクスポート中: {output_score_path}")
    try:
        with open(output_score_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["順位", "モデルID", "SMAPE(平均対称絶対誤差 %)", "MASE(平均絶対スケーリング誤差)", "OWA(総合加重精度スコア)"])
            for rank, item in enumerate(results_list, 1):
                writer.writerow([
                    rank,
                    item["id"],
                    f"{item['mape']:.4f}%",
                    f"{item['mase']:.5f}",
                    f"{item['owa']:.5f}"
                ])
        print(f"[+] OWA精度ランキングCSVを出力しました。")
    except Exception as e:
        print(f"[-] 精度ランキングの出力に失敗しました: {e}")

    # 2. 各系列・時点ごとの「実績値」と「表示線選択予測値」を結合した時系列予測生CSV
    print(f"[*] 時点ごとの予測推移データをエクスポート中: {output_forecast_path}")
    try:
        num_series = len(train_data)
        horizon = len(actual_val_data[0]) if actual_val_data else 0
        
        with open(output_forecast_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            # ヘッダー構成: [系列ID, 時点タイプ(学習/評価), 時点番号, 実実績値, モデル1予測値, モデル2予測値...]
            headers = ["系列ID", "時間タイプ", "時点番号(t+x)", "実績値"]
            model_ids = [item["id"] for item in results_list]
            headers.extend(model_ids)
            writer.writerow(headers)

            # 各時系列ごとに実数と全モデルの予測値をCSVのレコードとして格納
            for s in range(num_series):
                train_len = len(train_data[s])
                
                # A. 学習データ期間 (予測値は存在しないため空欄)
                for i, val in enumerate(train_data[s]):
                    row = [f"series_{s}", "学習用過去", f"t+{i+1}", f"{val:.4f}"]
                    row.extend([""] * len(model_ids))
                    writer.writerow(row)
                
                # B. 評価期間 (実績値に加え、全評価されたモデルの予測値をマッピング)
                for i in range(horizon):
                    actual_val = actual_val_data[s][i]
                    row = [f"series_{s}", "評価予測", f"t+{train_len + i + 1}", f"{actual_val:.4f}"]
                    
                    for m_id in model_ids:
                        # 対象モデルの系列s、時点iにおける予測値を取得
                        model_item = next((item for item in results_list if item["id"] == m_id), None)
                        pred_val = model_item["forecasts"][s][i] if model_item else None
                        row.append(f"{pred_val:.4f}" if pred_val is not None else "")
                    writer.writerow(row)
                    
        print(f"[+] 予測時系列データCSVを出力しました。")
    except Exception as e:
        print(f"[-] 予測データの出力に失敗しました: {e}")

def main():
    print("="*60)
    print(" 複数系列対応時系列予測システム: 自動API検証ドライバーツール")
    print("="*60)
    
    # 設定ファイルの判定
    config_file = "test_input_template.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        
    print(f"[*] 設定ファイル読み込み元: {config_file}")
    config = load_json_config(config_file)
    
    # FastAPIの評価実行
    response = run_evaluation(config)
    
    # エクスポート出力ファイルの決定
    score_out = "benchmark_scores_output.csv"
    forecast_out = "forecast_values_output.csv"
    
    export_results_to_csv(response, score_out, forecast_out)
    print("\n[+] 全てのAPI統合バッチテスト処理が正常に終了しました。")
    print("="*60)

if __name__ == "__main__":
    main()