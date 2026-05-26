import json
import os
import csv
import sys
import requests
from typing import Dict, Any, List
import os

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


def load_csv_series(filepath: str):
    """
    CSVから時系列データを読み込みます。
    - ヘッダーに `系列名` 列がある場合は系列ごとにグループ化して `data_series` を返します。
    - ヘッダーに `予測` 列がある場合は同じ構造で `existing_rule_series` を返します。
    返り値: (data_series: List[List[float]], existing_rule_series: Optional[List[List[float]]])
    """
    from collections import OrderedDict

    if not os.path.exists(filepath):
        return [], None

    try:
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            has_series_col = '系列名' in headers
            has_actual_col = '実績' in headers
            has_pred_col = '予測' in headers

            # グループ化するためのマップ
            if has_series_col:
                groups = OrderedDict()
                for row in reader:
                    series_key = row.get('系列名', '').strip()
                    if series_key == '':
                        series_key = 'series_0'
                    if series_key not in groups:
                        groups[series_key] = {'actual': [], 'pred': []}

                    # 実績の抽出
                    val = None
                    if has_actual_col:
                        try:
                            val = float(str(row.get('実績', '')).strip())
                        except Exception:
                            val = None
                    else:
                        # 実績列が無ければ、最後のカラムを数値として試す
                        try:
                            last = list(row.values())[-1]
                            val = float(str(last).strip())
                        except Exception:
                            val = None

                    if val is not None:
                        groups[series_key]['actual'].append(val)

                    # 予測の抽出（存在すれば）
                    if has_pred_col:
                        try:
                            p = float(str(row.get('予測', '')).strip())
                        except Exception:
                            p = None
                        if p is not None:
                            groups[series_key]['pred'].append(p)

                data_series = [v['actual'] for v in groups.values()]
                existing = [v['pred'] for v in groups.values()] if has_pred_col else None
                return data_series, existing
            else:
                # 系列指定なし: 単一系列扱い。行ごとに実績(または最後の数値列)を収集
                data = []
                preds = [] if has_pred_col else None
                for row in reader:
                    if has_actual_col:
                        try:
                            a = float(str(row.get('実績', '')).strip())
                        except Exception:
                            a = None
                    else:
                        try:
                            # DictReaderのrowはOrderedDict順、最後の値を実績とみなす
                            a = float(list(row.values())[-1])
                        except Exception:
                            a = None
                    if a is not None:
                        data.append(a)

                    if has_pred_col:
                        try:
                            p = float(str(row.get('予測', '')).strip())
                        except Exception:
                            p = None
                        if p is not None:
                            if preds is None:
                                preds = []
                            preds.append(p)

                return [data] if data else [], ( [preds] if preds else None )
    except Exception as e:
        print(f"[-] CSVの読み込みに失敗しました: {e}")
        return [], None

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
        # response = requests.post(api_url, json=payload, timeout=60)
        response = requests.post(api_url, json=payload)
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
    
    # CSVファイルパスは設定ファイルで指定可能（キー: csv_path または csv_file）。指定が無ければ data/test.csv を試す
    configured_csv = config.get("csv_path") or config.get("csv_file")
    if configured_csv:
        csv_path = configured_csv
    else:
        csv_path = os.path.join("data", "test.csv")

    # 相対パス指定時は data/ 下を優先して探索
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        alt = os.path.join("data", csv_path)
        if os.path.exists(alt):
            csv_path = alt

    csv_series, existing_series = load_csv_series(csv_path)
    if csv_series:
        if existing_series:
            print(f"[*] {csv_path} を検出しました。CSVデータと既存予測列を評価に使用します。")
        else:
            print(f"[*] {csv_path} を検出しました。CSVデータを評価に使用します。既存予測列は存在しません。")
        config["data_series"] = csv_series
        config["existing_rule_series"] = existing_series if existing_series else None

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