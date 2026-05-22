# **複数系列対応 予測ベンチマークシステム (Time Series Forecasting Benchmarking System)**

本システムは、学術論文「時系列予測手法選定のための実験フレームワークと精度比較実験」に基づいて設計された、包括的な予測モデル評価プラットフォームです。

実務で広く使われる伝統的な統計時系列モデル（14手法）から、機械学習や深層学習モデル（14手法）、さらにこれらを動的に組み合わせてブレンドする高精度なアンサンブルモデル（6手法）の全34手法を一元的に評価・比較することができます。

## **1\. システム構成とディレクトリ構造**

システムは、クリーンアーキテクチャおよびカプセル化（疎結合）の思想に基づき、各役割ごとにモジュールが細分化されています。

ts-forecasting-benchmark/  
├── models\_config.json      \# 全34予測手法の定義、カラーコード、Combブレンド規則の設定ファイル  
├── manual.json             \# 画面上で閲覧できる、全機能を網羅した日本語ユーザーマニュアル  
├── preprocessing.py        \# データ前処理（欠損値穴埋め・Z-Score標準化スケーラー）  
├── feature\_engineering.py  \# 教師あり学習用ラグ特徴量（Sliding Window）生成  
├── evaluation\_metrics.py   \# SMAPE、MASE、MASE分母基準の数理計算  
├── ranking\_system.py       \# OWA (Overall Weighted Average) の集計・ソート・ランキングシステム  
├── statistical\_models.py   \# 統計予測アルゴリズムの処理（ProphetやARIMA系を含む14種）  
├── machine\_learning\_models.py \# 機械学習・深層学習の予測処理および自己再帰推論ループ（14種）  
├── main.py                 \# FastAPI Web APIサーバー（コントローラー、CORS設定、動的Comb計算）  
├── test\_driver.py          \# APIサーバー検証用自動テストドライバースクリプト  
├── test\_input\_template.json \# テストドライバー用入力パラメータ定義JSONファイル  
├── benchmark\_app.html      \# HTML/JS フロントエンド（Chart.jsズーム、クリック値ロック、目玉トグル搭載）  
└── streamlit\_app.py        \# Streamlit (Python) フロントエンド（Plotly描画、動的構成同期対応）

### **各ファイルの役割詳細**

| ファイル名 | 区分 | 詳細説明 |
| :---- | :---- | :---- |
| **models\_config.json** | 設定 | 手法選定チェックボックスの項目、中分類、グラフ線カラー、論文OWA値、説明文を完全に一括管理するマスター設定ファイル。 Comb 1〜6のブレンド生成ルール（抽出戦略）もここに外部定義。 |
| **manual.json** | 設定 | 画面右上のボタンから閲覧できるヘルプ用ユーザーマニュアルの構造化テキストファイル。機能が追加されてもプログラムを修正せずマニュアルを改訂可能。 |
| **preprocessing.py** | データ前処理 | 時系列データに不可欠な、前方/後方/平均による欠損値補完（Imputation）や、機械学習の適合精度を最大化するZ-Score標準化スケーラー（fit/transform/inverse\_transform）の実装。 |
| **feature\_engineering.py** | 特徴量作成 | 1次元時系列データ配列から、過去 of ラグ（Sliding Window幅）を自動算出して教師あり学習用の特徴量行列 ![][image1] とターゲット配列 ![][image2] を生成する。 |
| **evaluation\_metrics.py** | 評価指標選択 | 評価誤差の計算を行います。M4競争基準である **SMAPE**（対称平均絶対パーセント誤差）および **MASE**（平均絶対スケーリング誤差）、MASE計算用の学習データベースライン分母（sp対応）の数理計算。 |
| **ranking\_system.py** | ランキング | 各個別系列で計算されたSMAPEとMASEを集約し、ベースラインモデル（Naive 2）の相対性能比としての **OWA（総合加重平均スコア）** を全モデル分算出してランキング。 |
| **statistical\_models.py** | 予測手法選択（統計モデル） | 伝統的な時系列モデルの実学習および予測。sktime や prophet と連携。未インストール時は自前のモックシミュレータに安全フォールバック。 |
| **machine\_learning\_models.py** | 予測手法選択（機械学習モデル） | 機械学習および PyTorchベースのディープラーニング（RNN/DeepAR）モデルを内包。make\_reductionを使用せず、自作の「再帰的予測ループ（Recursive Multi-step Forecasting）」を回すことで、将来値を1期ずつ再帰推論。 |
| **main.py** | API | FastAPIのAPIルーター。クレンジング後のデータを各予測器へ供給し、結果を受け取って models\_config.json 内の定義規則に基づいて Comb 1〜Comb 6 の動的ブレンド予測値平均を算出。ランキングを集計してフロントへ配信。 |
| **test\_driver.py** | テスト | FastAPIサーバー起動状態で実行するAPI自動検証・バッチテストスクリプト。設定値JSONをインポートしてAPI推論をかけ、結果スコア・予測系列の双方をCSVに出力。 |
| **test\_input\_template.json** | テスト | test\_driver.py がインポートする、フロントエンド側の各種マッピングパラメータ、予測期間、入力時系列データ、及び評価対象アルゴリズムを記述したテンプレートJSON。 |
| **benchmark\_app.html** | 画面 | **HTML/JS版フロントエンド。** ドラッグでのCSVアップ、Chart.jsによるズーム＆パン、縦型追跡線、クリックロックによる値固定表示パネル、モデル名の完全文字折り返し、テーブルの非表示スイッチ（一括目玉トグル）を備えた高性能Web。 |
| **streamlit\_app.py** | 画面 | **Python Streamlit版フロントエンド。** Plotlyを用いた動的プロット、サイドバー接続スイッチ、アコーディオン形式の中分類手法マスタを全自動レンダリングするWebアプリ。 |

## **2\. 動作環境とインストール要件**

システムおよびテストドライバーを動作させるための要件です。プロジェクトのルートディレクトリで以下のコマンドを実行することで、必要なすべてのPython環境を一括で構築できます。

pip install \-r requirements.txt

### **インストール対象パッケージの役割説明**

requirements.txt にリスト化されているライブラリ群は、システム内で以下の役割を担っています。

#### **① Webフレームワーク & API関連**

* **fastapi / uvicorn / pydantic**:  
  * バックエンドWeb APIサーバー（main.py）の構築に使用します。  
  * リクエスト/レスポンスの厳格なスキーマ検証、自動バリデーション、および高速な非同期処理サービングを実現します。  
* **requests**:  
  * 自動テストドライバー（test\_driver.py）やフロントエンド（streamlit\_app.py）がバックエンドAPIと通信するためのHTTPクライアントライブラリです。

#### **② データ処理・分析の基本ライブラリ**

* **numpy / pandas**:  
  * 時系列データの読み込み、整形、スライシング、およびラグ特徴量行列の基礎数値演算を担当するコアエンジンです。

#### **③ 統計時系列予測アルゴリズム**

* **sktime / statsmodels / pmdarima**:  
  * クラシックな時系列統計モデル（ETS、Damped、TBATS、ARMA、ARIMA、SARIMA）を学習・推論するための統合予測ライブラリ群です。最適な次数決定や自動トレンド検出を行います。  
* **prophet**:  
  * トレンドの変化点、週・年周期、イベント効果を非線形加法モデルとして頑健に学習・予測するProphetアルゴリズムの実体です。

#### **④ 機械学習 & 回帰モデル**

* **scikit-learn**:  
  * Random ForestやSupport Vector Machine（SVR）、多層パーセプトロン（MLP）、決定木、正則化線形回帰（Ridge、Lasso、ElasticNet）および標準化スケーラー（Z-Score）として幅広く活躍します。  
* **pygam**:  
  * 各入力ラグ変数（特徴量）と予測ターゲットとの間の滑らかな非線形スプライン関数を適合・学習する一般化加法モデル（GAM）の構成に使用します。  
* **xgboost / lightgbm**:  
  * 決定木ベースの勾配ブースティング（GBDT）として、時系列ラグ回帰における高いパターン認識と予測精度を実現します。

#### **⑤ 深層学習モデル (RNN / DeepAR用)**

* **torch (PyTorch)**:  
  * ニューラルネットワーク層の演算プラットフォームです。LSTM/RNNモデルの学習、およびDeepAR風の確率的正規サンプリングによる時系列自己回帰の推論エンジンとして機能します。

#### **⑥ フロントエンド & グラフ可視化**

* **plotly**:  
  * Streamlit版アプリ上で、拡大・縮小・パンを直感的に行えるインタラクティブな予測結果プロットを美しく可視化します。  
* **streamlit**:  
  * Pythonコードのみでリッチで洗練されたWebアプリケーションUIを構築し、トグルテーブルやスライド時点固定などの高度なフロントエンドUXを提供します。

## **3\. アプリケーションの起動と使い方**

システムは、実稼働APIによる正確な計算と、APIがオフラインの際にも動作する「ローカル模擬予測（フォールバック）」の双方に対応しています。

### **STEP 1: バックエンドAPIの起動**

プロジェクトのルートディレクトリで以下のコマンドを実行し、FastAPIサーバーを起動します。

uvicorn main:app \--host 0.0.0.0 \--port 8000 \--reload

起動すると、http://localhost:8000/api/config（構成情報の取得）や http://localhost:8000/api/manual（マニュアルの取得）が配信可能になり、API連携準備が整います。

### **STEP 2: フロントエンドの起動・使用方法**

#### **A. HTML/JavaScript版 Webアプリ (benchmark\_app.html)**

1. benchmark\_app.html をブラウザでダブルクリックして開きます。  
2. **API未連携時（ローカル模擬）**:  
   * ヘッダーの「API連携」が「ローカル模擬予測」になっていることを確認。  
   * 「ベンチマーク評価を実行」をクリックすると、内蔵された擬似予測ジェネレータが動き、動的Combを含むすべてのOWA性能ランキング、クリックによる値ロックパネルなどが即時に体験できます。  
3. **API連携時**:  
   * ヘッダーの「API連携」を「FastAPIサーバー」に変更。エンドポイントURLに http://localhost:8000/api/evaluate を指定。  
   * 実測データ（CSVなど）をアップロード、あるいは疑似データを設定し「FastAPI 評価プロセスを実行」をクリック。  
   * バックエンドサーバー上で実際に20種類以上の学習、特徴量ラグ生成、予測、および論文に基づいた動的Comb構築がミリ秒単位で処理され、本物のベンチマーク結果が再描画されます。

#### **B. Streamlit版 Webアプリ (streamlit\_app.py)**

以下のコマンドを実行し、ブラウザで立ち上げます。

streamlit run streamlit\_app.py

* 左側サイドバーの「接続モード」を「FastAPI バックエンド連携」に切り替えることで、起動したFastAPIサーバーと通信を行いリアルタイムベンチマークを実行できます。

## **4\. 自動テストドライバーによるバッチ評価手順**

フロントエンドのUIを使用せずに、自動化バッチ処理やコマンドライン上から一括して全系列・全モデルに対するテストを実行し、その評価結果（OWA精度順位、生予測カーブ数値など）をエクスポートできます。

### **実行条件**

* バックエンドサーバー（main.py）が **localhost:8000** 等で正常に起動している必要があります。

### **実行コマンド手順**

以下のコマンドを実行することで、指定されたテスト設定を読み込んで推論評価を自律実行します。

\# デフォルトテンプレート設定ファイル (test\_input\_template.json) を使用して評価  
python test\_driver.py

\# 任意の独自テスト設定 (例: custom\_input.json) を指定して評価  
python test\_driver.py custom\_input.json

### **出力される成果物**

テストが完了すると、カレントディレクトリに以下の2つのCSVファイルが自動的に生成されます。

1. **benchmark\_scores\_output.csv**:  
   モデルごとにすべての系列の予測に対する「平均SMAPE誤差」「平均MASE誤差」を求め、ナイーブモデル2を1.0とした「OWA加重相対比スコア」を昇順（精度が高い順）に一覧にした総合精度ランキング。  
2. **forecast\_values\_output.csv**:  
   各時系列系列（series 0, series 1...）における、すべての期間の実績値（学習、および評価用実数値）と、その時の**全評価対象に選んだすべての予測アルゴリズムの予測線プロット数値**をタイムステップ（t+x）ごとに1レコードに結合した時系列予測推移データ表。

## **5\. 全34アルゴリズムと中分類グループ定義**

本システムは、論文の「表1：採用した30手法」の分類規則に基づき、中分類（2列目のカテゴリ）ごとに整理して描画します。

### **統計時系列モデル (14手法)**

1. **ナイーブ (Naive)**  
   * Naive 1: 将来の値が最後に知られた観測値と同じになると仮定する。  
   * Naive S: 季節性を考慮し、前年同周期の最後の値を将来の予測値と仮定。  
   * Naive 2 (ベースライン): 必要に応じて乗法分解で季節調整した上で Naive 1 を適用。  
2. **指数平滑 (Exponential Smoothing)**  
   * SES: トレンドや季節性のないデータに適した単純指数平滑化。  
   * Holt: 指数平滑法に線形トレンド成分を追加して外挿。  
   * Damped: 将来の線形トレンドが無限に増加するのを抑制する減衰トレンドモデル。  
   * Theta法: ドリフトを伴う単純指数平滑化をベースにした高精度数理モデル。  
   * AutoETS: AIC等の基準を元に、最適な状態空間（エラー・トレンド・季節性）パラメータを自動探索。  
   * TBATS: Box-Cox変換や複雑な複数季節周期に対応した指数平滑状態空間モデル。  
3. **ARIMA (Autoregressive Integrated Moving Average)**  
   * ARMA: 差分階数 ![][image3] の自己回帰移動平均モデル。  
   * ARIMA: 標準的なパラメータ ![][image4] に固定した季節調整なしのARIMA。  
   * SARIMA: データの季節周期（![][image5]）を明示的に組み込んだ季節自己回帰和分移動平均モデル。  
   * AutoARIMA: ステップワイズ法でARIMAおよびSARIMAの最適な次数を情報量基準を元に自動探索。  
4. **状態空間モデル (State Space)**  
   * Prophet: トレンド、複数の周期（年・週）、および祝日効果などを頑健に加法分離するモデル。

### **機械学習・深層学習モデル (14手法)**

5. **線形回帰 (Linear Regression)**  
   * Linear: 多変量の線形回帰。過去のラグ特徴量に対するフィッティング。  
   * Ridge: 过学習を防ぐためのL2正則化項を付与した線形自己回帰。  
   * Lasso: 不要なラグをゼロ化して特徴量削減を行うL1正則化を付与した線形自己回帰。  
   * Elastic-Net: L1とL2の正則化項をバランスよく併用した線形回帰。  
   * GAM: 入力と予測値の非線形関係を滑らかな加法関数で表現する一般化加法モデル。  
6. **決定木 (Decision Tree)**  
   * Decision Tree: 過去のラグ情報を分岐境界でセグメントする回帰決定木。  
   * Random Forest: 複数のブートストラップ決定木を並列構築してブレンド平均する。  
   * GBDT: 直前の決定木の予測残差を次の決定木で逐次学習して補正する勾配ブースティング。  
   * XGBoost: 高度な正則化と二次勾配を用いた最適化型分散勾配ブースティング。  
   * LightGBM: 葉（Leaf-wise）単位で効率的に分岐木を成長させ、学習時間とメモリを最適化したGBDT。  
7. **SVM (Support Vector Machine)**  
   * Support Vector Machine: 高次元カーネル法によって余白（マージン）を最大化して回帰面を決定するSVR。  
8. **ニューラルネットワーク (Neural Network)**  
   * MLP: 多層の全結合順伝播型ニューラルネットワーク。  
   * RNN: 時系列の順序関係を隠れ状態で記憶・伝達できる再帰型ニューラルネット（LSTM等）。  
   * DeepAR: 自己回帰予測の出力をサンプリングし、確率的な未来の予測軌道を生成する自己回帰深層モデル。

## **6\. アンサンブル Comb 1〜Comb 6 の動的構築ルール**

本システム内の Comb 2〜Comb 6 は、アップロードされたCSVデータの特性（仮ランキングの精度）に基づいて、**ブレンドメンバーをリアルタイムかつ自律的に構成を変化**させます。すべて予測値の平均で算出します。

| 手法名 | 論文の定義 | API内の動的な自律構成アクション |
| :---- | :---- | :---- |
| **Comb 1** | 代表的な指数平滑の組み合わせ | 固定型（fixed）：SES, Holt, Dampedの3予測値を単純平均。 |
| **Comb 2** | 精度の高い3つの手法の組み合わせ | 動的型（top\_n）：全個別18手法（統計・ML）のうち、仮スコアが最も良かった**総合精度上位3モデル**を動的に特定し平均。 |
| **Comb 3** | 異なる中分類で精度の高い3手法の組み合わせ | 動的型（top\_subcats）：中分類の8カテゴリの中から、最もOWAスコアが良かった異なる3つの中分類のそれぞれのNo.1手法（計3モデル）を自動選出して平均。 |
| **Comb 4** | 2つの精度の高い統計モデルの組み合わせ | 動的型（top\_n）：統計14手法の中から、仮スコアが最も良かった**統計上位2モデル**を動的に自動選出して平均。 |
| **Comb 5** | 2つの精度の高い機械学習モデルの組み合わせ | 動的型（top\_n）：機械学習14手法の中から、仮スコアが最も良かった**機械学習上位2モデル**を動的に自動選出して平均。 |
| **Comb 6** | 精度の高い統計上位2位と、機械学習第1位の組み合わせ | 動的型（top\_n\_multi）：統計モデルの中から**上位2モデル**、機械学習モデルの中から**第1位の1モデル**（合計3モデル）を動的に自動選出して平均。 |

## **7\. 保守とアルゴリズムの拡張（拡張性設計）**

本システムは、ハードコーディングを一切排除した拡張性の高い設計をとっています。新しい予測手法を追加する場合は、以下の手順で行います。

1. **models\_config.json への追記**:  
   * models 配列に、追加したい手法の id, name, category（statistical / ml / ensemble）, subcat（中分類ID）, color（カラーコード）, desc（説明文）を記載。  
   * subcategories や subcat\_groups に新規中分類があればIDを追記。  
2. **モデルロジックの追加**:  
   * 統計的なモデルであれば statistical\_models.py 内の predict\_statistical\_model 内の if model\_id \== 'xxx' に処理を追記。  
   * 機械学習・DL系のラグ回帰モデルであれば machine\_learning\_models.py 内の predict\_ml\_model 内の elif model\_id \== 'xxx' に回帰モデル（Regressor）を初期化するロジックを1行追記。

APIやフロントエンド（HTML/JS版、Streamlit版双方）は、FastAPIから起動時に配信される構成設定情報（/api/config）およびアンサンブル推論処理（/api/evaluate）を自動的に読み込んでアコーディオンやチェックボックス、OWAランキング結果などのUIを全自動・ノンコーディングで動的更新するため、フロントエンドの書き換えや再ビルドは一切不要です。

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAYCAYAAADzoH0MAAABJklEQVR4XmNgGAWDDMjJyWnJy8vfAeL/SPibrKysLUgeyJ6MJncf3QwwAEpYAvFPIL4NxJIwcRUVFXYgfz3QonpxcXFuZD0oQEZGhhOoaAdQ8T8FBQUPqDAjkF8KwiA2snqsAKgxAurM5cbGxqxQzd0gNrparEBRUVEcqOE6EL8H4mYgnky0ZhgAamoFuQLomkNKSkr86PIEATUM8ALif0B8gmQDgJo0gXgf0PZbIEOQYoMwAAagPFDDRmB0qiDHBlCKBV0tBgA5Fah4FdAQMxAfOTaAKVIHXT0KgGpeB8TeyOJAVzRAA7MBWRwFABUogpwNxIXocsBUaQOU/w2UOyUtLS2MLhkLlPwF9ScI/wUq9IfJA/lZIDFkeaCencDwEUI2ZyQDAPVkWUGOojRDAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAWCAYAAAD5Jg1dAAABEUlEQVR4XmNgGAWDGzDLyckZKyoq2hkbG7MiSygpKfGDxYAKXOXl5WtUVFT4gPQBBQWFlSCNIEVAfjQQf5OVlTVlAEpkAk3SBwpYggSB/AiYaUD+JCC+KiUlJQK3AqigASj4BIgVQXygTYJA9mkgXgrkMoIViYqK8oCsBeI1QC4LSAxqyyegAelw04ACkkD8EGhKOZIYyH2/gWI2cIVA3eJAwbswhSCfAvl7MNwHBIxAwWKoG+dC3fYfqHE+SA6uCugODhAGuRVqOigEQO6DhwCDtLS0DFDwOsgqcXFxbqAQC9Ck6UD+FWVlZTG4QqjvXgJxDlRRPtCkW0BsAFcEAqDogbrvOAgDFXSCYglFERAAADEZP/QwjoKmAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAaCAYAAADxNd/XAAACJUlEQVR4Xu2WPUhbURTHDbHQUkFamgbzktx8QItgIRAcBBEEFwelaifbwUmXLp1KW5yCCIUuXQTBwalLVkFR0FFwUfADnIoITm52qB30d+q9+npKPhxeIvL+8Ofde8655+ve3JuWlhAhQtxvxGKxtnw+/5xhVOvuNIwx/fA3vIBr8Xj8sbYJEslk8hFxx+E8/JrNZl9qm5rwPC/J4mM4o3VBIpfLtRNzFZbkBGQymQLjfTimbasinU73yi6kUqlhrQsSxP1I3C2+T5yM+Vt4wE7E/bZVYR2dwk6tCwqStCRP1xf9cprYjfysVjOjkixOhqRSxmW4IduoDYOCxJem6QLIqYj8l6l0nO3CbQxn4aSM4Tn8rm2DhEu0UgFa/hd026A4hJ+YRkTGeIIFF7W2DJtpnB/dgsvcME+1Hwf0QxJXJ1qtgFYUP8zVbZN1wnQTzr+AeIO3KkAStImWmbZasRTV8PMvqJRoJbl/y6acjC32kP00dZx/KRC7jnpZ61Unnxx2JzpRVwD87JdfFwAHfTK5///AUdiDsw/+NX5Q7Cvs39RLiSOvrPbjYBuyAZeI+9DJWTuA7Fy+fnu5X7tQnLofq30F16RaW/WX/xYFDOK9g0fm5jcZYVyCm5LfP8ZW+R7uwAVz9YSPwF2crPD9ViwWH+hFQULiEXuO2Ovswmub/B7jgra9hv7XKU4SicQzN28CInT7hRw7rvm+RjcxRIgQIerDJYKprGhwbgmNAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHwAAAAaCAYAAAB1szj5AAAE80lEQVR4Xu2aW4gcRRSGe9kIioqOui57m5qZDSzxgsqoKKxGIcruwyqGoIG8CAF9UATvaJDowyLCKl7AB0GjDxJh46PiZcGAQYN5ScSgiIJKoogkQiDiBbJ+f7pKa8uZnu7ZtmcI/cNPV9ep/ru6TtWp0z0TRSVKlChRokSJUwZDQ0Nn1Wq108P6Er1Hs9k8rdFonBPWdw0cfXm1Wt2Rp6gm0OTk5IUUB0NbiWyQw/HR88aYjaEtM8bGxsYR+giHXxzaugFaN8I/4DJcGh4ePjNs00cY5LlvZzBroSEj/ncd/HQ+4/luvV6/OrRlwYBmDnwyNKwGdhIdgvOhrdcYHx8/g0Gd45lfoH+H4XHOm2G7TuiFDm1m4AeKnqEtFSYmJi7hBl/pGNpWAzo8rVWO7i2hrdfQANO3WQbuBvq5PWmAk9ALHW252D+h7ebQlgoIP4rAO7WckzWrewSuC239BNvPtgOcFkXqYJ+HuyiuCW2JkJPlbPh4aIvivWTa3lhJl86b7B83t9mTB+Vc2szRZlgdgrvD0KPkg1V/JW2uV1laCv9+myKRZoDToEgd7LPw+8zjxkUjulBOamHbqn2d4z7sr8uBHO/juA0ewGHGa7sO7sf+NLxLZfgnfNHXRG89/Jz6By33wG/gFr9dkUgzwGlQpI5stDnEwrkqtCXCXvgzx2m/fnR09ALqX7MrVeH+a/aZtbJNTU2dTd3HzplyvOzwMU4HVEf5TuzL/v6tzFL3wnaHrVKy+Eynh8P+BPYfMvA9+npeqNMO1RQDnAZF6piEhZoIidpBWiHOgF2K4D0cxzh+55xrr6lwvg8uVSoVJRA7TZyN1702K/ZvhXXKu+Fe7z1/gHY7qDuoCeauLRppBjgNitQx1uEma2Rs53CHaotMW060znwT+0W27CcQa3RuvP2bdhs4P2G8VzQ3ceT0yEaGXiDNAKdBkTrmX4c/ENoS0cnhEoQ/YW94dVvgMuH4burnXNnZ20QFPcQJOd6r03Zy3L+2FWx00AOmYtYve2kGOA2K1LHP2lVIb3Dhj3A2tEUtVqqtUwj/Uhmic7h/fTWOCn/BjfBaHHq/fYhj7OOXuXYmnjiJDyZoe6HNprRUX/ReG+q0g+1bu34MagLpbSY0hChI5ySqyX5rD5ucHWy1yryV+s++S2i/jvNf5Ex7ro82R1zItx8FllyHOW7juAH9GcpH3UO4RE/37uX+LdgB/q1Vxkv9VhNP6J1Rh3feInQcZKPNt6aLbxwucVrx+iRU7f5t4oxcCdci3I+j1nvNBqi7Fx6Ar8IP4W3wC65/n+OzetcWKS/APVz/BsdP4a+6tzQ8vUKgd3/u/RZ9OGpiRzhq1Tzn2lXjCPa7iQd3xNcQitZxMHF0/M83jlTAAZvhZ9yw4tfbmXYy05awPtxHbZwT/iomB9uVu2Iv5T7n2h8AroDHWkWWfoPCMH19Sa+ooS0L8tKJ7LZa6/a3D+sArbwZr7rV/p0b7AxNDFv9AvpZhwtRh1DcCXnp6HsIOnvdd5GuoD0YkUUlO1qd2mtNHE4Ws2a9SdAst7+ivQ0PK4lTXdiuj6DPxQuMx02hISPy0tHHqqfQeUTl0JgF2osftlS4fcUR8e15rXI9sK8NXzZdJB5Fgb6NMMC3Rqsc3Lx0bNK8K5c/qdjE6iE6dk1oK9F72Lem+VycXaJEiVMMfwMveNwiPkKrIgAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAaCAYAAAC3g3x9AAABs0lEQVR4Xu2SwSsEcRTHd7OKSMQadnZ2jJ3LHpSaOIkL4iLJQSkpxZkDRUlJ+QekNiUnB7tHJw5cRC4O68LFQZy0Z0nr83Z/vzW2PVhX++rb7/e+7/vem/nOBALVqEY1Ko9gVyEmHccZ9DyvVhdc122ybXs8Ho9bkhuG0YBmNBaLeaQ1WlcMaaZhDxwjmgab3NPRaLReoGrr4IWFu5wpMKvOpH95PhCNUbhiUAtpSAaD+0gk0gY3zH0DJMAbOJAl0mdZVh95VjQ/BkKsUXjnXDRNs5WGHoFaNs/r9dM8gebT38x9AO4DrHxPKxRGRAxyCmf4FvZr4HbAE4tMzbFsSemn/Np8IHQZvEoxo0TFrfIRyM9BijSkaLFGPBQbEnqIGJ5m0wNfsF048RHuVrarRuG64V7FGs2RO+AZbj+gl+CNAfkIeaiNZlAv3J08sW5W/uWobSkqSL4NMsywtS5fQLQsTwSS3I8YfsM55BfZBf+y4BqcoLnkTKHr8OuKQaEOQSdnc2ktHA43UrsAp/IW8hcIV6r7dZTz788hTyL/pvjHOVfuDSoKBi2ItxoMnCnV/PP4Au9EccPP11bpAAAAAElFTkSuQmCC>
