# Empirical Asset Pricing via Machine Learning

## Project ID
proj_a9791753

## Taxonomy
ResidualFactors

## Current Cycle
5

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
Traditional linear asset pricing models, such as the Capital Asset Pricing Model (CAPM) and Fama-French factor models, often fail to capture the complex, non-linear relationships between firm characteristics and stock returns. This results in significant pricing errors and an incomplete understanding of what drives risk premia. This paper addresses this limitation by employing a suite of machine learning models, including tree-based methods and neural networks, to construct a more powerful and flexible Stochastic Discount Factor (SDF). The primary goal is to leverage ML's ability to model high-dimensional, non-linear interactions to achieve lower pricing errors, more stable risk premia estimates, and superior out-of-sample return predictability compared to conventional models.

### Datasets
S&P 500 constituent stock data from yfinance. We will also fetch the list of S&P 500 tickers from Wikipedia.

### Targets
The primary target for the ML models is the next month's excess return for each individual stock (R_{i, t+1} - R_{f, t+1}).

### Model
The paper evaluates multiple ML models. This reproduction will focus on two main types: a tree-based model (LightGBM) and a feed-forward neural network (NN). The LightGBM is effective at capturing non-linear interactions with minimal tuning. The NN will be a simple multi-layer perceptron (MLP) with 3-5 hidden layers, ReLU activations, and Dropout for regularization. These models will take firm characteristics as input to predict next-month returns.

### Training
The models will be trained and evaluated using a walk-forward validation scheme to respect the time-series nature of the data and prevent look-ahead bias. For each validation split, the model is trained on a rolling or expanding window of past data (e.g., 60 months) and tested on the subsequent period (e.g., 12 months). Hyperparameters will be tuned on the first training fold only.

### Evaluation
Model performance will be evaluated primarily on out-of-sample (OOS) predictive R-squared. We will also evaluate the economic significance of the predictions by forming a long-short portfolio (long top decile of predicted returns, short bottom decile) and calculating its annualized Sharpe Ratio, net of transaction costs. Finally, we will compare the models' Mean Absolute Pricing Error (MAPE) against a traditional linear model baseline.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## ★ 今回のタスク (Cycle 2)


### Phase 2: データパイプライン構築（S&P 500） [Track ]

**Track**:  (A=論文再現 / B=近傍改善 / C=独自探索)
**ゴール**: yfinanceからS&P 500の株価データを取得し、月次リターンと基本特徴量を計算するパイプラインを構築する。

**具体的な作業指示**:
1. `src/data/sp500_loader.py`を作成します。`wikipedia`ライブラリを使いS&P 500のティッカーリストを取得し、`yfinance`で過去15年分の日次株価データをダウンロードする関数を実装します。
2. `src/features/build_features.py`を作成します。日次データを月次にリサンプリングし、ターゲットである翌月リターンを計算します。また、基本的な特徴量（例：過去1, 3, 6, 12ヶ月のリターン（モメンタム）、過去30日のボラティリティ）を計算する関数を実装します。
3. `src/main.py`に`build-dataset`コマンドを追加し、処理済みデータを`data/processed/sp500_monthly_features.parquet`として保存します。

**期待される出力ファイル**:
- src/data/sp500_loader.py
- src/features/build_features.py
- data/processed/sp500_monthly_features.parquet

**受入基準 (これを全て満たすまで完了としない)**:
- build-datasetコマンドが正常に終了すること。
- sp500_monthly_features.parquetが生成され、400銘柄以上、10年以上のデータが含まれていること。
- データフレームに'target_return'と複数の特徴量カラムが存在すること。










## 全体Phase計画 (参考)

✓ Phase 1: コアアルゴリズム実装（LightGBM on Synthetic Data） — 合成データを用いて、リターン予測を行うLightGBMモデルの基本構造を実装する。
✓ Phase 2: データパイプライン構築（S&P 500） — yfinanceからS&P 500の株価データを取得し、月次リターンと基本特徴量を計算するパイプラインを構築する。
✓ Phase 3: ウォークフォワード評価フレームワークの実装 — LightGBMモデルを実データで評価するためのウォークフォワード検証機能を実装する。
✓ Phase 4: ニューラルネットワークモデルの実装と評価 — 論文のもう一つの主要モデルであるニューラルネットワークを実装し、同じウォークフォワード検証で評価する。
→ Phase 5: ハイパーパラメータ最適化 — LGBMとNNモデルの主要なハイパーパラメータをOptunaで最適化する。
  Phase 6: フルスケール評価とベースライン比較 — 最適化済みモデルと線形モデルを、より多くのスプリット数でウォークフォワード評価し、性能を比較する。
  Phase 7: ポートフォリオ戦略とシャープレシオ計算 — モデルの予測に基づきロングショートポートフォリオを構築し、取引コスト考慮後のシャープレシオを計算する。
  Phase 8: 特徴量重要度の分析 — 最も性能の良いモデル（LGBM）について、どの特徴量がリターン予測に寄与しているかを分析する。
  Phase 9: 代替モデルの探索（Gated Recurrent Unit） — 論文で言及されていない時系列モデル（GRU）を実装し、性能改善の可能性を探る。
  Phase 10: 価格誤差（Pricing Error）の分析 — 特性でソートしたポートフォリオの価格誤差を計算し、MLモデルが線形モデルより優れているかを確認する。
  Phase 11: 最終レポート生成 — すべての分析結果を統合し、比較表やグラフを含む包括的なMarkdownレポートを生成する。
  Phase 12: 最終化とエグゼクティブサマリー — コードベースをクリーンアップし、非技術者向けの要約を追加してプロジェクトを完成させる。


## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項
- 未来情報を特徴量やシグナルに使わない
- 全サンプル統計でスケーリングしない (train-onlyで)
- テストセットでハイパーパラメータを調整しない
- コストなしのgross PnLだけで判断しない
- 時系列データにランダムなtrain/test splitを使わない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_5/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_5/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新（セットアップ手順、主要な結果、使い方など）
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合（エラー、データ不足、期間の短さ等）
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
