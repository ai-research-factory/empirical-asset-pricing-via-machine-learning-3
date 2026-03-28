# Phase 5: Hyperparameter Optimization — Technical Findings

## Objective

Optimize the main hyperparameters of LightGBM and Neural Network models using Optuna, following the paper's approach of tuning on the first training fold only.

## Implementation

### Models Implemented

1. **LightGBM** (`src/models/lgbm_model.py`): Gradient boosted decision trees for cross-sectional return prediction. Walk-forward evaluation with long-short decile portfolios.

2. **Neural Network** (`src/models/nn_model.py`): Multi-layer perceptron with ReLU activations and Dropout. Early stopping based on training loss with patience=10.

3. **Optuna Optimization** (`src/models/optimize.py`): Hyperparameter search on the first walk-forward fold only (no test-set contamination). Search space constrained near paper-default values.

### Walk-Forward Configuration

- **Windows**: 5 configured, 3 valid (first 2 skipped due to min_train_size=60 months constraint)
- **Training**: Expanding window from earliest available data
- **Features**: mom_1m, mom_3m, mom_6m, mom_12m, volatility_30d
- **Standardization**: Train-only mean/std applied to both train and test
- **Portfolio**: Long top decile, short bottom decile, equal-weight within decile

## Results

### Walk-Forward Sharpe Ratios (Net of Costs)

| Model | Window 0 | Window 1 | Window 2 | Average |
|-------|----------|----------|----------|---------|
| LGBM Default | -0.053 | 1.534 | 1.283 | 0.921 |
| LGBM Optimized | 0.565 | 1.584 | 1.884 | **1.345** |
| NN Default | -0.038 | 1.300 | 1.090 | 0.784 |
| NN Optimized | 0.049 | 1.209 | -0.089 | 0.390 |

### Best Model: LightGBM Optimized

- **Avg OOS Net Sharpe**: 1.3445
- **Positive Windows**: 3/3 (100%)
- **Annual Return**: 12.96%
- **Max Drawdown**: -6.3%
- **Hit Rate**: 66.7%

### Optimized Hyperparameters

**LightGBM** (30 Optuna trials on fold 1):
- num_leaves: 63 (default: 31)
- learning_rate: 0.062 (default: 0.05)
- feature_fraction: 0.825 (default: 0.8)
- bagging_fraction: 0.815 (default: 0.8)
- bagging_freq: 4 (default: 5)
- max_depth: 7 (default: 6)
- min_child_samples: 20 (default: 20, unchanged)
- n_estimators: 500 (default: 200)

**Neural Network** (20 Optuna trials on fold 1):
- hidden_dims: [96, 48] (default: [64, 32, 16])
- dropout: 0.423 (default: 0.3)
- learning_rate: 0.002 (default: 0.001)
- weight_decay: 1.4e-5 (default: 1e-4)
- batch_size: 128 (default: 256)

## Key Observations

1. **LGBM benefits from optimization**: Sharpe improved from 0.921 → 1.345 (+46%). Larger model capacity (more leaves, estimators, depth) captures non-linear patterns better.

2. **NN optimization hurt OOS performance**: Despite better fold-1 tuning, the optimized NN degraded from 0.784 → 0.390 on the full walk-forward. This suggests the NN is more prone to overfitting to the tuning fold's regime.

3. **Window 0 is hardest**: All models struggled with the 2020-2022 test period (COVID recovery → rate hikes). This is expected given the unusual market regime.

4. **LGBM dominates NN**: Across all configurations, LightGBM outperforms the neural network. With only 5 features, tree-based models may have a natural advantage over neural nets which typically benefit from higher-dimensional inputs.

5. **Transaction costs are minimal**: The long-short portfolio rebalances monthly with decile sorts. The 15 bps round-trip cost has negligible impact on monthly returns.

## Limitations

- Tuning is performed on the first fold only. A more robust approach would use nested cross-validation, but this follows the paper's prescribed methodology.
- Only 3 walk-forward windows are available given the ~14 years of data and 60-month minimum training requirement.
- The NN search space could be expanded (more architectures, regularization techniques), but the paper constrains us to near-default parameters.
- Survivorship bias from using current S&P 500 constituents persists (see docs/open_questions.md).
