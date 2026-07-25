# EMA SAR Quantitative Optimization System (Elite v4)

An enterprise-grade, multi-asset quantitative backtesting, optimization, and analytics framework built in Python. Designed for rigorous strategy evaluation, risk modeling, and institutional reporting.

---

## Key Capabilities

### 1. Vectorized Dual-EMA Grid Optimization
- Computes Exponential Moving Averages across periods **2 to 300** using digital filtering (`scipy.signal.lfilter`).
- **Coarse-to-Fine Grid Search**:
  - *Coarse Pass*: Scans parameter combinations in step sizes of 5.
  - *Fine Pass*: Refines search in a $\pm 10$ radius around optimal candidate pairs.
- **ProcessPoolExecutor Parallel Processing**: Utilizes all available CPU cores for high-throughput computation across asset/timeframe matrix.

### 2. Elimination of Lookahead Bias
- **1-Bar Execution Delay**: Post-signal execution is shifted by 1 bar using `np.roll(raw_signal, 1)`.
- **Lookahead Bias Delta**: Compares same-bar vs shifted execution to measure artificial profit inflation.

### 3. Institutional Risk & Statistical Analytics Engine
- **Kelly Criterion ($f^*$)**: Computes optimal fractional bet sizing:
  $$f^* = \frac{p \cdot b - q}{b}$$
- **Risk of Ruin (RoR)**: Calculates analytical probability of 50% equity drawdown:
  $$\text{RoR} = \left(\frac{1 - \text{edge}}{1 + \text{edge}}\right)^N$$
- **Hurst Exponent ($H$)**: Evaluates market regime dynamics via R/S analysis ($H > 0.5$ trending, $H < 0.5$ mean-reverting).
- **Ljung-Box Q-Test**: Tests trade returns for serial dependence / autocorrelation ($p < 0.05$).
- **Advanced Ratios**:
  - **Sharpe Ratio** (Total Volatility Risk-Adjusted Return)
  - **Sortino Ratio** (Downside Volatility Risk-Adjusted Return)
  - **Calmar Ratio** (Annual Return vs Maximum Drawdown)
  - **Omega Ratio** (Probability-weighted gain/loss distribution ratio)
  - **Ulcer Index** (Sustained drawdown depth and duration pain metric)
  - **Tail Ratio** (95th percentile return vs 5th percentile loss)
  - **System Quality Number (SQN)** (Van Tharp strategy quality rating)
- **Walk-Forward 3-Fold OOS Test**: Validates in-sample parameters on expanding out-of-sample windows.
- **Monte Carlo 500x Bootstrapping**: Simulates 500 resampled trade sequences to produce P5/P50/P95 equity bands.
- **Parameter Robustness Score**: Scans $\pm 5$ neighborhood to report percentage of surrounding profitable EMA pairs (anti-overfitting check).

---

## Workflow Architecture

```
[Input: market_data.xlsx]
          │
          ▼
[Pre-Test Profiling Engine] ───► Volatility, Hurst Exponent, Skewness, Kurtosis
          │
          ▼
[ProcessPoolExecutor Grid Engine] ───► Coarse Pass ➔ Fine Pass
          │
          ▼
[Realistic Execution Engine] ───► 1-Bar Shift, Trade Log, MAE/MFE, Long/Short Split
          │
          ▼
[Statistical & Risk Engine] ───► Kelly, Monte Carlo, Walk-Forward, Ljung-Box, Ulcer
          │
          ▼
[Reporting & Visualization Output] ───► Excel Master, HTML Dashboard, CSVs, PNGs, ZIP
```

---

## Directory Structure

```
z:\just run\
├── ema_sar_optimizer.py      # Core quantitative engine & pipeline
├── data/
│   └── market_data.xlsx       # Input OHLCV market data
├── output/
│   ├── dashboard.html         # Interactive HTML Chart.js dashboard
│   ├── optimization_report.xlsx # Master Excel workbook (68+ sheets)
│   ├── quant_bundle.zip       # Standalone ZIP package containing all outputs
│   ├── csv/                   # CSV exports of all tables & trade logs
│   │   ├── Summary.csv
│      ├── WithWithout.csv
│      ├── LongShort.csv
│      ├── OmegaUlcer.csv
│      ├── Profiles.csv
│      └── Trades_*.csv
│   └── *.png                  # 490+ High-resolution chart PNGs
└── README.md                  # System documentation
```

---

## Execution Guide

### Prerequisites
Ensure Python 3.8+ is installed with required packages:
```bash
pip install numpy pandas scipy openpyxl matplotlib
```

### 1. Synthetic Demo Run
Test the entire pipeline using auto-generated synthetic data:
```bash
python ema_sar_optimizer.py --demo --mode coarse
```

### 2. Coarse-to-Fine Grid Search (Recommended)
Run optimization on your Excel dataset (`./data/market_data.xlsx`):
```bash
python ema_sar_optimizer.py --input ./data/market_data.xlsx --mode coarse
```

### 3. Exhaustive Full Grid Search
Run full parameter grid search across all periods from 2 to 300:
```bash
python ema_sar_optimizer.py --input ./data/market_data.xlsx --mode full
```

---

## Output Deliverables

1. **Interactive Web Dashboard (`dashboard.html`)**: Chart.js risk-return scatter plots, performance bars, and metric breakdowns.
2. **Excel Master Workbook (`optimization_report.xlsx`)**: 68+ sheets covering strategy summaries, trade logs, yearly/session/regime breakdowns, and formula specifications.
3. **CSV Export Directory (`./output/csv/`)**: Structured data files ready for database import or Python/R pipelines.
4. **PNG Visualization Suite (`./output/*.png`)**:
   - Equity Curve & Underwater Drawdown
   - Parameter Space Heatmaps
   - MAE vs MFE Execution Quality Scatter
   - Trade Return Distributions
   - Session Performance Breakdown
   - Monte Carlo 500x Fan Bands
   - Hour × Weekday PnL Heatmaps
   - Rolling Sharpe Ratios
   - Cross-Asset Equity Correlation Matrix
   - Universe Risk-Return Frontier
