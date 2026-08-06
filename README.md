# EMA SAR Quantitative Optimization System

An enterprise-grade multi-asset quantitative backtesting, optimization, and analytics engine.  
Implements **Continuous Kelly**, **Purged K-Fold CV**, **Block Bootstrap**, **Bayesian Optimization**, **Regime-Adaptive Signals**, **ATR Dynamic Stops**, **Almgren-Chriss Market Impact**, **CPCV/PBO**, and **DSR** — all validated against quant literature (Thorp 2006, López de Prado 2018, Bailey 2012, Lo-MacKinlay 1988).

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas scipy matplotlib openpyxl statsmodels

# 2. Health-check (no data needed, ~5 seconds)
python validate.py

# 3. Run with auto-install + demo data
python run.py --demo

# 4. Run on your data
python run.py --mode coarse
```

---

## Run Modes

| Command | Description |
|---------|-------------|
| `--demo` | Synthetic demo data, full pipeline |
| `--mode coarse` | **Recommended** — Coarse+Fine grid |
| `--mode bo` | Bayesian Optimization (Sobol + RBF surrogate) |
| `--mode full` | Exhaustive full grid (slow, most thorough) |
| `--risk-aversion 2.0` | CLI risk-aversion λ for utility-theoretic ranking |

---

## Architecture

```
market_data.xlsx
    │
    ├─► profile_dataset()        Hurst, ADF, VR, volatility, regime
    │
    ├─► Optimization Engine
    │       coarse_fine()        Step-5 grid → ±10 fine refinement (S2: boundary-aware)
    │       bayesian_opt()       Sobol sampling + RBF thin-plate surrogate (F2)
    │
    ├─► backtest_pair()          1-bar shifted execution, SL/TSL/TP/ATR, regime signals (C1-C6)
    │       generate_strategy_signals()   Hurst/ADF gate → EMA or Bollinger Band MR (F1, Q1)
    │       apply_stop_loss()    Vectorized: Fixed SL → TP → TSL priority (P2, U1, U2)
    │
    ├─► Risk & Statistics
    │       continuous_kelly()   f* = μ/σ² continuous-time formula (F5)
    │       deflated_sharpe_ratio()  Bailey-LdP 2012 DSR (M2)
    │       monte_carlo()        Block bootstrap for autocorrelated returns (F4, P4)
    │       purged_kfold_cv()    Purged K-Fold with EMA cache (F3, P1)
    │       compute_cpcv_pbo()   Combinatorial PCV — PBO estimate (S1)
    │       variance_ratio_test() Multi-lag k∈{2,4,8,16} (M4)
    │
    ├─► SL Grid (run_sl_grid)
    │       225+ configs: fixed / trailing / dual / dual_tp / atr_dynamic
    │       3D space: SL × TSL × TP_RR (U3)
    │       ATR-normalized multipliers: [0.5→3.0] (U1)
    │       Regime-conditional adjustments (U2)
    │       Walk-Forward OOS validation per config (U4)
    │
    ├─► Portfolio (compute_efficient_frontier)
    │       Daily-resampled equity curves → covariance (F6)
    │       SLSQP max-Sharpe portfolio weights
    │
    └─► Reporting
            dashboard.html       Chart.js interactive report
            optimization_report.xlsx  68+ sheets
            output/csv/          All tables as CSV
            output/*.png         490+ charts
            output/strategy_*.json   Live-trading configs (Q7)
```

---

## Implemented Research Upgrades

| ID | Reference | Implementation |
|----|-----------|----------------|
| F1 | Hurst 1951, Lo-MacKinlay 1988 | Hurst/ADF gate → BB mean-reversion in sideways markets |
| F2 | Bayesian Optimization | Sobol QMC + RBF surrogate model |
| F3 | López de Prado 2018 | Purged K-Fold CV with temporal embargo |
| F4 | Politis-Romano 1994 | Moving Block Bootstrap for autocorrelated returns |
| F5 | Thorp 2006 | Continuous Kelly: f* = μ/σ² |
| F6 | Markowitz 1952 | Daily-resampled multi-frequency portfolio frontier |
| F7 | Schwager | Gain-to-Pain = Σ(positive rets) / Σ(|negative rets|) |
| M2 | Bailey-LdP 2012 | Deflated Sharpe Ratio: trial-count & kurtosis correction |
| M4 | Lo-MacKinlay 1988 | Multi-lag Variance Ratio Test k∈{2,4,8,16} |
| M5 | Kaufman 2013 | Diffusion approximation risk of ruin |
| Q6 | Benjamini-Hochberg | FDR multiple-comparison correction on 90,000 p-values |
| Q8 | von Neumann-Morgenstern 1944 | Expected utility: U = E[R] − λ/2 · Var[R] |
| S1 | López de Prado 2018 Ch.12 | CPCV: Probability of Backtest Overfitting (PBO) |
| U1 | Volatility-adaptive SL | ATR(14) normalized stop-loss multipliers |
| U3 | Risk/Reward theory | 3D grid: SL × TSL × TP_RR |

---

## Output Files

| File | Description |
|------|-------------|
| `output/dashboard.html` | Interactive Chart.js dashboard |
| `output/optimization_report.xlsx` | Master 68-sheet workbook |
| `output/csv/Summary.csv` | Best EMA pair per asset/TF |
| `output/csv/SL_Permutations_Full.csv` | All SL configs ranked |
| `output/csv/SL_Conclusion_Master.csv` | Best SL with OOS validation |
| `output/csv/RR_Surface.csv` | Risk-Reward 3D surface |
| `output/strategy_<SYM>_<TF>.json` | ccxt/IBKR deploy config |
| `output/*.png` | 490+ charts (equity, heatmap, MC bands…) |

---

## File Overview

| File | Purpose |
|------|---------|
| `ema_sar_optimizer.py` | Core engine (2,450 lines) |
| `export_and_bundle.py` | CSV export + HTML report + ZIP bundler |
| `run.py` | Quick launcher with dep-check |
| `validate.py` | Static health-check (~30 assertions, no data needed) |
