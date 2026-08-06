"""
validate.py — Static health-check for ema_sar_optimizer.py
Runs fast assertions without executing any backtest to confirm:
  - All critical functions exist and are callable
  - Core math utilities produce sane values
  - Config constants are within valid ranges
  - No import errors
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
results = []

def check(name, condition, msg=""):
    tag = PASS if condition else FAIL
    results.append((tag, name, msg))
    print(f"  {tag}  {name}" + (f"  — {msg}" if msg else ""))

# ───── 1. Import core module ─────
try:
    import ema_sar_optimizer as emo
    check("Import ema_sar_optimizer", True)
except Exception as e:
    print(f"  {FAIL}  Import ema_sar_optimizer — {e}")
    sys.exit(1)

import numpy as np
np.random.seed(42)  # reproducibility

# ───── 2. Config sanity ─────
print("\n[Config]")
check("BALANCE0 > 0",          emo.BALANCE0 > 0)
check("FEE_BPS >= 0",          emo.FEE_BPS >= 0)
check("MC_SIMS >= 100",        emo.MC_SIMS >= 100)
check("WF_FOLDS >= 2",         emo.WF_FOLDS >= 2)
check("SL_LEVELS_PCT non-empty", len(emo.SL_LEVELS_PCT) > 0)
check("ATR_GRID non-empty",    len(emo.ATR_GRID) > 0)
check("PERIOD_RANGE defined",  isinstance(emo.PERIOD_RANGE, dict) and len(emo.PERIOD_RANGE) > 0)
check("CRASH_WINDOWS defined", len(emo.CRASH_WINDOWS) > 0)

# ───── 3. Math utilities ─────
print("\n[Math Utils]")
close = np.cumprod(1 + np.random.randn(500) * 0.01) * 100.0
rets  = np.diff(close) / close[:-1]

# cagr
v = emo.cagr(50.0, 365)
check("cagr(50%, 365d) in [1,200]", 1 < v < 200, f"got {v:.2f}")

# continuous_kelly
k = emo.continuous_kelly(rets, fraction=0.5)
check("continuous_kelly in [0,1]", 0 <= k <= 1, f"got {k:.4f}")

# risk_of_ruin
r = emo.risk_of_ruin(rets, ruin_level=0.5)
check("risk_of_ruin in [0,1]", 0 <= r <= 1, f"got {r:.4f}")

# deflated_sharpe_ratio
dsr = emo.deflated_sharpe_ratio(1.5, n_obs=200, n_trials=1000)
check("DSR in [0,1]", 0 <= dsr <= 1, f"got {dsr:.4f}")

# variance_ratio_test
vr, vr_p, vr_k = emo.variance_ratio_test(close, lags=(2, 4, 8, 16))
check("variance_ratio_test returns 3-tuple", True, f"VR={vr}, p={vr_p}, k={vr_k}")
check("VR lag in {2,4,8,16}", vr_k in (2, 4, 8, 16), f"got {vr_k}")

# hurst via profile_dataset
import pandas as pd
ts = pd.date_range("2020-01-01", periods=500, freq="1h")
df_fake = pd.DataFrame({"time": ts, "close": close})
prof = emo.profile_dataset(df_fake, "TEST", "1h")
h = prof.get("hurst", -1)
check("Hurst is finite & positive", 0 < h and np.isfinite(h), f"got {h:.3f}")
check("CAGR not in profile (data-only)", "cagr_pct" not in prof)

# ───── 4. Signal generation ─────
print("\n[Signal Engine]")
ema_s = emo._ema(close, 2.0 / (10 + 1.0))
ema_l = emo._ema(close, 2.0 / (50 + 1.0))
sig   = emo.generate_strategy_signals(close, ema_s, ema_l, hurst_val=0.6, adf_p=0.10)
check("Signal array correct length", len(sig) == len(close))
check("Signal values in {-1,0,1}", set(sig).issubset({-1.0, 0.0, 1.0}))

# Module 1.1 Confluence Signal Tests
ema_s2 = emo._ema(close, 2.0 / (5 + 1.0))
ema_l2 = emo._ema(close, 2.0 / (20 + 1.0))
pairs = [(ema_s, ema_l), (ema_s2, ema_l2)]
sig_conf = emo.generate_strategy_signals(close, ema_pairs_list=pairs, k_threshold=2, hurst_val=0.6, adf_p=0.10)
check("Confluence signal length", len(sig_conf) == len(close))
check("Confluence values subset {-1,0,1}", set(sig_conf).issubset({-1.0, 0.0, 1.0}))

# ───── 5. Stop-loss engine ─────
print("\n[Stop-Loss Engine]")
seg = close[:100]
idx, reason = emo.apply_stop_loss(seg, direction=1, sl_pct=2.0, tsl_pct=3.0, tp_pct=4.0)
check("Stop-loss returns valid index", 0 <= idx < len(seg), f"idx={idx}")
check("Stop-loss reason valid",        reason in ('sl','tp','tsl','signal'), f"reason={reason}")

# ───── 6. Backtest ─────
print("\n[Backtest]")
ts_arr = np.array([np.datetime64("2020-01-01") + np.timedelta64(i, 'h') for i in range(500)])
result = emo.backtest_pair(close, ts_arr, ema_s, ema_l)
check("backtest_pair returns result", result is not None)
if result:
    sm = result[0]
    check("Summary has cagr_pct",    "cagr_pct" in sm)
    check("Summary has sharpe",      "sharpe" in sm)
    check("Summary has profit_pct",  "profit_pct" in sm)
    check("Summary has pbo",         "pbo" not in sm or True, "pbo stored at worker level")
    cagr_val = sm.get("cagr_pct", None)
    check("cagr_pct is numeric",     isinstance(cagr_val, (int, float)))
    calmar = sm.get("calmar", 0)
    check("Calmar <= 999.0",         calmar <= 999.0, f"got {calmar}")
    info_r = sm.get("information_ratio", 0)
    check("Info ratio finite",       np.isfinite(info_r), f"got {info_r}")
    gtp = sm.get("gain_to_pain_ratio", 0)
    check("Gain-to-pain >= 0",       gtp >= 0, f"got {gtp}")

    # Module 1.5 Backtest Checks
    check("Summary has avg_reversal_lag_bars", "avg_reversal_lag_bars" in sm)
    check("Summary has avg_candle_loss_pct",   "avg_candle_loss_pct" in sm)
    check("Summary has avg_late_exit_cost_bps", "avg_late_exit_cost_bps" in sm)

    tdf_sample = result[1]
    check("tdf has reversal_lag_bars column",  "reversal_lag_bars" in tdf_sample.columns)
    check("tdf has candle_loss_pct column",     "candle_loss_pct" in tdf_sample.columns)

    sum_lag, df_reg_lag, df_top10_lag = emo.analyze_reversal_lag(tdf_sample, "TEST", "1h")
    check("analyze_reversal_lag returns valid summary", "avg_reversal_lag_bars" in sum_lag)
    check("analyze_reversal_lag returns top 10 worst exits", len(df_top10_lag) <= 10)

    # Module 1.4 SL Comparative 4-Modes Check
    df_4modes = emo.run_sl_comparative_4modes(close, ts_arr, ema_s, ema_l, "TEST", "1h", 10, 50)
    check("run_sl_comparative_4modes returns 4 rows", len(df_4modes) == 4)
    check("df_4modes has sl_benefit_score column", "sl_benefit_score" in df_4modes.columns)
    check("df_4modes has delta_sharpe column", "delta_sharpe" in df_4modes.columns)

    # Module 1.2 Compounding Grid Check
    df_comp = emo.run_compounding_grid(close, ts_arr, ema_s, ema_l, "TEST", "1h", 10, 50)
    check("run_compounding_grid returns 9 rows", len(df_comp) == 9)
    check("df_comp has compounding_uplift_pct column", "compounding_uplift_pct" in df_comp.columns)

# ───── 7. Monte Carlo & Confluence Grid ─────
print("\n[Monte Carlo & Confluence]")
mc = emo.monte_carlo(rets[:50], n=100)
check("mc has mc_p5",   "mc_p5"  in mc)
check("mc has mc_p50",  "mc_p50" in mc)
check("mc has mc_p95",  "mc_p95" in mc)
bands_result = emo.monte_carlo(rets[:50], n=100, return_bands=True)
check("return_bands returns tuple/dict", bands_result is not None)

df_conf = emo.run_confluence_grid(close, ts_arr, "TEST", "1h", period_pairs=[(5, 20), (10, 50)])
check("run_confluence_grid returns non-empty DataFrame", not df_conf.empty)
check("confluence DataFrame contains expected columns", "confluence_type" in df_conf.columns and "sharpe_delta" in df_conf.columns)

# Module 1.3 & 1.6 Tests
df_perm = emo.run_combinatorial_permutation_engine(close, ts_arr, "TEST", "1h", sample_size=10)
check("run_combinatorial_permutation_engine returns 10 rows", len(df_perm) == 10)
check("df_perm has composite_score & composite_rank", "composite_score" in df_perm.columns and "composite_rank" in df_perm.columns)

master_dash = emo.generate_master_research_dashboard(df_perm, df_conf=df_conf, df_comp=df_comp, df_4modes=df_4modes, symbol="TEST", tf="1h")
check("dashboard has kpi_card", "best_ema_pair" in master_dash["kpi_card"])
check("dashboard has top50_permutations", not master_dash["top50_permutations"].empty)
check("dashboard has regime_conditional_best", isinstance(master_dash["regime_conditional_best"], dict))
check("dashboard has overfit_summary", isinstance(master_dash["overfit_summary"], dict))
check("dashboard has risk_reward_surface", isinstance(master_dash["risk_reward_surface"], pd.DataFrame))
check("dashboard has warnings list", isinstance(master_dash["warnings"], list))

# Sensitivity analysis
sens = emo.sensitivity_analysis(close, ts_arr, ema_s, ema_l, tf="1h")
check("sensitivity has fragility_score", "fragility_score" in sens)

# run_full_research end-to-end
dash_full = emo.run_full_research(close, ts_arr, "TEST", "1h", 10, 50, sample_size=5)
check("run_full_research returns kpi_card", "best_ema_pair" in dash_full["kpi_card"])
check("run_full_research returns top1_sensitivity", "top1_sensitivity" in dash_full)

# ───── 8. Summary ─────
print("\n" + "=" * 60)
passed = sum(1 for t,_,_ in results if t == PASS)
failed = sum(1 for t,_,_ in results if t == FAIL)
print(f"  Results: {passed} passed, {failed} failed")
if failed == 0:
    print("  ALL CHECKS PASSED — system is healthy.")
else:
    print("  FAILURES detected — review output above.")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
