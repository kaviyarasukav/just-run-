import os
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------- CONFIG ---------------------------------------
SYMBOLS    = ["ETH", "SOL", "XRP", "XAU", "XAG", "OIL", "TSLA", "GOOGL", "NVDA"]
TIMEFRAMES = ["5m", "15m", "30m", "1h", "2h", "3h", "4h"]
CUTOFF     = "2024-01-01"
BALANCE0   = 100_000.0
MIN_P      = 2
MAX_P      = 300
FEE_BPS    = 0.0
DATA_XLSX  = "./data/market_data.xlsx"
OUT_DIR    = "./output"
CSV_DIR    = "./output/csv"
MC_SIMS    = 500
WF_FOLDS   = 3
REGIME_WIN = 50
ROBUST_R   = 5

SESSION_BOUNDS = [
    ("Asia", 0, 8), ("London", 8, 13), ("NY_overlap", 13, 16),
    ("NY", 16, 21), ("Late", 21, 24),
]

def session_of_hour(h):
    for name, lo, hi in SESSION_BOUNDS:
        if lo <= h < hi:
            return name
    return "Unknown"


# ----------------------------- DATA PROFILING -------------------------------
def profile_dataset(df, symbol, tf):
    close = df["close"].values
    ts    = pd.to_datetime(df["time"]).values
    n     = len(df)
    s0, sn = float(close[0]), float(close[-1])
    bh     = (sn / s0 - 1.0) * 100.0
    t0     = pd.to_datetime(ts[0]); tn = pd.to_datetime(ts[-1])
    days   = max(1.0, (tn - t0).total_seconds() / 86400.0)
    bpd    = n / days
    br     = np.diff(close) / close[:-1]
    vol_b  = float(br.std() * 100.0) if len(br) else 0.0
    ann_v  = vol_b * np.sqrt(bpd * 365)
    regime = "Bullish" if bh > 10 else ("Bearish" if bh < -10 else "Sideways")

    def hurst(x, min_len=8):
        if len(x) < min_len * 2:
            return 0.5
        splits = max(2, len(x) // min_len)
        chunk  = len(x) // splits
        rs_vals = []
        for i in range(splits):
            seg = x[i*chunk:(i+1)*chunk]
            dev = seg - seg.mean()
            R   = dev.cumsum().max() - dev.cumsum().min()
            S   = seg.std()
            if S > 0:
                rs_vals.append(R / S)
        return 0.5 if not rs_vals else float(np.log(np.mean(rs_vals)) / np.log(chunk + 1e-9))

    return dict(
        symbol=symbol, timeframe=tf, bars=n,
        start=t0.strftime("%Y-%m-%d"), end=tn.strftime("%Y-%m-%d"),
        days=round(days, 1), start_price=s0, end_price=sn,
        bh_return_pct=round(bh, 2), bar_vol_pct=round(vol_b, 4),
        ann_vol_pct=round(ann_v, 2), regime=regime,
        hurst=round(hurst(close), 3),
        skewness=round(float(stats.skew(br)), 3),
        kurtosis=round(float(stats.kurtosis(br)), 3),
    )


# ----------------------------- CORE ENGINE ----------------------------------
def _ema(close, alpha):
    y, _ = lfilter([alpha], [1, -(1-alpha)], close,
                   zi=np.array([close[0] * (1-alpha)]))
    return y

def ema_matrix(close, periods):
    out = np.empty((len(periods), len(close)), dtype=np.float64)
    for i, p in enumerate(periods):
        out[i] = _ema(close, 2.0 / (p + 1.0))
    return out

def _max_consec(arr, val):
    mx = cur = 0
    for x in arr:
        cur = (cur + 1) if x == val else 0
        mx  = max(mx, cur)
    return mx

def _drawdown(eq):
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / peak

def kelly_criterion(win_rate, payoff_ratio):
    q = 1.0 - win_rate
    if payoff_ratio <= 0:
        return 0.0
    return max(0.0, (win_rate * payoff_ratio - q) / payoff_ratio)

def risk_of_ruin(win_rate, payoff_ratio, n_trades=100):
    if payoff_ratio <= 0 or win_rate <= 0:
        return 1.0
    edge = win_rate - (1.0 - win_rate) / payoff_ratio
    if edge >= 1.0:
        return 0.0
    if edge <= 0.0:
        return 1.0
    return float(((1 - edge) / (1 + edge)) ** n_trades)

def cagr(total_return_pct, days):
    return float(((1 + total_return_pct / 100.0) ** (365.0 / max(1, days)) - 1) * 100.0)

def ljung_box_p(rets, lags=10):
    n = len(rets)
    if n <= lags:
        return 1.0
    ac = np.array([np.corrcoef(rets[:-k], rets[k:])[0, 1] for k in range(1, lags + 1)])
    Q  = n * (n + 2) * np.sum(ac**2 / (n - np.arange(1, lags + 1)))
    return float(stats.chi2.sf(Q, df=lags))

def regime_labels(close, window=REGIME_WIN):
    out = []
    for i in range(len(close)):
        r = (close[i] / close[max(0, i - window)] - 1.0) * 100.0
        out.append("Bullish" if r > 3.0 else "Bearish" if r < -3.0 else "Sideways")
    return out

def _unshifted_profit(close, raw_sig, fee):
    n = len(close)
    flips = np.where(raw_sig[1:] != raw_sig[:-1])[0] + 1
    ei = np.concatenate(([0], flips))
    xi = np.concatenate((flips, [n - 1]))
    eq = BALANCE0
    for i0, i1 in zip(ei, xi):
        if i1 > i0:
            eq += eq * ((close[i1] / close[i0] - 1.0) * raw_sig[i0] - 2 * fee)
    return float((eq / BALANCE0 - 1.0) * 100.0)


# ----------------------------- MONTE CARLO ----------------------------------
def monte_carlo(trade_rets, n=MC_SIMS):
    if len(trade_rets) < 2:
        return dict(mc_p5=BALANCE0, mc_p50=BALANCE0, mc_p95=BALANCE0,
                    mc_prob_ruin=0.0, mc_sharpe_p5=0.0, mc_sharpe_p95=0.0)
    rng    = np.random.default_rng(42)
    T      = len(trade_rets)
    sims   = rng.choice(trade_rets, (n, T), replace=True)
    finals = BALANCE0 * np.prod(1 + sims, axis=1)
    m      = sims.mean(axis=1); s = sims.std(axis=1)
    sharps = np.where(s > 0, m / s * np.sqrt(T), 0.0)
    return dict(
        mc_p5  =round(float(np.percentile(finals, 5)), 2),
        mc_p50 =round(float(np.percentile(finals, 50)), 2),
        mc_p95 =round(float(np.percentile(finals, 95)), 2),
        mc_prob_ruin   =round(float((finals < BALANCE0 * 0.5).mean() * 100), 2),
        mc_sharpe_p5   =round(float(np.percentile(sharps, 5)), 2),
        mc_sharpe_p95  =round(float(np.percentile(sharps, 95)), 2),
    )

def mc_equity_bands(trade_rets, n=MC_SIMS):
    rng  = np.random.default_rng(42)
    T    = len(trade_rets)
    sims = rng.choice(trade_rets, (n, T), replace=True)
    curves = BALANCE0 * np.cumprod(1 + sims, axis=1)
    return (np.percentile(curves, 5,  axis=0),
            np.percentile(curves, 50, axis=0),
            np.percentile(curves, 95, axis=0))


# ----------------------------- WALK-FORWARD OOS -----------------------------
def walk_forward(close, ts, bx, by, n_folds=WF_FOLDS, fee=FEE_BPS / 10000.0):
    n = len(close); fs = n // (n_folds + 1)
    if fs < MAX_P + 10:
        return None
    oos = []
    for k in range(1, n_folds + 1):
        s0, s1 = k * fs, min((k + 1) * fs, n)
        if s1 <= s0:
            continue
        seg  = close[s0:s1]
        ts_s = ts[s0:s1]
        all_p = sorted({bx, by})
        mats  = ema_matrix(seg, np.array(all_p))
        pmap  = {p: i for i, p in enumerate(all_p)}
        out   = backtest_pair(seg, ts_s, mats[pmap[bx]], mats[pmap[by]])
        if out:
            oos.append(out[0]["profit_pct"])
    if not oos:
        return None
    return dict(
        wf_oos_avg=round(np.mean(oos), 2),
        wf_oos_min=round(np.min(oos), 2),
        wf_oos_max=round(np.max(oos), 2),
        wf_oos_positive_folds=int(sum(r > 0 for r in oos)),
        wf_oos_total_folds   =len(oos),
    )


# ----------------------------- ROBUSTNESS SCORE -----------------------------
def robustness_score(close, ts, bx, by, radius=ROBUST_R, fee=FEE_BPS / 10000.0):
    xs = range(max(MIN_P, bx - radius), min(MAX_P, bx + radius) + 1)
    ys = range(max(MIN_P, by - radius), min(MAX_P, by + radius) + 1)
    all_p = sorted(set(xs) | set(ys))
    mats  = ema_matrix(close, np.array(all_p))
    pmap  = {p: i for i, p in enumerate(all_p)}
    total = ok = 0
    for x in xs:
        for y in ys:
            if x >= y:
                continue
            out = backtest_pair(close, ts, mats[pmap[x]], mats[pmap[y]])
            if out:
                total += 1
                if out[0]["profit_pct"] > 0:
                    ok += 1
    return round(float(ok / total * 100) if total > 0 else 0.0, 1)


# ----------------------------- REGIME-SPECIFIC BEST EMA ---------------------
def regime_best_emas(df, symbol, tf, coarse_step=10):
    close  = df["close"].to_numpy(dtype=np.float64)
    ts_arr = pd.to_datetime(df["time"]).to_numpy()
    rlab   = np.array(regime_labels(close))
    result = {}
    periods = list(range(MIN_P, MAX_P + 1, coarse_step))
    all_p   = sorted(set(periods))
    mats    = ema_matrix(close, np.array(all_p))
    pmap    = {p: i for i, p in enumerate(all_p)}

    for regime in ["Bullish", "Bearish", "Sideways"]:
        idx = np.where(rlab == regime)[0]
        if len(idx) < 50:
            continue
        seg_close = close[idx]
        seg_ts    = ts_arr[idx]
        best_pf   = -1.0; best_pair = None
        for x in periods:
            for y in periods:
                if x >= y:
                    continue
                out = backtest_pair(seg_close, seg_ts,
                                    mats[pmap[x]][idx], mats[pmap[y]][idx])
                if out and out[0]["profit_factor"] > best_pf:
                    best_pf   = out[0]["profit_factor"]
                    best_pair = (x, y)
        if best_pair:
            result[regime] = dict(ema_x=best_pair[0], ema_y=best_pair[1],
                                   profit_factor=round(best_pf, 2))
    return result


# ----------------------------- BACKTEST ENGINE ------------------------------
def backtest_pair(close, ts, ema_s, ema_l,
                  balance0=BALANCE0, fee_bps=FEE_BPS):
    n       = len(close)
    raw_sig = np.where(ema_s > ema_l, 1.0, -1.0)
    fee     = fee_bps / 10000.0
    no_shift_ret = _unshifted_profit(close, raw_sig, fee)

    position = np.roll(raw_sig, 1); position[0] = 0.0
    flips    = np.where(raw_sig[1:] != raw_sig[:-1])[0] + 1
    ei       = np.concatenate(([1], flips))
    xi       = np.concatenate((flips, [n - 1]))
    ei, xi   = ei[ei < n], xi[:len(ei)]
    reg_lab  = regime_labels(close)

    trades = []
    equity = balance0
    for i0, i1 in zip(ei, xi):
        if i1 <= i0:
            continue
        d   = position[i0]
        p0, p1 = close[i0], close[i1]
        seg_u  = ((close[i0:i1+1] / p0 - 1.0) * d) - fee
        mfe    = float(seg_u.max() * 100.0)
        mae    = float(seg_u.min() * 100.0)
        tr     = float(seg_u[-1])
        pnl    = equity * tr
        equity += pnl
        eff    = (tr * 100.0 / mfe) if mfe > 0 else 0.0
        edge_r = (mfe / abs(mae)) if mae != 0 else mfe
        trades.append((i0, i1, ts[i0], ts[i1], p0, p1, int(d),
                       tr, pnl, int(i1 - i0), mae, mfe, eff, edge_r, reg_lab[i0]))

    if not trades:
        return None

    tdf = pd.DataFrame(trades, columns=[
        "entry_i","exit_i","entry_time","exit_time","entry_price","exit_price",
        "direction","ret","pnl","hold_bars","mae_pct","mfe_pct",
        "trade_efficiency_pct","edge_ratio","entry_regime"
    ])
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["exit_time"]  = pd.to_datetime(tdf["exit_time"])
    tdf["hold_sec"]   = (tdf["exit_time"] - tdf["entry_time"]).dt.total_seconds()
    tdf["hour"]       = tdf["entry_time"].dt.hour
    tdf["weekday"]    = tdf["entry_time"].dt.day_name()
    tdf["session"]    = tdf["hour"].apply(session_of_hour)
    tdf["year"]       = tdf["entry_time"].dt.year
    tdf["month"]      = tdf["entry_time"].dt.month
    tdf["equity"]     = balance0 + tdf["pnl"].cumsum()
    tdf["cum_profit"] = tdf["pnl"].clip(lower=0).cumsum()
    tdf["cum_loss"]   = (-tdf["pnl"].clip(upper=0)).cumsum()

    wins   = tdf[tdf.pnl > 0];  losses = tdf[tdf.pnl <= 0]
    gp     = float(wins.pnl.sum());  gl = float(-losses.pnl.sum())
    pf     = gp / gl if gl > 0 else np.inf
    total_ret = (equity / balance0 - 1.0) * 100.0
    bh_ret    = (close[-1] / close[0] - 1.0) * 100.0
    alpha_ret = total_ret - bh_ret

    eq_curve = np.concatenate(([balance0], tdf["equity"].values))
    dd_arr   = _drawdown(eq_curve)
    mdd      = float(dd_arr.min() * 100.0)
    mx_dd_bars = c = 0
    for v in (dd_arr < 0):
        c = (c+1) if v else 0; mx_dd_bars = max(mx_dd_bars, c)

    tr_rets  = tdf["ret"].values
    mu, sig  = tr_rets.mean(), tr_rets.std()
    sharpe   = (mu / sig * np.sqrt(len(tr_rets))) if sig > 0 else 0.0
    dstd     = tr_rets[tr_rets < 0].std() if len(tr_rets[tr_rets < 0]) > 0 else 0.0
    sortino  = (mu / dstd * np.sqrt(len(tr_rets))) if dstd > 0 else 0.0
    calmar   = (total_ret / abs(mdd)) if mdd < 0 else np.inf

    # New Analytics: Omega, Ulcer, Recovery Factor, Tail Ratio
    pos_sum  = float(tr_rets[tr_rets > 0].sum()) if len(tr_rets[tr_rets > 0]) else 0.0
    neg_sum  = float(abs(tr_rets[tr_rets < 0].sum())) if len(tr_rets[tr_rets < 0]) else 0.0
    omega    = (pos_sum / neg_sum) if neg_sum > 0 else np.inf
    ulcer    = float(np.sqrt(np.mean((dd_arr * 100.0)**2)))
    recovery = (total_ret / abs(mdd)) if mdd < 0 else total_ret
    p95      = float(np.percentile(tr_rets, 95)) if len(tr_rets) > 5 else 0.0
    p5       = float(abs(np.percentile(tr_rets, 5))) if len(tr_rets) > 5 else 1e-9
    tail_r   = (p95 / p5) if p5 > 0 else 0.0

    wr       = len(wins) / len(tdf)
    lr       = 1.0 - wr
    avg_w    = float(wins["ret"].mean() * 100.0) if len(wins) else 0.0
    avg_l    = float(losses["ret"].mean() * 100.0) if len(losses) else 0.0
    avg_wd   = float(wins["pnl"].mean()) if len(wins) else 0.0
    avg_ld   = float(losses["pnl"].mean()) if len(losses) else 0.0
    payoff   = (avg_wd / abs(avg_ld)) if avg_ld < 0 else np.inf
    exp_pct  = wr * avg_w + lr * avg_l
    exp_dol  = wr * avg_wd + lr * avg_ld
    sqn      = (exp_pct / (sig * 100.0) * np.sqrt(len(tdf))) if sig > 0 else 0.0

    # Long vs Short split
    longs  = tdf[tdf.direction == 1]; shorts = tdf[tdf.direction == -1]
    l_win  = (longs.pnl > 0).mean() * 100.0 if len(longs) else 0.0
    s_win  = (shorts.pnl > 0).mean() * 100.0 if len(shorts) else 0.0
    l_prof = (np.prod(1 + longs.ret.values) - 1.0) * 100.0 if len(longs) else 0.0
    s_prof = (np.prod(1 + shorts.ret.values) - 1.0) * 100.0 if len(shorts) else 0.0

    # Kelly equity curve
    kelly_f = kelly_criterion(wr, payoff if not np.isinf(payoff) else 10.0)
    k_eq    = [balance0]
    for r_i in tr_rets:
        k_eq.append(k_eq[-1] * (1.0 + kelly_f * r_i))
    kelly_prof = float((k_eq[-1] / balance0 - 1.0) * 100.0)

    # Statistical tests
    t_stat, p_val  = stats.ttest_1samp(tr_rets, 0) if len(tr_rets) > 1 else (0.0, 1.0)
    lb_p           = ljung_box_p(tr_rets)
    is_significant = bool(p_val < 0.05)
    has_autocorr   = bool(lb_p < 0.05)
    ror            = risk_of_ruin(wr, payoff if not np.isinf(payoff) else 10.0)

    mc      = monte_carlo(tr_rets)
    reg_prf = {}
    for regime, grp in tdf.groupby("entry_regime"):
        rp = grp["ret"].values
        reg_prf[regime] = dict(
            trades=len(grp),
            profit_pct=round(float((np.prod(1+rp)-1)*100), 2),
            win_rate_pct=round(float((rp>0).mean()*100), 2),
        )

    monthly = tdf.groupby(["year","month"]).agg(
        trades=("pnl","count"),
        profit_pct=("ret", lambda r: round((np.prod(1+r)-1)*100, 2)),
        cum_pnl=("pnl","sum"),
    ).reset_index()
    monthly["period"] = monthly["year"].astype(str)+"-"+monthly["month"].astype(str).str.zfill(2)

    summary = dict(
        final_balance   =round(equity, 2),
        profit_pct      =round(total_ret, 2),
        profit_pct_no_shift=round(no_shift_ret, 2),
        lookahead_delta =round(no_shift_ret - total_ret, 2),
        bh_return_pct   =round(bh_ret, 2),
        alpha_pct       =round(alpha_ret, 2),
        profit_factor   =round(pf, 2) if not np.isinf(pf) else 999.0,
        sharpe          =round(sharpe, 2),
        sortino         =round(sortino, 2),
        calmar          =round(calmar, 2) if not np.isinf(calmar) else 999.0,
        omega_ratio     =round(omega, 2) if not np.isinf(omega) else 999.0,
        ulcer_index     =round(ulcer, 2),
        recovery_factor =round(recovery, 2),
        tail_ratio      =round(tail_r, 2),
        max_drawdown_pct=round(mdd, 2),
        max_dd_bars     =mx_dd_bars,
        total_trades    =len(tdf),
        win_rate_pct    =round(wr * 100.0, 2),
        payoff_ratio    =round(payoff, 2) if not np.isinf(payoff) else 999.0,
        expectancy_pct  =round(exp_pct, 2),
        expectancy_dol  =round(exp_dol, 2),
        sqn             =round(sqn, 2),
        kelly_fraction  =round(kelly_f, 4),
        kelly_profit_pct=round(kelly_prof, 2),
        risk_of_ruin_pct=round(ror * 100.0, 2),
        long_trades     =len(longs),
        long_win_rate   =round(l_win, 2),
        long_profit_pct =round(l_prof, 2),
        short_trades    =len(shorts),
        short_win_rate  =round(s_win, 2),
        short_profit_pct=round(s_prof, 2),
        max_consec_wins =_max_consec((tdf.pnl > 0).values, True),
        max_consec_loss =_max_consec((tdf.pnl > 0).values, False),
        avg_win_pct     =round(avg_w, 2),
        avg_loss_pct    =round(avg_l, 2),
        avg_hold_bars   =round(float(tdf.hold_bars.mean()), 1),
        avg_mae_pct     =round(float(tdf.mae_pct.mean()), 2),
        avg_mfe_pct     =round(float(tdf.mfe_pct.mean()), 2),
        avg_edge_ratio  =round(float(tdf.edge_ratio.mean()), 2),
        t_stat          =round(float(t_stat), 3),
        p_value         =round(float(p_val), 4),
        is_significant  =is_significant,
        ljung_box_p     =round(lb_p, 4),
        has_autocorrelation=has_autocorr,
        **mc,
    )
    return summary, tdf, eq_curve, reg_prf, monthly


# ----------------------------- GRID ENGINE ----------------------------------
@dataclass
class GridResult:
    rows: list = field(default_factory=list)
    def add(self, s, t, x, y, sm):
        self.rows.append(dict(symbol=s, timeframe=t, ema_x=x, ema_y=y, **sm))
    def to_df(self):
        return pd.DataFrame(self.rows)

def run_grid(df, symbol, tf, xp, yp):
    close = df["close"].to_numpy(dtype=np.float64)
    ts    = pd.to_datetime(df["time"]).to_numpy()
    allp  = sorted(set(xp)|set(yp))
    mats  = ema_matrix(close, np.array(allp))
    pmap  = {p: i for i, p in enumerate(allp)}
    res   = GridResult(); best = None
    for x in xp:
        for y in yp:
            if x >= y: continue
            out = backtest_pair(close, ts, mats[pmap[x]], mats[pmap[y]])
            if out is None: continue
            sm, tdf, eq_c, reg_p, mty = out
            res.add(symbol, tf, int(x), int(y), sm)
            if best is None or sm["profit_factor"] > best[0]["profit_factor"]:
                best = (sm, tdf, eq_c, int(x), int(y), reg_p, mty)
    return res, best

def coarse_fine(df, symbol, tf, step=5, r=ROBUST_R):
    cp = list(range(MIN_P, MAX_P+1, step))
    if cp[-1] != MAX_P: cp.append(MAX_P)
    res, best = run_grid(df, symbol, tf, cp, cp)
    if best is None: return res, best
    bx, by = best[3], best[4]
    fx = list(range(max(MIN_P, bx-r*2), min(MAX_P, bx+r*2)+1))
    fy = list(range(max(MIN_P, by-r*2), min(MAX_P, by+r*2)+1))
    fres, fbest = run_grid(df, symbol, tf, fx, fy)
    combined = GridResult(); combined.rows = res.rows + fres.rows
    return combined, (fbest if (fbest and fbest[0]["profit_factor"] >= best[0]["profit_factor"]) else best)


# ----------------------------- DATA LOADING --------------------------------
def load_all(path):
    xl  = pd.ExcelFile(path); out = {}
    for sh in xl.sheet_names:
        if "_" not in sh: continue
        sym, tf = sh.rsplit("_", 1)
        df = xl.parse(sh)
        if "time" not in df.columns or "close" not in df.columns: continue
        df["time"] = pd.to_datetime(df["time"])
        df = df[df["time"] < CUTOFF].sort_values("time").reset_index(drop=True)
        if len(df) < MIN_P + 10: continue
        out[(sym, tf)] = df[["time","close"]]
    return out

def make_demo(path=DATA_XLSX, bars=4000, seed=42):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng  = np.random.default_rng(seed)
    freq = {"5m":"5min","15m":"15min","30m":"30min","1h":"h","2h":"2h","3h":"3h","4h":"4h"}
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for sym in SYMBOLS:
            base  = rng.uniform(20, 3000)
            vol   = rng.uniform(0.006, 0.02)
            drift = rng.uniform(-0.0001, 0.0003)
            for tf in TIMEFRAMES:
                t     = pd.date_range("2022-01-01", periods=bars, freq=freq[tf])
                price = base * np.exp(np.cumsum(rng.normal(drift, vol, bars)))
                pd.DataFrame({"time": t, "close": price}).to_excel(
                    w, sheet_name=f"{sym}_{tf}"[:31], index=False)
    print(f"[demo] synthetic data -> {path}")


# ----------------------------- VISUALS --------------------------------------
def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#0a0f1e", "axes.facecolor": "#0d1526",
        "axes.edgecolor": "#1e3a5f",   "axes.labelcolor": "#94a3b8",
        "xtick.color": "#64748b",      "ytick.color": "#64748b",
        "text.color": "#e2e8f0",       "grid.color": "#1e3a5f",
        "grid.linewidth": 0.6,         "font.family": "DejaVu Sans",
    })

def make_visuals(symbol, tf, grid_df, bx, by, eq_curve, tdf):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()

    # 1. Equity + Drawdown
    dd = _drawdown(eq_curve) * 100.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(eq_curve, color="#38bdf8", lw=1.8, label="Equity")
    ax1.fill_between(range(len(eq_curve)), BALANCE0, eq_curve,
                     where=np.array(eq_curve) >= BALANCE0, alpha=0.10, color="#38bdf8")
    ax1.fill_between(range(len(eq_curve)), BALANCE0, eq_curve,
                     where=np.array(eq_curve) < BALANCE0,  alpha=0.14, color="#f87171")
    ax1.axhline(BALANCE0, color="#475569", ls="--", lw=0.8)
    ax1.set_title(f"{symbol} {tf} - EMA({bx},{by}) Equity & Drawdown", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Balance ($)"); ax1.legend(fontsize=9); ax1.grid(True)
    ax2.fill_between(range(len(dd)), dd, 0, color="#f87171", alpha=0.55)
    ax2.set_ylabel("DD %"); ax2.set_xlabel("Trade #"); ax2.grid(True)
    fig.tight_layout(); fig.savefig(p:=f"{OUT_DIR}/equity_{symbol}_{tf}.png", dpi=130); plt.close(fig)
    eq_png = p

    # 2. Parameter Heatmap
    piv = grid_df.pivot_table("profit_factor", "ema_x", "ema_y", aggfunc="max")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="plasma",
                   extent=[piv.columns.min(), piv.columns.max(),
                            piv.index.min(),   piv.index.max()])
    fig.colorbar(im, ax=ax, label="Profit Factor")
    ax.scatter([by], [bx], color="#4ade80", s=90, zorder=5, label=f"Best ({bx},{by})")
    ax.set_xlabel("EMA y"); ax.set_ylabel("EMA x")
    ax.set_title(f"{symbol} {tf} - Parameter Space Heatmap", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(p:=f"{OUT_DIR}/heatmap_{symbol}_{tf}.png", dpi=130); plt.close(fig)
    hm_png = p

    # 3. MAE vs MFE Scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(tdf.mae_pct, tdf.mfe_pct, c=tdf["ret"]*100,
                    cmap="RdYlGn", edgecolors="none", alpha=0.75, s=36)
    fig.colorbar(sc, ax=ax, label="Trade Return %")
    ax.axhline(0, color="#475569", lw=0.7); ax.axvline(0, color="#475569", lw=0.7)
    ax.set_xlabel("MAE %"); ax.set_ylabel("MFE %")
    ax.set_title(f"{symbol} {tf} - Execution Quality (MAE vs MFE)", fontsize=11, fontweight="bold")
    ax.grid(True); fig.tight_layout()
    fig.savefig(p:=f"{OUT_DIR}/mae_mfe_{symbol}_{tf}.png", dpi=130); plt.close(fig)
    mae_png = p

    # 4. Return Distribution
    rp = tdf["ret"] * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rp[rp > 0], bins=25, color="#4ade80", alpha=0.80, label="Wins", edgecolor="none")
    ax.hist(rp[rp <= 0], bins=25, color="#f87171", alpha=0.80, label="Losses", edgecolor="none")
    ax.axvline(rp.mean(), color="#38bdf8", ls="--", lw=1.6, label=f"Mean {rp.mean():.2f}%")
    ax.axvline(0, color="#475569", lw=0.8)
    ax.set_xlabel("Trade Return %"); ax.set_ylabel("Count")
    ax.set_title(f"{symbol} {tf} - Trade Return Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True); fig.tight_layout()
    fig.savefig(p:=f"{OUT_DIR}/dist_{symbol}_{tf}.png", dpi=130); plt.close(fig)
    dist_png = p

    # 5. Session Breakdown
    sess = tdf.groupby("session").agg(
        profit_pct=("ret", lambda r: (np.prod(1+r)-1)*100),
        win_rate  =("ret", lambda r: (r>0).mean()*100),
    ).reindex(["Asia","London","NY_overlap","NY","Late"]).fillna(0)
    fig, ax1 = plt.subplots(figsize=(7, 4)); ax2 = ax1.twinx()
    colors = ["#4ade80" if v >= 0 else "#f87171" for v in sess.profit_pct]
    ax1.bar(sess.index, sess.profit_pct, color=colors, alpha=0.8)
    ax2.plot(sess.index, sess.win_rate, color="#fbbf24", marker="o", lw=2)
    ax1.set_ylabel("Profit %"); ax2.set_ylabel("Win Rate %")
    ax1.set_title(f"{symbol} {tf} - Session Breakdown", fontsize=11, fontweight="bold")
    ax1.grid(True); fig.tight_layout()
    fig.savefig(p:=f"{OUT_DIR}/session_{symbol}_{tf}.png", dpi=130); plt.close(fig)
    sess_png = p

    # 6. Monte Carlo Fan Chart
    tr = tdf["ret"].values
    p5, p50, p95 = mc_equity_bands(tr)
    actual = BALANCE0 * np.cumprod(1 + tr)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(tr))
    ax.fill_between(x, p5, p95, alpha=0.18, color="#38bdf8", label=f"P5-P95 ({MC_SIMS} sims)")
    ax.plot(p50,   color="#38bdf8", lw=1.5, label="MC Median")
    ax.plot(actual, color="#fbbf24", lw=1.8, label="Actual")
    ax.axhline(BALANCE0, color="#475569", ls="--", lw=0.8)
    ax.set_xlabel("Trade #"); ax.set_ylabel("Balance ($)")
    ax.set_title(f"{symbol} {tf} - Monte Carlo Equity Fan", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True); fig.tight_layout()
    fig.savefig(p:=f"{OUT_DIR}/mc_{symbol}_{tf}.png", dpi=130); plt.close(fig)
    mc_png = p

    # 7. Hour x Weekday PnL Heatmap
    try:
        wday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = tdf.pivot_table("pnl", index="weekday", columns="hour", aggfunc="sum")
        pivot = pivot.reindex([d for d in wday_order if d in pivot.index])
        fig, ax = plt.subplots(figsize=(14, 4))
        im = ax.imshow(pivot.fillna(0).values, aspect="auto", cmap="RdYlGn")
        fig.colorbar(im, ax=ax, label="Total PnL ($)")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title(f"{symbol} {tf} - PnL by Hour x Weekday", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(p:=f"{OUT_DIR}/heatmap_time_{symbol}_{tf}.png", dpi=130); plt.close(fig)
        time_hm_png = p
    except Exception:
        time_hm_png = ""

    # 8. Rolling Sharpe
    try:
        window_r = max(10, len(tdf) // 6)
        roll_s   = tdf["ret"].rolling(window_r).apply(
            lambda r: (r.mean()/r.std()*np.sqrt(len(r))) if r.std()>0 else 0)
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(roll_s.values, color="#a78bfa", lw=1.6)
        ax.axhline(0, color="#475569", lw=0.8, ls="--")
        ax.fill_between(range(len(roll_s)), roll_s, 0,
                        where=roll_s >= 0, alpha=0.15, color="#4ade80")
        ax.fill_between(range(len(roll_s)), roll_s, 0,
                        where=roll_s < 0,  alpha=0.15, color="#f87171")
        ax.set_xlabel("Trade #"); ax.set_ylabel("Rolling Sharpe")
        ax.set_title(f"{symbol} {tf} - Rolling Sharpe (window={window_r})", fontsize=11, fontweight="bold")
        ax.grid(True); fig.tight_layout()
        fig.savefig(p:=f"{OUT_DIR}/rolling_sharpe_{symbol}_{tf}.png", dpi=130); plt.close(fig)
        rs_png = p
    except Exception:
        rs_png = ""

    return eq_png, hm_png, mae_png, dist_png, sess_png, mc_png, time_hm_png, rs_png


def make_portfolio_visuals(full_df, equity_curves):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()

    best = full_df.loc[full_df.groupby(["symbol","timeframe"])["profit_factor"].idxmax()]

    # Risk-Return Frontier
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(best["max_drawdown_pct"].abs(), best["profit_pct"],
                    c=best["sqn"], cmap="viridis", s=130, edgecolors="#0a0f1e", linewidths=0.6, alpha=0.92)
    fig.colorbar(sc, ax=ax, label="SQN Score")
    for _, r in best.iterrows():
        ax.annotate(f"{r.symbol} {r.timeframe}",
                    (abs(r.max_drawdown_pct), r.profit_pct),
                    xytext=(5,4), textcoords="offset points", fontsize=7.5, color="#94a3b8")
    ax.set_xlabel("Max Drawdown % (Risk)"); ax.set_ylabel("Total Return % (Reward)")
    ax.set_title("Universe Risk-Return Frontier (colour = SQN)", fontsize=12, fontweight="bold")
    ax.grid(True); fig.tight_layout()
    fig.savefig(rr:=f"{OUT_DIR}/portfolio_rr.png", dpi=130); plt.close(fig)

    # Profit Factor Matrix
    mat = best.pivot_table("profit_factor","symbol","timeframe",aggfunc="max")
    fig, ax = plt.subplots(figsize=(9,6))
    im = ax.imshow(mat.values, aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax, label="Profit Factor")
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(len(mat.index)));   ax.set_yticklabels(mat.index)
    ax.set_title("Universe Profit Factor Matrix", fontsize=12, fontweight="bold")
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            v = mat.iloc[i,j]
            if not np.isnan(v):
                ax.text(j,i,f"{v:.1f}",ha="center",va="center",
                        color="white" if v>2 else "#fbbf24", fontsize=8)
    fig.tight_layout(); fig.savefig(mxp:=f"{OUT_DIR}/universe_matrix.png", dpi=130); plt.close(fig)

    # Summary Bar
    bs = best.sort_values("profit_pct", ascending=False)
    labels = bs["symbol"] + " " + bs["timeframe"]
    fig, ax = plt.subplots(figsize=(12, max(4, len(labels)*0.3)))
    ax.barh(labels, bs["profit_pct"],
            color=["#4ade80" if v>=0 else "#f87171" for v in bs["profit_pct"]])
    ax.set_xlabel("Strategy Return %")
    ax.set_title("Best EMA Strategy Return - All Assets", fontsize=12, fontweight="bold")
    ax.invert_yaxis(); ax.axvline(0, color="#475569", lw=0.8)
    fig.tight_layout(); fig.savefig(sp:=f"{OUT_DIR}/summary_bar.png", dpi=130); plt.close(fig)

    # Cross-Asset Equity Correlation
    corr_series = {}
    for (sym, tf), eq in equity_curves.items():
        corr_series[f"{sym}_{tf}"] = pd.Series(eq)
    if len(corr_series) > 1:
        corr_df = pd.DataFrame(corr_series).pct_change().dropna()
        corr_mat = corr_df.corr()
        fig, ax = plt.subplots(figsize=(max(7, len(corr_mat)*0.6+2),
                                        max(6, len(corr_mat)*0.5+2)))
        im = ax.imshow(corr_mat.values, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, label="Pearson Correlation")
        ax.set_xticks(range(len(corr_mat.columns))); ax.set_xticklabels(corr_mat.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(corr_mat.index)));   ax.set_yticklabels(corr_mat.index, fontsize=7)
        ax.set_title("Cross-Asset Equity Curve Correlation", fontsize=12, fontweight="bold")
        for i in range(len(corr_mat.index)):
            for j in range(len(corr_mat.columns)):
                ax.text(j,i,f"{corr_mat.iloc[i,j]:.2f}",ha="center",va="center",fontsize=6,color="white")
        fig.tight_layout(); fig.savefig(corrp:=f"{OUT_DIR}/cross_asset_corr.png", dpi=130); plt.close(fig)
    else:
        corrp = ""

    return sp, rr, mxp, corrp


# ----------------------------- CHART.JS DASHBOARD ---------------------------
def build_dashboard(profiles_df, full_df, best_by_combo, out_path):
    bdf = pd.DataFrame([
        dict(symbol=s, timeframe=t, best_ema_x=bx, best_ema_y=by, **sm)
        for (s,t),(bx,by,_,_,sm,*_) in best_by_combo.items()
    ]).sort_values("profit_pct", ascending=False)

    def jd(x): return json.dumps([round(v, 2) if isinstance(v, float) else v for v in x])
    labs  = jd([(f"{r.symbol} {r.timeframe}") for _,r in bdf.iterrows()])
    pcts  = jd(bdf["profit_pct"].tolist())
    bh    = jd(bdf.get("bh_return_pct", pd.Series([0]*len(bdf))).tolist())
    pf    = jd(bdf["profit_factor"].tolist())
    sh    = jd(bdf["sharpe"].tolist())
    so    = jd(bdf["sortino"].tolist())
    sq    = jd(bdf["sqn"].tolist())
    dd    = jd(bdf["max_drawdown_pct"].abs().tolist())
    kf    = jd(bdf["kelly_fraction"].tolist())
    ror   = jd(bdf["risk_of_ruin_pct"].tolist())

    top_html  = bdf.head(15).to_html(index=False, classes="tbl")
    prof_html = profiles_df.to_html(index=False, classes="tbl")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EMA SAR - Elite Quant Dashboard v4</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#070d1b;color:#e2e8f0;font-family:'Inter',sans-serif}}
.nav{{background:rgba(7,13,27,.95);backdrop-filter:blur(16px);border-bottom:1px solid #1e3a5f;
      padding:12px 28px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100}}
.nav h1{{font-size:1rem;font-weight:700;color:#38bdf8;letter-spacing:-.4px}}
.pill{{background:#0f2744;color:#38bdf8;padding:3px 11px;border-radius:20px;font-size:.7rem;font-weight:600}}
.main{{padding:20px 28px}}
.kpi{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:20px}}
.kcard{{background:#0d1526;border:1px solid #1e3a5f;border-radius:10px;padding:14px 10px;text-align:center}}
.klabel{{font-size:.6rem;color:#64748b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px}}
.kval{{font-size:1.4rem;font-weight:700;color:#38bdf8}}
.ksub{{font-size:.65rem;color:#475569;margin-top:2px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}}
.card{{background:#0d1526;border:1px solid #1e3a5f;border-radius:10px;padding:16px}}
.sec{{font-size:.8rem;font-weight:600;color:#38bdf8;margin-bottom:10px;
      border-bottom:1px solid #1e3a5f;padding-bottom:6px}}
.ch{{position:relative;height:260px}}
.ch-tall{{position:relative;height:340px}}
.tbl{{width:100%;border-collapse:collapse;font-size:.68rem}}
.tbl th{{background:#050b18;color:#38bdf8;padding:6px 8px;text-align:left;
          text-transform:uppercase;letter-spacing:.4px;border-bottom:1px solid #1e3a5f}}
.tbl td{{padding:5px 8px;border-bottom:1px solid rgba(30,58,95,.4);color:#cbd5e1}}
.tbl tr:hover td{{background:rgba(56,189,248,.04)}}
img{{width:100%;border-radius:8px;border:1px solid #1e3a5f}}
.formula{{background:#050b18;border:1px solid #1e3a5f;border-radius:8px;
           padding:14px;font-family:monospace;font-size:.73rem;color:#7dd3fc;line-height:2}}
.green{{color:#4ade80;font-weight:600}} .red{{color:#f87171;font-weight:600}}
</style>
</head>
<body>
<nav class="nav">
  <h1>EMA SAR - Elite Quant Dashboard v4</h1>
  <span class="pill">Walk-Forward {WF_FOLDS}-Fold OOS</span>
  <span class="pill">Monte Carlo {MC_SIMS}x</span>
  <span class="pill">Kelly Criterion</span>
  <span class="pill">Autocorrelation</span>
  <span class="pill">Robustness +-{ROBUST_R}</span>
  <span class="pill">Hurst Exponent</span>
  <span class="pill">{len(profiles_df)} Datasets</span>
</nav>
<div class="main">

<!-- KPIs -->
<div class="kpi">
  <div class="kcard"><div class="klabel">Best Return</div>
    <div class="kval green">{bdf['profit_pct'].max():.1f}%</div>
    <div class="ksub">{bdf.iloc[0]['symbol']} {bdf.iloc[0]['timeframe']}</div></div>
  <div class="kcard"><div class="klabel">Best Profit Factor</div>
    <div class="kval">{bdf['profit_factor'].max():.2f}x</div>
    <div class="ksub">Gross P / Gross L</div></div>
  <div class="kcard"><div class="klabel">Best Sharpe</div>
    <div class="kval">{bdf['sharpe'].max():.2f}</div>
    <div class="ksub">Risk-Adjusted</div></div>
  <div class="kcard"><div class="klabel">Best SQN</div>
    <div class="kval">{bdf['sqn'].max():.2f}</div>
    <div class="ksub">Van Tharp</div></div>
  <div class="kcard"><div class="klabel">Datasets</div>
    <div class="kval">{len(profiles_df)}</div>
    <div class="ksub">Symbol x Timeframe</div></div>
  <div class="kcard"><div class="klabel">Pairs Tested</div>
    <div class="kval">{len(full_df):,}</div>
    <div class="ksub">EMA Combinations</div></div>
</div>

<!-- Row 1 Charts -->
<div class="grid2">
  <div class="card"><div class="sec">Strategy Return vs Buy & Hold</div>
    <div class="ch-tall"><canvas id="retChart"></canvas></div></div>
  <div class="card"><div class="sec">Risk-Adjusted Metrics (Sharpe / Sortino / SQN)</div>
    <div class="ch-tall"><canvas id="riskChart"></canvas></div></div>
</div>

<!-- Row 2 Charts -->
<div class="grid3">
  <div class="card"><div class="sec">Profit Factor by Strategy</div>
    <div class="ch"><canvas id="pfChart"></canvas></div></div>
  <div class="card"><div class="sec">Max Drawdown Comparison</div>
    <div class="ch"><canvas id="ddChart"></canvas></div></div>
  <div class="card"><div class="sec">Kelly Fraction & Risk of Ruin %</div>
    <div class="ch"><canvas id="kellyChart"></canvas></div></div>
</div>

<!-- Portfolio Visuals -->
<div class="grid2" style="margin-bottom:16px">
  <div class="card"><div class="sec">Risk-Return Frontier (SQN colored)</div>
    <img src="portfolio_rr.png"></div>
  <div class="card"><div class="sec">Universe Profit Factor Matrix</div>
    <img src="universe_matrix.png"></div>
</div>
<div class="grid2" style="margin-bottom:16px">
  <div class="card"><div class="sec">Cross-Asset Equity Correlation</div>
    <img src="cross_asset_corr.png"></div>
  <div class="card"><div class="sec">Strategy Returns Summary</div>
    <img src="summary_bar.png"></div>
</div>

<!-- Top Strategies Table -->
<div class="card" style="margin-bottom:16px">
  <div class="sec">Top 15 Strategies - Full Analytics</div>
  <div style="overflow-x:auto">{top_html}</div>
</div>

<!-- Sample Visuals Grid -->
<div class="grid2" style="margin-bottom:16px">
  <div class="card"><div class="sec">Equity & Drawdown - ETH 4h</div>
    <img src="equity_ETH_4h.png"></div>
  <div class="card"><div class="sec">Monte Carlo Fan - ETH 4h</div>
    <img src="mc_ETH_4h.png"></div>
  <div class="card"><div class="sec">Hour x Weekday PnL Heatmap - ETH 4h</div>
    <img src="heatmap_time_ETH_4h.png"></div>
  <div class="card"><div class="sec">Rolling Sharpe - ETH 4h</div>
    <img src="rolling_sharpe_ETH_4h.png"></div>
</div>

<!-- Formulas -->
<div class="card" style="margin-bottom:16px">
  <div class="sec">Mathematical Engine - All Formulas</div>
  <div class="formula">
Trade Return     = (P_exit/P_entry - 1) x Direction - 2xFee          [realistic, 1-bar shift]
CAGR             = (1 + Return%)^(365/Days) - 1
Sharpe           = Mean(R) / Std(R) x sqrt(N)
Sortino          = Mean(R) / Downside_Std x sqrt(N)
Calmar           = Return% / |MaxDD%|
Omega            = Sum(Positive Returns) / Sum(|Negative Returns|)
Ulcer Index      = sqrt(Mean(Drawdown%^2))
SQN              = Expectancy% / Std% x sqrt(N)                       [Van Tharp]
Kelly f*         = (p x b - q) / b   where b=payoff, q=1-p              [fractional bet]
Risk of Ruin     = ((1-edge)/(1+edge))^N   edge = p - q/b
Hurst Exponent   = R/S method  (>0.5 trending | <0.5 mean-reverting)
Ljung-Box Q      = n(n+2) Sum(rho_k^2 / (n-k))   p<0.05 -> autocorrelation
Walk-Forward     = {WF_FOLDS}-fold rolling IS->OOS, report avg/min/max OOS return
Monte Carlo      = {MC_SIMS}x bootstrap resample -> P5/P50/P95 equity bands
Robustness       = % of +-{ROBUST_R} EMA neighbours with profit_pct > 0
Regime           = Rolling {REGIME_WIN}-bar return  >3% Bull | <-3% Bear | else Sideways
  </div>
</div>

<!-- Dataset Profiles -->
<div class="card">
  <div class="sec">Dataset Metadata (Pre-Test Profile)</div>
  <div style="overflow-x:auto">{prof_html}</div>
</div>
</div>

<script>
const C=(id)=>document.getElementById(id).getContext('2d');
const labs={labs};
const opts={{responsive:true,maintainAspectRatio:false,
  plugins:{{legend:{{labels:{{color:'#94a3b8',font:{{size:10}}}}}}}},
  scales:{{x:{{ticks:{{color:'#64748b',font:{{size:9}},maxRotation:50}},grid:{{color:'rgba(30,58,95,.4)'}}}},
           y:{{ticks:{{color:'#64748b',font:{{size:9}}}},grid:{{color:'rgba(30,58,95,.4)'}}}}}}
}};
const pcts={pcts}; const bh={bh}; const pf={pf}; const sh={sh};
const so={so}; const sq={sq}; const dd={dd}; const kf={kf}; const ror={ror};

new Chart(C('retChart'),{{type:'bar',data:{{labels:labs,datasets:[
  {{label:'Strategy %',data:pcts,backgroundColor:pcts.map(v=>v>=0?'rgba(74,222,128,.75)':'rgba(248,113,113,.75)'),borderRadius:3}},
  {{label:'Buy&Hold %',data:bh,backgroundColor:'rgba(56,189,248,.35)',borderRadius:3}}
]}},options:opts}});

new Chart(C('riskChart'),{{type:'bar',data:{{labels:labs,datasets:[
  {{label:'Sharpe', data:sh,  backgroundColor:'rgba(56,189,248,.75)',borderRadius:3}},
  {{label:'Sortino',data:so,  backgroundColor:'rgba(167,139,250,.75)',borderRadius:3}},
  {{label:'SQN',   data:sq,  backgroundColor:'rgba(251,191,36,.75)', borderRadius:3}}
]}},options:opts}});

new Chart(C('pfChart'),{{type:'bar',data:{{labels:labs,datasets:[
  {{label:'Profit Factor',data:pf,
    backgroundColor:pf.map(v=>v>=2?'rgba(74,222,128,.8)':v>=1?'rgba(251,191,36,.7)':'rgba(248,113,113,.7)'),borderRadius:3}}
]}},options:opts}});

new Chart(C('ddChart'),{{type:'bar',data:{{labels:labs,datasets:[
  {{label:'Max DD %',data:dd,backgroundColor:'rgba(248,113,113,.7)',borderRadius:3}}
]}},options:opts}});

new Chart(C('kellyChart'),{{type:'bar',data:{{labels:labs,datasets:[
  {{label:'Kelly f*',   data:kf, backgroundColor:'rgba(56,189,248,.75)',borderRadius:3}},
  {{label:'Risk of Ruin %',data:ror,backgroundColor:'rgba(248,113,113,.6)',borderRadius:3}}
]}},options:opts}});
</script>
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[dashboard] -> {out_path}")


# ----------------------------- REPORT WRITER --------------------------------
def write_reports(profiles_df, full_df, best_by_combo, doc_df, portfolio_pngs):
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XI
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils.dataframe import dataframe_to_rows

    os.makedirs(CSV_DIR, exist_ok=True)

    summary_rows = []
    for (s,t),(bx,by,_,_,sm,*_) in best_by_combo.items():
        summary_rows.append(dict(symbol=s,timeframe=t,ema_x=bx,ema_y=by,**sm))
    sdf = pd.DataFrame(summary_rows).sort_values("profit_pct", ascending=False)

    # WithWithout comparison dataframe
    ww_rows = []
    for (s,t),(bx,by,_,_,sm,*_) in best_by_combo.items():
        ww_rows.append(dict(
            symbol=s, timeframe=t, ema_x=bx, ema_y=by,
            shifted_profit=sm.get("profit_pct", 0),
            no_shift_profit=sm.get("profit_pct_no_shift", 0),
            lookahead_delta=sm.get("lookahead_delta", 0),
            kelly_profit=sm.get("kelly_profit_pct", 0),
            long_profit=sm.get("long_profit_pct", 0),
            short_profit=sm.get("short_profit_pct", 0)
        ))
    ww_df = pd.DataFrame(ww_rows)

    # Long Short split dataframe
    ls_rows = []
    for (s,t),(bx,by,_,_,sm,*_) in best_by_combo.items():
        ls_rows.append(dict(
            symbol=s, timeframe=t, ema_x=bx, ema_y=by,
            long_trades=sm.get("long_trades", 0),
            long_win_rate=sm.get("long_win_rate", 0),
            long_profit_pct=sm.get("long_profit_pct", 0),
            short_trades=sm.get("short_trades", 0),
            short_win_rate=sm.get("short_win_rate", 0),
            short_profit_pct=sm.get("short_profit_pct", 0)
        ))
    ls_df = pd.DataFrame(ls_rows)

    # Omega Ulcer dataframe
    ou_rows = []
    for (s,t),(bx,by,_,_,sm,*_) in best_by_combo.items():
        ou_rows.append(dict(
            symbol=s, timeframe=t, ema_x=bx, ema_y=by,
            omega_ratio=sm.get("omega_ratio", 0),
            ulcer_index=sm.get("ulcer_index", 0),
            recovery_factor=sm.get("recovery_factor", 0),
            tail_ratio=sm.get("tail_ratio", 0),
            sharpe=sm.get("sharpe", 0),
            sortino=sm.get("sortino", 0),
            sqn=sm.get("sqn", 0)
        ))
    ou_df = pd.DataFrame(ou_rows)

    for name, df in [("Formulas",doc_df),("Profiles",profiles_df),("Summary",sdf),
                     ("WithWithout",ww_df),("LongShort",ls_df),("OmegaUlcer",ou_df),
                     ("Full_Grid",full_df),("Top50",full_df.sort_values("profit_factor",ascending=False).head(50))]:
        df.to_csv(f"{CSV_DIR}/{name}.csv", index=False)
    for (s,t),(bx,by,tdf,*_) in best_by_combo.items():
        tdf.to_csv(f"{CSV_DIR}/Trades_{s}_{t}.csv", index=False)
    print(f"[csv] -> {CSV_DIR}/")

    wb  = Workbook(); wb.remove(wb.active)
    hf  = Font(bold=True, color="FFFFFF")
    hfill = PatternFill("solid", fgColor="07091c")

    def ws(name, df):
        sheet = wb.create_sheet(name[:31])
        for row in dataframe_to_rows(df, index=False, header=True):
            sheet.append(row)
        for c in sheet[1]: c.font = hf; c.fill = hfill
        for col in sheet.columns:
            sheet.column_dimensions[col[0].column_letter].width = min(30, max(10, max(len(str(c.value or "")) for c in col)+2))
        return sheet

    ws("Formulas",   doc_df)
    ws("Profiles",   profiles_df)
    ws("Summary",    sdf)
    ws("WithWithout",ww_df)
    ws("LongShort",  ls_df)
    ws("OmegaUlcer", ou_df)
    ws("Full_Grid",  full_df)
    ws("Top50",      full_df.sort_values("profit_factor",ascending=False).head(50))

    for (s,t),(bx,by,tdf,eq_curve,sm,reg_prf,monthly,*pngs) in best_by_combo.items():
        cols = ["entry_time","exit_time","entry_price","exit_price","direction","ret","pnl",
                "hold_bars","mae_pct","mfe_pct","trade_efficiency_pct","edge_ratio","entry_regime",
                "session","weekday","hour","year","month","equity","cum_profit","cum_loss"]
        ws(f"Trades_{s}_{t}", tdf[[c for c in cols if c in tdf.columns]])
        yearly = tdf.groupby("year").agg(
            trades=("pnl","count"),profit_pct=("ret",lambda r:round((np.prod(1+r)-1)*100,2)),
            cum_pnl=("pnl","sum"),avg_mae=("mae_pct","mean"),avg_mfe=("mfe_pct","mean"),
        ).reset_index()
        ws(f"Yearly_{s}_{t}", yearly)
        sess = tdf.groupby("session").agg(
            trades=("pnl","count"),profit_pct=("ret",lambda r:round((np.prod(1+r)-1)*100,2)),
            win_rate_pct=("pnl",lambda p:round((p>0).mean()*100,2)),cum_pnl=("pnl","sum"),
        ).reset_index()
        ws(f"Session_{s}_{t}", sess)
        ws(f"Regime_{s}_{t}", pd.DataFrame([dict(regime=k,**v) for k,v in reg_prf.items()]))
        ws(f"Monthly_{s}_{t}", monthly)

    # Charts sheet
    cs = wb.create_sheet("Charts"); row = 1
    for p in portfolio_pngs:
        if p and os.path.exists(p):
            cs.add_image(XI(p), f"A{row}"); row += 22
    for (s,t), payload in best_by_combo.items():
        *_, eq_p, hm_p, mae_p, dist_p, sess_p, mc_p, thm_p, rs_p = payload
        cs.cell(row=row,column=1,value=f"{s} {t}").font = Font(bold=True,size=11); row+=1
        for col, png in [("A",eq_p),("L",hm_p),("W",mc_p)]:
            if png and os.path.exists(png):
                cs.add_image(XI(png), f"{col}{row}")
        row += 22

    xlsx = f"{OUT_DIR}/optimization_report.xlsx"
    wb.save(xlsx); print(f"[excel] -> {xlsx}")

    zpath = f"{OUT_DIR}/quant_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for root,_,files in os.walk(OUT_DIR):
            for f in files:
                if f.endswith(".zip"): continue
                fp = os.path.join(root,f)
                zf.write(fp, os.path.relpath(fp, OUT_DIR))
    print(f"[zip] -> {zpath}  ({os.path.getsize(zpath)//1024//1024} MB)")


# ----------------------------- WORKER ---------------------------------------
def worker(symbol, tf, df, mode):
    prof = profile_dataset(df, symbol, tf)
    res, best = (run_grid(df, symbol, tf,
                          list(range(MIN_P, MAX_P+1)), list(range(MIN_P, MAX_P+1)))
                 if mode == "full" else coarse_fine(df, symbol, tf))
    grid_df = res.to_df()
    if best is None:
        return symbol, tf, prof, grid_df, None
    sm, tdf, eq_curve, bx, by, reg_prf, monthly = best[0], best[1], best[2], best[3], best[4], best[5], best[6]

    # Robustness score
    close_arr = df["close"].to_numpy(dtype=np.float64)
    ts_arr    = pd.to_datetime(df["time"]).to_numpy()
    rob       = robustness_score(close_arr, ts_arr, bx, by)
    sm["robustness_score"] = rob

    # CAGR
    prof_days = prof["days"]
    sm["cagr_pct"] = round(cagr(sm["profit_pct"], prof_days), 2)

    # Walk-Forward OOS
    wf = walk_forward(close_arr, ts_arr, bx, by)
    if wf: sm.update(wf)

    # Regime-specific best EMAs
    reg_best = regime_best_emas(df, symbol, tf)
    sm["regime_best_emas"] = str(reg_best)

    pngs = make_visuals(symbol, tf, grid_df, bx, by, eq_curve, tdf)
    return symbol, tf, prof, grid_df, (bx, by, tdf, eq_curve, sm, reg_prf, monthly, *pngs)


# ----------------------------- MAIN -----------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode",  choices=["full","coarse"], default="coarse")
    pa.add_argument("--demo",  action="store_true")
    pa.add_argument("--input", default=DATA_XLSX)
    args = pa.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.demo or not os.path.exists(args.input):
        make_demo(args.input)

    data = load_all(args.input)
    if not data:
        print("No usable sheets found."); return

    doc_df = pd.DataFrame([
        ("CAGR",                "(1+R%)^(365/Days)-1",             "Annualised compound growth"),
        ("Trade Return",        "(P1/P0-1)xDir - 2xFee",           "Net per-trade return (1-bar shift)"),
        ("Profit Factor",       "Sum(Wins)/Sum(|Losses|)",         "Gross profit / gross loss"),
        ("Sharpe",              "mu/sigma x sqrt(N)",              "Total-vol risk-adjusted return"),
        ("Sortino",             "mu/sigma_down x sqrt(N)",         "Downside-vol risk-adjusted return"),
        ("Calmar",              "Return%/|MaxDD%|",                "Return vs worst drawdown"),
        ("Omega Ratio",         "Sum(Pos)/Sum(|Neg|)",             "Probability-weighted ratio"),
        ("Ulcer Index",         "sqrt(Mean(DD%^2))",               "Sustained drawdown pain metric"),
        ("Recovery Factor",     "Return%/|MaxDD%|",                "Capital recovery performance"),
        ("Tail Ratio",          "P95_ret/P5_ret",                  "Extreme win vs loss ratio"),
        ("SQN",                 "(Expectancy/Std) x sqrt(N)",      "Van Tharp System Quality Number"),
        ("Kelly f*",            "(p x b - q)/b",                   "Optimal fractional bet size"),
        ("Risk of Ruin",        "((1-e)/(1+e))^N",                 "Analytic probability of 50% ruin"),
        ("Hurst",               "R/S method",                      ">0.5 trending, <0.5 mean-reverting"),
        ("Ljung-Box Q",         "n(n+2)Sum(rho_k^2/(n-k))",        "Autocorrelation test; p<0.05 serial"),
        ("Robustness",          "% +-5 EMA neighbours profitable", "Overfit guard / parameter stability"),
        ("Walk-Forward OOS",    f"{WF_FOLDS}-fold IS->OOS",        "True out-of-sample validation"),
        ("Monte Carlo",         f"{MC_SIMS}x bootstrap P5/P50/P95","Equity distribution under resampling"),
        ("Regime",              f"Rolling {REGIME_WIN}-bar return", "Bull/Bear/Sideways market label"),
        ("Lookahead Delta",     "Unshifted - Shifted return",      "Phantom profit from same-bar exec"),
    ], columns=["Metric","Formula","Description"])

    all_frames, all_profiles, best_by_combo, eq_curves = [], [], {}, {}

    with ProcessPoolExecutor() as ex:
        futs = {ex.submit(worker, s, t, df, args.mode): (s,t) for (s,t),df in data.items()}
        for fut in as_completed(futs):
            s, t = futs[fut]
            try:
                sym, tf, prof, grid_df, best = fut.result()
            except Exception as e:
                print(f"[FAIL] {s} {t}: {e}"); continue
            all_profiles.append(prof); all_frames.append(grid_df)
            if best is not None:
                bx, by, tdf, eq, sm = best[0], best[1], best[2], best[3], best[4]
                best_by_combo[(sym,tf)] = best
                eq_curves[(sym,tf)]     = eq
                sig = "YES" if sm.get("is_significant") else "NO"
                ac  = "AC" if sm.get("has_autocorrelation") else "  "
                print(
                    f"[{sym:5} {tf:3}] EMA({bx},{by}) | "
                    f"PF={sm['profit_factor']:5.2f} | "
                    f"CAGR={sm.get('cagr_pct',0):+6.1f}% | "
                    f"Sharpe={sm['sharpe']:5.2f} | "
                    f"Kelly={sm['kelly_fraction']:.3f} | "
                    f"Rob={sm['robustness_score']:5.1f}% | "
                    f"WF={sm.get('wf_oos_avg','N/A')}% | "
                    f"Sig={sig} {ac}"
                )
            else:
                print(f"[{s:5} {t:3}] no profitable pair")

    if not all_frames: print("No results."); return

    prof_df = pd.DataFrame(all_profiles).sort_values(["symbol","timeframe"])
    full_df = pd.concat(all_frames, ignore_index=True)

    _, rr, mx, corr = make_portfolio_visuals(full_df, eq_curves)
    build_dashboard(prof_df, full_df, best_by_combo, f"{OUT_DIR}/dashboard.html")
    write_reports(prof_df, full_df, best_by_combo, doc_df, [rr, mx, corr])

    print(f"\nCompleted -> {OUT_DIR}/dashboard.html | {OUT_DIR}/quant_bundle.zip")


if __name__ == "__main__":
    main()
