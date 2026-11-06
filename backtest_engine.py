"""Core backtest engine for IDX Trend-Pullback strategy."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Config:
    fractal_n: int = 5
    min_conf: int = 3
    adx_min: float = 20.0
    ema_len: int = 200
    rsi_lo: float = 40.0
    rsi_hi: float = 65.0
    use_dist: bool = True
    dist_min: float = 1.0
    dist_max: float = 3.0
    sl_buf: float = 0.1
    rr: float = 2.0
    trail: bool = False
    max_bars: int = 40
    struct_expiry: int = 20
    risk_pct: float = 0.01
    daily_loss: float = 0.03
    loss_streak: int = 3
    pause: int = 10
    fee: float = 0.002
    slip: float = 0.001
    max_lev: float = 1.0
    initial: float = 100_000_000.0
    lot_size: int = 100

VARIANTS = {
    "A_inti":     Config(min_conf=0),
    "B_2dari6":   Config(min_conf=2),
    "C_3dari6":   Config(min_conf=3),
    "D_4dari6":   Config(min_conf=4),
}

def rma(x, n):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) == 0:
        return out
    f = valid[0]
    if len(x) - f < n:
        return out
    out[f + n - 1] = x[f:f + n].mean()
    for i in range(f + n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out

def true_range(h, l, c):
    tr = np.empty(len(c))
    tr[0] = h[0] - l[0]
    pc = c[:-1]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - pc), np.abs(l[1:] - pc)])
    return tr

def adx_calc(h, l, c, n=14):
    up = np.zeros(len(h))
    dn = np.zeros(len(h))
    up[1:] = h[1:] - h[:-1]
    dn[1:] = l[:-1] - l[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = rma(true_range(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * rma(pdm, n) / atr
        mdi = 100 * rma(mdm, n) / atr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return rma(dx, n)

def rsi_calc(c, n=14):
    d = np.diff(c, prepend=c[0])
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    g, ls = rma(gain, n), rma(loss, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = g / ls
        out = 100 - 100 / (1 + rs)
    out = np.where((ls == 0) & ~np.isnan(g), 100.0, out)
    return out

def supertrend(h, l, c, length, factor):
    n = len(c)
    atr = rma(true_range(h, l, c), length)
    hl2 = (h + l) / 2
    ub = hl2 + factor * atr
    lb = hl2 - factor * atr
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    d = np.zeros(n, dtype=int)
    st = np.full(n, np.nan)
    valid = np.where(~np.isnan(atr))[0]
    if len(valid) == 0:
        return st, d
    s = valid[0]
    for i in range(s, n):
        if i == s:
            fu[i], fl[i] = ub[i], lb[i]
            d[i] = 1 if c[i] >= hl2[i] else -1
        else:
            fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
            if d[i - 1] == 1:
                d[i] = -1 if c[i] < fl[i] else 1
            else:
                d[i] = 1 if c[i] > fu[i] else -1
        st[i] = fl[i] if d[i] == 1 else fu[i]
    return st, d

def backtest(df, cfg, name=""):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    n = len(df)
    idx = df.index
    dates = np.array(idx.date)
    has_vol = np.nansum(v) > 0

    atr = rma(true_range(h, l, c), 14)
    adx = adx_calc(h, l, c, 14)
    A, dA = supertrend(h, l, c, 5, 1.0)
    B, dB = supertrend(h, l, c, 20, 1.8)
    C, dC = supertrend(h, l, c, 30, 2.5)
    cs = pd.Series(c)
    ema = cs.ewm(span=cfg.ema_len, adjust=False).mean().to_numpy()
    rsi = rsi_calc(c, 14)
    macd = cs.ewm(span=12, adjust=False).mean() - cs.ewm(span=26, adjust=False).mean()
    hist = (macd - macd.ewm(span=9, adjust=False).mean()).to_numpy()
    volma = pd.Series(v).rolling(20).mean().to_numpy()

    k = (cfg.fractal_n - 1) // 2
    fh = [None, None]
    fl = [None, None]
    last_structure_bar = -10 ** 9

    equity = cfg.initial
    pos = None
    pending = None
    trades = []
    streak = 0
    pause_until = -1
    day_pnl = {}
    warm = max(cfg.ema_len + 10, 60)

    for i in range(n):
        j = i - k
        if j >= k:
            wh = h[j - k:i + 1]
            wl = l[j - k:i + 1]
            new_high = new_low = False
            if (wh < h[j]).sum() == len(wh) - 1:
                fh = [fh[1], h[j]]
                new_high = True
            if (wl > l[j]).sum() == len(wl) - 1:
                fl = [fl[1], l[j]]
                new_low = True
            if None not in fh and None not in fl and fh[1] > fh[0] and fl[1] > fl[0]:
                if new_high or new_low:
                    last_structure_bar = i

        if pending is not None and pos is None:
            sl0, atr0, conf = pending
            pending = None
            entry = o[i] * (1 + cfg.slip)
            risk = entry - sl0
            ok = risk > 0
            if ok and cfg.use_dist:
                ok = cfg.dist_min * atr0 <= risk <= cfg.dist_max * atr0
            if ok and day_pnl.get(dates[i], 0.0) <= -cfg.daily_loss:
                ok = False
            if ok:
                raw_qty = min(equity * cfg.risk_pct / risk, equity * cfg.max_lev / entry)
                qty = max(cfg.lot_size, int(raw_qty // cfg.lot_size) * cfg.lot_size) if raw_qty >= cfg.lot_size else 0
                if qty > 0:
                    tp = None if cfg.trail else entry + cfg.rr * risk
                    pos = dict(entry=entry, sl=sl0, sl0=sl0, tp=tp, qty=qty, risk=risk,
                               i0=i, eq0=equity, be=False, conf=conf)

        if pos is not None:
            exit_p, reason = None, None
            if o[i] <= pos["sl"]:
                exit_p, reason = o[i], "SL"
            elif l[i] <= pos["sl"]:
                exit_p, reason = pos["sl"], "SL"
            elif pos["tp"] is not None and h[i] >= pos["tp"]:
                exit_p, reason = pos["tp"], "TP"
            elif cfg.max_bars and i - pos["i0"] >= cfg.max_bars:
                exit_p, reason = c[i], "TIME"
            elif i == n - 1:
                exit_p, reason = c[i], "EOD"

            if exit_p is None and cfg.trail:
                if h[i] - pos["entry"] >= pos["risk"]:
                    pos["be"] = True
                if pos["be"]:
                    new = pos["entry"]
                    if dA[i] == 1 and not np.isnan(A[i]) and A[i] > new:
                        new = A[i]
                    if new > pos["sl"]:
                        pos["sl"] = new
            elif exit_p is not None:
                exit_p = exit_p * (1 - cfg.slip)
                pnl = pos["qty"] * (exit_p - pos["entry"]) - cfg.fee * pos["qty"] * (pos["entry"] + exit_p)
                pnl_pct = pnl / pos["eq0"]
                equity += pnl
                trades.append(dict(
                    ticker=name, entry_time=idx[pos["i0"]], exit_time=idx[i],
                    entry=pos["entry"], exit=exit_p, sl=pos["sl0"], reason=reason,
                    conf=pos["conf"], r=pnl / (pos["qty"] * pos["risk"]),
                    pnl_pct=pnl_pct, bars=i - pos["i0"]))
                day_pnl[dates[i]] = day_pnl.get(dates[i], 0.0) + pnl_pct
                streak = streak + 1 if pnl < 0 else 0
                if streak >= cfg.loss_streak:
                    pause_until = i + cfg.pause
                    streak = 0
                pos = None

        if (pos is None and pending is None and warm <= i < n - 1 and i >= pause_until
                and not np.isnan([A[i], B[i], C[i], atr[i]]).any()):
            have = None not in fh and None not in fl
            if have and (i - last_structure_bar) <= cfg.struct_expiry and fh[1] > fh[0] and fl[1] > fl[0]:
                if dB[i] == 1 and l[i] <= B[i] and c[i] > B[i] and c[i] > C[i]:
                    checks = [
                        c[i] > ema[i],
                        bool(rsi[i] > rsi[i - 1] and cfg.rsi_lo <= rsi[i] <= cfg.rsi_hi),
                        bool(has_vol and v[i] >= volma[i]),
                        bool(hist[i] > hist[i - 1]),
                        bool(adx[i] > cfg.adx_min),
                        dA[i] == 1,
                    ]
                    conf = int(sum(bool(x) for x in checks))
                    if conf >= cfg.min_conf:
                        pending = (C[i] - cfg.sl_buf * atr[i], atr[i], conf)

    return pd.DataFrame(trades)

def pf_r(r):
    r = np.asarray(r, float)
    gp, gl = r[r > 0].sum(), -r[r <= 0].sum()
    return gp / gl if gl > 0 else np.inf

def boot_ci(r, n_boot=3000, seed=1):
    r = np.asarray(r, float)
    if len(r) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
