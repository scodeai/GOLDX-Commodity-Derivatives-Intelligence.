
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from scipy import stats as sps
    HAS_SCIPY = True
except Exception:  # SciPy optional at runtime, listed in requirements
    sps, HAS_SCIPY = None, False
try:
    from sklearn.ensemble import IsolationForest
    HAS_SK = True
except Exception:
    HAS_SK = False

st.set_page_config(page_title="GOLDX | Commodity Derivatives Intelligence", page_icon="🥇",
                   layout="wide", initial_sidebar_state="expanded")

GOLD, GREEN, RED, BLUE, PURPLE, GREY = "#D4A017", "#2ECC71", "#E74C3C", "#4DA3FF", "#B388FF", "#8B98A9"
REF_PURITY = 999
STORE_DIR = Path("goldx_store")

# Contract economics from the PPT: lot size (g), quotation basis (price is quoted per N grams), purity
DEFAULT_SPECS = pd.DataFrame({
    "symbol": ["GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"],
    "lot_g": [100, 10, 8, 1],
    "quote_g": [10, 10, 8, 1],
    "purity": [995, 999, 999, 999],
})
SYMBOLS = DEFAULT_SPECS["symbol"].tolist()
PAIRS = [("GOLDM", "GOLDTEN"), ("GOLDM", "GOLDGUINEA"), ("GOLDM", "GOLDPETAL"),
         ("GOLDTEN", "GOLDGUINEA"), ("GOLDTEN", "GOLDPETAL"), ("GOLDGUINEA", "GOLDPETAL")]

CSS = """
<style>
.stApp{background:#0B1220;color:#E6EDF3}
[data-testid=stSidebar]{background:#0F1A2E;border-right:1px solid #1E2B45}
h1,h2,h3,h4{color:#F3F6FA}
.gx-hero{background:linear-gradient(120deg,#16264A 0%,#0B1220 65%);border:1px solid #1E2B45;border-radius:16px;
 padding:18px 26px;margin-bottom:14px}
.gx-hero h1{margin:0;font-size:2.1rem;letter-spacing:.5px}
.gx-hero .g{color:#D4A017}
.gx-sub{color:#8B98A9;font-size:.95rem;margin-top:4px}
.gx-card{background:#111C33;border:1px solid #1E2B45;border-radius:14px;padding:14px 16px;height:100%}
.gx-card .l{color:#8B98A9;font-size:.74rem;text-transform:uppercase;letter-spacing:.8px}
.gx-card .v{font-size:1.55rem;font-weight:700;color:#F3F6FA;line-height:1.35}
.gx-card .s{font-size:.78rem;color:#8B98A9}
.gx-pill{display:inline-block;padding:2px 11px;border-radius:999px;font-size:.74rem;font-weight:700}
.p-cheap{background:#12372A;color:#2ECC71}.p-exp{background:#3A1A1A;color:#E74C3C}
.p-neu{background:#22304A;color:#9FB0C8}.p-no{background:#2A2A2A;color:#aaa}
.gx-alert{background:#111C33;border:1px solid #1E2B45;border-left:4px solid #D4A017;border-radius:10px;
 padding:12px 16px;margin-bottom:10px;font-size:.92rem}
.gx-flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0 10px 0}
.gx-step{background:#111C33;border:1px solid #1E2B45;border-radius:10px;padding:8px 12px;font-size:.82rem}
.gx-step b{color:#D4A017}
.gx-arrow{color:#D4A017;font-weight:700}
.gx-verdict{border-radius:12px;padding:14px 18px;font-size:1rem;margin:8px 0}
.v-good{background:#10301F;border:1px solid #2ECC71}.v-mid{background:#33290F;border:1px solid #D4A017}
.v-bad{background:#341616;border:1px solid #E74C3C}
.gx-quote{color:#D4A017;font-style:italic}
.stTabs [data-baseweb=tab]{font-weight:600}
</style>
"""

# =====================================================================================
# 1. DATA COLLECTION  (demo generator + Bhavcopy ingestion)
# =====================================================================================
@st.cache_data(show_spinner=False)
def make_demo(seed: int = 11) -> pd.DataFrame:
    """Synthetic MCX-style bhavcopy (clearly NOT real prices) with injected data issues for the validator."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-11-01", "2026-10-23")
    n = len(dates)
    lr = np.log(62000) + np.cumsum(rng.normal(0, 0.0085, n))
    lr = lr + (np.log(147000) - lr[-1]) * np.arange(n) / (n - 1)  # ends near Rs 1.47 L / 10g
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.97 * ou[i - 1] + rng.normal(0, 0.0015)
    carry = 0.07 + ou

    months = pd.date_range("2023-11-01", "2027-06-01", freq="MS")
    cfg = {
        "GOLDM": dict(quote_g=10, pur=995, bv=12000, bo=30000, hz=185, exps=[m + pd.Timedelta(days=4) for m in months]),
        "GOLDTEN": dict(quote_g=10, pur=999, bv=600, bo=15500, hz=185,
                        exps=[m + pd.Timedelta(days=4) for m in months if m.month % 2 == 0]),
        "GOLDGUINEA": dict(quote_g=8, pur=999, bv=800, bo=22000, hz=95,
                           exps=[m + pd.offsets.MonthEnd(0) for m in months]),
        "GOLDPETAL": dict(quote_g=1, pur=999, bv=9000, bo=205000, hz=95,
                          exps=[m + pd.offsets.MonthEnd(0) for m in months]),
    }
    dev, rows = {}, []
    for i, d in enumerate(dates):
        spot_g = math.exp(lr[i]) / 10
        for sym, c in cfg.items():
            live = [e for e in c["exps"] if 0 < (e - d).days <= c["hz"]]
            for r, e in enumerate(live):
                dte = (e - d).days
                x = 0.9 * dev.get((sym, e), 0.0) + rng.normal(0, 0.0006)
                dev[(sym, e)] = x
                fwd = spot_g * (1 + carry[i] * dte / 365) * (1 + x + rng.normal(0, 0.0002))
                close = round(fwd * c["pur"] / 999 * c["quote_g"])
                op = close * (1 + rng.normal(0, 0.002))
                hi = max(op, close) * (1 + abs(rng.normal(0, 0.0015)))
                lo = min(op, close) * (1 - abs(rng.normal(0, 0.0015)))
                vol = c["bv"] * math.exp(-1.1 * r) * rng.lognormal(0, 0.35) * (0.35 if dte <= 4 else 1)
                oi = c["bo"] * math.exp(-0.7 * r) * rng.lognormal(0, 0.15) * min(1, dte / 12)
                rows.append((d, sym, e, round(op), round(hi), round(lo), close, int(vol), int(oi)))
    df = pd.DataFrame(rows, columns=["date", "symbol", "expiry", "open", "high", "low", "close", "volume", "oi"])
    idx = rng.choice(len(df), size=30, replace=False)
    df.loc[idx[:5], "close"] = np.nan                                   # missing settlement
    df = pd.concat([df, df.loc[idx[5:10]]], ignore_index=True)           # duplicates
    df.loc[idx[10:13], "expiry"] = df.loc[idx[10:13], "date"] - pd.Timedelta(days=3)  # expired rows
    df.loc[idx[13:16], "close"] = df.loc[idx[13:16], "close"] * 1.08     # one-day price spikes
    df.loc[idx[16:19], "volume"] = 0                                     # stale settlement
    return df.sort_values(["date", "symbol", "expiry"]).reset_index(drop=True)


# =====================================================================================
# 2. VALIDATION & CLEANING  +  3. CONTRACT IDENTIFICATION (parse symbol & expiry)
# =====================================================================================
def _squash(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


ALIASES = {
    "date": ["date", "tradedate", "tradingdate", "timestamp"],
    "symbol": ["symbol", "commodity", "commoditysymbol", "contract", "scrip"],
    "expiry": ["expirydate", "expiry", "expdate"],
    "instrument": ["instrumentname", "instrument", "instrumenttype"],
    "open": ["open", "openprice"], "high": ["high", "highprice"], "low": ["low", "lowprice"],
    "close": ["close", "closeprice", "settlementprice", "settleprice", "settle", "ltp"],
    "volume": ["volumeinlots", "volume", "qtylots", "tradedlots", "noofcontracts"],
    "oi": ["openinterestinlots", "openinterest", "oi", "openint"],
}


def map_columns(df):
    cols = {_squash(c): c for c in df.columns}
    out = {}
    for std, al in ALIASES.items():
        for a in al:
            if a in cols:
                out[std] = cols[a]
                break
    return out


SYM_RE = re.compile(r"^([A-Z]+?)[\s_\-]*(\d{1,2})[\s_\-]*([A-Z]{3})[\s_\-]*(\d{2,4})$")


def parse_symbol(s):
    """'GOLDM05DEC2026' -> ('GOLDM', 2026-12-05); plain 'GOLDM' -> ('GOLDM', NaT)."""
    s = str(s).strip().upper()
    m = SYM_RE.match(s)
    if m:
        base, d, mon, y = m.groups()
        y = int(y) + (2000 if int(y) < 100 else 0)
        try:
            return base, pd.Timestamp(year=y, month=pd.to_datetime(mon, format="%b").month, day=int(d))
        except Exception:
            return base, pd.NaT
    return s, pd.NaT


def parse_dates(s):
    s = s.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(s[iso], errors="coerce")
    if (~iso).any():
        out[~iso] = pd.to_datetime(s[~iso], errors="coerce", dayfirst=True)
    return out


@st.cache_data(show_spinner=False)
def run_validation(raw: pd.DataFrame):
    """Returns (clean_df, checks_df, flags_df, meta, error)."""
    checks, flags = [], []

    def add(name, n, status, action):
        checks.append({"Check": name, "Rows affected": int(n), "Status": status, "Action": action})

    cmap = map_columns(raw)
    miss = [c for c in ["date", "symbol", "close"] if c not in cmap]
    if miss:
        return None, None, None, None, f"Missing required columns: {miss}. Found: {list(raw.columns)}"
    d = pd.DataFrame({k: raw[v].values for k, v in cmap.items()})
    n_raw = len(d)
    add("Schema / required fields", 0, "PASS", "Mapped: " + ", ".join(cmap))

    if "instrument" in d:
        m = d["instrument"].astype(str).str.upper().str.startswith("FUT")
        add("Futures-only filter (FUTCOM)", (~m).sum(), "PASS" if m.all() else "INFO", "Option rows dropped")
        d = d[m]

    d["date"] = parse_dates(d["date"])
    bad = d["date"].isna()
    add("Trading date parse", bad.sum(), "PASS" if not bad.any() else "WARN", "Unparseable dates dropped")
    d = d[~bad]

    sym = d["symbol"].astype(str).str.strip().str.upper()
    pm = {u: parse_symbol(u) for u in sym.unique()}
    d["symbol"] = sym.map(lambda s: pm[s][0])
    psexp = pd.to_datetime(sym.map(lambda s: pm[s][1]))
    d["expiry"] = parse_dates(d["expiry"]).fillna(psexp) if "expiry" in d else psexp
    other = ~d["symbol"].isin(SYMBOLS)
    add("Contract identification (GOLDM/GOLDTEN/GOLDGUINEA/GOLDPETAL)", other.sum(),
        "PASS" if not other.any() else "INFO", "Other commodities / contracts ignored (out of Phase-1 scope)")
    d = d[~other]
    if d.empty:
        return None, None, None, None, "No GOLDM / GOLDTEN / GOLDGUINEA / GOLDPETAL rows found in the data."

    noexp = d["expiry"].isna()
    add("Expiry date present / parsed from symbol", noexp.sum(), "PASS" if not noexp.any() else "FAIL", "Dropped")
    d = d[~noexp]

    for c in ["open", "high", "low", "close", "volume", "oi"]:
        if c in d:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")
    n_rel = len(d)
    badp = d["close"].isna() | (d["close"] <= 0)
    add("Missing / non-positive settlement price", badp.sum(), "PASS" if not badp.any() else "WARN", "Dropped")
    d = d[~badp]

    for c in ["volume", "oi"]:
        if c not in d:
            d[c] = 0.0
            add(f"'{c}' column provided", len(d), "WARN", "Not in file; set to 0 (liquidity filters will block)")
        else:
            nn = d[c].isna().sum()
            add(f"Missing {c}", nn, "PASS" if nn == 0 else "WARN", "Filled with 0")
            d[c] = d[c].fillna(0)

    exp = d["expiry"] < d["date"]
    add("Expiry before trading date", exp.sum(), "PASS" if not exp.any() else "FAIL", "Dropped")
    d = d[~exp]
    dup = d.duplicated(["date", "symbol", "expiry"], keep="last")
    add("Duplicate (date, symbol, expiry)", dup.sum(), "PASS" if not dup.any() else "WARN", "Kept last")
    d = d[~dup].sort_values(["symbol", "expiry", "date"]).reset_index(drop=True)

    def flag(mask, issue):
        if mask.any():
            flags.append(d.loc[mask, ["date", "symbol", "expiry", "close", "volume"]].assign(issue=issue))

    if {"high", "low"} <= set(d.columns):
        inc = (d["close"] > d["high"] * 1.0005) | (d["close"] < d["low"] * 0.9995)
        add("OHLC consistency (low <= close <= high)", inc.sum(), "PASS" if not inc.any() else "WARN", "Flagged, kept")
        flag(inc, "OHLC inconsistent")
    zv = d["volume"] <= 0
    add("Zero-volume rows (stale settlement)", zv.sum(), "PASS" if not zv.any() else "WARN",
        "Flagged; liquidity filter blocks signals")
    flag(zv, "Zero volume")
    ret = d.groupby(["symbol", "expiry"])["close"].pct_change().abs()
    jump = ret > 0.05
    add("Daily price jumps > 5% within a contract", jump.sum(), "PASS" if not jump.any() else "WARN",
        "Flagged; persistence filter suppresses one-day spikes")
    flag(jump.fillna(False), "Price jump >5%")
    ud = np.sort(d["date"].unique())
    gaps = int((np.diff(ud).astype("timedelta64[D]").astype(int) > 4).sum()) if len(ud) > 1 else 0
    add("Trading-date gaps (> 4 calendar days)", gaps, "PASS" if gaps == 0 else "WARN", "Review missing bhavcopy days")

    d["exp_month"] = d["expiry"].dt.strftime("%Y-%m")
    d["dte"] = (d["expiry"] - d["date"]).dt.days
    flags_df = pd.concat(flags, ignore_index=True) if flags else pd.DataFrame(
        columns=["date", "symbol", "expiry", "close", "volume", "issue"])
    nflag = flags_df[["date", "symbol", "expiry"]].drop_duplicates().shape[0] if len(flags_df) else 0
    score = max(0.0, 100.0 * (len(d) - nflag) / max(n_rel, 1))
    meta = dict(n_raw=n_raw, n_rel=n_rel, n_out=len(d), flagged=nflag, score=round(score, 1))
    return d, pd.DataFrame(checks), flags_df, meta, None


# =====================================================================================
# 4. PRICE NORMALIZATION  (unit + quotation + purity -> Rs per gram of 999 gold)
# =====================================================================================
def normalize(d, specs):
    s = specs.set_index("symbol")
    d = d.copy()
    d["lot_g"] = d["symbol"].map(s["lot_g"])
    d["quote_g"] = d["symbol"].map(s["quote_g"])
    d["purity"] = d["symbol"].map(s["purity"])
    d["norm"] = d["close"] / d["quote_g"] * REF_PURITY / d["purity"]       # Rs / gram @ 999
    d["lot_notional"] = d["close"] / d["quote_g"] * d["lot_g"]             # Rs per lot
    d["vol_kg"] = d["volume"] * d["lot_g"] / 1000.0
    d["oi_kg"] = d["oi"] * d["lot_g"] / 1000.0
    return d


# =====================================================================================
# 5. RELATIVE-VALUE ENGINE  (contract-specific, matched by expiry month, carry-adjusted)
# =====================================================================================
def _pvalue(z):
    z = np.asarray(z, dtype=float)
    if HAS_SCIPY:
        return 2 * sps.norm.sf(np.abs(z))
    return np.array([math.erfc(abs(v) / math.sqrt(2)) if not np.isnan(v) else np.nan for v in z])


@st.cache_data(show_spinner=False)
def build_pairs(d, carry, window, metric):
    cols = ["date", "exp_month", "expiry", "close", "norm", "volume", "oi", "dte"]
    syms = set(d["symbol"].unique())
    frames = []
    for a, b in PAIRS:
        if a not in syms or b not in syms:
            continue
        A, B = d.loc[d["symbol"] == a, cols], d.loc[d["symbol"] == b, cols]
        m = A.merge(B, on=["date", "exp_month"], suffixes=("_a", "_b"))
        if m.empty:
            continue
        m["pair"], m["a"], m["b"] = f"{a} vs {b}", a, b
        m["exp_gap"] = (m["expiry_a"] - m["expiry_b"]).dt.days
        m["norm_b_adj"] = m["norm_b"] * (1 + carry * m["exp_gap"] / 365.0)   # carry-adjust B to A's expiry date
        m["spread"] = m["norm_a"] - m["norm_b_adj"]                           # Rs / g
        m["ratio"] = m["norm_a"] / m["norm_b_adj"]
        m["raw_gap"] = m["close_a"] - m["close_b"]
        frames.append(m)
    if not frames:
        return pd.DataFrame()
    P = pd.concat(frames, ignore_index=True).sort_values(["pair", "exp_month", "date"]).reset_index(drop=True)
    P["metric"] = P["spread"] if metric.startswith("Spread") else P["ratio"]
    g = P.groupby(["pair", "exp_month"], sort=False)
    mp = max(10, int(window * 0.7))
    roll = lambda col, f: g[col].transform(lambda x: getattr(x.shift(1).rolling(window, min_periods=mp), f)())
    P["mu"], P["sd"] = roll("metric", "mean"), roll("metric", "std")      # shift(1): baseline excludes today
    P["z"] = (P["metric"] - P["mu"]) / P["sd"].replace(0, np.nan)
    P["mu_spread"] = roll("spread", "mean")
    P["dev_rs"] = (P["spread"] - P["mu_spread"]).abs()
    P["dspread"] = g["spread"].diff()
    P["pval"] = _pvalue(P["z"].values)
    P["avg_px"] = (P["norm_a"] + P["norm_b"]) / 2
    P["dte_min"] = np.minimum(P["dte_a"], P["dte_b"])
    return P


def half_life(x):
    x = pd.Series(x).dropna().values[-120:]
    if len(x) < 30:
        return np.nan
    b = np.polyfit(x[:-1] - x[:-1].mean(), np.diff(x), 1)[0]
    return -math.log(2) / math.log(1 + b) if -1 < b < 0 else np.nan


# =====================================================================================
# 6. EXPIRY + LIQUIDITY FILTERS  ->  SIGNAL ENGINE (explainable)
# =====================================================================================
def apply_filters(P, p):
    F = pd.DataFrame(index=P.index)
    F["rt_cost"] = 4 * (p["fee"] + p["slip"]) * p["cost_mult"] / 1e4 * P["avg_px"]   # Rs/g round-trip (4 leg-sides)
    F["f_dev"] = P["z"].abs() >= p["entry"]
    F["f_liq"] = ((P["volume_a"] >= p["min_vol"]) & (P["volume_b"] >= p["min_vol"]) &
                  (P["oi_a"] >= p["min_oi"]) & (P["oi_b"] >= p["min_oi"]))
    F["f_dte"] = P["dte_min"] >= p["min_dte"]
    F["f_cost"] = P["dev_rs"] >= p["hurdle"] * F["rt_cost"]
    s = pd.Series(np.where(F["f_dev"], np.sign(P["z"]), 0.0), index=P.index)
    ss = s.groupby([P["pair"], P["exp_month"]]).transform(
        lambda x: x.rolling(p["persist"], min_periods=p["persist"]).sum())
    F["f_per"] = ss.abs() >= p["persist"]
    F["entry_ok"] = F[["f_dev", "f_liq", "f_dte", "f_cost", "f_per"]].all(axis=1)
    blocked = pd.Series("", index=P.index)
    for c, n in {"f_liq": "liquidity", "f_dte": "expiry proximity", "f_cost": "cost hurdle", "f_per": "persistence"}.items():
        blocked = pd.Series(np.where(F["f_dev"] & ~F[c], blocked + n + "; ", blocked), index=P.index)
    F["blocked_by"] = blocked.str.rstrip("; ")
    F["signal"] = np.select([F["entry_ok"] & (P["z"] < 0), F["entry_ok"] & (P["z"] > 0), P["z"].abs() < 0.5],
                            ["CHEAP", "EXPENSIVE", "NEUTRAL"], default="NO SIGNAL")
    return F


def explain(r, p):
    cheap = r["signal"] == "CHEAP"
    side = f"Long {r['a']} / Short {r['b']}" if cheap else f"Short {r['a']} / Long {r['b']}"
    return (f"<b>{r['a']} looks {'cheap' if cheap else 'rich'} vs {r['b']}</b> on the <b>{r['exp_month']}</b> expiry "
            f"(idea: {side}). Normalized spread ₹{r['spread']:,.1f}/g vs rolling mean ₹{r['mu_spread']:,.1f}/g "
            f"→ z = {r['z']:+.2f} (p = {r['pval']:.3f}). Expiry gap {int(r['exp_gap']):+d}d carry-adjusted at "
            f"{p['carry']:.1%}. Liquidity OK (vol {r['volume_a']:,.0f}/{r['volume_b']:,.0f} lots, OI "
            f"{r['oi_a']:,.0f}/{r['oi_b']:,.0f}); {int(r['dte_min'])}d to nearest expiry. Expected reversion "
            f"₹{r['dev_rs']:,.1f}/g vs round-trip cost ₹{r['rt_cost']:,.1f}/g. Persisted ≥ {p['persist']} day(s).")


# =====================================================================================
# 7. WALK-FORWARD BACKTEST  (decide at close t, fill t+delay, costs, no look-ahead)
# =====================================================================================
def simulate(P, F, p):
    n = len(P)
    z, ok, dte = P["z"].values, F["entry_ok"].values, P["dte_min"].values
    pos = np.zeros(n)
    for idx in P.groupby(["pair", "exp_month"], sort=False).indices.values():
        cur, locked = 0, False
        for i in idx:
            zi = z[i]
            if not np.isnan(zi):
                if locked and abs(zi) < p["entry"]:
                    locked = False
                if cur == 0:
                    if ok[i] and not locked:
                        cur = 1 if zi < 0 else -1
                elif abs(zi) <= p["exit"] or dte[i] < p["exit_dte"]:
                    cur = 0
                elif abs(zi) >= p["stop"]:
                    cur, locked = 0, True
            pos[i] = cur
    S = P[["date", "pair", "exp_month", "dspread", "avg_px", "z"]].copy()
    S["pos"] = pos
    keys = [S["pair"], S["exp_month"]]
    S["filled"] = S.groupby(keys, sort=False)["pos"].shift(p["delay"]).fillna(0)     # fill after delay
    S["held"] = S.groupby(keys, sort=False)["filled"].shift(1).fillna(0)             # position earning today
    S["gross"] = S["held"] * S["dspread"].fillna(0)
    S["cost"] = (S["filled"] - S["held"]).abs() * 2 * (p["fee"] + p["slip"]) * p["cost_mult"] / 1e4 * S["avg_px"]
    S["net"] = S["gross"] - S["cost"]
    return S


def trades_of(S):
    if S.empty:
        return pd.Series(dtype=float)
    new = ((S["filled"] != 0) & (S["held"] == 0)).astype(int)
    seg = new.groupby([S["pair"], S["exp_month"]]).cumsum()
    act = (S["filled"] != 0) | (S["held"] != 0)
    return S.loc[act, "net"].groupby([S.loc[act, "pair"], S.loc[act, "exp_month"], seg[act]]).sum()


def perf_stats(S, exposure):
    empty = dict(net=0.0, gross=0.0, cost=0.0, trades=0, win=np.nan, avg=np.nan, pf=np.nan,
                 sharpe=np.nan, maxdd=0.0, pval=np.nan)
    if S is None or S.empty:
        return empty, pd.DataFrame(columns=["gross", "cost", "net"]), pd.Series(dtype=float)
    daily = S.groupby("date")[["gross", "cost", "net"]].sum() * exposure
    tr = trades_of(S) * exposure
    eq = daily["net"].cumsum()
    sd = daily["net"].std()
    gains, losses = tr[tr > 0].sum(), -tr[tr < 0].sum()
    pval = np.nan
    if HAS_SCIPY and len(daily) > 5 and sd and sd > 0:
        pval = float(sps.ttest_1samp(daily["net"], 0.0, alternative="greater").pvalue)
    st_ = dict(net=daily["net"].sum(), gross=daily["gross"].sum(), cost=daily["cost"].sum(), trades=len(tr),
               win=(tr > 0).mean() if len(tr) else np.nan, avg=tr.mean() if len(tr) else np.nan,
               pf=gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan),
               sharpe=daily["net"].mean() / sd * math.sqrt(252) if sd and sd > 0 else np.nan,
               maxdd=(eq - eq.cummax()).min(), pval=pval)
    return st_, daily, tr


@st.cache_data(show_spinner="Running walk-forward validation…")
def run_wf(P, p, grid, train, test):
    P = P.reset_index(drop=True)
    dates = np.sort(P["date"].unique())
    FE = {e: apply_filters(P, {**p, "entry": e}) for e in grid}
    folds, oos, i = [], [], 0
    while i + train + 10 <= len(dates):
        tr_d, te_d = dates[i:i + train], dates[i + train:i + train + test]
        mtr, mte = P["date"].isin(tr_d).values, P["date"].isin(te_d).values
        best_e, best_v = None, 0.0
        for e in grid:
            v = simulate(P[mtr], FE[e][mtr], {**p, "entry": e})["net"].sum()
            if v > best_v:
                best_e, best_v = e, v
        row = dict(Fold=len(folds) + 1, Train=f"{pd.Timestamp(tr_d[0]):%d %b %y} → {pd.Timestamp(tr_d[-1]):%d %b %y}",
                   Test=f"{pd.Timestamp(te_d[0]):%d %b %y} → {pd.Timestamp(te_d[-1]):%d %b %y}",
                   Entry_z=best_e if best_e is not None else "stand aside",
                   Train_net=round(best_v * p["exposure"], 0), Test_net=0.0, Test_trades=0)
        if best_e is not None:
            S = simulate(P[mte], FE[best_e][mte], {**p, "entry": best_e})
            S["fold"] = row["Fold"]
            oos.append(S)
            row["Test_net"] = round(S["net"].sum() * p["exposure"], 0)
            row["Test_trades"] = len(trades_of(S))
        folds.append(row)
        i += test
    S_oos = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    S_is = simulate(P, apply_filters(P, p), p)
    sens = []
    for m in [0, 0.5, 1, 2, 3]:
        q = {**p, "cost_mult": m}
        s_ = simulate(P, apply_filters(P, q), q)
        sens.append(dict(cost_multiple=m, net=perf_stats(s_, p["exposure"])[0]["net"]))
    return dict(folds=pd.DataFrame(folds), S_oos=S_oos, S_is=S_is, sens=pd.DataFrame(sens))


# =====================================================================================
# UI helpers
# =====================================================================================
_k = [0]
_VER = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])


def show(fig, h=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.14, x=0),
                      font=dict(color="#C9D4E3"), **({"height": h} if h else {}))
    _k[0] += 1
    key = f"pc{_k[0]}"
    if _VER >= (1, 50):
        try:
            st.plotly_chart(fig, width="stretch", key=key)
            return
        except Exception:
            pass
    st.plotly_chart(fig, use_container_width=True, key=key)


def table(df, height=None):
    kw = {"hide_index": True, **({"height": height} if height else {})}
    if _VER >= (1, 50):
        try:
            st.dataframe(df, width="stretch", **kw)
            return
        except Exception:
            pass
    st.dataframe(df, use_container_width=True, **kw)


def card(col, label, value, sub="", color="#F3F6FA"):
    col.markdown(f'<div class="gx-card"><div class="l">{label}</div><div class="v" style="color:{color}">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def pill(sig):
    cls = {"CHEAP": "p-cheap", "EXPENSIVE": "p-exp", "NEUTRAL": "p-neu"}.get(sig, "p-no")
    return f'<span class="gx-pill {cls}">{sig}</span>'


def flow(steps):
    html = '<div class="gx-flow">'
    for i, (t, s) in enumerate(steps):
        html += f'<div class="gx-step"><b>{t}</b><br><span style="color:#8B98A9">{s}</span></div>'
        if i < len(steps) - 1:
            html += '<span class="gx-arrow">➜</span>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def inr(x, d=0):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"₹{x:,.{d}f}"



# =====================================================================================
# GOLDX PROJECT MODE: actual same-underlying, different-expiry analysis
# =====================================================================================
def render_goldx_project_dashboard(market_raw: pd.DataFrame, signals_raw: pd.DataFrame):
    """Render the existing dashboard shell using the project's exported GOLDX CSVs."""
    market = market_raw.copy()
    sig = signals_raw.copy()

    required_market = {"Date", "Symbol", "Expiry Date", "Close", "Volume(Lots)"}
    required_signals = {"Date", "Spread", "Z_Score", "Potential_Anomaly", "Signal", "Liquidity_Pair"}
    missing_m = sorted(required_market - set(market.columns))
    missing_s = sorted(required_signals - set(sig.columns))
    if missing_m or missing_s:
        st.error(f"CSV schema mismatch. Market data missing: {missing_m}; signals missing: {missing_s}")
        st.stop()

    market["Date"] = pd.to_datetime(market["Date"], errors="coerce")
    market["Expiry Date"] = pd.to_datetime(market["Expiry Date"], errors="coerce")
    sig["Date"] = pd.to_datetime(sig["Date"], errors="coerce")
    for c in ["Close", "Volume(Lots)", "Open Interest(Lots)"]:
        if c in market.columns:
            market[c] = pd.to_numeric(market[c], errors="coerce")
    for c in ["Spread", "Z_Score", "Historical_Mean", "Historical_Std",
              "Near_Volume", "Far_Volume", "Near_Open_Interest", "Far_Open_Interest"]:
        if c in sig.columns:
            sig[c] = pd.to_numeric(sig[c], errors="coerce")
    sig["Potential_Anomaly"] = sig["Potential_Anomaly"].astype(str).str.lower().isin(
        ["true", "1", "yes", "y"])
    if "Both_Legs_Traded" in sig.columns:
        sig["Both_Legs_Traded"] = sig["Both_Legs_Traded"].astype(str).str.lower().isin(
            ["true", "1", "yes", "y"])
    market = market.dropna(subset=["Date", "Expiry Date", "Close"]).sort_values("Date")
    sig = sig.dropna(subset=["Date"]).sort_values("Date")
    if market.empty or sig.empty:
        st.error("The uploaded CSVs contain no usable dated records.")
        st.stop()

    last = market["Date"].max()
    n_dates = market["Date"].nunique()
    n_expiries = market["Expiry Date"].nunique()
    anomalies = sig[sig["Potential_Anomaly"]]
    valid_spreads = sig["Spread"].notna().sum()
    valid_z = sig["Z_Score"].notna().sum()
    both_traded = sig["Both_Legs_Traded"].sum() if "Both_Legs_Traded" in sig else np.nan

    st.markdown("### GOLDX — Real Project Data")
    st.info("Real-data mode: charts and alerts below use the uploaded GOLDX CSV exports. "
            "This view does not claim live market access or proven trading profitability.")
    k = st.columns(5)
    card(k[0], "Market rows", f"{len(market):,}", "after date/close parsing")
    card(k[1], "Trading dates", f"{n_dates:,}", f"{market['Date'].min():%d %b} – {last:%d %b %Y}")
    card(k[2], "Expiry contracts", n_expiries, "same underlying: GOLD")
    card(k[3], "Valid spread rows", f"{valid_spreads:,}", f"{valid_z:,} rows with z-score")
    card(k[4], "Flagged candidates", len(anomalies), "candidate anomalies; not trade recommendations",
         GOLD if len(anomalies) else GREY)

    tabs = st.tabs(["Overview", "Relative Value", "Liquidity & Data Quality", "Candidate Alerts", "Download"])
    t_over, t_rv, t_liq, t_alert, t_download = tabs

    with t_over:
        st.markdown("#### GOLD futures settlements by expiry")
        mplot = market.copy()
        mplot["Expiry"] = mplot["Expiry Date"].dt.strftime("%d %b %Y")
        fig = go.Figure()
        for expiry, group in mplot.groupby("Expiry", sort=True):
            fig.add_scatter(x=group["Date"], y=group["Close"], mode="lines+markers",
                            name=f"Expiry {expiry}")
        fig.update_layout(title="GOLD settlement price by expiry", xaxis_title="Trading date",
                          yaxis_title="Settlement price", legend_title="Contract expiry")
        show(fig, 380)

        st.markdown("#### Trading volume by expiry")
        v = market.groupby("Expiry Date", as_index=False).agg(
            Total_Volume=("Volume(Lots)", "sum"),
            Zero_Volume_Dates=("Volume(Lots)", lambda x: int((x.fillna(0) <= 0).sum())),
            Rows=("Date", "count"))
        v["Expiry"] = v["Expiry Date"].dt.strftime("%d %b %Y")
        fig = go.Figure(go.Bar(x=v["Expiry"], y=v["Total_Volume"], text=v["Total_Volume"],
                               textposition="outside"))
        fig.update_layout(title="Total recorded volume by expiry", yaxis_title="Lots")
        show(fig, 300)
        table(v[["Expiry", "Rows", "Total_Volume", "Zero_Volume_Dates"]])

    with t_rv:
        pairs = sorted(sig["Liquidity_Pair"].dropna().astype(str).unique())
        selected = st.selectbox("Expiry pair", pairs, key="goldx_project_pair")
        ss = sig[sig["Liquidity_Pair"].astype(str) == selected].sort_values("Date")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_scatter(x=ss["Date"], y=ss["Spread"], mode="lines+markers", name="Spread")
            if "Historical_Mean" in ss:
                fig.add_scatter(x=ss["Date"], y=ss["Historical_Mean"], mode="lines",
                                name="Historical mean")
            fig.update_layout(title=f"{selected}: spread and historical mean",
                              xaxis_title="Trading date", yaxis_title="Spread (source units)")
            show(fig, 340)
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=ss["Date"], y=ss["Z_Score"], mode="lines+markers", name="Rolling z-score")
            fig.add_hline(y=2, line_dash="dash", annotation_text="+2 threshold")
            fig.add_hline(y=-2, line_dash="dash", annotation_text="-2 threshold")
            fig.add_hline(y=0, line_color="#8B98A9")
            fig.update_layout(title=f"{selected}: rolling z-score", xaxis_title="Trading date",
                              yaxis_title="Z-score")
            show(fig, 340)
        st.caption("The z-score thresholds are visual reference lines. A flagged candidate is not proof of mispricing.")
        cols = [c for c in ["Date", "Spread", "Historical_Mean", "Historical_Std", "Z_Score",
                            "Signal", "Potential_Anomaly", "Liquidity_Status"] if c in ss.columns]
        table(ss[cols].round(4))

    with t_liq:
        st.markdown("#### Liquidity coverage by expiry pair")
        agg = sig.groupby("Liquidity_Pair", dropna=False).agg(
            Signal_Rows=("Date", "count"),
            Valid_Spread_Rows=("Spread", "count"),
            Valid_Z_Rows=("Z_Score", "count"),
            Candidate_Count=("Potential_Anomaly", "sum")).reset_index()
        if "Both_Legs_Traded" in sig:
            traded = sig.groupby("Liquidity_Pair")["Both_Legs_Traded"].sum().rename("Both_Legs_Traded_Rows")
            agg = agg.merge(traded, on="Liquidity_Pair", how="left")
        table(agg)
        if {"Near_Volume", "Far_Volume"} <= set(sig.columns):
            liq = sig.groupby("Liquidity_Pair", as_index=False).agg(
                Mean_Near_Volume=("Near_Volume", "mean"),
                Mean_Far_Volume=("Far_Volume", "mean"),
                Minimum_Far_Volume=("Far_Volume", "min"))
            fig = go.Figure()
            fig.add_bar(x=liq["Liquidity_Pair"], y=liq["Mean_Near_Volume"], name="Mean near volume")
            fig.add_bar(x=liq["Liquidity_Pair"], y=liq["Mean_Far_Volume"], name="Mean far volume")
            fig.update_layout(barmode="group", title="Average volume per leg", yaxis_title="Lots")
            show(fig, 320)
            table(liq.round(2))
        zero = market[market["Volume(Lots)"].fillna(0) <= 0]
        if len(zero):
            st.warning(f"{len(zero)} market rows have zero or missing volume. Treat those observations cautiously; "
                       "do not interpret a stale settlement as an executable opportunity.")
            zcols = [c for c in ["Date", "Symbol", "Expiry Date", "Close", "Volume(Lots)", "Open", "High", "Low"]
                     if c in zero.columns]
            table(zero[zcols].head(100))

    with t_alert:
        st.markdown("#### Candidate anomalies from the exported signal file")
        if anomalies.empty:
            st.info("No candidate anomaly is flagged in the uploaded signal file.")
        else:
            for _, r in anomalies.sort_values("Date", ascending=False).iterrows():
                z = r.get("Z_Score", np.nan)
                st.markdown(
                    f'<div class="gx-alert"><b>{r.get("Liquidity_Pair", "Unknown pair")}</b> · '
                    f'{r["Date"]:%d %b %Y}<br>Signal: <b>{r.get("Signal", "Candidate")}</b> · '
                    f'Spread: {r.get("Spread", np.nan):,.2f} · Z-score: {z:+.3f}<br>'
                    f'Liquidity status: {r.get("Liquidity_Status", "Not provided")}</div>',
                    unsafe_allow_html=True)
            st.caption("These are statistical candidates from historical observations, not confirmed arbitrage, "
                       "investment advice, or evidence of profitability.")
        alert_cols = [c for c in ["Date", "Liquidity_Pair", "Spread", "Z_Score", "Signal",
                                  "Near_Volume", "Far_Volume", "Liquidity_Status"]
                      if c in anomalies.columns]
        if not anomalies.empty and alert_cols:
            table(anomalies[alert_cols].round(4))

    with t_download:
        st.markdown("#### Export the integrated source data")
        c1, c2 = st.columns(2)
        c1.download_button("Download market data CSV", market_raw.to_csv(index=False).encode("utf-8"),
                           "goldx_application_market_data.csv", "text/csv")
        c2.download_button("Download signals CSV", signals_raw.to_csv(index=False).encode("utf-8"),
                           "goldx_application_signals.csv", "text/csv")
        st.markdown("#### Data notes")
        st.write(f"- Market data records supplied: **{len(market_raw)}**")
        st.write(f"- Signal records supplied: **{len(signals_raw)}**")
        st.write("- The February–April pair contains zero-volume observations; the dashboard shows these explicitly.")
        st.write("- Backtest/profitability claims from synthetic demo mode are not reused as real-data results.")

# =====================================================================================
# MAIN APP
# =====================================================================================
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="gx-hero"><h1>🥇 <span class="g">GOLD</span>X &nbsp;·&nbsp; Commodity Derivatives Intelligence</h1>
    <div class="gx-sub">Contract-aware relative-pricing for MCX gold futures &nbsp;|&nbsp; Normalize → Detect → Validate
    &nbsp;|&nbsp; Team TechNova · Hack in Hills</div>
    <div class="gx-sub gx-quote">We don't search for a signal every day. We search for a deviation that survives scrutiny.</div>
    </div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.markdown("### 🥇 GOLDX Control Panel")
        src = st.radio("Data source", ["GOLDX Project CSVs (real data)", "Demo (synthetic)", "Upload MCX Bhavcopy"])
        raw = None
        project_market = project_signals = None
        if src == "GOLDX Project CSVs (real data)":
            market_file = st.file_uploader("Upload goldx_application_market_data.csv", type=["csv"], key="project_market")
            signals_file = st.file_uploader("Upload goldx_application_signals.csv", type=["csv"], key="project_signals")
            if market_file is not None and signals_file is not None:
                try:
                    project_market = pd.read_csv(market_file)
                    project_signals = pd.read_csv(signals_file)
                except Exception as e:
                    st.error(f"Could not read GOLDX project CSVs: {e}")
        elif src.startswith("Upload"):
            files = st.file_uploader("Daily Bhavcopy (CSV / Parquet, multiple days allowed)",
                                     type=["csv", "parquet"], accept_multiple_files=True)
            parts = []
            for f in files or []:
                try:
                    parts.append(pd.read_parquet(f) if f.name.lower().endswith(".parquet") else pd.read_csv(f))
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            raw = pd.concat(parts, ignore_index=True) if parts else None
        else:
            raw = make_demo()
            st.caption("Synthetic data in MCX bhavcopy layout (includes deliberately injected data errors).")
    if src == "GOLDX Project CSVs (real data)":
        if project_market is None or project_signals is None:
            st.info("Upload both GOLDX project CSV files above to open the real-data dashboard.")
            st.stop()
        render_goldx_project_dashboard(project_market, project_signals)
        st.stop()
    if raw is None:
        st.info("Upload MCX daily bhavcopy files (OHLC + Volume + OI + Symbol + Expiry) or switch to Demo data.")
        st.stop()

    d, checks, flags, meta, err = run_validation(raw)
    if err:
        st.error(err)
        st.stop()

    with st.sidebar:
        with st.expander("📐 Contract specifications", expanded=False):
            specs = st.data_editor(DEFAULT_SPECS, hide_index=True, disabled=["symbol"], key="specs")
            st.caption("Quote basis = grams the quoted price refers to. Purity normalised to 999.")
        with st.expander("📊 Relative-value engine", expanded=True):
            metric = st.selectbox("Deviation metric", ["Spread (₹/g)", "Ratio"])
            window = st.slider("Rolling window (days)", 15, 90, 30)
            carry = st.slider("Cost-of-carry for expiry alignment (% p.a.)", 0.0, 12.0, 7.0, 0.5) / 100
            entry = st.slider("Entry |z|", 1.0, 3.5, 2.0, 0.1)
            exit_ = st.slider("Exit |z|", 0.0, 1.0, 0.5, 0.1)
            stop = st.slider("Stop |z|", 3.0, 8.0, 4.5, 0.5)
            persist = st.slider("Persistence (consecutive days)", 1, 5, 2)
        with st.expander("🛡️ Liquidity & expiry filters", expanded=False):
            min_vol = st.number_input("Min volume per leg (lots)", 0, 100000, 25)
            min_oi = st.number_input("Min open interest per leg (lots)", 0, 1000000, 1000)
            min_dte = st.slider("No new entries within N days of expiry", 1, 20, 7)
            exit_dte = st.slider("Force exit within N days of expiry", 1, 10, 3)
        with st.expander("💸 Cost & execution model", expanded=False):
            fee = st.number_input("Brokerage + exchange fees (bps / leg-side)", 0.0, 20.0, 0.5, 0.1)
            slip = st.number_input("Slippage (bps / leg-side)", 0.0, 30.0, 1.0, 0.1)
            hurdle = st.slider("Reversion must exceed N × round-trip cost", 0.0, 5.0, 1.5, 0.5)
            delay = st.slider("Execution delay (days after signal)", 1, 3, 1)
            exposure = st.number_input("Exposure per leg (grams)", 1, 100000, 100)
        with st.expander("🧪 Walk-forward settings", expanded=False):
            train = st.slider("Train window (days)", 60, 250, 120, 10)
            test = st.slider("Test window (days)", 20, 90, 40, 5)
            grid = tuple(st.multiselect("Entry-z candidates (chosen on train only)", [1.5, 2.0, 2.5, 3.0, 3.5],
                                        default=[1.5, 2.0, 2.5, 3.0]) or [2.0])
            scope = st.radio("Backtest scope", ["Selected pair (all expiries)", "All pairs"])

    p = dict(entry=entry, exit=exit_, stop=stop, persist=persist, min_vol=min_vol, min_oi=min_oi, min_dte=min_dte,
             exit_dte=exit_dte, fee=fee, slip=slip, hurdle=hurdle, delay=delay, exposure=exposure,
             cost_mult=1.0, carry=carry)

    dn = normalize(d, specs)
    P = build_pairs(dn, carry, window, metric)
    if P.empty:
        st.warning("No matching expiry-month pairs between the gold contracts in this data.")
        st.stop()
    F = apply_filters(P, p)
    Pf = P.join(F)

    with st.sidebar:
        st.markdown("---")
        pairs_av = sorted(P["pair"].unique())
        sel_pair = st.selectbox("Pair (A vs B)", pairs_av)
        last = P["date"].max()
        act_m = sorted(P.loc[(P["pair"] == sel_pair) & (P["date"] == last), "exp_month"].unique())
        months = sorted(P.loc[P["pair"] == sel_pair, "exp_month"].unique())
        sel_month = st.selectbox("Expiry month", months, index=months.index(act_m[0]) if act_m else len(months) - 1)
        st.caption(f"Data: {d['date'].min():%d %b %Y} → {last:%d %b %Y}")

    Pscope = P if scope == "All pairs" else P[P["pair"] == sel_pair]
    wf = run_wf(Pscope, p, grid, train, test)
    oos_st, oos_daily, oos_tr = perf_stats(wf["S_oos"], exposure)
    is_st, _, _ = perf_stats(wf["S_is"], exposure)

    sel = Pf[(Pf["pair"] == sel_pair) & (Pf["exp_month"] == sel_month)].sort_values("date")
    today = Pf[Pf["date"] == last]
    final_today = today[today["signal"].isin(["CHEAP", "EXPENSIVE"])].sort_values("z", key=lambda s: -s.abs())

    tabs = st.tabs(["🏠 Command Center", "🧹 Data Pipeline", "⚖️ Normalization", "📈 Relative Value",
                    "⏳ Expiry & Liquidity", "🔔 Signals", "🧪 Walk-Forward", "ℹ️ About"])
    t_home, t_data, t_norm, t_rv, t_exp, t_sig, t_wf, t_about = tabs

    # ================================================================== 1. COMMAND CENTER
    with t_home:
        raw_today = int((today["f_dev"]).sum())
        c = st.columns(5)
        card(c[0], "Latest settlement date", f"{last:%d %b %Y}", f"{today.groupby(['pair','exp_month']).ngroups} contract-pair series live")
        card(c[1], "Live contracts", int(dn[dn["date"] == last].shape[0]), "symbol × expiry tracked individually")
        card(c[2], "Raw deviations today", raw_today, f"|z| ≥ {entry}")
        card(c[3], "Signals after scrutiny", len(final_today), "survive liquidity · expiry · cost · persistence",
             GREEN if len(final_today) else GREY)
        card(c[4], "Data quality score", f"{meta['score']}%", f"{meta['flagged']} rows flagged")
        st.write("")
        st.markdown("#### 🔔 Today's explainable alerts")
        if len(final_today):
            for _, r in final_today.head(8).iterrows():
                st.markdown(f'<div class="gx-alert">{pill(r["signal"])} &nbsp; <b>{r["pair"]}</b> · {r["exp_month"]}<br>'
                            f'{explain(r, p)}</div>', unsafe_allow_html=True)
        else:
            st.success("All quiet. No deviation survives contract, liquidity, expiry, cost and persistence checks today — "
                       "that is by design.")
            wl = today[today["f_dev"]].sort_values("z", key=lambda s: -s.abs()).head(8)
            if len(wl):
                st.caption("Watchlist: raw deviations that were filtered out (and why)")
                table(wl[["pair", "exp_month", "z", "spread", "dte_min", "blocked_by"]].round(2))
        st.markdown("#### Signal survival (all history)")
        daily = Pf.groupby("date").agg(raw=("f_dev", "sum"), final=("entry_ok", "sum")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=daily["date"], y=daily["raw"], name="Raw |z| deviations", marker_color="#3B4A66")
        fig.add_bar(x=daily["date"], y=daily["final"], name="Survived all filters", marker_color=GOLD)
        fig.update_layout(barmode="overlay", title="Quiet by design: most raw deviations are structural noise")
        show(fig, 300)
        surv = 100 * Pf["entry_ok"].sum() / max(Pf["f_dev"].sum(), 1)
        c2 = st.columns(4)
        card(c2[0], "Signal survival rate", f"{surv:.1f}%", "final ÷ raw deviations")
        card(c2[1], "OOS net P&L", inr(oos_st["net"]), f"{oos_st['trades']} trades · after costs",
             GREEN if oos_st["net"] > 0 else RED)
        card(c2[2], "OOS Sharpe (ann.)", "—" if np.isnan(oos_st["sharpe"]) else f"{oos_st['sharpe']:.2f}", "walk-forward")
        card(c2[3], "Execution assumption", f"T+{delay}", "settlement ≠ executable fill")

    # ================================================================== 2. DATA PIPELINE
    with t_data:
        st.markdown("#### Data flow")
        flow([("MCX Bhavcopy", "OHLC + Vol + OI"), ("Validation & Cleaning", "dates · expiry · symbol · missing"),
              ("Contract ID", "GOLDM · TEN · GUINEA · PETAL"), ("Price Normalization", "unit · quotation · purity"),
              ("Relative-Value Engine", "spreads · ratios · z"), ("Expiry + Liquidity", "DTE · volume · OI"),
              ("Walk-Forward", "no look-ahead · costs"), ("Dashboard & Alerts", "signal + context")])
        c = st.columns(4)
        card(c[0], "Raw rows", f"{meta['n_raw']:,}", "as ingested")
        card(c[1], "Gold rows analysed", f"{meta['n_rel']:,}", "after contract identification")
        card(c[2], "Clean rows", f"{meta['n_out']:,}", f"{meta['n_rel'] - meta['n_out']:,} dropped")
        card(c[3], "Quality score", f"{meta['score']}%", "clean & unflagged ÷ analysed",
             GREEN if meta["score"] >= 95 else GOLD)
        st.markdown("#### Validation report")
        ck = checks.copy()
        ck["Status"] = ck["Status"].map({"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL", "INFO": "ℹ️ INFO"})
        table(ck)
        if len(flags):
            st.markdown("#### Flagged rows")
            iss = st.multiselect("Issue types", sorted(flags["issue"].unique()), default=sorted(flags["issue"].unique()))
            table(flags[flags["issue"].isin(iss)].head(500))
        cnt = d.groupby("date").size().reset_index(name="rows")
        fig = go.Figure(go.Bar(x=cnt["date"], y=cnt["rows"], marker_color=BLUE))
        fig.update_layout(title="Contracts per trading day (completeness)")
        show(fig, 260)
        st.markdown("#### Storage (CSV / Parquet)")
        s1, s2, s3 = st.columns(3)
        s1.download_button("⬇️ Clean data (CSV)", d.to_csv(index=False).encode(), "goldx_clean.csv", "text/csv")
        try:
            import io
            buf = io.BytesIO()
            d.to_parquet(buf, index=False)
            s2.download_button("⬇️ Clean data (Parquet)", buf.getvalue(), "goldx_clean.parquet")
        except Exception:
            s2.caption("Install pyarrow for Parquet export.")
        if s3.button("💾 Save to ./goldx_store"):
            STORE_DIR.mkdir(exist_ok=True)
            d.to_csv(STORE_DIR / "goldx_clean.csv", index=False)
            try:
                d.to_parquet(STORE_DIR / "goldx_clean.parquet", index=False)
            except Exception:
                pass
            st.success(f"Saved to {STORE_DIR.resolve()}")

    # ================================================================== 3. NORMALIZATION
    with t_norm:
        st.markdown("#### Why raw prices aren't comparable")
        sp = specs.copy()
        sp["quotation"] = "₹ per " + sp["quote_g"].astype(str) + " g"
        sp["lot"] = sp["lot_g"].astype(str) + " g"
        table(sp[["symbol", "lot", "quotation", "purity"]].rename(columns=str.title))
        st.latex(r"\text{Normalized price (₹/g @999)}=\frac{\text{Settlement}}{\text{Quotation grams}}\times\frac{999}{\text{Purity}}")
        near = dn.sort_values(["date", "expiry"]).groupby(["date", "symbol"], as_index=False).first()
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for s_ in SYMBOLS:
                x = near[near["symbol"] == s_]
                fig.add_scatter(x=x["date"], y=x["close"], name=s_, mode="lines")
            fig.update_layout(title="RAW settlement (different units → not comparable)", yaxis_type="log")
            show(fig, 340)
        with c2:
            fig = go.Figure()
            for s_ in SYMBOLS:
                x = near[near["symbol"] == s_]
                fig.add_scatter(x=x["date"], y=x["norm"], name=s_, mode="lines")
            fig.update_layout(title="NORMALIZED (₹ per gram @ 999)")
            show(fig, 340)
        snap = dn[(dn["date"] == last)].sort_values(["symbol", "expiry"]).groupby("symbol", as_index=False).first()
        fig = go.Figure(go.Bar(x=snap["symbol"], y=snap["lot_notional"] / 1000, marker_color=GOLD,
                               text=(snap["lot_notional"] / 1000).round(0), textposition="outside"))
        fig.update_layout(title="Approx. notional per lot (₹ thousand, log scale)", yaxis_type="log")
        show(fig, 320)
        t = snap[["symbol", "expiry", "close", "norm", "lot_notional", "volume", "oi"]].copy()
        t["expiry"] = t["expiry"].dt.strftime("%d %b %Y")
        t.columns = ["Symbol", "Near expiry", "Raw settle", "Normalized ₹/g", "Notional/lot ₹", "Volume (lots)", "OI (lots)"]
        table(t.round(1))

    # ================================================================== 4. RELATIVE VALUE
    with t_rv:
        if sel.empty:
            st.info("No data for this selection.")
        else:
            lastrow = sel.iloc[-1]
            hl = half_life(sel["spread"])
            c = st.columns(5)
            card(c[0], "Normalized spread", f"₹{lastrow['spread']:,.1f}/g", f"rolling mean ₹{lastrow['mu_spread']:,.1f}"
                 if not np.isnan(lastrow["mu_spread"]) else "warming up")
            card(c[1], "Z-score", "—" if np.isnan(lastrow["z"]) else f"{lastrow['z']:+.2f}",
                 "" if np.isnan(lastrow["pval"]) else f"p = {lastrow['pval']:.3f}")
            card(c[2], "Signal", pill(lastrow["signal"]), lastrow["blocked_by"] or "passes filters" if lastrow["f_dev"] else "")
            card(c[3], "Mean-reversion half-life", "—" if np.isnan(hl) else f"{hl:.1f} d", "AR(1) on spread")
            card(c[4], "Expiry gap A−B", f"{int(lastrow['exp_gap']):+d} d", f"{lastrow['dte_min']:.0f}d to nearest expiry")
            st.write("")
            fig = go.Figure()
            fig.add_scatter(x=sel["date"], y=sel["norm_a"], name=f"{sel_pair.split(' vs ')[0]} (₹/g)", line=dict(color=GOLD))
            fig.add_scatter(x=sel["date"], y=sel["norm_b_adj"], name=f"{sel_pair.split(' vs ')[1]} carry-adj (₹/g)",
                            line=dict(color=BLUE))
            fig.update_layout(title=f"Normalized prices · {sel_pair} · {sel_month} expiry")
            show(fig, 300)
            fig = go.Figure()
            fig.add_scatter(x=sel["date"], y=sel["metric"], name=metric, line=dict(color="#E6EDF3", width=2))
            fig.add_scatter(x=sel["date"], y=sel["mu"], name="Rolling mean", line=dict(color=GREY, dash="dash"))
            fig.add_scatter(x=sel["date"], y=sel["mu"] + entry * sel["sd"], name=f"+{entry}σ", line=dict(color=RED, width=1))
            fig.add_scatter(x=sel["date"], y=sel["mu"] - entry * sel["sd"], name=f"−{entry}σ", line=dict(color=GREEN, width=1),
                            fill="tonexty", fillcolor="rgba(120,140,170,0.08)")
            for sg, colr in (("CHEAP", GREEN), ("EXPENSIVE", RED)):
                x = sel[sel["signal"] == sg]
                fig.add_scatter(x=x["date"], y=x["metric"], mode="markers", name=sg,
                                marker=dict(color=colr, size=10, line=dict(color="white", width=1)))
            x = sel[sel["f_dev"] & ~sel["entry_ok"]]
            fig.add_scatter(x=x["date"], y=x["metric"], mode="markers", name="Filtered deviation",
                            marker=dict(color="#8B98A9", symbol="x", size=8), text=x["blocked_by"],
                            hovertemplate="%{x|%d %b}<br>blocked by: %{text}<extra></extra>")
            fig.update_layout(title=f"{metric} with statistical bands (baseline excludes today → no look-ahead)")
            show(fig, 380)
            fz = go.Figure()
            fz.add_scatter(x=sel["date"], y=sel["z"], name="z", line=dict(color=PURPLE))
            for lv, cl in ((entry, RED), (-entry, GREEN), (exit_, GREY), (-exit_, GREY)):
                fz.add_hline(y=lv, line_dash="dot", line_color=cl)
            if HAS_SK and st.checkbox("ML cross-check: IsolationForest (diagnostic only, in-sample, not used for signals)"):
                X = sel[["z", "dspread"]].dropna()
                if len(X) > 30:
                    flag_ = IsolationForest(contamination=0.05, random_state=0).fit_predict(X) == -1
                    fz.add_scatter(x=sel.loc[X.index[flag_], "date"], y=sel.loc[X.index[flag_], "z"], mode="markers",
                                   name="ML outlier", marker=dict(color=GOLD, symbol="diamond", size=9))
            fz.update_layout(title="Rolling z-score")
            show(fz, 280)
            st.caption("Settlement prices are analytical evidence, not executable fills. A grey ✕ is a deviation that was "
                       "suppressed by a filter (hover to see why).")

    # ================================================================== 5. EXPIRY & LIQUIDITY
    with t_exp:
        alld = sorted(dn["date"].dt.date.unique())
        sd_ = st.select_slider("Snapshot date", options=alld, value=alld[-1])
        snap = dn[dn["date"].dt.date == sd_]
        fig = go.Figure()
        for s_ in SYMBOLS:
            x = snap[snap["symbol"] == s_].sort_values("expiry")
            fig.add_scatter(x=x["expiry"], y=x["norm"], mode="lines+markers", name=s_)
        fig.update_layout(title=f"Term structure · normalized ₹/g · {sd_:%d %b %Y} (each point = one contract)")
        show(fig, 340)
        c1, c2 = st.columns(2)
        with c1:
            ls = st.selectbox("Contract symbol", SYMBOLS, index=SYMBOLS.index(sel_pair.split(" vs ")[0]))
        with c2:
            lm = st.selectbox("Contract expiry month", sorted(dn.loc[dn["symbol"] == ls, "exp_month"].unique()),
                              index=max(0, len(sorted(dn.loc[dn["symbol"] == ls, "exp_month"].unique())) - 6))
        lc = dn[(dn["symbol"] == ls) & (dn["exp_month"] == lm)].sort_values("dte")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=lc["dte"], y=lc["volume"], name="Volume (lots)", marker_color="#3B6EA8", secondary_y=False)
        fig.add_scatter(x=lc["dte"], y=lc["oi"], name="Open interest (lots)", line=dict(color=GOLD), secondary_y=True)
        fig.add_vline(x=min_dte, line_dash="dot", line_color=RED, annotation_text="entry cut-off")
        fig.update_xaxes(autorange="reversed", title="Days to expiry")
        fig.update_layout(title=f"Contract life-cycle · {ls} {lm}: liquidity decays into expiry")
        show(fig, 320)
        st.markdown("##### Expiry effect: naive continuous series vs contract-aware series")
        a_, b_ = sel_pair.split(" vs ")
        na = dn[dn["symbol"] == a_].sort_values(["date", "expiry"]).groupby("date")["norm"].first()
        nb = dn[dn["symbol"] == b_].sort_values(["date", "expiry"]).groupby("date")["norm"].first()
        naive = (na - nb).dropna()
        fig = go.Figure()
        fig.add_scatter(x=naive.index, y=naive, name="Naive near-month spread", line=dict(color=RED, width=1.5))
        first = True
        for _, gdf in Pf[Pf["pair"] == sel_pair].groupby("exp_month"):
            fig.add_scatter(x=gdf["date"], y=gdf["spread"], name="Contract-aware (per expiry)", showlegend=first,
                            line=dict(color=GREEN, width=1))
            first = False
        fig.update_layout(title="Rolling contracts create artificial jumps; expiry-matched series do not (₹/g)")
        show(fig, 340)
        st.markdown("##### Liquidity scorecard (latest date)")
        L = dn[dn["date"] == last].groupby("symbol").agg(contracts=("expiry", "nunique"), volume_lots=("volume", "sum"),
                                                          oi_lots=("oi", "sum"), volume_kg=("vol_kg", "sum"),
                                                          oi_kg=("oi_kg", "sum")).reset_index()
        table(L.round(1))
        pr = Pf.groupby("pair").agg(Liquidity_pass=("f_liq", "mean"), Expiry_pass=("f_dte", "mean"),
                                    Cost_hurdle_pass=("f_cost", "mean")).mul(100).round(1).reset_index()
        st.caption("Share of observations passing each filter (%)")
        table(pr)

    # ================================================================== 6. SIGNALS
    with t_sig:
        hd = sorted(Pf["date"].dt.date.unique())
        hdate = st.select_slider("Heatmap date", options=hd, value=hd[-1], key="hd")
        hm = Pf[Pf["date"].dt.date == hdate].pivot_table(index="pair", columns="exp_month", values="z")
        if not hm.empty:
            fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, zmin=-4, zmax=4,
                                       colorscale=[[0, GREEN], [0.5, "#16213A"], [1, RED]],
                                       text=np.round(hm.values, 1), texttemplate="%{text}", colorbar=dict(title="z")))
            fig.update_layout(title=f"Z-score map · pair × expiry month · {hdate:%d %b %Y} (green = A cheap, red = A rich)")
            show(fig, 320)
        stages = [("Observations with z", Pf["z"].notna()), (f"|z| ≥ {entry}", Pf["f_dev"])]
        cum = Pf["f_dev"].copy()
        for nm, col in (("+ Liquidity", "f_liq"), ("+ Expiry proximity", "f_dte"), ("+ Cost hurdle", "f_cost"),
                        ("+ Persistence", "f_per")):
            cum = cum & Pf[col]
            stages.append((nm, cum.copy()))
        fig = go.Figure(go.Funnel(y=[s[0] for s in stages], x=[int(s[1].sum()) for s in stages],
                                  marker=dict(color=[BLUE, "#5A6FA0", PURPLE, "#C77DFF", GOLD, GREEN])))
        fig.update_layout(title="Signal funnel: how deviations are filtered to explainable signals")
        show(fig, 360)
        hist = Pf[Pf["signal"].isin(["CHEAP", "EXPENSIVE"])].sort_values("date", ascending=False)
        st.markdown(f"#### Signal log ({len(hist)} signals)")
        f1, f2 = st.columns(2)
        fp = f1.multiselect("Pairs", sorted(hist["pair"].unique()), default=sorted(hist["pair"].unique()))
        fs = f2.multiselect("Direction", ["CHEAP", "EXPENSIVE"], default=["CHEAP", "EXPENSIVE"])
        lg = hist[hist["pair"].isin(fp) & hist["signal"].isin(fs)]
        out = lg[["date", "pair", "exp_month", "signal", "z", "pval", "spread", "mu_spread", "dev_rs", "rt_cost", "dte_min"]].copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out = out.round(3)
        table(out, 320)
        st.download_button("⬇️ Download signals (CSV)", out.to_csv(index=False).encode(), "goldx_signals.csv", "text/csv")
        if len(lg):
            st.markdown("##### Explain a signal")
            pick = st.selectbox("Pick", [f"{r['date']:%d %b %Y} · {r['pair']} · {r['exp_month']} · {r['signal']}"
                                         for _, r in lg.head(100).iterrows()])
            r = lg.head(100).iloc[[f"{x['date']:%d %b %Y} · {x['pair']} · {x['exp_month']} · {x['signal']}"
                                   for _, x in lg.head(100).iterrows()].index(pick)]
            st.markdown(f'<div class="gx-alert">{pill(r["signal"])} &nbsp; {explain(r, p)}</div>', unsafe_allow_html=True)

    # ================================================================== 7. WALK-FORWARD
    with t_wf:
        st.markdown("#### Walk-forward validation · cost-aware · no look-ahead")
        flow([("Train window", f"{train}d: pick entry z"), ("Test window", f"{test}d unseen: trade"),
              ("Roll forward", "repeat"), ("Costs", f"{fee + slip:.1f} bps/leg-side"), ("Fill", f"T+{delay} settlement")])
        if wf["folds"].empty:
            st.warning(f"Not enough history for walk-forward (need ≥ {train + 10} trading days of spreads).")
        else:
            net, tp = oos_st["net"], oos_st["pval"]
            if net > 0 and oos_st["trades"] >= 10 and (np.isnan(tp) or tp < 0.10):
                cls, msg = "v-good", "Edge survives out-of-sample after costs (preliminary evidence — keep monitoring)."
            elif net > 0:
                cls, msg = "v-mid", "Positive out-of-sample P&L but statistically weak (few trades / not significant)."
            else:
                cls, msg = "v-bad", ("Edge does NOT survive normalization, costs, liquidity and validation — reported as-is. "
                                     "GOLDX would stay quiet.")
            st.markdown(f'<div class="gx-verdict {cls}"><b>Verdict:</b> {msg}</div>', unsafe_allow_html=True)
            c = st.columns(6)
            card(c[0], "OOS net P&L", inr(net), f"gross {inr(oos_st['gross'])}", GREEN if net > 0 else RED)
            card(c[1], "Costs paid", inr(oos_st["cost"]), "fees + slippage")
            card(c[2], "Trades", oos_st["trades"], "" if np.isnan(oos_st["win"]) else f"win rate {oos_st['win'] * 100:.0f}%")
            card(c[3], "Sharpe (ann.)", "—" if np.isnan(oos_st["sharpe"]) else f"{oos_st['sharpe']:.2f}", "daily OOS")
            card(c[4], "Max drawdown", inr(oos_st["maxdd"]), "OOS")
            card(c[5], "t-test p-value", "—" if np.isnan(tp) else f"{tp:.3f}", "H₁: mean daily P&L > 0")
            st.write("")
            fig = go.Figure()
            if not oos_daily.empty:
                eq = oos_daily["net"].cumsum()
                fig.add_scatter(x=oos_daily.index, y=oos_daily["gross"].cumsum(), name="Gross", line=dict(color=BLUE, dash="dot"))
                fig.add_scatter(x=oos_daily.index, y=eq, name="Net (after costs)", line=dict(color=GOLD, width=3))
                fig.add_scatter(x=oos_daily.index, y=eq - eq.cummax(), name="Drawdown", fill="tozeroy",
                                line=dict(color=RED, width=1), fillcolor="rgba(231,76,60,0.2)")
            fig.update_layout(title=f"Out-of-sample equity (₹, {exposure} g per leg)")
            show(fig, 380)
            c1, c2 = st.columns(2)
            with c1:
                fs_ = wf["folds"]
                fig = go.Figure()
                fig.add_bar(x=fs_["Fold"], y=fs_["Train_net"], name="Train net", marker_color="#3B4A66")
                fig.add_bar(x=fs_["Fold"], y=fs_["Test_net"], name="Test net (unseen)", marker_color=GOLD)
                fig.update_layout(barmode="group", title="In-sample vs out-of-sample by fold (₹)")
                show(fig, 320)
            with c2:
                fig = go.Figure(go.Scatter(x=wf["sens"]["cost_multiple"], y=wf["sens"]["net"], mode="lines+markers",
                                           line=dict(color=PURPLE)))
                fig.add_hline(y=0, line_dash="dot", line_color=GREY)
                fig.update_layout(title="Cost sensitivity (full-sample, fixed params): net P&L vs cost multiple",
                                  xaxis_title="× assumed cost")
                show(fig, 320)
            st.markdown("##### Fold detail")
            table(wf["folds"])
            if len(oos_tr):
                fig = go.Figure(go.Histogram(x=oos_tr.values, nbinsx=30, marker_color=GOLD))
                fig.update_layout(title="Per-trade net P&L distribution (₹, OOS)")
                show(fig, 260)
            ov = st.columns(2)
            card(ov[0], "Full-sample (in-sample) net", inr(is_st["net"]), "optimistic: no walk-forward")
            card(ov[1], "Out-of-sample net", inr(oos_st["net"]), "honest estimate", GOLD)
            st.caption("Positions are decided at the settlement close, filled after the execution delay, forced flat before "
                       "expiry, charged fees + slippage on every leg change. Entry-z is chosen on the training window only; "
                       "if nothing is profitable in training, the model stands aside for that test window.")
            if len(wf["S_oos"]):
                st.download_button("⬇️ Download OOS trades log (CSV)", wf["S_oos"].to_csv(index=False).encode(),
                                   "goldx_oos_backtest.csv", "text/csv")

    # ================================================================== 8. ABOUT
    with t_about:
        st.markdown("#### The problem")
        st.info("MCX lists several gold futures for the same metal, but their prices are not directly comparable: contract "
                "size, purity, quotation basis, expiry and liquidity all differ. A raw price gap can be a specification, "
                "expiry or liquidity effect, not a genuine opportunity.")
        c = st.columns(3)
        card(c[0], "False signals", "Raw gaps mislead", "→ normalize first")
        card(c[1], "Expiry effects", "Artificial jumps", "→ track every contract by expiry")
        card(c[2], "Backtest risk", "Settlement ≠ fill", "→ costs + execution delay")
        st.markdown("#### Solution flow")
        flow([("MCX Bhavcopy", "raw data"), ("Contract Normalization", "unit · purity · quotation"),
              ("Relative Price Analysis", "spreads / ratios"), ("Anomaly Detection", "statistical deviation"),
              ("Liquidity + Expiry Filter", "volume · OI · expiry"), ("Walk-Forward Validation", "no look-ahead"),
              ("Explainable Signal", "actionable insight")])
        st.markdown("#### Why GOLDX")
        c = st.columns(5)
        for col, (a, b) in zip(c, [("Contract-aware", "individual expiries"), ("Normalized", "unit · quotation · purity"),
                                   ("No look-ahead", "walk-forward testing"), ("Cost-aware", "costs · liquidity · execution"),
                                   ("Quiet signals", "meaningful deviations only")]):
            card(col, a, "✔", b, GOLD)
        st.markdown("#### Target users & revenue model")
        c = st.columns(4)
        for col, (a, b) in zip(c, [("Proprietary traders", "contract-level relative value"),
                                   ("Commodity brokers", "monitoring & research tools"),
                                   ("Hedgers / treasury", "expiry & carry exposure"),
                                   ("Researchers / PMs", "normalized history + walk-forward")]):
            card(col, a, "", b)
        st.caption("B2B subscription (dashboard, alerts, history) → premium analytics (custom signals, backtesting) → "
                   "enterprise / API (broker integration, institutional dashboards).")
        st.markdown("#### Roadmap")
        c = st.columns(4)
        for col, (a, b) in zip(c, [("Phase 1 · Gold", "GOLDM · GOLDTEN · GOLDGUINEA · GOLDPETAL"),
                                   ("Phase 2 · Multi-commodity", "Silver · base metals · energy"),
                                   ("Phase 3 · Advanced", "adaptive models · regimes · liquidity-aware · better costs"),
                                   ("Phase 4 · Institutional", "trader dashboard · alerts · broker/API")]):
            card(col, a, "", b)
        st.markdown("#### Team TechNova")
        c = st.columns(3)
        card(c[0], "Wahaaj Ali", "", "Quantitative Analysis & ML")
        card(c[1], "Sweta Chauhan", "", "Data Engineering & Market Research")
        card(c[2], "Jatin Chandra", "", "Product & Visualization")
        st.markdown("#### Principles")
        st.markdown("- Contract-specific expiry tracking, never a continuous near-month series\n"
                    "- No look-ahead bias: baseline excludes today, fills occur after the decision\n"
                    "- Transaction costs and slippage on every trade\n"
                    "- Settlement ≠ guaranteed executable price\n"
                    "- If the edge disappears after normalization, costs, liquidity and validation — we report it")
        st.caption("Demo mode uses synthetic data. GOLDX is a research tool, not investment advice. "
                   "Stack: Python · Pandas · NumPy · SciPy · scikit-learn (optional) · Plotly · Streamlit · CSV/Parquet.")


main()