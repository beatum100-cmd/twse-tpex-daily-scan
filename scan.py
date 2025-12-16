from __future__ import annotations

import os
import re
import time
import signal
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import requests
import yfinance as yf


# =========================================================
# USER RULES (your filters)
# =========================================================
MIN_VOL_SHARES = 2_000_000
PRICE_MIN = 38.0
PRICE_MAX = 88.0
MIN_CHG_PCT = 0.10  # +0.10% and above vs previous close

# MA30 Strategy (v9) parameters
MA_LEN = 30
SLOPE_LEN = 5
MIN_SLOPE_PCT = 0.20     # maSlopePct > 0.20
BUFFER_PCT = 0.10        # close > MA30 * (1 + 0.10%)
REQUIRE_NO_TOUCH = True  # low > MA30

# Performance knobs
MAX_TICKERS_FOR_MA = 500          # cap for yfinance to avoid long runs
YF_CHUNK_SIZE = 30               # small chunks to reduce hang risk
YF_DOWNLOAD_TIMEOUT_SEC = 120    # per yfinance chunk timeout
YF_MAX_RETRIES_PER_CHUNK = 2

# TPEx publish detection / retries
TAIPEI_TZ = ZoneInfo("Asia/Taipei")
LOOKBACK_DAYS_FOR_PUBLISH = 14
WAIT_FOR_PUBLISH_MINUTES = 40
RETRY_EVERY_SECONDS = 300  # 5 minutes

# Output
OUT_DIR = "output"


# =========================================================
# DATA SOURCE (TPEx date-based close quote endpoint)
# =========================================================
TPEX_URL = "https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php"


# =========================================================
# Helpers
# =========================================================
def _num(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s in {"", "--", "None", "null", "NULL", "NaN"}:
        return None
    s = s.replace(",", "")
    s = re.sub(r"[^\d\.\-\+]", "", s)
    try:
        return float(s)
    except Exception:
        return None


def _is_4digit_stock(code: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", (code or "").strip()))


def _tick_size(price: float) -> float:
    # Common TW stock tick table
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.10
    if price < 500:
        return 0.50
    if price < 1000:
        return 1.0
    return 5.0


def _round_up_to_tick(px: float) -> float:
    t = _tick_size(px)
    return (int(px / t + 0.999999999) * t)


def _is_excluded_instrument(code: str, name: str) -> bool:
    # You asked: exclude ETFs/warrants
    # On TPEx, easiest reliable method is name keywords
    kw = [
        "ETF", "ETN", "受益", "反向", "槓桿", "指數", "債", "存託", "信託", "期貨",
        "權證", "認購", "認售", "牛熊"
    ]
    n = (name or "")
    u = n.upper()
    if "ETF" in u or "ETN" in u:
        return True
    for k in kw:
        if k in n:
            return True
    return False


def _tpex_roc_date(dt: datetime) -> str:
    # ROC date format: 114/12/16
    roc_year = dt.year - 1911
    return f"{roc_year}/{dt.month:02d}/{dt.day:02d}"


def _safe_get(row: list, idx: int):
    try:
        return row[idx]
    except Exception:
        return None


def _request_json_with_retries(url: str, params: Dict[str, Any], tries: int = 3) -> Optional[Dict[str, Any]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60, headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = str(e)
            time.sleep(1 + i * 2)
    print(f"TPEx request failed after retries. Last error: {last_err}", flush=True)
    return None


@dataclass
class PublishDay:
    date: datetime
    df: pd.DataFrame


def fetch_tpex_day(dt: datetime) -> pd.DataFrame:
    d_roc = _tpex_roc_date(dt)
    params = {"l": "zh-tw", "d": d_roc, "s": "0,asc,0"}

    j = _request_json_with_retries(TPEX_URL, params=params, tries=3)
    if not j:
        return pd.DataFrame()

    data = j.get("aaData") or []
    if not data:
        return pd.DataFrame()

    rows = []
    for row in data:
        code = str(_safe_get(row, 0)).strip()
        name = str(_safe_get(row, 1)).strip()

        close = _num(_safe_get(row, 2))
        chg_amt = _num(_safe_get(row, 3))
        open_ = _num(_safe_get(row, 4))
        high = _num(_safe_get(row, 5))
        low = _num(_safe_get(row, 6))

        # Volume index can vary, try several common positions
        vol = None
        for idx in [8, 7, 9, 10, 11]:
            v = _num(_safe_get(row, idx))
            if v is not None and v > 0:
                vol = v
                break

        rows.append([code, name, open_, high, low, close, chg_amt, vol])

    df = pd.DataFrame(
        rows,
        columns=["code", "name", "open", "high", "low", "close", "change_amt", "volume_shares"]
    )
    df.insert(0, "market", "TPEX")

    df = df.dropna(subset=["code", "name", "close", "change_amt"])
    df["prev_close"] = df["close"] - df["change_amt"]
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["prev_close"])
    df["change_pct"] = (df["close"] / df["prev_close"] - 1.0) * 100.0

    return df


def find_latest_published_tpex_day() -> PublishDay:
    now = datetime.now(TAIPEI_TZ)
    today = datetime(now.year, now.month, now.day, tzinfo=TAIPEI_TZ)

    for back in range(LOOKBACK_DAYS_FOR_PUBLISH):
        d = today - timedelta(days=back)
        df = fetch_tpex_day(d)
        if len(df) > 0:
            return PublishDay(date=d, df=df)

    raise RuntimeError("Could not find a published TPEx day in lookback window.")


def apply_universe_filters(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x[x["code"].map(_is_4digit_stock)]
    x = x[~x.apply(lambda r: _is_excluded_instrument(r["code"], r["name"]), axis=1)]

    x = x.dropna(subset=["close", "high", "low", "open", "volume_shares", "change_pct", "prev_close"])
    x = x[
        (x["volume_shares"] >= MIN_VOL_SHARES) &
        (x["close"] >= PRICE_MIN) &
        (x["close"] <= PRICE_MAX) &
        (x["change_pct"] >= MIN_CHG_PCT)
    ]
    return x


# =========================================================
# yfinance downloading with timeout and safe fallback
# =========================================================
class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout("yfinance download timed out")


def _yf_download_chunk(symbols: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
    if not symbols:
        return None

    old = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        signal.alarm(YF_DOWNLOAD_TIMEOUT_SEC)
        df = yf.download(
            tickers=" ".join(symbols),
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads=False,     # important for stability
            progress=False,
        )
        return df if df is not None and len(df) > 0 else None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _download_history_safe(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Returns dict: symbol -> OHLCV dataframe
    Uses chunk download, and if a chunk fails, splits it until it works.
    """
    out: Dict[str, pd.DataFrame] = {}

    def handle_chunk(chunk: List[str], depth: int = 0):
        if not chunk:
            return

        for attempt in range(YF_MAX_RETRIES_PER_CHUNK + 1):
            try:
                df = _yf_download_chunk(chunk, start, end)
                if df is None:
                    raise RuntimeError("Empty yfinance response")
                # MultiIndex columns expected for multi-ticker; handle single too
                if isinstance(df.columns, pd.MultiIndex):
                    for sym in chunk:
                        try:
                            sub = df[sym].dropna()
                            if len(sub) > 0:
                                out[sym] = sub
                        except Exception:
                            continue
                else:
                    # single ticker case
                    out[chunk[0]] = df.dropna()
                return
            except Exception as e:
                err = str(e)
                print(f"yfinance chunk failed (n={len(chunk)}), attempt {attempt+1}: {err}", flush=True)
                time.sleep(2 + attempt * 2)

        # If still failing, split the chunk
        if len(chunk) == 1:
            print(f"Skipping {chunk[0]} after repeated failures.", flush=True)
            return
        mid = len(chunk) // 2
        handle_chunk(chunk[:mid], depth + 1)
        handle_chunk(chunk[mid:], depth + 1)

    total = len(symbols)
    for i in range(0, total, YF_CHUNK_SIZE):
        chunk = symbols[i:i + YF_CHUNK_SIZE]
        print(f"Downloading history chunk {(i // YF_CHUNK_SIZE) + 1} / {((total + YF_CHUNK_SIZE - 1) // YF_CHUNK_SIZE)} "
              f"({len(chunk)} tickers)", flush=True)
        handle_chunk(chunk)

    return out


# =========================================================
# MA30 v9 logic (daily)
# =========================================================
def compute_ma30_v9_features(
    history: pd.DataFrame,
    trade_date: datetime,
) -> Optional[Dict[str, Any]]:
    """
    history: dataframe with columns Open, High, Low, Close, Volume indexed by datetime
    Use bars up to trade_date (Taipei date).
    """
    if history is None or len(history) < (MA_LEN + SLOPE_LEN + 5):
        return None

    dfp = history.sort_index().copy()
    # Align to trade_date (avoid using a later bar if yahoo already has it)
    td = trade_date.date()
    dfp = dfp[dfp.index.date <= td]
    if len(dfp) < (MA_LEN + SLOPE_LEN + 5):
        return None

    dfp["MA30"] = dfp["Close"].rolling(MA_LEN).mean()
    dfp["SlopePct"] = (dfp["MA30"] - dfp["MA30"].shift(SLOPE_LEN)) / dfp["MA30"].shift(SLOPE_LEN) * 100.0

    # Need last 3 bars: D-2, D-1, D
    if len(dfp) < 3:
        return None
    D = dfp.iloc[-1]
    Dm1 = dfp.iloc[-2]
    Dm2 = dfp.iloc[-3]

    if pd.isna(D["MA30"]) or pd.isna(Dm1["MA30"]) or pd.isna(D["SlopePct"]) or pd.isna(Dm2["MA30"]):
        return None

    slope_ok = float(D["SlopePct"]) > MIN_SLOPE_PCT

    # bullPrev: crossover happened on D-1
    # Equivalent to: close(D-1) > ma(D-1) and close(D-2) <= ma(D-2)
    bull_prev = (Dm1["Close"] > Dm1["MA30"]) and (Dm2["Close"] <= Dm2["MA30"])

    # confirmOK on day D
    ph_ok = D["Close"] > Dm1["High"]
    buf_ok = D["Close"] > D["MA30"] * (1.0 + BUFFER_PCT / 100.0)
    no_touch_ok = (not REQUIRE_NO_TOUCH) or (D["Low"] > D["MA30"])

    pass_v9 = bool(bull_prev and ph_ok and buf_ok and no_touch_ok and slope_ok)

    return {
        "last_bar_date": str(dfp.index[-1].date()),
        "MA30": float(D["MA30"]),
        "SlopePct": float(D["SlopePct"]),
        "bullPrev": bool(bull_prev),
        "phOK": bool(ph_ok),
        "bufOK": bool(buf_ok),
        "noTouchOK": bool(no_touch_ok),
        "slopeOK": bool(slope_ok),
        "pass_ma30_v9": pass_v9,
    }


def run_scan_once() -> Tuple[datetime, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pub = find_latest_published_tpex_day()
    trade_date = pub.date
    all_df = pub.df.copy()

    uni = apply_universe_filters(all_df)
    print(f"Published TPEx date: {trade_date.strftime('%Y-%m-%d')}", flush=True)
    print(f"Universe after filters (before MA cap): {len(uni)}", flush=True)

    # Cap to keep yfinance fast
    uni = uni.sort_values("volume_shares", ascending=False).head(MAX_TICKERS_FOR_MA).copy()
    print(f"Universe sent to MA30 check (capped): {len(uni)}", flush=True)

    # Prepare Yahoo symbols
    uni["yf_symbol"] = uni["code"].astype(str).str.strip() + ".TWO"
    symbols = uni["yf_symbol"].unique().tolist()

    start = (trade_date - timedelta(days=120)).date().isoformat()
    end = (trade_date + timedelta(days=1)).date().isoformat()

    histories = _download_history_safe(symbols, start, end)

    feats = []
    for r in uni.itertuples(index=False):
        sym = r.yf_symbol
        hist = histories.get(sym)
        f = compute_ma30_v9_features(hist, trade_date)
        if f is None:
            f = {
                "last_bar_date": None,
                "MA30": None,
                "SlopePct": None,
                "bullPrev": None,
                "phOK": None,
                "bufOK": None,
                "noTouchOK": None,
                "slopeOK": None,
                "pass_ma30_v9": False,
            }
        f["yf_symbol"] = sym
        feats.append(f)

    feat_df = pd.DataFrame(feats)

    scored = uni.merge(feat_df, on="yf_symbol", how="left")
    scored["pass_ma30_v9"] = scored["pass_ma30_v9"].fillna(False)

    # Suggested stop-buy: today high rounded up to tick
    scored["suggest_stop_buy"] = scored["high"].map(_round_up_to_tick)

    # Ranking: MA pass first, then change%, then volume
    scored = scored.sort_values(
        ["pass_ma30_v9", "change_pct", "volume_shares"],
        ascending=[False, False, False],
    )

    candidates = scored[scored["pass_ma30_v9"]].copy()
    print(f"Candidates (MA30 v9): {len(candidates)}", flush=True)

    return trade_date, all_df, scored, candidates


def write_excel(trade_date: datetime, all_df: pd.DataFrame, scored: pd.DataFrame, candidates: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_date = trade_date.strftime("%Y-%m-%d")
    out_path = os.path.join(OUT_DIR, f"tpex_scan_{out_date}.xlsx")
    latest_path = os.path.join(OUT_DIR, "latest.xlsx")

    meta = pd.DataFrame(
        [
            ["run_timestamp_taipei", datetime.now(TAIPEI_TZ).strftime("%Y-%m-%d %H:%M:%S")],
            ["published_trade_date_taipei", out_date],
            ["market_scope", "TPEX only"],
            ["min_volume_shares", MIN_VOL_SHARES],
            ["price_range", f"{PRICE_MIN} to {PRICE_MAX}"],
            ["min_change_pct", MIN_CHG_PCT],
            ["ma_len", MA_LEN],
            ["slope_len", SLOPE_LEN],
            ["min_slope_pct", MIN_SLOPE_PCT],
            ["buffer_pct", BUFFER_PCT],
            ["require_no_touch", REQUIRE_NO_TOUCH],
            ["max_tickers_for_ma", MAX_TICKERS_FOR_MA],
        ],
        columns=["key", "value"],
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        meta.to_excel(w, index=False, sheet_name="Meta")
        all_df.to_excel(w, index=False, sheet_name="AllRaw")
        scored.to_excel(w, index=False, sheet_name="UniverseFiltered")
        candidates.to_excel(w, index=False, sheet_name="Candidates")

    with pd.ExcelWriter(latest_path, engine="openpyxl") as w:
        meta.to_excel(w, index=False, sheet_name="Meta")
        all_df.to_excel(w, index=False, sheet_name="AllRaw")
        scored.to_excel(w, index=False, sheet_name="UniverseFiltered")
        candidates.to_excel(w, index=False, sheet_name="Candidates")

    print(f"Wrote: {out_path}", flush=True)
    print(f"Wrote: {latest_path}", flush=True)


def main():
    print("Starting TPEx scan...", flush=True)

    start_ts = datetime.now(TAIPEI_TZ)
    deadline = start_ts + timedelta(minutes=WAIT_FOR_PUBLISH_MINUTES)
    last_err = None

    while datetime.now(TAIPEI_TZ) <= deadline:
        try:
            trade_date, all_df, scored, candidates = run_scan_once()
            write_excel(trade_date, all_df, scored, candidates)
            return
        except Exception as e:
            last_err = str(e)
            print(f"Not ready yet, will retry. Reason: {last_err}", flush=True)
            time.sleep(RETRY_EVERY_SECONDS)

    raise RuntimeError(f"Timed out waiting for TPEx publish. Last error: {last_err}")


if __name__ == "__main__":
    main()
