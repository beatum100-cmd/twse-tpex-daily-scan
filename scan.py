from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf


# -------------------------
# User settings (your rules)
# -------------------------
MIN_VOL_SHARES = 2_000_000
PRICE_MIN = 38.0
PRICE_MAX = 88.0
MIN_CHG_PCT = 0.10  # +0.10% and above (today vs prev close)

# MA30 v9 settings
MA_LEN = 30
SLOPE_LEN = 5
MIN_SLOPE_PCT = 0.20     # maSlopePct > 0.20
BUFFER_PCT = 0.10        # close > MA30 * (1 + 0.10%)
REQUIRE_NO_TOUCH = True  # low > MA30

# Runtime behavior
TAIPEI_TZ = ZoneInfo("Asia/Taipei")
LOOKBACK_DAYS_FOR_DATE = 14
RETRY_MINUTES = 40
RETRY_EVERY_SECONDS = 300  # 5 minutes

# -------------------------
# Data sources
# -------------------------
# TWSE: MI_INDEX daily close table (one day, whole market)
TWSE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"

# TPEx: daily close quotes
TPEX_URL = "https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php"


# -------------------------
# Helpers
# -------------------------
def _num(x):
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


def _tick_size(price: float) -> float:
    # Taiwan stock tick table (common rule-of-thumb)
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


def _is_4digit_stock(code: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", code or ""))


def _is_etf_like(code: str, name: str) -> bool:
    # Practical exclusion: most TW ETFs are 00xx, 00xxx range (especially 0050, 006xxx, 008xxx, 009xxx)
    # Plus keyword exclusion.
    try:
        n = int(code)
    except Exception:
        n = 0

    keywords = ["ETF", "ETN", "受益", "反向", "槓桿", "指數", "債", "存託", "信託", "期貨", "權證", "認購", "認售", "牛熊"]
    name_s = (name or "").upper()
    if any(k in (name or "") for k in keywords):
        return True
    if "ETF" in name_s or "ETN" in name_s:
        return True
    # heuristic: exclude codes < 1000 (almost never common stocks)
    if n < 1000:
        return True
    # also exclude 00xx/006xxx/008xxx/009xxx patterns that are overwhelmingly funds
    if code.startswith(("00", "006", "008", "009")):
        return True
    return False


@dataclass
class MarketDay:
    date: datetime  # Taipei date (00:00)
    twse: pd.DataFrame
    tpex: pd.DataFrame


def fetch_twse_day(dt: datetime) -> pd.DataFrame:
    ymd = dt.strftime("%Y%m%d")
    params = {"response": "json", "date": ymd, "type": "ALL"}
    r = requests.get(TWSE_URL, params=params, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    j = r.json()

    tables = j.get("tables") or []
    if not tables:
        return pd.DataFrame()

    # Find the table that looks like daily close quotes list
    picked = None
    for t in tables:
        fields = t.get("fields") or []
        data = t.get("data") or []
        if data and any("證券代號" in f for f in fields) and any("收盤價" in f for f in fields):
            picked = t
            break
    if not picked:
        return pd.DataFrame()

    df = pd.DataFrame(picked["data"], columns=picked["fields"])

    # Expected columns (Chinese)
    code_col = "證券代號"
    name_col = "證券名稱"
    vol_col = "成交股數"
    close_col = "收盤價"
    sign_col = "漲跌(+/-)"
    chg_col = "漲跌價差"
    high_col = "最高價"
    low_col = "最低價"
    open_col = "開盤價"

    for c in [code_col, name_col, vol_col, close_col, chg_col, high_col, low_col, open_col]:
        if c not in df.columns:
            # if format changes, return empty so retry/backoff can handle it
            return pd.DataFrame()

    df_out = pd.DataFrame()
    df_out["market"] = "TWSE"
    df_out["code"] = df[code_col].astype(str).str.strip()
    df_out["name"] = df[name_col].astype(str).str.strip()

    df_out["open"] = df[open_col].map(_num)
    df_out["high"] = df[high_col].map(_num)
    df_out["low"] = df[low_col].map(_num)
    df_out["close"] = df[close_col].map(_num)

    df_out["volume_shares"] = df[vol_col].map(_num)

    # Change sign and amount
    sign = df[sign_col].astype(str).str.strip() if sign_col in df.columns else ""
    chg_amt = df[chg_col].map(_num)
    # sign can be "+", "-", "X" etc; treat "-" as negative
    df_out["change_amt"] = chg_amt
    df_out.loc[sign == "-", "change_amt"] = -abs(df_out.loc[sign == "-", "change_amt"])

    # prev_close and change_pct
    df_out["prev_close"] = df_out["close"] - df_out["change_amt"]
    df_out["change_pct"] = (df_out["close"] / df_out["prev_close"] - 1.0) * 100.0

    return df_out


def fetch_tpex_day(dt: datetime) -> pd.DataFrame:
    # TPEx uses ROC date like 114/12/16
    roc_year = dt.year - 1911
    d_roc = f"{roc_year}/{dt.month:02d}/{dt.day:02d}"

    params = {"l": "zh-tw", "d": d_roc, "s": "0,asc,0"}
    r = requests.get(TPEX_URL, params=params, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    j = r.json()

    data = j.get("aaData") or []
    fields = j.get("aaData")  # not used

    if not data:
        return pd.DataFrame()

    # The TPEx table is positional; we map by commonly used positions.
    # Typical columns: code, name, close, change, open, high, low, volume (shares) ...
    # We will try to be defensive and only pull what we can.
    def safe_get(row, idx):
        try:
            return row[idx]
        except Exception:
            return None

    rows = []
    for row in data:
        code = str(safe_get(row, 0)).strip()
        name = str(safe_get(row, 1)).strip()
        close = _num(safe_get(row, 2))
        chg = _num(safe_get(row, 3))
        open_ = _num(safe_get(row, 4))
        high = _num(safe_get(row, 5))
        low = _num(safe_get(row, 6))
        vol = _num(safe_get(row, 8))  # often 成交股數 is around index 8
        rows.append([code, name, open_, high, low, close, chg, vol])

    df_out = pd.DataFrame(rows, columns=["code", "name", "open", "high", "low", "close", "change_amt", "volume_shares"])
    df_out.insert(0, "market", "TPEX")

    df_out["prev_close"] = df_out["close"] - df_out["change_amt"]
    df_out["change_pct"] = (df_out["close"] / df_out["prev_close"] - 1.0) * 100.0

    return df_out


def find_latest_common_day() -> MarketDay:
    now = datetime.now(TAIPEI_TZ)
    today = datetime(now.year, now.month, now.day, tzinfo=TAIPEI_TZ)

    for back in range(0, LOOKBACK_DAYS_FOR_DATE):
        d = today - timedelta(days=back)
        tw = fetch_twse_day(d)
        tp = fetch_tpex_day(d)

        if len(tw) > 0 and len(tp) > 0:
            return MarketDay(date=d, twse=tw, tpex=tp)

    raise RuntimeError("Could not find a common published day for TWSE and TPEx in lookback window.")


def apply_universe_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["code"].map(_is_4digit_stock)]
    df = df[df["code"].notna()]

    df = df[~df.apply(lambda r: _is_etf_like(r["code"], r["name"]), axis=1)]

    df = df.dropna(subset=["close", "volume_shares", "change_pct", "high", "low", "open", "prev_close"])
    df = df[
        (df["volume_shares"] >= MIN_VOL_SHARES) &
        (df["close"] >= PRICE_MIN) &
        (df["close"] <= PRICE_MAX) &
        (df["change_pct"] >= MIN_CHG_PCT)
    ]

    return df


def yfinance_symbol(code: str, market: str) -> str:
    # Yahoo Finance: TWSE = .TW, TPEx = .TWO
    return f"{code}.TW" if market == "TWSE" else f"{code}.TWO"


def compute_ma30_v9_candidates(universe: pd.DataFrame, trade_date: datetime) -> pd.DataFrame:
    if len(universe) == 0:
        return universe.assign(pass_ma30_v9=False)

    tickers = [yfinance_symbol(r.code, r.market) for r in universe.itertuples(index=False)]
    # Pull enough bars to compute MA30 and slope, plus two latest days
    # Use 60 calendar days to cover ~40 trading days
    start = (trade_date - timedelta(days=90)).date().isoformat()
    end = (trade_date + timedelta(days=1)).date().isoformat()

    # yfinance can download many tickers at once; split to be safer
    chunks = []
    CHUNK = 200
    for i in range(0, len(tickers), CHUNK):
        chunks.append(tickers[i:i + CHUNK])

    results = []
    for ch in chunks:
        data = yf.download(
            tickers=" ".join(ch),
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if data is None or len(data) == 0:
            continue

        for sym in ch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    dfp = data[sym].dropna()
                else:
                    # single ticker case
                    dfp = data.dropna()
                if len(dfp) < (MA_LEN + SLOPE_LEN + 2):
                    continue

                # Use last two trading days available in yf (not necessarily equal to trade_date if holiday)
                dfp = dfp.sort_index()
                # compute MA30 on Close
                dfp["MA30"] = dfp["Close"].rolling(MA_LEN).mean()
                dfp["SlopePct"] = (dfp["MA30"] - dfp["MA30"].shift(SLOPE_LEN)) / dfp["MA30"].shift(SLOPE_LEN) * 100.0

                # last day = D, prev day = D-1
                D = dfp.iloc[-1]
                Dm1 = dfp.iloc[-2]

                if pd.isna(D["MA30"]) or pd.isna(Dm1["MA30"]) or pd.isna(D["SlopePct"]):
                    continue

                slope_ok = D["SlopePct"] > MIN_SLOPE_PCT

                # bullCross on D-1: close crosses above MA30
                bull_prev = (Dm1["Close"] > Dm1["MA30"]) and (dfp.iloc[-3]["Close"] <= dfp.iloc[-3]["MA30"])

                # confirm day D conditions
                ph_ok = D["Close"] > Dm1["High"]
                buf_ok = D["Close"] > D["MA30"] * (1.0 + BUFFER_PCT / 100.0)
                no_touch_ok = (not REQUIRE_NO_TOUCH) or (D["Low"] > D["MA30"])

                pass_v9 = bool(bull_prev and ph_ok and buf_ok and no_touch_ok and slope_ok)

                results.append(
                    {
                        "yf_symbol": sym,
                        "last_bar_date": str(dfp.index[-1].date()),
                        "MA30": float(D["MA30"]),
                        "SlopePct": float(D["SlopePct"]),
                        "bullPrev": pass_v9 and True or bool(bull_prev),
                        "phOK": bool(ph_ok),
                        "bufOK": bool(buf_ok),
                        "noTouchOK": bool(no_touch_ok),
                        "slopeOK": bool(slope_ok),
                        "pass_ma30_v9": pass_v9,
                    }
                )
            except Exception:
                continue

    feat = pd.DataFrame(results)
    if len(feat) == 0:
        return universe.assign(pass_ma30_v9=False)

    out = universe.copy()
    out["yf_symbol"] = out.apply(lambda r: yfinance_symbol(r["code"], r["market"]), axis=1)
    out = out.merge(feat, on="yf_symbol", how="left")
    out["pass_ma30_v9"] = out["pass_ma30_v9"].fillna(False)

    return out


def main():
    start_ts = datetime.now(TAIPEI_TZ)

    # Retry loop (handles delayed publishing)
    deadline = start_ts + timedelta(minutes=RETRY_MINUTES)
    last_err = None

    while datetime.now(TAIPEI_TZ) <= deadline:
        try:
            day = find_latest_common_day()
            trade_date = day.date

            all_df = pd.concat([day.twse, day.tpex], ignore_index=True)

            uni = apply_universe_filters(all_df)

            # MA30 v9 candidates
            scored = compute_ma30_v9_candidates(uni, trade_date)

            # Stop-buy suggestion for next session
            scored["suggest_stop_buy"] = scored["high"].map(_round_up_to_tick)

            # Rank: first by MA30 pass, then by change_pct and volume
            scored["rank_key_pass"] = scored["pass_ma30_v9"].astype(int)
            scored = scored.sort_values(["rank_key_pass", "change_pct", "volume_shares"], ascending=[False, False, False])

            candidates = scored[scored["pass_ma30_v9"]].copy()

            meta = pd.DataFrame(
                [
                    ["run_timestamp_taipei", start_ts.strftime("%Y-%m-%d %H:%M:%S")],
                    ["published_trade_date_taipei", trade_date.strftime("%Y-%m-%d")],
                    ["universe_rows_after_filters", len(uni)],
                    ["candidates_rows_ma30_v9", len(candidates)],
                    ["min_volume_shares", MIN_VOL_SHARES],
                    ["price_range", f"{PRICE_MIN} to {PRICE_MAX}"],
                    ["min_change_pct", MIN_CHG_PCT],
                    ["ma_len", MA_LEN],
                    ["slope_len", SLOPE_LEN],
                    ["min_slope_pct", MIN_SLOPE_PCT],
                    ["buffer_pct", BUFFER_PCT],
                    ["require_no_touch", REQUIRE_NO_TOUCH],
                ],
                columns=["key", "value"],
            )

            os.makedirs("output", exist_ok=True)
            out_date = trade_date.strftime("%Y-%m-%d")
            out_path = f"output/scan_{out_date}.xlsx"
            latest_path = "output/latest.xlsx"

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

            print(f"Wrote: {out_path}")
            print(f"Wrote: {latest_path}")
            return

        except Exception as e:
            last_err = str(e)
            print(f"Not ready yet, will retry. Reason: {last_err}")
            time.sleep(RETRY_EVERY_SECONDS)

    raise RuntimeError(f"Timed out waiting for TWSE+TPEx common publish. Last error: {last_err}")


if __name__ == "__main__":
    main()
