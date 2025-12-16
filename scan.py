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
# Your universe filters
# -------------------------
MIN_VOL_SHARES = 2_000_000
PRICE_MIN = 38.0
PRICE_MAX = 88.0
MIN_CHG_PCT = 0.10  # +0.10% and above (today vs prev close)

# MA30 v9 settings (daily)
MA_LEN = 30
SLOPE_LEN = 5
MIN_SLOPE_PCT = 0.20
BUFFER_PCT = 0.10
REQUIRE_NO_TOUCH = True

# Runtime behavior
TAIPEI_TZ = ZoneInfo("Asia/Taipei")
LOOKBACK_DAYS_FOR_DATE = 14
RETRY_MINUTES = 40
RETRY_EVERY_SECONDS = 300  # 5 minutes

# TPEx daily close quotes (date-based)
TPEX_URL = "https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php"


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
    # TPEx has fewer ETFs than TWSE, but still exclude fund-like instruments by name.
    keywords = ["ETF", "ETN", "受益", "反向", "槓桿", "指數", "債", "存託", "信託", "期貨", "權證", "認購", "認售", "牛熊"]
    n = (name or "")
    if any(k in n for k in keywords):
        return True
    return False


@dataclass
class MarketDay:
    date: datetime  # Taipei date (00:00)
    tpex: pd.DataFrame


def fetch_tpex_day(dt: datetime) -> pd.DataFrame:
    # TPEx uses ROC date like 114/12/16
    roc_year = dt.year - 1911
    d_roc = f"{roc_year}/{dt.month:02d}/{dt.day:02d}"

    params = {"l": "zh-tw", "d": d_roc, "s": "0,asc,0"}
    try:
        r = requests.get(TPEX_URL, params=params, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        j = r.json()
    except Exception:
        return pd.DataFrame()

    data = j.get("aaData") or []
    if not data:
        return pd.DataFrame()

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

        # volume position can vary; try a few common indices and pick the first numeric
        vol = None
        for idx in [8, 7, 9, 10]:
            v = _num(safe_get(row, idx))
            if v is not None and v > 0:
                vol = v
                break

        rows.append([code, name, open_, high, low, close, chg, vol])

    df = pd.DataFrame(rows, columns=["code", "name", "open", "high", "low", "close", "change_amt", "volume_shares"])
    df.insert(0, "market", "TPEX")

    df = df.dropna(subset=["close", "change_amt"])
    df["prev_close"] = df["close"] - df["change_amt"]
    df["change_pct"] = (df["close"] / df["prev_close"] - 1.0) * 100.0

    return df


def find_latest_tpex_day() -> MarketDay:
    now = datetime.now(TAIPEI_TZ)
    today = datetime(now.year, now.month, now.day, tzinfo=TAIPEI_TZ)

    for back in range(0, LOOKBACK_DAYS_FOR_DATE):
        d = today - timedelta(days=back)
        tp = fetch_tpex_day(d)
        if len(tp) > 0:
            return MarketDay(date=d, tpex=tp)

    raise RuntimeError("Could not find a published TPEx day in lookback window.")


def apply_universe_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["code"].map(_is_4digit_stock)]
    df = df[~df.apply(lambda r: _is_etf_like(r["code"], r["name"]), axis=1)]

    df = df.dropna(subset=["close", "volume_shares", "change_pct", "high", "low", "open", "prev_close"])
    df = df[
        (df["volume_shares"] >= MIN_VOL_SHARES) &
        (df["close"] >= PRICE_MIN) &
        (df["close"] <= PRICE_MAX) &
        (df["change_pct"] >= MIN_CHG_PCT)
    ]
    return df


def compute_ma30_v9_candidates(universe: pd.DataFrame, trade_date: datetime) -> pd.DataFrame:
    if len(universe) == 0:
        return universe.assign(pass_ma30_v9=False)

    universe = universe.copy()
    universe["yf_symbol"] = universe["code"].astype(str).str.strip() + ".TWO"

    tickers = universe["yf_symbol"].unique().tolist()

    start = (trade_date - timedelta(days=90)).date().isoformat()
    end = (trade_date + timedelta(days=1)).date().isoformat()

    results = []
    CHUNK = 200
    for i in range(0, len(tickers), CHUNK):
        ch = tickers[i:i + CHUNK]
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
                dfp = data[sym].dropna() if isinstance(data.columns, pd.MultiIndex) else data.dropna()
                if len(dfp) < (MA_LEN + SLOPE_LEN + 3):
                    continue

                dfp = dfp.sort_index()
                dfp["MA30"] = dfp["Close"].rolling(MA_LEN).mean()
                dfp["SlopePct"] = (dfp["MA30"] - dfp["MA30"].shift(SLOPE_LEN)) / dfp["MA30"].shift(SLOPE_LEN) * 100.0

                D = dfp.iloc[-1]
                Dm1 = dfp.iloc[-2]
                Dm2 = dfp.iloc[-3]

                if pd.isna(D["MA30"]) or pd.isna(Dm1["MA30"]) or pd.isna(D["SlopePct"]):
                    continue

                slope_ok = D["SlopePct"] > MIN_SLOPE_PCT

                # bullCross on D-1: close crossed above MA30
                bull_prev = (Dm1["Close"] > Dm1["MA30"]) and (Dm2["Close"] <= Dm2["MA30"])

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
                        "bullPrev": bool(bull_prev),
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
    out = universe.merge(feat, on="yf_symbol", how="left")
    out["pass_ma30_v9"] = out["pass_ma30_v9"].fillna(False)

    return out


def main():
    start_ts = datetime.now(TAIPEI_TZ)
    deadline = start_ts + timedelta(minutes=RETRY_MINUTES)
    last_err = None

    while datetime.now(TAIPEI_TZ) <= deadline:
        try:
            day = find_latest_tpex_day()
            trade_date = day.date

            all_df = day.tpex.copy()
            uni = apply_universe_filters(all_df)
            scored = compute_ma30_v9_candidates(uni, trade_date)

            scored["suggest_stop_buy"] = scored["high"].map(_round_up_to_tick)

            scored = scored.sort_values(
                ["pass_ma30_v9", "change_pct", "volume_shares"],
                ascending=[False, False, False],
            )
            candidates = scored[scored["pass_ma30_v9"]].copy()

            meta = pd.DataFrame(
                [
                    ["run_timestamp_taipei", start_ts.strftime("%Y-%m-%d %H:%M:%S")],
                    ["published_trade_date_taipei", trade_date.strftime("%Y-%m-%d")],
                    ["market_scope", "TPEX only"],
                    ["universe_rows_after_filters", len(uni)],
                    ["candidates_rows_ma30_v9", len(candidates)],
                ],
                columns=["key", "value"],
            )

            os.makedirs("output", exist_ok=True)
            out_date = trade_date.strftime("%Y-%m-%d")
            out_path = f"output/tpex_scan_{out_date}.xlsx"
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

    raise RuntimeError(f"Timed out waiting for TPEx publish. Last error: {last_err}")


if __name__ == "__main__":
    main()
