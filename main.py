import os
import json
import time
import logging
from datetime import datetime, timezone

import requests
import pandas as pd

# ==========================================================
# BTCUSDT BITGET USDT-M FUTURES SIGNAL BOT
# Timeframes: 4H + 1H + 15M + 5M
#
# IMPORTANT:
# - Signals only; this script does NOT place/manage live orders.
# - Standard FINAL signal requires BOTH:
#       15M BOS in signal direction AND 5M confirmation.
# - RSI exhaustion, volume, momentum and higher-timeframe filters
#   are used to reduce weak/chasing entries.
# - FVG / order block are optional context, never compulsory.
# - Fast Momentum monitors 15M, 30M, 1H, 2H and 3H moves.
# - Early / Fast warnings are NOT entries.
#
# Required GitHub Secrets:
#   TELEGRAM_TOKEN, CHAT_ID
#
# Optional GitHub Variables:
#   MIN_SCORE=7
#   EARLY_SCORE=4
#   MIN_VOLUME_RATIO=1.15
#   FAST_VOLUME_MULT=1.30
#   FAST_ADX_MIN=20
#   FAST_MAX_EXTENSION_ATR=3.0
# ==========================================================

BASE_URL = "https://api.bitget.com"
SYMBOL = "BTCUSDT"
PRODUCT_TYPE = "USDT-FUTURES"
STATE_FILE = "bot_state.json"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

MIN_SCORE = int(os.getenv("MIN_SCORE", "7"))
EARLY_SCORE = int(os.getenv("EARLY_SCORE", "4"))
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", "1.15"))
FAST_VOLUME_MULT = float(os.getenv("FAST_VOLUME_MULT", "1.30"))
FAST_ADX_MIN = float(os.getenv("FAST_ADX_MIN", "20"))
FAST_MAX_EXTENSION_ATR = float(os.getenv("FAST_MAX_EXTENSION_ATR", "3.0"))

REQUEST_TIMEOUT = 20
CANDLE_LIMIT = 250

# Multi-window impulse thresholds. These are starting thresholds,
# not guarantees. Each row: label, number of closed 5M candles,
# minimum move percentage, minimum absolute USD move.
FAST_WINDOWS = [
    ("15M", 3, 0.80, 700),
    ("30M", 6, 1.20, 1000),
    ("1H", 12, 1.60, 1300),
    ("2H", 24, 2.20, 1800),
    ("3H", 36, 2.80, 2200),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("BTC-BOT")


# ==========================================================
# TELEGRAM
# ==========================================================

def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise RuntimeError("Telegram secrets missing: TELEGRAM_TOKEN / CHAT_ID")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": CHAT_ID, "text": message},
        timeout=REQUEST_TIMEOUT
    )
    log.info("Telegram HTTP status: %s", response.status_code)
    response.raise_for_status()

    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegram error: {result}")

    log.info("TELEGRAM MESSAGE SENT SUCCESSFULLY")


# ==========================================================
# INDICATORS
# ==========================================================

def indicators(df):
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    # RSI (Wilder smoothing)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    # True range / ATR
    previous_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)
    df["tr"] = tr
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # Volume baseline excludes current candle to avoid diluting the ratio.
    df["vol_avg"] = df["volume"].rolling(20).mean().shift(1)
    df["vol_ratio"] = df["volume"] / df["vol_avg"].replace(0, float("nan"))

    # Directional movement / ADX
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        [u if (u > d and u > 0) else 0 for u, d in zip(up_move, down_move)],
        index=df.index
    )
    minus_dm = pd.Series(
        [d if (d > u and d > 0) else 0 for u, d in zip(up_move, down_move)],
        index=df.index
    )

    atr_smooth = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_smooth.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_smooth.replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return (
        df.replace([float("inf"), float("-inf")], float("nan"))
          .dropna()
          .reset_index(drop=True)
    )


# ==========================================================
# CANDLE DATA
# ==========================================================

def fetch_candles(granularity, limit=CANDLE_LIMIT):
    url = BASE_URL + "/api/v2/mix/market/candles"
    params = {
        "symbol": SYMBOL,
        "productType": PRODUCT_TYPE,
        "granularity": granularity,
        "limit": str(limit)
    }

    intervals = {
        "5m": 300000,
        "15m": 900000,
        "1H": 3600000,
        "4H": 14400000
    }
    if granularity not in intervals:
        raise ValueError(f"Unsupported granularity: {granularity}")

    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()

            if payload.get("code") != "00000":
                raise RuntimeError(payload.get("msg", "Bitget API error"))

            rows = payload.get("data", [])
            if not rows:
                raise RuntimeError("Empty candle data")

            clean_rows = [
                list(row[:7])
                for row in rows
                if isinstance(row, (list, tuple)) and len(row) >= 7
            ]

            df = pd.DataFrame(
                clean_rows,
                columns=[
                    "timestamp", "open", "high", "low",
                    "close", "volume", "turnover"
                ]
            )

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = (
                df.dropna()
                  .sort_values("timestamp")
                  .drop_duplicates("timestamp")
                  .reset_index(drop=True)
            )

            # Closed candles only. 3-second safety buffer.
            interval = intervals[granularity]
            now_ms = int(time.time() * 1000)
            df = df[
                df["timestamp"] + interval <= now_ms - 3000
            ].reset_index(drop=True)

            if len(df) < 205:
                raise RuntimeError(f"Insufficient closed candles: {len(df)}")

            log.info("%s candles loaded: %s", granularity, len(df))
            return indicators(df)

        except Exception as error:
            last_error = error
            log.warning("%s attempt %s failed: %s", granularity, attempt + 1, error)
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"{granularity} data failed: {last_error}")


# ==========================================================
# MARKET STRUCTURE
# ==========================================================

def market_structure(df):
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    swing_highs = []
    swing_lows = []

    # Confirmed pivots only; three candles on the right are required.
    for i in range(3, len(df) - 3):
        h = highs[i - 3:i + 4]
        l = lows[i - 3:i + 4]

        if highs[i] == max(h) and list(h).count(highs[i]) == 1:
            swing_highs.append(float(highs[i]))
        if lows[i] == min(l) and list(l).count(lows[i]) == 1:
            swing_lows.append(float(lows[i]))

    last = df.iloc[-1]
    previous = df.iloc[-2]

    swing_high = swing_highs[-1] if swing_highs else float(df.high.iloc[-30:-3].max())
    swing_low = swing_lows[-1] if swing_lows else float(df.low.iloc[-30:-3].min())

    return {
        "bos_up": bool(last.close > swing_high and previous.close <= swing_high),
        "bos_down": bool(last.close < swing_low and previous.close >= swing_low),
        "swing_high": swing_high,
        "swing_low": swing_low
    }


# ==========================================================
# TIMEFRAME CONFIRMATION
# ==========================================================

def timeframe_confirmation(df, lookback=5):
    last = df.iloc[-1]
    previous_high = float(df.high.iloc[-lookback - 1:-1].max())
    previous_low = float(df.low.iloc[-lookback - 1:-1].min())

    # Require directional candle, EMA alignment, RSI and volume.
    candle_range = max(float(last.high - last.low), 1e-9)
    body_ratio = abs(float(last.close - last.open)) / candle_range

    long_confirm = (
        last.close > previous_high
        and last.close > last.ema9 > last.ema21
        and last.rsi >= 52
        and last.close > last.open
        and body_ratio >= 0.45
    )

    short_confirm = (
        last.close < previous_low
        and last.close < last.ema9 < last.ema21
        and last.rsi <= 48
        and last.close < last.open
        and body_ratio >= 0.45
    )

    return {
        "long": bool(long_confirm),
        "short": bool(short_confirm),
        "price": float(last.close),
        "rsi": float(last.rsi),
        "timestamp": int(last.timestamp),
        "ema9": float(last.ema9),
        "ema21": float(last.ema21),
        "vol_ratio": float(last.vol_ratio),
        "adx": float(last.adx),
        "plus_di": float(last.plus_di),
        "minus_di": float(last.minus_di),
        "body_ratio": body_ratio,
        "break_high": previous_high,
        "break_low": previous_low
    }


# ==========================================================
# OPTIONAL CONFLUENCE
# ==========================================================

def order_block(df, direction):
    start = max(2, len(df) - 18)
    for i in range(len(df) - 2, start - 1, -1):
        candle = df.iloc[i]
        if direction == "LONG" and candle.close < candle.open:
            return float(candle.low), float(candle.open)
        if direction == "SHORT" and candle.close > candle.open:
            return float(candle.open), float(candle.high)
    return None


def fair_value_gap(df):
    for i in range(len(df) - 1, max(2, len(df) - 25), -1):
        first = df.iloc[i - 2]
        third = df.iloc[i]
        if first.high < third.low:
            return "BULLISH", float(first.high), float(third.low)
        if first.low > third.high:
            return "BEARISH", float(third.high), float(first.low)
    return None


# ==========================================================
# RSI EXHAUSTION + VOLUME FILTERS
# ==========================================================

def rsi_not_exhausted(side, rsi_value):
    # Do not initiate a fresh trade into an already stretched RSI.
    if side == "LONG":
        return 30 < rsi_value < 75
    return 25 < rsi_value < 70


def volume_direction_ok(row, side, minimum=MIN_VOLUME_RATIO):
    if float(row.vol_ratio) < minimum:
        return False
    if side == "LONG":
        return float(row.close) > float(row.open) and float(row.plus_di) > float(row.minus_di)
    return float(row.close) < float(row.open) and float(row.minus_di) > float(row.plus_di)


# ==========================================================
# MULTI-WINDOW FAST MOMENTUM DETECTOR
# Detects sharp impulses over 15M / 30M / 1H / 2H / 3H.
# Warning can trigger on a move; confirmed entry needs:
# 15M BOS + 5M confirmation + volume + ADX + HTF support,
# RSI not exhausted and price not excessively extended.
# ==========================================================

def fast_momentum(d5, d15, bull4, bear4, bull1, bear1, bos15):
    last5 = d5.iloc[-1]
    last15 = d15.iloc[-1]

    detected = []
    for label, candles, min_pct, min_usd in FAST_WINDOWS:
        if len(d5) <= candles:
            continue
        start = d5.iloc[-(candles + 1)]
        move_usd = float(last5.close - start.close)
        base = float(start.close)
        move_pct = (move_usd / base * 100) if base else 0.0

        if abs(move_usd) >= min_usd and abs(move_pct) >= min_pct:
            detected.append({
                "window": label,
                "candles": candles,
                "move_usd": move_usd,
                "move_pct": move_pct
            })

    if not detected:
        return {
            "side": None, "confirmed": False, "window": None,
            "move_usd": 0.0, "move_pct": 0.0,
            "volume_ratio": float(last5.vol_ratio),
            "adx": float(last5.adx), "rsi": float(last5.rsi),
            "atr": float(last5.atr), "price": float(last5.close),
            "timestamp": int(last5.timestamp), "extension_atr": 0.0,
            "not_overextended": True, "reasons": [],
            "bos15": bos15
        }

    # Use the strongest percentage impulse; direction follows its sign.
    strongest = max(detected, key=lambda item: abs(item["move_pct"]))
    side = "LONG" if strongest["move_usd"] > 0 else "SHORT"

    long5 = (
        last5.close > last5.ema9 > last5.ema21
        and last5.rsi >= 52
        and last5.plus_di > last5.minus_di
    )
    short5 = (
        last5.close < last5.ema9 < last5.ema21
        and last5.rsi <= 48
        and last5.minus_di > last5.plus_di
    )
    long15 = last15.close > last15.ema9 > last15.ema21 and last15.rsi >= 50
    short15 = last15.close < last15.ema9 < last15.ema21 and last15.rsi <= 50

    volume_ok = volume_direction_ok(last5, side, FAST_VOLUME_MULT)
    adx_ok = float(last5.adx) >= FAST_ADX_MIN

    if side == "LONG":
        momentum_ok = long5 and long15
        htf_ok = (bull4 or bull1) and not (bear4 and bear1)
        bos_ok = bool(bos15["bos_up"])
    else:
        momentum_ok = short5 and short15
        htf_ok = (bear4 or bear1) and not (bull4 and bull1)
        bos_ok = bool(bos15["bos_down"])

    atr5 = float(last5.atr)
    extension = abs(float(last5.close - last5.ema21) / atr5) if atr5 > 0 else 999.0
    not_overextended = extension <= FAST_MAX_EXTENSION_ATR
    rsi_ok = rsi_not_exhausted(side, float(last5.rsi))

    # Confirmed fast entry is deliberately strict. Sharp move alone is warning.
    confirmed = all([
        volume_ok, adx_ok, momentum_ok, htf_ok,
        bos_ok, not_overextended, rsi_ok
    ])

    reasons = [
        f"{strongest['window']} impulse detected",
        f"Move {strongest['move_usd']:+,.0f} USD ({strongest['move_pct']:+.2f}%)",
        f"Volume {'PASS' if volume_ok else 'FAIL'}",
        f"ADX {'PASS' if adx_ok else 'FAIL'}",
        f"15M BOS {'PASS' if bos_ok else 'FAIL'}",
        f"5M+15M momentum {'PASS' if momentum_ok else 'FAIL'}",
        f"HTF {'PASS' if htf_ok else 'FAIL'}",
        f"RSI filter {'PASS' if rsi_ok else 'FAIL'}",
        f"Extension {'PASS' if not_overextended else 'FAIL'}"
    ]

    return {
        "side": side,
        "confirmed": bool(confirmed),
        "window": strongest["window"],
        "move_usd": strongest["move_usd"],
        "move_pct": strongest["move_pct"],
        "volume_ratio": float(last5.vol_ratio),
        "adx": float(last5.adx),
        "rsi": float(last5.rsi),
        "atr": atr5,
        "price": float(last5.close),
        "timestamp": int(last5.timestamp),
        "extension_atr": extension,
        "not_overextended": bool(not_overextended),
        "reasons": reasons,
        "bos15": bos15
    }


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

def analyze():
    d4 = fetch_candles("4H")
    d1 = fetch_candles("1H")
    d15 = fetch_candles("15m")
    d5 = fetch_candles("5m")

    a4 = d4.iloc[-1]
    a1 = d1.iloc[-1]
    a15 = d15.iloc[-1]
    a5 = d5.iloc[-1]

    s1 = market_structure(d1)
    s15 = market_structure(d15)

    c15 = timeframe_confirmation(d15, lookback=5)
    c5 = timeframe_confirmation(d5, lookback=5)

    bull4 = bool(a4.ema20 > a4.ema50 and a4.close > a4.ema200)
    bear4 = bool(a4.ema20 < a4.ema50 and a4.close < a4.ema200)
    bull1 = bool(a1.ema20 > a1.ema50 and a1.close > a1.ema200)
    bear1 = bool(a1.ema20 < a1.ema50 and a1.close < a1.ema200)

    price = float(a15.close)
    atr15 = float(a15.atr)

    range_high = float(d1.high.tail(50).max())
    range_low = float(d1.low.tail(50).min())
    midpoint = (range_high + range_low) / 2

    discount = price < midpoint
    premium = price > midpoint

    previous_high = float(d15.high.iloc[-21:-1].max())
    previous_low = float(d15.low.iloc[-21:-1].min())

    sweep_low = bool(a15.low < previous_low and price > previous_low)
    sweep_high = bool(a15.high > previous_high and price < previous_high)

    fvg = fair_value_gap(d15)
    ob_long = order_block(d15, "LONG")
    ob_short = order_block(d15, "SHORT")

    long_score = 0
    short_score = 0
    long_reasons = []
    short_reasons = []

    def add(condition, points, side, reason):
        nonlocal long_score, short_score
        if not condition:
            return
        if side == "LONG":
            long_score += points
            long_reasons.append(reason)
        else:
            short_score += points
            short_reasons.append(reason)

    add(bull4, 2, "LONG", "4H bullish")
    add(bear4, 2, "SHORT", "4H bearish")
    add(bull1, 2, "LONG", "1H bullish")
    add(bear1, 2, "SHORT", "1H bearish")
    add(s1["bos_up"], 2, "LONG", "1H BOS UP")
    add(s1["bos_down"], 2, "SHORT", "1H BOS DOWN")
    add(s15["bos_up"], 2, "LONG", "15M BOS UP")
    add(s15["bos_down"], 2, "SHORT", "15M BOS DOWN")

    add(45 <= a15.rsi <= 68, 1, "LONG", "15M RSI supportive")
    add(32 <= a15.rsi <= 55, 1, "SHORT", "15M RSI supportive")

    if float(a15.vol_ratio) >= MIN_VOLUME_RATIO:
        add(True, 1, "LONG" if a15.close > a15.open else "SHORT", "15M volume expansion")

    add(sweep_low, 2, "LONG", "Liquidity sweep low")
    add(sweep_high, 2, "SHORT", "Liquidity sweep high")
    add(discount, 1, "LONG", "Discount zone")
    add(premium, 1, "SHORT", "Premium zone")
    add(bool(fvg and fvg[0] == "BULLISH"), 1, "LONG", "Bullish FVG")
    add(bool(fvg and fvg[0] == "BEARISH"), 1, "SHORT", "Bearish FVG")

    # Both confirmations are required for standard FINAL signal.
    long_confirmation = bool(c15["long"] and c5["long"])
    short_confirmation = bool(c15["short"] and c5["short"])

    if c15["long"]:
        add(True, 1, "LONG", "15M confirmation")
    if c5["long"]:
        add(True, 2, "LONG", "5M confirmation")
    if c15["short"]:
        add(True, 1, "SHORT", "15M confirmation")
    if c5["short"]:
        add(True, 2, "SHORT", "5M confirmation")

    # Early warning is a heads-up only; never an entry.
    early = None
    if long_score >= EARLY_SCORE and long_score > short_score:
        early = "LONG"
    elif short_score >= EARLY_SCORE and short_score > long_score:
        early = "SHORT"

    long_htf_ok = (bull4 or bull1) and not (bear4 and bear1)
    short_htf_ok = (bear4 or bear1) and not (bull4 and bull1)

    # Volume direction and RSI exhaustion filters on both lower timeframes.
    long_volume_ok = (
        volume_direction_ok(a15, "LONG")
        and volume_direction_ok(a5, "LONG")
    )
    short_volume_ok = (
        volume_direction_ok(a15, "SHORT")
        and volume_direction_ok(a5, "SHORT")
    )

    long_rsi_ok = (
        rsi_not_exhausted("LONG", float(a15.rsi))
        and rsi_not_exhausted("LONG", float(a5.rsi))
    )
    short_rsi_ok = (
        rsi_not_exhausted("SHORT", float(a15.rsi))
        and rsi_not_exhausted("SHORT", float(a5.rsi))
    )

    final = None
    final_timeframes = []

    if (
        long_confirmation and s15["bos_up"] and long_htf_ok
        and long_volume_ok and long_rsi_ok
        and long_score >= MIN_SCORE and long_score > short_score
    ):
        final = "LONG"
        final_timeframes = ["15M BOS", "15M", "5M"]
    elif (
        short_confirmation and s15["bos_down"] and short_htf_ok
        and short_volume_ok and short_rsi_ok
        and short_score >= MIN_SCORE and short_score > long_score
    ):
        final = "SHORT"
        final_timeframes = ["15M BOS", "15M", "5M"]

    levels = None
    if final == "LONG":
        entry = float(a5.close)
        risk = 1.5 * float(a5.atr)
        sl = entry - risk
        levels = (entry, sl, entry + risk, entry + 2 * risk, entry + 3 * risk)
    elif final == "SHORT":
        entry = float(a5.close)
        risk = 1.5 * float(a5.atr)
        sl = entry + risk
        levels = (entry, sl, entry - risk, entry - 2 * risk, entry - 3 * risk)

    fast = fast_momentum(d5, d15, bull4, bear4, bull1, bear1, s15)

    fast_levels = None
    if fast["confirmed"] and fast["side"] in ("LONG", "SHORT") and fast["atr"] > 0:
        entry = fast["price"]
        risk = 1.5 * fast["atr"]
        if fast["side"] == "LONG":
            fast_levels = (entry, entry - risk, entry + risk, entry + 2 * risk, entry + 3 * risk)
        else:
            fast_levels = (entry, entry + risk, entry - risk, entry - 2 * risk, entry - 3 * risk)

    return {
        "price": price, "atr": atr15, "early": early, "final": final,
        "levels": levels, "final_timeframes": final_timeframes,
        "long_score": long_score, "short_score": short_score,
        "long_reasons": long_reasons, "short_reasons": short_reasons,
        "bull4": bull4, "bear4": bear4, "bull1": bull1, "bear1": bear1,
        "bos_up": bool(s15["bos_up"]), "bos_down": bool(s15["bos_down"]),
        "confirm15_long": c15["long"], "confirm15_short": c15["short"],
        "confirm5_long": c5["long"], "confirm5_short": c5["short"],
        "rsi": float(a15.rsi), "rsi5": float(a5.rsi),
        "vol15": float(a15.vol_ratio), "vol5": float(a5.vol_ratio),
        "long_volume_ok": long_volume_ok, "short_volume_ok": short_volume_ok,
        "long_rsi_ok": long_rsi_ok, "short_rsi_ok": short_rsi_ok,
        "range_high": range_high, "range_low": range_low, "midpoint": midpoint,
        "sweep_low": sweep_low, "sweep_high": sweep_high,
        "fvg": fvg, "ob_long": ob_long, "ob_short": ob_short,
        "candle_ts": int(a15.timestamp), "candle5_ts": int(a5.timestamp),
        "fast": fast, "fast_levels": fast_levels
    }


# ==========================================================
# DIAGNOSTICS
# ==========================================================

def diagnostic(result):
    log.info("========== BTC DIAGNOSTICS ==========")
    log.info("LONG SCORE=%s | SHORT SCORE=%s", result["long_score"], result["short_score"])
    log.info("4H BULL=%s BEAR=%s", result["bull4"], result["bear4"])
    log.info("1H BULL=%s BEAR=%s", result["bull1"], result["bear1"])
    log.info("15M BOS UP=%s DOWN=%s", result["bos_up"], result["bos_down"])
    log.info("15M CONFIRM L=%s S=%s", result["confirm15_long"], result["confirm15_short"])
    log.info("5M CONFIRM L=%s S=%s", result["confirm5_long"], result["confirm5_short"])
    log.info("VOLUME L=%s S=%s | RSI L=%s S=%s",
             result["long_volume_ok"], result["short_volume_ok"],
             result["long_rsi_ok"], result["short_rsi_ok"])
    log.info("EARLY=%s FINAL=%s", result["early"], result["final"])
    log.info("FAST=%s", result["fast"])
    log.info("LONG REASONS: %s", result["long_reasons"])
    log.info("SHORT REASONS: %s", result["short_reasons"])
    log.info("========== END DIAGNOSTICS ==========")


# ==========================================================
# STATE
# ==========================================================

def load_state():
    try:
        with open(STATE_FILE, "r") as file:
            state = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}

    state.setdefault("sent", {})
    state.setdefault("early_sent", {})
    state.setdefault("fast_sent", {})
    return state


def save_state(state):
    with open(STATE_FILE, "w") as file:
        json.dump(state, file, indent=2)


# ==========================================================
# MESSAGE HELPERS
# ==========================================================

def fmt(value):
    return f"{value:,.2f}"


def utc_string(timestamp_ms):
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def make_message(result, stage):
    if stage == "CONFIRMED":
        side = result["final"]
        title = "BTC FINAL CONFIRMED SIGNAL"
    else:
        side = result["early"]
        title = "BTC EARLY WARNING - NOT AN ENTRY"

    reasons = result["long_reasons"] if side == "LONG" else result["short_reasons"]

    lines = [
        title, "", "BITGET USDT-M FUTURES", f"BTCUSDT: {side}",
        f"Price: {fmt(result['price'])}",
        f"LONG Score: {result['long_score']}",
        f"SHORT Score: {result['short_score']}",
        f"15M RSI: {result['rsi']:.1f}",
        f"5M RSI: {result['rsi5']:.1f}",
        f"15M Volume Ratio: {result['vol15']:.2f}x",
        f"5M Volume Ratio: {result['vol5']:.2f}x",
        "",
        f"4H Bullish: {result['bull4']}",
        f"4H Bearish: {result['bear4']}",
        f"1H Bullish: {result['bull1']}",
        f"1H Bearish: {result['bear1']}",
        f"15M BOS UP: {result['bos_up']}",
        f"15M BOS DOWN: {result['bos_down']}",
        f"15M LONG Confirm: {result['confirm15_long']}",
        f"15M SHORT Confirm: {result['confirm15_short']}",
        f"5M LONG Confirm: {result['confirm5_long']}",
        f"5M SHORT Confirm: {result['confirm5_short']}"
    ]

    if stage == "CONFIRMED" and result["levels"]:
        entry, sl, tp1, tp2, tp3 = result["levels"]
        lines += [
            "", "CONFIRMATION: 15M BOS + 15M + 5M",
            f"ENTRY: {fmt(entry)}", f"STOP LOSS: {fmt(sl)}",
            f"TP1: {fmt(tp1)}", f"TP2: {fmt(tp2)}", f"TP3: {fmt(tp3)}",
            "", "Volume and RSI filters passed."
        ]
    else:
        lines += ["", "Wait for final confirmation.", "Do not enter on Early Warning."]

    lines += [
        "", "Reasons: " + "; ".join(reasons),
        "FVG is optional, not compulsory.",
        "Signals are not guaranteed. Paper-test first.",
        "15M Candle UTC: " + utc_string(result["candle_ts"]),
        "5M Candle UTC: " + utc_string(result["candle5_ts"])
    ]
    return "\n".join(lines)


def make_fast_message(result):
    fast = result["fast"]
    side = fast["side"]

    title = (
        "BTC FAST MOMENTUM - CONFIRMED ENTRY"
        if fast["confirmed"]
        else "BTC FAST MOMENTUM WARNING - NOT AN ENTRY"
    )

    lines = [
        title, "", "BITGET USDT-M FUTURES",
        f"BTCUSDT: {side}",
        f"Current Price: {fmt(fast['price'])}",
        f"Detected Window: {fast['window']}",
        f"Move: {fast['move_usd']:+,.2f} USD",
        f"Move: {fast['move_pct']:+.2f}%",
        f"5M RSI: {fast['rsi']:.1f}",
        f"5M Volume Ratio: {fast['volume_ratio']:.2f}x",
        f"5M ADX: {fast['adx']:.1f}",
        f"Distance from EMA21: {fast['extension_atr']:.2f} ATR",
        f"15M BOS UP: {fast['bos15']['bos_up']}",
        f"15M BOS DOWN: {fast['bos15']['bos_down']}"
    ]

    if fast["confirmed"] and result["fast_levels"]:
        entry, sl, tp1, tp2, tp3 = result["fast_levels"]
        lines += [
            "", f"ENTRY: {fmt(entry)}", f"STOP LOSS: {fmt(sl)}",
            f"TP1: {fmt(tp1)}", f"TP2: {fmt(tp2)}", f"TP3: {fmt(tp3)}",
            "", "Fast momentum confirmation checks passed."
        ]
    else:
        lines += [
            "", "Sharp movement detected, but confirmation is incomplete.",
            "This is NOT an entry. Do not chase the move."
        ]

    lines += [
        "", "Checks: " + "; ".join(fast["reasons"]),
        "Fast signal is not guaranteed. Paper-test first.",
        "5M Candle UTC: " + utc_string(fast["timestamp"])
    ]
    return "\n".join(lines)


# ==========================================================
# MAIN
# ==========================================================

def main():
    log.info("BTC MULTI-TIMEFRAME BOT STARTING")

    result = analyze()
    diagnostic(result)

    state = load_state()
    sent = state["sent"]
    early_sent = state["early_sent"]
    fast_sent = state["fast_sent"]

    # Early warning: once per 15M candle and direction.
    if result["early"]:
        key = f"{result['candle_ts']}:EARLY:{result['early']}"
        if key not in early_sent:
            send_telegram(make_message(result, "EARLY"))
            early_sent[key] = int(time.time())
        else:
            log.info("EARLY WARNING ALREADY SENT FOR THIS CANDLE")

    # Standard final: both 15M BOS and 15M + 5M confirmations required.
    if result["final"] and result["levels"]:
        key = f"{result['candle_ts']}:CONFIRMED:{result['final']}"
        if key not in sent:
            send_telegram(make_message(result, "CONFIRMED"))
            sent[key] = int(time.time())
        else:
            log.info("FINAL SIGNAL ALREADY SENT FOR THIS CANDLE")
    else:
        log.info("STANDARD FINAL CONFIRMATION NOT MET")

    # Fast alerts are deduplicated by latest closed 5M candle and side/stage.
    fast = result["fast"]
    if fast["side"]:
        stage = "FAST_CONFIRMED" if fast["confirmed"] else "FAST_WARNING"
        key = f"{fast['timestamp']}:{stage}:{fast['side']}:{fast['window']}"
        if key not in fast_sent:
            send_telegram(make_fast_message(result))
            fast_sent[key] = int(time.time())
        else:
            log.info("FAST ALERT ALREADY SENT FOR THIS 5M CANDLE")
    else:
        log.info("NO FAST MOMENTUM THRESHOLD REACHED")

    # Keep state file compact.
    for name in ("sent", "early_sent", "fast_sent"):
        entries = state[name]
        if len(entries) > 500:
            state[name] = dict(sorted(entries.items(), key=lambda item: item[1])[-300:])

    save_state(state)
    log.info("BTC BOT FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("BTC BOT FAILED")
        raise
