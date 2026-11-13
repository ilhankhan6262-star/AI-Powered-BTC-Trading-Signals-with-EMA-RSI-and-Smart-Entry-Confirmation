import os
import json
import time
import logging
from datetime import datetime, timezone

import requests
import pandas as pd

# ==========================================
# BTC FUTURES MULTI-TIMEFRAME BOT
# 4H + 1H + 15M + 5M
# ==========================================

BASE_URL = "https://api.bitget.com"
SYMBOL = "BTCUSDT"
PRODUCT_TYPE = "USDT-FUTURES"
STATE_FILE = "bot_state.json"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

MIN_SCORE = 6
EARLY_SCORE = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

log = logging.getLogger("BTC-BOT")


# ==========================================
# TELEGRAM
# ==========================================

def send_telegram(message):

    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise RuntimeError("Telegram secrets missing")

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
    )

    response = requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=20
    )

    log.info(
        "Telegram HTTP status: %s",
        response.status_code
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(f"Telegram error: {result}")

    log.info("TELEGRAM MESSAGE SENT SUCCESSFULLY")


# ==========================================
# CANDLE DATA
# ==========================================

def fetch_candles(granularity, limit=250):

    url = BASE_URL + "/api/v2/mix/market/candles"

    params = {
        "symbol": SYMBOL,
        "productType": PRODUCT_TYPE,
        "granularity": granularity,
        "limit": str(limit)
    }

    last_error = None

    for attempt in range(3):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=20
            )

            response.raise_for_status()

            payload = response.json()

            if payload.get("code") != "00000":
                raise RuntimeError(payload.get("msg"))

            rows = payload.get("data", [])

            if not rows:
                raise RuntimeError("Empty candle data")

            clean_rows = [
                list(row[:7])
                for row in rows
                if isinstance(row, (list, tuple))
                and len(row) >= 7
            ]

            df = pd.DataFrame(
                clean_rows,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover"
                ]
            )

            for col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            df = (
                df.dropna()
                .sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True)
            )

            intervals = {
                "5m": 300000,
                "15m": 900000,
                "1H": 3600000,
                "4H": 14400000
            }

            interval = intervals[granularity]

            now = int(time.time() * 1000)

            # Closed candles only.
            df = df[
                df["timestamp"] + interval <= now - 3000
            ].reset_index(drop=True)

            if len(df) < 205:
                raise RuntimeError(
                    f"Insufficient closed candles: {len(df)}"
                )

            log.info(
                "%s candles loaded: %s",
                granularity,
                len(df)
            )

            return indicators(df)

        except Exception as error:

            last_error = error

            log.warning(
                "%s attempt %s failed: %s",
                granularity,
                attempt + 1,
                error
            )

            time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"{granularity} data failed: {last_error}"
    )


# ==========================================
# INDICATORS
# ==========================================

def indicators(df):

    df = df.copy()

    close = df["close"]

    df["ema20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["ema200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    delta = close.diff()

    gain = delta.clip(
        lower=0
    ).ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    loss = (-delta.clip(
        upper=0
    )).ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = gain / loss.replace(
        0,
        float("nan")
    )

    df["rsi"] = (
        100 - 100 / (1 + rs)
    ).fillna(50)

    previous = close.shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous).abs(),
            (df["low"] - previous).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    df["vol_avg"] = df["volume"].rolling(20).mean()

    return (
        df.replace(
            [float("inf"), float("-inf")],
            float("nan")
        )
        .dropna()
        .reset_index(drop=True)
    )


# ==========================================
# MARKET STRUCTURE
# ==========================================

def market_structure(df):

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    swing_highs = []
    swing_lows = []

    for i in range(3, len(df) - 3):

        h = highs[i - 3:i + 4]
        l = lows[i - 3:i + 4]

        if (
            highs[i] == max(h)
            and list(h).count(highs[i]) == 1
        ):
            swing_highs.append(float(highs[i]))

        if (
            lows[i] == min(l)
            and list(l).count(lows[i]) == 1
        ):
            swing_lows.append(float(lows[i]))

    last = df.iloc[-1]
    previous = df.iloc[-2]

    swing_high = (
        swing_highs[-1]
        if swing_highs
        else float(df.high.iloc[-30:-3].max())
    )

    swing_low = (
        swing_lows[-1]
        if swing_lows
        else float(df.low.iloc[-30:-3].min())
    )

    bos_up = (
        last.close > swing_high
        and previous.close <= swing_high
    )

    bos_down = (
        last.close < swing_low
        and previous.close >= swing_low
    )

    return {
        "bos_up": bool(bos_up),
        "bos_down": bool(bos_down),
        "swing_high": swing_high,
        "swing_low": swing_low
    }


# ==========================================
# 5 MINUTE FINAL CONFIRMATION
# ==========================================

def confirmation_5m(df):

    last = df.iloc[-1]

    previous_high = float(
        df.high.iloc[-6:-1].max()
    )

    previous_low = float(
        df.low.iloc[-6:-1].min()
    )

    long_confirm = (
        last.close > previous_high
        and last.close > last.ema20
        and last.ema20 > last.ema50
        and last.rsi >= 50
    )

    short_confirm = (
        last.close < previous_low
        and last.close < last.ema20
        and last.ema20 < last.ema50
        and last.rsi <= 50
    )

    return {
        "long": bool(long_confirm),
        "short": bool(short_confirm),
        "price": float(last.close),
        "rsi": float(last.rsi),
        "timestamp": int(last.timestamp),
        "break_high": previous_high,
        "break_low": previous_low
    }


# ==========================================
# ORDER BLOCK
# ==========================================

def order_block(df, direction):

    start = max(2, len(df) - 18)

    for i in range(len(df) - 2, start - 1, -1):

        candle = df.iloc[i]

        if (
            direction == "LONG"
            and candle.close < candle.open
        ):
            return (
                float(candle.low),
                float(candle.open)
            )

        if (
            direction == "SHORT"
            and candle.close > candle.open
        ):
            return (
                float(candle.open),
                float(candle.high)
            )

    return None


# ==========================================
# FAIR VALUE GAP
# ==========================================

def fair_value_gap(df):

    for i in range(
        len(df) - 1,
        max(2, len(df) - 25),
        -1
    ):

        first = df.iloc[i - 2]
        third = df.iloc[i]

        if first.high < third.low:

            return (
                "BULLISH",
                float(first.high),
                float(third.low)
            )

        if first.low > third.high:

            return (
                "BEARISH",
                float(third.high),
                float(first.low)
            )

    return None


# ==========================================
# ANALYSIS
# ==========================================

def analyze():

    d4 = fetch_candles("4H")
    d1 = fetch_candles("1H")
    d15 = fetch_candles("15m")
    d5 = fetch_candles("5m")

    a4 = d4.iloc[-1]
    a1 = d1.iloc[-1]
    a15 = d15.iloc[-1]

    s1 = market_structure(d1)
    s15 = market_structure(d15)

    c5 = confirmation_5m(d5)

    bull4 = (
        a4.ema20 > a4.ema50
        and a4.close > a4.ema200
    )

    bear4 = (
        a4.ema20 < a4.ema50
        and a4.close < a4.ema200
    )

    bull1 = (
        a1.ema20 > a1.ema50
        and a1.close > a1.ema200
    )

    bear1 = (
        a1.ema20 < a1.ema50
        and a1.close < a1.ema200
    )

    price = float(a15.close)
    atr = float(a15.atr)

    range_high = float(d1.high.tail(50).max())
    range_low = float(d1.low.tail(50).min())

    midpoint = (range_high + range_low) / 2

    discount = price < midpoint
    premium = price > midpoint

    previous_high = float(
        d15.high.iloc[-21:-1].max()
    )

    previous_low = float(
        d15.low.iloc[-21:-1].min()
    )

    sweep_low = (
        a15.low < previous_low
        and price > previous_low
    )

    sweep_high = (
        a15.high > previous_high
        and price < previous_high
    )

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

    add(
        45 <= a15.rsi <= 68,
        1,
        "LONG",
        "15M RSI supportive"
    )

    add(
        32 <= a15.rsi <= 55,
        1,
        "SHORT",
        "15M RSI supportive"
    )

    add(
        a15.volume > a15.vol_avg * 1.15,
        1,
        "LONG" if a15.close > a15.open else "SHORT",
        "Volume expansion"
    )

    add(sweep_low, 2, "LONG", "Liquidity sweep low")
    add(sweep_high, 2, "SHORT", "Liquidity sweep high")

    add(discount, 1, "LONG", "Discount zone")
    add(premium, 1, "SHORT", "Premium zone")

    add(
        bool(fvg and fvg[0] == "BULLISH"),
        1,
        "LONG",
        "Bullish FVG"
    )

    add(
        bool(fvg and fvg[0] == "BEARISH"),
        1,
        "SHORT",
        "Bearish FVG"
    )

    # Early warning: score-based, not final entry.

    early = None

    if (
        long_score >= EARLY_SCORE
        and long_score > short_score
    ):
        early = "LONG"

    elif (
        short_score >= EARLY_SCORE
        and short_score > long_score
    ):
        early = "SHORT"

    # Final confirmation.

    final = None

    if (
        bull4
        and bull1
        and s15["bos_up"]
        and long_score >= MIN_SCORE
        and long_score > short_score
        and c5["long"]
    ):
        final = "LONG"

    elif (
        bear4
        and bear1
        and s15["bos_down"]
        and short_score >= MIN_SCORE
        and short_score > long_score
        and c5["short"]
    ):
        final = "SHORT"

    # Entry and risk levels.

    levels = None

    if atr > 0 and final == "LONG":

        entry = c5["price"]

        sl = min(
            entry - 1.2 * atr,
            float(a15.low) - 0.15 * atr
        )

        risk = entry - sl

        levels = (
            entry,
            sl,
            entry + risk,
            entry + 2 * risk,
            entry + 3 * risk
        )

    elif atr > 0 and final == "SHORT":

        entry = c5["price"]

        sl = max(
            entry + 1.2 * atr,
            float(a15.high) + 0.15 * atr
        )

        risk = sl - entry

        levels = (
            entry,
            sl,
            entry - risk,
            entry - 2 * risk,
            entry - 3 * risk
        )

    return {
        "price": price,
        "atr": atr,
        "early": early,
        "final": final,
        "levels": levels,
        "long_score": long_score,
        "short_score": short_score,
        "long_reasons": long_reasons,
        "short_reasons": short_reasons,
        "bull4": bool(bull4),
        "bear4": bool(bear4),
        "bull1": bool(bull1),
        "bear1": bool(bear1),
        "bos_up": s15["bos_up"],
        "bos_down": s15["bos_down"],
        "confirm5_long": c5["long"],
        "confirm5_short": c5["short"],
        "rsi": float(a15.rsi),
        "rsi5": c5["rsi"],
        "range_high": range_high,
        "range_low": range_low,
        "midpoint": midpoint,
        "sweep_low": sweep_low,
        "sweep_high": sweep_high,
        "fvg": fvg,
        "ob_long": ob_long,
        "ob_short": ob_short,
        "candle_ts": int(a15.timestamp),
        "candle5_ts": c5["timestamp"]
    }


# ==========================================
# DIAGNOSTIC LOGS
# ==========================================

def diagnostic(result):

    log.info("========== BTC DIAGNOSTICS ==========")

    log.info(
        "LONG SCORE=%s | SHORT SCORE=%s",
        result["long_score"],
        result["short_score"]
    )

    log.info(
        "4H BULL=%s BEAR=%s",
        result["bull4"],
        result["bear4"]
    )

    log.info(
        "1H BULL=%s BEAR=%s",
        result["bull1"],
        result["bear1"]
    )

    log.info(
        "15M BOS UP=%s DOWN=%s",
        result["bos_up"],
        result["bos_down"]
    )

    log.info(
        "5M LONG CONFIRM=%s SHORT CONFIRM=%s",
        result["confirm5_long"],
        result["confirm5_short"]
    )

    log.info(
        "EARLY=%s FINAL=%s",
        result["early"],
        result["final"]
    )

    log.info(
        "LONG REASONS: %s",
        result["long_reasons"]
    )

    log.info(
        "SHORT REASONS: %s",
        result["short_reasons"]
    )

    log.info("========== END DIAGNOSTICS ==========")


# ==========================================
# STATE
# ==========================================

def load_state():

    try:
        with open(STATE_FILE, "r") as file:
            state = json.load(file)

    except (
        FileNotFoundError,
        json.JSONDecodeError
    ):
        state = {}

    state.setdefault("sent", {})
    state.setdefault("early_sent", {})

    return state


def save_state(state):

    with open(STATE_FILE, "w") as file:
        json.dump(state, file, indent=2)


# ==========================================
# MESSAGE
# ==========================================

def fmt(value):

    return f"{value:,.2f}"


def make_message(result, stage):

    side = (
        result["final"]
        if stage == "CONFIRMED"
        else result["early"]
    )

    if stage == "CONFIRMED":

        title = "BTC FINAL CONFIRMED SIGNAL"

    else:

        title = "BTC EARLY WARNING - NOT AN ENTRY"

    lines = [
        title,
        "",
        "BITGET USDT-M FUTURES",
        f"BTCUSDT: {side}",
        f"Price: {fmt(result['price'])}",
        f"LONG Score: {result['long_score']}",
        f"SHORT Score: {result['short_score']}",
        f"15M RSI: {result['rsi']:.1f}",
        f"5M RSI: {result['rsi5']:.1f}",
        "",
        f"4H Bullish: {result['bull4']}",
        f"4H Bearish: {result['bear4']}",
        f"1H Bullish: {result['bull1']}",
        f"1H Bearish: {result['bear1']}",
        f"15M BOS UP: {result['bos_up']}",
        f"15M BOS DOWN: {result['bos_down']}",
        f"5M LONG Confirm: {result['confirm5_long']}",
        f"5M SHORT Confirm: {result['confirm5_short']}"
    ]

    if stage == "CONFIRMED" and result["levels"]:

        entry, sl, tp1, tp2, tp3 = result["levels"]

        lines += [
            "",
            f"ENTRY: {fmt(entry)}",
            f"STOP LOSS: {fmt(sl)}",
            f"TP1: {fmt(tp1)}",
            f"TP2: {fmt(tp2)}",
            f"TP3: {fmt(tp3)}",
            "",
            "All timeframe confirmations passed."
        ]

    else:

        lines += [
            "",
            "Wait for final confirmation.",
            "Do not enter on Early Warning."
        ]

    reasons = (
        result["long_reasons"]
        if side == "LONG"
        else result["short_reasons"]
    )

    lines += [
        "",
        "Reasons: " + "; ".join(reasons),
        "Signals are not guaranteed. Paper-test first."
    ]

    candle_time = datetime.fromtimestamp(
        result["candle_ts"] / 1000,
        timezone.utc
    )

    lines.append(
        "15M Candle UTC: " +
        candle_time.strftime("%Y-%m-%d %H:%M")
    )

    return "\n".join(lines)


# ==========================================
# MAIN
# ==========================================

def main():

    log.info("BTC MULTI-TIMEFRAME BOT STARTING")

    result = analyze()

    diagnostic(result)

    state = load_state()

    sent = state["sent"]
    early_sent = state["early_sent"]

    # Early warning once per 15M candle and direction.

    if result["early"]:

        early_key = (
            f"{result['candle_ts']}:EARLY:{result['early']}"
        )

        if early_key not in early_sent:

            log.info(
                "EARLY WARNING CONDITION PASSED: %s",
                result["early"]
            )

            send_telegram(
                make_message(result, "EARLY")
            )

            early_sent[early_key] = int(time.time())

        else:

            log.info("EARLY WARNING ALREADY SENT FOR THIS CANDLE")

    else:

        log.info("EARLY WARNING CONDITION NOT MET")

    # Final confirmed signal once per 15M candle and direction.

    if result["final"]:

        key = (
            f"{result['candle_ts']}:"
            f"CONFIRMED:{result['final']}"
        )

        if key not in sent:

            log.info(
                "FINAL SIGNAL CONDITION PASSED: %s",
                result["final"]
            )

            send_telegram(
                make_message(result, "CONFIRMED")
            )

            sent[key] = int(time.time())

        else:

            log.info("FINAL SIGNAL ALREADY SENT FOR THIS CANDLE")

    else:

        log.info("FINAL CONFIRMATION NOT MET")

    # Keep state file small.

    if len(sent) > 500:

        state["sent"] = dict(
            sorted(
                sent.items(),
                key=lambda item: item[1]
            )[-300:]
        )

    if len(early_sent) > 500:

        state["early_sent"] = dict(
            sorted(
                early_sent.items(),
                key=lambda item: item[1]
            )[-300:]
        )

    save_state(state)

    log.info("BTC BOT FINISHED SUCCESSFULLY")


# ==========================================
# START
# ==========================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        log.exception("BTC BOT FAILED")

        raise
