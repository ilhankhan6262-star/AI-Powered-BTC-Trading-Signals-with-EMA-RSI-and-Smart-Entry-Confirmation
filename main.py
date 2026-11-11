import os
import json
import time
import logging
from datetime import datetime, timezone

import requests
import pandas as pd

BASE_URL = "https://api.bitget.com"
SYMBOL = "BTCUSDT"
STATE_FILE = "bot_state.json"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

MIN_SCORE = 6
EARLY_SCORE = 4

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BTC-BOT")


def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise RuntimeError("Telegram secrets missing")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=20
    )

    response.raise_for_status()

    if not response.json().get("ok"):
        raise RuntimeError("Telegram message failed")


def fetch_candles(granularity, limit=250):

    url = f"{BASE_URL}/api/v2/spot/market/candles"

    params = {
        "symbol": SYMBOL,
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

            df = pd.DataFrame(
                rows,
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

            for col in [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]:
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

            interval = {
                "15min": 15 * 60000,
                "1h": 60 * 60000,
                "4h": 240 * 60000
            }[granularity]

            now = int(time.time() * 1000)

            df = df[
                df["timestamp"] + interval <= now - 3000
            ].reset_index(drop=True)

            if len(df) < 80:
                raise RuntimeError("Not enough closed candles")

            return indicators(df)

        except Exception as error:
            last_error = error
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"Bitget data error: {last_error}"
    )


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

    gain = delta.clip(lower=0).ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    rs = gain / loss.replace(0, float("nan"))

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
        alpha=1/14,
        adjust=False
    ).mean()

    df["vol_avg"] = df["volume"].rolling(20).mean()

    return df.dropna().reset_index(drop=True)


def pivots(df):

    highs = []
    lows = []

    h = df["high"].to_numpy()
    l = df["low"].to_numpy()

    for i in range(3, len(df) - 3):

        section_h = h[i-3:i+4]
        section_l = l[i-3:i+4]

        if h[i] == max(section_h) and list(section_h).count(h[i]) == 1:
            highs.append(float(h[i]))

        if l[i] == min(section_l) and list(section_l).count(l[i]) == 1:
            lows.append(float(l[i]))

    return highs, lows


def market_structure(df):

    highs, lows = pivots(df)

    last = df.iloc[-1]
    previous = df.iloc[-2]

    swing_high = highs[-1] if highs else float(df["high"].iloc[-30:-3].max())
    swing_low = lows[-1] if lows else float(df["low"].iloc[-30:-3].min())

    bos_up = (
        last.close > swing_high
        and previous.close <= swing_high
    )

    bos_down = (
        last.close < swing_low
        and previous.close >= swing_low
    )

    trend_up = last.ema20 > last.ema50
    trend_down = last.ema20 < last.ema50

    return {
        "bos_up": bos_up,
        "bos_down": bos_down,
        "choch_up": bos_up and trend_down,
        "choch_down": bos_down and trend_up,
        "swing_high": swing_high,
        "swing_low": swing_low
    }


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

    for i in range(len(df) - 1, max(1, len(df) - 25), -1):

        first = df.iloc[i-2]
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


def analyze():

    d4 = fetch_candles("4h")
    d1 = fetch_candles("1h")
    d15 = fetch_candles("15min")

    a4 = d4.iloc[-1]
    a1 = d1.iloc[-1]
    a15 = d15.iloc[-1]

    s1 = market_structure(d1)
    s15 = market_structure(d15)

    price = float(a15.close)
    atr = float(a15.atr)

    bull4 = a4.ema20 > a4.ema50 and a4.close > a4.ema200
    bear4 = a4.ema20 < a4.ema50 and a4.close < a4.ema200

    bull1 = a1.ema20 > a1.ema50 and a1.close > a1.ema200
    bear1 = a1.ema20 < a1.ema50 and a1.close < a1.ema200

    range_high = float(d1.high.tail(50).max())
    range_low = float(d1.low.tail(50).min())

    midpoint = (range_high + range_low) / 2

    discount = price < midpoint
    premium = price > midpoint

    previous_high = float(d15.high.iloc[-21:-1].max())
    previous_low = float(d15.low.iloc[-21:-1].min())

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

    add(bull4, 2, "LONG", "4H bullish trend")
    add(bear4, 2, "SHORT", "4H bearish trend")

    add(bull1, 2, "LONG", "1H bullish trend")
    add(bear1, 2, "SHORT", "1H bearish trend")

    add(s1["bos_up"], 2, "LONG", "1H BOS")
    add(s1["bos_down"], 2, "SHORT", "1H BOS")

    add(s15["bos_up"], 2, "LONG", "15M BOS")
    add(s15["bos_down"], 2, "SHORT", "15M BOS")

    add(s15["choch_up"], 1, "LONG", "15M CHOCH")
    add(s15["choch_down"], 1, "SHORT", "15M CHOCH")

    add(
        45 <= a15.rsi <= 68,
        1,
        "LONG",
        "RSI supportive"
    )

    add(
        32 <= a15.rsi <= 55,
        1,
        "SHORT",
        "RSI supportive"
    )

    add(
        a15.volume > a15.vol_avg * 1.15,
        1,
        "LONG" if a15.close > a15.open else "SHORT",
        "Volume expansion"
    )

    add(sweep_low, 2, "LONG", "Sell-side liquidity sweep")
    add(sweep_high, 2, "SHORT", "Buy-side liquidity sweep")

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

    early = None
    final = None

    if bull4 and bull1 and long_score >= EARLY_SCORE and long_score > short_score:
        early = "LONG"

    elif bear4 and bear1 and short_score >= EARLY_SCORE and short_score > long_score:
        early = "SHORT"

    if bull4 and bull1 and s15["bos_up"] and long_score >= MIN_SCORE and long_score > short_score:
        final = "LONG"

    elif bear4 and bear1 and s15["bos_down"] and short_score >= MIN_SCORE and short_score > long_score:
        final = "SHORT"

    levels = None

    side = final or early

    if atr > 0 and side == "LONG":

        sl = min(
            price - 1.2 * atr,
            float(a15.low) - 0.15 * atr
        )

        risk = price - sl

        levels = (
            price,
            sl,
            price + risk,
            price + 2 * risk,
            price + 3 * risk
        )

    elif atr > 0 and side == "SHORT":

        sl = max(
            price + 1.2 * atr,
            float(a15.high) + 0.15 * atr
        )

        risk = sl - price

        levels = (
            price,
            sl,
            price - risk,
            price - 2 * risk,
            price - 3 * risk
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
        "rsi": float(a15.rsi),
        "range_high": range_high,
        "range_low": range_low,
        "midpoint": midpoint,
        "sweep_low": sweep_low,
        "sweep_high": sweep_high,
        "fvg": fvg,
        "ob_long": ob_long,
        "ob_short": ob_short,
        "candle_ts": int(a15.timestamp)
    }


def load_state():

    try:
        with open(STATE_FILE, "r") as file:
            return json.load(file)

    except (FileNotFoundError, json.JSONDecodeError):
        return {"sent": {}}


def save_state(state):

    with open(STATE_FILE, "w") as file:
        json.dump(state, file, indent=2)


def fmt(value):
    return f"{value:,.2f}"


def make_message(result, stage):

    side = result["final"] if stage == "CONFIRMED" else result["early"]

    if stage == "CONFIRMED":
        title = "✅ FINAL CONFIRMED SIGNAL"
    else:
        title = "⚠️ EARLY WARNING — NOT AN ENTRY"

    lines = [
        title,
        f"BTC/USDT | {side}",
        f"Price: {fmt(result['price'])}",
        f"LONG score: {result['long_score']}",
        f"SHORT score: {result['short_score']}",
        f"15M RSI: {result['rsi']:.1f}"
    ]

    if stage == "CONFIRMED" and result["levels"]:

        entry, sl, tp1, tp2, tp3 = result["levels"]

        lines += [
            f"Entry: {fmt(entry)}",
            f"Stop Loss: {fmt(sl)}",
            f"TP1: {fmt(tp1)}",
            f"TP2: {fmt(tp2)}",
            f"TP3: {fmt(tp3)}",
            "Confirmation: closed 15M BOS + aligned 4H/1H trend"
        ]

    else:
        lines.append(
            "Abhi entry na lein. Final confirmation ka intezar karein."
        )

    lines += [
        f"1H range: {fmt(result['range_low'])} - {fmt(result['range_high'])}",
        f"Premium/Discount midpoint: {fmt(result['midpoint'])}",
        f"Sell-side sweep: {result['sweep_low']}",
        f"Buy-side sweep: {result['sweep_high']}",
        "FVG: " + (
            f"{result['fvg'][0]} {fmt(result['fvg'][1])}-{fmt(result['fvg'][2])}"
            if result["fvg"] else "Not detected"
        ),
        "Order Block: " + (
            f"{fmt(result['ob_long'][0])}-{fmt(result['ob_long'][1])}"
            if side == "LONG" and result["ob_long"]
            else f"{fmt(result['ob_short'][0])}-{fmt(result['ob_short'][1])}"
            if side == "SHORT" and result["ob_short"]
            else "Not detected"
        ),
        "Reasons: " + "; ".join(
            result["long_reasons"] if side == "LONG"
            else result["short_reasons"]
        ),
        "Risk: Signal guaranteed nahi. Pehle paper trading karein.",
        "Candle UTC: " + datetime.fromtimestamp(
            result["candle_ts"] / 1000,
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M")
    ]

    return "\n".join(lines)


def main():

    result = analyze()

    state = load_state()
    sent = state.setdefault("sent", {})

    current_early = result["early"]
    active_early = state.get("active_early")

    if current_early and current_early != active_early:

        send_telegram(make_message(result, "EARLY"))

        state["active_early"] = current_early

    elif not current_early:
        state["active_early"] = None

    if result["final"]:

        key = (
            f"{result['candle_ts']}:"
            f"CONFIRMED:{result['final']}"
        )

        if key not in sent:

            send_telegram(
                make_message(result, "CONFIRMED")
            )

            sent[key] = int(time.time())

    if len(sent) > 500:

        recent = sorted(
            sent.items(),
            key=lambda item: item[1]
        )[-300:]

        state["sent"] = dict(recent)

    save_state(state)


if __name__ == "__main__":

    try:
        main()

    except Exception:
        log.exception("Bot failed")
        raise
