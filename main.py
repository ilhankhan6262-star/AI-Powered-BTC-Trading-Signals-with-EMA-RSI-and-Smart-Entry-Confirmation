import requests
import pandas as pd
import os

# BTC STRONG ALERT BOT

SYMBOL = "BTCUSDT"
TIMEFRAMES = ["4h", "1h", "15m"]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def get_data(timeframe):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": SYMBOL,
        "interval": timeframe,
        "limit": 250
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_buy", "taker_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df


def analyze(df):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["EMA200"] = df["close"].ewm(span=200).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VOL_AVG"] = df["volume"].rolling(20).mean()

    return df.iloc[-2]


def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram credentials missing")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    requests.post(url, data={
        "chat_id": CHAT_ID,
        "text": message
    }, timeout=20).raise_for_status()


def main():
    results = {}

    for tf in TIMEFRAMES:
        df = get_data(tf)
        results[tf] = analyze(df)

    h4 = results["4h"]
    h1 = results["1h"]
    m15 = results["15m"]

    price = float(m15["close"])

    bullish = (
        h4["EMA20"] > h4["EMA50"] > h4["EMA200"]
        and h1["EMA20"] > h1["EMA50"]
        and m15["close"] > m15["EMA20"]
        and 50 < m15["RSI"] < 68
        and m15["volume"] > m15["VOL_AVG"]
    )

    bearish = (
        h4["EMA20"] < h4["EMA50"] < h4["EMA200"]
        and h1["EMA20"] < h1["EMA50"]
        and m15["close"] < m15["EMA20"]
        and 32 < m15["RSI"] < 50
        and m15["volume"] > m15["VOL_AVG"]
    )

    if bullish or bearish:
        direction = "LONG" if bullish else "SHORT"

        atr = (get_data("15m")["high"] -
               get_data("15m")["low"]).rolling(14).mean().iloc[-2]

        risk = max(float(atr), price * 0.001)

        if bullish:
            sl = price - risk * 1.5
            tp1 = price + risk
            tp2 = price + risk * 2
            tp3 = price + risk * 3
        else:
            sl = price + risk * 1.5
            tp1 = price - risk
            tp2 = price - risk * 2
            tp3 = price - risk * 3

        message = f"""
🚨 BTC STRONG ALERT

Direction: {direction}
Entry: {price:.2f}
Stop Loss: {sl:.2f}

TP1: {tp1:.2f}
TP2: {tp2:.2f}
TP3: {tp3:.2f}

EMA 20/50/200: Checked
RSI 14: {m15['RSI']:.2f}
Volume: Confirmed

Timeframes: 4H + 1H + 15M

⚠️ Signal only. Not financial advice.
"""
        send_telegram(message)
        print(message)

    else:
        print("No confirmed setup. WAIT.")


if __name__ == "__main__":
    main()
