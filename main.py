import os
import json
import time
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np

# ============================================================
# MASTER GOLD + TOP 10 CRYPTO SMC SIGNAL BOT
# Bitget USDT-M Futures | 4H / 1H / 15M / 5M
#
# Telegram sends ONLY final confirmed LONG/SHORT signals.
# EMA, Premium/Discount, SMC and momentum are analyzed internally.
# This is an analysis/alert bot; it does NOT place orders.
# ============================================================

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
STATE_FILE = os.getenv("STATE_FILE", "bot_state.json")

TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()

GOLD_SYMBOL = "XAUUSDT"
CRYPTO_TOP_N = 10
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "10000000"))
CANDLE_LIMIT = 240
MAX_WORKERS = 1  # Sequential scanning reduces Bitget HTTP 429s
HTTP_TIMEOUT = 20
RETRIES = 4

# Quality gates. Increase these for fewer, stricter signals.
MIN_TOTAL_SCORE = 8
MIN_VOLUME_RATIO = 1.15
MIN_ADX = 16
MIN_BODY_RATIO = 0.48
MIN_MOMENTUM_PCT_30M = 0.35
MIN_MOMENTUM_PCT_1H = 0.55
MIN_MOMENTUM_PCT_2H = 0.80

# Preferred coins are ranked first only if they are valid, online
# Bitget USDT-M crypto contracts. Remaining places are volume-ranked.
PREFERRED_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"
]

EXCLUDED_BASES = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDE", "DAI",
    "USDD", "UST", "EUR", "GBP", "JPY", "XAG"
}
EXCLUDED_SYMBOL_FRAGMENTS = (
    "3L", "3S", "5L", "5S", "2L", "2S", "BULL", "BEAR",
    "UPUSDT", "DOWNUSDT"
)
# Explicitly exclude tokenized stocks and non-crypto RWA instruments.
EXCLUDED_STOCK_BASES = {
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOG",
    "GOOGL", "SNDK", "SOXL", "SOXS", "SPCX", "KORU", "SNXX"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("master-smc-bot")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Master-SMC-Alert-Bot/2.1"})
API_MIN_INTERVAL = float(os.getenv("API_MIN_INTERVAL", "0.45"))
_API_LOCK = __import__("threading").Lock()
_LAST_API_CALL = 0.0


def api_get(path, params=None):
    """Rate-limited Bitget public GET with exponential retry/backoff."""
    global _LAST_API_CALL
    url = BASE_URL + path
    last_error = None
    for attempt in range(RETRIES):
        try:
            with _API_LOCK:
                now = time.monotonic()
                delay = API_MIN_INTERVAL - (now - _LAST_API_CALL)
                if delay > 0:
                    time.sleep(delay)
                _LAST_API_CALL = time.monotonic()
                response = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    wait = max(float(retry_after), 2 ** (attempt + 1))
                except (TypeError, ValueError):
                    wait = 2 ** (attempt + 1)
                raise requests.HTTPError(f"HTTP 429; retry delay {wait:.1f}s")
            if response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("code")) != "00000":
                raise RuntimeError(f"Bitget API error {payload.get('code')}: {payload.get('msg')}")
            return payload.get("data", [])
        except Exception as exc:
            last_error = exc
            wait = min(2 ** (attempt + 1), 30)
            log.warning("API retry %s/%s for %s: %s; waiting %.1fs",
                        attempt + 1, RETRIES, path, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Bitget request failed: {path}: {last_error}")


def get_contracts_and_tickers():
    contracts = api_get(
        "/api/v2/mix/market/contracts",
        {"productType": PRODUCT_TYPE}
    )
    tickers = api_get(
        "/api/v2/mix/market/tickers",
        {"productType": PRODUCT_TYPE}
    )
    return contracts, tickers


def is_valid_crypto_contract(contract):
    symbol = str(contract.get("symbol", "")).upper()
    base = str(contract.get("baseCoin", "")).upper()
    quote = str(contract.get("quoteCoin", "")).upper()
    status = str(contract.get("symbolStatus", contract.get("status", ""))).lower()

    if not symbol.endswith("USDT") or quote != "USDT" or not base:
        return False
    if status and status not in ("normal", "online", "listed"):
        return False
    if base in EXCLUDED_BASES or base in EXCLUDED_STOCK_BASES:
        return False
    if any(fragment in symbol for fragment in EXCLUDED_SYMBOL_FRAGMENTS):
        return False
    if base.startswith(("1000", "10000")):
        # Meme coins with contract multipliers are legitimate, so don't
        # exclude them just for a numeric prefix.
        pass
    contract_type = str(contract.get("支持合约类型", contract.get("contractType", ""))).lower()
    if contract_type and contract_type not in ("perpetual", "永续"):
        return False
    return True


def select_symbols():
    contracts, tickers = get_contracts_and_tickers()
    ticker_map = {str(t.get("symbol", "")).upper(): t for t in tickers if t.get("symbol")}
    valid_contracts = {
        str(c.get("symbol", "")).upper(): c
        for c in contracts if is_valid_crypto_contract(c)
    }

    gold_available = GOLD_SYMBOL in valid_contracts and GOLD_SYMBOL in ticker_map
    if not gold_available:
        log.error("%s is not available as a Bitget %s contract; no substitute will be used.",
                  GOLD_SYMBOL, PRODUCT_TYPE)

    # Strict whitelist prevents stocks, indices, commodities and unrelated contracts.
    selected = [s for s in PREFERRED_SYMBOLS if s in valid_contracts and s in ticker_map]
    missing = [s for s in PREFERRED_SYMBOLS if s not in selected]
    if missing:
        log.warning("Approved major crypto contracts unavailable: %s", ", ".join(missing))

    symbols = selected[:CRYPTO_TOP_N]
    if gold_available:
        symbols.insert(0, GOLD_SYMBOL)
    log.info("Selected %s symbols (target: 1 Gold + 10 crypto): %s",
             len(symbols), ", ".join(symbols))
    if len(selected) < CRYPTO_TOP_N:
        log.warning("Only %s/%s approved major crypto contracts are available.",
                    len(selected), CRYPTO_TOP_N)
    return symbols


def fetch_candles(symbol, granularity, limit=CANDLE_LIMIT):
    data = api_get(
        "/api/v2/mix/market/candles",
        {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "granularity": granularity,
            "limit": str(limit)
        }
    )
    if not data:
        raise RuntimeError(f"No candle data for {symbol} {granularity}")

    rows = []
    for row in data:
        if len(row) < 6:
            continue
        try:
            rows.append([
                int(row[0]), float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), float(row[5]),
                float(row[6]) if len(row) > 6 else 0.0
            ])
        except (TypeError, ValueError):
            continue

    df = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    )
    if df.empty:
        raise RuntimeError(f"Invalid candle data for {symbol} {granularity}")

    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # Remove the currently forming candle. All confirmations use closed candles.
    interval_ms = {
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "1H": 60 * 60_000,
        "4H": 4 * 60 * 60_000
    }[granularity]
    now_ms = int(time.time() * 1000)
    df = df[df["timestamp"] + interval_ms <= now_ms].copy()
    if len(df) < 80:
        raise RuntimeError(
            f"Not enough closed candles for {symbol} {granularity}: {len(df)}"
        )
    return add_indicators(df)


def add_indicators(df):
    df = df.copy()
    for period in (9, 21, 50, 200):
        df[f"ema{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_safe = df["atr"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["plus_di"] = plus_di.fillna(0)
    df["minus_di"] = minus_di.fillna(0)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)

    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["vol_sma20"].replace(0, np.nan)
    df["volume_ratio"] = df["volume_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["body_ratio"] = (
        (df["close"] - df["open"]).abs() /
        (df["high"] - df["low"]).replace(0, np.nan)
    ).fillna(0)
    return df


def ema_direction(df):
    last = df.iloc[-1]
    bullish = (
        last["close"] > last["ema50"] > last["ema200"]
        and last["ema9"] > last["ema21"]
    )
    bearish = (
        last["close"] < last["ema50"] < last["ema200"]
        and last["ema9"] < last["ema21"]
    )
    return "BULLISH" if bullish else "BEARISH" if bearish else "MIXED"


def market_structure(df, lookback=30):
    recent = df.iloc[-(lookback + 1):-1]
    last = df.iloc[-1]
    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())

    bos_up = float(last["close"]) > swing_high
    bos_down = float(last["close"]) < swing_low

    # CHOCH proxy: close breaks the opposite side of a recent local range.
    prev = df.iloc[-2]
    prior_high = float(df["high"].iloc[-(lookback + 2):-2].max())
    prior_low = float(df["low"].iloc[-(lookback + 2):-2].min())
    choch_up = float(last["close"]) > prior_high and float(prev["close"]) <= prior_high
    choch_down = float(last["close"]) < prior_low and float(prev["close"]) >= prior_low
    return {
        "bos_up": bos_up, "bos_down": bos_down,
        "choch_up": choch_up, "choch_down": choch_down,
        "swing_high": swing_high, "swing_low": swing_low
    }


def premium_discount(df, lookback=50):
    recent = df.iloc[-lookback:]
    high = float(recent["high"].max())
    low = float(recent["low"].min())
    midpoint = (high + low) / 2
    price = float(df.iloc[-1]["close"])
    if price < midpoint:
        zone = "DISCOUNT"
    elif price > midpoint:
        zone = "PREMIUM"
    else:
        zone = "EQUILIBRIUM"
    return {
        "zone": zone, "high": high, "low": low,
        "equilibrium": midpoint, "price": price
    }


def liquidity_analysis(df, lookback=25):
    recent = df.iloc[-lookback - 1:-1]
    last = df.iloc[-1]
    prior_high = float(recent["high"].max())
    prior_low = float(recent["low"].min())
    sweep_high = float(last["high"]) > prior_high and float(last["close"]) < prior_high
    sweep_low = float(last["low"]) < prior_low and float(last["close"]) > prior_low
    return {
        "sweep_high": sweep_high,
        "sweep_low": sweep_low,
        "equal_high": prior_high,
        "equal_low": prior_low
    }


def detect_fvg(df):
    if len(df) < 4:
        return None
    a = df.iloc[-3]
    c = df.iloc[-1]
    if float(c["low"]) > float(a["high"]):
        return {"type": "BULLISH_FVG", "low": float(a["high"]), "high": float(c["low"])}
    if float(c["high"]) < float(a["low"]):
        return {"type": "BEARISH_FVG", "low": float(c["high"]), "high": float(a["low"])}
    return None


def detect_order_block(df, side, lookback=12):
    # Simple last-opposite-candle zone before a strong directional close.
    recent = df.iloc[-(lookback + 1):-1]
    last = df.iloc[-1]
    body = abs(float(last["close"]) - float(last["open"]))
    atr = max(float(last["atr"]), 1e-12)
    if body < 0.65 * atr:
        return None

    if side == "LONG" and float(last["close"]) > float(last["open"]):
        opposite = recent[recent["close"] < recent["open"]]
    elif side == "SHORT" and float(last["close"]) < float(last["open"]):
        opposite = recent[recent["close"] > recent["open"]]
    else:
        return None

    if opposite.empty:
        return None
    candle = opposite.iloc[-1]
    return {
        "low": float(candle["low"]),
        "high": float(candle["high"])
    }


def momentum_analysis(df5):
    """Detect unusual 30m / 1h / 2h closed-candle price expansion."""
    last_price = float(df5.iloc[-1]["close"])
    results = {}
    for label, bars, threshold in (
        ("30M", 6, MIN_MOMENTUM_PCT_30M),
        ("1H", 12, MIN_MOMENTUM_PCT_1H),
        ("2H", 24, MIN_MOMENTUM_PCT_2H)
    ):
        if len(df5) <= bars:
            results[label] = {"direction": "NONE", "change_pct": 0.0, "strong": False}
            continue
        start_price = float(df5.iloc[-(bars + 1)]["close"])
        change_pct = ((last_price - start_price) / start_price) * 100 if start_price else 0
        window = df5.iloc[-bars:]
        vol_avg = float(window["volume"].mean())
        base_vol = float(df5["volume"].iloc[-(bars + 21):-bars].mean()) if len(df5) > bars + 20 else 0
        vol_expansion = (vol_avg / base_vol) if base_vol > 0 else 1.0
        direction = "UP" if change_pct > 0 else "DOWN" if change_pct < 0 else "NONE"
        # Threshold is volatility-aware: a high ATR market needs a larger move.
        atr_pct = (float(df5.iloc[-1]["atr"]) / last_price * 100) if last_price else 0
        adaptive_threshold = max(threshold, atr_pct * (1.1 if label == "30M" else 1.5))
        strong = abs(change_pct) >= adaptive_threshold and vol_expansion >= 1.05
        results[label] = {
            "direction": direction,
            "change_pct": change_pct,
            "volume_expansion": vol_expansion,
            "strong": strong
        }
    return results


def candle_confirmation(df5, side):
    last = df5.iloc[-1]
    prev = df5.iloc[-2]
    body_ratio = float(last["body_ratio"])
    vol_ratio = float(last["volume_ratio"])
    atr = max(float(last["atr"]), 1e-12)
    body = abs(float(last["close"]) - float(last["open"]))
    if side == "LONG":
        bullish = float(last["close"]) > float(last["open"])
        engulf = (
            bullish and float(prev["close"]) < float(prev["open"])
            and float(last["close"]) >= float(prev["open"])
            and float(last["open"]) <= float(prev["close"])
        )
        breakout = float(last["close"]) > float(df5["high"].iloc[-7:-1].max())
        continuation = bullish and float(last["close"]) > float(prev["high"])
        return (
            bullish and body_ratio >= MIN_BODY_RATIO and body >= 0.45 * atr
            and (engulf or breakout or continuation)
            and vol_ratio >= MIN_VOLUME_RATIO
        )
    bearish = float(last["close"]) < float(last["open"])
    engulf = (
        bearish and float(prev["close"]) > float(prev["open"])
        and float(last["close"]) <= float(prev["open"])
        and float(last["open"]) >= float(prev["close"])
    )
    breakout = float(last["close"]) < float(df5["low"].iloc[-7:-1].min())
    continuation = bearish and float(last["close"]) < float(prev["low"])
    return (
        bearish and body_ratio >= MIN_BODY_RATIO and body >= 0.45 * atr
        and (engulf or breakout or continuation)
        and vol_ratio >= MIN_VOLUME_RATIO
    )


def build_trade_levels(df5, side):
    last = df5.iloc[-1]
    entry = float(last["close"])
    atr = max(float(last["atr"]), entry * 0.0005)
    risk = max(1.35 * atr, entry * 0.0015)
    if side == "LONG":
        sl = entry - risk
        return {
            "entry": entry, "area_low": entry - 0.20 * atr,
            "area_high": entry + 0.15 * atr, "sl": sl,
            "tp1": entry + risk, "tp2": entry + 2 * risk,
            "tp3": entry + 3 * risk
        }
    sl = entry + risk
    return {
        "entry": entry, "area_low": entry - 0.15 * atr,
        "area_high": entry + 0.20 * atr, "sl": sl,
        "tp1": entry - risk, "tp2": entry - 2 * risk,
        "tp3": entry - 3 * risk
    }


def analyze_symbol(symbol):
    try:
        # Four independent timeframe datasets; all use closed candles.
        df4 = fetch_candles(symbol, "4H")
        df1 = fetch_candles(symbol, "1H")
        df15 = fetch_candles(symbol, "15m")
        df5 = fetch_candles(symbol, "5m")

        trend4 = ema_direction(df4)
        trend1 = ema_direction(df1)
        structure15 = market_structure(df15)
        pd_zone = premium_discount(df15)
        liquidity = liquidity_analysis(df15)
        momentum = momentum_analysis(df5)
        last5 = df5.iloc[-1]

        # Both higher timeframes must agree. No 5m-only countertrend entries.
        if trend4 == "BULLISH" and trend1 == "BULLISH":
            side = "LONG"
        elif trend4 == "BEARISH" and trend1 == "BEARISH":
            side = "SHORT"
        else:
            return None

        # 15m structure must agree with the higher timeframe.
        if side == "LONG" and not (structure15["bos_up"] or structure15["choch_up"]):
            return None
        if side == "SHORT" and not (structure15["bos_down"] or structure15["choch_down"]):
            return None

        # 5m closed candle confirmation is mandatory.
        if not candle_confirmation(df5, side):
            return None

        # Strong-move detector: allow either a strong 30m/1h/2h impulse
        # OR a robust 5m volume/body expansion. Momentum alone is never an entry.
        momentum_aligned = any(
            item["strong"] and (
                (side == "LONG" and item["direction"] == "UP")
                or (side == "SHORT" and item["direction"] == "DOWN")
            )
            for item in momentum.values()
        )
        strong_5m = (
            float(last5["volume_ratio"]) >= 1.50
            and float(last5["body_ratio"]) >= 0.62
            and abs(float(last5["close"]) - float(last5["open"])) >= 0.65 * float(last5["atr"])
        )
        if not (momentum_aligned or strong_5m):
            return None

        score = 0
        score += 2  # 4H and 1H agree
        score += 2  # 15m BOS/CHOCH
        score += 2  # 5m candle confirmation
        score += 1 if float(last5["adx"]) >= MIN_ADX else 0
        score += 1 if float(last5["volume_ratio"]) >= MIN_VOLUME_RATIO else 0
        score += 1 if momentum_aligned else 0
        score += 1 if (
            (side == "LONG" and liquidity["sweep_low"])
            or (side == "SHORT" and liquidity["sweep_high"])
        ) else 0
        score += 1 if (
            (side == "LONG" and pd_zone["zone"] == "DISCOUNT")
            or (side == "SHORT" and pd_zone["zone"] == "PREMIUM")
        ) else 0

        if score < MIN_TOTAL_SCORE:
            return None

        # Avoid chasing a move after it is already overextended.
        price = float(last5["close"])
        atr5 = max(float(last5["atr"]), 1e-12)
        ema21 = float(last5["ema21"])
        if abs(price - ema21) > 2.8 * atr5:
            return None

        fvg = detect_fvg(df15)
        ob = detect_order_block(df15, side)
        levels = build_trade_levels(df5, side)

        return {
            "symbol": symbol,
            "side": side,
            "score": score,
            "entry": levels["entry"],
            "area_low": levels["area_low"],
            "area_high": levels["area_high"],
            "sl": levels["sl"],
            "tp1": levels["tp1"],
            "tp2": levels["tp2"],
            "tp3": levels["tp3"],
            "candle_ts": int(last5["timestamp"]),
            "trend4": trend4,
            "trend1": trend1,
            "pd_zone": pd_zone["zone"],
            "momentum": momentum,
            "fvg": fvg,
            "ob": ob
        }
    except Exception as exc:
        log.exception("Analysis failed for %s: %s", symbol, exc)
        return None


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("State file could not be read; starting empty: %s", exc)
        return {}


def save_state(state):
    temp_file = STATE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(temp_file, STATE_FILE)


def fmt_price(value):
    value = float(value)
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:.5f}".rstrip("0").rstrip(".")
    return f"{value:.8f}".rstrip("0").rstrip(".")


def build_telegram_message(signal):
    direction = "🟢 LONG" if signal["side"] == "LONG" else "🔴 SHORT"
    return (
        f"✅ FINAL CONFIRMED {direction}\n\n"
        f"Coin: {signal['symbol']}\n"
        f"Entry Area: {fmt_price(signal['area_low'])} - {fmt_price(signal['area_high'])}\n"
        f"Entry: {fmt_price(signal['entry'])}\n"
        f"Stop Loss: {fmt_price(signal['sl'])}\n"
        f"TP1: {fmt_price(signal['tp1'])}\n"
        f"TP2: {fmt_price(signal['tp2'])}\n"
        f"TP3: {fmt_price(signal['tp3'])}\n\n"
        f"Signal Score: {signal['score']}/11\n"
        f"Confirmation: 4H + 1H trend, 15M structure, 5M closed candle\n"
        f"Risk note: Signal is an algorithmic estimate, not a guarantee."
    )


def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials missing; signal will be logged only.")
        log.info("\n%s", message)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = SESSION.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "disable_web_page_preview": True
            },
            timeout=HTTP_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(str(payload))
        return True
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)
        return False


def main():
    log.info("Starting Master Gold + Top 10 Crypto SMC Bot")
    symbols = select_symbols()
    if not symbols:
        log.error("No eligible symbols found. Exiting.")
        return

    state = load_state()
    signals = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    signals.append(result)
                    log.info("Confirmed candidate: %s %s score=%s",
                             symbol, result["side"], result["score"])
                else:
                    log.info("No final confirmation: %s", symbol)
            except Exception as exc:
                log.exception("Worker error for %s: %s", symbol, exc)

    # Highest-quality candidates first. State key prevents repeats on the
    # same closed 5m candle for the same symbol and direction.
    signals.sort(key=lambda item: item["score"], reverse=True)
    sent_count = 0
    for signal in signals:
        key = f"{signal['symbol']}:{signal['side']}"
        candle_key = str(signal["candle_ts"])
        if state.get(key) == candle_key:
            log.info("Duplicate skipped: %s %s", signal["symbol"], signal["side"])
            continue

        message = build_telegram_message(signal)
        if send_telegram(message):
            state[key] = candle_key
            save_state(state)
            sent_count += 1
            log.info("Telegram signal sent: %s %s",
                     signal["symbol"], signal["side"])

    log.info(
        "Scan complete. Available symbols scanned=%s | Confirmed=%s | Sent=%s",
        len(symbols), len(signals), sent_count
    )


if __name__ == "__main__":
    main()
