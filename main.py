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
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))  # Keep at 1 to avoid API bursts
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
API_MIN_INTERVAL = float(os.getenv("API_MIN_INTERVAL", "0.60"))
_API_LOCK = __import__("threading").Lock()
_LAST_API_CALL = 0.0


def api_get(path, params=None):
    """Rate-limited Bitget GET. Retry only transient failures, not bad parameters."""
    global _LAST_API_CALL
    url = BASE_URL + path
    last_error = None

    for attempt in range(RETRIES):
        retry_wait = min(2 ** (attempt + 1), 30)
        try:
            with _API_LOCK:
                now = time.monotonic()
                delay = API_MIN_INTERVAL - (now - _LAST_API_CALL)
                if delay > 0:
                    time.sleep(delay)
                _LAST_API_CALL = time.monotonic()
                response = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)

            # A 400 is normally an invalid parameter/symbol/interval. Retrying
            # the identical request cannot fix it, so expose the actual response.
            if response.status_code == 400:
                raise ValueError(
                    f"Bitget HTTP 400 (bad request) for {path}; "
                    f"params={params}; response={response.text[:500]}"
                )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    retry_wait = max(float(retry_after), retry_wait) if retry_after else retry_wait
                except (TypeError, ValueError):
                    pass
                raise requests.HTTPError(
                    f"Bitget HTTP 429 rate limit; retrying after {retry_wait:.1f}s"
                )
            if response.status_code in (500, 502, 503, 504):
                raise requests.HTTPError(
                    f"Bitget HTTP {response.status_code}: {response.text[:300]}"
                )

            response.raise_for_status()
            payload = response.json()
            if str(payload.get("code")) != "00000":
                code = str(payload.get("code"))
                message = str(payload.get("msg"))
                # Bitget parameter errors are permanent; do not hammer endpoint.
                if code.startswith("4"):
                    raise ValueError(f"Bitget API parameter error {code}: {message}; params={params}")
                raise RuntimeError(f"Bitget API error {code}: {message}")
            return payload.get("data", [])

        except ValueError:
            raise
        except Exception as exc:
            last_error = exc
            log.warning(
                "API retry %s/%s for %s: %s; waiting %.1fs",
                attempt + 1, RETRIES, path, exc, retry_wait
            )
            if attempt + 1 < RETRIES:
                time.sleep(retry_wait)

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
    """Select XAUUSDT plus the 10 highest-24h-turnover eligible crypto contracts."""
    contracts, tickers = get_contracts_and_tickers()
    ticker_map = {str(t.get("symbol", "")).upper(): t for t in tickers if t.get("symbol")}
    contract_map = {
        str(c.get("symbol", "")).upper(): c
        for c in contracts if c.get("symbol")
    }

    # Gold is checked separately and never counted among the 10 crypto slots.
    gold_contract = contract_map.get(GOLD_SYMBOL)
    gold_available = bool(
        gold_contract and GOLD_SYMBOL in ticker_map
        and str(gold_contract.get("quoteCoin", "")).upper() == "USDT"
    )
    if not gold_available:
        log.error("%s is not available as a Bitget %s contract; it will be skipped.",
                  GOLD_SYMBOL, PRODUCT_TYPE)

    ranked = []
    for symbol, contract in contract_map.items():
        if symbol == GOLD_SYMBOL or not is_valid_crypto_contract(contract):
            continue
        ticker = ticker_map.get(symbol)
        if not ticker:
            continue
        try:
            quote_volume = float(ticker.get("quoteVolume", 0) or 0)
            last_price = float(ticker.get("lastPr", 0) or 0)
        except (TypeError, ValueError):
            continue
        if last_price <= 0 or quote_volume < MIN_QUOTE_VOLUME:
            continue
        ranked.append((symbol, quote_volume))

    ranked.sort(key=lambda item: item[1], reverse=True)
    crypto_symbols = [symbol for symbol, _ in ranked[:CRYPTO_TOP_N]]

    # If the volume floor leaves fewer than ten, fill remaining slots from
    # the next most liquid eligible live crypto contracts (still not stocks,
    # stablecoins, leveraged tokens, silver, or Gold).
    if len(crypto_symbols) < CRYPTO_TOP_N:
        all_ranked = []
        for symbol, contract in contract_map.items():
            if symbol == GOLD_SYMBOL or not is_valid_crypto_contract(contract):
                continue
            ticker = ticker_map.get(symbol)
            if not ticker or symbol in crypto_symbols:
                continue
            try:
                volume = float(ticker.get("quoteVolume", 0) or 0)
                price = float(ticker.get("lastPr", 0) or 0)
            except (TypeError, ValueError):
                continue
            if price > 0 and volume > 0:
                all_ranked.append((symbol, volume))
        all_ranked.sort(key=lambda item: item[1], reverse=True)
        for symbol, _ in all_ranked:
            if len(crypto_symbols) >= CRYPTO_TOP_N:
                break
            crypto_symbols.append(symbol)

    symbols = ([GOLD_SYMBOL] if gold_available else []) + crypto_symbols
    log.info("Selected %s symbols (target: 1 Gold + 10 crypto): %s",
             len(symbols), ", ".join(symbols))
    log.info("Crypto selected by 24h quote volume: %s/%s",
             len(crypto_symbols), CRYPTO_TOP_N)
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
    if len(df) < 205:
        raise RuntimeError(
            f"Not enough closed candles for {symbol} {granularity}: {len(df)}; need 205 for EMA200"
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


def rapid_move_retest_confirmation(df5, side):
    """Confirm a fast impulse by a closed 5m retest or controlled continuation.

    This avoids chasing an extended impulse candle. The breakout level is the
    prior 6-candle range; a retest must touch/approach that level and close back
    on the breakout side. A fresh continuation breakout is also accepted.
    """
    if len(df5) < 30:
        return False, "insufficient 5M history"

    last = df5.iloc[-1]
    prev = df5.iloc[-2]
    prior = df5.iloc[-7:-1]
    level_high = float(prior["high"].max())
    level_low = float(prior["low"].min())
    close = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    atr = max(float(last["atr"]), 1e-12)
    body_ratio = float(last["body_ratio"])
    volume_ratio = float(last["volume_ratio"])

    if side == "LONG":
        touched_retest = low <= level_high + 0.20 * atr and close > level_high
        continuation = (
            close > level_high and close > float(last["open"])
            and body_ratio >= 0.55 and volume_ratio >= 1.10
            and close > float(prev["high"])
        )
        return (touched_retest or continuation), ("5M retest" if touched_retest else "5M continuation" if continuation else "no 5M retest/continuation")

    touched_retest = high >= level_low - 0.20 * atr and close < level_low
    continuation = (
        close < level_low and close < float(last["open"])
        and body_ratio >= 0.55 and volume_ratio >= 1.10
        and close < float(prev["low"])
    )
    return (touched_retest or continuation), ("5M retest" if touched_retest else "5M continuation" if continuation else "no 5M retest/continuation")


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
        # All decisions are based on completed candles only.
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

        # Determine directional structure from higher timeframes + 15m BOS/CHOCH.
        long_structure = structure15["bos_up"] or structure15["choch_up"]
        short_structure = structure15["bos_down"] or structure15["choch_down"]
        long_momentum = any(x["strong"] and x["direction"] == "UP" for x in momentum.values())
        short_momentum = any(x["strong"] and x["direction"] == "DOWN" for x in momentum.values())
        htf_long_count = int(trend4 == "BULLISH") + int(trend1 == "BULLISH")
        htf_short_count = int(trend4 == "BEARISH") + int(trend1 == "BEARISH")

        # Normal route: both HTFs agree. Rapid route: at least one HTF agrees,
        # 15m structure agrees, and a strong 30m/1h/2h impulse is present.
        if htf_long_count == 2 and long_structure:
            side, rapid_route = "LONG", False
        elif htf_short_count == 2 and short_structure:
            side, rapid_route = "SHORT", False
        elif htf_long_count >= 1 and long_structure and long_momentum:
            side, rapid_route = "LONG", True
        elif htf_short_count >= 1 and short_structure and short_momentum:
            side, rapid_route = "SHORT", True
        else:
            return None

        # Standard 5m candle confirmation is required in both routes.
        standard_5m = candle_confirmation(df5, side)
        rapid_5m, rapid_reason = rapid_move_retest_confirmation(df5, side)
        if rapid_route:
            if not (standard_5m or rapid_5m):
                return None
        elif not standard_5m:
            return None

        momentum_aligned = long_momentum if side == "LONG" else short_momentum
        strong_5m = (
            float(last5["volume_ratio"]) >= 1.50
            and float(last5["body_ratio"]) >= 0.62
            and abs(float(last5["close"]) - float(last5["open"])) >= 0.65 * float(last5["atr"])
        )
        if not (momentum_aligned or strong_5m):
            return None

        # Score: rapid route has a slightly lower HTF score, but must still
        # pass 15m structure, momentum and 5m confirmation gates.
        score = 0
        score += 2 if htf_long_count == 2 and side == "LONG" or htf_short_count == 2 and side == "SHORT" else 1
        score += 2  # aligned 15m BOS/CHOCH
        score += 2 if standard_5m else 1  # closed 5m confirmation / retest
        score += 1 if float(last5["adx"]) >= MIN_ADX else 0
        score += 1 if float(last5["volume_ratio"]) >= MIN_VOLUME_RATIO else 0
        score += 1 if momentum_aligned else 0
        score += 1 if ((side == "LONG" and liquidity["sweep_low"]) or (side == "SHORT" and liquidity["sweep_high"])) else 0
        score += 1 if ((side == "LONG" and pd_zone["zone"] == "DISCOUNT") or (side == "SHORT" and pd_zone["zone"] == "PREMIUM")) else 0

        required_score = MIN_TOTAL_SCORE - (1 if rapid_route else 0)
        if score < required_score:
            return None

        # Do not chase a move that is already too far from its short EMA.
        price = float(last5["close"])
        atr5 = max(float(last5["atr"]), 1e-12)
        ema21 = float(last5["ema21"])
        max_extension = 3.5 if rapid_route else 2.8
        if abs(price - ema21) > max_extension * atr5:
            return None

        fvg = detect_fvg(df15)
        ob = detect_order_block(df15, side)
        levels = build_trade_levels(df5, side)
        return {
            "symbol": symbol, "side": side, "score": score,
            "entry": levels["entry"], "area_low": levels["area_low"],
            "area_high": levels["area_high"], "sl": levels["sl"],
            "tp1": levels["tp1"], "tp2": levels["tp2"], "tp3": levels["tp3"],
            "candle_ts": int(last5["timestamp"]),
            "trend4": trend4, "trend1": trend1,
            "ema_values": {
                "4H": {f"EMA{n}": float(df4.iloc[-1][f"ema{n}"]) for n in (9, 21, 50, 200)},
                "1H": {f"EMA{n}": float(df1.iloc[-1][f"ema{n}"]) for n in (9, 21, 50, 200)},
                "15M": {f"EMA{n}": float(df15.iloc[-1][f"ema{n}"]) for n in (9, 21, 50, 200)},
                "5M": {f"EMA{n}": float(df5.iloc[-1][f"ema{n}"]) for n in (9, 21, 50, 200)}
            },
            "pd_zone": pd_zone["zone"], "momentum": momentum,
            "fvg": fvg, "ob": ob,
            "entry_mode": "RAPID RETEST/CONTINUATION" if rapid_route else "STANDARD SMC",
            "confirmation": rapid_reason if rapid_route else "5M closed-candle confirmation"
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
        f"Entry Mode: {signal.get('entry_mode', 'STANDARD SMC')}\n"
        f"Confirmation: {signal.get('confirmation', '5M closed-candle confirmation')}\n"
        f"4H Trend: {signal['trend4']} | 1H Trend: {signal['trend1']}\n"
        f"Premium/Discount: {signal['pd_zone']}\n"
        f"EMA 4H: 9={fmt_price(signal['ema_values']['4H']['EMA9'])} | 21={fmt_price(signal['ema_values']['4H']['EMA21'])} | 50={fmt_price(signal['ema_values']['4H']['EMA50'])} | 200={fmt_price(signal['ema_values']['4H']['EMA200'])}\n"
        f"EMA 1H: 9={fmt_price(signal['ema_values']['1H']['EMA9'])} | 21={fmt_price(signal['ema_values']['1H']['EMA21'])} | 50={fmt_price(signal['ema_values']['1H']['EMA50'])} | 200={fmt_price(signal['ema_values']['1H']['EMA200'])}\n"
        f"EMA 15M: 9={fmt_price(signal['ema_values']['15M']['EMA9'])} | 21={fmt_price(signal['ema_values']['15M']['EMA21'])} | 50={fmt_price(signal['ema_values']['15M']['EMA50'])} | 200={fmt_price(signal['ema_values']['15M']['EMA200'])}\n"
        f"EMA 5M: 9={fmt_price(signal['ema_values']['5M']['EMA9'])} | 21={fmt_price(signal['ema_values']['5M']['EMA21'])} | 50={fmt_price(signal['ema_values']['5M']['EMA50'])} | 200={fmt_price(signal['ema_values']['5M']['EMA200'])}\n"
        f"Momentum: 30M {signal['momentum']['30M']['change_pct']:+.2f}% | "
        f"1H {signal['momentum']['1H']['change_pct']:+.2f}% | "
        f"2H {signal['momentum']['2H']['change_pct']:+.2f}%\n\n"
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
    log.info("API pacing=%.2fs | workers=1 | candle confirmation=closed 5M", API_MIN_INTERVAL)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials are missing; alerts cannot be delivered until repository secrets are set.")
    symbols = select_symbols()
    if not symbols:
        log.error("No eligible symbols found. Exiting.")
        return

    state = load_state()
    signals = []

    with ThreadPoolExecutor(max_workers=1) as executor:
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
