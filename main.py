import os
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np

# ============================================================
# MASTER GOLD + TOP 10 CRYPTO SMC SIGNAL BOT
# Bitget USDT-M Futures | 4H / 1H / 15M / 5M
#
# Telegram: ONLY final confirmed LONG/SHORT signals.
# Diagnostics: every rejected symbol logs the exact failed gate.
# This bot analyzes and sends alerts; it does NOT place orders.
# ============================================================

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
STATE_FILE = os.getenv("STATE_FILE", "bot_state.json")

TELEGRAM_BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or ""
).strip()
TELEGRAM_CHAT_ID = (
    os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or ""
).strip()

GOLD_SYMBOL = "XAUUSDT"
CRYPTO_TOP_N = 10
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "10000000"))
CANDLE_LIMIT = 240
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
HTTP_TIMEOUT = 20
RETRIES = 4

# Signal quality filters
MIN_TOTAL_SCORE = 8
MIN_VOLUME_RATIO = 1.15
MIN_ADX = 16
MIN_BODY_RATIO = 0.48
MIN_MOMENTUM_PCT_30M = 0.35
MIN_MOMENTUM_PCT_1H = 0.55
MIN_MOMENTUM_PCT_2H = 0.80

# These are preferred liquid crypto symbols, but the final selection is
# still limited to valid Bitget USDT-M crypto contracts and ranked by volume.
PREFERRED_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"
]

EXCLUDED_BASES = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDE", "DAI",
    "USDD", "UST", "EUR", "GBP", "JPY", "XAG"
}

# Explicitly exclude leveraged tokens and common leveraged-token suffixes.
EXCLUDED_SYMBOL_FRAGMENTS = (
    "3LUSDT", "3SUSDT", "5LUSDT", "5SUSDT", "2LUSDT", "2SUSDT",
    "BULLUSDT", "BEARUSDT", "UPUSDT", "DOWNUSDT"
)

# Known stock/equity-linked or tokenized-equity bases. This list is
# intentionally separate from ordinary crypto assets.
EXCLUDED_STOCK_BASES = {
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOG",
    "GOOGL", "SNDK", "SOXL", "SOXS", "SPCX", "KORU", "SNXX",
    "MU", "SKHY", "SKHYNIX"
}

# Metadata text containing these terms is treated as non-crypto/equity-linked.
NON_CRYPTO_METADATA_TERMS = (
    "tokenized stock", "tokenised stock", "stock token", "equity token",
    "share token", "stock", "equity", "shares", "etf", "leveraged etf"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("master-smc-bot")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Master-SMC-Alert-Bot/2.2"})
API_MIN_INTERVAL = float(os.getenv("API_MIN_INTERVAL", "0.60"))
_API_LOCK = threading.Lock()
_LAST_API_CALL = 0.0


def api_get(path, params=None):
    """Rate-limited Bitget GET with retries for transient failures."""
    global _LAST_API_CALL
    url = BASE_URL + path
    last_error = None

    for attempt in range(RETRIES):
        retry_wait = min(2 ** (attempt + 1), 30)
        try:
            with _API_LOCK:
                delay = API_MIN_INTERVAL - (time.monotonic() - _LAST_API_CALL)
                if delay > 0:
                    time.sleep(delay)
                _LAST_API_CALL = time.monotonic()
                response = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)

            if response.status_code == 400:
                raise ValueError(
                    f"Bitget HTTP 400 for {path}; params={params}; "
                    f"response={response.text[:500]}"
                )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    if retry_after:
                        retry_wait = max(float(retry_after), retry_wait)
                except (TypeError, ValueError):
                    pass
                raise requests.HTTPError(
                    f"Bitget HTTP 429 rate limit; retry after {retry_wait:.1f}s"
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
                if code.startswith("4"):
                    raise ValueError(
                        f"Bitget API parameter error {code}: {message}; params={params}"
                    )
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
    """Reject non-USDT, inactive, stablecoin, leveraged and stock-like contracts."""
    symbol = str(contract.get("symbol", "")).upper()
    base = str(contract.get("baseCoin", "")).upper()
    quote = str(contract.get("quoteCoin", "")).upper()
    status = str(
        contract.get("symbolStatus", contract.get("status", ""))
    ).lower()

    if not symbol.endswith("USDT") or quote != "USDT" or not base:
        return False
    if status and status not in ("normal", "online", "listed"):
        return False
    if base in EXCLUDED_BASES or base in EXCLUDED_STOCK_BASES:
        return False
    if any(symbol.endswith(fragment) for fragment in EXCLUDED_SYMBOL_FRAGMENTS):
        return False

    # Inspect descriptive contract metadata when Bitget supplies it.
    metadata_fields = (
        "symbolName", "symbolDisplayName", "baseCoinName",
        "underlying", "underlyingType", "category", "assetType",
        "marginCoin"
    )
    metadata = " ".join(
        str(contract.get(field, "")) for field in metadata_fields
    ).lower()
    if any(term in metadata for term in NON_CRYPTO_METADATA_TERMS):
        return False

    contract_type = str(
        contract.get("contractType", contract.get("支持合约类型", ""))
    ).lower()
    if contract_type and contract_type not in ("perpetual", "永续"):
        return False

    return True


def select_symbols():
    """Select XAUUSDT plus 10 eligible crypto contracts by 24h quote volume."""
    contracts, tickers = get_contracts_and_tickers()
    ticker_map = {
        str(t.get("symbol", "")).upper(): t
        for t in tickers if t.get("symbol")
    }
    contract_map = {
        str(c.get("symbol", "")).upper(): c
        for c in contracts if c.get("symbol")
    }

    gold_contract = contract_map.get(GOLD_SYMBOL)
    gold_available = bool(
        gold_contract and GOLD_SYMBOL in ticker_map
        and str(gold_contract.get("quoteCoin", "")).upper() == "USDT"
    )
    if not gold_available:
        log.error(
            "%s is not available as a Bitget %s contract; it will be skipped.",
            GOLD_SYMBOL, PRODUCT_TYPE
        )

    ranked = []
    excluded_stock_like = []
    for symbol, contract in contract_map.items():
        if symbol == GOLD_SYMBOL:
            continue

        base = str(contract.get("baseCoin", "")).upper()
        if base in EXCLUDED_STOCK_BASES:
            excluded_stock_like.append(symbol)
            continue
        if not is_valid_crypto_contract(contract):
            continue

        ticker = ticker_map.get(symbol)
        if not ticker:
            continue
        try:
            quote_volume = float(ticker.get("quoteVolume", 0) or 0)
            last_price = float(ticker.get("lastPr", 0) or 0)
        except (TypeError, ValueError):
            continue

        if last_price > 0 and quote_volume >= MIN_QUOTE_VOLUME:
            ranked.append((symbol, quote_volume))

    ranked.sort(key=lambda item: item[1], reverse=True)
    crypto_symbols = [symbol for symbol, _ in ranked[:CRYPTO_TOP_N]]

    # If fewer than ten meet the volume floor, fill from the remaining
    # eligible live contracts in descending volume order.
    if len(crypto_symbols) < CRYPTO_TOP_N:
        fallback = []
        for symbol, contract in contract_map.items():
            if symbol == GOLD_SYMBOL or symbol in crypto_symbols:
                continue
            if not is_valid_crypto_contract(contract):
                continue
            ticker = ticker_map.get(symbol)
            if not ticker:
                continue
            try:
                volume = float(ticker.get("quoteVolume", 0) or 0)
                price = float(ticker.get("lastPr", 0) or 0)
            except (TypeError, ValueError):
                continue
            if price > 0 and volume > 0:
                fallback.append((symbol, volume))

        fallback.sort(key=lambda item: item[1], reverse=True)
        for symbol, _ in fallback:
            if len(crypto_symbols) >= CRYPTO_TOP_N:
                break
            crypto_symbols.append(symbol)

    symbols = ([GOLD_SYMBOL] if gold_available else []) + crypto_symbols
    log.info(
        "Selected %s symbols (target: 1 Gold + 10 crypto): %s",
        len(symbols), ", ".join(symbols)
    )
    log.info(
        "Eligible crypto selected: %s/%s | volume floor: %s",
        len(crypto_symbols), CRYPTO_TOP_N, MIN_QUOTE_VOLUME
    )
    if excluded_stock_like:
        log.info(
            "Excluded known stock/equity-like contracts: %s",
            ", ".join(sorted(set(excluded_stock_like)))
        )
    if len(crypto_symbols) < CRYPTO_TOP_N:
        log.warning(
            "Only %s eligible crypto contracts found; Bitget may not have "
            "10 qualifying contracts under current filters.",
            len(crypto_symbols)
        )
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
        rows,
        columns=[
            "timestamp", "open", "high", "low",
            "close", "volume", "turnover"
        ]
    )
    if df.empty:
        raise RuntimeError(f"Invalid candle data for {symbol} {granularity}")

    df = (
        df.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

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
            f"Not enough closed candles for {symbol} {granularity}: "
            f"{len(df)}; need 205 for EMA200"
        )
    return add_indicators(df)


def add_indicators(df):
    df = df.copy()
    for period in (9, 21, 50, 200):
        df[f"ema{period}"] = df["close"].ewm(
            span=period, adjust=False
        ).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0), 0.0
    )
    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0), 0.0
    )
    atr_safe = df["atr"].replace(0, np.nan)
    plus_di = (
        100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    )
    minus_di = (
        100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    )
    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )
    df["plus_di"] = plus_di.fillna(0)
    df["minus_di"] = minus_di.fillna(0)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)

    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = (
        df["volume"] / df["vol_sma20"].replace(0, np.nan)
    )
    df["volume_ratio"] = (
        df["volume_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (
        (df["close"] - df["open"]).abs() / candle_range
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

    prev = df.iloc[-2]
    prior_high = float(df["high"].iloc[-(lookback + 2):-2].max())
    prior_low = float(df["low"].iloc[-(lookback + 2):-2].min())
    choch_up = (
        float(last["close"]) > prior_high
        and float(prev["close"]) <= prior_high
    )
    choch_down = (
        float(last["close"]) < prior_low
        and float(prev["close"]) >= prior_low
    )
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
    zone = (
        "DISCOUNT" if price < midpoint
        else "PREMIUM" if price > midpoint
        else "EQUILIBRIUM"
    )
    return {
        "zone": zone, "high": high, "low": low,
        "equilibrium": midpoint, "price": price
    }


def liquidity_analysis(df, lookback=25):
    recent = df.iloc[-lookback - 1:-1]
    last = df.iloc[-1]
    prior_high = float(recent["high"].max())
    prior_low = float(recent["low"].min())
    return {
        "sweep_high": (
            float(last["high"]) > prior_high
            and float(last["close"]) < prior_high
        ),
        "sweep_low": (
            float(last["low"]) < prior_low
            and float(last["close"]) > prior_low
        ),
        "equal_high": prior_high,
        "equal_low": prior_low
    }


def detect_fvg(df):
    if len(df) < 4:
        return None
    a = df.iloc[-3]
    c = df.iloc[-1]
    if float(c["low"]) > float(a["high"]):
        return {
            "type": "BULLISH_FVG",
            "low": float(a["high"]),
            "high": float(c["low"])
        }
    if float(c["high"]) < float(a["low"]):
        return {
            "type": "BEARISH_FVG",
            "low": float(c["high"]),
            "high": float(a["low"])
        }
    return None


def detect_order_block(df, side, lookback=12):
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
    """Detect unusual 30m / 1h / 2h movement from closed 5m candles."""
    last_price = float(df5.iloc[-1]["close"])
    results = {}

    for label, bars, threshold in (
        ("30M", 6, MIN_MOMENTUM_PCT_30M),
        ("1H", 12, MIN_MOMENTUM_PCT_1H),
        ("2H", 24, MIN_MOMENTUM_PCT_2H)
    ):
        if len(df5) <= bars:
            results[label] = {
                "direction": "NONE", "change_pct": 0.0,
                "volume_expansion": 0.0, "strong": False
            }
            continue

        start_price = float(df5.iloc[-(bars + 1)]["close"])
        change_pct = (
            ((last_price - start_price) / start_price) * 100
            if start_price else 0.0
        )
        window = df5.iloc[-bars:]
        vol_avg = float(window["volume"].mean())
        base_window = df5["volume"].iloc[-(bars + 21):-bars]
        base_vol = float(base_window.mean()) if len(base_window) else 0.0
        vol_expansion = vol_avg / base_vol if base_vol > 0 else 1.0

        direction = (
            "UP" if change_pct > 0
            else "DOWN" if change_pct < 0
            else "NONE"
        )
        atr_pct = (
            float(df5.iloc[-1]["atr"]) / last_price * 100
            if last_price else 0.0
        )
        adaptive_threshold = max(
            threshold,
            atr_pct * (1.1 if label == "30M" else 1.5)
        )
        strong = (
            abs(change_pct) >= adaptive_threshold
            and vol_expansion >= 1.05
        )
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
            bullish
            and float(prev["close"]) < float(prev["open"])
            and float(last["close"]) >= float(prev["open"])
            and float(last["open"]) <= float(prev["close"])
        )
        breakout = (
            float(last["close"]) > float(df5["high"].iloc[-7:-1].max())
        )
        continuation = (
            bullish and float(last["close"]) > float(prev["high"])
        )
        return (
            bullish and body_ratio >= MIN_BODY_RATIO
            and body >= 0.45 * atr
            and (engulf or breakout or continuation)
            and vol_ratio >= MIN_VOLUME_RATIO
        )

    bearish = float(last["close"]) < float(last["open"])
    engulf = (
        bearish
        and float(prev["close"]) > float(prev["open"])
        and float(last["close"]) <= float(prev["open"])
        and float(last["open"]) >= float(prev["close"])
    )
    breakout = (
        float(last["close"]) < float(df5["low"].iloc[-7:-1].min())
    )
    continuation = (
        bearish and float(last["close"]) < float(prev["low"])
    )
    return (
        bearish and body_ratio >= MIN_BODY_RATIO
        and body >= 0.45 * atr
        and (engulf or breakout or continuation)
        and vol_ratio >= MIN_VOLUME_RATIO
    )


def rapid_move_retest_confirmation(df5, side):
    """Confirm fast impulse by closed 5m retest or controlled continuation."""
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
        if touched_retest:
            return True, "5M retest"
        if continuation:
            return True, "5M continuation"
        return False, "no 5M retest/continuation"

    touched_retest = high >= level_low - 0.20 * atr and close < level_low
    continuation = (
        close < level_low and close < float(last["open"])
        and body_ratio >= 0.55 and volume_ratio >= 1.10
        and close < float(prev["low"])
    )
    if touched_retest:
        return True, "5M retest"
    if continuation:
        return True, "5M continuation"
    return False, "no 5M retest/continuation"


def build_trade_levels(df5, side):
    last = df5.iloc[-1]
    entry = float(last["close"])
    atr = max(float(last["atr"]), entry * 0.0005)
    risk = max(1.35 * atr, entry * 0.0015)

    if side == "LONG":
        return {
            "entry": entry,
            "area_low": entry - 0.20 * atr,
            "area_high": entry + 0.15 * atr,
            "sl": entry - risk,
            "tp1": entry + risk,
            "tp2": entry + 2 * risk,
            "tp3": entry + 3 * risk
        }

    return {
        "entry": entry,
        "area_low": entry - 0.15 * atr,
        "area_high": entry + 0.20 * atr,
        "sl": entry + risk,
        "tp1": entry - risk,
        "tp2": entry - 2 * risk,
        "tp3": entry - 3 * risk
    }


def momentum_summary(momentum):
    parts = []
    for label in ("30M", "1H", "2H"):
        item = momentum[label]
        parts.append(
            f"{label}={item['direction']} {item['change_pct']:+.2f}% "
            f"(vol x{item.get('volume_expansion', 0):.2f}, "
            f"strong={item['strong']})"
        )
    return " | ".join(parts)


def analyze_symbol(symbol):
    """Return a confirmed signal, or log the exact reason the symbol was rejected."""
    def reject(reason, details=""):
        suffix = f" | {details}" if details else ""
        log.info("DIAGNOSTIC %s | REJECT: %s%s", symbol, reason, suffix)
        return None

    try:
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

        long_structure = structure15["bos_up"] or structure15["choch_up"]
        short_structure = structure15["bos_down"] or structure15["choch_down"]
        long_momentum = any(
            item["strong"] and item["direction"] == "UP"
            for item in momentum.values()
        )
        short_momentum = any(
            item["strong"] and item["direction"] == "DOWN"
            for item in momentum.values()
        )
        htf_long_count = (
            int(trend4 == "BULLISH") + int(trend1 == "BULLISH")
        )
        htf_short_count = (
            int(trend4 == "BEARISH") + int(trend1 == "BEARISH")
        )

        diag = (
            f"4H={trend4}, 1H={trend1}, "
            f"15M_BOS_UP={structure15['bos_up']}, "
            f"15M_BOS_DOWN={structure15['bos_down']}, "
            f"15M_CHOCH_UP={structure15['choch_up']}, "
            f"15M_CHOCH_DOWN={structure15['choch_down']}, "
            f"PD={pd_zone['zone']}, "
            f"5M_vol={float(last5['volume_ratio']):.2f}, "
            f"5M_body={float(last5['body_ratio']):.2f}, "
            f"5M_ADX={float(last5['adx']):.1f}, "
            f"{momentum_summary(momentum)}"
        )

        # Normal route: both HTFs agree. Rapid route: one HTF plus
        # aligned 15M structure and strong multi-window momentum.
        if htf_long_count == 2 and long_structure:
            side, rapid_route = "LONG", False
        elif htf_short_count == 2 and short_structure:
            side, rapid_route = "SHORT", False
        elif htf_long_count >= 1 and long_structure and long_momentum:
            side, rapid_route = "LONG", True
        elif htf_short_count >= 1 and short_structure and short_momentum:
            side, rapid_route = "SHORT", True
        else:
            reasons = []
            if htf_long_count == 0 and htf_short_count == 0:
                reasons.append("4H/1H EMA trend mixed or not aligned")
            elif htf_long_count < 2 and htf_short_count < 2:
                reasons.append("both higher timeframes do not agree")
            if not long_structure and not short_structure:
                reasons.append("15M BOS/CHOCH not confirmed")
            if (
                (htf_long_count >= 1 and long_structure and not long_momentum)
                or (htf_short_count >= 1 and short_structure and not short_momentum)
            ):
                reasons.append("rapid route momentum not strong/aligned")
            return reject("; ".join(reasons) or "directional setup not aligned", diag)

        standard_5m = candle_confirmation(df5, side)
        rapid_5m, rapid_reason = rapid_move_retest_confirmation(df5, side)

        if rapid_route:
            if not (standard_5m or rapid_5m):
                return reject(
                    "rapid route blocked: 5M candle confirmation and retest/continuation both failed",
                    diag + f", rapid_reason={rapid_reason}"
                )
        elif not standard_5m:
            return reject(
                "standard route blocked: 5M closed-candle confirmation failed",
                diag
            )

        momentum_aligned = (
            long_momentum if side == "LONG" else short_momentum
        )
        strong_5m = (
            float(last5["volume_ratio"]) >= 1.50
            and float(last5["body_ratio"]) >= 0.62
            and abs(float(last5["close"]) - float(last5["open"]))
            >= 0.65 * float(last5["atr"])
        )
        if not (momentum_aligned or strong_5m):
            return reject(
                "momentum gate failed: neither aligned 30M/1H/2H impulse "
                "nor strong 5M volume/body",
                diag
            )

        score = 0
        score += 2 if (
            (htf_long_count == 2 and side == "LONG")
            or (htf_short_count == 2 and side == "SHORT")
        ) else 1
        score += 2
        score += 2 if standard_5m else 1
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

        required_score = MIN_TOTAL_SCORE - (1 if rapid_route else 0)
        if score < required_score:
            return reject(
                f"score too low: {score}/{11}, required {required_score}",
                diag
            )

        price = float(last5["close"])
        atr5 = max(float(last5["atr"]), 1e-12)
        ema21 = float(last5["ema21"])
        max_extension = 3.5 if rapid_route else 2.8
        extension = abs(price - ema21) / atr5
        if extension > max_extension:
            return reject(
                f"price overextended from 5M EMA21: {extension:.2f} ATR "
                f"(limit {max_extension:.2f})",
                diag
            )

        fvg = detect_fvg(df15)
        ob = detect_order_block(df15, side)
        levels = build_trade_levels(df5, side)

        signal = {
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
            "ema_values": {
                timeframe: {
                    f"EMA{n}": float(frame.iloc[-1][f"ema{n}"])
                    for n in (9, 21, 50, 200)
                }
                for timeframe, frame in (
                    ("4H", df4), ("1H", df1),
                    ("15M", df15), ("5M", df5)
                )
            },
            "pd_zone": pd_zone["zone"],
            "momentum": momentum,
            "fvg": fvg,
            "ob": ob,
            "entry_mode": (
                "RAPID RETEST/CONTINUATION" if rapid_route else "STANDARD SMC"
            ),
            "confirmation": (
                rapid_reason if rapid_route
                else "5M closed-candle confirmation"
            )
        }
        log.info(
            "DIAGNOSTIC %s | CONFIRMED %s | score=%s/11 | %s",
            symbol, side, score, diag
        )
        return signal

    except Exception as exc:
        log.exception("Analysis failed for %s: %s", symbol, exc)
        return None


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            state = json.load(file)
        return state if isinstance(state, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("State file could not be read; starting empty: %s", exc)
        return {}


def save_state(state):
    temp_file = STATE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)
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
    ema_lines = []
    for timeframe in ("4H", "1H", "15M", "5M"):
        values = signal["ema_values"][timeframe]
        ema_lines.append(
            f"EMA {timeframe}: "
            f"9={fmt_price(values['EMA9'])} | "
            f"21={fmt_price(values['EMA21'])} | "
            f"50={fmt_price(values['EMA50'])} | "
            f"200={fmt_price(values['EMA200'])}"
        )

    momentum_lines = []
    for label in ("30M", "1H", "2H"):
        item = signal["momentum"][label]
        momentum_lines.append(
            f"{label}: {item['direction']} {item['change_pct']:+.2f}% "
            f"| Vol x{item.get('volume_expansion', 0):.2f}"
        )

    return (
        f"✅ FINAL CONFIRMED {direction}\n\n"
        f"Coin: {signal['symbol']}\n"
        f"Entry Area: {fmt_price(signal['area_low'])} - "
        f"{fmt_price(signal['area_high'])}\n"
        f"Entry: {fmt_price(signal['entry'])}\n"
        f"Stop Loss: {fmt_price(signal['sl'])}\n"
        f"TP1: {fmt_price(signal['tp1'])}\n"
        f"TP2: {fmt_price(signal['tp2'])}\n"
        f"TP3: {fmt_price(signal['tp3'])}\n\n"
        f"Signal Score: {signal['score']}/11\n"
        f"Entry Mode: {signal['entry_mode']}\n"
        f"Confirmation: {signal['confirmation']}\n"
        f"4H Trend: {signal['trend4']} | 1H Trend: {signal['trend1']}\n"
        f"Premium/Discount: {signal['pd_zone']}\n"
        + "\n".join(ema_lines)
        + "\n\nMomentum:\n"
        + "\n".join(momentum_lines)
        + "\n\nRisk note: Algorithmic estimate, not a guarantee."
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
    log.info("Starting Master Gold + Top 10 Crypto SMC Bot v2.2")
    log.info(
        "API pacing=%.2fs | workers=%s | closed-candle confirmation=5M",
        API_MIN_INTERVAL, MAX_WORKERS
    )
    log.info(
        "Signal diagnostics enabled | Top crypto filter: USDT-M + "
        "active + non-stable + non-leveraged + non-equity"
    )

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning(
            "Telegram credentials missing; alerts cannot be delivered "
            "until repository secrets are set."
        )

    symbols = select_symbols()
    if not symbols:
        log.error("No eligible symbols found. Exiting.")
        return

    state = load_state()
    signals = []

    with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as executor:
        futures = {
            executor.submit(analyze_symbol, symbol): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    signals.append(result)
                    log.info(
                        "Confirmed candidate: %s %s score=%s",
                        symbol, result["side"], result["score"]
                    )
                else:
                    log.info("No final confirmation: %s (see DIAGNOSTIC above)",
                             symbol)
            except Exception as exc:
                log.exception("Worker error for %s: %s", symbol, exc)

    signals.sort(key=lambda item: item["score"], reverse=True)
    sent_count = 0

    for signal in signals:
        key = f"{signal['symbol']}:{signal['side']}"
        candle_key = str(signal["candle_ts"])
        if state.get(key) == candle_key:
            log.info(
                "Duplicate skipped: %s %s",
                signal["symbol"], signal["side"]
            )
            continue

        if send_telegram(build_telegram_message(signal)):
            state[key] = candle_key
            save_state(state)
            sent_count += 1
            log.info(
                "Telegram signal sent: %s %s",
                signal["symbol"], signal["side"]
            )

    log.info(
        "Scan complete. Available symbols scanned=%s | "
        "Confirmed=%s | Sent=%s",
        len(symbols), len(signals), sent_count
    )


if __name__ == "__main__":
    main()
