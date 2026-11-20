import os
import re
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np

# MASTER GOLD + TOP 10 CRYPTO SMC BOT v4.1
# Bitget USDT-M | 4H/1H/15M/5M | Telegram confirmed signals only
# Alert-only: this bot never places orders.

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
STATE_FILE = os.getenv("STATE_FILE", "bot_state.json")
TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()

GOLD = "XAUUSDT"
TOP_N = 10
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "10000000"))
CANDLE_LIMIT = 240
MAX_WORKERS = max(1, int(os.getenv("MAX_WORKERS", "1")))
API_INTERVAL = float(os.getenv("API_MIN_INTERVAL", "0.60"))
TIMEOUT, RETRIES = 20, 4

MIN_SCORE = 8
MIN_VOLUME_RATIO = 1.10
MIN_ADX = 15.0
MIN_BODY_RATIO = 0.45

MOMENTUM_THRESHOLDS = {"30M": (6, 0.35), "1H": (12, 0.55), "2H": (24, 0.80)}

PIVOT_LEFT, PIVOT_RIGHT = 3, 3

# FIX: BOS/CHOCH event can remain valid for six 15M closed candles,
# provided price continues to hold the breakout level.
STRUCTURE_MAX_AGE = 6
STRUCTURE_ATR_BUFFER = 0.04
STRUCTURE_HOLD_ATR_TOLERANCE = 0.15

STRONG_5M_VOLUME, STRONG_5M_BODY = 1.45, 0.60
MAX_EXTENSION_STANDARD, MAX_EXTENSION_RAPID = 2.8, 3.5

PREFERRED = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"
]

EXCLUDED_BASES = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDE", "DAI",
    "USDD", "UST", "EUR", "GBP", "JPY", "XAG"
}

BLOCKED_BASES = set(
    "NVDA TSLA AAPL MSFT AMZN META GOOG GOOGL SNDK SOXL SOXS SPCX "
    "KORU SNXX MU SKHY SKHYNIX OPENAI QQQ MSTR AMD TQQQ SQQQ UPRO "
    "SPXU UVXY SVXY LABU LABD NVDL NVDU TSLL TSLS CONL CONI FNGU "
    "FNGD BITX ETHU ETHD YINN YANG ARKK IWM SPY DIA VXX".split()
)
BLOCKED_SYMBOLS = {x + "USDT" for x in BLOCKED_BASES}
BLOCKED_TERMS = (
    "tokenized stock", "tokenised stock", "stock token", "equity token",
    "share token", "stock", "equity", "shares", "etf", "leveraged",
    "inverse etf", "exchange traded fund", "ultra long", "ultra short",
    "commodity", "forex"
)
BLOCKED_SUFFIXES = (
    "3LUSDT", "3SUSDT", "5LUSDT", "5SUSDT", "2LUSDT", "2SUSDT",
    "BULLUSDT", "BEARUSDT", "UPUSDT", "DOWNUSDT"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("master-smc-bot")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Master-SMC-Alert-Bot/4.1"})
API_LOCK = threading.Lock()
LAST_API_CALL = 0.0


def api_get(path, params=None):
    global LAST_API_CALL
    last_error = None

    for attempt in range(RETRIES):
        wait = min(2 ** (attempt + 1), 30)

        try:
            with API_LOCK:
                delay = API_INTERVAL - (time.monotonic() - LAST_API_CALL)
                if delay > 0:
                    time.sleep(delay)
                LAST_API_CALL = time.monotonic()
                response = SESSION.get(
                    BASE_URL + path, params=params, timeout=TIMEOUT
                )

            if response.status_code == 400:
                raise ValueError(
                    f"Bitget HTTP 400: {path} | {response.text[:300]}"
                )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
                raise requests.HTTPError("Bitget rate limit 429")

            if response.status_code in (500, 502, 503, 504):
                raise requests.HTTPError(
                    f"Bitget server error {response.status_code}"
                )

            response.raise_for_status()
            payload = response.json()

            if str(payload.get("code")) != "00000":
                code = str(payload.get("code"))
                msg = str(payload.get("msg"))
                if code.startswith("4"):
                    raise ValueError(
                        f"Bitget API parameter error {code}: {msg}"
                    )
                raise RuntimeError(f"Bitget API error {code}: {msg}")

            return payload.get("data", [])

        except ValueError:
            raise
        except Exception as exc:
            last_error = exc
            log.warning(
                "API retry %s/%s | %s | %s | wait %.1fs",
                attempt + 1, RETRIES, path, exc, wait
            )
            if attempt + 1 < RETRIES:
                time.sleep(wait)

    raise RuntimeError(f"Bitget request failed: {path}: {last_error}")


def get_contracts_tickers():
    contracts = api_get(
        "/api/v2/mix/market/contracts",
        {"productType": PRODUCT_TYPE}
    )
    tickers = api_get(
        "/api/v2/mix/market/tickers",
        {"productType": PRODUCT_TYPE}
    )
    return contracts, tickers


def valid_crypto_contract(contract):
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
    if base in EXCLUDED_BASES or base in BLOCKED_BASES:
        return False
    if symbol in BLOCKED_SYMBOLS:
        return False
    if any(symbol.endswith(suffix) for suffix in BLOCKED_SUFFIXES):
        return False

    fields = (
        "symbolName", "symbolDisplayName", "baseCoinName", "underlying",
        "underlyingType", "category", "assetType", "symbolDesc",
        "businessType"
    )
    metadata = " ".join(
        str(contract.get(key, "")) for key in fields
    ).lower()

    if any(term in metadata for term in BLOCKED_TERMS):
        return False

    if re.search(r"(2X|3X|5X|LONG|SHORT|BULL|BEAR)", base):
        return False

    contract_type = str(contract.get("contractType", "")).lower()
    if contract_type and contract_type not in ("perpetual", "永续"):
        return False

    return True


def select_symbols():
    contracts, tickers = get_contracts_tickers()
    contract_map = {
        str(c.get("symbol", "")).upper(): c
        for c in contracts if c.get("symbol")
    }
    ticker_map = {
        str(t.get("symbol", "")).upper(): t
        for t in tickers if t.get("symbol")
    }

    gold_ok = bool(
        GOLD in contract_map
        and GOLD in ticker_map
        and str(contract_map[GOLD].get("quoteCoin", "")).upper() == "USDT"
    )
    if not gold_ok:
        log.warning("%s unavailable on Bitget %s", GOLD, PRODUCT_TYPE)

    ranked = []
    excluded = []

    for symbol, contract in contract_map.items():
        if symbol == GOLD:
            continue

        base = str(contract.get("baseCoin", "")).upper()
        if base in BLOCKED_BASES or symbol in BLOCKED_SYMBOLS:
            excluded.append(symbol)
            continue

        if not valid_crypto_contract(contract):
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
            ranked.append((symbol, volume))

    volume_map = dict(ranked)
    eligible = {symbol for symbol, _ in ranked}

    selected = [
        symbol for symbol in PREFERRED
        if symbol in eligible
        and volume_map.get(symbol, 0) >= MIN_QUOTE_VOLUME
    ][:TOP_N]

    for symbol, volume in sorted(
        ranked, key=lambda item: item[1], reverse=True
    ):
        if len(selected) >= TOP_N:
            break
        if symbol not in selected and volume >= MIN_QUOTE_VOLUME:
            selected.append(symbol)

    symbols = ([GOLD] if gold_ok else []) + selected

    log.info(
        "Selected %s symbols: %s",
        len(symbols), ", ".join(symbols)
    )
    log.info(
        "Eligible liquid crypto: %s/%s | quote volume floor=%s",
        len(selected), TOP_N, MIN_QUOTE_VOLUME
    )
    log.info(
        "Leveraged/equity filter active; TQQQUSDT explicitly blocked"
    )

    if excluded:
        log.info(
            "Explicitly excluded stock/ETF contracts: %s",
            ", ".join(sorted(set(excluded)))
        )

    return symbols


def add_indicators(df):
    df = df.copy()

    for period in (9, 21, 50, 200):
        df[f"ema{period}"] = df.close.ewm(
            span=period, adjust=False
        ).mean()

    delta = df.close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    previous_close = df.close.shift(1)
    true_range = pd.concat(
        [
            df.high - df.low,
            (df.high - previous_close).abs(),
            (df.low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    up = df.high.diff()
    down = -df.low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    safe_atr = df.atr.replace(0, np.nan)

    plus_di = (
        100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / safe_atr
    )
    minus_di = (
        100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / safe_atr
    )
    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["plus_di"] = plus_di.fillna(0)
    df["minus_di"] = minus_di.fillna(0)
    df["adx"] = dx.ewm(
        alpha=1 / 14, adjust=False
    ).mean().fillna(0)

    df["volume_ratio"] = (
        df.volume
        / df.volume.rolling(20).mean().replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    candle_range = (df.high - df.low).replace(0, np.nan)
    df["body_ratio"] = (
        (df.close - df.open).abs() / candle_range
    ).fillna(0)

    return df


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
        raise RuntimeError(f"No candle data: {symbol} {granularity}")

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
        raise RuntimeError(f"Invalid candles: {symbol} {granularity}")

    df = (
        df.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    intervals = {
        "5m": 300000,
        "15m": 900000,
        "1H": 3600000,
        "4H": 14400000
    }
    interval = intervals[granularity]
    now_ms = int(time.time() * 1000)

    # Only fully closed candles are used.
    df = df[df.timestamp + interval <= now_ms].copy()

    if len(df) < 205:
        raise RuntimeError(
            f"Not enough closed candles for {symbol} "
            f"{granularity}: {len(df)}"
        )

    return add_indicators(df.reset_index(drop=True))


def ema_direction(df):
    row = df.iloc[-1]

    if row.close > row.ema50 > row.ema200 and row.ema9 > row.ema21:
        return "BULLISH"
    if row.close < row.ema50 < row.ema200 and row.ema9 < row.ema21:
        return "BEARISH"

    return "MIXED"


def confirmed_pivots(df):
    highs, lows = [], []

    # Pivot is confirmed only after PIVOT_RIGHT closed candles.
    for index in range(PIVOT_LEFT, len(df) - PIVOT_RIGHT):
        left = df.iloc[index - PIVOT_LEFT:index]
        right = df.iloc[index + 1:index + PIVOT_RIGHT + 1]

        high_value = float(df.iloc[index].high)
        low_value = float(df.iloc[index].low)

        if (
            high_value > float(left.high.max())
            and high_value >= float(right.high.max())
        ):
            highs.append((index, high_value))

        if (
            low_value < float(left.low.min())
            and low_value <= float(right.low.min())
        ):
            lows.append((index, low_value))

    return highs, lows


def structure_bias(highs, lows):
    if len(highs) < 2 or len(lows) < 2:
        return "UNKNOWN"

    higher_high = highs[-1][1] > highs[-2][1]
    higher_low = lows[-1][1] > lows[-2][1]
    lower_high = highs[-1][1] < highs[-2][1]
    lower_low = lows[-1][1] < lows[-2][1]

    if higher_high and higher_low:
        return "BULLISH"
    if lower_high and lower_low:
        return "BEARISH"

    return "RANGE"


def market_structure(df):
    """
    Improved 15M BOS/CHOCH logic.

    1. Detects a real close-crossing of the latest confirmed swing level.
    2. Classifies BOS vs CHOCH using structure bias before the break.
    3. Keeps the event valid for up to STRUCTURE_MAX_AGE closed candles.
    4. Checks every candle after the breakout for a close that loses the
       breakout level beyond a small ATR tolerance.
    5. Rejects failed breakouts; does not require a fresh BOS every scan.
    6. Uses only closed candles and confirmed pivots (no future lookahead).
    """

    empty = {
        "bos_up": False,
        "bos_down": False,
        "choch_up": False,
        "choch_down": False,
        "swing_high": 0.0,
        "swing_low": 0.0,
        "bias": "UNKNOWN",
        "event_age": None,
        "break_level": 0.0,
        "break_close": 0.0,
        "hold_ok": False,
        "hold_count": 0,
        "event_type": "NONE"
    }

    if len(df) < 50:
        return empty

    # age=0 means latest closed candle; age=6 means six candles ago.
    for age in range(STRUCTURE_MAX_AGE + 1):
        event_end = len(df) - age
        event_prefix = df.iloc[:event_end].reset_index(drop=True)

        if len(event_prefix) < 35:
            continue

        highs, lows = confirmed_pivots(event_prefix)
        if not highs or not lows:
            continue

        swing_high = float(highs[-1][1])
        swing_low = float(lows[-1][1])
        bias = structure_bias(highs, lows)

        event = event_prefix.iloc[-1]
        previous = event_prefix.iloc[-2]
        event_atr = max(float(event.atr), 1e-12)
        break_buffer = event_atr * STRUCTURE_ATR_BUFFER

        broke_up = (
            float(event.close) > swing_high + break_buffer
            and float(previous.close) <= swing_high + break_buffer
        )
        broke_down = (
            float(event.close) < swing_low - break_buffer
            and float(previous.close) >= swing_low - break_buffer
        )

        if not broke_up and not broke_down:
            continue

        if broke_up:
            event_type = "CHOCH_UP" if bias == "BEARISH" else "BOS_UP"
            hold_level = swing_high
            hold_tolerance = event_atr * STRUCTURE_HOLD_ATR_TOLERANCE
            subsequent = df.iloc[event_end:].copy()

            # Break candle is confirmed above level; every later close
            # must remain above level minus the small tolerance.
            hold_ok = bool(
                (subsequent.close > hold_level - hold_tolerance).all()
            )
            hold_count = len(subsequent)
            latest_close = float(df.iloc[-1].close)

            if not hold_ok:
                continue

            return {
                "bos_up": event_type == "BOS_UP",
                "bos_down": False,
                "choch_up": event_type == "CHOCH_UP",
                "choch_down": False,
                "swing_high": swing_high,
                "swing_low": swing_low,
                "bias": bias,
                "event_age": age,
                "break_level": hold_level,
                "break_close": float(event.close),
                "hold_ok": True,
                "hold_count": hold_count,
                "event_type": event_type,
                "latest_close": latest_close
            }

        if broke_down:
            event_type = "CHOCH_DOWN" if bias == "BULLISH" else "BOS_DOWN"
            hold_level = swing_low
            hold_tolerance = event_atr * STRUCTURE_HOLD_ATR_TOLERANCE
            subsequent = df.iloc[event_end:].copy()

            hold_ok = bool(
                (subsequent.close < hold_level + hold_tolerance).all()
            )
            hold_count = len(subsequent)
            latest_close = float(df.iloc[-1].close)

            if not hold_ok:
                continue

            return {
                "bos_up": False,
                "bos_down": event_type == "BOS_DOWN",
                "choch_up": False,
                "choch_down": event_type == "CHOCH_DOWN",
                "swing_high": swing_high,
                "swing_low": swing_low,
                "bias": bias,
                "event_age": age,
                "break_level": hold_level,
                "break_close": float(event.close),
                "hold_ok": True,
                "hold_count": hold_count,
                "event_type": event_type,
                "latest_close": latest_close
            }

    # No currently valid BOS/CHOCH event: still report latest structure.
    highs, lows = confirmed_pivots(df)
    if highs and lows:
        empty.update({
            "swing_high": float(highs[-1][1]),
            "swing_low": float(lows[-1][1]),
            "bias": structure_bias(highs, lows)
        })

    return empty


def premium_discount(df, lookback=50):
    recent = df.iloc[-lookback:]
    high = float(recent.high.max())
    low = float(recent.low.min())
    equilibrium = (high + low) / 2
    price = float(df.iloc[-1].close)

    zone = (
        "DISCOUNT" if price < equilibrium
        else "PREMIUM" if price > equilibrium
        else "EQUILIBRIUM"
    )

    return {
        "zone": zone,
        "high": high,
        "low": low,
        "equilibrium": equilibrium,
        "price": price
    }


def liquidity_analysis(df, lookback=25):
    recent = df.iloc[-lookback - 1:-1]
    current = df.iloc[-1]
    high = float(recent.high.max())
    low = float(recent.low.min())

    return {
        "sweep_high": bool(current.high > high and current.close < high),
        "sweep_low": bool(current.low < low and current.close > low),
        "equal_high": high,
        "equal_low": low
    }


def detect_fvg(df):
    if len(df) < 4:
        return None

    first = df.iloc[-3]
    last = df.iloc[-1]

    if last.low > first.high:
        return {
            "type": "BULLISH_FVG",
            "low": float(first.high),
            "high": float(last.low)
        }

    if last.high < first.low:
        return {
            "type": "BEARISH_FVG",
            "low": float(last.high),
            "high": float(first.low)
        }

    return None


def detect_order_block(df, side, lookback=12):
    recent = df.iloc[-(lookback + 1):-1]
    current = df.iloc[-1]

    if abs(float(current.close - current.open)) < (
        0.65 * max(float(current.atr), 1e-12)
    ):
        return None

    if side == "LONG" and current.close > current.open:
        opposite = recent[recent.close < recent.open]
    elif side == "SHORT" and current.close < current.open:
        opposite = recent[recent.close > recent.open]
    else:
        return None

    if opposite.empty:
        return None

    candle = opposite.iloc[-1]
    return {"low": float(candle.low), "high": float(candle.high)}


def momentum_analysis(df5):
    output = {}
    price = float(df5.iloc[-1].close)

    for label, (bars, threshold) in MOMENTUM_THRESHOLDS.items():
        start = float(df5.iloc[-bars - 1].close)
        change = (
            (price - start) / start * 100 if start > 0 else 0.0
        )

        window = df5.iloc[-bars:]
        baseline = df5.iloc[-(bars + 21):-bars]

        average_volume = float(window.volume.mean())
        baseline_volume = float(baseline.volume.mean())
        expansion = (
            average_volume / baseline_volume
            if baseline_volume > 0 else 0.0
        )

        direction = (
            "UP" if change > 0 else "DOWN" if change < 0 else "NONE"
        )

        atr_pct = (
            float(df5.iloc[-1].atr) / price * 100 if price > 0 else 0.0
        )
        adaptive_threshold = max(
            threshold,
            atr_pct * (1.1 if label == "30M" else 1.5)
        )

        output[label] = {
            "direction": direction,
            "change_pct": change,
            "volume_expansion": expansion,
            "strong": bool(
                abs(change) >= adaptive_threshold
                and expansion >= 1.05
            )
        }

    return output


def momentum_summary(momentum):
    return " | ".join(
        f"{key}={momentum[key]['direction']} "
        f"{momentum[key]['change_pct']:+.2f}% "
        f"(Vol x{momentum[key]['volume_expansion']:.2f}, "
        f"Strong={momentum[key]['strong']})"
        for key in ("30M", "1H", "2H")
    )


def candle_confirmation(df, side):
    current = df.iloc[-1]
    previous = df.iloc[-2]

    body = abs(float(current.close - current.open))
    atr = max(float(current.atr), 1e-12)
    body_ratio = float(current.body_ratio)
    volume_ratio = float(current.volume_ratio)

    if side == "LONG":
        directional = current.close > current.open
        engulf = (
            directional
            and previous.close < previous.open
            and current.close >= previous.open
            and current.open <= previous.close
        )
        breakout = current.close > float(df.high.iloc[-7:-1].max())
        continuation = directional and current.close > previous.high
    else:
        directional = current.close < current.open
        engulf = (
            directional
            and previous.close > previous.open
            and current.close <= previous.open
            and current.open >= previous.close
        )
        breakout = current.close < float(df.low.iloc[-7:-1].min())
        continuation = directional and current.close < previous.low

    return bool(
        directional
        and body_ratio >= MIN_BODY_RATIO
        and body >= 0.40 * atr
        and (engulf or breakout or continuation)
        and volume_ratio >= MIN_VOLUME_RATIO
    )


def rapid_retest_confirmation(df, side):
    if len(df) < 35:
        return False, "insufficient 5M history"

    current = df.iloc[-1]
    previous = df.iloc[-2]
    prior = df.iloc[-7:-1]
    earlier = df.iloc[-13:-7]

    atr = max(float(current.atr), 1e-12)
    body_ratio = float(current.body_ratio)
    volume_ratio = float(current.volume_ratio)

    if side == "LONG":
        level = float(earlier.high.max())
        prior_break = float(prior.close.max()) > level

        retest = (
            prior_break
            and current.low <= level + 0.20 * atr
            and current.close > level
            and current.close > current.open
            and body_ratio >= 0.45
            and volume_ratio >= 1.05
        )
        continuation = (
            current.close > float(prior.high.max())
            and current.close > current.open
            and current.close > previous.high
            and body_ratio >= 0.60
            and volume_ratio >= 1.35
        )
    else:
        level = float(earlier.low.min())
        prior_break = float(prior.close.min()) < level

        retest = (
            prior_break
            and current.high >= level - 0.20 * atr
            and current.close < level
            and current.close < current.open
            and body_ratio >= 0.45
            and volume_ratio >= 1.05
        )
        continuation = (
            current.close < float(prior.low.min())
            and current.close < current.open
            and current.close < previous.low
            and body_ratio >= 0.60
            and volume_ratio >= 1.35
        )

    if retest:
        return True, "5M breakout retest"
    if continuation:
        return True, "5M strong continuation"

    return False, "no valid retest/continuation"


def strong_5m(df, side):
    current = df.iloc[-1]
    directional = (
        current.close > current.open
        if side == "LONG"
        else current.close < current.open
    )

    return bool(
        directional
        and current.volume_ratio >= STRONG_5M_VOLUME
        and current.body_ratio >= STRONG_5M_BODY
        and abs(float(current.close - current.open))
        >= 0.60 * max(float(current.atr), 1e-12)
    )


def trade_levels(df, side):
    current = df.iloc[-1]
    entry = float(current.close)
    atr = max(float(current.atr), entry * 0.0005)
    risk = max(1.50 * atr, entry * 0.0015)

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


def analyze_symbol(symbol):
    def reject(reason, diagnostic=""):
        log.info(
            "DIAGNOSTIC %s | REJECT: %s%s",
            symbol,
            reason,
            (" | " + diagnostic) if diagnostic else ""
        )
        return None

    try:
        d4 = fetch_candles(symbol, "4H")
        d1 = fetch_candles(symbol, "1H")
        d15 = fetch_candles(symbol, "15m")
        d5 = fetch_candles(symbol, "5m")

        trend4 = ema_direction(d4)
        trend1 = ema_direction(d1)
        structure = market_structure(d15)
        pd_zone = premium_discount(d15)
        liquidity = liquidity_analysis(d15)
        momentum = momentum_analysis(d5)
        current = d5.iloc[-1]

        long_structure = (
            structure["bos_up"] or structure["choch_up"]
        )
        short_structure = (
            structure["bos_down"] or structure["choch_down"]
        )

        long_momentum = any(
            item["strong"] and item["direction"] == "UP"
            for item in momentum.values()
        )
        short_momentum = any(
            item["strong"] and item["direction"] == "DOWN"
            for item in momentum.values()
        )

        long_count = int(trend4 == "BULLISH") + int(trend1 == "BULLISH")
        short_count = int(trend4 == "BEARISH") + int(trend1 == "BEARISH")

        diagnostic = (
            f"4H={trend4}, 1H={trend1}, "
            f"15M BOS U/D={structure['bos_up']}/{structure['bos_down']}, "
            f"CHOCH U/D={structure['choch_up']}/{structure['choch_down']}, "
            f"bias={structure['bias']}, "
            f"event_age={structure['event_age']}, "
            f"event={structure['event_type']}, "
            f"break_level={structure['break_level']:.8g}, "
            f"hold={structure['hold_ok']} "
            f"(candles={structure['hold_count']}), "
            f"PD={pd_zone['zone']}, "
            f"5M vol={current.volume_ratio:.2f}, "
            f"body={current.body_ratio:.2f}, ADX={current.adx:.1f}, "
            f"{momentum_summary(momentum)}"
        )

        side = None
        rapid = False

        if long_count == 2 and long_structure:
            side = "LONG"
        elif short_count == 2 and short_structure:
            side = "SHORT"
        elif (
            long_count >= 1 and short_count == 0
            and long_structure and long_momentum
        ):
            side = "LONG"
            rapid = True
        elif (
            short_count >= 1 and long_count == 0
            and short_structure and short_momentum
        ):
            side = "SHORT"
            rapid = True

        if side is None:
            reasons = []

            if long_count == 0 and short_count == 0:
                reasons.append("4H/1H trend mixed")
            elif long_count < 2 and short_count < 2:
                reasons.append("HTF not aligned or opposing")

            if not long_structure and not short_structure:
                reasons.append(
                    "15M BOS/CHOCH absent, expired, or breakout hold failed"
                )

            if long_structure and long_count >= 1 and not long_momentum:
                reasons.append("bullish momentum weak")

            if short_structure and short_count >= 1 and not short_momentum:
                reasons.append("bearish momentum weak")

            return reject(
                "; ".join(reasons) or "directional setup not aligned",
                diagnostic
            )

        ema15_bullish = (
            d15.iloc[-1].ema9 > d15.iloc[-1].ema21
        )
        ema15_bearish = (
            d15.iloc[-1].ema9 < d15.iloc[-1].ema21
        )

        if side == "LONG" and not ema15_bullish:
            return reject("15M EMA disagrees with LONG", diagnostic)
        if side == "SHORT" and not ema15_bearish:
            return reject("15M EMA disagrees with SHORT", diagnostic)

        standard_confirmation = candle_confirmation(d5, side)
        rapid_ok, rapid_reason = rapid_retest_confirmation(d5, side)
        strong5 = strong_5m(d5, side)

        if rapid:
            if not (standard_confirmation or rapid_ok):
                return reject(
                    "rapid route: 5M confirmation failed",
                    diagnostic + " | " + rapid_reason
                )
        elif not standard_confirmation:
            return reject(
                "standard route: 5M closed-candle confirmation failed",
                diagnostic
            )

        aligned_momentum = (
            long_momentum if side == "LONG" else short_momentum
        )
        if not (aligned_momentum or strong5):
            return reject("momentum gate failed", diagnostic)

        score = (
            2 if (
                (side == "LONG" and long_count == 2)
                or (side == "SHORT" and short_count == 2)
            ) else 1
        )

        score += (
            2 if (
                (side == "LONG" and structure["bos_up"])
                or (side == "SHORT" and structure["bos_down"])
            ) else 1
        )
        score += 2 if (
            standard_confirmation or rapid_ok
        ) else 0
        score += int(float(current.adx) >= MIN_ADX)
        score += int(float(current.volume_ratio) >= MIN_VOLUME_RATIO)
        score += int(aligned_momentum)
        score += int(
            (side == "LONG" and liquidity["sweep_low"])
            or (side == "SHORT" and liquidity["sweep_high"])
        )
        score += int(
            (side == "LONG" and pd_zone["zone"] == "DISCOUNT")
            or (side == "SHORT" and pd_zone["zone"] == "PREMIUM")
        )
        score += 1  # 15M EMA alignment passed.

        if score < MIN_SCORE:
            return reject(
                f"score too low {score}/12; required {MIN_SCORE}",
                diagnostic
            )

        price = float(current.close)
        atr = max(float(current.atr), 1e-12)
        extension = abs(price - float(current.ema21)) / atr
        extension_limit = (
            MAX_EXTENSION_RAPID if rapid else MAX_EXTENSION_STANDARD
        )

        if extension > extension_limit:
            return reject(
                f"overextended from EMA21 {extension:.2f} ATR; "
                f"limit {extension_limit:.2f}",
                diagnostic
            )

        result = {
            "symbol": symbol,
            "side": side,
            "score": score,
            **trade_levels(d5, side),
            "candle_ts": int(current.timestamp),
            "trend4": trend4,
            "trend1": trend1,
            "ema_values": {
                timeframe: {
                    f"EMA{period}": float(
                        frame.iloc[-1][f"ema{period}"]
                    )
                    for period in (9, 21, 50, 200)
                }
                for timeframe, frame in [
                    ("4H", d4), ("1H", d1),
                    ("15M", d15), ("5M", d5)
                ]
            },
            "pd_zone": pd_zone["zone"],
            "momentum": momentum,
            "fvg": detect_fvg(d15),
            "ob": detect_order_block(d15, side),
            "liquidity": liquidity,
            "structure": structure,
            "entry_mode": (
                "RAPID RETEST/CONTINUATION"
                if rapid else "STANDARD SMC"
            ),
            "confirmation": (
                rapid_reason if rapid and rapid_ok
                else "5M closed-candle confirmation"
            ),
            "rsi5": float(current.rsi),
            "adx5": float(current.adx),
            "volume_ratio5": float(current.volume_ratio)
        }

        log.info(
            "DIAGNOSTIC %s | CONFIRMED %s | score=%s/12 | %s",
            symbol, side, score, diagnostic
        )
        return result

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
        log.warning("State read failed: %s", exc)
        return {}


def save_state(state):
    temporary_file = STATE_FILE + ".tmp"
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)
    os.replace(temporary_file, STATE_FILE)


def fmt(value):
    value = float(value)

    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:.5f}".rstrip("0").rstrip(".")

    return f"{value:.8f}".rstrip("0").rstrip(".")


def build_message(signal):
    direction = (
        "🟢 LONG" if signal["side"] == "LONG" else "🔴 SHORT"
    )

    ema_lines = []
    for timeframe in ("4H", "1H", "15M", "5M"):
        values = signal["ema_values"][timeframe]
        ema_lines.append(
            f"EMA {timeframe}: "
            f"9={fmt(values['EMA9'])} | "
            f"21={fmt(values['EMA21'])} | "
            f"50={fmt(values['EMA50'])} | "
            f"200={fmt(values['EMA200'])}"
        )

    momentum_lines = []
    for key in ("30M", "1H", "2H"):
        item = signal["momentum"][key]
        momentum_lines.append(
            f"{key}: {item['direction']} "
            f"{item['change_pct']:+.2f}% | "
            f"Vol x{item['volume_expansion']:.2f} | "
            f"Strong={item['strong']}"
        )

    structure = signal["structure"]
    fvg = signal["fvg"]
    order_block = signal["ob"]
    liquidity = signal["liquidity"]

    fvg_text = (
        f"{fvg['type']}: {fmt(fvg['low'])} - {fmt(fvg['high'])}"
        if fvg else "No fresh FVG detected"
    )
    ob_text = (
        f"{fmt(order_block['low'])} - {fmt(order_block['high'])}"
        if order_block else "No valid OB detected"
    )

    liquidity_text = (
        "Sell-side sweep detected"
        if liquidity["sweep_low"] else "No sell-side sweep"
    ) if signal["side"] == "LONG" else (
        "Buy-side sweep detected"
        if liquidity["sweep_high"] else "No buy-side sweep"
    )

    return (
        f"✅ FINAL CONFIRMED {direction}\n\n"
        f"Coin: {signal['symbol']}\n\n"
        f"Entry Area: {fmt(signal['area_low'])} - "
        f"{fmt(signal['area_high'])}\n"
        f"Entry: {fmt(signal['entry'])}\n\n"
        f"Stop Loss: {fmt(signal['sl'])}\n"
        f"TP1: {fmt(signal['tp1'])}\n"
        f"TP2: {fmt(signal['tp2'])}\n"
        f"TP3: {fmt(signal['tp3'])}\n\n"
        f"Signal Score: {signal['score']}/12\n"
        f"Entry Mode: {signal['entry_mode']}\n"
        f"Confirmation: {signal['confirmation']}\n\n"
        f"4H Trend: {signal['trend4']}\n"
        f"1H Trend: {signal['trend1']}\n"
        f"Premium/Discount: {signal['pd_zone']}\n\n"
        f"15M Structure: "
        f"BOS UP={structure['bos_up']} | "
        f"BOS DOWN={structure['bos_down']} | "
        f"CHOCH UP={structure['choch_up']} | "
        f"CHOCH DOWN={structure['choch_down']}\n"
        f"Structure Bias: {structure['bias']} | "
        f"Event: {structure['event_type']} | "
        f"Age: {structure['event_age']} candles\n"
        f"Break Level: {fmt(structure['break_level'])} | "
        f"Hold: {structure['hold_ok']} "
        f"({structure['hold_count']} candles)\n"
        f"Liquidity: {liquidity_text}\n"
        f"FVG: {fvg_text}\n"
        f"Order Block: {ob_text}\n\n"
        + "\n".join(ema_lines)
        + "\n\nMomentum:\n"
        + "\n".join(momentum_lines)
        + "\n\n"
        + f"5M RSI: {signal['rsi5']:.2f}\n"
        + f"5M ADX: {signal['adx5']:.2f}\n"
        + f"5M Volume Ratio: x{signal['volume_ratio5']:.2f}\n\n"
        + "Risk Note: Algorithmic analysis, not a guarantee of profit. "
        "No orders are placed automatically."
    )


def send_telegram(message):
    if not TOKEN or not CHAT_ID:
        log.error("Telegram credentials missing; message not delivered.")
        log.info("\n%s", message)
        return False

    try:
        response = SESSION.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": message,
                "disable_web_page_preview": True
            },
            timeout=TIMEOUT
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
    log.info("Starting Master Gold + Top 10 Crypto SMC Bot v4.1")
    log.info(
        "API pacing=%.2fs | workers=%s | 4H/1H/15M/5M",
        API_INTERVAL, MAX_WORKERS
    )
    log.info(
        "15M structure fix: BOS/CHOCH valid up to %s closed candles",
        STRUCTURE_MAX_AGE
    )
    log.info(
        "15M breakout hold validation enabled | ATR tolerance=%.2f",
        STRUCTURE_HOLD_ATR_TOLERANCE
    )
    log.info(
        "Closed-candle analysis + confirmed swing pivots enabled"
    )
    log.info(
        "5M confirmation + retest/continuation filters enabled"
    )
    log.info(
        "Telegram: FINAL CONFIRMED SIGNALS ONLY | no early warnings"
    )
    log.info(
        "Stock/equity/tokenized/leveraged/inverse ETF filters enabled"
    )

    if not TOKEN or not CHAT_ID:
        log.warning(
            "Telegram secrets missing; check GitHub Actions repository secrets."
        )

    try:
        symbols = select_symbols()
    except Exception as exc:
        log.exception("Symbol selection failed: %s", exc)
        return

    if not symbols:
        log.error("No eligible symbols found.")
        return

    state = load_state()
    signals = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(analyze_symbol, symbol): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    signals.append(result)
                    log.info(
                        "Confirmed candidate: %s %s | Score=%s/12",
                        symbol, result["side"], result["score"]
                    )
                else:
                    log.info("No final confirmation: %s", symbol)

            except Exception as exc:
                log.exception("Worker failed for %s: %s", symbol, exc)

    signals.sort(key=lambda item: item["score"], reverse=True)
    sent = 0

    for signal in signals:
        key = f"{signal['symbol']}:{signal['side']}"
        candle = str(signal["candle_ts"])

        if state.get(key) == candle:
            log.info(
                "Duplicate signal skipped: %s %s",
                signal["symbol"], signal["side"]
            )
            continue

        if send_telegram(build_message(signal)):
            state[key] = candle
            save_state(state)
            sent += 1
            log.info(
                "Telegram signal sent: %s %s",
                signal["symbol"], signal["side"]
            )

    log.info(
        "Scan complete | Available symbols scanned=%s | "
        "Confirmed=%s | Sent=%s",
        len(symbols), len(signals), sent
    )


if __name__ == "__main__":
    main()
