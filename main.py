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


# ============================================================
# MASTER GOLD + TOP 10 CRYPTO SMC BOT v5.0
# Bitget USDT-M Futures
# 4H / 1H / 15M / 5M
# FINAL CONFIRMED SIGNALS ONLY
# ALERT ONLY - NO AUTOMATIC TRADING
# ============================================================


BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"

STATE_FILE = os.getenv("STATE_FILE", "bot_state.json")

TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("TELEGRAM_TOKEN")
    or ""
).strip()

CHAT_ID = (
    os.getenv("TELEGRAM_CHAT_ID")
    or os.getenv("CHAT_ID")
    or ""
).strip()


GOLD = "XAUUSDT"
TOP_N = 10

MIN_QUOTE_VOLUME = float(
    os.getenv("MIN_QUOTE_VOLUME", "10000000")
)

CANDLE_LIMIT = 300

MAX_WORKERS = max(
    1, int(os.getenv("MAX_WORKERS", "1"))
)

API_INTERVAL = float(
    os.getenv("API_MIN_INTERVAL", "0.60")
)

TIMEOUT = 20
RETRIES = 4


# ============================================================
# SIGNAL QUALITY SETTINGS
# ============================================================

MIN_SCORE = 9

MIN_VOLUME_RATIO = 1.10
MIN_ADX = 15.0
MIN_BODY_RATIO = 0.45

STRONG_5M_VOLUME = 1.45
STRONG_5M_BODY = 0.60

WEAK_ADX_STRONG_VOLUME = 1.80
WEAK_ADX_STRONG_BODY = 0.65

MAX_EXTENSION_STANDARD = 2.8
MAX_EXTENSION_RAPID = 3.5


# 5-minute candles:
# 6 candles = 30 minutes
# 12 candles = 1 hour
# 24 candles = 2 hours

MOMENTUM_THRESHOLDS = {
    "30M": (6, 0.35),
    "1H": (12, 0.55),
    "2H": (24, 0.80)
}


# ============================================================
# MARKET STRUCTURE SETTINGS
# ============================================================

PIVOT_LEFT = 3
PIVOT_RIGHT = 3

STRUCTURE_MAX_AGE = 6

STRUCTURE_ATR_BUFFER = 0.04

STRUCTURE_HOLD_ATR_TOLERANCE = 0.15

MIN_POST_BREAK_CANDLES = 1


# ============================================================
# SYMBOL FILTERS
# ============================================================

PREFERRED = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT"
]


EXCLUDED_BASES = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD",
    "USDE", "DAI", "USDD", "UST", "EUR",
    "GBP", "JPY", "XAG"
}


BLOCKED_BASES = set(
    """
    NVDA TSLA AAPL MSFT AMZN META GOOG GOOGL
    SNDK SOXL SOXS SPCX KORU SNXX MU SKHY
    SKHYNIX OPENAI QQQ MSTR AMD TQQQ SQQQ
    UPRO SPXU UVXY SVXY LABU LABD NVDL NVDU
    TSLL TSLS CONL CONI FNGU FNGD BITX ETHU
    ETHD YINN YANG ARKK IWM SPY DIA VXX
    """.split()
)


BLOCKED_SYMBOLS = {
    base + "USDT" for base in BLOCKED_BASES
}


BLOCKED_TERMS = (
    "tokenized stock",
    "tokenised stock",
    "stock token",
    "equity token",
    "share token",
    "stock",
    "equity",
    "shares",
    "etf",
    "leveraged",
    "inverse etf",
    "exchange traded fund",
    "ultra long",
    "ultra short",
    "commodity",
    "forex"
)


BLOCKED_SUFFIXES = (
    "3LUSDT",
    "3SUSDT",
    "5LUSDT",
    "5SUSDT",
    "2LUSDT",
    "2SUSDT",
    "BULLUSDT",
    "BEARUSDT",
    "UPUSDT",
    "DOWNUSDT"
)


# ============================================================
# LOGGING AND SESSION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("master-smc-bot")

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "Master-SMC-Alert-Bot/5.0"
})

API_LOCK = threading.Lock()

LAST_API_CALL = 0.0


# ============================================================
# API REQUEST
# ============================================================

def api_get(path, params=None):

    global LAST_API_CALL

    last_error = None

    for attempt in range(RETRIES):

        wait = min(2 ** (attempt + 1), 30)

        try:

            with API_LOCK:

                delay = API_INTERVAL - (
                    time.monotonic() - LAST_API_CALL
                )

                if delay > 0:
                    time.sleep(delay)

                response = SESSION.get(
                    BASE_URL + path,
                    params=params,
                    timeout=TIMEOUT
                )

                LAST_API_CALL = time.monotonic()

            if response.status_code == 400:

                raise ValueError(
                    f"Bitget HTTP 400: {path} | "
                    f"{response.text[:300]}"
                )

            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:

                    try:
                        wait = max(
                            wait,
                            float(retry_after)
                        )
                    except ValueError:
                        pass

                raise requests.HTTPError(
                    "Bitget rate limit 429"
                )

            if response.status_code in (
                500, 502, 503, 504
            ):

                raise requests.HTTPError(
                    f"Bitget server error "
                    f"{response.status_code}"
                )

            response.raise_for_status()

            payload = response.json()

            if str(payload.get("code")) != "00000":

                code = str(payload.get("code"))
                msg = str(payload.get("msg"))

                if code.startswith("4"):

                    raise ValueError(
                        f"Bitget API parameter error "
                        f"{code}: {msg}"
                    )

                raise RuntimeError(
                    f"Bitget API error {code}: {msg}"
                )

            return payload.get("data", [])

        except ValueError:
            raise

        except Exception as exc:

            last_error = exc

            log.warning(
                "API retry %s/%s | %s | %s | wait %.1fs",
                attempt + 1,
                RETRIES,
                path,
                exc,
                wait
            )

            if attempt + 1 < RETRIES:
                time.sleep(wait)

    raise RuntimeError(
        f"Bitget request failed: {path}: {last_error}"
    )


# ============================================================
# CONTRACTS AND TICKERS
# ============================================================

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

    symbol = str(
        contract.get("symbol", "")
    ).upper()

    base = str(
        contract.get("baseCoin", "")
    ).upper()

    quote = str(
        contract.get("quoteCoin", "")
    ).upper()

    status = str(
        contract.get(
            "symbolStatus",
            contract.get("status", "")
        )
    ).lower()

    if not symbol.endswith("USDT"):
        return False

    if quote != "USDT" or not base:
        return False

    if status and status not in (
        "normal", "online", "listed"
    ):
        return False

    if base in EXCLUDED_BASES:
        return False

    if base in BLOCKED_BASES:
        return False

    if symbol in BLOCKED_SYMBOLS:
        return False

    if any(
        symbol.endswith(suffix)
        for suffix in BLOCKED_SUFFIXES
    ):
        return False

    fields = (
        "symbolName",
        "symbolDisplayName",
        "baseCoinName",
        "underlying",
        "underlyingType",
        "category",
        "assetType",
        "symbolDesc",
        "businessType"
    )

    metadata = " ".join(
        str(contract.get(key, ""))
        for key in fields
    ).lower()

    if any(
        term in metadata
        for term in BLOCKED_TERMS
    ):
        return False

    if re.search(
        r"(2X|3X|5X|LONG|SHORT|BULL|BEAR)",
        base
    ):
        return False

    contract_type = str(
        contract.get("contractType", "")
    ).lower()

    if contract_type and contract_type not in (
        "perpetual", "永续"
    ):
        return False

    return True


def select_symbols():

    contracts, tickers = get_contracts_tickers()

    contract_map = {
        str(c.get("symbol", "")).upper(): c
        for c in contracts
        if c.get("symbol")
    }

    ticker_map = {
        str(t.get("symbol", "")).upper(): t
        for t in tickers
        if t.get("symbol")
    }

    gold_ok = bool(
        GOLD in contract_map
        and GOLD in ticker_map
        and str(
            contract_map[GOLD].get("quoteCoin", "")
        ).upper() == "USDT"
    )

    if not gold_ok:

        log.warning(
            "%s unavailable on Bitget %s",
            GOLD,
            PRODUCT_TYPE
        )

    ranked = []

    for symbol, contract in contract_map.items():

        if symbol == GOLD:
            continue

        if not valid_crypto_contract(contract):
            continue

        ticker = ticker_map.get(symbol)

        if not ticker:
            continue

        try:

            volume = float(
                ticker.get("quoteVolume", 0) or 0
            )

            price = float(
                ticker.get("lastPr", 0) or 0
            )

        except (TypeError, ValueError):
            continue

        if (
            price > 0
            and volume >= MIN_QUOTE_VOLUME
        ):

            ranked.append((symbol, volume))

    volume_map = dict(ranked)

    eligible = set(volume_map)

    selected = [
        symbol
        for symbol in PREFERRED
        if symbol in eligible
    ][:TOP_N]

    for symbol, volume in sorted(
        ranked,
        key=lambda item: item[1],
        reverse=True
    ):

        if len(selected) >= TOP_N:
            break

        if symbol not in selected:
            selected.append(symbol)

    symbols = (
        [GOLD] if gold_ok else []
    ) + selected

    log.info(
        "Selected %s symbols: %s",
        len(symbols),
        ", ".join(symbols)
    )

    log.info(
        "Liquid crypto selected: %s/%s | "
        "minimum quote volume=%s",
        len(selected),
        TOP_N,
        MIN_QUOTE_VOLUME
    )

    log.info(
        "Equity, ETF, tokenized and leveraged "
        "contract filters enabled"
    )

    return symbols


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    for period in (9, 21, 50, 200):

        df[f"ema{period}"] = df.close.ewm(
            span=period,
            adjust=False
        ).mean()

    delta = df.close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    df["rsi"] = (
        100 - 100 / (1 + rs)
    ).fillna(50)

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
        alpha=1 / 14,
        adjust=False
    ).mean()

    up = df.high.diff()
    down = -df.low.diff()

    plus_dm = up.where(
        (up > down) & (up > 0),
        0.0
    )

    minus_dm = down.where(
        (down > up) & (down > 0),
        0.0
    )

    safe_atr = df.atr.replace(
        0, np.nan
    )

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean() /
        safe_atr
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean() /
        safe_atr
    )

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(
            0, np.nan
        )
    )

    df["plus_di"] = plus_di.fillna(0)
    df["minus_di"] = minus_di.fillna(0)

    df["adx"] = dx.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean().fillna(0)

    average_volume = df.volume.rolling(
        20,
        min_periods=10
    ).mean()

    df["volume_ratio"] = (
        df.volume /
        average_volume.replace(0, np.nan)
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    candle_range = (
        df.high - df.low
    ).replace(0, np.nan)

    df["body_ratio"] = (
        (df.close - df.open).abs()
        / candle_range
    ).fillna(0)

    return df


# ============================================================
# CANDLE DATA
# ============================================================

def fetch_candles(
    symbol,
    granularity,
    limit=CANDLE_LIMIT
):

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

        raise RuntimeError(
            f"No candle data: {symbol} {granularity}"
        )

    rows = []

    for row in data:

        if len(row) < 6:
            continue

        try:

            rows.append([
                int(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]) if len(row) > 6 else 0.0
            ])

        except (TypeError, ValueError):
            continue

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

    if df.empty:

        raise RuntimeError(
            f"Invalid candles: {symbol} {granularity}"
        )

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

    # Incomplete candles are removed.
    df = df[
        df.timestamp + interval <= now_ms
    ].copy()

    if len(df) < 220:

        raise RuntimeError(
            f"Not enough closed candles for "
            f"{symbol} {granularity}: {len(df)}"
        )

    return add_indicators(
        df.reset_index(drop=True)
    )


# ============================================================
# EMA TREND
# ============================================================

def ema_direction(df):

    row = df.iloc[-1]

    if (
        row.close > row.ema50 > row.ema200
        and row.ema9 > row.ema21
    ):
        return "BULLISH"

    if (
        row.close < row.ema50 < row.ema200
        and row.ema9 < row.ema21
    ):
        return "BEARISH"

    return "MIXED"


# ============================================================
# CONFIRMED PIVOTS
# ============================================================

def confirmed_pivots(df):

    highs = []
    lows = []

    for index in range(
        PIVOT_LEFT,
        len(df) - PIVOT_RIGHT
    ):

        left = df.iloc[
            index - PIVOT_LEFT:index
        ]

        right = df.iloc[
            index + 1:index + PIVOT_RIGHT + 1
        ]

        high_value = float(
            df.iloc[index].high
        )

        low_value = float(
            df.iloc[index].low
        )

        if (
            high_value > float(left.high.max())
            and high_value >= float(right.high.max())
        ):

            highs.append(
                (index, high_value)
            )

        if (
            low_value < float(left.low.min())
            and low_value <= float(right.low.min())
        ):

            lows.append(
                (index, low_value)
            )

    return highs, lows


# ============================================================
# STRUCTURE BIAS
# ============================================================

def structure_bias(highs, lows):

    if len(highs) < 2 or len(lows) < 2:
        return "UNKNOWN"

    higher_high = (
        highs[-1][1] > highs[-2][1]
    )

    higher_low = (
        lows[-1][1] > lows[-2][1]
    )

    lower_high = (
        highs[-1][1] < highs[-2][1]
    )

    lower_low = (
        lows[-1][1] < lows[-2][1]
    )

    if higher_high and higher_low:
        return "BULLISH"

    if lower_high and lower_low:
        return "BEARISH"

    return "RANGE"


# ============================================================
# IMPROVED BOS / CHOCH + HOLD VALIDATION
# ============================================================

def market_structure(df):

    empty = {
        "bos_up": False,
        "bos_down": False,
        "choch_up": False,
        "choch_down": False,

        "swing_high": 0.0,
        "swing_low": 0.0,

        "bias": "UNKNOWN",
        "bias_before": "UNKNOWN",

        "event_age": None,
        "break_level": 0.0,
        "break_close": 0.0,

        "hold_ok": False,
        "hold_count": 0,

        "event_type": "NONE",
        "latest_close": 0.0
    }

    if len(df) < 50:
        return empty

    # age=0 means current closed candle.
    # age=1 means one closed candle after breakout.
    #
    # We intentionally reject age=0.
    # At least one later closed candle must confirm the hold.

    for age in range(
        1,
        STRUCTURE_MAX_AGE + 1
    ):

        event_end = len(df) - age

        event_prefix = (
            df.iloc[:event_end]
            .reset_index(drop=True)
        )

        if len(event_prefix) < 35:
            continue

        highs, lows = confirmed_pivots(
            event_prefix
        )

        if not highs or not lows:
            continue

        swing_high = float(
            highs[-1][1]
        )

        swing_low = float(
            lows[-1][1]
        )

        bias_before = structure_bias(
            highs,
            lows
        )

        event = event_prefix.iloc[-1]
        previous = event_prefix.iloc[-2]

        event_atr = max(
            float(event.atr),
            1e-12
        )

        break_buffer = (
            event_atr * STRUCTURE_ATR_BUFFER
        )

        broke_up = (
            float(event.close)
            > swing_high + break_buffer

            and float(previous.close)
            <= swing_high
        )

        broke_down = (
            float(event.close)
            < swing_low - break_buffer

            and float(previous.close)
            >= swing_low
        )

        if not broke_up and not broke_down:
            continue

        subsequent = df.iloc[
            event_end:
        ].copy()

        hold_count = len(subsequent)

        if hold_count < MIN_POST_BREAK_CANDLES:
            continue

        latest_close = float(
            df.iloc[-1].close
        )

        if broke_up:

            if bias_before == "BEARISH":
                event_type = "CHOCH_UP"

            elif bias_before == "BULLISH":
                event_type = "BOS_UP"

            else:
                # Range/unknown structure is not accepted
                # as a high-confidence directional event.
                continue

            hold_tolerance = (
                event_atr *
                STRUCTURE_HOLD_ATR_TOLERANCE
            )

            hold_ok = bool(
                (
                    subsequent.close
                    > swing_high - hold_tolerance
                ).all()
                and
                latest_close
                > swing_high - hold_tolerance
            )

            if not hold_ok:
                continue

            return {
                "bos_up": event_type == "BOS_UP",
                "bos_down": False,

                "choch_up": event_type == "CHOCH_UP",
                "choch_down": False,

                "swing_high": swing_high,
                "swing_low": swing_low,

                "bias_before": bias_before,
                "bias": "BULLISH",

                "event_age": age,
                "break_level": swing_high,
                "break_close": float(event.close),

                "hold_ok": True,
                "hold_count": hold_count,

                "event_type": event_type,
                "latest_close": latest_close
            }

        if broke_down:

            if bias_before == "BULLISH":
                event_type = "CHOCH_DOWN"

            elif bias_before == "BEARISH":
                event_type = "BOS_DOWN"

            else:
                continue

            hold_tolerance = (
                event_atr *
                STRUCTURE_HOLD_ATR_TOLERANCE
            )

            hold_ok = bool(
                (
                    subsequent.close
                    < swing_low + hold_tolerance
                ).all()
                and
                latest_close
                < swing_low + hold_tolerance
            )

            if not hold_ok:
                continue

            return {
                "bos_up": False,
                "bos_down": event_type == "BOS_DOWN",

                "choch_up": False,
                "choch_down": event_type == "CHOCH_DOWN",

                "swing_high": swing_high,
                "swing_low": swing_low,

                "bias_before": bias_before,
                "bias": "BEARISH",

                "event_age": age,
                "break_level": swing_low,
                "break_close": float(event.close),

                "hold_ok": True,
                "hold_count": hold_count,

                "event_type": event_type,
                "latest_close": latest_close
            }

    # No valid breakout event.
    # Still return current structure context.

    highs, lows = confirmed_pivots(df)

    if highs and lows:

        bias = structure_bias(
            highs,
            lows
        )

        empty.update({
            "swing_high": float(highs[-1][1]),
            "swing_low": float(lows[-1][1]),
            "bias": bias,
            "bias_before": bias,
            "latest_close": float(df.iloc[-1].close)
        })

    return empty


# ============================================================
# PREMIUM / DISCOUNT
# ============================================================

def premium_discount(df, lookback=50):

    recent = df.iloc[-lookback:]

    high = float(
        recent.high.max()
    )

    low = float(
        recent.low.min()
    )

    equilibrium = (
        high + low
    ) / 2

    price = float(
        df.iloc[-1].close
    )

    if price < equilibrium:
        zone = "DISCOUNT"

    elif price > equilibrium:
        zone = "PREMIUM"

    else:
        zone = "EQUILIBRIUM"

    return {
        "zone": zone,
        "high": high,
        "low": low,
        "equilibrium": equilibrium,
        "price": price
    }


# ============================================================
# LIQUIDITY
# ============================================================

def liquidity_analysis(df, lookback=25):

    recent = df.iloc[
        -lookback - 1:-1
    ]

    current = df.iloc[-1]

    high = float(
        recent.high.max()
    )

    low = float(
        recent.low.min()
    )

    return {
        "sweep_high": bool(
            current.high > high
            and current.close < high
        ),

        "sweep_low": bool(
            current.low < low
            and current.close > low
        ),

        "equal_high": high,
        "equal_low": low
    }


# ============================================================
# FAIR VALUE GAP
# ============================================================

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


# ============================================================
# ORDER BLOCK
# ============================================================

def detect_order_block(
    df,
    side,
    lookback=12
):

    recent = df.iloc[
        -(lookback + 1):-1
    ]

    current = df.iloc[-1]

    if abs(
        float(current.close - current.open)
    ) < (
        0.65 *
        max(float(current.atr), 1e-12)
    ):
        return None

    if (
        side == "LONG"
        and current.close > current.open
    ):

        opposite = recent[
            recent.close < recent.open
        ]

    elif (
        side == "SHORT"
        and current.close < current.open
    ):

        opposite = recent[
            recent.close > recent.open
        ]

    else:
        return None

    if opposite.empty:
        return None

    candle = opposite.iloc[-1]

    return {
        "low": float(candle.low),
        "high": float(candle.high)
    }


# ============================================================
# FAST MOMENTUM DETECTOR
# ============================================================

def momentum_analysis(df5):

    output = {}

    price = float(
        df5.iloc[-1].close
    )

    for label, (
        bars,
        threshold
    ) in MOMENTUM_THRESHOLDS.items():

        start = float(
            df5.iloc[-bars - 1].close
        )

        change = (
            (price - start) / start * 100
            if start > 0
            else 0.0
        )

        window = df5.iloc[-bars:]

        baseline = df5.iloc[
            -(bars + 21):-bars
        ]

        average_volume = float(
            window.volume.mean()
        )

        baseline_volume = float(
            baseline.volume.mean()
        )

        expansion = (
            average_volume / baseline_volume
            if baseline_volume > 0
            else 0.0
        )

        direction = (
            "UP" if change > 0
            else "DOWN" if change < 0
            else "NONE"
        )

        atr_pct = (
            float(df5.iloc[-1].atr)
            / price * 100
            if price > 0
            else 0.0
        )

        adaptive_threshold = max(
            threshold,
            atr_pct * (
                1.1 if label == "30M" else 1.5
            )
        )

        strong = bool(
            abs(change) >= adaptive_threshold
            and expansion >= 1.05
        )

        output[label] = {
            "direction": direction,
            "change_pct": change,
            "volume_expansion": expansion,
            "strong": strong
        }

    return output


def momentum_summary(momentum):

    return " | ".join(
        (
            f"{key}={momentum[key]['direction']} "
            f"{momentum[key]['change_pct']:+.2f}% "
            f"(Vol x{momentum[key]['volume_expansion']:.2f}, "
            f"Strong={momentum[key]['strong']})"
        )
        for key in ("30M", "1H", "2H")
    )


# ============================================================
# 5M CANDLE CONFIRMATION
# ============================================================

def candle_confirmation(df, side):

    current = df.iloc[-1]
    previous = df.iloc[-2]

    body = abs(
        float(current.close - current.open)
    )

    atr = max(
        float(current.atr),
        1e-12
    )

    body_ratio = float(
        current.body_ratio
    )

    volume_ratio = float(
        current.volume_ratio
    )

    if side == "LONG":

        directional = (
            current.close > current.open
        )

        engulf = (
            directional
            and previous.close < previous.open
            and current.close >= previous.open
            and current.open <= previous.close
        )

        breakout = (
            current.close
            > float(df.high.iloc[-7:-1].max())
        )

        continuation = (
            directional
            and current.close > previous.high
        )

    else:

        directional = (
            current.close < current.open
        )

        engulf = (
            directional
            and previous.close > previous.open
            and current.close <= previous.open
            and current.open >= previous.close
        )

        breakout = (
            current.close
            < float(df.low.iloc[-7:-1].min())
        )

        continuation = (
            directional
            and current.close < previous.low
        )

    return bool(
        directional
        and body_ratio >= MIN_BODY_RATIO
        and body >= 0.40 * atr
        and (
            engulf
            or breakout
            or continuation
        )
        and volume_ratio >= MIN_VOLUME_RATIO
    )


# ============================================================
# RAPID MOVE RETEST / CONTINUATION
# ============================================================

def rapid_retest_confirmation(df, side):

    if len(df) < 35:

        return (
            False,
            "insufficient 5M history"
        )

    current = df.iloc[-1]
    previous = df.iloc[-2]

    prior = df.iloc[-7:-1]
    earlier = df.iloc[-13:-7]

    atr = max(
        float(current.atr),
        1e-12
    )

    body_ratio = float(
        current.body_ratio
    )

    volume_ratio = float(
        current.volume_ratio
    )

    if side == "LONG":

        level = float(
            earlier.high.max()
        )

        prior_break = (
            float(prior.close.max()) > level
        )

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

        level = float(
            earlier.low.min()
        )

        prior_break = (
            float(prior.close.min()) < level
        )

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

        return (
            True,
            "5M breakout retest"
        )

    if continuation:

        return (
            True,
            "5M strong continuation"
        )

    return (
        False,
        "no valid retest/continuation"
    )


# ============================================================
# STRONG 5M MOMENTUM
# ============================================================

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
        and abs(
            float(current.close - current.open)
        ) >= (
            0.60 *
            max(float(current.atr), 1e-12)
        )
    )


# ============================================================
# TRADE LEVELS
# ============================================================

def trade_levels(df, side):

    current = df.iloc[-1]

    entry = float(
        current.close
    )

    atr = max(
        float(current.atr),
        entry * 0.0005
    )

    risk = max(
        1.50 * atr,
        entry * 0.0015
    )

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


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(symbol):

    def reject(reason, diagnostic=""):

        log.info(
            "DIAGNOSTIC %s | REJECT: %s%s",
            symbol,
            reason,
            (" | " + diagnostic)
            if diagnostic else ""
        )

        return None

    try:

        # ----------------------------------------------------
        # FETCH TIMEFRAMES
        # ----------------------------------------------------

        d4 = fetch_candles(
            symbol,
            "4H"
        )

        d1 = fetch_candles(
            symbol,
            "1H"
        )

        d15 = fetch_candles(
            symbol,
            "15m"
        )

        d5 = fetch_candles(
            symbol,
            "5m"
        )

        # ----------------------------------------------------
        # HIGHER TIMEFRAME TREND
        # ----------------------------------------------------

        trend4 = ema_direction(d4)
        trend1 = ema_direction(d1)

        # ----------------------------------------------------
        # STRUCTURE AND CONTEXT
        # ----------------------------------------------------

        structure = market_structure(d15)

        pd_zone = premium_discount(d15)

        liquidity = liquidity_analysis(d15)

        momentum = momentum_analysis(d5)

        current = d5.iloc[-1]

        # ----------------------------------------------------
        # STRUCTURE DIRECTION VALIDATION
        # ----------------------------------------------------

        long_structure = bool(
            structure["hold_ok"]
            and structure["hold_count"] >= 1
            and structure["bias"] == "BULLISH"
            and (
                structure["bos_up"]
                or structure["choch_up"]
            )
        )

        short_structure = bool(
            structure["hold_ok"]
            and structure["hold_count"] >= 1
            and structure["bias"] == "BEARISH"
            and (
                structure["bos_down"]
                or structure["choch_down"]
            )
        )

        # ----------------------------------------------------
        # MOMENTUM DIRECTION
        # ----------------------------------------------------

        long_momentum = any(
            item["strong"]
            and item["direction"] == "UP"
            for item in momentum.values()
        )

        short_momentum = any(
            item["strong"]
            and item["direction"] == "DOWN"
            for item in momentum.values()
        )

        # ----------------------------------------------------
        # HTF ALIGNMENT
        # ----------------------------------------------------

        long_count = (
            int(trend4 == "BULLISH")
            + int(trend1 == "BULLISH")
        )

        short_count = (
            int(trend4 == "BEARISH")
            + int(trend1 == "BEARISH")
        )

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        diagnostic = (
            f"4H={trend4}, "
            f"1H={trend1}, "

            f"15M BOS U/D="
            f"{structure['bos_up']}/"
            f"{structure['bos_down']}, "

            f"CHOCH U/D="
            f"{structure['choch_up']}/"
            f"{structure['choch_down']}, "

            f"BiasBefore={structure['bias_before']}, "
            f"BiasAfter={structure['bias']}, "

            f"Event={structure['event_type']}, "
            f"Age={structure['event_age']}, "

            f"Break={structure['break_level']:.8g}, "

            f"Hold={structure['hold_ok']} "
            f"(candles={structure['hold_count']}), "

            f"PD={pd_zone['zone']}, "

            f"5M Vol={current.volume_ratio:.2f}, "
            f"Body={current.body_ratio:.2f}, "
            f"ADX={current.adx:.1f}, "

            f"{momentum_summary(momentum)}"
        )

        # ----------------------------------------------------
        # SIDE SELECTION
        # ----------------------------------------------------

        side = None
        rapid = False

        if (
            long_count == 2
            and long_structure
        ):

            side = "LONG"

        elif (
            short_count == 2
            and short_structure
        ):

            side = "SHORT"

        elif (
            long_count >= 1
            and short_count == 0
            and long_structure
            and long_momentum
        ):

            side = "LONG"
            rapid = True

        elif (
            short_count >= 1
            and long_count == 0
            and short_structure
            and short_momentum
        ):

            side = "SHORT"
            rapid = True

        if side is None:

            reasons = []

            if (
                long_count == 0
                and short_count == 0
            ):

                reasons.append(
                    "4H/1H trend mixed"
                )

            elif (
                long_count < 2
                and short_count < 2
            ):

                reasons.append(
                    "HTF trend not fully aligned"
                )

            if not long_structure and not short_structure:

                reasons.append(
                    "15M valid BOS/CHOCH or post-break hold absent"
                )

            if (
                long_structure
                and long_count >= 1
                and not long_momentum
            ):

                reasons.append(
                    "bullish momentum weak"
                )

            if (
                short_structure
                and short_count >= 1
                and not short_momentum
            ):

                reasons.append(
                    "bearish momentum weak"
                )

            return reject(
                "; ".join(reasons)
                or "directional setup not aligned",
                diagnostic
            )

        # ----------------------------------------------------
        # STRUCTURE SIDE MUST MATCH SIGNAL SIDE
        # ----------------------------------------------------

        if (
            side == "LONG"
            and structure["bias"] != "BULLISH"
        ):

            return reject(
                "15M post-break structure is not bullish",
                diagnostic
            )

        if (
            side == "SHORT"
            and structure["bias"] != "BEARISH"
        ):

            return reject(
                "15M post-break structure is not bearish",
                diagnostic
            )

        # ----------------------------------------------------
        # 15M EMA DIRECTION
        # ----------------------------------------------------

        ema15_bullish = (
            d15.iloc[-1].ema9
            > d15.iloc[-1].ema21
        )

        ema15_bearish = (
            d15.iloc[-1].ema9
            < d15.iloc[-1].ema21
        )

        if (
            side == "LONG"
            and not ema15_bullish
        ):

            return reject(
                "15M EMA disagrees with LONG",
                diagnostic
            )

        if (
            side == "SHORT"
            and not ema15_bearish
        ):

            return reject(
                "15M EMA disagrees with SHORT",
                diagnostic
            )

        # ----------------------------------------------------
        # 5M CONFIRMATION
        # ----------------------------------------------------

        standard_confirmation = (
            candle_confirmation(d5, side)
        )

        rapid_ok, rapid_reason = (
            rapid_retest_confirmation(d5, side)
        )

        strong5 = strong_5m(
            d5,
            side
        )

        if rapid:

            if not (
                standard_confirmation
                or rapid_ok
            ):

                return reject(
                    "rapid route: 5M confirmation failed",
                    diagnostic + " | " + rapid_reason
                )

        elif not standard_confirmation:

            return reject(
                "standard route: 5M closed-candle confirmation failed",
                diagnostic
            )

        # ----------------------------------------------------
        # MOMENTUM GATE
        # ----------------------------------------------------

        aligned_momentum = (
            long_momentum
            if side == "LONG"
            else short_momentum
        )

        if not (
            aligned_momentum
            or strong5
        ):

            return reject(
                "momentum gate failed",
                diagnostic
            )

        # ----------------------------------------------------
        # ADX QUALITY FILTER
        # ----------------------------------------------------

        adx_value = float(
            current.adx
        )

        weak_adx_exception = bool(
            strong5
            and current.volume_ratio
            >= WEAK_ADX_STRONG_VOLUME
            and current.body_ratio
            >= WEAK_ADX_STRONG_BODY
            and aligned_momentum
        )

        if (
            adx_value < MIN_ADX
            and not weak_adx_exception
        ):

            return reject(
                (
                    f"weak ADX {adx_value:.1f}; "
                    "strong 5M volume/body/momentum exception failed"
                ),
                diagnostic
            )

        # ----------------------------------------------------
        # PREMIUM / DISCOUNT LOCATION
        # ----------------------------------------------------

        favorable_location = (
            (
                side == "LONG"
                and pd_zone["zone"] == "DISCOUNT"
            )
            or
            (
                side == "SHORT"
                and pd_zone["zone"] == "PREMIUM"
            )
        )

        unfavorable_location = (
            (
                side == "LONG"
                and pd_zone["zone"] == "PREMIUM"
            )
            or
            (
                side == "SHORT"
                and pd_zone["zone"] == "DISCOUNT"
            )
        )

        # ----------------------------------------------------
        # SIGNAL SCORE
        # ----------------------------------------------------

        score = 0

        if (
            (
                side == "LONG"
                and long_count == 2
            )
            or
            (
                side == "SHORT"
                and short_count == 2
            )
        ):

            score += 2

        else:

            score += 1

        if (
            (
                side == "LONG"
                and structure["bos_up"]
            )
            or
            (
                side == "SHORT"
                and structure["bos_down"]
            )
        ):

            score += 2

        else:

            score += 1

        if (
            standard_confirmation
            or rapid_ok
        ):

            score += 2

        if adx_value >= MIN_ADX:
            score += 1

        if current.volume_ratio >= MIN_VOLUME_RATIO:
            score += 1

        if aligned_momentum:
            score += 1

        if (
            (
                side == "LONG"
                and liquidity["sweep_low"]
            )
            or
            (
                side == "SHORT"
                and liquidity["sweep_high"]
            )
        ):

            score += 1

        if favorable_location:
            score += 1

        # 15M EMA alignment already passed.
        score += 1

        # Opposite PD location reduces confidence.
        if unfavorable_location:
            score -= 1

        if score < MIN_SCORE:

            return reject(
                (
                    f"score too low: {score}/12; "
                    f"required {MIN_SCORE}"
                ),
                diagnostic
            )

        # ----------------------------------------------------
        # EXTENSION FILTER
        # ----------------------------------------------------

        price = float(
            current.close
        )

        atr = max(
            float(current.atr),
            1e-12
        )

        extension = abs(
            price - float(current.ema21)
        ) / atr

        extension_limit = (
            MAX_EXTENSION_RAPID
            if rapid
            else MAX_EXTENSION_STANDARD
        )

        if extension > extension_limit:

            return reject(
                (
                    f"overextended from EMA21 "
                    f"{extension:.2f} ATR; "
                    f"limit {extension_limit:.2f}"
                ),
                diagnostic
            )

        # ----------------------------------------------------
        # EMA VALUES FOR ALL TIMEFRAMES
        # ----------------------------------------------------

        ema_values = {}

        for timeframe, frame in [
            ("4H", d4),
            ("1H", d1),
            ("15M", d15),
            ("5M", d5)
        ]:

            ema_values[timeframe] = {
                f"EMA{period}": float(
                    frame.iloc[-1][f"ema{period}"]
                )
                for period in (
                    9, 21, 50, 200
                )
            }

        # ----------------------------------------------------
        # FINAL SIGNAL
        # ----------------------------------------------------

        result = {
            "symbol": symbol,
            "side": side,
            "score": score,

            **trade_levels(d5, side),

            "candle_ts": int(
                current.timestamp
            ),

            "trend4": trend4,
            "trend1": trend1,

            "ema_values": ema_values,

            "pd_zone": pd_zone,

            "momentum": momentum,

            "fvg": detect_fvg(d15),

            "ob": detect_order_block(
                d15,
                side
            ),

            "liquidity": liquidity,

            "structure": structure,

            "entry_mode": (
                "RAPID RETEST/CONTINUATION"
                if rapid
                else "STANDARD SMC"
            ),

            "confirmation": (
                rapid_reason
                if rapid and rapid_ok
                else "5M closed-candle confirmation"
            ),

            "rsi5": float(
                current.rsi
            ),

            "adx5": adx_value,

            "volume_ratio5": float(
                current.volume_ratio
            ),

            "favorable_location": favorable_location,

            "extension_atr": extension
        }

        log.info(
            "DIAGNOSTIC %s | CONFIRMED %s | "
            "score=%s/12 | %s",
            symbol,
            side,
            score,
            diagnostic
        )

        return result

    except Exception as exc:

        log.exception(
            "Analysis failed for %s: %s",
            symbol,
            exc
        )

        return None


# ============================================================
# STATE MANAGEMENT
# ============================================================

def load_state():

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            state = json.load(file)

        return (
            state
            if isinstance(state, dict)
            else {}
        )

    except FileNotFoundError:

        return {}

    except Exception as exc:

        log.warning(
            "State read failed: %s",
            exc
        )

        return {}


def save_state(state):

    temporary_file = (
        STATE_FILE + ".tmp"
    )

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            state,
            file,
            indent=2
        )

    os.replace(
        temporary_file,
        STATE_FILE
    )


# ============================================================
# PRICE FORMATTING
# ============================================================

def fmt(value):

    value = float(value)

    if value >= 1000:
        return f"{value:,.2f}"

    if value >= 1:
        return (
            f"{value:.5f}"
            .rstrip("0")
            .rstrip(".")
        )

    return (
        f"{value:.8f}"
        .rstrip("0")
        .rstrip(".")
    )


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def build_message(signal):

    direction = (
        "🟢 LONG"
        if signal["side"] == "LONG"
        else "🔴 SHORT"
    )

    ema_lines = []

    for timeframe in (
        "4H",
        "1H",
        "15M",
        "5M"
    ):

        values = signal[
            "ema_values"
        ][timeframe]

        ema_lines.append(
            f"EMA {timeframe}: "
            f"9={fmt(values['EMA9'])} | "
            f"21={fmt(values['EMA21'])} | "
            f"50={fmt(values['EMA50'])} | "
            f"200={fmt(values['EMA200'])}"
        )

    momentum_lines = []

    for key in (
        "30M",
        "1H",
        "2H"
    ):

        item = signal[
            "momentum"
        ][key]

        momentum_lines.append(
            f"{key}: "
            f"{item['direction']} "
            f"{item['change_pct']:+.2f}% | "
            f"Vol x{item['volume_expansion']:.2f} | "
            f"Strong={item['strong']}"
        )

    structure = signal["structure"]

    pd_info = signal["pd_zone"]

    fvg = signal["fvg"]

    order_block = signal["ob"]

    liquidity = signal["liquidity"]

    fvg_text = (
        (
            f"{fvg['type']}: "
            f"{fmt(fvg['low'])} - "
            f"{fmt(fvg['high'])}"
        )
        if fvg
        else "No fresh FVG detected"
    )

    ob_text = (
        (
            f"{fmt(order_block['low'])} - "
            f"{fmt(order_block['high'])}"
        )
        if order_block
        else "No valid OB detected"
    )

    if signal["side"] == "LONG":

        liquidity_text = (
            "Sell-side sweep detected"
            if liquidity["sweep_low"]
            else "No sell-side sweep"
        )

    else:

        liquidity_text = (
            "Buy-side sweep detected"
            if liquidity["sweep_high"]
            else "No buy-side sweep"
        )

    return (
        f"✅ FINAL CONFIRMED {direction}\n\n"

        f"Coin: {signal['symbol']}\n\n"

        f"Entry Area: "
        f"{fmt(signal['area_low'])} - "
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
        f"1H Trend: {signal['trend1']}\n\n"

        f"Premium/Discount: {pd_info['zone']}\n"

        f"Range High: {fmt(pd_info['high'])}\n"
        f"Range Low: {fmt(pd_info['low'])}\n"
        f"Equilibrium: {fmt(pd_info['equilibrium'])}\n\n"

        f"15M Structure:\n"

        f"BOS UP={structure['bos_up']} | "
        f"BOS DOWN={structure['bos_down']}\n"

        f"CHOCH UP={structure['choch_up']} | "
        f"CHOCH DOWN={structure['choch_down']}\n"

        f"Bias Before: {structure['bias_before']}\n"
        f"Bias After: {structure['bias']}\n"

        f"Event: {structure['event_type']}\n"

        f"Event Age: {structure['event_age']} candles\n"

        f"Break Level: {fmt(structure['break_level'])}\n"

        f"Hold: {structure['hold_ok']} "
        f"({structure['hold_count']} candles)\n\n"

        f"Liquidity: {liquidity_text}\n"

        f"FVG: {fvg_text}\n"

        f"Order Block: {ob_text}\n\n"

        + "\n".join(ema_lines)

        + "\n\nMomentum:\n"

        + "\n".join(momentum_lines)

        + "\n\n"

        + f"5M RSI: {signal['rsi5']:.2f}\n"

        + f"5M ADX: {signal['adx5']:.2f}\n"

        + f"5M Volume Ratio: "
        f"x{signal['volume_ratio5']:.2f}\n"

        + f"EMA21 Extension: "
        f"{signal['extension_atr']:.2f} ATR\n\n"

        + "Risk Note: Algorithmic analysis, "
        "not a guarantee of profit. "
        "No orders are placed automatically."
    )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TOKEN or not CHAT_ID:

        log.error(
            "Telegram credentials missing; "
            "message not delivered."
        )

        log.info("\n%s", message)

        return False

    try:

        response = SESSION.post(
            (
                "https://api.telegram.org/"
                f"bot{TOKEN}/sendMessage"
            ),

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

            raise RuntimeError(
                str(payload)
            )

        return True

    except Exception as exc:

        log.error(
            "Telegram send failed: %s",
            exc
        )

        return False


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    log.info(
        "Starting Master Gold + Top 10 Crypto SMC Bot v5.0"
    )

    log.info(
        "Timeframes: 4H / 1H / 15M / 5M"
    )

    log.info(
        "Closed candles only | "
        "Confirmed swing pivots enabled"
    )

    log.info(
        "15M BOS/CHOCH: maximum age=%s candles",
        STRUCTURE_MAX_AGE
    )

    log.info(
        "Post-break hold: minimum %s closed candle(s)",
        MIN_POST_BREAK_CANDLES
    )

    log.info(
        "15M breakout hold ATR tolerance=%.2f",
        STRUCTURE_HOLD_ATR_TOLERANCE
    )

    log.info(
        "5M confirmation + retest/continuation enabled"
    )

    log.info(
        "Weak ADX protection enabled"
    )

    log.info(
        "Premium/Discount location scoring enabled"
    )

    log.info(
        "30M/1H/2H momentum detector enabled"
    )

    log.info(
        "Telegram: FINAL CONFIRMED SIGNALS ONLY"
    )

    log.info(
        "Automatic trading: DISABLED"
    )

    if not TOKEN or not CHAT_ID:

        log.warning(
            "Telegram secrets missing; "
            "check GitHub Actions repository secrets."
        )

    # --------------------------------------------------------
    # SELECT SYMBOLS
    # --------------------------------------------------------

    try:

        symbols = select_symbols()

    except Exception as exc:

        log.exception(
            "Symbol selection failed: %s",
            exc
        )

        return

    if not symbols:

        log.error(
            "No eligible symbols found."
        )

        return

    # --------------------------------------------------------
    # LOAD STATE
    # --------------------------------------------------------

    state = load_state()

    signals = []

    # --------------------------------------------------------
    # SCAN ALL SYMBOLS
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as pool:

        futures = {
            pool.submit(
                analyze_symbol,
                symbol
            ): symbol

            for symbol in symbols
        }

        for future in as_completed(futures):

            symbol = futures[future]

            try:

                result = future.result()

                if result:

                    signals.append(result)

                    log.info(
                        "Confirmed candidate: "
                        "%s %s | Score=%s/12",
                        symbol,
                        result["side"],
                        result["score"]
                    )

                else:

                    log.info(
                        "No final confirmation: %s",
                        symbol
                    )

            except Exception as exc:

                log.exception(
                    "Worker failed for %s: %s",
                    symbol,
                    exc
                )

    # --------------------------------------------------------
    # SORT SIGNALS
    # --------------------------------------------------------

    signals.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    sent = 0

    # --------------------------------------------------------
    # SEND TELEGRAM SIGNALS
    # --------------------------------------------------------

    for signal in signals:

        key = (
            f"{signal['symbol']}:"
            f"{signal['side']}"
        )

        candle = str(
            signal["candle_ts"]
        )

        if state.get(key) == candle:

            log.info(
                "Duplicate signal skipped: "
                "%s %s",
                signal["symbol"],
                signal["side"]
            )

            continue

        message = build_message(
            signal
        )

        if send_telegram(message):

            state[key] = candle

            save_state(state)

            sent += 1

            log.info(
                "Telegram signal sent: "
                "%s %s",
                signal["symbol"],
                signal["side"]
            )

    # --------------------------------------------------------
    # SCAN SUMMARY
    # --------------------------------------------------------

    log.info(
        "Scan complete | "
        "Symbols scanned=%s | "
        "Confirmed=%s | "
        "Sent=%s",
        len(symbols),
        len(signals),
        sent
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
