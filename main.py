import os
import json
import time
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

# ============================================================
# GOLD + TOP 19 CRYPTO BITGET USDT-M FUTURES SMC BOT
# Total target: 1 Gold + 19 Crypto = 20 symbols
# Scans 4H / 1H / 15M / 5M. Sends CONFIRMED signals only.
# No Early Warning and no Fast Momentum alerts.
# ============================================================

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
STATE_FILE = os.getenv("STATE_FILE", "bot_state.json")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

GOLD_SYMBOL = "XAUUSDT"
CRYPTO_TOP_N = 19
TOP_N = 20

MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "10000000"))
MIN_SCORE = int(os.getenv("MIN_SCORE", "7"))
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", "1.15"))
MIN_SMC_SCORE = int(os.getenv("MIN_SMC_SCORE", "1"))
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", "250"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))

TOP_SYMBOLS = [
    x.strip().upper()
    for x in os.getenv("TOP_SYMBOLS", "").split(",")
    if x.strip()
]

INTERVAL_MS = {
    "5m": 300_000,
    "15m": 900_000,
    "1H": 3_600_000,
    "4H": 14_400_000,
}

# Stablecoins and non-target assets
BASE_COIN_EXCLUSIONS = {
    "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDE",
    "USDD", "USTC", "EUR", "GBP", "PAXG"
}

# Gold is included separately; Silver is excluded.
NON_CRYPTO_EXCLUSIONS = {
    "XAGUSDT",
    "NVDAUSDT",
    "SNDKUSDT",
    "SOXLUSDT",
    "SOXSUSDT",
    "KORUUSDT",
    "SNXXUSDT",
    "SPCXUSDT",
}

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s"
)

log = logging.getLogger("GOLD-TOP19-SMC-BOT")


# ---------------- HTTP / TELEGRAM ----------------

def api_get(path, params=None):
    last_error = None

    for attempt in range(RETRY_COUNT):
        try:
            response = requests.get(
                BASE_URL + path,
                params=params or {},
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "Gold-Top19-SMC-Bot/1.0"}
            )

            if response.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )

            response.raise_for_status()
            payload = response.json()

            if payload.get("code") != "00000":
                raise RuntimeError(
                    f"Bitget API error {payload.get('code')}: "
                    f"{payload.get('msg')}"
                )

            return payload.get("data", [])

        except Exception as exc:
            last_error = exc
            wait = min(2 ** attempt, 8)

            log.warning(
                "API retry %s/%s for %s: %s",
                attempt + 1, RETRY_COUNT, path, exc
            )

            if attempt + 1 < RETRY_COUNT:
                time.sleep(wait)

    raise RuntimeError(f"Bitget request failed: {last_error}")


def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise RuntimeError(
            "Missing TELEGRAM_TOKEN or CHAT_ID GitHub Secret"
        )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message,
            "disable_web_page_preview": True
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()
    payload = response.json()

    if not payload.get("ok"):
        raise RuntimeError(f"Telegram error: {payload}")


# ---------------- SYMBOL UNIVERSE ----------------

def is_eligible_crypto(symbol):
    symbol = str(symbol).upper()

    if not symbol.endswith("USDT"):
        return False

    if symbol == GOLD_SYMBOL:
        return False

    if symbol in NON_CRYPTO_EXCLUSIONS:
        return False

    base = symbol[:-4]

    if base in BASE_COIN_EXCLUSIONS:
        return False

    if any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES):
        return False

    return True


def select_top_symbols():
    """
    Select:
    1. Gold XAUUSDT
    2. Top 19 eligible crypto USDT futures by 24h quote volume.

    Silver and listed stock/leveraged symbols are excluded.
    If fewer than 19 coins meet the volume threshold, the
    remaining positions are filled from lower-volume eligible
    crypto contracts.
    """

    tickers = api_get(
        "/api/v2/mix/market/tickers",
        {"productType": PRODUCT_TYPE}
    )

    contracts = api_get(
        "/api/v2/mix/market/contracts",
        {"productType": PRODUCT_TYPE}
    )

    contract_map = {
        str(c.get("symbol", "")).upper(): c
        for c in contracts
        if c.get("symbol")
    }

    # Check that Gold exists on Bitget USDT-FUTURES.
    if GOLD_SYMBOL not in contract_map:
        raise RuntimeError(
            f"{GOLD_SYMBOL} Gold contract not found on Bitget USDT-FUTURES."
        )

    # Check Gold ticker availability.
    gold_ticker = next(
        (
            t for t in tickers
            if str(t.get("symbol", "")).upper() == GOLD_SYMBOL
        ),
        None
    )

    if gold_ticker is None:
        raise RuntimeError(
            f"{GOLD_SYMBOL} ticker not available from Bitget."
        )

    # Manual TOP_SYMBOLS can provide preferred crypto symbols.
    # They still pass the same exclusion and contract checks.
    preferred_symbols = []

    for symbol in TOP_SYMBOLS:
        if not is_eligible_crypto(symbol):
            continue

        if symbol not in contract_map:
            continue

        if symbol not in preferred_symbols:
            preferred_symbols.append(symbol)

    ranked = []

    for ticker in tickers:
        symbol = str(ticker.get("symbol", "")).upper()

        if not is_eligible_crypto(symbol):
            continue

        if symbol not in contract_map:
            continue

        contract = contract_map[symbol]

        status = str(
            contract.get(
                "symbolStatus",
                contract.get("status", "normal")
            )
        ).lower()

        if status and status not in (
            "normal", "listed", "trading", "online"
        ):
            continue

        try:
            quote_volume = float(
                ticker.get("quoteVolume", 0) or 0
            )

            last_price = float(
                ticker.get("lastPr", 0) or 0
            )

        except (TypeError, ValueError):
            continue

        if last_price <= 0 or quote_volume <= 0:
            continue

        ranked.append((symbol, quote_volume))

    ranked.sort(
        key=lambda item: item[1],
        reverse=True
    )

    volume_map = {
        symbol: volume
        for symbol, volume in ranked
    }

    # Preferred symbols first, if TOP_SYMBOLS is configured.
    selected_crypto = []

    for symbol in preferred_symbols:
        if symbol in volume_map and symbol not in selected_crypto:
            selected_crypto.append(symbol)

    # Fill remaining positions by highest volume.
    for symbol, volume in ranked:
        if symbol in selected_crypto:
            continue

        if len(selected_crypto) >= CRYPTO_TOP_N:
            break

        selected_crypto.append(symbol)

    if len(selected_crypto) < CRYPTO_TOP_N:
        log.warning(
            "Only %s eligible crypto contracts found. "
            "Target is 19.",
            len(selected_crypto)
        )

    # Gold always stays in the list.
    symbols = [GOLD_SYMBOL] + selected_crypto[:CRYPTO_TOP_N]

    log.info(
        "Selected %s symbols: %s",
        len(symbols),
        ", ".join(symbols)
    )

    log.info(
        "Gold included: %s | Crypto selected: %s",
        GOLD_SYMBOL,
        len(selected_crypto[:CRYPTO_TOP_N])
    )

    return symbols


# ---------------- INDICATORS ----------------

def indicators(df):
    df = df.copy()
    c, h, low = df.close, df.high, df.low

    for period in (9, 20, 21, 50, 200):
        df[f"ema{period}"] = c.ewm(
            span=period,
            adjust=False
        ).mean()

    delta = c.diff()

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

    previous_close = c.shift(1)

    tr = pd.concat(
        [
            h - low,
            (h - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    df["vol_avg"] = df.volume.rolling(20).mean().shift(1)

    df["vol_ratio"] = (
        df.volume / df.vol_avg.replace(0, float("nan"))
    )

    up, down = h.diff(), -low.diff()

    plus_dm = pd.Series(
        [
            u if u > d and u > 0 else 0
            for u, d in zip(up, down)
        ],
        index=df.index
    )

    minus_dm = pd.Series(
        [
            d if d > u and d > 0 else 0
            for u, d in zip(up, down)
        ],
        index=df.index
    )

    smoothed_atr = tr.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    plus_di = (
        100 * plus_dm.ewm(alpha=1/14, adjust=False).mean()
        / smoothed_atr.replace(0, float("nan"))
    )

    minus_di = (
        100 * minus_dm.ewm(alpha=1/14, adjust=False).mean()
        / smoothed_atr.replace(0, float("nan"))
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, float("nan"))
    )

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["adx"] = dx.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    return (
        df.replace(
            [float("inf"), float("-inf")],
            float("nan")
        )
        .dropna()
        .reset_index(drop=True)
    )


# ---------------- CLOSED CANDLES ----------------

def fetch_candles(symbol, granularity):
    if granularity not in INTERVAL_MS:
        raise ValueError(
            f"Unsupported interval: {granularity}"
        )

    rows = api_get(
        "/api/v2/mix/market/candles",
        {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "granularity": granularity,
            "limit": str(CANDLE_LIMIT)
        }
    )

    if not rows:
        raise RuntimeError(
            f"{symbol} {granularity}: empty candle response"
        )

    df = pd.DataFrame(
        [
            list(row[:7])
            for row in rows
            if isinstance(row, (list, tuple)) and len(row) >= 7
        ],
        columns=[
            "timestamp", "open", "high", "low",
            "close", "volume", "turnover"
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

    now_ms = int(time.time() * 1000)
    interval = INTERVAL_MS[granularity]

    df = df[
        df.timestamp + interval <= now_ms - 2000
    ].reset_index(drop=True)

    if len(df) < 205:
        raise RuntimeError(
            f"{symbol} {granularity}: only {len(df)} "
            f"closed candles; need 205"
        )

    result = indicators(df)

    if len(result) < 5:
        raise RuntimeError(
            f"{symbol} {granularity}: indicator data insufficient"
        )

    return result


# ---------------- MARKET STRUCTURE ----------------

def pivots(df, left=3, right=3, limit=15):
    highs = df.high.to_numpy()
    lows = df.low.to_numpy()

    swing_highs, swing_lows = [], []

    for i in range(left, len(df) - right):
        h_window = highs[i-left:i+right+1]
        l_window = lows[i-left:i+right+1]

        if (
            highs[i] == max(h_window)
            and list(h_window).count(highs[i]) == 1
        ):
            swing_highs.append({
                "index": i,
                "price": float(highs[i])
            })

        if (
            lows[i] == min(l_window)
            and list(l_window).count(lows[i]) == 1
        ):
            swing_lows.append({
                "index": i,
                "price": float(lows[i])
            })

    return swing_highs[-limit:], swing_lows[-limit:]


def market_structure(df):
    sh, sl = pivots(df)

    last = df.iloc[-1]
    previous = df.iloc[-2]

    sh = [
        p for p in sh
        if p["index"] < len(df) - 4
    ]

    sl = [
        p for p in sl
        if p["index"] < len(df) - 4
    ]

    hi = (
        sh[-1]["price"]
        if sh
        else float(df.high.iloc[-30:-3].max())
    )

    lo = (
        sl[-1]["price"]
        if sl
        else float(df.low.iloc[-30:-3].min())
    )

    trend = "NEUTRAL"

    if len(sh) >= 2 and len(sl) >= 2:
        higher_high = sh[-1]["price"] > sh[-2]["price"]
        higher_low = sl[-1]["price"] > sl[-2]["price"]

        lower_high = sh[-1]["price"] < sh[-2]["price"]
        lower_low = sl[-1]["price"] < sl[-2]["price"]

        if higher_high and higher_low:
            trend = "BULLISH"

        elif lower_high and lower_low:
            trend = "BEARISH"

    up = bool(
        last.close > hi
        and previous.close <= hi
    )

    down = bool(
        last.close < lo
        and previous.close >= lo
    )

    return {
        "swing_high": hi,
        "swing_low": lo,
        "swing_highs": sh,
        "swing_lows": sl,
        "trend": trend,
        "bos_up": bool(up and trend != "BEARISH"),
        "bos_down": bool(down and trend != "BULLISH"),
        "choch_up": bool(up and trend == "BEARISH"),
        "choch_down": bool(down and trend == "BULLISH")
    }


def equal_levels(points, atr, price, kind):
    if len(points) < 2:
        return []

    tolerance = max(
        price * 0.001,
        atr * 0.15
    )

    found = []

    for i, point in enumerate(points):
        group = [
            item["price"]
            for item in points[i:]
            if abs(item["price"] - point["price"]) <= tolerance
        ]

        if len(group) >= 2:
            level = sum(group) / len(group)

            if not any(
                abs(level - item["level"]) <= tolerance
                for item in found
            ):
                found.append({
                    "kind": kind,
                    "level": level,
                    "touches": len(group)
                })

    return found[-3:]


def liquidity_analysis(df, structure):
    last = df.iloc[-1]

    eqh = equal_levels(
        structure["swing_highs"],
        float(last.atr),
        float(last.close),
        "EQH"
    )

    eql = equal_levels(
        structure["swing_lows"],
        float(last.atr),
        float(last.close),
        "EQL"
    )

    highs = (
        [x["price"] for x in structure["swing_highs"][-5:]]
        + [x["level"] for x in eqh]
    )

    lows = (
        [x["price"] for x in structure["swing_lows"][-5:]]
        + [x["level"] for x in eql]
    )

    swept_high = next(
        (
            x for x in sorted(set(highs), reverse=True)
            if last.high > x and last.close < x
        ),
        None
    )

    swept_low = next(
        (
            x for x in sorted(set(lows))
            if last.low < x and last.close > x
        ),
        None
    )

    return {
        "eqh": eqh,
        "eql": eql,
        "high_pools": sorted(set(highs), reverse=True)[:6],
        "low_pools": sorted(set(lows))[:6],
        "sweep_high": swept_high is not None,
        "sweep_low": swept_low is not None,
        "sweep_high_level": swept_high,
        "sweep_low_level": swept_low
    }


# ---------------- ORDER BLOCKS / FVG ----------------

def detect_order_block(df, side, lookback=35):
    start = max(5, len(df) - lookback)
    zones = []

    for i in range(start + 3, len(df)):
        candle = df.iloc[i]

        atr = max(
            float(candle.atr),
            1e-9
        )

        body = abs(
            float(candle.close - candle.open)
        )

        if side == "LONG":
            impulse = (
                candle.close > candle.open
                and body >= 0.8 * atr
                and candle.close > df.high.iloc[i-3:i].max()
            )
        else:
            impulse = (
                candle.close < candle.open
                and body >= 0.8 * atr
                and candle.close < df.low.iloc[i-3:i].min()
            )

        if not impulse:
            continue

        ob_index = None

        for j in range(
            i - 1,
            max(start - 1, i - 8),
            -1
        ):
            prior = df.iloc[j]

            if (
                side == "LONG"
                and prior.close < prior.open
            ) or (
                side == "SHORT"
                and prior.close > prior.open
            ):
                ob_index = j
                break

        if ob_index is None:
            continue

        ob = df.iloc[ob_index]

        zone_low = float(ob.low)
        zone_high = float(ob.high)

        later = df.iloc[ob_index + 1:]

        if side == "LONG":
            penetration = (
                zone_high - float(later.low.min())
            ) / max(zone_high-zone_low, 1e-9)

            invalid = bool(
                (later.close < zone_low).any()
            )

        else:
            penetration = (
                float(later.high.max()) - zone_low
            ) / max(zone_high-zone_low, 1e-9)

            invalid = bool(
                (later.close > zone_high).any()
            )

        mitigation = round(
            max(0.0, min(1.0, penetration)) * 100,
            1
        )

        if invalid:
            status = "INVALIDATED"

        elif mitigation >= 100:
            status = "FULLY MITIGATED"

        elif mitigation > 0:
            status = "PARTIALLY MITIGATED"

        else:
            status = "FRESH"

        zones.append({
            "side": side,
            "low": zone_low,
            "high": zone_high,
            "mitigation_pct": mitigation,
            "status": status,
            "timestamp": int(ob.timestamp)
        })

    active = [
        z for z in zones
        if z["status"] in (
            "FRESH",
            "PARTIALLY MITIGATED"
        )
    ]

    return (active or zones)[-1] if zones else None


def detect_fvgs(df, lookback=40):
    zones = []

    for i in range(
        max(2, len(df)-lookback),
        len(df)
    ):
        a = df.iloc[i-2]
        b = df.iloc[i-1]
        c = df.iloc[i]

        body = abs(
            float(b.close - b.open)
        )

        atr = max(
            float(b.atr),
            1e-9
        )

        if (
            a.high < c.low
            and b.close > b.open
            and body >= 0.25 * atr
        ):
            low = float(a.high)
            high = float(c.low)

            later = df.iloc[i+1:]

            penetration = (
                (high - float(later.low.min()))
                / max(high-low, 1e-9)
                if len(later)
                else 0
            )

            invalid = (
                bool((later.close < low).any())
                if len(later)
                else False
            )

            fill = round(
                max(0.0, min(1.0, penetration))*100,
                1
            )

            status = (
                "INVALIDATED"
                if invalid
                else (
                    "FULLY FILLED"
                    if fill >= 100
                    else (
                        "PARTIALLY FILLED"
                        if fill > 0
                        else "FRESH"
                    )
                )
            )

            zones.append({
                "side": "BULLISH",
                "low": low,
                "high": high,
                "fill_pct": fill,
                "status": status
            })

        if (
            a.low > c.high
            and b.close < b.open
            and body >= 0.25 * atr
        ):
            low = float(c.high)
            high = float(a.low)

            later = df.iloc[i+1:]

            penetration = (
                (float(later.high.max()) - low)
                / max(high-low, 1e-9)
                if len(later)
                else 0
            )

            invalid = (
                bool((later.close > high).any())
                if len(later)
                else False
            )

            fill = round(
                max(0.0, min(1.0, penetration))*100,
                1
            )

            status = (
                "INVALIDATED"
                if invalid
                else (
                    "FULLY FILLED"
                    if fill >= 100
                    else (
                        "PARTIALLY FILLED"
                        if fill > 0
                        else "FRESH"
                    )
                )
            )

            zones.append({
                "side": "BEARISH",
                "low": low,
                "high": high,
                "fill_pct": fill,
                "status": status
            })

    return zones


def latest_fvg(df, side):
    zones = [
        z for z in detect_fvgs(df)
        if z["side"] == side
    ]

    active = [
        z for z in zones
        if z["status"] in (
            "FRESH",
            "PARTIALLY FILLED"
        )
    ]

    return (active or zones)[-1] if zones else None


def dealing_range(df, lookback=50, price_override=None):
    sample = df.tail(lookback)

    high = float(sample.high.max())
    low = float(sample.low.min())

    midpoint = (high + low) / 2
    span = max(high-low, 1e-9)

    price = float(
        df.close.iloc[-1]
        if price_override is None
        else price_override
    )

    return {
        "high": high,
        "low": low,
        "midpoint": midpoint,
        "premium": price > midpoint,
        "discount": price < midpoint,
        "position_pct": (price-low)/span*100
    }


def smc_engine(d1, d15):
    structure1 = market_structure(d1)
    structure15 = market_structure(d15)

    liquidity = liquidity_analysis(
        d15,
        structure15
    )

    ob_long = detect_order_block(d15, "LONG")
    ob_short = detect_order_block(d15, "SHORT")

    fvg_long = latest_fvg(d15, "BULLISH")
    fvg_short = latest_fvg(d15, "BEARISH")

    price = float(d15.close.iloc[-1])

    atr = max(
        float(d15.atr.iloc[-1]),
        1e-9
    )

    dealing = dealing_range(
        d1,
        50,
        price
    )

    def near(zone):
        if not zone or zone.get("status") not in (
            "FRESH",
            "PARTIALLY MITIGATED",
            "PARTIALLY FILLED"
        ):
            return False

        midpoint = (
            zone["low"] + zone["high"]
        ) / 2

        return (
            zone["low"] <= price <= zone["high"]
            or abs(price-midpoint) <= atr
        )

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    if structure15["bos_up"] or structure15["choch_up"]:
        long_score += 2
        long_reasons.append("15M bullish BOS/CHOCH")

    if structure15["bos_down"] or structure15["choch_down"]:
        short_score += 2
        short_reasons.append("15M bearish BOS/CHOCH")

    if liquidity["sweep_low"]:
        long_score += 2
        long_reasons.append("Sell-side liquidity sweep")

    if liquidity["sweep_high"]:
        short_score += 2
        short_reasons.append("Buy-side liquidity sweep")

    if near(ob_long):
        long_score += 1
        long_reasons.append("Near bullish order block")

    if near(ob_short):
        short_score += 1
        short_reasons.append("Near bearish order block")

    if near(fvg_long):
        long_score += 1
        long_reasons.append("Near bullish FVG")

    if near(fvg_short):
        short_score += 1
        short_reasons.append("Near bearish FVG")

    if dealing["discount"]:
        long_score += 1
        long_reasons.append("Discount range")

    if dealing["premium"]:
        short_score += 1
        short_reasons.append("Premium range")

    return {
        "structure1": structure1,
        "structure15": structure15,
        "liquidity": liquidity,
        "ob_long": ob_long,
        "ob_short": ob_short,
        "fvg_long": fvg_long,
        "fvg_short": fvg_short,
        "dealing": dealing,
        "long_score": long_score,
        "short_score": short_score,
        "long_reasons": long_reasons,
        "short_reasons": short_reasons,
        "long_ok": long_score >= MIN_SMC_SCORE,
        "short_ok": short_score >= MIN_SMC_SCORE
    }


# ---------------- CONFIRMATION / LEVELS ----------------

def timeframe_confirmation(df, lookback=5):
    last = df.iloc[-1]

    prior_high = float(
        df.high.iloc[-lookback-1:-1].max()
    )

    prior_low = float(
        df.low.iloc[-lookback-1:-1].min()
    )

    candle_range = max(
        float(last.high-last.low),
        1e-9
    )

    body_ratio = (
        abs(float(last.close-last.open))
        / candle_range
    )

    long_ok = (
        last.close > prior_high
        and last.close > last.ema9 > last.ema21
        and last.rsi >= 52
        and last.close > last.open
        and body_ratio >= 0.45
    )

    short_ok = (
        last.close < prior_low
        and last.close < last.ema9 < last.ema21
        and last.rsi <= 48
        and last.close < last.open
        and body_ratio >= 0.45
    )

    return {
        "long": bool(long_ok),
        "short": bool(short_ok),
        "price": float(last.close),
        "rsi": float(last.rsi),
        "timestamp": int(last.timestamp),
        "vol_ratio": float(last.vol_ratio),
        "adx": float(last.adx),
        "plus_di": float(last.plus_di),
        "minus_di": float(last.minus_di),
        "atr": float(last.atr)
    }


def volume_direction_ok(row, side):
    if float(row.vol_ratio) < MIN_VOLUME_RATIO:
        return False

    if side == "LONG":
        return (
            row.close > row.open
            and row.plus_di > row.minus_di
        )

    return (
        row.close < row.open
        and row.minus_di > row.plus_di
    )


def rsi_not_exhausted(side, rsi):
    if side == "LONG":
        return 30 < rsi < 75

    return 25 < rsi < 70


def build_levels(side, entry, atr):
    risk = max(
        1.5 * float(atr),
        entry * 0.0005
    )

    if side == "LONG":
        return (
            entry,
            entry-risk,
            entry+risk,
            entry+2*risk,
            entry+3*risk
        )

    return (
        entry,
        entry+risk,
        entry-risk,
        entry-2*risk,
        entry-3*risk
    )


def analyze_symbol(symbol):
    frames = {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(fetch_candles, symbol, tf): tf
            for tf in ("4H", "1H", "15m", "5m")
        }

        for future in as_completed(futures):
            tf = futures[future]
            frames[tf] = future.result()

    d4 = frames["4H"]
    d1 = frames["1H"]
    d15 = frames["15m"]
    d5 = frames["5m"]

    a4 = d4.iloc[-1]
    a1 = d1.iloc[-1]
    a15 = d15.iloc[-1]
    a5 = d5.iloc[-1]

    smc = smc_engine(d1, d15)

    structure15 = smc["structure15"]

    c15 = timeframe_confirmation(d15)
    c5 = timeframe_confirmation(d5)

    bull4 = bool(
        a4.ema20 > a4.ema50
        and a4.close > a4.ema200
    )

    bear4 = bool(
        a4.ema20 < a4.ema50
        and a4.close < a4.ema200
    )

    bull1 = bool(
        a1.ema20 > a1.ema50
        and a1.close > a1.ema200
    )

    bear1 = bool(
        a1.ema20 < a1.ema50
        and a1.close < a1.ema200
    )

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    def add(condition, points, side, reason):
        nonlocal long_score, short_score

        if condition:
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

    add(
        smc["structure1"]["bos_up"]
        or smc["structure1"]["choch_up"],
        2,
        "LONG",
        "1H BOS/CHOCH"
    )

    add(
        smc["structure1"]["bos_down"]
        or smc["structure1"]["choch_down"],
        2,
        "SHORT",
        "1H BOS/CHOCH"
    )

    add(
        structure15["bos_up"]
        or structure15["choch_up"],
        2,
        "LONG",
        "15M BOS/CHOCH"
    )

    add(
        structure15["bos_down"]
        or structure15["choch_down"],
        2,
        "SHORT",
        "15M BOS/CHOCH"
    )

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

    if a15.vol_ratio >= MIN_VOLUME_RATIO:
        add(
            True,
            1,
            "LONG" if a15.close > a15.open else "SHORT",
            "15M volume expansion"
        )

    long_score += smc["long_score"]
    short_score += smc["short_score"]

    long_reasons += [
        "SMC: " + x
        for x in smc["long_reasons"]
    ]

    short_reasons += [
        "SMC: " + x
        for x in smc["short_reasons"]
    ]

    add(
        c15["long"],
        1,
        "LONG",
        "15M candle confirmation"
    )

    add(
        c5["long"],
        2,
        "LONG",
        "5M candle confirmation"
    )

    add(
        c15["short"],
        1,
        "SHORT",
        "15M candle confirmation"
    )

    add(
        c5["short"],
        2,
        "SHORT",
        "5M candle confirmation"
    )

    long_htf = (
        (bull4 or bull1)
        and not (bear4 and bear1)
    )

    short_htf = (
        (bear4 or bear1)
        and not (bull4 and bull1)
    )

    long_volume = (
        volume_direction_ok(a15, "LONG")
        and volume_direction_ok(a5, "LONG")
    )

    short_volume = (
        volume_direction_ok(a15, "SHORT")
        and volume_direction_ok(a5, "SHORT")
    )

    long_rsi = (
        rsi_not_exhausted("LONG", float(a15.rsi))
        and rsi_not_exhausted("LONG", float(a5.rsi))
    )

    short_rsi = (
        rsi_not_exhausted("SHORT", float(a15.rsi))
        and rsi_not_exhausted("SHORT", float(a5.rsi))
    )

    final_side = None

    if (
        c15["long"]
        and c5["long"]
        and (
            structure15["bos_up"]
            or structure15["choch_up"]
        )
        and long_htf
        and long_volume
        and long_rsi
        and smc["long_ok"]
        and long_score >= MIN_SCORE
        and long_score > short_score
    ):
        final_side = "LONG"

    elif (
        c15["short"]
        and c5["short"]
        and (
            structure15["bos_down"]
            or structure15["choch_down"]
        )
        and short_htf
        and short_volume
        and short_rsi
        and smc["short_ok"]
        and short_score >= MIN_SCORE
        and short_score > long_score
    ):
        final_side = "SHORT"

    levels = (
        build_levels(
            final_side,
            float(a5.close),
            float(a5.atr)
        )
        if final_side
        else None
    )

    return {
        "symbol": symbol,
        "final": final_side,
        "levels": levels,
        "price": float(a15.close),
        "long_score": long_score,
        "short_score": short_score,
        "long_reasons": long_reasons,
        "short_reasons": short_reasons,
        "bull4": bull4,
        "bear4": bear4,
        "bull1": bull1,
        "bear1": bear1,
        "bos_up": structure15["bos_up"],
        "bos_down": structure15["bos_down"],
        "choch_up": structure15["choch_up"],
        "choch_down": structure15["choch_down"],
        "confirm15_long": c15["long"],
        "confirm15_short": c15["short"],
        "confirm5_long": c5["long"],
        "confirm5_short": c5["short"],
        "rsi15": float(a15.rsi),
        "rsi5": float(a5.rsi),
        "vol15": float(a15.vol_ratio),
        "vol5": float(a5.vol_ratio),
        "smc": smc,
        "candle_ts": int(a15.timestamp),
        "candle5_ts": int(a5.timestamp)
    }


# ---------------- TELEGRAM MESSAGE ----------------

def fmt(value):
    value = float(value)

    if abs(value) >= 1000:
        return f"{value:,.2f}"

    if abs(value) >= 1:
        return f"{value:,.5f}".rstrip("0").rstrip(".")

    return f"{value:.8f}".rstrip("0").rstrip(".")


def utc_string(timestamp_ms):
    return datetime.fromtimestamp(
        timestamp_ms/1000,
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")


def zone_text(zone, pct_key):
    if not zone:
        return "Not detected"

    pct = zone.get(pct_key, 0)

    label = (
        "Mitigation"
        if pct_key == "mitigation_pct"
        else "Fill"
    )

    return (
        f"{fmt(zone['low'])} - {fmt(zone['high'])} "
        f"[{zone['status']} | {label}: {pct:.1f}%]"
    )


def make_message(result):
    side = result["final"]

    smc = result["smc"]
    liquidity = smc["liquidity"]
    dealing = smc["dealing"]

    reasons = (
        result["long_reasons"]
        if side == "LONG"
        else result["short_reasons"]
    )

    entry, stop, tp1, tp2, tp3 = result["levels"]

    lines = [
        "✅ FINAL CONFIRMED SMC SIGNAL",
        "",
        "BITGET USDT-M FUTURES",
        f"{result['symbol']}: {side}",
        f"Price: {fmt(result['price'])}",
        (
            f"LONG Score: {result['long_score']} | "
            f"SHORT Score: {result['short_score']}"
        ),
        "",
        (
            f"4H Bull/Bear: "
            f"{result['bull4']} / {result['bear4']}"
        ),
        (
            f"1H Bull/Bear: "
            f"{result['bull1']} / {result['bear1']}"
        ),
        (
            f"15M BOS UP/DOWN: "
            f"{result['bos_up']} / {result['bos_down']}"
        ),
        (
            f"15M CHOCH UP/DOWN: "
            f"{result['choch_up']} / {result['choch_down']}"
        ),
        (
            f"15M Confirm L/S: "
            f"{result['confirm15_long']} / "
            f"{result['confirm15_short']}"
        ),
        (
            f"5M Confirm L/S: "
            f"{result['confirm5_long']} / "
            f"{result['confirm5_short']}"
        ),
        "",
        (
            f"15M RSI: {result['rsi15']:.1f} | "
            f"5M RSI: {result['rsi5']:.1f}"
        ),
        (
            f"15M Volume: {result['vol15']:.2f}x | "
            f"5M Volume: {result['vol5']:.2f}x"
        ),
        "",
        "----- SMC DETAILS -----",
        f"15M Structure: {smc['structure15']['trend']}",
        (
            f"Liquidity Sweep High/Low: "
            f"{liquidity['sweep_high']} / "
            f"{liquidity['sweep_low']}"
        ),
        (
            f"EQH/EQL pools: "
            f"{len(liquidity['eqh'])} / "
            f"{len(liquidity['eql'])}"
        ),
        (
            f"Bullish OB: "
            f"{zone_text(smc['ob_long'], 'mitigation_pct')}"
        ),
        (
            f"Bearish OB: "
            f"{zone_text(smc['ob_short'], 'mitigation_pct')}"
        ),
        (
            f"Bullish FVG: "
            f"{zone_text(smc['fvg_long'], 'fill_pct')}"
        ),
        (
            f"Bearish FVG: "
            f"{zone_text(smc['fvg_short'], 'fill_pct')}"
        ),
        (
            f"Dealing Range: "
            f"{fmt(dealing['low'])} - {fmt(dealing['high'])}"
        ),
        (
            f"Range Position: {dealing['position_pct']:.1f}% | "
            f"Premium: {dealing['premium']} | "
            f"Discount: {dealing['discount']}"
        ),
        "",
        "----- TRADE LEVELS (ATR MODEL) -----",
        f"ENTRY: {fmt(entry)}",
        f"STOP LOSS: {fmt(stop)}",
        f"TP1: {fmt(tp1)}",
        f"TP2: {fmt(tp2)}",
        f"TP3: {fmt(tp3)}",
        "",
        "Reasons: " + "; ".join(reasons[:12]),
        "",
        (
            "Algorithmic SMC estimates only; "
            "no guaranteed accuracy. Paper-test before live trading."
        ),
        "15
