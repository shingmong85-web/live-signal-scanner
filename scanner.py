import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# ============================================================
# CONFIG
# ============================================================

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
}

# Website থেকে নির্বাচিত pair পড়বে।
# না থাকলে EUR/USD default হবে।
SELECTED_PAIR = os.getenv("SELECTED_PAIR", "EUR/USD")

OUTPUT_FILE = "signal.json"


# ============================================================
# MARKET DATA
# ============================================================

def get_data(symbol, interval, range_="5d"):

    url = (
        "https://query1.finance.yahoo.com/"
        f"v8/finance/chart/{symbol}"
    )

    params = {
        "interval": interval,
        "range": range_,
        "includePrePost": "false"
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    result = response.json()["chart"]["result"][0]

    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "time": pd.to_datetime(
            result["timestamp"],
            unit="s",
            utc=True
        ),
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote.get("volume")
    })

    return (
        df
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index(drop=True)
    )


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # EMA
    df["EMA20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = df["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, 1e-10)

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # MACD
    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # ATR
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - previous_close).abs()
    tr3 = (df["low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    # Bollinger Bands
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()

    df["BB_MID"] = bb_mid
    df["BB_UPPER"] = bb_mid + (2 * bb_std)
    df["BB_LOWER"] = bb_mid - (2 * bb_std)

    # Stochastic
    lowest_low = df["low"].rolling(14).min()
    highest_high = df["high"].rolling(14).max()

    denominator = (
        highest_high - lowest_low
    ).replace(0, 1e-10)

    df["STOCH_K"] = (
        (df["close"] - lowest_low)
        / denominator
    ) * 100

    df["STOCH_D"] = (
        df["STOCH_K"]
        .rolling(3)
        .mean()
    )

    # Candle range
    df["RANGE"] = (
        df["high"] - df["low"]
    )

    return df


# ============================================================
# SMC / ICT STYLE ANALYSIS
# ============================================================

def smc_ict_analysis(df):

    if len(df) < 220:
        return {
            "bias": "NO SIGNAL",
            "score": 0,
            "max_score": 12,
            "reason": "Not enough data"
        }

    df = add_indicators(df)

    # Last CLOSED candle
    last = df.iloc[-2]
    prev = df.iloc[-3]

    up = 0
    down = 0

    reasons_up = []
    reasons_down = []

    # --------------------------------------------------------
    # 1. EMA TREND
    # --------------------------------------------------------

    if (
        last["EMA20"]
        > last["EMA50"]
        > last["EMA200"]
    ):
        up += 1
        reasons_up.append("EMA trend")

    elif (
        last["EMA20"]
        < last["EMA50"]
        < last["EMA200"]
    ):
        down += 1
        reasons_down.append("EMA trend")

    # --------------------------------------------------------
    # 2. RSI
    # --------------------------------------------------------

    if last["RSI"] > 50:
        up += 1
        reasons_up.append("RSI")

    elif last["RSI"] < 50:
        down += 1
        reasons_down.append("RSI")

    # --------------------------------------------------------
    # 3. MACD
    # --------------------------------------------------------

    if last["MACD"] > last["MACD_SIGNAL"]:
        up += 1
        reasons_up.append("MACD")

    elif last["MACD"] < last["MACD_SIGNAL"]:
        down += 1
        reasons_down.append("MACD")

    # --------------------------------------------------------
    # 4. MARKET STRUCTURE / BOS STYLE
    # --------------------------------------------------------

    if (
        last["high"] > prev["high"]
        and last["low"] >= prev["low"]
    ):
        up += 1
        reasons_up.append("Structure")

    elif (
        last["low"] < prev["low"]
        and last["high"] <= prev["high"]
    ):
        down += 1
        reasons_down.append("Structure")

    # --------------------------------------------------------
    # 5. LIQUIDITY SWEEP STYLE
    # --------------------------------------------------------

    recent_high = df["high"].iloc[-8:-2].max()
    recent_low = df["low"].iloc[-8:-2].min()

    bullish_sweep = (
        last["low"] < recent_low
        and last["close"] > recent_low
    )

    bearish_sweep = (
        last["high"] > recent_high
        and last["close"] < recent_high
    )

    if bullish_sweep:
        up += 1
        reasons_up.append("Liquidity sweep")

    if bearish_sweep:
        down += 1
        reasons_down.append("Liquidity sweep")

    # --------------------------------------------------------
    # 6. FVG STYLE
    # --------------------------------------------------------

    candle_a = df.iloc[-4]
    candle_c = df.iloc[-2]

    bullish_fvg = (
        candle_a["high"] < candle_c["low"]
    )

    bearish_fvg = (
        candle_a["low"] > candle_c["high"]
    )

    if bullish_fvg:
        up += 1
        reasons_up.append("FVG")

    if bearish_fvg:
        down += 1
        reasons_down.append("FVG")

    # --------------------------------------------------------
    # 7. DISPLACEMENT
    # --------------------------------------------------------

    avg_range = (
        df["RANGE"]
        .rolling(20)
        .mean()
        .iloc[-2]
    )

    if pd.notna(avg_range):

        if last["RANGE"] > avg_range * 1.3:

            if last["close"] > last["open"]:
                up += 1
                reasons_up.append("Displacement")

            elif last["close"] < last["open"]:
                down += 1
                reasons_down.append("Displacement")

    # --------------------------------------------------------
    # 8. ORDER BLOCK STYLE
    # --------------------------------------------------------

    bullish_ob = (
        prev["close"] < prev["open"]
        and last["close"] > last["open"]
        and last["close"] > prev["high"]
    )

    bearish_ob = (
        prev["close"] > prev["open"]
        and last["close"] < last["open"]
        and last["close"] < prev["low"]
    )

    if bullish_ob:
        up += 1
        reasons_up.append("Order block")

    if bearish_ob:
        down += 1
        reasons_down.append("Order block")

    # --------------------------------------------------------
    # 9. BOLLINGER BANDS
    # --------------------------------------------------------

    if last["close"] > last["BB_MID"]:
        up += 1
        reasons_up.append("Bollinger")

    elif last["close"] < last["BB_MID"]:
        down += 1
        reasons_down.append("Bollinger")

    # --------------------------------------------------------
    # 10. STOCHASTIC
    # --------------------------------------------------------

    if last["STOCH_K"] > last["STOCH_D"]:
        up += 1
        reasons_up.append("Stochastic")

    elif last["STOCH_K"] < last["STOCH_D"]:
        down += 1
        reasons_down.append("Stochastic")

    # --------------------------------------------------------
    # 11. CANDLE MOMENTUM
    # --------------------------------------------------------

    if last["close"] > last["open"]:
        up += 1
        reasons_up.append("Candle")

    elif last["close"] < last["open"]:
        down += 1
        reasons_down.append("Candle")

    # --------------------------------------------------------
    # 12. VOLATILITY FILTER
    # --------------------------------------------------------

    volatility_warning = False

    if pd.notna(last["ATR"]):

        if last["RANGE"] > last["ATR"] * 2.5:
            volatility_warning = True

    # --------------------------------------------------------
    # FINAL BIAS
    # --------------------------------------------------------

    max_score = 12

    if volatility_warning:

        bias = "NO SIGNAL"
        reason = "High volatility"

    elif up >= 7 and up > down:

        bias = "UP"
        reason = ", ".join(reasons_up)

    elif down >= 7 and down > up:

        bias = "DOWN"
        reason = ", ".join(reasons_down)

    else:

        bias = "NO SIGNAL"
        reason = "Insufficient confirmation"

    return {
        "bias": bias,
        "score": max(up, down),
        "max_score": max_score,
        "up": up,
        "down": down,
        "reason": reason,
        "price": float(last["close"]),
        "rsi": float(last["RSI"]),
        "ema20": float(last["EMA20"]),
        "ema50": float(last["EMA50"]),
        "ema200": float(last["EMA200"]),
        "macd": float(last["MACD"]),
        "macd_signal": float(last["MACD_SIGNAL"]),
        "bb_mid": float(last["BB_MID"]),
        "bb_upper": float(last["BB_UPPER"]),
        "bb_lower": float(last["BB_LOWER"]),
        "stoch_k": float(last["STOCH_K"]),
        "stoch_d": float(last["STOCH_D"])
    }


# ============================================================
# TIMEFRAME CONFIRMATION
# ============================================================

def get_bias(df):

    result = smc_ict_analysis(df)

    return result["bias"]


# ============================================================
# SELECTED PAIR SCAN
# ============================================================

def scan_selected_pair():

    if SELECTED_PAIR not in PAIRS:

        return {
            "status": "ERROR",
            "pair": SELECTED_PAIR,
            "bias": None,
            "score": 0,
            "tf5": "NO SIGNAL",
            "tf1": "NO SIGNAL",
            "reason": "Invalid pair"
        }

    symbol = PAIRS[SELECTED_PAIR]

    print("==========================================")
    print("      REAL MARKET PAPER ANALYSIS")
    print("==========================================")
    print()
    print("SELECTED PAIR:", SELECTED_PAIR)
    print()

    try:

        # 5M
        df5 = get_data(
            symbol,
            "5m",
            "5d"
        )

        result5 = smc_ict_analysis(df5)

        # 1M
        df1 = get_data(
            symbol,
            "1m",
            "1d"
        )

        result1 = smc_ict_analysis(df1)

        bias5 = result5["bias"]
        bias1 = result1["bias"]

        print(
            "5M:",
            bias5,
            f"({result5['score']}/{result5['max_score']})"
        )

        print(
            "1M:",
            bias1,
            f"({result1['score']}/{result1['max_score']})"
        )

        # ----------------------------------------------------
        # MULTI-TIMEFRAME CONFIRMATION
        # ----------------------------------------------------

        if (
            bias5 in ["UP", "DOWN"]
            and bias5 == bias1
        ):

            final_bias = bias5

            final_score = min(
                result5["score"],
                result1["score"]
            )

            status = "PAPER ANALYSIS"

        else:

            final_bias = "NO SIGNAL"

            final_score = max(
                result5["score"],
                result1["score"]
            )

            status = "NO SIGNAL"

        return {
            "updated": datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),

            "status": status,

            "pair": SELECTED_PAIR,

            "bias": final_bias,

            "score": final_score,

            "tf5": bias5,

            "tf1": bias1,

            "price": result5.get("price"),

            "rsi": result5.get("rsi"),

            "ema20": result5.get("ema20"),

            "ema50": result5.get("ema50"),

            "ema200": result5.get("ema200"),

            "macd": result5.get("macd"),

            "macd_signal": result5.get(
                "macd_signal"
            ),

            "bb_mid": result5.get("bb_mid"),

            "bb_upper": result5.get("bb_upper"),

            "bb_lower": result5.get("bb_lower"),

            "stoch_k": result5.get("stoch_k"),

            "stoch_d": result5.get("stoch_d"),

            "reason_5m": result5.get(
                "reason"
            ),

            "reason_1m": result1.get(
                "reason"
            ),

            "analysis_only": True
        }

    except Exception as e:

        print(
            "DATA ERROR:",
            e
        )

        return {
            "updated": datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),

            "status": "DATA ERROR",

            "pair": SELECTED_PAIR,

            "bias": "NO SIGNAL",

            "score": 0,

            "tf5": "NO SIGNAL",

            "tf1": "NO SIGNAL",

            "price": None,

            "rsi": None,

            "reason": str(e),

            "analysis_only": True
        }


# ============================================================
# SAVE RESULT
# ============================================================

result = scan_selected_pair()

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=2
    )

print()
print("------------------------------------------")
print("RESULT SAVED")
print("File:", OUTPUT_FILE)
print("------------------------------------------")

print(
    "PAIR:",
    result.get("pair")
)

print(
    "5M:",
    result.get("tf5")
)

print(
    "1M:",
    result.get("tf1")
)

print(
    "FINAL:",
    result.get("bias")
)

print(
    "SCORE:",
    result.get("score")
)

print("------------------------------------------")
print("REAL MARKET DATA")
print("PAPER ANALYSIS ONLY")
print("NO BROKER ORDER")
print("==========================================")
