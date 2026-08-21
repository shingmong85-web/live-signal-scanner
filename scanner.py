import pandas as pd
import requests
from datetime import datetime, timezone

REAL_PAIRS = {
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
    "ETH/USD": "ETH-USD"
}


def get_data(symbol, interval, range_="5d"):

    url = (
        "https://query1.finance.yahoo.com/"
        f"v8/finance/chart/{symbol}"
    )

    params = {
        "interval": interval,
        "range": range_
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
        "close": quote["close"]
    })

    return df.dropna().reset_index(drop=True)


def add_indicators(df):

    df = df.copy()

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

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    ).rolling(14).mean()

    loss = (-delta.clip(
        upper=0
    )).rolling(14).mean()

    rs = gain / loss.replace(
        0,
        1e-10
    )

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

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

    df["RANGE"] = (
        df["high"] -
        df["low"]
    )

    df["ATR"] = df["RANGE"].rolling(
        14
    ).mean()

    return df


def timeframe_bias(df):

    if len(df) < 220:
        return "NO SIGNAL"

    df = add_indicators(df)

    last = df.iloc[-2]

    bullish = 0
    bearish = 0

    # EMA trend
    if (
        last["EMA20"] >
        last["EMA50"] >
        last["EMA200"]
    ):
        bullish += 1

    elif (
        last["EMA20"] <
        last["EMA50"] <
        last["EMA200"]
    ):
        bearish += 1

    # RSI
    if last["RSI"] > 50:
        bullish += 1

    elif last["RSI"] < 50:
        bearish += 1

    # MACD
    if last["MACD"] > last["MACD_SIGNAL"]:
        bullish += 1

    elif last["MACD"] < last["MACD_SIGNAL"]:
        bearish += 1

    if bullish >= 2 and bullish > bearish:
        return "UP"

    if bearish >= 2 and bearish > bullish:
        return "DOWN"

    return "NO SIGNAL"


def smc_ict_score(df):

    if len(df) < 220:
        return {
            "bias": "NO SIGNAL",
            "score": 0
        }

    df = add_indicators(df)

    last = df.iloc[-2]
    prev = df.iloc[-3]

    up = 0
    down = 0

    # =====================================
    # 1 EMA TREND
    # =====================================

    if (
        last["EMA20"] >
        last["EMA50"] >
        last["EMA200"]
    ):
        up += 1

    elif (
        last["EMA20"] <
        last["EMA50"] <
        last["EMA200"]
    ):
        down += 1

    # =====================================
    # 2 RSI
    # =====================================

    if last["RSI"] > 50:
        up += 1

    elif last["RSI"] < 50:
        down += 1

    # =====================================
    # 3 MACD
    # =====================================

    if last["MACD"] > last["MACD_SIGNAL"]:
        up += 1

    elif last["MACD"] < last["MACD_SIGNAL"]:
        down += 1

    # =====================================
    # 4 MARKET STRUCTURE
    # =====================================

    if (
        last["high"] > prev["high"]
        and last["low"] >= prev["low"]
    ):
        up += 1

    elif (
        last["low"] < prev["low"]
        and last["high"] <= prev["high"]
    ):
        down += 1

    # =====================================
    # 5 LIQUIDITY SWEEP
    # =====================================

    recent_high = df["high"].iloc[-7:-2].max()
    recent_low = df["low"].iloc[-7:-2].min()

    if (
        last["low"] < recent_low
        and last["close"] > recent_low
    ):
        up += 1

    if (
        last["high"] > recent_high
        and last["close"] < recent_high
    ):
        down += 1

    # =====================================
    # 6 FVG STYLE CHECK
    # =====================================

    candle_a = df.iloc[-4]
    candle_c = df.iloc[-2]

    bullish_fvg = (
        candle_a["high"] <
        candle_c["low"]
    )

    bearish_fvg = (
        candle_a["low"] >
        candle_c["high"]
    )

    if bullish_fvg:
        up += 1

    if bearish_fvg:
        down += 1

    # =====================================
    # 7 DISPLACEMENT
    # =====================================

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

            elif last["close"] < last["open"]:
                down += 1

    # =====================================
    # 8 ORDER-BLOCK STYLE
    # =====================================

    if (
        prev["close"] < prev["open"]
        and last["close"] > last["open"]
        and last["close"] > prev["high"]
    ):
        up += 1

    elif (
        prev["close"] > prev["open"]
        and last["close"] < last["open"]
        and last["close"] < prev["low"]
    ):
        down += 1

    # =====================================
    # VOLATILITY FILTER
    # =====================================

    if pd.notna(last["ATR"]):

        if last["RANGE"] > last["ATR"] * 2.5:

            up = min(up, 5)
            down = min(down, 5)

    # =====================================
    # FINAL 5M RESULT
    # =====================================

    if up >= 6 and up > down:

        bias = "UP"
        score = up

    elif down >= 6 and down > up:

        bias = "DOWN"
        score = down

    else:

        bias = "NO SIGNAL"
        score = max(up, down)

    return {
        "bias": bias,
        "score": score,
        "up": up,
        "down": down,
        "price": float(last["close"]),
        "rsi": float(last["RSI"]),
        "ema20": float(last["EMA20"]),
        "ema50": float(last["EMA50"]),
        "ema200": float(last["EMA200"]),
        "macd": float(last["MACD"]),
        "macd_signal": float(last["MACD_SIGNAL"])
    }


# ==========================================
# MAIN SCANNER
# ==========================================

results = []

print("==========================================")
print("     SMC + ICT 1M / 5M BEST SINGLE")
print("==========================================")
print()

for pair, symbol in REAL_PAIRS.items():

    try:

        # 5-minute analysis
        df5 = get_data(
            symbol,
            "5m"
        )

        result5 = smc_ict_score(df5)

        # 1-minute confirmation
        df1 = get_data(
            symbol,
            "1m",
            "1d"
        )

        bias1 = timeframe_bias(df1)

        print(
            f"{pair}: "
            f"5M={result5['bias']} "
            f"({result5['score']}/8) "
            f"| 1M={bias1}"
        )

        # ----------------------------------
        # MTF CONFIRMATION
        # ----------------------------------

        confirmed = False

        if (
            result5["bias"] == "UP"
            and bias1 == "UP"
        ):
            confirmed = True

        elif (
            result5["bias"] == "DOWN"
            and bias1 == "DOWN"
        ):
            confirmed = True

        if confirmed:

            results.append({
                "pair": pair,
                **result5,
                "1m_bias": bias1
            })

    except Exception as e:

        print(
            f"{pair}: DATA ERROR - {e}"
        )


print()
print("------------------------------------------")


if not results:

    print("BEST SINGLE: NO SIGNAL")
    print("No confirmed 1M + 5M setup.")

else:

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = results[0]

    print("BEST SINGLE")
    print("------------------------------------------")

    print(
        "PAIR:",
        best["pair"]
    )

    print(
        "5M BIAS:",
        best["bias"]
    )

    print(
        "1M CONFIRMATION:",
        best["1m_bias"]
    )

    print(
        "SCORE:",
        f"{best['score']}/8"
    )

    print(
        "PRICE:",
        round(best["price"], 5)
    )

    print(
        "RSI:",
        round(best["rsi"], 2)
    )

    print(
        "EMA20:",
        round(best["ema20"], 5)
    )

    print(
        "EMA50:",
        round(best["ema50"], 5)
    )

    print(
        "EMA200:",
        round(best["ema200"], 5)
    )

    print(
        "MACD:",
        round(best["macd"], 6)
    )

    print(
        "MACD SIGNAL:",
        round(best["macd_signal"], 6)
    )


print("------------------------------------------")

print(
    "Updated:",
    datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
)

print("------------------------------------------")
print("REAL MARKET / PAPER ANALYSIS")
print("No broker order is sent.")
print("==========================================")
