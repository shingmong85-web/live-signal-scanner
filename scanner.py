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

def get_data(symbol, interval="5m", range_="5d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    params = {
        "interval": interval,
        "range": range_
    }

    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=20
    )

    r.raise_for_status()

    result = r.json()["chart"]["result"][0]

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


def indicators(df):

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

    df["ATR"] = (
        df["high"] - df["low"]
    ).rolling(14).mean()

    return df


def smc_ict_score(df):

    if len(df) < 220:
        return {
            "bias": "NO SIGNAL",
            "score": 0
        }

    df = indicators(df)

    # Last CLOSED candle
    last = df.iloc[-2]
    prev = df.iloc[-3]

    score_up = 0
    score_down = 0

    # --------------------------------------
    # 1. EMA TREND
    # --------------------------------------

    if (
        last["EMA20"] >
        last["EMA50"] >
        last["EMA200"]
    ):
        score_up += 1

    elif (
        last["EMA20"] <
        last["EMA50"] <
        last["EMA200"]
    ):
        score_down += 1

    # --------------------------------------
    # 2. RSI
    # --------------------------------------

    if last["RSI"] > 50:
        score_up += 1

    elif last["RSI"] < 50:
        score_down += 1

    # --------------------------------------
    # 3. MACD
    # --------------------------------------

    if last["MACD"] > last["MACD_SIGNAL"]:
        score_up += 1

    elif last["MACD"] < last["MACD_SIGNAL"]:
        score_down += 1

    # --------------------------------------
    # 4. BOS / CHoCH STYLE STRUCTURE
    # --------------------------------------

    if (
        last["high"] > prev["high"]
        and last["low"] >= prev["low"]
    ):
        score_up += 1

    elif (
        last["low"] < prev["low"]
        and last["high"] <= prev["high"]
    ):
        score_down += 1

    # --------------------------------------
    # 5. LIQUIDITY SWEEP STYLE CHECK
    # --------------------------------------

    recent_high = df["high"].iloc[-7:-2].max()
    recent_low = df["low"].iloc[-7:-2].min()

    if (
        last["low"] < recent_low
        and last["close"] > recent_low
    ):
        score_up += 1

    if (
        last["high"] > recent_high
        and last["close"] < recent_high
    ):
        score_down += 1

    # --------------------------------------
    # 6. FVG STYLE CHECK
    # --------------------------------------

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
        score_up += 1

    if bearish_fvg:
        score_down += 1

    # --------------------------------------
    # 7. DISPLACEMENT
    # --------------------------------------

    avg_range = (
        df["high"] -
        df["low"]
    ).rolling(20).mean().iloc[-2]

    current_range = (
        last["high"] -
        last["low"]
    )

    if current_range > avg_range * 1.3:

        if last["close"] > last["open"]:
            score_up += 1

        elif last["close"] < last["open"]:
            score_down += 1

    # --------------------------------------
    # 8. ORDER-BLOCK STYLE CANDLE
    # --------------------------------------

    if (
        prev["close"] < prev["open"]
        and last["close"] > last["open"]
        and last["close"] > prev["high"]
    ):
        score_up += 1

    elif (
        prev["close"] > prev["open"]
        and last["close"] < last["open"]
        and last["close"] < prev["low"]
    ):
        score_down += 1

    # --------------------------------------
    # 9. VOLATILITY FILTER
    # --------------------------------------

    atr = last["ATR"]

    if pd.notna(atr) and atr > 0:

        candle_range = (
            last["high"] -
            last["low"]
        )

        # Extremely large candle:
        # avoid chasing
        if candle_range > atr * 2.5:

            score_up = min(score_up, 5)
            score_down = min(score_down, 5)

    # --------------------------------------
    # FINAL DECISION
    # --------------------------------------

    if (
        score_up >= 6
        and score_up > score_down
    ):

        bias = "UP"
        score = score_up

    elif (
        score_down >= 6
        and score_down > score_up
    ):

        bias = "DOWN"
        score = score_down

    else:

        bias = "NO SIGNAL"
        score = max(
            score_up,
            score_down
        )

    return {
        "bias": bias,
        "score": score,
        "up": score_up,
        "down": score_down,
        "price": float(last["close"]),
        "rsi": float(last["RSI"]),
        "ema20": float(last["EMA20"]),
        "ema50": float(last["EMA50"]),
        "ema200": float(last["EMA200"]),
        "macd": float(last["MACD"]),
        "macd_signal": float(last["MACD_SIGNAL"])
    }


# ==========================================
# SCAN
# ==========================================

results = []

print("==========================================")
print("      SMC + ICT BEST SINGLE SCANNER")
print("==========================================")
print()

for pair, symbol in REAL_PAIRS.items():

    try:

        df = get_data(symbol)

        result = smc_ict_score(df)

        print(
            f"{pair}: "
            f"{result['bias']} "
            f"| Score "
            f"{result['score']}/8"
        )

        if result["bias"] != "NO SIGNAL":

            results.append({
                "pair": pair,
                **result
            })

    except Exception as e:

        print(
            f"{pair}: DATA ERROR - {e}"
        )


print()
print("------------------------------------------")


if not results:

    print("BEST SINGLE: NO SIGNAL")
    print("No strong SMC + ICT setup found.")

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
        "TIMEFRAME: 5M"
    )

    print(
        "BIAS:",
        best["bias"]
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
