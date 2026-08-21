import pandas as pd
import requests
from datetime import datetime, timezone

# ==========================================
# REAL MARKET PAIRS
# Yahoo Finance symbols
# ==========================================

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

# OTC আলাদা রাখা হয়েছে।
# OTC-এর জন্য এখানে real-market data ব্যবহার করা হচ্ছে না।
OTC_PAIRS = [
    "EUR/USD OTC",
    "GBP/USD OTC",
    "USD/JPY OTC",
    "AUD/USD OTC",
    "USD/CAD OTC",
    "USD/CHF OTC",
    "EUR/GBP OTC",
    "EUR/JPY OTC",
    "GBP/JPY OTC"
]


# ==========================================
# FETCH REAL 5-MINUTE MARKET DATA
# ==========================================

def get_market_data(symbol):

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    params = {
        "interval": "5m",
        "range": "5d"
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

    data = response.json()

    result = data["chart"]["result"][0]

    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "time": pd.to_datetime(
            timestamps,
            unit="s",
            utc=True
        ),
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"]
    })

    df = df.dropna()

    return df


# ==========================================
# TECHNICAL ANALYSIS
# ==========================================

def analyze(df):

    if len(df) < 200:
        return {
            "bias": "NO SIGNAL",
            "score": 0,
            "reason": "Not enough candles"
        }

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

    # Last closed candle
    last = df.iloc[-2]
    previous = df.iloc[-3]

    bullish = 0
    bearish = 0

    # EMA trend
    if last["EMA20"] > last["EMA50"]:
        bullish += 1
    else:
        bearish += 1

    if last["EMA50"] > last["EMA200"]:
        bullish += 1
    else:
        bearish += 1

    # RSI
    if last["RSI"] > 50:
        bullish += 1
    elif last["RSI"] < 50:
        bearish += 1

    # MACD
    if last["MACD"] > last["MACD_SIGNAL"]:
        bullish += 1
    else:
        bearish += 1

    # Candle direction
    if last["close"] > last["open"]:
        bullish += 1
    elif last["close"] < last["open"]:
        bearish += 1

    # Market structure
    if (
        last["high"] > previous["high"]
        and last["low"] > previous["low"]
    ):
        bullish += 1

    elif (
        last["high"] < previous["high"]
        and last["low"] < previous["low"]
    ):
        bearish += 1

    # ======================================
    # FINAL FILTER
    # ======================================

    if (
        bullish >= 5
        and bullish > bearish
    ):
        bias = "UP"

        score = bullish

    elif (
        bearish >= 5
        and bearish > bullish
    ):
        bias = "DOWN"

        score = bearish

    else:
        bias = "NO SIGNAL"

        score = max(
            bullish,
            bearish
        )

    return {
        "bias": bias,
        "score": score,
        "price": float(last["close"]),
        "rsi": float(last["RSI"]),
        "ema20": float(last["EMA20"]),
        "ema50": float(last["EMA50"]),
        "ema200": float(last["EMA200"]),
        "macd": float(last["MACD"]),
        "macd_signal": float(
            last["MACD_SIGNAL"]
        )
    }


# ==========================================
# SCAN REAL PAIRS
# ==========================================

results = []

print("========================================")
print("     REAL MARKET BEST SINGLE SCANNER")
print("========================================")
print()

for pair, symbol in REAL_PAIRS.items():

    try:

        df = get_market_data(symbol)

        analysis = analyze(df)

        if analysis["bias"] == "NO SIGNAL":

            print(
                f"{pair}: NO SIGNAL"
            )

            continue

        results.append({
            "pair": pair,
            "symbol": symbol,
            **analysis
        })

        print(
            f"{pair}: "
            f"{analysis['bias']} "
            f"| Score "
            f"{analysis['score']}/6"
        )

    except Exception as e:

        print(
            f"{pair}: DATA ERROR - {e}"
        )


# ==========================================
# BEST SINGLE
# ==========================================

print()
print("----------------------------------------")

if not results:

    print("BEST SINGLE: NO SIGNAL")
    print("No strong setup found.")

else:

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = results[0]

    print("BEST SINGLE")
    print("----------------------------------------")

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
        f"{best['score']}/6"
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

print("----------------------------------------")

print(
    "Updated:",
    datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
)

print("----------------------------------------")
print("REAL MARKET DATA / PAPER ANALYSIS")
print("No broker order is sent.")
print("========================================")
