import pandas as pd
import random

REAL_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY",
    "AUD/USD", "USD/CAD", "USD/CHF",
    "EUR/GBP", "EUR/JPY", "GBP/JPY"
]

OTC_PAIRS = [
    "EUR/USD OTC", "GBP/USD OTC", "USD/JPY OTC",
    "AUD/USD OTC", "USD/CAD OTC", "USD/CHF OTC",
    "EUR/GBP OTC", "EUR/JPY OTC", "GBP/JPY OTC"
]

ALL_PAIRS = REAL_PAIRS + OTC_PAIRS


def make_demo_data():
    price = 100.0
    rows = []

    for _ in range(600):
        change = random.uniform(-0.30, 0.30)

        o = price
        c = price + change
        h = max(o, c) + random.uniform(0, 0.10)
        l = min(o, c) - random.uniform(0, 0.10)

        rows.append({
            "open": o,
            "high": h,
            "low": l,
            "close": c
        })

        price = c

    return pd.DataFrame(rows)


def analyze(df):

    df["EMA20"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    df["EMA200"] = df["close"].ewm(
        span=200, adjust=False
    ).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    last = df.iloc[-1]

    bullish = 0
    bearish = 0

    if last["EMA20"] > last["EMA50"]:
        bullish += 1
    else:
        bearish += 1

    if last["EMA50"] > last["EMA200"]:
        bullish += 1
    else:
        bearish += 1

    if last["RSI"] > 50:
        bullish += 1
    elif last["RSI"] < 50:
        bearish += 1

    if last["MACD"] > last["MACD_SIGNAL"]:
        bullish += 1
    else:
        bearish += 1

    if bullish >= 3 and bullish > bearish:
        bias = "UP"
        score = bullish
    elif bearish >= 3 and bearish > bullish:
        bias = "DOWN"
        score = bearish
    else:
        bias = "NO SIGNAL"
        score = max(bullish, bearish)

    return bias, score


results = []

for pair in ALL_PAIRS:

    df = make_demo_data()

    bias, score = analyze(df)

    results.append({
        "pair": pair,
        "bias": bias,
        "score": score
    })


results.sort(
    key=lambda x: x["score"],
    reverse=True
)

best = results[0]

print("========================================")
print("       BEST SINGLE DEMO ANALYZER")
print("========================================")

print("Pair:", best["pair"])
print("Timeframe: 5M")
print("Bias:", best["bias"])
print("Score:", str(best["score"]) + "/4")

print("----------------------------------------")
print("All pair results:")

for item in results:
    print(
        item["pair"],
        "->",
        item["bias"],
        "| Score:",
        str(item["score"]) + "/4"
    )

print("----------------------------------------")
print("PAPER/DEMO ANALYSIS ONLY")
print("No broker order is sent.")
print("========================================")
