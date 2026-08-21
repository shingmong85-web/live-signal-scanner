import pandas as pd
import random

PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

# --------------------------------
# DEMO DATA GENERATOR
# --------------------------------

def generate_data():
    price = 100.0
    rows = []

    for _ in range(600):
        change = random.uniform(-0.30, 0.30)

        open_price = price
        close_price = price + change

        high_price = max(open_price, close_price) + random.uniform(0, 0.10)
        low_price = min(open_price, close_price) - random.uniform(0, 0.10)

        rows.append({
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price
        })

        price = close_price

    return pd.DataFrame(rows)


# --------------------------------
# ANALYZER
# --------------------------------

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
        signal = "UP BIAS"
    elif bearish >= 3 and bearish > bullish:
        signal = "DOWN BIAS"
    else:
        signal = "NO SIGNAL"

    return signal


# --------------------------------
# HISTORY TEST
# --------------------------------

history = []

for pair in PAIRS:

    df = generate_data()

    for i in range(200, len(df) - 1, 5):

        window = df.iloc[:i].copy()

        signal = analyze(window)

        current_price = df.iloc[i]["close"]
        next_price = df.iloc[i + 1]["close"]

        if signal == "UP BIAS":
            outcome = "CORRECT" if next_price > current_price else "WRONG"

        elif signal == "DOWN BIAS":
            outcome = "CORRECT" if next_price < current_price else "WRONG"

        else:
            outcome = "SKIPPED"

        history.append({
            "pair": pair,
            "signal": signal,
            "current_price": round(current_price, 5),
            "next_price": round(next_price, 5),
            "outcome": outcome
        })


# --------------------------------
# RESULTS
# --------------------------------

history_df = pd.DataFrame(history)

signals = history_df[
    history_df["outcome"] != "SKIPPED"
]

correct = len(
    signals[signals["outcome"] == "CORRECT"]
)

total = len(signals)

if total > 0:
    accuracy = (correct / total) * 100
else:
    accuracy = 0


print("================================")
print("DEMO SIGNAL HISTORY")
print("================================")

print("Pairs tested:", ", ".join(PAIRS))

print("Total signals:", total)
print("Correct:", correct)

print(
    "Accuracy:",
    round(accuracy, 2),
    "%"
)

print("--------------------------------")
print("Recent results:")
print(history_df.tail(10).to_string(index=False))

print("--------------------------------")
print("Paper/demo analysis only.")
print("No broker order is sent.")
print("================================")
