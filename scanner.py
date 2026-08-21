import pandas as pd
import random

# =============================
# DEMO 1-MINUTE DATA
# =============================

price = 100.0
rows = []

for i in range(500):
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

df = pd.DataFrame(rows)

# =============================
# 5-MINUTE CANDLES
# =============================

df["group"] = df.index // 5

df5 = df.groupby("group").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last"
}).reset_index(drop=True)

# =============================
# EMA
# =============================

df5["EMA20"] = df5["close"].ewm(span=20, adjust=False).mean()
df5["EMA50"] = df5["close"].ewm(span=50, adjust=False).mean()
df5["EMA200"] = df5["close"].ewm(span=200, adjust=False).mean()

# =============================
# RSI
# =============================

delta = df5["close"].diff()

gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()

rs = gain / loss.replace(0, 1e-10)

df5["RSI"] = 100 - (100 / (1 + rs))

# =============================
# MACD
# =============================

ema12 = df5["close"].ewm(span=12, adjust=False).mean()
ema26 = df5["close"].ewm(span=26, adjust=False).mean()

df5["MACD"] = ema12 - ema26
df5["MACD_SIGNAL"] = df5["MACD"].ewm(span=9, adjust=False).mean()

# =============================
# SUPPORT / RESISTANCE
# =============================

lookback = 20

support = df5["low"].rolling(lookback).min().iloc[-1]
resistance = df5["high"].rolling(lookback).max().iloc[-1]

# =============================
# MARKET STRUCTURE
# =============================

prev = df5.iloc[-2]
last = df5.iloc[-1]

if last["high"] > prev["high"] and last["low"] > prev["low"]:
    structure = "BULLISH"
elif last["high"] < prev["high"] and last["low"] < prev["low"]:
    structure = "BEARISH"
else:
    structure = "RANGE"

# =============================
# LIQUIDITY SWEEP
# =============================

recent_high = df5["high"].iloc[-11:-1].max()
recent_low = df5["low"].iloc[-11:-1].min()

bullish_sweep = (
    last["low"] < recent_low
    and last["close"] > recent_low
)

bearish_sweep = (
    last["high"] > recent_high
    and last["close"] < recent_high
)

# =============================
# FVG-STYLE GAP
# =============================

if len(df5) >= 3:

    c1 = df5.iloc[-3]
    c2 = df5.iloc[-2]
    c3 = df5.iloc[-1]

    bullish_fvg = c3["low"] > c1["high"]
    bearish_fvg = c3["high"] < c1["low"]

else:

    bullish_fvg = False
    bearish_fvg = False

# =============================
# ORDER-BLOCK STYLE CANDLE
# =============================

body = abs(last["close"] - last["open"])
range_size = last["high"] - last["low"]

strong_candle = (
    range_size > 0
    and body / range_size > 0.60
)

bullish_ob = (
    last["close"] > last["open"]
    and strong_candle
)

bearish_ob = (
    last["close"] < last["open"]
    and strong_candle
)

# =============================
# SCORING
# =============================

bullish_score = 0
bearish_score = 0

# EMA
if last["EMA20"] > last["EMA50"]:
    bullish_score += 1
else:
    bearish_score += 1

if last["EMA50"] > last["EMA200"]:
    bullish_score += 1
else:
    bearish_score += 1

# RSI
if last["RSI"] > 50:
    bullish_score += 1
elif last["RSI"] < 50:
    bearish_score += 1

# MACD
if last["MACD"] > last["MACD_SIGNAL"]:
    bullish_score += 1
else:
    bearish_score += 1

# Structure
if structure == "BULLISH":
    bullish_score += 1
elif structure == "BEARISH":
    bearish_score += 1

# Liquidity sweep
if bullish_sweep:
    bullish_score += 2

if bearish_sweep:
    bearish_score += 2

# FVG
if bullish_fvg:
    bullish_score += 1

if bearish_fvg:
    bearish_score += 1

# OB-style candle
if bullish_ob:
    bullish_score += 1

if bearish_ob:
    bearish_score += 1

# =============================
# FINAL RESULT
# =============================

if bullish_score >= 6 and bullish_score > bearish_score:
    result = "UP BIAS"

elif bearish_score >= 6 and bearish_score > bullish_score:
    result = "DOWN BIAS"

else:
    result = "NO SIGNAL"

# =============================
# OUTPUT
# =============================

print("================================")
print("ICT/MMC-STYLE DEMO ANALYZER")
print("================================")

print("Last Price:", round(last["close"], 5))

print("EMA20:", round(last["EMA20"], 5))
print("EMA50:", round(last["EMA50"], 5))
print("EMA200:", round(last["EMA200"], 5))

print("RSI:", round(last["RSI"], 2))

print("MACD:", round(last["MACD"], 5))
print("MACD Signal:", round(last["MACD_SIGNAL"], 5))

print("Support:", round(support, 5))
print("Resistance:", round(resistance, 5))

print("Market Structure:", structure)

print("Bullish Sweep:", bullish_sweep)
print("Bearish Sweep:", bearish_sweep)

print("Bullish FVG:", bullish_fvg)
print("Bearish FVG:", bearish_fvg)

print("Bullish OB-style:", bullish_ob)
print("Bearish OB-style:", bearish_ob)

print("--------------------------------")
print("Bullish Score:", bullish_score)
print("Bearish Score:", bearish_score)
print("--------------------------------")

print("RESULT:", result)

print("--------------------------------")
print("Paper/demo analysis only.")
print("No broker order is sent.")
print("================================")
