import streamlit as st
import requests, pandas as pd
from openai import OpenAI
import plotly.express as px
import re

# Custom exception
class DataFetchError(Exception):
    pass

# ───────────────────────────────────────────────
# ❶  OpenAI client
# ───────────────────────────────────────────────
client = OpenAI(api_key=st.secrets["openai_api_key"])

# ───────────────────────────────────────────────
# ❷  RapidAPI helpers
# ───────────────────────────────────────────────
RAPID_HEADERS = {
    "X-RapidAPI-Key":  st.secrets["rapidapi_key"],
    "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com",
}

@st.cache_data(ttl=3600, show_spinner="Fetching price data…")
@st.cache_data(ttl=3600, show_spinner="Fetching price data…")
def get_stock_data(ticker, range_="1mo", interval="1d"):
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-chart"
    params = {"symbol": ticker, "range": range_, "interval": interval, "region": "US"}
    resp   = requests.get(url, params=params, headers=RAPID_HEADERS, timeout=10)

    # Quota check
    if resp.status_code == 429:
        raise DataFetchError("RapidAPI quota exceeded (HTTP 429).")

    data = resp.json()

    # Structure validation
    if (
        not data
        or "chart" not in data
        or not data["chart"].get("result")
        or data["chart"]["result"][0] is None
    ):
        raise DataFetchError(
            f"Ticker ‘{ticker}’ returned no chart data. "
            "Check the symbol spelling or try again later."
        )

    ts     = data["chart"]["result"][0]
    closes = ts["indicators"]["quote"][0]["close"]
    df     = pd.DataFrame(
        {"Close": closes},
        index=pd.to_datetime(ts["timestamp"], unit="s")
    )
    return df


@st.cache_data(ttl=43200, show_spinner="Fetching fundamentals…")
def get_stock_info(ticker):
    """
    RapidAPI ‘get-summary’ → fundamentals dict similar to yfinance .info
    """
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-summary"
    params = {"symbol": ticker, "region": "US"}
    data = requests.get(url, params=params, headers=RAPID_HEADERS, timeout=10).json()
    return data

# ───────────────────────────────────────────────
# ❸  Key‑metrics extraction (unchanged)
# ───────────────────────────────────────────────
def extract_key_metrics(info):
    return {
        "Previous Close": info.get("price", {}).get("regularMarketPreviousClose", {}).get("raw", "N/A"),
        "Open":           info.get("price", {}).get("regularMarketOpen", {}).get("raw", "N/A"),
        "Bid":            info.get("summaryDetail", {}).get("bid", {}).get("raw", "N/A"),
        "Day's Range":    info.get("summaryDetail", {}).get("dayLow", {}).get("raw", "N/A"),
        "Average Volume": info.get("summaryDetail", {}).get("averageVolume", {}).get("raw", "N/A"),
        "Market Cap":     info.get("summaryDetail", {}).get("marketCap", {}).get("fmt", "N/A"),
        "Earnings Date":  info.get("calendarEvents", {}).get("earnings", {}).get("earningsDate", [{}])[0].get("fmt", "N/A"),
        "1-Year Target Estimate": info.get("financialData", {}).get("targetMeanPrice", {}).get("raw", "N/A"),
    }

# ───────────────────────────────────────────────
# ❹  LLM helpers (unchanged)
# ───────────────────────────────────────────────
@st.cache_data(ttl=3600)
def generate_explanation(ticker, metrics):
    metric_text = "\n".join(f"- {k}: {v}" for k, v in metrics.items())
    prompt = (
        f"A user has looked up the stock {ticker}. Here are some key metrics:\n"
        f"{metric_text}\n\n"
        "Please explain each term and its value in simple terms suitable for someone new to investing."
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
    )
    return resp.choices[0].message.content

@st.cache_data(ttl=3600)
def summarize_stock_data(ticker, history):
    prompt = (
        f"Based on recent stock data for {ticker}, summarize the short-term price trend "
        "and potential risks in no more than 2-3 beginner-friendly sentences."
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
    )
    return resp.choices[0].message.content

def generate_sentiment(ticker, history):
    prompt = (
        f"A beginner investor is considering buying {ticker}. Recent price data:\n"
        f"{history.tail(5).to_string()}\n\n"
        "Give a short 2-3 sentence reaction summarizing appeal, risk, and outlook in a casual tone."
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
    )
    return resp.choices[0].message.content

@st.cache_data(ttl=3600)
def get_random_stock_fact():
    prompt = (
        "Give me one short, surprising, or educational stock market fact a beginner might not know. "
        "Make it fun and easy to remember."
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
    )
    return resp.choices[0].message.content

# ───────────────────────────────────────────────
# ❺  UI header
# ───────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#1f77b4;'>Real‑Time LLM‑Powered AI Agent for Stock‑Market Beginners</h1>",
    unsafe_allow_html=True,
)

if st.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared! Please rerun the app.")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, AMZN):")

if ticker:
    st.info(f"💡 Did you know? {get_random_stock_fact()}")

# ───────────────────────────────────────────────
# ❻  Main action
# ───────────────────────────────────────────────
if st.button("Get Insights") and ticker:
    try:
        stock_data = get_stock_data(ticker)
        info       = get_stock_info(ticker)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

    key_metrics  = extract_key_metrics(info)
    summary      = summarize_stock_data(ticker, stock_data)
    explanation  = generate_explanation(ticker, key_metrics)
    sentiment    = generate_sentiment(ticker, stock_data)

    # Prep recent-data table
    recent = stock_data.tail(5).copy()
    recent["Daily Change %"] = (recent["Close"].pct_change().fillna(0) * 100).round(2)

    # Price-trend chart
    price_trend_fig = px.line(
        stock_data, x=stock_data.index, y="Close",
        title=f"{ticker.upper()} – Closing Price Over Time",
    )

    # Bold labels in explanation
    bolded_explanation = re.sub(r"- ([^:]+):", r"- **\\1**:", explanation)

    tab1, tab2 = st.tabs(["📊 Basics", "💡 Insights"])

    # — Basics —
    with tab1:
        st.subheader("🧠 Explanation of Key Terms")
        st.markdown(bolded_explanation)

    # — Insights —
    with tab2:
        st.plotly_chart(price_trend_fig)

        st.subheader("📌 Recent Stock Data")
        st.dataframe(
            recent.style
                  .highlight_max(subset=["Daily Change %"], color="green")
                  .highlight_min(subset=["Daily Change %"], color="red")
        )

        st.subheader("📝 Summary of Recent Prices")
        st.write(summary)

        st.subheader("🤔 Should I Buy This?")
        st.info(sentiment)

        st.subheader("📊 Key Metrics")
        st.json(key_metrics)
