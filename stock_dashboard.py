# ─────────────────────────────────────────────────────────────
# Streamlit Stock Helper — RapidAPI yahoo-finance15 edition
# ─────────────────────────────────────────────────────────────
import streamlit as st
import requests, pandas as pd
from openai import OpenAI
import plotly.express as px
import re

# ─── 0.  Secrets you need  ───────────────────────────────────
# st.secrets["rapidapi_key"]  – your RapidAPI key
# st.secrets["openai_api_key"] – your OpenAI key

# ─── 1.  RapidAPI header  ────────────────────────────────────
RAPID_HEADERS = {
    "X-RapidAPI-Key":  st.secrets["rapidapi_key"],
    "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com",
}

# ─── 2.  Custom error to simplify UI messages  ───────────────
class DataFetchError(Exception):
    """Raised when RapidAPI returns no usable data or quota is exceeded."""
    pass

# ─── 3.  Price history helper (historical)  ──────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching price data…")
def get_stock_data(ticker, period="1mo"):
    """
    Calls yahoo-finance15 /api/v2/historical
    Returns a DataFrame with Date index and 'Close' column.
    """
    url    = f"https://yahoo-finance15.p.rapidapi.com/api/v2/historical/{ticker}"
    params = {"period": period}     # 1d, 5d, 1mo, 3mo, 6mo, 1y, etc.
    resp   = requests.get(url, params=params, headers=RAPID_HEADERS, timeout=10)

    if resp.status_code == 429:
        raise DataFetchError("RapidAPI quota exceeded (HTTP 429).")

    data = resp.json()
    if not data or "items" not in data or not data["items"]:
        raise DataFetchError(f"No price data returned for “{ticker}”. Check symbol or period.")

    # Build DataFrame
    items = data["items"]
    df = (
        pd.DataFrame(items)
          .assign(Date=lambda d: pd.to_datetime(d["date"], unit="s"))
          .set_index("Date")
          [["close"]]
          .rename(columns={"close": "Close"})
          .sort_index()
    )
    return df

# ─── 4.  Fundamentals helper  ─────────────────────────────────
@st.cache_data(ttl=43200, show_spinner="Fetching fundamentals…")
def get_stock_info(ticker):
    url  = f"https://yahoo-finance15.p.rapidapi.com/api/v2/quote/{ticker}"
    resp = requests.get(url, headers=RAPID_HEADERS, timeout=10)

    if resp.status_code == 429:
        raise DataFetchError("RapidAPI quota exceeded (HTTP 429).")

    data = resp.json()
    if "price" not in data:
        raise DataFetchError(f"No fundamentals returned for “{ticker}”.")
    return data

# ─── 5.  Metric extraction (map new JSON structure)  ─────────
def extract_key_metrics(info):
    price   = info.get("price", {})
    summary = info.get("summaryDetail", {})
    calendar= info.get("calendarEvents", {}).get("earnings", {})
    target  = info.get("financialData", {}).get("targetMeanPrice", {})

    day_low  = summary.get("dayLow",  {}).get("raw", "N/A")
    day_high = summary.get("dayHigh", {}).get("raw", "N/A")

    return {
        "Previous Close": price.get("regularMarketPreviousClose", {}).get("raw", "N/A"),
        "Open":           price.get("regularMarketOpen", {}).get("raw", "N/A"),
        "Bid":            summary.get("bid", {}).get("raw", "N/A"),
        "Day's Range":    f"{day_low} – {day_high}",
        "Average Volume": summary.get("averageVolume", {}).get("raw", "N/A"),
        "Market Cap":     summary.get("marketCap", {}).get("fmt", "N/A"),
        "Earnings Date":  calendar[0].get("fmt", "N/A") if calendar else "N/A",
        "1‑Year Target Estimate": target.get("raw", "N/A"),
    }

# ─── 6.  OpenAI helpers (unchanged)  ─────────────────────────
client = OpenAI(api_key=st.secrets["openai_api_key"])

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
        "and potential risks in no more than 2–3 beginner-friendly sentences."
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
        "Give a short 2‑3 sentence reaction summarizing appeal, risk, and outlook in a casual tone."
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
    )
    return resp.choices[0].message.content

@st.cache_data(ttl=3600)
def get_random_stock_fact():
    prompt = (
        "Give me one short, surprising, or educational stock‑market fact a beginner might not know. "
        "Make it fun and easy to remember."
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
    )
    return resp.choices[0].message.content

# ─── 7.  UI header  ──────────────────────────────────────────
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

# ─── 8.  Main action  ───────────────────────────────────────
if st.button("Get Insights") and ticker:
    try:
        stock_data = get_stock_data(ticker)
        info       = get_stock_info(ticker)
    except DataFetchError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

    key_metrics  = extract_key_metrics(info)
    summary      = summarize_stock_data(ticker, stock_data)
    explanation  = generate_explanation(ticker, key_metrics)
    sentiment    = generate_sentiment(ticker, stock_data)

    # Recent price table
    recent = stock_data.tail(5).copy()
    recent["Daily Change %"] = (recent["Close"].pct_change().fillna(0) * 100).round(2)

    # Price trend chart
    price_trend_fig = px.line(
        stock_data, x=stock_data.index, y="Close",
        title=f"{ticker.upper()} – Closing Price Over Time",
    )

    # Bold metric names
    bolded_explanation = re.sub(r"- ([^:]+):", r"- **\\1**:", explanation)

    tab1, tab2 = st.tabs(["📊 Basics", "💡 Insights"])

    # Basics tab
    with tab1:
        st.subheader("🧠 Explanation of Key Terms")
        st.markdown(bolded_explanation)

    # Insights tab
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
