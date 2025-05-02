import streamlit as st
import pandas as pd
import plotly.express as px
import re
import time
import finnhub
from openai import OpenAI

# ────────── API Clients ──────────
finnhub_client = finnhub.Client(api_key=st.secrets["finnhub_api_key"])
openai_client = OpenAI(api_key=st.secrets["openai_api_key"])

# ────────── Caching ──────────
@st.cache_data(ttl=3600)
def get_stock_candles(symbol, resolution="D", count=30):
    now = int(time.time())
    past = now - count * 86400
    try:
        candles = finnhub_client.stock_candles(symbol, resolution, past, now)
        if candles['s'] != 'ok':
            return pd.DataFrame()
        df = pd.DataFrame({
            "Date": pd.to_datetime(candles["t"], unit="s"),
            "Open": candles["o"],
            "High": candles["h"],
            "Low": candles["l"],
            "Close": candles["c"],
            "Volume": candles["v"],
        })
        df.set_index("Date", inplace=True)
        return df
    except finnhub.exceptions.FinnhubAPIException:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_company_info(symbol):
    try:
        return finnhub_client.company_profile2(symbol=symbol)
    except:
        return {}

@st.cache_data(ttl=3600)
def generate_explanation(ticker, info):
    key_points = "\n".join([f"- {k}: {v}" for k, v in info.items() if isinstance(v, (str, int, float))])
    prompt = f"Explain the following company data for {ticker} in simple terms:\n{key_points}"
    resp = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
    )
    return resp.choices[0].message.content

@st.cache_data(ttl=3600)
def summarize_price_trend(ticker, history):
    prompt = f"Summarize the recent trend for {ticker} based on this data:\n{history.tail(5).to_string()}"
    resp = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
    )
    return resp.choices[0].message.content

@st.cache_data(ttl=3600)
def get_sentiment_opinion(ticker, history):
    prompt = f"Give a casual 2-3 sentence opinion on whether {ticker} seems appealing to a beginner investor based on this data:\n{history.tail(5).to_string()}"
    resp = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
    )
    return resp.choices[0].message.content

# ────────── UI ──────────
st.markdown("<h1 style='text-align:center;'>📊 Stock Insights for Beginners (Powered by Finnhub & GPT)</h1>", unsafe_allow_html=True)

if st.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):")

if ticker and st.button("Get Insights"):
    ticker = ticker.strip().upper()

    if not re.match(r"^[A-Z]{1,5}$", ticker):
        st.error("❌ Invalid ticker format. Please enter a valid stock symbol like AAPL or TSLA.")
        st.stop()

    with st.spinner("Fetching stock data and AI insights..."):
        info = get_company_info(ticker)
        history = get_stock_candles(ticker)

    if history.empty:
        st.error("❌ Could not fetch data. The ticker may be invalid or API is limited.")
    else:
        history["Daily Change %"] = (history["Close"].pct_change().fillna(0) * 100).round(2)

        tab1, tab2 = st.tabs(["📘 Basics", "🤖 AI-Powered Insights"])

        with tab1:
            st.subheader("🏢 Company Info")
            st.json(info)

        with tab2:
            st.subheader("📈 Price Trend")
            fig = px.line(history, x=history.index, y="Close", title=f"{ticker.upper()} – Closing Prices")
            st.plotly_chart(fig)

            st.subheader("🗂️ Recent Stock Data")
            st.dataframe(history.tail(5))

            explanation = generate_explanation(ticker, info)
            st.subheader("📘 What This Means")
            st.markdown(re.sub(r"- ([^:]+):", r"- **\1**:", explanation))

            summary = summarize_price_trend(ticker, history)
            st.subheader("📝 Price Summary")
            st.write(summary)

            sentiment = get_sentiment_opinion(ticker, history)
            st.subheader("🤔 Should I Buy?")
            st.info(sentiment)
