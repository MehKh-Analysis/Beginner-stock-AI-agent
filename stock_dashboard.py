import streamlit as st
import yfinance as yf
import pandas as pd
from openai import OpenAI
import plotly.express as px
import re

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai_api_key"])

# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="1mo", interval="1d"):
    stock = yf.Ticker(ticker)
    return stock.history(period=period, interval=interval)

def extract_key_metrics(info):
    return {
        "Previous Close":          info.get("previousClose", "N/A"),
        "Open":                    info.get("open", "N/A"),
        "Bid":                     info.get("bid", "N/A"),
        "Day's Range":             f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}",
        "Average Volume":          info.get("averageVolume", "N/A"),
        "Market Cap":              info.get("marketCap", "N/A"),
        "Earnings Date":           info.get("earningsDate", "N/A"),
        "1-Year Target Estimate":  info.get("targetMeanPrice", "N/A"),
    }

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
        "Give me one short, surprising, or educational stock-market fact a beginner might not know. "
        "Make it fun and easy to remember."
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
    )
    return resp.choices[0].message.content

# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#1f77b4;'>Real-Time LLM-Powered AI Agent for Stock Market Beginners</h1>",
    unsafe_allow_html=True,
)

if st.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared! Please rerun the app.")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, AMZN):")

if ticker:
    st.info(f"💡 Did you know? {get_random_stock_fact()}")

# ──────────────────────────────────────────────────────────────
# Main action
# ──────────────────────────────────────────────────────────────
if st.button("Get Insights") and ticker:
    stock_data = get_stock_data(ticker)
    info       = yf.Ticker(ticker).info
    key_metrics = extract_key_metrics(info)

    summary     = summarize_stock_data(ticker, stock_data)
    explanation = generate_explanation(ticker, key_metrics)
    sentiment   = generate_sentiment(ticker, stock_data)

    # Recent-data table
    recent = stock_data.tail(5).copy()
    recent["Daily Change %"] = (recent["Close"].pct_change().fillna(0) * 100).round(2)

    # Price-trend line chart
    price_trend_fig = px.line(
        stock_data, x=stock_data.index, y="Close",
        title=f"{ticker.upper()} – Closing Price Over Time",
    )

    # Bold metric names in explanation
    bolded_explanation = re.sub(r"- ([^:]+):", r"- **\\1**:", explanation)

    tab1, tab2 = st.tabs(["📊 Basics", "💡 Insights"])

    # — Basics —
    with tab1:
        st.subheader("🧠 Explanation of Key Terms You Really Need in the Market")
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
