import streamlit as st
import yfinance as yf
import pandas as pd
from openai import OpenAI
import plotly.express as px
import re

# Initialize OpenAI client with API key (replace with your actual key as needed)

client = OpenAI(api_key=st.secrets["openai_api_key"])

@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="1mo", interval="1d"):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period, interval=interval)
    return history

def extract_key_metrics(info):
    return {
        "Previous Close": info.get("previousClose", "N/A"),
        "Open": info.get("open", "N/A"),
        "Bid": info.get("bid", "N/A"),
        "Day's Range": f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}",
        "Average Volume": info.get("averageVolume", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "Earnings Date": info.get("earningsDate", "N/A"),
        "1-Year Target Estimate": info.get("targetMeanPrice", "N/A")
    }

@st.cache_data(ttl=3600)
def generate_explanation(ticker, metrics):
    metric_text = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
    prompt = (
        f"A user has looked up the stock {ticker}. Here are some key metrics:\n"
        f"{metric_text}\n\n"
        "Please explain each term and its value in simple terms suitable for someone new to investing."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

@st.cache_data(ttl=3600)
def summarize_stock_data(ticker, history):
    prompt = (
        f"Based on recent stock data for {ticker}, summarize the short-term price trend and potential risks. "
        "Keep it short and simple, no more than 2-3 sentences, suitable for a beginner investor."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_sentiment(ticker, history):
    prompt = (
        f"A beginner investor is considering buying {ticker}. Based on this stock's recent price data:\n"
        f"{history.tail(5).to_string()}\n\n"
        "Give a short 2-3 sentence reaction summarizing the investment appeal, risk level, and outlook in a casual tone."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

@st.cache_data(ttl=3600)
def get_random_stock_fact():
    prompt = (
        "Give me one short, surprising, or educational stock market fact that a beginner investor might not know. "
        "Make it fun and easy to remember."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

# Title
st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>Real-Time LLM-Powered AI Agent for Stock Market Beginners</h1>",
    unsafe_allow_html=True
)

if st.button('🔄 Clear Cache'):
    st.cache_data.clear()
    st.success("Cache cleared! Please rerun the app.")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, AMZN):")

if ticker:
    st.info(f"💡 Did you know? {get_random_stock_fact()}")

if st.button("Get Insights"):
    if ticker:
        stock_data = get_stock_data(ticker)
        info = yf.Ticker(ticker).info
        key_metrics = extract_key_metrics(info)

        summary = summarize_stock_data(ticker, stock_data)
        explanation = generate_explanation(ticker, key_metrics)
        sentiment = generate_sentiment(ticker, stock_data)

        # Prepare recent data for display
        recent = stock_data.tail(5).copy()
        recent["Daily Change %"] = recent["Close"].pct_change().fillna(0) * 100
        recent["Daily Change %"] = recent["Daily Change %"].round(2)

        # Create interactive charts for price trend and volume
        price_trend_fig = px.line(stock_data, x=stock_data.index, y='Close', title=f"{ticker.upper()} - Closing Price Over Time")
        volume_fig = px.bar(stock_data, x=stock_data.index, y='Volume', title=f"{ticker.upper()} - Daily Trading Volume")

        # Highlight key terms in the explanation
        bolded_explanation = re.sub(r"- ([^:]+):", r"- **\1**:", explanation)

        tab1, tab2 = st.tabs(["📊 Basics", "💡 Insights"])

        with tab1:
            st.subheader("🧠 Explanation of Key Terms You Really Need in the Market")
            st.markdown(bolded_explanation)

        st.subheader("📌 Recent Stock Data")
        st.dataframe(
            recent.style.highlight_max(subset=['Daily Change %'], color='green')
                         .highlight_min(subset=['Daily Change %'], color='red')
        )

        with tab2:
            st.plotly_chart(price_trend_fig)
            st.plotly_chart(volume_fig)

            st.subheader("📝 Summary of Recent Prices")
            st.write(summary)

            st.subheader("🤔 Should I Buy This?")
            st.info(sentiment)

            st.subheader("📊 Key Metrics")
            st.json(key_metrics)
