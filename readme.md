# 📈 LLM-Powered Stock Insights for Beginners

*Simplifying the Stock Market for Beginners*

---

## 🌟 Overview

A Streamlit web app that turns raw stock market data into plain-language explanations for new investors. Enter a ticker, and the app pulls live market data, charts it, and uses GPT-4 to explain the key metrics and summarize recent price movements in simple terms.

> **⚠️ Note:** Educational project only. Not financial advice. AI-generated summaries can be wrong and should not be used to make investment decisions.

---

## ✨ Features

| Feature | What it does |
| :--- | :--- |
| **Live market data** | Pulls the past month of daily prices, volume, and key metrics from Yahoo Finance |
| **Metrics explained simply** | GPT-4 explains terms like market cap, bid, and 1-year target estimate, using the stock's actual values |
| **Interactive charts** | Closing price and daily trading volume with Plotly |
| **Recent price table** | Last 5 trading days with daily % change, best and worst days highlighted |
| **Price trend summary** | A 2–3 sentence summary of the short-term trend and risks |
| **Quick take** | A casual, beginner-friendly reaction to the stock's recent performance |
| **Stock market fact** | A random educational fact each time you look up a ticker |

---

## 🛠️ Tech stack

- **App:** Streamlit
- **Data:** yfinance (Yahoo Finance)
- **LLM:** OpenAI GPT-4
- **Charts:** Plotly
- **Deployment:** Docker, dev container

---

## 🚀 How to run

### 1. Add your OpenAI API key
Create `.streamlit/secrets.toml`:

```toml
openai_api_key = "your-key-here"
```

### 2a. Run locally

```bash
pip install -r requirements.txt
streamlit run stock_dashboard.py
```

### 2b. Or run with Docker

```bash
docker build -t stock-insights .
docker run -p 8501:8501 stock-insights
```

Then open `http://localhost:8501`.

---

## ⚠️ Limitations

- **Not financial advice.** The LLM's commentary is generated from a small window of recent data and can be inaccurate.
- **Short data window.** Summaries use only recent daily prices, with no news, fundamentals analysis, or longer history.
- **Data availability.** `yfinance` is unofficial, so some metrics may be missing (shown as N/A) for certain tickers.