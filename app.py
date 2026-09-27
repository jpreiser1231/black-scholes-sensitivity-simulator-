from flask import Flask, jsonify, render_template, request
import yfinance as yf
import numpy as np
import pandas_datareader.data as web
import datetime

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/quote")
def quote():
    """Live spot price + trailing 1-year historical volatility for a ticker."""
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
            return jsonify({"error": f"No data found for '{ticker}'"}), 404

        price = float(hist["Close"].iloc[-1])
        log_returns = np.log(hist["Close"] / hist["Close"].shift(1))
        volatility = float(log_returns.std() * np.sqrt(252))

        return jsonify({
            "ticker": ticker,
            "price": round(price, 2),
            "volatility": round(volatility, 4),
            "as_of": hist.index[-1].strftime("%Y-%m-%d"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rate")
def rate():
    """Current 3-month Treasury yield from FRED, used as the risk-free rate."""
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=30)
        data = web.DataReader("DGS3MO", "fred", start).dropna()
        r = float(data.iloc[-1].values[0]) / 100
        return jsonify({"rate": round(r, 4), "as_of": data.index[-1].strftime("%Y-%m-%d")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
