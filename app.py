from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import datetime

app = Flask(__name__)

def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="2y")
    return data

def calculate_indicators(data):
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data.dropna(inplace=True)
    return data

def prepare_data_for_training(data):
    data['Target'] = data['Close'].shift(-1)
    features = ['MACD', 'Signal', 'RSI', 'SMA50', 'SMA200']
    data = data.dropna()
    X = data[features]
    y = data['Target']
    return X, y

def train_xgboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_next_day(model, data):
    latest_data = data.iloc[-1:]
    features = ['MACD', 'Signal', 'RSI', 'SMA50', 'SMA200']
    X_latest = latest_data[features]
    prediction = model.predict(X_latest)
    return prediction[0]

def is_market_open():
    today = datetime.datetime.today().weekday()
    return today < 5  # Return True if today is Monday to Friday (market is open)

def plot_market_chart(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data['Close'], label="Close Price", color='blue', alpha=0.8)
    ax.set_title('Past 1 Year Market Performance', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)
    return plot_url

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    market_status = None
    market_chart_url = None
    recent_close = None  # Variable to store the most recent closing price
    
    if request.method == "POST":
        symbol = request.form["symbol"]
        data = fetch_stock_data(symbol)
        
        if not data.empty:
            # Get the most recent closing price
            recent_close = data['Close'].iloc[-1]  # Last closing price
            
            # Check if the market is open, but allow charts to be displayed anyway
            if not is_market_open():
                market_status = "Market is closed today. Please check again on a weekday."
            
            # If market is open, proceed with prediction
            data = calculate_indicators(data)
            X, y = prepare_data_for_training(data)
            model = train_xgboost_model(X, y)
            predicted_price = predict_next_day(model, data)
            
            # Construct the prediction message
            prediction = f"The predicted closing price for tomorrow is {predicted_price:.2f}"
            
            # Generate and encode the market chart
            market_chart_url = plot_market_chart(data)
        else:
            prediction = "Invalid stock symbol or no data available."

    return render_template("index.html", 
                           prediction=prediction, 
                           market_status=market_status, 
                           market_chart_url=market_chart_url,
                           recent_close=recent_close)  # Pass recent close to the template



if __name__ == "__main__":
    app.run(debug=True)
