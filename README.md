# Stock-Market-Analyzer
Stock Market Analyzer using XGBoost
# Stock Market Prediction App

A Flask web application that predicts stock market prices using machine learning (XGBoost) and technical indicators.

## Features

- Real-time stock data fetching using yfinance
- Technical indicators calculation (MACD, RSI, EMAs, SMAs)
- Machine learning prediction using XGBoost
- Interactive web interface
- Visual stock price charts
- Market status checking

## Technical Indicators

The application calculates several technical indicators:
- EMA (12 and 26 periods)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- SMA (50 and 200 periods)

## Prerequisites

- Python 3.7+
- Flask
- yfinance
- pandas
- matplotlib
- XGBoost
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install required packages:
```bash
pip install flask yfinance pandas matplotlib xgboost scikit-learn
```

3. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Open the application in your web browser
2. Enter a valid stock symbol (e.g., AAPL, GOOGL)
3. Submit the form to get:
   - Current stock price
   - Predicted price for the next day
   - Market status
   - Historical price chart

## Model Details

The prediction model uses XGBoost with the following features:
- MACD
- Signal Line
- RSI
- 50-day SMA
- 200-day SMA

Model parameters:
- n_estimators: 100
- max_depth: 5
- learning_rate: 0.1

## Market Hours

The application checks if the market is currently open (Monday to Friday) and displays appropriate messages.

## Disclaimer

This application is for educational purposes only. Stock predictions are based on historical data and technical analysis, and should not be used as the sole basis for investment decisions.

## License

[MIT License](LICENSE)
