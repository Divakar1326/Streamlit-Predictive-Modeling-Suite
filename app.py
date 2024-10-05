import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,confusion_matrix, classification_report
from datetime import date
st.sidebar.title("📈 Stock Price Prediction App 💹")
stocks = ["AAPL", "GOOG", "MSFT", "AMZN", "📋 Enter Custom Stock"]
selected_stock = st.sidebar.selectbox("S💼 Choose one of the popular stocks below:", stocks)
if selected_stock == "📋 Enter Custom Stock":
    stock_ticker = st.sidebar.text_input("Enter stock here 👇 (e.g., TSLA):")
    st.image("C:/Users/diva1/OneDrive/Documents/yahoo.png", use_column_width=True)
else:
    stock_ticker = selected_stock
if stock_ticker:
    stock_data = yf.Ticker(stock_ticker)
    df = stock_data.history(period="10y")
    st.title(f"{stock_ticker}📊 Stock Price Prediction")
    st.subheader(f"ℹ️ General Information for {stock_ticker}")
    st.subheader("📄 Stock Summary")
    summary = stock_data.info['longBusinessSummary'][:300] + "..." if len(stock_data.info['longBusinessSummary']) > 300 else stock_data.info['longBusinessSummary']
    st.write(summary)
    st.subheader(f"📅 Stock Price Data for {stock_ticker} - Last 10 Year")
    st.line_chart(df['Close'])
    df['Date'] = df.index
    df['Day'] = np.arange(len(df))
    X = df[['Day']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    st.subheader("📈Prediction Report")
    st.write(f"**R² Score**: {r2:.2f}")
    st.write(f"**Mean Absolute Error (MAE)**: {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
    st.subheader("🧮 Confusion Matrix & Report (Example)")
    y_test_labels = np.where(y_test > np.median(y_test), 1, 0)
    y_pred_labels = np.where(y_pred > np.median(y_pred), 1, 0)
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    clf_report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    st.subheader("📑 Classification Report")
    st.write(pd.DataFrame(clf_report).transpose())
    st.subheader("🔥Correlation Heatmap")
    corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.subheader("🔮Make Predictions for New Data")
    future_days = st.number_input("Enter number of future days to predict:", min_value=1, max_value=365)
    if st.button("Predict"):
        future = np.array(range(len(df), len(df) + future_days)).reshape(-1, 1)
        future_pred = model.predict(future)
        st.write(f"Predicted stock prices for next {future_days} days:")
        st.line_chart(future_pred)
st.sidebar.subheader("ℹ️ App Information")
st.sidebar.write("🔍 This app predicts stock prices using live Yahoo Finance data.")
st.sidebar.write("💡 Try selecting popular stocks or entering your own stock ticker!")
