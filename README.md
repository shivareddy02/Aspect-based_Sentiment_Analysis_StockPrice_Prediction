# Stock Price Prediction using Aspect-based Sentiment Analysis

This project forecasts the stock price of Netflix and analyzes sentiment on the company by scraping Reddit reviews. It uses LSTM and sentiment analysis to aid stock investment decisions.

## Features
- Forecast future Netflix stock prices using LSTM time series model
- Scrape Reddit reviews of Netflix using PRAW
- Perform aspect-based sentiment analysis on reviews using finBERT NLP model 
- Analyze impact of sentiment on stock price fluctuations  
- Interactive visualizations of predicted stock prices and sentiment
- Streamlit web application for entering prediction time period

## Data Sources
- Netflix historical stock data from Kaggle
- Real-time Reddit reviews of Netflix using PRAW scraper

## Methodology
The web application is built using Streamlit and uses Flask in the backend to connect the front end UI to the Python code. The LSTM forecasting model and sentiment analysis pipeline are defined in Python scripts and called from within Streamlit. The web app architecture diagram explains the high level workings.

## Installation
```
pip install -r requirements.txt  
streamlit run app.py
```

## Usage
Once the app is running, the user can enter the number of days to get a stock price forecast. The predicted prices and sentiment score will be displayed in interactive visualizations.

## Team
Shruthi Senthilmani, Akash Patil, Siva Krishna Reddy Gundam
