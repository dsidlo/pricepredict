# ML Price Prediction - Swing Trading System

This is an ML Based Price Prediction Swing Trading System.
- Data is pulled from Yahoo Finance.
- A model is created for each Symbol/Ticker.
- The ML model generates Price Predictions and the deltas of price predictions is applied to the prior close.

This application is a web-based UI for a stock price prediction model. Here's a high-level overview of how it works:

1. User Interface: The application uses Streamlit to create a web-based user interface. Users can input stock symbols, manage groups, and perform various operations related to stock predictions.              
2. Data Management: The application manages stock symbols and their associated data using Pandas DataFrames stored in Streamlit's session state. It can load existing data from CSV files and update it with new
  symbols or groups.
3. Price Prediction: The core functionality involves predicting stock prices using a model defined in the PricePredict class. The application checks if a model exists for a given symbol and either loads it or
 trains a new one.
4. Model Training and Prediction: The PricePredict class handles data fetching, augmentation, scaling, model training, and prediction. It uses LSTM models to predict future stock prices based on historical   
  data.                                                          
5. Session State: The application uses Streamlit's session state to manage UI elements and data, ensuring that the state is preserved across user interactions.
6. Logging: The application logs various operations and errors to a file, which helps in debugging and monitoring the application's behavior.
7. File Management: The application saves and loads data and models from specific directories, allowing for persistence across sessions.

