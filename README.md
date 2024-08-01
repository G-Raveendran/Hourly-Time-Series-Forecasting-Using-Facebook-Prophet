# Hourly-Time-Series-Forecasting-Using-Facebook-Prophet

## Project Overview
In this project, we utilize Facebook's Prophet package to forecast hourly energy consumption data. The dataset used is the PJM East hourly power consumption data, which spans from 2002 to 2018. Energy consumption data typically exhibits unique seasonal patterns, trends, and other characteristics, making it an interesting case for time series forecasting.

This notebook walks through the process of data preprocessing, feature engineering, and model training, culminating in predictions of future energy consumption and the evaluation of model performance.

## Installation
To get started, you'll need to install the required packages. The most critical of these is Prophet, which can be installed via pip:


!pip install prophet
Other dependencies include:

## numpy
## pandas
## seaborn
## matplotlib
## scikit-learn
Install these via pip if they are not already installed in your environment.

## Data Description
The dataset PJME_hourly.csv contains the following columns:

## Datetime: The timestamp for each observation.
## PJME_MW: The corresponding energy consumption in megawatts.
The data covers the entire PJM East region from December 2002 to December 2018. The dataset is quite extensive, containing 145,366 hourly observations.

## Notebook Outline
## 1. Data Loading and Exploration
Import the dataset and visualize the time series data to understand its structure and trends.
Use various visualizations to identify patterns, seasonality, and potential anomalies.
## 2. Feature Engineering
Create time-based features such as hour, day of the week, month, and season, which are essential for capturing the temporal structure of the data.
Visualize how energy consumption varies across these features.
## 3. Train/Test Split
Split the data into a training set (2002-2014) and a test set (2015-2018) to evaluate the model's performance on unseen data.
## 4. Modeling with Prophet
Simple Prophet Model: Train a basic Prophet model using the training dataset and make predictions on the test set.
Prophet Model with Holidays: Enhance the model by incorporating US federal holidays, which are known to influence energy consumption patterns.
## 5. Model Evaluation
Evaluate the model's performance using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
Compare the model's predictions against actual data visually and quantitatively.
## 6. Forecasting Future Energy Consumption
Use the trained model to predict future energy consumption for a specified period.
Visualize the forecasted values and analyze the results.
## Results
The model performs reasonably well in capturing the general trends and seasonal patterns in the data, with an MAPE of around 16.5%. Incorporating holidays slightly improves the model's performance, indicating the importance of special dates in energy consumption forecasting.


## Model Optimization:
Further tuning the Prophet model parameters or experimenting with different time series models (e.g., ARIMA, LSTM) could yield better results.
## Feature Engineering:
Adding more relevant features or engineering existing ones to capture additional aspects of the data might enhance the model's performance.
## Conclusion
This project demonstrates the application of Facebook's Prophet for hourly energy consumption forecasting. While the model performs well in capturing general trends, there is room for improvement in terms of accuracy, especially during specific periods or around holidays. The results highlight the importance of incorporating domain-specific features in time series forecasting.

