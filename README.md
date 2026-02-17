# Pearls_AQI_Predictor
# 🌍 Karachi AQI Predictor: 72-Hour Forecasting System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20LSTM%20%7C%20ANN-orange)](https://xgboost.readthedocs.io/)
[![Database](https://img.shields.io/badge/Feature%20Store-MongoDB-green)](https://www.mongodb.com/)
[![CI/CD](https://img.shields.io/badge/Automation-GitHub%20Actions-black)](https://github.com/features/actions)

An end-to-end, serverless machine learning pipeline designed to predict the Air Quality Index (AQI) for **Karachi, Pakistan**. This project leverages real-time weather and pollutant data to provide accurate, interpretable 72-hour forecasts.

---

## 🚀 Live Dashboard
**View the real-time predictions and health advisories here:** 👉 **[CLICK HERE TO VIEW THE HOSTED APP](https://pearlsaqipredictor-ekdfurteh2cev8jhduidra.streamlit.app/)** ---

## 🛠 Project Workflow
The system operates through six specialized automated pipelines:

* **Backfill Pipeline:** Initializes the database with 6 months of historical data.
* **Feature Pipeline:** Hourly ingestion of weather/air data with automated feature engineering (rolling averages, lags, and smog indices).
* **Training Pipeline:** Daily model retraining using an ensemble of **XGBoost, ANN, and LSTM**, registering the "Champion Model" to DagsHub.
* **Inference Pipeline:** Generates a 72-hour future forecast every hour using live weather projections.
* **CI/CD Pipeline:** Fully automated workflow using **GitHub Actions** for seamless integration and deployment.
* **Web Dashboard:** An interactive **Streamlit** UI featuring real-time metrics, trend analysis, and health advisories.

