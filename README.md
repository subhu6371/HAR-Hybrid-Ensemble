# HAR-Hybrid-Ensemble
A Machine Learning project for recognisition human activity by mobile sensor data
# 🏃‍♂️ EdgeHAR: Hybrid XGBoost & CNN-LSTM Activity Recognition

An edge-optimized Human Activity Recognition (HAR) system designed to classify physical movements using raw smartphone accelerometer data. This project achieves **93.37% accuracy** with an ultra-low inference latency of **0.0059 ms**, making it ideal for continuous background deployment on mobile devices.

## ✨ Key Features
* **Hybrid Ensemble Architecture:** Fuses XGBoost (60% weight) for tabular statistical thresholds with a 1D CNN-LSTM (40% weight) for spatiotemporal sequence mapping.
* **Orientation-Invariant:** Extracts 17 mathematical features per time-window, including the Signal Magnitude Vector (SMV), negating smartphone orientation bias.
* **Edge-Ready Dashboard:** A lightweight Streamlit UI that accepts raw `.csv` or `.zip` sensor exports and performs real-time feature extraction and prediction.

## 🛠️ Tech Stack
* **Machine Learning:** TensorFlow/Keras (CNN-LSTM), XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Deployment:** Streamlit Web Framework
* **Dataset:** WISDM (Wireless Sensor Data Mining)

## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/subhu6371/HAR-Hybrid-Ensemble.git](https://github.com/subhu6371/HAR-Hybrid-Ensemble.git)
cd HAR-Hybrid-Ensemble
