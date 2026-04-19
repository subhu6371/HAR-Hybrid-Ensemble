import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Input

# --- PAGE SETUP ---
st.set_page_config(page_title="HAR Dashboard", page_icon="🏃‍♂️", layout="wide")
st.title("🏃‍♂️ AI Human Activity Recognition")
st.markdown("Upload raw accelerometer data from your smartphone to predict physical activity using Ensemble Deep Learning.")

# --- DYNAMIC MODEL BUILDER ---
def build_cnn_lstm():
    """Rebuilds the exact architecture manually to bypass Keras version errors."""
    model = Sequential([
        Input(shape=(1, 17)), 
        Conv1D(filters=128, kernel_size=1, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(6, activation='softmax')
    ])
    return model

# --- 1. LOAD AI ASSETS ---
@st.cache_resource
def load_assets():
    dest = r"D:\human activity recognition"
    feature_cols = joblib.load(os.path.join(dest, "feature_columns_list.pkl"))
    scaler = joblib.load(os.path.join(dest, "har_scaler_v1.pkl"))
    encoder = joblib.load(os.path.join(dest, "har_encoder_v1.pkl"))
    xgb_model = joblib.load(os.path.join(dest, "xgb_meta_model_final.pkl"))
    
    # THE BYPASS: Build the model empty, then load only the learned weights!
    cnn_model = build_cnn_lstm()
    
    # Updated to look for the modern .keras format exported in your final run
    try:
        cnn_model.load_weights(os.path.join(dest, "cnn_lstm_meta_model_final.keras"))
    except:
        # Fallback just in case you ever use the old file
        cnn_model.load_weights(os.path.join(dest, "cnn_lstm_meta_model_final.h5"))
        
    return feature_cols, scaler, encoder, xgb_model, cnn_model

try:
    feature_cols, scaler, encoder, xgb_model, cnn_model = load_assets()
    st.sidebar.success("✅ 17-Feature Ensemble Models Loaded!")
    st.sidebar.info("Methodology: XGBoost + CNN-LSTM\n\nAccuracy: 93.37%")
except Exception as e:
    st.sidebar.error(f"❌ Error loading AI assets: {e}")
    st.stop()

# --- 2. DYNAMIC FEATURE EXTRACTION ---
def extract_windows(raw_df, window_seconds=5, sample_rate_ms=20):
    """Slices raw data into time windows and explicitly calculates ALL 17 features."""
    # Standardize column names
    raw_df.columns = [col.strip().lower() for col in raw_df.columns]
    
    # Catch 'Sensor Logger' app naming conventions
    if 'acceleration_x' in raw_df.columns:
        raw_df = raw_df.rename(columns={'acceleration_x': 'x', 'acceleration_y': 'y', 'acceleration_z': 'z'})
        
    samples_per_window = int((window_seconds * 1000) / sample_rate_ms)
    features_list = []
    
    for i in range(0, len(raw_df), samples_per_window):
        window = raw_df.iloc[i : i + samples_per_window]
        
        if len(window) < (samples_per_window // 2): 
            break
            
        # Extract individual axes for cleaner math
        x, y, z = window['x'], window['y'], window['z']
            
        # EXPLICITLY CALCULATE ALL 17 FEATURES!
        win_features = {
            'xavg': x.mean(), 'yavg': y.mean(), 'zavg': z.mean(),
            'xpeak': x.max(), 'ypeak': y.max(), 'zpeak': z.max(),
            'xabsoldev': (x - x.mean()).abs().mean(),
            'yabsoldev': (y - y.mean()).abs().mean(),
            'zabsoldev': (z - z.mean()).abs().mean(),
            'xstanddev': x.std(), 'ystanddev': y.std(), 'zstanddev': z.std(),
            'xvar': x.var(), 'yvar': y.var(), 'zvar': z.var(),
            'smv': np.mean(np.sqrt(x**2 + y**2 + z**2)),
            'smv_calculated': np.mean(np.sqrt(x**2 + y**2 + z**2))
        }
        
        # Build the final row to match the exact training columns
        final_row = []
        for col in feature_cols:
            final_row.append(win_features.get(col.lower(), 0))
            
        features_list.append(final_row)
        
    return pd.DataFrame(features_list, columns=feature_cols).fillna(0)

# --- 3. MAIN DASHBOARD ---
st.write("---")
uploaded_file = st.file_uploader("📂 Upload Accelerometer.csv", type=["csv"])

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    st.write("### 🔍 Raw Sensor Data Preview")
    st.dataframe(raw_data.head())
    
    if st.button("🚀 Run AI Analysis"):
        with st.spinner("Extracting features and analyzing movements..."):
            
            # Start timer for IEEE latency tracking
            start_time = time.time()
            
            processed_df = extract_windows(raw_data)
            
            if len(processed_df) == 0:
                st.error("Not enough data. Please upload a recording that is at least a few seconds long.")
            else:
                X_scaled = scaler.transform(processed_df.values)
                X_cnn_live = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                
                p_xgb = xgb_model.predict_proba(X_scaled)
                p_cnn = cnn_model.predict(X_cnn_live, verbose=0)
                
                final_p = (0.6 * p_xgb) + (0.4 * p_cnn)
                top_pred_idx = np.argmax(final_p, axis=1)
                
                predicted_activities = encoder.inverse_transform(top_pred_idx)
                processed_df.insert(0, 'AI_Prediction', predicted_activities)
                
                # End timer
                inference_time = time.time() - start_time
                avg_latency = (inference_time / len(processed_df)) * 1000 # convert to ms
                
                st.success("✅ Analysis Complete!")
                
                col1, col2, col3 = st.columns(3)
                dominant_activity = processed_df['AI_Prediction'].mode()[0]
                col1.info(f"🏆 **Dominant Activity:** {dominant_activity}")
                col2.info(f"⏱️ **Windows Analyzed:** {len(processed_df)}")
                col3.info(f"⚡ **Hardware Latency:** {avg_latency:.2f} ms/window")
                
                st.write("### 🎯 Complete Inference Timeline")
                
                # Dynamic column selection to avoid missing SMV_Calculated case issues
                display_cols = ['AI_Prediction'] + [c for c in processed_df.columns if 'smv' in c.lower() or 'avg' in c.lower()][:4]
                st.dataframe(processed_df[display_cols])