

import warnings

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

st.title("Energy Efficiency Predictor")
st.markdown("Predict heating load for building designs.")
import os

@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    nn_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "energy_model.keras")
    )

    return scaler, nn_model


scaler, nn_model = load_models()

rc = st.sidebar.slider("Relative Compactness", 0.5, 1.0, 0.75)
sa = st.sidebar.slider("Surface Area", 500.0, 900.0, 700.0)
wa = st.sidebar.slider("Wall Area", 200.0, 400.0, 300.0)
ra = st.sidebar.slider("Roof Area", 100.0, 300.0, 200.0)
oh = st.sidebar.slider("Overall Height", 3.5, 7.0, 5.0)
ori = st.sidebar.selectbox("Orientation", [2, 3, 4, 5])
ga = st.sidebar.slider("Glazing Area", 0.0, 0.4, 0.2)
gad = st.sidebar.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5])

input_data = np.array([[rc, sa, wa, ra, oh, ori, ga, gad]])

# ----------------------------
# Preprocessing
# ----------------------------
input_scaled = scaler.transform(input_data)


# ----------------------------
# Prediction button
# ----------------------------
if st.button("Predict Energy Load"):
    prediction = nn_model.predict(input_scaled)
    st.success(f"Heating Load: {prediction[0, 0]:.2f}")
    st.success(f"Cooling Load: {prediction[0, 1]:.2f}")
    