import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset to get feature names
data = load_breast_cancer()
feature_names = data.feature_names

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

st.title("🩺 Breast Cancer Detection System")
st.markdown("ML-powered prediction using SVM model.")

st.markdown("---")
st.subheader("Enter Tumor Feature Values")

input_data = []

col1, col2, col3 = st.columns(3)

for i, feature in enumerate(feature_names):
    if i < 10:
        value = col1.number_input(feature, value=0.0)
    elif i < 20:
        value = col2.number_input(feature, value=0.0)
    else:
        value = col3.number_input(feature, value=0.0)

    input_data.append(value)

st.markdown("---")

if st.button("🔍 Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    confidence = np.max(probability) * 100

    st.subheader("Prediction Result")

    if prediction[0] == 0:
        st.error("⚠ Malignant (Cancer Detected)")
    else:
        st.success("✅ Benign (No Cancer)")

    st.write(f"Model Confidence: **{confidence:.2f}%**")

    # ROC Curve
    st.subheader("ROC Curve")
    y_scores = model.predict_proba(scaler.transform(data.data))[:,1]
    fpr, tpr, _ = roc_curve(data.target, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)