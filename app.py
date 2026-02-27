import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

st.title("🩺 Breast Cancer Detection System")
st.markdown("Compare multiple ML models for tumor classification.")

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
models = {
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=5000)
}

for model in models.values():
    model.fit(X_scaled, y)

# Sidebar input
st.sidebar.header("Enter Tumor Feature Values")

input_data = []
for feature in feature_names:
    value = st.sidebar.number_input(feature, value=0.0)
    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Model selection
selected_model_name = st.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

if st.button("🔍 Predict"):
    prediction = selected_model.predict(input_scaled)
    probability = selected_model.predict_proba(input_scaled)
    confidence = np.max(probability) * 100

    if prediction[0] == 0:
        st.error("⚠ Malignant (Cancer Detected)")
    else:
        st.success("✅ Benign (No Cancer)")

    st.write(f"Model Confidence: **{confidence:.2f}%**")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    y_pred = selected_model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # ROC Curve
    st.subheader("ROC Curve")
    y_scores = selected_model.predict_proba(X_scaled)[:,1]
    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# Dataset download
st.subheader("Download Dataset")
import pandas as pd
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False),
    file_name="breast_cancer_dataset.csv",
    mime="text/csv"
)