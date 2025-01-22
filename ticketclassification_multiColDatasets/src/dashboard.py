import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

# Load data
st.title("Model Performance Dashboard")

# Display Metrics
st.header("Metrics")
with open("reports/metrics.json", "r") as f:
    metrics = json.load(f)
st.json(metrics)

# Display Confusion Matrix
st.header("Confusion Matrix")
confusion_matrix = plt.imread("reports/confusion_matrix.png")
st.image(confusion_matrix, caption="Confusion Matrix")

# Display Accuracy/Loss Curve
st.header("Training Curves")
accuracy_loss_curve = plt.imread("reports/accuracy_loss_curve.png")
st.image(accuracy_loss_curve, caption="Accuracy vs. Loss")

# Show Predictions
st.header("Predictions")
predictions = pd.read_csv("reports/predictions.csv")
st.dataframe(predictions)
