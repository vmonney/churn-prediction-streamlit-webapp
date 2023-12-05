"""Streamlit app for predicting customer churn."""

# Import dependencies
import json
import logging
import pickle
from pathlib import Path

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from utils import transform_data

st.title("Predict Churn :rocket:")
st.write("Hit Predict to determine if your customer is likely to churn!")

# Load Schema
with Path("schema.json").open() as f:
    schema = json.load(f)

# Setup column orders
column_order_in = list(schema["column_info"].keys())[:-1]
column_order_out = list(schema["transformed_columns"]["transformed_columns"])

# SIDEBAR Section
st.sidebar.info("Update these values to predict based on your customer!")
# Collect user input features
options = {}
for column, column_properties in schema["column_info"].items():
    if column == "churn":
        pass
    # Create numerical sliders
    elif (
        column_properties["dtype"] == "int64" or column_properties["dtype"] == "float64"
    ):
        min_val, max_val = column_properties["values"]
        data_type = column_properties["dtype"]

        feature_mean = (min_val + max_val) / 2
        if data_type == "int64":
            feature_mean = int(feature_mean)

        options[column] = st.sidebar.slider(column, min_val, max_val, feature_mean)
    # Create categorical dropdowns
    elif column_properties["dtype"] == "object":
        options[column] = st.sidebar.selectbox(column, column_properties["values"])

# Load in model and encoder
model_path = "../models/experiment_2/gb.pkl"
with Path(model_path).open("rb") as f:
    model = pickle.load(f)

encoder_path = "../models/experiment_2/encoder.pkl"
with Path(encoder_path).open("rb") as f:
    onehot = pickle.load(f)

# Mean evening minutes value
mean_eve_mins = 200.29

# Make a Prediction
if st.button("Predict"):
    # Convet options to dataframe
    scoring_data = pd.Series(options).to_frame().T
    scoring_data.columns = column_order_in

    # Check datatypes
    for column, column_properties in schema["column_info"].items():
        if column != "churn":
            dtype = column_properties["dtype"]
            scoring_data[column] = scoring_data[column].astype(dtype)

    # Apply transformations
    scoring_sample = transform_data(
        scoring_data,
        column_order_out,
        mean_eve_mins,
        onehot,
    )

    # Render Predictions
    prediction = model.predict(scoring_sample)
    st.write("Predicted Outcome")
    st.write(prediction)
    st.write("Client Details")
    st.write(options)

# Save historical data
try:
    historical = pd.Series(options).to_frame().T
    historical["prediction"] = prediction
    if Path("historical_data.csv").is_file():
        historical.to_csv("historical_data.csv", mode="a", header=False, index=False)
    else:
        historical.to_csv("historical_data.csv", header=True, index=False)
except Exception:
    logging.exception("An error occurred while saving historical data")

st.header("Historical Outcomes")
if Path("historical_data.csv").is_file():
    hist = pd.read_csv("historical_data.csv")
    st.dataframe(hist)
    fig, ax = plt.subplots()
    sns.countplot(x="prediction", data=hist, ax=ax).set_title("Historical Predictions")
    st.pyplot(fig)
else:
    st.write("No historical data to display")
