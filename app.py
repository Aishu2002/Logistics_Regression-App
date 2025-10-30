import streamlit as st
import pickle
import pandas as pd

# Load model and columns
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("train_columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

st.title("Logistic Regression Model Deployment")
st.write("Enter values below:")

user_input = {}
for col in train_columns:
    user_input[col] = st.number_input(f"Enter {col}:", value=0.0)

input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Class: {prediction[0]}")
