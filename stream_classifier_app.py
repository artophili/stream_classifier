import streamlit as st
import numpy as np
import pickle
import os

# Define the correct paths to the model and scaler
model_path = os.path.join("models", "V1", "stream_predict.pkl")
scaler_path = os.path.join("models", "V1", "scaler.pkl")

# Load model and scaler
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Set page config
st.set_page_config(page_title="Student Stream Predictor", layout="centered")

st.title("📚 Student Stream Predictor")
st.write("Enter the student's marks below to get stream suggestions:")

# Input fields
subjects = [
    'English', 'Regional Language', 'Mathematics', 'Physics', 'Chemistry', 'Biology',
    'History', 'Geography', 'Civics', 'Economics', 'Computer Applications', 'Physical Education'
]

inputs = []
cols = st.columns(3)

for i, subject in enumerate(subjects):
    with cols[i % 3]:
        mark = st.number_input(f"{subject}", min_value=0, max_value=100, value=50, step=1)
        inputs.append(mark)

# Predict button
if st.button("Predict Stream"):
    sample_input = np.array(inputs).reshape(1, -1)
    
    try:
        sample_input_scaled = scaler.transform(sample_input)
        prediction = model.predict(sample_input_scaled)

        if prediction[0] == 0:
            st.success("🎨 The student may select **Arts**.")
        elif prediction[0] == 1:
            st.success("💼 The student may select **Commerce**.")
        else:
            st.success("🔬 The student may select **Science**.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")