import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Title
st.title("🎓 Student Performance Predictor")

# Load dataset
df = pd.read_csv("student_performance_dataset_1000.csv")

# Data preprocessing
features = ['Study_Hours', 'Attendance', 'Previous_Grade']
target = 'Final_Grade'

X = df[features]
y = df[target]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Sidebar for user input
st.sidebar.header("📥 Enter Student Details")

student_id = st.sidebar.text_input("Student ID")
name = st.sidebar.text_input("Name")
study_hours = st.sidebar.number_input("Study Hours", min_value=0.0, max_value=24.0, value=2.0)
attendance = st.sidebar.slider("Attendance (%)", min_value=0, max_value=100, value=75)
previous_grade = st.sidebar.number_input("Previous Grade", min_value=0.0, max_value=100.0, value=70.0)

# Create input DataFrame
input_df = pd.DataFrame({
    'Study_Hours': [study_hours],
    'Attendance': [attendance],
    'Previous_Grade': [previous_grade]
})

# Predict final grade
if st.sidebar.button("Predict Final Grade"):
    prediction = model.predict(input_df)[0]
    
    st.success(f"📊 Prediction for {name} (ID: {student_id})")
    st.write(f"**Predicted Final Grade:** {prediction:.2f} / 100")


