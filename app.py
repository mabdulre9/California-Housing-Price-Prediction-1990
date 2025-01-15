import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from joblib import load

# Define the custom function
def add_derived_features(data):
    data = data.copy()
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    return data

# Load the trained pipeline
pipeline = load("model_compressed.pkl")

# Define the function for user input
def user_input_features():
    st.sidebar.markdown(
        """
        <h2 style="color: #FF6F61;">Input Parameters</h2>
        """, unsafe_allow_html=True)
    
    longitude = st.sidebar.number_input("Longitude", value=-122.23, step=0.01)
    latitude = st.sidebar.number_input("Latitude", value=37.88, step=0.01)
    housing_median_age = st.sidebar.number_input("Housing Median Age", value=41, step=1)
    total_rooms = st.sidebar.number_input("Total Rooms", value=880, step=1)
    total_bedrooms = st.sidebar.number_input("Total Bedrooms", value=129, step=1)
    population = st.sidebar.number_input("Population", value=322, step=1)
    households = st.sidebar.number_input("Households", value=126, step=1)
    median_income = st.sidebar.number_input("Median Income", value=8.3252, step=0.01)
    ocean_proximity = st.sidebar.selectbox(
        "Ocean Proximity", 
        options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    )
    
    # Combine inputs into a DataFrame
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })
    return input_data

# Streamlit App Interface
st.markdown(
    """
    <div style="background-color: #0047AB; padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center;">California Housing Price Prediction 1990</h1>
        <h3 style="color: white; text-align: center;">by FourZero Productions</h3>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("""
This application predicts the **Median House Value** in California based on input data provided by the user.
""")

# Get user inputs
input_data = user_input_features()

# Display input data
st.markdown(
    """
    <div style="background-color: #F4F4F4; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #0047AB;">User Input Parameters</h3>
    </div>
    """, unsafe_allow_html=True
)
st.write(input_data)

# Make predictions
if st.button("Predict"):
    try:
        prediction = pipeline.predict(input_data)
        st.markdown(
            f"""
            <div style="background-color: #DFF0D8; padding: 15px; border-radius: 10px; margin-top: 20px;">
                <h3 style="color: #3C763D;">Prediction</h3>
                <p style="font-size: 18px;">The predicted Median House Value is <strong>${prediction[0]:,.2f}</strong>.</p>
            </div>
            """, unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
