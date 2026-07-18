# Streamlit is used to create the web application.
import streamlit as st

# Pandas is used for data manipulation.
import pandas as pd

# Joblib is used to load and save machine learning models.
import joblib

# Q2
# Load trained model and preprocessing files
model = joblib.load("LR_model.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

# Q3
# Configure the Streamlit page
st.set_page_config(
    page_title="Diamond Price Predictor",
    layout="centered"
)
# page_title sets the title shown on the browser tab.
# layout="centered" displays the application in the center for a clean and user-friendly interface.s

# Q4
# Title and Description
st.title("Diamond Price Predictor")
st.write("Enter the diamond details below to predict its price.")

# Q5
# Numerical Inputs
carat = st.number_input("Carat", min_value=0.0, value=1.0)
depth = st.number_input("Depth", min_value=0.0, value=61.0)
table = st.number_input("Table", min_value=0.0, value=55.0)
x = st.number_input("Length (x)", min_value=0.0, value=5.0)
y = st.number_input("Width (y)", min_value=0.0, value=5.0)
z = st.number_input("Height (z)", min_value=0.0, value=3.0)

# Q6
# Dropdown Inputs
cut = st.selectbox(
    "Cut",
    ["Fair", "Good", "Very Good", "Premium", "Ideal"]
)

color = st.selectbox(
    "Color",
    ["D", "E", "F", "G", "H", "I", "J"]
)

clarity = st.selectbox(
    "Clarity",
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
)

# Q7
# Predict Button
predict = st.button("Predict Price")

# Q8
if predict:

    # Create DataFrame from user inputs
    input_data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    # Perform One-Hot Encoding
    input_encoded = pd.get_dummies(input_data)

    # Align columns with training data
    input_encoded = input_encoded.reindex(
        columns=encoded_columns,
        fill_value=0
    )
    # Q9

    # Identify numerical columns
    numerical_columns = [
        "carat",
        "depth",
        "table",
        "x",
        "y",
        "z"
    ]

    # Apply StandardScaler
    input_encoded[numerical_columns] = scaler.transform(
        input_encoded[numerical_columns]
    )

    # Predict the diamond price
    prediction = model.predict(input_encoded)

    # Display predicted price
   
    # Q10

    # Predict Diamond Price
    prediction = model.predict(input_encoded)

    # Display Prediction
    st.success(f"Predicted Diamond Price: ₹{prediction[0]:,.2f}")