import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# =========================
# Load Dataset
# =========================

df = pd.read_csv("house price dataset.csv")

# Features and Target
X = df.drop("Price_Lakhs", axis=1)

y = df["Price_Lakhs"]

# Convert categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# =========================
# Streamlit UI
# =========================

st.title("House Price Predictor")

st.write("Enter house details below")

# User Inputs
area = st.number_input("Area in sqft", min_value=500)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

floors = st.number_input("Floors", min_value=1)

parking = st.number_input("Parking Spaces", min_value=0)

age = st.number_input("House Age", min_value=0)

school_distance = st.number_input("Nearby School Distance (km)", min_value=0.0)

furnished = st.selectbox(
    "Furnished?",
    ["Yes", "No"]
)

city = st.selectbox(
    "City",
    ["Ahmedabad", "Anand", "Rajkot", "Surat", "Vadodara"]
)

# =========================
# Prediction
# =========================

if st.button("Predict Price"):

    input_data = {
        'Area_sqft': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Floors': floors,
        'Parking': parking,
        'Age_Years': age,
        'Nearby_Schools_km': school_distance,

        'Furnished_Yes': 0,
        'City_Anand': 0,
        'City_Rajkot': 0,
        'City_Surat': 0,
        'City_Vadodara': 0
    }

    # Furnished Encoding
    if furnished == "Yes":
        input_data['Furnished_Yes'] = 1

    # City Encoding
    if city == "Anand":
        input_data['City_Anand'] = 1

    elif city == "Rajkot":
        input_data['City_Rajkot'] = 1

    elif city == "Surat":
        input_data['City_Surat'] = 1

    elif city == "Vadodara":
        input_data['City_Vadodara'] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Prediction
    predicted_price = model.predict(input_df)

    st.success(
        f"Predicted House Price: ₹ {predicted_price[0]:,.2f} Lakhs"
    )