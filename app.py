import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model from a pickle file
with open('house_price_model.pkl', 'rb') as file:
    model, feature_names = pickle.load(file)

# Define the Streamlit app
def main():
    st.title("House Price Prediction App")
    
    # Input fields for the features
    st.header("Input House Features")
    
    # Example features, replace with the actual features used in your model
    area = st.number_input("Area (in square feet)", min_value=0)
    bedrooms = st.number_input("Number of Bedrooms", min_value=0)
    bathrooms = st.number_input("Number of Bathrooms", min_value=0)
    stories = st.number_input("Number of Stories", min_value=0)
    garage = st.number_input("Number of Garages", min_value=0)
    
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'GrLivArea': [area],  # Replace 'GrLivArea' with the actual column name from your dataset
        'BedroomAbvGr': [bedrooms],  # Replace 'BedroomAbvGr' with the actual column name from your dataset
        'FullBath': [bathrooms],  # Replace 'FullBath' with the actual column name from your dataset
        'TotRmsAbvGrd': [stories],  # Replace 'TotRmsAbvGrd' with the actual column name from your dataset
        'GarageCars': [garage]  # Replace 'GarageCars' with the actual column name from your dataset
    })
    
    # Align input data with the model's training data
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Predict house price using the pre-trained model
    if st.button("Predict House Price"):
        prediction = model.predict(input_data)
        st.success(f"The predicted house price is: ${prediction[0]:,.2f}")

# Run the app
if __name__ == '__main__':
    main()
