
# House Price Prediction App

This Streamlit application predicts house prices based on user input features. The prediction model is a RandomForestRegressor trained on a dataset from a house prices competition.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Streamlit Application](#streamlit-application)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/house-price-prediction-app.git
    cd house-price-prediction-app
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the same version of scikit-learn used for model training:
    ```bash
    pip install scikit-learn==1.5.0
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Input the house features (e.g., area, number of bedrooms, number of bathrooms, etc.) and click the "Predict House Price" button to get the prediction.

## Model Training

The model is trained using a dataset from the house prices competition. Here are the steps to preprocess the data, train the model, and save it:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

# Load data
train_df = pd.read_csv('/mnt/data/train.csv')

# Preprocess data
def preprocess_data(df):
    # Handle missing values for numerical features by filling with median
    num_features = df.select_dtypes(include=[np.number]).columns
    df[num_features] = df[num_features].fillna(df[num_features].median())
    
    # Handle missing values for categorical features by filling with mode
    cat_features = df.select_dtypes(include=[object]).columns
    df[cat_features] = df[cat_features].fillna(df[cat_features].mode().iloc[0])
    
    # Convert categorical variables to dummy/indicator variables
    df = pd.get_dummies(df)
    
    return df

# Preprocess the train data
train_df = preprocess_data(train_df)

# Extract features and target variable
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

# Train a model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
val_predictions = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_predictions)
print(f'Validation MAE: {val_mae}')

# Save the model and the feature names
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump((model, X.columns), f)
```

## Streamlit Application

Here is the code for the Streamlit app (`app.py`):

```python
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
```

## Contributing

If you have suggestions for improving this project, please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
