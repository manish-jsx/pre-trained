import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

# Load data
train_df = pd.read_csv('data/train.csv')

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
