import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load and preprocess dataset
@st.cache_data
def load_data():
    file_path = "default_of_credit_card_clients.csv"  # Replace with your dataset path
    data = pd.read_csv(file_path)

    # Set column names and clean data
    data.columns = data.iloc[0]  # Set the first row as column names
    data = data[1:]  # Remove the first row after setting column names
    data.rename(columns={"default payment next month": "default"}, inplace=True)
    numeric_columns = data.columns[1:-1]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    data["default"] = pd.to_numeric(data["default"], errors='coerce')
    data_cleaned = data.dropna()  # Drop rows with missing values
    return data_cleaned

# Train the models
@st.cache_resource
def train_models(data):
    X = data.drop(columns=["ID", "default"])
    y = data["default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Train SVM
    svm_model = SVC(random_state=42, probability=True)  # Enable probability predictions
    svm_model.fit(X_train, y_train)

    # Train Logistic Regression
    logreg_model = LogisticRegression(random_state=42)
    logreg_model.fit(X_train, y_train)

    # Train ANN
    ann_model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    ann_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    return rf_model, svm_model, logreg_model, ann_model, X.columns

# Load data and train models
data = load_data()
rf_model, svm_model, logreg_model, ann_model, feature_names = train_models(data)

# Streamlit App UI
st.title("Credit Card Default Prediction")
st.write("Enter the details below to predict the default risk.")

# User inputs
user_input = {}
for feature in feature_names:
    # For repayment status columns (categorical)
    if feature.startswith("PAY_") and not feature.startswith("PAY_AMT"):
        user_input[feature] = st.selectbox(f"{feature} (Payment Status)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=2)
    # For payment amount columns (numeric)
    elif "PAY_AMT" in feature:
        user_input[feature] = st.number_input(f"{feature} (Payment Amount)", value=0.0)
    # For other numeric features
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Prediction
if st.button("Predict"):
    # Predictions from multiple models
    rf_pred = rf_model.predict(input_df)[0]
    svm_pred = svm_model.predict(input_df)[0]
    logreg_pred = logreg_model.predict(input_df)[0]
    ann_pred = ann_model.predict(input_df).round().astype(int)[0][0]
    
    # Display individual model predictions
    st.subheader("Prediction Results:")
    st.write(f"SVM Prediction: {'Default' if svm_pred == 1 else 'No Default'}")
    st.write(f"Logistic Regression Prediction: {'Default' if logreg_pred == 1 else 'No Default'}")
    st.write(f"ANN Prediction: {'Default' if ann_pred == 1 else 'No Default'}")
    
    # Final Majority Vote Prediction
    predictions = [rf_pred, svm_pred, logreg_pred, ann_pred]
    majority_vote = np.bincount(predictions).argmax()
    final_prediction = 'Default' if majority_vote == 1 else 'No Default'
    
    # Display final majority vote prediction
    st.write(f"Final Majority Vote Prediction: {final_prediction}")