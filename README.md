# CreditCard_Default_Predictor

Overview
The Credit Card Default Predictor uses machine learning techniques to predict the likelihood of customers defaulting on their credit card payments. The project aims to assist financial institutions in assessing credit risk and taking preventive measures to reduce financial losses.

Problem Statement
Credit card defaults pose a significant financial risk to institutions. Predicting defaults accurately allows financial companies to adjust credit policies, reduce risks, and improve customer management. This project focuses on analyzing historical customer data to create a predictive model.

Aim and Objectives
The project aims to build a robust ensemble-based machine learning system for accurate prediction of credit card defaults. Key objectives include:

-Combining machine learning algorithms like Random Forest, XGBoost, Artificial Neural Networks (ANN), Support Vector Machines (SVM), and Logistic Regression.
-Developing a majority-voting ensemble system to integrate predictions for higher accuracy.
-Providing an interactive tool for real-world applications.

Features

-Predicts default likelihood using financial and demographic data.
-Uses ensemble learning for enhanced prediction accuracy.
-Includes web-based deployment for practical usage.

Technologies Used

-Programming Language: Python
-Machine Learning Libraries: Scikit-learn, XGBoost, TensorFlow/Keras (ANN)
-Data Visualization: Matplotlib, Seaborn
-Web Deployment: Streamlit

Key Features

-Credit limit, Gender, Education, Marital Status, Age
-Repayment status (past months)
-Bill amounts and payments (past months)

Model Development

Steps:
-Data Preprocessing: Handle missing values, normalize numerical data, and encode categorical variables.
-Exploratory Data Analysis (EDA): Analyze feature importance and correlations.
-Model Training: Train individual models (Random Forest, XGBoost, ANN, SVM, Logistic Regression).
-Ensemble Method: Implement majority voting for robust predictions.

Usage

Installation:
1.Clone the repository:
git clone https://github.com/your-username/CreditCard_Default_Predictor.git

2.Navigate to the directory:
cd CreditCard_Default_Predictor

3.Install dependencies:
pip install -r requirements.txt

4.Run the Streamlit application:
streamlit run app.py


