# Flight-Fare-Prediction
Flight Ticket Price Prediction project

✈️ Flight Ticket Price Prediction

A machine learning project that predicts the ticket fare of domestic flights in India based on factors like airline, cities, class, duration, stops, and days left before departure.

📌 Project Overview

Flight ticket prices fluctuate based on several factors. This project analyses a large dataset of Indian flight fares and builds ML models to accurately predict flight prices.
We built an end-to-end system with:
✔ Data preprocessing
✔ Feature engineering
✔ Model training (Linear Regression, Random Forest, CatBoost, XGBoost etc.)
✔ Model comparison & evaluation
✔ Flask backend for prediction API
✔ Frontend (HTML/CSS/JS) for user input
✔ Deployment-ready Python code

📂 Tech Stack
Frontend-
HTML
CSS
JavaScript

Backend-
Python
Flask
Flask-CORS

Machine Learning-
Pandas
NumPy
Scikit-Learn
CatBoost
RandomForest

📊 Dataset Features

Model is trained on the following important features:
source_city
destination_city
departure_time
arrival_time
stops
class
duration
days_left

Note: Dataset does not contain weekday/weekend info. Prediction is based only on the above factors.

🧠 Models Used
1️⃣ Linear Regression
Baseline model
Simple & interpretable
Helps understand feature importance

2️⃣ Random Forest Regressor
Works well on non-linear data
Reduces overfitting
Better than Linear Regression

3️⃣ CatBoost Regressor
Handles categorical features automatically
Reduces need for one-hot encoding
Fast training
High accuracy

4️⃣ XGBoost Regressor (Best Model)
Most powerful model in the project
Gradient-boosted decision-tree algorithm
Captures complex patterns very well
Fast & optimized
Highest accuracy among all models

📊 Model Comparison Results
Model	R² Score	Accuracy (%)	MAE	RMSE
Linear Regression	0.8546	85.46%	4624.99	7014.31
Random Forest	0.9695	96.95%	2137.69	3967.47
XGBoost (Best)	0.9825	98.25%	1651.59	3000.00
CatBoost	0.9755	97.55%	1887.05	3550.99




📂 Project Structure
AIML HACKATHON/
│── app.py                      # Flask backend API
│── final flight ticket fare.ipynb   # Full ML model training notebook
│── index.html                  # Frontend (HTML + JS)
│── Indian Airlines.csv         # Dataset
│── label_encoders.pkl          # Encoded label mappings
│── model_columns.pkl           # Feature columns used in model
│── Random Forest_model.pkl     


🚀 How to Run the Project
1. Clone the repository
git clone https://github.com/asthanit1205/Flight-Fare-Prediction.git
cd Flight-Fare-Prediction.git
2. Install Required Libraries
pip install flask pandas numpy scikit-learn joblib flask-cors xgboost
3. Start Backend (Flask)
python app.py
4. Open Frontend
index.html

Conclusion-
XGBoost  model gives highest accuracy 
It outperformed Linear Regression, Random Forest, and CatBoost.
The major factors affecting flight fare are:
days_left
class
duration
airline
stops

The full pipeline includes data preprocessing, ML model training, Flask backend, and a frontend for user input.
