# import pandas as pd
# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# app = Flask(__name__)

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import numpy as np


app = Flask(__name__)
CORS(app)


# ------------------ LOAD EVERYTHING ------------------

# Load original dataset (needed for filtering airlines by route)
df_raw = pd.read_csv("Indian Airlines.csv")

# Load trained model
model = joblib.load("Random Forest_model.pkl")

# Load model column order (VERY IMPORTANT)
model_columns = joblib.load("model_columns.pkl")

# Load label encoders
# (These were created in your notebook under "label_encoders")
label_encoders = joblib.load("label_encoders.pkl")

print(" Model, encoders, and columns loaded successfully!")

# ------------------ UTIL: PREPROCESS USER INPUT ------------------

def preprocess_input(row_dict):
    """
    Converts one flight (dict) into encoded model-ready numpy array.
    Uses same LabelEncoders & column order as training.
    """

    row = row_dict.copy()

    # Apply LabelEncoder to all categorical columns
    for col in label_encoders:
        if col in row:
            row[col] = label_encoders[col].transform([str(row[col])])[0]

    # Build final row in correct order
    processed = []

    for col in model_columns:
        processed.append(row[col])

    return np.array(processed)


# ------------------ API ENDPOINT ------------------

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    src = data["source_city"]
    dest = data["destination_city"]

    # ---- Step 1: Find valid airlines for this route ----
    matching = df_raw[
        (df_raw["source_city"] == src) &
        (df_raw["destination_city"] == dest)
    ]

    valid_airlines = matching["airline"].unique()

    # If no flights exist:
    if len(valid_airlines) == 0:
        return jsonify({
            "predictions": [],
            "message": "No flights available for this route."
        })

    predictions = []

    # ---- Step 2: Filter airlines if Business class is selected ----
    # Only Vistara and Air India have Business class
    class_type = data.get("class", "Economy")
    airlines_to_process = valid_airlines
    
    if class_type == "Business":
        airlines_to_process = [airline for airline in valid_airlines 
                              if "Vistara" in airline or "Air_India" in airline]
        
        # If no business class airlines found for this route
        if len(airlines_to_process) == 0:
            return jsonify({
                "predictions": [],
                "message": "Business class is only available on Vistara and Air India flights. This route doesn't have Business class options."
            })

    # ---- Step 3: Make prediction for each airline ----
    for airline in airlines_to_process:

        row = {
            "airline": airline,
            "source_city": src,
            "departure_time": data["departure_time"],
            "stops": data["stops"],
            "arrival_time": data["arrival_time"],
            "destination_city": dest,
            "class": data["class"],
            "duration": float(data["duration"]),
            "days_left": int(data["days_left"])
        }

        # Preprocess
        X = preprocess_input(row)

        # Predict
        price = model.predict([X])[0]

        predictions.append({
            "airline": airline,
            "price": round(float(price), 2)
        })

    # Sort predictions by price (cheapest first)
    predictions.sort(key=lambda x: x["price"])
    
    # Mark the cheapest flight
    if len(predictions) > 0:
        predictions[0]["is_cheapest"] = True
        cheapest_price = predictions[0]["price"]
        
        # Calculate savings for other flights
        for pred in predictions[1:]:
            savings = pred["price"] - cheapest_price
            pred["savings"] = round(savings, 2)
            pred["is_cheapest"] = False

    return jsonify({"predictions": predictions})


# ------------------ MAIN ------------------

if __name__ == "__main__":
    print(" Backend server running... http://127.0.0.1:5000/")
    app.run(debug=True)
