import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the trained model
model_path = "model/random_forest_model.pkl"
model = joblib.load(model_path)

CATEGORICAL_FEATURES = {
    "model": ["kia", "nissan", "hyundai", "mercedes-benz", "toyota"],
    "motor_type": ["petrol", "gas", "petrol and gas"],
    "wheel": ["left", "right"],
    "color": ["black", "white", "silver", "blue", "gray", "other", "brown", "red", "green", "orange", "cherry", "skyblue", "clove", "beige"],
    "type": ["sedan", "SUV", "Universal", "Coupe", "hatchback"],
    "status": ["excellent", "normal", "good", "crashed", "new"]
}

# Initialize Flask app
app = Flask(__name__)
CORS(app)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        numerical_features = {
            "year": data.get("year"),
            "motor_volume": data.get("motor_volume"),
            "running_km": data.get("running_km")
        }

        # Create one-hot encoded features
        encoded_features = {}
        for feature, categories in CATEGORICAL_FEATURES.items():
            for category in categories:
                encoded_features[f"{feature}_{category}"] = 1 if data.get(feature) == category else 0

        final_features = {**numerical_features, **encoded_features}

        df = pd.DataFrame([final_features])

        feature_order = model.feature_names_in_  # Ensures correct order
        df = df.reindex(columns=feature_order, fill_value=0)

        # Make prediction
        prediction = model.predict(df)

        return jsonify({"predicted_price": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)