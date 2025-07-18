
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib

# Load the trained model
# In a real deployment, you would save the model after training and load it here.
# For this exercise, we assume the 'model' object is available in the environment
# after the previous training step. However, for a standalone app.py,
# you would need to save and load the model. Let's simulate loading a model.
model = joblib.load('model.pkl') # Uncomment and save the model in the training step

# Define the RH mapping
rh_mapping = {
    "Summer": 50,
    "Spring": 55,
    "Autumn": 60,
    "Winter": 65,
    "Rainy": 90
}

# Define the mappings for categorical features to match the training data's cat.codes
# These should be derived from the original df before the train/test split
# Assuming the order of categories is consistent with pandas .cat.codes
# You would typically save these mappings during the data preprocessing step
season_mapping = {
    "Summer": 0, # Assuming Summer was the first category alphabetically or in order of appearance
    "Spring": 1,
    "Autumn": 2,
    "Winter": 3,
    "Rainy": 4
}

packing_mapping = {
    "open to air": 0, # Assuming this was the first category
    "Closed packed": 1,
    "vacuum packed": 2
}


app = Flask(__name__)

# Add a root route to render the index.html template
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract input values
    moisture = data['Moisture']
    temperature = data['Storage Temperature in C']
    season = data['Season']
    rh = data['RH in percent']
    days_after_milling = data['Days passed after milling']
    ffa = data['Free Fatty acids in percent']
    packing = data['Packing']

    # Handle 'Moisture' if it contains '<'
    if isinstance(moisture, str) and '<' in moisture:
        try:
            moisture = float(moisture.replace('<', '').strip())
        except ValueError:
             return jsonify({'error': 'Invalid value for Moisture'}), 400
    try:
        moisture = float(moisture) # Ensure it's a float
    except ValueError:
         return jsonify({'error': 'Invalid value for Moisture'}), 400


    # Handle 'RH' if it's "not Known"
    if rh == 'not Known':
        rh = rh_mapping.get(season, None) # Use season to get RH
        if rh is None:
             return jsonify({'error': f'Unknown Season "{season}" for RH mapping'}), 400
    try:
        rh = float(rh) # Ensure it's a float
    except ValueError:
         return jsonify({'error': 'Invalid value for RH in percent'}), 400


    # Convert categorical inputs to numerical codes
    season_code = season_mapping.get(season, None) # Use .get with a default for safety
    packing_code = packing_mapping.get(packing, None) # Use .get with a default for safety

    # Check if categorical mappings were successful
    if season_code is None:
        return jsonify({'error': f'Invalid Season value: {season}'}), 400
    if packing_code is None:
         return jsonify({'error': f'Invalid Packing value: {packing}'}), 400

    # Ensure numerical inputs are indeed numbers
    try:
        temperature = float(temperature)
        days_after_milling = float(days_after_milling)
        ffa = float(ffa)
    except ValueError:
        return jsonify({'error': 'Invalid numerical input provided'}), 400


    # Create a feature vector in the correct order:
    # 'Moisture', 'Temperature', 'Season', 'RH', 'Days_after_milling', 'Packing', 'FFA'
    features = np.array([[moisture, temperature, season_code, rh, days_after_milling, packing_code, ffa]])

    # Make prediction - Ensure 'model' is available (e.g., loaded via joblib or passed in a real app)
    # If running this cell after training in the same notebook, 'model' will be in scope.
    # In a standalone app.py, you MUST uncomment joblib.load above.
    try:
        prediction = model.predict(features)
    except NameError:
         return jsonify({'error': 'Model not loaded. Ensure model.pkl exists or model is defined.'}), 500


    # Return the prediction as JSON
    return jsonify({'predicted_shelf_life_in_days': prediction[0]})

if __name__ == '__main__':
    # This is for running locally. For deployment, a production server like Gunicorn is used.
    # app.run(debug=True)
    pass # Avoid running the Flask dev server directly in this context

