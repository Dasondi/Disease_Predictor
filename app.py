# app.py

# 1. Import necessary libraries
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# 2. Initialize the Flask application
# This creates the web server.
app = Flask(__name__)

# 3. Load your trained model and binarizer
# The server loads these once when it starts up.
try:
    model = joblib.load("blood_model.pkl")
    mlb = joblib.load("label_binarizer.pkl")
    print("✅ Model and binarizer loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model files not found. Please run train_model.py first.")
    model = None
    mlb = None

# 4. Define the main page route
@app.route('/')
def home():
    """
    This function runs when someone visits the main URL (e.g., http://127.0.0.1:5000).
    It tells Flask to find 'index.html' in the 'templates' folder and show it.
    """
    return render_template('index.html')

# 5. Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """
    This function runs when the frontend sends data to the '/predict' URL.
    It only accepts POST requests, which is how forms send data.
    """
    if not model or not mlb:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        # Get the data sent from the browser's JavaScript
        data = request.get_json(force=True)

        # Convert the data into a pandas DataFrame in the correct order
        input_df = pd.DataFrame([data])

        # Use the model to predict probabilities
        probabilities = model.predict_proba(input_df)

        # Apply a threshold to decide which diseases are likely
        threshold = 0.3
        predicted_labels_binary = (probabilities >= threshold).astype(int)

        # Convert the binary predictions back to disease names
        predicted_conditions = mlb.inverse_transform(predicted_labels_binary)
        diseases = list(predicted_conditions[0])

        # Create the final text response
        if not diseases:
            result = "No Specific Disease Detected. General wellness is advised."
        else:
            result = ", ".join(diseases)

        # Send the result back to the browser as JSON
        return jsonify({'prediction': result})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 400

# 6. Run the application
if __name__ == '__main__':
    """
    This line makes sure the server only runs when you execute 'python app.py' directly.
    'debug=True' allows for helpful error messages in the browser while developing.
    """
    app.run(debug=True)
