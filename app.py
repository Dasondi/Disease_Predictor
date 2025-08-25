# app.py (NLP Version - Final GPU/CPU Fix)

import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

# --- NEW: Automatically detect the best device (GPU or CPU) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using GPU (Apple Metal)")
else:
    device = torch.device("cpu")
    print("✅ Using CPU")
# --- END OF NEW CODE ---


# --- 1. Load All Artifacts on Startup ---
print("🚀 Loading all necessary artifacts for the NLP model...")
try:
    diagnostic_model = joblib.load('diagnostic_model.joblib')
    embedding_data = joblib.load('symptom_embeddings.joblib')
    
    with open('model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
        
    with open('release_evidences.json', 'r') as f:
        evidence_data = json.load(f)
        
    # --- FIX: Load the model and move it to the detected device ---
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    
    symptom_codes = embedding_data['codes']
    
    # --- FIX: Convert embeddings to a tensor and move it to the same device ---
    symptom_embeddings = torch.tensor(embedding_data['embeddings']).to(device)

    feature_names = model_metadata['feature_names']
    label_encoder_classes = np.array(model_metadata['label_encoder_classes'])
    
    print("✅ All artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ FATAL ERROR: Could not find a required model artifact. Please run `create_embeddings.py` first.")
    exit()

# --- 2. Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_nlp', methods=['POST'])
def predict_nlp():
    try:
        data = request.get_json()
        user_text = data.get('text', '').lower()

        if not user_text:
            return jsonify({'error': 'No text provided.'}), 400

        # The model is already on the correct device, so the output will be too
        user_embedding = nlp_model.encode(user_text, convert_to_tensor=True)
        
        # This calculation will now work as both tensors are on the same device
        cosine_scores = util.cos_sim(user_embedding, symptom_embeddings)[0]
        
        matched_symptoms = []
        # Use torch.topk for efficiency, as scores is a tensor
        top_results = torch.topk(cosine_scores, k=5).indices

        print("\n--- NLP Symptom Matching ---")
        for idx in top_results:
            idx = idx.item() # Convert tensor index to integer
            score = cosine_scores[idx].item()
            code = symptom_codes[idx]
            if score > 0.4:
                symptom_name = evidence_data.get(code, {}).get('question_en', 'Unknown Symptom')
                print(f"   - Match Found: {code} ({symptom_name}) (Similarity: {score:.2f})")
                matched_symptoms.append(code)
        print("--------------------------")

        if not matched_symptoms:
            return jsonify({'error': "Sorry, I couldn't understand the symptoms. Please try describing them differently."})

        patient_vector = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
        for symptom_code in matched_symptoms:
            if symptom_code in patient_vector.columns:
                patient_vector[symptom_code] = 1

        probabilities = diagnostic_model.predict_proba(patient_vector)[0]
        top_5_indices = np.argsort(probabilities)[-5:][::-1]
        
        predictions = []
        for i in top_5_indices:
            disease_name = label_encoder_classes[i]
            probability = probabilities[i]
            if probability > 0.01:
                predictions.append({
                    "disease": disease_name,
                    "confidence_score": round(probability * 100, 2)
                })

        return jsonify(predictions)

    except Exception as e:
        print(f"🔥 Error in /predict_nlp: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
