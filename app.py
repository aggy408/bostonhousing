import pickle
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return 'Welcome to the Prediction API'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('data')
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Prepare data for prediction
    input_data = np.array(list(data.values())).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
    
