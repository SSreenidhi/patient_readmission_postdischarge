from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and encoders
model_path = os.path.join(os.getcwd(), 'xgboost_patient_readmission.pkl')
encoders_path = os.path.join(os.getcwd(), 'label_encoders.pkl')
scalers_path = os.path.join(os.getcwd(), 'scalers.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoders_path, 'rb') as le_file:
    label_encoders = pickle.load(le_file)

with open(scalers_path, 'rb') as scaler_file:
    scalers = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index1.html')  # Main page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        data = request.form

        # Convert input data to a NumPy array
        input_features = np.array([
            float(data['n_inpatient']), float(data['diabetes_med']), 
            float(data['n_emergency']), float(data['n_outpatient']), 
            float(data['glucose_test']), float(data['age']), 
            float(data['A1Ctest']), float(data['n_procedures']), 
            float(label_encoders['medical_specialty'].transform([data['medical_specialty']])[0]),
            float(data['time_in_hospital']), 
            float(label_encoders['primary_diagnosis'].transform([data['diag_1']])[0]), 
            float(label_encoders['sec_diagnosis'].transform([data['diag_2']])[0]), 
            float(label_encoders['additional_sec_diag'].transform([data['diag_3']])[0]), 
            float(data['n_medications']), float(data['n_lab_procedures']), 
            float(data['change'])
        ]).reshape(1, -1)

        # Apply scaling
        scaled_features = scalers['scaler'].transform(input_features)

        # Make a prediction
        prediction = model.predict(scaled_features)
        result = "Patient is likely to be readmitted." if prediction[0] == 1 else "Patient is unlikely to be readmitted."

        # Render a new HTML page with the result
        return render_template('result.html', prediction=result)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
