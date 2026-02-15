import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
scaler      = pickle.load(open("models/scaler.pkl", "rb"))

# ── Auto-read the EXACT column names the scaler was trained on ──
# This eliminates all feature name mismatch errors
FEATURE_COLUMNS = list(scaler.feature_names_in_)

# Print at startup so you can see the exact order in your terminal
print("\n✅ Scaler expects these features in this order:")
for i, col in enumerate(FEATURE_COLUMNS):
    print(f"   {i+1:02d}. {col}")
print()


# ── Route 1: Landing page ──────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ── Route 2: Prediction page — handles BOTH GET and POST ──
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = None

    if request.method == 'POST':
        try:
            # Raw values from form
            form_data = {
                'day':         int(request.form['day']),
                'month':       int(request.form['month']),
                'year':        int(request.form['year']),
                'Temperature': float(request.form['Temperature']),
                'RH':          float(request.form['RH']),
                'Ws':          float(request.form['Ws']),
                'Rain':        float(request.form['Rain']),
                'FFMC':        float(request.form['FFMC']),
                'DMC':         float(request.form['DMC']),
                'DC':          float(request.form['DC']),
                'ISI':         float(request.form['ISI']),
                'BUI':         float(request.form['BUI']),
                'FWI':         float(request.form['FWI']),
                # Encode Classes: "fire" → 1, "not fire" → 0
                'Classes':     1 if request.form['Classes'].strip().lower() == 'fire' else 0,
                'Region':      int(request.form['Region']),
            }

            # Build DataFrame using the EXACT column order the scaler knows
            # (auto-read from scaler.feature_names_in_ at startup)
            features_df = pd.DataFrame([form_data], columns=FEATURE_COLUMNS)

            scaled     = scaler.transform(features_df)
            prediction = ridge_model.predict(scaled)[0]

            prediction_text = f"Predicted FWI: {round(float(prediction), 2)}"

        except KeyError as e:
            prediction_text = f"Missing field: {str(e)}"
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('home.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)