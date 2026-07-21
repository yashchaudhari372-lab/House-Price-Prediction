import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model using pickle
try:
    with open('linear_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Embedded HTML Template with Modern CSS & Animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6C5CE7;
            --secondary: #a29bfe;
            --background: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text: #f8fafc;
            --accent: #00b894;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: radial-gradient(circle at top left, #1e1b4b, #0f172a);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.8s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            text-align: center;
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(45deg, #a29bfe, #74b9ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }

        form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group label {
            font-size: 0.85rem;
            margin-bottom: 0.4rem;
            color: #cbd5e1;
            font-weight: 500;
        }

        .input-group input {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 0.75rem 1rem;
            border-radius: 10px;
            color: #fff;
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }

        .input-group input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 10px rgba(108, 92, 231, 0.4);
            transform: translateY(-2px);
        }

        .btn-container {
            grid-column: 1 / -1;
            margin-top: 1rem;
            text-align: center;
        }

        button {
            width: 100%;
            max-width: 300px;
            padding: 0.9rem;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary), #0984e3);
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(108, 92, 231, 0.4);
        }

        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(108, 92, 231, 0.6);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(0, 184, 148, 0.15);
            border: 1px solid var(--accent);
            border-radius: 12px;
            text-align: center;
            animation: pulse 1s ease-in-out;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .result-box h2 {
            color: var(--accent);
            font-size: 1.8rem;
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>🏡 House Price Predictor</h1>
        <p class="subtitle">Enter the details below to estimate the valuation of the property</p>

        <form action="/predict" method="POST">
            <div class="input-group">
                <label>Number of Bedrooms</label>
                <input type="number" step="any" name="bedrooms" required placeholder="e.g. 3">
            </div>
            <div class="input-group">
                <label>Number of Bathrooms</label>
                <input type="number" step="any" name="bathrooms" required placeholder="e.g. 2">
            </div>
            <div class="input-group">
                <label>Living Area (sqft)</label>
                <input type="number" step="any" name="living_area" required placeholder="e.g. 2000">
            </div>
            <div class="input-group">
                <label>Lot Area (sqft)</label>
                <input type="number" step="any" name="lot_area" required placeholder="e.g. 5000">
            </div>
            <div class="input-group">
                <label>Number of Floors</label>
                <input type="number" step="any" name="floors" required placeholder="e.g. 1.5">
            </div>
            <div class="input-group">
                <label>Waterfront Present (0 or 1)</label>
                <input type="number" step="any" name="waterfront" required placeholder="0 for No, 1 for Yes">
            </div>
            <div class="input-group">
                <label>Number of Views</label>
                <input type="number" step="any" name="views" required placeholder="e.g. 0 to 4">
            </div>
            <div class="input-group">
                <label>Condition (1 to 5)</label>
                <input type="number" step="any" name="condition" required placeholder="e.g. 3">
            </div>
            <div class="input-group">
                <label>Grade (1 to 13)</label>
                <input type="number" step="any" name="grade" required placeholder="e.g. 7">
            </div>
            <div class="input-group">
                <label>Area Excluding Basement</label>
                <input type="number" step="any" name="area_ex_basement" required placeholder="e.g. 1500">
            </div>
            <div class="input-group">
                <label>Basement Area</label>
                <input type="number" step="any" name="basement_area" required placeholder="e.g. 500">
            </div>
            <div class="input-group">
                <label>Built Year</label>
                <input type="number" step="any" name="built_year" required placeholder="e.g. 1995">
            </div>
            <div class="input-group">
                <label>Renovation Year (0 if none)</label>
                <input type="number" step="any" name="renovation_year" required placeholder="e.g. 2010">
            </div>
            <div class="input-group">
                <label>Renovated Lot Area</label>
                <input type="number" step="any" name="lot_area_renov" required placeholder="e.g. 5000">
            </div>
            <div class="input-group">
                <label>Schools Nearby</label>
                <input type="number" step="any" name="schools" required placeholder="e.g. 3">
            </div>
            <div class="input-group">
                <label>Distance from Airport (km)</label>
                <input type="number" step="any" name="airport_dist" required placeholder="e.g. 15">
            </div>

            <div class="btn-container">
                <button type="submit">Predict Price</button>
            </div>
        </form>

        {% if prediction %}
        <div class="result-box">
            <h3>Estimated Property Value</h3>
            <h2>${{ prediction }}</h2>
        </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded correctly.", 500
    
    try:
        # Extract features from form in exact order expected by Scikit-Learn model
        features = [
            float(request.form['bedrooms']),
            float(request.form['bathrooms']),
            float(request.form['living_area']),
            float(request.form['lot_area']),
            float(request.form['floors']),
            float(request.form['waterfront']),
            float(request.form['views']),
            float(request.form['condition']),
            float(request.form['grade']),
            float(request.form['area_ex_basement']),
            float(request.form['basement_area']),
            float(request.form['built_year']),
            float(request.form['renovation_year']),
            float(request.form['lot_area_renov']),
            float(request.form['schools']),
            float(request.form['airport_dist'])
        ]
        
        # Convert to 2D numpy array
        final_features = np.array([features])
        
        # Perform prediction
        prediction = model.predict(final_features)
        formatted_price = f"{prediction[0]:,.2f}"
        
        return render_template_string(HTML_TEMPLATE, prediction=formatted_price)
    
    except Exception as e:
        return f"An error occurred during prediction: {e}", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
