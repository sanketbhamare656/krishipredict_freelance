from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from model_utils import SimpleYieldRiskModel  # Import the class

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
label_encoders = None

def load_model():
    global model, scaler, label_encoders
    try:
        print("Attempting to load model...")
        
        if os.path.exists('model.pkl'):
            print("Found model.pkl file")
            model_data = joblib.load('model.pkl')
            
            model = model_data['model']
            scaler = model_data['scaler']
            label_encoders = model_data['label_encoders']
            
            print("Model loaded successfully!")
            print(f"Available crops: {list(label_encoders['Crop'].classes_)}")
            print(f"Available soil types: {list(label_encoders['Soil_Type'].classes_)}")
            
            return True
        else:
            print("No model.pkl file found")
            return False
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())
        return False

# Routes for different pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/recommend')
def recommend_page():
    return render_template('recommend.html')

@app.route('/farmer-life')
def farmer_life():
    return render_template('farmer_life.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Ensure model is loaded
        if model is None:
            if not load_model():
                return jsonify({
                    'success': False,
                    'error': 'Model not available. Please run create_test_model.py first.'
                })
        
        data = request.json
        print(f"Received prediction request: {data}")
        
        # Validate required fields
        required_fields = ['crop_name', 'soil_type', 'rainfall', 'temperature', 'humidity', 'soil_ph', 'nitrogen', 'phosphorus', 'potassium']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                })
        
        # Prepare input data
        input_data = {
            'Crop': data['crop_name'],
            'Soil_Type': data['soil_type'],
            'Rainfall': float(data['rainfall']),
            'Temperature': float(data['temperature']),
            'Humidity': float(data['humidity']),
            'Soil_pH': float(data['soil_ph']),
            'N': float(data['nitrogen']),
            'P': float(data['phosphorus']),
            'K': float(data['potassium'])
        }
        
        print(f"Processing input: {input_data}")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for column in ['Crop', 'Soil_Type']:
            if column in label_encoders:
                le = label_encoders[column]
                if input_data[column] in le.classes_:
                    input_df[column] = le.transform([input_data[column]])
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Invalid {column}: {input_data[column]}. Available: {list(le.classes_)}'
                    })
        
        # Scale numerical features
        numerical_features = ['Rainfall', 'Temperature', 'Humidity', 'Soil_pH', 'N', 'P', 'K']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Make prediction
        yield_pred, risk_proba = model.predict(input_df)
        
        predicted_yield = round(float(yield_pred[0]), 2)
        risk_level_idx = np.argmax(risk_proba[0])
        risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        predicted_risk = risk_mapping[risk_level_idx]
        
        # Generate recommendations
        recommendations = generate_recommendations(input_data, predicted_risk, predicted_yield)
        
        response = {
            'success': True,
            'yield': predicted_yield,
            'risk_level': predicted_risk,
            'risk_probabilities': risk_proba[0].tolist(),
            'recommendations': recommendations
        }
        
        print(f"Prediction successful: {response}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f'Prediction error: {str(e)}'
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': error_msg
        })

def generate_recommendations(input_data, risk_level, yield_pred):
    recommendations = []
    
    # Soil pH recommendations
    ph = input_data['Soil_pH']
    if ph < 6.0:
        recommendations.append("Add lime to increase soil pH for better nutrient availability")
    elif ph > 7.5:
        recommendations.append("Add sulfur or organic matter to lower soil pH")
    
    # Nutrient recommendations
    n, p, k = input_data['N'], input_data['P'], input_data['K']
    if n < 50:
        recommendations.append("Apply nitrogen-rich fertilizers like urea")
    if p < 30:
        recommendations.append("Add phosphorus fertilizers like DAP")
    if k < 40:
        recommendations.append("Apply potassium fertilizers like MOP")
    
    # Risk-based recommendations
    if risk_level == 'High':
        recommendations.extend([
            "Consider crop insurance for risk mitigation",
            "Implement drip irrigation for water efficiency",
            "Use organic mulching to conserve soil moisture"
        ])
    elif risk_level == 'Medium':
        recommendations.extend([
            "Monitor weather forecasts regularly",
            "Maintain proper drainage systems",
            "Use integrated pest management"
        ])
    else:
        recommendations.extend([
            "Continue current agricultural practices",
            "Maintain soil health with crop rotation",
            "Regular soil testing recommended"
        ])
    
    # Yield optimization
    if yield_pred < 20:
        recommendations.append("Improve soil fertility and consider high-yield varieties")
    
    return recommendations

@app.route('/api/crops')
def get_crops():
    try:
        if label_encoders and 'Crop' in label_encoders:
            crops = list(label_encoders['Crop'].classes_)
        else:
            # Fallback
            crops = ['Wheat', 'Rice', 'Cotton', 'Sugarcane', 'Onion']
        
        return jsonify({'success': True, 'crops': crops})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/soil-types')
def get_soil_types():
    try:
        if label_encoders and 'Soil_Type' in label_encoders:
            soil_types = list(label_encoders['Soil_Type'].classes_)
        else:
            # Fallback
            soil_types = ['Black Cotton', 'Alluvial', 'Red']
        
        return jsonify({'success': True, 'soil_types': soil_types})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recommend-crops', methods=['POST'])
def api_recommend_crops():
    try:
        data = request.json
        print(f"Received recommendation request: {data}")
        
        # Validate required fields
        required_fields = ['soil_type', 'temperature', 'rainfall', 'humidity', 'soil_ph', 'nitrogen', 'phosphorus', 'potassium']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                })
        
        # Ensure model is loaded
        if model is None:
            if not load_model():
                return jsonify({
                    'success': False,
                    'error': 'Model not available.'
                })
        
        # Get all available crops
        available_crops = list(label_encoders['Crop'].classes_)
        print(f"Available crops for recommendation: {available_crops}")
        
        results = []
        
        # Predict for each crop
        for crop in available_crops:
            try:
                # Prepare input data for this crop
                input_data = {
                    'Crop': crop,
                    'Soil_Type': data['soil_type'],
                    'Rainfall': float(data['rainfall']),
                    'Temperature': float(data['temperature']),
                    'Humidity': float(data['humidity']),
                    'Soil_pH': float(data['soil_ph']),
                    'N': float(data['nitrogen']),
                    'P': float(data['phosphorus']),
                    'K': float(data['potassium'])
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical variables
                for column in ['Crop', 'Soil_Type']:
                    if column in label_encoders:
                        le = label_encoders[column]
                        if input_data[column] in le.classes_:
                            input_df[column] = le.transform([input_data[column]])
                        else:
                            continue  # Skip if crop not in encoder
                
                # Scale numerical features
                numerical_features = ['Rainfall', 'Temperature', 'Humidity', 'Soil_pH', 'N', 'P', 'K']
                input_df[numerical_features] = scaler.transform(input_df[numerical_features])
                
                # Make prediction
                yield_pred, risk_proba = model.predict(input_df)
                
                predicted_yield = round(float(yield_pred[0]), 2)
                risk_level_idx = np.argmax(risk_proba[0])
                risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
                model_risk = risk_mapping[risk_level_idx]
                
                # Calculate suitability score with improved risk calculation
                suitability_score, calculated_risk = calculate_suitability_score(
                    crop, data['soil_type'], float(data['temperature']), 
                    float(data['rainfall']), float(data['soil_ph']),
                    predicted_yield, model_risk, data.get('season')
                )
                
                # Get crop season
                crop_season = get_crop_season(crop)
                
                results.append({
                    'crop': crop,
                    'predicted_yield': predicted_yield,
                    'risk_level': calculated_risk,  # Use calculated risk instead of model risk
                    'suitability_score': suitability_score,
                    'season': crop_season,
                    'risk_probabilities': risk_proba[0].tolist()
                })
                
            except Exception as e:
                print(f"Error processing crop {crop}: {str(e)}")
                continue
        
        # Sort by suitability score (descending) and get top 5
        results.sort(key=lambda x: x['suitability_score'], reverse=True)
        top_crops = results[:5]
        
        response = {
            'success': True,
            'recommended_crops': top_crops,
            'input_conditions': {
                'soil_type': data['soil_type'],
                'temperature': float(data['temperature']),
                'rainfall': float(data['rainfall']),
                'humidity': float(data['humidity']),
                'soil_ph': float(data['soil_ph']),
                'nitrogen': float(data['nitrogen']),
                'phosphorus': float(data['phosphorus']),
                'potassium': float(data['potassium']),
                'season': data.get('season')
            }
        }
        
        print(f"Recommendation successful: {len(top_crops)} crops recommended")
        return jsonify(response)
            
    except Exception as e:
        error_msg = f'Recommendation error: {str(e)}'
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': error_msg
        })

def calculate_suitability_score(crop, soil_type, temperature, rainfall, soil_ph, yield_pred, model_risk, user_season=None):
    """
    Calculate how suitable a crop is for given conditions
    Higher score means more suitable
    """
    score = 0
    
    # Base score from predicted yield (normalized)
    score += min(yield_pred / 10, 15)  # Cap at 15 points
    
    # Improved risk calculation based on actual conditions
    risk_score = 0
    
    # Temperature check
    ideal_temps = {
        'Rice': 25, 'Wheat': 20, 'Cotton': 28, 'Sugarcane': 26, 'Onion': 20
    }
    if crop in ideal_temps:
        temp_diff = abs(temperature - ideal_temps[crop])
        if temp_diff <= 2:
            risk_score += 0  # Low risk
        elif temp_diff <= 5:
            risk_score += 1  # Medium risk
        else:
            risk_score += 2  # High risk
    
    # Rainfall check
    ideal_rainfall = {
        'Rice': 1200, 'Wheat': 500, 'Cotton': 500, 'Sugarcane': 1500, 'Onion': 400
    }
    if crop in ideal_rainfall:
        rain_diff = abs(rainfall - ideal_rainfall[crop]) / ideal_rainfall[crop]
        if rain_diff <= 0.2:
            risk_score += 0
        elif rain_diff <= 0.4:
            risk_score += 1
        else:
            risk_score += 2
    
    # Soil pH check
    ideal_ph = {
        'Rice': 5.5, 'Wheat': 6.8, 'Cotton': 7.2, 'Sugarcane': 7.5, 'Onion': 6.5
    }
    if crop in ideal_ph:
        ph_diff = abs(soil_ph - ideal_ph[crop])
        if ph_diff <= 0.5:
            risk_score += 0
        elif ph_diff <= 1.0:
            risk_score += 1
        else:
            risk_score += 2
    
    # Determine final risk level based on conditions
    if risk_score <= 2:
        final_risk = 'Low'
        score += 10  # Bonus for low risk
    elif risk_score <= 4:
        final_risk = 'Medium' 
        score += 5   # Small bonus for medium risk
    else:
        final_risk = 'High'
        score += 0   # No bonus for high risk
    
    # Season compatibility bonus
    crop_season = get_crop_season(crop)
    if user_season and crop_season in [user_season, 'Year-round']:
        score += 5
    
    # Soil type preference bonus
    preferred_soils = {
        'Cotton': ['Black Cotton'],
        'Rice': ['Alluvial', 'Black Cotton'],
        'Wheat': ['Alluvial', 'Black Cotton'],
        'Sugarcane': ['Alluvial', 'Black Cotton'],
        'Onion': ['Alluvial', 'Red']
    }
    
    if crop in preferred_soils and soil_type in preferred_soils[crop]:
        score += 3
    
    return round(score, 2), final_risk

def get_crop_season(crop_name):
    """Get the suitable season for a crop"""
    season_mapping = {
        'Kharif': ['Rice', 'Cotton', 'Soybean', 'Maize', 'Groundnut'],
        'Rabi': ['Wheat', 'Onion', 'Potato', 'Sunflower', 'Chickpea'],
        'Year-round': ['Sugarcane', 'Tomato', 'Brinjal', 'Cabbage', 'Cauliflower']
    }
    
    for season, crops in season_mapping.items():
        if crop_name in crops:
            return season
    
    # Default based on crop type
    if crop_name in ['Rice', 'Cotton', 'Maize']:
        return 'Kharif'
    elif crop_name in ['Wheat', 'Onion']:
        return 'Rabi'
    else:
        return 'Year-round'

@app.route('/api/model-status')
def model_status():
    status = {
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'label_encoders_loaded': label_encoders is not None,
        'model_file_exists': os.path.exists('model.pkl')
    }
    return jsonify(status)

if __name__ == '__main__':
    print("Starting KrishiPredict application...")
    
    # Try to load existing model
    if load_model():
        print("Model loaded successfully!")
    else:
        print("WARNING: Could not load model! Please run create_test_model.py first")
    
    print("Application starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)