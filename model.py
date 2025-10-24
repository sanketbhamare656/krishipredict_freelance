import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import os

def generate_maharashtra_crop_data(num_samples=2000):
    """Generate synthetic crop data for Maharashtra with realistic parameters"""
    
    np.random.seed(42)
    
    # Major crops of Maharashtra with realistic parameters
    maharashtra_crops = {
        'Jowar (Sorghum)': {'temp': 27, 'rainfall': 450, 'yield_base': 25, 'ph': 6.5, 'season': 'Kharif', 'soil_preference': ['Black Cotton', 'Alluvial']},
        'Bajra (Pearl Millet)': {'temp': 30, 'rainfall': 400, 'yield_base': 18, 'ph': 7.0, 'season': 'Kharif', 'soil_preference': ['Black Cotton', 'Red']},
        'Rice': {'temp': 25, 'rainfall': 1200, 'yield_base': 35, 'ph': 5.5, 'season': 'Kharif', 'soil_preference': ['Alluvial', 'Black Cotton']},
        'Wheat': {'temp': 20, 'rainfall': 500, 'yield_base': 30, 'ph': 6.8, 'season': 'Rabi', 'soil_preference': ['Alluvial', 'Black Cotton']},
        'Tur (Pigeon Pea)': {'temp': 26, 'rainfall': 600, 'yield_base': 12, 'ph': 6.5, 'season': 'Kharif', 'soil_preference': ['Black Cotton', 'Red']},
        'Chickpea': {'temp': 22, 'rainfall': 400, 'yield_base': 15, 'ph': 7.0, 'season': 'Rabi', 'soil_preference': ['Alluvial', 'Black Cotton']},
        'Soybean': {'temp': 24, 'rainfall': 650, 'yield_base': 20, 'ph': 6.2, 'season': 'Kharif', 'soil_preference': ['Black Cotton', 'Alluvial']},
        'Cotton': {'temp': 28, 'rainfall': 500, 'yield_base': 22, 'ph': 7.2, 'season': 'Kharif', 'soil_preference': ['Black Cotton']},
        'Sugarcane': {'temp': 26, 'rainfall': 1500, 'yield_base': 800, 'ph': 7.5, 'season': 'Year-round', 'soil_preference': ['Alluvial', 'Black Cotton']},
        'Groundnut': {'temp': 27, 'rainfall': 500, 'yield_base': 16, 'ph': 6.5, 'season': 'Kharif', 'soil_preference': ['Red', 'Alluvial']},
        'Sunflower': {'temp': 25, 'rainfall': 450, 'yield_base': 14, 'ph': 6.8, 'season': 'Rabi', 'soil_preference': ['Alluvial', 'Black Cotton']},
        'Maize': {'temp': 23, 'rainfall': 600, 'yield_base': 28, 'ph': 6.5, 'season': 'Kharif', 'soil_preference': ['Alluvial', 'Black Cotton']},
        'Moong (Green Gram)': {'temp': 26, 'rainfall': 500, 'yield_base': 10, 'ph': 6.8, 'season': 'Kharif', 'soil_preference': ['Alluvial', 'Red']},
        'Urad (Black Gram)': {'temp': 25, 'rainfall': 550, 'yield_base': 11, 'ph': 6.8, 'season': 'Kharif', 'soil_preference': ['Alluvial', 'Red']},
        'Safflower': {'temp': 22, 'rainfall': 350, 'yield_base': 8, 'ph': 7.0, 'season': 'Rabi', 'soil_preference': ['Black Cotton', 'Red']},
        'Onion': {'temp': 20, 'rainfall': 400, 'yield_base': 180, 'ph': 6.5, 'season': 'Rabi', 'soil_preference': ['Alluvial', 'Red']},
        'Tomato': {'temp': 22, 'rainfall': 600, 'yield_base': 250, 'ph': 6.2, 'season': 'Year-round', 'soil_preference': ['Alluvial', 'Red']},
        'Brinjal': {'temp': 24, 'rainfall': 550, 'yield_base': 200, 'ph': 6.0, 'season': 'Year-round', 'soil_preference': ['Alluvial', 'Red']},
        'Cabbage': {'temp': 18, 'rainfall': 500, 'yield_base': 220, 'ph': 6.5, 'season': 'Rabi', 'soil_preference': ['Alluvial', 'Red']},
        'Cauliflower': {'temp': 19, 'rainfall': 500, 'yield_base': 150, 'ph': 6.5, 'season': 'Rabi', 'soil_preference': ['Alluvial', 'Red']},
        'Potato': {'temp': 17, 'rainfall': 450, 'yield_base': 200, 'ph': 5.5, 'season': 'Rabi', 'soil_preference': ['Alluvial', 'Red']},
        'Grapes': {'temp': 25, 'rainfall': 600, 'yield_base': 300, 'ph': 6.5, 'season': 'Year-round', 'soil_preference': ['Alluvial', 'Red']},
        'Mango': {'temp': 27, 'rainfall': 800, 'yield_base': 80, 'ph': 6.5, 'season': 'Year-round', 'soil_preference': ['Alluvial', 'Red', 'Laterite']},
        'Banana': {'temp': 26, 'rainfall': 1200, 'yield_base': 400, 'ph': 6.0, 'season': 'Year-round', 'soil_preference': ['Alluvial', 'Black Cotton']},
        'Orange': {'temp': 23, 'rainfall': 700, 'yield_base': 120, 'ph': 6.2, 'season': 'Year-round', 'soil_preference': ['Alluvial', 'Red']},
        'Pomegranate': {'temp': 25, 'rainfall': 500, 'yield_base': 80, 'ph': 7.0, 'season': 'Year-round', 'soil_preference': ['Alluvial', 'Red']},
        'Sapota': {'temp': 26, 'rainfall': 1000, 'yield_base': 100, 'ph': 6.5, 'season': 'Year-round', 'soil_preference': ['Alluvial', 'Red']},
        'Custard Apple': {'temp': 24, 'rainfall': 600, 'yield_base': 60, 'ph': 6.8, 'season': 'Year-round', 'soil_preference': ['Red', 'Laterite']},
        'Guava': {'temp': 25, 'rainfall': 800, 'yield_base': 90, 'ph': 6.5, 'season': 'Year-round', 'soil_preference': ['Alluvial', 'Red']},
        'Papaya': {'temp': 26, 'rainfall': 1000, 'yield_base': 300, 'ph': 6.0, 'season': 'Year-round', 'soil_preference': ['Alluvial', 'Red']}
    }
    
    soil_types = ['Black Cotton', 'Laterite', 'Alluvial', 'Red', 'Mountain']
    
    data = []
    
    for crop_name, params in maharashtra_crops.items():
        for _ in range(num_samples // len(maharashtra_crops)):
            soil_type = np.random.choice(soil_types)
            
            # Generate features with realistic variations
            temperature = np.random.normal(params['temp'], 3)
            rainfall = np.random.normal(params['rainfall'], 100)
            humidity = np.random.normal(65, 10)
            soil_ph = np.random.normal(params['ph'], 0.5)
            
            # Generate NPK values based on crop requirements
            n = np.random.normal(60, 20)  # Nitrogen
            p = np.random.normal(40, 15)  # Phosphorus
            k = np.random.normal(50, 15)  # Potassium
            
            # Soil type modifiers for Maharashtra
            soil_modifiers = {
                'Black Cotton': 1.3,  # Best for cotton, good water retention
                'Laterite': 0.8,      # Lower fertility
                'Alluvial': 1.2,      # Good for most crops
                'Red': 1.0,           # Average
                'Mountain': 0.7       # Lower fertility
            }
            
            soil_modifier = soil_modifiers[soil_type]
            
            # Yield calculation with realistic factors
            temp_effect = 1 - abs(temperature - params['temp']) / 30
            rain_effect = 1 - abs(rainfall - params['rainfall']) / 800
            humidity_effect = 1 - abs(humidity - 65) / 40
            ph_effect = 1 - abs(soil_ph - params['ph']) / 2
            nutrient_effect = (n/80 + p/50 + k/60) / 3
            
            base_yield = params['yield_base']
            yield_value = (base_yield * soil_modifier * temp_effect * 
                         rain_effect * humidity_effect * ph_effect * nutrient_effect)
            
            # Add some noise
            yield_value += np.random.normal(0, base_yield * 0.15)
            
            # Determine risk level based on multiple factors
            variability = (abs(temperature - params['temp']) + 
                         abs(rainfall - params['rainfall']) + 
                         abs(soil_ph - params['ph']) * 10)
            
            if variability < 80:
                risk_level = 'Low'
            elif variability < 160:
                risk_level = 'Medium'
            else:
                risk_level = 'High'
            
            # Ensure realistic ranges
            temperature = max(15, min(35, temperature))
            rainfall = max(300, min(2000, rainfall))
            humidity = max(40, min(90, humidity))
            soil_ph = max(5.0, min(8.5, soil_ph))
            n, p, k = max(10, n), max(10, p), max(10, k)
            yield_value = max(5, yield_value)
            
            data.append({
                'Crop': crop_name,
                'Soil_Type': soil_type,
                'Rainfall': round(rainfall, 1),
                'Temperature': round(temperature, 1),
                'Humidity': round(humidity, 1),
                'Soil_pH': round(soil_ph, 2),
                'N': round(n, 1),
                'P': round(p, 1),
                'K': round(k, 1),
                'Yield': round(yield_value, 2),
                'Risk_Level': risk_level,
                'Season': params['season']
            })
    
    df = pd.DataFrame(data)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/maharashtra_crops_data.csv', index=False)
    print(f"Generated Maharashtra crop data with {len(data)} samples")
    
    return df

def train_model(data_path='data/maharashtra_crops_data.csv'):
    """Train the ML model for yield prediction and risk classification"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Prepare features and targets
    feature_columns = ['Crop', 'Soil_Type', 'Rainfall', 'Temperature', 'Humidity', 'Soil_pH', 'N', 'P', 'K']
    X = df[feature_columns]
    y_yield = df['Yield']
    y_risk = df['Risk_Level']
    
    # Encode categorical variables
    label_encoders = {}
    for column in ['Crop', 'Soil_Type']:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    # Encode risk levels
    risk_encoder = LabelEncoder()
    y_risk_encoded = risk_encoder.fit_transform(y_risk)
    label_encoders['Risk_Level'] = risk_encoder
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['Rainfall', 'Temperature', 'Humidity', 'Soil_pH', 'N', 'P', 'K']
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    # Split data
    X_train, X_test, y_yield_train, y_yield_test, y_risk_train, y_risk_test = train_test_split(
        X, y_yield, y_risk_encoded, test_size=0.2, random_state=42
    )
    
    # Custom model that handles both regression and classification
    class YieldRiskModel:
        def __init__(self):
            self.yield_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            self.risk_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            
        def fit(self, X, y):
            y_yield, y_risk = y
            self.yield_model.fit(X, y_yield)
            self.risk_model.fit(X, y_risk)
            return self
            
        def predict(self, X):
            yield_pred = self.yield_model.predict(X)
            risk_pred_proba = self.risk_model.predict_proba(X)
            return yield_pred, risk_pred_proba
    
    # Train model
    model = YieldRiskModel()
    model.fit(X_train, (y_yield_train, y_risk_train))
    
    # Evaluate model
    y_yield_pred, y_risk_pred_proba = model.predict(X_test)
    y_risk_pred = np.argmax(y_risk_pred_proba, axis=1)
    
    yield_mae = mean_absolute_error(y_yield_test, y_yield_pred)
    risk_accuracy = accuracy_score(y_risk_test, y_risk_pred)
    
    print(f"Yield Prediction MAE: {yield_mae:.2f}")
    print(f"Risk Classification Accuracy: {risk_accuracy:.2f}")
    
    # Save model and preprocessing objects
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'metrics': {
            'yield_mae': yield_mae,
            'risk_accuracy': risk_accuracy
        }
    }
    
    joblib.dump(model_data, 'model.pkl')
    print("Model trained and saved successfully")
    
    return model_data

def recommend_crops(soil_type, temperature, rainfall, humidity, soil_ph, nitrogen, phosphorus, potassium, season=None):
    """
    Recommend best crops based on user's field conditions
    Returns top 5 crops with their predicted yields and risk levels
    """
    try:
        # Load the trained model
        if not os.path.exists('model.pkl'):
            return {'success': False, 'error': 'Model not trained yet'}
            
        model_data = joblib.load('model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoders = model_data['label_encoders']
        
        # Load crop data to get all available crops
        df = pd.read_csv('data/maharashtra_crops_data.csv')
        all_crops = df['Crop'].unique()
        
        results = []
        
        # Predict for each crop
        for crop in all_crops:
            # Prepare input data
            input_data = {
                'Crop': crop,
                'Soil_Type': soil_type,
                'Rainfall': rainfall,
                'Temperature': temperature,
                'Humidity': humidity,
                'Soil_pH': soil_ph,
                'N': nitrogen,
                'P': phosphorus,
                'K': potassium
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for column in ['Crop', 'Soil_Type']:
                if column in label_encoders:
                    input_df[column] = label_encoders[column].transform(input_df[column])
            
            # Scale numerical features
            numerical_features = ['Rainfall', 'Temperature', 'Humidity', 'Soil_pH', 'N', 'P', 'K']
            input_df[numerical_features] = scaler.transform(input_df[numerical_features])
            
            # Make prediction
            prediction = model.predict(input_df)
            
            predicted_yield = round(prediction[0][0], 2)
            risk_probabilities = prediction[1][0]
            risk_level = np.argmax(risk_probabilities)
            
            risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
            predicted_risk = risk_mapping[risk_level]
            
            # Get crop season from our crop database
            crop_season = get_crop_season(crop)
            
            # Calculate suitability score (higher is better)
            suitability_score = calculate_suitability_score(
                crop, soil_type, temperature, rainfall, soil_ph, 
                predicted_yield, predicted_risk, season
            )
            
            results.append({
                'crop': crop,
                'predicted_yield': predicted_yield,
                'risk_level': predicted_risk,
                'suitability_score': suitability_score,
                'season': crop_season,
                'risk_probabilities': risk_probabilities.tolist()
            })
        
        # Sort by suitability score (descending) and get top 5
        results.sort(key=lambda x: x['suitability_score'], reverse=True)
        top_crops = results[:5]
        
        return {
            'success': True,
            'recommended_crops': top_crops,
            'input_conditions': {
                'soil_type': soil_type,
                'temperature': temperature,
                'rainfall': rainfall,
                'humidity': humidity,
                'soil_ph': soil_ph,
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'season': season
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_crop_season(crop_name):
    """Get the suitable season for a crop"""
    season_mapping = {
        'Kharif': ['Jowar (Sorghum)', 'Bajra (Pearl Millet)', 'Rice', 'Tur (Pigeon Pea)', 
                  'Soybean', 'Cotton', 'Groundnut', 'Maize', 'Moong (Green Gram)', 'Urad (Black Gram)'],
        'Rabi': ['Wheat', 'Chickpea', 'Sunflower', 'Safflower', 'Onion', 'Cabbage', 'Cauliflower', 'Potato'],
        'Year-round': ['Sugarcane', 'Tomato', 'Brinjal', 'Grapes', 'Mango', 'Banana', 'Orange', 
                      'Pomegranate', 'Sapota', 'Custard Apple', 'Guava', 'Papaya']
    }
    
    for season, crops in season_mapping.items():
        if crop_name in crops:
            return season
    return 'Unknown'

def calculate_suitability_score(crop, soil_type, temperature, rainfall, soil_ph, yield_pred, risk_level, user_season=None):
    """
    Calculate how suitable a crop is for given conditions
    Higher score means more suitable
    """
    score = 0
    
    # Base score from predicted yield (normalized)
    score += min(yield_pred / 50, 10)  # Cap at 10 points
    
    # Risk level scoring (lower risk = higher score)
    risk_scores = {'Low': 10, 'Medium': 5, 'High': 0}
    score += risk_scores.get(risk_level, 0)
    
    # Season compatibility
    crop_season = get_crop_season(crop)
    if user_season and crop_season in [user_season, 'Year-round']:
        score += 5
    
    # Soil type preference bonus
    preferred_soils = {
        'Cotton': ['Black Cotton'],
        'Rice': ['Alluvial', 'Black Cotton'],
        'Wheat': ['Alluvial', 'Black Cotton'],
        'Sugarcane': ['Alluvial', 'Black Cotton'],
        'Groundnut': ['Red', 'Alluvial'],
        'Mango': ['Alluvial', 'Red', 'Laterite']
    }
    
    if crop in preferred_soils and soil_type in preferred_soils[crop]:
        score += 3
    
    return round(score, 2)

if __name__ == '__main__':
    # Generate data and train model
    generate_maharashtra_crop_data()
    train_model()
    
    # Test crop recommendation
    print("\nTesting Crop Recommendation...")
    test_recommendation = recommend_crops(
        soil_type='Black Cotton',
        temperature=28,
        rainfall=500,
        humidity=65,
        soil_ph=7.0,
        nitrogen=60,
        phosphorus=40,
        potassium=50,
        season='Kharif'
    )
    
    if test_recommendation['success']:
        print("Top 5 Recommended Crops:")
        for i, crop in enumerate(test_recommendation['recommended_crops'], 1):
            print(f"{i}. {crop['crop']} - Yield: {crop['predicted_yield']} q/ha, "
                  f"Risk: {crop['risk_level']}, Score: {crop['suitability_score']}")