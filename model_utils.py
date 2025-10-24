import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Define the model class at the top level so it can be pickled
class SimpleYieldRiskModel:
    def __init__(self, yield_model, risk_model):
        self.yield_model = yield_model
        self.risk_model = risk_model
        
    def predict(self, X):
        yield_pred = self.yield_model.predict(X)
        risk_pred_proba = self.risk_model.predict_proba(X)
        return yield_pred, risk_pred_proba

# Create a simple test dataset
def create_test_data():
    np.random.seed(42)
    
    data = []
    crops = ['Wheat', 'Rice', 'Cotton', 'Sugarcane', 'Onion']
    soil_types = ['Black Cotton', 'Alluvial', 'Red']
    
    for _ in range(100):
        crop = np.random.choice(crops)
        soil_type = np.random.choice(soil_types)
        
        data.append({
            'Crop': crop,
            'Soil_Type': soil_type,
            'Rainfall': np.random.uniform(300, 1500),
            'Temperature': np.random.uniform(15, 35),
            'Humidity': np.random.uniform(40, 90),
            'Soil_pH': np.random.uniform(5.0, 8.5),
            'N': np.random.uniform(20, 100),
            'P': np.random.uniform(15, 80),
            'K': np.random.uniform(20, 90),
            'Yield': np.random.uniform(10, 100),
            'Risk_Level': np.random.choice(['Low', 'Medium', 'High'])
        })
    
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/maharashtra_crops_data.csv', index=False)
    print("Test data created successfully!")
    return df

# Simple model training function
def train_simple_model():
    # Load data
    df = pd.read_csv('data/maharashtra_crops_data.csv')
    
    # Create a copy to avoid SettingWithCopyWarning
    X = df[['Crop', 'Soil_Type', 'Rainfall', 'Temperature', 'Humidity', 'Soil_pH', 'N', 'P', 'K']].copy()
    y_yield = df['Yield'].copy()
    y_risk = df['Risk_Level'].copy()
    
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
    
    # Train models
    yield_model = RandomForestRegressor(n_estimators=50, random_state=42)
    risk_model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    yield_model.fit(X_train, y_yield_train)
    risk_model.fit(X_train, y_risk_train)
    
    # Create model instance using the top-level class
    model = SimpleYieldRiskModel(yield_model, risk_model)
    
    # Test prediction
    test_pred = model.predict(X_test[:1])
    print(f"Test prediction - Yield: {test_pred[0][0]}, Risk probabilities: {test_pred[1][0]}")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders
    }
    
    joblib.dump(model_data, 'model.pkl')
    print("Simple model trained and saved successfully!")
    
    # Print available crops and soil types
    print(f"Available crops: {list(label_encoders['Crop'].classes_)}")
    print(f"Available soil types: {list(label_encoders['Soil_Type'].classes_)}")
    
    return model_data