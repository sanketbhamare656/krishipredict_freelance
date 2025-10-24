# KrishiPredict - AI-based Yield & Risk Forecaster 🌱

KrishiPredict is an intelligent agricultural decision support system that uses machine learning to predict crop yields and assess risk levels based on environmental and soil conditions. The system helps farmers make informed decisions about crop selection and cultivation practices.

## 🚀 Features

- **Yield Prediction**: Predict crop yield in quintals per hectare
- **Risk Assessment**: Classify risk levels as Low, Medium, or High
- **Crop Recommendation**: Get AI-powered crop suggestions based on field conditions
- **Smart Recommendations**: Receive actionable farming advice
- **Maharashtra Focus**: Specialized for crops grown in Maharashtra region

## 🛠️ Tech Stack

- **Backend**: Flask, Python
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Model**: Random Forest (Regression + Classification)

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)

## 🏃‍♂️ How to Run

### 1. Clone and Setup
```bash
# Create project directory
mkdir KrishiPredict
cd KrishiPredict

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

If requirements.txt is not available, install manually:
```bash
pip install Flask==2.3.3 pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 joblib==1.3.2
```

### 3. Generate Model and Data
```bash
python create_test_model.py
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Application
Open your browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure
```
KrishiPredict/
├── app.py                 # Main Flask application
├── model_utils.py         # ML model utilities
├── create_test_model.py   # Model training script
├── requirements.txt       # Python dependencies
├── model.pkl             # Trained ML model (generated)
├── data/
│   └── maharashtra_crops_data.csv  # Crop dataset
├── templates/
│   ├── index.html        # Landing page
│   ├── predict.html      # Prediction page
│   ├── recommend.html    # Recommendation page
│   ├── farmer_life.html  # Farmer resources
│   ├── chatbot.html      # AI assistant
│   └── about.html        # About page
└── static/
    ├── css/
    │   └── style.css     # Stylesheets
    └── js/
        └── script.js     # JavaScript
```

## 🧪 Testing Data

### Yield Prediction Test Data

#### Test Case 1: Onion (Good Conditions)
```json
{
  "crop_name": "Onion",
  "soil_type": "Alluvial",
  "rainfall": 400,
  "temperature": 20,
  "humidity": 65,
  "soil_ph": 6.5,
  "nitrogen": 60,
  "phosphorus": 40,
  "potassium": 50
}
```

#### Test Case 2: Rice (Ideal Conditions)
```json
{
  "crop_name": "Rice",
  "soil_type": "Alluvial",
  "rainfall": 1200,
  "temperature": 25,
  "humidity": 75,
  "soil_ph": 5.5,
  "nitrogen": 70,
  "phosphorus": 45,
  "potassium": 55
}
```

#### Test Case 3: Cotton (Black Soil)
```json
{
  "crop_name": "Cotton",
  "soil_type": "Black Cotton",
  "rainfall": 500,
  "temperature": 28,
  "humidity": 60,
  "soil_ph": 7.2,
  "nitrogen": 55,
  "phosphorus": 35,
  "potassium": 45
}
```

### Crop Recommendation Test Data

#### Recommendation Test 1: Alluvial Soil (Good for multiple crops)
```json
{
  "soil_type": "Alluvial",
  "temperature": 25,
  "rainfall": 800,
  "humidity": 70,
  "soil_ph": 6.5,
  "nitrogen": 60,
  "phosphorus": 40,
  "potassium": 50,
  "season": "Kharif"
}
```

#### Recommendation Test 2: Black Cotton Soil (Cotton area)
```json
{
  "soil_type": "Black Cotton",
  "temperature": 28,
  "rainfall": 500,
  "humidity": 65,
  "soil_ph": 7.2,
  "nitrogen": 55,
  "phosphorus": 35,
  "potassium": 45,
  "season": "Kharif"
}
```

#### Recommendation Test 3: Low Rainfall Area
```json
{
  "soil_type": "Red",
  "temperature": 30,
  "rainfall": 350,
  "humidity": 55,
  "soil_ph": 6.8,
  "nitrogen": 45,
  "phosphorus": 30,
  "potassium": 40,
  "season": "Rabi"
}
```

## 🔧 API Endpoints

### Yield Prediction
- **URL**: `POST /api/predict`
- **Content-Type**: `application/json`
- **Input**: Crop and field parameters
- **Output**: Yield prediction and risk assessment

### Crop Recommendation
- **URL**: `POST /api/recommend-crops`
- **Content-Type**: `application/json`
- **Input**: Field conditions
- **Output**: Top 5 recommended crops

### Get Available Crops
- **URL**: `GET /api/crops`
- **Output**: List of supported crops

### Get Soil Types
- **URL**: `GET /api/soil-types`
- **Output**: List of supported soil types

### Model Status
- **URL**: `GET /api/model-status`
- **Output**: Model loading status

## 🌾 Supported Crops
- Wheat
- Rice
- Cotton
- Sugarcane
- Onion

## 🌱 Supported Soil Types
- Black Cotton
- Alluvial
- Red
- Laterite
- Mountain

## 🎯 Expected Results

### For Yield Prediction:
- **Good conditions**: High yield (40-80 q/ha), Low risk
- **Poor conditions**: Low yield (10-30 q/ha), High risk
- **Medium conditions**: Medium yield (30-50 q/ha), Medium risk

### For Crop Recommendation:
- **Alluvial soil**: Rice, Sugarcane, Wheat, Onion
- **Black Cotton soil**: Cotton, Soybean, Jowar
- **Red soil**: Groundnut, Pulses, Millets

## 🐛 Troubleshooting

### Common Issues:

1. **Model not loading**: Run `python create_test_model.py` first
2. **Port already in use**: Change port in `app.py` or kill existing process
3. **Import errors**: Check virtual environment activation
4. **404 errors**: Ensure all routes are properly defined in `app.py`

### Debug Steps:
```bash
# Check model status
curl http://localhost:5000/api/model-status

# Check available crops
curl http://localhost:5000/api/crops

# Test prediction API
curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" -d '{"crop_name":"Onion","soil_type":"Alluvial","rainfall":400,"temperature":20,"humidity":65,"soil_ph":6.5,"nitrogen":60,"phosphorus":40,"potassium":50}'
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model is properly trained
4. Check browser console for JavaScript errors

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Farming! 🌾🚀**

<script>
function copyReadme() {
    const readmeContent = `# KrishiPredict - AI-based Yield & Risk Forecaster 🌱

KrishiPredict is an intelligent agricultural decision support system that uses machine learning to predict crop yields and assess risk levels based on environmental and soil conditions. The system helps farmers make informed decisions about crop selection and cultivation practices.

## 🚀 Features

- **Yield Prediction**: Predict crop yield in quintals per hectare
- **Risk Assessment**: Classify risk levels as Low, Medium, or High
- **Crop Recommendation**: Get AI-powered crop suggestions based on field conditions
- **Smart Recommendations**: Receive actionable farming advice
- **Maharashtra Focus**: Specialized for crops grown in Maharashtra region

## 🛠️ Tech Stack

- **Backend**: Flask, Python
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Model**: Random Forest (Regression + Classification)

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)

## 🏃‍♂️ How to Run

### 1. Clone and Setup
\`\`\`bash
# Create project directory
mkdir KrishiPredict
cd KrishiPredict

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate
\`\`\`

### 2. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

If requirements.txt is not available, install manually:
\`\`\`bash
pip install Flask==2.3.3 pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 joblib==1.3.2
\`\`\`

### 3. Generate Model and Data
\`\`\`bash
python create_test_model.py
\`\`\`

### 4. Run the Application
\`\`\`bash
python app.py
\`\`\`

### 5. Access the Application
Open your browser and navigate to:
\`\`\`
http://localhost:5000
\`\`\`

## 📁 Project Structure
\`\`\`
KrishiPredict/
├── app.py                 # Main Flask application
├── model_utils.py         # ML model utilities
├── create_test_model.py   # Model training script
├── requirements.txt       # Python dependencies
├── model.pkl             # Trained ML model (generated)
├── data/
│   └── maharashtra_crops_data.csv  # Crop dataset
├── templates/
│   ├── index.html        # Landing page
│   ├── predict.html      # Prediction page
│   ├── recommend.html    # Recommendation page
│   ├── farmer_life.html  # Farmer resources
│   ├── chatbot.html      # AI assistant
│   └── about.html        # About page
└── static/
    ├── css/
    │   └── style.css     # Stylesheets
    └── js/
        └── script.js     # JavaScript
\`\`\`

## 🧪 Testing Data

### Yield Prediction Test Data

#### Test Case 1: Onion (Good Conditions)
\`\`\`json
{
  "crop_name": "Onion",
  "soil_type": "Alluvial",
  "rainfall": 400,
  "temperature": 20,
  "humidity": 65,
  "soil_ph": 6.5,
  "nitrogen": 60,
  "phosphorus": 40,
  "potassium": 50
}
\`\`\`

#### Test Case 2: Rice (Ideal Conditions)
\`\`\`json
{
  "crop_name": "Rice",
  "soil_type": "Alluvial",
  "rainfall": 1200,
  "temperature": 25,
  "humidity": 75,
  "soil_ph": 5.5,
  "nitrogen": 70,
  "phosphorus": 45,
  "potassium": 55
}
\`\`\`

#### Test Case 3: Cotton (Black Soil)
\`\`\`json
{
  "crop_name": "Cotton",
  "soil_type": "Black Cotton",
  "rainfall": 500,
  "temperature": 28,
  "humidity": 60,
  "soil_ph": 7.2,
  "nitrogen": 55,
  "phosphorus": 35,
  "potassium": 45
}
\`\`\`

### Crop Recommendation Test Data

#### Recommendation Test 1: Alluvial Soil (Good for multiple crops)
\`\`\`json
{
  "soil_type": "Alluvial",
  "temperature": 25,
  "rainfall": 800,
  "humidity": 70,
  "soil_ph": 6.5,
  "nitrogen": 60,
  "phosphorus": 40,
  "potassium": 50,
  "season": "Kharif"
}
\`\`\`

#### Recommendation Test 2: Black Cotton Soil (Cotton area)
\`\`\`json
{
  "soil_type": "Black Cotton",
  "temperature": 28,
  "rainfall": 500,
  "humidity": 65,
  "soil_ph": 7.2,
  "nitrogen": 55,
  "phosphorus": 35,
  "potassium": 45,
  "season": "Kharif"
}
\`\`\`

#### Recommendation Test 3: Low Rainfall Area
\`\`\`json
{
  "soil_type": "Red",
  "temperature": 30,
  "rainfall": 350,
  "humidity": 55,
  "soil_ph": 6.8,
  "nitrogen": 45,
  "phosphorus": 30,
  "potassium": 40,
  "season": "Rabi"
}
\`\`\`

## 🔧 API Endpoints

### Yield Prediction
- **URL**: \`POST /api/predict\`
- **Content-Type**: \`application/json\`
- **Input**: Crop and field parameters
- **Output**: Yield prediction and risk assessment

### Crop Recommendation
- **URL**: \`POST /api/recommend-crops\`
- **Content-Type**: \`application/json\`
- **Input**: Field conditions
- **Output**: Top 5 recommended crops

### Get Available Crops
- **URL**: \`GET /api/crops\`
- **Output**: List of supported crops

### Get Soil Types
- **URL**: \`GET /api/soil-types\`
- **Output**: List of supported soil types

### Model Status
- **URL**: \`GET /api/model-status\`
- **Output**: Model loading status

## 🌾 Supported Crops
- Wheat
- Rice
- Cotton
- Sugarcane
- Onion

## 🌱 Supported Soil Types
- Black Cotton
- Alluvial
- Red
- Laterite
- Mountain

## 🎯 Expected Results

### For Yield Prediction:
- **Good conditions**: High yield (40-80 q/ha), Low risk
- **Poor conditions**: Low yield (10-30 q/ha), High risk
- **Medium conditions**: Medium yield (30-50 q/ha), Medium risk

### For Crop Recommendation:
- **Alluvial soil**: Rice, Sugarcane, Wheat, Onion
- **Black Cotton soil**: Cotton, Soybean, Jowar
- **Red soil**: Groundnut, Pulses, Millets

## 🐛 Troubleshooting

### Common Issues:

1. **Model not loading**: Run \`python create_test_model.py\` first
2. **Port already in use**: Change port in \`app.py\` or kill existing process
3. **Import errors**: Check virtual environment activation
4. **404 errors**: Ensure all routes are properly defined in \`app.py\`

### Debug Steps:
\`\`\`bash
# Check model status
curl http://localhost:5000/api/model-status

# Check available crops
curl http://localhost:5000/api/crops

# Test prediction API
curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" -d '{"crop_name":"Onion","soil_type":"Alluvial","rainfall":400,"temperature":20,"humidity":65,"soil_ph":6.5,"nitrogen":60,"phosphorus":40,"potassium":50}'
\`\`\`

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model is properly trained
4. Check browser console for JavaScript errors

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Farming! 🌾🚀**`;

    navigator.clipboard.writeText(readmeContent).then(() => {
        alert('README.md content copied to clipboard!');
    }).catch(err => {
        console.error('Failed to copy: ', err);
    });
}
</script>

<button onclick="copyReadme()" style="position: fixed; top: 20px; right: 20px; padding: 10px 15px; background: #2e7d32; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 14px;">
    📋 Copy README
</button>

<style>
body {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    font-family: Arial, sans-serif;
    line-height: 1.6;
}

h1, h2, h3 {
    color: #2e7d32;
}

code {
    background: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
}

pre {
    
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    position: relative;
}

pre::after {
    content: "📋";
    position: absolute;
    top: 10px;
    right: 10px;
    cursor: pointer;
    font-size: 16px;
}

button:hover {
    background: #1b5e20;
    transform: translateY(-2px);
}
</style>