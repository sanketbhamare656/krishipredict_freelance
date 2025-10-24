// Navigation Toggle
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

if (hamburger) {
    hamburger.addEventListener('click', () => {
        hamburger.classList.toggle('active');
        navMenu.classList.toggle('active');
    });
}

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-link').forEach(n => n.addEventListener('click', () => {
    hamburger.classList.remove('active');
    navMenu.classList.remove('active');
}));

// Crop Data Management
class CropPredictor {
    constructor() {
        this.crops = [];
        this.init();
    }

    async init() {
        await this.loadCrops();
        this.setupEventListeners();
    }

    async loadCrops() {
        try {
            const response = await fetch('/api/crops');
            const data = await response.json();
            
            if (data.success) {
                this.crops = data.crops;
                this.populateCropDropdown();
            }
        } catch (error) {
            console.error('Error loading crops:', error);
        }
    }

    populateCropDropdown() {
        const cropSelect = document.getElementById('crop_name');
        if (cropSelect) {
            // Clear existing options except the first one
            while (cropSelect.options.length > 1) {
                cropSelect.remove(1);
            }

            // Add crop options
            this.crops.forEach(crop => {
                const option = document.createElement('option');
                option.value = crop;
                option.textContent = crop;
                cropSelect.appendChild(option);
            });
        }
    }

    setupEventListeners() {
        const predictionForm = document.getElementById('predictionForm');
        if (predictionForm) {
            predictionForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handlePrediction();
            });
        }

        // Setup batch file upload
        const batchFileInput = document.getElementById('batchFile');
        if (batchFileInput) {
            batchFileInput.addEventListener('change', (e) => {
                this.handleBatchUpload(e.target.files[0]);
            });
        }
    }

    async handlePrediction() {
        const form = document.getElementById('predictionForm');
        const formData = new FormData(form);
        
        const data = {
            crop_name: formData.get('crop_name'),
            soil_type: formData.get('soil_type'),
            rainfall: formData.get('rainfall'),
            temperature: formData.get('temperature'),
            humidity: formData.get('humidity'),
            soil_ph: formData.get('soil_ph'),
            nitrogen: formData.get('nitrogen'),
            phosphorus: formData.get('phosphorus'),
            potassium: formData.get('potassium')
        };

        // Show loading state
        const predictBtn = document.querySelector('.btn-predict');
        const originalText = predictBtn.innerHTML;
        predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
        predictBtn.disabled = true;

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                this.displayResults(result);
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('Network error. Please try again.');
        } finally {
            // Reset button
            predictBtn.innerHTML = originalText;
            predictBtn.disabled = false;
        }
    }

    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        const yieldValue = document.getElementById('yieldValue');
        const riskText = document.getElementById('riskText');
        const riskIndicator = document.getElementById('riskIndicator');
        const riskDescription = document.getElementById('riskDescription');
        const recommendationsList = document.getElementById('recommendationsList');

        // Update yield
        yieldValue.textContent = result.yield;

        // Update risk level with styling
        riskText.textContent = result.risk_level;
        riskIndicator.className = `risk-indicator risk-${result.risk_level.toLowerCase()}`;
        
        // Set risk description
        const riskDescriptions = {
            'Low': 'Favorable conditions with minimal risk factors.',
            'Medium': 'Moderate risk. Some factors need attention.',
            'High': 'High risk conditions detected. Immediate action recommended.'
        };
        riskDescription.textContent = riskDescriptions[result.risk_level];

        // Update recommendations
        recommendationsList.innerHTML = '';
        result.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationsList.appendChild(li);
        });

        // Show results section
        resultsSection.style.display = 'block';

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    showError(message) {
        alert('Error: ' + message);
    }

    async handleBatchUpload(file) {
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/batch_predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.displayBatchResults(result.results);
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('Error uploading file.');
        }
    }

    displayBatchResults(results) {
        const batchResults = document.getElementById('batchResults');
        const tableBody = document.querySelector('#batchResultsTable tbody');
        
        tableBody.innerHTML = '';
        
        results.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.Crop}</td>
                <td>${row.Soil_Type}</td>
                <td>${row.Rainfall}</td>
                <td>${row.Temperature}</td>
                <td>${row.Humidity}</td>
                <td>${row.Predicted_Yield}</td>
                <td>${row.Risk_Level}</td>
            `;
            tableBody.appendChild(tr);
        });

        batchResults.style.display = 'block';
    }

    async retrainModel() {
        try {
            const response = await fetch('/api/train', {
                method: 'POST'
            });

            const result = await response.json();

            if (result.success) {
                alert('Model retrained successfully!');
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('Error retraining model.');
        }
    }
}

// Other page-specific functions
function downloadTemplate() {
    const template = `Crop,Soil_Type,Rainfall,Temperature,Humidity,Soil_pH,N,P,K
Jowar (Sorghum),Black Cotton,450,27,65,6.5,60,40,50
Rice,Alluvial,1200,25,75,5.5,70,45,55
Cotton,Black Cotton,500,28,60,7.2,55,35,45`;

    const blob = new Blob([template], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'krishipredict_template.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function exportResults() {
    const results = []; // This would be populated with actual results
    // Implementation for exporting results
}

// Global functions for HTML onclick events
function retrainModel() {
    if (window.cropPredictor) {
        window.cropPredictor.retrainModel();
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.cropPredictor = new CropPredictor();
    
    // Add scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for animation
    document.querySelectorAll('.feature-card, .step, .crop-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});