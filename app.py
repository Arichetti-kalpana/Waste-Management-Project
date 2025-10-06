import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =================================================================
# 1. Configuration
# =================================================================
STATIC_UPLOADS = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = STATIC_UPLOADS

# Load the ML Model
MODEL_PATH = 'model/fine_tuned_vgg16.h5' 

# CRITICAL FIX: Define the 3 class names in the EXACT order your model was trained
# Based on your folder names (alphabetical is default):
CLASS_NAMES = [
    'Biodegradable Images',
    'Recyclable Images',
    'Trash Images',

]

model = None
try:
    model = load_model(MODEL_PATH)
    print("Machine learning model loaded successfully.")
except Exception as e:
    model = None
    print(f"[ERROR] Model loading failed: {e}")


# =================================================================
# 2. Prediction Functions
# =================================================================

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_and_predict(filepath):
    """Load image, preprocess it, and get prediction."""
    if model is None:
        results = {name: 0 for name in CLASS_NAMES}
        return "Model Error", results

    try:
        # Load and preprocess the image (224x224 and normalized)
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

        # Get prediction (8 probabilities)
        predictions = model.predict(img_array)
        score = predictions[0]

        # Get the class with the highest probability (CRITICAL FIX)
        predicted_class_index = np.argmax(score)
        # Use the index to retrieve the correct name from the 8-element list
        predicted_class_name = CLASS_NAMES[predicted_class_index] 

        # Format results for the webpage display (as percentages)
        results = {name: float(score[i] * 100) for i, name in enumerate(CLASS_NAMES)}

        return predicted_class_name, results
    
    except Exception as e:
        print(f"Prediction failed during processing: {e}")
        results_fail = {name: 0 for name in CLASS_NAMES}
        return "Prediction Failed (Processing)", results_fail

# =================================================================
# 3. Flask Routes
# =================================================================

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html', prediction_made=False, image_path=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('predict_page'))
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('predict_page'))

    filename = file.filename
    # Ensure 'static/uploads' exists
    if not os.path.exists(os.path.join(app.root_path, STATIC_UPLOADS)):
        os.makedirs(os.path.join(app.root_path, STATIC_UPLOADS))
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image_url = url_for('static', filename=f'uploads/{filename}')

    # Run Prediction
    predicted_class, results_dict = preprocess_and_predict(filepath)

    # Render the result
    return render_template(
        'predict.html',
        prediction_made=True,
        predicted_class_name=predicted_class,
        results=results_dict, # Contains all 8 classes
        image_path=image_url
    )

# Placeholder routes for completeness (add these if they were in your original app.py)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # This is where you process the form data (save to database, send an email, etc.)
        # request.form['name'], request.form['email'], etc.
        return 'Message sent successfully!' # or redirect to a thank-you page
    else:
        # This is where you just render the blank form
        return render_template('contact.html')

# =================================================================
# 4. Run Server
# =================================================================
if __name__ == '__main__':
    # Ensure the required folders exist
    for folder in [STATIC_UPLOADS, 'model']:
        if not os.path.exists(folder):
            os.makedirs(folder)

    app.run(debug=True)