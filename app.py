from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your MobileNetV2 model
model = load_model('mobilenet_fadc_asd_model.h5')

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess the image
    img = preprocess_image(filepath)
    prediction = model.predict(img)[0][0]

    print(f"Raw model output: {prediction:.4f}")

    # REVERSED INTERPRETATION
    if prediction > 0.5:
        label = "Non-Autistic"
        confidence = prediction
    else:
        label = "Autistic"
        confidence = 1 - prediction

    print(f"Predicted label: {label} with confidence {confidence * 100:.2f}%")

    return jsonify({'label': label, 'confidence': f"{confidence * 100:.2f}%"})



if __name__ == '__main__':
    app.run(debug=True)
