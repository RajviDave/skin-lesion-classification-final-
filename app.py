import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, url_for
import torch
from torchvision import transforms
from PIL import Image
from skin_lesion_classifier_pytorch import build_model
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model()
model.load_state_dict(torch.load('best_model.pth', map_location=device)['model_state_dict'])
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """Predict the class of an image using the trained model."""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            
        class_names = ['Common Nevus', 'Atypical Nevus', 'Melanoma']
        prediction = class_names[predicted.item()]
        
        # Get probability scores
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence = probabilities[predicted.item()].item() * 100
        
        return {
            'success': True,
            'prediction': prediction,
            'confidence': f"{confidence:.2f}%",
            'probabilities': {
                class_name: f"{prob.item()*100:.2f}%"
                for class_name, prob in zip(class_names, probabilities)
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        file.save(filename)
        
        # Make prediction
        result = predict_image(filename)
        result['image_path'] = url_for('static', filename='uploads/uploaded_image.jpg')
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Get the base64 image data from the request
        image_data = request.json['image']
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data received'})
        
        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image first
        image = Image.open(BytesIO(image_bytes))
        
        # Save the image
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg')
        image.save(filename, 'JPEG')
        
        # Make prediction
        result = predict_image(filename)
        result['image_path'] = url_for('static', filename='uploads/captured_image.jpg')
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print("Error in capture:", traceback.format_exc())  # Detailed error logging
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 