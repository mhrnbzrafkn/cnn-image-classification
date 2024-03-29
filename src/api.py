import torch
from PIL import Image
from flask_cors import CORS
from flask import Flask, request, jsonify
import torchvision.transforms as transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class_names = ['Cat', 'Cheetah', 'Dog', 
               'Fox', 'Leopard', 'Loin', 
               'Tiger', 'Wolf']

def predict_class(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)
        predicted_class_index = predicted.item()
        predicted_class_name = class_names[predicted_class_index]
        return probabilities.squeeze().cpu().numpy().tolist(), predicted_class_name

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Load the CNN model
model = torch.jit.load('./saved_models/CNN-scripted-20.pt').to(device)
model.eval()  # Set model to evaluation mode

# Define image transformation for preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define API endpoint for model inference
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read image file and apply transformation
        image = Image.open(file)
        image = transform(image)
        image = image.to(device)
        # Perform inference
        with torch.no_grad():
            probabilities, predicted_class = predict_class(model, image)

        # Convert output tensor to numpy array and return
        return jsonify({'Probabilities': probabilities, 'Predicted Class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)