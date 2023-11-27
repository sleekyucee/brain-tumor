from fastapi import FastAPI, UploadFile, File, Request
from PIL import Image
import torch
from torchvision import transforms
from custom_model import ResNet50
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
import io
import numpy as np
import cv2

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Initialize model
model_params = {'freeze_layers': False, 'weights': 'ResNet50_Weights'}
model = ResNet50(**model_params)

# Load the PyTorch model
model.load_state_dict(torch.load("resnet50.pth", map_location=device))
model.eval()

# Define class names
class_names = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']

# Define data transformation pipeline for preprocessing
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create the FastAPI application
app = FastAPI()

# Create a Jinja2 template object
templates = Jinja2Templates(directory="templates")

# Create a variable to store the current image for visualization
current_image = None

# Define the API endpoint for image classification
@app.post("/classify")
async def classify(image_file: UploadFile = File(...)):
    global current_image  # Use the global variable to store the current image
    # Load the image file
    image = Image.open(image_file.file)

    # Preprocess the image using the defined transformation pipeline
    preprocessed_image = val_transforms(image)

    # Check if the image is already normalized
    mean = preprocessed_image.mean()
    std = preprocessed_image.std()

    if mean < 0 or std < 0 or mean > 1 or std > 1:
        # Apply normalization if needed
        preprocessed_image = transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])(preprocessed_image)

    # Make predictions using the PyTorch model
    with torch.no_grad():
        predictions = model(preprocessed_image.unsqueeze(0))  # Add batch dimension

    # Get the class label and name
    class_label = predictions.argmax().item()
    class_name = class_names[class_label]
    confidence = torch.softmax(predictions, dim=1)[0][class_label].item()

    prediction_msg = f"This brain tumor belongs to class {class_label}.\nIt is a {class_name} type, and the model is {confidence * 100:.2f}% confident."

    # Set the current image for visualization
    current_image = preprocessed_image

    # Return the predictions
    return PlainTextResponse(content=prediction_msg)

# Create a page to display model metrics and ethical concerns
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    model_metrics = {
        'Accuracy': 95.14,
        'Specificity': 97.77,
        'Sensitivity': 93.68,
        'Precision': 97.60
    }
    ethical_concerns = "Our model is designed for research and educational purposes only. We acknowledge the importance of ethical considerations in AI, including data privacy and model biases."

    return templates.TemplateResponse("home.html", {"request": request, "model_metrics": model_metrics, "ethical_concerns": ethical_concerns})

# Start the Uvicorn server to make the API accessible
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
