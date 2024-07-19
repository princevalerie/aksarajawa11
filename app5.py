import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os

# Load the trained model
class_names = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 
               'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']  # 20 classes

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=20, bias=True)  # Adjusted for 20 classes
model.load_state_dict(torch.load('cnn_model1.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Function to predict the class
def predict(image, model, transform):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Function to preprocess Javanese script and segment characters
def preprocess_javanese_script(image):
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(image.convert('L'))

    # Apply threshold to get binary image
    _, binary_image = cv2.threshold(open_cv_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Segment characters and return as list of images
    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_image = binary_image[y:y+h, x:x+w]
        char_images.append(char_image)
    return char_images, contours

# Function to detect spaces between characters
def detect_spaces(contours):
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])  # Sort contours by x position
    spaces = []
    for i in range(1, len(contours)):
        x_prev, _, w_prev, _ = cv2.boundingRect(contours[i - 1])
        x_curr, _, _, _ = cv2.boundingRect(contours[i])
        space_width = x_curr - (x_prev + w_prev)
        if space_width > 20:  # Threshold to determine space
            spaces.append(space_width)
    return spaces

# Streamlit app
st.title("Aksara Jawa Detection")

# Camera input
image_data = st.camera_input("Take a picture")

if image_data is not None:
    # Load the image
    image = Image.open(io.BytesIO(image_data.getvalue()))
    
    # Display the image
    st.image(image, caption='Captured Image', use_column_width=True)
    
    # Segment characters from the image
    segmented_chars, contours = preprocess_javanese_script(image)
    
    if segmented_chars:
        # Detect spaces
        spaces = detect_spaces(contours)
        
        # Predict each character
        recognized_text = ""
        for i, char_image in enumerate(segmented_chars):
            char_image_pil = Image.fromarray(char_image)
            char_class = predict(char_image_pil, model, transform)
            recognized_text += char_class + " "
            if i < len(spaces) and spaces[i] > 20:  # Insert space if detected
                recognized_text += " "
        
        # Display the recognized text
        st.write(f"Recognized Text: {recognized_text.strip()}")
        st.write(f"Jumlah spasi yang terdeteksi: {len(spaces)}")
    else:
        st.write("No characters detected.")
