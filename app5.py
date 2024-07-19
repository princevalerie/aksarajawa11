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

# Fungsi untuk menampilkan gambar
def show_image(image, title=''):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def preprocess_javanese_script(image):
    # Konversi gambar PIL ke numpy array (RGB)
    image_np = np.array(image)
    
    # Konversi gambar dari RGB ke HSV
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # Definisikan rentang warna hitam dalam HSV
    lower_black = np.array([4, 4, 4])
    upper_black = np.array([180, 255, 50])
    
    # Buat mask untuk area hitam
    mask = cv2.inRange(hsv_image, lower_black, upper_black)
    
    # Ubah area selain hitam menjadi putih
    binary_image = cv2.bitwise_not(mask)
    
    # Temukan kontur
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Temukan kotak pembatas untuk setiap kontur
        char_image = binary_image[y:y+h, x:x+w]  # Potong gambar berdasarkan kotak pembatas
        char_images.append(char_image)
    
    return char_images, contours

# Deteksi spasi antar karakter
def detect_spaces(contours):
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])  # Urutkan kontur berdasarkan posisi x
    spaces = []
    for i in range(1, len(contours)):
        x_prev, _, w_prev, _ = cv2.boundingRect(contours[i - 1])
        x_curr, _, _, _ = cv2.boundingRect(contours[i])
        space_width = x_curr - (x_prev + w_prev)
        if space_width > 20:  # Ambang batas untuk menentukan spasi (sesuaikan sesuai kebutuhan)
            spaces.append(space_width)
    return spaces

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

# Streamlit app
st.title("Aksara Jawa Detection")

# Camera input
image_data = st.camera_input("Take a picture")

if image_data is not None:
    # Load the image
    image = Image.open(io.BytesIO(image_data.getvalue()))
    
    # Adjust the image (directly use the image for processing)
    adjusted_image = image  # Removed the adjust_image call
    
    # Display the adjusted image
    st.image(adjusted_image, caption='Adjusted Image', use_column_width=True)
    
    # Segment characters from the adjusted image
    segmented_chars, contours = preprocess_javanese_script(adjusted_image)
    
    if segmented_chars:
        # Detect spaces
        spaces = detect_spaces(contours)
        
        # Predict each character and form words
        recognized_text = ""
        word = ""
        for i, char_image in enumerate(segmented_chars):
            char_image_pil = Image.fromarray(char_image)
            char_class = predict(char_image_pil, model, transform)
            word += char_class
            if i < len(spaces) and spaces[i] > 20:  # Insert space if detected
                recognized_text += word + " "
                word = ""
        
        # Add the last word
        recognized_text += word
        
        # Display the recognized text
        st.write(f"Recognized Text: {recognized_text.strip()}")
        st.write(f"Jumlah spasi yang terdeteksi: {len(spaces)}")
        
        # Display each segmented character with its prediction
        st.write("Segmented Characters and Predictions:")
        for i, char_image in enumerate(segmented_chars):
            char_image_pil = Image.fromarray(char_image)
            char_class = predict(char_image_pil, model, transform)
            st.image(char_image, caption=f'Character {i}: {char_class}', use_column_width=True)
    else:
        st.write("No characters detected.")
