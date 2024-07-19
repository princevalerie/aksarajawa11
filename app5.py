import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import matplotlib.pyplot as plt

# Fungsi untuk menampilkan gambar
def show_image(image, title=''):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Fungsi untuk masking gambar
def mask_image(image):
    image_np = np.array(image)
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 135])
    mask = cv2.inRange(hsv_image, lower_black, upper_black)
    binary_image = cv2.bitwise_not(mask)
    return Image.fromarray(binary_image)  # Convert to PIL image

# Fungsi untuk preprocessing gambar dan segmentasi karakter
def preprocess_and_segment(image):
    image_np = np.array(image.convert('RGB'))
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_image = eroded_image[y:y+h, x:x+w]
        char_image_negated = cv2.bitwise_not(char_image)
        border_size = 10
        char_image_with_border = cv2.copyMakeBorder(
            char_image_negated,
            border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT,
            value=255
        )
        char_images.append((char_image_with_border, x, y, w, h))
    char_images = sorted(char_images, key=lambda x: x[1])
    return char_images, contours

# Deteksi spasi antar karakter dengan ambang batas dinamis
def detect_spaces(contours, min_space_width):
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    spaces = []
    positions = []
    for i in range(1, len(contours)):
        x_prev, _, w_prev, _ = cv2.boundingRect(contours[i - 1])
        x_curr, _, _, _ = cv2.boundingRect(contours[i])
        space_width = x_curr - (x_prev + w_prev)
        if space_width > min_space_width:
            spaces.append(space_width)
            positions.append((x_prev + w_prev, x_curr))
    return spaces, positions

# Fungsi untuk menghitung jumlah karakter di sebelah kiri setiap spasi
def count_chars_left_of_spaces(positions, contours):
    counts = []
    for (x1, x2) in positions:
        count = sum(1 for contour in contours if cv2.boundingRect(contour)[0] < x1)
        counts.append(count)
    return counts

# Load the trained model
class_names = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 
               'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=20, bias=True)
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
    image = Image.open(io.BytesIO(image_data.getvalue()))
    
    # Apply masking
    masked_image = mask_image(image)
    
    # Display the masked image
    st.image(masked_image, caption='Masked Image', use_column_width=True)
    
    # Segment characters from the masked image
    segmented_chars, contours = preprocess_and_segment(masked_image)
    
    # Set the minimum space width you want to detect
    min_space_width = 130  # Ubah nilai ini sesuai kebutuhan
    
    # Detect spaces
    spaces, positions = detect_spaces(contours, min_space_width)
    
    # Count characters left of each space
    char_counts_left_of_spaces = count_chars_left_of_spaces(positions, contours)
    
    # Display results
    st.write(f"Jumlah spasi yang terdeteksi: {len(spaces)}")
    st.write(f"Posisi spasi dan jumlah karakter di sebelah kirinya:")
    for i, (pos, count) in enumerate(zip(positions, char_counts_left_of_spaces)):
        st.write(f"Spasi {i + 1}: Dari {pos[0]} ke {pos[1]}, Jumlah karakter di sebelah kiri: {count}")
    
    # Visualize the result
    image_np = np.array(masked_image)
    for (x1, x2) in positions:
        cv2.rectangle(image_np, (x1, 0), (x2, image_np.shape[0]), (0, 255, 0), 2)
    
    st.image(image_np, caption='Detected Spaces', use_column_width=True)
    
    # Recognize and construct the text
    recognized_text = ""
    for char_image, x, y, w, h in segmented_chars:
        char_pil = Image.fromarray(char_image)
        predicted_char = predict(char_pil, model, transform)
        recognized_text += predicted_char + " "
    
    # Display the recognized text
    st.write(f"Recognized Text: {recognized_text.strip()}")
