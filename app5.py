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

# Define class names
class_names = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 
               'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

# Function to display image
def show_image(image, title=''):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to mask image
def mask_image(image):
    image_np = np.array(image)
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 135])
    mask = cv2.inRange(hsv_image, lower_black, upper_black)
    binary_image = cv2.bitwise_not(mask)
    return Image.fromarray(binary_image)  # Convert to PIL image

# Function to check if a character image is valid based on the proportion of black pixels
def is_valid_character(char_image):
    total_pixels = char_image.size
    black_pixels = np.sum(char_image == 0)
    black_ratio = black_pixels / total_pixels
    return 0.05 <= black_ratio <= 0.90

# Function for image preprocessing and character segmentation
def preprocess_and_segment(image):
    # Convert PIL image to numpy array (RGB)
    image_np = np.array(image.convert('RGB'))
    # Convert image from RGB to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Thresholding to get binary image
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Use morphological operations to thin characters
    kernel = np.ones((1, 1), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    # Find contours in binary image
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
        if is_valid_character(char_image_with_border):
            char_images.append((char_image_with_border, x))
    char_images = sorted(char_images, key=lambda x: x[1])
    return char_images, contours

# Function to detect spaces between characters based on Q3
def detect_spaces(contours):
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    spaces = []
    positions = []
    space_widths = []
    for i in range(1, len(contours)):
        x_prev, _, w_prev, _ = cv2.boundingRect(contours[i - 1])
        x_curr, _, _, _ = cv2.boundingRect(contours[i])
        space_width = x_curr - (x_prev + w_prev)
        space_widths.append(space_width)

    # Calculate Q3 (third quartile)
    if len(space_widths) > 0:
        q3 = np.percentile(space_widths, 75)
        for i in range(len(space_widths)):
            if space_widths[i] > q3:
                x_prev, _, w_prev, _ = cv2.boundingRect(contours[i])
                x_curr, _, _, _ = cv2.boundingRect(contours[i + 1])
                spaces.append(space_widths[i])
                positions.append((x_prev + w_prev, x_curr))

    return spaces, positions

# Function to count characters left of spaces
def count_chars_left_of_spaces(positions, contours):
    counts = []
    for (x1, x2) in positions:
        count = sum(1 for contour in contours if cv2.boundingRect(contour)[0] < x1)
        counts.append(count)
    return counts

# Function to add spaces to characters
def add_spaces_to_chars(segmented_chars, positions, char_counts_left_of_spaces):
    result = []
    char_index = 0
    for i, (char_image, x) in enumerate(segmented_chars):
        result.append((char_image, x))
        while char_index < len(char_counts_left_of_spaces) and i + 1 == char_counts_left_of_spaces[char_index]:
            # Ensure the index is within the valid range
            if char_index < len(positions):
                space_width = positions[char_index][1] - positions[char_index][0]
                space_image = np.ones((char_image.shape[0], space_width), dtype=np.uint8) * 255
                result.append((space_image, x))
            char_index += 1
    return result

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
    
    # Detect spaces
    spaces, positions = detect_spaces(contours)
    
    # Count characters left of each space
    char_counts_left_of_spaces = count_chars_left_of_spaces(positions, contours)
    
    # Add spaces to characters
    segmented_chars_with_spaces = add_spaces_to_chars(segmented_chars, positions, char_counts_left_of_spaces)
    
    if segmented_chars:
        # Predict each character and form words
        recognized_text = ""
        word = ""
        for i, (char_image, _) in enumerate(segmented_chars_with_spaces):
            # Ensure the index is within the valid range
            if i < len(positions) and char_image.shape[1] > 1 and char_image.shape[1] == positions[i][1] - positions[i][0]:
                recognized_text += word + " "
                word = ""
            else:
                char_image_pil = Image.fromarray(char_image)
                char_class = predict(char_image_pil, model, transform)
                word += char_class

        recognized_text += word
        st.write(f"Recognized Text: {recognized_text.strip()}")
        st.write(f"Jumlah spasi yang terdeteksi: {len(spaces)}")
        
        st.write("Segmented Characters and Predictions:")
        for i, (char_image, _) in enumerate(segmented_chars):
            char_image_pil = Image.fromarray(char_image)
            char_class = predict(char_image_pil, model, transform)
            st.image(char_image, caption=f'Character {i}: {char_class}', use_column_width=True)
    else:
        st.write("No characters detected.")
    
    # Visualize detected spaces
    image_np = np.array(masked_image)
    for (x1, x2) in positions:
        cv2.rectangle(image_np, (x1, 0), (x2, image_np.shape[0]), (0, 255, 0), 2)
    
    st.image(image_np, caption='Detected Spaces', use_column_width=True)
