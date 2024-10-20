import os
import pandas as pd
import numpy as np
import cv2  # For image processing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Specify the parent folder containing subfolders for each crop
parent_folder = 'crop_images'  # Adjust with the correct path if needed

# Helper function to retrieve all images and their corresponding crop types
def load_images_from_folder(folder_path):
    image_paths, crop_labels = [], []
    for crop_type in os.listdir(folder_path):  # Iterate over each crop folder
        crop_folder = os.path.join(folder_path, crop_type)
        if os.path.isdir(crop_folder):
            for image_file in os.listdir(crop_folder):
                image_path = os.path.join(crop_folder, image_file)
                image_paths.append(image_path)
                crop_labels.append(crop_type)  # Use folder name as label
    return image_paths, crop_labels

# Load all images and their corresponding crop types
image_paths, crop_labels = load_images_from_folder(parent_folder)

# Mock 'days_to_maturity' values (replace with your logic if available)
days_to_maturity = np.random.randint(50, 150, size=len(image_paths))  # Random days for example

# Preprocess images: resizing and normalization
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)  # Resize image
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return img_array

# Preprocess all images
X = np.array([preprocess_image(path) for path in image_paths])
y = np.array(days_to_maturity)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Predicting days to maturity
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# Predict on the test set and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f} days')

# Predict days to maturity for a new image
def predict_days_to_maturity(image_path):
    img_array = preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return int(prediction[0][0])

# Example usage: Predict for an image from the maize folder
sample_image = '/content/crop_images/maize/maize001a.jpeg'  # Update path if needed
days = predict_days_to_maturity(sample_image)
print(f'Estimated days to maturity: {days} days')