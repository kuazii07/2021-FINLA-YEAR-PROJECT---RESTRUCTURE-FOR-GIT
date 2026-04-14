# src/evaluate.py

import os
import pickle
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import classification_report

# ---------------------------
# Config
# ---------------------------
IMG_SIZE = 64
DATA_DIR = "data"
EXTERNAL_FOLDERS = ["External_Data"]  # Put test images here

CATEGORIES = [
    "Adialer.C","Agent.FYI","Allaple.A","Allaple.L","Alueron.gen!J",
    "Autorun.K","BenginPe","C2LOP.gen!g","C2LOP.P","Dialplatform.B",
    "Dontovo.A","Fakerean","Instantaccess","Lolyda.AA1","Lolyda.AA2",
    "Lolyda.AA3","Lolyda.AT","Malex.gen!J","Obfuscator.AD","Rbot!gen",
    "Skintrim.N","Swizzor.gen!E","Swizzor.gen!I","VB.AT","Wintrim.BX","Yuner.A"
]

# ---------------------------
# Helper function
# ---------------------------
def prepare(filepath):
    """Read an image, convert to grayscale, resize, and normalize"""
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        return None
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

# ---------------------------
# Load model
# ---------------------------
print("Loading trained model...")
model = tf.keras.models.load_model("models/Arch1_Model.h5", compile=False)

# ---------------------------
# Evaluate on test data
# ---------------------------
print("Loading test data...")
X_test = pickle.load(open("X_test.pickle", "rb"))
Y_test = pickle.load(open("Y_test.pickle", "rb"))

# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n========== Classification Report ==========\n")
print(classification_report(Y_test, y_pred_classes, zero_division=0))

# ---------------------------
# External testing
# ---------------------------
print("\n========== External Testing ==========\n")

for folder in EXTERNAL_FOLDERS:
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        continue

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        prepared_img = prepare(img_path)
        if prepared_img is None:
            print(f"Skipping invalid image: {img_file}")
            continue

        prediction = model.predict(prepared_img)
        pred_class = np.argmax(prediction)
        print(f"File: {img_file} → Prediction: {CATEGORIES[pred_class]}")