# src/utils.py

import cv2
import numpy as np

IMG_SIZE = 64  # Make sure this matches your training size

CATEGORIES = [
    "Adialer.C","Agent.FYI","Allaple.A","Allaple.L","Alueron.gen!J",
    "Autorun.K","BenginPe","C2LOP.gen!g","C2LOP.P","Dialplatform.B",
    "Dontovo.A","Fakerean","Instantaccess","Lolyda.AA1","Lolyda.AA2",
    "Lolyda.AA3","Lolyda.AT","Malex.gen!J","Obfuscator.AD","Rbot!gen",
    "Skintrim.N","Swizzor.gen!E","Swizzor.gen!I","VB.AT","Wintrim.BX","Yuner.A"
]

def prepare(filepath, img_size=IMG_SIZE):
    """
    Read an image file, convert to grayscale, resize, normalize, and reshape for model input.
    
    Args:
        filepath (str): Path to the image file.
        img_size (int): Target size for the image (img_size x img_size).
    
    Returns:
        np.ndarray or None: Preprocessed image array ready for model prediction, or None if invalid.
    """
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        return None
    resized_array = cv2.resize(img_array, (img_size, img_size))
    return resized_array.reshape(-1, img_size, img_size, 1) / 255.0

def decode_prediction(pred):
    """
    Convert model prediction probabilities to class label.
    
    Args:
        pred (np.ndarray): Model prediction output (softmax probabilities).
    
    Returns:
        str: Predicted malware category.
    """
    pred_class = np.argmax(pred)
    return CATEGORIES[pred_class]
