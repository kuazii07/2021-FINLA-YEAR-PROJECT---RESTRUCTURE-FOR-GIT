# src/preprocess.py

import os
import cv2
import numpy as np
import pickle
import random

# ---------------------------
# Config
# ---------------------------
DATADIR = "data/malimg_paper_dataset_imgs"  # Dataset folder (not included in GitHub)
CATEGORIES = [
    "Adialer.C","Agent.FYI","Allaple.A","Allaple.L","Alueron.gen!J",
    "Autorun.K","BenginPe","C2LOP.gen!g","C2LOP.P","Dialplatform.B",
    "Dontovo.A","Fakerean","Instantaccess","Lolyda.AA1","Lolyda.AA2",
    "Lolyda.AA3","Lolyda.AT","Malex.gen!J","Obfuscator.AD","Rbot!gen",
    "Skintrim.N","Swizzor.gen!E","Swizzor.gen!I","VB.AT","Wintrim.BX","Yuner.A"
]
IMG_SIZE = 64

# ---------------------------
# Functions
# ---------------------------

def create_training_data():
    training_data = []
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        
        if not os.path.exists(path):
            print(f"Warning: Folder missing: {path}")
            continue
        
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img_array is None:
                    continue
                
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([resized_array, class_num])
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return training_data

# ---------------------------
# Main Processing
# ---------------------------

if __name__ == "__main__":
    print("Creating training data...")
    training_data = create_training_data()
    print(f"Total samples: {len(training_data)}")

    print("Shuffling data...")
    random.shuffle(training_data)

    X = []
    Y = []

    for features, label in training_data:
        X.append(features)
        Y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize
    Y = np.array(Y, dtype=np.int32)

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    print("Saving preprocessed data...")
    with open("X.pickle", "wb") as f:
        pickle.dump(X, f)
    
    with open("Y.pickle", "wb") as f:
        pickle.dump(Y, f)

    print("Preprocessing completed successfully!")