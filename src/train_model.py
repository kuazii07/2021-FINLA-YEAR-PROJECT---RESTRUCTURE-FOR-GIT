# src/train_model.py

import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

# ---------------------------
# Config
# ---------------------------
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10  # You can increase for better results
TEST_SIZE = 0.1  # 10% test split

# ---------------------------
# Load preprocessed data
# ---------------------------
print("Loading preprocessed data...")
X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int32)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)

# ---------------------------
# TensorBoard setup
# ---------------------------
NAME = f"MalwareArch1-{int(time.time())}"
log_dir = os.path.join("logs", NAME)
tensorboard = TensorBoard(log_dir=log_dir)

# ---------------------------
# Define CNN model
# ---------------------------
model = Sequential()

# 1st Conv + Pool
model.add(Conv2D(64, (5,5), input_shape=X.shape[1:], padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

# 2nd Conv + Dropout + Pool
model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

# 3rd Conv + Pool
model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(2048))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(len(np.unique(Y))))
model.add(Activation("softmax"))

# Compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# ---------------------------
# Train model
# ---------------------------
print("Starting training...")
history = model.fit(
    X_train, Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, Y_test),
    callbacks=[tensorboard]
)

# ---------------------------
# Evaluate model
# ---------------------------
print("Evaluating model on test set...")
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# ---------------------------
# Save model and test data
# ---------------------------
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "Arch1_Model.h5")
model.save(model_path)
print(f"Model saved to {model_path}")

# Save test data for evaluation script
with open("X_test.pickle", "wb") as f:
    pickle.dump(X_test, f)
with open("Y_test.pickle", "wb") as f:
    pickle.dump(Y_test, f)

print("Training completed successfully!")