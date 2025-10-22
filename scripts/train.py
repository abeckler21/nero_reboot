"""
Train a CNN on the Sign Language MNIST dataset and save it as models/model01.keras
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import os

# ============================================================
# Load and preprocess data
# ============================================================

train_df = pd.read_csv("images/sign_mnist_train.csv")
test_df = pd.read_csv("images/sign_mnist_test.csv")

# Extract labels and images
y_train = train_df["label"].values
X_train = train_df.drop("label", axis=1).values
y_test = test_df["label"].values
X_test = test_df.drop("label", axis=1).values

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape to 28x28x1
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels (fix skipped letters J and Z)
# Map labels 0–25 → contiguous 0–23 range
unique_labels = sorted(np.unique(y_train))
label_map = {old: new for new, old in enumerate(unique_labels)}

y_train = np.array([label_map[y] for y in y_train])
y_test = np.array([label_map[y] for y in y_test])
num_classes = len(unique_labels)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Remapped labels to 0–{num_classes-1} (24 total classes).")

# ============================================================
# Build model
# ============================================================

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# ============================================================
# Compile and train
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=128,
    verbose=1
)

# ============================================================
# Save trained model
# ============================================================

os.makedirs("models", exist_ok=True)
model.save("models/model01.keras")
print("Model saved to models/model01.keras")
