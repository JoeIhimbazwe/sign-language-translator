import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Load all data
print("Loading data...")

X = []  # landmarks
y = []  # sign names

data_folder = "data"

for sign_name in os.listdir(data_folder):
    sign_folder = f"{data_folder}/{sign_name}"
    
    for sample_file in os.listdir(sign_folder):
        filepath = f"{sign_folder}/{sample_file}"
        
        with open(filepath, 'r') as f:
            landmarks = json.load(f)
        
        # Flatten landmarks to a 1D array
        flat = np.array(landmarks).flatten()
        X.append(flat)
        y.append(sign_name)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples for {len(set(y))} signs: {set(y)}")

# Encode labels (hello=0, yes=1, no=2)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save label names for later
np.save("labels.npy", encoder.classes_)
print(f"Labels: {encoder.classes_}")

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Build the model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(63,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Check accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save("sign_model.h5")
print("\nModel saved as sign_model.h5")