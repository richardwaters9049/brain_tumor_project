import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Load preprocessed data
X_train = np.load("./data/processed/X_train.npy")
X_test = np.load("./data/processed/X_test.npy")
y_train = np.load("./data/processed/y_train.npy")
y_test = np.load("./data/processed/y_test.npy")

# Verify the shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator()

# Fit parameters
train_datagen.fit(X_train)
val_datagen.fit(X_test)

# Define the model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),  # Added dropout layer
        Dense(2, activation="softmax"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

# Train the model with data augmentation
model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    validation_data=val_datagen.flow(X_test, y_test, batch_size=32),
    epochs=20,
    callbacks=[lr_scheduler],
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
test_acc_percentage = test_acc * 100
print(f"Test accuracy: {test_acc_percentage:.2f}%")
