import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Load preprocessed data
X_train = np.load("./data/processed/X_train.npy")
X_test = np.load("./data/processed/X_test.npy")
y_train = np.load("./data/processed/y_train.npy")
y_test = np.load("./data/processed/y_test.npy")

# Convert grayscale images to RGB
X_train = np.repeat(X_train, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

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

# Define the model with VGG16 as base
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers:
    layer.trainable = False

model = Sequential(
    [
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
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
