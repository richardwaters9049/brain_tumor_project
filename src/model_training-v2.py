import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

import os

# Set global parameters
INPUT_SHAPE = (128, 128, 3)
BATCH_SIZE = 256
EPOCHS = 10
DATA_DIR = "./brain_tumor_project/data/processed"
MODEL_SAVE_PATH = "./brain_tumor_project/models/best_model.h5"

def load_data(data_dir):
    """Load and preprocess the data from the specified directory."""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    # Convert grayscale images to RGB
    X_train = np.repeat(X_train, 3, axis=-1)
    X_test = np.repeat(X_test, 3, axis=-1)

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test

def augment_data(X, y):
    """Create augmented data using ImageDataGenerator."""
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        fill_mode="nearest",
        brightness_range=[0.8, 1.2],
        preprocessing_function=lambda x: x + np.random.normal(loc=0.0, scale=0.1, size=x.shape)  # Adding noise
    )

    # Generate additional data by applying transformations and duplicating the data
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        augmented_image = datagen.random_transform(X[i])
        X_augmented.append(augmented_image)
        y_augmented.append(y[i])

    # Convert to numpy arrays and concatenate with original data
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)

    X_combined = np.concatenate((X, X_augmented), axis=0)
    y_combined = np.concatenate((y, y_augmented), axis=0)

    # Shuffle the combined data
    return shuffle(X_combined, y_combined)

def build_model(input_shape):
    """Build and compile the EfficientNetB0-based model."""
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)

    # Unfreeze the top layers of the base model for fine-tuning
    for layer in base_model.layers:  # Unfreeze the last 20 layers
        layer.trainable = True

    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.85),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, model_save_path):
    """Train the model with augmented data and specified callbacks."""
    # Callbacks for learning rate reduction, early stopping, and model checkpointing
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True)

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[lr_scheduler, early_stopping, model_checkpoint],
    )

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print the accuracy."""
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    return test_acc

def main():
    # Load and augment data
    X_train, X_test, y_train, y_test = load_data(DATA_DIR)
    X_train, y_train = augment_data(X_train, y_train)

    # Build the model
    model = build_model(INPUT_SHAPE)

    # Train the model
    train_model(model, X_train, y_train, X_test, y_test, BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()