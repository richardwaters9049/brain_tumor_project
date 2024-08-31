import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def load_data(data_dir):
    categories = ["no", "yes"]
    data = []
    labels = []

    for category in categories:
        path = os.path.join(data_dir, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = img_to_array(
                load_img(img_path, color_mode="grayscale", target_size=(128, 128))
            )
            data.append(img_array)
            labels.append(categories.index(category))

    return np.array(data), np.array(labels)


def preprocess_data(data_dir):
    data, labels = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test):
    os.makedirs("./data/processed", exist_ok=True)
    np.save("./data/processed/X_train.npy", X_train)
    np.save("./data/processed/X_test.npy", X_test)
    np.save("./data/processed/y_train.npy", y_train)
    np.save("./data/processed/y_test.npy", y_test)


if __name__ == "__main__":
    data_dir = "./data/raw"
    X_train, X_test, y_train, y_test = preprocess_data(data_dir)
    save_data(X_train, X_test, y_train, y_test)
    print("Data preprocessing and saving completed.")
