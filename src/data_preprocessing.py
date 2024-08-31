import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    print(f"Loading data from: {data_dir}")  # Debug statement
    categories = ["no", "yes"]
    data = []

    for category in categories:
        path = os.path.join(data_dir, category)
        print(f"Processing category: {category}, path: {path}")  # Debug statement
        class_num = categories.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_img = cv2.resize(img_array, (128, 128))
                data.append([resized_img, class_num])
            except Exception as e:
                print(f"Error loading image: {e}")
                pass

    return data


def preprocess_data(data_dir):
    data = load_data(data_dir)
    np.random.shuffle(data)

    X = []
    y = []

    for features, label in data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, 128, 128, 1) / 255.0  # Normalize the data
    y = np.array(y)

    print(f"Shapes: X={X.shape}, y={y.shape}")  # Debugging shapes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data_dir = "./data/raw"
    X_train, X_test, y_train, y_test = preprocess_data(data_dir)

    # Debugging
    print(f"Saving data: X_train.shape={X_train.shape}, X_test.shape={X_test.shape}")

    np.save("./data/processed/X_train.npy", X_train)
    np.save("./data/processed/X_test.npy", X_test)
    np.save("./data/processed/y_train.npy", y_train)
    np.save("./data/processed/y_test.npy", y_test)

    print("Data saved successfully.")


if __name__ == "__main__":
    data_dir = "./data/raw"
    X_train, X_test, y_train, y_test = preprocess_data(data_dir)
    np.save("./data/processed/X_train.npy", X_train)
    np.save("./data/processed/X_test.npy", X_test)
    np.save("./data/processed/y_train.npy", y_train)
    np.save("./data/processed/y_test.npy", y_test)
