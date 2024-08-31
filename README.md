# Brain Tumour Classification Project

## Overview

This project is focused on the classification of brain tumour images using Convolutional Neural Networks (CNNs). The primary objective is to accurately distinguish between MRI images that show the presence of brain tumours and those that do not, leveraging deep learning techniques. The model is constructed using TensorFlow and Keras, incorporating a pre-trained VGG16 architecture for effective feature extraction and classification. This project highlights the potential of deep learning in medical image analysis, particularly in the early detection and diagnosis of brain tumours.

## Project Structure

- **`src/`**: Contains all the scripts necessary for running the project.
  - **`data_preprocessing.py`**: Script for loading, preprocessing, and preparing the dataset.
  - **`model_creation.py`**: Script for constructing the CNN model.
  - **`model_training.py`**: Script for training the model and saving results.
- **`data/`**: Directory for storing the dataset.
  - **`raw/`**: Contains the original dataset in its unprocessed form.
  - **`processed/`**: Contains preprocessed data in numpy format (`X.npy`, `y.npy`).
- **`results/`**: Stores the output of the model training, including accuracy and loss metrics, as well as any model checkpoints.
- **`README.md`**: Overview and instructions for the project.
- **`requirements.txt`**: List of Python dependencies required to run the project.

## Dataset

### Origin

The dataset used in this project is sourced from the **Kaggle Brain MRI Images Dataset**, which is a publicly available dataset that consists of labelled MRI images of human brain scans. This dataset was selected due to its relevance, size, and the quality of the images, all of which are crucial for training a reliable model.

### Structure

- **Categories**:
  - **Tumour**: Images that indicate the presence of a brain tumour.
  - **No Tumour**: Images that do not show any signs of a brain tumour.
- **Format**:
  - The images are provided in grayscale, simplifying the processing pipeline and reducing computational overhead.
  - Each image is resized to 128x128 pixels, ensuring uniformity across the dataset and compatibility with the input layer of the CNN model.

### Utilisation

The dataset is divided into two main parts:

1. **Training Set**:

   - Comprising 80% of the dataset.
   - Used to train the CNN model, allowing it to learn the distinguishing features of tumour-positive and tumour-negative images.

2. **Testing Set**:
   - Comprising 20% of the dataset.
   - Used to evaluate the model’s performance on unseen data, providing an unbiased estimate of its accuracy and generalisation capability.

### Preprocessing Steps

- **Grayscale Conversion**: All images are converted to grayscale to reduce dimensionality while retaining important features.
- **Resizing**: Images are resized to 128x128 pixels to ensure consistency across the dataset.
- **Normalisation**: Pixel values are normalised to the range [0, 1], which helps improve the efficiency and stability of the training process.
- **Data Augmentation**: Techniques like random rotation, shifting, and flipping are applied to the training data to artificially increase the dataset size, thereby improving model robustness and reducing the risk of overfitting.

## Model Creation

The model architecture is based on the VGG16 CNN, a well-established model in the field of image classification. VGG16 is renowned for its deep architecture, which allows it to capture complex features in images.

### Model Architecture

1. **Base Model**:

   - **VGG16**: The VGG16 model pre-trained on the ImageNet dataset as the base. The top layers of VGG16 are removed to allow for customisation specific to our binary classification task.

2. **Custom Layers**:
   - **Flatten Layer**: Converts the 3D output of VGG16 into a 1D vector.
   - **Fully Connected Layers**: Additional dense layers are added to adapt the model to the specific task of brain tumour classification.
   - **Dropout Layers**: Implemented to reduce overfitting by randomly dropping units during training.
   - **Output Layer**: A dense layer with a softmax activation function is used to classify the images into tumour or no-tumour categories.

### Compilation

- **Optimizer**: The Adam optimizer with a learning rate of `0.0001` is used for training. Adam is chosen for its ability to adaptively adjust the learning rate, making the training process more efficient.
- **Loss Function**: Categorical cross-entropy is used as the loss function, suitable for binary classification tasks.
- **Metrics**: Accuracy is used as the primary metric to evaluate model performance.

## Model Training

### Data Splitting

The dataset is split into training and testing sets, with 80% of the data used for training and 20% reserved for testing. This split ensures that the model is trained on a majority of the data while having sufficient unseen data to evaluate its performance.

### Training Process

1. **Batch Size**: We use a batch size of 32, which strikes a balance between memory efficiency and training speed.
2. **Epochs**: The model is trained over 25 epochs, allowing it to learn from the data while minimising the risk of overfitting.
3. **Early Stopping**: Early stopping is implemented to halt training if the validation accuracy stops improving, preventing overfitting and conserving computational resources.

### Evaluation

After training, the model is evaluated on the test set to measure its accuracy, precision, recall, and F1-score. The final model achieves an accuracy of over 90%, indicating strong performance in distinguishing between images with and without brain tumours.

## Usage

### Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.7+**
- **TensorFlow 2.x**
- **Keras**
- **NumPy**
- **Matplotlib**
- **scikit-learn**

You can install the required dependencies using the following command:

```bash
pip3 install -r requirements.txt
```

### Running the Project

1. Clone the repository:

```bash
git clone https://github.com/richardwaters9049/brain_tumor_project
```

2. Navigate to the project directory:

```bash
cd brain-tumor_project
```

3. Run the main script:

```bash
python3 src/data_preprocessing.py
```

4. Model Creation:

```bash
python3 src/model_creation.py
```

### Results

The trained model achieves an accuracy of over 90% on the test set, demonstrating its effectiveness in distinguishing between images with and without brain tumours.

### Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests if you have suggestions for improvements or new features.

### License

This project is licensed under the MIT License.
