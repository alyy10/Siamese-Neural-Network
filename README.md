# Siamese Neural Network for Document Image Similarity

This project implements a **Siamese Neural Network** using **TensorFlow** and **Keras** to determine the similarity between document images, such as driving licenses, social security numbers, and other documents. The pipeline leverages data augmentation, processes images from a dataset, and visualizes similar and dissimilar images based on a distance matrix.

![image](https://github.com/user-attachments/assets/22005815-b4c2-4fb1-b4cd-3b56ff260071)


## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Step-by-Step Execution](#step-by-step-execution)
  - [Step 1: Set Up the Environment](#step-1-set-up-the-environment)
  - [Step 2: Understand the Dataset](#step-2-understand-the-dataset)
  - [Step 3: Load and Prepare Data](#step-3-load-and-prepare-data)
  - [Step 4: Build the Siamese Model](#step-4-build-the-siamese-model)
  - [Step 5: Train the Model](#step-5-train-the-model)
  - [Step 6: Test and Make Predictions](#step-6-test-and-make-predictions)
  - [Step 7: Visualize Results](#step-7-visualize-results)
  - [Step 8: Draw Conclusions](#step-8-draw-conclusions)
- [Model Architecture](#model-architecture)
- [Security Features](#security-features)
- [Performance Optimization](#performance-optimization)
- [Potential Challenges and Solutions](#potential-challenges-and-solutions)
- [Sample Visualization](#sample-visualization)


## Project Overview
This project builds a Siamese Neural Network to compare document images and identify similarities based on their embeddings. Key objectives include:
- Understanding the dataset containing images of driving licenses, social security numbers, and other documents.
- Applying data augmentation to enhance model robustness.
- Building and training a Siamese Neural Network to compute image similarity.
- Visualizing similar and dissimilar images using a distance matrix.
- Drawing insights from model predictions.

The pipeline processes a dataset of 600 training images and 150 test images, applies data augmentation, trains the model, and visualizes results using Matplotlib.

## Prerequisites
Before starting, ensure you have:
- **Python 3.8+**: Installed on your system.
- **Dataset**: Available at `E:\dataset\project-pro-siamese-nn-1` or your specified path.
- **Dependencies**: Install required packages listed in the Tech Stack.
- **Hardware**: GPU recommended for faster training (TensorFlow supports CPU/GPU).
- **Cost Awareness**: No cloud costs involved; ensure sufficient local storage for the dataset.

## Tech Stack
- **Language**: Python
- **Packages**: TensorFlow, Keras, NumPy, Matplotlib, TensorFlow Addons
- **Environment**: Local Python environment (e.g., Anaconda, virtualenv)

## Project Structure
```
siamese-nn-document-similarity/
├── data/
│   ├── project-pro-siamese-nn-1/
│   │   ├── train/ (600 images)
│   │   ├── test/ (150 images)
├── scripts/
│   ├── main.ipynb
│   ├── show.py
│   ├── utils.py
├── README.md
```

- `data/`: Contains the dataset with train (600 images) and test (150 images) folders.
- `scripts/`: Includes the Jupyter notebook (`main.ipynb`) and Python scripts (`show.py`, `utils.py`) for visualization and utility functions.
- `README.md`: This file.

## Step-by-Step Execution

### Step 1: Set Up the Environment
1. **Install Python**:
   - Download and install Python 3.8+ from `https://www.python.org`.
   - Optionally, use Anaconda for environment management.

2. **Install Dependencies**:
   - Run:
     ```bash
     pip install tensorflow==2.9.1 tensorflow-addons numpy matplotlib
     ```

3. **Verify TensorFlow Installation**:
   - Run the following in Python to check TensorFlow version and available devices:
     ```python
     import tensorflow as tf
     print(f"tensorflow: {tf.__version__}")
     print("Available devices:", *[d.name for d in tf.config.list_physical_devices()])
     ```

### Step 2: Understand the Dataset
1. **Dataset Overview**:
   - **Structure**: Contains three classes: Driving License, Social Security Number, and Other Documents.
   - **Training Set**: 600 images.
   - **Test Set**: 150 images.

2. **Data Augmentation Parameters**:
   - Define augmentation parameters for robustness:
     ```python
     DATA_AUG_PARAMS = {
         'BRIGHTNESS': 0.5,
         'HUE': 0.5,
         'CONTRAST_MIN': 0.5,
         'CONTRAST_MAX': 1.5,
         'SATURATION_MIN': 0.5,
         'SATURATION_MAX': 1.5,
         'ZOOM_FACTOR': 0.5,
         'ROTATION_FACTOR': 0.2
     }
     ```

### Step 3: Load and Prepare Data
1. **Set Random Seed**:
   - Ensure reproducibility:
     ```python
     SEED = 42
     tf.random.set_seed(SEED)
     np.random.seed(SEED)
     random.seed(SEED)
     ```

2. **Load Dataset**:
   - Use TensorFlow’s `image_dataset_from_directory` to load images:
     ```python
     train_ds = tf.keras.preprocessing.image_dataset_from_directory(
         f"{DATASET_PATH}/train",
         image_size=(IMG_SIZE, IMG_SIZE),
         batch_size=32,
         seed=SEED
     )
     test_ds = tf.keras.preprocessing.image_dataset_from_directory(
         f"{DATASET_PATH}/test",
         image_size=(IMG_SIZE, IMG_SIZE),
         batch_size=32,
         seed=SEED
     )
     ```

3. **Apply Data Augmentation**:
   - Implement augmentation using `tf.keras.Sequential`:
     ```python
     data_augmentation = tf.keras.Sequential([
         tf.keras.layers.RandomBrightness(DATA_AUG_PARAMS['BRIGHTNESS']),
         tf.keras.layers.RandomContrast((DATA_AUG_PARAMS['CONTRAST_MIN'], DATA_AUG_PARAMS['CONTRAST_MAX'])),
         tf.keras.layers.RandomZoom(DATA_AUG_PARAMS['ZOOM_FACTOR']),
         tf.keras.layers.RandomRotation(DATA_AUG_PARAMS['ROTATION_FACTOR'])
     ])
     train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
     ```

4. **Normalize Images**:
   - Scale pixel values to [0, 1]:
     ```python
     train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
     test_ds = test_ds.map(lambda x, y: (x / 255.0, y))
     ```

### Step 4: Build the Siamese Model
1. **Base Model**:
   - Use a pre-trained model (e.g., MobileNetV2) as the feature extractor:
     ```python
     base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
     base_model.trainable = False
     ```

2. **Siamese Network**:
   - Create a Siamese model to compare pairs of images:
     ```python
     def create_siamese_network():
         input_a = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
         input_b = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
         processed_a = base_model(input_a)
         processed_b = base_model(input_b)
         pooled_a = tf.keras.layers.GlobalAveragePooling2D()(processed_a)
         pooled_b = tf.keras.layers.GlobalAveragePooling2D()(processed_b)
         distance = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([pooled_a, pooled_b])
         outputs = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
         return tf.keras.Model(inputs=[input_a, input_b], outputs=outputs)
     model = create_siamese_network()
     ```

3. **Compile Model**:
   - Use binary cross-entropy loss and Adam optimizer:
     ```python
     model.compile(
         optimizer=tf.keras.optimizers.Adam(),
         loss=tf.keras.losses.BinaryCrossentropy(),
         metrics=['accuracy']
     )
     ```

### Step 5: Train the Model
1. **Prepare Training Pairs**:
   - Generate pairs of images (positive: same class, negative: different class) using a custom function (not shown in provided code but assumed in `utils.py`).

2. **Train**:
   - Train the model for 15 epochs:
     ```python
     history = model.fit(
         train_pairs, train_labels,
         validation_data=(val_pairs, val_labels),
         epochs=EPOCHS,
         batch_size=32
     )
     ```

### Step 6: Test and Make Predictions
1. **Test Dataset**:
   - Evaluate the model on the test set:
     ```python
     test_loss, test_accuracy = model.evaluate(test_pairs, test_labels)
     print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
     ```

2. **Generate Embeddings**:
   - Extract embeddings for visualization:
     ```python
     embeddings = base_model.predict(test_ds.map(lambda x, y: x))
     d2 = utils.get_squared_distance_matrix(embeddings)
     ```

### Step 7: Visualize Results
1. **Show Similar Images**:
   - Use `show.py` to visualize similar and dissimilar images:
     ```python
     show.show_similar_images(imgs, d2, num_images=10, num_pos=3, num_neg=3, figsize=(10, 20))
     ```

2. **Interpret Output**:
   - Displays a grid with reference images, their closest (positive) matches, and furthest (negative) matches based on the distance matrix.

### Step 8: Draw Conclusions
1. **Model Performance**:
   - Analyze training/validation accuracy and loss to assess model performance.
   - Check if the model correctly identifies similar documents (e.g., two driving licenses) and distinguishes dissimilar ones (e.g., driving license vs. social security number).

2. **Insights**:
   - Data augmentation improves robustness to variations in brightness, contrast, etc.
   - The Siamese network effectively learns image similarity through embeddings.
   - Potential improvements: Fine-tune the base model or increase training data.

## Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet) for feature extraction.
- **Siamese Network**: Two inputs processed by the shared base model, followed by global average pooling, distance computation (absolute difference), and a sigmoid-activated dense layer.
- **Loss Function**: Binary cross-entropy for pair-wise similarity classification.
- **Optimizer**: Adam for efficient gradient-based optimization.

## Security Features
- **Local Execution**: No cloud services involved, reducing data exposure risks.
- **Data Handling**: Ensure dataset is stored securely and not publicly accessible.
- **Dependencies**: Use trusted package versions (e.g., TensorFlow 2.9.1) to avoid vulnerabilities.

## Performance Optimization
- **GPU Acceleration**: Leverage GPU for faster training (verified via `tf.config.list_physical_devices()`).
- **Data Pipeline**: Use `tf.data.Dataset` with prefetching and batching for efficient data loading:
  ```python
  train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
  test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
  ```
- **Model Efficiency**: Freeze the base model (`trainable = False`) to reduce computation.

## Potential Challenges and Solutions
- **Overfitting**:
  - **Issue**: Model may overfit due to limited dataset size (600 train images).
  - **Solution**: Increase data augmentation intensity or use regularization (e.g., dropout).
- **Data Augmentation Artifacts**:
  - **Issue**: Excessive augmentation may distort document features.
  - **Solution**: Tune augmentation parameters (e.g., reduce `BRIGHTNESS` or `ZOOM_FACTOR`).
- **Distance Matrix Computation**:
  - **Issue**: High memory usage for large datasets in `get_squared_distance_matrix`.
  - **Solution**: Process embeddings in batches or use approximate nearest neighbor methods.
- **GPU Availability**:
  - **Issue**: Lack of GPU slows training.
  - **Solution**: Use a cloud-based GPU (e.g., Google Colab) or reduce model complexity.

## Sample Visualization
A sample visualization from `show_similar_images` displays a grid with:
- **Reference Image**: A document image (e.g., driving license).
- **Positive Matches**: Top 3 similar images (e.g., other driving licenses).
- **Negative Matches**: Top 3 dissimilar images (e.g., social security numbers).
Below is a sample configuration for a similar visualization using Matplotlib:
```python
fig, ax = plt.subplots(10, 7, figsize=(10, 20))
ax[0, 0].set_title('Reference')
for i in range(3):
    ax[0, 1 + i].set_title(f'Pos {i}')
for i in range(3):
    ax[0, 4 + i].set_title(f'Neg {i}')
```

