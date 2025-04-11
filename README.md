# Image Embedding
# Image Embedding with Deep Neural Network Architectures

**Steps through run the code**

1. PATH is the path where this .ipynb should be placed. In this code PATH = '/content/drive/MyDrive/Colab Notebooks'
2. There should be folder named '/embedding-logs' in the PATH folder that would contain the log files for embedding.
3. Xception Model [1] has been used to extract the features and classification.
4. Bangla Handwritten Nurerals Dataset has been used in this example. But any other image dataset of which the labels are separated by different folders can be visualized with this script. 
5. The image dimension is kept (3, 28, 28). 
6. After run the TensorBoard, select "Projector" option to visualize the embedding. 

# 3D Visualization of Embedding
![3D Visualization](https://user-images.githubusercontent.com/9729244/120443913-ef33bf00-c3a8-11eb-996f-256aec4762e7.PNG)


# 2D Visualization of Embedding
![2D Visualization](https://user-images.githubusercontent.com/9729244/120444041-0f637e00-c3a9-11eb-9259-f1c16c5f96ce.PNG)


# Image Embedding with Classification

This project implements a deep learning pipeline for generating image embeddings using pre-trained convolutional neural networks (CNNs), and visualizing them using TensorBoard. It also includes a simple classifier built on top of the extracted embeddings for supervised classification tasks.

## 📌 Features

- Uses pre-trained CNN architectures (e.g., Xception, VGG16) for feature extraction
- Visualizes 2D and 3D embeddings using TensorBoard projector
- Trains a classifier on top of the embedding layer for image classification
- Includes scripts for data preprocessing, embedding generation, training, and visualization

## 🛠 Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy, pandas, Matplotlib
- TensorBoard
- Scikit-learn

## 📁 Project Structure

```
Embedding/
├── data/                # Input images or datasets
├── embeddings/          # Generated embeddings in .tsv format
├── logs/                # TensorBoard logs for visualization
├── models/              # Saved models (optional)
├── scripts/
│   ├── extract_embeddings.py
│   ├── train_classifier.py
│   ├── visualize_tensorboard.py
├── utils/
│   ├── data_loader.py
│   ├── model_utils.py
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/CopotronicRifat/Embedding.git
cd Embedding
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset

Organize images in a folder-based structure like:

```
data/
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
├── class2/
│   ├── img3.jpg
│   ├── img4.jpg
```

### 4. Run embedding extraction

```bash
python scripts/extract_embeddings.py
```

### 5. Train the classifier

```bash
python scripts/train_classifier.py
```

### 6. Launch TensorBoard to visualize embeddings

```bash
tensorboard --logdir=logs/
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

## 🧠 Sample Use Case

This pipeline is useful for visualizing high-dimensional image data in 2D/3D space and understanding how embeddings separate based on class. It can be adapted for:
- Transfer learning
- Fine-tuning classification models
- Data exploration and label quality analysis

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Author

**S M Rafiuddin Rifat**  
PhD Student in Computer Science  
[LinkedIn](https://www.linkedin.com/in/copotronicrifat) | [Portfolio](https://copotronicrifat.github.io)


# Reference

[1] Chollet, François. "Xception: Deep learning with depthwise separable convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

