# Activity-Monitoring-Project
# 🤟 Sign Language Digits Recognition using PyTorch

This project is a deep learning implementation of a **Sign Language Digit Recognizer** using a fully connected neural network (NN) built in **PyTorch**. It classifies hand gesture images representing digits (0–9) from the **Sign Language Digits Dataset**.

---

## 📁 Dataset

We use the [Sign Language Digits Dataset](https://www.kaggle.com/datamunge/sign-language-mnist) which contains images of hands representing digits 0 through 9. The images are stored in subfolders named `0`, `1`, ..., `9`.

### 📂 Folder Structure

---

## 🧠 Model Architecture

A simple fully connected (dense) neural network is used:
- **Input Layer**: 64×64 RGB image flattened to 12288 features
- **Hidden Layers**:
  - FC1: 12288 → 256 + ReLU
  - FC2: 256 → 64 + ReLU
- **Output Layer**: 64 → 10 (no activation, raw logits for CrossEntropyLoss)

---

## ⚙️ Features

- Custom `Dataset` class for loading images and labels
- Train/Test split: 70% training, 30% testing
- Uses `DataLoader` for batching and shuffling
- Accuracy checking on both training and test sets
- Written in modular and clean Python code using PyTorch

---

## 📦 Requirements

- Python 3.7+
- PyTorch
- torchvision
- PIL
- numpy

Install them via pip:
pip install torch torchvision pillow
Run the training script:


python sign_language_digits.py
This will:

Load and preprocess the dataset

Train the neural network over 6 epochs

Display accuracy for both train and test sets (on one batch)

📊 Accuracy (Example Output)
Epoch: 0
Epoch: 1
...
Epoch: 5

Test accuracy: 
Got 53 / 64 with accuracy 82.81%
Train accuracy: 
Got 50 / 64 with accuracy 78.12%
Note: Only one batch is evaluated in the accuracy function for demonstration.






