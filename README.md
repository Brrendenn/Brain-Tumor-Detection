# 🧠 Brain Tumor Detection using Deep Learning (CNN)

This project uses Convolutional Neural Networks (CNNs) to detect the presence of brain tumors from MRI images. Built with deep learning techniques, it classifies MRI scans into two categories: **Yes Tumor** and **No Tumor**.

## 📁 Dataset

We used the [BR35H Brain Tumor Detection Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle, which contains labeled MRI images of patients with and without brain tumors.

## 🔍 Project Overview

- 🧠 **Model**: CNN-based binary image classifier  
- 🖼️ **Input**: MRI image (.jpg/.png)  
- ✅ **Output**: Prediction - "Yes Tumor" or "No Tumor"  
- 📊 **Accuracy**: Achieved high accuracy through model training and validation  
- 🧪 **Libraries Used**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, etc.

## 🚀 Features

- Upload MRI images for instant tumor detection  
- Clean and simple architecture  
- Trained on a balanced dataset  

## 🏗️ Model Architecture

- 3 Convolutional layers with ReLU activation  
- MaxPooling layers to reduce dimensionality  
- Fully connected dense layers  
- Softmax/Sigmoid output for classification
