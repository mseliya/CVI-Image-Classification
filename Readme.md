# 📘 CVI – Image Classification (Assignment 2)

**Student:** Seliya Marahatta  
**Course:** CVI – Computer Vision  
**Assignment:** 2

This repository contains my solutions for Assignment 2 of the CVI course.  
The assignment covers two image classification tasks using classical machine learning methods and neural networks.

---

## 🐱🐶 Q1 – Cat vs Dog Classification

### 📌 Overview

In Q1, I built a classifier to distinguish between cat and dog images.  
The dataset contained approximately **1000 cat images** and **1000 dog images**, divided into train/test folders (not included in this repository due to size limits).

### 🧪 Methods Attempted

I experimented with several ML techniques taught in the course:

- **Logistic Regression**
- **MLP Neural Network (raw pixels)**
- **MLP Neural Network (with HOG features)**
- **SVC (Support Vector Classifier) with HOG features** ← **Best performing model**

### ⭐ Final Method Selected

I selected **SVC + HOG features** because it performed the best and produced stable, consistent results.

### 📊 Final Results

- **Validation Accuracy:** ~80%
- **Test Accuracy:** ~80%
- Correctly classified most internet images as well.

### ▶ How to Run Q1

In the `Q1` folder:

```bash
python Q1.py

Q1/
 ├── train/
 │     ├── cat/
 │     └── dog/
 ├── test/
 │     ├── cat/
 │     └── dog/
 └── internet/

```

## 🔢 Q2 – MNIST Handwritten Digit Classification

### 📌 Overview

Q2 uses the MNIST dataset, provided as CSV files containing flattened **28×28 pixel grayscale images**.
The goal was to classify digits from **0 to 9** and achieve at least **90% accuracy**.

---

### 🧪 Methods Used

I trained and compared the following models:

- **Logistic Regression**
- **MLP Neural Network**

---

### 📊 Final Accuracy

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | 92.6%     |
| MLP Neural Network  | **98.1%** |

Both models exceeded the 90% requirement, with the **MLP neural network performing the best**.

---

### ▶ How to Run Q2

Inside the `Q2` folder, run:

```bash
python Q2.py

Q2/
 ├── mnist_train.csv
 └── mnist_test.csv

```
