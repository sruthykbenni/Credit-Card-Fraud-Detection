# 🔗 Credit Card Fraud Detection Using Graph Neural Networks (GNNs)

This repository contains the implementation of a Graph Neural Network (GNN)-based approach for detecting credit card fraud using transaction data. The project leverages **PyTorch Geometric** to model transactions as a graph, allowing the model to capture hidden patterns and relationships that traditional models might miss.

## 📌 Project Overview

Traditional fraud detection techniques often fail to detect collusive or structured fraud involving multiple connected transactions. To address this, we:

- Treat each transaction as a node in a graph
- Connect similar transactions using **K-Nearest Neighbors (KNN)** based on cosine similarity
- Train a **Graph Convolutional Network (GCN)** to classify transactions as fraudulent or legitimate

This approach improves the detection of rare and structurally complex fraudulent behaviors.

---

## 📁 Repository Contents

- `Credit_Card_Fraud_Detection.ipynb` – Main Jupyter notebook containing the full GCN pipeline
- `README.md` – This file
- `.gitignore` – To ignore large or unnecessary files

---

## 📥 Dataset

Due to file size restrictions, the dataset is **not included** in this repository.

Please download the dataset manually from Kaggle:

🔗 **[Credit Card Fraud Detection Dataset (2023) – Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data)**

Once downloaded, place the `creditcard_2023.csv` file in your project directory or adjust the file path in the notebook as needed.

---

## 🛠️ Dependencies

Install all required packages with:

```bash
pip install torch torchvision torchaudio torch-geometric pandas numpy scikit-learn matplotlib
```
Note: PyTorch Geometric has specific installation instructions based on your OS and CUDA version. Follow the official guide here: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

## 🚀 How to Run
Clone this repository:

bash
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
Make sure the dataset file creditcard_2023.csv is present in the working directory.

Run the notebook:

-In Jupyter Notebook / Jupyter Lab

-Or on Google Colab (recommended for easy GPU access)

## 📊 Model Performance
Architecture: 2-layer GCN using GCNConv

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

Includes: Confusion Matrix, ROC Curve, Classification Report

Our GCN model demonstrated strong capability in detecting fraudulent transactions even under severe class imbalance, outperforming traditional shallow models in both interpretability and detection sensitivity.

## 🧠 Future Scope
Integration of dynamic graphs for real-time fraud detection

Use of heterogeneous GNNs to model multiple entity types (customers, merchants, cards)

Deployment as a web API for live inference

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🙌 Acknowledgments
Dataset by Nelgiriyewithana on Kaggle: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

PyTorch Geometric for the powerful GNN framework


