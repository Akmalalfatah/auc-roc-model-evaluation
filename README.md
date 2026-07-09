# Binary Classification Evaluation: Calculating AUC-ROC Performance

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZZFZBFNGFIQpZ-Gv591ZoYa7ie0UtlEY?usp=sharing)

A dedicated machine learning valuation repository demonstrating how to implement, optimize, and evaluate binary classification thresholds. This project uses a Support Vector Classifier (SVC) to construct a Receiver Operating Characteristic (ROC) curve and calculate its Area Under the Curve (AUC) metric.

---

## 📈 Model Performance & Diagnostic Metrics

The Support Vector Machine (SVM) pipeline was trained on standard feature matrices and comprehensively validated over key statistical performance boundaries: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **AUC-ROC**.

### Core Performance Summary:
* **Classification Accuracy:** High-accuracy boundary separation on the test partition.
* **AUC-ROC Integrity:** The calculated Area Under the Curve (AUC) demonstrates robust discriminative ability, confirming excellent true-positive vs. false-positive separation across all decision thresholds.
* **Error Analysis:** Generates a detailed confusion matrix and classification report highlighting minimized false negatives and false positives.

---

## 🛠️ Pipeline Architecture & Workflow

### 1. Data Preprocessing & Partitioning
* **Feature Scaling:** Employs a standard feature transformation layer (`StandardScaler`) to eliminate variance discrepancies and balance column weights before model exposure.
* **Stratified Split:** Partitions datasets into clean training and testing matrices via `train_test_split` to prevent evaluation bias.

### 2. Modeling & Decision Optimization
* **Model Selection:** Implements a robust Support Vector Classifier (`SVC`) optimized for continuous high-dimensional hyperplanes.
* **Threshold Diagnostics:** Extracts continuous probability estimates and decision functions to feed the ROC evaluation engine.

---

## 📊 Analytical Reports Generated

The script automatically executes a terminal reporting suite and saves persistent visualization artifacts to the workspace directory:
1. **Full Classification Report:** Comprehensive output print containing exact Precision, Recall, and F1-Scores for both true/false binary classes.
2. **Confusion Matrix Mapping:** Tabulated output showing strict True Negative, False Positive, False Negative, and True Positive distributions.
3. **ROC Curve Graphing:** Plotting of the True Positive Rate (TPR) against the False Positive Rate (FPR), explicitly computing and printing the integrated final AUC value.

---

## 🚀 How to Run the Environment

### Prerequisites
Install the required machine learning, statistics, and plotting dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
