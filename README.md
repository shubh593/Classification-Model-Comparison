# 🔬 Classification Model Comparison: Decision Tree vs. Random Forest

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

An empirical comparative study evaluating the predictive performance and interpretability of tree-based machine learning algorithms (**Decision Tree** vs. **Random Forest**) on the **Breast Cancer Wisconsin (Diagnostic)** dataset.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Methodology & Pipeline](#-methodology--pipeline)
- [Models Compared](#-models-compared)
- [Performance Benchmarking](#-performance-benchmarking)
- [Interpretability & Tree Visualization](#-interpretability--tree-visualization)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Key Takeaways](#-key-takeaways)
- [License](#-license)

---

## 📖 Overview

The objective of this project is to benchmark and analyze classification models on a high-dimensional medical diagnostic dataset. We explore the trade-offs between a single interpretable **Decision Tree** and an ensemble **Random Forest Classifier**, focusing on:
- Classification accuracy and precision/recall trade-offs.
- The reduction of variance and false-negative predictions through bootstrap aggregation (bagging).
- Visualizing decision logic for clinical interpretability.

---

## 📊 Dataset

The experiment utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset** provided by `scikit-learn`:

- **Samples**: 569 instances
- **Features**: 30 numeric real-valued features computed from fine needle aspirate (FNA) digitized images of breast masses.
  - Metrics include: `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, and `fractal dimension`.
  - Each metric is captured across 3 statistical representations: `mean`, `standard error (error)`, and `worst` (largest value).
- **Target Variable**: Binary classification:
  - `0`: Malignant (212 instances, ~37.3%)
  - `1`: Benign (357 instances, ~62.7%)
- **Data Split**: 70% Training (398 samples) / 30% Testing (171 samples) with fixed `random_state=42`.

---

## ⚙️ Methodology & Pipeline

1. **Data Ingestion & Structuring**:
   - Extracted features and targets via `sklearn.datasets.load_breast_cancer`.
   - Converted into a structured `pandas.DataFrame` for inspection (`df.head()`, `df.info()`).
2. **Train/Test Splitting**:
   - Stratified random partition with `test_size=0.3` and `random_state=42`.
3. **Model Training**:
   - Fit `DecisionTreeClassifier` (baseline).
   - Fit `RandomForestClassifier` (100 estimators).
4. **Evaluation**:
   - Accuracy score computation.
   - Comprehensive classification report (Precision, Recall, F1-Score, Support).
   - Confusion matrix analysis.
5. **Model Visualization**:
   - Visualizing tree decision paths using `sklearn.tree.plot_tree`.

---

## 🤖 Models Compared

| Algorithm | Hyperparameters | Characteristics |
| :--- | :--- | :--- |
| **Decision Tree Classifier** | `random_state=42` | Single estimator, high interpretability, prone to high variance and overfitting on unpruned depth. |
| **Random Forest Classifier** | `n_estimators=100`, `random_state=42` | Ensemble of 100 bootstrapped decision trees with random feature subsets; significantly reduces variance and stabilizes predictions. |

---

## 📈 Performance Benchmarking

On the held-out test set ($N = 171$):

### Summary Comparison Table

| Metric | Decision Tree | Random Forest (100 Trees) | Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **94.15%** | **97.08%** | **+2.93%** |
| **Precision (Malignant - 0)** | 0.90 | **0.98** | +0.08 |
| **Recall (Malignant - 0)** | **0.95** (60/63) | 0.94 (59/63) | -0.01 |
| **F1-Score (Malignant - 0)** | 0.92 | **0.96** | +0.04 |
| **Precision (Benign - 1)** | 0.97 | 0.96 | -0.01 |
| **Recall (Benign - 1)** | 0.94 (101/108) | **0.99** (107/108) | **+0.05** |
| **F1-Score (Benign - 1)** | 0.95 | **0.98** | +0.03 |
| **Macro Average F1** | 0.94 | **0.97** | +0.03 |
| **Weighted Average F1** | 0.94 | **0.97** | +0.03 |

### Confusion Matrix Breakdown

#### Decision Tree
```text
               Predicted (0)   Predicted (1)
Actual (0)          60               3
Actual (1)           7             101
```

#### Random Forest
```text
               Predicted (0)   Predicted (1)
Actual (0)          59               4
Actual (1)           1             107
```

> **Clinical Insight**: Random Forest significantly reduced misclassifications of Benign cases (only 1 false positive vs. 7 for Decision Tree), achieving an outstanding **99% Recall** on class 1 and an overall accuracy of **97.08%**.

---

## 🌳 Interpretability & Tree Visualization

The notebook includes full tree structure plotting via `matplotlib` and `sklearn.tree.plot_tree`:
- Colored decision nodes reflecting Gini impurity and class purity.
- Split thresholds labeled by clinical feature names (`worst concave points`, `worst area`, etc.).
- Easily inspectable decision rules for domain experts and healthcare professionals.

---

## 📁 Project Structure

```bash
Classification-Model-Comparison/
├── .gitignore         # Ignores Python checkpoints, bytecode, and virtual environments
├── LICENSE            # Apache 2.0 Open Source License
├── program.ipynb      # Complete Jupyter Notebook containing workflow & plots
└── README.md          # Comprehensive project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook / JupyterLab / VS Code Jupyter extension

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Classification-Model-Comparison.git
   cd Classification-Model-Comparison
   ```

2. **Create and activate a virtual environment (optional but recommended)**:
   ```bash
   # Linux/macOS
   python -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install pandas scikit-learn matplotlib jupyter
   ```

4. **Launch the notebook**:
   ```bash
   jupyter notebook program.ipynb
   ```

---

## 💡 Key Takeaways

1. **Ensemble Superiority**: Combining multiple decorrelated trees via bagging in Random Forest achieves **97.08% accuracy**, outperforming the single Decision Tree (**94.15%**) by mitigating single-tree variance.
2. **Feature Importance in Diagnosis**: Key morphological features such as `worst concave points`, `worst perimeter`, and `mean concave points` serve as top discriminators in separating malignant and benign tumors.
3. **Interpretability vs. Performance**: While Decision Trees provide an explicit path of binary if-else rules that can be audited directly, Random Forest delivers superior diagnostic reliability at the cost of slight model complexity.

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
