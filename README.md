# Flood-Prediction-ML
A machine learning project for predicting flood probability using environmental data. The dataset is processed, cleaned, and evaluated with three models: Support Vector Classifier, Logistic Regression, and Random Forest. Includes accuracy, classification reports, and confusion matrix visualization.


# 🌊 Flood Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting the **probability of floods** using machine learning classification models. The dataset contains multiple environmental and climatic features, which are used to train and evaluate different algorithms.

The models compared in this project are:

* **Support Vector Classifier (SVC)**
* **Logistic Regression**
* **Random Forest Classifier**

The goal is to analyze which model performs better in predicting flood risks.

---

## 📂 Dataset

* The dataset used: `flood.csv`
* Target column: **FloodProbability**
* Label encoding:

  * Values `>= 0.50` → **1 (High Flood Risk)**
  * Values `< 0.50` → **0 (Low Flood Risk)**

---

## ⚙️ Steps in the Project

1. **Load Dataset** → Read CSV file using `pandas`.
2. **Data Cleaning** → Checked missing values, dataset info, and statistical summary.
3. **Feature Engineering** → Separated features `X` and target variable `y`.
4. **Train-Test Split** → 80% training and 20% testing data.
5. **Model Training** → Implemented 3 models:

   * **SVC with StandardScaler (Pipeline)**
   * **Logistic Regression**
   * **Random Forest Classifier**
6. **Evaluation** →

   * Accuracy Score
   * Classification Report (Precision, Recall, F1-Score)
   * Confusion Matrix (with heatmap visualization using `seaborn`)

---

## 📊 Results

| Model                               | Evaluation Metrics                                |
| ----------------------------------- | ------------------------------------------------- |
| **Support Vector Classifier (SVC)** | Accuracy:0.9912, Classification Report:
                precision    recall  f1-score   support

           0       0.99      0.99      0.99      4821
           1       0.99      0.99      0.99      5179

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000, 


| **Logistic Regression**             | Accuracy, Classification Report, Confusion Matrix |
| **Random Forest Classifier**        | Accuracy, Classification Report, Confusion Matrix |

*(Replace with actual results once you run all models.)*

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy** (Data Handling)
* **Scikit-Learn** (ML Models & Evaluation)
* **Matplotlib, Seaborn** (Visualization)

---

## 🚀 How to Run the Project

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/flood-prediction.git
   ```
2. Navigate to the project folder:

   ```bash
   cd flood-prediction
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook:

   ```bash
   jupyter notebook "Flood Prediction.ipynb"
   ```

---

## 📌 Future Improvements

* Apply hyperparameter tuning (GridSearchCV, RandomizedSearchCV).
* Add more advanced models (XGBoost, Gradient Boosting).
* Deploy the model with a web app (Flask/Django/Streamlit).

---

## 📜 License

This project is licensed under the MIT License.
