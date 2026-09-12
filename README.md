# Liver Patient Prediction & Diagnostic Dashboard

A comprehensive Data Science capstone project focusing on predicting liver disease using clinical patient records and biochemical laboratory measurements. This project includes a complete exploratory data analysis, comparative machine learning models, and an interactive diagnostic web application.

---

## 📌 Project Overview
Liver disease is a major healthcare challenge, often linked to lifestyle factors, alcohol consumption, and hepatitis. Early diagnosis significantly increases survival rates. This project leverages the **Indian Liver Patient Dataset (ILPD)** to build and compare multiple machine learning classifiers to predict whether a patient has liver disease.

### Key Objectives:
1. **Exploratory Data Analysis**: Perform detailed research on target distribution, gender splits, enzyme correlations, and outliers.
2. **Model Training & Comparison**: Implement and compare six baseline algorithms (Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and Gradient Boosting).
3. **Interactive Dashboard**: Build a lightweight Python web application to visualize results, analyze feature relationships dynamically, and offer a diagnostic tool.

---

## 📸 Dashboard Screenshots

| 1. Overview Page | 2. Dataset Analysis Page |
| :---: | :---: |
| ![Overview](Liver%20Patient%20Prediction/DS9-1.png) | ![Dataset Analysis](Liver%20Patient%20Prediction/DS9-2.png) |

| 3. Feature Relationships | 4. Model Performance & Evaluation |
| :---: | :---: |
| ![Feature Relationships](Liver%20Patient%20Prediction/DS9-3.png) | ![Model Performance](Liver%20Patient%20Prediction/DS9-4.png) |

| 5. Diagnostic Prediction Tool | 6. Diagnostic Prediction Result |
| :---: | :---: |
| ![Prediction Tool](Liver%20Patient%20Prediction/DS9-5.png) | ![Prediction Result](Liver%20Patient%20Prediction/DS9-6.png) |

| 7. About Project Page |
| :---: |
| ![About Project](Liver%20Patient%20Prediction/DS9-7.png) |

---

## 📊 Dataset Profile
The dataset contains **583 records** with **10 clinical features** and a binary target:
- **Demographics**: Age, Gender
- **Bilirubin Levels**: Total Bilirubin, Direct Bilirubin
- **Enzymes (Liver Function)**: Alkaline Phosphotase, Alamine Aminotransferase (SGPT), Aspartate Aminotransferase (SGOT)
- **Proteins**: Total Proteins, Albumin, Albumin and Globulin Ratio
- **Target Variable**: `Dataset` (Converted to: `1` = Liver Disease, `0` = Healthy)

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Hetk8406/Liver-Patient-Prediction-Data-Science-Project-9.git
   cd "Liver Patient Prediction"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**:
   ```bash
   python dashboard.py
   ```
   *The application will automatically open in your default browser at `http://127.0.0.1:8050/`.*

---

## 📁 Repository Structure
```text
├── Data/
│   └── Indian Liver Patient Dataset (ILPD).csv   # Source Dataset
├── Liver Patient Prediction/                     # Application Screenshots
│   ├── DS9-1.png
│   ├── DS9-2.png
│   ├── DS9-3.png
│   ├── DS9-4.png
│   ├── DS9-5.png
│   ├── DS9-6.png
│   └── DS9-7.png
├── Liver_Patient_Prediction.ipynb                # Capstone Project Notebook
├── dashboard.py                                  # Dash Application Code
├── requirements.txt                              # Required Libraries
└── README.md                                     # Project Documentation
```

---

## 📈 Model Performance
All classifiers were trained on stratified, scaled data:

| Classifier | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** | 72.81% | 74.04% | 95.06% | 83.24% | 75.01% |
| **Support Vector Machine** | 71.05% | 71.05% | 100.00% | 83.08% | 65.10% |
| **Random Forest** | 71.05% | 73.08% | 93.83% | 82.16% | 74.75% |
| **Logistic Regression** | 71.05% | 74.49% | 90.12% | 81.56% | 80.28% |
| **K-Nearest Neighbors** | 69.30% | 74.47% | 86.42% | 80.00% | 68.13% |
| **Decision Tree** | 62.28% | 73.17% | 74.07% | 73.62% | 53.70% |

### Key Takeaways:
- **Gradient Boosting** showed the best overall performance with an F1-score of **83.24%**.
- **Random Forest** was selected as the most stable clinical model due to its high robustness to enzyme outliers and strong recall performance.
- The dashboard automatically detects and loads the saved model (`best_model.pkl`), scaler (`scaler.pkl`), and encoder (`encoder.pkl`) if generated from notebook execution.

---

## 💻 Dashboard Features
The Dash application runs completely in a single Python file and features a clean, student-friendly responsive interface:
- **KPI Summary**: Showcases total records, patient counts, and accuracy of the best model dynamically.
- **Dataset Analysis**: Live summary statistics table, missing-value audit, correlation heatmap, and interactive distribution plots with outlier boxplots.
- **Feature Relationships**: Interactive scatter plot with Gender and Diagnosis filters.
- **Model Performance**: Full comparative metrics tables, confusion matrix heatmap, and ROC curves.
- **Diagnostic Tool**: Form to input new clinical values and calculate liver disease probability dynamically.

---

## 🎓 Developer Info
* **Project**: Capstone Data Science Project
* **Author**: Het Kikani
