# 🩺 Diabetes Prediction with Logistic Regression

This project uses a Logistic Regression model to predict whether a patient is diabetic based on medical attributes from the Pima Indians Diabetes Dataset.

---

## 📂 Files

- `panda.py` – Main script for data loading, training, prediction, and evaluation.
- Dataset – Automatically loaded from a GitHub-hosted CSV.

---

## 🔍 Dataset

The dataset is sourced from: Kaggle


**Features Used:**

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

**Target:**
- Outcome (1: Diabetic, 0: Not Diabetic)

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn


**🚀 How to Run**
python panda.py


📊 Output Includes
Descriptive statistics of the dataset

Accuracy score of the model

Classification report

Sample prediction for a single patient

**🧠 Model**
Model: Logistic Regression

**Solver: liblinear**

**train_test_split with 80-20 ratio**

**Max iterations: 500**

**📈 Sample Prediction Output**
Sample Prediction: Diabetic


✅ Future Improvements
* Add user input or web form for live predictions
* Add model saving/loading with joblib
* Plot ROC curve or confusion matrix for better insights
* Deploy using Streamlit or Flask

## Author
Private-Fox7

