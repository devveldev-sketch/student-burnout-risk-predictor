# 🎓 Student Burnout Risk Predictor

AI-based system to predict:
- Burnout Risk (Low / Medium / High)
- Dropout Probability (%)
- Disengagement Level

---

## 📌 Problem Statement
Students face increasing academic pressure, leading to burnout, dropout, and disengagement.  
This project aims to **predict early risk signals** using behavioral and academic data.

---

## 📊 Dataset & Features
Dataset includes behavioral and academic features:

- Attendance (%)
- GPA
- Study Hours
- Sleep Hours
- Screen Time
- Assignment Delay
- Participation Score
- Study Consistency
- GPA Trend
- Attendance Trend

📁 Dataset: `student_burnout_data.csv`

---

## ⚙️ Approach & Methodology

1. Data preprocessing (scaling + encoding)
2. Feature engineering (trend + behavioral patterns)
3. Model training using XGBoost
4. Multi-output predictions:
   - Classification (Burnout, Disengagement)
   - Regression (Dropout %)
5. SHAP for explainability
6. Dashboard visualization

---

## 🤖 Models Used

- XGBoost Classifier → Burnout
- XGBoost Classifier → Disengagement
- XGBoost Regressor → Dropout %

---

## 🧠 Feature Engineering

- Converted trends into numerical signals (-1, 0, +1)
- Derived behavioral indicators:
  - Low sleep → higher burnout
  - High delay → dropout risk
- Normalized inputs for model consistency

---

## 📈 Evaluation Metrics

- Accuracy: 88%
- Precision: 88.11%
- Recall: 88%
- R² Score (Dropout): 0.59

---

## 🧠 Behavioural Insights

- Sleep is the strongest burnout reducer
- Assignment delay increases dropout risk
- Participation improves engagement
- Study consistency improves academic stability

---

## 💡 Output Explanation

The system outputs:
- Burnout level + confidence
- Dropout probability %
- Disengagement level
- SHAP explanation (feature impact)
- Recommendations

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python app.py
