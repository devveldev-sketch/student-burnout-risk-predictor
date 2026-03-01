# 🎓 Student Burnout Risk Predictor

AI-based system to predict:
- Burnout Risk (Low / Medium / High)
- Dropout Probability (%)
- Disengagement Level

---

## 🚀 Key Highlights

- Multi-task learning (classification + regression)
- Explainable AI using SHAP
- Behavioral analytics-driven insights
- Interactive dashboard for decision support

---

## 📌 Problem Statement

Students face increasing academic pressure, which can lead to burnout, disengagement, and even dropout.

Traditional systems fail to identify these risks early.

👉 This project aims to **predict early warning signals** using behavioral and academic data, enabling timely intervention.

---

## 📊 Dataset & Features

The dataset contains student behavioral and academic attributes:

- Attendance (%)
- GPA (0–10 scale)
- Study Hours (daily)
- Sleep Hours (daily)
- Screen Time (daily)
- Assignment Delay (days)
- Participation Score (1–10)
- Study Consistency (0 = irregular, 1 = consistent)
- GPA Trend (-1 ↓, 0 →, 1 ↑)
- Attendance Trend (-1 ↓, 0 →, 1 ↑)

📁 Dataset file: `student_burnout_data.csv`

---

## 🔄 Data Generation (Simulation)

Since real-world student behavioral datasets are limited, synthetic data was generated using `generate_data.py`.

- Data is created based on realistic academic patterns
- Logical relationships are embedded:
  - Low sleep + high delay → higher burnout
  - High attendance + consistency → lower risk
- Random noise is added to simulate real-world variability

👉 This ensures meaningful learning while maintaining generalization.

---

## ⚙️ Approach & Methodology

1. Data preprocessing:
   - Normalization of numerical features
   - Encoding of categorical/trend features

2. Feature engineering:
   - Behavioral pattern extraction
   - Trend conversion (-1, 0, +1)

3. Model training:
   - XGBoost for classification & regression

4. Multi-output predictions:
   - Burnout → Classification
   - Disengagement → Classification
   - Dropout → Regression

5. Rule-based enhancement:
   - Overrides extreme conditions (e.g., very low sleep)

6. Explainability:
   - SHAP used for feature contribution analysis

7. Visualization:
   - Interactive dashboard using Flask + Chart.js

---

## 🏗️ Model Architecture

```
User Input (Attendance, GPA, Sleep, etc.)
        ↓
Data Preprocessing (Scaling + Encoding)
        ↓
XGBoost Models (Pattern Learning)
        ↓
Rule-Based Layer (Critical Overrides)
        ↓
Predictions:
   - Burnout (Classification)
   - Dropout % (Regression)
   - Disengagement (Classification)
        ↓
SHAP Explainability (Feature Impact)
        ↓
Dashboard Output (Insights + Recommendations)
```

---

## 🤖 Models Used

- XGBoost Classifier → Burnout Prediction
- XGBoost Classifier → Disengagement Prediction
- XGBoost Regressor → Dropout Prediction

👉 Chosen for:
- High performance on structured data
- Ability to capture non-linear relationships
- Robustness to feature interactions

---

## 🧠 Feature Engineering

- Converted trend features into numerical signals (-1, 0, +1)
- Created behavioral relationships:
  - Low sleep → increases burnout risk
  - High assignment delay → increases dropout risk
  - High participation → reduces disengagement
- Normalized all inputs for consistent model performance

---

## 📈 Evaluation Metrics

- Accuracy: 88%
- Precision: 88.11%
- Recall: 88%
- R² Score (Dropout Regression): 0.59

👉 Indicates strong classification performance with moderate regression accuracy.

---

## 🧠 Behavioural Insights

- Sleep is the strongest factor in reducing burnout
- Assignment delay significantly increases dropout probability
- Participation improves engagement levels
- Study consistency stabilizes academic performance
- Screen time indirectly affects burnout through reduced sleep

---

## 💡 Output Explanation

The system provides:

- Burnout level (Low / Medium / High) + confidence
- Dropout probability (%)
- Disengagement level
- SHAP explanation (feature impact visualization)
- Personalized recommendations for improvement

---

## 📦 Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/devveldev-sketch/student-burnout-risk-predictor.git
cd student-burnout-risk-predictor
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the application:

```
python app.py
```

4. Open in browser:

```
http://127.0.0.1:5000
```

---

## 📊 Output Dashboard

The dashboard displays:

- Risk scores and classifications
- Gauge visualization
- Behavioral insights
- Recommendations
- SHAP explainability graphs
- Feature importance analysis

---

## 🌍 Practical Impact

- Helps identify at-risk students early
- Enables proactive academic interventions
- Supports data-driven decision making for institutions
- Improves student well-being and retention

---

## 🔮 Future Scope

- Integration with real-time student data systems
- Deep learning models for improved prediction
- Mobile app deployment
- Personalized intervention recommendations using NLP
- Integration with LMS platforms (Moodle, Blackboard)

---

## 👩‍💻 Author

Devadharshini S  
Integrated M.Tech CSE (Business Analytics)  
VIT Chennai
