from flask import Flask, render_template, request
import numpy as np
import pickle
import shap

app = Flask(__name__)

# LOAD MODELS
model = pickle.load(open("model.pkl", "rb"))
dropout_model = pickle.load(open("dropout_model.pkl", "rb"))
dis_model = pickle.load(open("disengagement_model.pkl", "rb"))
metrics = pickle.load(open("metrics.pkl", "rb"))

feature_names = [
    "Attendance", "GPA", "Study_Hours", "Sleep_Hours",
    "Screen_Time", "Assignment_Delay", "Participation_Score",
    "GPA_Trend", "Attendance_Trend", "Study_Consistency"
]

explainer = shap.TreeExplainer(model)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [
            float(request.form.get("attendance", 0)),
            float(request.form.get("gpa", 0)),
            float(request.form.get("study_hours", 0)),
            float(request.form.get("sleep_hours", 0)),
            float(request.form.get("screen_time", 0)),
            float(request.form.get("assignment_delay", 0)),
            float(request.form.get("participation", 0)),
            float(request.form.get("gpa_trend", 0)),
            float(request.form.get("attendance_trend", 0)),
            float(request.form.get("consistency", 0)),
        ]

        input_data = np.array(data).reshape(1, -1)

        # MODEL
        pred_class = int(model.predict(input_data)[0])
        probs = model.predict_proba(input_data)[0]
        confidence = float(np.max(probs) * 100)

        dropout = float(dropout_model.predict(input_data)[0] * 100)
        dropout = round(min(max(dropout, 0), 100), 2)

        dis_pred = int(dis_model.predict(input_data)[0])

        attendance, gpa, study, sleep, screen, delay, part, gtrend, atrend, cons = data

        # 🔥 HYBRID RULE SCORING
        risk_score_rule = 0
        if sleep < 5: risk_score_rule += 25
        if delay > 7: risk_score_rule += 25
        if part < 4: risk_score_rule += 20
        if study < 2: risk_score_rule += 15
        if gtrend < 0: risk_score_rule += 15
        if attendance < 60: risk_score_rule += 10

        if risk_score_rule >= 60:
            pred_class = 2
            confidence = max(confidence, 85)
        elif risk_score_rule >= 35:
            pred_class = 1
            confidence = max(confidence, 70)

        labels = ["Low", "Medium", "High"]
        risk = labels[pred_class]
        disengagement = labels[dis_pred]

        score = int(risk_score_rule)

        # -------------------------------
        # SHAP
        # -------------------------------
        shap_values = explainer.shap_values(input_data)

        if isinstance(shap_values, list):
            shap_values = shap_values[pred_class]

        shap_values = shap_values[0]
        shap_values = np.array(shap_values).astype(float).flatten().tolist()

        shap_pairs = list(zip(feature_names, shap_values))
        shap_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)

        shap_labels = [x[0] for x in shap_sorted[:5]]
        shap_values_top = [float(x[1]) for x in shap_sorted[:5]]

        # -------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------
        importances = model.feature_importances_
        feat_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

        feat_labels = [x[0] for x in feat_pairs[:5]]
        feat_values = [float(x[1]) for x in feat_pairs[:5]]

        # -------------------------------
        # 🔥 EXPLANATIONS
        # -------------------------------
        explanations = []
        for f, v in shap_sorted[:3]:
            if v > 0:
                explanations.append(f"{f} is increasing burnout risk")
            else:
                explanations.append(f"{f} is reducing burnout risk")

        # -------------------------------
        # 🔥 TRIGGERS (DATA-DRIVEN)
        # -------------------------------
        triggers = [f for f, v in shap_sorted[:3] if v > 0]
        if not triggers:
            triggers = ["No major risk factors detected"]

        # -------------------------------
        # 🔥 SMART RECOMMENDATIONS
        # -------------------------------
        recommendation = []
        if sleep < 6:
            recommendation.append(f"Increase sleep from {sleep}h → 7h")
        if delay > 5:
            recommendation.append(f"Reduce assignment delay (current: {delay})")
        if part < 5:
            recommendation.append(f"Improve participation (current: {part})")

        if not recommendation:
            recommendation.append("Maintain current performance")

        # -------------------------------
        # 🔄 WHAT-IF
        # -------------------------------
        what_if = []
        if sleep < 6:
            what_if.append("If sleep improves to 7h, burnout may reduce")
        if delay > 6:
            what_if.append("Reducing delays can lower risk level")

        return render_template(
            "result.html",
            risk=risk,
            confidence=round(confidence, 2),
            score=score,
            dropout=dropout,
            disengagement=disengagement,
            triggers=triggers,
            recommendation=recommendation,
            shap_labels=shap_labels,
            shap_values=shap_values_top,
            feat_labels=feat_labels,
            feat_values=feat_values,
            metrics=metrics,
            explanations=explanations,
            what_if=what_if
        )

    except Exception as e:
        return f"ERROR: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)