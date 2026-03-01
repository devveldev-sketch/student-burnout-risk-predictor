import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score
import pickle

# Load dataset
df = pd.read_csv("student_burnout_data.csv")

# Encode labels
df["Burnout_Risk"] = df["Burnout_Risk"].map({"Low": 0, "Medium": 1, "High": 2})
df["Disengagement_Level"] = df["Disengagement_Level"].map({"Low": 0, "Medium": 1, "High": 2})

# Features
X = df.drop(["Student_ID", "Burnout_Risk", "Dropout_Probability", "Disengagement_Level"], axis=1)

# Targets
y_class = df["Burnout_Risk"]
y_reg = df["Dropout_Probability"]
y_dis = df["Disengagement_Level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_dis_train, y_dis_test = train_test_split(X, y_dis, test_size=0.2, random_state=42)

# 🔥 OPTIMIZED MODELS

clf = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    eval_metric='mlogloss'
)

reg = XGBRegressor(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

dis_model = XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    eval_metric='mlogloss'
)

# Train models
clf.fit(X_train, y_train)
reg.fit(X_train, y_reg_train)
dis_model.fit(X_train, y_dis_train)

# Predictions
y_pred = clf.predict(X_test)
y_dis_pred = dis_model.predict(X_test)
y_reg_pred = reg.predict(X_test)

# Metrics
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
precision = round(precision_score(y_test, y_pred, average='weighted') * 100, 2)
recall = round(recall_score(y_test, y_pred, average='weighted') * 100, 2)
dis_acc = round(accuracy_score(y_dis_test, y_dis_pred) * 100, 2)
r2 = round(r2_score(y_reg_test, y_reg_pred), 2)

# Save models
pickle.dump(clf, open("burnout_model.pkl", "wb"))
pickle.dump(reg, open("dropout_model.pkl", "wb"))
pickle.dump(dis_model, open("disengagement_model.pkl", "wb"))

# Save metrics
metrics = {
    "Burnout Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Disengagement Accuracy": dis_acc,
    "Dropout R2 Score": r2
}
pickle.dump(metrics, open("metrics.pkl", "wb"))

# Feature importance
pd.DataFrame({
    "Feature": X.columns,
    "Importance": clf.feature_importances_
}).sort_values(by="Importance", ascending=False)\
 .to_csv("feature_importance.csv", index=False)

print("🔥 Training complete with optimized models!")
print(metrics)