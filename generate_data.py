import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000
data = []

for i in range(n):
    attendance = np.random.randint(40, 100)
    gpa = round(np.random.uniform(4.5, 10), 2)
    study_hours = round(np.random.uniform(0.5, 6), 1)
    sleep_hours = round(np.random.uniform(4, 9), 1)
    screen_time = round(np.random.uniform(2, 10), 1)
    assignment_delay = np.random.randint(0, 10)
    participation = np.random.randint(1, 10)

    gpa_trend = np.random.choice([-1, 0, 1])
    attendance_trend = np.random.choice([-1, 0, 1])
    study_consistency = np.random.choice([0, 1])

    # 🔹 Burnout score
    score = 0
    if attendance < 60: score += 2
    if gpa < 6: score += 2
    if study_hours < 2: score += 1
    if sleep_hours < 5: score += 2
    if screen_time > 7: score += 1
    if assignment_delay > 5: score += 2
    if participation < 4: score += 1
    if gpa_trend == -1: score += 2
    if attendance_trend == -1: score += 2
    if study_consistency == 0: score += 1

    if score >= 9:
        burnout = "High"
        dropout = round(np.random.uniform(0.7, 0.95), 2)
    elif score >= 5:
        burnout = "Medium"
        dropout = round(np.random.uniform(0.4, 0.7), 2)
    else:
        burnout = "Low"
        dropout = round(np.random.uniform(0.1, 0.4), 2)

    # 🔹 Disengagement score
    disengage_score = 0
    if participation < 4: disengage_score += 2
    if assignment_delay > 5: disengage_score += 2
    if gpa_trend == -1: disengage_score += 2
    if attendance_trend == -1: disengage_score += 2
    if study_consistency == 0: disengage_score += 1

    if disengage_score >= 6:
        disengagement = "High"
    elif disengage_score >= 3:
        disengagement = "Medium"
    else:
        disengagement = "Low"

    # 🔥 ADD NOISE (VERY IMPORTANT)
    if np.random.rand() < 0.1:
        disengagement = np.random.choice(["Low", "Medium", "High"])

    data.append([
        i+1, attendance, gpa, study_hours, sleep_hours,
        screen_time, assignment_delay, participation,
        gpa_trend, attendance_trend, study_consistency,
        burnout, dropout, disengagement
    ])

columns = [
    "Student_ID", "Attendance", "GPA", "Study_Hours",
    "Sleep_Hours", "Screen_Time", "Assignment_Delay",
    "Participation_Score", "GPA_Trend", "Attendance_Trend",
    "Study_Consistency", "Burnout_Risk",
    "Dropout_Probability", "Disengagement_Level"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("student_burnout_data.csv", index=False)

print("✅ Dataset generated with noise!")