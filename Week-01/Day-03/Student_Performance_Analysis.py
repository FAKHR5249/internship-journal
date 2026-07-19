import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

students = pd.read_csv("student_performance.csv")

print("=" * 50)
print("STUDENT PERFORMANCE ANALYSIS")
print("=" * 50)
print(students)

print("\n========== 1. BASIC STATISTICS ==========")

exam_marks = students["Exam_Score"]

avg_marks = exam_marks.mean()
mid_value = exam_marks.median()
most_common = exam_marks.mode().iloc[0]
spread = exam_marks.var()
deviation = exam_marks.std()

print(f"Average Marks: {avg_marks:.2f}")
print(f"Median Marks : {mid_value}")
print(f"Most Common  : {most_common}")
print(f"Variance     : {spread:.2f}")
print(f"Std Deviation: {deviation:.2f}")

print("\n========== 2. PERCENTILES ==========")

quartiles = np.percentile(exam_marks, [25, 50, 75])

lowest_score = exam_marks.min()
highest_score = exam_marks.max()

print("Q1 (25%) :", quartiles[0])
print("Q2 (50%) :", quartiles[1])
print("Q3 (75%) :", quartiles[2])

print("Lowest Score :", lowest_score)
print("Highest Score:", highest_score)

print(f"Middle 50% Scores: {quartiles[0]} - {quartiles[2]}")


print("\n========== 3. CORRELATION ==========")

features = ["Study_Hours", "Attendance", "Assignment_Score"]

for item in features:
    value = students[item].corr(exam_marks)
    print(f"{item} vs Exam Score = {value:.3f}")

plt.figure(figsize=(6, 4))
plt.scatter(students["Assignment_Score"], exam_marks)
plt.title("Exam Score vs Assignment Score")
plt.xlabel("Assignment Marks")
plt.ylabel("Exam Marks")
plt.grid(True)
plt.show()


print("\n========== 4. PROBABILITY ==========")

total_students = students.shape[0]

prob_pass = (exam_marks >= 50).sum() / total_students
prob_high = (exam_marks >= 80).sum() / total_students
prob_hours = (students["Study_Hours"] > 5).sum() / total_students

print("Passing Probability :", prob_pass)
print("80+ Score Probability:", prob_high)
print("Study Hours >5 Probability:", prob_hours)


print("\n========== 5. NORMAL DISTRIBUTION ==========")

plt.figure(figsize=(7, 4))
plt.hist(exam_marks, bins=8)
plt.title("Exam Score Distribution")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()


students["Standard_Score"] = zscore(exam_marks)

print("\nExam Scores with Z-Score")
print(students[["Student_ID", "Exam_Score", "Standard_Score"]])

print("\nPossible Extreme Values (|Z| > 2)")
print(students[students["Standard_Score"].abs() > 2])


print("\n========== 6. OUTLIER DETECTION ==========")

q1 = quartiles[0]
q3 = quartiles[2]

iqr = q3 - q1

lower_limit = q1 - (1.5 * iqr)
upper_limit = q3 + (1.5 * iqr)

print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

outlier_data = students[
    (exam_marks < lower_limit) |
    (exam_marks > upper_limit)
]

print("\nDetected Outliers")
print(outlier_data)


plt.figure(figsize=(6, 4))
plt.boxplot(exam_marks)
plt.title("Exam Score Box Plot")
plt.show()