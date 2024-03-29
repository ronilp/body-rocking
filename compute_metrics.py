# File: compute_metrics.py
# Author: Ronil Pancholia
# Date: 3/25/19
# Time: 11:06 PM
from sklearn.metrics import classification_report, accuracy_score

prediction_files = ["Session05.txt", "Session16.txt"]
ground_truth_files = ["data/TrainingData/test/Session05/detection.txt", "data/TrainingData/test/Session16/detection.txt"]

predictions = []
ground_truth = []

print("Reading predictions")
for file_name in prediction_files:
    with open(file_name) as f:
        for line in f:
            predictions.append(float(line.strip()))

    print(len(predictions))

print("Reading ground truth")
for file_name in ground_truth_files:
    with open(file_name) as f:
        for line in f:
            ground_truth.append(float(line.strip()))

    print(len(ground_truth))

print(classification_report(ground_truth, predictions))
print(accuracy_score(ground_truth, predictions))
