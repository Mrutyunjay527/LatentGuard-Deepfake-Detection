import csv
import os

results = [
    [
        "Baseline",
        "Original Test",
        0.9554,
        0.9545,
        0.9763,
        0.9653,
        0.9890
    ],
    [
        "Baseline",
        "Firefly",
        0.6625,
        0.6066,
        0.9250,
        0.7327,
        0.8531
    ],
    [
        "Baseline + Calibration",
        "Firefly",
        0.6875,
        0.6316,
        0.9000,
        0.7423,
        0.8531
    ],
    [
        "Robust",
        "Original Test",
        0.9526,
        0.9545,
        0.9763,
        0.9653,
        0.9890
    ],
    [
        "Robust",
        "Firefly @ 0.50",
        0.6875,
        0.6293,
        0.9125,
        0.7449,
        0.8527
    ],
    [
        "Robust + Calibration",
        "Firefly @ 0.55",
        0.7125,
        0.6667,
        0.8500,
        0.7473,
        0.8527
    ]
]

headers = [
    "Model",
    "Dataset",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC"
]

os.makedirs("outputs", exist_ok=True)

csv_path = "outputs/final_results_summary.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(headers)
    writer.writerows(results)

print("Final results summary saved to:")
print(csv_path)