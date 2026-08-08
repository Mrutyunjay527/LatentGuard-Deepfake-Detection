import matplotlib.pyplot as plt
import numpy as np

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

baseline = [
    66.25,
    60.66,
    92.50,
    73.27,
    85.33
]

robust = [
    71.25,
    66.67,
    85.00,
    74.73,
    85.27
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    baseline,
    width,
    label="Baseline LatentGuard"
)

plt.bar(
    x + width / 2,
    robust,
    width,
    label="Robust LatentGuard"
)

plt.ylabel("Score (%)")
plt.xlabel("Evaluation Metric")
plt.title("Baseline vs Robust LatentGuard on Unseen Firefly Dataset")

plt.xticks(x, metrics)
plt.ylim(0, 100)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "plots/firefly_metric_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Metric comparison graph saved successfully!")