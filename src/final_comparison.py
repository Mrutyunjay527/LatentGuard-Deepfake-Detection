import matplotlib.pyplot as plt
import numpy as np

# Results
models = [
    "Baseline",
    "Robust"
]

original_accuracy = [95.54, 95.26]
firefly_accuracy = [66.25, 71.25]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8, 6))

plt.bar(
    x - width/2,
    original_accuracy,
    width,
    label="Original Test"
)

plt.bar(
    x + width/2,
    firefly_accuracy,
    width,
    label="Firefly Unseen Test"
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Baseline vs Robust LatentGuard")

plt.xticks(x, models)
plt.ylim(0, 100)

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plots/baseline_vs_robust_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Comparison graph saved successfully!")