import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Replace these with YOUR confusion matrix values


cm = [
    [532, 57],
    [22, 1204]
]

labels = ["Real", "Fake"]

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - LatentGuard")

plt.tight_layout()

plt.savefig(
    "plots/confusion_matrix.png",
    dpi=300,
     bbox_inches="tight"
)

plt.show()

print("Confusion matrix saved successfully!")