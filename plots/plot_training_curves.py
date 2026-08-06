import torch
import matplotlib.pyplot as plt

# Load history
history = torch.load("outputs/history.pth")

train_loss = history["train_loss"]
val_loss = history["val_loss"]

train_acc = history["train_accuracy"]
val_acc = history["val_accuracy"]

epochs = range(1, len(train_loss) + 1)

# ---------------- LOSS ----------------
plt.figure(figsize=(8,5))

plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

plt.grid(True)

plt.tight_layout()
plt.savefig("plots/loss_curve.png", dpi=300)
plt.close()

# ---------------- ACCURACY ----------------
plt.figure(figsize=(8,5))

plt.plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
plt.plot(epochs, val_acc, label="Validation Accuracy", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy")
plt.legend()

plt.grid(True)

plt.tight_layout()
plt.savefig("plots/accuracy_curve.png", dpi=300)
plt.close()

print("Training curves saved successfully!")