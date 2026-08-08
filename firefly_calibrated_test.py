import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from config import *

from src.dataset import ImageDataset
from src.transforms import val_transform

from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder
from models.latent_guard import LatentGuard


def test():

    print("=" * 60)
    print("LATENTGUARD - CALIBRATED FIREFLY TEST")
    print("=" * 60)

    print(f"\nUsing Device: {DEVICE}")

    
    # Load calibrated threshold
    threshold_file = "outputs/calibrated_threshold.json"

    with open(threshold_file, "r") as f:
        calibration = json.load(f)

    # threshold = calibration["threshold"]
    threshold = 0.50

    print(f"Calibrated Threshold: {threshold:.2f}")

    
    # Firefly dataset
    FIREFLY_TEST_DIR = "data/FireFly_test"

    test_dataset = ImageDataset(
        FIREFLY_TEST_DIR,
        transform=val_transform
    )

    print(
        "Firefly Testing Images:",
        len(test_dataset)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

   
    # Load model
    print("\nLoading LatentGuard...")

    image_encoder = ImageEncoder().to(DEVICE)

    vae_encoder = VAEEncoder().to(DEVICE)

    model = LatentGuard(
        image_encoder=image_encoder,
        vae_encoder=vae_encoder
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_SAVE_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print("Model Loaded Successfully!")

    
    # Evaluation
    labels_list = []
    probabilities = []
    predictions = []

    print("\nEvaluating unseen Firefly dataset...\n")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probs = torch.softmax(
                outputs,
                dim=1
            )

            fake_probs = probs[:, 1]

            probabilities.extend(
                fake_probs.cpu().numpy()
            )

            labels_list.extend(
                labels.numpy()
            )

    labels_list = np.array(labels_list)

    probabilities = np.array(probabilities)

    
    # Apply calibrated threshold
    predictions = (
        probabilities >= threshold
    ).astype(int)

    
    # Metrics
    accuracy = accuracy_score(
        labels_list,
        predictions
    )

    precision = precision_score(
        labels_list,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels_list,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels_list,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        labels_list,
        predictions
    )

    report = classification_report(
        labels_list,
        predictions,
        target_names=["Real", "Fake"],
        zero_division=0
    )

    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(
        labels_list,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

  
    # Display
    print("\n")
    print("=" * 60)
    print("CALIBRATED FIREFLY RESULTS")
    print("=" * 60)

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC AUC   : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(report)

    
    # Save results
    os.makedirs(
        "outputs",
        exist_ok=True
    )

    with open(
        "outputs/firefly_calibrated_results.txt",
        "w"
    ) as f:

        f.write(
            "LATENTGUARD - CALIBRATED FIREFLY RESULTS\n"
        )

        f.write("=" * 50 + "\n\n")

        f.write(
            f"Threshold : {threshold:.4f}\n"
        )

        f.write(
            f"Accuracy  : {accuracy:.4f}\n"
        )

        f.write(
            f"Precision : {precision:.4f}\n"
        )

        f.write(
            f"Recall    : {recall:.4f}\n"
        )

        f.write(
            f"F1 Score  : {f1:.4f}\n"
        )

        f.write(
            f"ROC AUC   : {roc_auc:.4f}\n\n"
        )

        f.write(
            "Confusion Matrix\n"
        )

        f.write(
            str(cm)
        )

        f.write("\n\n")

        f.write(
            "Classification Report\n"
        )

        f.write(
            report
        )

   
    # Confusion Matrix
    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.title(
        "Confusion Matrix - Calibrated Firefly Test"
    )

    plt.tight_layout()

    plt.savefig(
        "plots/firefly_calibrated_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

   
    # ROC Curve
    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"LatentGuard (AUC = {roc_auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "ROC Curve - Calibrated Firefly Test"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "plots/firefly_calibrated_roc_curve.png",
        dpi=300
    )

    plt.close()

    print("\nSaved:")
    print(
        "outputs/firefly_calibrated_results.txt"
    )

    print(
        "plots/firefly_calibrated_confusion_matrix.png"
    )

    print(
        "plots/firefly_calibrated_roc_curve.png"
    )


if __name__ == "__main__":
    test()