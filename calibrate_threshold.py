import os
import json
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import *

from src.dataset import ImageDataset
from src.transforms import val_transform

from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder
from models.latent_guard import LatentGuard



# Get probabilities from a dataset
def get_predictions(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    labels = []
    probabilities = []

    model.eval()

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probs = torch.softmax(
                outputs,
                dim=1
            )

            fake_probability = probs[:, 1]

            probabilities.extend(
                fake_probability.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    return (
        np.array(labels),
        np.array(probabilities)
    )



# Find best threshold using validation data
def find_best_threshold(labels, probabilities):

    best_threshold = 0.5
    best_score = 0.0

    results = []

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        balanced_acc = balanced_accuracy_score(
            labels,
            predictions
        )

        results.append(
            (threshold, balanced_acc)
        )

        if balanced_acc > best_score:

            best_score = balanced_acc
            best_threshold = threshold

    return best_threshold, best_score, results



# Evaluate
def evaluate_threshold(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        labels,
        predictions
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "predictions": predictions
    }



# Main
def main():

    print("=" * 60)
    print("LATENTGUARD THRESHOLD CALIBRATION")
    print("=" * 60)

    print(
        f"\nUsing Device: {DEVICE}"
    )

   
    # Create output folders
    os.makedirs(
        "outputs",
        exist_ok=True
    )

    os.makedirs(
        "plots",
        exist_ok=True
    )

    
    # Load validation dataset
    val_dataset = ImageDataset(
        VAL_DIR,
        transform=val_transform
    )

    print(
        "Validation Images:",
        len(val_dataset)
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

    print(
        "Model loaded successfully!"
    )

    
    # Get validation probabilities
    print(
        "\nRunning validation inference..."
    )

    val_labels, val_probabilities = get_predictions(
        model,
        val_dataset
    )

   
    # Find threshold
    print(
        "\nSearching for optimal threshold..."
    )

    best_threshold, best_score, threshold_results = (
        find_best_threshold(
            val_labels,
            val_probabilities
        )
    )

    print(
        f"\nBest Threshold : {best_threshold:.2f}"
    )

    print(
        f"Validation Balanced Accuracy : "
        f"{best_score:.4f}"
    )

   
    # Save threshold
    threshold_file = (
        "outputs/calibrated_threshold.json"
    )

    with open(
        threshold_file,
        "w"
    ) as f:

        json.dump(
            {
                "threshold": float(
                    best_threshold
                ),
                "selection_metric":
                    "balanced_accuracy"
            },
            f,
            indent=4
        )

    print(
        f"\nThreshold saved to: "
        f"{threshold_file}"
    )

    
    # Validation results
    val_results = evaluate_threshold(
        val_labels,
        val_probabilities,
        best_threshold
    )

    print("\n")
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"Accuracy          : "
        f"{val_results['accuracy']:.4f}"
    )

    print(
        f"Precision         : "
        f"{val_results['precision']:.4f}"
    )

    print(
        f"Recall            : "
        f"{val_results['recall']:.4f}"
    )

    print(
        f"F1 Score          : "
        f"{val_results['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{val_results['balanced_accuracy']:.4f}"
    )

    print(
        f"ROC-AUC           : "
        f"{val_results['roc_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        val_results["confusion_matrix"]
    )

    
    # Threshold graph
    thresholds = [
        x[0]
        for x in threshold_results
    ]

    scores = [
        x[1]
        for x in threshold_results
    ]

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        thresholds,
        scores,
        linewidth=2
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        label=f"Best threshold = {best_threshold:.2f}"
    )

    plt.xlabel(
        "Decision Threshold"
    )

    plt.ylabel(
        "Balanced Accuracy"
    )

    plt.title(
        "Threshold Calibration - Validation Set"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "plots/threshold_calibration.png",
        dpi=300
    )

    plt.close()

    print(
        "\nThreshold calibration plot saved!"
    )

    
    # Done
    print("\n")
    print("=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)

    print(
        f"Use threshold: {best_threshold:.2f}"
    )

    print(
        "\nNext step:"
    )

    print(
        "Run the calibrated threshold on the "
        "Firefly unseen dataset."
    )


if __name__ == "__main__":
    main()