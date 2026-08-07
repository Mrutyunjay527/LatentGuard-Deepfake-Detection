import os
import torch
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

    print(f"Using Device : {DEVICE}")

    
    # FIREFLY DATASET
    FIREFLY_TEST_DIR = "data/FireFly_test"

    print("\nChecking Firefly dataset...")
    print("Path:", os.path.abspath(FIREFLY_TEST_DIR))
    print("Exists:", os.path.exists(FIREFLY_TEST_DIR))

    if not os.path.exists(FIREFLY_TEST_DIR):
        print("ERROR: Firefly dataset folder not found!")
        return

    print("Folders:", os.listdir(FIREFLY_TEST_DIR))

    for folder in os.listdir(FIREFLY_TEST_DIR):

        folder_path = os.path.join(
            FIREFLY_TEST_DIR,
            folder
        )

        if os.path.isdir(folder_path):

            images = os.listdir(folder_path)

            print(
                f"{folder}: {len(images)} files"
            )

    
    # LOAD DATASET
    test_dataset = ImageDataset(
        FIREFLY_TEST_DIR,
        transform=val_transform
    )

    print(
        "\nFirefly Testing Images :",
        len(test_dataset)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    
    # LOAD MODEL
    print("\nLoading Image Encoder...")

    image_encoder = ImageEncoder().to(DEVICE)

    print("Loading VAE Encoder...")

    vae_encoder = VAEEncoder().to(DEVICE)

    print("Loading LatentGuard...")

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

    
    # EVALUATION
    predictions = []
    labels_list = []
    probabilities = []

    print("\nEvaluating FIREFLY dataset...\n")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            probs = torch.softmax(
                outputs,
                dim=1
            )

            probabilities.extend(
                probs[:, 1].cpu().numpy()
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            labels_list.extend(
                labels.cpu().numpy()
            )

   
    # METRICS
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
        zero_division=0
    )

   
    # ROC
    fpr, tpr, _ = roc_curve(
        labels_list,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    
    # PRINT RESULTS
    print("\n")
    print("=" * 60)
    print("LATENTGUARD - FIREFLY UNSEEN TEST")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(report)

    
    # SAVE RESULTS
    with open(
        "outputs/firefly_test_results.txt",
        "w"
    ) as f:

        f.write(
            "LATENTGUARD - FIREFLY UNSEEN TEST\n"
        )

        f.write("=" * 50 + "\n\n")

        f.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        f.write(
            f"Precision: {precision:.4f}\n"
        )

        f.write(
            f"Recall   : {recall:.4f}\n"
        )

        f.write(
            f"F1 Score : {f1:.4f}\n"
        )

        f.write(
            f"ROC AUC  : {roc_auc:.4f}\n\n"
        )

        f.write("Confusion Matrix\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("Classification Report\n")
        f.write(report)

    
    # CONFUSION MATRIX
    plt.figure(figsize=(6, 5))

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
        "Confusion Matrix - Firefly Unseen Test"
    )

    plt.tight_layout()

    plt.savefig(
        "plots/firefly_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ROC CURVE
    plt.figure(figsize=(7, 6))

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
        "ROC Curve - Firefly Unseen Test"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "plots/firefly_roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nResults saved:")
    print("outputs/firefly_test_results.txt")
    print("plots/firefly_confusion_matrix.png")
    print("plots/firefly_roc_curve.png")


if __name__ == "__main__":
    test()