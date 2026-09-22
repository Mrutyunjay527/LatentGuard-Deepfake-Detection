import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from config import *

from src.dataset import ImageDataset
from src.transforms import val_transform

from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder
from models.latent_guard import LatentGuard


# --------------------------------------------------
# FIREfly UNSEEN-GENERATOR TEST
# --------------------------------------------------

FIREFLY_DIR = "data/FireFly_test"


def test_firefly():

    print("=" * 60)
    print("LATENTGUARD - FIREFLY UNSEEN GENERATOR TEST")
    print("=" * 60)

    print(f"\nUsing Device : {DEVICE}")

    # -----------------------------------------------
    # Load Firefly Dataset
    # -----------------------------------------------

    firefly_dataset = ImageDataset(
        FIREFLY_DIR,
        transform=val_transform
    )

    print("Firefly Images :", len(firefly_dataset))

    firefly_loader = DataLoader(
        firefly_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # -----------------------------------------------
    # Load Encoders
    # -----------------------------------------------

    print("\nLoading Image Encoder...")
    image_encoder = ImageEncoder().to(DEVICE)

    print("Loading VAE Encoder...")
    vae_encoder = VAEEncoder().to(DEVICE)

    # -----------------------------------------------
    # Load LatentGuard
    # -----------------------------------------------

    print("Loading LatentGuard Model...")

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

    # -----------------------------------------------
    # Evaluation
    # -----------------------------------------------

    predictions = []
    labels_list = []
    probabilities = []

    print("\nEvaluating Firefly images...\n")

    with torch.no_grad():

        for images, labels in firefly_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            predictions.extend(
                predicted.cpu().numpy()
            )

            labels_list.extend(
                labels.cpu().numpy()
            )

            probabilities.extend(
                probs[:, 1].cpu().numpy()
            )

    # -----------------------------------------------
    # Metrics
    # -----------------------------------------------

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

    roc_auc = roc_auc_score(
        labels_list,
        probabilities
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

    # -----------------------------------------------
    # Print Results
    # -----------------------------------------------

    print("=" * 60)
    print("LATENTGUARD FIREFLY RESULTS")
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

    # -----------------------------------------------
    # Save Results
    # -----------------------------------------------

    with open(
        "outputs/firefly_test_results.txt",
        "w"
    ) as f:

        f.write(
            "LATENTGUARD FIREFLY UNSEEN GENERATOR RESULTS\n"
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

        f.write("\n\nClassification Report\n")
        f.write(report)

    print(
        "\nResults saved to "
        "outputs/firefly_test_results.txt"
    )


if __name__ == "__main__":
    test_firefly()