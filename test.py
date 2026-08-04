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

# Dataset
from src.dataset import ImageDataset
from src.transforms import val_transform

# Models
from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder
from models.latent_guard import LatentGuard


def test():

    print(f"Using Device : {DEVICE}")

   
    # Load Test Dataset
    test_dataset = ImageDataset(
        TEST_DIR,
        transform=val_transform
    )

    print("Testing Images :", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    
    # Build Model
    print("Loading Image Encoder...")
    image_encoder = ImageEncoder().to(DEVICE)

    print("Loading VAE Encoder...")
    vae_encoder = VAEEncoder().to(DEVICE)

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

    print("Model Loaded Successfully!\n")


    # Evaluation
    predictions = []
    labels_list = []
    probabilities = []

    print("Evaluating Model...\n")

    with torch.no_grad():

        for images, labels in test_loader:

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

    
    # Metrics
    accuracy = accuracy_score(
        labels_list,
        predictions
    )

    precision = precision_score(
        labels_list,
        predictions
    )

    recall = recall_score(
        labels_list,
        predictions
    )

    f1 = f1_score(
        labels_list,
        predictions
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
        predictions
    )

    
    # Display Results
    print("=" * 50)
    print("LATENTGUARD TEST RESULTS")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(report)

    
    # Save Results
    with open("outputs/test_results.txt", "w") as f:

        f.write("LATENTGUARD TEST RESULTS\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"ROC AUC  : {roc_auc:.4f}\n\n")

        f.write("Confusion Matrix\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("Classification Report\n")
        f.write(report)

    print("\nResults saved to outputs/test_results.txt")


if __name__ == "__main__":
    test()