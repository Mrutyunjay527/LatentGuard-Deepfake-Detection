import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from config import *
from src.dataset import ImageDataset
from src.transforms import val_transform

from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder
from models.latent_guard import LatentGuard


def firefly_test():

    print("=" * 60)
    print("Adobe Firefly Image 4 - Unseen Evaluation")
    print("=" * 60)

    test_dataset = ImageDataset(
        "data/FireFly_test",
        transform=val_transform
    )

    print("Total Images:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    image_encoder = ImageEncoder().to(DEVICE)
    vae_encoder = VAEEncoder().to(DEVICE)

    model = LatentGuard(
        image_encoder=image_encoder,
        vae_encoder=vae_encoder
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    )

    model.eval()

    predictions = []
    labels_list = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())

            labels_list.extend(labels.numpy())

    print("\nRESULTS")
    print("-" * 40)

    print("Accuracy :", accuracy_score(labels_list, predictions))
    print("Precision:", precision_score(labels_list, predictions))
    print("Recall   :", recall_score(labels_list, predictions))
    print("F1 Score :", f1_score(labels_list, predictions))

    print("\nConfusion Matrix")
    print(confusion_matrix(labels_list, predictions))

    print("\nClassification Report")
    print(classification_report(labels_list, predictions))


if __name__ == "__main__":
    firefly_test()