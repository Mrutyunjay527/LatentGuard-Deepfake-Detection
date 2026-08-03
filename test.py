import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from config import *

from src.dataset import ImageDataset

from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder
from models.latent_guard import LatentGuard


def test():

    test_dataset = ImageDataset(TEST_DIR)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    image_encoder = ImageEncoder()

    vae_encoder = VAEEncoder()

    model = LatentGuard(
        image_encoder=image_encoder,
        vae_encoder=vae_encoder
    )

    model.load_state_dict(
        torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    )

    model.to(DEVICE)

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

    print("Accuracy :", accuracy_score(labels_list, predictions))
    print("Precision:", precision_score(labels_list, predictions))
    print("Recall   :", recall_score(labels_list, predictions))
    print("F1 Score :", f1_score(labels_list, predictions))
    print("Confusion Matrix")
    print(confusion_matrix(labels_list, predictions))


if __name__ == "__main__":
    test()