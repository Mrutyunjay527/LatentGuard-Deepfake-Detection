import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

from config import *

from src.dataset import ImageDataset
from src.transforms import val_transform

from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder
from models.latent_guard import LatentGuard


FIREFLY_DIR = "data/FireFly_test"
ERROR_DIR = "outputs/error_analysis"


def error_analysis():

    os.makedirs(ERROR_DIR, exist_ok=True)

    print("=" * 60)
    print("LATENTGUARD - FIREFLY ERROR ANALYSIS")
    print("=" * 60)

    print("\nUsing Device:", DEVICE)

    # --------------------------------------------------
    # Load Firefly Dataset
    # --------------------------------------------------

    dataset = ImageDataset(
        FIREFLY_DIR,
        transform=val_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print("Firefly images:", len(dataset))

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------

    print("\nLoading Image Encoder...")
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

    print("Model loaded successfully.")

    # --------------------------------------------------
    # Find Misclassified Images
    # --------------------------------------------------

    errors = []

    image_index = 0

    print("\nEvaluating Firefly images...\n")

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            for i in range(len(images)):

                true_label = labels[i].item()
                predicted_label = predictions[i].item()

                if true_label != predicted_label:

                    confidence = probabilities[
                        i,
                        predicted_label
                    ].item()

                    errors.append({
                        "index": image_index,
                        "true": true_label,
                        "predicted": predicted_label,
                        "confidence": confidence
                    })

                image_index += 1

    # --------------------------------------------------
    # Print Error Summary
    # --------------------------------------------------

    print("\nTotal misclassified images:", len(errors))

    for error in errors:

        print(
            f"Image {error['index']} | "
            f"True: {error['true']} | "
            f"Predicted: {error['predicted']} | "
            f"Confidence: {error['confidence']:.4f}"
        )

    # --------------------------------------------------
    # Save Error Details
    # --------------------------------------------------

    error_file = os.path.join(
        ERROR_DIR,
        "error_details.txt"
    )

    with open(error_file, "w") as f:

        f.write(
            "LATENTGUARD FIREFLY ERROR ANALYSIS\n"
        )

        f.write("=" * 50 + "\n\n")

        f.write(
            f"Total errors: {len(errors)}\n\n"
        )

        for error in errors:

            f.write(
                f"Image index: {error['index']}\n"
            )

            f.write(
                f"True label: {error['true']}\n"
            )

            f.write(
                f"Predicted label: {error['predicted']}\n"
            )

            f.write(
                f"Prediction confidence: "
                f"{error['confidence']:.4f}\n"
            )

            f.write("-" * 40 + "\n")

    print(
        "\nError details saved to:"
        f"\n{error_file}"
    )

    # --------------------------------------------------
    # Create Error Montage
    # --------------------------------------------------

    if len(errors) == 0:

        print("\nNo errors found.")
        return

    rows = 4
    cols = 4

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 12)
    )

    axes = axes.flatten()

    # Hide all subplot axes first
    for ax in axes:
        ax.axis("off")

    # --------------------------------------------------
    # Display Misclassified Images
    # --------------------------------------------------

    for i, error in enumerate(errors):

        idx = error["index"]

        # Your ImageDataset stores:
        # self.images = [(image_path, label), ...]
        image_path = dataset.images[idx][0]

        image = Image.open(
            image_path
        ).convert("RGB")

        axes[i].imshow(image)

        axes[i].set_title(
            f"True: {error['true']} | "
            f"Pred: {error['predicted']}\n"
            f"Conf: {error['confidence']:.2f}"
        )

        axes[i].axis("off")

    plt.tight_layout()

    # --------------------------------------------------
    # Save Montage
    # --------------------------------------------------

    montage_path = os.path.join(
        ERROR_DIR,
        "firefly_error_montage.png"
    )

    plt.savefig(
        montage_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nError montage saved to:"
        f"\n{montage_path}"
    )

    print("\nError analysis completed successfully.")


if __name__ == "__main__":
    error_analysis()