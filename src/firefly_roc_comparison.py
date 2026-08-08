import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

from config import *
from src.dataset import ImageDataset
from src.transforms import val_transform

from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder
from models.latent_guard import LatentGuard


FIREfly_DIR = "data/FireFly_test"

BATCH_SIZE_LOCAL = BATCH_SIZE


def get_scores(model_path):

    dataset = ImageDataset(
        FIREfly_DIR,
        transform=val_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_LOCAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    image_encoder = ImageEncoder().to(DEVICE)
    vae_encoder = VAEEncoder().to(DEVICE)

    model = LatentGuard(
        image_encoder=image_encoder,
        vae_encoder=vae_encoder
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.eval()

    labels_all = []
    scores_all = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            fake_probability = probabilities[:, 1]

            labels_all.extend(
                labels.numpy()
            )

            scores_all.extend(
                fake_probability.cpu().numpy()
            )

    return labels_all, scores_all


def main():

    print("Using Device:", DEVICE)

    print("\nEvaluating Baseline...")
    
    baseline_labels, baseline_scores = get_scores(
        "outputs/latentguard_baseline.pth"
    )

    print("Baseline evaluation complete.")

    print("\nEvaluating Robust Model...")

    robust_labels, robust_scores = get_scores(
        "outputs/latentguard_robust.pth"
    )

    print("Robust evaluation complete.")

    # ROC curves

    baseline_fpr, baseline_tpr, _ = roc_curve(
        baseline_labels,
        baseline_scores
    )

    robust_fpr, robust_tpr, _ = roc_curve(
        robust_labels,
        robust_scores
    )

    baseline_auc = auc(
        baseline_fpr,
        baseline_tpr
    )

    robust_auc = auc(
        robust_fpr,
        robust_tpr
    )

    # Plot

    plt.figure(figsize=(8, 7))

    plt.plot(
        baseline_fpr,
        baseline_tpr,
        linewidth=2,
        label=f"Baseline LatentGuard (AUC = {baseline_auc:.4f})"
    )

    plt.plot(
        robust_fpr,
        robust_tpr,
        linewidth=2,
        label=f"Robust LatentGuard (AUC = {robust_auc:.4f})"
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
        "ROC Comparison - Unseen Firefly Dataset"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "plots/firefly_baseline_vs_robust_roc.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "=" * 55)
    print("FIREfly ROC COMPARISON")
    print("=" * 55)

    print(
        f"Baseline ROC-AUC : {baseline_auc:.4f}"
    )

    print(
        f"Robust ROC-AUC   : {robust_auc:.4f}"
    )

    print(
        f"Difference       : {robust_auc - baseline_auc:+.4f}"
    )

    print(
        "\nROC comparison saved successfully!"
    )


if __name__ == "__main__":
    main()