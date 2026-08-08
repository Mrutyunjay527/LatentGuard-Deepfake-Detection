import os
import shutil
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import *
from src.transforms import val_transform

from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder
from models.latent_guard import LatentGuard



# Dataset
class FireflyErrorDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        # 0 = Real
        # 1 = Fake

        for class_name, label in [
            ("Real", 0),
            ("Fake", 1)
        ]:

            class_dir = os.path.join(
                root_dir,
                class_name
            )

            if not os.path.exists(class_dir):
                print(
                    f"WARNING: {class_dir} not found"
                )
                continue

            for filename in os.listdir(class_dir):

                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp")
                ):

                    path = os.path.join(
                        class_dir,
                        filename
                    )

                    self.samples.append(
                        (path, label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, path



# Main
def analyze_errors():

    print(f"Using Device : {DEVICE}")

   
    # Paths
    FIREfly_DIR = "data/FireFly_test"

    ERROR_DIR = "outputs/firefly_errors"

    REAL_AS_FAKE_DIR = os.path.join(
        ERROR_DIR,
        "real_as_fake"
    )

    FAKE_AS_REAL_DIR = os.path.join(
        ERROR_DIR,
        "fake_as_real"
    )

    os.makedirs(
        REAL_AS_FAKE_DIR,
        exist_ok=True
    )

    os.makedirs(
        FAKE_AS_REAL_DIR,
        exist_ok=True
    )

    
    # Dataset
    dataset = FireflyErrorDataset(
        FIREfly_DIR,
        transform=val_transform
    )

    print(
        "\nTotal Firefly images:",
        len(dataset)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

   
    # Load Model
    print("\nLoading model...")

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

    print("Model loaded successfully!")

  
    # Error storage
    errors = []

    real_as_fake = 0
    fake_as_real = 0
    THRESHOLD = 0.55

   
    # Evaluation
   
    print("\nFinding misclassified images...\n")

    with torch.no_grad():

        for images, labels, paths in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            fake_probabilities = probabilities[:, 1]

            predictions = (
                fake_probabilities >= THRESHOLD
            ).long()

            for i in range(len(images)):

                true_label = labels[i].item()

                predicted_label = predictions[i].item()

                confidence = probabilities[
                    i,
                    predicted_label
                ].item()

               
                # Misclassification

                if true_label != predicted_label:

                    original_path = paths[i]

                    filename = os.path.basename(
                        original_path
                    )

                    if true_label == 0 and predicted_label == 1:

                        destination = os.path.join(
                            REAL_AS_FAKE_DIR,
                            filename
                        )

                        real_as_fake += 1

                        error_type = "Real_as_Fake"

                    else:

                        destination = os.path.join(
                            FAKE_AS_REAL_DIR,
                            filename
                        )

                        fake_as_real += 1

                        error_type = "Fake_as_Real"

                    shutil.copy2(
                        original_path,
                        destination
                    )

                    errors.append([
                        filename,
                        original_path,
                        "Real" if true_label == 0 else "Fake",
                        "Real" if predicted_label == 0 else "Fake",
                        round(confidence, 4),
                        error_type
                    ])

    
    # Save CSV

    csv_path = os.path.join(
        ERROR_DIR,
        "firefly_errors.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Image",
            "Original Path",
            "True Label",
            "Predicted Label",
            "Confidence",
            "Error Type"
        ])

        writer.writerows(errors)

   
    # Summary
    print("=" * 55)
    print("FIREFLY FAILURE CASE ANALYSIS")
    print("=" * 55)

    print(
        f"Real → Fake errors : {real_as_fake}"
    )

    print(
        f"Fake → Real errors : {fake_as_real}"
    )

    print(
        f"Total errors       : {len(errors)}"
    )

    print("\nSaved to:")

    print(
        "outputs/firefly_errors/real_as_fake/"
    )

    print(
        "outputs/firefly_errors/fake_as_real/"
    )

    print(
        "\nCSV:",
        csv_path
    )


if __name__ == "__main__":
    analyze_errors()