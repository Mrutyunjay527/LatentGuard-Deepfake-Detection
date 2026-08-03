import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import *

# Dataset
from src.dataset import ImageDataset
from src.transforms import train_transform, val_transform

# Encoders
from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder

# Main Model
from models.latent_guard import LatentGuard


# Set Random Seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def train():

    print(f"Using Device : {DEVICE}")

    
    # Dataset
    train_dataset = ImageDataset(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = ImageDataset(
        VAL_DIR,
        transform=val_transform
    )
    

    print("Training Images :", len(train_dataset))
    
    print("Validation Images :", len(val_dataset))

    # data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )


    # Build Model
    print("Creating Image Encoder...")
    image_encoder = ImageEncoder().to(DEVICE)
    image_encoder.eval()
    print("Image Encoder Loaded")

    print("Creating VAE Encoder...")
    vae_encoder = VAEEncoder().to(DEVICE)
    vae_encoder.eval()
    print("VAE Encoder Loaded")

    print("Building LatentGuard...")

    model = LatentGuard(
        image_encoder=image_encoder,
        vae_encoder=vae_encoder
    ).to(DEVICE)

    print("LatentGuard Ready")

  
    # Loss
    criterion = nn.CrossEntropyLoss()

    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

   
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_loss = float("inf")

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    print("Starting Training Loop...")
    
    # Training Loop
    for epoch in range(EPOCHS):
        print(f"\n===== Starting Epoch {epoch+1} =====")

        model.train()

        image_encoder.eval()
        vae_encoder.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            
            if batch_idx == 0:
                    print("First batch loaded successfully")

            if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)}")       

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        scheduler.step()

       
        # Validation
        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        epoch_loss = running_loss / len(train_loader)

        epoch_acc = 100 * correct / total

        # Save history
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        train_accuracies.append(epoch_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS}"
            f" | Train Loss: {epoch_loss:.4f}"
            f" | Train Acc: {epoch_acc:.2f}%"
            f" | Val Loss: {val_loss:.4f}"
            f" | Val Acc: {val_acc:.2f}%"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print("Best model saved!")


        # Save Training History
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies
    }

    torch.save(history, "outputs/history.pth")

    
    # Display Training History
    # print("\nTraining History")
    # print(train_losses)
    # print(val_losses)

    print("\nTraining Finished Successfully!")


if __name__ == "__main__":
    train()