import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import *

# Dataset
from src.dataset import ImageDataset

# Robust Transforms
from src.transforms_robust import (
    train_transform_robust,
    val_transform_robust
)

# Encoders
from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder

# Main Model
from models.latent_guard import LatentGuard



# SET RANDOM SEED

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



# TRAINING FUNCTION

def train():

    print(f"Using Device : {DEVICE}")


    
    # DATASET
   
    train_dataset = ImageDataset(
        TRAIN_DIR,
        transform=train_transform_robust
    )

    val_dataset = ImageDataset(
        VAL_DIR,
        transform=val_transform_robust
    )

    print("Training Images :", len(train_dataset))
    print("Validation Images :", len(val_dataset))


   
    # DATA LOADERS

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


    
    # BUILD MODEL

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


   
    # LOSS FUNCTION

    criterion = nn.CrossEntropyLoss()


   
    # OPTIMIZER

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )


 
    # LEARNING RATE SCHEDULER

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )


   
    # RESUME SETTINGS

    start_epoch = 0

    best_loss = float("inf")

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []


 
    # RESUME FROM LATEST CHECKPOINT

    if os.path.exists(LATEST_CHECKPOINT_PATH):

        print(
            f"\nFound checkpoint: "
            f"{LATEST_CHECKPOINT_PATH}"
        )

        try:

            checkpoint = torch.load(
                LATEST_CHECKPOINT_PATH,
                map_location=DEVICE
            )


            # Load Model
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )


            # Load Optimizer
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )


            # Load Scheduler
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )


            # Resume From Next Epoch
            start_epoch = checkpoint["epoch"] + 1


            # Load Best Loss
            best_loss = checkpoint["best_loss"]


            # Load Training History
            train_losses = checkpoint.get(
                "train_losses",
                []
            )

            val_losses = checkpoint.get(
                "val_losses",
                []
            )

            train_accuracies = checkpoint.get(
                "train_accuracies",
                []
            )

            val_accuracies = checkpoint.get(
                "val_accuracies",
                []
            )


            print(
                "\nSuccessfully resumed training."
            )

            print(
                f"Next Epoch: "
                f"{start_epoch + 1}/{EPOCHS}"
            )


        except Exception as e:

            print(
                f"\nError loading checkpoint: {e}"
            )

            print(
                "Starting training from scratch."
            )


   
    # START TRAINING

    print("\nStarting Training Loop...")


   
    # TRAINING LOOP

    for epoch in range(start_epoch, EPOCHS):

        print(
            f"\n===== Starting Epoch "
            f"{epoch + 1}/{EPOCHS} ====="
        )


    
        # TRAINING MODE
    
        model.train()

        # Keep Encoders in Evaluation Mode
        image_encoder.eval()
        vae_encoder.eval()


        running_loss = 0.0

        correct = 0
        total = 0


      
        # TRAINING BATCHES
      
        for batch_idx, (images, labels) in enumerate(
            train_loader
        ):

            if batch_idx == 0:

                print(
                    "First batch loaded successfully"
                )


            if batch_idx % 100 == 0:

                print(
                    f"Batch "
                    f"{batch_idx}/{len(train_loader)}"
                )


            # Move Data To Device
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)


            # Reset Gradients
            optimizer.zero_grad()


            # Forward Pass
            outputs = model(images)


            # Calculate Loss
            loss = criterion(
                outputs,
                labels
            )


            # Backpropagation
            loss.backward()


            # Update Model
            optimizer.step()


            # Save Loss
            running_loss += loss.item()


            # Predictions
            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


        # Update Scheduler
        scheduler.step()


        # VALIDATION

        model.eval()


        val_running_loss = 0.0

        val_correct = 0
        val_total = 0


        with torch.no_grad():

            for images, labels in val_loader:


                images = images.to(DEVICE)
                labels = labels.to(DEVICE)


                # Forward Pass
                outputs = model(images)


                # Validation Loss
                loss = criterion(
                    outputs,
                    labels
                )


                val_running_loss += loss.item()


                # Predictions
                _, predicted = torch.max(
                    outputs,
                    1
                )


                val_total += labels.size(0)

                val_correct += (
                    predicted == labels
                ).sum().item()


        
        # CALCULATE METRICS

        epoch_loss = (
            running_loss / len(train_loader)
        )

        epoch_acc = (
            100 * correct / total
        )


        val_loss = (
            val_running_loss / len(val_loader)
        )

        val_acc = (
            100 * val_correct / val_total
        )


       
        # SAVE HISTORY

        train_losses.append(
            epoch_loss
        )

        val_losses.append(
            val_loss
        )

        train_accuracies.append(
            epoch_acc
        )

        val_accuracies.append(
            val_acc
        )


       
        # DISPLAY RESULTS
       
        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
            f" | Train Loss: {epoch_loss:.4f}"
            f" | Train Acc: {epoch_acc:.2f}%"
            f" | Val Loss: {val_loss:.4f}"
            f" | Val Acc: {val_acc:.2f}%"
        )


        # SAVE BEST MODEL

        if val_loss < best_loss:

            best_loss = val_loss


            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )


            print(
                "Best model saved!"
            )


    
        # CREATE CHECKPOINT

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_loss":
                best_loss,

            "train_losses":
                train_losses,

            "val_losses":
                val_losses,

            "train_accuracies":
                train_accuracies,

            "val_accuracies":
                val_accuracies,
        }



        # SAVE EPOCH-SPECIFIC CHECKPOINT

        epoch_checkpoint_path = os.path.join(

            CHECKPOINT_DIR,

            f"checkpoint_epoch_"
            f"{epoch + 1}.pth"

        )


        torch.save(
            checkpoint,
            epoch_checkpoint_path
        )


        print(
            f"Checkpoint saved: "
            f"{epoch_checkpoint_path}"
        )


       
        # UPDATE LATEST CHECKPOINT

        torch.save(
            checkpoint,
            LATEST_CHECKPOINT_PATH
        )


        print(
            "Latest checkpoint updated."
        )


      
        # SAVE CURRENT HISTORY
      
        history = {

            "train_loss":
                train_losses,

            "val_loss":
                val_losses,

            "train_accuracy":
                train_accuracies,

            "val_accuracy":
                val_accuracies,

        }


        torch.save(
            history,
            os.path.join(
                OUTPUT_DIR,
                "history_20epoch.pth"
            )
        )



    # TRAINING COMPLETE

    print(
        "\nTraining Finished Successfully!"
    )

    print(
        f"Best model saved at: "
        f"{MODEL_SAVE_PATH}"
    )


# MAIN

if __name__ == "__main__":

    try:

        train()


    except KeyboardInterrupt:

        print(
            "\n\nTraining stopped by user."
        )

        print(
            "The latest completed epoch "
            "checkpoint has been saved."
        )

        print(
            "Run the script again to "
            "continue training."
        )