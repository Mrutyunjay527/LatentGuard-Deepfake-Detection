import os
import random
import shutil

random.seed(42)

SOURCE_DIR = "data/labelled_deepfake"
OUTPUT_DIR = "data"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

classes = os.listdir(SOURCE_DIR)

for cls in classes:

    source_folder = os.path.join(SOURCE_DIR, cls)

    if not os.path.isdir(source_folder):
        continue

    images = os.listdir(source_folder)
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split_name, split_images in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):

        dst_folder = os.path.join(OUTPUT_DIR, split_name, cls)

        os.makedirs(dst_folder, exist_ok=True)

        for img in split_images:

            shutil.copy(
                os.path.join(source_folder, img),
                os.path.join(dst_folder, img)
            )

print("Dataset split completed successfully!")