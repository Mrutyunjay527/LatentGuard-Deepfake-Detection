import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.transform = transform
        self.images = []

        folders = os.listdir(root_dir)

        for folder in folders:

            folder_path = os.path.join(root_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            label = 0 if folder.lower() == "real" else 1

            for image_name in os.listdir(folder_path):

                image_path = os.path.join(folder_path, image_name)

                self.images.append((image_path, label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path, label = self.images[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label