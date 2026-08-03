import io
import numpy as np
from PIL import Image
from PIL import ImageFilter


def jpeg_compression(image, quality=50):

    buffer = io.BytesIO()

    image.save(buffer, format="JPEG", quality=quality)

    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


def gaussian_blur(image, radius=2):

    return image.filter(ImageFilter.GaussianBlur(radius))


def resize_image(image, down_size=(128, 128), final_size=(224, 224)):

    image = image.resize(down_size)

    image = image.resize(final_size)

    return image


def gaussian_noise(image, mean=0, std=15):

    img = np.array(image)

    noise = np.random.normal(mean, std, img.shape)

    noisy = img + noise

    noisy = np.clip(noisy, 0, 255)

    return Image.fromarray(noisy.astype(np.uint8))