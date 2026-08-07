import os
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = "outputs/firefly_errors"

REAL_FAKE_DIR = os.path.join(
    BASE_DIR,
    "real_as_fake"
)

FAKE_REAL_DIR = os.path.join(
    BASE_DIR,
    "fake_as_real"
)


def create_montage(
    folder,
    output_file,
    title,
    columns=5,
    image_size=180
):

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    if len(files) == 0:
        print("No images found:", folder)
        return

    rows = (len(files) + columns - 1) // columns

    title_height = 60

    canvas = Image.new(
        "RGB",
        (
            columns * image_size,
            rows * (image_size + 25) + title_height
        ),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 15),
        title,
        fill="black"
    )

    for i, filename in enumerate(files):

        path = os.path.join(
            folder,
            filename
        )

        try:

            image = Image.open(path).convert("RGB")

            image.thumbnail(
                (image_size - 10, image_size - 10)
            )

            x = (i % columns) * image_size
            y = (
                i // columns
            ) * (image_size + 25) + title_height

            image_x = (
                x + (image_size - image.width) // 2
            )

            image_y = (
                y + (image_size - image.height) // 2
            )

            canvas.paste(
                image,
                (image_x, image_y)
            )

            draw.text(
                (
                    x + 5,
                    y + image_size
                ),
                str(i + 1),
                fill="black"
            )

        except Exception as e:

            print(
                "Could not open:",
                filename,
                e
            )

    canvas.save(
        output_file,
        quality=95
    )

    print(
        "Saved:",
        output_file
    )


create_montage(
    REAL_FAKE_DIR,
    "outputs/firefly_real_as_fake_montage.jpg",
    "Real Images Misclassified as Fake"
)


create_montage(
    FAKE_REAL_DIR,
    "outputs/firefly_fake_as_real_montage.jpg",
    "Fake Images Misclassified as Real"
)