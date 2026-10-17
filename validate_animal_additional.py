import os
from PIL import Image
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ADDITIONAL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_additional"
)

CLASSES = [
    "Bacterial_dermatosis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity_allergic_dermatosis"
]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def calculate_hash(image_path):
    """Calculate file hash for duplicate detection."""

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


print("\n" + "=" * 70)
print("DERMACARE AI - ADDITIONAL ANIMAL DATASET VALIDATION")
print("=" * 70)

print("\nDataset:")
print(os.path.abspath(ADDITIONAL_DIR))


if not os.path.exists(ADDITIONAL_DIR):

    print("\nERROR: Additional dataset folder not found!")

    raise SystemExit


all_hashes = set()

total_valid = 0
total_invalid = 0
total_duplicates = 0


for class_name in CLASSES:

    class_dir = os.path.join(
        ADDITIONAL_DIR,
        class_name
    )

    print("\n" + "-" * 70)
    print("CLASS:", class_name)
    print("-" * 70)

    if not os.path.exists(class_dir):

        print("Folder missing!")

        continue

    files = os.listdir(class_dir)

    image_files = [
        file for file in files
        if os.path.splitext(file)[1].lower()
        in VALID_EXTENSIONS
    ]

    print(
        "Image Files:",
        len(image_files)
    )

    valid = 0
    invalid = 0
    duplicates = 0

    for filename in image_files:

        image_path = os.path.join(
            class_dir,
            filename
        )

        try:

            # ---------------------------------
            # IMAGE CHECK
            # ---------------------------------

            with Image.open(image_path) as image:

                image.verify()

            # ---------------------------------
            # DUPLICATE CHECK
            # ---------------------------------

            image_hash = calculate_hash(
                image_path
            )

            if image_hash in all_hashes:

                duplicates += 1
                total_duplicates += 1

                print(
                    "DUPLICATE:",
                    filename
                )

                continue

            all_hashes.add(image_hash)

            valid += 1
            total_valid += 1

        except Exception as error:

            invalid += 1
            total_invalid += 1

            print(
                "INVALID:",
                filename,
                "|",
                error
            )

    print(
        "Valid:",
        valid
    )

    print(
        "Invalid:",
        invalid
    )

    print(
        "Duplicates:",
        duplicates
    )


print("\n" + "=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

print(
    "Total Valid:",
    total_valid
)

print(
    "Total Invalid:",
    total_invalid
)

print(
    "Total Duplicates:",
    total_duplicates
)

print("\n" + "=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)