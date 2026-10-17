import os
import shutil
import hashlib
import random
from pathlib import Path
from collections import defaultdict

# ============================================================
# DERMACARE AI
# ANIMAL DATASET - DUPLICATE CLEAN + GROUP SAFE SPLIT
# ============================================================

SOURCE = Path("dataset/animal_combined")
OUTPUT = Path("dataset/animal_clean_split")

TRAIN_DIR = OUTPUT / "train"
VAL_DIR = OUTPUT / "validation"

TRAIN_RATIO = 0.80
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = [
    "Bacterial_dermatosis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity_allergic_dermatosis"
]


# ============================================================
# IMAGE HASH
# ============================================================

def file_hash(path):
    """Return SHA256 hash of image file contents."""

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# DETECT DOG GROUP
# ============================================================

def detect_dog_group(filename):
    """
    Detect dog ID from filenames such as:

    additional_dog210612_03_01_13_pic0.jpg
    additional_Dog210619_01_01_44_pic5.jpg

    Returns:
        dog210612
        dog210619
        None if no detectable dog ID
    """

    name = Path(filename).stem.lower()

    # Search for dog followed by digits
    import re

    match = re.search(r"dog(\d+)", name)

    if match:
        return "dog" + match.group(1)

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DERMACARE AI - ANIMAL DATASET CLEANING")
    print("=" * 70)

    print()
    print("Source:")
    print(SOURCE.resolve())

    print()
    print("Output:")
    print(OUTPUT.resolve())

    # --------------------------------------------------------
    # CHECK SOURCE
    # --------------------------------------------------------

    if not SOURCE.exists():

        print()
        print("ERROR: Source dataset does not exist!")
        print(SOURCE.resolve())
        return

    # --------------------------------------------------------
    # REMOVE OLD OUTPUT
    # --------------------------------------------------------

    if OUTPUT.exists():

        print()
        print("Removing previous clean dataset...")

        shutil.rmtree(OUTPUT)

    TRAIN_DIR.mkdir(parents=True)
    VAL_DIR.mkdir(parents=True)

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total_original = 0
    total_unique = 0
    total_duplicates = 0

    total_train = 0
    total_validation = 0

    global_hashes = {}

    print()
    print("=" * 70)
    print("STEP 1 - FINDING UNIQUE IMAGES")
    print("=" * 70)

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_name in CLASSES:

        source_class = SOURCE / class_name

        if not source_class.exists():

            print()
            print("WARNING: Missing class:", class_name)

            continue

        print()
        print("-" * 70)
        print("CLASS:", class_name)
        print("-" * 70)

        files = [
            p for p in source_class.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print("Original images:", len(files))

        total_original += len(files)

        # ----------------------------------------------------
        # HASH GROUPS
        # ----------------------------------------------------

        hash_groups = defaultdict(list)

        for image_path in files:

            try:
                h = file_hash(image_path)
                hash_groups[h].append(image_path)

            except Exception as e:

                print()
                print("ERROR reading:")
                print(image_path)
                print("Reason:", e)

        unique_files = []

        duplicate_count = 0

        for h, group in hash_groups.items():

            # Keep first image only
            unique_files.append(group[0])

            if len(group) > 1:

                duplicate_count += len(group) - 1

        total_duplicates += duplicate_count
        total_unique += len(unique_files)

        print("Unique images:", len(unique_files))
        print("Exact duplicates removed:", duplicate_count)

        # ----------------------------------------------------
        # STORE UNIQUE FILES
        # ----------------------------------------------------

        # Structure:
        #
        # dog group -> list of images
        # OTHER -> list of images

        dog_groups = defaultdict(list)
        other_images = []

        for image_path in unique_files:

            dog_id = detect_dog_group(image_path.name)

            if dog_id:

                dog_groups[dog_id].append(image_path)

            else:

                other_images.append(image_path)

        print("Detected dog groups:", len(dog_groups))
        print("OTHER images:", len(other_images))

        # ----------------------------------------------------
        # SORT DOG GROUPS
        # ----------------------------------------------------

        groups = list(dog_groups.items())

        random.Random(SEED).shuffle(groups)

        # ----------------------------------------------------
        # GROUP-WISE SPLIT
        # ----------------------------------------------------

        target_train = int(len(unique_files) * TRAIN_RATIO)

        train_files = []
        validation_files = []

        # First allocate dog groups
        current_train = 0

        for dog_id, group_files in groups:

            if current_train < target_train:

                train_files.extend(group_files)
                current_train += len(group_files)

            else:

                validation_files.extend(group_files)

        # ----------------------------------------------------
        # HANDLE OTHER IMAGES
        # ----------------------------------------------------

        other_copy = list(other_images)

        random.Random(SEED).shuffle(other_copy)

        remaining_train = target_train - len(train_files)

        if remaining_train > 0:

            train_files.extend(other_copy[:remaining_train])
            validation_files.extend(other_copy[remaining_train:])

        else:

            validation_files.extend(other_copy)

        # ----------------------------------------------------
        # CREATE DIRECTORIES
        # ----------------------------------------------------

        train_class = TRAIN_DIR / class_name
        val_class = VAL_DIR / class_name

        train_class.mkdir(parents=True, exist_ok=True)
        val_class.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # COPY FILES
        # ----------------------------------------------------

        for index, src in enumerate(train_files):

            destination = train_class / src.name

            # Avoid filename collision
            if destination.exists():

                destination = train_class / f"copy_{index}_{src.name}"

            shutil.copy2(src, destination)

        for index, src in enumerate(validation_files):

            destination = val_class / src.name

            if destination.exists():

                destination = val_class / f"copy_{index}_{src.name}"

            shutil.copy2(src, destination)

        print()
        print("Training images:", len(train_files))
        print("Validation images:", len(validation_files))
        print("Total:", len(train_files) + len(validation_files))

        total_train += len(train_files)
        total_validation += len(validation_files)

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("CLEAN DATASET CREATED")
    print("=" * 70)

    print()
    print("Original images:", total_original)
    print("Unique images:", total_unique)
    print("Exact duplicates removed:", total_duplicates)

    print()
    print("Training images:", total_train)
    print("Validation images:", total_validation)

    print()
    print("Grand total:", total_train + total_validation)

    print()
    print("Training directory:")
    print(TRAIN_DIR.resolve())

    print()
    print("Validation directory:")
    print(VAL_DIR.resolve())

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print()
    print("Exact duplicate images were reduced to one copy.")

    print()
    print("Detected dog groups were kept together.")

    print()
    print("Therefore:")
    print("Same dog -> TRAIN")
    print("OR")
    print("Same dog -> VALIDATION")

    print()
    print("The original dataset was NOT modified.")

    print()
    print("=" * 70)
    print("CLEANING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()