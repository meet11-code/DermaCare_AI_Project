import os
import shutil
import random

# ============================================================
# ANIMAL DATASET TRAIN / VALIDATION SPLIT
# ============================================================

SOURCE_DIR = "dataset/animal"
OUTPUT_DIR = "dataset/animal_split"

TRAIN_RATIO = 0.80

CLASSES = [
    "Bacterial_dermatosis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity_allergic_dermatosis"
]

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

RANDOM_SEED = 42


# ============================================================
# START
# ============================================================

print("\n" + "=" * 70)
print("DERMACARE AI - ANIMAL DATASET SPLIT")
print("=" * 70)

print(
    "Source:",
    os.path.abspath(SOURCE_DIR)
)

print(
    "Output:",
    os.path.abspath(OUTPUT_DIR)
)


# ============================================================
# CHECK SOURCE
# ============================================================

if not os.path.exists(SOURCE_DIR):

    print("\nERROR: Source dataset not found!")

    exit()


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for split in ["train", "validation"]:

    for class_name in CLASSES:

        folder = os.path.join(
            OUTPUT_DIR,
            split,
            class_name
        )

        os.makedirs(
            folder,
            exist_ok=True
        )


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_name in CLASSES:

    print("\n" + "-" * 70)
    print("CLASS:", class_name)
    print("-" * 70)

    source_class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    if not os.path.exists(source_class_dir):

        print(
            "ERROR: Class folder not found:",
            source_class_dir
        )

        continue


    # ========================================================
    # GET IMAGES
    # ========================================================

    images = []

    for filename in os.listdir(
        source_class_dir
    ):

        file_path = os.path.join(
            source_class_dir,
            filename
        )

        if not os.path.isfile(file_path):
            continue

        if not filename.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue

        images.append(filename)


    # ========================================================
    # SHUFFLE
    # ========================================================

    random.shuffle(images)


    # ========================================================
    # SPLIT
    # ========================================================

    train_count = int(
        len(images) * TRAIN_RATIO
    )

    train_images = images[:train_count]

    validation_images = images[train_count:]


    print(
        "Total:",
        len(images)
    )

    print(
        "Training:",
        len(train_images)
    )

    print(
        "Validation:",
        len(validation_images)
    )


    # ========================================================
    # COPY TRAINING IMAGES
    # ========================================================

    train_dir = os.path.join(
        OUTPUT_DIR,
        "train",
        class_name
    )

    for filename in train_images:

        source = os.path.join(
            source_class_dir,
            filename
        )

        destination = os.path.join(
            train_dir,
            filename
        )

        shutil.copy2(
            source,
            destination
        )


    # ========================================================
    # COPY VALIDATION IMAGES
    # ========================================================

    validation_dir = os.path.join(
        OUTPUT_DIR,
        "validation",
        class_name
    )

    for filename in validation_images:

        source = os.path.join(
            source_class_dir,
            filename
        )

        destination = os.path.join(
            validation_dir,
            filename
        )

        shutil.copy2(
            source,
            destination
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL DATASET SPLIT COMPLETED")
print("=" * 70)

print(
    "Output Directory:",
    os.path.abspath(OUTPUT_DIR)
)

print("\nNext Step:")
print("Animal AI model training")

print("=" * 70)