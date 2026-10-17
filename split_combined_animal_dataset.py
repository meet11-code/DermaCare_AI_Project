import os
import shutil
import random

# ============================================================
# DERMACARE AI - COMBINED ANIMAL DATASET SPLIT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_combined"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_combined_split"
)

TRAIN_DIR = os.path.join(
    OUTPUT_DIR,
    "train"
)

VALIDATION_DIR = os.path.join(
    OUTPUT_DIR,
    "validation"
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
    ".bmp",
    ".webp"
}

TRAIN_RATIO = 0.80
SEED = 42

random.seed(SEED)

print("\n" + "=" * 70)
print("DERMACARE AI - COMBINED ANIMAL DATASET SPLIT")
print("=" * 70)

print("\nSource:")
print(os.path.abspath(SOURCE_DIR))

print("\nOutput:")
print(os.path.abspath(OUTPUT_DIR))


# ============================================================
# CHECK SOURCE
# ============================================================

if not os.path.exists(SOURCE_DIR):

    print("\nERROR: Combined dataset not found!")
    raise SystemExit


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for class_name in CLASSES:

    os.makedirs(
        os.path.join(
            TRAIN_DIR,
            class_name
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            VALIDATION_DIR,
            class_name
        ),
        exist_ok=True
    )


# ============================================================
# SPLIT DATA
# ============================================================

total_train = 0
total_validation = 0

for class_name in CLASSES:

    class_source = os.path.join(
        SOURCE_DIR,
        class_name
    )

    if not os.path.exists(class_source):

        print(
            f"\nERROR: Missing class: {class_name}"
        )

        continue

    files = []

    for filename in os.listdir(class_source):

        file_path = os.path.join(
            class_source,
            filename
        )

        if not os.path.isfile(file_path):
            continue

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in VALID_EXTENSIONS:
            files.append(filename)

    random.shuffle(files)

    split_index = int(
        len(files) * TRAIN_RATIO
    )

    train_files = files[:split_index]
    validation_files = files[split_index:]

    # --------------------------------------------------------
    # COPY TRAINING FILES
    # --------------------------------------------------------

    for filename in train_files:

        source_file = os.path.join(
            class_source,
            filename
        )

        destination_file = os.path.join(
            TRAIN_DIR,
            class_name,
            filename
        )

        shutil.copy2(
            source_file,
            destination_file
        )

    # --------------------------------------------------------
    # COPY VALIDATION FILES
    # --------------------------------------------------------

    for filename in validation_files:

        source_file = os.path.join(
            class_source,
            filename
        )

        destination_file = os.path.join(
            VALIDATION_DIR,
            class_name,
            filename
        )

        shutil.copy2(
            source_file,
            destination_file
        )

    total_train += len(train_files)
    total_validation += len(validation_files)

    print("\n" + "-" * 70)
    print("CLASS:", class_name)
    print("-" * 70)

    print("Total:", len(files))
    print("Training:", len(train_files))
    print("Validation:", len(validation_files))


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMBINED ANIMAL DATASET SPLIT COMPLETED")
print("=" * 70)

print("\nTotal Training Images:", total_train)
print("Total Validation Images:", total_validation)

print(
    "\nGrand Total:",
    total_train + total_validation
)

print("\nTraining Directory:")
print(os.path.abspath(TRAIN_DIR))

print("\nValidation Directory:")
print(os.path.abspath(VALIDATION_DIR))

print("\n" + "=" * 70)
print("NEXT STEP: IMPROVED ANIMAL MODEL TRAINING")
print("=" * 70)