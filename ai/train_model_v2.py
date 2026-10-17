import os
import random
import shutil
import tensorflow as tf

# ============================================================
# DERMACARE AI - CORRECT 6 CLASS DATASET SPLIT
# ============================================================

print("=" * 60)
print("DERMACARE AI - DATASET SPLIT V2")
print("=" * 60)

# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "human"
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "human_split"
)

TRAIN_DIR = os.path.join(
    SPLIT_DIR,
    "train"
)

VAL_DIR = os.path.join(
    SPLIT_DIR,
    "validation"
)

# ============================================================
# SETTINGS
# ============================================================

CLASSES = [
    "Acne",
    "Eczema",
    "Healthy",
    "Melanoma",
    "Psoriasis",
    "Ringworm"
]

VALIDATION_RATIO = 0.20
SEED = 42

random.seed(SEED)

# ============================================================
# PRINT PATHS
# ============================================================

print("\nSource Dataset:")
print(SOURCE_DIR)

print("\nTraining Dataset:")
print(TRAIN_DIR)

print("\nValidation Dataset:")
print(VAL_DIR)

# ============================================================
# CHECK SOURCE DATASET
# ============================================================

if not os.path.exists(SOURCE_DIR):

    print("\nERROR: Source dataset not found!")

    exit()

print("\nSource dataset found successfully!")

# ============================================================
# REMOVE OLD SPLIT DATASET
# ============================================================

if os.path.exists(SPLIT_DIR):

    print("\nRemoving old split dataset...")

    shutil.rmtree(
        SPLIT_DIR
    )

# ============================================================
# CREATE TRAIN AND VALIDATION FOLDERS
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
            VAL_DIR,
            class_name
        ),
        exist_ok=True
    )

# ============================================================
# SPLIT DATASET
# ============================================================

print("\n" + "=" * 60)
print("SPLITTING DATASET")
print("=" * 60)

for class_name in CLASSES:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    train_class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    val_class_dir = os.path.join(
        VAL_DIR,
        class_name
    )

    if not os.path.exists(
        source_class_dir
    ):

        print(
            f"\nWARNING: {class_name} folder not found!"
        )

        continue

    # Get image files only

    files = []

    for filename in os.listdir(
        source_class_dir
    ):

        file_path = os.path.join(
            source_class_dir,
            filename
        )

        if os.path.isfile(
            file_path
        ):

            files.append(
                filename
            )

    # Shuffle files

    random.shuffle(
        files
    )

    # Calculate validation count

    validation_count = int(
        len(files) *
        VALIDATION_RATIO
    )

    validation_files = files[
        :validation_count
    ]

    train_files = files[
        validation_count:
    ]

    # Copy training files

    for filename in train_files:

        source_path = os.path.join(
            source_class_dir,
            filename
        )

        destination_path = os.path.join(
            train_class_dir,
            filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )

    # Copy validation files

    for filename in validation_files:

        source_path = os.path.join(
            source_class_dir,
            filename
        )

        destination_path = os.path.join(
            val_class_dir,
            filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )

    # Print results

    print(
        f"\n{class_name}"
    )

    print(
        f"Total Images      : {len(files)}"
    )

    print(
        f"Training Images   : {len(train_files)}"
    )

    print(
        f"Validation Images : {len(validation_files)}"
    )

# ============================================================
# VERIFY DATASET
# ============================================================

print("\n" + "=" * 60)
print("DATASET SPLIT COMPLETED")
print("=" * 60)

print("\nTraining Dataset:")

total_train = 0

for class_name in CLASSES:

    class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    count = len(
        os.listdir(
            class_dir
        )
    )

    total_train += count

    print(
        f"{class_name}: {count}"
    )

print(
    f"\nTotal Training Images: {total_train}"
)

print("\nValidation Dataset:")

total_val = 0

for class_name in CLASSES:

    class_dir = os.path.join(
        VAL_DIR,
        class_name
    )

    count = len(
        os.listdir(
            class_dir
        )
    )

    total_val += count

    print(
        f"{class_name}: {count}"
    )

print(
    f"\nTotal Validation Images: {total_val}"
)

print("\n" + "=" * 60)
print("DATASET READY FOR TRAINING")
print("=" * 60)