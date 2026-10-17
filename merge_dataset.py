import os
import shutil

# ============================================================
# DERMACARE AI - MERGE NEW DATASET
# Healthy class will NOT be changed
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Existing original dataset
SOURCE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "human"
)

# New cleaned dataset
NEW_DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "new_data"
)

# Classes to merge
# Healthy intentionally excluded
CLASSES = [
    "Acne",
    "Eczema",
    "Melanoma",
    "Psoriasis",
    "Ringworm"
]

print("=" * 60)
print("DERMACARE AI - MERGING NEW DATASET")
print("=" * 60)

print("\nExisting Dataset:")
print(SOURCE_DIR)

print("\nNew Dataset:")
print(NEW_DATA_DIR)

# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(SOURCE_DIR):
    print("\nERROR: Existing dataset not found!")
    exit()

if not os.path.exists(NEW_DATA_DIR):
    print("\nERROR: New dataset not found!")
    exit()

# ============================================================
# MERGE
# ============================================================

for class_name in CLASSES:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    new_class_dir = os.path.join(
        NEW_DATA_DIR,
        class_name
    )

    # Create original class folder if missing
    os.makedirs(
        source_class_dir,
        exist_ok=True
    )

    if not os.path.exists(new_class_dir):

        print(
            f"\nWARNING: {class_name} new data folder not found!"
        )

        continue

    files = [
        f
        for f in os.listdir(new_class_dir)
        if os.path.isfile(
            os.path.join(
                new_class_dir,
                f
            )
        )
    ]

    print("\n" + "-" * 60)
    print(f"Class: {class_name}")
    print(f"New Images: {len(files)}")

    copied = 0
    skipped = 0

    for index, filename in enumerate(files):

        source_path = os.path.join(
            new_class_dir,
            filename
        )

        # New unique filename
        name, extension = os.path.splitext(
            filename
        )

        new_filename = (
            f"new_{index:05d}_{name}{extension}"
        )

        destination_path = os.path.join(
            source_class_dir,
            new_filename
        )

        # Safety check
        if os.path.exists(destination_path):

            skipped += 1

            continue

        shutil.copy2(
            source_path,
            destination_path
        )

        copied += 1

    print(f"Copied : {copied}")
    print(f"Skipped: {skipped}")

# ============================================================
# FINAL DATASET COUNT
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET COUNTS")
print("=" * 60)

for class_name in [
    "Acne",
    "Eczema",
    "Healthy",
    "Melanoma",
    "Psoriasis",
    "Ringworm"
]:

    class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    if os.path.exists(class_dir):

        count = len([
            f
            for f in os.listdir(class_dir)
            if os.path.isfile(
                os.path.join(
                    class_dir,
                    f
                )
            )
        ])

        print(
            f"{class_name}: {count} images"
        )

print("\n" + "=" * 60)
print("DATASET MERGE COMPLETED")
print("=" * 60)

print(
    "\nIMPORTANT: Healthy class was NOT modified."
)
