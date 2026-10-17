import os
import shutil
import random
import re
from collections import defaultdict

# ============================================================
# DERMACARE AI - GROUPED ANIMAL DATASET SPLIT
# ============================================================

SOURCE_DIR = r"C:\Users\meett\OneDrive\Desktop\DermaCare_AI\dataset\animal_combined"

OUTPUT_DIR = r"C:\Users\meett\OneDrive\Desktop\DermaCare_AI\dataset\animal_grouped_split"

TRAIN_RATIO = 0.80
RANDOM_SEED = 42

CLASSES = [
    "Bacterial_dermatosis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity_allergic_dermatosis"
]

random.seed(RANDOM_SEED)


print("=" * 70)
print("DERMACARE AI - GROUPED ANIMAL DATASET SPLIT")
print("=" * 70)

print()
print("Source:")
print(SOURCE_DIR)

print()
print("Output:")
print(OUTPUT_DIR)

print()
print("-" * 70)
print("PREPARING OUTPUT DIRECTORIES")
print("-" * 70)

# ------------------------------------------------------------
# Create directories
# ------------------------------------------------------------

for split in ["train", "validation"]:
    for class_name in CLASSES:
        path = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(path, exist_ok=True)

# ------------------------------------------------------------
# Detect animal group from filename
# ------------------------------------------------------------

def get_group_id(filename):

    base = os.path.splitext(filename)[0]

    match = re.match(
        r"^(?:additional_)?(Dog\d+|dog\d+)",
        base,
        re.IGNORECASE
    )

    if match:
        return match.group(1).lower()

    return None


# ------------------------------------------------------------
# Process each class
# ------------------------------------------------------------

grand_train = 0
grand_validation = 0

print()
print("=" * 70)
print("PROCESSING CLASSES")
print("=" * 70)

for class_name in CLASSES:

    class_source = os.path.join(SOURCE_DIR, class_name)

    if not os.path.exists(class_source):
        print(f"\nERROR: Class folder not found: {class_source}")
        continue

    print()
    print("-" * 70)
    print(f"CLASS: {class_name}")
    print("-" * 70)

    files = [
        f for f in os.listdir(class_source)
        if os.path.isfile(os.path.join(class_source, f))
    ]

    print(f"Total images: {len(files)}")

    # --------------------------------------------------------
    # Group images by animal ID
    # --------------------------------------------------------

    groups = defaultdict(list)
    other_files = []

    for filename in files:

        group_id = get_group_id(filename)

        if group_id is not None:
            groups[group_id].append(filename)
        else:
            other_files.append(filename)

    print(f"Identified animal groups: {len(groups)}")
    print(f"OTHER images: {len(other_files)}")

    # --------------------------------------------------------
    # Split animal groups
    # --------------------------------------------------------

    group_ids = list(groups.keys())

    random.shuffle(group_ids)

    target_train = int(len(files) * TRAIN_RATIO)

    train_groups = []
    validation_groups = []

    train_count = 0

    for group_id in group_ids:

        group_size = len(groups[group_id])

        # Keep entire animal group together
        if train_count + group_size <= target_train:
            train_groups.append(group_id)
            train_count += group_size
        else:
            validation_groups.append(group_id)

    # --------------------------------------------------------
    # If validation accidentally becomes too small,
    # move one complete group
    # --------------------------------------------------------

    if len(validation_groups) == 0 and len(train_groups) > 1:

        moved_group = train_groups.pop()

        train_count -= len(groups[moved_group])

        validation_groups.append(moved_group)

    # --------------------------------------------------------
    # Split OTHER files individually
    # --------------------------------------------------------

    random.shuffle(other_files)

    remaining_train_needed = max(
        0,
        target_train - train_count
    )

    other_train = other_files[:remaining_train_needed]
    other_validation = other_files[remaining_train_needed:]

    # --------------------------------------------------------
    # Build final file lists
    # --------------------------------------------------------

    train_files = []

    for group_id in train_groups:
        train_files.extend(groups[group_id])

    train_files.extend(other_train)

    validation_files = []

    for group_id in validation_groups:
        validation_files.extend(groups[group_id])

    validation_files.extend(other_validation)

    # --------------------------------------------------------
    # Copy files
    # --------------------------------------------------------

    train_destination = os.path.join(
        OUTPUT_DIR,
        "train",
        class_name
    )

    validation_destination = os.path.join(
        OUTPUT_DIR,
        "validation",
        class_name
    )

    for filename in train_files:

        source = os.path.join(class_source, filename)
        destination = os.path.join(train_destination, filename)

        shutil.copy2(source, destination)

    for filename in validation_files:

        source = os.path.join(class_source, filename)
        destination = os.path.join(validation_destination, filename)

        shutil.copy2(source, destination)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("Group split:")
    print(f"  Training groups:   {len(train_groups)}")
    print(f"  Validation groups: {len(validation_groups)}")

    print()
    print("Images:")
    print(f"  Training:          {len(train_files)}")
    print(f"  Validation:        {len(validation_files)}")
    print(f"  Total:             {len(train_files) + len(validation_files)}")

    grand_train += len(train_files)
    grand_validation += len(validation_files)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("GROUPED DATASET SPLIT COMPLETED")
print("=" * 70)

print()
print(f"Total Training Images:   {grand_train}")
print(f"Total Validation Images: {grand_validation}")

print()
print(f"Grand Total:             {grand_train + grand_validation}")

print()
print("Training Directory:")
print(os.path.join(OUTPUT_DIR, "train"))

print()
print("Validation Directory:")
print(os.path.join(OUTPUT_DIR, "validation"))

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print("""
Identified Dog groups were kept completely together.

Therefore:
    Same dog -> TRAIN
OR
    Same dog -> VALIDATION

The same identified dog will NOT appear in both sets.

OTHER images were split individually because their filenames
do not contain a detectable Dog ID.
""")

print("=" * 70)