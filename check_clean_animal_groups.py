import os
import re
from collections import defaultdict

TRAIN_DIR = r"dataset\animal_clean_split\train"
VAL_DIR = r"dataset\animal_clean_split\validation"


def find_dog_group(filename):
    """
    Detect Dog ID from filenames such as:
    Dog210612
    dog210612
    additional_Dog210612_03_01_13_pic1
    """

    match = re.search(r"(?:^|_)(dog\d+)", filename, re.IGNORECASE)

    if match:
        return match.group(1).lower()

    return None


def collect_groups(directory):
    groups = defaultdict(list)

    for root, _, files in os.walk(directory):

        for file in files:

            group = find_dog_group(os.path.splitext(file)[0])

            if group:
                filepath = os.path.join(root, file)
                groups[group].append(filepath)

    return groups


print("=" * 70)
print("DERMACARE AI - CLEAN ANIMAL DOG GROUP LEAKAGE AUDIT")
print("=" * 70)

print("\nScanning CLEAN TRAIN dataset...")
train_groups = collect_groups(TRAIN_DIR)

print("Scanning CLEAN VALIDATION dataset...")
val_groups = collect_groups(VAL_DIR)


# ---------------------------------------------------------
# FIND GROUPS PRESENT IN BOTH DATASETS
# ---------------------------------------------------------

common_groups = sorted(
    set(train_groups.keys()) & set(val_groups.keys())
)


# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"\nDetected TRAIN dog groups:      {len(train_groups)}")
print(f"Detected VALIDATION dog groups: {len(val_groups)}")

print(f"\nDog groups in BOTH datasets:    {len(common_groups)}")


# ---------------------------------------------------------
# SHOW LEAKAGE
# ---------------------------------------------------------

if common_groups:

    print("\n" + "-" * 70)
    print("CRITICAL: SAME DOG EXISTS IN TRAIN AND VALIDATION")
    print("-" * 70)

    for i, group in enumerate(common_groups, 1):

        print(f"\nGroup {i}: {group}")

        print("TRAIN:")

        for file in train_groups[group]:
            print(f"  {file}")

        print("VALIDATION:")

        for file in val_groups[group]:
            print(f"  {file}")

else:

    print("\n" + "=" * 70)
    print("GOOD NEWS")
    print("=" * 70)

    print("\nNo detected dog group exists in both TRAIN and VALIDATION.")


print("\n" + "=" * 70)
print("DOG GROUP LEAKAGE AUDIT COMPLETED")
print("=" * 70)