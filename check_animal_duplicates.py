import os
import hashlib
from collections import defaultdict

TRAIN_DIR = r"dataset\animal_clean_split\train"
VAL_DIR = r"dataset\animal_clean_split\validation"


def get_hash(filepath):
    hasher = hashlib.md5()

    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def collect_hashes(directory):
    hashes = defaultdict(list)

    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)

            try:
                file_hash = get_hash(filepath)
                hashes[file_hash].append(filepath)
            except Exception as e:
                print(f"ERROR: {filepath}")
                print(e)

    return hashes


print("=" * 70)
print("DERMACARE AI - ANIMAL DATASET DUPLICATE AUDIT")
print("=" * 70)

print("\nScanning TRAIN dataset...")
train_hashes = collect_hashes(TRAIN_DIR)

print("Scanning VALIDATION dataset...")
val_hashes = collect_hashes(VAL_DIR)


# ---------------------------------------------------------
# DUPLICATES INSIDE TRAIN
# ---------------------------------------------------------

train_duplicates = {
    h: files for h, files in train_hashes.items()
    if len(files) > 1
}


# ---------------------------------------------------------
# DUPLICATES INSIDE VALIDATION
# ---------------------------------------------------------

val_duplicates = {
    h: files for h, files in val_hashes.items()
    if len(files) > 1
}


# ---------------------------------------------------------
# SAME IMAGE IN TRAIN + VALIDATION
# ---------------------------------------------------------

cross_duplicates = []

for file_hash in train_hashes:
    if file_hash in val_hashes:

        cross_duplicates.append(
            (
                file_hash,
                train_hashes[file_hash],
                val_hashes[file_hash]
            )
        )


print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"\nUnique TRAIN image hashes:      {len(train_hashes)}")
print(f"Unique VALIDATION image hashes: {len(val_hashes)}")

print(f"\nDuplicate groups inside TRAIN:       {len(train_duplicates)}")
print(f"Duplicate groups inside VALIDATION:  {len(val_duplicates)}")

print(f"\nExact duplicates TRAIN <-> VALIDATION: {len(cross_duplicates)}")


# ---------------------------------------------------------
# SHOW TRAIN DUPLICATES
# ---------------------------------------------------------

if train_duplicates:

    print("\n" + "-" * 70)
    print("DUPLICATES INSIDE TRAIN")
    print("-" * 70)

    for i, files in enumerate(train_duplicates.values(), 1):

        print(f"\nGroup {i}:")

        for file in files:
            print(f"  {file}")


# ---------------------------------------------------------
# SHOW VALIDATION DUPLICATES
# ---------------------------------------------------------

if val_duplicates:

    print("\n" + "-" * 70)
    print("DUPLICATES INSIDE VALIDATION")
    print("-" * 70)

    for i, files in enumerate(val_duplicates.values(), 1):

        print(f"\nGroup {i}:")

        for file in files:
            print(f"  {file}")


# ---------------------------------------------------------
# SHOW CROSS DUPLICATES
# ---------------------------------------------------------

if cross_duplicates:

    print("\n" + "-" * 70)
    print("CRITICAL: SAME IMAGE IN TRAIN AND VALIDATION")
    print("-" * 70)

    for i, (_, train_files, val_files) in enumerate(cross_duplicates, 1):

        print(f"\nGroup {i}")

        print("TRAIN:")
        for file in train_files:
            print(f"  {file}")

        print("VALIDATION:")
        for file in val_files:
            print(f"  {file}")

else:

    print("\n" + "=" * 70)
    print("GOOD NEWS")
    print("=" * 70)
    print("\nNo exact duplicate image exists between TRAIN and VALIDATION.")


print("\n" + "=" * 70)
print("DUPLICATE AUDIT COMPLETED")
print("=" * 70)