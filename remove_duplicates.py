import os
import hashlib
import shutil

# ============================================================
# DERMACARE AI - SAFE DUPLICATE IMAGE REMOVER
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "new_data"
)

DUPLICATE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "duplicates_removed"
)

# Disease classes
CLASSES = [
    "Acne",
    "Eczema",
    "Melanoma",
    "Psoriasis",
    "Ringworm"
]

# Create duplicate backup folder
os.makedirs(
    DUPLICATE_DIR,
    exist_ok=True
)

print("=" * 60)
print("DERMACARE AI - DUPLICATE IMAGE CHECK")
print("=" * 60)

total_images = 0
unique_images = 0
duplicate_images = 0

# ============================================================
# CHECK EACH DISEASE SEPARATELY
# ============================================================

for class_name in CLASSES:

    source_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    duplicate_class_dir = os.path.join(
        DUPLICATE_DIR,
        class_name
    )

    os.makedirs(
        duplicate_class_dir,
        exist_ok=True
    )

    if not os.path.exists(source_dir):

        print(
            f"\nWARNING: {class_name} folder not found!"
        )

        continue

    print("\n" + "-" * 60)
    print(
        f"Checking: {class_name}"
    )
    print("-" * 60)

    seen_hashes = set()

    class_total = 0
    class_unique = 0
    class_duplicates = 0

    files = os.listdir(
        source_dir
    )

    for filename in files:

        file_path = os.path.join(
            source_dir,
            filename
        )

        # Only process files
        if not os.path.isfile(
            file_path
        ):
            continue

        try:

            # Read image
            with open(
                file_path,
                "rb"
            ) as f:

                file_hash = hashlib.md5(
                    f.read()
                ).hexdigest()

            class_total += 1
            total_images += 1

            # Check duplicate
            if file_hash in seen_hashes:

                class_duplicates += 1
                duplicate_images += 1

                # Move duplicate to backup folder
                destination = os.path.join(
                    duplicate_class_dir,
                    filename
                )

                # Avoid filename collision
                counter = 1

                while os.path.exists(
                    destination
                ):

                    name, extension = os.path.splitext(
                        filename
                    )

                    destination = os.path.join(
                        duplicate_class_dir,
                        f"{name}_duplicate_{counter}{extension}"
                    )

                    counter += 1

                shutil.move(
                    file_path,
                    destination
                )

            else:

                seen_hashes.add(
                    file_hash
                )

                class_unique += 1
                unique_images += 1

        except Exception as e:

            print(
                f"Error processing {filename}: {e}"
            )

    print(
        f"Total      : {class_total}"
    )

    print(
        f"Unique     : {class_unique}"
    )

    print(
        f"Duplicates : {class_duplicates}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE CHECK COMPLETED")
print("=" * 60)

print(
    f"\nTotal Images Checked : {total_images}"
)

print(
    f"Unique Images        : {unique_images}"
)

print(
    f"Duplicate Images     : {duplicate_images}"
)

print(
    "\nDuplicate images were MOVED to:"
)

print(
    DUPLICATE_DIR
)

print("\n" + "=" * 60)
print("SAFE CLEANUP COMPLETED")
print("=" * 60)