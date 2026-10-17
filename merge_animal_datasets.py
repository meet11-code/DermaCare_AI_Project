import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ORIGINAL_DIR = os.path.join(
    BASE_DIR, "dataset", "animal"
)

ADDITIONAL_DIR = os.path.join(
    BASE_DIR, "dataset", "animal_additional"
)

COMBINED_DIR = os.path.join(
    BASE_DIR, "dataset", "animal_combined"
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

print("\n" + "=" * 70)
print("DERMACARE AI - MERGE ANIMAL DATASETS")
print("=" * 70)

print("\nOriginal:")
print(ORIGINAL_DIR)

print("\nAdditional:")
print(ADDITIONAL_DIR)

print("\nCombined:")
print(COMBINED_DIR)


# ------------------------------------------------------------
# CHECK SOURCE DATASETS
# ------------------------------------------------------------

if not os.path.exists(ORIGINAL_DIR):
    print("\nERROR: Original animal dataset not found!")
    raise SystemExit

if not os.path.exists(ADDITIONAL_DIR):
    print("\nERROR: Additional animal dataset not found!")
    raise SystemExit


# ------------------------------------------------------------
# CREATE COMBINED DIRECTORIES
# ------------------------------------------------------------

os.makedirs(COMBINED_DIR, exist_ok=True)

for class_name in CLASSES:

    os.makedirs(
        os.path.join(
            COMBINED_DIR,
            class_name
        ),
        exist_ok=True
    )


# ------------------------------------------------------------
# COPY FUNCTION
# ------------------------------------------------------------

def copy_images(source_dir, dataset_prefix):

    copied = 0

    for class_name in CLASSES:

        source_class = os.path.join(
            source_dir,
            class_name
        )

        destination_class = os.path.join(
            COMBINED_DIR,
            class_name
        )

        if not os.path.exists(source_class):

            print(
                f"\nWARNING: Missing class: "
                f"{source_class}"
            )

            continue


        for filename in os.listdir(source_class):

            source_file = os.path.join(
                source_class,
                filename
            )

            if not os.path.isfile(source_file):
                continue


            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in VALID_EXTENSIONS:
                continue


            # Add prefix so files from different
            # datasets cannot overwrite each other
            new_filename = (
                dataset_prefix
                + "_"
                + filename
            )

            destination_file = os.path.join(
                destination_class,
                new_filename
            )


            shutil.copy2(
                source_file,
                destination_file
            )

            copied += 1


        print(
            f"{dataset_prefix} -> "
            f"{class_name}: "
            f"{len(os.listdir(source_class))} files"
        )


    return copied


# ------------------------------------------------------------
# COPY ORIGINAL DATASET
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("COPYING ORIGINAL DATASET")
print("-" * 70)

original_count = copy_images(
    ORIGINAL_DIR,
    "original"
)


# ------------------------------------------------------------
# COPY ADDITIONAL DATASET
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("COPYING ADDITIONAL DATASET")
print("-" * 70)

additional_count = copy_images(
    ADDITIONAL_DIR,
    "additional"
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MERGE COMPLETED")
print("=" * 70)

print(
    "\nOriginal images copied:",
    original_count
)

print(
    "Additional images copied:",
    additional_count
)

print(
    "Total images:",
    original_count + additional_count
)


print("\nImages by class:")

total = 0

for class_name in CLASSES:

    class_dir = os.path.join(
        COMBINED_DIR,
        class_name
    )

    count = len([
        f for f in os.listdir(class_dir)
        if os.path.isfile(
            os.path.join(class_dir, f)
        )
    ])

    total += count

    print(
        f"{class_name}: {count}"
    )


print("\nFinal total:", total)

print("\nCombined dataset:")
print(
    os.path.abspath(
        COMBINED_DIR
    )
)

print("\n" + "=" * 70)