import os
from PIL import Image

# ============================================================
# ANIMAL DATASET ANALYSIS
# ============================================================

DATASET_PATH = "dataset/animal"

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


print("\n" + "=" * 70)
print("DERMACARE AI - ANIMAL DATASET ANALYSIS")
print("=" * 70)

print("Dataset Path:")
print(os.path.abspath(DATASET_PATH))


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Animal dataset folder not found!")

    print(
        "Expected:",
        os.path.abspath(DATASET_PATH)
    )

    exit()


total_images = 0
total_invalid = 0


# ============================================================
# CHECK EACH CLASS
# ============================================================

for class_name in CLASSES:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    print("\n" + "-" * 70)
    print("CLASS:", class_name)
    print("-" * 70)

    if not os.path.exists(class_path):

        print("ERROR: Class folder not found!")

        continue


    image_files = []

    invalid_files = []


    # ========================================================
    # GET FILES
    # ========================================================

    for filename in os.listdir(class_path):

        file_path = os.path.join(
            class_path,
            filename
        )


        if not os.path.isfile(file_path):
            continue


        if not filename.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue


        image_files.append(file_path)


    print(
        "Total Image Files:",
        len(image_files)
    )


    # ========================================================
    # VALIDATE IMAGES
    # ========================================================

    valid_count = 0


    for image_path in image_files:

        try:

            with Image.open(image_path) as img:

                img.verify()


            valid_count += 1


        except Exception:

            invalid_files.append(
                image_path
            )


    invalid_count = len(
        invalid_files
    )


    print(
        "Valid Images:",
        valid_count
    )

    print(
        "Invalid/Corrupted Images:",
        invalid_count
    )


    # ========================================================
    # INVALID FILES
    # ========================================================

    if invalid_files:

        print("\nInvalid Files:")

        for file_path in invalid_files[:20]:

            print(
                os.path.basename(file_path)
            )


    total_images += valid_count

    total_invalid += invalid_count


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    "Total Valid Images:",
    total_images
)

print(
    "Total Invalid Images:",
    total_invalid
)

print(
    "Total Classes:",
    len(CLASSES)
)

print("\nClasses:")

for index, class_name in enumerate(CLASSES):

    print(
        index,
        "->",
        class_name
    )


print("\n" + "=" * 70)
print("DATASET ANALYSIS COMPLETED")
print("=" * 70)