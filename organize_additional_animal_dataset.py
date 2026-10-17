import os
import shutil

# ============================================================
# DERMACARE AI - ORGANIZE ADDITIONAL ANIMAL DATASET
# ============================================================

SOURCE_DIR = r"C:\Users\meett\Downloads\5dbht54kw7-1 (1)\5dbht54kw7-1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_additional"
)

LABEL_FILE = os.path.join(
    SOURCE_DIR,
    "image_label.txt"
)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

CLASSES = [
    "Bacterial_dermatosis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity_allergic_dermatosis"
]

print("\n" + "=" * 70)
print("DERMACARE AI - ORGANIZE ADDITIONAL ANIMAL DATASET")
print("=" * 70)

print("\nSource:")
print(SOURCE_DIR)

print("\nDestination:")
print(DEST_DIR)


# ============================================================
# CHECK SOURCE
# ============================================================

if not os.path.exists(SOURCE_DIR):
    print("\nERROR: Source dataset not found!")
    raise SystemExit

if not os.path.exists(LABEL_FILE):
    print("\nERROR: image_label.txt not found!")
    raise SystemExit


# ============================================================
# CREATE DESTINATION FOLDERS
# ============================================================

for class_name in CLASSES:

    class_dir = os.path.join(
        DEST_DIR,
        class_name
    )

    os.makedirs(
        class_dir,
        exist_ok=True
    )


# ============================================================
# READ LABEL FILE
# ============================================================

labels = {}

with open(
    LABEL_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        folder_name = parts[0]
        class_name = parts[1]

        labels[folder_name.lower()] = class_name


print("\nFolders found in label file:", len(labels))


# ============================================================
# COPY IMAGES
# ============================================================

class_counts = {
    class_name: 0
    for class_name in CLASSES
}

folders_processed = 0
folders_not_found = 0
images_copied = 0


for folder_name_lower, class_name in labels.items():

    actual_folder = None

    # Find actual folder ignoring upper/lower case
    for item in os.listdir(SOURCE_DIR):

        item_path = os.path.join(
            SOURCE_DIR,
            item
        )

        if (
            os.path.isdir(item_path)
            and item.lower() == folder_name_lower
        ):

            actual_folder = item_path
            break


    if actual_folder is None:

        print(
            "\nWARNING: Folder not found:",
            folder_name_lower
        )

        folders_not_found += 1
        continue


    folders_processed += 1

    destination_class_dir = os.path.join(
        DEST_DIR,
        class_name
    )


    # Find images inside folder
    image_files = []

    for root, dirs, files in os.walk(actual_folder):

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in VALID_EXTENSIONS:

                image_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )


    print(
        f"\n{os.path.basename(actual_folder)}"
        f" -> {class_name}"
        f" -> {len(image_files)} images"
    )


    # Copy images
    for image_path in image_files:

        original_name = os.path.basename(
            image_path
        )

        folder_base = os.path.basename(
            actual_folder
        )

        new_name = (
            folder_base
            + "_"
            + original_name
        )

        destination_path = os.path.join(
            destination_class_dir,
            new_name
        )


        # Avoid accidental overwrite
        counter = 1

        while os.path.exists(destination_path):

            name, extension = os.path.splitext(
                new_name
            )

            destination_path = os.path.join(
                destination_class_dir,
                f"{name}_{counter}{extension}"
            )

            counter += 1


        shutil.copy2(
            image_path,
            destination_path
        )

        class_counts[class_name] += 1
        images_copied += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ORGANIZATION SUMMARY")
print("=" * 70)

print(
    "\nFolders processed:",
    folders_processed
)

print(
    "Folders not found:",
    folders_not_found
)

print(
    "Total images copied:",
    images_copied
)

print("\nImages by class:")

for class_name in CLASSES:

    print(
        f"{class_name}: "
        f"{class_counts[class_name]}"
    )


print("\nDestination:")
print(
    os.path.abspath(
        DEST_DIR
    )
)

print("\n" + "=" * 70)
print("ORGANIZATION COMPLETED")
print("=" * 70)