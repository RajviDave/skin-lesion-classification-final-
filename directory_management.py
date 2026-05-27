import os
import shutil

path = "PH2Dataset/PH2_DATASET_IMAGES"
destination_path = "DATASET"

dermo_path = os.path.join(destination_path, "dermoscopic")
lesion_path = os.path.join(destination_path, "lesion")

os.makedirs(dermo_path, exist_ok=True)
os.makedirs(lesion_path, exist_ok=True)

image_folders = os.listdir(path)

for folder1 in image_folders:

    folder1_path = os.path.join(path, folder1)

    # IMD002_Dermoscopic_Image etc
    subfolders = os.listdir(folder1_path)

    for folder2 in subfolders:

        folder2_path = os.path.join(folder1_path, folder2)

        # image inside folder
        images = os.listdir(folder2_path)

        for image in images:

            image_path = os.path.join(folder2_path, image)

            if "Dermoscopic" in folder2:
                shutil.copy(image_path,
                            os.path.join(dermo_path, image))

            elif "lesion" in folder2:
                shutil.copy(image_path,
                            os.path.join(lesion_path, image))

print("Done")