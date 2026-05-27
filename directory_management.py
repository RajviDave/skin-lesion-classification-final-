import os
import shutil

path="PH2Dataset/PH2_DATASET_IMAGES"
destination_path="DATASET"

dermo_path=os.path.join(destination_path,"dermoscopic")
lesion_path=os.path.join(destination_path,"lesion")

os.makedirs(dermo_path,exist_ok=True)
os.makedirs(lesion_path,exist_ok=True)

image_folders=os.listdir(path)
for image_folders1 in image_folders:
    path1=os.path.join(path,image_folders1)
    for image_folder2 in path1:
        path2=os.path.join(path1,image_folder2)
        print(path2)
