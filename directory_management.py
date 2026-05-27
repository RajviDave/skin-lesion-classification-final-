import os

# path="PH2Dataset\PH2_DATASET_IMAGES"
# folders=os.listdir(path)

path="PH2Dataset/PH2_DATASET_IMAGES"
destination_path="DATASET"

dermo_path=os.path.join(destination_path,"dermoscopic")
lesion_path=os.path.join(destination_path,"lesion")

os.makedirs(dermo_path,exist_ok=True)
os.makedirs(lesion_path,exist_ok=True)

