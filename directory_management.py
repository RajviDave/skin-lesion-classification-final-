
from pathlib import Path

folder_name = Path("PH2Dataset/PH2_DATASET_IMAGES")

subfolders = [f for f in folder_name.iterdir() if f.is_dir()]

for folder in subfolders:
    print(folder.name)

