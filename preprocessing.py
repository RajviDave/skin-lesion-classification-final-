import pandas as pd
import os

images={"Images":os.listdir("DATASET/dermoscopic")}
df=pd.DataFrame(images)
data=pd.read_excel('PH2Dataset/PH2_dataset.xlsx',skiprows=12)
