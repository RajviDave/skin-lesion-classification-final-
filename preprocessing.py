import pandas as pd
import os

images={"Images":os.listdir("DATASET/dermoscopic")}
df=pd.DataFrame(images)
data=pd.read_excel('PH2Dataset/PH2_dataset.xlsx')
print(data)
common_nevu=data[2]
atypical_nevu=data[3]
melanoma=data[4]

print(common_nevu[50])