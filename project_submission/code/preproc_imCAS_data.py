import glob
import os
import numpy as np
np.random.seed(42)


all_paths = sorted(glob.glob("/cluster/work/felixzr/TDT4265_StarterCode_2024/data/imageCAS/*"))
paths = np.random.choice(all_paths, size=150, replace=False)

target_path = "/cluster/work/felixzr/TDT4265_StarterCode_2024/data_flat_ImageCAS"

for i, path in enumerate(paths):
    name = path.split("/")[-1]

    os.system(f"cp -r {path}/img.nii.gz {target_path}/{name}_img.nii.gz")
    os.system(f"cp -r {path}/label.nii.gz {target_path}/{name}_label.nii.gz")

os.system("gunzip /cluster/work/felixzr/TDT4265_StarterCode_2024/data_flat_ImageCAS/*.gz")


import json

image_paths = [target_path+"/"+i.split("/")[-1]+"_img.nii" for i in paths]
label_paths = [target_path+"/"+i.split("/")[-1]+"_label.nii" for i in paths]

data = {"training": [], "testing": []}
num_training = int(0.8 * len(image_paths))      # train/val/test = 0.7/0.1/0.2

# Add the training data to the dictionary
for i in range(num_training):
    data["training"].append({
        "image": image_paths[i],
        "label": label_paths[i]
    })

# Add the testing data to the dictionary
for i in range(num_training, len(image_paths)):
    data["testing"].append({
        "image": image_paths[i],
        "label": label_paths[i]
    })

# write to JSON file
with open('/cluster/work/felixzr/TDT4265_StarterCode_2024/project/model1_auto3dseg/data_ImageCAS.json', 'w') as f:
    json.dump(data, f, indent=4)
