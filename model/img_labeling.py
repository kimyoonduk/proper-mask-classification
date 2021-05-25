import os
import pandas as pd
import numpy as np
import joblib
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

img_folder = "../data/img/"

# target count per class
target_ct = 100

image_paths = sorted(list(paths.list_images(img_folder)))

data = pd.DataFrame()
labels = []

cdict = {
    "c1-nomask": {"ct": 0, "label": "c1-nm"},
    "c2-correct": {"ct": 0, "label": "c2-cm"},
    "c3-incorrect": {"ct": 0, "label": "c3-im"},
}

# sample target_ct images for each class and remove the rest
for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
    cname = image_path.split(os.path.sep)[-2]
    ccount = cdict[cname]["ct"]

    if ccount < target_ct:
        labels.append(cdict[cname]["label"])
        cdict[cname]["ct"] += 1
        data.loc[i, "image_path"] = image_path
    else:
        os.remove(image_path)


labels = np.array(labels)
# one hot encode
lb = LabelBinarizer()
labels_mat = lb.fit_transform(labels)

print(f"Total instances: {len(labels_mat)}")
for i in range(len(labels_mat)):
    index = np.argmax(labels_mat[i])
    data.loc[i, "target"] = int(index)

# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# print count for each label
print(data.pivot_table(index="target", aggfunc=len))

# save as csv file
data.to_csv("../data/input/data.csv", index=False)
# pickle the label binarizer
joblib.dump(lb, "../data/input/lb.pkl")
