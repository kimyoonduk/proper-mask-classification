import os
import pandas as pd
import numpy as np
import joblib
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

import seaborn as sns
from PIL import Image, ImageDraw


def generate_box(obj):
    xmin = int(obj.find("xmin").text)
    ymin = int(obj.find("ymin").text)
    xmax = int(obj.find("xmax").text)
    ymax = int(obj.find("ymax").text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    if obj.find("name").text == "with_mask":
        return 1
    elif obj.find("name").text == "mask_weared_incorrect":
        return 2
    return 0


def crop_face(datarow, newsize):

    box = datarow["box"]
    box_width = datarow["width"]
    box_height = datarow["height"]
    img_id = datarow["id"]

    if box_width > newsize:
        newsize = box_width
    elif box_height > newsize:
        newsize = box_height

    img_path = img_folder + img_id + ".png"
    with Image.open(img_path) as im:
        if im.width < newsize or im.height < newsize:
            print(f"{img_id} is too small")
            return (np.nan, (0, 0))

        offset_w = int((newsize - box_width) / 2)
        offset_h = int((newsize - box_height) / 2)

        pos_left = box[0] - offset_w
        pos_top = box[1] - offset_h
        pos_right = box[2] + offset_w
        pos_bot = box[3] + offset_h

        # calculate border positions
        limit_left = max(0, pos_left)
        limit_top = max(0, pos_top)
        limit_right = min(im.width, pos_right)
        limit_bot = min(im.height, pos_bot)

        # if one side is oob, compensate on the other side
        if limit_left == 0:
            limit_right = min(limit_right - pos_left, im.width)
        elif limit_right == im.width:
            limit_left = max(limit_left - (pos_right - im.width), 0)

        if limit_top == 0:
            limit_bot = min(limit_bot - pos_top, im.height)
        elif limit_bot == im.height:
            limit_top = max(limit_top - (pos_bot - im.height), 0)

        newbox = [limit_left, limit_top, limit_right, limit_bot]

        im = im.crop(newbox)

        # img_np = np.array(im)
        img_dim = im.size

    return (im, img_dim)


def main():

    anno_folder = "../data/kaggle/annotations/"
    img_folder = "../data/kaggle/images/"
    save_folder = "../data/kaggle/new/"

    anno_paths = sorted(list(paths.list_files(anno_folder)))
    anno_paths = [ap for ap in anno_paths if ap.endswith(".xml")]

    img_paths = sorted(list(paths.list_images(img_folder)))

    print(len(anno_paths))

    imgids = []
    labels = []
    boxes = []
    widths = []
    heights = []

    # no mask: 52
    # len([lb for lb in labels if lb == 0])

    for i, anno_path in tqdm(enumerate(anno_paths), total=len(anno_paths)):
        with open(anno_path) as f:
            data = f.read()
            soup = BeautifulSoup(data, "xml")
            objects = soup.find_all("object")
            imgid = re.match(r".+\/(.+)\.xml", anno_path)[1]

            for obj in objects:
                box = generate_box(obj)
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]

                imgids.append(imgid)
                widths.append(box_width)
                heights.append(box_height)
                labels.append(generate_label(obj))
                boxes.append(box)

    kaggle_dict = {
        "id": imgids,
        "label": labels,
        "box": boxes,
        "width": widths,
        "height": heights,
    }

    kaggle_df = pd.DataFrame.from_dict(kaggle_dict)

    # total number of faces in dataset
    len(kaggle_df)

    print("distribution of widths and heights for face area box")
    sns.distplot(kaggle_df["width"])
    sns.distplot(kaggle_df["height"])

    # filter incorrectly worn examples
    filtered = kaggle_df[kaggle_df["label"] != 2]

    # face should be large enough
    filtered = filtered[filtered["height"] >= 60]

    # photograph should contain only one person
    filtered = filtered[filtered["label"].duplicated(keep=False)]

    # TEMP: filtering out data from previous iteration
    # v1 = filtered[filtered["cropped_dim"] != (0, 0)]
    # v1_list = v1["id"]
    # filtered = filtered[~filtered["id"].isin(v1_list)]

    filtered.pivot_table(index="label", aggfunc=len)

    newsize = 200
    dim_list = []

    for idx, row in filtered.iterrows():
        new_img, img_dim = crop_face(row, newsize)
        if img_dim != (0, 0):
            new_img = new_img.convert("RGB")
            img_id = row["id"]
            label_str = "cw_" if row["label"] == 1 else "nm_"
            save_path = save_folder + label_str + img_id + ".jpg"
            new_img.save(save_path, "JPEG")
        dim_list.append(img_dim)

    filtered["cropped_dim"] = dim_list
    filtered[filtered["cropped_dim"] != (0, 0)].pivot_table(index="label", aggfunc=len)

    ## BOX TEST ##
    datarow = filtered.iloc[1]

    img_path = img_folder + datarow["id"] + ".png"
    im = Image.open(img_path)

    box = temp["box"]

    box_path = [
        (box[0], box[1]),
        (box[0], box[3]),
        (box[2], box[3]),
        (box[2], box[1]),
        (box[0], box[1]),
    ]

    d = ImageDraw.Draw(im, "RGBA")
    d.line(box_path, fill=(150, 0, 0, 180), width=2)
