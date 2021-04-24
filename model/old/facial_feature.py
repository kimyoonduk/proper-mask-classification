import pandas as pd
import os
import numpy as np
import joblib
import tqdm

import face_recognition
import random
import torch
from PIL import Image, ImageDraw


def seed_everything(SEED=1337):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


def main():

    # set variables
    SEED = 1337
    seed_everything(SEED=SEED)
    target_ct = 2000
    img_size = 256
    test_size = 0.2
    batch_size = 32
    data_path = "../data/input/data.csv"

    # set computation device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Computation device: {device}")

    # read the data.csv file and get the image paths and labels
    df = pd.read_csv(data_path)
    # X = df.image_path.values
    # y = df.target.values
    # lb = joblib.load("../data/input/lb.pkl")
    # print(lb.classes_)

    X = df.image_path.values

    file_path_col = []
    face_locations_col = []
    num_faces_col = []
    face_landmarks_col = []
    face_encodings_col = []
    missing_img = []

    for i, img_path in tqdm.tqdm(enumerate(X), total=len(X)):
        # print(X)
        # file_path = img_dir + img_path.split('/img/')[1]
        file_path = img_path
        try:
            face_img = face_recognition.load_image_file(file_path)
        except FileNotFoundError:
            print(f"File Not Found: {file_path}")
            missing_img.append(file_path)
        face_locations = face_recognition.face_locations(face_img)
        num_faces = len(face_locations)

        if num_faces > 0:
            face_landmarks = face_recognition.face_landmarks(
                face_img, face_locations=face_locations, model="large"
            )[0]
            face_encodings = face_recognition.face_encodings(
                face_img, known_face_locations=face_locations, model="large"
            )[0]
            if num_faces > 1:
                print(f"More than one face: {file_path}, defaulting to first..")
        else:
            face_landmarks = []
            face_encodings = []
            print(f"No faces detected: {file_path}")

        file_path_col.append(file_path)
        face_locations_col.append(face_locations)
        num_faces_col.append(num_faces)
        face_landmarks_col.append(face_landmarks)
        face_encodings_col.append(face_encodings)

    encodings = {
        "file_path_col": file_path_col,
        "face_locations_col": face_locations_col,
        "num_faces_col": num_faces_col,
        "face_landmarks_col": face_landmarks_col,
        "face_encodings_col": face_encodings_col,
    }

    enc_df = pd.DataFrame.from_dict(encodings)
    enc_df.to_pickle("/content/drive/MyDrive/cis522/encodings.pickle")

    ## VISUALIZATION ##

    correct_mask = X[y == 1]

    # example: no recognized face
    # sunglass_and_mask = correct_mask[1]

    mask_img = face_recognition.load_image_file(correct_mask[2])
    pil_image = Image.fromarray(mask_img)
    d = ImageDraw.Draw(pil_image, "RGBA")

    for feature in face_landmarks.keys():
        if feature in ["chin", "nose_bridge"]:
            d.line(face_landmarks[feature], fill=(150, 0, 0, 180), width=5)
        else:
            d.polygon(face_landmarks[feature], fill=(0, 180, 0, 128))
