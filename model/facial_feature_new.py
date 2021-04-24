import cv2
import pandas as pd
import os
import numpy as np
import joblib
import tqdm

import face_recognition
import random
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


data_path = "../data/input/data.csv"
df = pd.read_csv(data_path)
X = df.image_path.values

img_size = 256


def get_dlib_encoding(X, img_size):

    file_path_col = []
    face_locations_col = []
    num_faces_col = []
    face_landmarks_col = []
    face_encodings_col = []
    missing_img = []

    for i, img_path in tqdm.tqdm(enumerate(X), total=len(X)):
        # print(X)
        file_path = img_path
        im = Image.open(file_path)
        im = im.convert("RGB")
        im = im.resize((img_size, img_size))
        face_img = np.array(im)

        face_locations = face_recognition.face_locations(face_img)
        num_faces = len(face_locations)

        # if more than one faces are detected, only the first is returned by default
        if num_faces > 0:
            face_landmarks = face_recognition.face_landmarks(
                face_img, face_locations=face_locations, model="large"
            )[0]
            face_encodings = face_recognition.face_encodings(
                face_img, known_face_locations=face_locations, model="large"
            )[0]
            face_locations = face_locations[0]
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
        "file_path": file_path_col,
        "face_locations": face_locations_col,
        "num_faces": num_faces_col,
        "face_landmarks": face_landmarks_col,
        "face_encodings": face_encodings_col,
    }

    enc_df = pd.DataFrame.from_dict(encodings)
    enc_df.index = [path.split("/")[-1].split(".")[0] for path in enc_df["file_path"]]
    enc_df.to_pickle("../data/input/dlib_encodings.pickle")


def get_top_n(feature_array, n):
    boxes = feature_array[0]
    if len(boxes) == 0:
        # return np.asarray([[-1, -1, -1, -1]] * n)
        return np.nan
    detect_counts = feature_array[1].flatten()

    top_indices = detect_counts.argsort()[(-1) * n :][::-1]
    top_boxes = boxes[top_indices]

    return top_boxes


# TODO
def is_inside(feature_array, face):
    x, y, w, h = face_box
    # [y : y + h, x : x + w]


def get_haar_features(X, img_size):

    face_model = cv2.CascadeClassifier(
        "./haar_embeddings/haarcascade_frontalface_default.xml"
    )
    eye_model = cv2.CascadeClassifier("./haar_embeddings/haarcascade_eye.xml")
    nose_model = cv2.CascadeClassifier("./haar_embeddings/haarcascade_mcs_nose.xml")
    mouth_model = cv2.CascadeClassifier("./haar_embeddings/mouth2.xml")

    file_path_col = []
    num_faces_col = []
    face_col = []
    eyes_col = []
    nose_col = []
    mouth_col = []

    for img_path in X:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # returns a list of (x,y,w,h) tuples

        faces = face_model.detectMultiScale2(img, scaleFactor=1.1, minNeighbors=3)
        eyes = eye_model.detectMultiScale2(img, scaleFactor=1.1, minNeighbors=4)
        noses = nose_model.detectMultiScale2(img, scaleFactor=1.1, minNeighbors=6)
        mouths = mouth_model.detectMultiScale2(img, scaleFactor=1.1, minNeighbors=8)

        num_faces = len(faces[0])

        # get specified number of features
        face_box = get_top_n(faces, 1)
        eye_box = get_top_n(eyes, 2)
        nose_box = get_top_n(noses, 1)
        mouth_box = get_top_n(mouths, 1)

        file_path_col.append(file_path)
        num_faces_col.append(num_faces)
        face_col.append(face_box)
        eyes_col.append(eye_box)
        nose_col.append(nose_box)
        mouth_col.append(mouth_box)

    encodings = {
        "file_path": file_path_col,
        "num_faces": num_faces_col,
        "face": face_col,
        "eyes": eyes_col,
        "nose": nose_col,
        "mouth": mouth_col,
    }

    enc_df = pd.DataFrame.from_dict(encodings)
    enc_df.index = [path.split("/")[-1].split(".")[0] for path in enc_df["file_path"]]
    enc_df.to_pickle("../data/input/haar_encodings.pickle")

    # VISUALIZATION ##
    # out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # for (x, y, w, h) in faces[0]:
    #     cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # for (x, y, w, h) in noses[0]:
    #     cv2.rectangle(out_img, (x, y), (x + w, y + h), (255, 0, 255), 1)
    # for (x, y, w, h) in eyes[0]:
    #     cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # for (x, y, w, h) in mouths[0]:
    #     cv2.rectangle(out_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    # plt.figure(figsize=(12, 12))
    # plt.imshow(out_img)

