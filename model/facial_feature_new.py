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


def get_dlib_encoding(X):

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
        face_img = face_recognition.load_image_file(file_path)
        face_img = face_img.resize((img_size, img_size))

        face_locations = face_recognition.face_locations(face_img)
        num_faces = len(face_locations)

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
        "file_path_col": file_path_col,
        "face_locations_col": face_locations_col,
        "num_faces_col": num_faces_col,
        "face_landmarks_col": face_landmarks_col,
        "face_encodings_col": face_encodings_col,
    }

    enc_df = pd.DataFrame.from_dict(encodings)
    enc_df.to_pickle("../data/input/dlib_encodings.pickle")


def get_haar_features(X):

    face_model = cv2.CascadeClassifier(
        "./haar_embeddings/haarcascade_frontalface_default.xml"
    )
    eye_model = cv2.CascadeClassifier("./haar_embeddings/haarcascade_eye.xml")
    nose_model = cv2.CascadeClassifier("./haar_embeddings/haarcascade_mcs_nose.xml")
    mouth_model = cv2.CascadeClassifier("./haar_embeddings/mouth2.xml")
    smile_model = cv2.CascadeClassifier("./haar_embeddings/haarcascade_smile.xml")

    features = []
    img_path = X[20]

    # for img_path in X:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # returns a list of (x,y,w,h) tuples
    faces = face_model.detectMultiScale2(img, scaleFactor=1.1, minNeighbors=3)

    eyes = eye_model.detectMultiScale2(img, scaleFactor=1.1, minNeighbors=4)
    noses = nose_model.detectMultiScale2(img, scaleFactor=1.1, minNeighbors=6)
    mouths = mouth_model.detectMultiScale2(img, scaleFactor=1.1, minNeighbors=8)

    # features.append((faces, eyes, noses, mouths))

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
    #     roi_gray = gray[y : y + h, x : x + w]
    #     smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for (x, y, w, h) in faces[0]:
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    for (x, y, w, h) in noses[0]:
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (255, 0, 255), 1)
    for (x, y, w, h) in eyes[0]:
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    for (x, y, w, h) in mouths[0]:
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    plt.figure(figsize=(12, 12))
    plt.imshow(out_img)

    joblib.dump(features, "../data/input/haar_features.pkl")
