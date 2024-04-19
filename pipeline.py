# import the necessary packages
from pyimagesearch.helpers import convert_and_trim_bb
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import face_recognition as fr
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torchvision
import torch

# classifier = resnet18(pretrained=True)
# classifier.fc = nn.Linear(512, 7)
# classifier.load_state_dict(torch.load("faces_resnet.pth"))

from facenet_pytorch import InceptionResnetV1

classifier = classifier = InceptionResnetV1(
    pretrained="vggface2", classify=True, num_classes=7
)
classifier.load_state_dict(torch.load("faces_resnet_vgg.pth"))

import pandas as pd

df = pd.read_csv("faces_database.csv")
names = df["Name"]
labels = df["Label"]
label_name_dict = {}
for i in range(len(names)):
    label_name_dict[labels[i]] = names[i]

print("[INFO] loading CNN face detector...")
model = "mmod_human_face_detector.dat"
upsample = 0
detector = dlib.cnn_face_detection_model_v1(model)

p = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

i = 0
preds = []
while i < 10:
    ret, frame = cap.read()
    image = frame
    image = imutils.resize(image, width=500)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start = time.time()
    print("[INFO[ performing face detection with dlib...")
    results = detector(rgb, upsample)
    end = time.time()
    print("[INFO] face detection took {:.4f} seconds".format(end - start))

    boxes = [convert_and_trim_bb(image, r.rect) for r in results]
    face_images = []

    for x, y, w, h in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    count = 0
    for face in results:
        shape = predictor(image, face.rect)
        main_landmarks = {}
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(
                image,
                str(i),
                (x, y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.5,
                (0, 255, 0),
                1,
            )
            if i == 36:
                main_landmarks["left_eye1"] = (x, y)
            elif i == 39:
                main_landmarks["left_eye2"] = (x, y)
            elif i == 42:
                main_landmarks["right_eye1"] = (x, y)
            elif i == 45:
                main_landmarks["right_eye2"] = (x, y)

        lef_eye_center = np.array(
            [
                int(
                    (main_landmarks["left_eye1"][0] + main_landmarks["left_eye2"][0])
                    / 2
                ),
                int(
                    (main_landmarks["left_eye1"][1] + main_landmarks["left_eye2"][1])
                    / 2
                ),
            ]
        )

        right_eye_center = np.array(
            [
                int(
                    (main_landmarks["right_eye1"][0] + main_landmarks["right_eye2"][0])
                    / 2
                ),
                int(
                    (main_landmarks["right_eye1"][1] + main_landmarks["right_eye2"][1])
                    / 2
                ),
            ]
        )

        angle = np.degrees(
            np.arctan2(
                right_eye_center[1] - lef_eye_center[1],
                right_eye_center[0] - lef_eye_center[0],
            )
        )
        desired_left_eye = (0.35, 0.35)
        desired_right_eye_x = 1.0 - desired_left_eye[0]

        dist = np.sqrt(
            (right_eye_center[0] - lef_eye_center[0]) ** 2
            + (right_eye_center[1] - lef_eye_center[1]) ** 2
        )
        desiredDist = desired_right_eye_x - desired_left_eye[0]
        desiredDist *= 256
        scale = desiredDist / dist

        eyesCenter = (
            (lef_eye_center[0] + right_eye_center[0]) // 2,
            (lef_eye_center[1] + right_eye_center[1]) // 2,
        )
        eyesCenter = (int(eyesCenter[0]), int(eyesCenter[1]))

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        tX = 256 * 0.5
        tY = 256 * 0.35

        M[0, 2] += tX - eyesCenter[0]
        M[1, 2] += tY - eyesCenter[1]

        print(face)

        (w, h) = (256, 256)

        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        image_to_classify = torch.from_numpy(image)
        image_to_classify = image_to_classify.permute(2, 0, 1)

        image_to_classify = torchvision.transforms.functional.resize(
            image_to_classify, (224, 224)
        )
        image_to_classify = image_to_classify[0:3]
        image_to_classify = image_to_classify.to(torch.float32)
        classifier.eval()
        out = classifier(image_to_classify.unsqueeze(0))
        probs = F.softmax(out, dim=1)
        probs = sorted(probs[0].tolist(), reverse=True)
        delta = probs[0] - probs[1]
        _, predicted = torch.max(out, 1)

        if delta > 0.005:
            preds.append(predicted.item())
        else:
            preds.append(10)
    i += 1

preds = np.array(preds)
# Take the mode of the predictions
names = []
for i in range(len(preds)):
    if preds[i] == 10:
        names.append("Unknown")
    else:
        names.append(label_name_dict[preds[i]])
mode = np.bincount(preds).argmax()
if mode == 10:
    print("Person Unknown")
else:
    print("Are you : ", label_name_dict[mode])

print("The names are: ", names)

cap.release()
cv2.destroyAllWindows()
