# import the necessary packages
from pyimagesearch.helpers import convert_and_trim_bb
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import face_recognition as fr
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str, required=True,
# 	help="path to input image")
# ap.add_argument("-m", "--model", type=str,
# 	default="mmod_human_face_detector.dat",
# 	help="path to dlib's CNN face detector model")
# ap.add_argument("-u", "--upsample", type=int, default=1,
# 	help="# of times to upsample")
# args = vars(ap.parse_args())

# load dlib's CNN face detector
print("[INFO] loading CNN face detector...")
model = "mmod_human_face_detector.dat"
upsample =0 
detector = dlib.cnn_face_detection_model_v1(model)

p = "shape_predictor_68_face_landmarks.dat"
predictor  = dlib.shape_predictor(p)

# load the input image from disk, resize it, and convert it from
# BGR to RGB channel ordering (which is what dlib expects)
# image = cv2.imread(args["image"])
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image = frame
    # image = cv2.imread("images/man.jpg")
    image = imutils.resize(image, width=500)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # perform face detection using dlib's face detector
    start = time.time()
    print("[INFO[ performing face detection with dlib...")
    results = detector(rgb, upsample)
    end = time.time()
    print("[INFO] face detection took {:.4f} seconds".format(end - start))

    # convert the resulting dlib rectangle objects to bounding boxes,
    # then ensure the bounding boxes are all within the bounds of the
    # input image
    boxes = [convert_and_trim_bb(image, r.rect) for r in results]
    face_images = []
    # loop over the bounding boxes
    for (x, y, w, h) in boxes:
        # draw the bounding box on our image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # image = imutils.resize(image[y:y+h, x:x+w], width=256)
    # '''Affine transformation of the face'''

    # cv2.affine

    '''Face landmarks'''
    #Using shape_predictor_68_face_landmarks.dat model, mark the facial landmarks on the face
    

    # main_landmarks = {}
    count = 0 
    for face in results:
        shape = predictor(image, face.rect)
        main_landmarks = {}
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)
            if i == 36:
                main_landmarks["left_eye1"] = (x,y)
            elif i == 39:
                main_landmarks["left_eye2"] = (x,y)
            elif i == 42:
                main_landmarks["right_eye1"] = (x,y)
            elif i == 45:
                main_landmarks["right_eye2"] = (x,y)
        # main_landmarks[face] = main_landmarks
        
        #Left eye center
        lef_eye_center = np.array([int((main_landmarks["left_eye1"][0] + main_landmarks["left_eye2"][0])/2), int((main_landmarks["left_eye1"][1] + main_landmarks["left_eye2"][1])/2)])
        #Right eye center
        right_eye_center = np.array([int((main_landmarks["right_eye1"][0] + main_landmarks["right_eye2"][0])/2), int((main_landmarks["right_eye1"][1] + main_landmarks["right_eye2"][1])/2)])

        #angle between the eyes
        angle = np.degrees(np.arctan2(right_eye_center[1] - lef_eye_center[1], right_eye_center[0] - lef_eye_center[0])) 
        #desired left eye x-coordinate
        desired_left_eye = (0.35, 0.35)
        #desired left eye x-coordinate
        desired_right_eye_x = 1.0 - desired_left_eye[0]

        dist = np.sqrt((right_eye_center[0] - lef_eye_center[0])**2 + (right_eye_center[1] - lef_eye_center[1])**2)
        desiredDist = (desired_right_eye_x - desired_left_eye[0])
        desiredDist *= 256
        scale = desiredDist / dist

        eyesCenter = ((lef_eye_center[0] + right_eye_center[0])//2, (lef_eye_center[1] + right_eye_center[1])//2)
        eyesCenter = (int(eyesCenter[0]), int(eyesCenter[1]))

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        tX = 256 * 0.5
        tY = 256 * 0.35

        M[0, 2] += tX - eyesCenter[0]
        M[1, 2] += tY - eyesCenter[1]

        print(face)

        (w,h) = (256,256)

        output = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC)

        #show the output image
        # cv2.imshow("Input", image)
        cv2.imshow("Output"+str(count), output)
        count += 1
        # cv2.waitKey(0)

    # show the output image
    cv2.imshow("Input", image)
    


    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()