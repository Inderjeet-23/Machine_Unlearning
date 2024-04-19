import cv2
import face_recognition
import pickle
import os 

#load all the images from the folder images/*/* and encode them

#load the images
imagePaths = list()
for root, dirs, files in os.walk("images/archive/lfw_funneled"):
    for file in files:
        if file.endswith("0001.jpg"):
            imagePaths.append(os.path.join(root, file))
print(imagePaths)

known_embeddings = list()
known_names = list()

# #Check the code for first 10 images: 
# imagePaths = imagePaths[:3]
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        known_embeddings.append(encoding)
        known_names.append(name)
data = {"embeddings": known_embeddings, "names": known_names}
f = open("output/embeddings.p", "wb")
f.write(pickle.dumps(data))
f.close()
