import numpy as np
import pandas as pd
import os

# Create a DataFrame of all images in the folder 'Faces' with their corresponding image names and labels

df = pd.DataFrame(columns=["Image", "Name", "Label"])

# Path to the folder containing the images
path = "Faces/"

# List of all the images in the folder
images = os.listdir(path)

# Loop through all the images and add them to the DataFrame
cnt = -1
name_dict = {}
for image in images:
    name = image.split(".")[0]
    name = "".join(filter(str.isalpha, name))
    if name in name_dict:
        label = cnt
    else:
        name_dict[name] = cnt
        cnt += 1
        label = cnt
    df = df.append({"Image": image, "Name": name, "Label": label}, ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv("Faces/Train_Faces.csv", index=False)

# Create forget df where there is only label == 2
Forget_df = df[df["Label"] == 2]
Forget_df.to_csv("Faces/Forget_Faces.csv", index=False)

Retained_df = df[df["Label"] != 2]
Retained_df.to_csv("Faces/Retained_Faces.csv", index=False)
