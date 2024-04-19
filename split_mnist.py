# Split train.csv into forget.csv and retain.csv

import pandas as pd
import numpy as np
import os

# # Read train.csv
# train = pd.read_csv("FashionMNIST/data/train.csv")

# # Split train.csv into forget.csv and retain.csv
# forget = train.sample(frac=0.5, random_state=0)
# retain = train.drop(forget.index)

# # Save forget.csv and retain.csv
# forget.to_csv("FashionMNIST/data/forget.csv", index=False)
# retain.to_csv("FashionMNIST/data/retain.csv", index=False)

# Read train.csv
train = pd.read_csv("MiniFashionMNIST/train.csv")

# Sort train.csv by Image_File
train = train.sort_values(by=["Image_File"])

# Drop all rows after 1000 rows
train = train.iloc[:1200]

# First 1000 are train images and rest are val images
val = train.iloc[1000:1200]
train = train.iloc[:1000]

# Split the dataset with forget set having only class 9 and 7 and retain set having all other classes

# forget = train[(train["Class"] == 9) | (train["Class"] == 7)]
# retain = train[(train["Class"] != 9) & (train["Class"] != 7)]

# Randomly split the dataset into forget and retain
forget = train.sample(frac=0.2, random_state=0)
retain = train.drop(forget.index)

# Save train.csv
train.to_csv("MiniFashionMNIST/train_mini.csv", index=False)
val.to_csv("MiniFashionMNIST/val_mini.csv", index=False)

# # Split train.csv into forget.csv and retain.csv
# forget = train.sample(frac=0.2, random_state=0)
# retain = train.drop(forget.index)

# Save forget.csv and retain.csv
forget.to_csv("MiniFashionMNIST/forget_mini.csv", index=False)
retain.to_csv("MiniFashionMNIST/retain_mini.csv", index=False)
