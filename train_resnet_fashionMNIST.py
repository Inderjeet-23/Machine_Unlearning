import os
import subprocess

import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def load_example(df_row):
    image = torchvision.io.read_image(df_row["image_path"])

    result = {"image": image, "class": df_row["Class"]}
    return result


class HiddenDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        self.examples = []
        self.transform = None

        df = pd.read_csv(f"FashionMNIST/{split}_mini.csv")
        df["image_path"] = df["Image_File"].apply(
            lambda x: os.path.join("FashionMNIST/train/", x)
        )
        df = df.sort_values(by="image_path")
        df.apply(lambda row: self.examples.append(load_example(row)), axis=1)
        if len(self.examples) == 0:
            raise ValueError("No examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image = example["image"]
        image = image.to(torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        example["image"] = image
        return example


def get_dataset(batch_size):
    train_ds = HiddenDataset(split="train")
    retain_ds = HiddenDataset(split="retain")
    forget_ds = HiddenDataset(split="forget")

    retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True)
    forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    return retain_loader, forget_loader, train_loader


retain_loader, forget_loader, train_loader = get_dataset(64)


model = resnet18(pretrained=False, num_classes=10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for i, sample in enumerate(train_loader):

        inputs = sample["image"]
        labels = sample["class"]
        optimizer.zero_grad()

        # Make 3 channels for RGB
        inputs = torch.cat((inputs, inputs, inputs), 1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

checkpoint_path = "learned_resnet.pth"
torch.save(model.state_dict(), checkpoint_path)
