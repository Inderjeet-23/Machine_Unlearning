import numpy as np
import tensorflow as tf
from keras import layers, Model
from keras import backend as K
import keras
import random
import os
import cv2
# Define the triplet loss function
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
    return loss

# Define the CNN architecture
def create_base_network(input_shape):
    base_model = keras.applications.MobileNetV2(input_shape=input_shape,
                                                    include_top=False,
                                                    weights='imagenet')
    base_model.trainable = False
    global_average_layer = layers.GlobalAveragePooling2D()
    return Model(inputs=base_model.input, outputs=global_average_layer(base_model.output))

# Define the triplet network
def create_triplet_network(input_shape):
    anchor_input = layers.Input(shape=input_shape, name='anchor_input')
    positive_input = layers.Input(shape=input_shape, name='positive_input')
    negative_input = layers.Input(shape=input_shape, name='negative_input')

    base_network = create_base_network(input_shape)

    anchor_embedding = base_network(anchor_input)
    positive_embedding = base_network(positive_input)
    negative_embedding = base_network(negative_input)

    triplet_output = layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=-1)
    return Model(inputs=[anchor_input, positive_input, negative_input], outputs=triplet_output)

# Generate triplets from dataset
def generate_triplets(images, labels, num_triplets=10000):
    triplets = []
    for _ in range(num_triplets):
        # Select anchor
        anchor_idx = np.random.choice(np.where(labels == np.random.choice(labels))[0])
        anchor = images[anchor_idx]
        # Select positive
        positive_idx = np.random.choice(np.where(labels == labels[anchor_idx])[0])
        positive = images[positive_idx]
        # Select negative
        negative_idx = np.random.choice(np.where(labels != labels[anchor_idx])[0])
        negative = images[negative_idx]
        triplets.append((anchor, positive, negative))
    return np.array(triplets)

def load_images_from_directory(root_dir):
    # Initialize empty lists to store images and labels
    images = []
    labels = []
    count = 0 
    # Iterate through each person's directory
    for person_dir in os.listdir(root_dir):
        person_path = os.path.join(root_dir, person_dir)
        if os.path.isdir(person_path):
            label = count
            # Iterate through each image in the person's directory
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                # Load image and append to images list
                images.append(img_path)
                # Append corresponding label to labels list
                labels.append(label)
            count += 1

    return images, labels

def preprocess_image(image):
    resized_image = cv2.resize(image, (256, 256))
    # Convert the image to float32 and normalize pixel values to range [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0
    # Optionally, you can perform other preprocessing steps such as mean subtraction or standardization
    
    return normalized_image

#Traverse through the dataset and generate triplets
images,labels = load_images_from_directory("lfw_funneled")

images = np.array([cv2.imread(image) for image in images])
images = np.array([preprocess_image(image) for image in images])

labels = np.array(labels)

triplets = generate_triplets(images, labels)

input_shape = images.shape[1:]

triplet_network = create_triplet_network(input_shape)
triplet_network.compile(optimizer=keras.optimizers.legacy.Adam(), loss=triplet_loss)

# Train the model
triplet_network.fit([triplets[:, 0], triplets[:, 1], triplets[:, 2]], np.zeros(triplets.shape[0]), epochs=10, batch_size=32)
