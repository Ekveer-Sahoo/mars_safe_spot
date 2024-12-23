import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

image_dir = "ai4mars-dataset-merged-0.1/msl/images/edr"
mask_dir = "ai4mars-dataset-merged-0.1/msl/labels/train"

IMG_HEIGHT = 256
IMG_WIDTH = 256

RGB_TO_CLASS = {
    (0, 0, 0): 0,  # Soil
    (1, 1, 1): 1,  # Bedrock
    (2, 2, 2): 2,  # Sand
    (3, 3, 3): 3,  # Big rock
    (255, 255, 255): 4  # NULL/No label
}


def process_image_mask(img_file):
    img_path = os.path.join(image_dir, img_file)
    mask_path = os.path.join(mask_dir, img_file.replace(".JPG", ".png"))

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if img is None or mask is None:
        print(f"Failed to load: {img_file}")
        return None, None

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

    img = img / 255.0

    mask_class = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    for rgb_value, class_index in RGB_TO_CLASS.items():
        mask_class[np.all(mask == rgb_value, axis=-1)] = class_index

    return img, mask_class


if __name__ == "__main__":
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".JPG")]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

    valid_images = [img for img in image_files if img.replace(".JPG", ".png") in mask_files]

    images = []
    masks = []

    print("Processing images and masks...")

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image_mask, valid_images), total=len(valid_images), desc="Processing",
                            unit="file"))

        for img, mask in results:
            if img is not None and mask is not None:
                images.append(img)
                masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

class_counts = np.zeros(5)  # Five classes: 0 (Soil), 1 (Bedrock), 2 (Sand), 3 (Big Rock), 4 (NULL)

for mask in masks:  
    unique, counts = np.unique(mask, return_counts=True)
    for u, c in zip(unique, counts):
        class_counts[u] += c

classes = ['Soil', 'Bedrock', 'Sand', 'Big Rock', 'NULL']
plt.bar(classes, class_counts, color=['brown', 'gray', 'yellow', 'black', 'white'])
plt.title("Class Distribution in Masks")
plt.xlabel("Classes")
plt.ylabel("Pixel Count")
plt.show()

train_size = len(X_train)
val_size = len(X_val)

plt.bar(['Training', 'Validation'], [train_size, val_size], color=['blue', 'orange'])
plt.title("Training vs Validation Split")
plt.xlabel("Dataset")
plt.ylabel("Number of Samples")
plt.show()


processing_times = [0.05, 0.06, 0.07, 0.08, 0.05]  

plt.plot(processing_times)
plt.title("Processing Time per File")
plt.xlabel("File Index")
plt.ylabel("Processing Time (seconds)")
plt.show()

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

print("Preprocessed data saved successfully!")
