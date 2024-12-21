import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def process_mask(mask_file):
    mask_dir = "ai4mars-dataset-merged-0.1/msl/labels/train"  # Define mask directory inside the function
    mask_path = os.path.join(mask_dir, mask_file)

    mask = cv2.imread(mask_path)

    if mask is None:
        print(f"Failed to load mask: {mask_file}")
        return set()

    reshaped_mask = mask.reshape(-1, 3)
    return set(map(tuple, reshaped_mask))


if __name__ == "__main__":
    mask_dir = "ai4mars-dataset-merged-0.1/msl/labels/train"

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

    unique_colors = set()

    print("Processing masks to find unique RGB values...")
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(process_mask, mask_files), total=len(mask_files), desc="Masks Processed", unit="file"))

        for color_set in results:
            unique_colors.update(color_set)

    unique_colors = sorted(unique_colors)

    print("\nUnique RGB values found in all masks:")
    for color in unique_colors:
        print(color)
