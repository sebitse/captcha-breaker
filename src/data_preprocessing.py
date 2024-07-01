import os
import cv2
import numpy as np
import pandas as pd

RECTANGLE_DIMENSION = (20, 20)


def preprocess_image(image_path):
    """
    Preprocesses a CAPTCHA image by converting it to grayscale,
    applying binary inversion thresholding, finding contours,
    and extracting each character as a separate image.

    Args:
    image_path (str): The file path to the CAPTCHA image.

    Returns:
    np.array: An array of character images.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rects = []

    for contour in contours:
        rect = cv2.boundingRect(contour)
        bounding_rects.append(rect)

    def sort_by_x(rect):
        return rect[0]

    bounding_boxes = sorted(bounding_rects, key=sort_by_x)
    characters = []

    for x, y, w, h in bounding_boxes:
        char = thresh[y:y + h, x:x + w]
        resized_char = cv2.resize(char, (20, 20))
        characters.append(resized_char)

    # Ensure exactly 5 characters per CAPTCHA image
    if len(characters) != 5:
        raise ValueError(f"Expected 5 characters, but found {len(characters)} in image {image_path}")

    return np.array(characters)

def load_data(data_dir, labels_dir):
    """
    Loads CAPTCHA images from the specified directory and their corresponding labels from the labels directory.

    Args:
    data_dir (str): The directory containing CAPTCHA images.
    labels_dir (str): The directory containing the labels.csv file.

    Returns:
    tuple: A tuple containing the feature array X and the label array y.
    """
    labels_csv_path = os.path.join(labels_dir, 'train_labels.csv')

    labels_df = pd.read_csv(labels_csv_path)

    X = []
    y = []

    for _, row in labels_df.iterrows():
        image_path = os.path.join(data_dir, row['filename'])
        label = row['label']

        # Ensure the label has exactly 5 characters
        if len(label) != 5:
            raise ValueError(f"Expected 5 characters in label, but found {len(label)} in label {label}")

        # Preprocess the image to extract characters
        characters = preprocess_image(image_path)
        X.extend(characters)

        # Convert label to numerical format
        # Assuming label is a string of alphanumeric characters
        for char in label:
            if char.isalpha():
                y.append(ord(char) - ord('A'))
            else:
                y.append(ord(char) - ord('0') + 26)

    X = np.array(X)
    y = np.array(y)
    return X, y


# Example usage
data_dir = '../data/train'
labels_dir = '../data/labels'
X_train, y_train = load_data(data_dir, labels_dir)

# Preprocessing the data
X_train = X_train.reshape(-1, 20, 20, 1).astype('float32') / 255

# Print shapes for verification
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)