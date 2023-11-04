"""
Student:        Cian Herlihy - R00205604
Title:          Supervised Learning
Due Date:       Friday, 17th November 2023
"""

import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_NAME = "fashion-mnist_train.csv"


def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df


def separate_labels(df):
    label_dfs_dict = {}
    unique_labels = df['label'].unique()

    for label in unique_labels:
        label_dfs_dict[label] = df[df['label'] == label]

    return label_dfs_dict


def main():
    df = load_data(CSV_FILE_NAME)
    label_data = separate_labels(df)
    first_label = label_data[0]
    sample_image_vector = first_label.iloc[0, 1:].values  # Assuming label 0 for tshirt/top
    sample_label = first_label.iloc[0, 0]  # Assuming label is in the first column

    # Reshape the image vector to a 28x28 format
    sample_image = sample_image_vector.reshape(28, 28)

    # Display the image
    plt.imshow(sample_image, cmap='gray')
    plt.title(f'Label: {sample_label}')
    plt.show()


if __name__ == "__main__":
    main()
