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


def display_vector_per_category(label_data):
    for label, data in label_data.items():
        sample_image_vector = data.iloc[0, 1:].values
        sample_label = data.iloc[0, 0]

        # Reshape the image vector to a 28x28 format
        sample_image = sample_image_vector.reshape(28, 28)

        # Display the image
        plt.imshow(sample_image, cmap='gray')
        plt.title(f'Label: {sample_label}')
        plt.show()


def main():
    df = load_data(CSV_FILE_NAME)
    # Task 1
    label_data = separate_labels(df)
    display_vector_per_category(label_data)
    # Task 2


if __name__ == "__main__":
    main()
