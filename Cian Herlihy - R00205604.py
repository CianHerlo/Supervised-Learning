"""
Student:        Cian Herlihy - R00205604
Title:          Supervised Learning
Due Date:       Friday, 17th November 2023
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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
        sample_image = sample_image_vector.reshape(28, 28)

        # Display the image
        plt.imshow(sample_image, cmap='gray')
        plt.title(f'Label: {sample_label}')
        plt.show()


def k_fold_cross_validation(df, initial_train_size=8000, increase_per_fold=10000, initial_test_size=4000, increase_test_per_fold=2000):
    vectors = df.iloc[:, 1:]  # Vector images
    categories = df.iloc[:, 0]   # Categories
    current_train_size = initial_train_size
    current_test_size = initial_test_size
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(vectors, categories), 1):  # Split data into folds
        train_index = train_index[:current_train_size]
        test_index = test_index[:current_test_size]
        x_train, x_test = vectors.iloc[train_index], vectors.iloc[test_index]   # Extracts vectors for train/test
        y_train, y_test = categories.iloc[train_index], categories.iloc[test_index]   # Extracts vectors for train/test
        print(f"Fold {fold} - Training Size: {current_train_size}, Evaluation Size: {len(test_index)}")
        current_train_size += increase_per_fold
        current_test_size += increase_test_per_fold


def main():
    df = load_data(CSV_FILE_NAME)
    # Task 1
    label_data = separate_labels(df)
    # display_vector_per_category(label_data)
    # Task 2
    k_fold_cross_validation(df)


if __name__ == "__main__":
    main()
