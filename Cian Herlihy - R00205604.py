"""
Student:        Cian Herlihy - R00205604
Title:          Supervised Learning
Due Date:       Friday, 17th November 2023
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import time

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


def k_fold_cross_validation(df, classifier_type='perceptron'):
    data_sizes = []
    train_times = []
    eval_times = []
    all_elapsed_train_times = []
    all_elapsed_eval_times = []
    all_elapsed_pred_times = []
    all_elapsed_confusion_matrix_times = []
    all_eval_accuracies = []
    initial_train_size = 8000  # Initial training size
    increase_per_fold = 10000  # Increase training size per fold
    initial_test_size = 4000  # Initial test size
    increase_test_per_fold = 2000  # Increase test size per fold
    vectors = df.iloc[:, 1:]  # Vector images
    categories = df.iloc[:, 0]  # Categories
    current_train_size = initial_train_size  # Initial size of train size before folds
    current_test_size = initial_test_size  # Initial size of test size before folds

    if classifier_type == 'perceptron':
        classifier = Perceptron()
    elif classifier_type == 'decision_tree':
        classifier = DecisionTreeClassifier()
    elif classifier_type == 'k_nearest_neighbor':
        classifier = KNeighborsClassifier()
    else:
        print(f"Invalid classifier type: {classifier_type}")
        exit()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(vectors, categories), 1):  # Split data into folds
        train_index = train_index[:current_train_size]
        test_index = test_index[:current_test_size]
        x_train, x_test = vectors.iloc[train_index], vectors.iloc[test_index]  # Extracts vectors for train/test
        y_train, y_test = categories.iloc[train_index], categories.iloc[test_index]  # Extracts vectors for train/test

        # Training
        start_train_time = time.time()  # Start time for training
        scores = cross_val_score(classifier, x_train, y_train, cv=5)  # Train model
        end_train_time = time.time()  # End time for training
        elapsed_train_time = end_train_time - start_train_time  # Training time duration
        all_elapsed_train_times.append(elapsed_train_time)  # Add duration to array

        # Evaluation
        start_eval_time = time.time()  # Start time for evaluation
        classifier.fit(x_train, y_train)  # Fit classifier on the entire training set for evaluation
        eval_accuracy = classifier.score(x_test, y_test)  # Evaluate the test set
        end_eval_time = time.time()  # End time for evaluation
        elapsed_eval_time = end_eval_time - start_eval_time  # Evaluation time duration
        all_elapsed_eval_times.append(elapsed_eval_time)  # Add duration to array
        all_eval_accuracies.append(eval_accuracy)  # Add accuracy to array

        # Evaluation Prediction
        start_pred_time = time.time()  # Start time for prediction
        y_pred = classifier.predict(x_test)  # Predict on the test set
        end_pred_time = time.time()  # End time for prediction
        elapsed_pred_time = end_pred_time - start_pred_time  # Prediction time duration
        all_elapsed_pred_times.append(elapsed_pred_time)  # Add duration to array

        # Confusion Matrix
        start_confusion_matrix_time = time.time()  # Record start time for confusion matrix
        cm = confusion_matrix(y_test, y_pred)  # Generate confusion matrix
        end_confusion_matrix_time = time.time()  # Record end time for confusion matrix
        elapsed_confusion_matrix_time = end_confusion_matrix_time - start_confusion_matrix_time  # Generate confusion matrix duration
        all_elapsed_confusion_matrix_times.append(elapsed_confusion_matrix_time)  # Add duration to array

        # Collect data for plotting
        data_sizes.append(current_train_size)
        train_times.append(elapsed_train_time)
        eval_times.append(elapsed_eval_time)

        # Printing
        print_fold_results(fold, current_train_size, test_index, scores, elapsed_train_time, elapsed_eval_time,
                           elapsed_pred_time, elapsed_confusion_matrix_time, eval_accuracy)

        # Increase data size of train and test for next fold
        current_train_size += increase_per_fold
        current_test_size += increase_test_per_fold

    # Printing statistics from total folds
    print_overall_results(all_elapsed_train_times, all_elapsed_eval_times, all_elapsed_pred_times,
                          all_elapsed_confusion_matrix_times, all_eval_accuracies)

    # Plotting
    plt.figure(figsize=(10, 5))  # Graph size = 1,000px x 500px
    plt.plot(data_sizes, train_times, label='Training Duration')  # Data plotting with legend title for line
    plt.plot(data_sizes, eval_times, label='Evaluation Duration')  # Data plotting with legend title for line
    plt.xlabel('Data Size (vectors)')  # X axis Label
    plt.ylabel('Time (seconds)')  # Y axis label
    plt.title(f'Relationship Between Data Size and Run Duration for {classifier_type.capitalize()} Classifier')  # Graph title
    plt.legend()  # Show legend for line titles
    plt.show()  # Show graph


def print_fold_results(fold, current_train_size, test_index, scores, elapsed_train_time, elapsed_eval_time,
                       elapsed_pred_time, elapsed_confusion_matrix_time, eval_accuracy):
    print(f"Fold {fold} - Training Size: {current_train_size}, Evaluation Size: {len(test_index)}")
    print(f"Cross-Validation Accuracy: {scores.mean():.4f}")
    print(f"Time to train fold: {elapsed_train_time:.2f} seconds")
    print(f"Time to evaluate fold: {elapsed_eval_time:.2f} seconds")
    print(f"Time for Prediction on Test Set: {elapsed_pred_time:.4f} seconds")
    print(f"Time for Confusion Matrix: {elapsed_confusion_matrix_time:.4f} seconds")
    print(f"Prediction Accuracy on Test Set: {eval_accuracy:.4f}\n")


def print_overall_results(all_elapsed_train_times, all_elapsed_eval_times, all_elapsed_pred_times,
                          all_elapsed_confusion_matrix_times, all_eval_accuracies):
    print("\nOverall Results:")
    print(f"{'='*50}")
    print(f"{'-' * 10}Training{'-' * 10}")
    print(f"Min Training Time: {min(all_elapsed_train_times):.2f} seconds")
    print(f"Max Training Time: {max(all_elapsed_train_times):.2f} seconds")
    print(f"Average Training Time: {np.mean(all_elapsed_train_times):.2f} seconds")
    print(f"{'-' * 10}Evaluation{'-' * 10}")
    print(f"Min Evaluation Time: {min(all_elapsed_eval_times):.2f} seconds")
    print(f"Max Evaluation Time: {max(all_elapsed_eval_times):.2f} seconds")
    print(f"Average Evaluation Time: {np.mean(all_elapsed_eval_times):.2f} seconds")
    print(f"{'-' * 10}Prediction{'-' * 10}")
    print(f"Min Prediction Time: {min(all_elapsed_pred_times):.4f} seconds")
    print(f"Max Prediction Time: {max(all_elapsed_pred_times):.4f} seconds")
    print(f"Average Prediction Time: {np.mean(all_elapsed_pred_times):.4f} seconds")
    print(f"{'-' * 10}Confusion Matrix{'-' * 10}")
    print(f"Min Confusion Matrix Time: {min(all_elapsed_confusion_matrix_times):.4f} seconds")
    print(f"Max Confusion Matrix Time: {max(all_elapsed_confusion_matrix_times):.4f} seconds")
    print(f"Average Confusion Matrix Time: {np.mean(all_elapsed_confusion_matrix_times):.4f} seconds")
    print(f"{'-' * 10}Accuracy{'-' * 10}")
    print(f"Min Prediction Accuracy: {min(all_eval_accuracies):.4f}")
    print(f"Max Prediction Accuracy: {max(all_eval_accuracies):.4f}")
    print(f"Average Prediction Accuracy: {np.mean(all_eval_accuracies):.4f}\n")


def main():
    print(f"{'_'*20}  Prep  {'_'*20}")
    df = load_data(CSV_FILE_NAME)
    # Task 1
    print(f"{'_' * 20} Task 1 {'_' * 20}")
    label_data = separate_labels(df)
    # display_vector_per_category(label_data)
    # Task 2
    print(f"{'_' * 20} Task 2 {'_' * 20}")
    # k_fold_cross_validation(df) - Task 2 should only develop function to be used
    # Task 3
    print(f"{'_' * 20} Task 3 {'_' * 20}")
    # k_fold_cross_validation(df, "perceptron")
    # Task 4
    print(f"{'_' * 20} Task 4 {'_' * 20}")
    # k_fold_cross_validation(df, "decision_tree")
    # Task 5
    print(f"{'_' * 20} Task 5 {'_' * 20}")
    # k_fold_cross_validation(df, "k_nearest_neighbor")
    # Task 6
    print(f"{'_' * 20} Task 6 {'_' * 20}")
    # Task 7
    print(f"{'_' * 20} Task 7 {'_' * 20}")


if __name__ == "__main__":
    main()
