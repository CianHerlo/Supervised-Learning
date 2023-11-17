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
from sklearn.svm import SVC
import time

CSV_FILE_NAME = "fashion-mnist_train.csv"   # Constant for file name


def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)     # read csv in to dataframe
    return df                           # return dataframe


def separate_labels(df):
    label_dfs_dict = {}                                     # Dictionary for all categories
    unique_labels = df['label'].unique()                    # Get all unique labels(categories) 0-9

    for label in unique_labels:                             # loop through all categories
        label_dfs_dict[label] = df[df['label'] == label]    # Gather all matching vectors and insert value as dataframe
    return label_dfs_dict                                   # Return Dictionary full of dataframes


def display_vector_per_category(label_data):
    for label, data in label_data.items():              # Loop through dictionary full of categories' dataframes
        image_values = data.iloc[0, 1:].values          # Grab values for image vector
        image_label = data.iloc[0, 0]                   # grab label for vector
        reshaped_image = image_values.reshape(28, 28)   # Reshape values into 28x28 image

        # Display the image
        plt.imshow(reshaped_image, cmap='gray')         # Plot image in Grayscale
        plt.title(f'Label: {image_label}')              # Set title for image as category of image
        plt.show()                                      # Display plot in window


def k_fold_cross_validation(df, classifier_type='perceptron', k=5, gamma=1, initial_train_size=8000,
                            increase_per_fold=10000, initial_test_size=4000, increase_test_per_fold=2000):
    data_sizes = []                             # Total images used in each fold
    train_times = []                            # Training times
    eval_times = []                             # Evaluation times
    all_elapsed_train_times = []                # All durations for training
    all_elapsed_eval_times = []                 # All durations for evaluations
    all_elapsed_pred_times = []                 # All durations for prediction of image
    all_elapsed_confusion_matrix_times = []     # All durations for confusion matrix generations
    all_eval_accuracies = []                    # All evaluations accuracies
    vectors = df.iloc[:, 1:]                    # Vector images
    categories = df.iloc[:, 0]                  # Categories
    current_train_size = initial_train_size     # Initial size of train size before folds
    current_test_size = initial_test_size       # Initial size of test size before folds

    # Switch like if block to determine classifier to be used
    if classifier_type == 'perceptron':
        classifier = Perceptron()
    elif classifier_type == 'decision_tree':
        classifier = DecisionTreeClassifier()
    elif classifier_type == 'k_nearest_neighbor':
        classifier = KNeighborsClassifier(n_neighbors=k)        # K can be set in parameter. Defaults to 5
    elif classifier_type == 'support_vector_machine':
        classifier = SVC(kernel='rbf', gamma=gamma)             # Gamma can be set in parameter. Defaults to 1
    else:                                                       # Should be unreachable
        print(f"Invalid classifier type: {classifier_type}")    # Unknown classifier handling statement
        exit()                                                  # Exit code due to error

    kf = KFold(n_splits=5, shuffle=True, random_state=42)       # Create KFold with 5 splits
    for fold, (train_index, test_index) in enumerate(kf.split(vectors, categories), 1):     # Split data into folds
        train_index = train_index[:current_train_size]                                      # Limit size of data used for training
        test_index = test_index[:current_test_size]                                         # Limit size of data used for test/eval
        x_train, x_test = vectors.iloc[train_index], vectors.iloc[test_index]               # Extracts vectors for train/test
        y_train, y_test = categories.iloc[train_index], categories.iloc[test_index]         # Extracts vectors for train/test

        # Training
        start_train_time = time.time()                                  # Start time for training
        scores = cross_val_score(classifier, x_train, y_train, cv=5)    # Train model
        end_train_time = time.time()                                    # End time for training
        elapsed_train_time = end_train_time - start_train_time          # Training time duration
        all_elapsed_train_times.append(elapsed_train_time)              # Add duration to array

        # Evaluation
        start_eval_time = time.time()                           # Start time for evaluation
        classifier.fit(x_train, y_train)                        # Fit classifier on the entire training set for evaluation
        eval_accuracy = classifier.score(x_test, y_test)        # Evaluate the test set
        end_eval_time = time.time()                             # End time for evaluation
        elapsed_eval_time = end_eval_time - start_eval_time     # Evaluation time duration
        all_elapsed_eval_times.append(elapsed_eval_time)        # Add duration to array
        all_eval_accuracies.append(eval_accuracy)               # Add accuracy to array

        # Evaluation Prediction
        start_pred_time = time.time()                           # Start time for prediction
        y_pred = classifier.predict(x_test)                     # Predict on the test set
        end_pred_time = time.time()                             # End time for prediction
        elapsed_pred_time = end_pred_time - start_pred_time     # Prediction time duration
        all_elapsed_pred_times.append(elapsed_pred_time)        # Add duration to array

        # Confusion Matrix
        start_confusion_matrix_time = time.time()   # Record start time for confusion matrix
        cm = confusion_matrix(y_test, y_pred)       # Generate confusion matrix
        end_confusion_matrix_time = time.time()     # Record end time for confusion matrix
        elapsed_confusion_matrix_time = end_confusion_matrix_time - start_confusion_matrix_time     # Generate confusion matrix duration
        all_elapsed_confusion_matrix_times.append(elapsed_confusion_matrix_time)                    # Add duration to array

        # Collect data for plotting
        data_sizes.append(current_train_size)   # Add image count to list
        train_times.append(elapsed_train_time)  # Add training duration to list
        eval_times.append(elapsed_eval_time)    # Add evaluation duration to list

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
    plt.figure(figsize=(10, 5))                                             # Graph size = 1,000px x 500px
    plt.plot(data_sizes, train_times, label='Training Duration')      # Data plotting with legend title for line
    plt.plot(data_sizes, eval_times, label='Evaluation Duration')     # Data plotting with legend title for line
    plt.xlabel('Data Size (vectors)')   # X axis Label
    plt.ylabel('Time (seconds)')        # Y axis label
    plt.title(
        f'Relationship Between Data Size and Run Duration for {classifier_type.capitalize()} Classifier')  # Graph title
    plt.legend()    # Show legend for line titles
    plt.show()      # Show graph

    # Create a dictionary to store mean values
    mean_results = {
        'mean_train_time': np.mean(all_elapsed_train_times),
        'mean_eval_time': np.mean(all_elapsed_eval_times),
        'mean_pred_time': np.mean(all_elapsed_pred_times),
        'mean_confusion_matrix_time': np.mean(all_elapsed_confusion_matrix_times),
        'mean_accuracy': np.mean(all_eval_accuracies)
    }
    return mean_results     # Return dictionary containing mean results


def find_best_k(accuracies):    # Find the best value for K-Nearest Neighbour classifier
    best_k = max(accuracies, key=lambda k: accuracies[k]["mean_accuracy"])      # Get max accuracy from each k
    best_accuracy = accuracies[best_k]["mean_accuracy"]                         # Get best mean accuracy achieved
    print(f"Best Mean k: {best_k}, Best Mean Accuracy: {best_accuracy:.4f}")    # Print selected info
    return accuracies[best_k]   # Return dictionary for best k value


def find_best_gamma(gamma_accuracies):  # Find the best Gamma value for Support Vector Machine classifier
    best_gamma = max(gamma_accuracies, key=lambda gamma: gamma_accuracies[gamma]["mean_accuracy"])  # Get max accuracy from each gamma
    best_accuracy = gamma_accuracies[best_gamma]["mean_accuracy"]                                   # Get best mean accuracy achieved
    print(f"Best Mean gamma: {best_gamma}, Best Mean Accuracy: {best_accuracy:.4f}\n")              # Print selected info
    return gamma_accuracies[best_gamma]     # Return dictionary for best gamma value


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
    print("\nOverall Results")
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


def final_comparison(mean_perceptron_results, mean_decision_tree_results, best_mean_k_nearest_neighbor_results,
                     best_mean_support_vector_machine_results):
    headers = ["Classifier", "Train", "Evaluation", "Predict",
               "Conf. Matrix", "Accuracy"]  # List of headers for printing
    classifiers = [
        ("Perceptron", mean_perceptron_results),
        ("Decision Tree", mean_decision_tree_results),
        ("K Nearest Neighbor", best_mean_k_nearest_neighbor_results),
        ("Support Vector Machine", best_mean_support_vector_machine_results)
    ]   # List of classifiers for printing

    # Display headers
    print("{:<30}".format(headers[0]), end="")  # End with "" allows next print to same line
    for header in headers[1:]:                  # Loop through headers list
        print("{:<15}".format(header), end="")  # Prints each header aligned left with 20 spaces
    print()                                     # Move to the next line
    for classifier_name, result in classifiers:                                         # Loop through classifiers
        print("{:<30}".format(classifier_name), end="")                                 # Print classifier's title
        print("{:<15}".format(f"{result['mean_train_time']:.2f}s"), end="")             # Print each mean training time
        print("{:<15}".format(f"{result['mean_eval_time']:.2f}s"), end="")              # Print each mean evaluation time
        print("{:<15}".format(f"{result['mean_pred_time']:.4f}s"), end="")              # Print each mean prediction time
        print("{:<15}".format(f"{result['mean_confusion_matrix_time']:.4f}s"), end="")  # Print each mean confusion matrix gen time

        # Convert accuracy to percentage and print
        accuracy_percentage = result['mean_accuracy'] * 100
        print("{:<15}".format(f"{accuracy_percentage:.2f}%"))


def main():
    print(f"{'_' * 20}  Prep  {'_' * 20}")
    df = load_data(CSV_FILE_NAME)
    # Task 1
    print(f"{'_' * 20} Task 1 {'_' * 20}")
    label_data = separate_labels(df)
    display_vector_per_category(label_data)
    # Task 2
    print(f"{'_' * 20} Task 2 {'_' * 20}")
    # k_fold_cross_validation(df) - Task 2 should only develop function to be used
    # Task 3
    print(f"{'_' * 20} Task 3 {'_' * 20}")
    mean_perceptron_results = k_fold_cross_validation(df, "perceptron")
    # Task 4
    print(f"{'_' * 20} Task 4 {'_' * 20}")
    mean_decision_tree_results = k_fold_cross_validation(df, "decision_tree")
    # Task 5
    print(f"{'_' * 20} Task 5 {'_' * 20}")
    all_k_accuracies = {}       # Empty dictionary for storing k values with dictionary for results
    for k_val in range(1, 6):   # Loop to give k values (1, 2, 3, 4, 5) - 4 was the best k value
        mean_k_nearest_neighbor_results = k_fold_cross_validation(df, "k_nearest_neighbor", k=k_val)
        all_k_accuracies[k_val] = mean_k_nearest_neighbor_results           # Add mean results to dictionary
    best_mean_k_nearest_neighbor_results = find_best_k(all_k_accuracies)    # Find best mean results from all dictionaries
    # Task 6
    print(f"{'_' * 20} Task 6 {'_' * 20}")
    all_gamma_accuracies = {}               # Empty dictionary for storing k values with dictionary for results
    for gamma_val in range(10, 41, 10):     # Loop to give gamma values (10, 20, 30, 40)
        mean_support_vector_machine_results = k_fold_cross_validation(df,
                                                                      "support_vector_machine",
                                                                      gamma=gamma_val,
                                                                      initial_train_size=1000,
                                                                      increase_per_fold=1000,
                                                                      initial_test_size=1000,
                                                                      increase_test_per_fold=1000)  # Took too long so narrowed the data sample
        all_gamma_accuracies[gamma_val] = mean_support_vector_machine_results           # Add mean results to dictionary
    best_mean_support_vector_machine_results = find_best_gamma(all_gamma_accuracies)    # Find best mean results from all dictionaries
    # Task 7
    print(f"{'_' * 20} Task 7 {'_' * 20}")
    final_comparison(mean_perceptron_results, mean_decision_tree_results, best_mean_k_nearest_neighbor_results,
                     best_mean_support_vector_machine_results)
    """
    Upon doing all these tasks and trying out these classifiers, I discovered some pro's and con's to the models in my opinion.
    Perceptron was a much faster model and allowed me to use all the images available within a reasonable time frame and
    yielded a high accuracy. I also like the simplicity of the model which for example you do not need extra parameters
    to use the classifier. This classifier is great for easily split data into sections so it can apply a weight to
    the images to predict. Given that this project consisted of data that was easily separated into 10 different sections
    0-9 categories then it enabled this model to perform very well.
    
    Decision Tree classifier was another easy to use model by importing the correct classifier and implement into my 
    code with no additional data needed. As the name suggests it uses a tree structure with each branch representing a 
    decision choice. The recursive functions it uses made it a faster model to work with, however, my pc did sometimes 
    struggle and would give me weird readings like being 10x slower the Perceptron for example. This model is also known 
    for struggling with unseen data which does make the whole predictability of it kind of null, therefore I wouldn't have
    chose this model even with the high accuracies it yielded for me.
    
    K-Nearest Neighbours was another classifier like Perceptron that was very fast for training on my pc and found it 
    quite often to be faster. It does, however, add complexity in implementing it over Perceptron such as more parameters.
    This model was my most accurate classifier and also fastest meaning it is in my opinion and experience in this project,
    the best fit for this data. It is a highly flexible model and predicts based on data points distances and how a
    data point being closer to another will impact heavily on what it predicts. This does make the prediction times slower.
    The parameters you can adjust may yield better results too and in my testing, 4 was the best value for k out of 1-5.
    
    Finally the Support Vector Machine classifier. I struggled with this classifier and getting it to work for me. It 
    has given me extremely low accuracies and is by far the worst for timing. So bad that I had to lower the training
    sample size and evaluation size dramatically. I rarely got into the double digit percentages although probably due 
    to my smaller sample sizes but the model with the same sample size as the other 3 would take well over 30 minutes
    for 1 of 5 folds. I then wanted to test more gamma ranges which meant an extra 5 folds for each gamma range. I tried
    gamma ranges from 0.1-1.0, I tried gamma ranges 1-10 and even tried gamma ranges 10-100. All resulting in extremely
    slow training times and evaluations. I spent countless hours trying to make this work and would love to see a working
    example of this being done because I could not figure out what I was doing wrong. I am sorry but I am open to 
    being corrected and educated in doing so. The model itself and how it works I even find harder to explain in its
    understanding but from my experience now, I find it very difficult to work with.
    
    Overall, K-Nearest Neighbour is my recommended classifier with a mean average score of 86.79% in my latest run. 
    The mean times are brought down by the scaled up sample rates but if you take the table I print off it was 
    16.33 seconds for training, 5.43 seconds for evaluation and sub 1 seconds for prediction and generating of 
    confusion matrix's combined.
    
    Ranking:
    1. K-Nearest Neighbour
    2. Perceptron
    3. Decision Tree
    4. Support Vector Machine (By Default)
    """


if __name__ == "__main__":
    main()
