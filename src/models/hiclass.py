"""
Comprehensive utilities for hierarchical prediction. 
Utilizes the scikit-learn-contrib/hiclass package.
"""

import os
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from multiprocessing import Pool
from hiclass import metrics


def get_data_hiclass(
    data,
    true_column="TRUE_COICOP",
):
    """
    get_data_hiclass:
        Prepares the input data for hierarchical predictions

    @param data (pd.DataFrame): The input DataFrame that needs to be processed for hierarchical predictions.
    @param true_column (str): The name of the column in the input DataFrame that contains the true labels.
    @param root_name (str): The name that will be used for the new 'ROOT' column.
    @param cat_coicop_dict (dict): A dictionary that specifies how to create new columns from the 'POSTE' column.
    """
    cat_coicop_dict = {
        "SOUS-CLASSE": 1,
        "CLASSE": 2,
        "GROUPE": 3,
        "DIVISION": 4,
    }

    # Remove rows where the true_column field is NaN and rename it to 'POSTE'
    data = data.loc[data[true_column].notna()].rename(columns={true_column: "POSTE"})

    # Create new columns in the DataFrame by extracting subsets of characters from the 'POSTE' column
    # The starting position of the subset and the column name is determined by the cat_coicop_dict
    for cat in cat_coicop_dict.keys():
        data[cat] = data["POSTE"].str[: 10 - 2 * cat_coicop_dict[cat]]

    # Select only the columns needed for the hierarchical predictions
    data = data[
        [
            "DESCRIPTION_EAN_FINAL",
            "DIVISION",
            "GROUPE",
            "CLASSE",
            "SOUS-CLASSE",
            "POSTE",
        ]
    ]

    # Reset the index of the DataFrame and return it
    return data.reset_index(drop=True)


def evaluation_hiclass_hierarchical(
    y_true, y_predicted, print_results=True, return_score=False
):
    """
    evaluation_hiclass_hierarchical:
        Evaluates the performance of hiclass classifier

    @param y_true (np.array or pandas.series): The ground truth (correct) labels for the data.
    @param y_predicted (np.array or pandas.series): The predicted labels, as returned by a classifier.
    @param print_results (bool, optional): Whether to print the evaluation results. Default is True.
    @param return_score (bool, optional): Whether to return the evaluation scores. Default is False.
    """
    # Calculate the precision, recall, and F1 score
    precision = metrics.precision(y_true, y_predicted)
    recall = metrics.recall(y_true, y_predicted)
    f1 = metrics.f1(y_true, y_predicted)

    # Print the calculated metrics if required
    if print_results:
        print("Hierarchical Precision: {:.3%}".format(precision))
        print("Hierarchical Recall: {:.3%}".format(recall))
        print("Hierarchical F1: {:.3%}".format(f1))

    # Return the scores if required
    if return_score:
        return precision, recall, f1


def monte_carlo_evaluation_hiclass(pipeline, n_simulations, X, y, random_seed=42):
    """
    monte_carlo_evaluation_hiclass:
        Performs a Monte Carlo simulation to evaluate a hierarchical classification model.

    @param n_simulations (int): The number of Monte Carlo iterations to run.
    @param random_seed (int): Seed for the random number generator to ensure reproducibility.
    @param X (pd.DataFrame or np.array): The features for the hierarchical classification.
    @param y (pd.DataFrame or np.array): The labels for the hierarchical classification.
    """
    # Generate a sequence of random integers to use as random states for train-test splits
    random_integers = np.random.choice(10 * n_simulations, n_simulations)

    # Initialize lists to store the precision, recall, and F1 score for each iteration
    precision_flat_list, recall_flat_list, f1_flat_list = [], [], []
    precision_hierarchical_list, recall_hierarchical_list, f1_hierarchical_list = (
        [],
        [],
        [],
    )

    # Perform n_simulations iterations
    for i in random_integers:
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i
        )

        # Fit the model on the training data
        pipeline.fit(X_train, y_train)

        # Make predictions on the test data
        predictions = pipeline.predict(X_test)

        # Compute precision, recall, and F1 score FLAT
        precision_flat, recall_flat, f1_flat = evaluation_hiclass_flat(
            y_true=y_test["POSTE"],
            y_predicted=pd.DataFrame(predictions)[4],
            print_results=False,
            return_score=True,
        )

        # Compute precision, recall, and F1 score HIERARCHICAL
        (
            precision_hierarchical,
            recall_hierarchical,
            f1_hierarchical,
        ) = evaluation_hiclass_hierarchical(
            y_true=np.array(y_test),
            y_predicted=predictions,
            print_results=False,
            return_score=True,
        )

        # Append the scores to the respective lists
        precision_flat_list.append(precision_flat)
        recall_flat_list.append(recall_flat)
        f1_flat_list.append(f1_flat)
        precision_hierarchical_list.append(precision_hierarchical)
        recall_hierarchical_list.append(recall_hierarchical)
        f1_hierarchical_list.append(f1_hierarchical)

    # Print the average precision, recall, and F1 score over all iterations
    print(
        "Over {} iterations, the average flat precision is {:.3%}".format(
            n_simulations, np.mean(precision_flat_list)
        )
    )
    print(
        "Over {} iterations, the average flat recall is {:.3%}".format(
            n_simulations, np.mean(recall_flat_list)
        )
    )
    print(
        "Over {} iterations, the average flat f1 score is {:.3%}".format(
            n_simulations, np.mean(f1_flat_list)
        )
    )
    print(
        "Over {} iterations, the average hierarchical precision is {:.3%}".format(
            n_simulations, np.mean(precision_hierarchical_list)
        )
    )
    print(
        "Over {} iterations, the average hierarchical recall is {:.3%}".format(
            n_simulations, np.mean(recall_hierarchical_list)
        )
    )
    print(
        "Over {} iterations, the average hierarchical f1 score is {:.3%}".format(
            n_simulations, np.mean(f1_hierarchical_list)
        )
    )


def process_pipeline(max_df, ngram_range, C, kernel, gamma, kf, pipeline, X, y):
    """
    Function to train and validate a machine learning model in a pipeline.

    Arguments:
    max_df: The maximum frequency within the documents a given feature can have to be used in the tf-idf matrix.
    ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted.
    C: Inverse of regularization strength in SVM; smaller values specify stronger regularization.
    kernel: Specifies the kernel type to be used in the SVM algorithm.
    gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’ in SVM.
    kf: KFold object. The KFold cross-validator.
    pipeline: The scikit-learn Pipeline object.

    Returns:
    f1_cv: The mean F1 score calculated over all folds.
    """
    fold_scores = []

    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        pipeline.set_params(
            tfidf__max_df=max_df,
            tfidf__ngram_range=ngram_range,
            clf__local_classifier__C=C,
            clf__local_classifier__kernel=kernel,
        )

        # Set the gamma parameter for 'rbf' and 'sigmoid' kernels
        if kernel in ["rbf", "sigmoid"]:
            pipeline.set_params(clf__local_classifier__gamma=gamma)

        pipeline.fit(X_train_fold, y_train_fold)
        predictions = pipeline.predict(X_val_fold)

        # Evaluate predictions and get the F1 score
        precision, recall, f1 = evaluation_hiclass_flat(
            y_val_fold["POSTE"],
            pd.DataFrame(predictions)[4],
            print_results=False,
            return_score=True,
        )

        fold_scores.append(f1)

    # Compute the mean F1 score
    f1_cv = np.mean(fold_scores)
    return f1_cv


def worker(params, pipeline, X, y, kf):
    """
    Inner function to be executed in parallel for each set of hyperparameters.

    Parameters:
    params: Tuple containing the set of hyperparameters to use for training.
    pipeline: A sklearn.pipeline.Pipeline object representing the machine learning model pipeline.
    X: DataFrame containing the input features for the model.
    y: Series containing the target variable for the model.
    kf: A sklearn.model_selection.KFold object for splitting the data into training and validation sets.

    Returns:
    A tuple containing the set of hyperparameters and the corresponding F1 score obtained on the validation set.
    """
    try:
        print(f"Processing {params} in process id {os.getpid()}")
        max_df, ngram_range, C, kernel, gamma = params

        # Set hyperparameters for the pipeline
        pipeline.set_params(
            tfidf__max_df=max_df,
            tfidf__ngram_range=ngram_range,
            clf__local_classifier__C=C,
            clf__local_classifier__kernel=kernel,
        )

        # Set the gamma parameter for 'rbf' and 'sigmoid' kernels
        if kernel in ["rbf", "sigmoid"]:
            pipeline.set_params(clf__local_classifier__gamma=gamma)

        # Use the process_pipeline function to fit the model, make predictions, and get the F1 score
        score = process_pipeline(
            max_df, ngram_range, C, kernel, gamma, kf, pipeline, X, y
        )
        return params, score
    except Exception as e:
        print(f"Error in worker {os.getpid()}: {e}")
        return params, None


def tune_hyperparameters(
    max_dfs,
    ngram_ranges,
    Cs,
    kernels,
    gammas,
    pipeline,
    n_cpus,
    X,
    y,
    kf=KFold(n_splits=5),
):
    """
    Function to tune hyperparameters of a machine learning model in a pipeline using multiprocessing.

    Arguments:
    max_dfs: The maximum frequencies to consider for tf-idf matrix.
    ngram_ranges: The ngram ranges to consider.
    Cs: The regularization strengths to consider for SVM.
    kernels: The kernel types to consider for SVM.
    gammas: The kernel coefficients to consider for SVM.
    pipeline: The scikit-learn Pipeline object.
    n_cpus: Integer. The number of CPUs to use for multiprocessing.
    X: The input data for the model.
    y: The output data for the model.
    kf: KFold object. The KFold cross-validator.
    """
    # Create a list of all possible combinations of hyperparameters
    params_list = []

    for max_df, ngram_range, C, kernel in itertools.product(
        max_dfs, ngram_ranges, Cs, kernels
    ):
        if kernel in ["rbf", "sigmoid"]:
            for gamma in gammas:
                params_list.append((max_df, ngram_range, C, kernel, gamma))
        else:
            params_list.append((max_df, ngram_range, C, kernel, None))

    # Use functools.partial to create a new function that has some of the parameters preset
    worker_func = partial(worker, pipeline=pipeline, X=X, y=y, kf=kf)

    # Use multiprocessing to test all combinations of hyperparameters
    with Pool(n_cpus) as pool:
        results = pool.map(worker_func, params_list)

    # Check if any results were obtained
    if results:
        results_dict = {params: score for params, score in results if score is not None}

        # Find the combination of hyperparameters with the highest F1 score
        best_params = max(results_dict, key=lambda x: results_dict[x])
        best_score = results_dict[best_params]

        print("\nThe best parameters are:", best_params)
        print(f"The best F1 score is:", best_score)

        # Save best params and score to a file
        with open("hyperparameters_hiclass.txt", "w") as f:
            f.write("Best parameters are:\n")
            f.write(str(best_params))

    else:
        print("No results to process.")
