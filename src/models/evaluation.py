"""
The functions evaluate a classification model's performance using metrics like accuracy, precision, and F1-score, optionally weighted by sales revenue, and provide a breakdown by COICOP categories.
"""

import pandas as pd
import numpy as np
from sklearn import metrics


def get_weighted_metrics(df, true_column, pred_column, weights_column):
    """
    Calculate precision, recall, and F1-score weighted by a given column (e.g., sales revenue).

    Parameters:
    - df: DataFrame containing the data
    - true_column: Column name for true labels
    - pred_column: Column name for predicted labels
    - weights_column: Column name for the weights (e.g., sales revenue)

    Returns:
    - None, but prints the weighted precision, recall, and F1-score
    """

    # Extracting columns
    y_true = df[true_column].values
    y_pred = df[pred_column].values
    weights = df[weights_column].values

    # Initializations
    weighted_TP = 0
    weighted_FP = 0
    weighted_FN = 0

    # Calculate weights for each class
    for c in np.unique(y_true):
        TP_weights = weights[(y_true == c) & (y_pred == c)]
        FP_weights = weights[(y_true != c) & (y_pred == c)]
        FN_weights = weights[(y_true == c) & (y_pred != c)]

        weighted_TP += TP_weights.sum()
        weighted_FP += FP_weights.sum()
        weighted_FN += FN_weights.sum()

    # Compute weighted metrics
    precision_weighted = weighted_TP / (weighted_TP + weighted_FP)
    recall_weighted = weighted_TP / (weighted_TP + weighted_FN)
    f1_weighted = (
        2
        * (precision_weighted * recall_weighted)
        / (precision_weighted + recall_weighted)
    )

    # Print results
    print("Weighted (sales revenue) precision: {:.2%}".format(precision_weighted))
    print("Weighted (sales revenue) recall: {:.2%}".format(recall_weighted))
    print("Weighted (sales revenue) F1-score: {:.2%}".format(f1_weighted))


def get_classification_metrics(df, true_column, pred_column):
    """
    Evaluates the accuracy, weighted precision, weighted recall, and weighted f1-score of predictions.

    Parameters:
    - df (DataFrame): The dataframe containing the true values and predictions.
    - true_column (str): The column name for the true values.
    - pred_column (str): The column name for the predicted values.
    - weights_column: Column name for the weights (e.g., sales revenue)

    Returns:
    None. Prints the evaluation metrics.
    """

    # Extracting true values and predictions from the dataframe
    y_true = df[true_column]
    y_pred = df[pred_column]

    # Computing weighted precision
    precision = metrics.precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    print("Precision: {:.2%}".format(precision))

    # Computing weighted recall
    recall = metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
    print("Recall: {:.2%}".format(recall))

    # Computing weighted f1-score
    f1 = metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print("F1-score: {:.2%}".format(f1))


def get_evaluation_model(
    data,
    pred_column,
    true_column,
    sales_column=None,
):
    """
    evaluation_model:
        Returns the correct prediction rate for different COICOP categories and their share in sales revenue.

    @param data (pd.DataFrame): Data to use for evaluation.
    @param pred_column (str): Name of the column containing the predicted COICOP.
    @param true_column (str): Name of the column containing the true COICOP.
    @param sales_column (str): Name of the column containing sales revenue.
    """
    # Dictionary mapping COICOP categories to their respective levels
    cat_coicop_dict = {
        "poste": 0,
        "sous-classe": 1,
        "classe": 2,
        "groupe": 3,
        "division": 4,
    }

    # Filtering out rows where true COICOP values are not missing
    data = data.loc[data[true_column].notna()]

    get_classification_metrics(data, true_column, pred_column)
    print("------")

    if sales_column:
        get_weighted_metrics(data, true_column, pred_column, sales_column)
        print("------")

        for cat in cat_coicop_dict.keys():
            if cat == "poste":
                # Counting the number of correct predictions
                n_pred = len(data.loc[(data[true_column] == data[pred_column])])

                # Calculating the total sales revenue for correct predictions
                n_sales = data.loc[
                    (data[true_column] == data[pred_column]), sales_column
                ].sum()

            else:
                # Counting the number of correct predictions at different levels
                n_pred = len(
                    data.loc[
                        (
                            data[true_column].str[: -2 * cat_coicop_dict[cat]]
                            == data[pred_column].str[: -2 * cat_coicop_dict[cat]]
                        )
                    ]
                )
                # Calculating the total sales revenue for correct predictions
                n_sales = data.loc[
                    (
                        data[true_column].str[: -2 * cat_coicop_dict[cat]]
                        == data[pred_column].str[: -2 * cat_coicop_dict[cat]]
                    ),
                    sales_column,
                ].sum()

            # Printing the results for each COICOP category
            print(
                "The model correctly predicted {:.2%} of {} digits COICOP codes ('{}')".format(
                    n_pred / len(data), 6 - cat_coicop_dict[cat], cat
                )
            )
            print(
                "Correctly predicted '{}' represents {:.2%} of sales revenue".format(
                    cat, n_sales / data[sales_column].sum()
                )
            )
            print("------")
