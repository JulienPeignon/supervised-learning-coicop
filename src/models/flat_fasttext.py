"""
These functions automate the process of text classification using FastText, focusing on transforming product descriptions into COICOP 6-digit codes for categorization. They handle data preparation, model training, and result integration.
"""

import os
import datetime
import pandas as pd
import fasttext

from src.features.functions_get_true_label import (
    get_desc_coicop,
    get_coicop_dict,
)


def pretrained_model_prediction_fasttext(
    data,
    model,
    label_column="DESCRIPTION_EAN_FINAL",
    prediction_column="COICOP_PREDICTED",
    description_column="DESC_COICOP_PRED",
):
    """
    Returns the input DataFrame with the predicted COICOP 6-digit code and the corresponding description of labels.

    Parameters:
    - data (pd.DataFrame): Data to use for prediction.
    - model (fasttext.FastText._FastText): Pre-trained model for prediction.
    - label_column (str): Name of the column containing the labels.
    - prediction_column (str): Name of the output column for the predicted COICOP 6-digit code.
    - description_column (str): Name of the output column for the description of the predicted COICOP code.

    Returns:
    - pd.DataFrame: DataFrame with predicted COICOP codes and descriptions.
    """
    # Load the COICOP dictionary
    coicop_dict = get_coicop_dict()

    # Group the data by label and aggregate the indices into lists
    data_group = (
        data.reset_index().groupby(label_column).agg({"index": list}).reset_index()
    )

    # Predict using the model and extract the first prediction for each group
    predictions = [
        k[0] for k in model.predict(data_group[label_column].tolist(), k=1)[0]
    ]

    # Remove "__label__" prefix from the predictions
    data_group[prediction_column] = [
        pred.replace("__label__", "") for pred in predictions
    ]
    # Add description for each prediction based on COICOP dictionary
    data_group = get_desc_coicop(
        data_group, prediction_column, description_column, category="poste"
    )

    # Merge predicted data back to the original dataframe
    data = (
        data.reset_index()
        .merge(
            data_group.explode("index")[
                [prediction_column, description_column, "index"]
            ],
            on="index",
        )
        .drop(columns="index")
    )

    print("-- Prediction done! --\n")
    return data


def prepare_fasttext_data(df, label_column, desc_column):
    """Prepare data in FastText format."""
    return "__label__" + df[label_column] + " " + df[desc_column]


def process_data(file_path):
    """Processes the data from a text file, extracting the descriptions."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.readlines()

    descriptions = [line.split(" ", 1)[1].rstrip("\n") for line in data]
    return descriptions


def train_model(input_file, params):
    return fasttext.train_supervised(input=input_file, **params)


def predict_category(model, descriptions):
    """
    Predict category using the provided model for a list of descriptions.

    Parameters:
    - model: The trained FastText model.
    - descriptions (list[str]): A list of descriptions to predict.

    Returns:
    - tuple: (list of predicted labels, list of confidence scores)
    """
    labels_and_probs = [model.predict(desc) for desc in descriptions]
    # Extract labels and probabilities separately
    predictions = [label[0][0] for label in labels_and_probs]
    confidences = [prob[1][0] for prob in labels_and_probs]
    return predictions, confidences


def format_predictions(df, pred_column):
    df[pred_column] = df[pred_column].astype(str)
    df[pred_column] = df[pred_column].str.replace("__label__", "")
    return df


def get_flat_prediction_fasttext(df_train, df_test, coicop_column, desc_column, params):
    """
    Train a FastText model using provided training data, and then use it to predict labels for test data.
    Merges the predicted labels with the original test dataframe.

    Parameters:
    - df_train (pd.DataFrame): Training data.
    - df_test (pd.DataFrame): Test data.
    - coicop_column (str): Column containing class labels in train and test DataFrames.
    - desc_column (str): Column containing description data in train and test DataFrames.
    - params (dict): Training parameters for the FastText model.

    Returns:
    - pd.DataFrame: DataFrame containing predictions, descriptions, and actual labels.
    """

    # Generate a unique timestamp for filenames
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_filename = f"train_fasttext_{now}.txt"
    test_filename = f"test_fasttext_{now}.txt"

    # Convert data to FastText format and save to files
    prepare_fasttext_data(df_train, coicop_column, desc_column).to_csv(
        train_filename, header=None, index=None, sep="\t", mode="a"
    )
    prepare_fasttext_data(df_test, coicop_column, desc_column).to_csv(
        test_filename, header=None, index=None, sep="\t", mode="a"
    )

    # Train the model with the train data
    model = train_model(train_filename, params)

    # Extract descriptions from the test file
    descriptions = process_data(test_filename)

    # Predict the labels for the test descriptions
    predictions, confidences = predict_category(model, descriptions)

    # Create a DataFrame with predictions and their corresponding descriptions
    df = pd.DataFrame(
        {"FLAT_PRED": predictions, desc_column: descriptions, "CONFIDENCE": confidences}
    )
    df = format_predictions(df, "FLAT_PRED")

    # Merge the predictions with the original test data
    df_final = df_test.reset_index(drop=True).join(
        df[["FLAT_PRED", "CONFIDENCE"]], how="inner"
    )

    # Clean up by removing temporary files
    os.remove(train_filename)
    os.remove(test_filename)

    return df_final
