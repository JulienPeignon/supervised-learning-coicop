"""
These functions automate hierarchical (Local Classifier Per Parent Node) text classification using FastText. They prepare data, train models for various classification levels, and merge the predictions back into the original dataset, focusing on COICOP codes for product categorization.
"""

import os
import pandas as pd
import fasttext
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.models.hiclass import get_data_hiclass
from src.models.flat_fasttext import (
    prepare_fasttext_data,
    process_data,
    train_model,
    predict_category,
    format_predictions,
)


def write_to_file(df: pd.DataFrame, category: str, file_type: str) -> None:
    """Writes a DataFrame to a file in fasttext format if the dataframe has more than 2 entries.

    Args:
    - df (pd.DataFrame): DataFrame to be written to the file.
    - category (str): Category name for the filename.
    - file_type (str): Type of file ('train' or 'test').

    Returns:
    - None
    """
    directory_name = "fastText_files"

    # Create the directory if it doesn't exist
    os.makedirs(directory_name, exist_ok=True)

    # Write DataFrame to a file
    df.to_csv(
        os.path.join(directory_name, f"{file_type}_fasttext_{category}.txt"),
        header=False,
        index=False,
        sep="\t",
        mode="a",
    )


def get_LCPN_fasttext_files(
    df_train: pd.DataFrame, df_test: pd.DataFrame, coicop_col: str, desc_col: str
) -> None:
    """
    Prepares fasttext formatted training and test files from input DataFrames.

    Args:
    df_train (pd.DataFrame): Training data.
    df_test (pd.DataFrame): Testing data.
    coicop_col (str): Column name for coicop categories.
    desc_col (str): Column name for descriptions.
    """

    category_lengths = {"POSTE": 2, "SOUS-CLASSE": 4, "CLASSE": 6, "GROUPE": 8}

    # Calculate the number of models to train
    num_models = sum(
        [
            df_train[coicop_col].str[:-length].nunique()
            for length in category_lengths.values()
        ]
    )
    print(f"There are {num_models} models to train.")

    # Creating txt files for each category length
    for length in tqdm(category_lengths.values(), desc="Creating txt files"):
        unique_categories = df_train[coicop_col].str[:-length].unique()

        # Process each category
        for category in unique_categories:
            for df_type, original_df in zip(["train", "test"], [df_train, df_test]):
                sub_df = original_df[
                    original_df[coicop_col].str[:-length] == category
                ].copy()

                # Update category data based on current category length
                if length != 2:
                    sub_df[coicop_col] = sub_df[coicop_col].str[: -length + 2]
                sub_df[coicop_col] = sub_df[coicop_col].str[-1]

                fasttext_df = prepare_fasttext_data(sub_df, coicop_col, desc_col)
                write_to_file(fasttext_df, category, df_type)

    # Handle division category separately
    for df_type, original_df in zip(["train", "test"], [df_train, df_test]):
        sub_df = original_df.copy()
        sub_df[coicop_col] = sub_df[coicop_col].str[:-8]
        fasttext_df = "__label__" + sub_df[coicop_col] + " " + sub_df[desc_col]
        write_to_file(fasttext_df, "division", df_type)


def process_category_data(
    category: str, params: dict, pred_column: str, desc_column: str, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process a specific category, train model and make predictions.

    Args:
    category (str): Specific category to process.
    params (dict): FastText parameters.
    pred_column (str): Column name for predictions.
    desc_column (str): Column name for descriptions.
    df (pd.DataFrame): DataFrame with data.

    Returns:
    pd.DataFrame: DataFrame with updated predictions.
    """
    input_file = f"fastText_files/train_fasttext_{category}.txt"
    test_file = f"fastText_files/test_fasttext_{category}.txt"

    # Train the model
    try:
        model = train_model(input_file, params)
    except ValueError:
        return df

    # Make predictions
    descriptions = process_data(test_file)
    predictions, confidences = predict_category(model, descriptions)

    # Prepare DataFrame with predictions and descriptions
    df_pred = pd.DataFrame(
        {pred_column: predictions, desc_column: descriptions}
    ).drop_duplicates(subset=desc_column)
    df_pred = format_predictions(df_pred, pred_column)

    # Merge original dataframe with new predictions
    df = merge_and_update_predictions(df, df_pred, pred_column, desc_column)

    return df


def merge_and_update_predictions(
    df: pd.DataFrame, df_pred: pd.DataFrame, pred_column: str, desc_column: str
) -> pd.DataFrame:
    """
    Merge original DataFrame with new predictions and update.

    Args:
    df (pd.DataFrame): Original DataFrame.
    df_pred (pd.DataFrame): DataFrame with new predictions.
    pred_column (str): Column name for predictions.
    desc_column (str): Column name for descriptions.

    Returns:
    pd.DataFrame: Updated DataFrame.
    """
    # Merge DataFrames on description column
    merged_df = pd.merge(
        df, df_pred, on=desc_column, how="inner", suffixes=("", "_new")
    )
    merged_df[pred_column] = (
        merged_df[pred_column] + "." + merged_df[f"{pred_column}_new"]
    )

    # Update original DataFrame with new predictions
    df.set_index(desc_column, inplace=True)
    merged_df.set_index(desc_column, inplace=True)
    df.update(merged_df[pred_column])
    df.reset_index(inplace=True)

    return df


def delete_all_files(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def get_LCPN_prediction_fasttext(
    data_train, data_test, coicop_column, desc_column, params
) -> pd.DataFrame:
    """
    Train a fastText model using the training data and predicts categories on the test data.

    Parameters:
    - data_train (pd.DataFrame): Training data DataFrame.
    - data_test (pd.DataFrame): Test data DataFrame.
    - coicop_column (str): Name of the column containing COICOP (Classification of Individual Consumption by Purpose) categories.
    - desc_column (str): Name of the column containing descriptions.
    - params (dict): Parameters for training the fastText model.

    Returns:
    - pd.DataFrame: Test data DataFrame enriched with the predicted categories.
    """
    # Generate fastText files
    get_LCPN_fasttext_files(
        df_train=data_train,
        df_test=data_test,
        coicop_col=coicop_column,
        desc_col=desc_column,
    )

    # Setup for division-level predictions
    input_file = f"fastText_files/train_fasttext_division.txt"
    test_file = f"fastText_files/test_fasttext_division.txt"
    pred_column = "LCPN_PRED"

    # Train model and make predictions for division level
    model = train_model(input_file, params)
    descriptions = process_data(test_file)
    predictions, confidences = predict_category(model, descriptions)

    # Prepare DataFrame with predictions and descriptions
    df = pd.DataFrame(
        {pred_column: predictions, desc_column: descriptions}
    ).drop_duplicates(subset=desc_column)
    df = format_predictions(df, pred_column)

    # Process each category length
    for i in [2, 4, 6, 8]:
        for category in tqdm(
            df[pred_column].unique(), desc="Training & testing fastText"
        ):
            if len(category) == i:
                df = process_category_data(
                    category, params, pred_column, desc_column, df
                )

    # Filter by final category length
    df = df.loc[df[pred_column].str.len() == 10].copy()

    # Merge predictions with test data
    df_final = pd.merge(
        df,
        data_test.drop_duplicates(subset=desc_column).reset_index(drop=True),
        on=desc_column,
        how="inner",
    )

    # Suppress the temporary files
    delete_all_files("fastText_files")

    return df_final
