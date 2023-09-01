"""
These functions automate hierarchical (Local Classifier Per Level) text classification using FastText. They prepare data, train models for various classification levels, and merge the predictions back into the original dataset, focusing on COICOP codes for product categorization.

Experimental, not used in practice. The main hierarchical classifier is the Local Classifier Per Parent Node (LCPN)
"""

import os
import pandas as pd

from src.models.flat_fasttext import (
    prepare_fasttext_data,
    train_model,
    process_data,
    format_predictions,
)


def get_LCPL_fasttext_files(
    df_train,
    df_test,
    coicop_column="CODE_COICOP",
    desc_column="DESCRIPTION_EAN_FINAL",
    cat_coicop_list=["POSTE", "SOUS-CLASSE", "CLASSE", "GROUPE", "DIVISION"],
):
    """
    Generate training and test datasets for hierarchical text classification using fastText.

    Parameters:
    ----------
    df_train : DataFrame
        Training data containing coicop_column and desc_column.

    df_test : DataFrame
        Test data containing coicop_column and desc_column.

    coicop_column : str, optional
        Column in data containing the COICOP codes. Default is 'CODE_COICOP'.

    desc_column : str, optional
        Column in data containing the descriptions. Default is 'DESCRIPTION_EAN_FINAL'.

    cat_coicop_list : list, optional
        Categories for hierarchical classification. Default is ['POSTE', 'SOUS-CLASSE', 'CLASSE', 'GROUPE', 'DIVISION'].

    Returns:
    --------
    None
        Writes datasets to disk in fastText format.
    """

    # Preprocess the data
    data_hiclass_train = get_data_hiclass(df_train, true_column=coicop_column)
    data_hiclass_test = get_data_hiclass(df_test, true_column=coicop_column)

    # Convert to fastText format and save
    for cat in cat_coicop_list:
        train_fasttext = prepare_fasttext_data(data_hiclass_train, cat, desc_column)
        test_fasttext = prepare_fasttext_data(data_hiclass_test, cat, desc_column)

        train_fasttext.to_csv(
            f"train_fasttext_{cat}.txt", header=None, index=None, sep="\t", mode="a"
        )
        test_fasttext.to_csv(
            f"test_fasttext_{cat}.txt", header=None, index=None, sep="\t", mode="a"
        )


def find_string(row):
    """
    Finds the first string value among the prediction columns.

    Parameters:
    ----------
    row : Series
        A row from a DataFrame.

    Returns:
    --------
    str
        The found string or an empty string if none found.
    """
    for col in [
        "POSTE_PRED",
        "SOUS-CLASSE_PRED",
        "CLASSE_PRED",
        "GROUPE_PRED",
        "DIVISION_PRED",
    ]:
        val = row[col]
        if isinstance(val, str) or (isinstance(val, list) and len(val) == 1):
            return val[0] if isinstance(val, list) else val
    return ""


def get_LCPL_prediction_fasttext(
    df_train,
    df_test,
    parameters,
    confidence_threshold,
    coicop_column="CODE_COICOP",
    desc_column="DESCRIPTION_EAN_FINAL",
    cat_coicop_list=["POSTE", "SOUS-CLASSE", "CLASSE", "GROUPE", "DIVISION"],
):
    """
    Train a fastText model and get predictions for hierarchical text data.

    Parameters:
    ----------
    df_train : DataFrame
        Training data.

    df_test : DataFrame
        Test data.

    parameters : dict
        Parameters for 'fasttext.train_supervised' function.

    confidence_threshold : float
        Confidence threshold for predictions.

    coicop_column : str, optional
        COICOP code column. Default is 'CODE_COICOP'.

    desc_column : str, optional
        Description column. Default is 'DESCRIPTION_EAN_FINAL'.

    cat_coicop_list : list, optional
        Hierarchical classification categories. Default is ['POSTE', 'SOUS-CLASSE', 'CLASSE', 'GROUPE', 'DIVISION'].

    Returns:
    --------
    DataFrame
        Contains true labels, descriptions, predictions, and their probabilities.
    """
    # Prepare data
    get_LCPL_fasttext_files(
        df_train, df_test, coicop_column, desc_column, cat_coicop_list
    )

    dfs = []

    # Train and predict for each category
    for cat in cat_coicop_list:
        model = train_model(f"train_fasttext_{cat}.txt", parameters)
        descriptions = process_data(f"test_fasttext_{cat}.txt")
        predictions = predict_category(model, descriptions)
        probabilities = [list(model.predict(desc, k=2)[1]) for desc in descriptions]

        df_prediction = pd.DataFrame(
            {
                f"{cat}_PRED": predictions,
                f"{cat}_PROB": probabilities,
                desc_column: descriptions,
            }
        )

        df_prediction = format_predictions(df_prediction, f"{cat}_PRED")
        df_prediction[f"DIFF_PROB_{cat}"] = df_prediction[f"{cat}_PROB"].apply(
            lambda x: x[0] - x[1]
        )
        mask = df_prediction[f"DIFF_PROB_{cat}"] > confidence_threshold
        df_prediction.loc[mask, f"{cat}_PRED"] = df_prediction.loc[
            mask, f"{cat}_PRED"
        ].apply(lambda x: x[0])

        dfs.append(df_prediction)

    # Merging and finalizing
    df_final = pd.concat(dfs, axis=1)
    df_final = df_final.loc[
        :, ~df_final.columns.duplicated()
    ]  # Remove duplicate columns
    df_final["CONFIDENCE_PRED"] = df_final.apply(find_string, axis=1)
    df_final = df_final.merge(df_test, on=desc_column, how="inner")

    return df_final
