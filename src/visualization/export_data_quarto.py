"""
These functions are used to export data in the correct format for generating accuracy, Sankey, and confusion matrix graphs using Observable.
"""

import pandas as pd
import numpy as np

from src.features.functions_get_true_label import get_desc_coicop

def export_accuracy_data(
    data,
    true_coicop_column="TRUE_COICOP",
    pred_column="COICOP_PREDICTED",
    file_name="accuracy",
):
    """
    export_accuracy_data:
        Exports accuracy data to a Parquet file.

    @param data (pd.DataFrame): Data to export.
    @param true_coicop_column (str): Name of the column containing the true COICOP.
    @param pred_column (str): Name of the column containing the predicted COICOP.
    @param cat_coicop_dict (dict): Dictionary mapping COICOP categories to their levels.
    @param file_name (str): Name of the output file (without file extension).
    """
    cat_coicop_dict = {
        "poste": 0,
        "sous-classe": 1,
        "classe": 2,
        "groupe": 3,
        "division": 4,
    }

    data_graph = pd.DataFrame()

    # Iterate over COICOP categories
    for category in cat_coicop_dict.keys():
        if category == "poste":
            # For "poste" category
            data_concatenate = (
                pd.DataFrame(
                    data.loc[
                        data[true_coicop_column] == data[pred_column],
                        true_coicop_column,
                    ].value_counts()
                )
                .reset_index()
                .rename(
                    columns={"index": "COICOP", true_coicop_column: "NB_CORRECT_PRED"}
                )
            )
            data_concatenate["NB_OCCURENCE"] = data_concatenate["COICOP"].map(
                data[true_coicop_column].value_counts()
            )
        else:
            # For other categories
            data_concatenate = (
                pd.DataFrame(
                    data.loc[
                        data[true_coicop_column].str[: -2 * cat_coicop_dict[category]]
                        == data[pred_column].str[: -2 * cat_coicop_dict[category]],
                        true_coicop_column,
                    ]
                    .str[: -2 * cat_coicop_dict[category]]
                    .value_counts()
                )
                .reset_index()
                .rename(
                    columns={"index": "COICOP", true_coicop_column: "NB_CORRECT_PRED"}
                )
            )
            data_concatenate["NB_OCCURENCE"] = data_concatenate["COICOP"].map(
                data[true_coicop_column]
                .str[: -2 * cat_coicop_dict[category]]
                .value_counts()
            )

        # Calculate frequency of correct predictions
        data_concatenate["FREQ_CORRECT_PRED"] = (
            data_concatenate["NB_CORRECT_PRED"] / data_concatenate["NB_OCCURENCE"]
        )

        # Retrieve COICOP descriptions
        data_concatenate = get_desc_coicop(
            data_concatenate, "COICOP", "COICOP_DESC", category=category
        )

        # Concatenate data for different categories
        data_graph = pd.concat([data_graph, data_concatenate])

    # Remove unwanted characters from COICOP descriptions
    data_graph["COICOP_DESC"] = data_graph["COICOP_DESC"].str.replace(
        " n.c.a.", "", regex=False
    )

    # Map COICOP lengths to categories
    data_graph["CATEGORY"] = np.select(
        [
            data_graph["COICOP"].str.len() == 2,
            data_graph["COICOP"].str.len() == 4,
            data_graph["COICOP"].str.len() == 6,
            data_graph["COICOP"].str.len() == 8,
            data_graph["COICOP"].str.len() == 10,
        ],
        ["Division", "Groupe", "Classe", "Sous-classe", "Poste"],
    )

    # Drop any rows with missing values and save data to Parquet file
    data_graph.dropna().to_parquet(file_name + ".parquet", index=False)


def transform_dict(column):
    """
    transform_dict:
        Transforms a column into a dictionary with unique values as keys and their counts as values.

    @param column (pd.Series): Column to transform.

    Returns:
        dict: Transformed dictionary.
    """
    unique, counts = np.unique(column, return_counts=True)
    return dict(zip(unique, counts))


def export_sankey_data(
    data,
    true_coicop_column="TRUE_COICOP",
    pred_column="COICOP_PREDICTED",
    file_name="sankey",
):
    """
    export_sankey_data:
        Exports data for a Sankey diagram to a Parquet file.

    @param data (pd.DataFrame): Data to export.
    @param true_coicop_column (str): Name of the column containing the true COICOP.
    @param pred_column (str): Name of the column containing the predicted COICOP.
    @param cat_coicop_dict (dict): Dictionary mapping COICOP categories to their levels.
    @param file_name (str): Name of the output file (without file extension).
    """
    cat_coicop_dict = {
        "poste": 0,
        "sous-classe": 1,
        "classe": 2,
        "groupe": 3,
        "division": 4,
    }

    data_copy = data.loc[
        (data[true_coicop_column].notna()) & (data[true_coicop_column].str.len() == 10)
    ]
    data_graph = pd.DataFrame()

    # Iterate over COICOP categories
    for category in cat_coicop_dict.keys():
        data_concatenate = data_copy.copy()

        if category != "poste":
            # Reduce COICOP codes based on the category level
            data_concatenate[true_coicop_column] = data_concatenate[
                true_coicop_column
            ].str[: -2 * cat_coicop_dict[category]]
            data_concatenate[pred_column] = data_concatenate[pred_column].str[
                : -2 * cat_coicop_dict[category]
            ]

        # Group by predicted COICOP and aggregate true COICOP values
        data_concatenate = (
            data_concatenate.groupby(pred_column)
            .agg({true_coicop_column: list})
            .reset_index()
        )

        # Transform true COICOP values into dictionaries
        data_concatenate[true_coicop_column] = data_concatenate[
            true_coicop_column
        ].apply(transform_dict)

        # Extract keys and values from true COICOP dictionaries
        data_concatenate["true_coicop"] = data_concatenate[true_coicop_column].apply(
            lambda x: list(x.keys())
        )
        data_concatenate["value"] = data_concatenate[true_coicop_column].apply(
            lambda x: list(x.values())
        )

        # Explode the lists into separate rows
        data_concatenate = (
            data_concatenate.explode(["true_coicop", "value"])
            .drop(columns=true_coicop_column)
            .reset_index(drop=True)
        )

        # Retrieve COICOP descriptions for predicted and true COICOP values
        data_concatenate = get_desc_coicop(
            data_concatenate, pred_column, "target", category=category
        )
        data_concatenate = get_desc_coicop(
            data_concatenate, "true_coicop", "source", category=category
        )

        # Map COICOP lengths to categories
        data_concatenate["CATEGORY"] = np.select(
            [
                data_concatenate["true_coicop"].str.len() == 2,
                data_concatenate["true_coicop"].str.len() == 4,
                data_concatenate["true_coicop"].str.len() == 6,
                data_concatenate["true_coicop"].str.len() == 8,
                data_concatenate["true_coicop"].str.len() == 10,
            ],
            ["Division", "Groupe", "Classe", "Sous-classe", "Poste"],
        )

        # Select relevant columns and rename them
        data_concatenate = data_concatenate[["source", "target", "value", "CATEGORY"]]

        # Append a space to the target COICOP codes for visualization purposes
        data_concatenate["target"] = data_concatenate["target"] + " "

        # Sort the data by the source COICOP codes
        data_concatenate = data_concatenate.sort_values("source")

        # Concatenate the data for different categories
        data_graph = pd.concat([data_graph, data_concatenate])

    # Remove unwanted characters from COICOP descriptions
    data_graph["target"] = data_graph["target"].str.replace(" n.c.a.", "", regex=False)
    data_graph["source"] = data_graph["source"].str.replace(" n.c.a.", "", regex=False)

    # Convert the "value" column to numeric type
    data_graph["value"] = pd.to_numeric(data_graph["value"])

    # Save the data to a Parquet file
    data_graph.to_parquet(file_name + ".parquet", index=False)


def export_confusion_matrix_data(
    data,
    file_name="confusion_matrix",
):
    """
    export_confusion_matrix_data:
        Exports data for a confusion matrix to a Parquet file.

    @param data (pd.DataFrame): Data to export.
    @param file_name (str): Name of the output file (without file extension).
    """
    data_graph = data.loc[
        (data["TRUE_COICOP"].notna()) & (data["TRUE_COICOP"].str.len() == 10)
    ]
    data_graph = data_graph[["TRUE_COICOP", "COICOP_PREDICTED"]].reset_index(drop=True)

    # Group by true and predicted COICOP and count occurrences
    data_graph = (
        data_graph.groupby(["TRUE_COICOP", "COICOP_PREDICTED"])
        .size()
        .reset_index()
        .rename(columns={0: ""})
    )
    data_graph = data_graph.rename(
        columns={"TRUE_COICOP": "true", "COICOP_PREDICTED": "pred", "": "nb"}
    )

    # Generate all combinations of true and predicted COICOP codes
    all_combinations = pd.MultiIndex.from_product(
        [data_graph["true"].unique(), data_graph["pred"].unique()],
        names=["true", "pred"],
    )
    new_df = pd.DataFrame(index=all_combinations).reset_index()

    # Merge with the original data to fill missing combinations
    merged_df = pd.merge(data_graph, new_df, on=["true", "pred"], how="outer")
    merged_df["nb"].fillna(0, inplace=True)
    data_graph = merged_df.copy()

    # Retrieve COICOP descriptions for true and predicted COICOP codes
    data_graph = get_desc_coicop(data_graph, "true", "desc_true")
    data_graph = get_desc_coicop(data_graph, "pred", "desc_pred")

    # Sort the data by true and predicted COICOP codes
    data_graph = data_graph.sort_values(["true", "pred"])

    # Extract the first 4 digits of true COICOP codes
    data_graph["true_short"] = data_graph["true"].str[:4]

    # Retrieve COICOP descriptions for the group level
    data_graph = get_desc_coicop(
        data_graph, "true_short", "desc_groupe", category="groupe"
    )

    # Save the data to a Parquet file
    data_graph.to_parquet(file_name + ".parquet", index=False)
