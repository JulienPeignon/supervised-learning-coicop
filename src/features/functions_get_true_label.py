"""
Collection of functions aimed at labelizing initially unlabeled data, leveraging field data and IRI families for this task.

The function 'get_true_label' ultimately aggregates these methods for comprehensive data labelizing.
"""

import pandas as pd
import numpy as np
import yaml
import os

def import_yaml_config(location: str) -> dict:
    """
    import_yaml_config:
        Imports a yaml file and returns its contents as a dictionary.

    @param location (str): Path to the yaml file.
    """
    with open(location, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config


# Load configuration from a yaml file.
config = import_yaml_config("configuration/config.yaml")
path_data = config["path"]["path_data"]


def get_coicop_dict(
    category="poste",
):  # Options: poste, sous-classe, classe, groupe, division
    """
    get_coicop_dict:
        Returns a dictionary that maps COICOP 6-digit codes to their descriptions.

    @param category (str): The category of COICOP codes.
    """
    # Load COICOP data from Excel file.
    path_coicop = path_data + config["coicop"][category]
    coicop = pd.read_excel(path_coicop, header=1)
    # Clean up COICOP codes
    coicop["Code"] = coicop["Code"].str.replace("'", "")
    # Create and return dictionary of COICOP codes and their descriptions.
    return dict(zip(coicop.Code, coicop.Libellé))


def get_desc_coicop(
    data,
    coicop_column,
    output_column,
    category="poste",
):
    """
    get_desc_coicop:
        Adds a new column to the DataFrame, containing COICOP descriptions.

    @param data (pd.DataFrame): DataFrame to add the new column to.
    @param coicop_column (str): Column containing 6-digit COICOP codes.
    @param output_column (str): New column for COICOP descriptions.
    """
    # Get COICOP code-description mapping
    coicop_dict = get_coicop_dict(category=category)
    # Map COICOP codes to their descriptions and create a new column
    data[output_column] = data[coicop_column].map(coicop_dict)
    # Remove unwanted substrings and handle missing values
    data[output_column] = data[output_column].str.replace(" n.c.a.", "", regex=False)
    # data[output_column] = data[output_column].fillna("Non défini")
    return data


def same_iri(lst):
    """
    same_iri:
        Helper function that replaces all elements of a list with the unique element if it exists.
        If not, returns the original list.
    """
    unique = list(set([i for i in lst if i is not np.nan]))
    return unique * len(lst) if len(unique) == 1 else lst


def add_iri_from_ean(
    data,
    path_conversion_ean_iri=path_data
    + config["conversion_files"]["conversion_ean_iri"],
    iri_column="ID_FAMILLE",
    ean_column="EAN_FINAL",
    label_column="DESCRIPTION_EAN_FINAL"
):
    """
    add_iri_from_ean:
        Assigns IRI family to data using their EAN, filling in missing values.

    @param data: the dataframe with products to be labeled
    @param path_conversion_ean_iri: path to the file with the conversion rule from EANs to IRI families
    @param iri_column: the name of the column in the data that contains the IRI families
    @param ean_column: the name of the column in the data that contains the EANs
    @param label_column (str): Column name holding the labels.
    """
    initial_number_iri = len(data.loc[data[iri_column].notna()])
    print("-- Initial number of IRI family available: {} --".format(initial_number_iri))
    # Read EAN to IRI family conversion data
    conversion_ean_iri = pd.read_csv(
        path_conversion_ean_iri, dtype={"ean": str, "id_famille": str}
    )
    conversion_ean_iri_dict = dict(
        zip(conversion_ean_iri.ean, conversion_ean_iri.id_famille)
    )
    # Group by EAN and generate new IRI family IDs
    data_group = (
        data.reset_index().groupby(ean_column).agg({"index": list}).reset_index()
    )
    data_group["NEW_ID_FAMILLE"] = data_group[ean_column].map(conversion_ean_iri_dict)
    # Merge new IRI family IDs into original data
    data = (
        data.reset_index()
        .merge(
            data_group.drop(columns=ean_column).explode("index"),
            on="index",
        )
        .drop(columns="index")
    )
    # Fill missing IRI family IDs with new ones
    data[iri_column] = data[iri_column].fillna(data["NEW_ID_FAMILLE"])
    # Ensure same IRI for all instances of a cleaned label
    data_group = (
        data.reset_index()
        .groupby(label_column)
        .agg({iri_column: list, "index": list})
        .reset_index()
    )
    data_group["TEST_NEW_IRI"] = data_group[iri_column].apply(same_iri)
    # Merge new IRI family IDs into original data
    data = (
        data.reset_index()
        .merge(
            data_group.drop(columns=[label_column, iri_column]).explode(
                ["TEST_NEW_IRI", "index"]
            ),
            on="index",
        )
        .drop(columns="index")
    )
    # Fill missing IRI family IDs with new ones
    data[iri_column] = data[iri_column].fillna(data["TEST_NEW_IRI"])
    # Report on IRI family ID additions
    final_number_iri = len(data.loc[data[iri_column].notna()])
    print(
        "-- New IRI family added: {} --".format(final_number_iri - initial_number_iri)
    )
    return data.drop(columns=["TEST_NEW_IRI", "NEW_ID_FAMILLE"])


def true_label_from_iri(
    data,
    path_conversion_iri=path_data + config["conversion_files"]["conversion_iri"],
    iri_column="ID_FAMILLE",
):
    """
    true_label_from_iri:
        Creates a new column with the true COICOP by using the IRI family (if available).

    @param data: the dataframe with products to be labeled
    @param path_conversion_iri: path to the file with the conversion rule from IRI families to COICOP codes
    @param iri_column: the name of the column in the data that contains the IRI families
    """
    # Read IRI family to COICOP conversion data
    conversion_iri = pd.read_csv(
        path_conversion_iri,
        dtype={"code_coicop_poste": str, "id_famille": str},
        sep=";",
    )
    # Add "0" in front of code of length 5
    condition_len_5 = conversion_iri["code_coicop_poste"].str.len() == 5
    conversion_iri.loc[condition_len_5, "code_coicop_poste"] = (
        "0" + conversion_iri.loc[condition_len_5, "code_coicop_poste"]
    )
    # Add "." to obtain the code in the right format
    conversion_iri.loc[
        conversion_iri["code_coicop_poste"].notna(), "code_coicop_poste"
    ] = conversion_iri.loc[
        conversion_iri["code_coicop_poste"].notna(), "code_coicop_poste"
    ].apply(
        lambda x: x[0] + ".".join(x[i : i + 1] for i in range(1, len(x)))
    )
    conversion_iri_dict = dict(
        zip(
            conversion_iri.id_famille, conversion_iri.code_coicop_poste
        )  # formule_classement: COICOP code (10 digits)
    )
    # Create new column for COICOP codes
    data["TRUE_COICOP"] = data[iri_column].map(conversion_iri_dict)
    # Clean up COICOP codes and handle missing values
    data["TRUE_COICOP"] = data["TRUE_COICOP"].replace("nan", np.nan)
    number_label = len(data.loc[data["TRUE_COICOP"].notna() & data[iri_column].notna()])
    number_iri = len(data.loc[data[iri_column].notna()])
    print(
        "-- Label created from IRI families: {} labels, i.e. it converted {:.2%} of IRI families at our disposal --".format(
            number_label, number_label / number_iri
        )
    )
    nb_9 = len(data.loc[data["TRUE_COICOP"] == "99.9.9.9.9"])
    print(
        "-- There are {} products out of the range of study (99.9.9.9.9) --".format(
            nb_9
        )
    )
    # Add description of true COICOP codes
    return get_desc_coicop(data, "TRUE_COICOP", "DESC_COICOP_TRUE")


def true_label_from_field(
    data,
    path_data_field=path_data + config["data_raw"]["sample_field"],
    path_conversion_variete=path_data
    + config["conversion_files"]["conversion_variete"],
    brand_column="ENSEIGNE",
    brand="LIDL",
    ean_column="CODE_BARRES",
    label_column="LIBELLE",
    iri_column="ID_FAMILLE",
):
    """
    true_label_from_field:
        This function attempts to labelize products based on available field data. It first loads the field data,
        which presumably contains accurate labels for certain products. This labeling will be used in later steps
        to supplement or correct the labels assigned through other means. The function then maps the field labels
        to a COICOP classification using a predefined conversion dictionary.

    @param data: the dataframe with products to be labeled
    @param path_data_field: path to the field data file
    @param path_conversion_variete: path to the file with the conversion rule from field labels to COICOP codes
    @param brand_column: the name of the column in the data that contains the brand information
    @param brand: the brand name to be used for filtering the data
    @param ean_column: the name of the column in the data that contains the EANs
    @param label_column: the name of the column in the data that contains the labels
    """

    # Load the field data
    data_field = pd.read_csv(
        path_data_field, dtype={ean_column: "str"}, on_bad_lines="skip", sep=","
    )

    # Filter the field data to only keep entries for the specified brand
    data_field = data_field.loc[data_field[brand_column] == brand].rename(
        columns={
            ean_column: "EAN_FINAL",
            label_column: "LABEL_VARIETE",
        }
    )

    data_field = data_field.drop_duplicates(subset="EAN_FINAL", keep="first")

    # Merge the product data with the field data, joining on EAN
    data = data.merge(
        data_field[["EAN_FINAL", "LABEL_VARIETE"]], how="left", on="EAN_FINAL"
    ).rename(str.upper, axis="columns")

    # Load the conversion rules from field labels to COICOP codes
    conversion_variete_poste = pd.read_csv(path_conversion_variete).rename(
        columns={"LIBVARIETE": "LABEL_VARIETE", "IDPOSTE": "COICOP_FIELD"}
    )

    # Clean the field labels and join the conversion rules to the product data
    conversion_variete_poste["LABEL_VARIETE"] = conversion_variete_poste[
        "LABEL_VARIETE"
    ].str.replace("DC_", "")

    conversion_variete_poste = conversion_variete_poste.drop_duplicates(
        subset="LABEL_VARIETE", keep="first"
    )

    data = data.merge(
        conversion_variete_poste[["COICOP_FIELD", "LABEL_VARIETE"]],
        how="left",
        on="LABEL_VARIETE",
    )

    # Handle quirks in the format of the COICOP codes
    condition_end_0 = data["COICOP_FIELD"].notna() & data["COICOP_FIELD"].astype(
        str
    ).str.endswith(".0")
    data.loc[condition_end_0, "COICOP_FIELD"] = (
        data.loc[condition_end_0, "COICOP_FIELD"]
        .astype(str)
        .str.replace(".0", "", regex=False)
    )
    condition_len_5 = data["COICOP_FIELD"].str.len() == 5
    data.loc[condition_len_5, "COICOP_FIELD"] = (
        "0" + data.loc[condition_len_5, "COICOP_FIELD"]
    )

    # Add the "." in the coicop code
    data.loc[data["COICOP_FIELD"].notna(), "COICOP_FIELD"] = data.loc[
        data["COICOP_FIELD"].notna(), "COICOP_FIELD"
    ].apply(lambda x: x[0] + ".".join(x[i : i + 1] for i in range(1, len(x))))

    # Add descriptions of the COICOP codes
    data = get_desc_coicop(data, "COICOP_FIELD", "DESC_COICOP_FIELD")

    # Handle potential conflicts between field labels and other labels
    mask = (
        data["TRUE_COICOP"].notna()
        & data["COICOP_FIELD"].notna()
        & (data["TRUE_COICOP"] != data["COICOP_FIELD"])
    )
    data.loc[mask, "TRUE_COICOP_CONFLICT"] = data["COICOP_FIELD"]
    data.loc[mask, "DESC_COICOP_TRUE_CONFLICT"] = data["DESC_COICOP_FIELD"]

    # Fill in missing labels with field labels
    data["TRUE_COICOP"] = data["TRUE_COICOP"].fillna(data["COICOP_FIELD"])
    data["DESC_COICOP_TRUE"] = data["DESC_COICOP_TRUE"].fillna(
        data["DESC_COICOP_FIELD"]
    )

    # Print out statistics and potentially problematic labels
    number_label_total = len(
        data.loc[data["TRUE_COICOP"].notna() & data[iri_column].isna()]
    )
    number_match_field = len(data.loc[data["LABEL_VARIETE"].notna()])
    number_doublon = len(
        data.loc[
            data[iri_column].notna()
            & data["COICOP_FIELD"].notna()
            & (data["COICOP_FIELD"] == data["TRUE_COICOP"])
        ]
    )
    number_multiple_label = len(data.loc[mask])
    n = number_label_total
    print(
        "-- Label created from field data: {} labels, i.e. it converted {:.2%} of field data at our disposal --".format(
            n,
            (n + number_doublon + number_multiple_label) / number_match_field,
        )
    )
    print(
        "-- There are {} labels from field data are equals to labels from IRI families --".format(
            number_doublon
        )
    )
    print(
        "-- There are {} labels from field data that conflicts with labels from IRI families --".format(
            number_multiple_label
        )
    )
    label_not_found_field = data.loc[
        data["COICOP_FIELD"].isna() & data["LABEL_VARIETE"].notna()
    ]["LABEL_VARIETE"].unique()
    if label_not_found_field:
        print(
            "-- Labels from field data that we couldn't match to a COICOP code are: --"
        )
        for label in label_not_found_field:
            print("- " + label)
    return data.drop(columns=["COICOP_FIELD", "DESC_COICOP_FIELD", "LABEL_VARIETE"])


def get_true_label(
    data,
    path_conversion_iri=path_data + config["conversion_files"]["conversion_iri"],
    path_data_field=path_data + config["data_raw"]["sample_field"],
    path_conversion_variete=path_data
    + config["conversion_files"]["conversion_variete"],
    path_conversion_ean_iri=path_data
    + config["conversion_files"]["conversion_ean_iri"],
    brand_column="ENSEIGNE",
    brand="LIDL",
    iri_column="ID_FAMILLE",
    ean_column="CODE_BARRES",
    label_column="LIBELLE",
):
    """
    get_true_label:
        Executes a series of labeling functions to categorize the data.

    @param data (pd.DataFrame): data to use
    @param path_conversion_iri (str): path to conversion csv for IRI family to COICOP
    @param path_data_field (str): path to field data
    @param path_conversion_variete (str): path to conversion csv for 10 digits COICOP to 6 digits
    @param brand_column (str): column name with the brand name of the supermarket
    @param brand (str): name of the brand
    @param iri_column (str): column name with IRI family
    @param ean_column (str): column name with EANs
    @param label_column (str): column name with labels
    """

    # Add IRI family to data using the corresponding EAN
    data = add_iri_from_ean(
        data,
        path_conversion_ean_iri=path_conversion_ean_iri,
        iri_column=iri_column,
        ean_column="EAN_FINAL",
        label_column="DESCRIPTION_EAN_FINAL"
    )

    # Labelize data with true 6 digits COICOP code based on IRI family data
    data = true_label_from_iri(
        data,
        path_conversion_iri=path_conversion_iri,
        iri_column=iri_column,
    )

    # Use field data to labelize data with true 6 digits COICOP code
    data = true_label_from_field(
        data,
        path_data_field=path_data_field,
        path_conversion_variete=path_conversion_variete,
        brand_column=brand_column,
        brand=brand,
        ean_column=ean_column,
        label_column=label_column,
        iri_column=iri_column,
    )
    # Print proportion of data-set that has been labelized
    n = len(data.loc[data["TRUE_COICOP"].notna()])
    print("-- Labelized data represent {:.2%} of the data-set-- ".format(n / len(data)))
    print()

    return data
