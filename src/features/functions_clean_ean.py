"""
Comprehensive suite of functions intended for pre-modeling label cleansing. These methods are collectively utilized in the 'total_clean_ean' function.
"""

import numpy as np
import pylcs


def cleaning_ean(data, ean_column_input="EAN", ean_column_output="EAN_CLEAN"):
    """
    cleaning_ean:
        Strips whitespace from EAN strings and trims EANs to their appropriate length.

    @param data (pd.DataFrame): DataFrame to clean.
    @param ean_column_input (str): Column containing the original EANs.
    @param ean_column_output (str): Column to hold the cleaned EANs.
    """
    print("-- Identifying EANs of length 8 --")
    # Stripping whitespace
    data[ean_column_input] = data[ean_column_input].str.strip()

    # Using np.select to apply conditions for selecting last 8 or 13 characters based on conditions.
    data[ean_column_output] = np.select(
        condlist=[
            (data[ean_column_input].str.len() > 13)
            & ~(data[ean_column_input].str[:-7].str.match(r"^550+2")),
            (data[ean_column_input].str.len() > 8)
            & (data[ean_column_input].str[:-7].str.match(r"^(55)?0+2")),
        ],
        choicelist=[
            data[ean_column_input].str[-13:],
            data[ean_column_input].str[-8:],
        ],
        default=data[ean_column_input],
    )
    return data


def similar_ean(ean_grouped, lcs_min=4):
    """
    similar_ean:
        Changes non-standard EANs (not 8 or 13 digits long) to their similar counterpart in the group.

    @param ean_grouped (list): List of EANs to potentially change.
    @param lcs_min (int): Minimum length of common subsequence to qualify EANs as 'similar'.
    """
    # Create list of EANs not of length 8 or 13
    tested_ean = [x for x in ean_grouped if len(x) not in [8, 13]]
    for i in tested_ean:
        # Create list of EANs of length 8 or 13
        filtered_group = [x for x in ean_grouped if len(x) in [8, 13]]
        for j in filtered_group:
            # Check longest common subsequence length
            lcs = pylcs.lcs_sequence_idx(i, j)
            # If LCS length is long enough, replace original EAN with its similar counterpart
            if (len(lcs) >= lcs_min) & (lcs != "0" * len(lcs)):
                ean_grouped[ean_grouped.index(i)] = j
                break
    return ean_grouped


def reconstruct_similar_ean(
    data,
    label_column="DESCRIPTION_EAN",
    ean_column_input="EAN_CLEAN",
    ean_column_output="EAN_RECONSTRUCTED",
):
    """
    reconstruct_similar_ean:
        Apply the function 'similar_ean' to each group of EANs that share the same label.

    @param data (pd.DataFrame): DataFrame to apply reconstruction to.
    @param label_column (str): Column with labels based on which EANs are grouped.
    @param ean_column_input (str): Column with EANs to reconstruct.
    @param ean_column_output (str): Column to hold reconstructed EANs.
    """
    print("-- Reconstructing similar EANs --")
    # Group by label and get list of EANs and indices
    data_group = (
        data.reset_index()
        .groupby(label_column)
        .agg({ean_column_input: list, "index": list})
        .reset_index()
    )

    # Apply 'similar_ean' to each group's list of EANs
    data_group[ean_column_output] = data_group[ean_column_input].apply(similar_ean)

    # Merge back to original DataFrame
    data = data.reset_index().merge(
        data_group.explode([ean_column_output, "index"]).drop(
            [label_column, ean_column_input], axis=1
        ),
        on="index",
    )
    return data.drop("index", axis=1)


def remove_abnormal_ean(
    data, ean_column_input="EAN_RECONSTRUCTED", ean_column_remove="EAN_REMOVE"
):
    """
    remove_abnormal_ean:
        Removes EANs that are not of length 8 or 13.

    @param data (pd.DataFrame): DataFrame to remove abnormal EANs from.
    @param ean_column_input (str): Column with potentially abnormal EANs.
    @param ean_column_remove (str): Column indicating whether the corresponding EAN should be removed.
    """
    print("-- Removing abnormals EANs --")
    # Add boolean column indicating whether the EAN is not of length 8 or 13
    data[ean_column_remove] = ~data[ean_column_input].str.len().isin([8, 13])

    # Return two DataFrames: one without abnormal EANs, and one with only abnormal EANs.
    return (
        data[data[ean_column_remove] == False]
        .drop(ean_column_remove, axis=1)
        .reset_index(drop=True),
        data[data[ean_column_remove] == True].reset_index(drop=True),
    )


def total_clean_ean(
    data,
    drop=True,
    ean_column_input="EAN",
    ean_column_output="EAN_FINAL",
    column_not_to_drop=["DESCRIPTION_EAN", "ID_FAMILLE", "CA", "CA_PRIX_QTE"],
):
    """
    total_clean_ean:
        Orchestrates the EAN cleaning process by applying all the defined functions.

    @param data (pd.DataFrame): DataFrame to clean.
    @param drop (boolean): If True, drops all columns not defined in 'column_not_to_drop'.
    @param ean_column_input (str): Column containing the original EANs.
    @param ean_column_output (str): Column to hold the final cleaned EANs.
    @param column_not_to_drop (list): Columns not to drop in the final cleaned DataFrame.
    """
    # Start EAN cleaning
    data = cleaning_ean(data, ean_column_input=ean_column_input)

    # Reconstruct similar EANs
    data = reconstruct_similar_ean(data)

    # Remove abnormal EANs, saving them into a separate DataFrame
    data, data_remove = remove_abnormal_ean(data)

    # Rename the column to indicate these are the final, cleaned EANs
    data = data.rename(columns={"EAN_RECONSTRUCTED": ean_column_output})

    # Optionally drop unnecessary columns
    if drop:
        # Add the output column to the list of columns not to drop
        column_not_to_drop.append(ean_column_output)

        # Calculate the columns to drop
        columns_to_drop = set(data.columns) - set(column_not_to_drop)

        # Drop the unnecessary columns
        data = data.drop(columns=columns_to_drop)

    print("-- EANs cleaned! --")
    print()

    # Return the cleaned DataFrame
    return data
