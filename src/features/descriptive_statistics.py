"""
Collection of functions dedicated to generating descriptive statistics.

These are combined in the 'descriptive_statistics' function for comprehensive data analysis.
"""


def nb_occurences_ean(data, ean_column="EAN_FINAL"):
    """
    nb_occurences_ean:
    Returns the number and frequency of EANs appearing X times (X in [1,...,10+])

    @param data (pd.DataFrame): dataframe to use
    @param ean_column (str): name of the column containing the EANs
    """
    freq = data.value_counts(subset=ean_column).value_counts()

    # Iterate over each unique counts
    for item in freq.items():
        if item[0] < 10:
            print(
                "Nb. of EANs appearing {} times: {}, i.e. {:.2%}".format(
                    item[0], item[1], item[1] / freq.sum()
                )
            )
        else:
            break

    # Print count and frequency of EANs appearing 10+ times
    print(
        "Nb. of EANs appearing 10+ times: {}, i.e. {:.2%}".format(
            freq[freq.index > 9].sum(), freq[freq.index > 9].sum() / freq.sum()
        )
    )


def nb_occurences_label(data, label_column="DESCRIPTION_EAN_FINAL"):
    """
    nb_occurences_label:
    Returns the number and frequency of labels appearing X times (X in [1,...,10+])

    @param data (pd.DataFrame): dataframe to use
    @param label_column (str): name of the column containing the labels
    """
    freq = data.value_counts(subset=label_column).value_counts()

    # Iterate over each unique counts
    for item in freq.items():
        if item[0] < 10:
            print(
                "Nb. of labels appearing {} times: {}, i.e. {:.2%}".format(
                    item[0], item[1], item[1] / freq.sum()
                )
            )
        else:
            break

    # Print count and frequency of labels appearing 10+ times
    print(
        "Nb. of labels appearing 10+ times: {}, i.e. {:.2%}".format(
            freq[freq.index > 9].sum(), freq[freq.index > 9].sum() / freq.sum()
        )
    )


def number_labels_per_ean(
    data, ean_column="EAN_FINAL", label_column="DESCRIPTION_EAN_FINAL"
):
    """
    number_labels_per_ean:
    Returns the number and frequency of EANs having X labels (X in [1,...,10+])

    @param data (pd.DataFrame): dataframe to use
    @param ean_column (str): name of the column containing the EANs
    @param label_column (str): name of the column containing the labels
    """
    # Group by EAN and calculate number of unique labels for each EAN
    nb_labels = (
        data[[ean_column, label_column]]
        .groupby(ean_column)
        .agg({label_column: list})[label_column]
        .apply(lambda x: list(set(x)))
        .str.len()
        .value_counts()
    )

    # Iterate over each unique count of labels
    for item in nb_labels.items():
        # If an EAN has less than 10 unique labels, print the count and frequency
        if item[0] < 10:
            print(
                "Nb. of EANs with {} labels: {}, i.e. {:.2%}".format(
                    item[0], item[1], item[1] / nb_labels.sum()
                )
            )
        else:
            break

    # Print count and frequency of EANs having 10+ unique labels
    print(
        "Nb. of EANs with 10+ labels: {}, i.e. {:.2%}".format(
            nb_labels[nb_labels.index > 9].sum(),
            nb_labels[nb_labels.index > 9].sum() / nb_labels.sum(),
        )
    )


def number_ean_per_label(
    data, ean_column="EAN_FINAL", label_column="DESCRIPTION_EAN_FINAL"
):
    """
    number_ean_per_label:
    Returns the number and frequency of labels having X EANs (X in [1,...,10+])

    @param data (pd.DataFrame): dataframe to use
    @param ean_column (str): name of the column containing the EANs
    @param label_column (str): name of the column containing the labels
    """
    # Group by label and calculate number of unique EANs for each label
    nb_eans = (
        data[[ean_column, label_column]]
        .groupby(label_column)
        .agg({ean_column: list})[ean_column]
        .apply(lambda x: list(set(x)))
        .str.len()
        .value_counts()
    )

    # Iterate over each unique count of EANs
    for item in nb_eans.items():
        # If a label has less than 10 unique EANs, print the count and frequency
        if item[0] < 10:
            print(
                "Nb. of labels with {} EANs: {}, i.e. {:.2%}".format(
                    item[0], item[1], item[1] / nb_eans.sum()
                )
            )
        else:
            break

    # Print count and frequency of labels having 10+ unique EANs
    print(
        "Nb. of labels with 10+ EANs: {}, i.e. {:.2%}".format(
            nb_eans[nb_eans.index > 9].sum(),
            nb_eans[nb_eans.index > 9].sum() / nb_eans.sum(),
        )
    )


def descriptive_statistics(
    data,
    ean_column="EAN_FINAL",
    label_column="DESCRIPTION_EAN_FINAL",
    coicop_column="TRUE_COICOP",
    sales_column="CA",
):
    """
    descriptive_statistics:
    Main function to generate descriptive statistics for the dataset. It leverages
    previously defined functions for detailed breakdowns.

    @param data (pd.DataFrame): The dataframe containing the data
    @param ean_column (str): The column in the dataframe that contains EANs
    @param label_column (str): The column in the dataframe that contains labels
    @param coicop_column (str): The column in the dataframe containing the true COICOP 6 digits code
    @param sales_column (str): The column in the dataframe containing sales revenue
    """

    # Count the total number of rows in the data
    n = len(data)
    print("The data-set contains {} rows".format(n))
    print("------")

    # Count the unique EANs in the data by dropping duplicates
    n = len(data.drop_duplicates(subset=ean_column))
    print("The data-set contains {} unique EANs".format(n))
    print("------")

    # Count the unique labels in the data by dropping duplicates
    n = len(data.drop_duplicates(subset=label_column))
    print("The data-set contains {} unique labels".format(n))
    print("------")

    # Count the number of non-missing values in the COICOP column
    n = data[coicop_column].notna().sum()
    n_freq = n / len(data)
    print(
        "The data-set contains {} products with known COICOP 6 digits code, i.e {:.2%}".format(
            n, n_freq
        )
    )

    # Calculate the proportion of sales revenue from products with a known COICOP code
    n = data[sales_column][data[coicop_column].notna()].sum() / data[sales_column].sum()
    print("Products with known COICOP code represent {:.2%} of sales revenue".format(n))
    print("------")

    # Count the number of EANs of length 8
    n = len(data.loc[data[ean_column].str.len() == 8])
    n_freq = n / len(data)
    print("The data-set contains {} EANs of length 8, i.e {:.2%}".format(n, n_freq))

    # Calculate the proportion of sales revenue from EANs of length 8
    n = (
        data.loc[data[ean_column].str.len() == 8][sales_column].sum()
        / data[sales_column].sum()
    )
    print("EANs of length 8 represent {:.2%} of sales revenue".format(n))
    print("------")

    # Count the number of EANs of length 13
    n = len(data.loc[data[ean_column].str.len() == 13])
    n_freq = n / len(data)
    print("The data-set contains {} EANs of length 13, i.e {:.2%}".format(n, n_freq))

    # Calculate the proportion of sales revenue from EANs of length 13
    n = (
        data.loc[data[ean_column].str.len() == 13][sales_column].sum()
        / data[sales_column].sum()
    )
    print("EANs of length 13 represent {:.2%} of sales revenue".format(n))
    print("------")

    # Count the number of EANs that start with "0" (considered as questionable EANs)
    n = len(data.loc[data[ean_column].str.match(r"^0+")])
    n_freq = n / len(data)
    print("The data-set contains {} questionable EANs, i.e {:.2%}".format(n, n_freq))

    # Calculate the proportion of sales revenue from questionable EANs
    n = (
        data.loc[data[ean_column].str.match(r"^0+")][sales_column].sum()
        / data[sales_column].sum()
    )
    print("Questionable EANs represent {:.2%} of sales revenue".format(n))
    print("------")

    # Generate detailed breakdown of EAN occurrence statistics
    nb_occurences_ean(data, ean_column=ean_column)
    print("------")

    # Generate detailed breakdown of label occurrence statistics
    nb_occurences_label(data, label_column=label_column)
    print("------")

    # Generate detailed breakdown of the number of labels per EAN
    number_labels_per_ean(data, ean_column=ean_column, label_column=label_column)
    print("------")

    # Generate detailed breakdown of the number of EANs per label
    number_ean_per_label(data, ean_column=ean_column, label_column=label_column)
