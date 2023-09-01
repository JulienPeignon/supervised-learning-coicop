"""
Collection of functions designed for label preprocessing prior to model application.

While certain functions target the specifics of scanner data (like remove_edition, remove_short, pre_process), others perform more standard NLP cleaning tasks (such as remove_stopwords, lemmatisation).

The function 'total_clean_label' ultimately aggregates these methods for comprehensive label cleaning.
"""

import pandas as pd
import spacy
import nltk
import re

from multiprocessing import Pool, cpu_count
from rapidfuzz import fuzz
from unidecode import unidecode
from nltk.stem.snowball import SnowballStemmer
from src.features.dic_cleaning_labels import get_dictio_cleaning

spacy_stopwords = spacy.blank("fr")
stopwords = [word.upper() for word in spacy_stopwords.Defaults.stop_words]

spacy_lemmatisation = spacy.load("fr_core_news_lg")

stemmer = SnowballStemmer(language="french")


def remove_edition(data, ean_column="EAN_FINAL"):
    """
    remove_edition:
        Discards rows where EAN starts with certain codes ("378", "977", "978", "979") indicating an edition.

    @param data (pd.DataFrame): DataFrame to clean.
    @param ean_column (str): Column name holding EANs.
    """
    # Create an 'Edition' column marking rows where EAN starts with the given codes
    data["Edition"] = data[ean_column].str[:3].isin(["378", "977", "978", "979"])

    # Separate rows with 'Edition' EANs for reporting purposes
    data_edition = data.loc[data["Edition"]]
    print(f"-- Edition : {data_edition.shape[0]} line(s) --")

    # Keep only the rows not marked as 'Edition'
    return data[data["Edition"] == False]


def remove_similar(data, c=75, label_column="DESCRIPTION_EAN", ean_column="EAN_FINAL"):
    """
    remove_similar:
        Eliminates labels that are too similar, preferring to keep the longest ones.

    @param data (pd.DataFrame): DataFrame to clean.
    @param c (float): Threshold for similarity ratio to consider labels "too similar".
    @param label_column (str): Column name holding the labels.
    @param ean_column (str): Column name holding EANs.
    """
    # Find rows with duplicate EANs
    doublons = data[[ean_column, label_column]].loc[
        data.duplicated(subset=[ean_column], keep=False)
    ]

    rm_indices = []  # indices to remove
    # Loop over each unique EAN in the duplicates
    for ean in doublons[ean_column].unique():
        items = data.loc[data[ean_column] == ean]
        # Compare each pair of labels with the same EAN
        for (index1, row1), (index2, row2) in zip(
            items[:-1].iterrows(), items[1:].iterrows()
        ):
            fz_similarity = fuzz.ratio(row1[label_column], row2[label_column])
            if fz_similarity > c:  # If labels are too similar
                # Keep the one with longer label
                rm_indices.append(
                    index2
                    if len(row1[label_column]) > len(row2[label_column])
                    else index1
                )

    print(f"-- Labels too similar : {len(rm_indices)} line(s) --")
    # Drop rows with too similar labels
    return data.drop(rm_indices, axis=0)


def pre_process(data, replace_values_ean, label_column="DESCRIPTION_EAN"):
    """
    pre_process:
        Performs initial cleaning operations on labels (making them upper case, replacing certain characters, etc.).

    @param data (pd.DataFrame): DataFrame to clean.
    @param replace_values_ean (dict): Dictionary defining character replacement rules.
    @param label_column (str): Column name holding the labels.
    """
    print("-- Replacement of certain characters --")
    # Convert labels to uppercase
    data[label_column] = data[label_column].str.upper()

    # Replace characters in labels according to the rules
    data.replace({label_column: replace_values_ean}, regex=True, inplace=True)

    # Replace multiple spaces with a single space
    replace_whitespaces = {r"([ ]{2,})": " "}
    data.replace({label_column: replace_whitespaces}, inplace=True, regex=True)
    return data


def remove_short(data, len_min=3, label_column="DESCRIPTION_EAN"):
    """
    remove_short:
        Discards rows with labels that are too short.

    @param data (pd.DataFrame): DataFrame to clean.
    @param label_column (str): Column name holding the labels.
    @param len_min (int): Minimum acceptable length for labels.
    """
    length = data[label_column].str.len()
    print(f"-- Labels too short : {sum(length<len_min)} line(s) --")
    # Keep only rows with labels of sufficient length
    return data[length >= len_min]


def remove_stopwords(data, label_column="DESCRIPTION_EAN", stop=stopwords):
    """
    remove_stopwords:
        Removes stopwords from labels.

    @param data (pd.DataFrame): DataFrame to clean.
    @param label_column (str): Column name holding the labels.
    @param stop (list): List of stopwords to be removed.
    """
    print("-- Removing stopwords --")
    # Update labels by removing stopwords from each
    data[label_column] = data[label_column].apply(
        lambda x: " ".join([word for word in x.split() if word not in (stop)])
    )
    return data


def stemming(data, label_column="DESCRIPTION_EAN"):
    """
    stemming:
        Applies stemming to labels using nltk.

    @param data (pd.DataFrame): DataFrame to clean.
    @param label_column (str): Column name holding the labels.
    """
    print("-- Stemming --")
    # Update labels by replacing each word with its stem
    data[label_column] = (
        data[label_column]
        .apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
        .str.upper()
    )
    return data


def lematise_text(text):
    """
    lematise_text:
        Helper function to apply lemmatization to a given text.

    @param text (str): Text to lemmatize.
    """
    # Use Spacy to lemmatize the text, replace multi-space sequences with single space, and convert to uppercase
    tokens = [token.lemma_ for token in spacy_lemmatisation(text)]
    processed_text = re.sub("#\s", "#", " ".join(tokens).upper())
    return processed_text


def lemmatisation(data, label_column="DESCRIPTION_EAN"):
    """
    lemmatisation:
        Applies lemmatization to labels using Spacy.

    @param data (pd.DataFrame): DataFrame to clean.
    @param label_column (str): Column name holding the labels.
    """
    print("-- Lemmatisation --")
    # Lemmatize each row in the dataframe
    data[label_column] = data[label_column].apply(lematise_text)
    return data


def total_clean_label(
    data,
    drop=True,
    stem=False,
    ean_column="EAN_FINAL",
    label_column="DESCRIPTION_EAN",
    label_column_final="DESCRIPTION_EAN_FINAL",
    column_not_to_drop=["EAN_FINAL", "ID_FAMILLE", "CA", "CA_PRIX_QTE"],
):
    """
    total_clean_label:
        Applies all defined cleaning functions to the labels.

    @param data (pd.DataFrame): DataFrame to clean.
    @param drop (boolean): If True, discards columns not listed in column_not_to_drop.
    @param stem (boolean): If True, applies stemming, otherwise applies lemmatization.
    @param ean_column (str): Column name holding the EANs.
    @param label_column (str): Column name holding the labels.
    @param label_column_final (str): New column name for the cleaned labels.
    @param column_not_to_drop (list): Column names not to be dropped.
    """
    df = data.copy()
    # Remove editions
    df = remove_edition(df, ean_column=ean_column)
    # Preprocess the data
    df = pre_process(df, get_dictio_cleaning(), label_column=label_column)
    # Remove stopwords
    df = remove_stopwords(df, label_column=label_column)
    # Decode any non-ASCII characters
    df[label_column] = df[label_column].apply(unidecode)
    # Remove digits from labels
    df[label_column] = df[label_column].str.replace(r"\d", "", regex=True)
    # Remove short words from labels
    df[label_column] = df[label_column].str.replace(
        r"\b(?!(?:ST|WC)\b)\w{1,2}\b", "", regex=True
    )
    # Replace multiple spaces with single space
    df[label_column] = df[label_column].str.replace(r"([ ]{2,})", " ", regex=True)
    # Apply either stemming or lemmatization
    if stem:
        df = stemming(df, label_column=label_column)
    else:
        df = lemmatisation(df, label_column=label_column)
    # Remove short labels
    df = remove_short(df, label_column=label_column)
    # Rename column to indicate final labels
    df = df.rename(columns={label_column: label_column_final})
    # Optionally, drop unwanted columns
    if drop:
        column_not_to_drop.append(label_column_final)
        columns_to_drop = set(df.columns) - set(column_not_to_drop)
        df = df.drop(columns=columns_to_drop)
    print("-- Labels cleaned! --")
    print()
    # Convert column names to upper case
    return df.rename(columns=str.upper)
