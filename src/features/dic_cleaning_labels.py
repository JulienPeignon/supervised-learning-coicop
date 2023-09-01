"""
A dictionary encompassing label cleaning rules, leveraging regular expressions (regex) for pattern matching.

Crafted by Jeremy Lhour.
"""

# List of "publishing" EAN beginnings (books, periodicals, etc.)
liste_ean_edition = ["378", "977", "978", "979"]

# Cleaning 'Family description
replace_values_famille = {" ": "_"}

# Cleaning multiple spaces
replace_whitespaces = {r"([ ]{2,})": " "}


# Label cleaning, just before model training
def get_dictio_cleaning(mode="base"):
    """
    get_dictio_cleaning: returns the cleaning dictionary.

    @param mode (str): selected dictionary, only "base" currently implemented.
    """
    if mode == "base":
        replace_values_ean = {
            "NON RENSEIGNE": "",
            "NON CONNUE": "",
            "INCONNU": "",
            "QUEL GROUPE:": "",
            "FIN DE SERIE": "",
            "THIRD PARTY ITEM": "",
            "DECOTE": "",
            "SOLDES": "",
            "DEGAGEMENT": "",
            "PROMO DATE COURTE": "",
            "DLC COURTE": "",
            "FIELD COLLECTED IMAGE RECEIVED": "",
            "&AMP": " ",
            "&QT": "",
            "&QUOT": "",
            "&TILT": "",
            "&GT": "",
            r"[a-zA-Z]\'": "",
            r"\'": " ",
            "\.": " ",
            ",": " ",
            ";": " ",
            r"\(": "",
            r"\)": "",
            r"\*": "",
            r"\-": "",
            r"\!": "",
            r"\?": "",
            "&": " ",
            "/": " ",
            "\+": " ",
            r"\b(LR|AOP|IGP|VPF|VBF|LAB. ROUGE|BIO|HALAL|VVF)\b": "#LABEL",  # Update
            r"\b\d*(MOIS|JOUR|SEMAINE)[S]?\b": "#MATURATION",  # Update
            r"\b(HOMME|FEMME|GARÇON|FILLE)\b": "#SEXE",  # Update
            r"^\d+(?!\S*\/)\b": "#LOT",  # Update: replace "3 savons" by "#LOT savons"
            r"\d+\s?(V|W)\b": "#PUISSANCE",
            r"\d+\s?(X|\*)\s?\d*\s?(G)": "#LOT #POIDS",  # Update: replace '3x140g' by '#LOT #POIDS' instead of '3X #POIDS'
            r"\b\d*\.?\d*\s?(K?GR?)\b": " #POIDS ",
            r"\d+\s?(C?M?)\s?(X)\s?\d+\s?(C|M|CM)+\b": "#DIMENSION",  # Update: consider dimensions like '130x160 cm'
            r"\d+\.?\d*\s?(C?MM?)\b": " #DIMENSION ",
            r"\d+\.?\d*\s?([CM]?L)\b": " #VOLUME ",
            r"\d+\.?\d*\s?%": " #POURCENTAGE ",
            r"\b\d+\s?(P)\b": "#LOT",  # Update: replace 'Boxer 3p' by 'BOXER #LOT' instead of 'BOXER 3P'
            r"\d+\s?(X|\*)\s?\d*\b": " #LOT ",
            r"\d*\s?(X|\*)\s?\d+\b": " #LOT ",
            r"\b(LOT)\s(DE)\b": "#LOT",  # Update: replace 'lot de XXX' by '#LOT'
            r"\d+\.?\d*\s?(CT)\b": " #UNITE ",
            r"(\sX*S\b)|(\sM\b)|(\sX*L\b)": " #TAILLE ",
            r"\s\d{2,}\/\d{2,}\b": " #TAILLE ",
            r"\s\d+\b": " ",
            r"^\d+ ": "",
            r"^[0-9]+$": "",  # to eliminate inadvertently introduced barcodes
            r"[^a-zA-Z\d\s#À-ÖØ-öø-ÿ]": "",  # deletes all non-alphanumeric characters (Update: except accents)
            r"\b(\d+)\b": "",  # Update: remove numbers
        }
    else:
        raise ValueError("Mode selected is non-existent.")
    return replace_values_ean


if __name__ == "__main__":
    pass
