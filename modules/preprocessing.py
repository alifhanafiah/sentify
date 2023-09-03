import os
import pandas as pd

from modules.preprocessing_utils import (
    case_folding,
    remove_tweet_special,
    remove_number,
    remove_punctuation,
    remove_whitespace_LT,
    remove_whitespace_multiple,
    remove_single_char,
    word_tokenize_wrapper,
    stopwords_removal,
    get_stemmed_term,
)


def preprocessed():
    # open the dataset
    filename = "tweets_dataset.csv"
    save_location = os.path.join("database", filename)
    df = pd.read_csv(save_location, delimiter=";")

    # case folding
    df["Text_Lower"] = df["Text"].apply(case_folding)

    # cleaning
    df["Text_Cleaning"] = df["Text_Lower"].apply(remove_tweet_special)
    df["Text_Cleaning"] = df["Text_Cleaning"].apply(remove_number)
    df["Text_Cleaning"] = df["Text_Cleaning"].apply(remove_punctuation)
    df["Text_Cleaning"] = df["Text_Cleaning"].apply(remove_whitespace_LT)
    df["Text_Cleaning"] = df["Text_Cleaning"].apply(remove_whitespace_multiple)
    df["Text_Cleaning"] = df["Text_Cleaning"].apply(remove_single_char)

    # tokenizing
    df["Text_Token"] = df["Text_Cleaning"].apply(word_tokenize_wrapper)

    # stopword removal
    df["Text_Token_Stop"] = df["Text_Token"].apply(stopwords_removal)

    # stemming
    df["Text_Token_Stop_Stem"] = df["Text_Token_Stop"].apply(get_stemmed_term)

    # save preprocessed dataset after preprocessing
    df.to_csv(os.path.join("database", "preprocessed_dataset.csv"), index=False)

    # choose the column to show
    df_selected = df[
        [
            "Text",
            "Text_Lower",
            "Text_Cleaning",
            "Text_Token",
            "Text_Token_Stop",
            "Text_Token_Stop_Stem",
            "Label",
        ]
    ]

    # Customize the DataFrame columns and index names
    columns = [
        "Tweets",
        "Case Folding",
        "Cleaning",
        "Tokenization",
        "Stopwords Removal",
        "Stemming",
        "Label",
    ]
    df_selected.columns = columns

    # Add a new column 'No' with the desired index values
    df_selected.insert(0, "No", range(1, len(df_selected) + 1))

    # Set the 'No' column as the index
    df_selected.set_index("No")

    # choose the head of the document
    df_head = df_selected.head()

    # Convert the DataFrame to an HTML table
    data_preprocessed_head = df_head.to_html(index=False)

    return data_preprocessed_head
