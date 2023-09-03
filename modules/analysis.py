import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from modules.analysis_utils import (
    generate_pie_chart,
    convert,
    join_text_list,
    calculate_tfidf,
    convert_to_percent,
)


def analyze():
    # open the dataset
    filename = "preprocessed_dataset.csv"
    save_location = os.path.join("database", filename)
    df = pd.read_csv(save_location)

    # Get the pie chart image as an in-memory file
    chart_img_path = generate_pie_chart(df)

    # convert the label
    df["Label_Encode"] = df["Label"].apply(convert)

    # convert to string before tf idf
    df["Text_String"] = df["Text_Token_Stop_Stem"].apply(join_text_list)

    # applying tf idf
    cvect, tfidf, IDF_vector, tfidf_mat, data_ranking = calculate_tfidf(
        df["Text_String"]
    )

    # choose the column to show
    df_selected = df[["Text_Token_Stop_Stem", "Text_String", "Label", "Label_Encode"]]

    # Customize the DataFrame columns and index names
    columns = ["Tweets", "Tweets Convert String", "Label", "Encoding"]
    columns_rank = ["Term", "TF", "IDF", "Importance"]
    df_selected.columns = columns
    data_ranking.columns = columns_rank

    # Add a new column 'No' with the desired index values
    df_selected.insert(0, "No", range(1, len(df_selected) + 1))
    data_ranking.insert(0, "No", range(1, len(data_ranking) + 1))

    # Set the 'No' column as the index
    df_selected.set_index("No")
    data_ranking.set_index("No")

    # choose the head of the document
    df_analyze_head = df_selected.head()
    df_ranking_head = data_ranking.head()

    # Convert the DataFrame to an HTML table
    data_analyze_head = df_analyze_head.to_html(index=False)
    data_ranking_head = df_ranking_head.to_html(index=False)

    # make a variable for classification
    X = tfidf_mat
    y = df["Label"]

    # split the dataset
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    # find out the amount of split
    split_amount = convert_to_percent(test_size)

    # amount of data
    training_data = len(X_train)
    testing_data = len(X_test)

    # Save the data variable to a pickle file
    filename = "data.pkl"
    save_location = os.path.join("database", filename)
    with open(save_location, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)

    # Save the vectorizer to a pickle file
    filename = "vectorizer.pkl"
    save_location = os.path.join("database", filename)
    with open(save_location, "wb") as f:
        pickle.dump((cvect, tfidf, IDF_vector), f)

    return (
        chart_img_path,
        data_analyze_head,
        data_ranking_head,
        training_data,
        testing_data,
        split_amount,
    )
