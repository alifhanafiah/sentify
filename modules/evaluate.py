import os
import pandas as pd
import pickle
from flask import request
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from werkzeug.utils import secure_filename

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
from modules.file_utils import allowed_file
from modules.analysis_utils import (
    join_text_list,
)
from modules.evaluate_utils import plot_confusion_matrix, generate_pie_chart_result


def evaluate_model_and_predict():
    # Load the variables from the pickle file
    filename = "data.pkl"
    save_location = os.path.join("database", filename)

    # show the chart if the file already exist
    if os.path.exists(save_location):
        with open(save_location, "rb") as f:
            X_train, X_test, y_train, y_test = pickle.load(f)

        # training data
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # predict data
        y_preds = model.predict(X_test)

        # accuracy score
        acc_score = accuracy_score(y_test, y_preds)
        acc_score_percentage = f"{acc_score * 100:.2f}%"

        # macro: This calculates the metric for each class independently and
        # then takes the unweighted average of all class scores.
        # Precision score (macro average)
        precision = precision_score(y_test, y_preds, average="macro")
        precision_percentage = f"{precision * 100:.2f}%"

        # Recall score (macro average)
        recall = recall_score(y_test, y_preds, average="macro")
        recall_percentage = f"{recall * 100:.2f}%"

        # F1-score (macro average)
        f1 = f1_score(y_test, y_preds, average="macro")
        f1_percentage = f"{f1 * 100:.2f}%"

        # generate confussion matrix plot
        classes = ["Negative", "Neutral", "Positive"]
        chart_img_path = plot_confusion_matrix(y_test, y_preds, classes)

        # save the model
        filename = "classifier_model.pkl"
        save_location = os.path.join("database", filename)
        with open(save_location, "wb") as f:
            pickle.dump(model, f)
    else:
        acc_score_percentage = "None"
        precision_percentage = "None"
        recall_percentage = "None"
        f1_percentage = "None"
        chart_img_path = "None"

    """
        PREDICT
    """

    # variables for the parameter
    error_msg = None
    success_msg = None
    data_table = None

    chart_img_path_result = None

    filename = "new_dataset.csv"
    save_location_pred = os.path.join("database", filename)

    # predict new dataset
    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            error_msg = "No file selected."
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            try:
                file.save(save_location_pred)

                success_msg = "File uploaded successfully."

            except Exception as e:
                error_msg = "An error occurred while saving the file."
        else:
            error_msg = "Invalid file. Only CSV files are allowed."

    # show the table if the file already exist
    if os.path.exists(save_location_pred):
        # load the vectorizer
        filename = "vectorizer.pkl"
        save_location_vec = os.path.join("database", filename)
        with open(save_location_vec, "rb") as f:
            cvect, tfidf, IDF_vector = pickle.load(f)

        # load the model
        filename = "classifier_model.pkl"
        save_location_model = os.path.join("database", filename)
        with open(save_location_model, "rb") as f:
            model = pickle.load(f)

        # Read the CSV file using pandas
        df_pred = pd.read_csv(save_location_pred, delimiter=";")

        # preprocessing
        # case folding
        df_pred["Text_Lower"] = df_pred["Text"].apply(case_folding)

        # cleaning
        df_pred["Text_Cleaning"] = df_pred["Text_Lower"].apply(remove_tweet_special)
        df_pred["Text_Cleaning"] = df_pred["Text_Cleaning"].apply(remove_number)
        df_pred["Text_Cleaning"] = df_pred["Text_Cleaning"].apply(remove_punctuation)
        df_pred["Text_Cleaning"] = df_pred["Text_Cleaning"].apply(remove_whitespace_LT)
        df_pred["Text_Cleaning"] = df_pred["Text_Cleaning"].apply(
            remove_whitespace_multiple
        )
        df_pred["Text_Cleaning"] = df_pred["Text_Cleaning"].apply(remove_single_char)

        # tokenizing
        df_pred["Text_Token"] = df_pred["Text_Cleaning"].apply(word_tokenize_wrapper)

        # stopword removal
        df_pred["Text_Token_Stop"] = df_pred["Text_Token"].apply(stopwords_removal)

        # stemming
        df_pred["Text_Token_Stop_Stem"] = df_pred["Text_Token_Stop"].apply(
            get_stemmed_term
        )

        # convert to string before tf idf
        df_pred["Text_String"] = df_pred["Text_Token_Stop_Stem"].apply(join_text_list)

        # tf idf
        TF_vector_new = cvect.transform(df_pred["Text_String"])
        normalized_TF_vector_new = normalize(TF_vector_new, norm="l1", axis=1)
        tfidf_mat_new = normalized_TF_vector_new.multiply(IDF_vector).toarray()

        # predict the result
        predict_result = model.predict(tfidf_mat_new)
        df_pred["Predict_Result"] = predict_result

        # plot the result
        chart_img_path_result = generate_pie_chart_result(df_pred)

        # choose the column to show
        df_pred_selected = df_pred[
            [
                "Text",
                "Text_Lower",
                "Text_Cleaning",
                "Text_Token",
                "Text_Token_Stop",
                "Text_Token_Stop_Stem",
                "Predict_Result",
            ]
        ]

        # Customize the DataFrame columns and index names
        columns_pred = [
            "Tweets",
            "Case Folding",
            "Cleaning",
            "Tokenization",
            "Stopwords Removal",
            "Stemming",
            "Result",
        ]
        df_pred_selected.columns = columns_pred

        # saving the result
        filename = "predicted_dataset.csv"
        save_location_pred = os.path.join("database", filename)
        df_pred_selected.to_csv(save_location_pred, index=False)

        # check the size
        dataset_size = df_pred_selected.shape

        # Add a new column 'No' with the desired index values
        df_pred_selected.insert(0, "No", range(1, len(df_pred_selected) + 1))

        # Set the 'No' column as the index
        df_pred_selected.set_index("No")

        # take just the head
        df_pred_selected_head = df_pred_selected.head()

        # Convert the DataFrame to an HTML table
        data_table = df_pred_selected_head.to_html(index=False)
    else:
        dataset_size = 0

    return (
        acc_score_percentage,
        precision_percentage,
        recall_percentage,
        f1_percentage,
        chart_img_path,
        chart_img_path_result,
        error_msg,
        success_msg,
        dataset_size,
        data_table,
    )
