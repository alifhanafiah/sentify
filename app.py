from flask import Flask, render_template

from config import Config
from modules import data_handling, preprocessing, analysis, evaluate

app = Flask(__name__)
app.config.from_object(Config)


# index
@app.route("/")
def index():
    return render_template("pages/home.html")


# dataset
@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    (
        columns,
        error_msg,
        success_msg,
        dataset_size,
        data_table,
    ) = data_handling.handle_uploaded_data()

    return render_template(
        "pages/dataset.html",
        columns=columns,
        error=error_msg,
        success=success_msg,
        dataset_size=dataset_size,
        data_table=data_table,
    )


# train
@app.route("/train")
def train():
    data_preprocessed_head = preprocessing.preprocessed()

    (
        chart_img_path,
        data_analyze_head,
        data_ranking_head,
        training_data,
        testing_data,
        split_amount,
    ) = analysis.analyze()

    return render_template(
        "pages/train.html",
        data_preprocessed_head=data_preprocessed_head,
        data_analyze_head=data_analyze_head,
        data_ranking_head=data_ranking_head,
        chart_img_path=chart_img_path,
        training_data=training_data,
        testing_data=testing_data,
        split_amount=split_amount,
    )


# result
@app.route("/result", methods=["GET", "POST"])
def result():
    (
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
    ) = evaluate.evaluate_model_and_predict()

    return render_template(
        "pages/result.html",
        acc_score_percentage=acc_score_percentage,
        precision_percentage=precision_percentage,
        recall_percentage=recall_percentage,
        f1_percentage=f1_percentage,
        chart_img_path=chart_img_path,
        chart_img_path_result=chart_img_path_result,
        error=error_msg,
        success=success_msg,
        dataset_size=dataset_size,
        data_table=data_table,
    )


if __name__ == "__main__":
    app.run()
