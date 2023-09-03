import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

sns.set()
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_test, y_preds, classes):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_preds)

    # Create a DataFrame for the confusion matrix with custom row and column labels
    cmat_df = pd.DataFrame(cm, index=classes, columns=classes)

    # Create the heatmap using seaborn
    ax = sns.heatmap(cmat_df, annot=True, cmap="Blues", fmt="d")

    # Set the x-axis label and make it bold
    ax.set_xlabel("Predicted", fontweight="bold")

    # Set the y-axis label and make it bold
    ax.set_ylabel("Actual", fontweight="bold")

    # Move the x-axis tick labels to the top
    ax.xaxis.tick_top()

    # Set the position of the x-axis label to the top
    ax.xaxis.set_label_position("top")

    # Save the plot as a temporary image file
    image_path = os.path.join("static", "img/confmat.png")
    plt.savefig(image_path, format="png", dpi=300)

    # Close the plot to release memory
    plt.close()

    return image_path

# Function to generate the pie chart and return it as an image
def generate_pie_chart_result(df):
    label_counts = df["Predict_Result"].value_counts()

    # Plot the pie chart and add a legend with percentages
    plt.figure(figsize=(10, 8))  # Adjust the size of the pie chart (optional)

    # Customize the colors of the slices (optional)
    colors = ["#FACF32", "#FA6368", "#21CCAC"]

    # Plot the pie chart with custom colors and labels
    # Also, calculate the percentages for each category
    percentages = (label_counts / label_counts.sum()) * 100
    legend_labels = label_counts.index

    plt.pie(label_counts, colors=colors, startangle=90)

    # Add a title
    plt.title("Distribution of Sentiments")

    # Add the legend with percentages on the top left
    plt.legend(
        title="Label",
        loc="upper left",
        labels=[
            f"{label} ({percentage:.1f}% - {count})"
            for label, percentage, count in zip(
                label_counts.index, percentages, label_counts
            )
        ],
    )

    # Save the plot as a temporary image file
    image_path = os.path.join("static", "img/chart_result.png")
    plt.savefig(image_path, format="png", dpi=300)
    plt.close()

    return image_path