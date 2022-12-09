"""
Mass testing file that grabs ML models and runs all their predictions for each given value.
Used to compare results to see how different predictions are.
Compares all predicted values for all classifiers, for visual reference.
"""

# Import pandas
import pandas as pd

# Import the models
import model_DecisionTree as DT
import model_KNN as KNN
import model_GradientBoost as GB
import model_LogisticRegression as LRG
import model_NaiveBayes as NB
import model_NeuralNetworkMLP as MLP
import model_AdaBoost as ADA
import model_NeuralNetworkRBM as RBM
import model_RandomForest as RF
import model_SVM as SVM

# Import the main dataset processing stuff
from classifier_specification import decimal, model_evaluation
from featureset_specification import TIERS_SET
from copy import deepcopy
from matplotlib import pyplot as plt


METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1-Score"]
LABEL_INDEX = "classifier_label"


def generate_all_evaluation_tables():

    for tiers in TIERS_SET:
        print("\n")
        model_params = [
            DT.with_tiers(tiers),
            GB.with_tiers(tiers),
            KNN.with_tiers(tiers),
            LRG.with_tiers(tiers),
            NB.with_tiers(tiers),
            MLP.with_tiers(tiers),
            RBM.with_tiers(tiers),
            ADA.with_tiers(tiers),
            RF.with_tiers(tiers),
            SVM.with_tiers(tiers),
        ]

        print("  |Tiers|  =  {}\n\n".format(len(tiers)))
        param_result_pairs = []
        for params in model_params:
            params["verbose"] = False
            params["final_evaluation"] = True
            param_result_pairs.append((params, model_evaluation(**params)))

        generate_evaluation_table(param_result_pairs)
        generate_evaluation_plots(param_result_pairs, tiers)

    print("\n")


def generate_evaluation_plots(param_result_pairs, tiers):

    for params, result in param_result_pairs:
        plot_details = {
            "class_names": tiers,
            "cmap": plt.cm.Blues,
            "report": result,
            "title": params[LABEL_INDEX],
            "with_avg_total": False,
        }
        save_classification_plot(**plot_details)


def generate_evaluation_table(param_result_pairs):
    model_parms, _ = zip(*param_result_pairs)

    num_column = len(METRIC_KEYS)
    max_column = len(max(METRIC_KEYS, key=lambda keyval: len(keyval)))
    max_label = len(
        max(model_parms, key=lambda params: len(params[LABEL_INDEX]))[LABEL_INDEX]
    )
    border_str = (
        "|:"
        + "-" * max_label
        + "-"
        + (("|:" + "-" * max_column + ":") * num_column)
        + "|"
    )
    format_str = (
        "| {:<"
        + str(max_label)
        + "} "
        + (("| {:^" + str(max_column) + "} ") * num_column)
        + "|"
    )

    headers = deepcopy(METRIC_KEYS)
    headers.insert(0, "")
    header_str = format_str.format(*headers)

    print(header_str)
    print(border_str)
    for params, result in param_result_pairs:
        line_items = [params[LABEL_INDEX]]
        for key in METRIC_KEYS:
            line_items.append(getDecimal(result, key))

        print(format_str.format(*line_items))


def getDecimal(d, k):
    txt = "{:<06}"
    key = k.lower()
    val = d[key] if k == METRIC_KEYS[0] else d["macro avg"][key]
    return txt.format(decimal(val))


def save_classification_plot(
    report,
    class_names,
    title="Classification report ",
    with_avg_total=False,
    cmap=plt.cm.Blues,
):
    class_amount = len(class_names)
    x_tick_marks = range(3)
    y_tick_marks = range(class_amount)

    image_dotspi = 150
    image_format = "png"
    image_detail = {
        "fname": str(class_amount) + " - " + title + "." + image_format,
        "dpi": image_dotspi,
        "format": image_format,
        "pad_inches": 0,
        "transparent": True,
    }

    metric_matrix = []
    for i in class_names:
        row = []
        for j in METRIC_KEYS[1:]:
            row.append(report[i][j.lower()])
        metric_matrix.append(row)

    plt.figure(figsize=(3, 7), dpi=image_dotspi)
    plt.imshow(metric_matrix, interpolation="nearest", cmap=cmap, origin="lower")
    plt.title(title)
    plt.colorbar()
    plt.xticks(x_tick_marks, ["Precision", "Recall", "F1 Score"], rotation=90)
    plt.yticks(y_tick_marks, class_names)
    plt.tight_layout()
    #    plt.ylabel('Tiers')
    #    plt.xlabel('Measures')
    plt.savefig(**image_detail)
    plt.clf()


def main():
    generate_all_evaluation_tables()


if __name__ == "__main__":
    main()
