"""
Mass testing file that grabs ML models and runs all their predictions for each given value.
Used to compare results to see how different predictions are.
Compares all predicted values for all classifiers, for visual reference.
"""
# Import the main dataset processing stuff
from classifier_specification import model_evaluation
from featureset_specification import TIERS_SET
from copy import deepcopy
import time

# Import the models
import model_AdaBoost as ADA
# import model_DecisionTree as DT
# import model_KNN as KNN
# import model_LogisticRegression as LRG
# import model_NaiveBayes as NB
# import model_NeuralNetworkMLP as MLP
# import model_NeuralNetworkRBM as RBM
import model_RandomForest as RF
# import model_SVM as SVM


def generate_all_evaluation_tables():

    for tier_size in TIERS_SET:
        print("\n")
        param_list = [
            ADA.elo_tier_bins(tier_size),
            # DT.elo_tier_bins(tier_size),
            # KNN.elo_tier_bins(tier_size),
            # LRG.elo_tier_bins(tier_size),
            # NB.elo_tier_bins(tier_size),
            # MLP.elo_tier_bins(tier_size),
            # RBM.elo_tier_bins(tier_size),
            RF.elo_tier_bins(tier_size),
            # SVM.elo_tier_bins(tier_size),
        ]
        print("Tier size:\t{}\n\n".format(tier_size))
        generate_evaluation_table(param_list)

    print("\n")


def generate_evaluation_table(param_list):
    label_index = "classifier_label"
    eval_results = []
    for params in param_list:
        params["verbose"] = False
        params["final_evaluation"] = True
        eval_results.append((params, model_evaluation(**params)))

    keys_wlog = list(eval_results[0][1].keys())
    num_column = len(keys_wlog)

    print(keys_wlog)

    max_column = len(max(keys_wlog, key=lambda keyval: len(keyval)))
    max_label = len(
        max(param_list, key=lambda params: len(params[label_index]))[label_index]
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

    headers = deepcopy(keys_wlog)
    headers.insert(0, "")
    header_str = format_str.format(*headers)

    print(header_str)
    print(border_str)
    for params, result in eval_results:
        for key in keys_wlog:
            result[key] = getDecimal(result, key)

        outputs = [params[label_index]] + list(result.values())
        print(format_str.format(*outputs))


def getDecimal(d, k):
    double_str = "{:<06}"
    d[k] = double_str.format(d[k])
    return d[k]


def main():
    start_time = time.time()
    generate_all_evaluation_tables()
    # print(f"Running time: {time.time() - start_time} seconds.")
    print(time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start_time)))


if __name__ == "__main__":
    main()
