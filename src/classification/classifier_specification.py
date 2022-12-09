import featureset_specification as datum
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

# It is nice to remove the deprecation warnings.
# They really distract from the important output!
# Also affects subprocesses, forcing proper behavior with parallelism.
import os
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


# A random seed, but fixed seed to use in randomized function calls.
STATIC_SEED = 0xF90B36C2
METRIC_MULTI_SEARCH = True
METRIC_AVERAGING = "macro"

decimal = lambda v: round(v, 4)


def model_evaluation(
    classifier_label,
    classifier,
    dataset_params,
    final_evaluation=False,
    hyperspace_params=None,
    best_hyperparameters=None,
    verbose=True,
):

    class_labels = dataset_params["class_names"]

    #################################
    ###   Data preparation
    #################################

    monster_data = datum.retrieve_monster_dataset(
        **dataset_params
    )  # (dataset_params['class_names'], dataset_params['decorrelate'], dataset_params['textual'])

    # Split off the last column as the label vector.
    X, Y = seperate_data(monster_data)

    # Partition out data into training, validation and testing sets.
    (
        X_train_full,
        X_train_part,
        X_valid,
        X_test,
        Y_train_full,
        Y_train_part,
        Y_valid,
        Y_test,
    ) = train_valid_test(X, Y, 0.2, 0.2)

    #################################
    ###   Model Selection
    #################################

    # If we don't already have best parameters...
    # Let's go find them!
    if best_hyperparameters is None:
        hint = "If you do not specify the 'best_hyperparameters', then you must specify 'hyperspace_params'!"
        if hyperspace_params is None:
            raise ValueError(hint)

        search_spec = {
            "classifier_label": classifier_label,
            "classifier": classifier,
            "param_grid": hyperspace_params,
            "X_train_part": X_train_part,
            "Y_train_part": Y_train_part,
        }

        best_hyperparameters = classifier_specification(**search_spec)

    #################################
    ###   Evaluate Classifier
    #################################

    # Build a new classifier incorperating what we learned from hyperparameter tuning
    # and then evaluate the classifier's perfomance.
    X_learn = None
    X_eval = None
    Y_learn = None
    Y_eval = None
    label_construction = None
    label_prediction = None

    # If we are not using the finalized model,
    # only evaluate using the validation dataset!
    if not final_evaluation:
        X_learn = X_train_part
        X_eval = X_valid
        Y_learn = Y_train_part
        Y_eval = Y_valid
        label_construction = "Partial"
        label_prediction = "Validation"

    # If we are evaluating the finalized model, use the whole dataset.
    else:
        X_learn = X_train_full
        X_eval = X_test
        Y_learn = Y_train_full
        Y_eval = Y_test
        label_construction = "Full"
        label_prediction = "Testing"

    classifier_model = classifier.set_params(**best_hyperparameters)
    classifier_model.fit(X_learn, Y_learn)
    if verbose:
        describe_model(classifier_label, best_hyperparameters)
        describe_data_set(
            X_learn, "  Using " + label_construction + " training dataset containing:"
        )

    Y_score = classifier_model.predict(X_eval)

    if verbose:
        print("Generated predictions for evaluation\n")
        describe_data_set(
            X_test, "  Using " + label_prediction + " dataset containing:"
        )
    return evaluate_predictions(Y_eval, Y_score, class_labels, verbose)


# Given a Pandas data frame, partition the data frame into two segements.
# The first segment contains all but the last column.
# The second segment contains only the last column.
# The partitioned data frame represents the feature observation matrix
def seperate_data(data_frame):
    # Shuffle the input data to ensure there are no ordering biases
    data_frame = data_frame.sample(frac=1, random_state=STATIC_SEED)
    labelColumn = data_frame.columns[-1]
    X = data_frame.loc[:, data_frame.columns != labelColumn]
    Y = data_frame[labelColumn]
    return X, Y


# Takes a feature observation matrix and a label vector along with
# a specification for the relative sizes of the requested partitions.
# Returns the inputs partitioned into 4 sets, respectively:
#   - Full Training
#   - Partial Training
#   - Validation
#   - Testing
#
# The partitions have the following relationships:
#   - Full Training ∪ Testing = Input
#   - Full Training = Partial Training ∪ Validation
#
# Intended to be convieient for model selection and tuning.
def train_valid_test(X_in, Y_in, validation_size, test_size):
    splitter = lambda x, y, n: train_test_split(
        x, y, test_size=n, random_state=STATIC_SEED, stratify=y
    )
    X_full, X_test, Y_full, Y_test = splitter(X_in, Y_in, test_size)
    X_train, X_valid, Y_train, Y_valid = splitter(X_full, Y_full, validation_size)
    return X_full, X_train, X_valid, X_test, Y_full, Y_train, Y_valid, Y_test


# We tune the model by determining which hyperparamaters perform best.
def classifier_specification(
    classifier_label, classifier, param_grid, X_train_part, Y_train_part
):

    # Conditionally:
    #   1. Use Area Under the Receive Operation Characteristic Curve
    # ... OR ...
    #   2. Consider all of:
    #         - Confusion matrix structure
    #         - F1-score
    #         - Jaccard similarity coefficient
    #         - Matthews correlation coefficient
    #         - Precision + Recall + F-score support
    #      Before refitting based on the Area Under the Receive Operation Characteristic Curve
    fitting = True
    measure = lambda x: (x + "_" + METRIC_AVERAGING)
    #    metrics = measure(precision_recall_fscore_support)
    metrics = "balanced_accuracy"

    if METRIC_MULTI_SEARCH:
        fitting = "Area under ROC"
        metrics = {
            fitting: "roc_auc_ovo_weighted",
            "Accuracy": "balanced_accuracy",
            "Precision": measure("precision"),
            "Recall": measure("recall"),
            "F1 Score": measure("f1"),
            "Jaccard": measure("jaccard"),
        }
    #        fitting = "Precision + recall + F-score support"
    #        metrics = {
    #            fitting: measure(precision_recall_fscore_support),
    #            "Accuraccy (balanced)": measure(balanced_accuracy_score),
    #
    #            "Recall": measure(recall_score),
    #            "F1 Score": measure(f1_score),
    #            "Jaccard score": measure(jaccard_score),
    #            "Matthews correlation coefficient": make_scorer(matthews_corrcoef),
    #            #            "Area under ROC": make_scorer(roc_auc_score(multi_class='ovo', average=METRIC_AVERAGING)),
    #        }
    #    hyperparameterSearch = GridSearchCV(
    #        classifier,
    #        param_grid,
    #        scoring=metrics,
    #        refit=fitting,
    #        cv=4,
    #        verbose=1,
    #        n_jobs=-1,
    #        return_train_score=True,
    #    )

    if METRIC_MULTI_SEARCH:
        #        hyperparameterSearch = GridSearchCV(
        hyperparameterSearch = GridSearchCVProgressBar(
            classifier,
            param_grid,
            scoring=metrics,
            refit=fitting,
            cv=4,
            verbose=1,
            n_jobs=4,
            return_train_score=True,
        )
    else:
        hyperparameterSearch = GridSearchCV(
            classifier,
            param_grid,
            scoring="balanced_accuracy",
            cv=4,
            verbose=1,
            n_jobs=-1,
        )
    titleOf = fitting if METRIC_AVERAGING else metrics

    hyperparameterSearch.fit(X_train_part, Y_train_part)
    best_hyperparameters = hyperparameterSearch.best_params_
    best_score = hyperparameterSearch.best_score_

    print("For classifier", classifier_label)
    print("Hyperparameters space sampled over cross-validation search.")
    print("Using", titleOf, "as final fitting metric.")
    print("Best found hyperparameters in search results with:")
    print("    Score: ", decimal(best_score))
    print("    Value: ")
    print(best_hyperparameters)
    print()

    #    scoring = metrics
    #    results = hyperparameterSearch.cv_results_
    #    print("\nResult keys:")
    #    for k in sorted(results.keys()):
    #        print("\t", k)
    #
    #    plt.figure(figsize=(13, 13))
    #    plt.title(
    #        "GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16
    #    )
    #
    #    x_name = "learning_rate"
    #
    #    plt.xlabel(x_name)
    #    plt.ylabel("Score")
    #
    #    ax = plt.gca()
    #    ax.set_xlim(10 ** (-4), 1.0)
    #    ax.set_ylim(0.1, 1)
    #    plt.xscale("log")
    #
    #    # Get the regular numpy array from the MaskedArray
    #    X_axis = np.array(results["param_" + x_name].data, dtype=float)
    #    print("\nX-Axis")
    #    print(X_axis)
    #
    #    for scorer, color in zip(sorted(scoring), ["r", "g", "b", "c", "m", "y", "k"]):
    #        for sample, style in (("train", "--"), ("test", "-")):
    #            sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
    #            sample_score_std = results["std_%s_%s" % (sample, scorer)]
    #            ax.fill_between(
    #                X_axis,
    #                sample_score_mean - sample_score_std,
    #                sample_score_mean + sample_score_std,
    #                alpha=0.1 if sample == "test" else 0,
    #                color=color,
    #            )
    #            ax.plot(
    #                X_axis,
    #                sample_score_mean,
    #                style,
    #                color=color,
    #                alpha=1 if sample == "test" else 0.7,
    #                label="%s (%s)" % (scorer, sample),
    #            )
    #
    #        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
    #        best_score = results["mean_test_%s" % scorer][best_index]
    #
    #        # Plot a dotted vertical line at the best score for that scorer marked by x
    #        ax.plot(
    #            [
    #                X_axis[best_index],
    #            ]
    #            * 2,
    #            [0, best_score],
    #            linestyle="-.",
    #            color=color,
    #            marker="x",
    #            markeredgewidth=3,
    #            ms=8,
    #        )
    #
    #        # Annotate the best score for that scorer
    #        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))
    #
    #    plt.legend(loc="best")
    #    plt.grid(False)
    #    plt.show()

    return best_hyperparameters


# Nicely renders the model and it's hyper parameters
def describe_model(classifier_label, best_hyperparameters):
    max_len = max(map(len, best_hyperparameters))

    print("")
    print("Constructed", classifier_label, "model")
    print("")
    print("  Using hyperparameters:")

    for k, v in best_hyperparameters.items():
        print("   ", ("{:<" + str(max_len) + "} =").format(k), v)

    print("")


# Define a reusable descriptor for data sets.
# Nicely renders the dimensions of the provided data set.
def describe_data_set(X, label):
    rStr = str(X.shape[0])
    cStr = str(X.shape[1])
    mLen = max(len(rStr), len(cStr))

    print(label)
    print("   ", rStr.rjust(mLen), "observations")
    print("   ", cStr.rjust(mLen), "features")
    print("")


def evaluate_predictions(Y_eval, Y_score, class_labels, verbose=True):
    report_details = {
        "digits": 4,
        "target_names": class_labels,
        "output_dict": True,
        "y_pred": Y_score,
        "y_true": Y_eval,
    }
    result = classification_report(**report_details)
    if verbose:
        inspect_confusion_matrix(Y_eval, Y_score)
        print("")
        print("  Accuracy Score: ", result["Accuracy"])
        print("  Precision Score:", result["Precision"])
        print("  Recall Score:   ", result["Recall"])
        print("  F1 Score:       ", result["F1"])
        print("")

    return result


def inspect_confusion_matrix(Y_true, Y_pred):
    matrix = confusion_matrix(Y_true, Y_pred)
    maxVal = max(np.concatenate(matrix).flat, key=lambda x: x)
    padLen = len(str(maxVal))

    print("  Confusion matrix:")
    for row in matrix:
        print("    ", sep="", end="")
        for col in row:
            print(str(col).rjust(padLen), " ", sep="", end="")
        print()
