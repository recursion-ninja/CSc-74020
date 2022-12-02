import featureset_specification as datum
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, GridSearchCV

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


def model_evaluation(
    classifier_label,
    classifier,
    dataset_params,
    final_evaluation=False,
    hyperspace_params=None,
    best_hyperparameters=None,
    verbose=True,
):

    #################################
    ###   Data preparation
    #################################

    monster_data = datum.retrieve_monster_dataset(**dataset_params)

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
        if hyperspace_params is None:
            raise ValueError(
                "If you do not specify the 'best_hyperparameters', then you must specify 'hyperspace_params'!"
            )
        best_hyperparameters = classifier_specification(
            classifier, hyperspace_params, X_train_part, Y_train_part
        )
        print("For classifier:", classifier_label)
        print("Found best hyperparameters:\n", best_hyperparameters)

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
    return evaluate_predictions(
        Y_eval, Y_score, dataset_params["standardized_label_classes"], verbose
    )


# Given a Pandas data frame, partition the data frame into two segements.
# The first segment contains all but the last column.
# The second segment contains only the last column.
# The partitioned data frame represents the feature observation matrix
def seperate_data(data_frame):
    # Shuffle the input data to ensure
    # there are no ordering biases
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
    label_classes = list(sorted(Y_in.unique()))
    splitter = lambda x, y, n: train_test_split(
        x, y, test_size=n, random_state=STATIC_SEED, stratify=y
    )
    X_full, X_test, Y_full, Y_test = splitter(X_in, Y_in, test_size)
    X_train, X_valid, Y_train, Y_valid = splitter(X_full, Y_full, validation_size)
    return X_full, X_train, X_valid, X_test, Y_full, Y_train, Y_valid, Y_test


# We tune the model by determining which hyperparamaters perform best.
def classifier_specification(classifier, param_grid, X_train_part, Y_train_part):
    result_classifier = GridSearchCV(
        classifier,
        param_grid,
        scoring="accuracy",
        cv=4,
        verbose=1,
        n_jobs=-1,
    )
    result_classifier.fit(X_train_part, Y_train_part)
    best_hyperparameters = result_classifier.best_params_
    print("Best accuracy score found:  ", round(result_classifier.best_score_, 4))
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


def evaluate_predictions(Y_eval, Y_score, num_classes, verbose=True):
    result = {
        "Accuracy": round(metrics.accuracy_score(Y_eval, Y_score), 4),
        "Precision": round(
            metrics.precision_score(Y_eval, Y_score, average="weighted"), 4
        ),
        "Recall": round(metrics.recall_score(Y_eval, Y_score, average="weighted"), 4),
        "F1": round(metrics.f1_score(Y_eval, Y_score, average="weighted"), 4)
        #                , 'LRAP'           : round(metrics.label_ranking_average_precision_score(Y_eval, Y_score  ), 4)
        #                , 'Converge Error' : round(metrics.coverage_error(     Y_eval, Y_score                    ), 4)
        #                , 'Ranking Loss'   : round(metrics.label_ranking_loss( Y_eval, Y_score                    ), 4)
        #                , 'ROC AUC'   : round(metrics.roc_auc_score(  Y_eval, Y_score, average='weighted', multi_class='ovo', labels=list(range(0,num_classes))), 4)
    }

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
    matrix = metrics.confusion_matrix(Y_true, Y_pred)
    maxVal = max(np.concatenate(matrix).flat, key=lambda x: x)
    padLen = len(str(maxVal))

    print("  Confusion matrix:")
    for row in matrix:
        print("    ", sep="", end="")
        for col in row:
            print(str(col).rjust(padLen), " ", sep="", end="")
        print()
