from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import (
    NON_BINARY_COLUMNS,
    default_feature_specification,
    which_set,
)
from numpy import linspace
from sklearn.svm import SVC
from copy import deepcopy
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#########################################
###   Model Specific Definitions:
#########################################

classifySVM = SVC()
inputScaler = ColumnTransformer(
    [("columnScaler", StandardScaler(copy=True), NON_BINARY_COLUMNS)]
)

classifier = Pipeline(
    steps=[
        ("inputScaler", inputScaler),
        ("classifySVM", classifySVM),
    ]
)

classifier = SVC()

designation = "Support Vector Classification"

hyperparameter_values = [
    #    {
    #        "C": 0.04625,
    #        "decision_function_shape": "ovo",
    #        "gamma": "scale",
    #        "kernel": "linear",
    #        "probability": True,
    #        "random_state": STATIC_SEED,
    #        "shrinking": False,
    #    },
    None,
    {
        "C": 16.457142857142856,
        "class_weight": "balanced",
        "decision_function_shape": "ovo",
        "gamma": "scale",
        "kernel": "rbf",
        "probability": True,
        "random_state": 4178261698,
        "shrinking": True,
    }
    # {'C': 0.002870967741935484, 'class_weight': 'balanced', 'decision_function_shape': 'ovo', 'kernel': 'linear', 'probability': True, 'random_state': 4178261698, 'shrinking': False}
]


# search_grid_options = {
#    "C": list(linspace(0.002838709677419355 - 0.001, 0.002838709677419355 + 0.001, num=32)),
#    "class_weight": ['balanced'],
#    "decision_function_shape": ["ovo"],
#    "kernel": ["linear"], # [ "rbf", "sigmoid"], # [ "poly" ],
#    "probability": [True],
#    "random_state": [STATIC_SEED],
#    "shrinking": [True],
# }


search_grid_options = {
    "C": [10 ** (-1 * i) for i in range(4)],
    "class_weight": ["balanced", None],
    "decision_function_shape": ["ovo"],
    "gamma": ["scale", "auto"],
    "kernel": ["rbf"],  # [ "rbf", "sigmoid"], # [ "poly" ],
    "probability": [True],
    "random_state": [STATIC_SEED],
    "shrinking": [True],
}


# search_grid_options = {
#    "C": list(linspace(0.01, 0.7, num=9)) + [0.6],
#    "class_weight": [None],
#    "decision_function_shape": ["ovo"],
#    "gamma": ["scale"],
#    "kernel": ["sigmoid"],
#    "probability": [True],
#    "random_state": [STATIC_SEED],
#    "shrinking": [True],
# }


# search_grid_options = {
#    "C": (
#        [10 ** (-1 * i) for i in [1,2] ]
##
##        + list(linspace(0.005, 0.001, num=31))
##        + [0.04625]
#    ),
#    "coef0": list(linspace(-7, -5, num=3)),
#    "class_weight": ['balanced'],
#    "decision_function_shape": ["ovo"],
#    "degree": [2,3,4],
#    "gamma": ["scale", "auto"],
#    "kernel": [ "poly" ],
#    "probability": [False],
#    "random_state": [STATIC_SEED],
#    "shrinking": [False],
# }


#########################################
###   Generic Definitions:
#########################################


evaluation_parameters = {
    "classifier_label": designation,
    "classifier": classifier,
    "dataset_params": default_feature_specification,
    "hyperspace_params": search_grid_options,
    "best_hyperparameters": hyperparameter_values,
}


def best_classifier():
    return classifier.set_params(hyperparameter_values)


def with_tiers(tiers):
    params = deepcopy(evaluation_parameters)
    params["dataset_params"]["class_names"] = tiers
    params["best_hyperparameters"] = hyperparameter_values[which_set(tiers)]
    return params


def main():
    model_evaluation(**evaluation_parameters)


if __name__ == "__main__":
    main()
