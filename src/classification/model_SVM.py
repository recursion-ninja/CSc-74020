from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification, which_set
from numpy import linspace
from sklearn.svm import SVC
from copy import deepcopy


#########################################
###   Model Specific Definitions:
#########################################


classifier = SVC()

designation = "Support Vector Classification"

hyperparameter_values = [
    {
        "C": 0.04625,
        "decision_function_shape": "ovo",
        "gamma": "scale",
        "kernel": "linear",
        "probability": True,
        "random_state": STATIC_SEED,
        "shrinking": False,
    },
    None,
]

search_grid_options = {
    "C": (
        [10 ** (-1 * i) for i in range(0, 9)]
        + list(linspace(0.005, 0.001, num=31))
        + [0.04625]
    ),
    "decision_function_shape": ["ovo", "ovr"],
    "degree": range(2, 17),
    "gamma": ["scale", "auto"],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "probability": [False, True],
    "random_state": [STATIC_SEED],
    "shrinking": [False, True],
}


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
