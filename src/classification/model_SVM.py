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
    "kernel": "linear",
    "gamma": "scale",
    "shrinking": False,
    "probability": True,
    "decision_function_shape": "ovo",
    "random_state": STATIC_SEED,
    },
    None
]

search_grid_options = {
    "C": (
        [10 ** (-1 * i) for i in range(0, 9)]
        + list(linspace(0.005, 0.001, num=31))
        + [0.04625]
    ),
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": range(2, 17),
    "gamma": ["scale", "auto"],
    "shrinking": [False, True],
    "probability": [False, True],
    "decision_function_shape": ["ovo", "ovr"],
    "random_state": [STATIC_SEED],
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
