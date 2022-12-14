from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification, which_set
from numpy import linspace
from sklearn.linear_model import LogisticRegression
from copy import deepcopy


#########################################
###   Model Specific Definitions:
#########################################


classifier = LogisticRegression()

designation = "Logistic Regression"

hyperparameter_values = [
    {
        "C": 0.0307,
        "l1_ratio": 0.0,
        "max_iter": 100000,
        "penalty": "l2",
        "random_state": STATIC_SEED,
        "solver": "newton-cg",
        "tol": 0.1,
    },
    # None,
    {
        "C": 0.0025,
        "l1_ratio": 0.0,
        "max_iter": 10000,
        "penalty": "l2",
        "random_state": 4178261698,
        "solver": "lbfgs",
        "tol": 0.1,
    },
]

search_grid_options = {
    "C": list(linspace(0.001, 0.1, num=11)),  # [20 ** (-1 * i) for i in range(1, 6)],
    "l1_ratio": list(linspace(0, 0.1, num=11)),  # linspace(0, 1, num=13),
    "max_iter": [
        10 ** (4 + i) for i in range(1, 3)
    ],  # [10 ** (1 + i) for i in range(1, 4)],
    "penalty": ["l2"],  # ["elasticnet", "l1", "l2"],
    "random_state": [STATIC_SEED],
    "solver": [
        "newton-cg",
        "lbfgs",
    ],  # ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "tol": [
        0.1,
        0.01,
    ],  # list(linspace(0.01,0.1,num=11)) # [10 ** (-1 * i) for i in range(1, 6)],
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
