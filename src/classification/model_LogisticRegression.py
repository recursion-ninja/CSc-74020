from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification
from numpy import linspace
from sklearn.linear_model import LogisticRegression
from copy import deepcopy


#########################################
###   Model Specific Definitions:
#########################################


classifier = LogisticRegression()

designation = "Logistic Regression"

search_grid_options = {
    "penalty": ["elasticnet", "l1", "l2"],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "C": [20 ** (-1 * i) for i in range(1, 6)],
    "tol": [10 ** (-1 * i) for i in range(1, 6)],
    "max_iter": [10 ** (1 + i) for i in range(1, 4)],
    "random_state": [STATIC_SEED],
    "l1_ratio": linspace(0, 1, num=13),
}
hyperparameter_values = {
    "penalty": "l2",
    "solver": "lbfgs",
    "C": 0.05,
    "tol": 0.1,
    "max_iter": 10000,
    "random_state": STATIC_SEED,
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


def tier_parameters(elo_bound):
    params = deepcopy(evaluation_parameters)
    params["dataset_params"]["standardized_label_classes"] = elo_bound
    return params


def main():
    model_evaluation(**evaluation_parameters)


if __name__ == "__main__":
    main()