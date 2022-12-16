from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification, which_set
from numpy import linspace
from sklearn.neural_network import MLPClassifier
from copy import deepcopy


#########################################
###   Model Specific Definitions:
#########################################


classifier = MLPClassifier()

designation = "Multi-layer Perceptron"

beta_candidates_vals = (
    [10 ** (-1 * i) for i in range(1, 4)]
    + list(linspace(0.2, 0.8, num=7))
    + [(((10**i) - 1) / (10**i)) for i in range(1, 4)]
)

hyperparameter_values = [
    {
        "activation": "logistic",
        "alpha": 0.1,
        "beta_1": 0.8,
        "beta_2": 0.99,
        "early_stopping": False,
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "random_state": STATIC_SEED,
        "solver": "adam",
    },
    #    None,
    {
        "activation": "logistic",
        "alpha": 0.1,
        "beta_1": 0.999,
        "beta_2": 0.99,
        "early_stopping": False,
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "random_state": 4178261698,
        "solver": "adam",
    },
]

search_grid_options = {
    "activation": ["logistic"],
    "alpha": [0.1],
    "beta_1": beta_candidates_vals,
    "beta_2": beta_candidates_vals,
    "early_stopping": [False],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "learning_rate_init": [10 ** (i - 5) for i in range(0, 10)],
    "random_state": [STATIC_SEED],
    "solver": ["adam"],
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
    return classifier.set_params(**hyperparameter_values)


def with_tiers(tiers):
    params = deepcopy(evaluation_parameters)
    params["dataset_params"]["class_names"] = tiers
    params["best_hyperparameters"] = hyperparameter_values[which_set(tiers)]
    return params


def main():
    model_evaluation(**evaluation_parameters)


if __name__ == "__main__":
    main()
