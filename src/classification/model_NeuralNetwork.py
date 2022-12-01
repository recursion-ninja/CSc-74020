from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification
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
hyperparameter_values = {
    "solver": "adam",
    "activation": "logistic",
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "alpha": 0.1,
    "beta_1": 0.8,
    "beta_2": 0.99,
    "early_stopping": False,
    "random_state": STATIC_SEED,
}
search_grid_options = {
    "activation": ["logistic"],
    "solver": ["adam"],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "learning_rate_init": [10 ** (i - 5) for i in range(0, 10)],
    "alpha": [0.1],
    "beta_1": beta_candidates_vals,
    "beta_2": beta_candidates_vals,
    "early_stopping": [False],
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
    return classifier.set_params(**hyperparameter_values)


def elo_tier_bins(elo_bound):
    params = deepcopy(evaluation_parameters)
    params["dataset_params"]["standardized_label_classes"] = elo_bound
    return params


def main():
    model_evaluation(**evaluation_parameters)


#    clf = best_classifier()
#    print("Layers  :", clf.n_layers_      )
#    print("Outputs :", clf.n_outputs_     )
#    print("Function:", clf.out_activation_)


if __name__ == "__main__":
    main()
