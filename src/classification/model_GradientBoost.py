from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification, which_set
from copy import deepcopy
from numpy import linspace
from sklearn.ensemble import GradientBoostingClassifier

from matplotlib import pyplot as plt

#########################################
###   Model Specific Definitions:
#########################################

classifier = GradientBoostingClassifier()

designation = "Gradient Boosting"

hyperparameter_values = [
    {
        #    "learning_rate": 0.0001,
        "loss": "log_loss",
        #    "max_depth": 2,
        "max_features": None,
        "max_leaf_nodes": None,
        #    "min_samples_leaf": 2,
        #    "min_samples_split": 2,
        #    "n_estimators": 250,
        "random_state": STATIC_SEED,
        #    "subsample": 0.4,
        "warm_start": True,
    },
    None
]


search_grid_options = {
    "learning_rate": [10.0 ** ((i - 9) / 2) for i in range(1, 5)],
    "loss": ["log_loss"],
    "max_depth": [2, 4],  # list(linspace(2, 5, num=4).astype(int)),
    "max_features": [None],
    "max_leaf_nodes": [None],
    "min_samples_leaf": [2, 4],  # list(linspace(2, 5, num=4).astype(int)),
    "min_samples_split": [2],
    "n_estimators": range(200, 501, 50),
    "random_state": [STATIC_SEED],
    "subsample": list(linspace(0.4, 0.8, num=5)),
    "warm_start": [True],
}

# {
#    "ccp_alpha": [0] + [2**i for i in range(2, 5)],
#    "learning_rate": [10 ** ((i - 4) / 2) for i in range(0, 12)],
#    "learning_rate": [10 ** (i - 4) for i in range(0, 6)],
#    "learning_rate": list(linspace(0.01, 0.2, num=5)),
#    "loss": ["log_loss"],  # , "deviance", "exponential"],
#    "max_depth": list(linspace(1, 5, num=5).astype(int)),
#    "max_features": [None],
#    "max_leaf_nodes": list(linspace(2, 10, num=5).astype(int)) + [None],
#    "min_samples_leaf": list(linspace(1, 5, num=5).astype(int)),
#    "min_samples_split": list(linspace(2, 20, num=7).astype(int)),
#    "min_weight_fraction_leaf": list(linspace(0.0, 0.5, num=5)),
#    "n_estimators":
#    "n_iter_no_change": list(linspace(1, 5, num=3).astype(int)) + [None],
#    "random_state": [STATIC_SEED],
#    "subsample": list(linspace(0.0, 1, num=11).astype(int))[1:],  # [ 0.1, 0.2, .., 0.9, 1.0 ]
#    "tol": [10 ** ((-1 / 2) * i) for i in range(4, 8)],
#    "validation_fraction": list(linspace(0.05, 0.15, num=3)),
#    "warm_start": [True],
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
