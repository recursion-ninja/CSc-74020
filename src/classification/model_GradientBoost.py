from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification
from copy import deepcopy
from numpy import linspace
from sklearn.ensemble import GradientBoostingClassifier


#########################################
###   Model Specific Definitions:
#########################################

classifier = GradientBoostingClassifier()

designation = "Gradient Boosting"

hyperparameter_values = None

search_grid_options = {
    "loss": ["log_loss"],  # , "deviance", "exponential"],
    #    "learning_rate": [10 ** ((i - 3) / 2) for i in range(0, 12)],
    "learning_rate": list(linspace(0.01, 0.2, num=5)),
    "n_estimators": [2**i for i in range(5, 7)],
    #    "subsample": list(linspace(0.0, 1, num=11).astype(int))[1:],  # [ 0.1, 0.2, .., 0.9, 1.0 ]
    "subsample": list(linspace(0.2, 0.8, num=4)),
    #    "min_samples_split": list(linspace(2, 20, num=7).astype(int)),
    #    "min_samples_split": [2,4],
    #    "min_samples_leaf": list(linspace(1, 5, num=5).astype(int)),
    #    "min_weight_fraction_leaf": list(linspace(0.0, 0.5, num=5)),
    #    "max_depth": list(linspace(1, 5, num=5).astype(int)),
    "random_state": [STATIC_SEED],
    "max_features": ["sqrt", None],
    "max_leaf_nodes": list(linspace(6, 10, num=5).astype(int)) + [None],
    "warm_start": [True],
    #    "validation_fraction": list(linspace(0.05, 0.15, num=3)),
    #    "n_iter_no_change": list(linspace(1, 5, num=3).astype(int)) + [None],
    #    "tol": [10 ** ((-1 / 2) * i) for i in range(4, 8)],
    #    "ccp_alpha": [0] + [2**i for i in range(2, 5)],
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


def elo_tier_bins(elo_bound):
    params = deepcopy(evaluation_parameters)
    params["dataset_params"]["standardized_label_classes"] = elo_bound
    return params


def main():
    model_evaluation(**evaluation_parameters)


if __name__ == "__main__":
    main()
