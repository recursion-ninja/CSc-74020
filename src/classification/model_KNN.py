from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification, which_set
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy


#########################################
###   Model Specific Definitions:
#########################################

classifier = KNeighborsClassifier()

designation = "K Nearest Neighbors"

hyperparameter_values = [
    {
        "algorithm": "ball_tree",
        "leaf_size": 5,
        "n_neighbors": 30,
        "p": 1,
        "weights": "distance",
    },
    {
        "algorithm": "ball_tree",
        "leaf_size": 15,
        "n_neighbors": 33,
        "p": 1,
        "weights": "distance",
    },
]


search_grid_options = {
    "n_neighbors": list(range(1, 23, 2)) + list(range(23, 32)) + list(range(33, 38, 2)),
    "p": range(1, 5),
    "weights": ["distance", "uniform"],
    "algorithm": ["auto", "ball_tree", "kd_tree"],
    "leaf_size": list(range(1, 10)) + list(range(10, 51, 5)),
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
