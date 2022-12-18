from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification, which_set
from sklearn.naive_bayes import MultinomialNB
from copy import deepcopy
from numpy import linspace


#########################################
###   Model Specific Definitions:
#########################################


classifier = MultinomialNB()

designation = "Multinomial Naïve Bayes"

hyperparameter_values = [
    {"alpha": 0.0001, "fit_prior": False},
    # None,
    {"alpha": 1, "fit_prior": True},
]

search_grid_options = {
    "alpha": list(
        linspace(0.001, 0.00005, 33)
    ),  # [10 ** (i - 4) for i in range(0, 9)],
    "fit_prior": [False],  # [False, True],
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
