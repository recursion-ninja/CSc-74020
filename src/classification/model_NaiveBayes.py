from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification
from sklearn.naive_bayes import MultinomialNB
from copy import deepcopy


#########################################
###   Model Specific Definitions:
#########################################


classifier = MultinomialNB()

designation = "Multinomial Naïve Bayes"

hyperparameter_values = {"alpha": 0.01, "fit_prior": False}
search_grid_options = {
    "alpha": [10 ** (i - 4) for i in range(0, 9)],
    "fit_prior": [False, True],
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
