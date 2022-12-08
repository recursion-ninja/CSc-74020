from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification, which_set
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy


#########################################
###   Model Specific Definitions:
#########################################

classifier = DecisionTreeClassifier()

designation = "Decision Tree"

hyperparameter_values = [
    {
    "criterion": "gini",
    "splitter": "best",
    "max_features": "log2",
    "random_state": STATIC_SEED,
    },
    None
]

search_grid_options = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_features": ["auto", "sqrt", "log2"],
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
