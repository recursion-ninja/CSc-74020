from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification, which_set
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy


#########################################
###   Model Specific Definitions:
#########################################


classifier = RandomForestClassifier()

designation = "Random Forest"

hyperparameter_values = [
    {
        "bootstrap": True,
        "class_weight": "balanced",
        "criterion": "entropy",
        "max_features": "auto",
        "n_estimators": 150,
        "oob_score": False,
        "random_state": STATIC_SEED,
    },
    #    None,
    {
        "bootstrap": True,
        "class_weight": None,
        "criterion": "gini",
        "max_features": "auto",
        "n_estimators": 150,
        "oob_score": False,
        "random_state": 4178261698,
    },
]

search_grid_options = {
    "bootstrap": [True],  # [False, True],
    "class_weight": [None],  # , "balanced", "balanced_subsample"],
    "criterion": ["gini"],  # ["gini", "entropy"],
    "max_features": ["auto"],  # ["auto", "sqrt", "log2"],
    "n_estimators": [10 * i for i in range(10, 26)],
    "oob_score": [False],  # [False, True],
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


def with_tiers(tiers):
    params = deepcopy(evaluation_parameters)
    params["dataset_params"]["class_names"] = tiers
    params["best_hyperparameters"] = hyperparameter_values[which_set(tiers)]
    return params


def main():
    model_evaluation(**evaluation_parameters)


#    clf = best_classifier()
#    print("Features :", clf.n_features_in_)
#    print("Inputs   :", clf.feature_names_in_)
#    print("Outputs  :", clf.n_outputs_)
#    print("Important:", clf.feature_importances_)


if __name__ == "__main__":
    main()
