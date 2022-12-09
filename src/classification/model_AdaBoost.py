from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification, which_set
from numpy import linspace
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from copy import deepcopy


#########################################
###   Model Specific Definitions:
#########################################


classifier = AdaBoostClassifier()

designation = "AdaBoost"

hyperparameter_values = [
    None, # 12 tiers (TERC)
    None  # 22 tiers (TER)
]
#  Tier size 4: {'algorithm': 'SAMME', 'base_estimator': DecisionTreeClassifier(max_depth=8), 'learning_rate': 1.8085714285714287, 'n_estimators': 988, 'random_state': 4178261698}

# Example to refine a search arought 10 ^ -2
# Values to consider 10 ^ i in [-3, -2.8, ..., -1.2, -1.0]
#
# [10**(-2 + 0.2 * (i-5)) for i in range (0, 11) ]
#
search_grid_options = {
    "base_estimator": [DecisionTreeClassifier(max_depth=i) for i in range(7, 10)],
    "learning_rate": list(linspace(1.62, 2.5, num=15)),
    "n_estimators": list(linspace(400, 2048, num=15).astype(int)),
    "algorithm": ['SAMME', 'SAMME.R'],
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
#    print("Layers  :", clf.n_layers_      )
#    print("Outputs :", clf.n_outputs_     )
#    print("Function:", clf.out_activation_)


if __name__ == "__main__":
    main()
