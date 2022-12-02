from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification
from numpy import linspace
from sklearn.neural_network import BernoulliRBM
from copy import deepcopy


#########################################
###   Model Specific Definitions:
#########################################


classifier = BernoulliRBM()

designation = "Bernoulli Restricted Boltzmann Machine"

hyperparameter_values = None

search_grid_options = {
    "n_components": [2 ** i for i in range(3, 10)],
    "learning_rate": [10 ** ((-1/2)*i) for i in range(0, 7)],
    "batch_size": list(linspace(2, 16, num=8).astype(int)),
    "n_iter": list(linspace(2, 16, num=8).astype(int)),
    "random_state": [STATIC_SEED]
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
