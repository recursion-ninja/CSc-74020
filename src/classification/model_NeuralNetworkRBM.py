from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification
from numpy import linspace
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from copy import deepcopy


#########################################
###   Model Specific Definitions:
#########################################

bernouliRMB = BernoulliRBM()
inputScaler = MinMaxScaler(copy=False)
logisticReg = LogisticRegression(solver="newton-cg", tol=1)

classifier = Pipeline(
    steps=[("inputScaler", inputScaler), ("bernouliRMB", bernouliRMB), ("logisticReg", logisticReg)]
)

designation = "Bernoulli Restricted Boltzmann Machine"

hyperparameter_values = None

search_grid_options = {
    "bernouliRMB__n_components": [2**i for i in range(3, 10)],
    "bernouliRMB__learning_rate": [10 ** ((-1 / 2) * i) for i in range(0, 7)],
    "bernouliRMB__batch_size": list(linspace(2, 16, num=8).astype(int)),
    "bernouliRMB__n_iter": list(linspace(2, 16, num=8).astype(int)),
    "bernouliRMB__random_state": [STATIC_SEED],
    "logisticReg__penalty": ["l2"],
    "logisticReg__solver": ["lbfgs"],
    "logisticReg__C": [0.05],
    "logisticReg__tol": [0.1],
    "logisticReg__max_iter": [10000],
    "logisticReg__random_state": [STATIC_SEED],
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


    clf = best_classifier()
    print("Layers  :", clf.n_layers_      )
    print("Outputs :", clf.n_outputs_     )
    print("Function:", clf.out_activation_)


if __name__ == "__main__":
    main()