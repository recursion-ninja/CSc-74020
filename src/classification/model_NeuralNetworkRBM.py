from classifier_specification import STATIC_SEED, model_evaluation
from featureset_specification import default_feature_specification, which_set
from copy import deepcopy
from numpy import linspace
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


#########################################
###   Model Specific Definitions:
#########################################

bernouliRMB = BernoulliRBM()
inputScaler = MinMaxScaler(copy=False)
logisticReg = LogisticRegression(solver="newton-cg", tol=1)

classifier = Pipeline(
    steps=[
        ("inputScaler", inputScaler),
        ("bernouliRMB", bernouliRMB),
        ("logisticReg", logisticReg),
    ]
)

designation = "Bernoulli Restricted Boltzmann Machine"

hyperparameter_values = [
    {
    "bernouliRMB__batch_size": 16,
    "bernouliRMB__learning_rate": 1.0,
    "bernouliRMB__n_components": 16,
    "bernouliRMB__n_iter": 14,
    "bernouliRMB__random_state": 4178261698,
    "logisticReg__C": 0.05,
    "logisticReg__max_iter": 10000,
    "logisticReg__penalty": "l2",
    "logisticReg__random_state": 4178261698,
    "logisticReg__solver": "lbfgs",
    "logisticReg__tol": 0.1,
    },
    None
]

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


def with_tiers(tiers):
    params = deepcopy(evaluation_parameters)
    params["dataset_params"]["class_names"] = tiers
    params["best_hyperparameters"] = hyperparameter_values[which_set(tiers)] 
    return params


if __name__ == "__main__":
    main()
