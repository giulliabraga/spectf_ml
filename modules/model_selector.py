import json
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from parzen_classifier import ParzenClassifier  
from density_kernel_classifier import KernelDensityClassifier

'''
This module allows for an easy selection of a model with its respective parameter grid for a randomized search cross-validation.

To make your own setup, go to './utils/parameters_dict.json' and edit it according to the models and parameters you want to assess.

You will also need to change the MODEL_MAP in this module.
'''

# Loading the file which contains each model's identifier and its parameters dictionary for the randomized search
with open('./utils/parameter_dictionaries.json', 'r') as file:
    parameter_dicts = json.load(file)

# Mapping of model names to their corresponding classes
MODEL_MAP = {
    "KNN": KNeighborsClassifier,
    "LR": LogisticRegression,
    "QDA": QuadraticDiscriminantAnalysis,
    "Parzen": ParzenClassifier,
    "KDE": KernelDensityClassifier
}

# Mapping parameter dictionaries
PARAM_DICT_MAP = {
    param["model"]: param["params_dict"] for param in parameter_dicts
}

def get_model_and_params(model_name):
    """
    Retrieve a model instance and its parameter dictionary for RandomizedSearchCV.

    Parameters:
    - model_name (str): The name of the model, as a string. Must be one of the keys in MODEL_MAP.

    Returns:
    - model (estimator): An instance of the requested model.
    - param_dict (dict): The parameter dictionary for RandomizedSearchCV.
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model '{model_name}' is not recognized. Available models are: {list(MODEL_MAP.keys())}")

    if model_name not in PARAM_DICT_MAP:
        raise ValueError(f"Model '{model_name}' does not have a parameter dictionary. Available models are: {list(PARAM_DICT_MAP.keys())}")

    model = MODEL_MAP[model_name]()
    param_dict = PARAM_DICT_MAP[model_name]
    
    return model, param_dict

def get_majority_voting_classifier():
    """
    Build and return a majority voting classifier and its associated parameter grid for RandomizedSearchCV.
    
    Returns:
    - voting_clf (VotingClassifier): The majority voting classifier.
    - param_dict (dict): The dictionary containing the parameters for RandomizedSearchCV.
    """
    knn = KNeighborsClassifier()
    lr = LogisticRegression() 
    qda = QuadraticDiscriminantAnalysis()
    parzen = ParzenClassifier()
    
    voting_clf = VotingClassifier(estimators=[
        ('KNN', knn),
        ('LR', lr),
        ('QDA', qda),
        ('Parzen', parzen)
    ], voting='hard')  
    
    # This is the parameter dictionary for RandomizedSearchCV
    param_dict = {
        'KNN__' + key: value for key, value in PARAM_DICT_MAP["KNN"].items()
    }
    param_dict.update({
        'LR__' + key: value for key, value in PARAM_DICT_MAP["LR"].items()
    })
    param_dict.update({
        'QDA__' + key: value for key, value in PARAM_DICT_MAP["QDA"].items()
    })
    param_dict.update({
        'Parzen__' + key: value for key, value in PARAM_DICT_MAP["Parzen"].items()
    })
    
    return voting_clf, param_dict