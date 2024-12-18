from classifier_cv import ClassifierCrossValidation
import pandas as pd
from model_selector import get_model_and_params, get_majority_voting_classifier

'''
Configure the experimental settings
'''

finetune = True

model_name = "KDE"

dataset = pd.read_csv('./data/preprocessed/SPECTF_preprocessed.csv')

# dataset.sample(frac=0.1) # Uncomment this if you want to test the cross-validation with a dataset small subsample

# Get the features and target variable
X = dataset.drop(columns='target')
y = dataset['target']


'''
Run the experiment
'''

if model_name == 'Voting': # approach is different if you want the ensemble method
    model, param_dict = get_majority_voting_classifier()
else:
    model, param_dict = get_model_and_params(model_name)


cv = ClassifierCrossValidation(X, y, model, param_dict)

metrics = cv.cross_validation(finetune=finetune)

metrics.to_csv(f'./results/metrics_{model_name}_tuning_{finetune}.csv', index=False)