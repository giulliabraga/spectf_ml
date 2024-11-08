from scipy.stats import randint
from classifiers_comparison import ClassifiersComparison
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('./data/preprocessed/SPECTF_preprocessed.csv')

# dataset.sample(frac=0.1) # Uncomment this if you want to test the cross-validation with a dataset small subsample

# Get the features and target variable
X = dataset.drop(columns='target')
y = dataset['target']

# Set the model and parameters dictionary you want to use
model = KNeighborsClassifier()
param_dict = {
    'n_neighbors': randint(3,10),
    'weights':['uniform', 'distance'],
    'algorithm':['auto', 'ball_tree','kd_tree','brute'],
    'metric' : ['euclidean', 'manhattan', 'chebyshev'],
}

comparison = ClassifiersComparison(X,y,model)

metrics = comparison.cross_validation(finetune=False)

metrics.to_csv('./results/metrics_knn_with_tuning.csv')

