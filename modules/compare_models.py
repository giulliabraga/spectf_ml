import pandas as pd
from statistical_methods import StatisticalMethods

knn_results = pd.read_csv('./results/metrics_KNN_tuning_False.csv')
gnb_results = pd.read_csv('./results/metrics_GNB_tuning_False.csv')
lr_results = pd.read_csv('./results/metrics_LR_tuning_False.csv')

list_of_model_results = [knn_results, gnb_results, lr_results]
list_of_model_names = ['KNN', 'GNB', 'LR']
list_of_metrics_names = ['error_rate', 'test_accuracy', 'coverage', 'f1_score']

comp = StatisticalMethods(list_of_model_results, list_of_model_names, list_of_metrics_names)

friedman_results, metrics_with_difference = comp.friedman_test()

nemenyi_results = comp.nemenyi_test(metrics_with_difference)

print(f'Friedman results:\n {friedman_results}')
print(f'\n Nemenyi results: \n {nemenyi_results}')