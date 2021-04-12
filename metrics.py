import numpy as np
import pandas as pd

def calculate_distance(data_matrix, method):
    if method == 'pearson':
        distance_matrix = pearson_based(data_matrix)
    
    elif method == 'theil_index':
        distance_matrix = theil_based(data_matrix)
    
    elif method == 'atkinson_index':
        distance_matrix = theil_based(data_matrix, atkinson=True)

    return distance_matrix

def pearson_based(data_matrix):
    corr_matrix = data_matrix.corr()
    distance_matrix = np.sqrt(2*(1-corr_matrix))
    
    return distance_matrix

def theil_based(data_matrix, atkinson = False):
    # Theil index based Manhattan Distance
    data_mean = data_matrix.mean()
    mean_fraq = data_matrix / data_mean
    log_mean_fraq = mean_fraq.applymap(lambda x: np.log(x) if x!=0 else 0)

    Th = (mean_fraq * log_mean_fraq).sum() / len(mean_fraq)
    index_metric = pd.DataFrame(Th).T
    if atkinson:
        index_metric = 1-np.exp(-index_metric) # Atkinson index
    distance_matrix = manhattan_distance(index_metric)
    return distance_matrix

def manhattan_distance(pd_data):
    distance = lambda col1, col2: np.abs(col1 - col2).sum() / len(col1)
    result = pd_data.apply(lambda col1: pd_data.apply(lambda col2: distance(col1, col2)))
    return result


if __name__ == '__main__':

    data = pd.DataFrame({'A': [1,2,3,4,5,6,7,8],
                         'B': [1.1,2.1,3.5,4.12,1.1,5.02,7.12,8.001],
                         'C': [0.1,.52,123,77.4,45.5,69,7.12,108]})
    
    pearson_based_result = calculate_distance(data, method='pearson')
    theil_based_result = calculate_distance(data, method='theil_index')
    atkinson_based_result = calculate_distance(data, method='atkinson_index')
    
