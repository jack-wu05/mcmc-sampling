import numpy as np
import numpy.linalg as linalg

## A variable in the dataset
class Node:
    def __init__(self, label, data):
        self.label = label
        self.data = data
        self.edges = []

## A relationship between two variables weighted by mutual information
class Edge:
    def __init__(self, node_to, node_from, weight):
        self.node_to = node_to
        self.node_from = node_from
        self.weight = weight
        node_to.edges.append(self)
        node_from.edges.append(self)
        
## Find max spanning tree
def kruskals(): pass

## Fit a Gaussian to each variable
def fit_gaussian(data):
    X = data - np.mean(data, axis=0)
    
    return (X.T @ X) / (X.shape[0]-1)

## Compute mutual information weight
def mutual_info(node1, node2):
    Sigma_XX = fit_gaussian(node1.data)
    Sigma_YY = fit_gaussian(node2.data)

    X = node1.data - np.mean(node1.data,axis=0)
    Y = node2.data - np.mean(node2.data,axis=0)
    
    Sigma_XY = (X.T @ Y) / (X.shape[0]-1)
    Sigma_YX = Sigma_XY.T
    
    Sigma = np.block([
        [Sigma_XX, Sigma_XY],
        [Sigma_YX, Sigma_YY]
    ])
    
    return 0.5 * np.log((linalg.det(Sigma_XX) * linalg.det(Sigma_YY))/linalg.det(Sigma))
    
## Chow-Liu algorithm
def tree_decomposition(samples):
    list_of_nodes = [Node(i, samples.iloc[:,i]) for i in range(samples.shape[1])]
    
    n = len(list_of_nodes)
    for i in range(n):
        for j in range(i+1,n):
            
    
    
    
