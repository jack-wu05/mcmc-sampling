import numpy as np
import numpy.linalg as linalg
import networkx as nx
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

## A variable in the dataset
class Node:
    def __init__(self, label, data):
        self.label = label
        self.data = data
        self.edges = []

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
    nodes = [Node("X" + str(i), np.stack(samples.iloc[:,i])) for i in range(samples.shape[1])]
    
    G = nx.Graph()
    
    n = len(nodes)
    for i in range(n):
        for j in range(i+1,n):
            node1 = nodes[i]
            node2 = nodes[j]
            
            G.add_edge(node1.label, node2.label, weight= -mutual_info(node1,node2))
    
    edges = list(G.edges(data=True))
    
    print()
    print("All edges:", edges)
    print()
    print("Num all edges:", len(edges))
    print()
    
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    edges = list(mst.edges(data=True))
    
    print("Selected edges:", edges)
    print()
    print("Num selected edges:", len(edges))
    print()
            

def generate_data(num_samples):
    data = []
    
    for i in range(num_samples):
        sample = []
        
        mean0 = [0,0]
        cov0 = [[1,2],[3,4]]
        sample0 = np.random.multivariate_normal(mean0,cov0,size=1)[0]
        sample.append(sample0)
        
        cov1 = [[5,6],[7,8]]
        sample1 = np.random.multivariate_normal(sample0,cov1,size=1)[0]
        sample.append(sample1)
    
        cov2 = [[9,10],[11,12]]
        sample2 = np.random.multivariate_normal(sample0,cov2,size=1)[0]
        sample.append(sample2)
    
        cov3 = [[13,14],[15,16]]
        sample3 = np.random.multivariate_normal(sample2,cov3,size=1)[0]
        sample.append(sample3)
        
        data.append(sample)
    
    return data
        

data = generate_data(50)
df = pd.DataFrame(data, columns=['X0', 'X1', 'X2', 'X3'])
tree_decomposition(df)
