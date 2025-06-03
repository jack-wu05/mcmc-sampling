import numpy as np
import numpy.linalg as linalg
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.stats import multivariate_normal
warnings.filterwarnings("ignore")

np.random.seed(1)

global_nodes = []

## A variable in the dataset
class Node:
    def __init__(self, label, data):
        self.label = label
        self.data = data
        self.edges = []


## Fit a Gaussian to each variable
def fit_gaussian(data):
    mean = np.mean(data, axis=0)
    X = data - mean
    
    return mean, (X.T @ X) / (X.shape[0]-1)


## Compute mutual information weight
def mutual_info(node1, node2):
    mean_XX, Sigma_XX = fit_gaussian(node1.data)
    mean_YY, Sigma_YY = fit_gaussian(node2.data)

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
    global global_nodes
    
    nodes = [Node(str(i), np.stack(samples.iloc[:,i])) for i in range(samples.shape[1])]
    global_nodes = nodes.copy()
    
    G = nx.Graph()
    
    n = len(nodes)
    for i in range(n):
        for j in range(i+1,n):
            node1 = nodes[i]
            node2 = nodes[j]
            
            G.add_edge(node1.label, node2.label, weight= -mutual_info(node1,node2))
    
    edges = list(G.edges(data=True))
    
    print()
    print("Num all edges:", len(edges))
    print("All edges:", edges)
    print()
    
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    edges = list(mst.edges(data=True))
    
    print("Num selected edges:", len(edges))
    print("Selected edges:", edges)
    print()
    
    return mst


## Generate 2D data vectors from an already Gaussian and already tree-like dependency structure:
##  X0 --> X2 --> X3
##   |
##   V
##  X1
## The dependency is in the mean parameter; the covariances are arbitrarily fixed
def generate_data(num_samples):
    data = []
    
    for i in range(num_samples):
        sample = []
        
        mean0 = [0,0]
        cov0 = [[1,0],[0,1]]
        sample0 = np.random.multivariate_normal(mean0,cov0,size=1)[0]
        sample.append(sample0)
        
        cov1 = [[2,1],[1,1]]
        sample1 = np.random.multivariate_normal(sample0,cov1,size=1)[0]
        sample.append(sample1)
    
        cov2 = [[10,6],[6,9]]
        sample2 = np.random.multivariate_normal(sample0,cov2,size=1)[0]
        sample.append(sample2)
    
        cov3 = [[5,8],[8,20]]
        sample3 = np.random.multivariate_normal(sample2,cov3,size=1)[0]
        sample.append(sample3)
        
        data.append(sample)
    
    return data


## Assign edge directions pointing away from the arbitrarily fixed root X0
def directed_graph(tree):
    root = '0'
    
    x = nx.DiGraph()
    for parent, child in nx.bfs_edges(tree, root):
       x.add_edge(parent,child)
    
    return x


## Construct conditional pdf of two multivariate Gaussians: f(node1 = val1 | node2 = val2)
def unnormalized_cond_gaussian(node1, node2, val1, val2):
    mean_XX, Sigma_XX = fit_gaussian(node1.data)    
    mean_YY, Sigma_YY = fit_gaussian(node2.data)
    
    X = node1.data - np.mean(node1.data,axis=0)
    Y = node2.data - np.mean(node2.data,axis=0)
    
    Sigma_XY = (X.T @ Y) / (X.shape[0]-1)
    Sigma_YX = Sigma_XY.T
    
    new_mu = mean_XX + Sigma_XY @ linalg.inv(Sigma_YY) @ (val2 - mean_YY)
    new_Sigma = Sigma_XX - Sigma_XY @ linalg.inv(Sigma_YY) @ Sigma_YX
    
    return multivariate_normal.pdf(val1, mean=new_mu, cov=new_Sigma)

    
    
## Evaluate approximate Chow-Liu joint density at input_x
def tree_pdf(directed_tree, input_x):
    global global_nodes
    
    pdf = 1
    temp1, temp2 = fit_gaussian(global_nodes[0].data)
    pdf *= multivariate_normal.pdf(input_x[0], temp1, temp2)
    
    for edge in list(directed_tree.edges()):
        parent = int(edge[0])
        child = int(edge[1])
        
        pdf *= unnormalized_cond_gaussian(global_nodes[child], global_nodes[parent], input_x[child], input_x[parent])
    
    return pdf


#### Toy example 1
# num_samples = 2000
# data = generate_data(num_samples)
# df = pd.DataFrame(data, columns=['X0', 'X1', 'X2', 'X3'])
# tree = tree_decomposition(df)
# directed_tree = directed_graph(tree)

# input_x = np.array([[0,0],[0,0],[0,0],[0,0]])
# print(tree_pdf(directed_tree, input_x))


