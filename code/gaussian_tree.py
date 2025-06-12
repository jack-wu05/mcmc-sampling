import numpy as np
import numpy.linalg as linalg
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.stats import multivariate_normal
from scipy.stats import norm

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
    data = np.asarray(data)
    
    mean = np.mean(data)
    std = np.std(data)
    
    # mean = np.mean(data, axis=0)
    # X = data - mean
    # cov = (X.T @ X) / (X.shape[0]-1)
    
    return mean, std


## Compute mutual information weight
def mutual_info(node1, node2):
    X = np.asarray(node1.data)
    Y = np.asarray(node2.data)
    
    rho = np.corrcoef(X, Y)[0, 1]
    
    return -0.5 * np.log(1 - rho**2)
    
    # mean_XX, Sigma_XX = fit_gaussian(node1.data)
    # mean_YY, Sigma_YY = fit_gaussian(node2.data)

    # X = node1.data - np.mean(node1.data,axis=0)
    # Y = node2.data - np.mean(node2.data,axis=0)
    
    # Sigma_XY = (X.T @ Y) / (X.shape[0]-1)
    # Sigma_YX = Sigma_XY.T
    
    # Sigma = np.block([
    #     [Sigma_XX, Sigma_XY],
    #     [Sigma_YX, Sigma_YY]
    # ])
    
    # return 0.5 * np.log((linalg.det(Sigma_XX) * linalg.det(Sigma_YY))/linalg.det(Sigma))
    

## Chow-Liu algorithm
def tree_decomposition(samples):
    global global_nodes
    
    samples = np.asarray(samples)
    nodes = [Node(str(i), np.array(list(samples[:,i]))) for i in range(samples.shape[1])]
    global_nodes = nodes.copy()
    
    G = nx.Graph()
    
    n = len(nodes)
    for i in range(n):
        for j in range(i+1,n):
            node1 = nodes[i]
            node2 = nodes[j]
            
            G.add_edge(node1.label, node2.label, weight= -mutual_info(node1,node2))
    
    edges = list(G.edges(data=True))
    
    # print()
    # print("Num all edges:", len(edges))
    # print("All edges:", edges)
    # print()
    
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    edges = list(mst.edges(data=True))
    
    # print("Num selected edges:", len(edges))
    # print("Selected edges:", edges)
    # print()
    
    return mst


## Generate 2D data vectors from an already Gaussian and already tree-like dependency structure:
##  X0 --> X2 --> X3
##   |
##   V
##  X1
## The dependency is in the mean parameter; the covariances are arbitrarily fixed
def generate_data1(num_samples):
    data = []
    
    for i in range(num_samples):
        sample0 = np.random.normal(0,1)
        sample1 = np.random.normal(sample0,2)
        sample2 = np.random.normal(sample0,5)
        sample3 = np.random.normal(sample2,3)
        
        data.append([sample0, sample1, sample2, sample3])
    
    return data


## Assign edge directions pointing away from the arbitrarily fixed root X0
def directed_graph(tree):
    root = '0'
    
    x = nx.DiGraph()
    for parent, child in nx.bfs_edges(tree, root):
       x.add_edge(parent,child)
    
    return x


## Construct conditional pdf of two Normals: f(node1 = val1 | node2 = val2)
def log_cond_gaussian(node1, node2, val1, val2):
    X = np.asarray(node1.data)
    Y = np.asarray(node2.data)
    
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    var_X = np.var(X)
    var_Y = np.var(Y)
    
    rho = np.corrcoef(X,Y)[0,1]
    
    new_mu = mean_X + rho * (np.sqrt(var_X) / np.sqrt(var_Y)) * (val2 - mean_Y)
    new_sigma = np.sqrt((1-rho**2) * var_X)
    
    return norm.logpdf(val1, new_mu, new_sigma)
    
    
    # mean_XX, Sigma_XX = fit_gaussian(node1.data)    
    # mean_YY, Sigma_YY = fit_gaussian(node2.data)
    
    # X = node1.data - np.mean(node1.data,axis=0)
    # Y = node2.data - np.mean(node2.data,axis=0)
    
    # Sigma_XY = (X.T @ Y) / (X.shape[0]-1)
    # Sigma_YX = Sigma_XY.T

    # new_mu = mean_XX + Sigma_XY @ linalg.inv(Sigma_YY) @ (val2 - mean_YY)
    # new_Sigma = Sigma_XX - Sigma_XY @ linalg.inv(Sigma_YY) @ Sigma_YX
    
    # return multivariate_normal.pdf(val1, mean=new_mu, cov=new_Sigma)


## Evaluate approximate log Chow-Liu joint density at input_x
def tree_logpdf(directed_tree, input_x):
    global global_nodes
    
    pdf = 0
    temp1, temp2 = fit_gaussian(global_nodes[0].data)
    # pdf *= multivariate_normal.pdf(input_x[0], temp1, temp2)
    pdf += norm.logpdf(input_x[0], temp1, temp2)
    
    for edge in list(directed_tree.edges()):
        parent = int(edge[0])
        child = int(edge[1])
        
        pdf += log_cond_gaussian(global_nodes[child], global_nodes[parent], input_x[child], input_x[parent])
    
    return pdf

