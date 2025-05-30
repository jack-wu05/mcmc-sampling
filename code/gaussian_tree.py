import numpy as np
import numpy.linalg as linalg
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

global_nodes = NULL

## A variable in the dataset
class Node:
    def __init__(self, label, data):
        self.label = label
        self.data = data
        self.edges = []

## Fit a Gaussian to each variable
def fit_gaussian(data):
    mean = np.mean(data, axis=0)
    X = data - np.mean(data, axis=0)
    
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
    
    return mst, list(mst.vertices(data=True))


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

def directed_graph(tree, kept_nodes):
    root = kept_nodes[0]
    
    x = nx.DiGraph()
    for parent, child in nx.bfs_edges(tree, root):
       x.add_edge(parent,child)
    
    return x
    
    
    
def unnormalized_gaussian(mu, Sigma):
    return lambda x: -0.5 * (x-mu).T @ linalg.inv(Sigma) @ (x-mu)

def unnormalized_cond_gaussian(node1, node2):
    mean_XX, Sigma_XX = fit_gaussian(node1.data)    
    mean_YY, Sigma_YY = fit_gaussian(node2.data)
    
    X = node1.data - np.mean(node1.data,axis=0)
    Y = node2.data - np.mean(node2.data,axis=0)
    
    Sigma_XY = (X.T @ Y) / (X.shape[0]-1)
    Sigma_YX = Sigma_XY.T
    
    new_mu = lambda y: mean_XX + Sigma_XY @ linalg.inv(Sigma_YY) @ (y - mean_YY)
    new_Sigma = Sigma_XX - Sigma_XY @ linalg.inv(Sigma_YY) @ Sigma_YX
    
    return lambda x, y: -0.5 * (x - new_mu(y)).T @ linalg.inv(new_Sigma) @ (x - new_mu(y))
    
    
    
def tree_pdf(directed_tree, kept_nodes):
    root = int(kept_nodes[0])
    
    pdf = lambda x: 1
    pdf *= global_nodes[]
    
    
    
    

data = generate_data(50)
df = pd.DataFrame(data, columns=['X0', 'X1', 'X2', 'X3'])
tree, kept_nodes = tree_decomposition(df)
directed_tree = directed_graph(tree, kept_nodes)
pdf = tree_pdf(directed_tree, kept_nodes)
print(pdf)











