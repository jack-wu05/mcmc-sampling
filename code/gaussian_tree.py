import numpy as np

## A variable in the dataset
class Node:
    def __init__(self, label):
        self.label = label
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

## Compute mutual information weight
def mutual_info(): pass

## Chow-Liu algorithm
def tree_decomposition(): pass
    
        
    