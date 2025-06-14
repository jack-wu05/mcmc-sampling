import numpy as np
from mks_test import mkstest
from scipy.stats import ks_2samp

import gaussian_tree
import var_pt



### Multivar KS test for correctness of Chow-Liu
## The null is that data comes from same distribution
d = 4
num_samples = 200
iid_data = gaussian_tree.generate_data1(num_samples)
iid_data = np.asarray(iid_data)

temp_tree, global_nodes = gaussian_tree.tree_decomposition(iid_data)
cl_tree = gaussian_tree.directed_graph(temp_tree)
tree_logpdf = lambda x: gaussian_tree.tree_logpdf(cl_tree, x, global_nodes)

kernel_samples = np.zeros((num_samples,d))
for i in range(num_samples):
    print(i)
    kernel_samples[i] = var_pt.RWMH_exploration_kernel(tree_logpdf, iid_data[i], 50)[-1]

print(mkstest(iid_data, kernel_samples, alpha=0.05))
## Output: False
## Can't reject the null