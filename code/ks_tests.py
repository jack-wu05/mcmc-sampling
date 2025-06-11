import numpy as np
from mks_test import mkstest
from scipy.stats import ks_2samp

import gaussian_tree
import var_pt



### Kolmogorov-Smirnov test for kernel pi-invariance using N(0,1)
# log_gamma = lambda x: -x**2 / 2
# num_samples = 3000
# iid_samples = np.random.normal(size=num_samples)

# kernel_samples = np.zeros(num_samples)
# for i in range(num_samples):
#     kernel_samples[i] = var_pt.RWMH_exploration_kernel(log_gamma, iid_samples[i], 4000)[-1]

# ks_result = ks_2samp(iid_samples, kernel_samples)
# print("Kernel test p-value:", ks_result.pvalue)



### Multivar KS test for correctness of Chow-Liu
d = 4
num_samples = 10
iid_data = gaussian_tree.generate_data1(num_samples)
iid_data = np.asarray(iid_data)

cl_tree = gaussian_tree.directed_graph(
                gaussian_tree.tree_decomposition(iid_data)
                )    
tree_pdf = lambda x: gaussian_tree.tree_pdf(cl_tree, x)

kernel_samples = np.zeros((num_samples,d))
for i in range(num_samples):
    print(i)
    kernel_samples[i] = var_pt.RWMH_exploration_kernel(tree_pdf, iid_data[i], 10)[-1]

print(mkstest(iid_data, kernel_samples, alpha=0.05))
