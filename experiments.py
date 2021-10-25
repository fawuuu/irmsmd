##### Code for "Implicit Regularization in Matrix Sensing via Mirror Descent" #####
# Experiments for Figure 1 and Figure 2

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

from algorithms import gen_data, gd, md, cvx_opt, effrank

########################################
# Run experiments for Figure 1 and Figure 2
########################################
# Set parameters
# For Figure 2, set compl = True
np.random.seed(1)       # Random seed for reproducibility             
n = 50                  # Dimension of X*
r = 5                   # Rank of X*
compl = False           # Matrix sensing with Gaussian matrices or matrix completion
if compl:
    eta = 2000          # Step size for matrix completion
else:
    eta = 1             # Step size for matrix sensing with Gaussian matrices
iter = 5000             # Number of iterations

# Experiments with 3nr measurements
m = int(3 * n * r)
# Generate data
X_star, A, y  = gen_data(n, r, m, completion = compl)

# Compute ground truth and nuclear norm minimization
ground_truth = np.linalg.norm(X_star, ord = "nuc")
min_nuc, nuc_matrix = cvx_opt(A, y, eps = 1.e-5)

nuc_gt, nuc_nuc, nuc_gd, nuc_md, nuc_gd_sd, nuc_md_sd = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)
eff_gt, eff_nuc, eff_gd, eff_md, eff_gd_sd, eff_md_sd = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)
rec_gt, rec_nuc, rec_gd, rec_md, rec_gd_sd, rec_md_sd = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)

# Run gradient descent and mirror descent for varying initialization sizes
for i in range(10):
    res_gd = gd(A, y, alpha = 10**(-(i+1)), step = eta/4, iter = iter)[-1]
    res_md = md(A, y, alpha = 10**(-(i+1)), step = eta, iter = iter)[-1]
    
    # Save nuclear norms
    nuc_gt[i] = ground_truth
    nuc_nuc[i] = min_nuc
    nuc_gd[i] = np.linalg.norm(res_gd, ord = "nuc")
    nuc_md[i] = np.linalg.norm(res_md, ord = "nuc")

    # Save reconstruction errors
    rec_gt[i] = 0
    rec_nuc[i] = np.linalg.norm(nuc_matrix - X_star)
    rec_gd[i] = np.linalg.norm(res_gd - X_star)
    rec_md[i] = np.linalg.norm(res_md - X_star)

    # Save effective ranks
    eff_gt[i] = effrank(X_star)
    eff_nuc[i] = effrank(nuc_matrix)
    eff_gd[i] = effrank(res_gd)
    eff_md[i] = effrank(res_md)

# Experiments with nr measurements
m = int(n * r)
# Generate data
X_star, A, y  = gen_data(n, r, m, completion = compl)

# Compute ground truth and nuclear norm minimization
ground_truth = np.linalg.norm(X_star, ord = "nuc")
min_nuc, nuc_matrix = cvx_opt(A, y, eps = 1.e-5)

nuc_gt2, nuc_nuc2, nuc_gd2, nuc_md2, nuc_gd_sd2, nuc_md_sd2 = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)
eff_gt2, eff_nuc2, eff_gd2, eff_md2, eff_gd_sd2, eff_md_sd2 = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)
rec_gt2, rec_nuc2, rec_gd2, rec_md2, rec_gd_sd2, rec_md_sd2 = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)

# Run gradient descent and mirror descent for varying initialization sizes
for i in range(10):
    res_gd = gd(A, y, alpha = 10**(-(i+1)), step = eta/4, iter = iter)[-1]
    res_md = md(A, y, alpha = 10**(-(i+1)), step = eta, iter = iter)[-1]

    # Save nuclear norms
    nuc_gt2[i] = ground_truth
    nuc_nuc2[i] = min_nuc
    nuc_gd2[i] = np.linalg.norm(res_gd, ord = "nuc")
    nuc_md2[i] = np.linalg.norm(res_md, ord = "nuc") 

    # Save reconstruction errors
    rec_gt2[i] = 0
    rec_nuc2[i] = np.linalg.norm(nuc_matrix - X_star) / np.linalg.norm(X_star)
    rec_gd2[i] = np.linalg.norm(res_gd - X_star) / np.linalg.norm(X_star)
    rec_md2[i] = np.linalg.norm(res_md - X_star) / np.linalg.norm(X_star)

    # Save effective ranks
    eff_gt2[i] = effrank(X_star)
    eff_nuc2[i] = effrank(nuc_matrix)
    eff_gd2[i] = effrank(res_gd)
    eff_md2[i] = effrank(res_md)

########################################
# Plot Figure 1 and Figure 2
########################################
fig, axs = plt.subplots(2, 3, figsize=(15, 6))
fig.tight_layout(w_pad=4, h_pad=2)

def format_func(value, tick_number):
    N = int(value + 1)
    if N == 1:
        return r"0.1"
    return r"$0.1^{0}$".format(N)

# Plot nuclear norm for 3nr measurements
axs[0, 0].plot(range(10), nuc_gt, "--", color = "black", marker = "+", label='ground truth')
axs[0, 0].plot(range(10), nuc_nuc, ":", color = "brown", marker = "v", markersize = 4, label='nuclear min')
axs[0, 0].plot(range(10), nuc_gd, ":", color = "blue", marker = "x", label='gradient descent')
axs[0, 0].plot(range(10), nuc_md, "--", color = "red", marker = "o", markersize = 4, label='mirror descent')
axs[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
axs[0, 0].set_ylabel('nuclear norm', fontsize = 16)
axs[0, 0].set_xlabel('initialization size', fontsize = 16)
axs[0, 0].grid()
axs[0, 0].legend(prop={'size': 13})

# Plot effective rank for 3nr measurements
axs[0, 1].plot(range(10), eff_gt, "--", color = "black", marker = "+", label='ground truth')
axs[0, 1].plot(range(10), eff_nuc, ":", color = "brown", marker = "v", markersize = 4, label='nuclear min')
axs[0, 1].plot(range(10), eff_gd, ":", color = "blue", marker = "x", label='gradient descent')
axs[0, 1].plot(range(10), eff_md, "--", color = "red", marker = "o", markersize = 4, label='mirror descent')
axs[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
axs[0, 1].set_ylabel('effective rank', fontsize = 16)
axs[0, 1].set_xlabel('initialization size', fontsize = 16)
axs[0, 1].grid()
axs[0, 1].legend(prop={'size': 13})

# Plot reconstruction error for 3nr measurements
axs[0, 2].plot(range(10), rec_gt, "--", color = "black", marker = "+", label='ground truth')
axs[0, 2].plot(range(10), rec_nuc, ":", color = "brown", marker = "v", markersize = 4, label='nuclear min')
axs[0, 2].plot(range(10), rec_gd, ":", color = "blue", marker = "x", label='gradient descent')
axs[0, 2].plot(range(10), rec_md, "--", color = "red", marker = "o", markersize = 4, label='mirror descent')
axs[0, 2].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
axs[0, 2].set_ylabel('reconstruction error', fontsize = 16)
axs[0, 2].set_xlabel('initialization size', fontsize = 16)
axs[0, 2].grid()
axs[0, 2].legend(prop={'size': 13})

# Plot nuclear norm for nr measurements
axs[1, 0].plot(range(10), nuc_gt2, "--", color = "black", marker = "+", label='ground truth')
axs[1, 0].plot(range(10), nuc_nuc2, ":", color = "brown", marker = "v", markersize = 4, label='nuclear min')
axs[1, 0].plot(range(10), nuc_gd2, ":", color = "blue", marker = "x", label='gradient descent')
axs[1, 0].plot(range(10), nuc_md2, "--", color = "red", marker = "o", markersize = 4, label='mirror descent')
axs[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
axs[1, 0].set_ylabel('nuclear norm', fontsize = 16)
axs[1, 0].set_xlabel('initialization size', fontsize = 16)
axs[1, 0].grid()
axs[1, 0].legend(prop={'size': 13})

# Plot effective rank for nr measurements
axs[1, 1].plot(range(10), eff_gt2, "--", color = "black", marker = "+", label='ground truth')
axs[1, 1].plot(range(10), eff_nuc2, ":", color = "brown", marker = "v", markersize = 4, label='nuclear min')
axs[1, 1].plot(range(10), eff_gd2, ":", color = "blue", marker = "x", label='gradient descent')
axs[1, 1].plot(range(10), eff_md2, "--", color = "red", marker = "o", markersize = 4, label='mirror descent')
axs[1, 1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
axs[1, 1].set_ylabel('effective rank', fontsize = 16)
axs[1, 1].set_xlabel('initialization size', fontsize = 16)
axs[1, 1].grid()
axs[1, 1].legend(prop={'size': 13})

# Plot reconstruction error for nr measurements
axs[1, 2].plot(range(10), rec_gt2, "--", color = "black", marker = "+", label='ground truth')
axs[1, 2].plot(range(10), rec_nuc2, ":", color = "brown", marker = "v", markersize = 4, label='nuclear min')
axs[1, 2].plot(range(10), rec_gd2, ":", color = "blue", marker = "x", label='gradient descent')
axs[1, 2].plot(range(10), rec_md2, "--", color = "red", marker = "o", markersize = 4, label='mirror descent')
axs[1, 2].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
axs[1, 2].set_ylabel('reconstruction error', fontsize = 16)
axs[1, 2].set_xlabel('initialization size', fontsize = 16)
axs[1, 2].grid()
axs[1, 2].legend(prop={'size': 13})

fig.align_ylabels()

fig.savefig("figure1.pdf", bbox_inches='tight', transparent=True)