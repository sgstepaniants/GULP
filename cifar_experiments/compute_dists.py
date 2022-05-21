import sys
import os
import pickle
import math
from itertools import product
import numpy
import torch
import torchvision
import torchvision.transforms as transforms

sys.path.append(os.path.abspath("../"))
from distance_functions import *

lmbda_range = np.power(10.0, range(-8, 3))
lmbda_range[0] = 0
def evaluate_distances(A, B):
    evals_a, evecs_a = np.linalg.eigh(A @ A.T)
    evals_b, evecs_b = np.linalg.eigh(B @ B.T)
    u, s, vh, transformed_a, transformed_b = cca_decomp(A, B, evals_a, evecs_a, evals_b, evecs_b)
    
    all_dists = {}
    all_dists['mean_sq_cca_e2e'] = mean_sq_cca_corr(s)
    all_dists['mean_cca_e2e'] = mean_cca_corr(s)
    all_dists['pwcca_dist_e2e'] = pwcca_dist(A, s, transformed_a)
    
    all_dists['lin_cka_dist'] = lin_cka_dist(A, B)
    all_dists['lin_cka_prime_dist'] = lin_cka_prime_dist(A, B)
    
    all_dists['procrustes'] = procrustes(A, B)
    
    for lmbda in lmbda_range:
        all_dists[f'predictor_dist_{lmbda}'] = predictor_dist(A, B, evals_a, evecs_a, evals_b, evecs_b, lmbda=lmbda)
    
    return all_dists

seeds = np.arange(1, 17)
epochs = np.arange(50)
#layers = np.array(["layer1.0", "layer1.1", "layer2.0", "layer2.1", "layer3.0", "layer3.1", "layer4.0", "layer4.1", "last"])
layers = np.array(["last"])
num_seeds = len(seeds)
num_epochs = len(epochs)
num_layers = len(layers)

batch_num = int(sys.argv[1])
total_batches = int(sys.argv[2])
prod_inds = []
for i1 in range(num_seeds):
    for j1 in range(num_epochs):
        for k1 in range(num_layers):
            for i2 in range(num_seeds):
                for j2 in range(num_epochs):
                    for k2 in range(num_layers):
                        if i1 > i2:
                            continue
                        if i1 == i2 and j1 > j2:
                            continue
                        if i1 == i2 and j1 == j2 and k1 >= k2:
                            continue
                        prod_inds.append((i1, j1, k1, i2, j2, k2))
total_pairs = len(prod_inds)
batch_size = math.ceil(total_pairs / total_batches)
print(batch_size)

batch_inds = prod_inds[(batch_num-1)*batch_size:batch_num*batch_size]
print(len(batch_inds))
for pair in batch_inds:
    i1, j1, k1, i2, j2, k2 = pair
    seed1 = seeds[i1]
    epoch1 = epochs[j1]
    layer1 = layers[k1]
    seed2 = seeds[i2]
    epoch2 = epochs[j2]
    layer2 = layers[k2]
    print(f'Computing {i1}, {j1}, {k1}, {i2}, {j2}, {k2}', flush=True)
    
    f1 = open(f'cifar{seed1}/test/epoch{epoch1}/latents.pkl', 'rb')
    out1 = pickle.load(f1)
    rep1 = out1[layer1]
    rep1 = rep1.reshape(rep1.shape[0], -1).T
    # center and normalize
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep1 = rep1 / np.linalg.norm(rep1)

    f2 = open(f'cifar{seed2}/test/epoch{epoch2}/latents.pkl', 'rb')
    out2 = pickle.load(f2)
    rep2 = out2[layer2]
    rep2 = rep2.reshape(rep2.shape[0], -1).T
    # center and normalize
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
    rep2 = rep2 / np.linalg.norm(rep2)
    
    all_dists = evaluate_distances(rep1, rep2)
    for dist_name in all_dists:
        dist = all_dists[dist_name]
        print(f'{dist_name}: {dist}', flush=True)
    print(flush=True)
