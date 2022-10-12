import sys
import os
import math
import random
from itertools import product
import numpy
import torch
import torchvision
import torchvision.transforms as transforms

sys.path.append(os.path.abspath("../"))
from distance_functions import *

lmbda_range = np.power(10.0, range(-20, 5))
lmbda_range = np.concatenate((lmbda_range, [0]))

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


batch_num = int(sys.argv[1])
total_batches = int(sys.argv[2])

n = 10000


layers = np.array([1, 2, 3, 4, 5])
widths = np.array([200, 400, 600, 800, 1000])
trials = np.array([0, 1, 2, 3, 4])

total_models = len(layers) * len(widths) * len(trials)

model_names = []
for layer in layers:
    for width in widths:
        for trial in trials:
            model_names.append(f'fc_cifar_{layer}_{width}_{trial}')

dist_pairs_saved = np.zeros((total_models, total_models), dtype=bool)
if os.path.exists(f"distances/{n}/stats.npz"):
    dist_pairs_saved = np.load(f"distances/{n}/stats.npz")["dist_pairs_saved"]
    print(f"{np.sum(dist_pairs_saved)} existing pairs", flush=True)

model_pairs = []
for i in range(total_models):
    for j in range(i+1, total_models):
        if not dist_pairs_saved[i, j]:
            model_pairs.append((model_names[i], model_names[j]))
random.Random(4).shuffle(model_pairs)

total_pairs = len(model_pairs)
batch_size = math.ceil(total_pairs / total_batches)
print(batch_size, flush=True)

batch_pairs = model_pairs[(batch_num-1)*batch_size:batch_num*batch_size]
print(len(batch_pairs), flush=True)

for pair in batch_pairs:
    name1, name2 = pair
    print(f'Computing {name1}, {name2}', flush=True)
    
    rep1 = np.load(f"reps/{n}/{name1}_rep.npy")
    print(rep1.shape)
    # center and normalize
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep1 = math.sqrt(n) * rep1 / np.linalg.norm(rep1)
    
    rep2 = np.load(f"reps/{n}/{name2}_rep.npy")
    print(rep2.shape)
    # center and normalize
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
    rep2 = math.sqrt(n) * rep2 / np.linalg.norm(rep2)
    
    all_dists = evaluate_distances(rep1, rep2)
    for dist_name in all_dists:
        dist = all_dists[dist_name]
        print(f'{dist_name}: {dist}', flush=True)
    print(flush=True)

