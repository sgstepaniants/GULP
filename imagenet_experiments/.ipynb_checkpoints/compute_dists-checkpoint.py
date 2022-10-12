import sys
import os
import math
import random
from itertools import product
import numpy
import torch
import torchvision
import torchvision.transforms as transforms

sys.path.append(os.path.abspath("../sim_metric/dists"))
from scoring import *

lmbda_range = np.power(10.0, range(-15, 1))
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

pretrained = False
subset = "train"
mode = "eval"
n = 3000

untrained_reps = []
pretrained_reps = []
reps_folder = f"reps/{subset}/{n}_{mode}"
filenames = os.listdir(reps_folder)
for filename in filenames:
    if "pretrained" in filename:
        pretrained_reps.append(filename[:-4])
    elif "untrained" in filename:
        untrained_reps.append(filename[:-4])
untrained_reps = np.sort(untrained_reps)
pretrained_reps = np.sort(pretrained_reps)

model_names = untrained_reps
folder = f"distances/{subset}/untrained"
if pretrained:
    model_names = pretrained_reps
    folder = f"distances/{subset}/pretrained"
total_models = len(model_names)

dist_pairs_saved = np.zeros((total_models, total_models), dtype=bool)
if os.path.exists(f"{folder}/stats.npz"):
    dist_pairs_saved = np.load(f"{folder}/stats.npz")["dist_pairs_saved"]
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
    
    rep1 = np.load(f"{reps_folder}/{name1}.npy")
    # center and normalize
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep1 = np.sqrt(n) * rep1 / np.linalg.norm(rep1)

    rep2 = np.load(f"{reps_folder}/{name2}.npy")
    # center and normalize
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
    rep2 = np.sqrt(n) * rep2 / np.linalg.norm(rep2)
    
    all_dists = evaluate_distances(rep1, rep2)
    for dist_name in all_dists:
        dist = all_dists[dist_name]
        print(f'{dist_name}: {dist}', flush=True)
    print(flush=True)
