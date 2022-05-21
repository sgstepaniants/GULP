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

input_size = 784 # 28x28

# Import MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                        train=True,
                                        transform=transforms.ToTensor())

# Generate matrix of all train data images
full_train_data = []
for i in range(len(train_dataset)):
    full_train_data.append(train_dataset[i][0].reshape(-1, 28*28))
full_train_data = torch.vstack(full_train_data)

num_data = len(train_dataset)

width_subset = 100*np.arange(1, 11)
depth_subset = np.arange(1, 11)

batch_num = int(sys.argv[1])
total_batches = int(sys.argv[2])
model_names = []
modeldir = "models/widthdepth/"
filenames = os.listdir(modeldir)
for filename in filenames:
    name = filename.split(".")[0]
    splits = name.split('_')
    width = int(splits[0][5:])
    depth = int(splits[1][5:])
    if width in width_subset and depth in depth_subset:
        model_names.append(name)
model_names = np.sort(model_names)
total_models = len(model_names)

dist_pairs_saved = np.zeros((total_models, total_models), dtype=bool)
if os.path.exists("distances/widthdepth/stats.npz"):
    dist_pairs_saved = np.load("distances/widthdepth/stats.npz")["dist_pairs_saved"]
    print(f"{np.sum(dist_pairs_saved)} existing pairs", flush=True)

model_pairs = []
for i in range(total_models):
    for j in range(total_models):
        if j > i and not dist_pairs_saved[i, j]:
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

    model1_path = f'{modeldir}{name1}.pth'
    model1_lastlayer = torch.load(model1_path)[0:-1]
    rep1 = model1_lastlayer(full_train_data).detach().numpy().T
    # center and normalize
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep1 = math.sqrt(num_data) * rep1 / np.linalg.norm(rep1)

    model2_path = f'{modeldir}{name2}.pth'
    model2_lastlayer = torch.load(model2_path)[0:-1]
    rep2 = model2_lastlayer(full_train_data).detach().numpy().T
    # center and normalize
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
    rep2 = math.sqrt(num_data) * rep2 / np.linalg.norm(rep2)
    
    all_dists = evaluate_distances(rep1, rep2)
    for dist_name in all_dists:
        dist = all_dists[dist_name]
        print(f'{dist_name}: {dist}', flush=True)
    print(flush=True)
