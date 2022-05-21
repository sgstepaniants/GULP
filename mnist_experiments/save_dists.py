import sys
import os
import numpy as np

width_subset = 100*np.arange(1, 11)
depth_subset = np.arange(1, 11)

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

all_filenames = os.listdir()
filenames = []
batches = []
total_batches = []
for filename in all_filenames:
    if filename.startswith("distcomp-"):
        filenames.append(filename)
        splits = filename[:-4].split("-")
        batches.append(int(splits[1]))
        total_batches.append(int(splits[2]))
filenames = np.array(filenames)
batches = np.array(batches)

num_batches = np.unique(total_batches)
if len(num_batches) == 0:
    num_batches = 0
elif len(num_batches) == 1:
    num_batches = num_batches[0]
else:
    raise Exception("Mixed batch sizes")

sorted_inds = np.argsort(batches)
batches = batches[sorted_inds]
filenames = filenames[sorted_inds]

dist_pairs_saved = np.zeros((total_models, total_models), dtype=bool)
if os.path.exists("distances/widthdepth/stats.npz"):
    dist_pairs_saved = np.load("distances/widthdepth/stats.npz")["dist_pairs_saved"]

dists = {}
existing_dist_files = os.listdir("distances/widthdepth/")
existing_dist_names = [name[:-4] for name in existing_dist_files if name.endswith(".npy")]
for dist_name in existing_dist_names:
    dists[dist_name] = np.load(f'distances/widthdepth/{dist_name}.npy')

print(f'existing pairs {np.sum(dist_pairs_saved)} / {int(total_models * (total_models - 1) / 2)}', flush=True)

for filename in filenames:
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        if line.startswith("Computing"):
            splits = line.split(", ")
            name1 = splits[0][10:].strip()
            name2 = splits[1].strip()
            
            i1 = np.where(model_names == name1)[0][0]
            i2 = np.where(model_names == name2)[0][0]
        elif line.count(": ") == 1:
            splits = line.split(": ")
            dist_name = splits[0]
            dist = float(splits[1])
            if dist_name not in dists:
                dists[dist_name] = np.zeros((total_models, total_models))
                dists[dist_name][:] = np.nan
            dists[dist_name][i1, i2] = dist

for i1 in range(total_models):
    for i2 in range(i1+1, total_models):
        dist_pairs_saved[i1, i2] = True
        for dist_name in dists:
            if np.isnan(dists[dist_name][i1, i2]):
                dist_pairs_saved[i1, i2] = False

for dist_name in dists:
    print(dist_name, flush=True)
    np.save(f'distances/widthdepth/{dist_name}.npy', dists[dist_name])
np.savez('distances/widthdepth/stats.npz', model_names=model_names, dist_pairs_saved=dist_pairs_saved)
