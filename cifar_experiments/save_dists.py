import sys
import os
import numpy as np

n = 10000

all_filenames = os.listdir()
filenames = []
for filename in all_filenames:
    if filename.startswith("distcomp-"):
        filenames.append(filename)
filenames = np.array(filenames)


layers = np.array([1, 2, 3, 4, 5])
widths = np.array([200, 400, 600, 800, 1000])
trials = np.array([0, 1, 2, 3, 4])

total_models = len(layers) * len(widths) * len(trials)
model_names = []
for layer in layers:
    for width in widths:
        for trial in trials:
            model_names.append(f'fc_cifar_{layer}_{width}_{trial}')
model_names = np.array(model_names)

folder = f"distances/{n}"
dist_pairs_saved = np.zeros((total_models, total_models), dtype=bool)
if os.path.exists(f"{folder}/stats.npz"):
    dist_pairs_saved = np.load(f"{folder}/stats.npz")["dist_pairs_saved"]

dists = {}
print(folder)
if os.path.exists(folder):
    existing_dist_files = os.listdir(folder)
    existing_dist_names = [name[:-4] for name in existing_dist_files if name.endswith(".npy")]
    for dist_name in existing_dist_names:
        dists[dist_name] = np.load(f'{folder}/{dist_name}.npy')
else:
    os.mkdir(folder)

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
        elif line.count(": ") == 1 and line.count(" ") == 1:
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
    np.save(f'{folder}/{dist_name}.npy', dists[dist_name])
np.savez(f'{folder}/stats.npz', model_names=model_names, dist_pairs_saved=dist_pairs_saved)
