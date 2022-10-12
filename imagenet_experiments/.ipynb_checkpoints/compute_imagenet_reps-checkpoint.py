import os
import sys
import math
import numpy as np
import torch
from torchvision import datasets, transforms
import random

# number of samples in each representation
n = 3000

subset = "val"
eval_mode = True

reps_folder = f"reps/{subset}/{n}_eval2"
if not os.path.exists(reps_folder):
    os.makedirs(reps_folder)

batch_num = int(sys.argv[1])
total_batches = int(sys.argv[2])

model_names = []
file_names = os.listdir("models")
for filename in file_names:
    if filename.endswith(".pth"):
        model_names.append(filename[:-4])
random.Random(4).shuffle(model_names)

total_models = len(model_names)
batch_size = math.ceil(total_models / total_batches)
#print(batch_size, flush=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
dataset = datasets.ImageFolder(f"/home/gridsan/groups/datasets/ImageNet/{subset}/", transform=transform)

batch_names = model_names[(batch_num-1)*batch_size:batch_num*batch_size]
for model_name in batch_names:
    print(f"Computing representation for {model_name}", flush=True)
    
    model = torch.load(f"models/{model_name}.pth")
    if eval_mode:
        model.eval()
    print(model, flush=True)
    
    d = model(dataset[0][0][None, :, :, :]).shape[1]
    print(f"d = {d}", flush=True)
    
    g_cpu = torch.Generator()
    g_cpu.manual_seed(1234)
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, generator=g_cpu, num_workers=0)
    
    rep = np.zeros((n, d))
    i = 0
    for batch_data, batch_labels in loader:
        if i >= n:
            break
        nb = min(batch_data.shape[0], n-i)
        with torch.no_grad():
            rep[i:i+nb, :] = model(batch_data)[0:nb, :]
        i += nb
        print(f"{i} / {n}", flush=True)
        print(batch_labels, flush=True)
        print(flush=True)
    
    np.save(f"{reps_folder}/{model_name}_rep.npy", rep.T)
