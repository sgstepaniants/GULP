# GULP: a prediction-based metric between representations
This repository contains the code to replicate the results in the paper: "GULP: a prediction-based metric between representations".


The experiments are split into three groups which study distances between network representations on MNIST, ImageNet, and CIFAR. For all experiments, distances between network representations are presaved in this repository. However, if the user wishes to recompute these distances or access the network representations themselves (which are memory intensive), they will need to load/train the corresponding network architectures and recompute the distances between them. We have provided the necessary scripts to load/train all models and compute distances between them in parallel on a slurm cluster.

## MNIST Experiments (`mnist_experiments/`):
* Distances between fully-connected ReLU networks of varying widths and depths are saved in `distances/widthdepth/`.
* To recompute these distances, first clear the folder `distances/widthdepth/` and load the MNIST dataset into the `data/MNIST/` placeholder folder. Then run the slurm script `fit_loop.sh` which will train fully-connected ReLU networks on MNIST of varying width and depth and save these architectures to `models/widthdepth/`. Finally, run the slurm script `dist_loop.sh` to compute all pairwise distance between the final-layer representations of these networks which will be saved in `distances/widthdepth/`.
* All visualizations of MNIST networks in Figures 5 and 11 of the paper can be reproduced in the notebook `width_depth_embedding.ipynb`.

## ImageNet Experiments (`imagenet_experiments/`):
* Distances between pretrained and untrained state-of-the-art ImageNet networks taken from https://pytorch.org/vision/stable/models.html#classification are saved in `distances/pretrained/` and `distances/untrained/` respectively.
* To recompute these distances, first clear the folders `distances/pretrained` and `distances/untrained`. Load all untrained and pretrained PyTorch ImageNet models by running `load_models.py`. Then load the ImageNet dataset into a local folder and save its path. Paste this path into the file `compute_reps.py` and run the slurm script `rep_loop.sh` which will save the final-layer representations of all pretrained and untrained ImageNet networks loaded from PyTorch. Finally, run the slurm script `dist_loop.sh` to compute all pairwise distance between these final-layer representations which will be saved in `distances/train/pretrained` and `distances/train/untrained` respectively.
* All visualizations of ImageNet networks in Figures 6, 12, 13 and 14 of the paper can be reproduced in the notebook `embed_models.ipynb`.

## CIFAR Experiments (`cifar_experiments/`):
