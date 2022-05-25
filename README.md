# GULP: a prediction-based metric between representations
This repository contains the code to replicate the results in the paper: "GULP: a prediction-based metric between representations".


The experiments are split into three groups which study distances between network representations on MNIST, ImageNet, and CIFAR. For all experiments, distances between network representations are presaved in this repository. However, if the user wishes to recompute these distances or access the network representations themselves (which are memory intensive), they will need to load/train the corresponding network architectures and recompute the distances between them. 

## MNIST Experiments (`mnist_experiments/`)
* Distances between fully-connected ReLU networks of varying widths and depths are saved in `distances/widthdepth/`.
* To recompute these distances, first clear the folder `distances/widthdepth/` and load the MNIST dataset into the `data/MNIST/` placeholder folder. Then run the slurm script `fit_loop.sh` which will train fully-connected ReLU networks on MNIST of varying width and depth and save these architectures to `models/widthdepth/`. Finally, run the slurm script `dist_loop.sh` to compute all pairwise distance between the final-layer representations of these networks which will be saved in `distances/widthdepth/`.
* All visualizations of MNIST networks in Figures 5 and 11 of the paper can be reproduced in the notebook `width_depth_embedding.ipynb`.
* The relationships between distances on MNIST models in Figure 9 can be reproduced in the notebook `Compare_other_distances_to_GULP.ipynb`.

## ImageNet Experiments (`imagenet_experiments/`)
* Distances between pretrained and untrained state-of-the-art ImageNet networks taken from https://pytorch.org/vision/stable/models.html#classification are saved in `distances/pretrained/` and `distances/untrained/` respectively.
* To recompute these distances, first clear the folders `distances/pretrained` and `distances/untrained`. Load all untrained and pretrained PyTorch ImageNet models by running `load_models.py`. Then load the ImageNet dataset into a local folder and save its path. Paste this path into the file `compute_reps.py` and run the slurm script `rep_loop.sh` which will save the final-layer representations of all pretrained and untrained ImageNet networks loaded from PyTorch. Finally, run the slurm script `dist_loop.sh` to compute all pairwise distance between these final-layer representations which will be saved in `distances/train/pretrained` and `distances/train/untrained` respectively.
* All visualizations of ImageNet networks in Figures 1, 6, 12, 13 and 14 of the paper can be reproduced in the notebook `embed_models.ipynb`.
* The relationships between distances on ImageNet models in Figures 2 and 8 can be reproduced in the notebook `Compare_other_distances_to_GULP.ipynb`.
* The convergence of the plug-in estimator in Figures 3 and 10 can be reproduced in the notebook `Convergence_of_the_plug_in_estimator.ipynb'.
* How GULP captures generalization performance on linear predictors in Figure 4 can be reproduced in the notebook `GULP_versus_linear_predictor_generalization.ipynb`. This requires loading the ImageNet representations (see above).
* The GULP distance versus generalization performance on logistic predictors in Figure 17 can be reproduced in the notebook `GULP_versus_logistic_predictor_generalization.ipynb`. This requires loading the ImageNet representations (see above).

## CIFAR Experiments (`cifar_experiments/`)
* Distances between Resnet18 networks trained on CIFAR10 at every epoch (of 50 epochs) of training are included in the zip file cifar_dists.zip. To plot these distances, adjust the working directory "dirwithreps" based on the unzipped file in the notebook plot_CIFAR_distances_during_training.ipynb and run all cells.
* We do not include code to retrain the networks, but refer to the FFCV package and/or any standard training code for training Resnet18 architectures on CIFAR10 and saving the final representations. Given representations saved in the hierarchy e.g. cifar1/test/epoch3/latents.pkl, where 1 indicates the index of the model, 3 is the epoch, test is the split of the data on which the representations were generated, and latents.pkl contains a dictionary with key 'last', the script compute_CIFAR_distances_from_saved_representations.py recomputes the provided distances.
* The GULP distance between networks during training in Figures 7 and 15 of the paper can be reproduced in the notebook `plot_CIFAR_distances_during_training.ipynb`.
