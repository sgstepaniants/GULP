import torch
import sys
import os
sys.path.append(os.path.abspath("../"))
import distance_functions
from distance_functions import cca_decomp, mean_sq_cca_corr, mean_cca_corr, pwcca_dist, lin_cka_dist, lin_cka_prime_dist, procrustes, predictor_dist
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import os.path

if __name__ == "__main__":
    parser = argparse.ArgumentParser('demo')
    parser.add_argument('--center', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--run_name', type=str, default=None) #GIVE A RUN NAME!
    parser.add_argument('--last_epoch',action='store_true',default=False)
    parser.add_argument('--save_evals',action='store_true',default=False)
    parser.add_argument('--just_test',action='store_true',default=False)
    parser.add_argument('--min_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=50)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # where to save results of these distance computations
    img_dir = f'/saved/' 
    
    # assume cifar representations have been saved here,
    # under the hierarchy e.g. cifar_representations/cifar2/test/epoch3/latents.pkl 
    # for the second trained network, epoch 3, test reps
    # 
    # latents.pkl should contain a file with key 'last' containing the last hidden layer representations
    # To generate Figure 7, we trained CIFAR Resnets using https://github.com/MadryLab/failure-directions, which relies
    # on FFCV (https://github.com/libffcv/ffcv) for fast training 
    dir_for_saved_reps = '/saved/cifar_representations/' 

    if args.run_name is not None:
        img_dir = img_dir + args.run_name + '/'

    if args.center:
        if args.normalize:
            img_dir = img_dir + 'centered_normalized/'
        else:
            img_dir = img_dir + 'centered/'
    else:
        if args.normalize:
            img_dir = img_dir + 'normalized/'

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        
    def mean_sq_cca_e2e(A,B):
        u, s, vh, transformed_a, transformed_b = cca_decomp(A,B)
        return mean_sq_cca_corr(s)

    def mean_cca_e2e(A,B):
        u, s, vh, transformed_a, transformed_b = cca_decomp(A,B)
        return mean_cca_corr(s)

    def pwcca_dist_e2e(A,B):
        u, s, vh, transformed_a, transformed_b = cca_decomp(A,B)
        return pwcca_dist(A, s, transformed_a)

    lambda_range = np.power(10.0,range(-8,3))
    lambda_range[0]=0

    def predictor_dist_range(A,B):
        res=[]
        for lmb in lambda_range:
            res.append(predictor_dist(A,B,lmbda=lmb))
        return res
        

    all_dist_funcs = {'mean_sq_cca_e2e': mean_sq_cca_e2e, 'mean_cca_e2e':mean_cca_e2e, 'pwcca_dist_e2e':pwcca_dist_e2e, 'lin_cka_dist':lin_cka_dist, 'lin_cka_prime_dist':lin_cka_prime_dist, 'procrustes':procrustes, 'predictor_dist_range':predictor_dist_range}

    
    def get_rep_path(ind, loadername, epoch):
        fname = dir_for_saved_reps + f'cifar{ind}/{loadername}/epoch{epoch}/latents.pkl'
        return fname

    def get_epoch_dir(ind, loadername, epoch):
        return dir_for_saved_reps + f'cifar{ind}/{loadername}/epoch{epoch}/'

    def get_last_reps(ind, loadername, epoch):
        fname = get_rep_path(ind, loadername, epoch) 
        with open(fname,'rb') as f: 
            out = pickle.load(f)
        last_latents = out['last']
        return last_latents
    
    maxsubdirnum = 16

    if args.save_evals:
        # FIRST PASS: SAVE ALL EIGENVALS, EIGENVECTORS
        
        full_res = {}
        for loadername in ['train','test']:
            loader_res = {}
            print('\nLOADER', loadername)
            for ind1 in range(1, maxsubdirnum+1):
                print('IND1', ind1)
                total_epochs = 50
                if args.last_epoch:
                    epoch_vals = range(49,50)
                else:
                    epoch_vals = range(args.min_epoch, args.max_epoch)
                print('min', args.min_epoch)
                for epoch in epoch_vals:
                    print('\n\n  EPOCH', epoch,'\n\n')
                    
                    if not os.path.isfile(get_epoch_dir(ind1, loadername, epoch) + 'eigenvals.pkl'):

                        lat1 = get_last_reps(ind1, loadername, epoch).T
                        if args.center:
                            lat1 = lat1 - lat1.mean(axis=1, keepdims=True)
                        if args.normalize:
                            lat1 = lat1 * np.sqrt(lat1.shape[1]) / np.linalg.norm(lat1)
                        print('lat1', lat1.shape)
                        evals_a, evecs_a = np.linalg.eigh(lat1 @ lat1.T)

                        with open(get_epoch_dir(ind1, loadername, epoch) + 'eigenvals.pkl', 'wb') as f:
                            pickle.dump({'evals':evals_a, 'evecs':evecs_a}, f)


    # Now, compute the distances

    full_res = {}
    if args.just_test:
        loadernames = ['test']
    else:
        loadernames = ['train','test']
    for loadername in loadernames:
        loader_res = {}
        print('\nLOADER', loadername)
        
        total_epochs = 50
        if args.last_epoch:
            epoch_vals = range(49,50)
        else:
            epoch_vals = range(args.min_epoch, total_epochs)
        for epoch in epoch_vals:
            this_epoch_res = {}
            print('\n\n  EPOCH', epoch,'\n\n')
            for ind1 in range(1, maxsubdirnum):
                lat1 = get_last_reps(ind1, loadername, epoch).T
                if args.center:
                    lat1 = lat1 - lat1.mean(axis=1, keepdims=True)
                if args.normalize:
                    lat1 = lat1 * np.sqrt(lat1.shape[1]) / np.linalg.norm(lat1)

                A = lat1
                with open(get_epoch_dir(ind1, loadername, epoch) + 'eigenvals.pkl', 'rb') as f:
                    out = pickle.load(f)
                    evals_a = out['evals']
                    evecs_a = out['evecs']

                for ind2 in range(ind1+1, maxsubdirnum+1):
                    print('--------ind1',ind1,'ind2',ind2)
                    lat2 = get_last_reps(ind2, loadername, epoch).T
                    if args.center:
                        lat2 = lat2 - lat2.mean(axis=1, keepdims=True)
                    if args.normalize:
                        lat2 = lat2 * np.sqrt(lat2.shape[1]) / np.linalg.norm(lat2)
                    B = lat2 
                    with open(get_epoch_dir(ind2, loadername, epoch) + 'eigenvals.pkl', 'rb') as f:
                        out = pickle.load(f)
                        evals_b = out['evals']
                        evecs_b = out['evecs']
                    
                    all_dists = {}
                    u, s, vh, transformed_a, transformed_b = cca_decomp(A, B, evals_a, evecs_a, evals_b, evecs_b)
                    
                    all_dists = {}
                    all_dists['mean_sq_cca_e2e'] = mean_sq_cca_corr(s)
                    all_dists['mean_cca_e2e'] = mean_cca_corr(s)
                    all_dists['pwcca_dist_e2e'] = pwcca_dist(A, s, transformed_a)
                    
                    all_dists['lin_cka_dist'] = lin_cka_dist(A, B)
                    all_dists['lin_cka_prime_dist'] = lin_cka_prime_dist(A, B)
                    
                    all_dists['procrustes'] = procrustes(A, B)
                    
                    for lmbda in lambda_range:
                        all_dists[f'predictor_dist_{lmbda}'] = predictor_dist(A, B, evals_a, evecs_a, evals_b, evecs_b, lmbda=lmbda)
                    
                    loader_res[f'e{epoch}_{ind1}_{ind2}'] = all_dists
                    this_epoch_res[f'{ind1}_{ind2}'] = all_dists
                    if not os.path.exists(img_dir + loadername + f'/e{epoch}'):
                        os.makedirs(img_dir + loadername + f'/e{epoch}')
                    with open(img_dir + loadername + f'/e{epoch}/dists.pkl', 'wb') as f:
                        pickle.dump(this_epoch_res, f)





        if not os.path.exists(img_dir + loadername):
            os.makedirs(img_dir + loadername)
        with open(img_dir + loadername + '/dists.pkl', 'wb') as f:
            pickle.dump(loader_res, f)
        full_res[loadername] = loader_res
    with open(img_dir + '/dists.pkl', 'wb') as f:
        pickle.dump(full_res, f)

                        

        
                
                    

                    
                    


