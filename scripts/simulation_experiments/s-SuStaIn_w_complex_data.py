import sys
sys.path.insert(0, '/home2/rtandon32/ebm/s-SuStaIn/sim')

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cbook as cbook

import os
import pandas as pd
from simfuncs import *
from functools import partial, partialmethod
from kde_ebm.mixture_model import fit_all_gmm_models, fit_all_kde_models
from kde_ebm import plotting
import warnings
from sSuStaIn.sEBMSustain import sEBMSustain
import sklearn.model_selection
import pylab
import json
from joblib import Parallel, delayed
import numpy.ma as ma


read_dir = "/home/rtandon32/ebm/s-SuStain-outputs/simulation_experiments/complex/input_data"
# write_dir = "/home/rtandon32/ebm/s-SuStain-outputs/simulation_experiments/complex/output"
write_dir = "/home/rtandon32/ebm/s-SuStain-outputs/simulation_experiments/complex/output_runtime"

def get_read_path(n_sample, dim, seed, nc, root_path=read_dir):
    read_path = os.path.join(root_path, "{}_samples".format(n_sample), "{}_dims".format(dim),
                              "{}_components".format(nc), "params_rep_{}.json".format(seed))
    return read_path

def read_json(file_path):
    with open(file_path) as f:
        hp = json.load(f)
    return hp

def extract_fields(hp):
    X = np.array(hp["X_data"])
    y = np.array(hp["y_data"])
    gt_order = hp["gt_dict"]
    mixture_fractions = np.array(hp["mixture_p"])
    subtype = np.array(hp["subtype_from_seed"])
    return X, y, gt_order, mixture_fractions, subtype

def process_L(L, min_val=0):
    mx = ma.masked_less_equal(L,min_val)
    min_masked = mx.min(axis=0)
    L_new = mx.filled(fill_value=min_masked)
    return L_new

def run_SuStaIn(run_path):
    hp = read_json(run_path)
    X, y, gt_order, mixture_fractions, subtype = extract_fields(hp)
    L_yes = np.zeros(X.shape)
    L_no  = np.zeros(X.shape)
    dim = X.shape[1]
    n_samples = X.shape[0]
    nc = len(gt_order)
    seed = run_path.split(".")[-2][-1]

    # Create mixture models
    mixtures = fit_all_gmm_models(X, y)
    for i in range(dim):
        L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, X[:, i])
    L_no = process_L(L_no)
    L_yes = process_L(L_yes)
    
    # s-SuStaIn Params
    rep=1
    n_stages = 5
    assert dim % n_stages == 0
    stage_sizes = [dim // n_stages] * n_stages
    min_clust_size = int(dim/(2*n_stages))
    p_absorb = 0.3
    SuStaInLabels = ['BM ' + str(i) for i in range(dim)]

    # MCMC params
    N_iterations_MCMC_init=1e4
    N_iterations_MCMC = int(1e3)
    N_startpoints = 1
    N_em = 100
    N_S_max = nc
    use_parallel_startpoints = False

    # Output paths
    dataset_name = 'simulation_rep_{}'.format(seed)
    output_dir = os.path.join(write_dir, "{}_samples".format(n_samples), "{}_dims".format(dim), "{}_components".format(nc), "SuStaIn_sEBM")
    output_folder = os.path.join(output_dir, dataset_name)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    sustain = sEBMSustain(L_yes, L_no, n_stages, stage_sizes, min_clust_size, p_absorb, rep, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC_init,N_iterations_MCMC, N_em, output_folder, dataset_name, use_parallel_startpoints)
    samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = sustain.run_sustain_algorithm(plot=True)


if __name__ == "__main__":
    samples = [200]
    samples = np.array(samples).astype(int)
    dims = [50, 100, 150, 200, 250, 300]
    dims = np.array(dims).astype(int)
    seeds = list(range(3))
    n_components = [4]

    run_objects = []
    for n_sample in samples:
        for n_dim in dims:
            for s in seeds:
                for nc in n_components:
                    path = get_read_path(n_sample, n_dim, s, nc, read_dir)
                    run_objects.append(path)

    Parallel(n_jobs=min(18, len(run_objects)))(delayed(run_SuStaIn)(obj) for obj in run_objects)