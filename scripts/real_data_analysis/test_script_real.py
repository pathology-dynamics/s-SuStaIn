import time
import os
import warnings
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from functools import partial, partialmethod
from kde_ebm.mixture_model import fit_all_gmm_models, fit_all_kde_models
from kde_ebm import plotting
from sSuStaIn.sEBMSustain import sEBMSustain
import numpy.ma as ma

path = "/home/rtandon32/ebm/ebm_experiments/experiment_scripts/real_data/dfMri_D12_ebm_final_n327.csv"
df = pd.read_csv(path)


k=119
exclude_idx = [21,  71,  22,  19, 111,  64,  96,  44,  90,  50,  80,  57,   9,
        77,  33, 113,  76,  98, 105,  79,  41,   4,  87, 107, 114,  20,
         2, 110,  84,  23,  58,   1,  60,  56,  10,  94,  49,  74,   0,
        92,  91,  18,  61,  43,  45, 108,  78, 118,  62,   8,   5,  46,
         3,  47,  48,  53,  42, 112,  73]

select_cols = [_ for _ in range(k) if _ not in exclude_idx]
data = df.iloc[:,select_cols]
bm_names = data.columns.tolist()
data = data.values
labels = df["DX"].map({"Dementia":1, "CN":0})


sustainType             = 'mixture_GMM'
if sustainType == "mixture_GMM":
    mixtures            = fit_all_gmm_models(data, labels)
elif sustainType == "mixture_KDE":
    mixtures            = fit_all_kde_models(data, labels)

L_yes                   = np.zeros(data.shape)
L_no                    = np.zeros(data.shape)

# for i in range(k):
for i in range(len(select_cols)):
    if sustainType == "mixture_GMM":
        L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])
    elif sustainType   == "mixture_KDE":
        L_no[:, i], L_yes[:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1))

def process_L(L, min_val=0):
    mx = ma.masked_less_equal(L,min_val)
    min_masked = mx.min(axis=0)
    L_new = mx.filled(fill_value=min_masked)
    return L_new

L_no = process_L(L_no)
L_yes = process_L(L_yes)



stage_sizes = [12,12,12,12,12]
N_startpoints           = 40
N_S_max                 = 1
SuStaInLabels = bm_names
rep = 3
N_iterations_MCMC_init = int(2e3)
N_iterations_MCMC       = int(3e3)  #Generally recommend either 1e5 or 1e6 (the latter may be slow though) in practice
n_stages = 5
min_clust_size = 12
p_absorb = 0.0
N_em = 100


dataset_name            = 'sim_tadpole14'
output_dir              = '/home/rtandon32/ebm/s-SuStain-outputs'
output_folder           = os.path.join(output_dir, dataset_name + '_' + sustainType)
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
use_parallel_startpoints = True



sustain = sEBMSustain(L_yes, L_no, n_stages, stage_sizes, min_clust_size, p_absorb, rep, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC_init, N_iterations_MCMC, N_em, output_folder, dataset_name, use_parallel_startpoints)

# sustain = sEBMSustain(L_yes, L_no, 5, stage_sizes, 15, 0.4, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC, output_folder, dataset_name, use_parallel_startpoints)

samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = sustain.run_sustain_algorithm(plot=True)
# print(samples_sequence, samples_f)