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
#warnings.filterwarnings("ignore",category=cbook.mplDeprecation)

# from pySuStaIn.sEBMSustain import sEBMSustainData
from sSuStaIn.sEBMSustain import sEBMSustain
# from pySuStaIn.ZscoreSustain  import ZscoreSustain
# from pySuStaIn.MixtureSustain import MixtureSustain

import sklearn.model_selection

import pylab

N                       = 10         # number of biomarkers
M                       = 200       # number of observations ( e.g. subjects )
N_S_ground_truth        = 3         # number of ground truth subtypes
stage_sizes = [2,3,3,2]
# the fractions of the total number of subjects (M) belonging to each subtype
ground_truth_fractions = np.array([0.5, 0.30, 0.20])

#create some generic biomarker names
BiomarkerNames           = ['Biomarker ' + str(i) for i in range(N)]

#***************** parameters for SuStaIn-based inference of subtypes
use_parallel_startpoints = True

# number of starting points
N_startpoints           = 25
# maximum number of inferred subtypes - note that this could differ from N_S_ground_truth
N_S_max                 = 2
N_iterations_MCMC       = int(1e6)  #Generally recommend either 1e5 or 1e6 (the latter may be slow though) in practice

#labels for plotting are biomarker names
SuStaInLabels           = BiomarkerNames

# cross-validation
validate                = True
N_folds                 = 3         #Set low to speed things up here, but generally recommend 10 in practice

#either 'mixture_GMM' or 'mixture_KDE' or 'zscore'
sustainType             = 'mixture_GMM'


dataset_name            = 'simx'
output_folder           = os.path.join(os.getcwd(), dataset_name + '_' + sustainType)
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

ground_truth_subj_ids   = list(np.arange(1, M+1).astype('str'))
ground_truth_sequences  = generate_random_mixture_sustain_model(N, N_S_ground_truth)

ground_truth_subtypes   = np.random.choice(range(N_S_ground_truth), M, replace=True, p=ground_truth_fractions).astype(int)

N_stages                = N

ground_truth_stages_control = np.zeros((int(np.round(M * 0.25)), 1))
ground_truth_stages_other   = np.random.randint(1, N_stages+1, (int(np.round(M * 0.75)), 1))
ground_truth_stages         = np.vstack((ground_truth_stages_control, ground_truth_stages_other)).astype(int)

data, data_denoised     = generate_data_mixture_sustain(ground_truth_subtypes, ground_truth_stages, ground_truth_sequences, sustainType)

# choose which subjects will be cases and which will be controls
MIN_CASE_STAGE          = np.round((N + 1) * 0.8)
index_case              = np.where(ground_truth_stages >=  MIN_CASE_STAGE)[0]
index_control           = np.where(ground_truth_stages ==  0)[0]

labels                  = 2 * np.ones(data.shape[0], dtype=int)     # 2 - intermediate value, not used in mixture model fitting
labels[index_case]      = 1                                         # 1 - cases
labels[index_control]   = 0                                         # 0 - controls

data_case_control       = data[labels != 2, :]
labels_case_control     = labels[labels != 2]

if sustainType == "mixture_GMM":
    mixtures            = fit_all_gmm_models(data, labels)
elif sustainType == "mixture_KDE":
    mixtures            = fit_all_kde_models(data, labels)

# fig, ax                 = plotting.mixture_model_grid(data_case_control, labels_case_control, mixtures, SuStaInLabels)#, plotting_font_size=20)
# fig.show()
# fig.savefig(os.path.join(output_folder, 'kde_fits.png'))

L_yes                   = np.zeros(data.shape)
L_no                    = np.zeros(data.shape)
for i in range(N):
    if sustainType == "mixture_GMM":
        L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])
    elif sustainType   == "mixture_KDE":
        L_no[:, i], L_yes[:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1))

sustain = sEBMSustain(L_yes, L_no, stage_sizes, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC, output_folder, dataset_name, use_parallel_startpoints)

rng = np.random.default_rng(0)
seq_init = sustain._initialise_sequence(sustain._sEBMSustain__sustainData, rng)
flattened = sustain._flatten_S_dict(seq_init[0])
print("flattened seq init", flattened)

seq_gt1 = {0: np.array([1,4]), 1: np.array([5,7,2]), 2: np.array([9,0,8]), 3: np.array([6,3])}
seq_gt2 = {0: np.array([7,9]), 1: np.array([5,6,1]), 2: np.array([4,0,3]), 3: np.array([2,8])}

a, b, c, _, _, _ = sustain._perform_em(sustain._sEBMSustain__sustainData, [seq_gt1, seq_gt2], [0.2,0.8], rng)
# print("a", a)
# print("b", b)
print("ground truth sequences", ground_truth_sequences)
print("ground truth fractions", ground_truth_fractions)
a, b, c, d, e, f = sustain._estimate_ml_sustain_model_nplus1_clusters(sustain._sEBMSustain__sustainData, a, b)
print("seq init", a)
print("f init", b)
print("ll", c)
# print("c", c)
# print("d", d)
# print("e", e)
# print("f", f)
# tt = sustain._flatten_sequence([seq_gt1, seq_gt2])
# print(tt)
# print(type(tt))

a, b, c, d, e, f = sustain._perform_mcmc(sustain._sEBMSustain__sustainData, a, b, 200000, 1, 0.01)
print("MCMC seq", a)
print("MCMC f", b)
print("MCMC ll", c)