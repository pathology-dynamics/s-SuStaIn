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
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

data_save_path = "/nethome/rtandon32/ebm/real_datasets/ehbs"

def process_L(L, min_val=0):
    mx = ma.masked_less_equal(L,min_val)
    min_masked = mx.min(axis=0)
    L_new = mx.filled(fill_value=min_masked)
    return L_new

def readData():
	peptide_fname = os.path.join(data_save_path, "Peptide Area Report_BSR2020-102_80pep.csv")
	skyline_fname = os.path.join(data_save_path, "SkylineRatios-FullPercesion_2021_0608.csv")
	skyline_df = pd.read_csv(skyline_fname, index_col=0)
	peptide_df = pd.read_csv(peptide_fname)
	# Get the labels for all subjects in the data
	label_df = readLabels(peptide_df)
	# Impute missing values
	Ximp = imputeData(skyline_df)
	ss = StandardScaler()
	scaledX = ss.fit_transform(Ximp)
	scaledX = pd.DataFrame(scaledX, index=Ximp.index, columns=Ximp.columns)
	scaledX = scaledX.reindex(label_df["sbj"])
	Ximp = Ximp.reindex(label_df["sbj"])
	return scaledX, Ximp, label_df

def readLabels(peptide_df):
	label_df = peptide_df[["Replicate", "Condition"]]
	label_df = label_df[label_df["Condition"].isin(["AD", "Control", "AsymAD"])]
	label_dict = dict(zip(label_df["Replicate"], label_df["Condition"]))
	label_df = pd.DataFrame.from_dict(label_dict, orient="index").reset_index()
	label_df.columns = ["sbj", "DX"]
	return label_df

def imputeData(x):
	x_imp = x.T.values.copy()
	n_neighbors=10
	x_imp = imptKNN(x_imp, n_neighbors)
	Ximp = pd.DataFrame(x_imp, index=x.columns, columns=x.index)
	return Ximp

def imptKNN(x, n_neighbors, weights="uniform"):
	imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
	x_imputed = imputer.fit_transform(x)
	return x_imputed

def extract_classes(X, label_df, dx):
	merged = pd.merge(X, label_df, left_index=True, right_index=True)
	merged_subset = merged[merged["DX"].isin(dx)]
	# y = merged_subset["DX"].map({"Control":0, "AD":1}).to_numpy()
	y = merged_subset["DX"]
	x = merged_subset[merged_subset.columns.difference(["DX"])]
	x = x.to_numpy().astype(float)
	return x, y

scaledX, Ximp, label_df = readData()
label_df.set_index("sbj", inplace=True)

X, y = extract_classes(Ximp, label_df, ["Control", "AD"])
print("outputs from extract classes \n", X.shape, y.shape)
y = y.map({"Control":0, "AD":1}).to_numpy()
bm_names = Ximp.columns.tolist()



mixture_type = "gmm"
if mixture_type=="kde":
    mm_fit = fit_all_kde_models
elif mixture_type == "gmm":
    mm_fit = fit_all_gmm_models
	
mixtures = mm_fit(X, y)

L_yes                   = np.zeros(X.shape)
L_no                    = np.zeros(X.shape)

for i in range(X.shape[1]):
    if mixture_type == "gmm":
        L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, X[:, i])
    elif mixture_type == "kde":
        L_no[:, i], L_yes[:, i] = mixtures[i].pdf(X[:, i].reshape(-1, 1))

# print("L_no zeroes \n", (L_no==0.0).sum(axis=0))
# print("L_yes zeroes \n", (L_yes==0.0).sum(axis=0))


# L_no_zeros = (L_no==0.0).sum(axis=0)
# L_yes_zeros = (L_yes==0.0).sum(axis=0)
# print(np.nonzero(L_no_zeros), np.nonzero(L_yes_zeros))



L_no = process_L(L_no)
L_yes = process_L(L_yes)

# print(bm_names[6], bm_names[42])

stage_sizes = [15,15,15,15,15]
N_startpoints           = 50
N_S_max                 = 4
# SuStaInLabels = df.columns[select_cols].tolist()
SuStaInLabels = bm_names
rep = 20
N_iterations_MCMC_init = int(2e4)
N_iterations_MCMC       = int(1e6)  #Generally recommend either 1e5 or 1e6 (the latter may be slow though) in practice
n_stages = 5
min_clust_size = 6
p_absorb = 0.3


dataset_name            = 'sim_ehbs1'
output_dir              = '/home/rtandon32/ebm/s-SuStain-outputs'
output_folder           = os.path.join(output_dir, dataset_name + '_' + mixture_type)
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
use_parallel_startpoints = True


# Create the sEBMSuStaIn object
sustain = sEBMSustain(L_yes, L_no, n_stages, stage_sizes, min_clust_size, p_absorb, rep, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC_init, N_iterations_MCMC, output_folder, dataset_name, use_parallel_startpoints)

# Run SuStaIn algorithm
samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = sustain.run_sustain_algorithm(plot=True)