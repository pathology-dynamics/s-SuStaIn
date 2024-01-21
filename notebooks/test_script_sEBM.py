"""
Setting-up run environment
"""
import os
import sys
import json
import pdb
# Load settings and run parameters
f = open("/nethome/rtandon32/ebm/ebm_experiments/experiment_scripts/run_params.json")
run_params = json.load(f)
output_path = run_params["save_path"]
# Extend system paths
sys.path.extend(run_params["add_paths"])
sys.path.extend(["/nethome/rtandon32/ebm/ebm_experiments/experiment_scripts/awkde/"])
"""
Importing general purpose modules
"""
import time
import warnings
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from matplotlib import pyplot as plt
from joblib import Parallel, delayed


"""
Importing ebm related modules
"""

from kde_ebm import mcmc
from kde_ebm import plotting
from kde_ebm import datasets
from kde_ebm import mixture_model
# from read_xy import LahData, SimulationParams, RunObject
# import partial_ranking_scores as prs
# import kt_analysis as kt
# from kde_ebm.event_order import EventOrder, EventOrderCustomSet, EventOrderFS

run_name="tadpole_sEBM_test"
path = "/home/rtandon32/ebm/ebm_experiments/experiment_scripts/real_data/dfMri_D12_ebm_final_n327.csv"
save_dir = os.path.join(output_path, run_name)
df = pd.read_csv(path)

k=119
X = df.iloc[:,:k].values
bm_names = df.columns[:k].tolist()
y = df["DX"].map({"Dementia":1, "CN":0})

mm_fit = mixture_model.fit_all_gmm_models
mixture_models = mm_fit(X, y)
# mcmc params
clust_sizes = [59,12,12,12,12,12]
# clust_sizes = [39,16,16,16,16,16]
# clust_sizes = [59] + [10]*6
assert X.shape[1] == sum(clust_sizes)
mcmc_samples, ll, fig0 = mcmc.mcmc(X, mixture_models, clust_sizes, 
				n_iter=1000000, 
				greedy_n_init=100,
				greedy_n_iter=500, 
				fs=True,
				keep_last_iters=100000)

pdb.set_trace()
