import os
import pickle
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import itertools
from math import comb
import seaborn as sns

samples = [200, 400, 600]
dims = [50, 100, 150, 200, 250, 300]
seeds = list(range(6))
n_components = [2, 3, 4]
sim_type = ["sEBM", "classic"]
read_dir = "/home/rtandon32/ebm/s-SuStain-outputs/simulation_experiments/complex/input_data"

def pair_scores(seq1, seq2, shape, penalty=0.5):
    assert set(seq1) == set(seq2)
    N_b = seq1.shape[0]
    seq1_pos = np.zeros(N_b)
    seq2_pos = np.zeros(N_b)
    seq1_pos[seq1] = [i for i, j in enumerate(shape) for _ in range(j)]
    seq2_pos[seq2] = [i for i, j in enumerate(shape) for _ in range(j)]
    pairs = list(itertools.combinations(seq1,2))
    kendall_tau = 0
    for i, j in pairs:
        rank_1i = seq1_pos[i]
        rank_1j = seq1_pos[j]
        rank_2i = seq2_pos[i]
        rank_2j = seq2_pos[j]
        sign1 = np.sign(rank_1j - rank_1i)
        sign2 = np.sign(rank_2j - rank_2i)
        sign_product = sign1*sign2
        sign_sum = sign1+sign2
        if sign_product != 0:
            if sign_product > 0:
                k = 0
            else:
                k = 1
        else:
            if sign_sum == 0:
                k = 0
            else:
                k = penalty
        # print((i, j), sign_product, sign_sum, k)
        kendall_tau += k
    return kendall_tau

def _flatten_S_dict(S_dict):
    # S_dict is dictionary, NOT a list of dictionaries
    flatten_S = []
    stages = len(S_dict)
    for k in range(stages):
        flatten_S.append(S_dict[k])
    return np.hstack(flatten_S)

def get_ground_truth_path(n_sample, dim, nc, rep, root_path=read_dir):
    read_path = os.path.join(root_path, "{}_samples".format(n_sample), "{}_dims".format(dim),
                              "{}_components".format(nc), "params_rep_{}.json".format(rep))
    return read_path

def read_json(file_path):
    with open(file_path) as f:
        hp = json.load(f)
    return hp

def extract_fields(hp):
    gt_order = hp["gt_dict"]
    mixture_fractions = np.array(hp["mixture_p"])
    subtype = np.array(hp["subtype_from_seed"])
    return gt_order, mixture_fractions, subtype


# Functions for the sustain results
sustain_output_root = "/home/rtandon32/ebm/s-SuStain-outputs/simulation_experiments/complex/output/"


def get_sustain_results_path(n_samples, n_dims, nc, rep, SuStaIn_type, root_path=sustain_output_root):
    path = os.path.join(root_path, 
                 "{}_samples".format(n_samples),
                 "{}_dims".format(n_dims),
                 "{}_components".format(nc),
                 "SuStaIn_{}".format(SuStaIn_type),
                 "simulation_rep_{}".format(rep),
                 "pickle_files",
                 "simulation_rep_{}_subtype{}.pickle".format(rep, nc-1))
    return path

def read_sustain_results(path):
    with open(path, "rb") as input_file:
        pkl = pickle.load(input_file)
        return pkl

def get_output_seq(pkl_obj, sustain_type):
    seq = pkl_obj["ml_sequence_EM"]
    fraction = pkl_obj["ml_f_EM"]
    shape = None
    idx = fraction.argsort()[::-1]
    if sustain_type == "sEBM":
        shape = pkl_obj["shape_seq"]
        seq = np.array([_flatten_S_dict(s) for s in seq])
        shape = shape[idx]
    seq = seq.astype(int)
    return seq[idx], fraction[idx], shape

def get_output_seq(pkl_obj, sustain_type):
    seq = pkl_obj["ml_sequence_EM"]
    fraction = pkl_obj["ml_f_EM"]
    shape = None
    idx = fraction.argsort()[::-1]
    if sustain_type == "sEBM":
        shape = pkl_obj["shape_seq"]
        shape = shape[idx]
        seq = np.array([_flatten_S_dict(s) for s in seq])
    seq = seq.astype(int)
    return seq[idx], fraction[idx], shape




def get_metrics(n_sample, n_dim, n_comp, rep):

    path_gt = get_ground_truth_path(n_sample, n_dim, n_comp, rep)
    hp_gt = read_json(path_gt)
    gt_order, gt_fractions, gt_subtype = extract_fields(hp_gt)
    path_classic = get_sustain_results_path(n_sample, n_dim, n_comp, rep, "classic")
    path_sEBM = get_sustain_results_path(n_sample, n_dim, n_comp, rep, "sEBM")
    pkl_classic = read_sustain_results(path_classic)
    pkl_sEBM = read_sustain_results(path_sEBM)
    sEBM_sim = get_output_seq(pkl_sEBM, "sEBM")
    classic_sim = get_output_seq(pkl_classic, "classic")
    keys = np.array(list(gt_order.keys()))
    keys.sort()
    max_dist = comb(n_dim, 2)
    kt_dist = []
    for nc_idx in range(n_comp):
        ps_sebm = pair_scores(sEBM_sim[0][nc_idx], gt_order[str(keys[nc_idx])], sEBM_sim[2][nc_idx]) / max_dist
        ps_classic = pair_scores(classic_sim[0][nc_idx], gt_order[str(keys[nc_idx])], sEBM_sim[2][nc_idx]) / max_dist
        ps_classic2 = pair_scores(classic_sim[0][nc_idx], gt_order[str(keys[nc_idx])], [1]*n_dim) / max_dist
        kt_dist.append((ps_sebm, ps_classic, ps_classic2))
    kt_dist = np.array(kt_dist)
    assert n_comp == gt_fractions.shape[0]
    weighted_kt = kt_dist * gt_fractions.reshape(-1,1)
    weighted_kt = weighted_kt.sum(axis=0)
    cross_entropy_sEBM = np.dot(gt_fractions, -np.log(sEBM_sim[1]))
    cross_entropy_classic = np.dot(gt_fractions, -np.log(classic_sim[1]))
    KL_sEBM = np.dot(gt_fractions, np.log(gt_fractions/sEBM_sim[1]))
    KL_classic = np.dot(gt_fractions, np.log(gt_fractions/classic_sim[1]))
    metrics = np.hstack([weighted_kt, [cross_entropy_sEBM, cross_entropy_classic, KL_sEBM, KL_classic]])
    return metrics

n_samples = [200]
n_dims = [200]
n_comps = [2, 3, 4]
reps = list(range(6))

exp_results = []
for n_s in n_samples:
    for n_d in n_dims:
        for n_c in n_comps:
            for r in reps:
                obs = [n_s, n_d, n_c, r]
                metrics = get_metrics(n_s, n_d, n_c, r)
                row = np.hstack([obs, metrics])
                exp_results.append(row)

cols = ["N_s", "N_dims", "N_comps", "seed", "KT_sEBM", "KT_classic", "KT_classic2", "entropy_sEBM", "entropy_classic", "KL_sEBM", "KL_classic"]
exp_results_df = pd.DataFrame(np.vstack(exp_results), columns=cols)
df_kt = pd.melt(exp_results_df[["N_s", "N_dims", "N_comps", "seed", "KT_sEBM", "KT_classic"]],
                    value_vars=["KT_sEBM", "KT_classic"],
                    id_vars=["N_s", "N_dims", "N_comps", "seed"],
                    var_name = "SuStaIn type",
                    value_name = "KT")

df_entropy = pd.melt(exp_results_df[["N_s", "N_dims", "N_comps", "seed", "entropy_sEBM", "entropy_classic"]],
                    value_vars=["entropy_sEBM", "entropy_classic"],
                    id_vars=["N_s", "N_dims", "N_comps", "seed"],
                    var_name = "SuStaIn type",
                    value_name = "cross-entropy")

df_KL = pd.melt(exp_results_df[["N_s", "N_dims", "N_comps", "seed", "KL_sEBM", "KL_classic"]],
                    value_vars=["KL_sEBM", "KL_classic"],
                    id_vars=["N_s", "N_dims", "N_comps", "seed"],
                    var_name = "SuStaIn type",
                    value_name = "KL")

# fig, ax = plt.subplot(figsize=(5,5))
barplot_kt = sns.barplot(df_kt, x="N_comps", y="KT", hue="SuStaIn type", errorbar="sd")
fig1 = barplot_kt.get_figure()
fig1.savefig("/home/rtandon32/ebm/s-SuStain-outputs/data_dump/figures/KT.png", dpi=300, transparent=True)
plt.close()

barplot_H = sns.barplot(df_entropy, x="N_comps", y="cross-entropy", hue="SuStaIn type", errorbar="sd")
fig2 = barplot_H.get_figure()
fig2.savefig("/home/rtandon32/ebm/s-SuStain-outputs/data_dump/figures/H.png", dpi=300, transparent=True)
plt.close()

barplot_H = sns.barplot(df_KL, x="N_comps", y="KL", hue="SuStaIn type", errorbar="sd")
fig3 = barplot_H.get_figure()
ax = fig3.get_axes()
ax[0].set_ylim([0,0.1])
fig3.savefig("/home/rtandon32/ebm/s-SuStain-outputs/data_dump/figures/KL.png", dpi=300, transparent=True)
plt.close()