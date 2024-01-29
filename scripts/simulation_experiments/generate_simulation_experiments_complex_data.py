import numpy as np
import os
import json
import sys
import itertools
import pdb
# Read already created data from sEBM synthetic experiments

root_path = "/home/rtandon32/ebm/ebm_experiments/output/jumbo_panel_fs.5/"
seeds = list(range(6))
random_state=0

# Mixture simulation parameters
rep = 6
n_mixtures = [2,3,4]
mixture_p = {3:[0.6, 0.3, 0.1], 2:[0.6,0.4], 4:[0.4,0.3,0.2,0.1]}
output_root = "/home/rtandon32/ebm/s-SuStain-outputs/simulation_experiments/"


def get_read_path(n_sample, dim, s, root_path=root_path):
    path = os.path.join(root_path, "{}_sample".format(n_sample), "{}_dim".format(dim), "5_nclust","{}_seed".format(s), "modified", "hp.json")
    return path

def read_json(file_path):
    with open(file_path) as f:
        hp = json.load(f)
    return hp

def extract_fields(hp):
    X = np.array(hp["X"])
    y = np.array(hp["y"])
    gt_order = np.array(hp["gt_order"])
    return X, y, gt_order

def create_mixture_data(data_dict, seed_combination, mixture_p, random_state=random_state):
    mixture_data = {}
    X_data = []
    y_data = []
    seed_idx = []
    gt_dict = {}
    for e, s in enumerate(seed_combination):
        X, y, gt_order = data_dict[s]
        f = mixture_p[e]
        n_sample = X.shape[0]
        np.random.seed(random_state)
        sample_idx = np.random.choice(n_sample, int(n_sample*f), replace=False )
        sample_idx.sort()
        X_data.append(X[sample_idx])
        y_data.append(y[sample_idx])
        seed_idx.append([s]*int(n_sample*f))
        gt_dict[int(s)] = gt_order.tolist()
    seed_idx = np.hstack(seed_idx).tolist()
    X_data = np.vstack(X_data).tolist()
    y_data = np.hstack(y_data).tolist()
    mixture_data["seed_combination"] = seed_combination.tolist()
    mixture_data["mixture_p"] = mixture_p
    mixture_data["X_data"] = X_data
    mixture_data["y_data"] = y_data
    mixture_data["subtype_from_seed"] = seed_idx
    mixture_data["gt_dict"] = gt_dict
    assert len(gt_dict) == seed_combination.shape[0]
    return mixture_data

def get_mixture_seeds(n_mixtures, seeds, rep, random_state=random_state):
    combinations = list(itertools.combinations(seeds, n_mixtures))
    combinations = np.array(combinations)
    np.random.seed(random_state)
    mixture_seeds_idx = np.random.choice(combinations.shape[0], rep, replace=False)
    mixture_seeds_idx.sort()
    mixture_seeds = combinations[mixture_seeds_idx]
    return mixture_seeds

samples = [200, 400, 600]
samples = np.array(samples).astype(int)
dims = [50, 100, 150, 200, 250, 300]
dims = np.array(dims).astype(int)


def get_data_dict(n_sample, dim, seeds):
    data_dict = {}
    for s in seeds:
        path = get_read_path(n_sample, dim, s)
        hp = read_json(path)
        X, y, gt_order = extract_fields(hp)
        assert max(gt_order) == dim-1
        assert X.shape[0] == y.shape[0] == n_sample
        assert X.shape[1] == gt_order.shape[0] == dim
        data_dict[s] = (X, y, gt_order)
    assert len(data_dict) == len(seeds)
    return data_dict

for n_sample in samples:
    for dim in dims:
        data_dict = get_data_dict(n_sample, dim, seeds)
        for nmc in n_mixtures:
            mixture_seeds = get_mixture_seeds(nmc, seeds, rep)
            for idx, mc in enumerate(mixture_seeds):
                mixture_data = create_mixture_data(data_dict, mc, mixture_p[nmc])
                write_dir = os.path.join(output_root, "complex", "input_data", "{}_samples".format(n_sample), "{}_dims".format(dim), "{}_components".format(nmc))
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)
                file_path = os.path.join(write_dir, "params_rep_{}.json".format(idx))
                with open(file_path, "w") as f:
                    json.dump(mixture_data, f)
        del data_dict
