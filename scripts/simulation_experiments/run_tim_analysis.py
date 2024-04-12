import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

sustain_output_root = "/home/rtandon32/ebm/s-SuStain-outputs/simulation_experiments/complex/output_runtime/"

def get_sustain_results_path(n_samples, n_dims, nc, rep, SuStaIn_type, root_path=sustain_output_root):
    path = os.path.join(root_path, 
                 "{}_samples".format(n_samples),
                 "{}_dims".format(n_dims),
                 "{}_components".format(4),
                 "SuStaIn_{}".format(SuStaIn_type),
                 "simulation_rep_{}".format(rep),
                 "pickle_files",
                 "simulation_rep_{}_subtype{}.pickle".format(rep, nc))
    return path

def read_sustain_results(path):
    if os.path.exists(path):
        with open(path, "rb") as input_file:
            pkl = pickle.load(input_file)
    else:
        pkl = None
    return pkl
    
def get_run_times(n_s, n_d, n_c, r, s_t):
    path = get_sustain_results_path(n_s, n_d, n_c, r, s_t)
    pkl = read_sustain_results(path)
    if pkl is not None:
        run_times = pkl["run_times"]
    else:
        run_times = [None, None]
    return run_times
    
n_samples = [200]
n_dims = [50, 100,150, 200]
n_comps = 4
reps = list(range(3))
sustain_types = ["classic", "sEBM"]
exp_results = []
for n_s in n_samples:
    for n_d in n_dims:
        for n_c in range(1,n_comps):
            for r in reps:
                for s_t in sustain_types:
                    obs = [n_s, n_d, n_c, r, s_t]
                    run_times = get_run_times(n_s, n_d, n_c, r, s_t)
                    row = np.hstack([obs, run_times])
                    exp_results.append(row)

cols = ["N_samples", "N_dim", "N_components", "seed", "sustain-type", "opt-time", "mcmc-time"]
exp_results_df = pd.DataFrame(np.vstack(exp_results), columns=cols)
col_diff = exp_results_df.columns.difference(["sustain-type"])
exp_results_df[col_diff] = exp_results_df[col_diff].astype("float")
x = exp_results_df


df_list = []
for s_t in sustain_types:
    for n_d in n_dims:
        for r in reps:
            mini_df = x[(x["sustain-type"].isin([s_t])) & (x["N_dim"].isin([n_d])) & (x["seed"]==r)]
            mini_df.sort_values("N_components", inplace=True)
            mini_df["opt-time"] = mini_df["opt-time"].cumsum()
            df_list.append(mini_df)

y = pd.concat(df_list, axis=0)
y.to_csv("/nethome/rtandon32/ebm/s-SuStain-outputs/data_dump/figures/y.csv", index=False)
gb_mean = y.groupby(["sustain-type", "N_dim", "N_components"]).mean()["opt-time"]
gb_sem = y.groupby(["sustain-type", "N_dim", "N_components"]).sem()["opt-time"]

fig, ax  = plt.subplots(1,4,sharex=True, figsize=(7.3,1.55))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
clr_dict = dict(zip(sustain_types, colors[:2][::-1]))
for idx, n_d in enumerate(n_dims):
    for s_t in sustain_types:
        mean_time = gb_mean[s_t][n_d]
        sem_time = gb_sem[s_t][n_d]
        ax[idx].plot(np.array(mean_time.index.tolist())+1, mean_time, marker="D", ms=6, c=clr_dict[s_t])
        ax[idx].fill_between(np.array(mean_time.index.tolist())+1, mean_time + sem_time, 
                             mean_time - sem_time, alpha=0.3, color=clr_dict[s_t])
        ax[idx].set_title("biomarkers (N) = {}".format(int(n_d)), fontsize=10)
        ax[idx].set_xticks(np.array(mean_time.index.tolist())+1)
        ax[idx].set_yscale("log")
        ax[idx].spines['top'].set_visible(False)
        ax[idx].spines['right'].set_visible(False)

ax[0].set_ylabel("Opt. time (s)", fontsize=11)
# fig.supxlabel("sub-types (T)")

fig.tight_layout()
fig.savefig("/home/rtandon32/ebm/s-SuStain-outputs/data_dump/figures/run_time.png", transparent=True, dpi=300)

fig, ax = plt.subplots(1,1,figsize=(3,3))
assert gb_mean["sEBM"].index.equals(gb_mean["classic"].index)
r = gb_mean["classic"] / gb_mean["sEBM"]
r_df = r.unstack()
markers = {1:"o", 2:"s", 3:"P"}
for _ in range(1,4):
    ax.plot(r_df[_].index.tolist(), r_df[_], 
            label="{} {}".format(_+1, "subtypes"), marker=markers[_])
ax.legend(framealpha=0.3)
ax.set_xlabel("biomarkers (N)")
ax.set_ylabel(r'Speed up factor ($\frac{T_{SuStaIn}}{T_{s-SuStaIn}}$)')
ax.set_xticks(r_df[_].index.tolist())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig("/home/rtandon32/ebm/s-SuStain-outputs/data_dump/figures/run_time_factor2.png", transparent=True, dpi=300)

print(clr_dict)
# print(exp_results_df)