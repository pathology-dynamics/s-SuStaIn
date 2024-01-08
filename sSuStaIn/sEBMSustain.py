###
# pySuStaIn: a Python implementation of the Subtype and Stage Inference (SuStaIn) algorithm
#
# If you use pySuStaIn, please cite the following core papers:
# 1. The original SuStaIn paper:    https://doi.org/10.1038/s41467-018-05892-0
# 2. The pySuStaIn software paper:  https://doi.org/10.1016/j.softx.2021.100811
#
# Please also cite the corresponding progression pattern model you use:
# 1. The piece-wise linear z-score model (i.e. ZscoreSustain):  https://doi.org/10.1038/s41467-018-05892-0
# 2. The event-based model (i.e. MixtureSustain):               https://doi.org/10.1016/j.neuroimage.2012.01.062
#    with Gaussian mixture modeling (i.e. 'mixture_gmm'):       https://doi.org/10.1093/brain/awu176
#    or kernel density estimation (i.e. 'mixture_kde'):         https://doi.org/10.1002/alz.12083
# 3. The model for discrete ordinal data (i.e. OrdinalSustain): https://doi.org/10.3389/frai.2021.613261
#
# Thanks a lot for supporting this project.
#
# Authors:      Peter Wijeratne (p.wijeratne@ucl.ac.uk) and Leon Aksman (leon.aksman@loni.usc.edu)
# Contributors: Arman Eshaghi (a.eshaghi@ucl.ac.uk), Alex Young (alexandra.young@kcl.ac.uk), Cameron Shand (c.shand@ucl.ac.uk)
###
import pdb
import os
from pathlib import Path
import pickle
import csv
import os
import multiprocessing
from functools import partial, partialmethod

import time
import pathos
import warnings
from tqdm.auto import tqdm
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from scipy.special import logsumexp

from sSuStaIn.AbstractSustain import AbstractSustainData
from sSuStaIn.AbstractSustain import AbstractSustain

#*******************************************
#The data structure class for MixtureSustain. It holds the positive/negative likelihoods that get passed around and re-indexed in places.
class sEBMSustainData(AbstractSustainData):

    def __init__(self, L_yes, L_no, n_stages):

        assert(L_yes.shape[0] == L_no.shape[0] and L_yes.shape[1] == L_no.shape[1])

        self.L_yes          = L_yes
        self.L_no           = L_no
        self.n_stages       = n_stages
        self.L_yes_log      = np.log(L_yes)
        self.L_no_log       = np.log(L_no)

    def getNumSamples(self):
        return self.L_yes.shape[0]

    def getNumBiomarkers(self):
        return self.L_no.shape[1]

    def getNumStages(self):
        return self.n_stages

    def reindex(self, index):
        return sEBMSustainData(self.L_yes[index,], self.L_no[index,], self.n_stages)

#*******************************************
#An implementation of the AbstractSustain class with mixture model based events
class sEBMSustain(AbstractSustain):

    def __init__(self,
                 L_yes,
                 L_no,
                 n_stages, 
                 stage_size_init, 
                 min_stage_size,
                 p_absorb,
                 biomarker_labels,
                 N_startpoints,
                 N_S_max,
                 N_iterations_MCMC,
                 output_folder,
                 dataset_name,
                 use_parallel_startpoints,
                 seed=None):
        # The initializer for the mixture model based events implementation of AbstractSustain
        # Parameters:
        #   L_yes                       - probability of positive class for all subjects across all biomarkers (from mixture modelling)
        #                                 dim: number of subjects x number of biomarkers
        #   L_no                        - probability of negative class for all subjects across all biomarkers (from mixture modelling)
        #                                 dim: number of subjects x number of biomarkers
        #   biomarker_labels            - the names of the biomarkers as a list of strings
        #   N_startpoints               - number of startpoints to use in maximum likelihood step of SuStaIn, typically 25
        #   N_S_max                     - maximum number of subtypes, should be 1 or more
        #   N_iterations_MCMC           - number of MCMC iterations, typically 1e5 or 1e6 but can be lower for debugging
        #   output_folder               - where to save pickle files, etc.
        #   dataset_name                - for naming pickle files
        #   use_parallel_startpoints    - boolean for whether or not to parallelize the maximum likelihood loop
        #   seed                        - random number seed

        N                               =  L_yes.shape[1] # number of biomarkers
        assert (len(biomarker_labels) == N), "number of labels should match number of biomarkers"

        self.biomarker_labels           = biomarker_labels
        self.n_stages                   = n_stages
        self.__sustainData              = sEBMSustainData(L_yes, L_no, self.n_stages)
        self.stage_size_init            = stage_size_init
        self.min_stage_size             = min_stage_size
        self.p_absorb                   = p_absorb
        assert self.n_stages == len(stage_size_init), "number of stages should match with the number of elements in stage_size_init"
        assert min(self.stage_size_init) >= self.min_stage_size, "no stage should have fewer biomarkers than what are required by min_stage_size"
        assert self.p_absorb < 1 and self.p_absorb >= 0, "the probability should be less than 1, but can include 0"


        super().__init__(self.__sustainData,
                         N_startpoints,
                         N_S_max,
                         N_iterations_MCMC,
                         output_folder,
                         dataset_name,
                         use_parallel_startpoints,
                         seed)

    def _initialise_sequence(self, sustainData, rng):
        # Randomly initialises a sequence

        S = rng.permutation(sustainData.getNumBiomarkers()).astype(int)
        S_init = [self._dictionarize_sequence(S, self.stage_size_init)]
        return S_init
    
    def _get_shape(self, S_dict):
        assert type(S_dict) == dict
        N_stages = len(S_dict)
        assert N_stages == self.n_stages, "Number of stages should remain the same"
        shape = [len(S_dict[_]) for _ in range(N_stages)]
        assert min(shape) >= self.min_stage_size, "Each stage should have biomarkers greater than or equal to the minimum size"
        return shape
    
    # def _dictionarize_sequence(self, S):
    #     # S is array
    #     stage_size = self.stage_size
    #     stages_cumsum = np.cumsum(stage_size)
    #     S_dict = {}

    #     for _ in range(len(stage_size),0,-1):
    #         # if _ == 0:
    #         #     idx = (0, stages_cumsum[_])
    #         # else:
    #         idx = (stages_cumsum[_-1], stages_cumsum[_])
    #         stage = S[idx[0]: idx[1]]
    #         S_dict[_] = stage
    #     return S_dict

    def _dictionarize_sequence(self, S, stage_size):
        # stage_size = self.stage_size
        stages_cumsum = np.cumsum(stage_size, dtype=int)
        S_dict = {}

        for _ in range(len(stage_size)-1,0,-1):
            idx = (stages_cumsum[_-1], stages_cumsum[_])
            stage = S[idx[0]: idx[1]]
            S_dict[_] = stage
        S_dict[0] = S[0:idx[0]]
        S_dict_ = {i:S_dict[i] for i in range(len(stage_size))}
        return S_dict_
            
    def _flatten_sequence(self, S):
        flatten_S = np.vstack([self._flatten_S_dict(s) for s in S])
        return flatten_S


    def _flatten_S_dict(self, S_dict):
        # S_dict is dictionary, NOT a list of dictionaries
        flatten_S = []
        stages = len(S_dict)
        for k in range(stages):
            flatten_S.append(S_dict[k])
        return np.hstack(flatten_S)

    def _calculate_likelihood_stage(self, sustainData, S, stage_size):
        '''
        S - Should be a dictionary
        Computes the likelihood of a single event based model
        stage_size - gives the shape of S (number of biomarkers in each cluster)

        Inputs:
        =======
        sustainData - a MixtureData type that contains:
            L_yes - likelihood an event has occurred in each subject
                    dim: number of subjects x number of biomarkers
            L_no -  likelihood an event has not occurred in each subject
                    dim: number of subjects x number of biomarkers
            S -     the current (dict) ordering for a particular subtype
                    dim: 1 x number of events
        Outputs:
        ========
         p_perm_k - the probability of each subjects data at each stage of a particular subtype
         in the SuStaIn model
        '''

        M = sustainData.getNumSamples()
        N = sustainData.getNumStages()
        N_b = sustainData.getNumBiomarkers()
        ss = self._get_shape(S)
        assert ss == stage_size, "passed stage shape should correspond to the dictionary shape"
        S = self._flatten_S_dict(S) # Flatten the dictionary form of S
        assert len(ss) == N, "the number of biomarker clusters should match the number of stages"
        assert sum(ss) == N_b, "sum of cluster sizes should be equal to total number of biomarkers"
        sample_idx = np.cumsum(ss[:-1])
        S_int = S.astype(int)
        arange_Np1 = np.arange(0, N+1) # redundant (leaving due to legacy)
        p_perm_k_log = np.zeros((M, N+1))

        #**** THIS VERSION IS ROUGHLY 10x FASTER THAN THE ONE BELOW
        cp_yes = np.cumsum(sustainData.L_yes_log[:, S_int], 1)
        cp_no = np.cumsum(sustainData.L_no_log[:,  S_int[::-1]],  1)   #do the cumulative product from the end of the sequence

        # Even faster version to avoid loops
        p_perm_k_log[:, 0] = cp_no[:, -1]
        p_perm_k_log[:, -1] = cp_yes[:, -1]
        p_perm_k_log[:, 1:-1] =  cp_yes[:, :-1][:,sample_idx - 1] + cp_no[:, :-1][:,int(N_b) - sample_idx - 1]

        p_perm_k_log += np.log(1 / (N + 1))

        return p_perm_k_log


    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        # Optimise the parameters of the SuStaIn model

        M                                   = sustainData.getNumSamples()
        N_S                                 = len(S_init)
        N                                   = sustainData.getNumStages()
        N_b                                 = sustainData.getNumBiomarkers()
        # ss                                  = self.stage_size_init

        S_opt                               = S_init.copy()  # have to copy or changes will be passed to S_init
        f_opt                               = np.array(f_init).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        p_perm_k_log                        = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            shape_S = self._get_shape(S_opt[s])
            p_perm_k_log[:, :, s]               = self._calculate_likelihood_stage(sustainData, S_opt[s], shape_S)

        p_perm_k_weighted                   = p_perm_k * f_val_mat
        # the second summation axis is different to Matlab version
        #p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))
        # adding 1e-250 fixes divide by zero problem that happens rarely
        p_perm_k_norm                       = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)

        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        order_seq                           = rng.permutation(N_S)    #np.random.permutation(N_S)  # this will produce different random numbers to Matlab
        rep = 5

        for s in order_seq:
            order_bio                       = rng.permutation(N_b) #np.random.permutation(N)  # this will produce different random numbers to Matlab
            for i in order_bio:
                current_sequence            = S_opt[s]
                current_shape               = self._get_shape(current_sequence)
                current_sequence_flatten = self._flatten_S_dict(current_sequence)
                assert(current_sequence_flatten.shape[0]==N_b)
                current_location            = np.array([0] * N_b)
                # print("CURRENT SEQ", current_sequence)
                current_location[current_sequence_flatten.astype(int)] = [loc_i for loc_i, size in enumerate(current_shape) for _ in range(size)]

                possible_positions          = np.arange(N)
                possible_sequences          = np.zeros((len(possible_positions), N_b, rep))
                possible_likelihood         = np.zeros((len(possible_positions), rep))
                possible_shapes             = np.zeros((N, self.n_stages, rep))
                possible_p_perm_k           = np.zeros((M, N + 1, len(possible_positions), rep))
                for index in range(len(possible_positions)):
                    for r in range(rep):
                        selected_event = i
                        move_event_from = current_location[selected_event]
                        new_sequence = S_opt[s].copy()
                        stage_shape = current_shape.copy()

                        #choose a position in the sequence to move an event to
                        move_event_to           = possible_positions[index]

                        if move_event_from > move_event_to:
                            step = 1
                        elif move_event_from < move_event_to:
                            step = -1
                        else:
                            step = 0
                        
                        if step != 0:
                            # print("ns1", new_sequence)
                            if new_sequence[move_event_from].shape[0] > self.min_stage_size:
                                expand_stage = rng.binomial(1, self.p_absorb)
                            else:
                                expand_stage = 0

                            new_sequence[move_event_from] = np.delete(new_sequence[move_event_from], 
                                                                    np.where(new_sequence[move_event_from] == selected_event))
                            

                            if not expand_stage:
                                for _ in range(move_event_to, move_event_from, step):
                                    start_cluster = new_sequence[_]
                                    rng.shuffle(start_cluster)
                                    shift_event = start_cluster[0]
                                    new_sequence[_] = np.delete(np.append(start_cluster, selected_event), 0)
                                    selected_event = shift_event
                                new_sequence[_+step] = np.append(new_sequence[_+step], selected_event)
                            else:
                                new_sequence[move_event_to] = np.append(new_sequence[move_event_to], selected_event)
                                stage_shape[move_event_from] -= 1
                                stage_shape[move_event_to] += 1
                        possible_shapes[index,:,r] = stage_shape
                        ns_flatten = self._flatten_S_dict(new_sequence)
                        possible_sequences[index,:,r] = ns_flatten
                        possible_p_perm_k[:,:,index,r] = self._calculate_likelihood_stage(sustainData, new_sequence, stage_shape)
                        p_perm_k[:,:,s] = possible_p_perm_k[:, :, index, r]
                        total_prob_stage        = np.sum(p_perm_k * f_val_mat, 2)
                        total_prob_subj         = np.sum(total_prob_stage, 1)
                        possible_likelihood[index, r] = np.sum(np.log(total_prob_subj + 1e-250))
                
                idx_max, r_max = np.unravel_index(np.argmax(possible_likelihood, axis=None), possible_likelihood.shape)
                max_likelihood = possible_likelihood[idx_max, r_max]
                this_S = possible_sequences[idx_max, :, r_max].astype(int)
                shape_S = possible_shapes[idx_max, :, r_max]
                S_opt[s] = self._dictionarize_sequence(this_S, shape_S)
                p_perm_k[:,:,s] = possible_p_perm_k[:,:,idx_max, r_max]
            
            S_opt[s] = self._dictionarize_sequence(this_S, shape_S)

        p_perm_k_weighted                   = p_perm_k * f_val_mat
        p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))  # the second summation axis is different to Matlab version
        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)

        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

        f_opt                               = f_opt.reshape(N_S)
        total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
        total_prob_subj                     = np.sum(total_prob_stage, 1)

        likelihood_opt                      = np.sum(np.log(total_prob_subj + 1e-250))

        return S_opt, f_opt, likelihood_opt

    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma):
        # Take MCMC samples of the uncertainty in the SuStaIn model parameters

        M                                   = sustainData.getNumSamples()
        N                                   = sustainData.getNumStages()
        N_b                                 = sustainData.getNumBiomarkers()
        N_S                                 = len(seq_init)
        shape_S                             = np.vstack([self._get_shape(s) for s in seq_init])
        # ss = self.stage_size
        ss_cumsum = np.cumsum(shape_S, axis=1)
        stage_idx = []
        for _ in ss_cumsum:
            stage_idx_ = {}
            init_idx = 0
            for stage, size in enumerate(_):
                stage_idx_[stage] = np.arange(init_idx, size)
                init_idx = size
            stage_idx.append(stage_idx_)


        seq_init_flatten                    = self._flatten_sequence(seq_init) 
        if isinstance(f_sigma, float):  # FIXME: hack to enable multiplication
            f_sigma                         = np.array([f_sigma])

        samples_sequence                    = np.zeros((N_S, N_b, n_iterations))
        samples_f                           = np.zeros((N_S, n_iterations))
        samples_likelihood                  = np.zeros((n_iterations, 1))
        samples_sequence[:, :, 0]           = seq_init_flatten  # don't need to copy as we don't write to 0 index
        samples_f[:, 0]                     = f_init
        sample_prob                         = shape_S / shape_S.sum(axis=1).reshape(-1,1)

        # Reduce frequency of tqdm update to 0.1% of total for larger iteration numbers
        tqdm_update_iters = int(n_iterations/1000) if n_iterations > 100000 else None 

        for i in tqdm(range(n_iterations), "MCMC Iteration", n_iterations, miniters=tqdm_update_iters):
            if i > 0:
                # seq_order                   = self.global_rng.permutation(N_S)
                # this function returns different random numbers to Matlab

                # Abstract out seq_order loop
                # move_event_from_stage = np.random.choice(np.arange(N).astype(int), N_S, p=sample_prob, replace=True)
                move_event_from_stage = np.array([np.random.choice(np.arange(N).astype(int), 1, p=s)[0] for s in sample_prob])
                move_event_from_idx = np.array([np.random.choice(stage_idx[i][j], 1)[0] for i, j in enumerate(move_event_from_stage)])

                # move_event_from = np.ceil(N * self.global_rng.random(N_S)).astype(int) - 1
                current_sequence = samples_sequence[:, :, i - 1]

                selected_event = current_sequence[np.arange(N_S), move_event_from_idx]

                # possible_positions = np.arange(N) + np.zeros((len(seq_order),1))
                bm_pos = np.zeros((N_S, N_b))
                for s in range(N_S):
                    bm_pos[s][current_sequence[s].astype(int)] = [loc_i for loc_i, size in enumerate(shape_S[s]) for _ in range(size)]

                distance = bm_pos - move_event_from_stage[:, np.newaxis]

                weight = AbstractSustain.calc_coeff(seq_sigma) * AbstractSustain.calc_exp(distance, 0., seq_sigma)
                weight = np.divide(weight, weight.sum(1)[:, None])

                move_event_to_idx = [self.global_rng.choice(np.arange(N_b), 1, replace=True, p=row)[0] for row in weight]
                # move_event_to_idx = [np.random.choice(stage_idx[i][j], 1)[0] for i, j in enumerate(index)]

                # move_event_to = np.arange(N)[index]

                

                r = current_sequence.shape[0]
                # Don't need to copy, but doing it for clarity
                new_seq = current_sequence.copy()
                new_seq[np.arange(r), move_event_from_idx] = new_seq[np.arange(r), move_event_to_idx]
                new_seq[np.arange(r), move_event_to_idx] = selected_event

                samples_sequence[:, :, i] = new_seq

                new_f                       = samples_f[:, i - 1] + f_sigma * self.global_rng.standard_normal()
                # TEMP: MATLAB comparison
                #new_f                       = samples_f[:, i - 1] + f_sigma * stats.norm.ppf(np.random.rand(1,N_S))

                new_f                       = (np.fabs(new_f) / np.sum(np.fabs(new_f)))
                samples_f[:, i]             = new_f
            S                               = samples_sequence[:, :, i]

            #f                               = samples_f[:, i]
            #likelihood_sample, _, _, _, _   = self._calculate_likelihood(sustainData, S, f)

            p_perm_k                        = np.zeros((M, N+1, N_S))
            for s in range(N_S):
                S_dict = self._dictionarize_sequence(S[s,:], shape_S[s])
                p_perm_k[:,:,s]             = self._calculate_likelihood_stage(sustainData, S_dict, shape_S[s].tolist())


            #NOTE: added extra axes to get np.tile to work the same as Matlab's repmat in this 3D tiling
            f_val_mat                       = np.tile(samples_f[:,i, np.newaxis, np.newaxis], (1, N+1, M))
            f_val_mat                       = np.transpose(f_val_mat, (2, 1, 0))

            total_prob_stage                = np.sum(p_perm_k * f_val_mat, 2)
            total_prob_subj                 = np.sum(total_prob_stage, 1)

            likelihood_sample               = np.sum(np.log(total_prob_subj + 1e-250))

            samples_likelihood[i]           = likelihood_sample

            if i > 0:
                ratio                           = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
                if ratio < self.global_rng.random():
                    samples_likelihood[i]       = samples_likelihood[i - 1]
                    samples_sequence[:, :, i]   = samples_sequence[:, :, i - 1]
                    samples_f[:, i]             = samples_f[:, i - 1]

        # perm_index                          = np.where(samples_likelihood == np.max(samples_likelihood))
        # perm_index                          = perm_index[0][0]
        perm_index                          = np.argmax(samples_likelihood)
        # ml_likelihood                       = np.max(samples_likelihood)
        ml_likelihood                       = samples_likelihood[perm_index]
        ml_sequence                         = samples_sequence[:, :, perm_index]
        ml_f                                = samples_f[:, perm_index]

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

    def _plot_sustain_model(self, *args, **kwargs):
        return sEBMSustain.plot_positional_var(*args, **kwargs)


    # ********************* STATIC METHODS
    @staticmethod
    def plot_positional_var(ml_sequence_EM, samples_sequence, samples_f, n_samples, biomarker_labels=None, ml_f_EM=None, cval=False, subtype_order=None, biomarker_order=None, title_font_size=12, stage_font_size=10, stage_label="Event Position", stage_rot=0, stage_interval=1, label_font_size=10, label_rot=0, cmap="Oranges", biomarker_colours=None, figsize=None, subtype_titles=None, separate_subtypes=False, save_path=None, save_kwargs={}):
        # Get the number of subtypes
        def _get_shape(S_dict):
            assert type(S_dict) == dict
            N_stages = len(S_dict)
            shape = [len(S_dict[_]) for _ in range(N_stages)]
            return shape
        shape_S = np.vstack([_get_shape(_) for _ in ml_sequence_EM])
        print("shape_S", shape_S)
        N_S = samples_sequence.shape[0]
        # Get the number of features/biomarkers
        N_bio = samples_sequence.shape[1]
        # Check that the number of labels given match
        if biomarker_labels is not None:
            assert len(biomarker_labels) == N_bio
        # Set subtype order if not given
        if subtype_order is None:
            # Determine order if info given
            if ml_f_EM is not None:
                subtype_order = np.argsort(ml_f_EM)[::-1]
            # Otherwise determine order from samples_f
            else:
                subtype_order = np.argsort(np.mean(samples_f, 1))[::-1]
        # Warn user of reordering if labels and order given
        if biomarker_labels is not None and biomarker_order is not None:
            warnings.warn(
                "Both labels and an order have been given. The labels will be reordered according to the given order!"
            )
        # Use default order if none given
        if biomarker_order is None:
            biomarker_order = np.arange(N_bio)
        # If no labels given, set dummy defaults
        if biomarker_labels is None:
            biomarker_labels = [f"Biomarker {i}" for i in range(N_bio)]
        # Otherwise reorder according to given order (or not if not given)
        else:
            biomarker_labels = [biomarker_labels[i] for i in biomarker_order]
        # Check number of subtype titles is correct if given
        if subtype_titles is not None:
            assert len(subtype_titles) == N_S
        # Check biomarker label colours
        # If custom biomarker text colours are given
        if biomarker_colours is not None:
            biomarker_colours = AbstractSustain.check_biomarker_colours(
            biomarker_colours, biomarker_labels
        )
        # Default case of all-black colours
        # Unnecessary, but skips a check later
        else:
            biomarker_colours = {i:"black" for i in biomarker_labels}

        # Flag to plot subtypes separately
        print("separate subtypes", separate_subtypes)
        if separate_subtypes:
            nrows, ncols = 1, 1
        else:
            # Determine number of rows and columns (rounded up)
            if N_S == 1:
                nrows, ncols = 1, 1
            elif N_S < 4:
                nrows, ncols = 1, N_S
            elif N_S < 7:
                nrows, ncols = 2, int(np.ceil(N_S / 2))
            else:
                nrows, ncols = 3, int(np.ceil(N_S / 3))
        # Total axes used to loop over
        total_axes = nrows * ncols
        # Create list of single figure object if not separated
        if separate_subtypes:
            subtype_loops = N_S
        else:
            subtype_loops = 1
        # Container for all figure objects
        figs = []
        # Loop over figures (only makes a diff if separate_subtypes=True)
        print("subtype_loops", subtype_loops)
        for i in range(subtype_loops):
            # Create the figure and axis for this subtype loop
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
            figs.append(fig)
            # Loop over each axis
            for j in range(total_axes):
                # Normal functionality (all subtypes on one plot)
                if not separate_subtypes:
                    i = j
                # Handle case of a single array
                if isinstance(axs, np.ndarray):
                    ax = axs.flat[i]
                else:
                    ax = axs
                # Turn off axes from rounding up
                if i not in range(N_S):
                    ax.set_axis_off()
                    continue
                print("III", i)
                this_shape = shape_S[subtype_order[i]]
                shape_cumsum = np.cumsum(this_shape)
                shape_cumsum = np.insert(shape_cumsum,0,0)
                this_samples_sequence = samples_sequence[subtype_order[i],:,:].T
                N = this_samples_sequence.shape[1]

                # Construct confusion matrix (vectorized)
                # We compare `this_samples_sequence` against each position
                # Sum each time it was observed at that point in the sequence
                # And normalize for number of samples/sequences
                confus_matrix = (this_samples_sequence==np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[0]
                confus_matrix_cluster = np.zeros((confus_matrix.shape[0], len(this_shape)))
                for _ in range(shape_cumsum.shape[0] -1):
                    confus_matrix_cluster[:,_] = confus_matrix[:,shape_cumsum[_]:shape_cumsum[_+1]].sum(axis=1)
                print("Confusion Matrix shape", confus_matrix.shape)
                print(confus_matrix)
                print("Confusion Matrix cluster")
                print(confus_matrix_cluster)

                if subtype_titles is not None:
                    title_i = subtype_titles[i]
                else:
                    # Add axis title
                    if cval == False:
                        temp_mean_f = np.mean(samples_f, 1)
                        # Shuffle vals according to subtype_order
                        # This defaults to previous method if custom order not given
                        vals = temp_mean_f[subtype_order]

                        if n_samples != np.inf:
                            title_i = f"Subtype {i+1}\n(f={vals[i]:.2f}, n={np.round(vals[i] * n_samples):n})"
                        else:
                            title_i = f"Subtype {i+1}\n(f={vals[i]:.2f})"
                    else:
                        title_i = f"Subtype {i+1}\ncross-validated"

                # Plot the matrix
                # Manually set vmin/vmax to handle edge cases
                # and ensure consistent colourization across figures 
                # when certainty=1
                ax.imshow(
                    confus_matrix_cluster[biomarker_order, :],
                    interpolation='nearest',
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    aspect=0.25
                )
                # Add the xticks and labels
                stage_ticks = np.arange(0, this_shape.shape[0], stage_interval)
                ax.set_xticks(stage_ticks)
                ax.set_xticklabels(stage_ticks+1, fontsize=stage_font_size, rotation=stage_rot)
                # Add the yticks and labels
                ax.set_yticks(np.arange(N_bio))
                # Add biomarker labels to LHS of every row
                # if (i % ncols) == 0:
                ax.set_yticklabels(biomarker_labels, ha='right', fontsize=label_font_size - 4, rotation=label_rot)
                # Set biomarker label colours
                for tick_label in ax.get_yticklabels():
                    tick_label.set_color(biomarker_colours[tick_label.get_text()])
                # else:
                #     ax.set_yticklabels([])
                # Make the event label slightly bigger than the ticks
                ax.set_xlabel(stage_label, fontsize=stage_font_size+2)
                ax.set_title(title_i, fontsize=title_font_size)
            # Tighten up the figure
            fig.tight_layout()
            # Save if a path is given
            if save_path is not None:
                # Modify path for specific subtype if specified
                # Don't modify save_path!
                if separate_subtypes:
                    save_name = f"{save_path}_subtype{i}"
                else:
                    save_name = f"{save_path}_all-subtypes"
                # Handle file format, avoids issue with . in filenames
                if "format" in save_kwargs:
                    file_format = save_kwargs.pop("format")
                # Default to png
                else:
                    file_format = "png"
                # Save the figure, with additional kwargs
                fig.savefig(
                    f"{save_name}.{file_format}",
                    **save_kwargs
                )
        return figs, axs

    def subtype_and_stage_individuals_newData(self, L_yes_new, L_no_new, samples_sequence, samples_f, N_samples):

        numStages_new                   = L_yes_new.shape[1]    #number of stages == number of biomarkers here

        assert numStages_new == self.__sustainData.getNumStages(), "Number of stages in new data should be same as in training data"

        sustainData_newData             = MixtureSustainData(L_yes_new, L_no_new, numStages_new)

        ml_subtype,         \
        prob_ml_subtype,    \
        ml_stage,           \
        prob_ml_stage,      \
        prob_subtype,       \
        prob_stage,         \
        prob_subtype_stage          = self.subtype_and_stage_individuals(sustainData_newData, samples_sequence, samples_f, N_samples)

        return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage

    @staticmethod
    def linspace_local2(a, b, N, arange_N):
        return a + (b - a) / (N - 1.) * arange_N

    @staticmethod
    def calc_coeff(sig):
        return 1. / np.sqrt(np.pi * 2.0) * sig

    @staticmethod
    def calc_exp(x, mu, sig):
        x = (x - mu) / sig
        return np.exp(-.5 * x * x)

    # ********************* TEST METHODS
    @classmethod
    def test_sustain(cls, n_biomarkers, n_samples, n_subtypes, ground_truth_subtypes, sustain_kwargs, seed=42, mixture_type="mixture_GMM"):
        from pathlib import Path
        # Path to load/save the arrays
        array_path = Path.cwd() / "mixture_arrays.npz"
        # If not present, create and save the mixture arrays
        # NOTE: This will require kde_ebm to be installed, but should not be required by users
        if not Path(array_path).is_file():
            cls.create_mixture_data(n_biomarkers, n_samples, n_subtypes, ground_truth_subtypes, seed, mixture_type, save_path=array_path)
        # Load the saved arrays
        npzfile = np.load("mixture_arrays.npz")
        # Extract the arrays
        L_yes = npzfile['L_yes']
        L_no = npzfile['L_no']
        return cls(
            L_yes, L_no,
            **sustain_kwargs
        )

    # TODO: Refactor as Zscore func
    def generate_random_model(N_biomarkers, N_S):
        S                                   = np.zeros((N_S, N_biomarkers))
        #try 30 times to find a unique sequence for each subtype
        for i in range(30): 
            matched_others                  = False
            for s in range(N_S):
                S[s, :]                     = np.random.permutation(N_biomarkers)
                #compare to all previous sequences
                for i in range(s):
                    if np.all(S[s, :] == S[i, :]):
                        matched_others      = True
            #all subtype sequences are unique, so break
            if not matched_others:
                break
        if matched_others:
            print('WARNING: Iterated 30 times and could not find unique sequences for all subtypes.')
        return S

    @staticmethod
    def generate_data(subtypes, stages, gt_ordering, mixture_style):
        N_biomarkers                        = gt_ordering.shape[1]
        N_subjects                          = len(subtypes)
        #controls are always drawn from N(0, 1) distribution
        mean_controls                       = np.array([0]   * N_biomarkers)
        std_controls                        = np.array([0.25] * N_biomarkers)
        #mean and variance for cases
        #if using mixture_GMM, use normal distribution with mean 1 and std. devs sampled from a range
        if mixture_style == 'mixture_GMM':
            mean_cases                      = np.array([1.5] * N_biomarkers)
            std_cases                       = np.random.uniform(0.25, 0.50, N_biomarkers)
        #if using mixture_KDE, use log normal with mean 0.5 and std devs sampled from a range
        elif mixture_style == 'mixture_KDE':
            mean_cases                      = np.array([0.5] * N_biomarkers)
            std_cases                       = np.random.uniform(0.2, 0.5, N_biomarkers)

        data                                = np.zeros((N_subjects, N_biomarkers))
        data_denoised                       = np.zeros((N_subjects, N_biomarkers))

        stages                              = stages.astype(int)
        #loop over all subjects, creating measurment for each biomarker based on what subtype and stage they're in
        for i in range(N_subjects):
            S_i                             = gt_ordering[subtypes[i], :].astype(int)
            stage_i                         = stages[i].item()

            #fill in with ABNORMAL values up to the subject's stage
            for j in range(stage_i):

                if      mixture_style == 'mixture_KDE':
                    sample_j                = np.random.lognormal(mean_cases[S_i[j]], std_cases[S_i[j]])
                elif    mixture_style == 'mixture_GMM':
                    sample_j                = np.random.normal(mean_cases[S_i[j]], std_cases[S_i[j]])

                data[i, S_i[j]]             = sample_j
                data_denoised[i, S_i[j]]    = mean_cases[S_i[j]]

            # fill in with NORMAL values from the subject's stage+1 to last stage
            for j in range(stage_i, N_biomarkers):
                data[i, S_i[j]]             = np.random.normal(mean_controls[S_i[j]], std_controls[S_i[j]])
                data_denoised[i, S_i[j]]    = mean_controls[S_i[j]]
        return data, data_denoised #, stage_value

    @classmethod
    def create_mixture_data(cls, n_biomarkers, n_samples, n_subtypes, ground_truth_subtypes, seed, mixture_type, save_path):
        # Avoid import outside of testing
        from kde_ebm.mixture_model import fit_all_gmm_models, fit_all_kde_models #from mixture_model import fit_all_gmm_models, fit_all_kde_models
        # Set a global seed to propagate (particularly for mixture_model)
        np.random.seed(seed)

        ground_truth_sequences = cls.generate_random_model(n_biomarkers, n_subtypes)

        N_stages = n_biomarkers

        ground_truth_stages_control = np.zeros((int(np.round(n_samples * 0.25)), 1))
        ground_truth_stages_other = np.random.randint(1, N_stages+1, (int(np.round(n_samples * 0.75)), 1))
        ground_truth_stages = np.vstack(
            (ground_truth_stages_control, ground_truth_stages_other)
        ).astype(int)

        data, data_denoised = cls.generate_data(
            ground_truth_subtypes,
            ground_truth_stages,
            ground_truth_sequences,
            mixture_type
        )
        # choose which subjects will be cases and which will be controls
        MIN_CASE_STAGE = np.round((n_biomarkers + 1) * 0.8)
        index_case = np.where(ground_truth_stages >=  MIN_CASE_STAGE)[0]
        index_control = np.where(ground_truth_stages ==  0)[0]

        labels = 2 * np.ones(data.shape[0], dtype=int) # 2 - intermediate value, not used in mixture model fitting
        labels[index_case] = 1                         # 1 - cases
        labels[index_control] = 0                      # 0 - controls

        data_case_control = data[labels != 2, :]
        labels_case_control = labels[labels != 2]
        if mixture_type == "mixture_GMM":
            mixtures = fit_all_gmm_models(data, labels)
        elif mixture_type == "mixture_KDE":
            mixtures = fit_all_kde_models(data, labels)

        L_yes = np.zeros(data.shape)
        L_no = np.zeros(data.shape)
        for i in range(n_biomarkers):
            if mixture_type == "mixture_GMM":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])
            elif mixture_type == "mixture_KDE":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1))
        # Save the arrays
        np.savez(save_path, L_yes=L_yes, L_no=L_no)

    #********************* PUBLIC METHODS
    def run_sustain_algorithm(self, plot=False, plot_format="png", **kwargs):
        # Externally called method to start the SuStaIn algorithm after initializing the SuStaIn class object properly

        ml_sequence_prev_EM                 = []
        ml_f_prev_EM                        = []

        pickle_dir                          = os.path.join(self.output_folder, 'pickle_files')
        if not os.path.isdir(pickle_dir):
            os.mkdir(pickle_dir)
        if plot:
            fig0, ax0                           = plt.subplots()
        for s in range(self.N_S_max):

            pickle_filename_s               = os.path.join(pickle_dir, self.dataset_name + '_subtype' + str(s) + '.pickle')
            pickle_filepath                 = Path(pickle_filename_s)
            if pickle_filepath.exists():
                print("Found pickle file: " + pickle_filename_s + ". Using pickled variables for " + str(s) + " subtype.")

                pickle_file                 = open(pickle_filename_s, 'rb')

                loaded_variables            = pickle.load(pickle_file)

                #self.stage_zscore           = loaded_variables["stage_zscore"]
                #self.stage_biomarker_index  = loaded_variables["stage_biomarker_index"]
                #self.N_S_max                = loaded_variables["N_S_max"]

                samples_likelihood          = loaded_variables["samples_likelihood"]
                samples_sequence            = loaded_variables["samples_sequence"]
                samples_f                   = loaded_variables["samples_f"]

                ml_sequence_EM              = loaded_variables["ml_sequence_EM"]
                ml_sequence_prev_EM         = loaded_variables["ml_sequence_prev_EM"]
                ml_f_EM                     = loaded_variables["ml_f_EM"]
                ml_f_prev_EM                = loaded_variables["ml_f_prev_EM"]

                pickle_file.close()
            else:
                print("Failed to find pickle file: " + pickle_filename_s + ". Running SuStaIn model for " + str(s) + " subtype.")

                ml_sequence_EM,     \
                ml_f_EM,            \
                ml_likelihood_EM,   \
                ml_sequence_mat_EM, \
                ml_f_mat_EM,        \
                ml_likelihood_mat_EM        = self._estimate_ml_sustain_model_nplus1_clusters(self.__sustainData, ml_sequence_prev_EM, 
                                                                                              ml_f_prev_EM) #self.__estimate_ml_sustain_model_nplus1_clusters(self.__data, ml_sequence_prev_EM, ml_f_prev_EM)

                seq_init                    = ml_sequence_EM
                print("SEQ INIT", seq_init)
                f_init                      = ml_f_EM

                ml_sequence,        \
                ml_f,               \
                ml_likelihood,      \
                samples_sequence,   \
                samples_f,          \
                samples_likelihood          = self._estimate_uncertainty_sustain_model(self.__sustainData, seq_init, f_init)           #self.__estimate_uncertainty_sustain_model(self.__data, seq_init, f_init)
                ml_sequence_prev_EM         = ml_sequence_EM
                ml_f_prev_EM                = ml_f_EM

            # max like subtype and stage / subject
            shape_S = np.vstack([self._get_shape(_) for _ in seq_init])
            N_samples                       = 1000
            ml_subtype,             \
            prob_ml_subtype,        \
            ml_stage,               \
            prob_ml_stage,          \
            prob_subtype,           \
            prob_stage,             \
            prob_subtype_stage               = self.subtype_and_stage_individuals(self.__sustainData, shape_S, samples_sequence, samples_f, N_samples)   #self.subtype_and_stage_individuals(self.__data, samples_sequence, samples_f, N_samples)
            if not pickle_filepath.exists():

                if not os.path.exists(self.output_folder):
                    os.makedirs(self.output_folder)

                save_variables                          = {}
                save_variables["samples_sequence"]      = samples_sequence
                save_variables["samples_f"]             = samples_f
                save_variables["samples_likelihood"]    = samples_likelihood

                save_variables["ml_subtype"]            = ml_subtype
                save_variables["prob_ml_subtype"]       = prob_ml_subtype
                save_variables["ml_stage"]              = ml_stage
                save_variables["prob_ml_stage"]         = prob_ml_stage
                save_variables["prob_subtype"]          = prob_subtype
                save_variables["prob_stage"]            = prob_stage
                save_variables["prob_subtype_stage"]    = prob_subtype_stage

                save_variables["ml_sequence_EM"]        = ml_sequence_EM
                save_variables["ml_sequence_prev_EM"]   = ml_sequence_prev_EM
                save_variables["ml_f_EM"]               = ml_f_EM
                save_variables["ml_f_prev_EM"]          = ml_f_prev_EM

                pickle_file                 = open(pickle_filename_s, 'wb')
                pickle_output               = pickle.dump(save_variables, pickle_file)
                pickle_file.close()

            n_samples                       = self.__sustainData.getNumSamples() #self.__data.shape[0]

            if plot:
                # print("ml_f_EM", ml_f_EM)
                #order of subtypes displayed in positional variance diagrams plotted by _plot_sustain_model
                self._plot_subtype_order        = np.argsort(ml_f_EM)[::-1]
                #order of biomarkers in each subtypes' positional variance diagram
                # print("ml_sequence_EM", ml_sequence_EM)
                # print("_plot_subtype_order", self._plot_subtype_order)
                flatten_S = np.vstack([self._flatten_S_dict(s) for s in ml_sequence_EM])
                self._plot_biomarker_order      = flatten_S[self._plot_subtype_order[0], :].astype(int)

            # plot results
            if plot:
                figs, ax = self._plot_sustain_model(ml_sequence_EM=ml_sequence_EM, 
                    samples_sequence=samples_sequence,
                    samples_f=samples_f,
                    n_samples=n_samples,
                    biomarker_labels=self.biomarker_labels,
                    subtype_order=self._plot_subtype_order,
                    biomarker_order=self._plot_biomarker_order,
                    save_path=Path(self.output_folder) / f"{self.dataset_name}_subtype{s}_PVD.{plot_format}",
                    **kwargs
                )
                for fig in figs:
                    fig.show()

                ax0.plot(range(self.N_iterations_MCMC), samples_likelihood, label="Subtype " + str(s+1))

        # save and show this figure after all subtypes have been calculcated
        if plot:
            ax0.legend(loc='upper right')
            fig0.tight_layout()
            fig0.savefig(Path(self.output_folder) / f"MCMC_likelihoods.{plot_format}", bbox_inches='tight')
            fig0.show()

        return samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage