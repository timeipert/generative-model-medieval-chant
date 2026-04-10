import pymc as pm
from pymc.pytensorf import collect_default_updates
import numpy as np
import pytensor
import pytensor.tensor as pt
from models import base_model


class MarkovModeTransposedModel(base_model.Model):
    def __init__(self, data=None, coords=None):
        super().__init__(data=data, coords=coords)
        self.init_probs_prior = np.ones(self.V)
        self.base_transition_prior = np.ones((self.V, self.V))
        self.num_modes = 8

    @staticmethod
    def _scan_step_fn(current_time, prev_state, transition_matrix, sequence_lengths):
        num_states = transition_matrix.shape[0]
        valid_prev_state = pt.clip(prev_state, 0, num_states - 1)

        current_state = pm.Categorical.dist(p=transition_matrix[valid_prev_state])

        absorbing_state = pm.DiracDelta.dist(-1, shape=current_state.shape)
        current_state = pt.where(current_time >= sequence_lengths, absorbing_state, current_state)

        return current_state, collect_default_updates([current_state])

    @staticmethod
    def _scan_generative_dist(initial_state, transition_matrix, sequence_lengths, size):
        states, _ = pytensor.scan(
            fn=MarkovModeTransposedModel._scan_step_fn,
            sequences=[pt.arange(sequence_lengths.max())],
            outputs_info=[initial_state],
            non_sequences=[transition_matrix, sequence_lengths],
            strict=True,
        )
        return states

    @staticmethod
    def transpose_transition_matrix(base_transition_matrix, mode_shift, scale_func=lambda d: 0.5 ** d):
        num_states = base_transition_matrix.shape[0]
        row_idx, col_idx = pt.ogrid[:num_states, :num_states]

        dist_pos = pt.maximum(mode_shift - row_idx, mode_shift - col_idx).clip(0, num_states)
        mask_pos = (row_idx < mode_shift) | (col_idx < mode_shift)

        threshold_neg = num_states + mode_shift
        dist_neg = pt.maximum(row_idx - (threshold_neg - 1), col_idx - (threshold_neg - 1)).clip(0, num_states)
        mask_neg = (row_idx >= threshold_neg) | (col_idx >= threshold_neg)

        shifted_matrix = pt.roll(base_transition_matrix, mode_shift, axis=0)
        shifted_matrix = pt.roll(shifted_matrix, mode_shift, axis=1)

        distance = pt.switch(mode_shift > 0, dist_pos, dist_neg)
        mask = pt.switch(mode_shift > 0, mask_pos, mask_neg)

        base_scaled = base_transition_matrix * scale_func(distance)
        unnormalized_matrix = pt.switch(mask, base_scaled, shifted_matrix)

        row_sums = pt.sum(unnormalized_matrix, axis=1, keepdims=True)
        return unnormalized_matrix / row_sums

    def _calculate_conditional_sequence_logp(self, prev_states, current_states, valid_time_mask, base_transition_matrix,
                                             mode_indices):
        all_modes_transition_matrices = pt.stack([
            MarkovModeTransposedModel.transpose_transition_matrix(base_transition_matrix,
                                                                  pt.as_tensor(i, dtype="int32"))
            for i in range(self.num_modes)
        ])

        mode_broadcasted = pt.alloc(mode_indices, prev_states.shape[0], mode_indices.shape[0])
        step_log_probs = pt.log(all_modes_transition_matrices[mode_broadcasted, prev_states, current_states])

        return pt.sum(step_log_probs * valid_time_mask)

    def _calculate_marginal_sequence_logp(self, prev_states, current_states, valid_time_mask, base_transition_matrix,
                                          mode_probs):
        all_modes_transition_matrices = pt.stack([
            MarkovModeTransposedModel.transpose_transition_matrix(base_transition_matrix,
                                                                  pt.as_tensor(i, dtype="int32"))
            for i in range(self.num_modes)
        ])

        log_likelihoods = []
        for i in range(self.num_modes):
            step_log_probs = pt.log(all_modes_transition_matrices[i, prev_states, current_states])
            sequence_log_prob = pt.sum(step_log_probs * valid_time_mask, axis=0)
            log_likelihoods.append(pt.log(mode_probs[i]) + sequence_log_prob)

        return pt.sum(pm.math.logsumexp(pt.stack(log_likelihoods), axis=0))

    def _define_priors(self):
        mode_probs = pm.Dirichlet("mode_probs", a=np.ones(self.num_modes))
        base_transition_matrix = pm.Dirichlet("base_transition_matrix", a=self.base_transition_prior)
        init_probs = pm.Dirichlet("init_probs", a=self.init_probs_prior)

        return mode_probs, base_transition_matrix, init_probs

    def _define_inference_likelihood(self, mode_probs, base_transition_matrix, init_probs, initial_states_obs,
                                     sequence_states_obs, sequence_lengths_obs, mode_obs):
        pm.Categorical("initial_states", p=init_probs, observed=initial_states_obs)

        initial_states_padded = pt.shape_padleft(initial_states_obs)
        all_states = pt.concatenate([initial_states_padded, sequence_states_obs], axis=0)

        prev_states = all_states[:-1]
        current_states = all_states[1:]

        time_indices = pt.arange(sequence_states_obs.shape[0])
        valid_time_mask = time_indices[:, None] < sequence_lengths_obs[None, :]

        if mode_obs is not None:
            pm.Categorical("mode", p=mode_probs, observed=mode_obs)
            sequence_log_likelihood = self._calculate_conditional_sequence_logp(
                prev_states,
                current_states,
                valid_time_mask,
                base_transition_matrix,
                mode_obs
            )
            pm.Potential("conditional_sequence_ll", sequence_log_likelihood)
        else:
            marginal_log_likelihood = self._calculate_marginal_sequence_logp(
                prev_states,
                current_states,
                valid_time_mask,
                base_transition_matrix,
                mode_probs
            )
            pm.Potential("marginal_sequence_ll", marginal_log_likelihood)

    def _define_generative_likelihood(self, mode_probs, base_transition_matrix, init_probs, mode_obs=None):
        if mode_obs is not None:
            mode = pm.Categorical("mode", p=mode_probs, observed=mode_obs)
        else:
            mode = pm.Categorical("mode", p=mode_probs)

        active_transition_matrix = pm.Deterministic("active_transition_matrix",
                                                    self.transpose_transition_matrix(base_transition_matrix, mode))
        initial_states = pm.Categorical("initial_states", p=init_probs)
        sequence_lengths = pm.Poisson("sequence_lengths", self.mean_seq_len)

        pm.CustomDist(
            "generated_sequence",
            initial_states,
            active_transition_matrix,
            sequence_lengths,
            dist=MarkovModeTransposedModel._scan_generative_dist,
        )

    def build_model(self, initial_states_obs=None, sequence_states_obs=None, sequence_lengths_obs=None, mode_obs=None,
                    **kwargs):
        coords = getattr(self, "coords", None)

        with pm.Model(coords=coords) as self.model:
            mode_probs, base_transition_matrix, init_probs = self._define_priors()

            if sequence_states_obs is not None:
                self._define_inference_likelihood(
                    mode_probs,
                    base_transition_matrix,
                    init_probs,
                    initial_states_obs,
                    sequence_states_obs,
                    sequence_lengths_obs,
                    mode_obs
                )
            else:
                self._define_generative_likelihood(
                    mode_probs,
                    base_transition_matrix,
                    init_probs,
                    mode_obs
                )

        return self.model