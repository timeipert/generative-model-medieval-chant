import pymc as pm
from pymc.pytensorf import collect_default_updates
import numpy as np
import pytensor
import pytensor.tensor as pt
from models import base_model


class MarkovModeRegionIndependent(base_model.Model):
    def __init__(self, data=None, coords=None):
        super().__init__(data=data, coords=coords)
        self.init_probs_prior = np.ones(self.V)
        self.transition_prior = np.ones((self.V, self.V))
        self.num_modes = 8
        self.num_regions = 3

    @staticmethod
    def _scan_step_fn(current_time, prev_state, transition_matrix, sequence_lengths):
        num_states = transition_matrix.shape[-1]
        valid_prev_state = pt.clip(prev_state, 0, num_states - 1)

        if transition_matrix.ndim == 3:
            batch_idx = pt.arange(valid_prev_state.shape[0])
            p = transition_matrix[batch_idx, valid_prev_state]
        else:
            p = transition_matrix[valid_prev_state]

        current_state = pm.Categorical.dist(p=p)

        absorbing_state = pm.DiracDelta.dist(-1, shape=current_state.shape)
        current_state = pt.where(current_time >= sequence_lengths, absorbing_state, current_state)

        return current_state, collect_default_updates([current_state])

    @staticmethod
    def _scan_generative_dist(initial_state, transition_matrix, sequence_lengths, size):
        states, _ = pytensor.scan(
            fn=MarkovModeRegionIndependent._scan_step_fn,
            sequences=[pt.arange(sequence_lengths.max())],
            outputs_info=[initial_state],
            non_sequences=[transition_matrix, sequence_lengths],
            strict=True,
        )
        return states

    def _calculate_sequence_logp(self, prev_states, current_states, valid_time_mask, transition_matrices, mode_indices,
                                 region_indices):
        mode_broadcasted = pt.alloc(mode_indices, prev_states.shape[0], mode_indices.shape[0])
        region_broadcasted = pt.alloc(region_indices, prev_states.shape[0], region_indices.shape[0])

        step_log_probs = pt.log(transition_matrices[region_broadcasted, mode_broadcasted, prev_states, current_states])

        return pt.sum(step_log_probs * valid_time_mask)

    def _define_priors(self):
        global_mode_probs = pm.Dirichlet("global_mode_probs", a=np.ones(self.num_modes))
        kappa = pm.Exponential("kappa", 1.0)
        mode_probs = pm.Dirichlet("mode_probs", a=global_mode_probs * kappa, shape=(self.num_regions, self.num_modes))

        beta_expanded = np.tile(self.transition_prior, (self.num_regions, self.num_modes, 1, 1))
        region_mode_transition_matrices = pm.Dirichlet("region_mode_transition_matrices", a=beta_expanded)
        init_probs = pm.Dirichlet("init_probs", a=self.init_probs_prior)

        return mode_probs, region_mode_transition_matrices, init_probs

    def _define_inference_likelihood(self, mode_probs, region_mode_transition_matrices, init_probs, initial_states_obs,
                                     sequence_states_obs, sequence_lengths_obs, mode_obs, region_obs):
        pm.Categorical("initial_states", p=init_probs, observed=initial_states_obs)

        initial_states_padded = pt.shape_padleft(initial_states_obs)
        all_states = pt.concatenate([initial_states_padded, sequence_states_obs], axis=0)

        prev_states = all_states[:-1]
        current_states = all_states[1:]

        time_indices = pt.arange(sequence_states_obs.shape[0])
        valid_time_mask = time_indices[:, None] < sequence_lengths_obs[None, :]

        pm.Categorical("mode", p=mode_probs[region_obs], observed=mode_obs)

        sequence_log_likelihood = self._calculate_sequence_logp(
            prev_states,
            current_states,
            valid_time_mask,
            region_mode_transition_matrices,
            mode_obs,
            region_obs
        )
        pm.Potential("conditional_sequence_ll", sequence_log_likelihood)

    def _define_generative_likelihood(self, mode_probs, region_mode_transition_matrices, init_probs, mode_obs=None,
                                      region_obs=None):
        if region_obs is not None:
            region = pt.as_tensor_variable(region_obs)
            batch_shape = region.shape
        else:
            region = pm.Categorical("region", p=np.ones(self.num_regions) / self.num_regions)
            batch_shape = None

        if mode_obs is not None:
            mode = pm.Categorical("mode", p=mode_probs[region], observed=mode_obs)
        else:
            mode = pm.Categorical("mode", p=mode_probs[region], shape=batch_shape)

        active_transition_matrix = pm.Deterministic("active_transition_matrix",
                                                    region_mode_transition_matrices[region, mode])

        initial_states = pm.Categorical("initial_states", p=init_probs, shape=batch_shape)
        sequence_lengths = pm.Poisson("sequence_lengths", self.mean_seq_len, shape=batch_shape)

        pm.CustomDist(
            "generated_sequence",
            initial_states,
            active_transition_matrix,
            sequence_lengths,
            dist=MarkovModeRegionIndependent._scan_generative_dist,
        )

    def build_model(self, initial_states_obs=None, sequence_states_obs=None, sequence_lengths_obs=None, mode_obs=None,
                    region_obs=None, **kwargs):
        coords = getattr(self, "coords", None)

        with pm.Model(coords=coords) as self.model:
            mode_probs, region_mode_transition_matrices, init_probs = self._define_priors()

            if sequence_states_obs is not None:
                self._define_inference_likelihood(
                    mode_probs,
                    region_mode_transition_matrices,
                    init_probs,
                    initial_states_obs,
                    sequence_states_obs,
                    sequence_lengths_obs,
                    mode_obs,
                    region_obs
                )
            else:
                self._define_generative_likelihood(
                    mode_probs,
                    region_mode_transition_matrices,
                    init_probs,
                    mode_obs,
                    region_obs
                )

        return self.model