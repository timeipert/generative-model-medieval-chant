import pymc as pm
from pymc.pytensorf import collect_default_updates
import numpy as np
import pytensor
import pytensor.tensor as pt
from models import base_model


class MarkovModeIndependentModel(base_model.Model):
    def __init__(self, data=None, coords=None):
        super().__init__(data=data, coords=coords)
        self.init_probs_prior = np.ones(self.V)
        self.transition_prior = np.ones((self.V, self.V))
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
            fn=MarkovModeIndependentModel._scan_step_fn,
            sequences=[pt.arange(sequence_lengths.max())],
            outputs_info=[initial_state],
            non_sequences=[transition_matrix, sequence_lengths],
            strict=True,
        )
        return states

    def _calculate_sequence_logp(self, prev_states, current_states, valid_time_mask, mode_transition_matrices,
                                 mode_indices):
        mode_broadcasted = pt.alloc(mode_indices, prev_states.shape[0], mode_indices.shape[0])
        step_log_probs = pt.log(mode_transition_matrices[mode_broadcasted, prev_states, current_states])

        return pt.sum(step_log_probs * valid_time_mask)

    def _define_priors(self):
        mode_probs = pm.Dirichlet("mode_probs", a=np.ones(self.num_modes))

        beta_expanded = np.tile(self.transition_prior, (self.num_modes, 1, 1))
        mode_transition_matrices = pm.Dirichlet("mode_transition_matrices", a=beta_expanded)

        init_probs = pm.Dirichlet("init_probs", a=self.init_probs_prior)

        return mode_probs, mode_transition_matrices, init_probs

    def _define_inference_likelihood(self, mode_probs, mode_transition_matrices, init_probs, initial_states_obs,
                                     sequence_states_obs, sequence_lengths_obs, mode_obs):
        pm.Categorical("initial_states", p=init_probs, observed=initial_states_obs)

        initial_states_padded = pt.shape_padleft(initial_states_obs)
        all_states = pt.concatenate([initial_states_padded, sequence_states_obs], axis=0)

        prev_states = all_states[:-1]
        current_states = all_states[1:]

        time_indices = pt.arange(sequence_states_obs.shape[0])
        valid_time_mask = time_indices[:, None] < sequence_lengths_obs[None, :]

        pm.Categorical("mode", p=mode_probs, observed=mode_obs)

        sequence_log_likelihood = self._calculate_sequence_logp(
            prev_states,
            current_states,
            valid_time_mask,
            mode_transition_matrices,
            mode_obs
        )
        pm.Potential("conditional_sequence_ll", sequence_log_likelihood)

    def _define_generative_likelihood(self, mode_probs, mode_transition_matrices, init_probs, mode_obs=None):
        if mode_obs is not None:
            mode = pm.Categorical("mode", p=mode_probs, observed=mode_obs)
        else:
            mode = pm.Categorical("mode", p=mode_probs)

        active_transition_matrix = pm.Deterministic("active_transition_matrix", mode_transition_matrices[mode])
        initial_states = pm.Categorical("initial_states", p=init_probs)
        sequence_lengths = pm.Poisson("sequence_lengths", self.mean_seq_len)

        pm.CustomDist(
            "generated_sequence",
            initial_states,
            active_transition_matrix,
            sequence_lengths,
            dist=MarkovModeIndependentModel._scan_generative_dist,
        )

    def build_model(self, initial_states_obs=None, sequence_states_obs=None, sequence_lengths_obs=None, mode_obs=None,
                    **kwargs):
        coords = getattr(self, "coords", None)

        with pm.Model(coords=coords) as self.model:
            mode_probs, mode_transition_matrices, init_probs = self._define_priors()

            if sequence_states_obs is not None:
                self._define_inference_likelihood(
                    mode_probs,
                    mode_transition_matrices,
                    init_probs,
                    initial_states_obs,
                    sequence_states_obs,
                    sequence_lengths_obs,
                    mode_obs
                )
            else:
                self._define_generative_likelihood(
                    mode_probs,
                    mode_transition_matrices,
                    init_probs,
                    mode_obs
                )

        return self.model