import pymc as pm
from pymc.pytensorf import collect_default_updates
import numpy as np
import pytensor
import pytensor.tensor as pt
from models import base_model


class MelodicMarkovModel(base_model.Model):
    def __init__(self, data=None, coords=None):
        super().__init__(data=data, coords=coords)
        self.alpha_prior = np.full(self.V, 1.5)
        self.beta_prior = np.full((self.V, self.V), 1.5)

    @staticmethod
    def _scan_step_fn(current_time, prev_state, transition_matrix, sequence_lengths):
        current_state = pm.Categorical.dist(p=transition_matrix[prev_state])

        # Inject absorbing state (-1) when sequence length is reached
        # See: https://discourse.pymc.io/t/using-discretemarkovchain-with-batch-dimensions-and-data-of-different-lengths/16611/2
        absorbing_state = pm.DiracDelta.dist(-1, shape=current_state.shape)
        current_state = pt.where(current_time >= sequence_lengths, absorbing_state, current_state)

        return current_state, collect_default_updates([current_state])

    @staticmethod
    def _scan_generative_dist(initial_state, transition_matrix, sequence_lengths, size):
        states, _ = pytensor.scan(
            fn=MelodicMarkovModel._scan_step_fn,
            sequences=[pt.arange(sequence_lengths.max())],
            outputs_info=[initial_state],
            non_sequences=[transition_matrix, sequence_lengths],
            strict=True,
        )
        return states

    def _define_priors(self):
        init_probs = pm.Dirichlet("init_probs", a=self.alpha_prior)
        transition_matrix = pm.Dirichlet("transition_matrix", a=self.beta_prior)

        return init_probs, transition_matrix

    def _define_inference_likelihood(self, init_probs, transition_matrix, initial_states_obs, sequence_states_obs,
                                     sequence_lengths_obs):
        pm.Categorical("initial_states", p=init_probs, observed=initial_states_obs)

        initial_states_padded = pt.shape_padleft(initial_states_obs)
        all_states = pt.concatenate([initial_states_padded, sequence_states_obs], axis=0)

        states_t_minus_1 = all_states[:-1]
        states_t = all_states[1:]

        time_indices = pt.arange(sequence_states_obs.shape[0])
        valid_time_mask = time_indices[:, None] < sequence_lengths_obs[None, :]

        valid_prev_states = states_t_minus_1[valid_time_mask]
        valid_current_states = states_t[valid_time_mask]

        pm.Categorical(
            "transitions",
            p=transition_matrix[valid_prev_states],
            observed=valid_current_states
        )

    def _define_generative_likelihood(self, init_probs, transition_matrix):
        initial_states = pm.Categorical("initial_states", p=init_probs)
        sequence_lengths = pm.Poisson("sequence_lengths", self.mean_seq_len)

        pm.CustomDist(
            "generated_sequence",
            initial_states,
            transition_matrix,
            sequence_lengths,
            dist=MelodicMarkovModel._scan_generative_dist,
        )

    def build_model(self, initial_states_obs=None, sequence_states_obs=None, sequence_lengths_obs=None, **kwargs):
        # Passing coords directly to Model context if they exist in base_model
        coords = getattr(self, "coords", None)

        with pm.Model(coords=coords) as self.model:
            init_probs, transition_matrix = self._define_priors()

            if sequence_states_obs is not None:
                self._define_inference_likelihood(
                    init_probs,
                    transition_matrix,
                    initial_states_obs,
                    sequence_states_obs,
                    sequence_lengths_obs
                )
            else:
                self._define_generative_likelihood(
                    init_probs,
                    transition_matrix
                )

        return self.model