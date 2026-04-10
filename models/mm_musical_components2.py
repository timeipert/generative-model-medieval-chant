import pymc as pm
from pymc.pytensorf import collect_default_updates
import numpy as np
import pytensor
import pytensor.tensor as pt
from models import base_model


class MelodicComponentsIIMarkovModel(base_model.Model):
    def __init__(self, data=None, coords=None):
        super().__init__(data=data, coords=coords)
        self.init_probs_prior = np.full(self.V, 1.5)

    @staticmethod
    def _scan_step_fn(current_time, prev_state, transition_matrix, sequence_lengths):
        current_state = pm.Categorical.dist(p=transition_matrix[prev_state])

        absorbing_state = pm.DiracDelta.dist(-1, shape=current_state.shape)
        current_state = pt.where(current_time >= sequence_lengths, absorbing_state, current_state)

        return current_state, collect_default_updates([current_state])

    @staticmethod
    def _scan_generative_dist(initial_state, transition_matrix, sequence_lengths, size):
        states, _ = pytensor.scan(
            fn=MelodicComponentsIIMarkovModel._scan_step_fn,
            sequences=[pt.arange(sequence_lengths.max())],
            outputs_info=[initial_state],
            non_sequences=[transition_matrix, sequence_lengths],
            strict=True,
        )
        return states

    def _build_transition_matrix_components(self):
        from_pitch_idx = np.arange(self.V)[:, None]
        to_pitch_idx = np.arange(self.V)[None, :]

        interval_dist_sq = (from_pitch_idx - to_pitch_idx) ** 2
        weight_interval = pm.HalfNormal("weight_interval", sigma=1.0)

        register_center = pm.Normal("register_center", mu=self.V / 4.0, sigma=self.V / 4.0)
        pitch_dist_sq = (to_pitch_idx - register_center) ** 2
        weight_register = pm.HalfNormal("weight_register", sigma=1.0)

        is_repeat = (from_pitch_idx == to_pitch_idx).astype(float)
        weight_repeat = pm.Normal("weight_repeat", mu=0, sigma=1.0)

        direction = np.sign(to_pitch_idx - from_pitch_idx).astype(float)
        weight_direction = pm.Normal("weight_direction", mu=0, sigma=1.0)

        logits = (
                - (weight_interval * interval_dist_sq)
                - (weight_register * pitch_dist_sq)
                + (weight_repeat * is_repeat)
                + (weight_direction * direction)
        )

        return pm.Deterministic("transition_matrix", pm.math.softmax(logits, axis=-1))

    def _define_priors(self):
        init_probs = pm.Dirichlet("init_probs", a=self.init_probs_prior)
        transition_matrix = self._build_transition_matrix_components()

        return init_probs, transition_matrix

    def _define_inference_likelihood(self, init_probs, transition_matrix, initial_states_obs, sequence_states_obs,
                                     sequence_lengths_obs):
        pm.Categorical("initial_states", p=init_probs, observed=initial_states_obs)

        initial_states_padded = pt.shape_padleft(initial_states_obs)
        all_states = pt.concatenate([initial_states_padded, sequence_states_obs], axis=0)

        prev_states = all_states[:-1]
        current_states = all_states[1:]

        time_indices = pt.arange(sequence_states_obs.shape[0])
        valid_time_mask = time_indices[:, None] < sequence_lengths_obs[None, :]

        valid_prev_states = prev_states[valid_time_mask]
        valid_current_states = current_states[valid_time_mask]

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
            dist=MelodicComponentsIIMarkovModel._scan_generative_dist,
        )

    def build_model(self, initial_states_obs=None, sequence_states_obs=None, sequence_lengths_obs=None, **kwargs):
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