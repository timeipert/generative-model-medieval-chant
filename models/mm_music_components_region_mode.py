import pymc as pm
from pymc.pytensorf import collect_default_updates
import numpy as np
import pytensor
import pytensor.tensor as pt
from models import base_model


class MelodicComponentsRegionModeModel(base_model.Model):
    def __init__(self, data=None, coords=None):
        super().__init__(data=data, coords=coords)
        self.init_probs_prior = np.full(self.V, 1.5)
        self.num_modes = 8
        self.num_regions = 3  # West, East, Italy

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
            fn=MelodicComponentsRegionModeModel._scan_step_fn,
            sequences=[pt.arange(sequence_lengths.max())],
            outputs_info=[initial_state],
            non_sequences=[transition_matrix, sequence_lengths],
            strict=True,
        )
        return states

    def _build_transition_matrix_components(self, param_shape):
        from_pitch_idx = np.arange(self.V)[:, None]
        to_pitch_idx = np.arange(self.V)[None, :]

        base_logits = pm.Normal("base_logits", mu=0, sigma=1.0, shape=(*param_shape, self.V, self.V))

        weight_interval = pm.HalfNormal("weight_interval", sigma=1.0, shape=param_shape)
        weight_register = pm.HalfNormal("weight_register", sigma=1.0, shape=param_shape)
        weight_repeat = pm.Normal("weight_repeat", mu=0, sigma=1.0, shape=param_shape)
        weight_direction = pm.Normal("weight_direction", mu=0, sigma=1.0, shape=param_shape)
        register_center = pm.Normal("register_center", mu=self.V / 4.0, sigma=self.V / 4.0, shape=param_shape)

        w_int_exp = weight_interval[..., None, None]
        w_reg_exp = weight_register[..., None, None]
        w_rep_exp = weight_repeat[..., None, None]
        w_dir_exp = weight_direction[..., None, None]
        reg_center_exp = register_center[..., None, None]

        interval_dist_sq = (from_pitch_idx - to_pitch_idx) ** 2
        pitch_dist_sq = (to_pitch_idx - reg_center_exp) ** 2
        is_repeat = (from_pitch_idx == to_pitch_idx).astype(float)
        direction = np.sign(to_pitch_idx - from_pitch_idx).astype(float)

        logits = (
                base_logits
                - (w_int_exp * interval_dist_sq)
                - (w_reg_exp * pitch_dist_sq)
                + (w_rep_exp * is_repeat)
                + (w_dir_exp * direction)
        )

        return pm.Deterministic("transition_matrices", pm.math.softmax(logits, axis=-1))

    def _define_priors(self, use_modes=False, use_regions=False):
        param_shape = []
        if use_regions:
            param_shape.append(self.num_regions)
        if use_modes:
            param_shape.append(self.num_modes)
        param_shape = tuple(param_shape)

        transition_matrices = self._build_transition_matrix_components(param_shape)
        init_probs = pm.Dirichlet("init_probs", a=self.init_probs_prior)

        mode_probs = None
        if use_regions and use_modes:
            global_mode_probs = pm.Dirichlet("global_mode_probs", a=np.ones(self.num_modes))
            kappa = pm.Exponential("mode_probs_kappa", 1.0)
            mode_probs = pm.Dirichlet("mode_probs", a=global_mode_probs * kappa,
                                      shape=(self.num_regions, self.num_modes))
        elif use_modes:
            mode_probs = pm.Dirichlet("mode_probs", a=np.ones(self.num_modes))

        return init_probs, transition_matrices, mode_probs

    def _define_inference_likelihood(self, init_probs, transition_matrices, mode_probs, initial_states_obs,
                                     sequence_states_obs, sequence_lengths_obs, mode_obs=None, region_obs=None):
        use_modes = mode_obs is not None
        use_regions = region_obs is not None

        pm.Categorical("initial_states", p=init_probs, observed=initial_states_obs)

        if use_regions and use_modes:
            pm.Categorical("mode", p=mode_probs[region_obs], observed=mode_obs)
        elif use_modes:
            pm.Categorical("mode", p=mode_probs, observed=mode_obs)

        initial_states_padded = pt.shape_padleft(initial_states_obs)
        all_states = pt.concatenate([initial_states_padded, sequence_states_obs], axis=0)

        prev_states = all_states[:-1]
        current_states = all_states[1:]

        time_indices = pt.arange(sequence_states_obs.shape[0])
        valid_time_mask = time_indices[:, None] < sequence_lengths_obs[None, :]

        valid_prev_states = prev_states[valid_time_mask]
        valid_current_states = current_states[valid_time_mask]

        sequence_indices = pt.arange(sequence_states_obs.shape[1])[None, :]
        broadcasted_sequence_indices = pt.alloc(sequence_indices, prev_states.shape[0], prev_states.shape[1])
        valid_sequence_indices = broadcasted_sequence_indices[valid_time_mask]

        if use_modes:
            mode_obs_pt = pt.as_tensor_variable(mode_obs)
        if use_regions:
            region_obs_pt = pt.as_tensor_variable(region_obs)

        if use_regions and use_modes:
            valid_transition_probs = transition_matrices[
                region_obs_pt[valid_sequence_indices], mode_obs_pt[valid_sequence_indices], valid_prev_states]
        elif use_modes:
            valid_transition_probs = transition_matrices[mode_obs_pt[valid_sequence_indices], valid_prev_states]
        else:
            valid_transition_probs = transition_matrices[valid_prev_states]

        pm.Categorical("transitions", p=valid_transition_probs, observed=valid_current_states)

    def _define_generative_likelihood(self, init_probs, transition_matrices, mode_probs=None, mode_obs=None,
                                      region_obs=None, use_modes=False, use_regions=False):
        if use_regions and use_modes:
            if region_obs is not None:
                region = pt.as_tensor_variable(region_obs)
            else:
                region = pm.Categorical("region", p=np.ones(self.num_regions) / self.num_regions)

            if mode_obs is not None:
                mode = pt.as_tensor_variable(mode_obs)
            else:
                mode = pm.Categorical("mode",
                                      p=pm.Dirichlet("global_mode_probs_gen", a=np.ones(self.num_modes))[region])

            active_transition_matrix = transition_matrices[region, mode]
            batch_shape = region.shape

        elif use_modes:
            if mode_obs is not None:
                mode = pt.as_tensor_variable(mode_obs)
            else:
                mode = pm.Categorical("mode", p=pm.Dirichlet("mode_probs_gen", a=np.ones(self.num_modes)))

            active_transition_matrix = transition_matrices[mode]
            batch_shape = mode.shape

        else:
            active_transition_matrix = transition_matrices
            batch_shape = None

        initial_states = pm.Categorical("initial_states", p=init_probs, shape=batch_shape)
        sequence_lengths = pm.Poisson("sequence_lengths", self.mean_seq_len, shape=batch_shape)

        pm.CustomDist(
            "generated_sequence",
            initial_states,
            active_transition_matrix,
            sequence_lengths,
            dist=MelodicComponentsRegionModeModel._scan_generative_dist,
        )

    def build_model(self, initial_states_obs=None, sequence_states_obs=None, sequence_lengths_obs=None, mode_obs=None,
                    region_obs=None, use_modes=False, use_regions=False, **kwargs):
        coords = getattr(self, "coords", None)

        if sequence_states_obs is not None:
            use_modes = use_modes or (mode_obs is not None)
            use_regions = use_regions or (region_obs is not None)

        with pm.Model(coords=coords) as self.model:
            init_probs, transition_matrices, mode_probs = self._define_priors(use_modes=use_modes,
                                                                              use_regions=use_regions)

            if sequence_states_obs is not None:
                self._define_inference_likelihood(
                    init_probs,
                    transition_matrices,
                    mode_probs,
                    initial_states_obs,
                    sequence_states_obs,
                    sequence_lengths_obs,
                    mode_obs,
                    region_obs
                )
            else:
                self._define_generative_likelihood(
                    init_probs,
                    transition_matrices,
                    mode_probs,
                    mode_obs,
                    region_obs,
                    use_modes,
                    use_regions
                )

        return self.model