import pymc as pm
import numpy as np
import pytensor
import pytensor.tensor as pt
from models import base_model


class Model2(base_model.Model):
    def __init__(self, data=None, coords=None):
        super().__init__(data=data, coords=coords)
        self.alpha = np.ones(self.V)
        self.beta = np.ones((self.V, self.V))

    @staticmethod
    def step_fn(t, state_tm1, transition_p, end_t):
        V = transition_p.shape[0]
        idx = pt.clip(state_tm1, 0, V - 1)

        state_t = pm.Categorical.dist(p=transition_p[idx])
        # variable length like here: https://discourse.pymc.io/t/using-discretemarkovchain-with-batch-dimensions-and-data-of-different-lengths/16611/2
        end_state = pm.DiracDelta.dist(-1, shape=state_t.shape)
        state_t = pt.where(t >= end_t, end_state, state_t)

        return state_t, pm.pytensorf.collect_default_updates([state_t])

    @staticmethod
    def scan_dist(init_state, transition_p, end_t, size):
        states, _ = pytensor.scan(
            fn=Model2.step_fn,
            sequences=[pt.arange(end_t.max())],
            outputs_info=[init_state],
            non_sequences=[transition_p, end_t],
            strict=True,
        )
        return states

    @staticmethod
    def transpose_beta(A_0, m, scale_func=lambda d: 0.5 ** d):
        V = A_0.shape[0]
        i, j = pt.ogrid[:V, :V]

        dist_pos = pt.maximum(m - i, m - j).clip(0, V)
        mask_pos = (i < m) | (j < m)

        s_neg = V + m
        dist_neg = pt.maximum(i - (s_neg - 1), j - (s_neg - 1)).clip(0, V)
        mask_neg = (i >= s_neg) | (j >= s_neg)

        shifted = pt.roll(A_0, m, axis=0)
        shifted = pt.roll(shifted, m, axis=1)

        dist = pt.switch(m > 0, dist_pos, dist_neg)
        mask = pt.switch(m > 0, mask_pos, mask_neg)

        base_scaled = A_0 * scale_func(dist)
        unnormalized = pt.switch(mask, base_scaled, shifted)
        #unnormalized = unnormalized + 1e-8

        row_sums = pt.sum(unnormalized, axis=1, keepdims=True)
        return unnormalized / row_sums

    def build_model(self, x_0_obs=None, x_t_obs=None, lengths_obs=None):
        with pm.Model() as model:
            nu = pm.Dirichlet("nu", a=np.ones(4))
            m = pm.Categorical("m", p=nu)

            A_0 = pm.Dirichlet("A_0", a=self.beta)
            transition_p = pm.Deterministic("A_m", Model2.transpose_beta(A_0, m))

            pi = pm.Dirichlet("pi", a=self.alpha)

            if x_t_obs is not None:
                x_0 = pm.Categorical("x_0", p=pi, observed=x_0_obs)

                init_state = pt.as_tensor(x_0_obs, dtype="int64")
                end_t = pt.as_tensor(lengths_obs, dtype="int64")

                steps = pm.CustomDist(
                    "x_t+1",
                    init_state,
                    transition_p,
                    end_t,
                    dist=Model2.scan_dist,
                    observed=x_t_obs,
                )
            else:
                x_0 = pm.Categorical("x_0", p=pi)
                length = pm.Poisson("length", self.mean_seq_len)

                steps = pm.CustomDist(
                    "x_t+1",
                    x_0,
                    transition_p,
                    length,
                    dist=Model2.scan_dist,
                )
        #print(model.initial_point())
        self.model = model
        return model